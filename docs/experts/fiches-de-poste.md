# Fiches de poste — experts Kinetic

> Documents opérationnels pour recruter, cadrer ou instancier les profils
> décrits dans [`docs/experts.md`](../experts.md).
>
> Chaque fiche donne : la mission, le contexte, les compétences, le périmètre
> d'écriture, les fichiers de contexte, les commandes de mesure, les livrables
> 30/60/90 et les critères de succès.
>
> **Règle commune** : un seul changement par essai, une configuration par
> essai, contrôles avant et après, médiane de ≥ 6 échantillons, aucune
> régression > 3 %, et les gates d'exactitude avant toute publication.

---

## Fiche 1 — Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)

**Missions détaillées** : M02 (préfill Vulkan), M03 (MoE), M07 (HIP).
Voir [`missions.md`](missions.md).

**Priorité** : P0 — chemin critique.
**Rattachement** : équipe backend GPU.
**Mission** : combler l'écart de préfill Vulkan, débloquer le MoE, poser les
bases d'un backend HIP complet.

### Contexte

- Le décodage dense Vulkan est à 95 % de llama.cpp (38,4 contre 40,1 t/s sur
  un dense 27B IQ3_XXS), mais le préfill est à 23 % (261,9 contre 1154,2 t/s).
- `gemv_many.wgsl` garde 32 tokens en vol ; ce n'est pas un GEMM tuilé.
- `Commands::submit` soumet puis attend `vkWaitForFences` ; pas de timeline
  semaphores, pas de `VkPipelineCache` persisté.
- `KvCache` est en f16, non quantifié.
- `moe_gemv.wgsl` lit les experts un slot à la fois ; pas de GEMM groupé.
- Le MoE 30B A3B est à 55,7 t/s en décodage contre 181,9 pour llama.cpp, et
  489 contre 2637 t/s en préfill.
- La cible architecturale est **un seul backend complet par API** : tout
  Vulkan dans `kinetic-vulkan`, tout HIP dans `kinetic-hip` (à créer).

### Responsabilités

- Profiler les noyaux Vulkan et identifier memory-bound vs compute-bound.
- Concevoir et implémenter un GEMM de préfill tuilé (Q4_K, Q5_K, Q6_K, Q8_0,
  IQ3_XXS, IQ4_XS), bit-exact avec la passe mono-token.
- Ajouter coopmat (`VK_KHR_cooperative_matrix`), split-K et tuilage LDS.
- Remplacer la soumission synchrone par une soumission asynchrone avec
  timeline semaphores, sans casser l'ordre des dépendances.
- Persister le cache de pipelines et réduire le temps de démarrage.
- Implémenter un GEMM/GEMV groupé pour le MoE.
- Ajouter le KV q8_0 (puis q4_0) avec la même exactitude.
- Aligner Vulkan sur les classes de fonctionnalités du backend CUDA (grands
  lots de préfill, flash attention, MoE groupé, split multi-GPU) lorsque
  l'API le permet, sans copier le code.
- Porter les noyaux sur HIP/ROCm (`gfx1201`) dans `kinetic-hip`.
- Consigner chaque essai dans `PLAN.md` et `ledger.jsonl`.

### Compétences requises

- Vulkan compute 1.3 : descriptors, push constants, specialization constants,
  subgroups 32/64, barriers, command buffers, timeline semaphores,
  `VkPipelineCache`, cooperative matrix.
- WGSL, SPIR-V, naga ; RGP ; ISA AMDGPU ; RDNA 4 (wave32, LDS 64 Kio,
  VGPR/SGPR, occupancy, Infinity Cache).
- GEMM : tuilage, register blocking, split-K, swizzling, bank conflicts,
  épilogue fusionné, FP16/BF16/INT8 dot.
- Quantification : formats de blocs, dequant-in-registers, dot int8 packé
  (`v_dot4_i32_i8`, `sdot4`).
- Roofline et compteurs GPU.
- Exactitude : accumulation f32, batch-invariance, égalité passe prompt /
  passe mono-token.
- Rust, `ash`, `unsafe` aux frontières.

### Compétences appréciées

- HIP/ROCm, `hipcc`, rocWMMA, rocBLAS/hipBLASLt, hipGraph, rocprof.
- CUDA (pour partager les sources de noyaux, pas pour dupliquer).
- Metal/SPIR-V croisé.
- Expérience de llama.cpp, ggml, vLLM ou TensorRT-LLM.

### Périmètre d'écriture

- `crates/kinetic-vulkan/**`
- `crates/kinetic-hip/**` (à créer)
- `crates/kinetic-runtime/src/{device,capabilities,factory,memory}.rs`
  (propositions d'API uniquement)
- `PLAN.md` (entrées d'expériences), `ledger.jsonl`, `configs/`

### Fichiers de contexte

- `PLAN.md` §4 P0.2 et P0.3.
- `llama-kinetic.md` §7.2, §7.3, §7.10, §7.11.
- `crates/kinetic-vulkan/src/{commands,pipeline,gemv,attention,moe}.rs`.
- `crates/kinetic-vulkan/shaders/*.wgsl`.
- `crates/kinetic-models/src/vulkan/*` (à déplacer vers `kinetic-vulkan`).
- `docs/src/devices.md`, `docs/src/contributing.md`.
- `crates/kinetic-quant/src/{types,dequant}.rs` pour les formats.

### Commandes de mesure

```sh
# Build
cargo build --release --locked -p kinetic-cli --features vulkan

# Tests d'exactitude
cargo test --release --locked -p kinetic-vulkan
cargo test --release --locked -p kinetic-models

# Bench
target/release/kinetic bench MODEL.gguf --device vulkan --json -r 6
llama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6

# Profil
RGP capture sur pp512 (Radeon GPU Profiler)
rocm-smi --showuse --showtemp --showmemuse
```

### Livrables 30/60/90

- **30 jours** : baseline, roofline, profil RGP, prototype de GEMM tuilé
  Q4_K bit-exact, ≥ 1,5× sur pp512.
- **60 jours** : coopmat + split-K, ≥ 2× ; soumission asynchrone ; cache de
  pipelines ; pas de régression tg128.
- **90 jours** : GEMM MoE groupé ; KV q8_0 ; squelette HIP avec GEMV
  Q4_K/Q5_K/Q6_K/Q8_0 et attention FP16.

### Critères de succès

- pp512/pp2048 ≥ 2× le baseline Kinetic, sans régression > 3 % sur tg128.
- Tous les tests d'identité verts.
- `kinetic bench` dit `edit_identical: true`.
- `PLAN.md` contient chaque essai avec commande, commit, date et verdict.
- Aucun `unsafe` sans commentaire `SAFETY`.

### Interactions

- **Runtime** : interfaces de graphe, soumission asynchrone, KV.
- **Numérique** : formats, tolérances, gates.
- **QA** : protocole A/B, CI GPU, ledger.
- **Rust backend** : il possède le trait `ModelRunner` et le runner CPU ; le
  GPU AMD ne fait que proposer les capacités nécessaires.

### Profil / sourcing

HPC, ROCm, moteurs de jeu, compute Vulkan, équipes de noyaux ML. Un profil
CUDA-only n'est pas suffisant : il doit savoir lire du WGSL/SPIR-V et
raisonner RDNA.

### Signaux d'alerte

- Ne sait pas expliquer pourquoi un noyau est memory-bound.
- Ne sait pas lire un compteur RGP.
- Propose de « tout réécrire en CUDA ».
- N'a jamais écrit de GEMM tuilé.
- Ignore les contraintes d'exactitude.

---

## Fiche 2 — Architecte runtime d'inférence

**Missions détaillées** : M04 (KV cache), M05 (ordonnanceur).
Voir [`missions.md`](missions.md).

**Priorité** : P0 — chemin critique.
**Rattachement** : équipe runtime/serveur.
**Mission** : faire passer le serveur d'un modèle « N backends indépendants »
à un ordonnanceur token-level qui tient la concurrence, le TTFT et le p99.

**Chantier prioritaire** : M04 (KV cache) avant M05 (ordonnanceur). Le KV
est le prochain plafond mémoire et le prérequis de la concurrence.

### Contexte

- `Engine.free` est un `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>` : une
  requête par backend.
- `PREFILL_CHUNK = 256` existe, mais le chunking est par requête.
- `Convoy` mutualise des passes, mais ne planifie pas les requêtes.
- Le KV est contigu par séquence, non paginé.
- `MAX_CONNECTIONS = 64`, `Connection: close`, pas de keep-alive.
- Les technologies RAD-1, Warm Start, Zero Wait, Shortlist et Convoy doivent
  rester exactes.

### Responsabilités

- Concevoir un ordonnanceur token-level : admission, priorité, fairness,
  annulation, backpressure.
- Introduire un pool KV paginé, dimensionné pour N × contexte, avec
  défragmentation et éviction.
- Ajouter le chunked prefill global et le recouvrement préfill/décodage.
- Gérer le rollback KV du spéculatif et la réutilisation de préfixe.
- Intégrer le placement mémoire (layers, experts, tenseurs) et le multi-GPU.
- Ajouter les métriques par token : TTFT, tok/s, p50/p99, cache hit, queue
  depth.
- Préserver les invariants d'exactitude et les tests existants.

### Compétences requises

- Continuous batching, paged KV, prefix caching, chunked prefill, admission
  control, fairness, priorité, annulation, backpressure.
- Décodage spéculatif : draft/verify, arbre, rollback, réutilisation de KV.
- Rust concurrence : threads/async, `Mutex`/`Condvar`, channels, atomics,
  `Send`/`Sync`, scoped threads, panic safety.
- Mémoire : pool KV, fragmentation, mémoire unifiée, pinned host, paging des
  experts, watchdog OOM.
- Graph capture/replay, multi-stream.
- Observabilité et mesure de latence.

### Compétences appréciées

- Expérience vLLM, SGLang, TensorRT-LLM, llama.cpp server, ONNX Runtime.
- Connaissance des modèles hybrides SSM/GDN et des slots d'état.
- Expérience des systèmes locaux-first (pas cloud).

### Périmètre d'écriture

- `crates/kinetic-runtime/**` (à créer : ordonnanceur, pool KV, métriques)
- `crates/kinetic-server/src/{engine,sequence,backend,server}.rs`
- `crates/kinetic-models/src/{convoy,memory,per_layer}.rs`
- `crates/kinetic-models/src/warm/**`
- `PLAN.md`, `ledger.jsonl`, `configs/`

### Fichiers de contexte

- `PLAN.md` §4 P1.5 et P1.6.
- `llama-kinetic.md` §7.6, §7.9, §7.10.
- `crates/kinetic-server/src/engine.rs`, `sequence.rs`, `backend.rs`.
- `crates/kinetic-models/src/convoy.rs`, `memory.rs`, `per_layer.rs`.
- `crates/kinetic-models/src/warm/**`.
- `docs/src/how-it-is-fast.md` (RAD-1, Warm Start, Zero Wait, Convoy).
- `docs/src/devices.md` (layer split, MoE offload).

### Commandes de mesure

```sh
# Serveur
target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4

# Charge concurrente (exemple avec oha, ou un script maison)
oha -n 64 -c 4 --json http://127.0.0.1:8080/v1/chat/completions ...

# Exactitude sous charge
python scripts/same_answer_demo.py --model MODEL.gguf \
  --kinetic target/release/kinetic --kinetic-args "--device vulkan"

# Robustesse
python scripts/server_abuse.py --port 8091
```

### Livrables 30/60/90

- **30 jours** : cartographie du runtime actuel, design doc, métriques
  manquantes, prototype d'ordonnanceur.
- **60 jours** : pool KV N × contexte, chunked prefill global, annulation,
  métriques par token.
- **90 jours** : benchmarks c1/c4/c16/c64, p99 et TTFT documentés, aucune
  régression d'exactitude.

### Critères de succès

- Débit agrégé en hausse à c4/c16/c64.
- p99 et TTFT mesurés et reproductibles.
- `same_answer_demo.py` identique sous charge.
- Aucun test d'identité cassé.
- Aucune fuite mémoire ni cache non borné.

### Interactions

- **GPU AMD** : interfaces de graphe, KV, soumission.
- **Rust backend** : `kinetic-runtime`, découpage des responsabilités.
- **QA** : protocole de charge, p99, soak.
- **Serveur/API** : métriques, slots, keep-alive.

### Profil / sourcing

Équipes serving/LLM runtime, vLLM, SGLang, TensorRT-LLM, llama.cpp. Rare :
il faut du systèmes distribué local, pas du cloud.

### Signaux d'alerte

- Ne connaît que le batching statique.
- Ne sait pas expliquer le rollback KV.
- Propose un « rewrite » complet sans migration.
- Ignore la fairness et l'annulation.
- Ne mesure que le débit moyen, jamais le p99.

---

## Fiche 3 — Architecte Rust systèmes/backend

**Mission détaillée** : M06 (consolidation backend).
Voir [`missions.md`](missions.md).

**Priorité** : P0/P1 — fondation.
**Rattachement** : équipe cœur.
**Mission** : réduire la duplication CUDA/Vulkan, faire de chaque API un
backend complet et préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.

### Contexte

- `kinetic-models/src/cuda` ≈ 4,5 k LOC et `kinetic-cuda` ≈ 8,5 k LOC.
- `kinetic-models/src/vulkan` ≈ 2,2 k LOC et `kinetic-vulkan` ≈ 5,9 k LOC.
- `kinetic-models` a des features `cuda`/`vulkan` et des modules backend.
- Ajouter HIP créerait `kinetic-hip` + `kinetic-models/src/hip`.
- Les seams `Family`, `Target`, `Backend`, `Projector` sont bons et doivent
  être préservés.
- La surface `unsafe` est concentrée aux frontières SIMD/GPU/FFI.

### Responsabilités

- Créer `kinetic-runtime` : devices, capacités, budget mémoire, factory,
  ordonnanceur, pool KV, métriques. Pas de types de modèle.
- Extraire `ModelRunner` dans `kinetic-models` : charger, préfill, décoder,
  état de séquence, capacités. Le runner CPU devient la référence.
- Déplacer les exécuteurs CUDA/Vulkan dans leurs crates backend respectifs,
  via `ModelRunner` ; ce n'est pas un simple `git mv`.
- Retirer les features backend de `kinetic-models`.
- Ajouter une factory de backend dans `kinetic-server` / `kinetic-cli`.
- Ajouter un backend de test pour `kinetic-models`.
- Auditer les `unsafe` et les invariants `Send`/`Sync`.
- Améliorer le démarrage, le temps de build et la taille binaire.
- Documenter la règle « un backend complet par API » et son contrat de
  compatibilité.

### Compétences requises

- Rust avancé : lifetimes, traits, génériques vs `dyn`, zéro-copie,
  soundness `unsafe`, `Send`/`Sync`, `Arc`/`Rc`, mmap, allocators, layouts
  cache-friendly, monomorphisation.
- Architecture : ports/adapters, abstraction backend, graphe/dispatch,
  feature flags, compilation conditionnelle, FFI.
- Performance hôte : `perf`, flamegraph, heaptrack, criterion, LLVM remarks,
  LTO/PGO/BOLT.
- Build : workspaces Cargo, cross-compilation, toolchains CUDA/HIP, CI.
- Tests : unitaires, intégration, property-based, fuzz.

### Compétences appréciées

- Connaissance de CUDA/Vulkan/Metal au niveau API, pas nécessairement noyau.
- Expérience de refactor de moteurs de rendu ou de bases de données.
- Expérience PyO3/C ABI.

### Périmètre d'écriture

- `Cargo.toml`, `crates/*/Cargo.toml`
- `crates/kinetic-runtime/**` (à créer)
- `crates/kinetic-models/src/{lib,model}.rs` (retrait des modules backend)
- `crates/kinetic-cuda/**`, `crates/kinetic-vulkan/**` (accueil des
  exécuteurs)
- `crates/kinetic-server/src/{backend,lib}.rs`
- `docs/src/contributing.md`, `docs/src/building.md`

### Fichiers de contexte

- `docs/src/contributing.md` (layout, règles).
- `docs/src/building.md` (build, features, tests).
- `crates/kinetic-models/src/lib.rs`, `model.rs`.
- `crates/kinetic-models/src/{cuda,vulkan}/**`.
- `crates/kinetic-cuda/src/**`, `crates/kinetic-vulkan/src/**`.
- `crates/kinetic-server/src/{backend,engine}.rs`.
- `PLAN.md` §3 et §4.

### Commandes de mesure

```sh
cargo build --release --locked --workspace --all-features
cargo test --release --locked --workspace
cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
cargo tree -p kinetic-models
cargo bloat --release -p kinetic-cli   # taille binaire
perf record -g target/release/kinetic ...  # overhead hôte
```

### Livrables 30/60/90

- **30 jours** : cartographie de la duplication, design de `kinetic-runtime`,
  liste des `unsafe` à auditer.
- **60 jours** : déplacement mécanique d'un exécuteur (CUDA ou Vulkan) dans
  son crate, tests verts, pas de changement de comportement.
- **90 jours** : les deux exécuteurs déplacés, `kinetic-models` sans feature
  backend, factory de backend, backend de test.

### Critères de succès

- `kinetic-models` ne dépend plus de CUDA/Vulkan.
- Un seul crate par API contient tout le code de cette API.
- Les tests d'identité restent verts à chaque étape.
- Le temps de build et le démarrage ne régressent pas.
- Chaque `unsafe` a un commentaire `SAFETY` et un test associé.

### Interactions

- **GPU AMD** : accueil des exécuteurs, interfaces de graphe.
- **Runtime** : `kinetic-runtime`, scheduler.
- **QA** : tests de non-régression, build CI.
- **Build/release** : features, cross-compilation.

### Profil / sourcing

Systèmes, bases de données, moteurs de jeu, embedded. Rare : il faut du Rust
de production, pas du Rust d'exercice.

### Signaux d'alerte

- Veut introduire MLIR ou un compilateur de graphes.
- Ne connaît pas `unsafe`/`Send`/`Sync` en profondeur.
- Propose un rewrite complet.
- Ne sait pas mesurer le temps de build ou l'overhead hôte.
- Casse les seams existants.

---

## Fiche 4 — Ingénieur quantification et exactitude numérique

**Mission détaillée** : M08 (quantification et formats KV).
Voir [`missions.md`](missions.md).

**Priorité** : P0/P1 — garde-fou.
**Rattachement** : équipe numérique.
**Mission** : étendre la quantification et garantir que chaque gain de
vitesse reste exact.

### Contexte

- 5 cibles de quantification contre ~30 dans llama.cpp.
- Pas d'imatrix, pas d'IQ encode.
- NVFP4 et F8_E4M3 absents du runtime.
- KV : Vulkan f16 uniquement ; CUDA f16/q8_0 ; CPU f32.
- `kinetic-quant` est la définition de référence ; `kdot/tests.rs` compare
  chaque noyau.
- L'exactitude est le différenciateur produit.

### Responsabilités

- Implémenter NVFP4 et F8_E4M3 au runtime.
- Implémenter l'encodeur IQ et l'imatrix.
- Ajouter le KV q8_0/q4_0 avec tests d'exactitude.
- Définir les tolérances et les gates par format.
- Construire un corpus de conformance et des tests property/fuzz.
- Mesurer la qualité (KL/perplexité) et la vitesse par format.
- Refuser tout noyau qui ne passe pas la référence f64.

### Compétences requises

- GGUF, quantization par blocs, K-quants, grilles IQ, MXFP4, NVFP4,
  FP8 E4M3, imatrix.
- Analyse numérique : f64, propagation d'erreur, tolérances, ordre
  d'accumulation, fp16/bf16/fp8, arrondi stochastique.
- Encodeur `quantize`, politique par tenseur, requantification, split/merge.
- Tests différentiels, property-based, fuzzing, conformance.
- Sensibilité performance : dequant-in-registers, vectorisation.

### Compétences appréciées

- Expérience ggml/llama.cpp, GPTQ/AWQ, bitsandbytes.
- Connaissance des kernels CUDA/Vulkan.
- Expérience calibration et datasets.

### Périmètre d'écriture

- `crates/kinetic-quant/**`
- `crates/kinetic-gguf/**`
- `crates/kinetic-cpu/src/kdot/**`
- Tests de noyaux dans `crates/kinetic-cuda/tests/**` et
  `crates/kinetic-vulkan/tests/**`
- `docs/src/models.md`
- `PLAN.md`, `ledger.jsonl`, `configs/`

### Fichiers de contexte

- `docs/src/contributing.md` section « Adding a weight format or a kernel ».
- `crates/kinetic-quant/src/{types,dequant,encode}/**`.
- `crates/kinetic-cpu/src/kdot/tests.rs`.
- `llama-kinetic.md` §7.7.
- `PLAN.md` P0.4.
- `crates/kinetic-gguf/src/**`.

### Commandes de mesure

```sh
cargo test --release --locked -p kinetic-quant
cargo test --release --locked -p kinetic-cpu
cargo test --release --locked -p kinetic-vulkan
target/release/kinetic quantize MODEL.gguf --help
target/release/kinetic perplexity MODEL.gguf
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device cpu
```

### Livrables 30/60/90

- **30 jours** : audit des gates, corpus de conformance, liste des formats
  manquants.
- **60 jours** : NVFP4/FP8 runtime, KV q8_0, tests d'identité.
- **90 jours** : IQ encode + imatrix, politique par tenseur, métriques
  qualité.

### Critères de succès

- Chaque nouveau format a une référence f64 et un noyau testé.
- Aucune régression d'identité.
- La qualité est mesurée, pas supposée.
- Les chiffres sont reproductibles et consignés.

### Interactions

- **GPU AMD** : noyaux de quantification, layouts.
- **Runtime** : KV, mémoire.
- **QA** : gates, corpus.
- **Modèles** : formats des fichiers.

### Profil / sourcing

Compression ML, co-design matériel/logiciel, calcul numérique, ggml. Rare.

### Signaux d'alerte

- « La quantization change forcément la sortie, ce n'est pas grave. »
- Ne sait pas expliquer l'ordre d'accumulation.
- Ne teste que la vitesse, jamais la qualité.
- Ne connaît pas les formats de blocs.

---

## Fiche 5 — Ingénieur performance / qualification

**Mission détaillée** : M01 (baseline, théorie, harnais de mesure), puis
support des missions M02–M15.
Voir [`missions.md`](missions.md).

**Priorité** : P0 — crédibilité.
**Rattachement** : équipe qualité.
**Mission** : rendre la mesure automatisée, reproductible et infalsifiable.

### Contexte

- Le protocole existe dans `PLAN.md` et `llama-kinetic.md` §7.10, mais il est
  manuel.
- Pas de CI GPU ; pas de `THEORY.md` ; pas de `ledger.jsonl`.
- Les scripts existent : `bench_protocol.py`, `bench_vs_llamacpp.py`,
  `conformance.py`, `server_abuse.py`, `same_answer_demo.py`.
- Les chiffres sont la crédibilité du projet.

### Responsabilités

- Écrire `THEORY.md` : plafonds de bande passante, FLOPS, KV, TTFT.
- Créer `ledger.jsonl`, `configs/`, `best-config.json`.
- Automatiser le protocole A/B avec contrôles avant/après.
- Ajouter une CI GPU (Vulkan/ROCm) avec seuil de régression.
- Étendre `conformance.py` à Vulkan.
- Ajouter un test needle 50 %/100 % de contexte.
- Ajouter un soak concurrent et un watchdog mémoire.
- Qualifier chaque release : build propre, poids épinglés, gates.

### Compétences requises

- Méthodologie : médianes, intervalles de confiance, bruit, A/B, état
  thermique, isolation.
- Harness : Python/Rust, JSON, ledger, régression, CI GPU, artefacts pinnés.
- Profilage : RGP, rocprof, Nsight, `perf`, watchdogs.
- Tests : fuzzing, soak, needle, conformance, chaos.
- Reproductibilité : commits, révisions de poids, drivers, conteneurs.

### Compétences appréciées

- Expérience performance engineering/SRE.
- Connaissance de llama.cpp et de ses benchmarks.
- Expérience GitHub Actions self-hosted GPU.

### Périmètre d'écriture

- `scripts/**`
- `.github/workflows/**`
- `PLAN.md`, `THEORY.md`, `ledger.jsonl`, `configs/`, `best-config.json`
- `docs/src/benchmarks.md`

### Fichiers de contexte

- `PLAN.md` §2 et §7.
- `llama-kinetic.md` §7.10–7.12.
- `docs/src/benchmarks.md`.
- `scripts/bench_protocol.py`, `bench_vs_llamacpp.py`, `conformance.py`,
  `server_abuse.py`, `same_answer_demo.py`.
- `.github/workflows/ci.yml`.

### Commandes de mesure

```sh
python scripts/bench_protocol.py MODEL.gguf --llama-bin /path/to/llama-bench
python scripts/bench_vs_llamacpp.py MODEL.gguf --llama-dir /path/to/llama.cpp/bin
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device cpu
python scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091
python scripts/same_answer_demo.py --model MODEL.gguf --kinetic target/release/kinetic
rocm-smi --showuse --showtemp --showmemuse
```

### Livrables 30/60/90

- **30 jours** : `THEORY.md`, ledger, script A/B, inventaire des gates.
- **60 jours** : CI GPU, extension Vulkan de `conformance.py`, needle test.
- **90 jours** : dashboard de régression, qualification de release, rapport
  type.

### Critères de succès

- Un candidat à +5 % est distingué du bruit avec un protocole reproductible.
- Une régression est détectée automatiquement.
- Chaque chiffre publié a une commande, un commit, une date et un état
  machine.
- Les résultats négatifs sont conservés.

### Interactions

- **Tous les experts** : protocole, gates, publication.
- **Build/release** : CI, artefacts.
- **GPU AMD** : profils, compteurs.

### Profil / sourcing

Performance engineering, SRE, équipes benchmark. Rare.

### Signaux d'alerte

- « Best of one run ».
- Ne contrôle pas l'état machine.
- Ne connaît pas les médianes/IC.
- Confond benchmark et démonstration.
- Ne consigne pas les échecs.

---

## Fiche 6 — Ingénieur intégration des architectures

**Mission détaillée** : M09 (couverture des architectures).
Voir [`missions.md`](missions.md).

**Priorité** : P1 — couverture.
**Rattachement** : équipe modèles.
**Mission** : couvrir les architectures manquantes par briques réutilisables.

### Contexte

- 21/180 architectures.
- Les briques rentables : MLA/DSA, hybrides SSM, T5, Gemma 2, GPT-OSS.
- Les modèles locaux bloqués : `qwen3vl`, `deepseek`, `llama4`, `gemma2`,
  `phi3`, `command-r`, `dbrx`.
- Le registre `Family` et les blocs sont les bons points d'extension.

### Responsabilités

- Ajouter des familles dans `arch/<nom>.rs` + une ligne dans `FAMILIES`.
- Réutiliser `transformer.rs`, `blocks/moe.rs`, `blocks/attention.rs`.
- Ajouter les blocs manquants : MLA, DSA, Mamba 1, RWKV, T5, iRoPE, sinks.
- Mapper les tenseurs GGUF vers les blocs.
- Tester la conformance vs llama.cpp.
- Documenter `docs/src/models.md`.

### Compétences requises

- Zoo Transformer : MHA/GQA/MQA/MLA/DSA/sinks, RoPE/iRoPE/partial/scaling,
  MoE top-k/shared/sigmoid, SSM/hybride, norm/activation, T5.
- Lecture GGUF, mapping de tenseurs.
- Conformance vs llama.cpp.
- Discipline : un fichier par famille, pas de `match` central.

### Compétences appréciées

- Expérience llama.cpp/ggml.
- Connaissance des modèles récents (DeepSeek, Qwen, Llama 4, Gemma).
- Notions de quantization.

### Périmètre d'écriture

- `crates/kinetic-models/src/arch/**`
- `crates/kinetic-models/src/blocks/**`
- `docs/src/models.md`
- `scripts/new_family.py` (si nécessaire)

### Fichiers de contexte

- `docs/src/adding-a-family.md`.
- `llama-kinetic.md` §4.1, §5.1, §6.
- `crates/kinetic-models/src/arch/mod.rs` (`FAMILIES`).
- `crates/kinetic-models/src/arch/transformer.rs`.
- `crates/kinetic-models/src/blocks/**`.
- `scripts/new_family.py`.

### Commandes de mesure

```sh
target/release/kinetic info MODEL.gguf
target/release/kinetic run MODEL.gguf --device cpu "Hello"
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device cpu
cargo test --release --locked -p kinetic-models
```

### Livrables 30/60/90

- **30 jours** : une famille proche (phi3, gemma2 ou command-r) chargée et
  conforme.
- **60 jours** : qwen3vl + qwen2moe/phimoe, tests de conformance.
- **90 jours** : MLA/DSA ou hybride SSM, documentation.

### Critères de succès

- Le modèle se charge et génère.
- Les tokens sont identiques à llama.cpp sur le corpus de contrôle.
- Aucun `match` central ajouté.
- La documentation est à jour.

### Interactions

- **Numérique** : formats, qualité.
- **QA** : conformance.
- **Serveur** : templates/tool calls si nécessaire.

### Profil / sourcing

Communauté llama.cpp/transformers, ingénieurs modèles. Moyenne rareté.

### Signaux d'alerte

- Copie-colle un `match` central.
- Ne teste pas contre llama.cpp.
- Ignore les variantes RoPE/norm.
- Ne documente pas.

---

## Fiche 7 — Ingénieur tokenizers / templates / tool calls

**Mission détaillée** : M10 (tokenizers, templates, tool calls).
Voir [`missions.md`](missions.md).

**Priorité** : P1 — compatibilité.
**Rattachement** : équipe compatibilité.
**Mission** : atteindre la compatibilité des tokenizers et des templates.

### Contexte

- 13/68 prétokenizers, 55 manquants.
- UGM, RWKV, PLaMo, Tekken, Jais2, etc. manquants.
- Tokens EOT/EOM/FIM et types spéciaux incomplets.
- Templates intégrés et formats tool-call incomplets.
- L'issue #5 (tool calls dans le raisonnement) est typique.

### Responsabilités

- Ajouter les prétokenizers prioritaires.
- Implémenter UGM et les variantes BPE.
- Ajouter EOT/EOM/FIM et les tokens de contrôle.
- Ajouter les templates intégrés par nom.
- Ajouter les formats tool-call manquants.
- Tester contre `llama-tokenize` et `llama-server`.

### Compétences requises

- BPE, SentencePiece, WordPiece, UGM, Tekken.
- Prétokenizers regex, byte-level, tokens spéciaux.
- Jinja, tool-call formats, JSON Schema/GBNF.
- Tests différentiels.

### Compétences appréciées

- Expérience NLP infrastructure.
- Connaissance des templates Qwen/Llama/Mistral/DeepSeek/GLM/Kimi.

### Périmètre d'écriture

- `crates/kinetic-tokenizer/**`
- `crates/kinetic-server/src/{template,tool_calls,tool_grammar}.rs`
- `docs/src/models.md`

### Fichiers de contexte

- `llama-kinetic.md` §4.2, §5.3.
- `crates/kinetic-tokenizer/src/**`.
- `crates/kinetic-server/src/{template,tool_calls}.rs`.
- `scripts/conformance.py`.
- `docs/src/models.md`.

### Commandes de mesure

```sh
llama-tokenize -m MODEL.gguf -f corpus.txt
cargo test --release --locked -p kinetic-tokenizer
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device cpu
```

### Livrables 30/60/90

- **30 jours** : 5 prétokenizers prioritaires, corpus de contrôle.
- **60 jours** : 15 prétokenizers, UGM, tokens spéciaux.
- **90 jours** : templates intégrés, formats tool-call, split GGUF.

### Critères de succès

- Ids identiques sur le corpus.
- Tokens spéciaux corrects.
- Tool calls valides.
- Aucune régression de template.

### Interactions

- **Modèles** : familles et tokens.
- **Serveur** : templates, tool calls.
- **QA** : corpus.

### Profil / sourcing

Infrastructure NLP. Moyenne rareté.

### Signaux d'alerte

- Ne connaît que tiktoken.
- Ignore les tokens spéciaux.
- Ne teste pas contre llama.cpp.

---

## Fiche 8 — Ingénieur serveur / API / observabilité

**Missions détaillées** : M11 (API/observabilité), M14 (sécurité, en
pilote).
Voir [`missions.md`](missions.md).

**Priorité** : P1 — produit.
**Rattachement** : équipe serveur.
**Mission** : parité API, robustesse et observabilité.

### Contexte

- ~15/45 routes.
- Pas de keep-alive/TLS.
- 64 connexions thread-per-connection.
- `docker stop` à améliorer.
- Compatibilité clients perfectible.

### Responsabilités

- Ajouter les routes manquantes : `/props`, `/slots/:id_slot`, `/infill`,
  `/apply-template`, `/lora-adapters`, `n > 1`, `timings_per_token`.
- Ajouter keep-alive, TLS/API key, CORS, arrêt propre.
- Ajouter un router multi-modèle.
- Améliorer les métriques Prometheus.
- Tester la compatibilité llama-server/OpenAI/Anthropic/Ollama.
- Sécuriser les entrées.

### Compétences requises

- HTTP/1.1, SSE, keep-alive, chunked, timeouts, backpressure, TLS.
- APIs OpenAI/llama-server/Anthropic/Ollama.
- Prometheus, logs, tracing.
- Sécurité applicative.

### Compétences appréciées

- Rust async/threads.
- Expérience serveurs LLM.
- Connaissance llama.cpp server.

### Périmètre d'écriture

- `crates/kinetic-server/src/{server,http,api,llama,metrics}.rs`,
  `crates/kinetic-server/src/ollama/`
- `docs/src/server.md`, `docs/src/clients.md`

### Fichiers de contexte

- `llama-kinetic.md` §4.5, §5.9.
- `PLAN.md` P0.1.
- `scripts/llama_server_compat.py`, `server_abuse.py`.
- `docs/src/server.md`, `docs/src/clients.md`.

### Commandes de mesure

```sh
target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
python scripts/llama_server_compat.py --model MODEL.gguf \
  --llama-server /path/to/llama-server --kinetic target/release/kinetic
python scripts/server_abuse.py --port 8091
```

### Livrables 30/60/90

- **30 jours** : routes manquantes, keep-alive, tests de compatibilité.
- **60 jours** : `n > 1`, TLS/API key, router, métriques.
- **90 jours** : arrêt propre, CORS, documentation, tests de charge.

### Critères de succès

- Les clients existants fonctionnent sans modification.
- Les champs inconnus sont ignorés ou refusés clairement.
- Pas de crash sous `server_abuse.py`.
- Métriques exploitables.

### Interactions

- **Runtime** : slots, métriques, annulation.
- **Tokenizers** : templates, tool calls.
- **QA** : compatibilité, charge.

### Profil / sourcing

Backend/API engineering. Courante.

### Signaux d'alerte

- Ne connaît que REST basique.
- Ignore le streaming/SSE.
- Ne teste pas les clients réels.

---

## Fiche 9 — Ingénieur multimodal

**Mission détaillée** : M12 (vision, audio, OCR).
Voir [`missions.md`](missions.md).

**Priorité** : P2 — expansion.
**Rattachement** : équipe modèles.
**Mission** : vision, audio, OCR.

### Contexte

- 2/73 projecteurs.
- Pas d'audio/vidéo/OCR.
- `kinetic-vision` a un contrat `Projector` propre.
- Le multimodal est XL et hors chemin critique de performance.

### Responsabilités

- Ajouter des projecteurs vision.
- Ajouter le crate `kinetic-audio`.
- Ajouter le pipeline OCR.
- Tester contre `llama-mtmd-debug`.
- Documenter.

### Compétences requises

- ViT/SigLIP/CLIP, projecteurs, preprocessing.
- Vidéo, audio (mel, ASR/TTS), OCR.
- Noyaux GPU vision.

### Périmètre d'écriture

- `crates/kinetic-vision/**`
- `crates/kinetic-audio/**` (à créer)
- `crates/kinetic-models/src/input.rs`
- `docs/src/models.md`

### Fichiers de contexte

- `llama-kinetic.md` §4.10, §5.2.
- `crates/kinetic-vision/src/{lib,ops,preprocess}.rs`.
- `crates/kinetic-vision/examples/encode.rs`.
- `docs/src/models.md`.

### Commandes de mesure

```sh
cargo run --release -p kinetic-vision --example encode -- mmproj.gguf rainbow:512
llama-mtmd-debug --image rainbow -n 512 -m mmproj.gguf
target/release/kinetic serve MODEL.gguf --mmproj mmproj.gguf
```

### Livrables 30/60/90

- **30 jours** : 3 projecteurs prioritaires.
- **60 jours** : 10 projecteurs, tests de parité.
- **90 jours** : audio ou OCR, documentation.

### Critères de succès

- Embeddings proches de llama.cpp.
- Réponses cohérentes.
- Pas de régression texte.

### Interactions

- **Modèles** : architecture.
- **GPU AMD** : noyaux vision.
- **QA** : parité.

### Profil / sourcing

Vision/audio appliquée. Rare.

### Signaux d'alerte

- Ne connaît que des pipelines Python.
- Ignore le preprocessing.
- Ne compare pas à llama.cpp.

---

## Fiche 10 — Ingénieur build/release/bindings

**Missions détaillées** : M13 (build/release/bindings), M14 (sécurité, en
support), M15 (documentation/DX).
Voir [`missions.md`](missions.md).

**Priorité** : P2 — distribution.
**Rattachement** : équipe plateforme.
**Mission** : rendre le produit installable et intégrable.

### Contexte

- 10/200 fonctions C.
- Bindings Python partiels.
- Images Docker et packaging à finaliser.
- CI multi-OS existante.

### Responsabilités

- Compléter l'ABI C.
- Générer les bindings Python/Node/Go.
- Finaliser Docker, winget/scoop, archives.
- Améliorer la CI release.
- Réduire la taille et le temps de build.
- Maintenir la documentation de build.

### Compétences requises

- Cargo, cross-compilation, toolchains CUDA/HIP.
- Docker, GitHub Actions, signature de code.
- maturin/PyO3, ABI C, semver.
- mdBook.

### Périmètre d'écriture

- `.github/workflows/**`
- `docker/**`, `packaging/**`
- `bindings/**`
- `crates/kinetic-capi/**`
- `docs/src/{building,bindings}.md`

### Fichiers de contexte

- `docs/src/building.md`, `docs/src/bindings.md`.
- `.github/workflows/*.yml`.
- `crates/kinetic-capi/**`.
- `docker/Dockerfile*`.
- `packaging/**`.

### Commandes de mesure

```sh
cargo build --release --locked --workspace --all-features
cargo test --release --locked --workspace
cd bindings/python && maturin build --release --locked
docker build -f docker/Dockerfile -t kinetic .
```

### Livrables 30/60/90

- **30 jours** : audit ABI, build reproductible, taille binaire.
- **60 jours** : bindings générés, Docker, installateurs.
- **90 jours** : CI release, signature, documentation.

### Critères de succès

- Installation réussie sur Windows/Linux/macOS.
- ABI testée.
- Build reproductible.
- Documentation à jour.

### Interactions

- **Serveur/API** : ABI.
- **QA** : CI.
- **Rust backend** : features, build.

### Profil / sourcing

Release engineering, packaging. Moyenne rareté.

### Signaux d'alerte

- Ne connaît pas les builds cross-platform.
- Ignore les toolchains CUDA/HIP.
- Ne teste pas les artefacts.
