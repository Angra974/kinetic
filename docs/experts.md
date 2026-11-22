# Experts pour l'architecture et la performance de Kinetic

> **Statut** : analyse, 22 novembre 2026.
> **Snapshot** : `main` du fork `Angra974/kinetic`, complété par `PLAN.md` et
> `llama-kinetic.md`.
> **Portée** : le fork local (Ryzen 7 7800X3D, RX 9070 XT `gfx1201`, Vulkan
> RADV, ROCm 7.2), pas le plan CUDA de l'amont.
> **But** : déterminer les profils d'experts, leurs compétences, leur périmètre
> d'écriture, leurs fichiers de contexte et leurs commandes de mesure pour
> améliorer l'architecture et la performance de Kinetic.
> **Placement** : document interne, hors du mdBook publié (`docs/src/`).

---

## 0. Décision structurante : un seul backend complet par API

Le dépôt a aujourd'hui **deux couches par API GPU** :

- CUDA est éclaté entre `kinetic-cuda` (driver, noyaux, lanceurs) et
  `kinetic-models/src/cuda` (orchestration du modèle) ;
- Vulkan est éclaté entre `kinetic-vulkan` (loader, shaders, noyaux) et
  `kinetic-models/src/vulkan` (orchestration du modèle).

La cible est différente et plus simple :

> **Un seul backend complet par API GPU.** Tout CUDA vit dans `kinetic-cuda` :
> driver, noyaux, layouts, graphes, mémoire, exécution du modèle, capacités.
> Tout Vulkan vit dans `kinetic-vulkan`. Le CPU reste le chemin de référence :
> ses noyaux dans `kinetic-cpu`, son exécution générique dans `kinetic-models`.
> `kinetic-models` ne contient plus de code CUDA/Vulkan : il décrit les
> architectures, les blocs et les invariants, et expose une interface
> d'exécution **au niveau modèle** (charger, préfill, décoder, état de
> séquence, capacités), pas au niveau noyau.

Consolider ne signifie pas copier CUDA dans Vulkan. Chaque backend doit
offrir les mêmes **classes** de fonctionnalités — graphes/replay, attention
flash, MoE groupé, KV quantifié, multi-device — avec les mécanismes de son
API. La parité se juge en fonctionnalités, en exactitude et en mesures, pas
en lignes de code.

Les autres API (HIP, SYCL, WebGPU) sont des **backends de compatibilité** :
chacun est complet pour son API, mais n'est ajouté que lorsqu'une machine et
une CI peuvent le valider. Metal est un chemin **natif** du crate
`kinetic-vulkan` (feature `metal`, shaders Metal générés depuis WGSL), pas un
backend séparé et pas MoltenVK. L'utilisateur n'en charge qu'un seul à
l'exécution, selon sa carte. On fournit le socle (trait, capacités, tests de
conformance, modèle de build), pas une implémentation complète et
invérifiable de chaque API.

### 0.1 État actuel vs cible

| Aujourd'hui | Cible |
|---|---|
| CUDA : `kinetic-cuda` + `kinetic-models/src/cuda` | `kinetic-cuda` contient tout CUDA |
| Vulkan : `kinetic-vulkan` + `kinetic-models/src/vulkan` | `kinetic-vulkan` contient tout Vulkan |
| `kinetic-models` a des features `cuda` / `vulkan` et des modules backend | `kinetic-models` sans feature backend, expose un trait `ModelRunner` |
| `kinetic-server` a `cuda.rs`, `vulkan.rs`, `cpu.rs` | `kinetic-server` choisit un backend via une factory |
| Ajouter HIP = nouveau crate + nouveau module dans `kinetic-models` | Ajouter HIP = nouveau crate qui implémente `ModelRunner` |
| Les tests modèle connaissent les backends | Les tests modèle utilisent le runner CPU ou un mock |

### 0.2 Où vivent les traits

Deux couches, pas une :

- **`kinetic-runtime`** (à créer) : découverte des devices, registre/factory
  de backends, capacités, budget mémoire, ordonnanceur, pool KV, métriques.
  Il ne connaît **pas** les types de modèle.
- **`kinetic-models`** : architectures, blocs, modèle générique, trait
  `ModelRunner` au niveau modèle, runner CPU de référence.
- **`kinetic-cuda` / `kinetic-vulkan` / `kinetic-hip`** : implémentent
  `ModelRunner` et possèdent leurs noyaux, graphes et politiques de mémoire.
- **`kinetic-server`** : garde le trait `Backend` actuel (requête,
  `reuses`, `run_ahead`, `settle_some`) au-dessus de `ModelRunner`.

Cette séparation évite un cycle de dépendances et évite de transformer
`kinetic-runtime` en compilateur de graphes. Le trait `ModelRunner` reste
petit : charger, préfill, décoder, état de séquence, capacités. Chaque
backend garde la liberté de fusionner, grapher et ordonnancer ses noyaux
comme il veut.

**Point à trancher avant M06 (WP0)** : ce document place à la fois
l'ordonnanceur dans `kinetic-runtime` et le trait `ModelRunner` dans
`kinetic-models`. Pour éviter un cycle de dépendances, choisir :

- **Option A (recommandée)** : `kinetic-runtime` définit `ModelRunner`,
  `kinetic-models` implémente le runner CPU de référence, les backends
  implémentent le trait, l'ordonnanceur runtime appelle le trait. La factory
  qui connaît les crates concrets vit dans `kinetic-server`.
- **Option B** : `kinetic-models` définit `ModelRunner`, l'ordonnanceur vit
  dans `kinetic-server` (ou un crate `kinetic-scheduler`), `kinetic-runtime`
  se limite aux devices, capacités, mémoire et pool KV.

La décision doit être écrite dans un ADR avant le déplacement des exécuteurs.

### 0.3 Migration incrémentale

1. Créer `kinetic-runtime` avec les types de devices, capacités, budget
   mémoire et factory. Aucun changement de comportement.
2. Extraire dans `kinetic-models` un trait `ModelRunner` au niveau modèle et
   exposer les descripteurs nécessaires aux backends. Le runner CPU existant
   devient l'implémentation de référence.
3. Déplacer `kinetic-models/src/cuda` vers `kinetic-cuda/src/runner` et
   `kinetic-models/src/vulkan` vers `kinetic-vulkan/src/runner`. Ce n'est pas
   un simple `git mv` : il faut passer par le trait et les descripteurs
   exposés, et conserver les tests à chaque étape.
4. Retirer les features `cuda` / `vulkan` de `kinetic-models` ; il ne dépend
   plus des crates backend.
5. Ajouter une factory de backend dans `kinetic-server` / `kinetic-cli` :
   `auto | cpu | cuda | vulkan | hip`, un seul backend chargé.
6. Ajouter un backend de test (CPU ou mock) pour tester `kinetic-models`
   sans GPU.
7. Seulement ensuite, optimiser. Le refactor d'abord ; la performance vient
   après, avec les mêmes gates d'exactitude.

### 0.4 Contrat de compatibilité d'un backend

Un backend n'est « fourni » que s'il respecte ce contrat :

- **Découverte** : `probe()` liste les devices et leurs capacités réelles.
- **Capacités** : types de poids, formats KV, batch max, async, multi-device,
  mémoire disponible. Le serveur refuse clairement ce qui n'est pas supporté.
- **Conformance** : mêmes tokens que le runner CPU sur le corpus de contrôle,
  aux tolérances près, et mêmes gates d'identité.
- **Mesure** : `kinetic bench` et le protocole A/B fonctionnent sur ce
  backend, avec un seuil de régression.
- **Isolement** : un seul backend est chargé à l'exécution ; les autres ne
  coûtent ni mémoire ni temps de démarrage.
- **CI** : au moins un runner ou une machine de référence valide le backend
  avant qu'il ne soit activé par défaut.

Tant qu'un backend ne passe pas ce contrat, il reste **opt-in** et documenté
comme expérimental. C'est ainsi qu'on fournit les bases de HIP/SYCL/WebGPU
sans promettre une parité invérifiable.

### 0.5 Ce que cela change pour les experts

- **GPU AMD** travaille dans `kinetic-vulkan` et `kinetic-hip`, pas dans
  `kinetic-models`.
- **Runtime** travaille dans `kinetic-runtime` et `kinetic-server`, pas dans
  les crates backend.
- **Rust backend** pilote la migration et garde les seams `Family`,
  `Target`, `Backend`, `Projector`.
- **Quantification** travaille dans `kinetic-quant` et les noyaux, avec une
  seule définition de référence.
- **Intégration modèles** travaille dans `kinetic-models/src/arch` et
  `blocks`, sans code backend.
- **QA** teste chaque backend séparément avec le même corpus et les mêmes
  gates.

### 0.6 Non-objectifs

- Pas de compilateur de graphes généraliste (MLIR ou équivalent).
- Pas d'ABI de plugins dynamiques au départ : les features Cargo suffisent.
- Pas de noyau « universel » qui serait lent partout : chaque backend
  optimise son chemin.
- Pas de backend CUDA prioritaire sur cette machine : pas de GPU NVIDIA,
  donc pas de mesure locale.

---

## 1. Diagnostic

### 1.1 Ce que le dépôt fait bien

- Architecture en couches claire : `kinetic-gguf` → `kinetic-quant` →
  `kinetic-cpu` → `kinetic-models` (`arch/`, `blocks/`) → exécuteurs →
  `kinetic-server` → CLI/bindings.
- Registre `Family` : une architecture = un fichier + une ligne, pas de
  `match` central.
- Contrat `Target` pour le spéculatif : RAD-1, rollback, arbre de
  vérification, exactitude.
- Culture de l'exactitude : référence f64, batch-invariance, égalité passe
  mono-token / passe prompt, tests de non-régression.
- Différenciateurs réels : RAD-1, Warm Start, Zero Wait, Shortlist, Convoy,
  Flywheel, `.ktc` signé.
- Discipline : zéro `TODO/FIXME`, doc comments, CI Linux/Windows/macOS,
  `clippy -D warnings`, ~517 tests.

### 1.2 Les goulets mesurés

| Goulet | Preuve (22 nov. 2026) | Expert |
|---|---|---|
| Préfill Vulkan | 261,9 t/s contre 1154,2 pour llama.cpp Vulkan sur un dense 27B IQ3_XXS, soit **4,4×** | GPU AMD |
| MoE Vulkan | 55,7 contre 181,9 t/s en décodage, 489 contre 2637 t/s en préfill, soit **3,3× / 5,4×** | GPU AMD + runtime |
| Soumission synchrone | `vkWaitForFences` après chaque soumission, pas de timeline semaphores | GPU AMD |
| Cache de pipelines | `create_compute_pipelines(vk::PipelineCache::null(), …)` | GPU AMD |
| KV Vulkan non quantifié | `KvCache` Vulkan en f16 ; CUDA a déjà f16/q8_0 ; CPU en f32 ; pas de paged KV | GPU AMD + quantification |
| Concurrence | `Engine.free` = un backend par requête ; `Convoy` mutualise des passes mais ne planifie pas | Runtime |
| Duplication backend | `kinetic-models/src/cuda` ≈ 4,5 k LOC + `kinetic-cuda` ≈ 8,5 k LOC ; `vulkan` ≈ 2,2 k + `kinetic-vulkan` ≈ 5,9 k | Rust backend |
| Quantification | 5 cibles contre ~30 ; pas d'imatrix, pas d'IQ encode, pas de NVFP4/FP8 runtime | Quantification |
| Couverture | 21/180 architectures, 13/68 prétokenizers, ~15/45 routes, 2/73 projecteurs | Modèles / tokenizers / serveur |
| Mesure | protocole excellent mais manuel ; pas de CI GPU, pas de `THEORY.md`, pas de ledger automatisé | QA/perf |

### 1.3 Les goulets structurels

1. **La duplication backend** : ajouter HIP/SYCL/WebGPU va multiplier les
   exécuteurs si `kinetic-models` continue de contenir du code CUDA/Vulkan.
2. **Le runtime** : le serveur est conçu pour N backends indépendants, pas
   pour un ordonnanceur token-level avec KV paginé, chunked prefill global,
   priorité et annulation.
3. **Le préfill** : `gemv_many.wgsl` n'est pas un GEMM tuilé ; `PROMPT_BATCH`
   vaut 128 et le chemin « many tokens » garde 32 tokens en vol. C'est le
   seul écart de performance mesuré sur la machine locale.
4. **Le MoE** : le GEMV groupé lit les experts un slot à la fois ; le
   décodage est borné par les lectures d'experts en mémoire hôte, pas par le
   calcul GPU.
5. **La quantification** : le runtime lit les formats courants, mais
   l'encodeur, l'imatrix, NVFP4/FP8 et le KV quantifié manquent.
6. **La mesure** : les chiffres sont la crédibilité du projet ; ils doivent
   être automatisés, pas recopiés à la main.

### 1.4 Conséquence

Ce n'est pas un problème de « framework ML ». C'est un problème de
**systèmes GPU, de runtime et de numérique**. Les experts à recruter sont
donc des praticiens qui livrent du code, des noyaux et des mesures, pas des
architectes qui réécrivent.

---

## 2. Profils d'experts

### 2.1 Tableau de synthèse

| Profil | Priorité | Rareté | Mission | Périmètre principal |
|---|:-:|:-:|---|---|
| Ingénieur noyaux GPU AMD | **P0** | Très rare | Préfill Vulkan, MoE, async, KV, puis HIP | `kinetic-vulkan`, `kinetic-hip` |
| Architecte runtime d'inférence | **P0** | Très rare | Ordonnanceur, KV, concurrence, TTFT, p99 | `kinetic-runtime`, `kinetic-server` |
| Architecte Rust systèmes/backend | **P0/P1** | Rare | Un backend complet par API, dette, unsafe | workspace, `kinetic-runtime` |
| Ingénieur quantification/exactitude | **P0/P1** | Rare | NVFP4/FP8, IQ/imatrix, KV, gates | `kinetic-quant`, noyaux |
| Ingénieur performance/qualification | **P0** | Rare | CI GPU, `THEORY.md`, ledger, A/B | `scripts/`, `.github/` |
| Ingénieur intégration modèles | P1 | Moyenne | 159 architectures, MLA/DSA/SSM | `kinetic-models/src/arch` |
| Ingénieur tokenizers/templates | P1 | Moyenne | 55 prétokenizers, tool calls | `kinetic-tokenizer` |
| Ingénieur serveur/API | P1 | Courante | ~30 routes, TLS, keep-alive, `n > 1` | `kinetic-server` |
| Ingénieur multimodal | P2 | Rare | 71 projecteurs, audio, OCR | `kinetic-vision`, `kinetic-audio` |
| Ingénieur build/release/bindings | P2 | Moyenne | ABI C, Python, Docker, packaging | `bindings/`, `docker/`, CI |

### 2.2 P0 — Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)

**Mission** : faire passer le préfill Vulkan de 23 % à ≥ 50 % de llama.cpp,
aligner Vulkan sur les classes de fonctionnalités du backend CUDA lorsque
l'API le permet, puis porter les mêmes noyaux sur HIP.

**Compétences clés** :

- Vulkan compute 1.3 : descriptors, push constants, specialization constants,
  subgroups 32/64, barriers, command buffers, timeline semaphores,
  `VkPipelineCache`, `VK_KHR_cooperative_matrix`.
- WGSL / SPIR-V / naga ; RGP ; ISA AMDGPU ; RDNA 4 (wave32, LDS 64 Kio,
  pression VGPR/SGPR, occupancy, Infinity Cache).
- GEMM tuilé : tuiles en mémoire partagée, register blocking, split-K,
  swizzling, bank conflicts, épilogue fusionné, FP16/BF16/INT8 dot.
- Quantification : Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ4_XS, échelles de
  blocs, dequant-in-registers, dot int8 packé (`v_dot4_i32_i8`, `sdot4`).
- Roofline : mesurer la bande passante et les FLOPS réels, distinguer
  memory-bound et compute-bound avec des compteurs.
- Exactitude : accumulation f32, batch-invariance, égalité passe prompt /
  passe mono-token. Comprendre `Target` et RAD-1.
- Rust/ash/unsafe aux frontières des noyaux.

**Pourquoi ici** : `gemv_many.wgsl` n'est pas un GEMM tuilé ; `PROMPT_BATCH`
vaut 128 ; pas de coopmat ; `Commands::submit` attend le fence ; pas de cache
de pipelines ; KV Vulkan f16 seulement ; `moe_gemv.wgsl` lit un expert à la
fois. C'est le seul écart de performance mesuré sur la RX 9070 XT.

**Livrables 90 jours** :

1. Baseline reproductible, roofline, profil RGP de pp512.
2. GEMM tuilé Q4_K/IQ, bit-exact, ≥ 1,5×.
3. Coopmat + split-K, ≥ 2× ; soumission asynchrone ; cache de pipelines.
4. GEMM MoE groupé ; KV q8_0.
5. Portage HIP des mêmes noyaux.

**KPI** : pp512/pp2048 vs llama Vulkan, tg128 sans régression, gates
bit-exact, GB/s et TFLOPS atteints, temps de démarrage.

### 2.3 P0 — Architecte runtime d'inférence

**Mission** : faire passer le serveur de « N backends indépendants » à un
ordonnanceur token-level qui tient la concurrence, le TTFT et le p99, sans
casser RAD-1, Warm Start ni Convoy.

**Compétences clés** :

- Continuous batching, paged KV, prefix caching, chunked prefill, admission
  control, fairness, priorité, annulation, backpressure.
- Décodage spéculatif : ordonnancement draft/verify, vérification d'arbre,
  rollback, réutilisation de KV.
- Concurrence Rust : threads/async, `Mutex`/`Condvar`, channels, atomics,
  `Send`/`Sync`, scoped threads, panic safety.
- Mémoire : dimensionnement du pool KV (N × contexte), fragmentation/défrag,
  mémoire unifiée, pinned host, paging des experts MoE, watchdog OOM.
- Graph capture/replay, multi-stream, recouvrement copies/calcul.
- Observabilité : TTFT, tok/s, p50/p99, cache hit, profondeur de file.

**Pourquoi ici** : `Engine.free` est un
`Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>` : un pool de backends, une
requête par backend ; `PREFILL_CHUNK = 256` existe mais le chunking est par
requête ; `Convoy` mutualise des passes mais ne planifie pas ; le KV est
contigu par séquence, non paginé ; `MAX_CONNECTIONS = 64` ; en-tête
`Connection: close`.

**Livrables 90 jours** :

1. Design doc + prototype d'ordonnanceur token-level.
2. Pool KV N × contexte ; chunked prefill ; annulation ; métriques par token.
3. Benchmarks c4/c16/c64 avec p99 et TTFT.
4. Préservation de l'exactitude et des invariants Warm Start/Convoy.

**KPI** : débit agrégé, p99, TTFT, cache hit, zéro régression d'exactitude.

### 2.4 P0/P1 — Architecte Rust systèmes/backend

**Mission** : réduire la duplication CUDA/Vulkan, faire de chaque API un
backend complet et préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.

**Compétences clés** :

- Rust avancé : lifetimes, traits, génériques vs `dyn`, zéro-copie,
  soundness `unsafe`, `Send`/`Sync`, `Arc`/`Rc`, mmap, allocators, layouts
  cache-friendly, monomorphisation.
- Architecture : ports/adapters, abstraction backend, graphe/dispatch,
  feature flags, compilation conditionnelle, FFI.
- Performance hôte : `perf`, flamegraph, heaptrack, criterion, LLVM
  optimization remarks, LTO/PGO/BOLT, démarrage.
- Build : workspaces Cargo, cross-compilation, toolchains CUDA/HIP, CI.
- Tests : unitaires, intégration, property-based, fuzz, tests déterministes.

**Pourquoi ici** : `kinetic-models/src/cuda` ≈ 4,5 k LOC vs `vulkan` ≈ 2,2 k ;
`kinetic-cuda` ≈ 8,5 k vs `kinetic-vulkan` ≈ 5,9 k ; ajouter HIP va dupliquer
encore. Les bons seams existent déjà (`Family`, `Target`, `Backend`,
`Projector`) ; il faut les préserver. La surface `unsafe` est concentrée aux
frontières SIMD/GPU/FFI ; un audit de soundness est nécessaire.

**Livrables 90 jours** :

1. Cartographie de la duplication ; proposition d'une couche partagée
   minimale (`kinetic-runtime`).
2. Refactor d'un chemin comme preuve (FFN/attention) sans changer le
   comportement.
3. Audit `unsafe` avec commentaires `SAFETY`.
4. Amélioration du démarrage et du temps de build.
5. API prête pour 3+ backends.

**KPI** : LOC dupliquées, temps de build, démarrage, overhead hôte par token,
nombre d'`unsafe` justifiés.

### 2.5 P0/P1 — Ingénieur quantification et exactitude numérique

**Mission** : étendre la quantification et garantir que chaque gain de
vitesse reste exact.

**Compétences clés** :

- GGUF, quantization par blocs, K-quants, grilles IQ, MXFP4, NVFP4,
  FP8 E4M3, imatrix.
- Analyse numérique : référence f64, propagation d'erreur, tolérances, ordre
  d'accumulation, fp16/bf16/fp8, arrondi stochastique.
- Encodeur `quantize`, politique par tenseur, requantification, split/merge.
- Tests différentiels, property-based, fuzzing, conformance vs llama.cpp.
- Sensibilité performance : dequant-in-registers, vectorisation, layouts
  compatibles noyaux.

**Pourquoi ici** : 5 cibles de quantification contre ~30 ; pas d'imatrix ni
d'IQ encode ; NVFP4 et F8_E4M3 absents du runtime ; KV Vulkan f16 uniquement,
CUDA f16/q8_0, CPU f32. L'exactitude est le différenciateur produit : un gain
de vitesse qui change la sortie n'est pas mergeable.

**Livrables 90 jours** :

1. Runtime NVFP4/FP8.
2. IQ encode + imatrix.
3. KV q8/q4 correct.
4. Corpus de conformance, tests property/fuzz, gates d'identité.
5. Métriques qualité (KL/perplexité) par format.

**KPI** : qualité, vitesse, mémoire, zéro régression d'identité.

### 2.6 P0 — Ingénieur performance/qualification

**Mission** : rendre la mesure automatisée, reproductible et infalsifiable.

**Compétences clés** :

- Méthodologie : médianes, intervalles de confiance, bruit ±5–10 %, A/B avec
  contrôles avant/après, état thermique et fréquences, isolation.
- Harness : Python/Rust, JSON, `ledger.jsonl`, détection de régression, CI
  GPU, artefacts épinglés.
- Profilage : RGP, rocprof, Nsight, `perf`, watchdogs mémoire.
- Tests : fuzzing, soak, needle à 50 %/100 % de contexte, conformance,
  chaos (`server_abuse.py`).
- Reproductibilité : révisions de poids, commits moteur, versions de driver,
  conteneurs.

**Pourquoi ici** : le protocole existe dans `PLAN.md` et `llama-kinetic.md`
§7.10, mais il est manuel ; pas de CI GPU ; les chiffres sont la crédibilité
du projet ; les scripts existent déjà mais ne sont pas industrialisés.

**Livrables 90 jours** :

1. `THEORY.md` avec les plafonds mesurés (bande passante, FLOPS, KV).
2. `ledger.jsonl` + `configs/` + `best-config.json`.
3. A/B automatisé avec contrôles.
4. CI GPU (au moins un runner Vulkan/ROCm) avec seuil de régression.
5. Qualification de release : build propre, poids épinglés, gates.

**KPI** : taux de faux positifs/négatifs, temps de détection d'une
régression, couverture des gates.

### 2.7 P1 — Ingénieur intégration des architectures

**Mission** : couvrir les architectures manquantes par briques réutilisables,
pas par forks.

**Compétences clés** : zoo Transformer (MHA/GQA/MQA/MLA/DSA/sinks),
RoPE/iRoPE/partial/scaling, MoE top-k/shared/sigmoid, SSM/hybride (Mamba2,
RWKV, delta net), norm/activation, T5 encodeur-décodeur, têtes
embedding/rerank/TTS ; lecture GGUF ; mapping vers `blocks/` ; conformance
vs llama.cpp ; discipline « un fichier par famille ».

**Pourquoi ici** : 21/180 architectures ; les briques les plus rentables sont
MLA/DSA, les hybrides SSM et T5 ; les modèles locaux bloqués sont `qwen3vl`,
`deepseek`, `llama4`, `gemma2`, `phi3`, `command-r`, `dbrx`.

**Livrables** : lots d'architectures par brique réutilisable, tests de
conformance, documentation `models.md`.

**KPI** : modèles chargés, tokens identiques à llama.cpp, couverture.

### 2.8 P1 — Ingénieur tokenizers / templates / tool calls

**Mission** : atteindre la compatibilité des tokenizers et des templates.

**Compétences clés** : BPE, SentencePiece, WordPiece, UGM, Tekken,
prétokenizers regex, byte-level, tokens spéciaux EOT/EOM/FIM ; templates
Jinja ; formats de tool calls ; JSON Schema/GBNF ; tests différentiels vs
`llama-tokenize` et `llama-server`.

**Pourquoi ici** : 13/68 prétokenizers, 55 manquants ; templates intégrés et
formats tool-call incomplets ; l'issue #5 (tool calls dans le raisonnement)
est typique.

**Livrables** : 15 prétokenizers prioritaires, UGM, split GGUF, templates
intégrés, formats tool-call.

**KPI** : ids identiques sur un corpus de contrôle, requêtes serveur
compatibles.

### 2.9 P1 — Ingénieur serveur / API / observabilité

**Mission** : parité API, robustesse et observabilité.

**Compétences clés** : HTTP/1.1, SSE, keep-alive, chunked, timeouts,
backpressure, TLS, API keys, CORS, arrêt propre ; APIs OpenAI, llama-server,
Anthropic Messages, Ollama ; `n > 1`, `timings_per_token`, `/props`,
`/slots`, `/infill`, `/apply-template`, `/lora-adapters` ; Prometheus, logs
structurés, tracing, p99 ; sécurité (limites d'entrée, DoS, path traversal).

**Pourquoi ici** : ~15/45 routes, pas de keep-alive/TLS, 64 connexions
thread-per-connection, `docker stop` à améliorer, compatibilité clients
perfectible.

**Livrables** : routes manquantes, `n > 1`, TLS/API key, router multi-modèle,
métriques, arrêt propre.

**KPI** : conformité llama-server, overhead, stabilité sous charge.

### 2.10 P2 — Ingénieur multimodal

**Mission** : vision, audio, OCR.

**Compétences clés** : ViT/SigLIP/CLIP, projecteurs (MLP/merger/Q-Former),
preprocessing image, vidéo, audio (mel, ASR/TTS), OCR, noyaux GPU vision.

**Pourquoi ici** : 2/73 projecteurs, pas d'audio/vidéo/OCR. Impact produit
fort mais chantier XL, hors chemin critique de performance.

**Livrables** : 10 projecteurs majeurs, crate `kinetic-audio`, pipeline OCR.

**KPI** : parité des embeddings vs `llama-mtmd`, qualité des réponses.

### 2.11 P2 — Ingénieur build/release/bindings

**Mission** : rendre le produit installable et intégrable.

**Compétences clés** : Cargo, cross-compilation, toolchains CUDA/HIP, Docker,
GitHub Actions, signature de code, winget/scoop, maturin/PyO3, ABI C, semver,
mdBook.

**Pourquoi ici** : 10/200 fonctions C, bindings Python partiels, images
Docker et packaging à finaliser.

**Livrables** : ABI C complète, bindings générés, images Docker, installateurs,
CI release.

**KPI** : couverture ABI, temps/taille de build, taux d'installation réussie.

### 2.12 Mapping profils → missions

Les missions détaillées sont dans
[`docs/experts/missions.md`](experts/missions.md).

| Profil | Missions | Rôle principal |
|---|---|---|
| GPU AMD | M02, M03, M07 | Noyaux Vulkan/HIP |
| Runtime | M04, M05 | KV, ordonnanceur, concurrence |
| Rust backend | M06 | Consolidation backend |
| Numérique | M08 | Quantification, KV, exactitude |
| Performance/QA | M01 + support | Mesure, ledger, CI GPU |
| Intégration modèles | M09 | Architectures |
| Tokenizers | M10 | Tokenizers, templates, tool calls |
| Serveur/API | M11, M14 | API, observabilité, sécurité |
| Multimodal | M12 | Vision, audio, OCR |
| Build/release | M13, M14, M15 | Build, release, bindings, doc |

---

## 3. Composition et séquencement

### 3.1 Cellule « performance » minimale (5 personnes)

| Rôle | Mission | Justification |
|---|---|---|
| GPU AMD | Vulkan d'abord, HIP ensuite | Le seul écart mesuré est le préfill/MoE |
| Runtime | Ordonnanceur, KV, concurrence | Le prochain plafond après le préfill |
| Rust backend | Abstraction backend, dette | Évite de dupliquer HIP/SYCL/WebGPU |
| Numérique | Quantification + gates | Protège le différenciateur exactitude |
| Performance/QA | CI GPU, ledger, A/B | Sans mesure, pas de gain prouvé |

### 3.2 Équipe complète (9–10 personnes)

Ajouter : **intégration modèles**, **tokenizers/templates**, **serveur/API**,
**multimodal**, **build/release/bindings**.

### 3.3 Phases

| Phase | Durée | Rôles | Objectif |
|---|---|---|---|
| 0. Mesure | 2–4 sem. | QA + GPU | Baseline, `THEORY.md`, CI GPU |
| 1. Préfill | 6–10 sem. | GPU + numérique + QA | GEMM tuilé, coopmat, async, pp ≥ 2× |
| 2. MoE/HIP | 8–12 sem. | GPU + runtime + Rust | MoE groupé, squelette HIP, KV q8 |
| 3. Concurrence | 6–10 sem. | Runtime + Rust + QA | Ordonnanceur, cN, TTFT |
| 4. Couverture | continu | Modèles + tokenizers + serveur | Architectures, prétokenizers, routes |
| 5. Produit | continu | Multimodal + build | Vision/audio, bindings |

### 3.4 RACI simplifié

| Chantier | Pilote | Support | Porte de sortie |
|---|---|---|---|
| GEMM préfill Vulkan | GPU AMD | Numérique, QA | pp512 ≥ 2×, bit-exact |
| MoE Vulkan | GPU AMD | Runtime, Numérique | tg/pp, qualité |
| HIP/ROCm | GPU AMD/HIP | Rust backend | vs llama ROCm0 |
| Ordonnanceur/KV | Runtime | Rust backend, QA | cN, p99, TTFT |
| Quantification | Numérique | GPU/CPU kernel | Identité |
| Architectures | Intégration modèles | Numérique, QA | Conformance |
| Tokenizers | Tokenizers | Serveur | Ids identiques |
| API serveur | Serveur | QA | Conformance |
| CI perf | QA | Build/release | Régression |
| Bindings | Build/release | Serveur | Tests ABI |

---

## 4. Évaluation et recrutement

### 4.1 Exercices

Éviter les entretiens théoriques. Donner un exercice tiré du dépôt, avec
mesure et gate d'exactitude.

| Rôle | Exercice |
|---|---|
| GPU AMD | Implémenter un GEMM tuilé Q4_K dans un nouveau shader ; le profiler avec RGP ; battre l'existant de ≥ 1,5× ; passer le test bit-exact. |
| Runtime | Concevoir un ordonnanceur continuous batching pour `Engine`/`Sequence` ; expliquer le rollback KV, l'annulation, la fairness ; identifier les impacts sur RAD-1/Warm Start. |
| Rust backend | Refactoriser un chemin dupliqué CUDA/Vulkan derrière une abstraction minimale, sans changer les tests ; expliquer `Send`/`Sync` et les `unsafe`. |
| Numérique | Implémenter un format de bloc de la référence f64 au noyau, avec tests différentiels et mesure. |
| Intégration modèles | Ajouter une famille à partir d'un GGUF inconnu, en un fichier + une ligne, et la comparer à llama.cpp. |
| Tokenizers | Faire passer un corpus de contrôle avec les mêmes ids que `llama-tokenize`, tokens spéciaux compris. |
| QA | Concevoir un A/B qui détecte une régression de 5 % avec un bruit de ±8 % ; calculer la taille d'échantillon et les contrôles. |

### 4.2 Signaux d'alerte

- Uniquement PyTorch/TensorFlow, aucune notion de noyau ou de mémoire.
- CUDA uniquement, incapable de lire du WGSL/SPIR-V ou de raisonner RDNA.
- « Il suffit de réécrire en X » sans mesure ni migration incrémentale.
- Benchmarks sans contrôles, sans médiane, sans état machine.
- Aucune sensibilité à l'exactitude : « on verra plus tard ».
- Impossible de lire un profil GPU ou d'expliquer memory-bound vs
  compute-bound.

### 4.3 Rareté et sourcing

| Profil | Rareté | Où chercher |
|---|---|---|
| GPU AMD | Très rare | HPC, ROCm, moteurs de jeu, compute Vulkan, équipes de noyaux ML |
| Runtime | Très rare | vLLM, SGLang, TensorRT-LLM, llama.cpp, ONNX Runtime |
| Rust backend | Rare | Systèmes, bases de données, moteurs de jeu, embedded |
| Numérique | Rare | Compression ML, co-design matériel/logiciel, calcul numérique |
| QA/perf | Rare | Performance engineering, SRE, équipes de benchmark |
| Intégration modèles | Moyenne | Communauté llama.cpp/transformers |
| Tokenizers | Moyenne | Infrastructure NLP |
| Serveur/API | Courante | Backend/API engineering |
| Multimodal | Rare | Vision/audio appliquée |
| Build/release | Moyenne | Release engineering, packaging |

---

## 5. Commandes de mesure communes

### 5.1 Build et tests

```sh
cargo build --release --locked -p kinetic-cli --features vulkan
cargo test --release --locked --workspace
cargo test --release --locked -p kinetic-vulkan
cargo test --release --locked -p kinetic-models
cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
```

### 5.2 Inventaire

```sh
target/release/kinetic doctor
target/release/kinetic info MODEL.gguf
```

### 5.3 Bench Vulkan

```sh
target/release/kinetic bench MODEL.gguf --device vulkan --json -r 6
llama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6
```

`kinetic bench` mesure `pp512`, `tg128` et `edit256`. Pour un préfill plus
long, utiliser `--prompt 2048`. Les chiffres se comparent toujours sur le
même fichier, le même backend et le même état machine.

### 5.4 Protocole et conformance

```sh
python scripts/bench_protocol.py MODEL.gguf --llama-bin /path/to/llama-bench
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device cpu
```

`conformance.py` ne connaît aujourd'hui que `cpu` et `cuda`. L'extension à
`vulkan` est un livrable QA. En attendant, lancer
`kinetic serve --device vulkan` et comparer les tokens manuellement avec
`llama-server`.

### 5.5 Serveur

```sh
python scripts/llama_server_compat.py --model MODEL.gguf \
  --llama-server /path/to/llama-server --kinetic target/release/kinetic
python scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091
python scripts/same_answer_demo.py --model MODEL.gguf \
  --kinetic target/release/kinetic --kinetic-args "--device vulkan"
```

### 5.6 Vision

```sh
cargo run --release -p kinetic-vision --example encode -- mmproj.gguf rainbow:512
llama-mtmd-debug --image rainbow -n 512 -m mmproj.gguf
```

### 5.7 Précautions

- Un seul job GPU à la fois ; vérifier la machine idle (`rocm-smi`,
  `radeontop`, aucun processus `kinetic`/`llama`).
- Médiane de ≥ 6 échantillons ; contrôles avant et après chaque candidat.
- Accepter seulement si le candidat bat les deux contrôles sur sa métrique et
  ne régresse rien de plus de ~3 %.
- Consigner chaque essai dans `PLAN.md` avec commande, commit, date, état
  machine et verdict.
- Ne jamais forcer un serveur en pleine passe : annuler les requêtes, laisser
  drainer, puis arrêter avec un timeout.

---

## 6. Ce qu'il ne faut pas faire

- **Ne pas recruter un « expert LLM » générique en premier.** Le goulot est
  GPU/runtime, pas la théorie des transformers.
- **Ne pas commencer par HIP.** Le décodage Vulkan est déjà à 95 % ; le
  préfill est à 23 %. Le GEMM Vulkan est le meilleur premier chantier.
- **Ne pas recruter un expert CUDA-only pour ce fork.** La machine n'a pas de
  GPU NVIDIA ; il ne pourrait rien mesurer. CUDA est utile en amont/CI, pas
  en P0 local.
- **Ne pas lancer un chantier multimodal ou bindings avant la performance.**
  C'est XL et hors chemin critique.
- **Ne pas introduire un compilateur de graphes généraliste.** Une couche de
  dispatch/exécution minimale suffit.
- **Ne pas casser la culture d'exactitude.** Un gain non prouvé, ou qui change
  la sortie, doit être rejeté.

---

## 7. Documents associés

- [`docs/experts/README.md`](experts/README.md) : index et ordre de lecture.
- [`docs/experts/fiches-de-poste.md`](experts/fiches-de-poste.md) : fiches de
  poste détaillées pour les 10 profils.
- [`docs/experts/missions.md`](experts/missions.md) : 15 missions
  exécutables, avec work packages, dépendances, livrables et critères
  d'acceptation.
- [`docs/experts/prompts-agents.md`](experts/prompts-agents.md) : prompts
  d'agents spécialisés, avec périmètre d'écriture, fichiers de contexte,
  commandes de mesure et garde-fous.

---

## 8. Recommandation finale

Si je devais n'embaucher que **quatre personnes** pour ce dépôt :

1. **Ingénieur noyaux GPU AMD** — le plus gros gain mesuré, le plus rare.
2. **Architecte runtime d'inférence** — le prochain plafond de scalabilité.
3. **Ingénieur performance/qualification** — pour que les deux premiers
   prouvent leurs gains.
4. **Ingénieur quantification/exactitude** — pour que ces gains restent
   conformes à la promesse produit.

Si un seul profil : **GPU AMD avec une forte sensibilité
quantification/exactitude**. C'est le seul qui attaque directement l'écart de
4,4× en préfill et de 3,3–5,4× en MoE.

Si deux : ajouter **runtime**. Si trois : ajouter **QA/perf**. Si quatre :
ajouter **numérique**.

Le reste — architectures, tokenizers, serveur, multimodal, bindings — peut
être parallélisé, communautarisé ou recruté ensuite. Le projet a déjà les bons
seams (`Family`, `Target`, `Backend`, `Projector`) et une documentation
d'onboarding solide (`adding-a-family.md`, `contributing.md`) : un bon expert
doit pouvoir livrer une famille ou un noyau en quelques jours, pas en
quelques semaines.
