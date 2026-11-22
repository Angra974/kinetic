# Missions des experts Kinetic

> **Statut** : catalogue de missions, 22 novembre 2026.
> **But** : transformer les 10 profils de `docs/experts.md` en missions
> exécutables, avec objectif, périmètre, work packages, livrables, critères
> d'acceptation, dépendances et protocole de mesure.
>
> **Règle** : une mission = un agent = un périmètre d'écriture. Ne pas lancer
> deux agents sur les mêmes fichiers. M01 passe avant toutes les autres.

---

## 0. Comment utiliser ce document

1. Lire `docs/experts.md` pour l'analyse et l'architecture cible.
2. Choisir une mission ci-dessous.
3. Lire la fiche de poste correspondante dans
   `docs/experts/fiches-de-poste.md`.
4. Donner à l'agent le prompt correspondant dans
   `docs/experts/prompts-agents.md`, en y ajoutant l'ID de mission.
5. L'agent commence par lire son périmètre, mesure la baseline, propose un
   seul changement, mesure, consigne dans `PLAN.md` et le ledger, puis
   rapporte.

### 0.1 Règles communes

- Un seul changement par essai ; une configuration par essai.
- Baseline, candidat, puis contrôle ; médiane de ≥ 6 échantillons.
- Accepter seulement si le candidat bat les deux contrôles et ne régresse
  rien de plus de ~3 %.
- Toute optimisation doit préserver l'égalité batch / mono-token et les
  gates d'identité.
- Consigner les échecs et les résultats négatifs.
- Ne pas copier de code depuis llama.cpp ou un autre moteur.
- Ne pas modifier les fichiers hors du périmètre de la mission.
- Si `ledger.jsonl` ou `configs/` n'existent pas encore, créer une entrée
  minimale au format documenté par M01 ; ne pas restructurer.

### 0.2 Statuts

| Statut | Sens |
|---|---|
| `à lancer` | Mission prête, aucun agent actif |
| `en cours` | Un agent travaille dessus |
| `bloquée` | Dépend d'une décision ou d'une mission |
| `terminée` | Critères d'acceptation remplis et consignés |
| `abandonnée` | Résultat négatif documenté |

---

## 1. Index des missions

| ID | Mission | Priorité | Pilote | Support | Dépend de | Débloque | Durée |
|---|---|---|---|---|---|---|---|
| M01 | Baseline, théorie et harnais de mesure | P0 | Performance/QA | GPU, Runtime | — | Toutes | 2–4 sem. |
| M02 | Préfill Vulkan : GEMM tuilé et soumission asynchrone | P0 | GPU AMD | Numérique, QA | M01 | M03, M07 | 6–10 sem. |
| M03 | MoE Vulkan : GEMM groupé et pagination des experts | P0 | GPU AMD | Runtime, Numérique | M01, M02 | M07 | 6–8 sem. |
| M04 | KV cache : audit, pool unifié, pagination et partage | P0 | Runtime | GPU, Numérique, QA | M01 | M05, M08 | 6–10 sem. |
| M05 | Ordonnanceur runtime et concurrence | P0 | Runtime | Rust backend, QA | M01, M04 | M11 | 6–10 sem. |
| M06 | Consolidation backend : un seul backend complet par API | P0/P1 | Rust backend | GPU, Runtime, QA | M01 | M07, M13 | 8–12 sem. |
| M07 | Backend HIP/ROCm : squelette et noyaux de base | P1 | GPU AMD | Rust backend, QA | M06, M02 | — | 8–12 sem. |
| M08 | Quantification et formats KV | P0/P1 | Numérique | GPU, Runtime | M01, M04, M02 | — | 8–12 sem. |
| M09 | Couverture des architectures | P1 | Intégration modèles | Numérique, QA | M01 | M12 | continu |
| M10 | Tokenizers, templates et tool calls | P1 | Tokenizers | Serveur, QA | M01 | M11 | 6–10 sem. |
| M11 | Serveur, API et observabilité | P1 | Serveur/API | Runtime, Tokenizers, QA | M01, M05, M10 | M14 | 8–12 sem. |
| M12 | Multimodal : vision, audio, OCR | P2 | Multimodal | Modèles, GPU, QA | M01, M09 | — | XL |
| M13 | Build, release et bindings | P2 | Build/release | Serveur, Rust backend | M06 | M14 | 8–12 sem. |
| M14 | Sécurité | P1/P2 | Serveur/API | Rust backend, Build/release | M11, M13 | — | 4–6 sem. |
| M15 | Documentation et expérience développeur | P2 | Build/release | Tous | continu | — | continu |

### 1.1 Séquencement

| Phase | Durée | Missions | Porte de sortie |
|---|---|---|---|
| 0. Mesure | 2–4 sem. | M01 | Baseline, `THEORY.md`, ledger, CI GPU |
| 1. Performance | 6–10 sem. | M02, M04, M06 WP0 | pp ≥ 2×, KV instrumenté, design backend validé |
| 2. MoE et concurrence | 6–10 sem. | M03, M05, M06, M08 | MoE, cN, KV q8 Vulkan |
| 3. HIP et couverture | continu | M07, M09–M13 | HIP opt-in, familles, API, produit |
| 4. Consolidation | continu | M14, M15 | Sécurité, doc, release |

### 1.2 Chemin critique

- Performance : **M01 → M02 → M03 → M07**.
- Concurrence : **M01 → M04 → M05 → M08**.
- Architecture : **M01 → M06 → M07**.

M06 peut démarrer en parallèle de M02/M04 pour le design (WP0), mais le
déplacement des exécuteurs doit se faire dans une branche ou un worktree
dédié, après stabilisation des noyaux, pour éviter les conflits de fichiers.

---

## 2. Missions détaillées

## M01 — Baseline, théorie et harnais de mesure

| Champ | Valeur |
|---|---|
| Priorité | P0 |
| Pilote | Performance/QA (Agent 5) |
| Support | GPU AMD, Runtime |
| Dépend de | — |
| Débloque | Toutes les autres missions |
| Durée | 2–4 semaines |
| Statut | à lancer |

### Objectif

Rendre la mesure automatisée, reproductible et infalsifiable : plafonds
théoriques, baseline, ledger, configurations, protocole A/B, CI GPU,
conformance Vulkan, tests de contexte long et soak.

### État actuel

- Le protocole existe dans `PLAN.md` et `llama-kinetic.md` §7.10, mais il est
  manuel.
- Pas de CI GPU, pas de `THEORY.md`, pas de `ledger.jsonl`, pas de
  `configs/`, pas de `best-config.json`.
- `scripts/bench_protocol.py`, `bench_vs_llamacpp.py`, `conformance.py`,
  `server_abuse.py`, `same_answer_demo.py` existent.
- `conformance.py` ne connaît que `cpu` et `cuda` ; l'extension Vulkan est à
  faire.

### Périmètre

**Inclus** :

- `THEORY.md` : bande passante, FLOPS, KV, TTFT, plafonds par modèle.
- `ledger.jsonl` : une entrée par essai, avec commande, commit, date, état
  machine, résultat, verdict.
- `configs/` : configurations complètes et reproductibles.
- `best-config.json` : meilleure configuration connue par modèle/backend.
- Protocole A/B : contrôle, candidat, contrôle.
- CI GPU self-hosted (Vulkan/ROCm) avec seuil de régression.
- Extension de `conformance.py` à Vulkan.
- Test needle 50 %/100 % de contexte.
- Soak concurrent et watchdog mémoire.
- Dashboard de régression et rapport type.

**Exclu** :

- Modifier les noyaux ou le runtime pour améliorer un chiffre.
- Publier un chiffre sans commande, commit, date et état machine.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Écrire `THEORY.md` : bande passante mesurée, FLOPS, KV, TTFT, plafond par modèle | Document + formule de plafond |
| WP2 | Définir le schéma `ledger.jsonl` et `configs/` ; créer le squelette | Fichiers + exemple d'entrée |
| WP3 | Automatiser le protocole A/B avec contrôles avant/après | `scripts/ab_protocol.py` ou extension |
| WP4 | Étendre `conformance.py` à `--device vulkan` | Corpus + rapport de conformance |
| WP5 | Ajouter needle 50 %/100 %, soak concurrent, watchdog mémoire | Scripts + seuils |
| WP6 | Ajouter une CI GPU avec seuil de régression | Workflow + artefacts |
| WP7 | Dashboard et rapport type de régression | Rapport + historique |

### Livrables

- `THEORY.md`.
- `ledger.jsonl` et `configs/` avec au moins 10 essais de baseline.
- `best-config.json`.
- `scripts/` : A/B, needle, soak, watchdog.
- CI GPU opérationnelle.
- Rapport de baseline pour le dense 27B et le MoE 30B A3B.

### Critères d'acceptation

- Un candidat à +5 % est distingué du bruit avec un protocole reproductible.
- Une régression de 5 % est détectée automatiquement.
- Chaque chiffre publié a une commande, un commit, une date et un état
  machine.
- Les résultats négatifs sont conservés.
- `conformance.py` compare Vulkan au runner CPU sur le corpus de contrôle.

### Protocole de mesure

```sh
target/release/kinetic doctor
target/release/kinetic bench MODEL.gguf --device vulkan --json -r 6
llama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6
python scripts/bench_protocol.py MODEL.gguf --llama-bin /path/to/llama-bench
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device vulkan
python scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091
python scripts/same_answer_demo.py --model MODEL.gguf --kinetic target/release/kinetic
rocm-smi --showuse --showtemp --showmemuse
```

### Dépendances et handoffs

- Fournit le ledger et les configs à M02–M15.
- Reçoit des mesures de M02/M03/M04/M05.
- Valide les gates de M08/M11.

### Risques

| Risque | Mitigation |
|---|---|
| Bruit GPU/thermique | Machine idle, répétitions, contrôles, médiane |
| Driver ou ROCm instable | Épingler version, redémarrer, documenter |
| Mesure non représentative | Même fichier, même backend, même état machine |
| Dérive du ledger | Schéma strict, une entrée par essai, revue |

### Definition of done

- `THEORY.md` relu et accepté.
- Ledger rempli avec la baseline.
- CI GPU verte.
- `conformance.py` Vulkan fonctionne.
- Rapport de baseline publié dans `PLAN.md`.

---

## M02 — Préfill Vulkan : GEMM tuilé et soumission asynchrone

| Champ | Valeur |
|---|---|
| Priorité | P0 |
| Pilote | GPU AMD (Agent 1) |
| Support | Numérique (Agent 4), QA (Agent 5) |
| Dépend de | M01 |
| Débloque | M03, M07 |
| Durée | 6–10 semaines |
| Statut | à lancer |

### Objectif

Combler l'écart de préfill Vulkan : passer de 261,9 t/s à ≥ 2× la baseline
Kinetic sur pp512/pp2048, sans régression > 3 % sur tg128, avec exactitude
bit-exacte.

### État actuel

- Dense 27B IQ3_XXS : 261,9 t/s en préfill contre 1154,2 pour llama.cpp
  Vulkan, soit 4,4×.
- `gemv_many.wgsl` garde 32 tokens en vol ; ce n'est pas un GEMM tuilé.
- `Commands::submit` soumet puis attend `vkWaitForFences` ; pas de timeline
  semaphores.
- Pas de `VkPipelineCache` persisté.
- Le décodage dense est à 95 % de llama.cpp (38,4 contre 40,1 t/s).

### Périmètre

**Inclus** :

- `crates/kinetic-vulkan/**` : shaders, pipelines, commandes, GEMV, attention.
- Tests d'exactitude et de non-régression dans `kinetic-vulkan` et
  `kinetic-models`.
- `PLAN.md`, `ledger.jsonl`, `configs/` pour les essais.

**Exclu** :

- Déplacer les exécuteurs de `kinetic-models` (M06).
- Modifier les architectures ou les blocs.
- Modifier CUDA.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Profiler RGP sur pp512 ; roofline ; memory-bound vs compute-bound | Rapport de profil |
| WP2 | GEMM tuilé Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ4_XS | Shader + tests bit-exact |
| WP3 | Coopmat (`VK_KHR_cooperative_matrix`), split-K, tuilage LDS | Variantes mesurées |
| WP4 | Soumission asynchrone avec timeline semaphores | Commandes + tests |
| WP5 | `VkPipelineCache` persisté ; temps de démarrage | Cache + mesure |
| WP6 | Tests batch-invariance et égalité passe prompt / mono-token | Tests verts |

### Livrables

- GEMM de préfill tuilé pour les formats prioritaires.
- Variante coopmat et split-K.
- Soumission asynchrone sans casser les dépendances.
- Cache de pipelines persisté.
- Rapport de bench avant/après avec RGP.

### Critères d'acceptation

- pp512/pp2048 ≥ 2× la baseline Kinetic.
- tg128 sans régression > 3 %.
- Tous les tests d'identité verts.
- `kinetic bench` dit `edit_identical: true`.
- Chaque essai est dans `PLAN.md` avec commande, commit, date et verdict.

### Protocole de mesure

```sh
cargo build --release --locked -p kinetic-cli --features vulkan
cargo test --release --locked -p kinetic-vulkan
cargo test --release --locked -p kinetic-models
target/release/kinetic bench MODEL.gguf --device vulkan --json -r 6
llama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6
RGP capture sur pp512
rocm-smi --showuse --showtemp --showmemuse
```

### Dépendances et handoffs

- Reçoit le ledger et la baseline de M01.
- Fournit les noyaux de base à M03 et M07.
- Coordonne avec M06 : ne pas déplacer les exécuteurs pendant M02.

### Risques

| Risque | Mitigation |
|---|---|
| Coopmat mal supporté par RADV/RDNA 4 | Fallback tuilage classique, mesurer |
| Exactitude cassée par le tuilage | Test bit-exact avant/après |
| Régression décodage | Contrôle tg128 à chaque essai |
| Conflit avec M06 | Branche dédiée, pas de déplacement de fichiers |

### Definition of done

- KPI atteint et consigné.
- Tests verts.
- Shaders documentés.
- Ledger à jour.
- Rapport final avec métrique, baseline, maintenant, plafond, méthode.

---

## M03 — MoE Vulkan : GEMM groupé et pagination des experts

| Champ | Valeur |
|---|---|
| Priorité | P0 |
| Pilote | GPU AMD (Agent 1) |
| Support | Runtime (Agent 2), Numérique (Agent 4) |
| Dépend de | M01, M02 |
| Débloque | M07 |
| Durée | 6–8 semaines |
| Statut | à lancer |

### Objectif

Débloquer le MoE Vulkan : passer de 55,7 à ≥ 181,9 t/s en décodage et de
489 à ≥ 2637 t/s en préfill sur le MoE 30B A3B, sans régression dense.

### État actuel

- MoE 30B A3B : 55,7 t/s en décodage contre 181,9 pour llama.cpp, soit 3,3×.
- Préfill : 489 contre 2637 t/s, soit 5,4×.
- `moe_gemv.wgsl` lit les experts un slot à la fois ; pas de GEMM groupé.
- Le décodage est borné par les lectures d'experts en mémoire hôte, pas par
  le calcul GPU.
- Les experts sont paginés côté hôte, mais le noyau ne regroupe pas les
  tokens par expert.

### Périmètre

**Inclus** :

- `crates/kinetic-vulkan/src/moe.rs` et shaders MoE.
- Tests de qualité et de non-régression MoE.
- `PLAN.md`, `ledger.jsonl`, `configs/` pour les essais.

**Exclu** :

- Changer le format GGUF ou le routeur.
- Modifier les architectures.
- Déplacer les exécuteurs (M06).

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Profiler le routage et les accès experts ; identifier le goulot | Rapport RGP |
| WP2 | GEMM/GEMV groupé : regrouper les tokens par expert | Shader + tests |
| WP3 | Fusion gate/up/down et expert partagé | Variantes mesurées |
| WP4 | Pagination/préchargement des experts ; copies asynchrones | Réduction des stalls |
| WP5 | Tests de qualité vs CPU/llama.cpp ; non-régression dense | Tests + bench |

### Livrables

- GEMM groupé pour top-k experts.
- Expert partagé fusionné.
- Pagination/préchargement des experts.
- Rapport de bench décodage/préfill.

### Critères d'acceptation

- Décodage MoE ≥ 181,9 t/s ou ≥ 3× la baseline Kinetic.
- Préfill MoE ≥ 2637 t/s ou ≥ 5× la baseline Kinetic.
- Pas de régression > 3 % sur le dense.
- Qualité mesurée (KL/perplexité) et consignée.

### Protocole de mesure

```sh
cargo test --release --locked -p kinetic-vulkan
target/release/kinetic bench MOE.gguf --device vulkan --json -r 6
llama-bench -m MOE.gguf -dev Vulkan0 -p 512 -n 128 -r 6
rocm-smi --showuse --showtemp --showmemuse
```

### Dépendances et handoffs

- Reçoit le GEMM de préfill de M02.
- Fournit les noyaux MoE à M07.
- Coordonne avec M04 pour le KV des experts si nécessaire.

### Risques

| Risque | Mitigation |
|---|---|
| Déséquilibre des experts | Regroupement dynamique, mesures par batch |
| Bande passante mémoire | Préchargement, compression, layout |
| Qualité | Gate KL/perplexité avant publication |
| Conflit avec M06 | Branche dédiée |

### Definition of done

- KPI atteint et consigné.
- Tests verts.
- Rapport final avec métrique, baseline, maintenant, plafond, méthode.

---

## M04 — KV cache : audit, pool unifié, pagination et partage

| Champ | Valeur |
|---|---|
| Priorité | P0 |
| Pilote | Runtime (Agent 2) |
| Support | GPU AMD (Agent 1), Numérique (Agent 4), QA (Agent 5) |
| Dépend de | M01 |
| Débloque | M05, M08 |
| Durée | 6–10 semaines |
| Statut | à lancer |

### Objectif

Comprendre et redesigner le cache KV : mesurer l'existant, unifier le pool,
introduire la pagination et le partage de préfixe, quantifier le KV côté
Vulkan, et lever le mur mémoire qui limite le contexte et la concurrence.

### État actuel

| Backend | Stockage | Quantification | Slots | Partage | Pagination |
|---|---|---|---|---|---|
| CPU | `Vec<f32>` par couche | f32 | N, limité par la mémoire | Non | Non |
| CUDA | buffer device | f16 ou q8_0 | 1 à 4 | Copie pour shared KV | Non |
| Vulkan | buffer device | f16 uniquement | 1 | Non (re-projection) | Non |
| Warm Start | cache d'état hôte | exact | N/A | Préfixe exact, pas de blocs | Non |

- Le KV est contigu par séquence et réservé pour le contexte complet.
- `KvCache` Vulkan est en f16, non quantifié.
- CUDA a déjà q8_0 ; CPU est en f32.
- Pas de pool unifié, pas de block table, pas de partage de blocs entre
  requêtes.
- `Convoy` mutualise des passes, pas le KV.
- `Warm Start` est un cache d'état exact (1 Gio RAM / 2 Gio disque), pas un
  cache de blocs.
- Le rollback spéculatif est contraint par la copie de l'ancien K/V côté CPU
  et par `MAX_BATCH` côté GPU.

### Périmètre

**Inclus** :

- `crates/kinetic-runtime/**` : pool KV, block table, métriques.
- `crates/kinetic-models/src/{convoy,memory,per_layer}.rs` et `warm/**`.
- `crates/kinetic-server/src/{engine,sequence}.rs` pour l'intégration.
- Tests et mesures dans `kinetic-models`, `kinetic-vulkan`, `kinetic-cuda`.
- `PLAN.md`, `ledger.jsonl`, `configs/`.

**Exclu** :

- Modifier les noyaux d'attention sans coordination avec Agent 1.
- Remplacer Warm Start.
- Changer les contrats `Family`/`Target`.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Audit : cartographier chaque allocation KV, slot, cycle de vie | Cartographie + métriques |
| WP2 | Instrumenter : octets alloués/utilisés, fragmentation, cache hit, évictions | Métriques exposées |
| WP3 | Baseline : contexte max par slot, nombre de slots sur 16 Gio, time-to-OOM | Rapport de mur mémoire |
| WP4 | Pool unifié : budget, croissance dynamique, éviction, fallback hôte | Design + prototype |
| WP5 | Pagination : block table, taille de bloc, gather attention, fragmentation | Prototype + tests |
| WP6 | Partage de préfixe : hash de blocs, COW, shared KV zéro-copie | Design + prototype |
| WP7 | Quantification KV : q8_0 Vulkan, q4_0, politique par couche, K/V séparés | Formats + gates |
| WP8 | Rollback spéculatif : troncature du block table, checkpoint | Tests d'exactitude |
| WP9 | Offload froid et multi-GPU (P2) | Design + mesures |

### Livrables

- Design doc du pool KV et de la pagination.
- Métriques KV par requête et par backend.
- Prototype de pool unifié.
- Prototype de pagination avec tests d'exactitude.
- Design du partage de préfixe et du shared KV zéro-copie.
- Rapport de bench mémoire/contexte/concurrence.

### Critères d'acceptation

- Un contexte de 32k tient sur 16 Gio avec ≥ 4 slots, sans OOM.
- Les métriques KV sont exposées dans `kinetic doctor` ou le serveur.
- La pagination ne change pas les tokens sur le corpus de contrôle.
- q8_0 Vulkan double environ le contexte utile, avec qualité mesurée.
- Le partage de préfixe réduit la mémoire des préfixes communs.
- Aucune régression > 3 % sur les benches existants.

### Protocole de mesure

```sh
target/release/kinetic doctor
target/release/kinetic bench MODEL.gguf --device vulkan --json -r 6
target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
python scripts/same_answer_demo.py --model MODEL.gguf --kinetic target/release/kinetic
python scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091
rocm-smi --showmemuse --showuse --showtemp
```

### Dépendances et handoffs

- Reçoit la baseline et le ledger de M01.
- Fournit le pool KV à M05 et les formats KV à M08.
- Coordonne avec Agent 1 pour les noyaux d'attention paginée.
- Coordonne avec Agent 4 pour la quantification et les gates.

### Risques

| Risque | Mitigation |
|---|---|
| Complexité de la pagination | Prototype incrémental, tests d'abord |
| Fragmentation | Métriques, défragmentation, taille de bloc |
| Qualité q8_0/q4_0 | KL/perplexité, gate par format |
| Rollback spéculatif | Tests d'égalité, checkpoint |
| Conflit avec M05/M06 | Périmètre et branches séparés |

### Definition of done

- Design doc accepté.
- Prototype mesuré.
- Tests d'exactitude verts.
- Métriques disponibles.
- Ledger à jour.
- Rapport final avec mémoire, contexte, concurrence, qualité.

---

## M05 — Ordonnanceur runtime et concurrence

| Champ | Valeur |
|---|---|
| Priorité | P0 |
| Pilote | Runtime (Agent 2) |
| Support | Rust backend (Agent 3), QA (Agent 5) |
| Dépend de | M01, M04 |
| Débloque | M11 |
| Durée | 6–10 semaines |
| Statut | à lancer |

### Objectif

Faire passer le serveur d'un modèle « N backends indépendants » à un
ordonnanceur token-level qui tient la concurrence, le TTFT et le p99, sans
casser RAD-1, Warm Start ni Convoy.

### État actuel

- `Engine.free` est un `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>` : une
  requête par backend.
- `PREFILL_CHUNK = 256` existe, mais le chunking est par requête.
- `Convoy` mutualise des passes, mais ne planifie pas les requêtes.
- Le KV est contigu par séquence, non paginé.
- `MAX_CONNECTIONS = 64`, `Connection: close`, pas de keep-alive.
- Vulkan n'a qu'un slot ; CUDA/CPU ont jusqu'à 4 slots par défaut.
- Pas de métriques par token (TTFT, p50/p99, cache hit, queue depth).

### Périmètre

**Inclus** :

- `crates/kinetic-runtime/**` : ordonnanceur, admission, priorités,
  annulation, backpressure, métriques.
- `crates/kinetic-server/src/{engine,sequence,backend,server}.rs`.
- `crates/kinetic-models/src/{convoy,memory,per_layer}.rs`, `warm/**`.
- Tests de charge et d'exactitude sous charge.

**Exclu** :

- Modifier les noyaux GPU.
- Changer les contrats `Family`/`Target`/`Backend` sans décision.
- Introduire une dépendance async lourde sans justification.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Cartographier le cycle de vie requête/slot/KV/annulation | Design doc |
| WP2 | Admission, priorité, fairness, backpressure | Prototype |
| WP3 | Chunked prefill global et recouvrement préfill/décodage | Prototype |
| WP4 | Multi-slot Vulkan et keep-alive | Backend + tests |
| WP5 | Métriques par token : TTFT, tok/s, p50/p99, cache hit, queue depth | Métriques |
| WP6 | Exactitude sous charge : `same_answer_demo.py` | Rapport |
| WP7 | Soak concurrent et watchdog mémoire | Tests |

### Livrables

- Design doc de l'ordonnanceur.
- Prototype token-level.
- Pool KV N × contexte intégré (avec M04).
- Chunked prefill global.
- Annulation et backpressure.
- Métriques par token.
- Benchmarks c1/c4/c16/c64.

### Critères d'acceptation

- Débit agrégé en hausse à c4/c16/c64.
- p99 et TTFT mesurés et reproductibles.
- `same_answer_demo.py` identique sous charge.
- Aucun test d'identité cassé.
- Aucune fuite mémoire ni cache non borné.

### Protocole de mesure

```sh
target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
oha -n 64 -c 4 --json http://127.0.0.1:8080/v1/chat/completions ...
python scripts/same_answer_demo.py --model MODEL.gguf \
  --kinetic target/release/kinetic --kinetic-args "--device vulkan"
python scripts/server_abuse.py --port 8091
rocm-smi --showuse --showtemp --showmemuse
```

### Dépendances et handoffs

- Reçoit le pool KV de M04.
- Fournit les slots et métriques à M11.
- Coordonne avec M06 pour l'emplacement de l'ordonnanceur.

### Risques

| Risque | Mitigation |
|---|---|
| Complexité de l'ordonnanceur | Prototype incrémental, tests |
| Famine ou priorité injuste | Fairness mesurée, quotas |
| Rollback KV | Tests d'exactitude |
| Vulkan mono-slot | Multi-slot avant cN |
| Conflit avec M06 | Périmètre séparé |

### Definition of done

- Design doc accepté.
- cN mesuré.
- p99/TTFT documentés.
- Tests d'exactitude verts.
- Ledger à jour.

---

## M06 — Consolidation backend : un seul backend complet par API

| Champ | Valeur |
|---|---|
| Priorité | P0/P1 |
| Pilote | Rust backend (Agent 3) |
| Support | GPU AMD (Agent 1), Runtime (Agent 2), QA (Agent 5) |
| Dépend de | M01 |
| Débloque | M07, M13 |
| Durée | 8–12 semaines |
| Statut | à lancer |

### Objectif

Réduire la duplication CUDA/Vulkan, faire de chaque API un backend complet
et préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.

### État actuel

| Aujourd'hui | Cible |
|---|---|
| CUDA : `kinetic-cuda` + `kinetic-models/src/cuda` | `kinetic-cuda` contient tout CUDA |
| Vulkan : `kinetic-vulkan` + `kinetic-models/src/vulkan` | `kinetic-vulkan` contient tout Vulkan |
| `kinetic-models` a des features `cuda`/`vulkan` | `kinetic-models` sans feature backend |
| `kinetic-server` a `cuda.rs`, `vulkan.rs`, `cpu.rs` | factory, un seul backend chargé |
| Ajouter HIP = crate + module dans `kinetic-models` | crate qui implémente `ModelRunner` |

### WP0 — Décision à prendre avant tout code

Le document `docs/experts.md` place à la fois l'ordonnanceur dans
`kinetic-runtime` et le trait `ModelRunner` dans `kinetic-models`. Pour
éviter un cycle de dépendances, trancher :

- **Option A (recommandée)** : `kinetic-runtime` définit `ModelRunner`,
  `kinetic-models` implémente le runner CPU de référence, les backends
  implémentent `ModelRunner`, l'ordonnanceur runtime appelle le trait. La
  factory qui connaît les crates concrets vit dans `kinetic-server`.
- **Option B** : `kinetic-models` définit `ModelRunner`, l'ordonnanceur vit
  dans `kinetic-server` (ou `kinetic-scheduler`), `kinetic-runtime` se limite
  aux devices, capacités, mémoire et pool KV.

La décision doit être écrite dans un ADR avant le déplacement des
exécuteurs.

### Périmètre

**Inclus** :

- `Cargo.toml`, `crates/*/Cargo.toml`.
- `crates/kinetic-runtime/**` (à créer).
- `crates/kinetic-models/src/{lib,model}.rs` et `src/runner/**`.
- `crates/kinetic-cuda/**`, `crates/kinetic-vulkan/**`.
- `crates/kinetic-server/src/{backend,lib}.rs`.
- `docs/src/contributing.md`, `docs/src/building.md`.

**Exclu** :

- Modifier les noyaux ou les shaders.
- Changer le comportement observable.
- Introduire MLIR ou un compilateur de graphes.
- Casser les seams `Family`/`Target`/`Backend`/`Projector`.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP0 | Trancher la direction des dépendances et l'emplacement de la factory | ADR |
| WP1 | Créer `kinetic-runtime` : devices, capacités, mémoire, traits | Crate + tests |
| WP2 | Extraire `ModelRunner` et le runner CPU de référence | Trait + tests |
| WP3 | Déplacer l'exécuteur CUDA dans `kinetic-cuda/src/runner` | Déplacement + tests |
| WP4 | Déplacer l'exécuteur Vulkan dans `kinetic-vulkan/src/runner` | Déplacement + tests |
| WP5 | Retirer les features `cuda`/`vulkan` de `kinetic-models` | Build sans backend |
| WP6 | Factory de backend dans `kinetic-server`/`kinetic-cli` | `auto/cpu/cuda/vulkan/hip` |
| WP7 | Backend de test (CPU ou mock) | Tests sans GPU |
| WP8 | Mesurer build, démarrage, overhead hôte | Rapport |

### Livrables

- `kinetic-runtime` avec les traits et types de base.
- `ModelRunner` et runner CPU de référence.
- Exécuteurs CUDA et Vulkan déplacés dans leurs crates.
- `kinetic-models` sans feature backend.
- Factory de backend.
- Backend de test.
- ADR et documentation de la règle « un backend complet par API ».

### Critères d'acceptation

- `kinetic-models` ne dépend plus de CUDA/Vulkan.
- Un seul crate par API contient tout le code de cette API.
- Les tests d'identité restent verts à chaque étape.
- Le temps de build et le démarrage ne régressent pas.
- Chaque `unsafe` a un commentaire `SAFETY` et un test associé.
- Un seul backend est chargé à l'exécution.

### Protocole de mesure

```sh
cargo build --release --locked --workspace --all-features
cargo test --release --locked --workspace
cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
cargo tree -p kinetic-models
cargo bloat --release -p kinetic-cli
perf record -g target/release/kinetic ...
```

### Dépendances et handoffs

- Reçoit la baseline de M01.
- Coordonne avec M02/M03/M04 pour ne pas déplacer les fichiers en cours
  d'optimisation.
- Fournit le trait backend à M07.
- Fournit les features stabilisées à M13.

### Risques

| Risque | Mitigation |
|---|---|
| Refactor massif | Déplacement mécanique d'abord, optimisation ensuite |
| Conflit avec M02/M03/M04 | Branche/worktree dédié, coordination |
| Cycle de dépendances | WP0 tranché avant code |
| Régression de performance | Mesure build/démarrage/overhead à chaque étape |

### Definition of done

- ADR accepté.
- `kinetic-models` sans backend.
- Tests verts.
- Build et démarrage non régressés.
- Documentation à jour.

---

## M07 — Backend HIP/ROCm : squelette et noyaux de base

| Champ | Valeur |
|---|---|
| Priorité | P1 |
| Pilote | GPU AMD (Agent 1) |
| Support | Rust backend (Agent 3), QA (Agent 5) |
| Dépend de | M06, M02 |
| Débloque | — |
| Durée | 8–12 semaines |
| Statut | à lancer |

### Objectif

Poser les bases d'un backend HIP complet pour `gfx1201`, avec les noyaux de
base et une conformance vs llama.cpp ROCm, sans dupliquer l'exécuteur.

### État actuel

- Pas de backend HIP ; ROCm 7.2 est installé sur la machine locale.
- `kinetic-vulkan` fonctionne sur `gfx1201`.
- Le décodage Vulkan est à 95 % de llama.cpp ; le préfill est le premier
  chantier.
- HIP est une cible de compatibilité, pas un prérequis du chemin critique.

### Périmètre

**Inclus** :

- `crates/kinetic-hip/**` (à créer).
- Tests de conformance et de bench HIP.
- `PLAN.md`, `ledger.jsonl`, `configs/`.

**Exclu** :

- Remplacer Vulkan.
- Modifier `kinetic-models` ou les architectures.
- Promettre une parité complète avant validation CI.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Squelette : device, mémoire, capacités, build `hipcc` | Crate qui compile |
| WP2 | GEMV Q4_K, Q5_K, Q6_K, Q8_0 | Noyaux + tests |
| WP3 | Attention FP16 | Noyau + tests |
| WP4 | Conformance vs llama.cpp ROCm0 | Rapport de tokens |
| WP5 | Bench et baseline `gfx1201` | Rapport de perf |
| WP6 | CI/opt-in et documentation | Backend expérimental |

### Livrables

- `kinetic-hip` avec device, mémoire et capacités.
- GEMV de base.
- Attention FP16.
- Rapport de conformance et de perf.
- Documentation d'activation opt-in.

### Critères d'acceptation

- Un modèle se charge et génère sur `gfx1201`.
- Tokens identiques au runner CPU sur le corpus de contrôle.
- Baseline de perf consignée.
- Backend opt-in, pas activé par défaut sans CI.

### Protocole de mesure

```sh
cargo build --release --locked -p kinetic-cli --features hip
cargo test --release --locked -p kinetic-hip
target/release/kinetic bench MODEL.gguf --device hip --json -r 6
llama-bench -m MODEL.gguf -dev ROCm0 -p 512 -n 128 -r 6
rocm-smi --showuse --showtemp --showmemuse
```

### Dépendances et handoffs

- Reçoit le trait backend de M06.
- Reçoit les noyaux de base de M02.
- Fournit une alternative à Vulkan pour les utilisateurs ROCm.

### Risques

| Risque | Mitigation |
|---|---|
| Support ROCm incomplet | Rester opt-in, documenter |
| Portabilité des noyaux | Tests de conformance |
| Pas de CI GPU | Machine locale de référence |
| Duplication avec Vulkan | Partager les sources quand c'est possible, pas les exécuteurs |

### Definition of done

- Backend compile et tourne.
- Conformance verte.
- Baseline consignée.
- Documentation à jour.

---

## M08 — Quantification et formats KV

| Champ | Valeur |
|---|---|
| Priorité | P0/P1 |
| Pilote | Numérique (Agent 4) |
| Support | GPU AMD (Agent 1), Runtime (Agent 2) |
| Dépend de | M01, M04, M02 |
| Débloque | — |
| Durée | 8–12 semaines |
| Statut | à lancer |

### Objectif

Étendre la quantification et garantir que chaque gain de vitesse reste
exact : NVFP4, FP8 E4M3, IQ encode, imatrix, KV q8_0/q4_0 Vulkan, politique
par couche et qualité mesurée.

### État actuel

- 5 cibles de quantification contre ~30 dans llama.cpp.
- Pas d'imatrix, pas d'IQ encode.
- NVFP4 et F8_E4M3 absents du runtime.
- KV : CUDA f16/q8_0 ; Vulkan f16 uniquement ; CPU f32.
- `kinetic-quant` est la définition de référence ; `kdot/tests.rs` compare
  chaque noyau.
- L'exactitude est le différenciateur produit.

### Périmètre

**Inclus** :

- `crates/kinetic-quant/**`.
- `crates/kinetic-gguf/**`.
- `crates/kinetic-cpu/src/kdot/**`.
- Tests de noyaux dans `kinetic-cuda/tests/**` et `kinetic-vulkan/tests/**`.
- `docs/src/models.md`.
- `PLAN.md`, `ledger.jsonl`, `configs/`.

**Exclu** :

- Modifier les architectures ou le runtime.
- Accepter un noyau qui ne passe pas la référence f64.
- Changer un format existant sans versionner.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Auditer les gates et le corpus de conformance | Inventaire |
| WP2 | NVFP4 et FP8 E4M3 runtime | Noyaux + tests |
| WP3 | IQ encode et imatrix | Encodeur + tests |
| WP4 | KV q8_0 Vulkan, q4_0, politique par couche, K/V séparés | Formats + gates |
| WP5 | Qualité : KL/perplexité, corpus de calibration | Rapport |
| WP6 | Tests property/fuzz et conformance | Tests |

### Livrables

- NVFP4 et FP8 E4M3.
- IQ encode et imatrix.
- KV q8_0/q4_0 Vulkan.
- Politique de quantification par tenseur/couche.
- Rapport qualité/vitesse/mémoire.

### Critères d'acceptation

- Chaque nouveau format a une référence f64 et un noyau testé.
- Aucune régression d'identité.
- La qualité est mesurée, pas supposée.
- Les chiffres sont reproductibles et consignés.
- q8_0 Vulkan ne réimplémente pas q8_0 CUDA : il l'étend au backend Vulkan.

### Protocole de mesure

```sh
cargo test --release --locked -p kinetic-quant
cargo test --release --locked -p kinetic-cpu
cargo test --release --locked -p kinetic-vulkan
target/release/kinetic quantize MODEL.gguf --help
target/release/kinetic perplexity MODEL.gguf
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device vulkan
```

### Dépendances et handoffs

- Reçoit le pool KV de M04.
- Reçoit les noyaux de M02.
- Fournit les formats et gates à M01/M05.

### Risques

| Risque | Mitigation |
|---|---|
| Perte de qualité | KL/perplexité, gate par format |
| Exactitude | Référence f64, tests différentiels |
| Complexité IQ | Incrémental, corpus |
| Conflit avec M02 | Coordination des shaders/tests |

### Definition of done

- Formats implémentés.
- Gates verts.
- Rapport qualité/vitesse.
- Ledger à jour.

---

## M09 — Couverture des architectures

| Champ | Valeur |
|---|---|
| Priorité | P1 |
| Pilote | Intégration modèles (Agent 6) |
| Support | Numérique (Agent 4), QA (Agent 5) |
| Dépend de | M01 |
| Débloque | M12 |
| Durée | continu |
| Statut | à lancer |

### Objectif

Couvrir les architectures manquantes par briques réutilisables, sans
multiplier les `match` ni dupliquer `transformer.rs`.

### État actuel

- 21/180 architectures.
- Les briques rentables : MLA/DSA, hybrides SSM, T5, Gemma 2, GPT-OSS.
- Les modèles locaux bloqués : `qwen3vl`, `deepseek`, `llama4`, `gemma2`,
  `phi3`, `command-r`, `dbrx`.
- Le registre `Family` et les blocs sont les bons points d'extension.

### Périmètre

**Inclus** :

- `crates/kinetic-models/src/arch/**`.
- `crates/kinetic-models/src/blocks/**`.
- `docs/src/models.md`.
- `scripts/new_family.py` si nécessaire.

**Exclu** :

- Ajouter un `match` central.
- Dupliquer `transformer.rs`.
- Modifier les backends.
- Copier du code llama.cpp.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Choisir une famille proche (phi3, gemma2, command-r) | Famille chargée |
| WP2 | Mapper les tenseurs GGUF vers les blocs | Mapping + tests |
| WP3 | Ajouter une ligne dans `FAMILIES` | Registre |
| WP4 | Conformance vs llama.cpp | Rapport de tokens |
| WP5 | Documenter `models.md` | Doc |
| WP6 | Ajouter MLA/DSA, SSM, T5 par briques | Blocs réutilisables |

### Livrables

- Familles prioritaires chargées.
- Blocs manquants (MLA, DSA, Mamba 1, RWKV, T5, iRoPE, sinks).
- Tests de conformance.
- Documentation à jour.

### Critères d'acceptation

- Le modèle se charge et génère.
- Les tokens sont identiques à llama.cpp sur le corpus de contrôle.
- Aucun `match` central ajouté.
- La documentation est à jour.

### Protocole de mesure

```sh
target/release/kinetic info MODEL.gguf
target/release/kinetic run MODEL.gguf --device cpu "Hello"
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device cpu
cargo test --release --locked -p kinetic-models
```

### Dépendances et handoffs

- Reçoit la conformance de M01.
- Fournit les architectures à M12.
- Coordonne avec M10 pour les tokens spéciaux.

### Risques

| Risque | Mitigation |
|---|---|
| Scope creep | Prioriser les modèles locaux |
| Variantes RoPE/norm | Tests de conformance |
| Duplication | Un fichier par famille, blocs réutilisables |

### Definition of done

- Famille(s) chargée(s) et conforme(s).
- Tests verts.
- Doc à jour.

---

## M10 — Tokenizers, templates et tool calls

| Champ | Valeur |
|---|---|
| Priorité | P1 |
| Pilote | Tokenizers (Agent 7) |
| Support | Serveur (Agent 8), QA (Agent 5) |
| Dépend de | M01 |
| Débloque | M11 |
| Durée | 6–10 semaines |
| Statut | à lancer |

### Objectif

Atteindre la compatibilité des tokenizers et des templates : prétokenizers,
UGM, tokens spéciaux, templates intégrés et tool calls.

### État actuel

- 13/68 prétokenizers, 55 manquants.
- UGM, RWKV, PLaMo, Tekken, Jais2, etc. manquants.
- Tokens EOT/EOM/FIM et types spéciaux incomplets.
- Templates intégrés et formats tool-call incomplets.
- L'issue #5 (tool calls dans le raisonnement) est typique.

### Périmètre

**Inclus** :

- `crates/kinetic-tokenizer/**`.
- `crates/kinetic-server/src/{template,tool_calls,tool_grammar}.rs`.
- `docs/src/models.md`.

**Exclu** :

- Modifier les architectures ou les noyaux.
- Changer le contrat `Tokenizer::from_gguf/encode/decode`.
- Copier des regex sans NOTICE.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Corpus de contrôle et comparaison `llama-tokenize` | Corpus + rapport |
| WP2 | 5 prétokenizers prioritaires | Prétokenizers + tests |
| WP3 | UGM et variantes BPE | Implémentation + tests |
| WP4 | EOT/EOM/FIM et tokens de contrôle | Tests |
| WP5 | Templates intégrés par nom | Templates + tests |
| WP6 | Formats tool-call manquants | Tests client |

### Livrables

- 15 prétokenizers.
- UGM.
- Tokens spéciaux.
- Templates intégrés.
- Formats tool-call.
- Split GGUF si nécessaire.

### Critères d'acceptation

- Ids identiques sur le corpus.
- Tokens spéciaux corrects.
- Tool calls valides.
- Aucune régression de template.

### Protocole de mesure

```sh
llama-tokenize -m MODEL.gguf -f corpus.txt
cargo test --release --locked -p kinetic-tokenizer
python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
  --llama /path/to/llama.cpp/bin --device cpu
```

### Dépendances et handoffs

- Reçoit le corpus de M01.
- Fournit les templates et tool calls à M11.
- Coordonne avec M09 pour les tokens de famille.

### Risques

| Risque | Mitigation |
|---|---|
| Regex complexes | Tests différentiels, NOTICE |
| Tokens spéciaux | Corpus dédié |
| Templates | Tests client réels |

### Definition of done

- Corpus vert.
- Tool calls valides.
- Doc à jour.

---

## M11 — Serveur, API et observabilité

| Champ | Valeur |
|---|---|
| Priorité | P1 |
| Pilote | Serveur/API (Agent 8) |
| Support | Runtime (Agent 2), Tokenizers (Agent 7), QA (Agent 5) |
| Dépend de | M01, M05, M10 |
| Débloque | M14 |
| Durée | 8–12 semaines |
| Statut | à lancer |

### Objectif

Atteindre la parité API, la robustesse et l'observabilité : routes
manquantes, keep-alive, TLS/API key, multi-modèle, métriques et compatibilité
clients.

### État actuel

- ~15/45 routes.
- Pas de keep-alive/TLS.
- 64 connexions thread-per-connection.
- `docker stop` à améliorer.
- Compatibilité clients perfectible.

### Périmètre

**Inclus** :

- `crates/kinetic-server/src/{server,http,api,llama,metrics}.rs`.
- `crates/kinetic-server/src/ollama/`.
- `docs/src/server.md`, `docs/src/clients.md`.

**Exclu** :

- Modifier les noyaux ou le runtime.
- Casser les clients existants.
- Ajouter une dépendance réseau non nécessaire.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Comparer routes/champs avec llama-server | Inventaire |
| WP2 | Routes manquantes : `/props`, `/slots/:id`, `/infill`, `/apply-template`, `/lora-adapters`, `n > 1`, `timings_per_token` | Routes + tests |
| WP3 | Keep-alive, TLS/API key, CORS | Configuration |
| WP4 | Router multi-modèle | Design + tests |
| WP5 | Métriques Prometheus, logs, tracing | Métriques |
| WP6 | Arrêt propre et `docker stop` | Test |
| WP7 | Compatibilité llama-server/OpenAI/Anthropic/Ollama | Rapport |
| WP8 | Sécurité des entrées (limites, DoS, path traversal) | Tests + checklist |

### Livrables

- Routes manquantes.
- Keep-alive/TLS/API key.
- Router multi-modèle.
- Métriques.
- Arrêt propre.
- Rapport de compatibilité.

### Critères d'acceptation

- Les clients existants fonctionnent sans modification.
- Les champs inconnus sont ignorés ou refusés clairement.
- Pas de crash sous `server_abuse.py`.
- Métriques exploitables.
- Pas de régression d'exactitude sous charge.

### Protocole de mesure

```sh
target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
python scripts/llama_server_compat.py --model MODEL.gguf \
  --llama-server /path/to/llama-server --kinetic target/release/kinetic
python scripts/server_abuse.py --port 8091
python scripts/same_answer_demo.py --model MODEL.gguf \
  --kinetic target/release/kinetic --kinetic-args "--device vulkan"
```

### Dépendances et handoffs

- Reçoit les slots/métriques de M05.
- Reçoit les templates de M10.
- Fournit la surface à sécuriser à M14.

### Risques

| Risque | Mitigation |
|---|---|
| Compatibilité clients | Tests réels, corpus |
| Sécurité | M14, limites d'entrée |
| Concurrence | M05, multi-slot |
| Scope routes | Prioriser les clients utilisés |

### Definition of done

- Routes prioritaires livrées.
- Tests de compatibilité verts.
- Métriques disponibles.
- Doc à jour.

---

## M12 — Multimodal : vision, audio, OCR

| Champ | Valeur |
|---|---|
| Priorité | P2 |
| Pilote | Multimodal (Agent 9) |
| Support | Modèles (Agent 6), GPU (Agent 1), QA (Agent 5) |
| Dépend de | M01, M09 |
| Débloque | — |
| Durée | XL |
| Statut | à lancer |

### Objectif

Étendre la couverture multimodale : projecteurs vision, audio et OCR, avec
parité des embeddings et sans régression texte.

### État actuel

- 2/73 projecteurs.
- Pas d'audio/vidéo/OCR.
- `kinetic-vision` a un contrat `Projector` propre.
- Le multimodal est XL et hors chemin critique de performance.

### Périmètre

**Inclus** :

- `crates/kinetic-vision/**`.
- `crates/kinetic-audio/**` (à créer).
- `crates/kinetic-models/src/input.rs`.
- `docs/src/models.md`.

**Exclu** :

- Modifier le texte ou les backends.
- Élargir le contrat `Projector` pour l'audio.
- Copier du code llama.cpp.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | 3 projecteurs prioritaires | Projecteurs + tests |
| WP2 | 10 projecteurs majeurs | Couverture |
| WP3 | Crate `kinetic-audio` | Audio |
| WP4 | Pipeline OCR | OCR |
| WP5 | Parité `llama-mtmd-debug` | Rapport |
| WP6 | Documentation | Doc |

### Livrables

- Projecteurs vision.
- `kinetic-audio`.
- Pipeline OCR.
- Tests de parité.
- Documentation.

### Critères d'acceptation

- Embeddings proches de llama.cpp.
- Réponses cohérentes.
- Pas de régression texte.

### Protocole de mesure

```sh
cargo run --release -p kinetic-vision --example encode -- mmproj.gguf rainbow:512
llama-mtmd-debug --image rainbow -n 512 -m mmproj.gguf
target/release/kinetic serve MODEL.gguf --mmproj mmproj.gguf
```

### Dépendances et handoffs

- Reçoit les architectures de M09.
- Reçoit les noyaux GPU de M02.
- Fournit la parité à M01.

### Risques

| Risque | Mitigation |
|---|---|
| Scope XL | Prioriser 3 puis 10 projecteurs |
| Preprocessing | Tests de parité |
| Audio/OCR | Traiter après la vision |

### Definition of done

- Projecteurs prioritaires livrés.
- Parité mesurée.
- Doc à jour.

---

## M13 — Build, release et bindings

| Champ | Valeur |
|---|---|
| Priorité | P2 |
| Pilote | Build/release (Agent 10) |
| Support | Serveur (Agent 8), Rust backend (Agent 3) |
| Dépend de | M06 |
| Débloque | M14 |
| Durée | 8–12 semaines |
| Statut | à lancer |

### Objectif

Rendre le produit installable et intégrable : ABI C, bindings, Docker,
packaging, CI release, taille et temps de build.

### État actuel

- 10/200 fonctions C.
- Bindings Python partiels.
- Images Docker et packaging à finaliser.
- CI multi-OS existante.

### Périmètre

**Inclus** :

- `.github/workflows/**`.
- `docker/**`, `packaging/**`.
- `bindings/**`.
- `crates/kinetic-capi/**`.
- `docs/src/{building,bindings}.md`.

**Exclu** :

- Modifier les noyaux ou le runtime.
- Casser l'ABI existante sans versionner.
- Publier un artefact non testé.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Audit ABI et bindings | Inventaire |
| WP2 | Compléter l'ABI C | ABI + tests |
| WP3 | Bindings Python/Node/Go | Bindings + exemples |
| WP4 | Docker, winget/scoop, archives | Artefacts |
| WP5 | CI release | Workflow |
| WP6 | Réduire taille et temps de build | Mesures |
| WP7 | Documentation de build | Doc |

### Livrables

- ABI C complète.
- Bindings générés.
- Docker et installateurs.
- CI release.
- Documentation.

### Critères d'acceptation

- Installation réussie sur Windows/Linux/macOS.
- ABI testée.
- Build reproductible.
- Documentation à jour.

### Protocole de mesure

```sh
cargo build --release --locked --workspace --all-features
cargo test --release --locked --workspace
cd bindings/python && maturin build --release --locked
docker build -f docker/Dockerfile -t kinetic .
```

### Dépendances et handoffs

- Reçoit les features stabilisées de M06.
- Reçoit la surface serveur de M11.
- Fournit les artefacts à M14.

### Risques

| Risque | Mitigation |
|---|---|
| Toolchains CUDA/HIP | CI dédiée, conteneurs |
| Taille de build | LTO/PGO, features |
| ABI instable | Semver, tests |

### Definition of done

- Artefacts testés.
- CI release verte.
- Doc à jour.

---

## M14 — Sécurité

| Champ | Valeur |
|---|---|
| Priorité | P1/P2 |
| Pilote | Serveur/API (Agent 8) |
| Support | Rust backend (Agent 3), Build/release (Agent 10) |
| Dépend de | M11, M13 |
| Débloque | — |
| Durée | 4–6 semaines |
| Statut | à lancer |

### Objectif

Réduire la surface d'attaque du produit local-first : serveur, fichiers de
modèle, `unsafe`, dépendances et chaîne d'approvisionnement.

### État actuel

- Serveur local avec 64 connexions, pas de TLS/API key.
- Surface `unsafe` concentrée aux frontières SIMD/GPU/FFI.
- Fichiers GGUF et mmproj lus depuis le disque.
- Pas de threat model formel.
- `server_abuse.py` existe mais ne couvre pas tout.

### Périmètre

**Inclus** :

- `docs/src/security.md` (à créer).
- Tests et scripts de sécurité.
- Audit `unsafe` dans les crates backend.
- Dépendances et workflows.

**Exclu** :

- Refonte cryptographique.
- Promesse de sécurité cloud.
- Modifier les noyaux sans coordination.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Threat model local-first | Document |
| WP2 | Audit des entrées : HTTP, JSON, templates, chemins | Rapport |
| WP3 | Limites de taille, timeouts, DoS, path traversal | Tests |
| WP4 | Audit `unsafe` et `Send`/`Sync` | Rapport + SAFETY |
| WP5 | Audit dépendances et supply chain | Rapport |
| WP6 | Fuzzing des parseurs GGUF/tokenizer/template | Corpus |
| WP7 | Checklist CI sécurité | Workflow |

### Livrables

- Threat model.
- Rapport d'audit.
- Tests d'abus.
- Checklist CI.
- Documentation sécurité.

### Critères d'acceptation

- Aucune vulnérabilité critique ouverte.
- `server_abuse.py` ne crashe pas le serveur.
- Chaque `unsafe` a un commentaire `SAFETY` et un test.
- Les dépendances sont auditées.
- Les secrets ne sont jamais loggés.

### Protocole de mesure

```sh
python scripts/server_abuse.py --port 8091
cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
cargo audit
rg -n "unsafe" crates/kinetic-cuda crates/kinetic-vulkan crates/kinetic-cpu
```

### Dépendances et handoffs

- Reçoit la surface serveur de M11.
- Reçoit les artefacts de M13.
- Fournit les gates à M01.

### Risques

| Risque | Mitigation |
|---|---|
| Sur-ingénierie | Threat model local-first, prioriser |
| Faux sentiment de sécurité | Tests, revue, audit |
| Conflit avec les features | Checklist CI, pas de blocage |

### Definition of done

- Threat model accepté.
- Audits terminés.
- Tests verts.
- Doc à jour.

---

## M15 — Documentation et expérience développeur

| Champ | Valeur |
|---|---|
| Priorité | P2 |
| Pilote | Build/release (Agent 10) |
| Support | Tous |
| Dépend de | continu |
| Débloque | — |
| Durée | continu |
| Statut | à lancer |

### Objectif

Garder la documentation alignée avec le code, faciliter l'onboarding et
éviter la dérive entre les docs, les prompts et les missions.

### État actuel

- Documentation riche : `README.md`, `docs/src/`, `PLAN.md`,
  `llama-kinetic.md`.
- Risque de dérive entre `docs/experts.md`, les fiches, les prompts et les
  missions.
- Pas d'ADR pour les décisions structurantes.
- Pas de guide de migration backend.

### Périmètre

**Inclus** :

- `docs/**` (hors code).
- `README.md`, `CONTRIBUTING.md`.
- ADR.
- Exemples et guides.

**Exclu** :

- Modifier le code ou les noyaux.
- Réécrire l'historique.

### Work packages

| WP | Contenu | Sortie |
|---|---|---|
| WP1 | Index `docs/experts/README.md` | Index |
| WP2 | ADR « un backend complet par API » | ADR |
| WP3 | Guide de migration backend | Doc |
| WP4 | Mettre à jour `docs/src/contributing.md` | Doc |
| WP5 | Vérifier les liens et les commandes | CI doc |
| WP6 | Onboarding test : une famille ou un noyau en < 1 jour | Rapport |

### Livrables

- Index des docs experts.
- ADR.
- Guide de migration.
- Docs à jour.
- Rapport d'onboarding.

### Critères d'acceptation

- `mdbook build` passe.
- Les liens internes sont valides.
- Les commandes documentées fonctionnent.
- Un nouveau contributeur peut lancer un modèle et un bench en < 1 jour.

### Protocole de mesure

```sh
mdbook build docs
rg -n "TODO|FIXME|à créer" docs/experts
```

### Dépendances et handoffs

- Reçoit les décisions de M06.
- Reçoit les mesures de M01.
- Met à jour les docs de toutes les missions.

### Risques

| Risque | Mitigation |
|---|---|
| Dérive | Revue à chaque mission |
| Doc non testée | CI doc, exemples |
| Surcharge | Prioriser les docs critiques |

### Definition of done

- Docs à jour.
- Liens valides.
- Onboarding testé.

---

## 3. Mapping mission → agent

| Agent | Missions | Rôle principal |
|---|---|---|
| Agent 1 — GPU AMD | M02, M03, M07 | Noyaux Vulkan/HIP |
| Agent 2 — Runtime | M04, M05 | KV, ordonnanceur, concurrence |
| Agent 3 — Rust backend | M06 | Consolidation backend |
| Agent 4 — Numérique | M08 | Quantification, KV, exactitude |
| Agent 5 — Performance/QA | M01, support M02–M15 | Mesure, ledger, CI GPU |
| Agent 6 — Intégration modèles | M09 | Architectures |
| Agent 7 — Tokenizers | M10 | Tokenizers, templates, tool calls |
| Agent 8 — Serveur/API | M11, M14 | API, observabilité, sécurité |
| Agent 9 — Multimodal | M12 | Vision, audio, OCR |
| Agent 10 — Build/release | M13, M14, M15 | Build, release, bindings, doc |

---

## 4. Règles de lancement

1. **M01 d'abord.** Sans baseline, ledger et CI, les autres missions ne
   peuvent pas prouver leurs gains.
2. **Un agent par mission.** Ne pas mélanger GPU, runtime et modèles dans la
   même session.
3. **Un seul écrivain par fichier.** Si deux missions touchent le même
   fichier, séquencer ou utiliser une branche/worktree dédié.
4. **M06 WP0 avant le déplacement des exécuteurs.** Trancher la direction des
   dépendances et l'emplacement de la factory.
5. **Ne pas déplacer les exécuteurs pendant M02/M03/M04.** Le refactor
   attend la stabilisation des noyaux, ou se fait dans une branche dédiée.
6. **Le KV avant l'ordonnanceur.** M04 fournit le pool et les métriques dont
   M05 a besoin.
7. **La sécurité après l'API et le release.** M14 a besoin de M11 et M13.
8. **La documentation en continu.** M15 ne bloque rien, mais doit suivre
   chaque décision.

---

## 5. Ce qui n'est pas une mission

- Un benchmark isolé sans contrôle ni ledger.
- Une réécriture complète « pour faire propre ».
- Un portage CUDA-only sur une machine sans NVIDIA.
- Un backend HIP/Metal/SYCL complet sans machine ni CI pour le valider.
- Une optimisation qui change la sortie sans gate d'identité.

---

## 6. Prochaine action

Créer les fichiers de travail de M01 :

- `THEORY.md`
- `ledger.jsonl`
- `configs/`
- `best-config.json`

Puis lancer M01 avant M02/M04/M06.
