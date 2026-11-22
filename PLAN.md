# PLAN — comblement des lacunes de Kinetic face à llama.cpp

> Exécution de `llama-kinetic.md`. Le document mesure les écarts par domaine et fixe
> l'ordre (§8) ; ce journal suit cet ordre, une entrée par lacune, et n'inscrit un
> résultat qu'avec sa preuve.
>
> Une lacune est la capacité de llama.cpp qui marche et que Kinetic ne fait pas. La
> preuve est la même capacité qui réussit dans Kinetic, sur la même machine, avec le
> même fichier ou la même requête.

---

## 1. Preuve exigée par domaine

| Domaine | Preuve de clôture |
|---|---|
| Architecture | `kinetic info` nomme la famille, `kinetic run` sur le GPU (jamais de repli CPU), sortie non dégénérée, comparaison de prompts avec llama |
| Format / backend | même GGUF chargé sur GPU, `kinetic info`, sortie cohérente ; type listé dans `docs/src/models.md` |
| Tokenizer | mêmes ids que `llama-tokenize` sur un corpus de contrôle, tokens spéciaux compris |
| Modalité | même image/audio et même question qu'avec llama, réponse équivalente |
| Serveur | la requête llama-server passe sans modification, champ inconnu ignoré ou refusé clairement |
| Sampling | même paramètre, même graine, distribution vérifiée sur un prompt de contrôle |
| Spéculatif | sortie greedy identique à sans spéculation, gain mesuré |
| Outil CLI | l'artefact produit est relu par llama.cpp et par Kinetic |
| ABI / bindings | l'appel existe et un exemple compilé l'utilise |
| Performance | même fichier, même backend, `kinetic bench` contre `llama-bench`, un harnais par comparaison |

Une entrée close sans preuve reproductible est rouverte.

## 2. Règles de mesure

Reprises du §7.10 du document et du skill `inference-optimisation` :

- Un seul changement par essai, une config par essai, contrôles avant **et** après.
- Médiane de ≥ 6 échantillons ; le bruit est de ±5–10 %.
- Un candidat n'est gardé que s'il bat les deux contrôles sur sa métrique et ne
  régresse rien de plus de ~3 %.
- **Un seul job GPU à la fois.** Les faux échecs de test observés deux fois sur le
  GPU intégré venaient de mesures concurrentes.
- Exactitude : `cargo test --release -p kinetic-vulkan` (référence f64, bit-exact
  mono-token/passe prompt, garde-fou tous types).
- Plafonds avant optimisation (`THEORY.md`, à créer) : décodage borné par la bande
  passante **mesurée**, préfill par le calcul mesuré.
- Portes de stabilité avant de qualifier : needle à 50 % puis 100 % du contexte,
  équivalence greedy si le changement se dit exact.

## 3. Machine et dépôt

| Élément | Valeur (22 nov. 2026) |
|---|---|
| CPU | Ryzen 7 7800X3D (Zen 4, AVX2 + AVX-512 non exploité), 8C/16T |
| GPU | RX 9070 XT `gfx1201`, 64 CU, 16 Gio |
| iGPU | Radeon 610M `gfx1036`, 2 CU |
| Vulkan | 1.4.354 RADV / Mesa 26.1.6, sous-groupe 32–64 |
| ROCm | 7.2.0, `hipcc` gfx1201 |
| CUDA | **absent** : aucun GPU NVIDIA, Kinetic compilé sans CUDA — rien à construire ni à mesurer ici |
| llama.cpp de référence | `build-vulkan/bin/llama-bench` (Vulkan0) |

Dépôt : fork `Angra974/kinetic`, remote `fork`, branche unique `main`. `upstream`
(`Veloresearch/kinetic`) est l'amont, en lecture seule.

---

## 4. File de lacunes

### P0.1 — Sampling et routes serveur (S/M)

Le document les met en premier parce qu'ils sont petits et qu'un client existant en
dépend. Aucun backend à toucher.

**Sampling : fait (voir G3).** `dynatemp_range`, `dynatemp_exponent`, `min_keep`,
`adaptive_target`, `adaptive_decay` vivent dans le sampler partagé
(`kinetic-models/src/speculative/sampler.rs`, filtres dans `filters.rs`) ; `ignore_eos`
et `reasoning_budget_tokens` y vivent aussi, mais c'est le moteur qui les applique, là où
les tokens d'arrêt et l'état du raisonnement sont connus (`engine.rs`).

Deux choix assumés, qui ne copient pas llama.cpp :

- `ignore_eos` retire **tous** les tokens de fin (eos et marqueurs de fin de tour) et
  n'affiche jamais le texte des tokens de contrôle ; llama.cpp biaise leurs logits.
- `reasoning_budget_tokens` termine le tour quand le budget est dépensé. llama.cpp
  écrit le tag de fermeture à la place du modèle, puis laisse répondre ; Kinetic ne
  modifie pas la sortie du modèle, donc pas de réponse après le budget.

**Routes : reste à faire.** `POST /props`, `POST /slots/:id_slot`, `/infill`,
`/apply-template`, `/lora-adapters`, `n > 1`, `timings_per_token`.
- Preuve : requêtes de référence (llama-server) rejouées avec les mêmes champs.

### P0.2 — Vulkan (L)

- **Préfill** : GEMM tuilé puis coopmat. Mesure du 22 nov. sur `Qwen3.8-27B-IQ3_XXS` :
  261,9 t/s contre 1154,2 pour llama Vulkan (4,4×) ; décodage déjà à 95 %
  (38,4 contre 40,1). C'est le seul écart de performance mesuré.
- **KV quantifié** (q8_0 d'abord) : 2× de contexte sur 16 Gio, absent aujourd'hui.
- **Placement mémoire** : sous-allocation d'un grand buffer, cache de pipelines
  persisté, soumission asynchrone (5–15 % attendus, `commands.rs` 160–163).
- **MoE** : routage top-k + GEMV groupé ; conditionne `qwen3-coder-30b-a3b`.
- **Types restants** : IQ4_NL, IQ3_S, Q5_0/Q5_1/Q4_1, Q2_K/Q3_K, MXFP4.
- Preuve : pp/tg comparés à llama Vulkan après chaque levier, bit-exact conservé.

### P0.3 — `kinetic-hip` (L/XL)

Backend complet, sur le modèle de `kinetic-cuda`, un crate par backend. Après P0.2 :
le décodage Vulkan étant déjà à 95 %, HIP ne rendrait d'abord que l'écart de préfill
que le GEMM attaque moins cher.
- Preuve : GEMV Q4_K/Q5_K/Q6_K/Q8_0/IQ3_XXS/IQ4_XS, attention FP16, build gfx1201,
  puis comparaison à `llama-bench` ROCm0 sur le même fichier.

### P0.4 — Quantification de production (M/L)

- NVFP4 (type 40) et F8_E4M3 (type 43) au runtime, pour tous les backends.
- `imatrix`, cibles IQ (IQ4_XS, IQ3_XXS, IQ2_XS), `--output-tensor-type`,
  `--token-embedding-type`.
- Preuve : un fichier produit par `kinetic quantize` se charge dans llama.cpp et dans
  Kinetic avec le même `kinetic info`.

### P0.5 — Prétokenizers et split GGUF (S/M)

- **`tekken` (et `pixtral`) : fait (voir G4).** `helcyon-gpt-4o-v4.5` se charge et
  génère sur Vulkan.
- Restent UGM (T5), RWKV, PLaMo, et le variant `tekken-num3` (nombres par groupes de
  trois à droite) si un modèle l'exige.
- Tokens EOT/EOM/FIM et types de tokens spéciaux.
- Split GGUF en lecture.
- Preuve : mêmes ids que `llama-tokenize`.

### P0.6 — Architectures (L)

Le plus gros gisement : 159 noms. Ordre du document, par brique réutilisable, pas par
nom :

1. Transformers proches réutilisant `arch/transformer.rs` : `qwen2moe`, `qwen3moe`,
   `phi3`, `phimoe`, `gemma2`, `command-r`, `dbrx`, `seed_oss`, `olmoe`.
2. MoE en général (`blocks/moe.rs` existe, le noyau Vulkan manque) : c'est le
   déblocage de `qwen3-coder-30b-a3b`.
3. MLA/DSA (DeepSeek V2/V3/V4, Kimi, GLM4) : `blocks/mla.rs`, `blocks/dsa.rs`.
4. Hybrides SSM (Qwen3-Next, Mamba, Jamba, Nemotron-H, RWKV6/7).
5. Cas spéciaux : Llama 4/iRoPE, GPT-OSS/sinks, Gemma 2 softcap, Gemma 3n, T5, BitNet.
6. Multimodal : `qwen3vl` et `paddleocr` en premier (modèles présents localement).

### P1.1 — Modalités (L/XL)

Vision d'abord (2 projecteurs contre 73) : Qwen2/2.5/3-VL, InternVL, Pixtral, Gemma3,
SmolVLM, MiniCPM-V. Puis vidéo, audio (`kinetic-audio`), OCR (pipeline dédié, pas un
simple projecteur), TTS.

### P1.2 — Spéculatif (L)

Draft model séparé, EAGLE3, DFlash/DFlash2, variantes ngram-map, cache de lookup.
Toute source passe par le vérificateur exact existant.

### P1.3 — Outils CLI (S/M)

`imatrix`, `tokenize`/`detokenize`, `gguf-split`, `apply-template`, `export-lora`,
`cvector`, format `llama-bench` pour `bench`.

### P1.4 — ABI et bindings (L)

Groupes cohérents : modèle, vocab, session, sampling, grammar, état KV, LoRA,
multimodal, énumération des devices. Python/Node/Go générés depuis l'ABI.

### P1.5 — Multi-GPU et placement (XL)

Row/tensor split, `--override-tensor`, `--main-gpu`, cache MoE explicite. Sur cette
machine, seul le layer split dGPU + iGPU est exploitable (iGPU 2 CU).

### Hors périmètre de ce journal

CUDA, SYCL, Metal natif, CANN, MUSA, WebGPU, RPC, diffusion, entraînement. Ce sont des
capacités de llama.cpp, mais **aucune ne se construit ni ne se mesure sur cette
machine** : pas de GPU NVIDIA (donc CUDA inutilisable), pas de cible SYCL/Metal/CANN,
et pas de scénario d'entraînement. Les garder dans la file reviendrait à planifier du
travail invérifiable ; ils restent dans l'annexe E du document, pas ici.

Conséquence pour les tableaux du document : une colonne CUDA y décrit l'état du crate
`kinetic-cuda` (code existant, non exécutable ici), elle ne décrit pas un travail à
faire. NVFP4 et F8_E4M3 restent en P0.4 uniquement pour le runtime CPU/Vulkan/HIP —
les bénéficiaires NVIDIA ne sont pas concernés par ce journal.

---

## 5. Lacunes closes

| # | Lacune | Commit | Preuve | Effet |
|---|---|---|---|---|
| G1 | Types `IQ3_XXS`, `IQ4_XS`, `Q4_0` refusés sur Vulkan | `0127427` | 27B sur GPU, sortie cohérente, 6+20+1 tests | CPU 3,8 → 154,8 t/s pp512 |
| G2 | Passe prompt sans noyau many-token pour ces types | `85d29aa` | bit-exact conservé, tests verts | pp512 154,8 → 261,9 t/s ; résidu 4,4× vers P0.2 |
| G3 | Sampling de llama-server absent : `dynatemp_range/exponent`, `min_keep`, `adaptive_target/decay`, `ignore_eos`, `reasoning_budget_tokens` | non committé | mêmes requêtes rejouées sur les deux serveurs, 22 nov., L3-8B IQ4_XS et Qwen3.5-9B Q4_K_M ; 174 tests unitaires verts (91 modèles + 83 serveur) | champs acceptés des deux côtés ; `ignore_eos` passe de `stop` 4 car. à `length` 167 car. ; budget 8 → 8 tokens de raisonnement contre 200 sans budget |
| G4 | Prétokenizer `tekken` absent : `helcyon-gpt-4o-v4.5` refusé au chargement | non committé | mêmes ids que `llama-tokenize` sur 26 lignes de contrôle et sur un fichier entier (83 tokens, identiques), 22 nov. ; 28 tests tokenizer verts ; génération Vulkan 77,3 t/s | le modèle se charge et génère |

G1 et G2 ferment une partie de **§4.3 formats** et amorcent **P0.2** ; G3 ferme la
partie sampling de **P0.1**, pas ses routes ; G4 ferme la partie `tekken` de **P0.5**.

### Mesures de G3 (22 nov. 2026)

Même requête, même fichier, un seul serveur GPU à la fois ; llama.cpp `19bc41dbb`
(Vulkan0) contre kinetic `main` + ce lot.

| Requête | llama-server | kinetic |
|---|---|---|
| chat, température 0 | `stop`, « Done » (4 car.) | `stop`, « Done » (4 car.) |
| idem + `ignore_eos` | `length`, 19 car. | `length`, 167 car. |
| `dynatemp_range` 0,8 / exp 1 | HTTP 200, 101 car. | HTTP 200, 96 car. |
| `top_p` 0 + `min_keep` 5 | HTTP 200, 109 car. | HTTP 200, 109 car. |
| `adaptive_target` 0,5 / `decay` 0,9 | HTTP 200, 103 car. | HTTP 200, 96 car. |
| `reasoning_budget_tokens` 8 (Qwen3.5-9B, réflexion) | HTTP 200, sans effet sur ce couple | 8 tokens de raisonnement, `length` |

Les tirages ne sont pas identiques d'un serveur à l'autre (RNG différents) : la preuve
porte sur l'effet des champs, pas sur l'égalité des tokens.

## 6. Ce qui n'est pas dans ce journal

- Les listes exhaustives (159 architectures, 30 routes, 68 prétokenizers) : elles
  restent dans les annexes du document, source unique.
- Les estimations de gain du document : elles sont des hypothèses jusqu'à mesure.
- Les chiffres de performance détaillés : `llama-kinetic.md` §4.13 et §7.3, avec leur
  protocole.

## 7. Tenue

- Une entrée par lacune, dans l'ordre ci-dessus ; sauter une entrée demande une raison
  écrite ici.
- Un chiffre sans commande, commit et date n'entre pas.
- Une régression > 3 % annule l'essai.
- Une lacune est close par sa preuve, pas par son code mergé.
