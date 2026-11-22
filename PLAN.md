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
- **MoE : fait (voir G5).** Routage top-k et GEMV groupé vivent sur le GPU ;
  restent à optimiser : pp512 à 489 t/s contre 2637 pour llama, tg128 à 55,7
  contre 182 sur le même fichier. Le GEMV groupé lit les experts d'un slot à
  la fois et le préfill n'a pas de GEMM tuilé, d'où l'écart.
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

**Fait (voir G5) :** `qwen3moe` et la brique MoE Vulkan, qui débloquent
`qwen3-coder-30b-a3b`.

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
| G5 | `qwen3moe` refusé au chargement, experts routés et IQ3_S absents du Vulkan : le 30B A3B retombait sur le CPU | `6141b3b`, `7714f86`, `c729c30` | même fichier chargé sur Vulkan, 1er token identique à la CPU, sorties cohérentes ; passe routée comparée au host sur 3 tokens et 3 types d'experts ; 91 tests modèles + 21 GEMV + 1 MoE verts | le modèle tourne sur GPU : pp512 489 t/s, tg128 55,7 t/s |
| G6 | `qwen35moe` refusé sur Vulkan (expert partagé, routage sigmoïde) et experts Q4_K/Q6_K sans noyau groupé : le 35B A3B retombait sur le CPU | `5457f06` | 6 tests MoE contre le host (sigmoïde gated/plain, Q4_K, Q6_K) et 34 tests `kinetic-vulkan` verts ; le fichier se charge et génère sur Vulkan | génération 11,9 → 26,8 t/s, prompt 16,5 → 91 t/s |
| G7 | Un modèle plus gros que la VRAM déversait en RAM système tout ce qu'il téléversait en dernier, tête de sortie et poids denses compris, relus à chaque token par le bus | `7598ca2` | profil par timestamps GPU : tête 16,4 → 1,45 ms par token, couches 29-40 à 2,5 ms contre 0,66 ms pour les précédentes ; même sortie au token près ; 34 tests `kinetic-vulkan` verts | génération 25,7 → 33,5 t/s, préfill 92,6 → 92,9 t/s |
| G8 | Aucune découpe GPU/CPU sur Vulkan : `--gpu-layers` y était accepté et ignoré, et un modèle plus gros que la VRAM relisait ses couches par le PCIe | `2c6c03b` | `kinetic bench --gpu-layers 31` : 9 des 40 couches sur le CPU, sortie cohérente, 34 tests `kinetic-vulkan` verts, défaut inchangé au token près | découpe mesurée plus lente que le placement en place : tg128 16,8 contre 30,8 t/s, pp512 167 contre 205 ; elle reste optionnelle |
| G4 | Prétokenizer `tekken` absent : `helcyon-gpt-4o-v4.5` refusé au chargement | non committé | mêmes ids que `llama-tokenize` sur 26 lignes de contrôle et sur un fichier entier (83 tokens, identiques), 22 nov. ; 28 tests tokenizer verts ; génération Vulkan 77,3 t/s | le modèle se charge et génère |

G1 et G2 ferment une partie de **§4.3 formats** et amorcent **P0.2** ; G3 ferme la
partie sampling de **P0.1**, pas ses routes ; G4 ferme la partie `tekken` de **P0.5**.

### Mesures de G6 (22 nov. 2026)

`qwen3.8-35b-a3b` Q4_K_M (20,22 Gio) : 31 des 41 couches sur une RX 9070 XT de
16 Gio, un seul job GPU à la fois, même prompt (« Say hi in five words. ») et
40 tokens générés à température 0 ; llama.cpp `19bc41dbb`.

| Test | llama-server `-ngl 31` | kinetic `--gpu-layers 31` |
|---|---:|---:|
| génération | 39,2-40,7 t/s | 25,2-27,6 t/s |
| prompt, 18 tokens | 82,2 t/s | 86,2-89,3 t/s |

kinetic sur CPU pour la même requête : 11,9 t/s en génération, 16,5 t/s en
prompt. Le GPU double le débit de Kinetic et le préfill dépasse llama, mais le
décodage reste 1,5× sous llama.

Le balayage `--gpu-layers` 24 → 41 alors consigné (26,3 à 24,4 t/s, « plateau »)
ne mesurait rien : le backend Vulkan ne lit pas ce drapeau, place les 40 couches
d'office, et seule la taille de la VRAM décide de ce qui déborde. Les 2 t/s
d'écart entre les passes étaient du bruit. Le décodage est borné par ce que le bus
relit à chaque token, ce que G7 a mesuré par timestamps GPU puis corrigé par le
placement des poids.

### Profil de décodage du 35B (22 nov. 2026)

Où part le temps d'un token de `qwen3.8-35b-a3b` sur Vulkan, mesuré par
timestamps GPU (deux sondes : après les 41 couches, après la tête), même
prompt, 40 tokens à température 0, `-c 8192`.

| Poste | Avant | Après |
|---|---:|---:|
| tête de sortie, 248 320 × 2048 Q6_K | 16,4 ms | 1,45 ms |
| couches 1-28 | 0,66 ms par couche | 0,66 ms par couche |
| couches 29-40 | 2,5 ms par couche | — |
| génération | 25,7 t/s | 33,5 t/s |
| préfill, 18 tokens | 92,6 t/s | 92,9 t/s |

La cause n'est pas le calcul : 20,6 Go de poids pour 16 Go de VRAM, et
`Allocation::new` déverse en RAM système (`HOST_SPILL`) ce qui ne tient plus,
dans l'ordre du téléversement. La tête était donc téléversée en dernier après
les 41 couches, et les 12 dernières couches — tête de sortie comprise —
étaient relues à chaque token par le PCIe (`mem_info_vram_used` culminait à
16 202 Mio). Deux corrections de placement, sans toucher aux noyaux :

- la tête et sa norme prennent leur place avant les couches ;
- tous les poids denses (attention, expert partagé, routeur, normes) sont
téléversés avant les experts routés, qui prennent le débordement : un token
  en lit un sur `n_experts`, contre la totalité des poids denses.

`DeviceExperts` gagne l'allocation et la copie séparées (`reserve`, `fill`)
pour permettre ce second passage ; la copie d'un expert est inchangée. Le
préfill ne bouge pas : il est borné par le calcul, pas par le bus.

Ce qui reste : les experts hébergés en RAM système coûtent encore leur lecture
PCIe à chaque token. Les exécuter sur le CPU pour les couches concernées, comme
le fait llama.cpp avec `-ngl`, devait remplacer ce bus par la RAM à pleine
bande ; G8 a implémenté cette découpe et l'a mesurée : elle est plus lente que
le placement en place, la couche CPU de Kinetic coûtant plus cher que la
pénalité PCIe qu'elle évite.

### Mesures pp512 / tg128 du 35B (22 nov. 2026)

`kinetic bench --device vulkan -r 1` contre
`llama-bench -dev Vulkan0 -ngl 31 -p 512 -n 128 -r 1`, même fichier, un seul
job GPU à la fois, deux passes de chaque côté.

| Test | llama-bench `-ngl 31` | kinetic `bench` |
|---|---:|---:|
| pp512 | 642,67 t/s | 198,0-202,4 t/s |
| tg128 | 43,33 t/s | 29,9-30,2 t/s |
| pic VRAM | 15 981 Mio | 16 185 Mio |

Les deux configurations ne sont pas identiques, faute d'équivalent : llama garde
31 couches sur le périphérique et calcule les 10 autres, experts compris, sur le
CPU ; kinetic n'a pas de découpe de couches sur Vulkan, met les 40 sur le
périphérique et laisse 4,4 Go déborder en RAM système, relus par le bus une fois
par passe. `kinetic bench` n'a donc pas de `--gpu-layers` à proposer : le backend
Vulkan ne le lit pas (`kinetic run` l'accepte et l'ignore de même), c'est le
`HOST_SPILL` qui décide de ce qui reste en VRAM.

La passe many-token mesurée ici (512 tokens) nuance la lecture du prompt court de
G6 : à 18 tokens kinetic était devant (91 contre 82 t/s), mais à 512 il est 3,2×
sous llama. Le préfill est borné par le calcul, et c'est le résidu P0.2 des noyaux
many-token, pas le placement des poids — le débordement ne coûte qu'une lecture
par passe, pas par token.

### Découpe GPU/CPU du 35B (22 nov. 2026)

La découpe de llama.cpp, portée sur Vulkan : `--gpu-layers N` (défaut : toutes)
place les `N` dernières couches sur le périphérique, les premières tournent sur
le CPU avec les mêmes opérations que le chemin CPU, et seuls `n_embd` flottants
par token traversent. `kinetic bench --gpu-layers N` la mesure.

| Test | llama-bench `-ngl 31` | kinetic défaut | kinetic `--gpu-layers 31` |
|---|---:|---:|---:|
| pp512 | 642,67 t/s | 205,46 t/s | 167,11 t/s |
| tg128 | 43,33 t/s | 30,84 t/s | 16,83 t/s |
| génération, `run`, 18 tokens de prompt | 39,2-40,7 t/s | 33,6 t/s | 20,8 t/s (5 tokens, EOS) |

La découpe perd : 9 couches sur le CPU coûtent environ 4,3 ms par token chacune,
quand la même couche hébergée en RAM système ne coûte que sa lecture PCIe, de
l'ordre de 1,8 ms. llama.cpp ne gagne avec `-ngl` que parce que sa couche CPU lui
coûte bien moins, autour de 0,8 ms. Le préfill perd aussi (167 contre 205 t/s),
les couches hôte n'ayant pas de noyau many-token côté CPU.

Conséquence : la découpe est livrée mais **optionnelle**, le défaut reste le
placement de G7 — les 40 couches sur le périphérique, les experts prenant le
débordement —, au même débit et au token près. Sous `--gpu-layers`, la sortie
diverge de celle du défaut, comme entre les chemins CPU et Vulkan : les couches
hôte calculent en f32 sur le CPU, le périphérique non.

### Mesures de G5 (22 nov. 2026)

`qwen3-coder-30b-a3b-instruct` UD-IQ3_XXS (11,97 Gio), un seul job GPU à la fois,
`kinetic bench --device vulkan --prompt 512 -r 2` contre
`llama-bench -dev Vulkan0 -p 512 -n 128 -r 2` (llama.cpp `19bc41dbb`).

| Test | llama Vulkan0 | kinetic Vulkan |
|---|---:|---:|
| pp512 | 2637,4 t/s | 489,0 t/s |
| tg128 | 181,9 t/s | 55,7 t/s |

CPU kinetic pour la même requête greedy : 8,4 t/s en génération. Le GPU est donc
6,6× plus rapide que le CPU de Kinetic, mais reste 3,3× sous llama en décodage et
5,4× en préfill : c'est le chantier P0.2 (GEMM tuilé, GEMV groupé par plusieurs
tokens) qui reste ouvert, pas un repli CPU.

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
