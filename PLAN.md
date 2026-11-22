# PLAN — journal de comblement des lacunes de Kinetic face à llama.cpp

> Compagnon de `llama-kinetic.md` (§4 matrice, §6 lots, §7 modules, §10 annexes).
> Le document dit **où** Kinetic est en retard et dans quel ordre attaquer ; ce journal
> dit **ce qui a été comblé**, avec quelle preuve, et ce qui reste.
> Une entrée n'est close que par une preuve : une capacité de llama.cpp qui réussit
> maintenant dans Kinetic, sur la même machine et le même fichier.

---

## 1. Ce qu'on comble

La cible est la liste des capacités de llama.cpp, pas la liste de nos modèles. Les
conséquences — charger `qwen3moe`, `qwen3vl`, `tekken`, un PTQ1_0, et s'approcher des
débits de llama — découlent du comblement, elles ne le remplacent pas.

**Preuve d'une lacune comblée**, selon le domaine :

| Domaine | Preuve exigée |
|---|---|
| Format / backend | `kinetic info` + `kinetic run` sur le même GGUF que llama, sur le GPU (jamais de repli CPU), sortie non dégénérée |
| Architecture | idem, plus `kinetic info` qui nomme la famille, et le prompt de contrôle |
| Performance | même fichier, même backend, `kinetic bench --prompt` contre `llama-bench`, un harnais par comparaison |
| Tokenizer | `kinetic tokenize` (ou l'équivalent) donne les mêmes ids que `llama-tokenize` sur un corpus de contrôle |
| API / serveur | la requête llama.cpp passe sans modification, champs inconnus ignorés ou refusés clairement |
| Outillage | la commande équivalente existe et produit un artefact comparable (`quantize`, `imatrix`, `split`) |

Une entrée close sans preuve reproductible est rouverte.

## 2. Machine et état

| Élément | Valeur (relevé du 22 nov. 2026) |
|---|---|
| CPU | AMD Ryzen 7 7800X3D, 8 cœurs / 16 threads, AVX2, FMA, F16C |
| GPU | AMD Radeon RX 9070 XT, `gfx1201` (RDNA 4), 64 CU, 16 Gio |
| iGPU | AMD Radeon 610M, `gfx1036`, 2 CU, mémoire partagée |
| Vulkan | 1.4.354, RADV / Mesa 26.1.6, sous-groupe 32–64 |
| ROCm | 7.2.0, HIP 7.2, `hipcc` ciblant `gfx1201` |
| CUDA | absent |
| llama.cpp de référence | `build-vulkan/bin/llama-bench` (Vulkan0) |

Dépôt : fork `Angra974/kinetic`, remote `fork`, branche unique `main`. `upstream`
(`Veloresearch/kinetic`) est l'amont, en lecture seule.

Matrice de capacités de référence : `llama-kinetic.md` §4 (architectures, tokenizers,
formats, backends, serveur, sampling, spéculatif, outils, bindings, modalités).

## 3. Protocole

- Un backend par comparaison, jamais de mélange (§7.10 du document).
- Protocole des essais de performance : `python3 scripts/bench_protocol.py '<modèle>'
  --llama-bin <llama-bench>` pour le préfill, `kinetic bench -r 5` pour le décodage.
- Exactitude : `cargo test --release -p kinetic-vulkan` — référence f64 à 1e-5
  (`check`), égalité de bits mono-token / passe prompt (`check_prompt_pass`), garde-fou
  `every_gemv_type_survives_a_prompt_pass`.
- Un changement par essai, contrôle → candidat → contrôle. Un candidat n'est gardé que
  s'il comble la lacune visée **sans** régresser ce qui marchait (> 3 % = régression).

---

## 4. Lacunes closes

### L1 — Types IQ dans les noyaux Vulkan (formats)

- **Lacune** : llama.cpp charge `IQ3_XXS`, `IQ4_XS`, `Q4_0` sur Vulkan ; Kinetic refusait
  le modèle (`Vulkan has no kernel for IQ3_XXS`) et retombait sur le CPU.
- **Travail** : layouts planaires, un noyau GEMV par type (commits `0127427`, `85d29aa`).
- **Preuve** : `Qwen3.8-27B-IQ3_XXS` tourne sur `AMD Radeon RX 9070 XT (Vulkan)`,
  sortie cohérente ; tests 6 + 20 + 1, dont références f64 et passe prompt bit-exacte.
- **Effet mesuré** : CPU 3,8 t/s pp512 / 0,6 tg128 → Vulkan 154,8 / 38,1.
- **Statut** : close.

### L2 — Passe de prompt sur ces types (performance)

- **Lacune** : llama.cpp fait 1154 t/s de préfill sur ce fichier ; Kinetic en était à
  154,8, soit 7,5× derrière.
- **Travail** : noyaux many-token (`gemv_many.wgsl`, 32 tokens par passe, `PROMPT_BATCH`
  128) pour les trois types (`85d29aa`).
- **Preuve** : toujours bit-exact, `tg128` inchangé.
- **Effet mesuré** : pp512 154,8 → 261,9 ; pp2048 152,3 → 257,9. Reste 4,4× derrière
  llama (~23 % de son préfill).
- **Statut** : close pour l'écart de méthode (GEMV par ligne ×32 au lieu d'un GEMV par
  token) ; **l'écart résiduel de 4,4× reste ouvert** et devient L3.

---

## 5. File d'attente, par lacune

Ordre : d'abord ce qui débloque une classe entière de fichiers avec une preuve rapide,
ensuite ce qui demande un backend.

### L3 — Préfill quantifié (performance)

- **Lacune** : 261,9 t/s contre 1154,2 sur le même fichier et le même backend.
- **Hypothèse** : les poids sont relus une fois par ligne, sans tuile en mémoire
  partagée ni split-K ; la passe many-token garde 32 accumulateurs scalaires par
  invocation et refait ses sommes par sous-groupe de 32.
- **Cible** : pp512 ≥ 390 t/s (×1,5) sans toucher au décodage, bit-exact.
- **Note** : un GEMM tuilé qui change l'ordre des sommes casse `check_prompt_pass` ;
  la tuile doit préserver l'ordre par ligne, sinon l'essai est rejeté, pas toléré.

### L4 — MoE (architectures)

- **Lacune** : llama.cpp sert `qwen3moe`, `qwen2moe`, `llama4`, `deepseek2` ; Kinetic n'a
  ni routage top-k ni GEMV groupé, et `qwen3-coder-30b-a3b` retombe sur le CPU.
- **Preuve** : `qwen3-coder-30b-a3b` sur GPU, sortie cohérente, pp/tg comparés.
- **Pourquoi ici** : c'est la lacune d'architecture qui couvre le plus de fichiers
  réellement présents, et elle est mesurable sur cette machine.

### L5 — Prétokenizers (tokenizers)

- **Lacune** : `tekken` (et 14 autres du §6.2) absents ; `helcyon-gpt-4o-v4.5` ne charge
  pas alors que son format est déjà couvert.
- **Preuve** : mêmes ids que `llama-tokenize` sur un corpus de contrôle.

### L6 — Quantification de production (outillage)

- **Lacune** : llama.cpp produit IQ/NVFP4/FP8 avec imatrix ; Kinetic ne produit que
  Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0, et ne sait pas *encoder* IQ3_XXS/IQ4_XS.
- **Preuve** : un fichier produit par `kinetic quantize` se charge dans llama.cpp et
  dans Kinetic avec le même `kinetic info`.

### L7 — BF16 sur Vulkan, NVFP4/F8 (formats)

- **Lacune** : `paddleocr-vl-1.6` est en BF16 ; RADV rapporte `bf16: 0`, donc à valider
  avant d'annoncer. NVFP4 (type 40) et F8_E4M3 (type 43) manquent partout.

### L8 — API et sampling (serveur)

- **Lacune** : `ignore_eos`, `min_keep`, `dynatemp_*`, `adaptive_*`, `reasoning_budget` ;
  routes `/infill`, `/apply-template`, `POST /props`, `POST /slots/:id_slot`,
  `/lora-adapters`, `n > 1`.
- **Preuve** : requêtes de référence sur `llama-server` rejouées contre Kinetic.

### L9 — Backend HIP (backend)

- **Lacune** : llama.cpp a ROCm0 et Vulkan0 ; Kinetic n'a que Vulkan et le CPU.
- **Place** : après L3. Le décodage Vulkan est déjà à 95 % de llama Vulkan ; un backend
  complet coûte L/XL et ne rendrait que l'écart de préfill que L3 attaque pour moins cher.

---

## 6. Journal des essais

| # | Lacune | Commit | Mesure | Verdict |
|---|---|---|---|---|
| L1 | types IQ | `0127427` | 3,8 → 154,8 t/s pp512 | close |
| L2 | préfill IQ | `85d29aa` | 154,8 → 261,9 t/s pp512 | close (écart 4,4× → L3) |
| L3 | préfill tuilé | — | — | à faire |

Détail des essais L1 et L2 : voir l'historique `git log` des commits cités et
`llama-kinetic.md` §4.13. Les CSV de mesures (`bench-results.csv`, `bench-many.csv`)
sont restés dans l'ancien répertoire `/home/mrnemesys/Devs/kinetic-work`, non suivis.

## 7. Règles de tenue

- Une entrée par lacune, dans l'ordre de la file ; on ne saute pas L3 pour aller à L5
  sans raison écrite.
- Un chiffre sans commande, commit et date n'entre pas ici.
- Une lacune comblée sans preuve reproductible est rouverte.
- Une régression de plus de 3 % sur un cas déjà couvert annule l'essai.
