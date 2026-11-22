# Kinetic face à llama.cpp — analyse de parité et modules complémentaires

> **Statut** : analyse. Aucun code de Kinetic n'a été modifié ; les mesures citées au
> §4.13, §5.6, §7.3 et §8 viennent de la branche de travail et sont reproductibles.
> **Snapshot** : Kinetic `f3723b1` (19 nov. 2026) et sa branche de travail `kinetic-work`
> (commits `0127427` et `85d29aa`, 22 nov. 2026), llama.cpp `19bc41dbb` (21 nov. 2026).
> Les tableaux de types, de modèles et de performance portent leur commit de relevé.
> **But** : déterminer ce qui manque à Kinetic pour couvrir 100 % des capacités de
> llama.cpp, en conservant l'architecture de Kinetic, puis proposer les modules
> complémentaires qui donnent une performance égale ou supérieure.

---

## 0. Lecture rapide

Kinetic n'est pas un clone de llama.cpp. C'est un moteur volontairement étroit,
optimisé pour un usage précis : un seul modèle, un serveur local, des architectures
récentes, une exactitude vérifiable et des accélérateurs automatiques (RAD-1,
Warm Start, Zero Wait, Shortlist, Flywheel, Convoy). llama.cpp est une plateforme
généraliste : ~180 architectures, une quinzaine de familles de backends, le support
vision/audio/vidéo/OCR/TTS/diffusion, l'entraînement, le RPC, une API C complète et
un écosystème de bindings.

La couverture « 100 % » se joue donc sur quatre contrats distincts :

1. **Contrat modèle** — tout GGUF que llama.cpp charge, Kinetic le charge ou explique
   précisément pourquoi il ne le charge pas.
2. **Contrat d'exécution** — chaque matériel majeur a un backend Kinetic au moins
   aussi rapide que llama.cpp sur les charges courantes.
3. **Contrat d'API** — chaque route et chaque champ de llama-server est implémenté,
   ou renvoie une erreur documentée (501) au lieu d'un 404 silencieux.
4. **Contrat d'outillage** — quantize/imatrix/split/tokenize/bench/export-lora/
   bindings couvrent les mêmes workflows.

Estimation de couverture actuelle, pondérée par l'usage réel plutôt que par le nombre
de lignes de code :

| Domaine | Couverture | Commentaire |
|---|---:|---|
| Architectures (noms) | ~12 % | 21 noms sur 180 ; ~55–65 % pondéré par les modèles réellement téléchargés |
| Formats de poids (runtime) | ~92 % | tous les types courants sauf NVFP4 et F8_E4M3 ; BF16 CUDA manquant ; Vulkan à 9 types depuis `0127427` (F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ4_XS) |
| Quantification (production) | ~25 % | 5 cibles seulement, pas d'imatrix, pas d'IQ |
| Backends (matériel) | ~35–45 % | x86 AVX2, CUDA, Vulkan ; **pas de HIP/ROCm (priorité locale)**, SYCL, Metal natif, CANN, MUSA, WebGPU, RPC |
| Serveur (routes) | ~45 % | OpenAI et llama-server de base ; long tail absent |
| Sampling | ~85 % | manque dynatemp, min_keep, ignore_eos, adaptive, reasoning budget |
| Décodage spéculatif | ~60 % | excellent en automatique, pas de draft model/EAGLE3/DFlash |
| Outils CLI | ~30 % | pas d'imatrix, split, export-lora, tokenize, TTS, RPC |
| Bindings | ~15 % | C minimal, Python partiel, pas d'écosystème |
| Modalités | ~5 % | 2 projecteurs vision, pas d'audio/vidéo/OCR/TTS |

**Conclusion** : la parité littérale est un chantier pluriannuel et une cible mouvante
(llama.cpp ajoute des architectures chaque semaine). La bonne stratégie n'est pas de
recopier llama.cpp, mais de **définir un contrat de parité par domaine**, de combler
d'abord les 20 % qui couvrent 80 % des usages, puis d'ajouter les modules de
performance là où llama.cpp est aujourd'hui devant. Les sections 6 et 7 donnent ce
plan ; la section 10 donne les listes exhaustives.

### 0.1 Les vingt manques qui comptent le plus

1. **ROCm/HIP** : backend complet, priorité locale (runtime, GEMV quantifié, WMMA,
   flash, hipGraph).
2. **Vulkan** : GEMM de prompt, coopmat, KV quantifié, MoE, layer split (priorité
   locale). Les types IQ3_XXS/IQ4_XS/Q4_0 sont faits (`0127427`, `85d29aa`), donc
   l'écart restant est le préfill, pas le chargement.
3. **NVFP4 + F8_E4M3** runtime, **BF16/FP8** sur ROCm/Vulkan, **imatrix** et cibles
   **IQ** dans `kinetic quantize`.
4. **DeepSeek V2/V3/V3.2/V4** (MLA, DSA, MoE, MTP).
5. **Qwen2MoE, Qwen3MoE, Qwen3-Next, Qwen3-VL**.
6. **Llama 4** (iRoPE + MoE + vision).
7. **GLM4, GLM4-MoE, GLM-DSA, GLM5**.
8. **Mistral 3/4 + Pixtral**.
9. **Phi3/PhiMoE, Gemma 2, GPT-OSS**.
10. **Mamba/Mamba2/Jamba/RWKV6-7/Nemotron-H**.
11. **T5/T5-Encoder** et tokenizer **UGM**.
12. **Projecteurs vision** : Qwen2-VL/2.5-VL/3-VL, InternVL, Llama4, Pixtral,
    Gemma3, SmolVLM, MiniCPM-V.
13. **Audio** : Whisper/Voxtral/Qwen3-ASR en entrée, Kokoro/Zonos3/PocketTTS en sortie.
14. **OCR** : DeepSeek-OCR, PaddleOCR.
15. **Sampling** : `ignore_eos`, `min_keep`, `dynatemp_*`, `adaptive_*`,
    `reasoning_budget`.
16. **Routes serveur** : `/infill`, `/apply-template`, `POST /props`,
    `POST /slots/:id_slot`, `/lora-adapters`, `n > 1`.
17. **API Responses et Anthropic Messages**.
18. **Multi-GPU row/tensor split**, `--override-tensor`, cache MoE explicite.
19. **SYCL, Metal natif, CANN, MUSA, WebGPU, RPC**.
20. **ABI C/Python complètes** + 15 prétokenizers prioritaires.

Les **quick wins** réels (effort S/M, impact immédiat) sont 15, 16 et une partie de
20 : ils ne demandent pas de nouvelle architecture de code et débloquent des workflows
entiers. Les points 1 et 2 sont des déblocages à fort impact, mais demandent un travail
L/XL (backend HIP, kernels Vulkan).

### 0.2 Machine de référence : AMD, Vulkan et ROCm — pas de CUDA

Cette analyse est priorisée pour la machine locale, pas pour une machine NVIDIA
hypothétique. Inventaire relevé le 22 novembre 2026 :

| Élément | Valeur |
|---|---|
| CPU | AMD Ryzen 7 7800X3D, 8 cœurs / 16 threads, AVX2, FMA, F16C ; Zen 4 supporte aussi AVX-512, non exploité par Kinetic |
| RAM | 30,5 Gio, ~21 Gio libres |
| GPU | AMD Radeon RX 9070 XT, `gfx1201` (RDNA 4), 64 CU, 16 Gio, wavefront 32 |
| iGPU | AMD Radeon 610M, `gfx1036`, 2 CU, mémoire partagée |
| Vulkan | 1.4.354, RADV / Mesa 26.1.6, sous-groupe par défaut 64, plage 32–64, `VK_KHR_cooperative_matrix`, FP16, int8 dot, pas de BF16 ni FP4 natif |
| ROCm | 7.2.0, HIP 7.2, `hipcc` ciblant `gfx1201` |
| CUDA | **absent** : aucun GPU NVIDIA, `nvidia-smi` inexistant, Kinetic compilé sans CUDA |
| Builds llama.cpp | `build-hip/bin/llama-bench` (ROCm0) et `build-vulkan/bin/llama-bench` (Vulkan0) |

Conséquences directes pour ce document :

- **`kinetic-hip` (ROCm) est un P0 local**, pas un P2. C'est le backend qui peut
  exploiter les 64 CU et la bande passante GDDR6 sans passer par RADV.
- **Vulkan est le backend local actuel** : il doit d'abord combler ses manques
  structurels (GEMM de prompt, coopmat, types IQ, MoE, layer split) avant d'être
  comparé à llama.cpp.
- **CUDA reste dans le périmètre de parité** pour les utilisateurs NVIDIA, mais il ne
  doit pas bloquer les priorités locales. Les tableaux ci-dessous le marquent
  explicitement comme secondaire.
- Les comparaisons de performance doivent opposer, sur le même fichier :
  `kinetic --device vulkan` à `llama-bench` **Vulkan0**, puis `kinetic` HIP (quand il
  existera) à `llama-bench` **ROCm0**. Un chiffre CUDA n'a aucune valeur ici.

La machine expose aussi un iGPU `gfx1036` : il ne faut pas le compter comme un second
GPU de calcul sérieux (2 CU, mémoire système). Il peut servir de cible de layer split
pour les modèles qui ne tiennent pas en VRAM, pas de cible de tensor split.

---

## 1. Ce que « 100 % » veut dire

### 1.1 Parité fonctionnelle

Une capacité est « à parité » si un utilisateur de llama.cpp peut faire la même chose
avec Kinetic, avec le même fichier, la même ligne de commande ou le même client HTTP,
et un résultat équivalent. Cela inclut :

- charger le même GGUF et obtenir des tokens équivalents (pas nécessairement
  bit-à-bit avec llama.cpp, mais équivalents au sens greedy/seed) ;
- les mêmes options de quantification et de découpage ;
- les mêmes routes serveur et les mêmes champs JSON ;
- les mêmes modes de placement mémoire et multi-GPU ;
- les mêmes modalités (image, audio, vidéo, OCR, TTS) ;
- les mêmes bindings.

### 1.2 Parité de performance

Une capacité est « à performance égale » si, sur le même matériel et le même fichier,
Kinetic est dans la même classe de vitesse que llama.cpp (pp/tg, TTFT, débit
multi-requêtes, mémoire pic). « Supérieure » signifie qu'un module Kinetic apporte un
gain mesurable et reproductible, pas une victoire anecdotique.

### 1.3 Ce qui n'est pas un objectif

- **Bit-à-bit avec llama.cpp** : les ordres de sommation, les kernels et les
  arrondis diffèrent. Kinetic garantit l'exactitude *vis-à-vis de lui-même* (batch
  invariant, greedy identique avec/sans draft). Il faut conserver cette promesse et
  documenter qu'un même prompt peut différer de llama.cpp.
- **Remplacer les différenciateurs de Kinetic** par des flags optionnels de type
  llama.cpp. RAD-1, Warm Start, Zero Wait, Shortlist, Flywheel et Convoy doivent
  rester automatiques par défaut ; les modes llama.cpp s'ajoutent *en plus*, pas *à la
  place*.
- **Télémetrie, réseau, comptes** : Kinetic reste local. Les outils de type MCP/server
  tools devront être explicitement locaux et sans appel sortant.

---

## 2. Méthode et chiffres de référence

### 2.1 Sources inspectées

| Source | Ce qui a été comparé |
|---|---|
| `kinetic/docs/src/models.md` | architectures, formats, tokenizers, projecteurs |
| `kinetic/docs/src/devices.md` | CPU, CUDA, Vulkan, multi-GPU, offload |
| `kinetic/docs/src/server.md` | API OpenAI, llama-server, Ollama |
| `kinetic/docs/src/cli.md`, `llama-flags-table.md` | commandes et flags |
| `kinetic/docs/src/how-it-is-fast.md` | RAD-1, Warm Start, Zero Wait, Shortlist, Flywheel |
| `kinetic/crates/kinetic-models/src/arch/mod.rs` | registre `FAMILIES` |
| `kinetic/crates/kinetic-models/src/blocks/mod.rs` | briques réutilisables |
| `kinetic/crates/kinetic-models/src/speculative/` | draft, lookup, MTP, sampler |
| `kinetic/crates/kinetic-tokenizer/src/pretokenize.rs` | prétokenizers |
| `kinetic/crates/kinetic-quant/src/types.rs` | types GGUF |
| `kinetic/crates/kinetic-cli/src/quantize.rs` | cibles de quantification |
| `kinetic/crates/kinetic-server/src/llama.rs`, `server.rs` | routes réellement servies |
| `kinetic/crates/kinetic-capi/include/kinetic.h` | ABI C |
| `llama.cpp/src/llama-arch.cpp` | 180 noms d'architecture (302 entrées avec alias/dummies) |
| `llama.cpp/src/llama-vocab.h` | 68 prétokenizers |
| `llama.cpp/common/common.h` | sampling, spéculatif, multimodal, serveur |
| `llama.cpp/tools/server/server.cpp` | ~45 routes/alias |
| `llama.cpp/tools/mtmd/models/` | 73 fichiers de projecteurs/modèles multimodaux |
| `llama.cpp/ggml/src/` | 18 familles de backends |
| `llama.cpp/tools/` et `examples/` | outils et workflows |
| `lspci`, `rocminfo`, `vulkaninfo`, `hipcc --version` | matériel local, ROCm 7.2, Vulkan 1.4 RADV, gfx1201 |
| `llama.cpp/build-hip`, `build-vulkan` | backends réellement disponibles sur la machine |
| `kinetic-work`, commits `0127427` et `85d29aa` | noyaux Vulkan IQ3_XXS/IQ4_XS/Q4_0 (mono-token et many-token) et mesures du 22 nov. |

### 2.2 Conventions de comptage

- **Architectures llama.cpp** : 180 valeurs `general.architecture` réelles, hors
  `(unknown)`. Les 302 entrées de `llama-arch.cpp` incluent les alias, les architectures
  réservées et les entrées internes (`clip`, `eagle3`, `dflash`, etc.).
- **Architectures Kinetic** : 21 noms = 18 décodeurs + 3 encodeurs. Le registre
  `FAMILIES` contient 10 `Family`, mais certaines couvrent plusieurs noms
  (`qwen35`/`qwen35moe`, `lfm2`/`lfm2moe`, `granite`/`granitehybrid`/`granitemoe`).
- **Projecteurs** : 2 types (`qwen3vl_merger`, `gemma4v`) contre 73 fichiers mtmd,
  dont certains sont des encodeurs audio, des TTS ou des VAE.
- **Routes** : compte des enregistrements HTTP, alias compris. Kinetic en sert une
  quinzaine, llama.cpp une quarantaine.

### 2.3 Limites de l'analyse

- Les listes d'architectures évoluent vite ; les chiffres sont un instantané.
- Certaines architectures classées « proches » ont des variantes de RoPE, de MoE ou de
  normalisation qui demandent un fichier dédié. Le classement est indicatif.
- Les estimations d'effort sont en tailles (S/M/L/XL), pas en jours-homme.
- Les performances citées sont des ordres de grandeur à vérifier sur le matériel cible ;
  la section 7.10 propose le protocole de mesure.
- Les seuls chiffres locaux inclus sont ceux du 22 nov. (§4.13, §5.6, §7.3, §8), pris
  machine idle avec `llama-bench` Vulkan0 et `kinetic bench --prompt`, sur des commits
  nommés. Toute nouvelle mesure doit suivre la même règle : un harnais et un backend
  par comparaison, jamais CUDA ou ROCm pour commenter un chiffre Vulkan.
- L'arbre Kinetic contient des modifications non commitées (Vulkan, CLI) : les constats
  de code reflètent l'arbre de travail au 22 novembre, pas seulement le commit
  `f3723b1`.

---

## 3. Forces de Kinetic à préserver

Ces points sont des différenciateurs réels. Toute implémentation de parité doit être
additive et ne pas les dégrader.

| Force | Ce qu'elle apporte | Risque si on la remplace par le modèle llama.cpp |
|---|---|---|
| **RAD-1** | Décodage spéculatif exact, actif par défaut, sans modèle draft : lookup contexte, index inter-requêtes, bloc MTP, arbres, rejet exact, longueur adaptative | Perte du « zéro configuration » et de l'exactitude par construction |
| **Warm Start** | Cache de préfixe persistant, snapshots pour les couches récurrentes, LRU disque, reprise bit-à-bit | Perte du cache inter-redémarrage et de la reprise des modèles SSM |
| **Zero Wait** | Préremplissage pendant que l'utilisateur tape, via `/v1/prefill` | Perte de la latence perçue |
| **Shortlist** | Ne lit que les lignes de la couche de sortie qui peuvent gagner | Perte du gain sur les gros vocabulaires |
| **Flywheel** | Offload automatique des experts MoE, les experts chauds restent sur GPU | Perte du réglage automatique |
| **Convoy** | Mutualise les passes entre requêtes concurrentes, avec exactitude | Perte du débit multi-requêtes |
| **Exactitude batch** | Une batch donne les mêmes logits que token par token ; un arbre respecte chaque chemin | Perte de la garantie qui rend RAD-1 sûr |
| **`.ktc` signé** | BLAKE3 + Ed25519, manifeste, receipts rejouables | Perte de la chaîne de confiance |
| **UX mono-binaire** | `doctor`, `fit`, `pull`, `verify`, `verify-receipt`, `compare` | Perte de la simplicité |

**Règle d'or** : un nouveau backend ou un nouveau sampler doit soit préserver les
invariants d'exactitude, soit être explicitement opt-in et documenté comme changeant
les sorties.

---

## 4. Matrice de capacités

Légende : **P** = parité à faire ; **S** = module de performance ; **D** =
différenciateur à conserver.

### 4.1 Architectures

#### Déjà supportées (21 noms)

`qwen35`, `qwen35moe`, `lfm3`, `lfm2`, `lfm2moe`, `llama`, `qwen2`, `qwen3`,
`gemma3`, `gemma4`, `ernie4_5`, `hunyuan-dense`, `exaone4`, `olmo2`, `granite`,
`granitehybrid`, `granitemoe`, `falcon-h1`, `bert`, `nomic-bert`,
`gemma-embedding`.

#### Familles manquantes, par lot d'implémentation

| Lot | Architectures llama.cpp | Briques nécessaires | Effort |
|---|---|---|---|
| **Transformers proches** (réutilisent `arch/transformer.rs`) | `qwen`, `qwen2moe`, `qwen3moe`, `qwen4exp`, `deepseek`, `phi2`, `phi3`, `phimoe`, `gemma`, `gemma2`, `gemma4-assistant`, `olmo`, `olmoe`, `openelm`, `stablelm`, `bloom`, `gpt2`, `gptj`, `gptneox`, `mpt`, `baichuan`, `starcoder`, `starcoder2`, `refact`, `falcon`, `grok`, `deci`, `xverse`, `arctic`, `orion`, `internlm2`, `minicpm`, `minicpm3`, `command-r`, `cohere2`, `cohere2moe`, `dbrx`, `exaone`, `exaone-moe`, `jais`, `jais2`, `plamo`, `plamo2`, `plamo3`, `codeshell`, `chatglm`, `granite_swa`, `graniteswitch`, `smollm3`, `seed_oss`, `grovemoe`, `apertus`, `arcee`, `afmoe`, `laguna`, `bailingmoe`, `bailingmoe2`, `bailingmoe3`, `dots1`, `plm`, `metis16`, `zaya1`, `mellum`, `nanbeige`, `beam`, `kolibri1`, `ricursive`, `domynedge`, `inkling2`, `inkling2_loop`, `maincoder`, `maion-coder`, `pangu-embedded`, `xing4_0`, `spark2_5`, `k2-horizon`, `hrm_text`, `limite`, `talkie`, `mistral3` | `transformer.rs` + variantes RoPE/norm/activation, `moe.rs` pour les MoE, `head` pour les têtes spéciales ; `mistral3` demande en plus le projecteur Pixtral | M à XL selon les variantes |
| **MLA / attention compressée** | `deepseek2`, `deepseek32`, `deepseek4`, `deepseek41`, `glm4`, `glm4moe`, `glm-dsa`, `glm5-next`, `kimi-linear`, `kimi-k3`, `kimi-k31`, `minimax-01`, `minimax-m2`, `minimax-m3`, `mistral4`, `mistral-large4` | Nouveau `blocks/mla.rs` (KV compressé, projections absorbées), `blocks/dsa.rs` (indexeur + top-k), `blocks/lightning.rs` (MiniMax), variantes MoE ; vérifier au chargement si `mistral4`/`mistral-large4` utilisent réellement MLA | XL |
| **Hybrides SSM / linéaires** | `qwen3next`, `mamba`, `mamba2`, `jamba`, `maple`, `nemotron`, `nemotron_h`, `nemotron_h_moe`, `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7` | `blocks/mamba.rs` (Mamba1 ; Mamba2 existe déjà), `blocks/rwkv.rs` (time-mix), hybridation attention/SSM (Qwen3-Next, Nemotron-H), MoE | L à XL |
| **Llama 4 / iRoPE** | `llama4` | `rope.rs` : iRoPE (couches interleavées avec/sans RoPE), MoE, vision | L |
| **GPT-OSS / sinks** | `gpt-oss` | `blocks/attention.rs` : attention sinks, MXFP4, MoE | M |
| **Gemma 2/3n** | `gemma2`, `gemma3n` | Softcapping des logits, AltUp, LAuReL, per-layer embeddings, MatFormer | L |
| **T5 / encodeur-décodeur** | `t5`, `t5encoder` | `arch/t5.rs`, attention croisée, biais de position relatif, tokenizer UGM | L |
| **BitNet** | `bitnet` | Réutilise TQ1_0/TQ2_0 déjà présents ; arch dédiée | S |
| **Chameleon / early fusion** | `chameleon` | Tokens image dans la séquence, tokenizer dédié | M |
| **Multimodal texte+vision** | `qwen2vl`, `qwen3vl`, `qwen3vlmoe`, `hunyuan_vl`, `sensenova_u1`, `cogvlm`, `deepseek2-ocr`, `paddleocr`, `dots3note`, `mimo2`, `step35` | Projecteurs vision (section 4.10), `Input::Image` déjà présent, pipelines OCR | L à XL |
| **Audio / TTS** | `qwen3tts`, `pockettts`, `moshi`, `zonos3`, `kokoro`, `tiny-voca`, `wavtokenizer-dec`, `muse-spark`, `muse-glimmer` | Nouveau crate `kinetic-audio` : encodeurs audio, décodeurs TTS, mel, streaming | XL |
| **Diffusion** | `dream`, `llada`, `llada-moe` | Nouveau crate `kinetic-diffusion` : sampler itératif, masquage, ordonnancement | XL |
| **Embeddings / encodeurs** | `modern-bert`, `nomic-bert-moe`, `neo-bert`, `jina-bert-v2`, `jina-bert-v3`, `eurobert`, `llama-embed`, `gemma-embedding2` | Étendre `arch/bert.rs` et `head` ; pooling, normalisation, têtes de rerank | M |
| **Têtes de draft** | `eagle3`, `dflash` | `speculative/eagle.rs`, `speculative/dflash.rs` | L |
| **Divers** | `vllm-decision2` (modèle de décision, pas de génération autoregressive) | Adaptateur dédié, hors chemin `Family::decoder` | M |

**Point clé** : le registre `FAMILIES` et le contrat `Family` sont déjà le bon point
d'extension. Il ne faut pas ajouter un `match` central : chaque famille reste un
fichier `arch/<nom>.rs` avec `names`, `body`, `output_norm`, `sqrt_embd_scale`,
`logit_scale`, plus une ligne dans `FAMILIES`. Les variantes qui ne changent que la
tête ou le RoPE doivent réutiliser `transformer.rs` avec des champs de `Family`, pas
créer un duplicateur.

#### Architecture cible du registre

Aujourd'hui `Family` ne porte que cinq champs (noms, corps, norme de sortie, échelle
d'embedding, échelle de logits). La parité demande d'ajouter des **capacités
déclaratives** au `Family`, pas des branches dans le loader :

```rust
pub(crate) struct Family {
    pub names: &'static [&'static str],
    pub body: Body,
    pub output_norm: &'static str,
    pub sqrt_embd_scale: bool,
    pub logit_scale: bool,
    // nouveaux axes, tous optionnels et lus par les blocs :
    pub rope: RopeKind,             // standard, iRoPE, partial, none
    pub attention: AttentionKind,   // mha, gqa, mqa, mla, dsa, sinks
    pub moe: Option<MoeKind>,       // routed, shared, expert-parallel
    pub state: StateKind,           // none, mamba2, rwkv, short-conv
    pub head: HeadKind,             // lm, embedding, rerank, tts, diffusion
}
```

Chaque champ est consommé par un bloc existant (`attention.rs`, `moe.rs`,
`mamba2.rs`, `short_conv.rs`) ou par un nouveau bloc. Les fichiers d'architecture
restent de simples mappages de tenseurs.

### 4.2 Tokenizers

| Élément | llama.cpp | Kinetic | Écart |
|---|---|---|---|
| Modèles | BPE, SPM, WordPiece, UGM (T5), RWKV, PLaMo, Tekken, Jais2, Spark2.5, DeepSeek3, Youtu, Gemma4, MiniCPM5… | BPE (`gpt2`), SPM (`llama`), WordPiece (`bert`) | UGM, RWKV, PLaMo, Tekken et tokenizers spécialisés manquants |
| Prétokenizers | 68 (`llama-vocab.h`) | 13 (`qwen35`, `qwen2`, `llama3`, `llama-v3`, `llama-bpe`, `lfm2`, `falcon-h1`, `dbrx`, `smollm`, `refact`, `gpt-2`, `exaone4`, `hunyuan-dense`) | 55 manquants |
| Tokens spéciaux | BOS/EOS/EOT/EOM/UNK/SEP/FIM, tokens de contrôle, `tokenizer.ggml.token_type` | BOS/EOS/SEP, quelques spéciaux | EOT/EOM, FIM, tokens de contrôle, types de tokens |
| Templates de chat | Jinja complet (minja), templates intégrés, `--chat-template` | Jinja du fichier, `--chat-template-file` | Templates intégrés par nom, variantes de rendu |
| Tool calls | Formats Qwen, Llama, Mistral, Hermes, etc. | Qwen3.5, Qwen3, LFM3, LFM2, Llama 3, Mistral | Formats DeepSeek, GLM, Kimi, InternLM, Command-R, etc. |
| Tokenizer de vision/audio | `mtmd` | Aucun | Tout |

**Implémentation** : `crates/kinetic-tokenizer/src/pretokenize.rs` doit gagner des
`Pattern` réutilisables (Llama3-like, GPT-2-like, whitespace, digits, Han/kana,
contractions) et une table de correspondance `tokenizer.ggml.pre` → `Pattern`. Les
modèles non-BPE (UGM, RWKV, PLaMo) demandent des modules dédiés dans
`kinetic-tokenizer/src/` (`ugm.rs`, `rwkv.rs`, `plamo.rs`). Le contrat public
`Tokenizer::from_gguf`, `encode`, `decode` ne change pas.

### 4.3 Formats de poids et quantification

#### Runtime

| Type GGUF | CPU | Vulkan | HIP/ROCm | CUDA | llama.cpp | Écart local |
|---|:-:|:-:|:-:|:-:|:-:|---|
| F32, F16 | oui | oui | **non** | oui | oui | HIP à ajouter |
| BF16 | oui | **non** | **non** | **non** | oui | HIP/Vulkan/CUDA à ajouter |
| Q8_0, Q6_K, Q5_K, Q4_K | oui | oui | **non** | oui | oui | HIP à ajouter |
| Q4_0 | oui | oui | **non** | oui | oui | HIP à ajouter |
| Q5_0, Q5_1, Q4_1 | oui | **non** | **non** | oui | oui | Vulkan/HIP à ajouter |
| IQ4_XS | oui | oui | **non** | oui / experts seulement | oui | HIP à ajouter ; CUDA complet |
| IQ4_NL, IQ3_S | oui | **non** | **non** | oui / experts seulement | oui | Vulkan/HIP à ajouter ; CUDA complet |
| IQ3_XXS | oui | oui | **non** | oui / experts seulement | oui | HIP à ajouter ; CUDA complet |
| IQ2_S, IQ2_XS, IQ2_XXS | oui | **non** | **non** | oui / experts seulement | oui | Vulkan/HIP à ajouter ; CUDA complet |
| IQ1_M, IQ1_S, TQ1_0, TQ2_0 | oui | **non** | **non** | oui | oui | Vulkan/HIP à ajouter |
| MXFP4 | oui | **non** | **non** | oui | oui | Vulkan/HIP à ajouter |
| Q2_K, Q3_K | oui | **non** | **non** | oui | oui | Vulkan/HIP à ajouter |
| **NVFP4** | **non** | **non** | **non** | **non** | oui | type 40 à ajouter |
| **F8_E4M3** | **non** | **non** | **non** | **non** | oui | type 43 à ajouter |
| Q1_0, Q2_0, PQ2_0, PTQ1_0 | oui (source) | **non** | **non** | oui (source) | Q1_0/Q2_0 | docs à mettre à jour |

**Conséquence locale (relevé du 22 nov., `kinetic-work` `85d29aa`)** : Kinetic Vulkan
charge F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, IQ3_XXS et IQ4_XS, chacun avec un
noyau mono-token et un noyau many-token. Les modèles locaux en IQ4_XS (L3-8B Stheno)
et IQ3_XXS (Qwen3.8-27B) tournent donc sur GPU. Les blocages de format restants sont
IQ4_NL, IQ3_S, Q2_K/Q3_K, Q5_0/Q5_1/Q4_1, MXFP4, BF16, puis NVFP4 (type 40) et
F8_E4M3 (type 43) absents de tous les backends. Sur cette machine, le blocage n'est
plus le format mais l'architecture : `qwen3-coder-30b-a3b` est en IQ3_XXS et reste
inutilisable sans noyau MoE.

`kinetic-quant/src/types.rs` connaît déjà les identifiants 41, 42, 142, 143 mais pas
40 (NVFP4) ni 43 (F8_E4M3). La détection de modèle qui inspecte les formes de poids
doit continuer à n'utiliser que la première dimension (les modèles 4 bits peuvent
avoir une seconde dimension réduite).

#### Quantification (production)

| Capacité `llama-quantize` | Kinetic `quantize` | Écart |
|---|---|---|
| `--type Q4_0/Q8_0/Q4_K_M/Q5_K_M/Q6_K` | oui | — |
| Toutes les autres cibles (`Q2_K`…`IQ1_S`, `MXFP4`, `NVFP4`, `F8_E4M3`) | non | production IQ/NVFP4/FP8 |
| `--imatrix` | non | outil `kinetic imatrix` + import |
| `--output-tensor-type` | non | type de sortie |
| `--token-embedding-type` | non | embeddings séparés |
| `--tensor-type`, `--tensor-type-file` | non | politique par tenseur |
| `--include-weights`, `--exclude-weights` | non | filtres |
| `--prune-layers` | non | élagage |
| `--keep-split` | non | sortie multi-fichiers |
| `--allow-requantize` | non | requantification |
| `--leave-output-tensor`, `--pure` | non | contrôle fin |
| `--override-kv`, `--dry-run`, `--max-buffer-size` | non | méta/robustesse |

#### Cache KV

| Type | llama.cpp | Kinetic CPU | Kinetic Vulkan | Kinetic HIP | Kinetic CUDA |
|---|:-:|:-:|:-:|:-:|:-:|
| f32, f16 | oui | f16 | f16 | non | f16 |
| bf16 | oui | non | non | non | non |
| q8_0 | oui | non | non | non | oui |
| q4_0, q4_1, iq4_nl, q5_0, q5_1 | oui | non | non | non | non |

Le cache KV est un levier direct de mémoire et de vitesse en contexte long. Ajouter
`q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`, `bf16` et `f32` dans
`kinetic-hip/src/attention.rs`, `kinetic-vulkan/src/attention.rs`, le chemin CPU et
`kinetic-cuda/src/attention.rs` (`KvType`). Sur la RX 9070 XT, q8_0 est le premier
gain utile : il double la capacité de contexte sans changer les noyaux de calcul.

### 4.4 Backends et matériel

| Famille | llama.cpp | Kinetic | Priorité locale |
|---|---|---|---|
| CPU x86 | AVX2, AVX512, AVX512-VNNI, AMX, repack, BLAS, HBM, NUMA | AVX2/FMA/F16C, fallback scalaire | P1 (Zen 4 a AVX-512) |
| CPU ARM | NEON, SDOT, i8mm, SVE/SVE2 | NEON, SDOT | P3 |
| **HIP/ROCm** | oui | **non** | **P0** |
| **Vulkan** | coopmat/coopmat2, subgroup 64, MoE, layer split, flash, GEMM prompt | subgroup 32 imposé, 9 types de poids, GEMV mono-token et many-token (pas de GEMM tuilé), attention flash-style, pas de MoE/layer split | **P0** |
| CUDA | compute 5.x–12.x, BF16, FP8, NVFP4, IQ, MMQ, flash, graphs, multi-GPU | compute 6.0–12.0, MMA int8, flash, graphs, MoE, IQ partiel | P3 (aucun GPU NVIDIA ici) |
| Metal | natif | via Vulkan/Metal layer | P3 |
| SYCL | oui | non | P3 |
| CANN (Ascend) | oui | non | P3 |
| MUSA (Moore Threads) | oui | non | P3 |
| OpenCL | oui | non | P3 |
| WebGPU | oui | non | P3 |
| RPC | oui | non | P3 |
| BLAS | oui | non | P3 |
| Hexagon / OpenVINO / zDNN / ZenDNN / VirtGPU / ET | oui | non | niches |

Sur la machine de référence, **seules les lignes CPU x86 et Vulkan sont exécutables
aujourd'hui**. La ligne CUDA est conservée pour la parité de l'écosystème, mais elle
ne peut pas être mesurée localement et ne doit pas retarder HIP/ROCm ou Vulkan.

**Multi-GPU** : Kinetic sait répartir par couches (`--gpus 0,1`) et garder les experts
MoE en mémoire système (Flywheel). llama.cpp ajoute :

- `--split-mode {none,layer,row}` : le split `row` découpe les matrices et réduit la
  latence d'un seul modèle sur plusieurs GPU ;
- `--tensor-split` : proportion par GPU ;
- `--main-gpu` : GPU des embeddings/head ;
- `--override-tensor` (`-ot`) : placement par motif de tenseur ;
- `--cpu-moe`, `-ncmoe`, `--moe-cache-mib`, `--moe-cache-cpu-share` : politique MoE ;
- `--no-kv-offload`, `--no-op-offload`, `--defrag-thold` : contrôle du cache.

Ces options ne sont pas que du confort : elles changent la mémoire pic et le débit.
Il faut les ajouter comme réglages du planificateur de mémoire, pas comme flags
ignorés.

**Sur cette machine**, le second GPU est un iGPU `gfx1036` à 2 CU et mémoire système :
le row/tensor split n'a pas de second GPU sérieux pour l'exploiter. Le layer split
dGPU + iGPU reste utile pour les modèles qui ne tiennent pas en 16 Gio, mais il sera
limité par la bande passante de l'iGPU. Le vrai levier local est HIP sur le GPU
discret, puis Vulkan en repli.

### 4.5 Serveur et API

#### Routes OpenAI / llama-server

| Route | Kinetic | llama.cpp | Priorité |
|---|:-:|:-:|---|
| `GET /health`, `/v1/health` | oui | oui | — |
| `GET /metrics`, `/v1/metrics` | oui | oui | P2 (labels) |
| `GET /v1/models` | oui | oui | — |
| `POST /v1/chat/completions` | oui | oui | — |
| `POST /v1/completions` | oui | oui | — |
| `POST /v1/embeddings` | oui | oui | — |
| `POST /v1/rerank`, `/v1/reranking` | oui | oui | — |
| `POST /completion`, `/completions` | oui | oui | — |
| `POST /tokenize`, `/detokenize` | oui | oui | — |
| `GET /props` | oui | oui | — |
| `POST /props` | **non** | oui | P1 |
| `GET /slots` | oui | oui | — |
| `POST /slots/:id_slot` | **non** | oui | P1 |
| `POST /embedding`, `/embeddings` | oui | oui | — |
| `POST /infill` | **non** | oui | P1 |
| `POST /apply-template` | **non** | oui | P1 |
| `POST /v1/chat/completions/input_tokens` | **non** | oui | P2 |
| `POST /v1/responses` | **non** | oui | P2 |
| `POST /v1/messages` (Anthropic) | **non** | oui | P2 |
| `POST /v1/messages/count_tokens` | **non** | oui | P2 |
| `POST /tts`, `/v1/audio/speech` | **non** | oui | P3 |
| `POST /v1/audio/transcriptions` | **non** | oui | P3 |
| `WS /v1/realtime` | **non** | oui | P3 |
| `POST /v1/systemone` | **non** | oui | P3 |
| `POST /v1/chat/completions/control` | **non** | oui | P3 |
| `GET/POST /lora-adapters` | **non** | oui | P1 |
| Router `POST/DELETE /models`, `/models/load`, `/models/unload`, `/models/sse` | **non** | oui | P2 |
| `/v1/stream`, `/v1/streams/lookup` | **non** | oui | P3 |
| MCP / server tools | **non** | oui | P3 |
| GCP Vertex compat | **non** | oui | P3 |
| CORS proxy | **non** | oui | P3 |
| TLS natif | **non** (proxy recommandé) | oui | P2 |
| `--api-key-file` | **non** (clé en clair) | oui | P2 |

#### Champs et fonctionnalités serveur

| Fonction | Kinetic | llama.cpp | Écart |
|---|:-:|:-:|---|
| `n > 1` | non | oui | paralléliser les choix |
| `logprobs` / `top_logprobs` | oui | oui | — |
| `reasoning_content` | oui | oui | — |
| `grammar` / `response_format` | oui | oui | — |
| `tools` / `tool_choice` | oui | oui | formats à étendre |
| `parallel_tool_calls` | oui | oui | — |
| `ignore_eos` | non | oui | sampler |
| `min_keep` | non | oui | sampler |
| `dynatemp_range` / `dynatemp_exponent` | non | oui | sampler |
| `adaptive_target` / `adaptive_decay` | non | oui | sampler |
| `reasoning_budget` | non | oui | sampler |
| `grammar_lazy`, triggers | non | oui | grammaire |
| `chat_template_kwargs` | oui | oui | — |
| `cache_prompt` | implicite (Warm Start) | oui | documenter |
| `slot_id` | non | oui | slots |
| `id_slot` | non | oui | slots |
| `timings_per_token` | non | oui | métriques |
| `prompt_progress` | non | oui | streaming |
| `return_progress` | non | oui | streaming |
| `n_predict`, `n_keep` | partiel | oui | params |
| `samplers` (ordre) | non | oui | chaîne |
| `backend_sampling` | non | oui | GPU sampling |
| `xtc_*`, `dry_*`, `mirostat_*`, `top_n_sigma`, `typical_p` | oui | oui | — |
| `stop` | oui | oui | — |

### 4.6 Sampling et décodage

| Paramètre | Kinetic | llama.cpp | Action |
|---|:-:|:-:|---|
| temperature, top_k, top_p, min_p | oui | oui | — |
| repeat/presence/frequency penalty | oui | oui | — |
| typical_p, top_n_sigma, XTC, DRY | oui | oui | — |
| Mirostat 1/2 | oui | oui | — |
| logit_bias, seed, stop | oui | oui | — |
| dynamic temperature | non | oui | ajouter `dynatemp_range`, `dynatemp_exponent` |
| adaptive sampling | non | oui | ajouter `adaptive_target`, `adaptive_decay` |
| `min_keep` | non | oui | ajouter |
| `ignore_eos` | non | oui | ajouter |
| reasoning budget | non | oui | ajouter |
| grammar lazy/triggers | non | oui | ajouter |
| backend sampling | non | oui | ajouter (greedy/top-k d'abord) |
| `n > 1` | non | oui | ajouter |
| ordre de samplers configurable | non | oui | ajouter |

Le cœur est `crates/kinetic-models/src/speculative/sampler.rs` (`struct Sampling`,
`fn sample`). Il est partagé par le serveur et le CLI. Ajouter les champs et les
filtres dans `speculative/filters.rs` ; ne pas dupliquer la logique dans
`kinetic-server`.

### 4.7 Décodage spéculatif

| Type llama.cpp | Kinetic | Commentaire |
|---|:-:|---|
| `draft` (modèle séparé) | non | à ajouter comme source optionnelle de RAD-1 |
| `draft-eagle3` | non | tête EAGLE3 |
| `draft-mtp` | oui (bloc MTP) | déjà intégré |
| `draft-dflash` | non | tête DFlash |
| `draft-dflash2` | non | tête DFlash2 |
| `draft-dspark` | non | DFlash + tête Markov |
| `draft-lilicorr` | non | DFlash + scorer |
| `ngram-simple` | oui (lookup contexte) | RAD-1 |
| `ngram-map-k` | non | index n-gram par clés |
| `ngram-map-k4v` | non | clés + 4 valeurs |
| `ngram-mod` | non | variante modulo |
| `ngram-cache` | non | cache 3 niveaux |
| lookup cache statique/dynamique | non | persistance |
| `--spec-type`, `--draft-max/min`, `-ngld` | non (auto) | flags à accepter/documenter |
| backend sampling | non | échantillonnage GPU |
| speculative + batching | partiel | vérifier avec Convoy |

**Stratégie** : conserver RAD-1 automatique comme défaut. Ajouter un champ
`Sampling::draft` ou une politique `speculative/policy.rs` qui permet d'activer une
source supplémentaire (draft model, EAGLE3, ngram map) sans changer l'API. Les
nouvelles têtes doivent passer par le même vérificateur exact.

### 4.8 Outils CLI

| Commande llama.cpp | Kinetic | Écart |
|---|:-:|---|
| `llama-cli`, `llama-completion` | `run`, `chat` | — |
| `llama-server` | `serve` | routes à compléter |
| `llama-bench` | `bench` | options/format de sortie |
| `llama-perplexity` | `perplexity` | KL, fichiers |
| `llama-quantize` | `quantize` | cibles/imatrix |
| `llama-imatrix` | **non** | à ajouter |
| `llama-gguf-split` | **non** | merge/split |
| `llama-tokenize` | **non** | CLI tokenize/detokenize |
| `llama-export-lora` | **non** | export |
| `llama-cvector-generator` | **non** | control vectors |
| `llama-mtmd-debug` | **non** | debug multimodal |
| `llama-tts` | **non** | TTS |
| `llama-rpc-server` | **non** | RPC |
| `llama-fit-params` | `fit` (proche) | options |
| `llama-batched-bench` | **non** | benchmark batch |
| `llama-lookahead`, `llama-lookup` | **non** | spéculatif |
| `llama-passkey`, `llama-retrieval`, `llama-parallel` | **non** | exemples |
| `llama-simple`, `llama-speculative` | **non** | exemples |
| `llama-finetune`, `llama-train-text-from-scratch` | **non** | entraînement |
| `llama-diffusion` | **non** | diffusion |
| `gguf-hash`, `gguf` | `verify` (partiel) | hash |
| scripts de conversion HF | **non** (choix assumé) | à documenter |

**Flags llama.cpp non couverts** (au-delà de ceux déjà traduits/ignorés) :
`-sm/--split-mode`, `-ts/--tensor-split`, `-mg/--main-gpu`, `-ot/--override-tensor`,
`-cmoe/--cpu-moe`, `-ncmoe`, `--moe-cache-mib`, `--moe-cache-cpu-share`,
`--no-kv-offload`, `--no-op-offload`, `--defrag-thold`, `--lora-init-without-apply`,
`--control-vector`, `--control-vector-scaled`, `--grammar`, `--json-schema` (CLI),
`--spec-type`, `--draft-*`, `-ngld`, `--numa`, `--prio`, `--poll`, `--threads-http`,
`--api-key-file`, `--ssl-*`, `--path`, `--webui`, `--tools`, `--mcp`, `--connect`,
`--rpc`. `--cache-reuse` est déjà accepté mais ignoré : il faut l'implémenter ou
l'expliquer explicitement.

Le fichier `crates/kinetic-cli/src/llama_flags.rs` est le bon endroit pour traduire
ou expliquer ; il ne faut pas laisser un flag inconnu produire « unexpected
argument ».

### 4.9 Bindings et ABI

| Capacité | llama.cpp | Kinetic |
|---|---|---|
| API C | ~200 fonctions (`llama.h`) | 10 fonctions (`kinetic.h`) |
| Modèle : load/free/metadata | oui | load/free/version |
| Vocab : tokenize/detokenize/tokens | oui | non |
| Contexte : new/free/params | oui | session new/free |
| Batch : decode/encode | oui | non |
| Sampler chain | oui | params minimal |
| Grammar | oui | non |
| KV cache / state save/load | oui | non |
| LoRA | oui | non |
| Multimodal | oui (mtmd) | non |
| Backend/device enumeration | oui | non |
| Logging/perf | oui | non |
| Python | llama-cpp-python complet | partiel, `NotImplementedError` pour grammars/tools/response_format/logprobs |
| Go, Node, .NET, Java, Swift, Ruby, PHP, Rust, Flutter | oui | non |

L'ABI C de Kinetic est volontairement minimale. Pour la parité, il faut l'étendre par
groupes cohérents (modèle, vocab, session, sampling, grammar, état, multimodal) en
gardant les fonctions existantes et en ajoutant des versions/tailles explicites. Les
bindings Python/Node/Go peuvent être générés à partir de cette ABI, mais le serveur
HTTP reste le chemin le plus simple pour les nouveaux écosystèmes.

### 4.10 Modalités

| Modalité | llama.cpp (mtmd) | Kinetic |
|---|---|---|
| Image | ~40 projecteurs : Gemma3/4, SmolVLM, Pixtral, Qwen2/2.5/3-VL, InternVL, Llama4, Moondream, MiniCPM-V, GLM4V, HunyuanVL, CogVLM, etc. | 2 : `qwen3vl_merger`, `gemma4v` |
| Vidéo | oui (échantillonnage de frames) | non |
| Audio entrée | oui (Whisper, Ultravox, Voxtral, Qwen3-ASR, Granite-Speech, etc.) | non |
| Audio sortie / TTS | oui (Kokoro, Zonos3, PocketTTS, Qwen3TTS, etc.) | non |
| OCR | oui (DeepSeek-OCR, PaddleOCR, DotsOCR, etc.) | non |
| Projecteurs | 73 fichiers | 2 |
| Préprocessing | resize, normalize, mel, frames, layout OCR | resize/normalize image |

Le trait `Projector` de `kinetic-vision/src/lib.rs` est bon pour l'image :
`n_embd_out`, `prepare`, `encode`, et une ligne dans `PROJECTORS`. Pour l'audio et la
vidéo, il faut des traits distincts (ou un trait `Modality` avec des méthodes
spécifiques) et un crate `kinetic-audio` ; ne pas forcer l'audio dans le trait vision.

### 4.11 Entraînement, diffusion, RPC

| Domaine | llama.cpp | Kinetic | Recommandation |
|---|---|---|---|
| Fine-tuning | `examples/training`, `llama-finetune` | non | hors périmètre prioritaire ; crate `kinetic-train` si nécessaire |
| Train from scratch | oui | non | idem |
| Diffusion LLM | `examples/diffusion`, LLaDA/Dream | non | crate `kinetic-diffusion` |
| RPC backend | `ggml-rpc` | non | crate `kinetic-rpc` |
| Conversion HF→GGUF | scripts Python | non (choix) | documenter ou crate `kinetic-convert` |
| Modèles de décision | `vllm-decision2` | non | niche |
| Control vectors | oui | non | `kinetic-models/src/lora.rs` + vecteurs |
| Export LoRA | oui | non | CLI |

### 4.12 Synthèse chiffrée

| Axe | llama.cpp | Kinetic | Parité |
|---|---:|---:|---:|
| Noms d'architecture | 180 | 21 | 12 % |
| Prétokenizers | 68 | 13 | 19 % |
| Types de poids utiles | ~30 | ~28 | ~90 % |
| Cibles de quantification | ~30 | 5 | 17 % |
| Backends majeurs | 15+ | 3 | ~20 % |
| Routes serveur | ~45 | ~15 | 33 % |
| Paramètres de sampling | ~35 | ~20 | 57 % |
| Types spéculatifs | 14 | 3 (lookup, index, MTP) | 21 % |
| Projecteurs multimodaux | 73 | 2 | 3 % |
| Fonctions C | ~200 | 10 | 5 % |

Ces pourcentages ne sont pas une note de qualité : Kinetic est meilleur que llama.cpp
sur plusieurs axes (exactitude, cache, automatisation). Ils montrent l'ampleur du
travail de *couverture*.

**Lecture locale** : sur la RX 9070 XT, Kinetic ne dispose que de Vulkan et du CPU.
llama.cpp dispose de deux backends exploitables, HIP/ROCm et Vulkan. Le taux de
couverture « backends » pertinent n'est donc pas 3/15 mais **1/2** (Vulkan contre
HIP + Vulkan), et le principal écart de performance local est l'absence de HIP.

### 4.13 Modèles présents sur la machine et blocages réels

Relevé `kinetic models` du 22 novembre 2026. Il montre que les blocages locaux ne
sont pas théoriques : plusieurs modèles déjà téléchargés ne passent pas.

| Modèle local | Architecture | Quant | Kinetic Vulkan | Blocage |
|---|---|---|---|---|
| `gemma-4-12b-heretic-Q5_K_M` | `gemma4` | Q5_K_M | oui | — |
| `qwen3.5-4b` / `qwen3.5-9b` | `qwen35` | Q4_K_M | oui | — |
| `L3-8B-Stheno-v3.2` | `llama` | IQ4_XS | oui | `0127427` |
| `Qwen3.8-27B-IQ3_XXS` | `qwen35` | IQ3_XXS | oui | `85d29aa`, chiffres ci-dessous |
| `qwen3-coder-30b-a3b` | `qwen3moe` | IQ3_XXS | **non** | architecture MoE : aucun noyau MoE Vulkan |
| `qwen3-vl-4b` | `qwen3vl` | Q4_K_M | **non** | architecture `qwen3vl` non enregistrée |
| `paddleocr-vl-1.6` | `paddleocr` | BF16 | **non** | architecture OCR + BF16 Vulkan |
| `helcyon-gpt-4o-v4.5` | `llama` | Q5_K_M | **non** | prétokenizer `tekken` non supporté |
| `Offrail-27B-12G` | `qwen35` | PTQ1_0 | **non** | type PTQ1_0 absent des shaders |

Les deux modèles en IQ que Kinetic ne chargeait pas (`L3-8B-Stheno-v3.2` et
`Qwen3.8-27B-IQ3_XXS`) ont été débloqués le 22 nov. et servent de référence de
performance (protocole §7.10, un fichier, un backend par comparaison) :

| `Qwen3.8-27B-IQ3_XXS`, 11,47 Gio | kinetic Vulkan | llama.cpp Vulkan |
|---|---:|---:|
| pp512 | 261,9 t/s | 1154,2 t/s |
| pp2048 | 257,9 t/s | 1149,0 t/s |
| tg128 | 38,4 t/s | 40,1 t/s |
| avant les noyaux IQ (CPU) | 3,8 / 0,6 t/s | — |

Lecture : le décodage est à 95 % de llama.cpp, le préfill à 23 %. Le levier local n'est
donc plus le format ni le backend, mais le GEMM de prompt (§7.3). Les blocages qui
restent sur cette machine sont `qwen3vl` (architecture), `qwen3moe` (MoE), `paddleocr`
(BF16 + OCR), `Offrail-27B` (PTQ1_0) et `helcyon-gpt-4o` (prétokenizer `tekken`).

Sous `--device auto`, Kinetic essaie CUDA puis Vulkan et retombe sur le CPU avec un
message quand le backend ne peut pas charger le modèle. C'est ce qui se passe
aujourd'hui pour `qwen3-coder-30b-a3b` (MoE) ou un GGUF BF16 : le modèle « marche », mais sur le CPU,
donc à une fraction de la vitesse attendue. Un test de performance doit toujours
nommer le backend (`--device vulkan`) pour ne pas mesurer cette retombée.

---

## 5. Écarts détaillés par crate et point d'extension

### 5.1 `kinetic-models`

**Rôle actuel** : registre `FAMILIES`, blocs (`attention`, `delta_net`, `mamba2`,
`short_conv`, `moe`, `conv_window`, `folding`), chargement, tête, LoRA, Warm Start,
spéculatif.

**À ajouter** :

| Écart | Fichier(s) cible(s) | Effort |
|---|---|---|
| Bloc MLA (DeepSeek V2/V3/V3.2/V4, Kimi, GLM) | `blocks/mla.rs` | XL |
| Bloc DSA (DeepSeek Sparse Attention, GLM-DSA) | `blocks/dsa.rs` | XL |
| Bloc lightning attention (MiniMax) | `blocks/lightning.rs` | L |
| Bloc RWKV time-mix (RWKV6/7) | `blocks/rwkv.rs` | L |
| Mamba 1 (Mamba2 existe) | `blocks/mamba.rs` | M |
| Hybridation SSM/attention (Nemotron-H, Jamba, Falcon-H1 étendu) | `blocks/mod.rs`, `arch/*` | L |
| Attention sinks (GPT-OSS) | `blocks/attention.rs` | M |
| iRoPE (Llama 4) | `arch/rope.rs` | M |
| Softcapping logits (Gemma 2) | `head/mod.rs` | S |
| AltUp/LAuReL/per-layer embeddings (Gemma 3n) | `arch/gemma3n.rs` | L |
| T5 encoder-decoder | `arch/t5.rs`, `head` | L |
| BitNet | `arch/bitnet.rs` | S |
| Encodeurs ModernBERT/Jina/EuroBERT/NeoBERT | `arch/bert.rs` | M |
| Têtes EAGLE3/DFlash | `speculative/eagle.rs`, `speculative/dflash.rs` | L |
| Draft model séparé | `speculative/draft_model.rs` (nouveau `Drafter`), `ensemble.rs` | L |
| Ngram map/cache | `speculative/lookup.rs` | M |
| LoRA dynamique (scaling au runtime, hot swap) | `lora.rs` | M |
| Control vectors | `lora.rs` ou `control.rs` | M |
| Entrées audio/vidéo | `input.rs` | L |
| Cache KV quantifié (q4/q5/bf16) | `model.rs`, `blocks/attention.rs` | L |
| Split GGUF | `loader.rs` | S |
| `--override-tensor` | `per_layer.rs`, `memory.rs` | M |

**Contrainte** : ne pas ajouter de paramètres inutiles aux constructeurs de blocs.
Les variantes doivent être des champs de configuration lus par le bloc, pas des
branches dans le forward.

### 5.2 `kinetic-vision` et le nouveau `kinetic-audio`

**Vision** : ajouter des projecteurs dans `crates/kinetic-vision/src/` :

| Famille de projecteur | Modèles | Effort |
|---|---|---|
| Qwen2-VL / Qwen2.5-VL | `qwen2vl` | M |
| Qwen3-VL / Qwen3-VL-MoE | `qwen3vl`, `qwen3vlmoe` | M |
| Llama 4 | `llama4` | M |
| InternVL | `internvl` | M |
| Pixtral / Mistral3 | `pixtral`, `mistral3` | M |
| Gemma 3 | `gemma3` | S |
| SmolVLM / Idefics | `smolvlm` | S |
| MiniCPM-V | `minicpmv` | M |
| GLM-4V | `glm4v` | M |
| HunyuanVL | `hunyuanvl` | M |
| CogVLM | `cogvlm` | M |
| DeepSeek-OCR / PaddleOCR / DotsOCR | `deepseekocr`, `paddleocr`, `dotsocr` | L |
| Moondream, MobilenetV5, SigLIP, etc. | divers | S à M |

Chaque projecteur reste un module + une ligne dans `PROJECTORS`. Le contrat
`Projector` ne doit pas être élargi pour l'audio.

**Audio** : nouveau crate `kinetic-audio` avec :

- encodeurs : Whisper, Voxtral, Qwen3-ASR, Granite-Speech, Conformer, Parakeet,
  Nemotron-Transcribe, VibeVoice-ASR ;
- décodeurs TTS : Kokoro, Zonos3, PocketTTS, Qwen3TTS, Chatterbox, Fish-S2,
  Soprano, VoxCPM2, Yue2 ;
- frontends : mel spectrogram, resampling, tokenisation audio, streaming ;
- trait `AudioProjector` et registre `AUDIO_PROJECTORS`.

**Vidéo** : module `kinetic-vision/src/video.rs` (échantillonnage de frames,
timestamps, pooling temporal) et `Input::Video` dans `input.rs`.

**OCR** : pipeline dédié dans `kinetic-vision/src/ocr.rs` (layout, régions,
post-traitement), pas seulement un projecteur.

### 5.3 `kinetic-tokenizer`

| Écart | Fichier | Effort |
|---|---|---|
| 55 prétokenizers | `pretokenize.rs` | M |
| UGM (T5) | `ugm.rs` | M |
| RWKV | `rwkv.rs` | M |
| PLaMo 2/3 | `plamo.rs` | M |
| Tekken (Mistral/Helcyon) | `pretokenize.rs` + variante BPE | M |
| Jais2, Spark2.5, DeepSeek3, Youtu, Gemma4, MiniCPM5 | `pretokenize.rs` + variantes | M |
| Tokens EOT/EOM/FIM/contrôle | `special.rs`, `vocab.rs` | S |
| Templates intégrés par nom | `kinetic-server/src/template.rs` | M |
| Formats tool-call supplémentaires | `kinetic-server/src/tool_calls.rs` | M |

**Contrainte** : le contrat `Tokenizer::from_gguf` / `encode` / `decode` ne change
pas. Les nouveaux modèles sont des variantes de `Model` et de `Pattern`.

### 5.4 `kinetic-quant`, `kinetic-gguf`, `kinetic-ktc`

| Écart | Fichier | Effort |
|---|---|---|
| NVFP4 (type 40) | `types.rs`, `dequant.rs`, `encode/` | L |
| F8_E4M3 (type 43) | `types.rs`, `dequant.rs`, `encode/` | M |
| IQ encode (Q2..Q4) | `encode/` | XL |
| Imatrix | nouveau `imatrix.rs` + CLI | L |
| Politique par tenseur | `encode/policy.rs` | M |
| `--output-tensor-type`, `--token-embedding-type` | CLI | S |
| `--keep-split`, `--prune-layers`, `--pure`, `--dry-run` | CLI | S |
| Split GGUF en lecture | `kinetic-gguf` | S |
| Signature/convert `.ktc` CLI | `kinetic-ktc` | M |
| Hash/verify partiel | `kinetic-ktc` | S |

**Attention** : les identifiants 40 et 43 doivent être ajoutés à `from_id` et aux
tables de taille de bloc. La détection de modèle ne doit pas supposer que la seconde
dimension est pleine (NVFP4).

### 5.5 `kinetic-cpu`

| Écart | Fichier | Effort | Gain attendu |
|---|---|---|---|
| AVX512 / AVX512-VNNI | `kdot/avx512.rs`, dispatch | L | 1,3–2× prompt sur Zen4/Sapphire Rapids |
| AMX (int8/bf16) | `kdot/amx.rs` | XL | 2–4× prompt sur Xeon récents |
| ARM i8mm | `kdot/neon.rs` | M | 1,2–1,8× sur Graviton/Apple CPU |
| ARM SVE/SVE2 | `kdot/sve.rs` | L | 1,3–2× sur serveurs ARM |
| Repack runtime | `kinetic-quant/src/layout.rs` + CPU | M | 5–20 % décodage |
| BLAS fallback | `matmul.rs` | S | stabilité sur CPU exotiques |
| NUMA / affinité | `pool.rs`, `memory.rs` | M | jusqu'à 2× multi-socket |
| Threadpool dédié | `pool.rs` | M | 3–10 % petits batches |
| Huge pages / mlock / dio | `memory.rs` | S | moins de page faults |
| Cache KV q4/q5/bf16 | `ops.rs` | L | 2–4× mémoire contexte |

Le 7800X3D est un Zen 4 : AVX-512 est disponible et Kinetic ne l'exploite pas. C'est
un gain direct pour les prompts qui débordent de la VRAM ou pour les embeddings, sans
changer l'architecture du backend.

### 5.6 `kinetic-vulkan` (priorité locale)

État actuel (`kinetic-work` `85d29aa`) : noyaux GEMV pour F32, F16, Q8_0, Q4_0, Q4_K,
Q5_K, Q6_K, IQ3_XXS et IQ4_XS, avec un second noyau par type pour les passes de
`PROMPT_BATCH = 128` tokens (`shaders/gemv_many.wgsl`) ; attention
`attend_64/128/256/512` déjà écrite en flash-style (online softmax, `shaders/attend.wgsl`
ligne 6, GQA, SWA, ring slots) ; sous-groupe 32 imposé ; pas de GEMM tuilé, pas de
coopmat, pas de MoE, pas de layer split, pas de KV quantifié. Les commandes sont
enregistrées une fois (`commands::Graph`, l'équivalent d'un graphe CUDA), mais
`Commands::submit` soumet puis attend la fin (`commands.rs` 160–163) : le GPU n'a
jamais plusieurs pas en vol. C'est un levier mesurable avant même les noyaux.

| Écart | Fichier | Effort | Gain attendu |
|---|---|---|---|
| GEMM de prompt tuilé (mémoire partagée, split-K, dequant en registres) | `shaders/gemm*.wgsl`, `kernels.rs` | L | 2–4× prompt |
| coopmat KHR (`VK_KHR_cooperative_matrix` disponible sur RADV gfx1201) ; coopmat2 seulement si le driver l'expose | `shaders/gemm_coopmat*.wgsl`, `physical.rs`, `device.rs` | XL | 2–5× prompt AMD/Intel/NVIDIA |
| Sous-groupe 32 ou 64 par noyau (RADV expose 32–64) | `physical.rs`, `pipeline.rs` | M | 5–15 % sur les noyaux memory-bound |
| Types IQ4_NL, IQ3_S, Q5_0/Q5_1/Q4_1, Q2_K/Q3_K, MXFP4 (IQ3_XXS, IQ4_XS, Q4_0 faits) | `gemv.rs`, `layout.rs`, `shaders/` | S/M par format | charge les modèles restants |
| KV cache q8_0/q4_0/q5_0/bf16 | `attention.rs`, `shaders/attend*.wgsl` | L | 2–4× mémoire KV |
| Softcap d'attention (Gemma 2) ; online softmax, GQA, SWA et ring slots sont déjà là | `shaders/attend*.wgsl` | S | parité Gemma 2 |
| MoE : routage top-k, GEMV/GEMM groupé, experts en VRAM ou hôte | `shaders/moe.wgsl`, `ffn.rs` | L | parité MoE |
| Layer split (VRAM + RAM, ou dGPU + iGPU) | `memory.rs`, `batch.rs` | L | modèles > 16 Gio |
| Soumission asynchrone : plusieurs frames en vol, timeline semaphores (le graphe enregistré existe déjà) | `commands.rs`, `device.rs` | M | 5–15 % décodage |
| Cache de pipelines Vulkan persisté | `pipeline.rs` | S | démarrage |
| Sous-allocation d'un grand buffer au lieu d'une allocation par tenseur | `memory.rs` | M | fragmentation, pic mémoire |
| BF16/FP8 si le driver les expose (RADV rapporte `bf16: 0`, `fp4: 0`) | `gemv.rs`, `shaders/` | M | parité, sinon repli FP16 |
| Valider les limites RADV sur gfx1201 (sous-groupe, pas de BF16/FP4 natif) et les documenter | tests | S | stabilité |

### 5.7 `kinetic-hip` (nouveau, priorité locale)

Le backend manque entièrement. Il faut créer un crate `kinetic-hip` sur le modèle de
`kinetic-cuda` : un crate par backend, pas un fork de `kinetic-vulkan`. Les sources de
kernels peuvent être partagées avec CUDA par macros ou par un petit crate de sources
communes, mais l'API, le build et les tests restent propres à HIP.

| Écart | Fichier | Effort | Gain attendu |
|---|---|---|---|
| Runtime HIP, build `hipcc --offload-arch=gfx1201`, streams, events | `build.rs`, `driver.rs`, `kernels.rs` | L | base du backend |
| GEMV quantifié Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS/IQ3_XXS | `gemv_*.hip` | L | décodage |
| WMMA FP16 (BF16 à valider sur gfx1201), wave32 | `gemm_wmma.hip` | L | prompt |
| Dot products int8/FP16 (`v_dot2_f32_f16`, `v_dot4_i32_i8`) | `dot.hip` | M | GEMV quantifié |
| rocBLAS / hipBLASLt pour les GEMM de prompt | `blas.rs` | M | prompt, fallback |
| Flash attention : online softmax, GQA, SWA, paged KV | `attention.hip` | L | contexte long |
| MoE groupé (top-k, tuiles par expert) | `moe.hip` | L | MoE |
| hipGraph pour capturer le pas de décodage | `graph.rs` | M | décodage |
| Pipelines async, mémoire épinglée, recouvrement copies/calcul | `memory.rs` | M | 5–15 % |
| KV cache quantifié | `attention.hip` | L | 2× mémoire KV |
| Layer split dGPU + iGPU, sinon RAM | `multi.rs` | L | modèles > 16 Gio |
| Télémétrie `rocm-smi`, timings `hipEvent`, profilage `rocprof` | `info.rs` | S | ops |
| Validation gfx1201 (ROCm 7.2 récent) | tests | S | correction |

**Contrainte** : ne pas supposer que les noyaux CUDA fonctionnent tels quels. RDNA 4
est wave32, n'a pas les mêmes primitives que CDNA (pas de MFMA), et la disponibilité
BF16/FP8 doit être vérifiée sur `gfx1201`. Le chemin sûr est FP16 + int8, puis BF16
si `__hip_bfloat16` et les instructions correspondantes sont validées.

### 5.8 `kinetic-cuda` (parité NVIDIA, secondaire)

Ce backend reste utile pour les utilisateurs NVIDIA et comme référence de kernels,
mais il est **hors priorité locale** : aucun GPU CUDA n'est présent sur la machine.

Déjà présents et à ne pas recompter comme des manques : CUDA graphs (`graph.rs`,
utilisé par `kinetic-models/src/cuda/mod.rs`), MoE groupé (`moe.rs` + `moe.cuh`,
9 types d'experts, gate/up fusionné, entrées q8k), flash attention
(`attention_flash.cuh`, utilisée au-delà de `MAX_BATCH`), GEMV int8 MMA
(`gemv_mma.cuh`, exact, choisi à partir de 3–5 entrées), mémoire épinglée
(`pinned.rs`) et copies P2P (`peer.rs`). Les écarts ci-dessous sont des
compléments, pas des fondations.

| Écart | Fichier | Effort | Gain attendu |
|---|---|---|---|
| BF16 GEMM/GEMV/attention | `cuda/gemm_float.cuh`, `gemv_float.cuh`, `attention*.cuh` | M | 1,1–1,5× sur Ampere+ |
| NVFP4 | `cuda/gemm_nvfp4.cuh`, `gemv_nvfp4.cuh`, `mma.cuh` | XL | 2–4× vs Q4_K, mémoire réduite |
| FP8 (E4M3), poids et KV | `cuda/gemm_fp8.cuh`, `attention.cuh` | L | 1,5–2,5× sur Hopper/Blackwell ; 2× KV |
| IQ4_XS/IQ3_XXS/IQ2_XS/IQ2_XXS pour les linears denses (GEMV) | `cuda/gemv_iq.cuh`, `gemv.rs` | L | active les denses 2–4 bits |
| GEMM tuilé pour les types IQ (prompt) | `gemm.rs`, `gemm.cuh` | L | prompt des modèles IQ |
| Row/tensor split (le layer split multi-GPU existe déjà) et `--tensor-split`/`--main-gpu` | `device.rs`, `batch.rs`, `peer.rs` | XL | un modèle sur 2 GPU |
| Softcap d'attention (Gemma 2) | `cuda/attention.cuh`, `attention_flash.cuh` | S | parité Gemma 2 |
| KV q4_0/q4_1/iq4_nl/q5_0/q5_1/bf16 | `cuda/attention.cuh`, `attention.rs` | L | 2–4× mémoire KV |

**Contrainte** : ne pas caster le résultat d'une op optimisée vers le dtype d'entrée
sans contrat documenté. Les kernels doivent respecter les contrats de dtype des
backends existants et de `kinetic-quant` (principe déjà appliqué dans Kinetic).

### 5.9 `kinetic-server`

| Écart | Fichier cible | Effort |
|---|---|---|
| `POST /props` | `llama.rs` | S |
| `POST /slots/:id_slot` | `llama.rs` | M |
| `/infill` (FIM) | nouveau `infill.rs` | M |
| `/apply-template` | `template.rs` | S |
| `/v1/responses` | nouveau `openai/responses.rs` | L |
| `/v1/messages` (Anthropic) | nouveau `anthropic.rs` | L |
| `/v1/messages/count_tokens` | `anthropic.rs` | S |
| `/v1/chat/completions/input_tokens` | `api.rs` | S |
| `/v1/audio/*`, `/tts` | `kinetic-audio` + `audio.rs` | XL |
| `/v1/realtime` WS | `realtime.rs` | XL |
| `/lora-adapters` | `lora.rs` | M |
| `/v1/chat/completions/control` | `sequence.rs` | M |
| Router `/models/*` | nouveau `router.rs` | XL |
| MCP / tools | nouveau `mcp.rs`, `tools.rs` | XL |
| `n > 1` | `api.rs`, `engine.rs` | M |
| `ignore_eos`, `min_keep`, dynatemp, adaptive, reasoning budget | `sampler.rs` | M |
| Grammar lazy/triggers | `constrained.rs` | M |
| Backend sampling | `hip.rs`, `vulkan.rs`, `cuda.rs` | L |
| TLS / API key file | `http.rs` | M |
| CORS / proxy | `http.rs` | S |
| Métriques détaillées | `metrics.rs` | S |
| Timings par token | `api.rs` | S |
| Prompt progress | `sequence.rs` | M |

**Architecture** : garder `server.rs` comme registre de routes et `llama.rs` comme
adaptateur llama-server. Les nouvelles routes vont dans des modules dédiés, pas dans
un `match` géant.

### 5.10 `kinetic-cli`

| Écart | Fichier cible | Effort |
|---|---|---|
| `kinetic imatrix` | nouveau `imatrix.rs` | L |
| `kinetic tokenize` / `detokenize` | nouveau `tokenize.rs` | S |
| `kinetic apply-template` | `template.rs` | S |
| `kinetic gguf-split` | nouveau `gguf_split.rs` | S |
| `kinetic export-lora` | `library/` | M |
| `kinetic cvector` | nouveau | M |
| `kinetic tts` | `kinetic-audio` | L |
| `kinetic rpc` | `kinetic-rpc` | L |
| `kinetic convert` | `kinetic-convert` | XL |
| Flags multi-GPU/MoE | `llama_flags.rs`, `main.rs` | M |
| Flags spéculatifs | `llama_flags.rs` | S |
| Flags grammaire CLI | `main.rs` | S |
| `bench` format llama-bench | `bench.rs` | M |

### 5.11 `kinetic-capi` et bindings

| Écart | Fonctions à ajouter | Effort |
|---|---|---|
| Modèle | `kinetic_model_meta`, `n_params`, `n_ctx`, `arch`, `desc` | S |
| Vocab | `tokenize`, `token_to_piece`, `n_vocab`, `bos/eos` | M |
| Session | `set_sampling`, `set_grammar`, `set_lora`, `save_state`, `load_state` | L |
| Batch | `decode`, `encode`, `batch_*` | L |
| Sampler | chaîne complète, `sample`, `accept` | L |
| Multimodal | `image`, `audio`, `video` | XL |
| Backend | `backend_count`, `backend_info`, `device` | M |
| Perf/log | `perf_context`, `log_set` | S |

Les bindings Python/Node/Go peuvent être des wrappers de l'ABI ou du serveur HTTP.
Pour la parité avec `llama-cpp-python`, il faut au minimum : `Llama`, `create_chat_completion`,
`create_completion`, `embed`, `tokenize`, `detokenize`, `grammar`, `response_format`,
`logprobs`, `tools`, `Llama.from_pretrained`.

### 5.12 Nouveaux crates

| Crate | Rôle | Priorité | Justification |
|---|---|---|---|
| `kinetic-hip` | backend AMD ROCm | **P0 local** | seul moyen d'exploiter ROCm sans passer par RADV |
| `kinetic-audio` | encodeurs audio, TTS, ASR, mel | P1 | modalité distincte, comme `kinetic-vision` |
| `kinetic-diffusion` | LLaDA/Dream, sampler itératif | P2 | modèle non autoregressif |
| `kinetic-rpc` | backend distribué | P2 | backend, pas modèle |
| `kinetic-convert` | HF→GGUF | P2 | outil, pas runtime |
| `kinetic-train` | fine-tuning | P3 | optionnel, hors priorité |
| `kinetic-sycl` | backend Intel | P3 | idem |
| `kinetic-metal` | backend Apple natif | P3 | idem |

Ajouter un crate par backend est cohérent avec `kinetic-cpu`, `kinetic-cuda`,
`kinetic-vulkan`. Ne pas créer un crate fourre-tout « gpu ».

---

## 6. Plan d'implémentation sans casser l'architecture

### 6.1 Règles

1. **Registre d'abord** : toute architecture est un fichier `arch/<nom>.rs` + une
   ligne dans `FAMILIES`. Aucun `match` central.
2. **Blocs d'abord** : la math réutilisable va dans `blocks/`. Les fichiers
   d'architecture ne font que mapper des tenseurs et déclarer des capacités.
3. **Un backend par crate** : `kinetic-cpu`, `kinetic-cuda`, `kinetic-vulkan`, puis
   `kinetic-hip`, `kinetic-sycl`, `kinetic-metal`. Le code de quantization/layout est
   partagé dans `kinetic-quant`.
4. **Routes par module** : `server.rs` enregistre, `llama.rs` adapte, les nouvelles
   API vont dans des modules dédiés.
5. **Sampling partagé** : le cœur reste dans `kinetic-models/src/speculative/sampler.rs`,
   utilisé par le serveur et le CLI.
6. **Flags llama.cpp traduits** : `llama_flags.rs` doit soit traduire, soit expliquer,
   jamais échouer sur un flag connu.
7. **Exactitude préservée** : tout nouveau chemin doit passer les tests de
   batch-invariance et de draft exact, ou être opt-in.
8. **Pas de dépendance nouvelle** sans nécessité absolue. Pour HIP, réutiliser les
   sources de kernels communes via macros/build scripts plutôt que dupliquer ; pour
   SYCL, réutiliser la même couche de quantization/layout.

### 6.2 Lots

#### P0 — Déblocage local (S/M/L)

Objectif : sur la RX 9070 XT, Kinetic charge et sert les modèles courants via Vulkan,
puis via HIP, sans régression face à llama.cpp.

- Sampling : `ignore_eos`, `min_keep`, `dynatemp_range`, `dynatemp_exponent`,
  `adaptive_target`, `adaptive_decay`, `reasoning_budget`.
- Serveur : `POST /props`, `POST /slots/:id_slot`, `/infill`, `/apply-template`,
  `/lora-adapters`, `n > 1`, timings par token, `--api-key-file`.
- **Vulkan** : GEMM de prompt tuilé, coopmat, KV q8_0, soumission asynchrone
  (les types IQ3_XXS/IQ4_XS/Q4_0 sont faits depuis le 22 nov.). C'est le backend qui
  doit d'abord être comparé à llama.cpp Vulkan.
- **HIP/ROCm** : squelette `kinetic-hip`, runtime, GEMV Q4_K/Q5_K/Q6_K/Q8_0,
  attention FP16, build `gfx1201`. C'est le backend qui doit ensuite être comparé à
  llama.cpp ROCm.
- Quantization : NVFP4 + F8_E4M3 runtime (parité NVIDIA ; FP8 utile sur ROCm),
  imatrix, cibles IQ prioritaires (IQ4_XS, IQ3_XXS, IQ2_XS),
  `--output-tensor-type`, `--token-embedding-type`.
- Tokenizers : les 15 prétokenizers les plus fréquents (DeepSeek, Command-R,
  StarCoder, Falcon, MPT, ChatGLM, OLMo, StableLM2, GPT-4o, Llama4, Pixtral,
  Seed-Coder, Kimi-K2, Grok-2, MiniMax-M2).
- Architecture : `qwen3moe`, `qwen2moe`, `qwen3next`, `phi3`, `phimoe`, `gemma2`,
  `llama4` (si le bloc iRoPE est prêt), `gpt-oss`, `deepseek2`/`deepseek32`,
  `glm4`/`glm4moe`, `mistral3`/`mistral4`, `command-r`, `dbrx`, `seed_oss`.
- Split GGUF en lecture.

#### P1 — Couverture modèles, multimodal et backends locaux avancés (M/L)

- MLA/DSA/lightning/RWKV/Mamba/T5/BitNet.
- Projecteurs vision majeurs : Qwen2VL/2.5VL/3VL, Llama4, InternVL, Pixtral,
  Gemma3, SmolVLM, MiniCPM-V, GLM4V, HunyuanVL.
- Audio : Whisper/Voxtral/Qwen3-ASR en entrée, Kokoro/Zonos3/PocketTTS en sortie.
- OCR : DeepSeek-OCR, PaddleOCR.
- Draft model + EAGLE3 + ngram map.
- **HIP** : WMMA, flash attention, MoE groupé, hipGraph, KV quantifié, layer split.
- **Vulkan** : MoE, layer split, flash attention avancée, KV q4/q5.
- **CPU** : AVX-512/VNNI et repack (le 7800X3D les expose ; gain direct hors VRAM).
- Multi-GPU row/tensor split, `--override-tensor`, MoE cache explicite.
- Router mode.

#### P2 — Parité NVIDIA et autres plateformes (L/XL)

- CUDA BF16/NVFP4/FP8, IQ denses + GEMM IQ, row/tensor split, KV quantifié
  (pour les utilisateurs NVIDIA, pas pour cette machine).
- SYCL, Metal natif, CPU AMX/i8mm/SVE, NUMA.
- `/v1/responses`, `/v1/messages`, realtime WS, TTS.
- Bindings C complètes + Python.

#### P3 — Long tail (XL)

- Toutes les architectures restantes.
- Diffusion, training, RPC, conversion HF.
- MCP/tools, GCP compat, CORS proxy.
- WebGPU, CANN, MUSA, OpenCL, VirtGPU.

### 6.3 Critères d'acceptation

Pour chaque lot :

- **Modèle** : `kinetic info` charge le GGUF ; `kinetic run` greedy donne des tokens
  équivalents à llama.cpp sur un jeu de prompts ; `conformance.py` passe.
- **Serveur** : un client OpenAI/llama.cpp existant fonctionne sans modification ;
  les champs inconnus sont ignorés ou rejetés avec un message clair.
- **Performance** : `kinetic bench --device vulkan` contre `llama-bench -dev Vulkan0`,
  puis `kinetic` HIP contre `llama-bench -dev ROCm0`, sur le même fichier, le même
  matériel et les mêmes threads ; pas de régression > 5 % sur les cas déjà couverts.
- **Exactitude** : tests batch-invariance, draft exact, seed identique.
- **Mémoire** : pic VRAM/RAM mesuré, pas de fuite, pas de cache persistant non borné.
- **Noyau** : tout nouveau type de poids passe trois tests avant d'être gardé — la
  référence f64 à 1e-5 relatif (`check`), l'égalité de bits entre passe mono-token et
  passe prompt (`check_prompt_pass`), et le garde-fou
  `every_gemv_type_survives_a_prompt_pass` qui exige un noyau many-token pour chaque
  type admis. Un noyau qui échoue est retiré, pas toléré.
- **Documentation** : `docs/src/models.md`, `devices.md`, `server.md`,
  `llama-flags-table.md` mis à jour ; pas de contradiction (voir 9.6), et chaque
  tableau de capacités daté avec le commit de son relevé.

---

## 7. Modules complémentaires de performance

Cette section liste les modules qui ne sont pas nécessaires à la parité *fonctionnelle*
mais qui donnent une performance égale ou supérieure à llama.cpp. Chaque module indique
le crate cible et le gain attendu.

### 7.1 CPU

| Module | Description | Crate | Gain |
|---|---|---|---|
| AVX512 + VNNI | Dot products int8/bf16, masques, `_mm512_dpbusd_epi32` ; le 7800X3D les expose | `kinetic-cpu` | 1,3–2× prompt |
| AMX | Tuiles `tmm`, int8/bf16, pour les GEMM de prompt | `kinetic-cpu` | 2–4× prompt Xeon |
| i8mm / SVE | ARM server et Apple CPU | `kinetic-cpu` | 1,2–2× |
| Repack | Packing des poids par layout CPU (K-quants, IQ) | `kinetic-quant` + CPU | 5–20 % décodage |
| Threadpool | Barrière dédiée, affinité, spin court | `kinetic-cpu` | 3–10 % |
| NUMA | Allocation et duplication par nœud | `kinetic-cpu` | jusqu'à 2× multi-socket |
| Huge pages / mlock / dio | Moins de page faults, lectures directes | `kinetic-cpu` | 5–15 % chargement/long contexte |
| BLAS | Fallback prompt sur CPU exotiques | `kinetic-cpu` | stabilité |
| Cache KV quantifié | q4/q5/bf16, attention adaptée | CPU + HIP + Vulkan + CUDA | 2–4× mémoire KV |

Le CPU local est un Zen 4 : AVX-512 est disponible mais Kinetic ne l'utilise pas
encore. C'est un gain gratuit pour les cas où le modèle déborde de la VRAM ou pour
les embeddings.

### 7.2 ROCm/HIP (priorité locale)

| Module | Description | Gain |
|---|---|---|
| Runtime HIP | `hipcc --offload-arch=gfx1201`, streams, events, `hipMallocAsync` | base |
| GEMV quantifié | Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS/IQ3_XXS, dequant en registres | décodage |
| WMMA FP16/BF16 | Tuiles 16×16×16 wave32, `rocWMMA` | 2–4× prompt |
| Dot products | `v_dot2_f32_f16`, `v_dot4_i32_i8`, `__builtin_amdgcn_sdot4` | GEMV quantifié |
| rocBLAS / hipBLASLt | GEMM de prompt, fallback et autotuning | prompt |
| Flash attention | Online softmax, GQA, SWA, paged KV, split-K | 1,2–2× contexte long |
| MoE groupé | Top-k, tuiles par expert, fusion gate/up/down | 1,5–3× MoE |
| hipGraph | Capture du pas de décodage | souvent ~2× c1 ; 5–15 % si les noyaux dominent |
| Pipelines async | Plusieurs streams, mémoire épinglée, recouvrement copies/calcul | 5–15 % |
| KV quantifié | q8_0/q4_0/q5_0/bf16 | 2× mémoire KV |
| Layer split | dGPU + iGPU, sinon RAM | modèles > 16 Gio |
| Télémétrie | `rocm-smi`, `hipEvent`, `rocprof` | ops |
| Réglage | Occupation, LDS 64 Kio, taille de tuile, wave32 | 5–15 % |
| Validation gfx1201 | Tests de correction, pas seulement de vitesse | fiabilité |

**Approche** : ne pas dupliquer les kernels CUDA à la main. Extraire les sources
communes (quantization, layout, dispatch) dans un petit crate de sources partagées,
puis compiler avec `hipcc` dans `kinetic-hip`. Les primitives diffèrent réellement :
RDNA 4 est wave32, n'a pas MFMA (CDNA), et il faut valider BF16/FP8 sur `gfx1201`
avant de les activer.

### 7.3 Vulkan (priorité locale)

| Module | Description | Gain |
|---|---|---|
| GEMM de prompt | Tuiles en mémoire partagée, split-K, dequant en registres | 2–4× prompt |
| coopmat KHR | `VK_KHR_cooperative_matrix` (disponible sur RADV gfx1201) ; coopmat2 si exposé | 2–5× prompt |
| Sous-groupe 32/64 | Choisir par noyau ; RADV expose 32–64, le défaut est 64 | 5–15 % memory-bound |
| Types restants | IQ4_NL, IQ3_S, Q5_0/Q5_1/Q4_1, Q2_K/Q3_K, MXFP4 | charge les modèles restants |
| KV quantifié | q8_0/q4_0/q5_0/bf16 | 2–4× mémoire KV |
| Softcap | Softcap d'attention (Gemma 2) ; le reste de l'attention est déjà flash-style | parité Gemma 2 |
| MoE | Routage top-k, GEMV/GEMM groupé, experts VRAM/hôte | parité MoE |
| Layer split | VRAM + RAM, ou dGPU + iGPU | modèles > 16 Gio |
| Soumission asynchrone | Plusieurs frames en vol, timeline semaphores ; le graphe enregistré existe déjà | 5–15 % |
| Cache de pipelines | `VkPipelineCache` persisté | démarrage |
| Sous-allocation | Un grand buffer, sous-allocation par tenseur | fragmentation |
| BF16/FP8 | Seulement si le driver les expose ; RADV rapporte `bf16: 0` | parité, sinon FP16 |
| Validation RADV | Vérifier chaque chemin sur gfx1201 ; le driver impose ses limites (sous-groupe, BF16/FP4) | stabilité |

Le point le plus rentable est le **GEMM de prompt**. Mesure du 22 nov. sur
`Qwen3.8-27B-IQ3_XXS` (`kinetic-work` `85d29aa` contre `llama.cpp` Vulkan0) :

| | kinetic Vulkan | llama.cpp Vulkan | écart |
|---|---:|---:|---:|
| pp512 | 261,9 t/s | 1154,2 t/s | 4,4× |
| pp2048 | 257,9 t/s | 1149,0 t/s | 4,5× |
| tg128 | 38,4 t/s | 40,1 t/s | 1,04× |

`PROMPT_BATCH` vaut 128 et le chemin « many tokens » (`shaders/gemv_many.wgsl`) garde
le décodage des poids en vol pour 32 tokens, ce qui a porté le préfill de 154 à
262 t/s. Ce n'est toujours pas un GEMM tuilé : chaque ligne refait ses sommes par
sous-groupe de 32, sans tuile en mémoire partagée ni split-K, et les poids restent lus
une fois par ligne. C'est là que se joue l'écart de 4,4×, avant coopmat. L'attention,
elle, est déjà écrite en flash-style avec online softmax (`shaders/attend.wgsl` ligne 6),
GQA et SWA : il ne faut pas la compter comme un manque, seulement comme un noyau à
régler.

### 7.4 CUDA (parité NVIDIA, secondaire)

| Module | Description | Gain |
|---|---|---|
| BF16 | GEMM/GEMV/attention BF16, modèles BF16 | 1,1–1,5× |
| NVFP4 | Tensor cores Blackwell, échelle E4M3 | 2–4× vs Q4_K |
| FP8 | Hopper/Blackwell, poids et KV | 1,5–2,5× ; 2× KV |
| IQ denses | GEMV IQ4_XS/IQ3_XXS/IQ2_XS/IQ2_XXS pour les linears denses | active les denses bas bit |
| GEMM IQ | GEMM tuilé pour les types IQ | prompt des modèles IQ |
| Row/tensor split | Découpe des matrices, réduction, `--tensor-split` | latence 1 modèle |
| KV quantifié | q4_0/q4_1/iq4_nl/q5_0/q5_1/bf16 | 2–4× mémoire |
| Softcap | Softcap d'attention (Gemma 2) | parité Gemma 2 |

CUDA graphs, le MoE groupé, la flash attention, le GEMV int8 MMA et les copies P2P
sont déjà implémentés : ils ne doivent pas être présentés comme des gains à venir.
Les modules ci-dessus sont les compléments réels. Ils restent dans le plan de parité,
mais ils ne sont pas mesurables sur cette machine et ne doivent pas passer avant
HIP/ROCm ou Vulkan.

### 7.5 Autres backends

| Module | Description | Gain |
|---|---|---|
| SYCL | Intel GPU, oneAPI | parité Intel |
| Metal natif | simdgroup matrix, pas de couche Vulkan | 1,2–2× Apple |
| CANN | Ascend | reach |
| MUSA | Moore Threads | reach |
| OpenCL | fallback | reach |
| WebGPU | Navigateur | reach |
| RPC | Backend distribué | scale-out |
| BLAS | Fallback | stabilité |

**Approche** : ne pas maintenir 6 copies des kernels. Extraire les parties communes
(quantization, layout, dispatch) dans `kinetic-quant` et un petit crate de traits de
backend, puis garder un crate par backend pour la compilation et les tests. Si un
backend ne peut pas partager, le documenter.

### 7.6 Ordonnancement, cache et mémoire

| Module | Description | Crate | Gain |
|---|---|---|---|
| Chunked prefill | Découper les longs prompts, réduire TTFT | `kinetic-server` | TTFT |
| Priorité / fairness | Éviter qu'une requête affame les autres | `engine.rs` | latence p99 |
| Context checkpoints | Snapshots tous N tokens, SWA-aware | `warm/` | reprise longue conversation |
| Cache RAM configurable | `--cache-ram`, `--kv-unified` | `warm/`, `engine.rs` | hit rate |
| `--cache-reuse` | Réutilisation partielle de préfixe | `warm/` | prompt |
| MoE cache explicite | `--moe-cache-mib`, `--moe-cache-cpu-share` | `memory.rs` | MoE |
| `--override-tensor` | Placement par motif | `per_layer.rs` | mémoire/perf |
| `--defrag-thold` | Défragmentation KV | `memory.rs` | latence |
| Pinned host memory | Experts MoE épinglés | `memory.rs` | upload |
| Warm Start disque | Déjà présent ; ajouter compression/version | `warm/disk.rs` | hit rate |

### 7.7 Quantization

| Module | Description | Gain |
|---|---|---|
| Imatrix | Calibration d'importance par tenseur | qualité à bas bit |
| IQ encode | Q2/Q3/Q4 IQ | modèles plus petits |
| NVFP4/FP8 | Poids et KV | vitesse/mémoire |
| Politique par tenseur | Garder embeddings/attention en haut bit | qualité |
| Requantification | `--allow-requantize` | workflow |
| Split/merge | `gguf-split` | gros modèles |
| Calibration intégrée | `kinetic imatrix` + dataset | autonomie |

Sur cette machine, la priorité de *production* est **IQ4_XS/IQ3_XXS** : les modèles
locaux utilisent déjà ces formats, mais Kinetic ne sait pas encore les encoder — seule
la lecture est faite. **FP8 sur ROCm** vient ensuite si `gfx1201` le valide. NVFP4
reste une cible de parité NVIDIA.

### 7.8 Spéculatif

| Module | Description | Gain |
|---|---|---|
| Draft model | Source optionnelle, vérifiée exactement | 1,5–3× modèles sans MTP |
| EAGLE3 | Tête de draft | 2–3× |
| DFlash/DFlash2/DSpark/LiliCorr | Têtes récentes | 2–4× selon modèle |
| Ngram map/k4v/mod/cache | Meilleur hit rate agents | 1,2–2× |
| Lookup cache statique/dynamique | Persistance entre runs | hit rate |
| Backend sampling | Échantillonnage GPU | 5–15 % |
| Politique de coût | Choisir la source selon le gain mesuré | évite les régressions |

**Contrainte** : toute nouvelle source doit passer le même vérificateur exact. Un
draft model ne doit jamais changer la distribution ; le rejet exact reste la règle.
Sur AMD, mesurer la spéculation séparément en Vulkan et en HIP : les coûts de
vérification et de draft ne sont pas les mêmes.

### 7.9 Serveur et observabilité

| Module | Description | Gain |
|---|---|---|
| Router multi-modèles | Chargement/déchargement, SSE | UX/scale |
| Métriques Prometheus | Labels, histogrammes, slots | ops |
| Timings par token | TTFT, tok/s, cache hit | ops |
| `n > 1` | Choix multiples | UX |
| Responses/Anthropic | Compat clients | reach |
| Realtime WS | Audio streaming | reach |
| MCP/tools | Outils locaux | reach |
| TLS/API key file | Sécurité | ops |
| CORS/proxy | Intégration web | ops |

### 7.10 Plafonds théoriques et méthode de mesure

Les gains des tableaux 7.1 à 7.9 sont des **hypothèses** : ils doivent être validés
par la méthode ci-dessous, sur la machine cible, avant d'être annoncés.

#### Plafonds

**Décodage = bande passante mémoire.** Les octets lus par token émis sont :

```text
B_target = poids denses
         + octets_experts × (top_k / n_experts)
         + octets_lm_head
         + octets_kv_par_token × contexte
         + octets_état_récurrent × 2
```

Le `lm_head` pèse souvent 10–25 % du total sur les gros vocabulaires ; c'est
pourquoi Shortlist compte. Avec spéculation (k drafts, longueur acceptée `a`) :

```text
B_par_token ≈ (B_verify + k × (B_draft + B_draft_head)) / a
plafond_c1 ≈ bande_passante_mesurée / B_par_token
```

**Préfill = calcul.** `flops/token ≈ 2 × params linéaires actifs + 4 × couches
d'attention × têtes × head_dim × position`, puis `plafond ≈ 0,5 × FLOPS_mesurés /
flops_moyens`. Les tensor cores int8/FP8 doublent le pic sur les parties concernées ;
les petits GEMM MoE par expert font chuter l'efficacité.

**TTFT, démarrage, KV.** TTFT court ≈ une passe avant + tokenize + ordonnancement ;
démarrage ≈ octets de poids / débit disque + allocation ; KV :
`n_couches_kv × 2 × n_têtes_kv × head_dim × octets_par_élément` par token, puis
`streams_full_context = (mémoire_dispo − poids − états − runtime) / (KV_par_token ×
contexte)`.

Mesurer le matériel, pas les fiches techniques : copie device-to-device pour la
bande passante, GEMM 4096³ et 8192³ pour les FLOPS, fréquences sous charge.

**Exemple arithmétique pour la RX 9070 XT** (pas une mesure) : bande passante
nominale 640 Go/s (256 bits × 20 Gbit/s), à remplacer par la copie device-to-device
mesurée, par exemple ~600 Go/s utiles. Un dense 9B en Q4_K_M ≈ 5,3 Gio de poids, un
`lm_head` ≈ 0,4 Gio, un KV f16 ≈ 0,125 Mio/token, soit 1 Gio à 8k de contexte. À
600 Go/s : `(5,3 + 0,4 + 1,0)` Gio = 7,19 Go → ~83 tok/s à 8k, ~98 tok/s à contexte
court. Avec une spéculation qui accepte 2,5 tokens par vérification et un draft bon
marché, le plafond du target monte d'environ 2,5×, mais il faut retrancher les octets
du draft. Si la mesure réelle est à 60 % de ce plafond, le problème est dans les
noyaux ou l'ordonnancement, pas dans la bande passante.

Côté préfill, la RX 9070 XT annonce ~97 TFLOPS FP16 (64 CU, wave32) ; à 50 % pratique,
~48 TFLOPS. Pour 9B actifs, `2 × 9e9 = 18 GFLOP/token`, soit un plafond de l'ordre de
2 600 tok/s avant l'attention et les limites mémoire. C'est un plafond théorique : le
GEMM de prompt et la bande passante réelle décideront.

#### Bruit et protocole

- **Un seul changement par expérience**, un fichier de configuration par essai.
- **Contrôles des deux côtés** : contrôle → candidat → contrôle. Médiane de ≥ 6
  échantillons ; le bruit typique est de ±5–10 %.
- **Accepter** seulement si le candidat bat les deux contrôles sur la métrique cible
  et ne régresse rien de plus de ~3 %.
- **Même fichier, même matériel, mêmes threads** : `kinetic bench` et `llama-bench`
  avec `pp512`, `tg128`, `pp2048`, `tg512`, et un test d'édition (comme
  `docs/src/benchmarks.md`).
- **Multi-requêtes** : 1, 4, 16, 64 sessions ; débit agrégé et p99.
- **Contexte long** : 8k, 32k, 128k ; cache KV f16/q8/q4 ; vérifier la stabilité
  pleine longueur.
- **Backends, priorité locale** : Vulkan0 (RADV gfx1201) puis ROCm0 (HIP gfx1201) ;
  CPU x86 en repli. CUDA (Pascal/Turing/Ampere/Hopper/Blackwell) et les autres
  plateformes viennent après, pour la parité.
- **CI** : seuil de régression 5 %, rapport JSON, comparaison automatique.
- **Watchdog mémoire** : tuer la mesure si le pic dépasse la limite ; sur mémoire
  unifiée, mesurer la résidence driver, pas seulement `MemAvailable`.

#### Portes de stabilité

Un candidat n'est « meilleur » qu'après :

- une recherche de needle à ~50 % puis ~100 % du contexte, trouvée exactement ;
- un soak de requêtes concurrentes pleine longueur si le KV est dimensionné pour N ;
- une vérification d'équivalence greedy si le changement prétend être lossless ;
- une vérification que la sortie n'est pas dégénérée ou en boucle.

`kinetic bench` doit accepter les options de `llama-bench` (`-p`, `-n`, `-r`, `-b`,
`-ub`, `-t`, `-ngl`, `-sm`, `-fa`, `-mmp`, `-o json/csv/md`) pour que la comparaison
soit directe.

### 7.11 Ordre des leviers (à vérifier sur la machine cible)

1. **Caches persistants** (JIT, compilation, pipelines, autotune) : démarrage −50 à
   −70 % sur les stacks qui compilent. Kinetic compile ses kernels à la construction,
   mais Vulkan et HIP peuvent bénéficier d'un cache de pipeline persisté.
2. **Decode graphs** (hipGraph sur ROCm, soumission asynchrone sur Vulkan) : souvent
   ~2× en c1 ; 5–15 % quand les noyaux dominent déjà.
3. **Spéculation** : balayer la longueur de draft par concurrence et contexte ;
   élaguer le vocabulaire du `lm_head` du draft (le target vérifie toujours sur le
   vocabulaire complet) ; garder l'échantillonnage sur GPU ; vérifier avec une
   attention de type décodage, pas préfill.
4. **Concurrence et KV** : dimensionner N × contexte, les slots d'état des modèles
   hybrides (SSM/GDN) et le cache de préfixe ; confirmer `cached_tokens > 0`.
5. **Préfill** : chunk size (compromis débit/stall), attention plus rapide, GEMM
   int8/FP8 en trade assumé, éviter les falaises de padding des GEMM MoE groupés.
6. **Threads et tuiles des noyaux** : les défauts viennent souvent d'une autre carte.
7. **Travail noyau** : seulement après plateau des leviers de configuration ;
   profil par étape, test bit-exact ou tolérance serrée à chaque nombre de lignes M.

**Avertissement** : certains leviers perdent sur les machines à faible bande passante
ou les prompts difficiles. La spéculation peut régresser ; les MTP faibles au-delà de
la profondeur 1 existent ; le KV FP8 peut diviser le débit d'attention. Toujours
mesurer, jamais généraliser depuis une fiche technique.

### 7.12 Journal d'expériences et artefacts

Pour que « égal ou meilleur » soit vérifiable et reproductible, chaque campagne de
performance doit produire :

| Artefact | Contenu |
|---|---|
| `THEORY.md` | plafonds calculés pour la machine et le modèle, avec les mesures matérielles |
| `PLAN.md` | journal horodaté : baseline, chaque essai, verdict, raison, échecs inclus |
| `ledger.jsonl` | une ligne par exécution : configuration, métriques, médiane, écart-type |
| `configs/*.json` | une configuration par essai, dérivée du meilleur courant |
| `best-config.json` | configuration retenue après passage des portes de stabilité |

Le journal de campagne est `PLAN.md`, à la racine du dépôt : il contient l'état de la
machine, le protocole, les essais horodatés avec leur verdict et les écarts connus.
`ledger.jsonl`, `configs/` et `best-config.json` sont créés à la première campagne qui
en a besoin ; tant qu'aucun essai n'est automatisé, `PLAN.md` en tient lieu. Les deux
premiers essais (noyaux IQ, puis many-token) y sont consignés.

Le journal doit aussi consigner l'état de la machine : profil d'alimentation, autres
processus GPU, version du driver (RADV/Mesa, ROCm/HIP), commit Kinetic, révision des
poids, et le backend utilisé (`vulkan`, `hip`, `cpu`). Un essai dont l'état a changé
est invalide. Les résultats négatifs sont conservés : ils évitent de refaire le même
essai six mois plus tard.

**Ordre de priorité des campagnes** :

1. **Vulkan0** : décodage c1 sur un modèle dense, puis un MoE, avec et sans RAD-1.
2. **ROCm0** : les mêmes campagnes dès que `kinetic-hip` sait charger le modèle.
3. **Décodage cN** (4, 16, 64) avec Convoy et Warm Start, sur les deux backends.
4. **Préfill** à 512, 2k, 8k, 32k tokens ; c'est là que le GEMM de prompt et coopmat
   doivent se voir.
5. **TTFT** et démarrage à froid/chaud, cache de pipeline inclus.
6. **Mémoire** : pic, KV plein contexte, marge sous 16 Gio.
7. **Comparaison llama.cpp** : `llama-bench` Vulkan0 puis ROCm0, même fichier, mêmes
   réglages ; jamais un chiffre CUDA.

Une optimisation n'est publiée qu'après avoir été reproduite sur une build propre,
avec les poids épinglés par révision et le commit Kinetic épinglé. Les chiffres de
démonstration (thinking off, médianes) et les chiffres de porte (thinking on,
premières 30 s) ne sont pas comparables et doivent être étiquetés séparément.

---

## 8. Priorisation et effort

| Lot | Contenu | Impact | Effort | Risque |
|---|---|---|---|---|
| **P0.1** | Sampling manquant + routes serveur de base | élevé | S/M | faible |
| **P0.2** | **Vulkan** : GEMM de prompt tuilé, coopmat, KV q8_0, soumission asynchrone | très élevé | L | moyen |
| **P0.3** | **`kinetic-hip`** : runtime, GEMV Q4_K/Q5_K/Q6_K/Q8_0/IQ3_XXS/IQ4_XS, attention FP16, build gfx1201 | élevé | L/XL | élevé |
| **P0.4** | NVFP4/F8 runtime (parité), imatrix, cibles IQ | élevé | M/L | moyen |
| **P0.5** | 15 prétokenizers + split GGUF | moyen | S/M | faible |
| **P0.6** | 10–15 architectures majeures | très élevé | L | moyen |
| **P1.1** | MLA/DSA/SSM/T5/BitNet | élevé | XL | élevé |
| **P1.2** | Projecteurs vision majeurs | élevé | L | moyen |
| **P1.3** | Audio/OCR | élevé | XL | élevé |
| **P1.4** | Draft model/EAGLE3/ngram | moyen | L | moyen |
| **P1.5** | **HIP avancé** : WMMA, flash, MoE, hipGraph, KV quantifié | élevé | L/XL | élevé |
| **P1.6** | **Vulkan avancé** : MoE, layer split, flash, KV q4/q5 | élevé | L/XL | élevé |
| **P1.7** | Multi-GPU row/tensor + override | moyen | XL | élevé |
| **P1.8** | **CPU AVX-512/VNNI + repack** (Zen 4 local) | moyen | L | faible |
| **P2.1** | CUDA BF16/NVFP4/FP8, IQ denses, row/tensor split (NVIDIA) | élevé | L | moyen |
| **P2.2** | SYCL/Metal, CPU AMX/i8mm/SVE, NUMA | moyen | L/XL | moyen |
| **P2.3** | Responses/Anthropic/realtime/TTS | moyen | L/XL | moyen |
| **P3** | Long tail, diffusion, training, RPC, bindings | variable | XL | variable |

**Recommandation** : sur cette machine, P0.2 (GEMM Vulkan) passe avant P0.3 (HIP), et
les deux avant tout travail CUDA. La raison est mesurée, pas théorique : depuis le
22 nov., le décodage Vulkan est à 95 % de llama.cpp Vulkan (38,4 contre 40,1 t/s) et
le préfill à 23 % (262 contre 1154 t/s). Un backend HIP complet (L/XL, risque élevé)
récupérerait au mieux quelques pour cent de décodage, alors qu'un GEMM tuilé (L)
attaque le seul écart qui reste. P0.1, P0.4 et P0.5 avancent en parallèle car ils ne
dépendent pas du backend. P1.5/P1.6 sont la suite naturelle : ils transforment un
backend qui « fonctionne » en backend qui « gagne ». P2.1 (CUDA) reste utile pour la
parité, mais ne doit pas consommer le temps de l'équipe avant que Vulkan soit
compétitif.

---

## 9. Risques et notes de compatibilité

### 9.1 Cible mouvante

llama.cpp ajoute des architectures et des types chaque semaine. « 100 % » n'est
atteignable qu'avec un **contrat de parité** :

> Tout GGUF que llama.cpp charge est soit chargé par Kinetic, soit refusé avec un
> message qui nomme l'architecture, le type ou la fonctionnalité manquante, et un
> lien vers l'issue de suivi.

C'est plus réaliste qu'une course à l'exhaustivité, et c'est déjà l'esprit de
`kinetic info`.

### 9.2 Exactitude vs llama.cpp

Kinetic garantit l'exactitude vis-à-vis de lui-même. Il ne faut pas promettre des
tokens bit-à-bit identiques à llama.cpp : les kernels, les ordres de sommation et les
arrondis diffèrent. La promesse doit être :

- greedy : mêmes tokens *si* les logits sont équivalents à la tolérance près ;
- sampling : même distribution, même seed → même sortie *dans Kinetic* ;
- documenter les différences attendues (CPU vs GPU, quant types, KV quantifié).

### 9.3 Mémoire et dtype

- Les modèles 4 bits peuvent avoir une seconde dimension réduite : la détection ne
  doit lire que la première.
- BF16 CUDA manquant : ne pas ajouter de cast défensif dans le modèle ; le corriger
  dans la couche d'exécution. Sur cette machine, le bloqueur n'est pas CUDA mais
  **BF16/FP8 sur ROCm et Vulkan** : RADV rapporte `bf16: 0` et `fp4: 0`, donc le
  chemin Vulkan doit rester FP16 (ou FP32) tant que le driver ne les expose pas.
- NVFP4/FP8 : vérifier les échelles, les layouts et les contrats de dtype. NVFP4 est
  une cible NVIDIA ; sur `gfx1201`, valider FP8 avant de l'activer.
- KV quantifié : documenter que la sortie change (arrondis) ; ne pas l'activer par
  défaut.

### 9.4 Sécurité

- Les server tools/MCP exécutent des processus locaux. Si Kinetic les ajoute, il faut
  une allowlist, une confinement de chemins, pas de réseau, et une documentation
  explicite. Ne pas copier le modèle de llama.cpp sans réfléchir aux permissions.
- TLS : recommander un proxy, ou ajouter un mode natif avec certificats.
- API key : `--api-key-file` pour éviter les clés en clair dans la ligne de commande.

### 9.5 ABI et compatibilité

- L'ABI C actuelle est minimale : ajouter des fonctions est sûr, changer les
  signatures existantes ne l'est pas.
- Les noms de commandes, d'options et de routes existants doivent rester stables.
- Les flags llama.cpp doivent continuer à être traduits ou expliqués.

### 9.6 Dérive documentaire

Il y a déjà une contradiction : `docs/src/faq.md` dit « pas de macOS », alors que
`docs/src/devices.md` et `roadmap.md` décrivent macOS/Apple Silicon via Vulkan/Metal.
Il faut corriger avant d'ajouter de la documentation de parité. De même,
`docs/src/models.md` n'affiche pas Q1_0/Q2_0/PQ2_0/PTQ1_0 alors que
`kinetic-quant/src/types.rs` les connaît.

Ce document a lui-même dérivé : ses tableaux de types (§4.3) et de modèles (§4.13)
décrivaient encore l'état d'avant les noyaux IQ alors que la branche de travail les
avait déjà. Règle à tenir : chaque tableau de capacité porte son commit de relevé, et
un tableau non daté n'est pas une preuve.

### 9.7 Licence

Kinetic est Apache-2.0, llama.cpp est MIT. Le port de code est possible avec
attribution et conservation des notices. Les formats GGUF/GGML sont documentés et
réutilisables. Vérifier les licences des modèles et des projecteurs (certains sont
sous des licences spécifiques).

### 9.8 Coût de maintenance

Chaque backend et chaque architecture multiplient la matrice de tests. Il faut :

- des tests de conformance automatisés par architecture ;
- des tests de non-régression de performance ;
- des tests d'API (routes/champs) ;
- une CI multi-backend (au moins CPU + Vulkan + HIP, puis CUDA).

---

## 10. Annexes

### A. Architectures llama.cpp manquantes (159 noms)

Liste brute, à des fins de suivi. Les groupes sont indicatifs ; certains noms sont des
alias, des têtes de draft ou des tours multimodales.

**Transformers proches / MoE simples**
`afmoe`, `apertus`, `arcee`, `arctic`, `baichuan`, `bailingmoe`, `bailingmoe2`,
`bailingmoe3`, `beam`, `bloom`, `chatglm`, `clef`, `clover_moe`, `codeshell`,
`cohere2`, `cohere2moe`, `command-r`, `dbrx`, `deci`, `deepseek`, `dots1`,
`domynedge`, `ernie4_5-moe`, `exaone`, `exaone-moe`, `falcon`, `flint`, `gemma`,
`gemma2`, `gemma4-assistant`, `gpt2`, `gptj`, `gptneox`, `granite_swa`,
`graniteswitch`, `grok`, `grovemoe`, `hrm_text`, `hunyuan-moe`, `hy_v3`, `hy_v4`, `inkling2`,
`inkling2_loop`, `internlm2`, `jais`, `jais2`, `k2-horizon`, `kolibri1`, `laguna`,
`limite`, `maincoder`, `maion-coder`, `mellum`, `metis16`, `minicpm`, `minicpm3`,
`mistral3`, `mpt`, `nanbeige`, `olmo`, `olmoe`, `openelm`, `orion`, `pangu-embedded`,
`phi2`, `phi3`, `phimoe`, `plamo`, `plamo2`, `plamo3`, `plm`, `qwen`, `qwen2moe`,
`qwen3moe`, `qwen4exp`, `refact`, `ricursive`, `rnd1`, `seed_oss`, `smallthinker`,
`smollm3`, `spark2_5`, `stablelm`, `starcoder`, `starcoder2`, `talkie`, `xing4_0`,
`xverse`, `zaya1`.

**MLA / attention compressée / sparse**
`deepseek2`, `deepseek32`, `deepseek4`, `deepseek41`, `glm4`, `glm4moe`, `glm-dsa`,
`glm5-next`, `kimi-linear`, `kimi-k3`, `kimi-k31`, `minimax-01`, `minimax-m2`,
`minimax-m3`, `mistral4`, `mistral-large4` (vérifier le type d'attention de ces deux
derniers au chargement).

**SSM / linéaire / hybride**
`qwen3next`, `mamba`, `mamba2`, `jamba`, `maple`, `nemotron`, `nemotron_h`,
`nemotron_h_moe`, `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7`.

**Llama 4 / GPT-OSS / Gemma 2-3n / T5 / BitNet**
`llama4`, `gpt-oss`, `gemma3n`, `t5`, `t5encoder`, `bitnet`, `chameleon`.

**Multimodal vision / OCR**
`qwen2vl`, `qwen3vl`, `qwen3vlmoe`, `hunyuan_vl`, `sensenova_u1`, `cogvlm`,
`deepseek2-ocr`, `paddleocr`, `dots3note`, `mimo2`, `step35`, `clip`.

**Audio / TTS**
`qwen3tts`, `pockettts`, `moshi`, `zonos3`, `kokoro`, `tiny-voca`,
`wavtokenizer-dec`, `muse-spark`, `muse-glimmer`.

**Diffusion**
`dream`, `llada`, `llada-moe`.

**Embeddings / encodeurs**
`modern-bert`, `nomic-bert-moe`, `neo-bert`, `jina-bert-v2`, `jina-bert-v3`,
`eurobert`, `llama-embed`, `gemma-embedding2`.

**Têtes de draft / divers**
`eagle3`, `dflash`, `vllm-decision2`.

### B. Routes llama-server et statut Kinetic

Voir la table 4.5. Les routes manquantes sont :

`POST /props`, `POST /slots/:id_slot`, `POST /infill`, `POST /apply-template`,
`POST /v1/chat/completions/input_tokens`, `POST /v1/responses`,
`POST /v1/messages`, `POST /v1/messages/count_tokens`, `POST /tts`,
`POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `WS /v1/realtime`,
`POST /v1/systemone`, `POST /v1/chat/completions/control`,
`GET/POST /lora-adapters`, `POST/DELETE /models`, `POST /models/load`,
`POST /models/unload`, `GET /models/sse`, `GET/POST/DELETE /v1/stream`,
`POST /v1/streams/lookup`, MCP/tools, GCP compat, CORS proxy.

### C. Paramètres de sampling

Présents : `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`,
`presence_penalty`, `frequency_penalty`, `repeat_last_n`, `seed`, `logit_bias`,
`typical_p`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `dry_multiplier`,
`dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`,
`mirostat`, `mirostat_tau`, `mirostat_eta`, `stop`.

Manquants : `dynatemp_range`, `dynatemp_exponent`, `min_keep`, `ignore_eos`,
`adaptive_target`, `adaptive_decay`, `reasoning_budget_tokens`,
`reasoning_budget_start`, `reasoning_budget_end`, `reasoning_budget_forced`,
`reasoning_budget_message`, `grammar_lazy`, `grammar_triggers`, `preserved_tokens`,
`samplers` (ordre), `backend_sampling`, `n`.

### D. Types GGUF

Types connus de `kinetic-quant` : F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL,
IQ3_S, IQ2_S, IQ4_XS, I8, I16, I32, I64, F64, IQ1_M, BF16, TQ1_0, TQ2_0, MXFP4,
Q1_0, Q2_0, PQ2_0, PTQ1_0.

Manquants : **NVFP4 (40)**, **F8_E4M3 (43)**.

### E. Backends

Présents : CPU x86 (AVX2/FMA/F16C), CPU ARM (NEON/SDOT), CUDA (6.0–12.0), Vulkan
(subgroup 32), Metal via Vulkan.

**Sur la machine de référence** : Vulkan (RADV gfx1201) et CPU x86 uniquement.
ROCm/HIP est installé (7.2) mais Kinetic n'a pas de backend HIP. CUDA est absent.

Manquants : HIP/ROCm, SYCL, CANN, MUSA, OpenCL, WebGPU, RPC, BLAS, Hexagon,
OpenVINO, zDNN, ZenDNN, VirtGPU, ET, Metal natif, CPU AVX512/AMX/VNNI, ARM i8mm/SVE.

### F. Carte des fichiers à toucher

| Sujet | Fichiers principaux |
|---|---|
| Registre architectures | `crates/kinetic-models/src/arch/mod.rs`, `arch/*.rs` |
| Blocs | `crates/kinetic-models/src/blocks/*.rs` |
| Tokenizers | `crates/kinetic-tokenizer/src/*.rs` |
| Quantization | `crates/kinetic-quant/src/*.rs`, `crates/kinetic-cli/src/quantize.rs` |
| GGUF | `crates/kinetic-gguf/src/*.rs` |
| HIP/ROCm | `crates/kinetic-hip/src/*.rs`, `hip/*.hip` (nouveau) |
| CUDA | `crates/kinetic-cuda/src/*.rs`, `cuda/*.cuh` |
| Vulkan | `crates/kinetic-vulkan/src/*.rs`, `shaders/*.wgsl` |
| CPU | `crates/kinetic-cpu/src/*.rs`, `kdot/*.rs` |
| Serveur | `crates/kinetic-server/src/server.rs`, `llama.rs`, `api.rs`, `*.rs` |
| CLI | `crates/kinetic-cli/src/main.rs`, `llama_flags.rs`, `*.rs` |
| C API | `crates/kinetic-capi/include/kinetic.h`, `src/*.rs` |
| Vision | `crates/kinetic-vision/src/lib.rs`, `*/*.rs` |
| Warm Start | `crates/kinetic-models/src/warm/*.rs` |
| Spéculatif | `crates/kinetic-models/src/speculative/*.rs` |
| LoRA | `crates/kinetic-models/src/lora.rs` |
| Mémoire | `crates/kinetic-models/src/memory.rs`, `per_layer.rs` |

### G. Commandes de mesure locales

```bash
# Inventaire
lspci -nn | grep -Ei 'vga|3d|display'
rocminfo | grep -E 'Name:|Marketing Name:|gfx|Compute Unit|Wavefront'
vulkaninfo --summary
hipcc --version

# Vérifier que la machine est idle avant un benchmark
rocm-smi --showuse --showpids

# llama.cpp, mêmes réglages, un backend à la fois
./build-vulkan/bin/llama-bench -m <modele.gguf> -p 512 -n 128 -r 5 -ngl 99 -dev Vulkan0
./build-hip/bin/llama-bench   -m <modele.gguf> -p 512 -n 128 -r 5 -ngl 99 -dev ROCm0

# Kinetic, backend Vulkan
./target/release/kinetic bench <modele.gguf> --device vulkan -r 5

# Forcer le GPU discret pour HIP
HIP_VISIBLE_DEVICES=0 ./build-hip/bin/llama-bench ...

# Profilage des noyaux HIP
rocprof --stats ./build-hip/bin/llama-bench ...
```

Ne pas lancer ces commandes pendant qu'un autre bench tourne (le `kinetic bench`
CPU visible dans `ps` suffit à fausser les mesures). Sur mémoire unifiée ou partagée,
mesurer la résidence driver et la mémoire du processus, pas seulement `MemAvailable`.

---

## 11. Conclusion

Kinetic a déjà les bons points d'extension : un registre de familles, des blocs
réutilisables, un trait de projecteur, des crates par backend, des modules de routes,
un traducteur de flags et une ABI C. La parité avec llama.cpp ne demande pas de
réécrire cette architecture ; elle demande de **remplir les registres** et d'**ajouter
des backends et des modalités** derrière les mêmes contrats.

Les trois priorités qui changent le plus la perception sont :

1. **Les architectures et les tokenizers** : sans eux, un utilisateur ne peut même pas
   charger son modèle.
2. **Le serveur et le sampling** : sans eux, un client existant casse ou perd des
   fonctionnalités.
3. **Les backends et la quantization** : sans eux, la promesse « aussi rapide que
   llama.cpp » ne tient pas sur AMD, Intel, Apple, ARM ou les GPU récents.

Sur la machine de référence — RX 9070 XT, ROCm 7.2, Vulkan 1.4 RADV, pas de CUDA —
la priorité est encore plus nette :

1. **`kinetic-hip`** : le backend qui exploite réellement les 64 CU et la bande
   passante de la carte, sans la couche RADV.
2. **Vulkan** : GEMM de prompt, coopmat, types IQ, KV quantifié, MoE, layer split.
3. **Le reste de la parité** : architectures, tokenizers, sampling, routes, outils.

CUDA reste dans le plan pour les utilisateurs NVIDIA, mais il ne doit pas être le
premier chantier d'une équipe qui n'a pas de GPU NVIDIA. Les différenciateurs de
Kinetic — RAD-1, Warm Start, Zero Wait, Shortlist, Flywheel, Convoy, l'exactitude et
les receipts — doivent rester le cœur du produit. La parité doit les rendre **plus
utiles**, pas les remplacer.
