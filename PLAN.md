# PLAN — journal de campagne de performance

> Compagnon de `llama-kinetic.md` (§7.10 protocole, §7.12 artefacts).
> Une ligne par essai : hypothèse, configuration, mesure, verdict, raison.
> Aucun chiffre sans commande, commit et date.

---

## 1. Machine et état

| Élément | Valeur (relevé du 22 nov. 2026) |
|---|---|
| CPU | AMD Ryzen 7 7800X3D, 8 cœurs / 16 threads, AVX2, FMA, F16C |
| GPU | AMD Radeon RX 9070 XT, `gfx1201` (RDNA 4), 64 CU, 16 Gio, wavefront 32 |
| iGPU | AMD Radeon 610M, `gfx1036`, 2 CU, mémoire partagée |
| Vulkan | 1.4.354, RADV / Mesa 26.1.6, sous-groupe par défaut 64, plage 32–64 |
| ROCm | 7.2.0, HIP 7.2, `hipcc` ciblant `gfx1201` |
| CUDA | absent (aucun GPU NVIDIA) |
| llama.cpp mesuré | `build-vulkan/bin/llama-bench` (Vulkan0) |

Dépôt de travail : le fork `Angra974/kinetic`, remote `fork`. `main` porte ce journal et
l'analyse `llama-kinetic.md` ; la branche `vulkan-work` porte les noyaux et les mesures
(commits `0127427` et `85d29aa`). `upstream` (`Veloresearch/kinetic`) n'est que l'amont :
on y lit les nouveautés, on n'y pousse pas. Les deux branches partent de `f3723b1`.

## 2. Protocole

- Un backend par comparaison : `kinetic` Vulkan contre `llama-bench` Vulkan0, jamais
  de mélange avec ROCm ou CUDA.
- Machine idle, même fichier GGUF, un seul changement par essai.
- Préfill : `python3 scripts/bench_protocol.py '<modèle>' --llama-bin <llama-bench>`
  (écrit un CSV horodaté avec le commit ; kinetic y enregistre pp512/pp2048, llama
  pp512/pp2048/tg128).
- Décodage : `./target/release/kinetic bench '<modèle>' -r 5` (le protocole
  n'enregistre pas tg côté kinetic).
- Exactitude : `cargo test --release -p kinetic-vulkan` — référence f64 à 1e-5
  (`check`), égalité de bits mono-token / passe prompt (`check_prompt_pass`), et
  garde-fou `every_gemv_type_survives_a_prompt_pass`.
- Verdict : le candidat est gardé s'il bat le contrôle sur la métrique cible, ne
  régresse rien de plus de ~3 %, et passe les tests d'exactitude.

## 3. Modèle de référence

`Qwen3.8-27B-IQ3_XXS` — `qwen35`, 27,32 B paramètres, 65 couches, 11,47 Gio,
255 tenseurs IQ3_XXS + 44 IQ4_XS + 43 Q5_K + 38 Q4_K + 22 Q6_K + 8 Q4_0 + 456 F32.

---

## 4. Essais

### E0 — Contrôle : CPU seulement (avant tout noyau IQ)

- **Commit** : `110e6f2`. Vulkan refuse le modèle (`Vulkan has no kernel for
  IQ3_XXS`), tout retombe sur le CPU.
- **Commande** : `./target/release/kinetic bench 'local/Qwen3.8-27B-IQ3_XXS'
  --device cpu -r 1` (best of 1 : ~20 min pour la passe complète).
- **Mesure** : pp512 3,77 t/s, tg128 0,64 t/s, edit256 1,74 t/s. Pic mémoire
  11,59 Gio.
- **Verdict** : contrôle de départ.

### E1 — Noyaux Vulkan IQ3_XXS, IQ4_XS, Q4_0

- **Hypothèse** : le modèle ne tourne pas sur GPU parce qu'il mêle trois types sans
  noyau ; les ajouter (layout planaire, un noyau mono-token par type) suffit à
  passer du CPU au GPU.
- **Commit** : `0127427`.
- **Mesure (protocole)** : pp512 154,8 t/s, pp2048 152,3 t/s.
  **`bench -r 5`** : pp512 153,5, tg128 38,08, edit256 63,68. Pic 11,88 Gio.
  **llama.cpp Vulkan0** : pp512 1159,5, pp2048 1153,5, tg128 40,27.
- **Exactitude** : 6 + 20 + 1 tests OK, dont les références f64 des trois types et
  `check_prompt_pass`.
- **Verdict** : **gardé**. Décodage à 95 % de llama.cpp, préfill ×41 par rapport au
  CPU mais 7,5× derrière llama.cpp.
- **Faits saillants** : trois bugs corrigés avant validation — double comptage de
  l'offset de plan `qs` (IQ3_XXS), bit de parité des signes (IQ3_XXS), extraction du
  scale 6 bits `scales_l`/`scales_h` (IQ4_XS). Sans la référence f64, les trois
  passaient inaperçus.

### E2 — Noyaux many-token pour les mêmes types

- **Hypothèse** : le préfill reste lent parce que le décode des poids est refait pour
  chaque token ; un noyau « many tokens » (32 tokens par passe, `PROMPT_BATCH` 128)
  doit s'en approcher des K-quants.
- **Commit** : `85d29aa`.
- **Mesure (protocole)** : pp512 261,9 t/s, pp2048 257,9 t/s.
  **`bench -r 3`** : pp512 263,5, tg128 38,36, edit256 67,46.
  **llama.cpp Vulkan0** : pp512 1154,2, pp2048 1149,0, tg128 40,07.
- **Exactitude** : mêmes 6 + 20 + 1 tests, dont la passe prompt bit-exacte pour les
  trois types (mêmes bits que le noyau mono-token, qui reste la référence).
- **Verdict** : **gardé**. Préfill ×1,69, décodage inchangé, aucune régression.
- **Incident** : un passage a échoué sur le F32 du GPU intégré (7800X3D / RAPHAEL),
  sans lien avec les noyaux ; relancé à l'identique, tout passe. C'est la contention
  connue entre les deux GPU quand un bench tourne en parallèle, à ne pas confondre
  avec une régression.

### Récapitulatif

| Étape | Commit | pp512 | pp2048 | tg128 | llama pp512 | llama tg128 |
|---|---|---:|---:|---:|---:|---:|
| E0 CPU | `110e6f2` | 3,8 | — | 0,6 | — | — |
| E1 noyaux IQ | `0127427` | 154,8 | 152,3 | 38,1 | 1159,5 | 40,3 |
| E2 many-token | `85d29aa` | 261,9 | 257,9 | 38,4 | 1154,2 | 40,1 |

Artefacts : `bench-results.csv` (E1) et `bench-many.csv` (E2) sont restés dans l'ancien
répertoire de travail `/home/mrnemesys/Devs/kinetic-work`, non suivis par git.

---

## 5. Écarts connus après E2

| Écart | Effet mesuré ou attendu |
|---|---|
| Pas de GEMM tuilé (GEMV par ligne, sommes par sous-groupe) | préfill 4,4× derrière llama.cpp |
| `Commands::submit` soumet puis attend (`commands.rs` 160–163) | aucun pas en vol ; 5–15 % attendus |
| Pas de coopmat (`VK_KHR_cooperative_matrix` disponible sur RADV gfx1201) | 2–5× préfill, après le GEMM |
| KV cache f16 seulement | mémoire de contexte, pas la vitesse |
| Pas de MoE, pas de layer split | `qwen3-coder-30b-a3b` reste en CPU |
| Types IQ4_NL, IQ3_S, Q5_0/Q5_1/Q4_1, Q2_K/Q3_K, MXFP4 absents | modèles concernés en CPU |
| Décodage à 95 % | rien à gagner avant le prochain levier mémoire |

## 6. Prochain essai

### E3 — GEMM de prompt tuilé pour les types quantifiés

- **Hypothèse** : le préfill est limité par la relecture des poids une fois par ligne
  et l'absence de tuile ; un GEMM tuilé (mémoire partagée, split-K, déquant en
  registres) doit approcher le préfill de llama.cpp sans toucher au décodage.
- **Cible** : pp512 ≥ 390 t/s (×1,5), soit ≤ 3× llama.cpp.
- **Exactitude requise** : référence f64 pour chaque type, même tolérance ; la passe
  prompt doit rester bit-exacte vis-à-vis du noyau mono-token, sinon le chemin many
  est retiré.
- **Non-régression** : tg128 ≥ 37,2 t/s (−3 %), pic mémoire ≤ 12,1 Gio.
- **Mesure** : protocole identique à E1/E2, un fichier, Vulkan0, avant/après.

### E4 — Soumission asynchrone (après E3)

- **Hypothèse** : plusieurs frames en vol, timeline semaphores ; le graphe enregistré
  existe déjà.
- **Cible** : 5–15 % sur tg128 et pp512, sans changement de résultats.

## 7. Règles de tenue

- Tout essai ajoute une section datée ici avant d'être annoncé comme un gain.
- Un chiffre sans commande, commit et date n'entre pas dans ce journal.
- Un candidat qui échoue à l'exactitude est retiré du code, pas documenté comme une
  option.
