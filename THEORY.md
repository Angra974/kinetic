# THEORY — plafonds de la RX 9070 XT (gfx1201) et lecture des mesures

> M01 / Agent 5, 22 novembre 2026. Chaque chiffre dit s'il vient d'une mesure
> (commande, commit et date dans `ledger.jsonl`) ou d'une fiche technique
> (marqué *nominal*, jamais décisif). Le plafond commande la lecture : une
> mesure à 90 % du plafond est un mur matériel ; à 40 %, le problème est dans
> les noyaux ou l'ordonnancement, pas dans le matériel.

## 1. Machine mesurée

| Poste | Valeur | Source |
|---|---|---|
| GPU | Radeon RX 9070 XT, `gfx1201`, 64 CU, 16 304 Mio de VRAM | `rocm-smi`, `PLAN.md` §3 |
| GTT | 15 597 Mio accessibles côté pilote | `rocm-smi --showmeminfo gtt` |
| Pilote | Vulkan 1.4.354 RADV / Mesa 26.1.6 ; ROCm 7.2.0 (HIP absent du binaire) | `PLAN.md` §3 |
| CPU | Ryzen 7 7800X3D, 8C/16T, AVX2 (AVX-512 non exploité) | `PLAN.md` §3 |
| Débordement | RADV sur-alloue son tas local (VRAM + GTT ≈ 31,2 Gio) ; `HOST_SPILL` ne se déclenche pas sur cette machine | `PLAN.md` §5, commit `8f64626` |

## 2. Formules

**Décodage = bande passante mémoire.** Octets lus par token émis :

```text
B_t = poids_denses + octets_experts × (top_k / n_experts)
      + octets_lm_head + octets_KV_par_token × contexte
plafond_c1 = bande_passante_mesurée / B_t
```

**Préfill = calcul.** `FLOP/token ≈ 2 × params_linéaires_actifs`, puis
`plafond_pp ≈ FLOPS_mesurés / FLOP_par_token` ; les GEMM MoE par expert
écrêtent l'efficacité et l'attention ajoute son propre terme.

**Speculation** (hors baseline, §7.10 de `llama-kinetic.md`) :
`B_verify+k×B_draft)/a` par token accepté — à mesurer avant d'annoncer.

## 3. Bande passante : mesure dérivée, borne étalon

Pas de sonde de copie device-to-device disponible dans la boîte à outils
actuelle (`rocm-bandwidth-test` absent de ROCm 7.2.0). La borne étalon vient
du décodage dense de llama.cpp, qui lit tout le fichier à chaque token et dont
les noyaux GEMV sont optimisés :

| Modèle (fichier, résident) | tg llama | octets/token | bande effectivement exploitée |
|---|---:|---:|---:|
| Qwen3.8-27B IQ3_XXS (12,32 Gio, 100 % VRAM) | 40,7 t/s | 12,32 Go | **501 Go/s** |
| Qwen3-Coder-30B-A3B UD-IQ3_XXS (12,85 Gio, ~top-k/n_experts lu) | 201,5 t/s | ~1,32 Go | 280 Go/s |
| Qwen3.8-35B-A3B Q4_K_M (21,7 Gio, -ngl 31, spill) | 51,6 t/s | ~1,92 Go lu | 104 Go/s (bus + CPU) |

La valeur étalon du décodage dense est donc **~500 Go/s exploités** contre les
640 Go/s *nominaux* de la fiche. Le 30B MoE ne lit que ses experts actifs par
token, d'où un chiffre moindre à expert constant (aucun des deux n'atteint le
mur). Pour le 35B, la lecture passe par le PCIe pour les experts déversés :
c'est le bus, pas la VRAM, qui décide (conforme à G7/G8).

## 4. Calcul : efficacité mesurée du préfill

FLOP/token = 2 × params actifs (attention non décomptée, borne basse) :

| Mesure | t/s | TFLOPS effectifs |
|---|---:|---:|
| 27B dense, llama pp512 | 1167,4 | **63,8** |
| 27B dense, kinetic pp512 | 263,7 | 14,4 |
| 30B MoE (3,3B actifs), llama pp512 | 3522,2 | 23,2 |
| 30B MoE (3,3B actifs), kinetic pp512 | 484,3 | 3,2 |

llama exploite l'équivalent de ~64 TFLOPS sur un dense en quantizé contre
~97 TFLOPS *nominaux* FP16 : c'est le meilleur repère de calcul disponible.
Kinetic reste 4 à 7× dessous en préfill — c'est le résidu P0.2 (noyaux
many-token, GEMM tuilé), pas un plafond matériel.

## 5. Plafonds par modèle pivot (à contexte court, f16 KV oisif)

Borne haute du décodage = bande étalon 500 Go/s / octets par token :

| Modèle | B_t (Go, est.) | plafond c1 | mesuré kinetic | % du plafond | lecture |
|---|---:|---:|---:|---:|---|
| Qwen3.8-27B IQ3_XXS | 12,3 | ~41 t/s | 38,7 | **94 %** | mur matériel atteint côté décodage dense ; reste le préfill (14,4 vs ~64 TFLOPS) |
| Qwen3-Coder-30B-A3B | ~1,3 (top-k 8/128) | ~380 t/s | 55,4 | 15 % | noyaux MoE/Vulkan, pas la bande |
| Qwen3.8-35B-A3B | ~1,9 + lecture PCIe | ~260 t/s (VRAM seule) ; ~100 t/s avec spill | 31,2 | ~30 % | spill + noyaux |
| Qwen3.5-4B Q4_K_M | 3,0 | ~165 t/s | 153,3 | **93 %** | décodage dense petit modèle : au mur |
| Qwen3.5-9B Q4_K_M | 5,68 | ~88 t/s | 97,9 | au-dessus (111 %) | le 9B dépasse l'étalon : l'étalon dense 27B n'est pas un mur universel (GEMV moins dominés par le déquant) |

Estimations de B_t à affiner par `kinetic info` par modèle (tensions par
famille, taille du lm_head) ; les pourcentages valent à ±10 %.

## 6. Ce que la baseline dit (22 nov. 2026, commit `f000853`)

- `edit_identical: true` sur les 8 modèles benchés sur Vulkan.
- Le décodage dense est déjà au mur (4B 93 %, 27B 94 % de l'étalon) ; la marge
  de Kinetic est dans **le préfill** (4–7× sous llama) et **le décodage MoE**
  (15–30 % de l'étalon). C'est exactement le résidu P0.2, à confirmer par les
  mesures M02/M03 depuis ce ledger.
- VRAM/GTT relevés par modèle dans `ledger.jsonl` (`peak_memory_bytes`) ;
  le 35B culmine à 16 187 Mio de VRAM + 6 073 Mio de GTT, conforme au
  comportement RADV documenté en §5 de `PLAN.md`.

## 7. Ce qu'il manque à cette théorie

- Un sonde de copie D2D et un GEMM synthétique 4096³/8192³ pour remplacer les
  étalons dérivés par des mesures directes (WP1 reste ouvert sur ce point).
- Un relevé par modèle de `top_k`/`n_experts` et de la taille du `lm_head`
  pour remplacer l'estimation `fichier × fraction_active`.
- Les mesures à contexte long (KV f16/q8) et multi-requêtes (1/4/16/64),
  prévues par le protocole de `llama-kinetic.md` §7.10.
