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

## 3. Bande passante : mesurée directement (sonde `kinetic-vulkan --example bandwidth`)

La sonde de lecture de l'exemple `bandwidth` (512 Mio hors de tout cache,
3 essais, le meilleur compte) donne sur la RX 9070 XT, le 22 novembre 2026 :

| Mesure | Valeur |
|---|---:|
| **lecture atteignable (device-local, gros bloc)** | **638,8 Go/s** — 99,8 % des 640 Go/s nominaux |
| gros tensors hors cache (94 Mio F32 ; tête 521 Mio Q6_K) | 627,5–630,7 Go/s (98–99 % de la lecture atteignable) |
| petits tensors (13–47 Mio) | 817–1675 Go/s apparents — boostés par le cache, non représentatifs du débit DRAM |

L'étalon de décodage est donc **~630 Go/s soutenus** sur un fichier résident,
contre les 640 Go/s nominaux : la bande est saine, le pilote ne la bride pas.
L'étalon dérivé du décodage llama (~500 Go/s sur le 27B dense) vaut 79 % de
cette mesure : llama lui-même laisse ~20 % de la bande, le GEMV déquantisé
n'étant pas purement borné par la lecture.

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

Borne haute du décodage = bande étalon **630 Go/s soutenus** / octets par token
; efficacité préfill = 63,8 TFLOPS effectifs mesurés sur llama (§4) :

| Modèle | B_t (Go, est.) | plafond c1 | mesuré kinetic | % du plafond | lecture |
|---|---:|---:|---:|---:|---|
| Qwen3.8-27B IQ3_XXS | 13,2 | ~48 t/s | 38,7 | **80 %** | bande presque saturée ; marge ~20 % côté décodage, 4× côté préfill |
| Qwen3.5-4B Q4_K_M | 3,26 | ~193 t/s | 153,3 | **79 %** | idem, décodage dense petit modèle |
| Qwen3-Coder-30B-A3B | ~1,39 (top-k 8/128) | ~453 t/s | 55,4 | 12 % | noyaux MoE/Vulkan, pas la bande |
| Qwen3.8-35B-A3B | ~2,0 + lecture PCIe | ~315 t/s (VRAM seule) ; ~78 t/s avec spill | 31,2 | ~40 % | spill + noyaux |
| Qwen3.5-9B Q4_K_M | 6,10 | ~103 t/s | 97,9 | 95 % | au mur, à la précision de B_t (±10 %) |

Estimations de B_t à affiner par `kinetic info` par modèle (tensions par
famille, taille du lm_head) ; les pourcentages valent à ±10 %.

## 6. Ce que la baseline dit (22 nov. 2026, commit `f000853`)

- `edit_identical: true` sur les 8 modèles benchés sur Vulkan.
- Le décodage dense exploite 75–80 % de la bande mesurée (sonde directe
  638,8 Go/s) ; il reste ~20 % de marge, et le décodage MoE est à 12–40 % du
  plafond. L'écart massif est dans **le préfill** (4–7× sous llama, 3–14 TFLOPS
  effectifs contre 64) et **le décodage MoE** — le résidu P0.2, à confirmer par
  les mesures M02/M03 depuis ce ledger.
- VRAM/GTT relevés par modèle dans `ledger.jsonl` (`peak_memory_bytes`) ;
  le 35B culmine à 16 187 Mio de VRAM + 6 073 Mio de GTT, conforme au
  comportement RADV documenté en §5 de `PLAN.md`.
- Stabilité : needle PASS à 50 % et ~98 % du contexte à 8k et 32k (4B) ; soak
  4 et 16 slots × 512 tokens à 180 et 282 t/s agrégés, watchdog sans
  déclenchement ; conformance Vulkan 256/256 tokens contre llama.
- Essai A/B de bout en bout (35B, candidat `--gpu-layers 31`) : contrôles
  stables à 30,7 / 31,6 t/s (écart 2,9 %), candidat rejeté (18,2 t/s, −31 % de
  préfill) — conforme à G8, protocole et verdict au ledger. Première passe
  rendue inconclusive par un serveur kinetic leaké (dérive 10,7 %) : fuite
  corrigée dans `needle_test.py`, la porte de bruit a bien mordu.

## 7. Ce qu'il manque à cette théorie

- Un GEMM synthétique 4096³/8192³ pour un plafond de calcul direct : la sonde
  `bandwidth` mesure la lecture et les GEMV ; l'efficacité préfill reste
  dérivée de llama (63,8 TFLOPS effectifs).
- Un relevé par modèle de `top_k`/`n_experts` et de la taille du `lm_head`
  pour remplacer l'estimation `fichier × fraction_active`.
- Le runner auto-hébergé de la CI GPU est à brancher côté dépôt GitHub
  (Settings → Runners) ; le workflow et la gate sont prêts.
