# M01 — Analyse de contexte (Agent 5, Performance/Qualification)

> Créé le 22 novembre 2026, avant toute mesure et avant toute modification.
> Objet : état des lieux du dépôt, des artefacts et du harnais, par rapport aux
> livrables de M01. Aucun chiffre ici n'est une mesure : ce sont des constats de
> lecture (fichiers, chemins, drapeaux).

## 1. Sources lues

- `docs/experts/missions.md` — M01 (objectif, périmètre, 7 WP, livrables,
  critères d'acceptation, protocole, risques, DoD).
- `docs/experts/prompts-agents.md` — contrat commun + Agent 5 (périmètre
  d'écriture, interdits, boucle de travail, commandes, critères de succès).
- `PLAN.md` — §1 (preuve exigée), §2 (règles de mesure), §3 (machine), §5
  (lacunes closes G1–G8 et mesures du 22 nov.).
- `llama-kinetic.md` §7.10 (plafonds, bruit, portes de stabilité), §7.11 (ordre
  des leviers), §7.12 (artefacts attendus).
- `docs/src/benchmarks.md`, `scripts/bench_protocol.py`, `scripts/conformance.py`,
  `.github/workflows/ci.yml`, `kinetic bench --help`.

## 2. État des livrables M01

| Livrable | État | Détail |
|---|---|---|
| `THEORY.md` | **absent** | Plafonds seulement esquissés dans `llama-kinetic.md` §7.10 (exemple arithmétique « RX 9070 XT », explicitement « pas une mesure »). |
| `ledger.jsonl` | **absent** | `llama-kinetic.md` §7.12 le prévoit ; `PLAN.md` en tient lieu provisoirement. |
| `configs/` | **absent** | — |
| `best-config.json` | **absent** | Le rôle est tenu de fait par `/home/mrnemesys/Devs/models/presets.ini` (hors dépôt). |
| `scripts/ab_protocol.py` | **absent** | `bench_protocol.py` compare, mais ne fait pas contrôle → candidat → contrôle. |
| `conformance.py --device vulkan` | **non** | `choices=["cpu", "cuda"]` (ligne 150). |
| needle 50/100 %, soak, watchdog | **absents** | `server_abuse.py` existe (charge/abus), pas de needle ni de watchdog mémoire. |
| CI GPU | **absente** | `ci.yml` : fmt + clippy/tests sur runners GitHub hébergés (ubuntu/windows/macos) + bindings Python. Aucun runner self-hosted Vulkan/ROCm, aucun seuil de régression. |
| Rapport de baseline dense 27B / MoE 30B A3B | **partiel** | Chiffres du 22 nov. dans `PLAN.md` §5, mais `-r 1`/`-r 2`, sans contrôles avant/après, donc non qualifiables au sens §2. |

Conclusion : M01 est **au statut « à lancer »**, conforme à l'index des
missions. Le squelette ledger/configs/THEORY n'existe pas encore.

## 3. Machine et fichiers de modèle présents

- CPU Ryzen 7 7800X3D, GPU RX 9070 XT `gfx1201` (16 Gio), iGPU gfx1036,
  Vulkan 1.4.354 RADV/Mesa 26.1.6, ROCm 7.2.0, pas de CUDA — comme `PLAN.md` §3.
- `llama-bench` de référence : `/home/mrnemesys/Devs/llama.cpp/build-vulkan/bin/llama-bench`
  (`llama-cli` à côté). **Pas** dans le dépôt kinetic : `build-vulkan/` n'y existe pas.
- Modèles sous `~/.local/share/kinetic/models/local/` :
  - `qwen3.8-35b-a3b/Qwen3.8-35B-A3B-Q4_K_M.gguf` — 21 Gio (le fichier des mesures
    G6/G7/G8) ;
  - `qwen3-coder-30b-a3b-instruct/…-UD-IQ3_XXS.gguf` — 12 Gio (mesures G5) ;
  - `Qwen3.8-27B-IQ3_XXS/…` — 13 Gio (+ LoRA F16) ;
  - plus petits : gemma-4-12b Q5_K_M, helcyon-gpt-4o Q5_K_M, L3-8B IQ4_XS,
    qwen3.5-9b/4b, qwen3-vl-4b, paddleocr-vl, Offrail-27B.
- `target/release/kinetic` existe (version 0.1). `rocm-smi` en `/opt/rocm-7.2.0/bin`.

## 4. État du dépôt (impact sur la reproductibilité)

- Branche `main`, HEAD `8f64626` (« Record where the host spill really goes on
  Vulkan »), puis `6b6e4a0` (« Profile a CPU layer of the 35B and place its cost »).
- ~~Arbre de travail sale~~ **résolu** : l'instrumentation `KINETIC_PROFILE` a été
  retirée ; ses conclusions sont consignées dans `PLAN.md` par `6b6e4a0`. L'arbre
  est propre et le binaire `target/release/kinetic` correspond à HEAD.
- ~~Non suivis~~ **résolu** : tout le travail courant est committé (423372a) et
  poussé sur `fork/main`.
- ~~`PLAN.md` entrée périmée~~ **résolu** : la section « Mesures pp512 / tg128 du
  35B » a été remise au passé et renvoie à la mesure de découpe de G8 (ab0fe06).
- **Base vérifiée (f000853)** : `cargo fmt` appliqué partout, trois avertissements
  clippy corrigés (`div_ceil`, `too_many_arguments` sur `moe_combine` et le bench
  CLI, test MoE nettoyé) ; `cargo clippy --release -D warnings` vert,
  `cargo test --release --workspace` 65/65 binaires verts, scripts Python
  compilables. La baseline part donc d'un HEAD committé, poussé, et d'un binaire
  release reconstruit correspondant exactement à ce HEAD.

## 5. Audit du harnais existant

`scripts/bench_protocol.py`
- Appelle `kinetic bench --json --prompt P -r N` et `llama-bench -m … -p … -n 128 -r N`.
- **Ne passe pas** `-dev Vulkan0` ni `-ngl` à `llama-bench` : comparaison non
  appariée sur le backend, contrairement à la règle « même backend » de `PLAN.md` §1.
- `-r 5` par défaut (< 6), une seule passe : pas de contrôle avant/après, pas de
  médiane, pas d'écart-type → ne satisfait pas §2.
- N'écrit pas dans `ledger.jsonl` (CSV seulement, sans état machine ni pic VRAM).

`scripts/conformance.py`
- `--device` limité à `cpu|cuda` ; envoie `--gpu-layers 99` pour non-cpu (ligne 96),
  convention à réconcilier avec `--gpu-layers N` = N dernières couches de kinetic.

`kinetic bench`
- `--device {auto,cpu,cuda,vulkan}`, `--gpu-layers`, `-r` (défaut 3, « best is
  reported » → **max**, pas médiane).

`docs/src/benchmarks.md`
- Documente la méthode côté CPU (`bench_vs_llamacpp.py`), pas de section GPU
  chiffrée ni de protocole de régression.

## 6. Écarts à combler, dans l'ordre des WP de M01

1. **WP1 `THEORY.md`** : mesurer bande passante device-to-device, FLOPS GEMM
   4096³/8192³ sur gfx1201, puis plafond par modèle (35B A3B, 30B A3B, 27B).
2. **WP2 ledger + configs** : schéma JSONL (commande, commit, date, état machine,
   pic VRAM, médiane, écart-type, verdict), `configs/` dérivé de `presets.ini`.
3. **WP3 A/B** : `scripts/ab_protocol.py` contrôle/candidat/contrôle, `-r ≥ 6`,
   médiane + IC, garde de non-régression 3 %.
4. **WP4** `conformance.py --device vulkan` (corpus de contrôle, comparaison au CPU).
5. **WP5** needle 50 %/100 %, soak concurrent, watchdog mémoire.
6. **WP6** CI GPU self-hosted (`runs-on` local), seuil 5 %, artefacts JSON.
7. **WP7** dashboard/rapport type.

## 7. Propositions de baseline (à confirmer avant exécution)

- Périmètre de modèles : **tous** les modèles de
  `~/.local/share/kinetic/models/local/`, sans distinction (13 familles, de
  qwen3-vl-4b à qwen3.8-35b-a3b), chacun mesuré sur son backend de service
  (Vulkan quand il tient, découpe ou CPU sinon) ; dense 27B et MoE 30B A3B
  restent les pivots des rapports M01.
- Pour chacun : `pp512`, `pp2048`, `tg128` (médiane de 6, `-r 6`), pic VRAM
  (`rocm-smi`), TTFT et temps de chargement, contre `llama-bench -dev Vulkan0`
  avec le même nombre de couches sur le périphérique.
- Gates : `edit_identical: true` (test d'édition de `kinetic bench`), sortie
  gloutonne non dégénérée, un seul job GPU à la fois, machine idle.
- Prérequis levés : arbre propre committé et poussé (`f000853`), CI gates
  (fmt/clippy/tests) vertes, binaire release synchronisé avec HEAD.

## 8. Risques / points d'arrêt

- Arbre sale + instrumentation temporaire → biais et non-reproductibilité.
- `llama-bench` sans `-dev`/`-ngl` → comparaison non appariée (faux écarts).
- Pic VRAM vs débordement GTT : `PLAN.md` §5 montre que le pilote RADV sur-alloue
  en local (GTT), donc « pic VRAM » seul est trompeur ; mesurer aussi `mem_info_gtt_used`.
- `-r 6` sur le 35B : ~40 s par passe de bench, plus chargement → prévoir la durée ;
  ne pas paralléliser (règle « un seul job GPU »).
