# Experts et missions Kinetic

Index des documents de travail pour améliorer l'architecture et la
performance de Kinetic.

## Ordre de lecture

1. [`../experts.md`](../experts.md) — analyse, diagnostic, architecture
   cible et profils.
2. [`fiches-de-poste.md`](fiches-de-poste.md) — qui recruter, avec
   compétences, périmètre, livrables et critères.
3. [`missions.md`](missions.md) — quoi faire, dans quel ordre, avec work
   packages, dépendances et gates.
4. [`prompts-agents.md`](prompts-agents.md) — comment lancer un agent
   spécialisé.

## Règle de lancement

- **M01 d'abord** : baseline, `THEORY.md`, ledger, CI GPU.
- **Une mission = un agent = un périmètre d'écriture.**
- **Un seul écrivain par fichier.** Si deux missions touchent le même
  fichier, séquencer ou utiliser une branche/worktree dédié.
- **M06 WP0 avant le déplacement des exécuteurs** : trancher la direction
  des dépendances et l'emplacement de la factory.
- **Ne pas déplacer les exécuteurs pendant M02/M03/M04.**
- **Le KV (M04) avant l'ordonnanceur (M05).**
- **La sécurité (M14) après l'API (M11) et le release (M13).**

## Fichiers de travail à créer par M01

- `THEORY.md`
- `ledger.jsonl`
- `configs/`
- `best-config.json`

## Ce qui n'est pas une mission

- Un benchmark isolé sans contrôle ni ledger.
- Une réécriture complète « pour faire propre ».
- Un portage CUDA-only sur une machine sans NVIDIA.
- Un backend HIP/Metal/SYCL complet sans machine ni CI pour le valider.
- Une optimisation qui change la sortie sans gate d'identité.
