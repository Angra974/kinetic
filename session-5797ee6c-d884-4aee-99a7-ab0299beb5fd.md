# Prompts experts et missions

**Session ID:** ses_dfdd49814ffePmhNa4i5zo9UvP
**Created:** 11/22/2026, 11:32:08 AM
**Updated:** 11/22/2026, 11:49:45 AM

---

## User

Anlayse ./docs/experts/ et ./session-ad321db8-5ba0-40bf-b50d-4be596739cef.md pour voir si tout les prompts pour les experts et missions ont bien été crée. 

---

## Assistant

I'll analyze the docs/experts directory and the session file to check whether all expert prompts and missions were created.



**Tool: shell**

**Input:**
```json
{
  "command": "ls -la ./docs/experts/ 2>/dev/null; echo \"---\"; ls -la ./session-ad321db8-5ba0-40bf-b50d-4be596739cef.md 2>/dev/null; echo \"---\"; wc -l ./session-ad321db8-5ba0-40bf-b50d-4be596739cef.md 2>/dev/null"
}
```

total 68
drwxrwxr-x 2 mrnemesys mrnemesys  4096 nov.  22 11:15 .
drwxrwxr-x 5 mrnemesys mrnemesys  4096 nov.  22 11:14 ..
-rw-rw-r-- 1 mrnemesys mrnemesys 33840 nov.  22 11:22 fiches-de-poste.md
-rw-rw-r-- 1 mrnemesys mrnemesys 22454 nov.  22 11:22 prompts-agents.md
---
-rw-rw-r-- 1 mrnemesys mrnemesys 1142529 nov.  22 11:30 ./session-ad321db8-5ba0-40bf-b50d-4be596739cef.md
---
21479 ./session-ad321db8-5ba0-40bf-b50d-4be596739cef.md


**Tool: glob**

**Input:**
```json
{
  "pattern": "docs/experts/**/*"
}
```

/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md
/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md

---

## Assistant

_Thinking:_

Let me read the docs/experts files first.

**Tool: read**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md, lines 1-1048
1: # Fiches de poste — experts Kinetic
2: 
3: > Documents opérationnels pour recruter, cadrer ou instancier les profils
4: > décrits dans [`docs/experts.md`](../experts.md).
5: >
6: > Chaque fiche donne : la mission, le contexte, les compétences, le périmètre
7: > d'écriture, les fichiers de contexte, les commandes de mesure, les livrables
8: > 30/60/90 et les critères de succès.
9: >
10: > **Règle commune** : un seul changement par essai, une configuration par
11: > essai, contrôles avant et après, médiane de ≥ 6 échantillons, aucune
12: > régression > 3 %, et les gates d'exactitude avant toute publication.
13: 
14: ---
15: 
16: ## Fiche 1 — Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)
17: 
18: **Priorité** : P0 — chemin critique.
19: **Rattachement** : équipe backend GPU.
20: **Mission** : combler l'écart de préfill Vulkan, débloquer le MoE, poser les
21: bases d'un backend HIP complet.
22: 
23: ### Contexte
24: 
25: - Le décodage dense Vulkan est à 95 % de llama.cpp (38,4 contre 40,1 t/s sur
26:   un dense 27B IQ3_XXS), mais le préfill est à 23 % (261,9 contre 1154,2 t/s).
27: - `gemv_many.wgsl` garde 32 tokens en vol ; ce n'est pas un GEMM tuilé.
28: - `Commands::submit` soumet puis attend `vkWaitForFences` ; pas de timeline
29:   semaphores, pas de `VkPipelineCache` persisté.
30: - `KvCache` est en f16, non quantifié.
31: - `moe_gemv.wgsl` lit les experts un slot à la fois ; pas de GEMM groupé.
32: - Le MoE 30B A3B est à 55,7 t/s en décodage contre 181,9 pour llama.cpp, et
33:   489 contre 2637 t/s en préfill.
34: - La cible architecturale est **un seul backend complet par API** : tout
35:   Vulkan dans `kinetic-vulkan`, tout HIP dans `kinetic-hip` (à créer).
36: 
37: ### Responsabilités
38: 
39: - Profiler les noyaux Vulkan et identifier memory-bound vs compute-bound.
40: - Concevoir et implémenter un GEMM de préfill tuilé (Q4_K, Q5_K, Q6_K, Q8_0,
41:   IQ3_XXS, IQ4_XS), bit-exact avec la passe mono-token.
42: - Ajouter coopmat (`VK_KHR_cooperative_matrix`), split-K et tuilage LDS.
43: - Remplacer la soumission synchrone par une soumission asynchrone avec
44:   timeline semaphores, sans casser l'ordre des dépendances.
45: - Persister le cache de pipelines et réduire le temps de démarrage.
46: - Implémenter un GEMM/GEMV groupé pour le MoE.
47: - Ajouter le KV q8_0 (puis q4_0) avec la même exactitude.
48: - Aligner Vulkan sur les classes de fonctionnalités du backend CUDA (grands
49:   lots de préfill, flash attention, MoE groupé, split multi-GPU) lorsque
50:   l'API le permet, sans copier le code.
51: - Porter les noyaux sur HIP/ROCm (`gfx1201`) dans `kinetic-hip`.
52: - Consigner chaque essai dans `PLAN.md` et `ledger.jsonl`.
53: 
54: ### Compétences requises
55: 
56: - Vulkan compute 1.3 : descriptors, push constants, specialization constants,
57:   subgroups 32/64, barriers, command buffers, timeline semaphores,
58:   `VkPipelineCache`, cooperative matrix.
59: - WGSL, SPIR-V, naga ; RGP ; ISA AMDGPU ; RDNA 4 (wave32, LDS 64 Kio,
60:   VGPR/SGPR, occupancy, Infinity Cache).
61: - GEMM : tuilage, register blocking, split-K, swizzling, bank conflicts,
62:   épilogue fusionné, FP16/BF16/INT8 dot.
63: - Quantification : formats de blocs, dequant-in-registers, dot int8 packé
64:   (`v_dot4_i32_i8`, `sdot4`).
65: - Roofline et compteurs GPU.
66: - Exactitude : accumulation f32, batch-invariance, égalité passe prompt /
67:   passe mono-token.
68: - Rust, `ash`, `unsafe` aux frontières.
69: 
70: ### Compétences appréciées
71: 
72: - HIP/ROCm, `hipcc`, rocWMMA, rocBLAS/hipBLASLt, hipGraph, rocprof.
73: - CUDA (pour partager les sources de noyaux, pas pour dupliquer).
74: - Metal/SPIR-V croisé.
75: - Expérience de llama.cpp, ggml, vLLM ou TensorRT-LLM.
76: 
77: ### Périmètre d'écriture
78: 
79: - `crates/kinetic-vulkan/**`
80: - `crates/kinetic-hip/**` (à créer)
81: - `crates/kinetic-runtime/src/{device,capabilities,factory,memory}.rs`
82:   (propositions d'API uniquement)
83: - `crates/kinetic-models/src/runner/**` (à créer : trait `ModelRunner` et
84:   runner CPU)
85: - `PLAN.md` (entrées d'expériences), `ledger.jsonl`, `configs/`
86: 
87: ### Fichiers de contexte
88: 
89: - `PLAN.md` §4 P0.2 et P0.3.
90: - `llama-kinetic.md` §7.2, §7.3, §7.10, §7.11.
91: - `crates/kinetic-vulkan/src/{commands,pipeline,gemv,attention,moe}.rs`.
92: - `crates/kinetic-vulkan/shaders/*.wgsl`.
93: - `crates/kinetic-models/src/vulkan/*` (à déplacer vers `kinetic-vulkan`).
94: - `docs/src/devices.md`, `docs/src/contributing.md`.
95: - `crates/kinetic-quant/src/{types,dequant}.rs` pour les formats.
96: 
97: ### Commandes de mesure
98: 
99: ```sh
100: # Build
101: cargo build --release --locked -p kinetic-cli --features vulkan
102: 
103: # Tests d'exactitude
104: cargo test --release --locked -p kinetic-vulkan
105: cargo test --release --locked -p kinetic-models
106: 
107: # Bench
108: target/release/kinetic bench MODEL.gguf --device vulkan --json -r 6
109: llama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6
110: 
111: # Profil
112: RGP capture sur pp512 (Radeon GPU Profiler)
113: rocm-smi --showuse --showtemp --showmemuse
114: ```
115: 
116: ### Livrables 30/60/90
117: 
118: - **30 jours** : baseline, roofline, profil RGP, prototype de GEMM tuilé
119:   Q4_K bit-exact, ≥ 1,5× sur pp512.
120: - **60 jours** : coopmat + split-K, ≥ 2× ; soumission asynchrone ; cache de
121:   pipelines ; pas de régression tg128.
122: - **90 jours** : GEMM MoE groupé ; KV q8_0 ; squelette HIP avec GEMV
123:   Q4_K/Q5_K/Q6_K/Q8_0 et attention FP16.
124: 
125: ### Critères de succès
126: 
127: - pp512/pp2048 ≥ 2× le baseline Kinetic, sans régression > 3 % sur tg128.
128: - Tous les tests d'identité verts.
129: - `kinetic bench` dit `edit_identical: true`.
130: - `PLAN.md` contient chaque essai avec commande, commit, date et verdict.
131: - Aucun `unsafe` sans commentaire `SAFETY`.
132: 
133: ### Interactions
134: 
135: - **Runtime** : interfaces de graphe, soumission asynchrone, KV.
136: - **Numérique** : formats, tolérances, gates.
137: - **QA** : protocole A/B, CI GPU, ledger.
138: - **Rust backend** : migration des exécuteurs vers `kinetic-vulkan`.
139: 
140: ### Profil / sourcing
141: 
142: HPC, ROCm, moteurs de jeu, compute Vulkan, équipes de noyaux ML. Un profil
143: CUDA-only n'est pas suffisant : il doit savoir lire du WGSL/SPIR-V et
144: raisonner RDNA.
145: 
146: ### Signaux d'alerte
147: 
148: - Ne sait pas expliquer pourquoi un noyau est memory-bound.
149: - Ne sait pas lire un compteur RGP.
150: - Propose de « tout réécrire en CUDA ».
151: - N'a jamais écrit de GEMM tuilé.
152: - Ignore les contraintes d'exactitude.
153: 
154: ---
155: 
156: ## Fiche 2 — Architecte runtime d'inférence
157: 
158: **Priorité** : P0 — chemin critique.
159: **Rattachement** : équipe runtime/serveur.
160: **Mission** : faire passer le serveur d'un modèle « N backends indépendants »
161: à un ordonnanceur token-level qui tient la concurrence, le TTFT et le p99.
162: 
163: ### Contexte
164: 
165: - `Engine.free` est un `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>` : une
166:   requête par backend.
167: - `PREFILL_CHUNK = 256` existe, mais le chunking est par requête.
168: - `Convoy` mutualise des passes, mais ne planifie pas les requêtes.
169: - Le KV est contigu par séquence, non paginé.
170: - `MAX_CONNECTIONS = 64`, `Connection: close`, pas de keep-alive.
171: - Les technologies RAD-1, Warm Start, Zero Wait, Shortlist et Convoy doivent
172:   rester exactes.
173: 
174: ### Responsabilités
175: 
176: - Concevoir un ordonnanceur token-level : admission, priorité, fairness,
177:   annulation, backpressure.
178: - Introduire un pool KV paginé, dimensionné pour N × contexte, avec
179:   défragmentation et éviction.
180: - Ajouter le chunked prefill global et le recouvrement préfill/décodage.
181: - Gérer le rollback KV du spéculatif et la réutilisation de préfixe.
182: - Intégrer le placement mémoire (layers, experts, tenseurs) et le multi-GPU.
183: - Ajouter les métriques par token : TTFT, tok/s, p50/p99, cache hit, queue
184:   depth.
185: - Préserver les invariants d'exactitude et les tests existants.
186: 
187: ### Compétences requises
188: 
189: - Continuous batching, paged KV, prefix caching, chunked prefill, admission
190:   control, fairness, priorité, annulation, backpressure.
191: - Décodage spéculatif : draft/verify, arbre, rollback, réutilisation de KV.
192: - Rust concurrence : threads/async, `Mutex`/`Condvar`, channels, atomics,
193:   `Send`/`Sync`, scoped threads, panic safety.
194: - Mémoire : pool KV, fragmentation, mémoire unifiée, pinned host, paging des
195:   experts, watchdog OOM.
196: - Graph capture/replay, multi-stream.
197: - Observabilité et mesure de latence.
198: 
199: ### Compétences appréciées
200: 
201: - Expérience vLLM, SGLang, TensorRT-LLM, llama.cpp server, ONNX Runtime.
202: - Connaissance des modèles hybrides SSM/GDN et des slots d'état.
203: - Expérience des systèmes locaux-first (pas cloud).
204: 
205: ### Périmètre d'écriture
206: 
207: - `crates/kinetic-runtime/**` (à créer : ordonnanceur, pool KV, métriques)
208: - `crates/kinetic-server/src/{engine,sequence,backend,server}.rs`
209: - `crates/kinetic-models/src/{convoy,memory,per_layer}.rs`
210: - `crates/kinetic-models/src/warm/**`
211: - `PLAN.md`, `ledger.jsonl`, `configs/`
212: 
213: ### Fichiers de contexte
214: 
215: - `PLAN.md` §4 P1.5 et P1.6.
216: - `llama-kinetic.md` §7.6, §7.9, §7.10.
217: - `crates/kinetic-server/src/engine.rs`, `sequence.rs`, `backend.rs`.
218: - `crates/kinetic-models/src/convoy.rs`, `memory.rs`, `per_layer.rs`.
219: - `crates/kinetic-models/src/warm/**`.
220: - `docs/src/how-it-is-fast.md` (RAD-1, Warm Start, Zero Wait, Convoy).
221: - `docs/src/devices.md` (layer split, MoE offload).
222: 
223: ### Commandes de mesure
224: 
225: ```sh
226: # Serveur
227: target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
228: 
229: # Charge concurrente (exemple avec oha, ou un script maison)
230: oha -n 64 -c 4 --json http://127.0.0.1:8080/v1/chat/completions ...
231: 
232: # Exactitude sous charge
233: python scripts/same_answer_demo.py --model MODEL.gguf \
234:   --kinetic target/release/kinetic --kinetic-args "--device vulkan"
235: 
236: # Robustesse
237: python scripts/server_abuse.py --port 8091
238: ```
239: 
240: ### Livrables 30/60/90
241: 
242: - **30 jours** : cartographie du runtime actuel, design doc, métriques
243:   manquantes, prototype d'ordonnanceur.
244: - **60 jours** : pool KV N × contexte, chunked prefill global, annulation,
245:   métriques par token.
246: - **90 jours** : benchmarks c1/c4/c16/c64, p99 et TTFT documentés, aucune
247:   régression d'exactitude.
248: 
249: ### Critères de succès
250: 
251: - Débit agrégé en hausse à c4/c16/c64.
252: - p99 et TTFT mesurés et reproductibles.
253: - `same_answer_demo.py` identique sous charge.
254: - Aucun test d'identité cassé.
255: - Aucune fuite mémoire ni cache non borné.
256: 
257: ### Interactions
258: 
259: - **GPU AMD** : interfaces de graphe, KV, soumission.
260: - **Rust backend** : `kinetic-runtime`, découpage des responsabilités.
261: - **QA** : protocole de charge, p99, soak.
262: - **Serveur/API** : métriques, slots, keep-alive.
263: 
264: ### Profil / sourcing
265: 
266: Équipes serving/LLM runtime, vLLM, SGLang, TensorRT-LLM, llama.cpp. Rare :
267: il faut du systèmes distribué local, pas du cloud.
268: 
269: ### Signaux d'alerte
270: 
271: - Ne connaît que le batching statique.
272: - Ne sait pas expliquer le rollback KV.
273: - Propose un « rewrite » complet sans migration.
274: - Ignore la fairness et l'annulation.
275: - Ne mesure que le débit moyen, jamais le p99.
276: 
277: ---
278: 
279: ## Fiche 3 — Architecte Rust systèmes/backend
280: 
281: **Priorité** : P0/P1 — fondation.
282: **Rattachement** : équipe cœur.
283: **Mission** : réduire la duplication CUDA/Vulkan, faire de chaque API un
284: backend complet et préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.
285: 
286: ### Contexte
287: 
288: - `kinetic-models/src/cuda` ≈ 4,5 k LOC et `kinetic-cuda` ≈ 8,5 k LOC.
289: - `kinetic-models/src/vulkan` ≈ 2,2 k LOC et `kinetic-vulkan` ≈ 5,9 k LOC.
290: - `kinetic-models` a des features `cuda`/`vulkan` et des modules backend.
291: - Ajouter HIP créerait `kinetic-hip` + `kinetic-models/src/hip`.
292: - Les seams `Family`, `Target`, `Backend`, `Projector` sont bons et doivent
293:   être préservés.
294: - La surface `unsafe` est concentrée aux frontières SIMD/GPU/FFI.
295: 
296: ### Responsabilités
297: 
298: - Créer `kinetic-runtime` : devices, capacités, budget mémoire, factory,
299:   ordonnanceur, pool KV, métriques. Pas de types de modèle.
300: - Extraire `ModelRunner` dans `kinetic-models` : charger, préfill, décoder,
301:   état de séquence, capacités. Le runner CPU devient la référence.
302: - Déplacer les exécuteurs CUDA/Vulkan dans leurs crates backend respectifs,
303:   via `ModelRunner` ; ce n'est pas un simple `git mv`.
304: - Retirer les features backend de `kinetic-models`.
305: - Ajouter une factory de backend dans `kinetic-server` / `kinetic-cli`.
306: - Ajouter un backend de test pour `kinetic-models`.
307: - Auditer les `unsafe` et les invariants `Send`/`Sync`.
308: - Améliorer le démarrage, le temps de build et la taille binaire.
309: - Documenter la règle « un backend complet par API » et son contrat de
310:   compatibilité.
311: 
312: ### Compétences requises
313: 
314: - Rust avancé : lifetimes, traits, génériques vs `dyn`, zéro-copie,
315:   soundness `unsafe`, `Send`/`Sync`, `Arc`/`Rc`, mmap, allocators, layouts
316:   cache-friendly, monomorphisation.
317: - Architecture : ports/adapters, abstraction backend, graphe/dispatch,
318:   feature flags, compilation conditionnelle, FFI.
319: - Performance hôte : `perf`, flamegraph, heaptrack, criterion, LLVM remarks,
320:   LTO/PGO/BOLT.
321: - Build : workspaces Cargo, cross-compilation, toolchains CUDA/HIP, CI.
322: - Tests : unitaires, intégration, property-based, fuzz.
323: 
324: ### Compétences appréciées
325: 
326: - Connaissance de CUDA/Vulkan/Metal au niveau API, pas nécessairement noyau.
327: - Expérience de refactor de moteurs de rendu ou de bases de données.
328: - Expérience PyO3/C ABI.
329: 
330: ### Périmètre d'écriture
331: 
332: - `Cargo.toml`, `crates/*/Cargo.toml`
333: - `crates/kinetic-runtime/**` (à créer)
334: - `crates/kinetic-models/src/{lib,model}.rs` (retrait des modules backend)
335: - `crates/kinetic-cuda/**`, `crates/kinetic-vulkan/**` (accueil des
336:   exécuteurs)
337: - `crates/kinetic-server/src/{backend,lib}.rs`
338: - `docs/src/contributing.md`, `docs/src/building.md`
339: 
340: ### Fichiers de contexte
341: 
342: - `docs/src/contributing.md` (layout, règles).
343: - `docs/src/building.md` (build, features, tests).
344: - `crates/kinetic-models/src/lib.rs`, `model.rs`.
345: - `crates/kinetic-models/src/{cuda,vulkan}/**`.
346: - `crates/kinetic-cuda/src/**`, `crates/kinetic-vulkan/src/**`.
347: - `crates/kinetic-server/src/{backend,engine}.rs`.
348: - `PLAN.md` §3 et §4.
349: 
350: ### Commandes de mesure
351: 
352: ```sh
353: cargo build --release --locked --workspace --all-features
354: cargo test --release --locked --workspace
355: cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
356: cargo tree -p kinetic-models
357: cargo bloat --release -p kinetic-cli   # taille binaire
358: perf record -g target/release/kinetic ...  # overhead hôte
359: ```
360: 
361: ### Livrables 30/60/90
362: 
363: - **30 jours** : cartographie de la duplication, design de `kinetic-runtime`,
364:   liste des `unsafe` à auditer.
365: - **60 jours** : déplacement mécanique d'un exécuteur (CUDA ou Vulkan) dans
366:   son crate, tests verts, pas de changement de comportement.
367: - **90 jours** : les deux exécuteurs déplacés, `kinetic-models` sans feature
368:   backend, factory de backend, backend de test.
369: 
370: ### Critères de succès
371: 
372: - `kinetic-models` ne dépend plus de CUDA/Vulkan.
373: - Un seul crate par API contient tout le code de cette API.
374: - Les tests d'identité restent verts à chaque étape.
375: - Le temps de build et le démarrage ne régressent pas.
376: - Chaque `unsafe` a un commentaire `SAFETY` et un test associé.
377: 
378: ### Interactions
379: 
380: - **GPU AMD** : accueil des exécuteurs, interfaces de graphe.
381: - **Runtime** : `kinetic-runtime`, scheduler.
382: - **QA** : tests de non-régression, build CI.
383: - **Build/release** : features, cross-compilation.
384: 
385: ### Profil / sourcing
386: 
387: Systèmes, bases de données, moteurs de jeu, embedded. Rare : il faut du Rust
388: de production, pas du Rust d'exercice.
389: 
390: ### Signaux d'alerte
391: 
392: - Veut introduire MLIR ou un compilateur de graphes.
393: - Ne connaît pas `unsafe`/`Send`/`Sync` en profondeur.
394: - Propose un rewrite complet.
395: - Ne sait pas mesurer le temps de build ou l'overhead hôte.
396: - Casse les seams existants.
397: 
398: ---
399: 
400: ## Fiche 4 — Ingénieur quantification et exactitude numérique
401: 
402: **Priorité** : P0/P1 — garde-fou.
403: **Rattachement** : équipe numérique.
404: **Mission** : étendre la quantification et garantir que chaque gain de
405: vitesse reste exact.
406: 
407: ### Contexte
408: 
409: - 5 cibles de quantification contre ~30 dans llama.cpp.
410: - Pas d'imatrix, pas d'IQ encode.
411: - NVFP4 et F8_E4M3 absents du runtime.
412: - KV f16 seulement.
413: - `kinetic-quant` est la définition de référence ; `kdot/tests.rs` compare
414:   chaque noyau.
415: - L'exactitude est le différenciateur produit.
416: 
417: ### Responsabilités
418: 
419: - Implémenter NVFP4 et F8_E4M3 au runtime.
420: - Implémenter l'encodeur IQ et l'imatrix.
421: - Ajouter le KV q8_0/q4_0 avec tests d'exactitude.
422: - Définir les tolérances et les gates par format.
423: - Construire un corpus de conformance et des tests property/fuzz.
424: - Mesurer la qualité (KL/perplexité) et la vitesse par format.
425: - Refuser tout noyau qui ne passe pas la référence f64.
426: 
427: ### Compétences requises
428: 
429: - GGUF, quantization par blocs, K-quants, grilles IQ, MXFP4, NVFP4,
430:   FP8 E4M3, imatrix.
431: - Analyse numérique : f64, propagation d'erreur, tolérances, ordre
432:   d'accumulation, fp16/bf16/fp8, arrondi stochastique.
433: - Encodeur `quantize`, politique par tenseur, requantification, split/merge.
434: - Tests différentiels, property-based, fuzzing, conformance.
435: - Sensibilité performance : dequant-in-registers, vectorisation.
436: 
437: ### Compétences appréciées
438: 
439: - Expérience ggml/llama.cpp, GPTQ/AWQ, bitsandbytes.
440: - Connaissance des kernels CUDA/Vulkan.
441: - Expérience calibration et datasets.
442: 
443: ### Périmètre d'écriture
444: 
445: - `crates/kinetic-quant/**`
446: - `crates/kinetic-gguf/**`
447: - `crates/kinetic-cpu/src/kdot/**`
448: - Tests de noyaux dans `crates/kinetic-cuda/tests/**` et
449:   `crates/kinetic-vulkan/tests/**`
450: - `docs/src/models.md`
451: - `PLAN.md`, `ledger.jsonl`, `configs/`
452: 
453: ### Fichiers de contexte
454: 
455: - `docs/src/contributing.md` section « Adding a weight format or a kernel ».
456: - `crates/kinetic-quant/src/{types,dequant,encode}/**`.
457: - `crates/kinetic-cpu/src/kdot/tests.rs`.
458: - `llama-kinetic.md` §7.7.
459: - `PLAN.md` P0.4.
460: - `crates/kinetic-gguf/src/**`.
461: 
462: ### Commandes de mesure
463: 
464: ```sh
465: cargo test --release --locked -p kinetic-quant
466: cargo test --release --locked -p kinetic-cpu
467: cargo test --release --locked -p kinetic-vulkan
468: target/release/kinetic quantize MODEL.gguf --help
469: target/release/kinetic perplexity MODEL.gguf
470: python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
471:   --llama /path/to/llama.cpp/bin --device cpu
472: ```
473: 
474: ### Livrables 30/60/90
475: 
476: - **30 jours** : audit des gates, corpus de conformance, liste des formats
477:   manquants.
478: - **60 jours** : NVFP4/FP8 runtime, KV q8_0, tests d'identité.
479: - **90 jours** : IQ encode + imatrix, politique par tenseur, métriques
480:   qualité.
481: 
482: ### Critères de succès
483: 
484: - Chaque nouveau format a une référence f64 et un noyau testé.
485: - Aucune régression d'identité.
486: - La qualité est mesurée, pas supposée.
487: - Les chiffres sont reproductibles et consignés.
488: 
489: ### Interactions
490: 
491: - **GPU AMD** : noyaux de quantification, layouts.
492: - **Runtime** : KV, mémoire.
493: - **QA** : gates, corpus.
494: - **Modèles** : formats des fichiers.
495: 
496: ### Profil / sourcing
497: 
498: Compression ML, co-design matériel/logiciel, calcul numérique, ggml. Rare.
499: 
500: ### Signaux d'alerte
501: 
502: - « La quantization change forcément la sortie, ce n'est pas grave. »
503: - Ne sait pas expliquer l'ordre d'accumulation.
504: - Ne teste que la vitesse, jamais la qualité.
505: - Ne connaît pas les formats de blocs.
506: 
507: ---
508: 
509: ## Fiche 5 — Ingénieur performance / qualification
510: 
511: **Priorité** : P0 — crédibilité.
512: **Rattachement** : équipe qualité.
513: **Mission** : rendre la mesure automatisée, reproductible et infalsifiable.
514: 
515: ### Contexte
516: 
517: - Le protocole existe dans `PLAN.md` et `llama-kinetic.md` §7.10, mais il est
518:   manuel.
519: - Pas de CI GPU ; pas de `THEORY.md` ; pas de `ledger.jsonl`.
520: - Les scripts existent : `bench_protocol.py`, `bench_vs_llamacpp.py`,
521:   `conformance.py`, `server_abuse.py`, `same_answer_demo.py`.
522: - Les chiffres sont la crédibilité du projet.
523: 
524: ### Responsabilités
525: 
526: - Écrire `THEORY.md` : plafonds de bande passante, FLOPS, KV, TTFT.
527: - Créer `ledger.jsonl`, `configs/`, `best-config.json`.
528: - Automatiser le protocole A/B avec contrôles avant/après.
529: - Ajouter une CI GPU (Vulkan/ROCm) avec seuil de régression.
530: - Étendre `conformance.py` à Vulkan.
531: - Ajouter un test needle 50 %/100 % de contexte.
532: - Ajouter un soak concurrent et un watchdog mémoire.
533: - Qualifier chaque release : build propre, poids épinglés, gates.
534: 
535: ### Compétences requises
536: 
537: - Méthodologie : médianes, intervalles de confiance, bruit, A/B, état
538:   thermique, isolation.
539: - Harness : Python/Rust, JSON, ledger, régression, CI GPU, artefacts pinnés.
540: - Profilage : RGP, rocprof, Nsight, `perf`, watchdogs.
541: - Tests : fuzzing, soak, needle, conformance, chaos.
542: - Reproductibilité : commits, révisions de poids, drivers, conteneurs.
543: 
544: ### Compétences appréciées
545: 
546: - Expérience performance engineering/SRE.
547: - Connaissance de llama.cpp et de ses benchmarks.
548: - Expérience GitHub Actions self-hosted GPU.
549: 
550: ### Périmètre d'écriture
551: 
552: - `scripts/**`
553: - `.github/workflows/**`
554: - `PLAN.md`, `THEORY.md`, `ledger.jsonl`, `configs/`, `best-config.json`
555: - `docs/src/benchmarks.md`
556: 
557: ### Fichiers de contexte
558: 
559: - `PLAN.md` §2 et §7.
560: - `llama-kinetic.md` §7.10–7.12.
561: - `docs/src/benchmarks.md`.
562: - `scripts/bench_protocol.py`, `bench_vs_llamacpp.py`, `conformance.py`,
563:   `server_abuse.py`, `same_answer_demo.py`.
564: - `.github/workflows/ci.yml`.
565: 
566: ### Commandes de mesure
567: 
568: ```sh
569: python scripts/bench_protocol.py MODEL.gguf --llama-bin /path/to/llama-bench
570: python scripts/bench_vs_llamacpp.py MODEL.gguf --llama-dir /path/to/llama.cpp/bin
571: python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
572:   --llama /path/to/llama.cpp/bin --device cpu
573: python scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091
574: python scripts/same_answer_demo.py --model MODEL.gguf --kinetic target/release/kinetic
575: rocm-smi --showuse --showtemp --showmemuse
576: ```
577: 
578: ### Livrables 30/60/90
579: 
580: - **30 jours** : `THEORY.md`, ledger, script A/B, inventaire des gates.
581: - **60 jours** : CI GPU, extension Vulkan de `conformance.py`, needle test.
582: - **90 jours** : dashboard de régression, qualification de release, rapport
583:   type.
584: 
585: ### Critères de succès
586: 
587: - Un candidat à +5 % est distingué du bruit avec un protocole reproductible.
588: - Une régression est détectée automatiquement.
589: - Chaque chiffre publié a une commande, un commit, une date et un état
590:   machine.
591: - Les résultats négatifs sont conservés.
592: 
593: ### Interactions
594: 
595: - **Tous les experts** : protocole, gates, publication.
596: - **Build/release** : CI, artefacts.
597: - **GPU AMD** : profils, compteurs.
598: 
599: ### Profil / sourcing
600: 
601: Performance engineering, SRE, équipes benchmark. Rare.
602: 
603: ### Signaux d'alerte
604: 
605: - « Best of one run ».
606: - Ne contrôle pas l'état machine.
607: - Ne connaît pas les médianes/IC.
608: - Confond benchmark et démonstration.
609: - Ne consigne pas les échecs.
610: 
611: ---
612: 
613: ## Fiche 6 — Ingénieur intégration des architectures
614: 
615: **Priorité** : P1 — couverture.
616: **Rattachement** : équipe modèles.
617: **Mission** : couvrir les architectures manquantes par briques réutilisables.
618: 
619: ### Contexte
620: 
621: - 21/180 architectures.
622: - Les briques rentables : MLA/DSA, hybrides SSM, T5, Gemma 2, GPT-OSS.
623: - Les modèles locaux bloqués : `qwen3vl`, `deepseek`, `llama4`, `gemma2`,
624:   `phi3`, `command-r`, `dbrx`.
625: - Le registre `Family` et les blocs sont les bons points d'extension.
626: 
627: ### Responsabilités
628: 
629: - Ajouter des familles dans `arch/<nom>.rs` + une ligne dans `FAMILIES`.
630: - Réutiliser `transformer.rs`, `blocks/moe.rs`, `blocks/attention.rs`.
631: - Ajouter les blocs manquants : MLA, DSA, Mamba 1, RWKV, T5, iRoPE, sinks.
632: - Mapper les tenseurs GGUF vers les blocs.
633: - Tester la conformance vs llama.cpp.
634: - Documenter `docs/src/models.md`.
635: 
636: ### Compétences requises
637: 
638: - Zoo Transformer : MHA/GQA/MQA/MLA/DSA/sinks, RoPE/iRoPE/partial/scaling,
639:   MoE top-k/shared/sigmoid, SSM/hybride, norm/activation, T5.
640: - Lecture GGUF, mapping de tenseurs.
641: - Conformance vs llama.cpp.
642: - Discipline : un fichier par famille, pas de `match` central.
643: 
644: ### Compétences appréciées
645: 
646: - Expérience llama.cpp/ggml.
647: - Connaissance des modèles récents (DeepSeek, Qwen, Llama 4, Gemma).
648: - Notions de quantization.
649: 
650: ### Périmètre d'écriture
651: 
652: - `crates/kinetic-models/src/arch/**`
653: - `crates/kinetic-models/src/blocks/**`
654: - `docs/src/models.md`
655: - `scripts/new_family.py` (si nécessaire)
656: 
657: ### Fichiers de contexte
658: 
659: - `docs/src/adding-a-family.md`.
660: - `llama-kinetic.md` §4.1, §5.1, §6.
661: - `crates/kinetic-models/src/arch/mod.rs` (`FAMILIES`).
662: - `crates/kinetic-models/src/arch/transformer.rs`.
663: - `crates/kinetic-models/src/blocks/**`.
664: - `scripts/new_family.py`.
665: 
666: ### Commandes de mesure
667: 
668: ```sh
669: target/release/kinetic info MODEL.gguf
670: target/release/kinetic run MODEL.gguf --device cpu "Hello"
671: python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
672:   --llama /path/to/llama.cpp/bin --device cpu
673: cargo test --release --locked -p kinetic-models
674: ```
675: 
676: ### Livrables 30/60/90
677: 
678: - **30 jours** : une famille proche (phi3, gemma2 ou command-r) chargée et
679:   conforme.
680: - **60 jours** : qwen3vl + qwen2moe/phimoe, tests de conformance.
681: - **90 jours** : MLA/DSA ou hybride SSM, documentation.
682: 
683: ### Critères de succès
684: 
685: - Le modèle se charge et génère.
686: - Les tokens sont identiques à llama.cpp sur le corpus de contrôle.
687: - Aucun `match` central ajouté.
688: - La documentation est à jour.
689: 
690: ### Interactions
691: 
692: - **Numérique** : formats, qualité.
693: - **QA** : conformance.
694: - **Serveur** : templates/tool calls si nécessaire.
695: 
696: ### Profil / sourcing
697: 
698: Communauté llama.cpp/transformers, ingénieurs modèles. Moyenne rareté.
699: 
700: ### Signaux d'alerte
701: 
702: - Copie-colle un `match` central.
703: - Ne teste pas contre llama.cpp.
704: - Ignore les variantes RoPE/norm.
705: - Ne documente pas.
706: 
707: ---
708: 
709: ## Fiche 7 — Ingénieur tokenizers / templates / tool calls
710: 
711: **Priorité** : P1 — compatibilité.
712: **Rattachement** : équipe compatibilité.
713: **Mission** : atteindre la compatibilité des tokenizers et des templates.
714: 
715: ### Contexte
716: 
717: - 13/68 prétokenizers, 55 manquants.
718: - UGM, RWKV, PLaMo, Tekken, Jais2, etc. manquants.
719: - Tokens EOT/EOM/FIM et types spéciaux incomplets.
720: - Templates intégrés et formats tool-call incomplets.
721: - L'issue #5 (tool calls dans le raisonnement) est typique.
722: 
723: ### Responsabilités
724: 
725: - Ajouter les prétokenizers prioritaires.
726: - Implémenter UGM et les variantes BPE.
727: - Ajouter EOT/EOM/FIM et les tokens de contrôle.
728: - Ajouter les templates intégrés par nom.
729: - Ajouter les formats tool-call manquants.
730: - Tester contre `llama-tokenize` et `llama-server`.
731: 
732: ### Compétences requises
733: 
734: - BPE, SentencePiece, WordPiece, UGM, Tekken.
735: - Prétokenizers regex, byte-level, tokens spéciaux.
736: - Jinja, tool-call formats, JSON Schema/GBNF.
737: - Tests différentiels.
738: 
739: ### Compétences appréciées
740: 
741: - Expérience NLP infrastructure.
742: - Connaissance des templates Qwen/Llama/Mistral/DeepSeek/GLM/Kimi.
743: 
744: ### Périmètre d'écriture
745: 
746: - `crates/kinetic-tokenizer/**`
747: - `crates/kinetic-server/src/{template,tool_calls,tool_grammar}.rs`
748: - `docs/src/models.md`
749: 
750: ### Fichiers de contexte
751: 
752: - `llama-kinetic.md` §4.2, §5.3.
753: - `crates/kinetic-tokenizer/src/**`.
754: - `crates/kinetic-server/src/{template,tool_calls}.rs`.
755: - `scripts/conformance.py`.
756: - `docs/src/models.md`.
757: 
758: ### Commandes de mesure
759: 
760: ```sh
761: llama-tokenize -m MODEL.gguf -f corpus.txt
762: cargo test --release --locked -p kinetic-tokenizer
763: python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
764:   --llama /path/to/llama.cpp/bin --device cpu
765: ```
766: 
767: ### Livrables 30/60/90
768: 
769: - **30 jours** : 5 prétokenizers prioritaires, corpus de contrôle.
770: - **60 jours** : 15 prétokenizers, UGM, tokens spéciaux.
771: - **90 jours** : templates intégrés, formats tool-call, split GGUF.
772: 
773: ### Critères de succès
774: 
775: - Ids identiques sur le corpus.
776: - Tokens spéciaux corrects.
777: - Tool calls valides.
778: - Aucune régression de template.
779: 
780: ### Interactions
781: 
782: - **Modèles** : familles et tokens.
783: - **Serveur** : templates, tool calls.
784: - **QA** : corpus.
785: 
786: ### Profil / sourcing
787: 
788: Infrastructure NLP. Moyenne rareté.
789: 
790: ### Signaux d'alerte
791: 
792: - Ne connaît que tiktoken.
793: - Ignore les tokens spéciaux.
794: - Ne teste pas contre llama.cpp.
795: 
796: ---
797: 
798: ## Fiche 8 — Ingénieur serveur / API / observabilité
799: 
800: **Priorité** : P1 — produit.
801: **Rattachement** : équipe serveur.
802: **Mission** : parité API, robustesse et observabilité.
803: 
804: ### Contexte
805: 
806: - ~15/45 routes.
807: - Pas de keep-alive/TLS.
808: - 64 connexions thread-per-connection.
809: - `docker stop` à améliorer.
810: - Compatibilité clients perfectible.
811: 
812: ### Responsabilités
813: 
814: - Ajouter les routes manquantes : `/props`, `/slots/:id_slot`, `/infill`,
815:   `/apply-template`, `/lora-adapters`, `n > 1`, `timings_per_token`.
816: - Ajouter keep-alive, TLS/API key, CORS, arrêt propre.
817: - Ajouter un router multi-modèle.
818: - Améliorer les métriques Prometheus.
819: - Tester la compatibilité llama-server/OpenAI/Anthropic/Ollama.
820: - Sécuriser les entrées.
821: 
822: ### Compétences requises
823: 
824: - HTTP/1.1, SSE, keep-alive, chunked, timeouts, backpressure, TLS.
825: - APIs OpenAI/llama-server/Anthropic/Ollama.
826: - Prometheus, logs, tracing.
827: - Sécurité applicative.
828: 
829: ### Compétences appréciées
830: 
831: - Rust async/threads.
832: - Expérience serveurs LLM.
833: - Connaissance llama.cpp server.
834: 
835: ### Périmètre d'écriture
836: 
837: - `crates/kinetic-server/src/{server,http,api,llama,metrics}.rs`,
838:   `crates/kinetic-server/src/ollama/`
839: - `docs/src/server.md`, `docs/src/clients.md`
840: 
841: ### Fichiers de contexte
842: 
843: - `llama-kinetic.md` §4.5, §5.9.
844: - `PLAN.md` P0.1.
845: - `scripts/llama_server_compat.py`, `server_abuse.py`.
846: - `docs/src/server.md`, `docs/src/clients.md`.
847: 
848: ### Commandes de mesure
849: 
850: ```sh
851: target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
852: python scripts/llama_server_compat.py --model MODEL.gguf \
853:   --llama-server /path/to/llama-server --kinetic target/release/kinetic
854: python scripts/server_abuse.py --port 8091
855: ```
856: 
857: ### Livrables 30/60/90
858: 
859: - **30 jours** : routes manquantes, keep-alive, tests de compatibilité.
860: - **60 jours** : `n > 1`, TLS/API key, router, métriques.
861: - **90 jours** : arrêt propre, CORS, documentation, tests de charge.
862: 
863: ### Critères de succès
864: 
865: - Les clients existants fonctionnent sans modification.
866: - Les champs inconnus sont ignorés ou refusés clairement.
867: - Pas de crash sous `server_abuse.py`.
868: - Métriques exploitables.
869: 
870: ### Interactions
871: 
872: - **Runtime** : slots, métriques, annulation.
873: - **Tokenizers** : templates, tool calls.
874: - **QA** : compatibilité, charge.
875: 
876: ### Profil / sourcing
877: 
878: Backend/API engineering. Courante.
879: 
880: ### Signaux d'alerte
881: 
882: - Ne connaît que REST basique.
883: - Ignore le streaming/SSE.
884: - Ne teste pas les clients réels.
885: 
886: ---
887: 
888: ## Fiche 9 — Ingénieur multimodal
889: 
890: **Priorité** : P2 — expansion.
891: **Rattachement** : équipe modèles.
892: **Mission** : vision, audio, OCR.
893: 
894: ### Contexte
895: 
896: - 2/73 projecteurs.
897: - Pas d'audio/vidéo/OCR.
898: - `kinetic-vision` a un contrat `Projector` propre.
899: - Le multimodal est XL et hors chemin critique de performance.
900: 
901: ### Responsabilités
902: 
903: - Ajouter des projecteurs vision.
904: - Ajouter le crate `kinetic-audio`.
905: - Ajouter le pipeline OCR.
906: - Tester contre `llama-mtmd-debug`.
907: - Documenter.
908: 
909: ### Compétences requises
910: 
911: - ViT/SigLIP/CLIP, projecteurs, preprocessing.
912: - Vidéo, audio (mel, ASR/TTS), OCR.
913: - Noyaux GPU vision.
914: 
915: ### Périmètre d'écriture
916: 
917: - `crates/kinetic-vision/**`
918: - `crates/kinetic-audio/**` (à créer)
919: - `crates/kinetic-models/src/input.rs`
920: - `docs/src/models.md`
921: 
922: ### Fichiers de contexte
923: 
924: - `llama-kinetic.md` §4.10, §5.2.
925: - `crates/kinetic-vision/src/{lib,ops,preprocess}.rs`.
926: - `crates/kinetic-vision/examples/encode.rs`.
927: - `docs/src/models.md`.
928: 
929: ### Commandes de mesure
930: 
931: ```sh
932: cargo run --release -p kinetic-vision --example encode -- mmproj.gguf rainbow:512
933: llama-mtmd-debug --image rainbow -n 512 -m mmproj.gguf
934: target/release/kinetic serve MODEL.gguf --mmproj mmproj.gguf
935: ```
936: 
937: ### Livrables 30/60/90
938: 
939: - **30 jours** : 3 projecteurs prioritaires.
940: - **60 jours** : 10 projecteurs, tests de parité.
941: - **90 jours** : audio ou OCR, documentation.
942: 
943: ### Critères de succès
944: 
945: - Embeddings proches de llama.cpp.
946: - Réponses cohérentes.
947: - Pas de régression texte.
948: 
949: ### Interactions
950: 
951: - **Modèles** : architecture.
952: - **GPU AMD** : noyaux vision.
953: - **QA** : parité.
954: 
955: ### Profil / sourcing
956: 
957: Vision/audio appliquée. Rare.
958: 
959: ### Signaux d'alerte
960: 
961: - Ne connaît que des pipelines Python.
962: - Ignore le preprocessing.
963: - Ne compare pas à llama.cpp.
964: 
965: ---
966: 
967: ## Fiche 10 — Ingénieur build/release/bindings
968: 
969: **Priorité** : P2 — distribution.
970: **Rattachement** : équipe plateforme.
971: **Mission** : rendre le produit installable et intégrable.
972: 
973: ### Contexte
974: 
975: - 10/200 fonctions C.
976: - Bindings Python partiels.
977: - Images Docker et packaging à finaliser.
978: - CI multi-OS existante.
979: 
980: ### Responsabilités
981: 
982: - Compléter l'ABI C.
983: - Générer les bindings Python/Node/Go.
984: - Finaliser Docker, winget/scoop, archives.
985: - Améliorer la CI release.
986: - Réduire la taille et le temps de build.
987: - Maintenir la documentation de build.
988: 
989: ### Compétences requises
990: 
991: - Cargo, cross-compilation, toolchains CUDA/HIP.
992: - Docker, GitHub Actions, signature de code.
993: - maturin/PyO3, ABI C, semver.
994: - mdBook.
995: 
996: ### Périmètre d'écriture
997: 
998: - `.github/workflows/**`
999: - `docker/**`, `packaging/**`
1000: - `bindings/**`
1001: - `crates/kinetic-capi/**`
1002: - `docs/src/{building,bindings}.md`
1003: 
1004: ### Fichiers de contexte
1005: 
1006: - `docs/src/building.md`, `docs/src/bindings.md`.
1007: - `.github/workflows/*.yml`.
1008: - `crates/kinetic-capi/**`.
1009: - `docker/Dockerfile*`.
1010: - `packaging/**`.
1011: 
1012: ### Commandes de mesure
1013: 
1014: ```sh
1015: cargo build --release --locked --workspace --all-features
1016: cargo test --release --locked --workspace
1017: cd bindings/python && maturin build --release --locked
1018: docker build -f docker/Dockerfile -t kinetic .
1019: ```
1020: 
1021: ### Livrables 30/60/90
1022: 
1023: - **30 jours** : audit ABI, build reproductible, taille binaire.
1024: - **60 jours** : bindings générés, Docker, installateurs.
1025: - **90 jours** : CI release, signature, documentation.
1026: 
1027: ### Critères de succès
1028: 
1029: - Installation réussie sur Windows/Linux/macOS.
1030: - ABI testée.
1031: - Build reproductible.
1032: - Documentation à jour.
1033: 
1034: ### Interactions
1035: 
1036: - **Serveur/API** : ABI.
1037: - **QA** : CI.
1038: - **Rust backend** : features, build.
1039: 
1040: ### Profil / sourcing
1041: 
1042: Release engineering, packaging. Moyenne rareté.
1043: 
1044: ### Signaux d'alerte
1045: 
1046: - Ne connaît pas les builds cross-platform.
1047: - Ignore les toolchains CUDA/HIP.
1048: - Ne teste pas les artefacts.

**Tool: read**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md, lines 1-657
1: # Prompts d'agents spécialisés — Kinetic
2: 
3: > Prompts prêts à l'emploi pour des agents (OpenCode, subagents ou équipes
4: > humaines) chargés d'améliorer l'architecture et la performance de Kinetic.
5: >
6: > Chaque prompt définit : le rôle, la mission, les fichiers à lire, le
7: > périmètre d'écriture, les garde-fous, la boucle de travail, les commandes de
8: > mesure, le format de sortie et les conditions d'arrêt.
9: >
10: > **À lire avant d'utiliser un agent** : [`docs/experts.md`](../experts.md) et
11: > [`docs/experts/fiches-de-poste.md`](fiches-de-poste.md).
12: 
13: ---
14: 
15: ## Contrat commun à tous les agents
16: 
17: Tout agent Kinetic doit respecter les règles suivantes, même si son prompt
18: spécifique ne les répète pas.
19: 
20: ### Vérité et mesure
21: 
22: - Ne jamais annoncer un gain sans commande, commit, date et état machine.
23: - Mesurer la baseline avant de modifier quoi que ce soit.
24: - Un seul changement par essai ; une configuration par essai.
25: - Médiane de ≥ 6 échantillons pour le décodage ; contrôles avant et après.
26: - Accepter seulement si le candidat bat les deux contrôles sur sa métrique et
27:   ne régresse rien de plus de ~3 %.
28: - Consigner les échecs et les résultats négatifs dans `PLAN.md`.
29: - Si une mesure est suspecte (trop rapide, sortie dégénérée, GPU idle),
30:   arrêter et vérifier avant de conclure.
31: 
32: ### Exactitude
33: 
34: - Toute optimisation doit préserver l'égalité batch / mono-token et les gates
35:   d'identité.
36: - Un changement qui change la sortie n'est pas mergeable, sauf s'il est
37:   explicitement opt-in et documenté.
38: - Lancer les tests d'exactitude du backend touché avant de publier.
39: 
40: ### Périmètre
41: 
42: - Ne modifier que les fichiers listés dans le périmètre d'écriture.
43: - Ne pas toucher aux autres backends, aux autres crates ou aux fichiers de
44:   build sans demande explicite.
45: - Ne pas copier de code depuis llama.cpp ou un autre moteur ; lire pour
46:   comprendre est autorisé, copier non.
47: - Aucune connexion réseau, sauf `kinetic pull` demandé explicitement.
48: 
49: ### Processus
50: 
51: - Lire les fichiers de contexte avant d'écrire.
52: - Écrire un plan court avant une modification non triviale.
53: - Laisser le dépôt dans un état compilable et testé.
54: - Mettre à jour `PLAN.md` avec : objectif, commande, résultat, verdict,
55:   raison.
56: - Produire un rapport final avec : métrique, baseline, maintenant, plafond,
57:   méthode de mesure, limites.
58: 
59: ### Arrêt
60: 
61: - S'arrêter et demander si : une modification architecturale est nécessaire,
62:   un test d'exactitude échoue sans explication, une mesure contredit le
63:   modèle, ou le périmètre d'écriture doit être élargi.
64: 
65: ---
66: 
67: ## Agent 1 — GPU AMD (Vulkan/RDNA 4, puis HIP)
68: 
69: ```text
70: Tu es ingénieur noyaux GPU AMD pour Kinetic. Ta mission est de combler
71: l'écart de préfill Vulkan, de débloquer le MoE, d'aligner Vulkan sur les
72: classes de fonctionnalités du backend CUDA lorsque l'API le permet, puis de
73: porter les noyaux sur HIP/ROCm.
74: 
75: LECTURE OBLIGATOIRE
76: - PLAN.md, sections 2, 3 et 4 (P0.2, P0.3).
77: - llama-kinetic.md, sections 7.2, 7.3, 7.10, 7.11.
78: - docs/experts.md, section 0 (un seul backend complet par API).
79: - crates/kinetic-vulkan/src/{commands,pipeline,gemv,attention,moe}.rs.
80: - crates/kinetic-vulkan/shaders/*.wgsl.
81: - crates/kinetic-models/src/vulkan/* (à terme dans kinetic-vulkan).
82: - crates/kinetic-quant/src/{types,dequant}.rs.
83: 
84: PÉRIMÈTRE D'ÉCRITURE
85: - crates/kinetic-vulkan/**
86: - crates/kinetic-hip/** (à créer)
87: - PLAN.md, ledger.jsonl, configs/ (entrées d'expériences)
88: - Propositions d'API dans crates/kinetic-runtime/src/{device,capabilities,factory}.rs
89:   uniquement, sans implémentation.
90: 
91: INTERDITS
92: - Ne pas modifier kinetic-models/src/arch/** ni kinetic-models/src/blocks/**.
93: - Ne pas modifier kinetic-cuda/** sans demande.
94: - Ne pas changer la sémantique d'un noyau existant sans gate d'identité.
95: - Ne pas copier de code llama.cpp.
96: 
97: BOUCLE DE TRAVAIL
98: 1. Vérifier que la machine est idle (rocm-smi, aucun processus kinetic/llama).
99: 2. Mesurer la baseline : kinetic bench --device vulkan --json -r 6, et
100:    llama-bench -dev Vulkan0 -p 512 -n 128 -r 6 sur le même fichier.
101: 3. Calculer le plafond : bande passante mesurée, FLOPS mesurés, KV.
102: 4. Profiler le noyau dominant (RGP) et écrire l'hypothèse.
103: 5. Modifier un seul levier : GEMM tuilé, coopmat, split-K, async, cache de
104:    pipelines, MoE groupé, KV q8_0.
105: 6. Mesurer le candidat, puis re-mesurer le contrôle.
106: 7. Accepter/rejeter avec raison, consigner dans PLAN.md.
107: 8. Répéter.
108: 
109: COMMANDES
110: - cargo build --release --locked -p kinetic-cli --features vulkan
111: - cargo test --release --locked -p kinetic-vulkan
112: - cargo test --release --locked -p kinetic-models
113: - target/release/kinetic bench MODEL.gguf --device vulkan --json -r 6
114: - llama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6
115: - RGP capture sur pp512 ; rocm-smi --showuse --showtemp --showmemuse
116: 
117: CRITÈRES DE SUCCÈS
118: - pp512/pp2048 ≥ 2× la baseline, tg128 sans régression > 3 %.
119: - Tests d'identité verts ; edit_identical: true.
120: - Chaque essai dans PLAN.md avec commande, commit, date, verdict.
121: - Aucun unsafe sans commentaire SAFETY.
122: 
123: RAPPORT FINAL
124: Tableau : métrique | baseline | maintenant | plafond | méthode.
125: Puis : ce qui a marché, ce qui a échoué, ce qui reste, risques.
126: ```
127: 
128: ---
129: 
130: ## Agent 2 — Runtime d'inférence
131: 
132: ```text
133: Tu es architecte runtime d'inférence pour Kinetic. Ta mission est de faire
134: passer le serveur d'un modèle « N backends indépendants » à un ordonnanceur
135: token-level qui tient la concurrence, le TTFT et le p99, sans casser RAD-1,
136: Warm Start ni Convoy.
137: 
138: LECTURE OBLIGATOIRE
139: - PLAN.md, sections 2, 3 et 4 (P1.5, P1.6).
140: - llama-kinetic.md, sections 7.6, 7.9, 7.10.
141: - docs/experts.md, section 0 (un seul backend complet par API).
142: - crates/kinetic-server/src/{engine,sequence,backend,server}.rs.
143: - crates/kinetic-models/src/{convoy,memory,per_layer}.rs.
144: - crates/kinetic-models/src/warm/**.
145: - docs/src/how-it-is-fast.md.
146: 
147: PÉRIMÈTRE D'ÉCRITURE
148: - crates/kinetic-runtime/** (à créer : scheduler, pool KV, métriques)
149: - crates/kinetic-server/src/{engine,sequence,backend,server}.rs
150: - crates/kinetic-models/src/{convoy,memory,per_layer}.rs
151: - crates/kinetic-models/src/warm/**
152: - PLAN.md, ledger.jsonl, configs/
153: 
154: INTERDITS
155: - Ne pas modifier les noyaux GPU.
156: - Ne pas changer les contrats Family/Target/Backend sans demande.
157: - Ne pas casser l'égalité batch / mono-token.
158: - Ne pas introduire de dépendance async lourde sans justification.
159: 
160: BOUCLE DE TRAVAIL
161: 1. Cartographier le runtime actuel : slots, files, Convoy, Warm Start,
162:    Zero Wait, KV.
163: 2. Mesurer c1/c4/c16/c64, TTFT, p50/p99, cache hit.
164: 3. Identifier le prochain plafond (KV, ordonnancement, chunking, fairness).
165: 4. Écrire un design court avec les invariants d'exactitude.
166: 5. Implémenter un seul changement : pool KV, chunked prefill global,
167:    priorité, annulation, métriques.
168: 6. Mesurer, contrôler, consigner.
169: 7. Répéter.
170: 
171: COMMANDES
172: - target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
173: - oha -n 64 -c 4 --json http://127.0.0.1:8080/v1/chat/completions ...
174: - python scripts/same_answer_demo.py --model MODEL.gguf \
175:     --kinetic target/release/kinetic --kinetic-args "--device vulkan"
176: - python scripts/server_abuse.py --port 8091
177: 
178: CRITÈRES DE SUCCÈS
179: - Débit agrégé en hausse à c4/c16/c64.
180: - p99 et TTFT mesurés et reproductibles.
181: - same_answer_demo identique sous charge.
182: - Aucun test d'identité cassé, aucune fuite mémoire.
183: 
184: RAPPORT FINAL
185: Tableau : métrique | baseline | maintenant | plafond | méthode.
186: Puis : design retenu, invariants préservés, limites, prochains leviers.
187: ```
188: 
189: ---
190: 
191: ## Agent 3 — Rust systèmes/backend
192: 
193: ```text
194: Tu es architecte Rust systèmes/backend pour Kinetic. Ta mission est de
195: réduire la duplication CUDA/Vulkan, de faire de chaque API un backend complet
196: et de préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.
197: 
198: LECTURE OBLIGATOIRE
199: - docs/experts.md, section 0 (un seul backend complet par API).
200: - docs/src/contributing.md, docs/src/building.md.
201: - crates/kinetic-models/src/{lib,model}.rs.
202: - crates/kinetic-models/src/{cuda,vulkan}/**.
203: - crates/kinetic-cuda/src/**, crates/kinetic-vulkan/src/**.
204: - crates/kinetic-server/src/{backend,engine}.rs.
205: - PLAN.md, section 3.
206: 
207: PÉRIMÈTRE D'ÉCRITURE
208: - Cargo.toml, crates/*/Cargo.toml
209: - crates/kinetic-runtime/** (à créer : devices, capacités, factory)
210: - crates/kinetic-models/src/runner/** (à créer : trait ModelRunner et runner CPU)
211: - crates/kinetic-models/src/{lib,model}.rs (retrait des modules backend)
212: - crates/kinetic-cuda/**, crates/kinetic-vulkan/** (accueil des exécuteurs)
213: - crates/kinetic-server/src/{backend,lib}.rs
214: - docs/src/contributing.md, docs/src/building.md
215: 
216: INTERDITS
217: - Ne pas modifier les noyaux ou les shaders.
218: - Ne pas changer le comportement observable.
219: - Ne pas introduire MLIR ou un compilateur de graphes.
220: - Ne pas casser les seams Family/Target/Backend/Projector.
221: 
222: BOUCLE DE TRAVAIL
223: 1. Cartographier la duplication et les dépendances (cargo tree, wc -l).
224: 2. Écrire le design de kinetic-runtime (devices, capacités, factory, erreurs)
225:    et du trait ModelRunner (charger, préfill, décoder, état).
226: 3. Extraire ModelRunner dans kinetic-models, avec le runner CPU comme
227:    référence, tests verts.
228: 4. Déplacer un exécuteur GPU derrière le trait, tests verts.
229: 5. Mesurer build, démarrage, overhead hôte.
230: 6. Répéter pour le second exécuteur.
231: 7. Retirer les features backend de kinetic-models.
232: 8. Ajouter la factory de backend et le backend de test.
233: 
234: COMMANDES
235: - cargo build --release --locked --workspace --all-features
236: - cargo test --release --locked --workspace
237: - cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
238: - cargo tree -p kinetic-models
239: - cargo bloat --release -p kinetic-cli
240: - perf record -g target/release/kinetic ...
241: 
242: CRITÈRES DE SUCCÈS
243: - kinetic-models ne dépend plus de CUDA/Vulkan.
244: - Un seul crate par API contient tout le code de cette API.
245: - Tests d'identité verts à chaque étape.
246: - Build et démarrage non régressés.
247: - Chaque unsafe a un commentaire SAFETY et un test.
248: 
249: RAPPORT FINAL
250: Tableau : duplication avant/après, build, démarrage, unsafe.
251: Puis : migration effectuée, ce qui reste, risques.
252: ```
253: 
254: ---
255: 
256: ## Agent 4 — Quantification et exactitude numérique
257: 
258: ```text
259: Tu es ingénieur quantification et exactitude numérique pour Kinetic. Ta
260: mission est d'étendre la quantification et de garantir que chaque gain de
261: vitesse reste exact.
262: 
263: LECTURE OBLIGATOIRE
264: - docs/src/contributing.md, section « Adding a weight format or a kernel ».
265: - crates/kinetic-quant/src/{types,dequant,encode}/**.
266: - crates/kinetic-cpu/src/kdot/tests.rs.
267: - llama-kinetic.md, section 7.7.
268: - PLAN.md, P0.4.
269: - crates/kinetic-gguf/src/**.
270: 
271: PÉRIMÈTRE D'ÉCRITURE
272: - crates/kinetic-quant/**
273: - crates/kinetic-gguf/**
274: - crates/kinetic-cpu/src/kdot/**
275: - Tests de noyaux dans crates/kinetic-cuda/tests/** et
276:   crates/kinetic-vulkan/tests/**
277: - docs/src/models.md
278: - PLAN.md, ledger.jsonl, configs/
279: 
280: INTERDITS
281: - Ne pas modifier les architectures ni le runtime.
282: - Ne pas accepter un noyau qui ne passe pas la référence f64.
283: - Ne pas changer un format existant sans versionner.
284: - Ne pas copier de tables sans NOTICE.
285: 
286: BOUCLE DE TRAVAIL
287: 1. Auditer les gates existants et le corpus de conformance.
288: 2. Choisir un format : NVFP4, FP8 E4M3, IQ encode, imatrix, KV q8_0.
289: 3. Écrire la référence f64 et les tests différentiels.
290: 4. Implémenter le noyau CPU, puis GPU si applicable.
291: 5. Mesurer qualité (KL/perplexité), vitesse, mémoire.
292: 6. Consigner et recommander.
293: 
294: COMMANDES
295: - cargo test --release --locked -p kinetic-quant
296: - cargo test --release --locked -p kinetic-cpu
297: - cargo test --release --locked -p kinetic-vulkan
298: - target/release/kinetic quantize MODEL.gguf --help
299: - target/release/kinetic perplexity MODEL.gguf
300: - python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
301:     --llama /path/to/llama.cpp/bin --device cpu
302: 
303: CRITÈRES DE SUCCÈS
304: - Chaque format a une référence f64 et un noyau testé.
305: - Aucune régression d'identité.
306: - Qualité mesurée, pas supposée.
307: - Chiffres reproductibles et consignés.
308: 
309: RAPPORT FINAL
310: Tableau : format | qualité | vitesse | mémoire | gate.
311: Puis : formats ajoutés, limites, risques.
312: ```
313: 
314: ---
315: 
316: ## Agent 5 — Performance et qualification
317: 
318: ```text
319: Tu es ingénieur performance et qualification pour Kinetic. Ta mission est de
320: rendre la mesure automatisée, reproductible et infalsifiable.
321: 
322: LECTURE OBLIGATOIRE
323: - PLAN.md, sections 2 et 7.
324: - llama-kinetic.md, sections 7.10 à 7.12.
325: - docs/src/benchmarks.md.
326: - scripts/{bench_protocol,bench_vs_llamacpp,conformance,server_abuse,same_answer_demo}.py.
327: - .github/workflows/ci.yml.
328: 
329: PÉRIMÈTRE D'ÉCRITURE
330: - scripts/**
331: - .github/workflows/**
332: - PLAN.md, THEORY.md, ledger.jsonl, configs/, best-config.json
333: - docs/src/benchmarks.md
334: 
335: INTERDITS
336: - Ne pas modifier les noyaux ou le runtime pour « améliorer » un chiffre.
337: - Ne pas publier un chiffre sans commande, commit, date, état machine.
338: - Ne pas supprimer un résultat négatif.
339: 
340: BOUCLE DE TRAVAIL
341: 1. Écrire THEORY.md : bande passante, FLOPS, KV, TTFT.
342: 2. Créer le ledger et les configs.
343: 3. Automatiser le protocole A/B (contrôle, candidat, contrôle).
344: 4. Étendre conformance.py à Vulkan.
345: 5. Ajouter un test needle 50 %/100 %.
346: 6. Ajouter un soak concurrent et un watchdog mémoire.
347: 7. Ajouter une CI GPU avec seuil de régression.
348: 
349: COMMANDES
350: - python scripts/bench_protocol.py MODEL.gguf --llama-bin /path/to/llama-bench
351: - python scripts/bench_vs_llamacpp.py MODEL.gguf --llama-dir /path/to/llama.cpp/bin
352: - python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
353:     --llama /path/to/llama.cpp/bin --device cpu
354: - python scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091
355: - python scripts/same_answer_demo.py --model MODEL.gguf --kinetic target/release/kinetic
356: - rocm-smi --showuse --showtemp --showmemuse
357: 
358: CRITÈRES DE SUCCÈS
359: - Un candidat à +5 % est distingué du bruit.
360: - Une régression est détectée automatiquement.
361: - Chaque chiffre publié est reproductible.
362: - Les résultats négatifs sont conservés.
363: 
364: RAPPORT FINAL
365: Tableau : métrique | baseline | maintenant | plafond | méthode.
366: Puis : protocole, gates, faux positifs/négatifs, limites.
367: ```
368: 
369: ---
370: 
371: ## Agent 6 — Intégration des architectures
372: 
373: ```text
374: Tu es ingénieur intégration des architectures pour Kinetic. Ta mission est
375: de couvrir les architectures manquantes par briques réutilisables.
376: 
377: LECTURE OBLIGATOIRE
378: - docs/src/adding-a-family.md.
379: - llama-kinetic.md, sections 4.1, 5.1, 6.
380: - crates/kinetic-models/src/arch/mod.rs (FAMILIES).
381: - crates/kinetic-models/src/arch/transformer.rs.
382: - crates/kinetic-models/src/blocks/**.
383: - scripts/new_family.py.
384: 
385: PÉRIMÈTRE D'ÉCRITURE
386: - crates/kinetic-models/src/arch/**
387: - crates/kinetic-models/src/blocks/**
388: - docs/src/models.md
389: - scripts/new_family.py (si nécessaire)
390: 
391: INTERDITS
392: - Ne pas ajouter de match central.
393: - Ne pas dupliquer transformer.rs.
394: - Ne pas modifier les backends.
395: - Ne pas copier de code llama.cpp.
396: 
397: BOUCLE DE TRAVAIL
398: 1. Choisir une famille dans le plan.
399: 2. Lancer scripts/new_family.py sur un GGUF.
400: 3. Mapper les tenseurs vers les blocs.
401: 4. Ajouter une ligne dans FAMILIES.
402: 5. Tester kinetic info, kinetic run, conformance.
403: 6. Documenter.
404: 
405: COMMANDES
406: - target/release/kinetic info MODEL.gguf
407: - target/release/kinetic run MODEL.gguf --device cpu "Hello"
408: - python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
409:     --llama /path/to/llama.cpp/bin --device cpu
410: - cargo test --release --locked -p kinetic-models
411: 
412: CRITÈRES DE SUCCÈS
413: - Le modèle se charge et génère.
414: - Tokens identiques à llama.cpp sur le corpus.
415: - Aucun match central.
416: - Documentation à jour.
417: 
418: RAPPORT FINAL
419: Tableau : famille | chargement | tokens | tests | limites.
420: Puis : blocs ajoutés, réutilisation, prochaines familles.
421: ```
422: 
423: ---
424: 
425: ## Agent 7 — Tokenizers, templates et tool calls
426: 
427: ```text
428: Tu es ingénieur tokenizers/templates/tool calls pour Kinetic. Ta mission est
429: d'atteindre la compatibilité des tokenizers et des templates.
430: 
431: LECTURE OBLIGATOIRE
432: - llama-kinetic.md, sections 4.2, 5.3.
433: - crates/kinetic-tokenizer/src/**.
434: - crates/kinetic-server/src/{template,tool_calls}.rs.
435: - scripts/conformance.py.
436: - docs/src/models.md.
437: 
438: PÉRIMÈTRE D'ÉCRITURE
439: - crates/kinetic-tokenizer/**
440: - crates/kinetic-server/src/{template,tool_calls,tool_grammar}.rs
441: - docs/src/models.md
442: 
443: INTERDITS
444: - Ne pas modifier les architectures ni les noyaux.
445: - Ne pas changer le contrat Tokenizer::from_gguf/encode/decode.
446: - Ne pas copier de regex sans NOTICE.
447: 
448: BOUCLE DE TRAVAIL
449: 1. Construire un corpus de contrôle.
450: 2. Comparer avec llama-tokenize.
451: 3. Ajouter un prétokenizer ou un template.
452: 4. Tester les tokens spéciaux et les tool calls.
453: 5. Documenter.
454: 
455: COMMANDES
456: - llama-tokenize -m MODEL.gguf -f corpus.txt
457: - cargo test --release --locked -p kinetic-tokenizer
458: - python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
459:     --llama /path/to/llama.cpp/bin --device cpu
460: 
461: CRITÈRES DE SUCCÈS
462: - Ids identiques sur le corpus.
463: - Tokens spéciaux corrects.
464: - Tool calls valides.
465: 
466: RAPPORT FINAL
467: Tableau : prétokenizer/template | ids | tool calls | tests.
468: Puis : ajouts, limites, prochains formats.
469: ```
470: 
471: ---
472: 
473: ## Agent 8 — Serveur, API et observabilité
474: 
475: ```text
476: Tu es ingénieur serveur/API/observabilité pour Kinetic. Ta mission est la
477: parité API, la robustesse et l'observabilité.
478: 
479: LECTURE OBLIGATOIRE
480: - llama-kinetic.md, sections 4.5, 5.9.
481: - PLAN.md, P0.1.
482: - scripts/{llama_server_compat,server_abuse}.py.
483: - docs/src/server.md, docs/src/clients.md.
484: - crates/kinetic-server/src/{server,http,api,llama,metrics}.rs,
485:   crates/kinetic-server/src/ollama/.
486: 
487: PÉRIMÈTRE D'ÉCRITURE
488: - crates/kinetic-server/src/{server,http,api,llama,metrics}.rs,
489:   crates/kinetic-server/src/ollama/
490: - docs/src/server.md, docs/src/clients.md
491: 
492: INTERDITS
493: - Ne pas modifier les noyaux ni le runtime.
494: - Ne pas casser les clients existants.
495: - Ne pas ajouter de dépendance réseau non nécessaire.
496: 
497: BOUCLE DE TRAVAIL
498: 1. Comparer les routes et les champs avec llama-server.
499: 2. Ajouter une route ou un champ.
500: 3. Tester avec un client réel.
501: 4. Tester server_abuse.
502: 5. Documenter.
503: 
504: COMMANDES
505: - target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
506: - python scripts/llama_server_compat.py --model MODEL.gguf \
507:     --llama-server /path/to/llama-server --kinetic target/release/kinetic
508: - python scripts/server_abuse.py --port 8091
509: 
510: CRITÈRES DE SUCCÈS
511: - Clients existants fonctionnels.
512: - Champs inconnus ignorés ou refusés clairement.
513: - Pas de crash sous abuse.
514: - Métriques exploitables.
515: 
516: RAPPORT FINAL
517: Tableau : route | statut | test | limite.
518: Puis : ajouts, compatibilité, prochaines routes.
519: ```
520: 
521: ---
522: 
523: ## Agent 9 — Multimodal
524: 
525: ```text
526: Tu es ingénieur multimodal pour Kinetic. Ta mission est la vision, l'audio
527: et l'OCR.
528: 
529: LECTURE OBLIGATOIRE
530: - llama-kinetic.md, sections 4.10, 5.2.
531: - crates/kinetic-vision/src/{lib,ops,preprocess}.rs.
532: - crates/kinetic-vision/examples/encode.rs.
533: - docs/src/models.md.
534: 
535: PÉRIMÈTRE D'ÉCRITURE
536: - crates/kinetic-vision/**
537: - crates/kinetic-audio/** (à créer)
538: - crates/kinetic-models/src/input.rs
539: - docs/src/models.md
540: 
541: INTERDITS
542: - Ne pas modifier le texte ni les backends.
543: - Ne pas élargir le contrat Projector pour l'audio.
544: - Ne pas copier de code llama.cpp.
545: 
546: BOUCLE DE TRAVAIL
547: 1. Choisir un projecteur.
548: 2. Lire les poids et les hyperparamètres.
549: 3. Implémenter prepare/encode.
550: 4. Comparer à llama-mtmd-debug.
551: 5. Documenter.
552: 
553: COMMANDES
554: - cargo run --release -p kinetic-vision --example encode -- mmproj.gguf rainbow:512
555: - llama-mtmd-debug --image rainbow -n 512 -m mmproj.gguf
556: - target/release/kinetic serve MODEL.gguf --mmproj mmproj.gguf
557: 
558: CRITÈRES DE SUCCÈS
559: - Embeddings proches de llama.cpp.
560: - Réponses cohérentes.
561: - Pas de régression texte.
562: 
563: RAPPORT FINAL
564: Tableau : projecteur | parité | vitesse | tests.
565: Puis : ajouts, limites, prochains projecteurs.
566: ```
567: 
568: ---
569: 
570: ## Agent 10 — Build, release et bindings
571: 
572: ```text
573: Tu es ingénieur build/release/bindings pour Kinetic. Ta mission est de
574: rendre le produit installable et intégrable.
575: 
576: LECTURE OBLIGATOIRE
577: - docs/src/building.md, docs/src/bindings.md.
578: - .github/workflows/*.yml.
579: - crates/kinetic-capi/**.
580: - docker/Dockerfile*, packaging/**.
581: - Cargo.toml, bindings/python/Cargo.toml.
582: 
583: PÉRIMÈTRE D'ÉCRITURE
584: - .github/workflows/**
585: - docker/**, packaging/**
586: - bindings/**
587: - crates/kinetic-capi/**
588: - docs/src/{building,bindings}.md
589: 
590: INTERDITS
591: - Ne pas modifier les noyaux ni le runtime.
592: - Ne pas casser l'ABI existante sans versionner.
593: - Ne pas publier un artefact non testé.
594: 
595: BOUCLE DE TRAVAIL
596: 1. Auditer l'ABI et les bindings.
597: 2. Construire sur les trois OS.
598: 3. Tester les exemples C et Python.
599: 4. Réduire la taille et le temps de build.
600: 5. Documenter.
601: 
602: COMMANDES
603: - cargo build --release --locked --workspace --all-features
604: - cargo test --release --locked --workspace
605: - cd bindings/python && maturin build --release --locked
606: - docker build -f docker/Dockerfile -t kinetic .
607: 
608: CRITÈRES DE SUCCÈS
609: - Installation réussie sur Windows/Linux/macOS.
610: - ABI testée.
611: - Build reproductible.
612: - Documentation à jour.
613: 
614: RAPPORT FINAL
615: Tableau : artefact | build | tests | taille | limite.
616: Puis : ajouts, compatibilité, prochaines étapes.
617: ```
618: 
619: ---
620: 
621: ## Utilisation avec OpenCode
622: 
623: - Chaque prompt peut être donné à un subagent `general` ou à un agent
624:   spécialisé.
625: - Donner à l'agent **un seul rôle** à la fois : ne pas mélanger GPU,
626:   runtime et modèles dans la même session.
627: - Toujours fournir le chemin du modèle et le backend cible.
628: - Demander un rapport au format tableau + verdict + prochaines étapes.
629: - Pour les modifications de code, demander un diff minimal et les tests
630:   exécutés.
631: - Pour les mesures, demander la commande exacte, la date, le commit et
632:   l'état machine.
633: 
634: ### Exemple de lancement
635: 
636: ```text
637: Lis docs/experts/prompts-agents.md, section « Agent 1 — GPU AMD ».
638: Tu es cet agent. Le modèle est MODEL.gguf, le backend est vulkan.
639: Commence par la baseline et le profil RGP, puis propose un seul levier.
640: Ne modifie rien hors de ton périmètre. Termine par le rapport demandé.
641: ```
642: 
643: ### Exemple de garde-fou
644: 
645: ```text
646: Avant toute modification :
647: - vérifie que la machine est idle ;
648: - mesure la baseline ;
649: - écris ton hypothèse et le plafond calculé ;
650: - annonce le fichier exact que tu vas modifier.
651: 
652: Après toute modification :
653: - lance les tests d'exactitude ;
654: - mesure le candidat puis le contrôle ;
655: - consigne dans PLAN.md ;
656: - donne le verdict avec la raison.
657: ```

---

## Assistant

_Thinking:_

Now I need to read the session file to see what was requested. It's 1.1MB, 21,479 lines. Let me search for key things: mentions of experts, missions, prompts, fiches-de-poste, etc.

Let me get an overview of the session.

Now let me examine the session file to understand what was requested and produced.



**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && head -100 session-ad321db8-5ba0-40bf-b50d-4be596739cef.md"
}
```

# Quels experts pour inference-optimisation

**Session ID:** ses_dfdfbe085ffek91B5Cjyuv4l6V
**Created:** 11/22/2026, 10:49:14 AM
**Updated:** 11/22/2026, 11:30:23 AM

---

## User

@inference-optimisation Analyse le projet et détermine les experts ( quels compétences ) ceux-ci devraient avoir pour améliorer l'architecture et les performances de cette application. 

---

## Assistant

_Thinking:_

The user is asking (in French) to analyze the project and determine the experts (which skills/competencies) they should have to improve the architecture and performance of this application. The skill "inference-optimisation" was loaded, but the request is more about analyzing the project and determining what expert profiles/competencies would be needed to improve architecture and performance.

Let me first explore the project to understand what it is.

Working directory: /home/mrnemesys/Devs/kinetic

Let me explore.

**Tool: read**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic"
}
```

Read directory /home/mrnemesys/Devs/kinetic, entries 1-25
.git/
.github/
assets/
bindings/
crates/
docker/
docs/
packaging/
scripts/
target/
.gitattributes
.gitignore
Cargo.lock
Cargo.toml
CHANGELOG.md
CODE_OF_CONDUCT.md
CONTRIBUTING.md
LICENSE
llama-kinetic.md
NOTICE
PLAN.md
README.md
rust-toolchain.toml
SECURITY.md
TRADEMARKS.md

**Tool: shell**

**Input:**
```json
{
  "command": "ls -la /home/mrnemesys/Devs/kinetic && find /home/mrnemesys/Devs/kinetic -maxdepth 2 -type f -name \"*.md\" | head -50"
}
```

total 308
drwxrwxr-x 12 mrnemesys mrnemesys  4096 nov.  22 04:51 .
drwxrwxr-x 21 mrnemesys mrnemesys  4096 nov.  21 22:56 ..
drwxrwxr-x  5 mrnemesys mrnemesys  4096 nov.  21 12:36 assets
drwxrwxr-x  4 mrnemesys mrnemesys  4096 nov.  21 12:36 bindings
-rw-rw-r--  1 mrnemesys mrnemesys 42475 nov.  21 12:36 Cargo.lock
-rw-rw-r--  1 mrnemesys mrnemesys   266 nov.  21 12:36 Cargo.toml
-rw-rw-r--  1 mrnemesys mrnemesys  6113 nov.  21 12:36 CHANGELOG.md
-rw-rw-r--  1 mrnemesys mrnemesys  5480 nov.  21 12:36 CODE_OF_CONDUCT.md
-rw-rw-r--  1 mrnemesys mrnemesys  4848 nov.  21 12:36 CONTRIBUTING.md
drwxrwxr-x 16 mrnemesys mrnemesys  4096 nov.  21 12:36 crates
drwxrwxr-x  2 mrnemesys mrnemesys  4096 nov.  21 12:36 docker
drwxrwxr-x  4 mrnemesys mrnemesys  4096 nov.  21 12:36 docs
drwxrwxr-x  8 mrnemesys mrnemesys  4096 nov.  22 10:24 .git
-rw-rw-r--  1 mrnemesys mrnemesys    46 nov.  21 12:36 .gitattributes
drwxrwxr-x  4 mrnemesys mrnemesys  4096 nov.  21 12:36 .github
-rw-rw-r--  1 mrnemesys mrnemesys    88 nov.  21 12:36 .gitignore
-rw-rw-r--  1 mrnemesys mrnemesys 11358 nov.  21 12:36 LICENSE
-rw-rw-r--  1 mrnemesys mrnemesys 97899 nov.  22 05:41 llama-kinetic.md
-rw-rw-r--  1 mrnemesys mrnemesys  1356 nov.  21 12:36 NOTICE
drwxrwxr-x  4 mrnemesys mrnemesys  4096 nov.  21 12:36 packaging
-rw-rw-r--  1 mrnemesys mrnemesys 14470 nov.  22 10:15 PLAN.md
-rw-rw-r--  1 mrnemesys mrnemesys 37445 nov.  22 04:51 README.md
-rw-rw-r--  1 mrnemesys mrnemesys    66 nov.  21 12:36 rust-toolchain.toml
drwxrwxr-x  2 mrnemesys mrnemesys  4096 nov.  22 05:16 scripts
-rw-rw-r--  1 mrnemesys mrnemesys  1583 nov.  21 12:36 SECURITY.md
drwxrwxr-x  5 mrnemesys mrnemesys  4096 nov.  22 05:10 target
-rw-rw-r--  1 mrnemesys mrnemesys  2713 nov.  21 12:36 TRADEMARKS.md
/home/mrnemesys/Devs/kinetic/.github/PULL_REQUEST_TEMPLATE.md


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n \"^## \\|^# \\|^### \" session-ad321db8-5ba0-40bf-b50d-4be596739cef.md | head -200"
}
```

1:# Quels experts pour inference-optimisation
9:## User
15:## Assistant
115:## Assistant
723:## Assistant
945:## Assistant
1551:## Assistant
2069:## Assistant
2690:## Assistant
3196:# On macOS the executor drives the GPU through Metal itself, which needs no
3197:# driver installed; see the `metal` feature of kinetic-vulkan.
3210:# On macOS, drives the GPU through Metal itself instead of a Vulkan driver
3211:# layered on it. The kernels are the same WGSL, written out as Metal
3212:# shaders at build time, with GEMV kernels of its own from shaders/metal.
3213:# No effect on other systems.
3276:# Runs models on NVIDIA GPUs; the kernels are built when nvcc is found. The
3277:# driver itself is opened at runtime, so the binary still starts, on the
3278:# CPU, on machines without one.
3280:# Runs models on AMD, Intel and NVIDIA GPUs through Vulkan. The kernels are
3281:# compiled to SPIR-V at build time and the loader is opened at runtime, so
3282:# this needs no SDK to build and no driver to start.
3313:## Assistant
3675:## Assistant
4240:## Assistant
4937:## Assistant
5032:# Models and formats
5066:## Architectures
5095:### Embedding and ranking models
5111:### Vision
5147:### Tokenizers
5153:# Devices and offload
5161:## Choosing the device
5197:## CPU
5207:## Apple GPUs
5219:## NVIDIA GPUs
5251:## Vulkan
5264:## Dense models larger than the GPU
5279:## Several GPUs
5287:## Assistant
5305:## Project diagnosis
5335:### 1. GPU compute kernel engineer — Vulkan/RDNA (top priority)
5347:### 2. HIP/ROCm backend engineer (top priority locally, second wave)
5353:### 3. Inference runtime / scheduler architect
5365:### 4. Rust systems architect / performance engineer
5377:### 5. Quantization & numerical correctness engineer
5388:### 6. Model integration engineer (architectures/tokenizers)
5399:### 7. Server/API & platform engineer
5408:### 8. Multimodal engineer (vision/audio/OCR)
5414:### 9. Performance measurement / QA automation engineer (SRE for benchmarks)
5425:### 10. Build/release & bindings engineer
5431:### 11. Technical writer / developer experience (optional)
5434:### 12. Security engineer? 
5676:## Assistant
5893:# both servers answer directly rather than reasoning first
5931:## Assistant
6038:## Assistant
6062:## 1. Diagnostic synthétique
6093:### P0 — Expert 1: Ingénieur noyaux GPU AMD (Vulkan/RDNA4)
6104:### P0 — Expert 2: Architecte runtime d'inférence / ordonnanceur
6115:### P0/P1 — Expert 3: Architecte Rust systèmes / performance
6124:### P1 — Expert 4: Ingénieur quantification & exactitude numérique
6133:### P1 — Expert 5: Ingénieur backends HIP/ROCm (ou "portabilité GPU")
6141:### P1 — Expert 6: Ingénieur intégration modèles (architectures)
6150:### P1/P2 — Expert 7: Ingénieur tokenizers/templates/tool-calls
6157:### P2 — Expert 8: Ingénieur serveur/API & observabilité
6166:### P2 — Expert 9: Ingénieur multimodal (vision/audio/OCR)
6171:### P2 — Expert 10: Ingénieur mesure/qualification (perf & release)
6180:### P2 — Expert 11: Ingénieur build/release & bindings
6185:### Cross-cutting: sécurité
6298:### Profile 1 details
6321:### Profile 2 details
6339:### Profile 3 details
6356:### Profile 4 details
6373:### Profile 5 details
6389:### Profile 6 details
6405:### Profile 7 details
6417:### Profile 8 details
6429:### Profile 9 details
6441:### Profile 10 details
6453:### Profile 11 details
6695:## Assistant
7048:### P0 — 1. Ingénieur noyaux GPU AMD (Vulkan/RDNA4, puis HIP)
7357:# Kinetic — quels experts pour améliorer l’architecture et la performance
7359:## Réponse courte
7372:## 1. Ce que le dépôt montre
7411:## 2. Les profils d’experts
7413:### Tableau de synthèse
7430:### P0 — 1. Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)
7466:### P0 — 2. Architecte runtime d’inférence
7499:### P0/P1 — 3. Ingénieur quantification et exactitude numérique
7531:### P0/P1 — 4. Architecte Rust systèmes / backend
7563:### P0 — 5. Ingénieur performance / qualification
7594:### P1 — 6. Ingénieur intégration des architectures
7611:### P1 — 7. Ingénieur tokenizers / templates / tool calls
7627:### P1 — 8. Ingénieur serveur / API / observabilité
7644:### P2 — 9. Ingénieur multimodal
7656:### P2 — 10. Ingénieur build/release/bindings
7668:## 3. Composition d’équipe recommandée
7670:### Cellule « performance » minimale (5 personnes)
7680:### Équipe complète (9–10 personnes)
7684:### Séquencement
7695:### Matrice chantier → pilote
7712:## 4. Comment évaluer ces experts
7737:## 5. Ce qu’il ne faut pas faire
7748:## 6. Recommandation finale
7769:## User
7775:## Assistant
7856:### Main `docs/experts.md`
7870:### Architecture correction
7905:### Expert profiles
7911:### Measurement commands
8149:# Building from source
8154:## Requirements
8164:## Build
8207:## The libraries
8223:## Docker
8233:## Tests
8252:## Checks before a pull request
8263:## The documentation
8304:## Assistant
8351:## Assistant
8443:## Assistant
8489:### Main doc `docs/experts.md`
8494:# Experts pour l'architecture et la performance de Kinetic
8499:## 0. Résumé
8503:## 1. Diagnostic
8504:### 1.1 Ce que le dépôt fait bien
8505:### 1.2 Les goulets mesurés
8507:### 1.3 Les goulets structurels
8509:### 1.4 Conséquence : quels experts, pas quels frameworks
8511:## 2. Architecture cible : un backend complet par API
8512:### 2.1 Principe
8513:### 2.2 État actuel vs cible
8515:### 2.3 Migration incrémentale
8517:### 2.4 Règle pour les backends de compatibilité
8518:### 2.5 Ce que ça change pour les experts
8520:## 3. Profils d'experts
8521:### 3.1 Tableau de synthèse
8522:### 3.2 P0 — GPU AMD
8523:### 3.3 P0 — Runtime
8524:### 3.4 P0/P1 — Rust backend
8525:### 3.5 P0/P1 — Quantification/exactitude
8526:### 3.6 P0 — Performance/QA
8527:### 3.7 P1 — Intégration modèles
8528:### 3.8 P1 — Tokenizers
8529:### 3.9 P1 — Serveur/API
8530:### 3.10 P2 — Multimodal
8531:### 3.11 P2 — Build/release/bindings
8535:## 4. Composition et séquencement
8536:### 4.1 Cellule minimale
8537:### 4.2 Équipe complète
8538:### 4.3 Phases
8539:### 4.4 RACI
8541:## 5. Évaluation et recrutement
8542:### 5.1 Exercices
8543:### 5.2 Signaux d'alerte
8544:### 5.3 Rareté et sourcing
8546:## 6. Commandes de mesure communes
8547:### 6.1 Build/test
8548:### 6.2 Inventaire
8549:### 6.3 Bench
8550:### 6.4 Conformance/exactitude
8551:### 6.5 Serveur
8552:### 6.6 Précautions
8554:## 7. Livrables
8558:## 8. Ce qu'il ne faut pas faire
8573:### Architecture section details
8634:### Expert profiles: scope and context
8699:### Job descriptions
8704:# Fiche de poste — <titre>
8736:### Agent prompts
8741:## Agent <nom>
8771:### Need to update PLAN.md?
8775:### Check if `docs/experts/` directory exists
8779:### Language and style
8783:### Before writing, let's inspect `scripts/llama_server_compat.py` and `llama_presets.py` to give accurate commands.
8909:# GGUF value types, see the GGUF spec in ggml's docs/gguf.md
8998:## Assistant
9011:# Experts pour l'architecture et la performance de Kinetic
9088:# Build
9091:# Tests
9096:# Inventory
9100:# Bench
9104:# Protocol
9107:# Conformance
9110:# Server compatibility
9113:# Abuse
9116:# Same answer
9119:# Vision
9169:## Assistant
9279:## Assistant
9291:### Writing the files
9345:## Assistant
9390:## Assistant
9462:## Assistant
9558:# Summary
9562:# Getting started
9568:# Reference
9578:# Under the hood
9584:# Development


---

## Assistant

_Thinking:_

Let me continue reading the headings.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n \"^## \\|^# \\|^### \" session-ad321db8-5ba0-40bf-b50d-4be596739cef.md | sed -n '200,400p'"
}
```

9584:# Development
9590:# More
9618:## Assistant
9892:## Assistant
10187:## Assistant
10310:## Assistant
10434:## Assistant
10541:## Assistant
10778:# On macOS the executor drives the GPU through Metal itself, which needs no
10779:# driver installed; see the `metal` feature of kinetic-vulkan.
11140:## Assistant
11358:## Assistant
11908:## Assistant
12040:## Assistant
12131:## Assistant
12142:## 0. Décision structurante : un seul backend complet par API
12164:### 0.1 État actuel vs cible
12175:### 0.2 Où vivent les traits
12192:### 0.3 Migration incrémentale
12212:### 0.4 Contrat de compatibilité d'un backend
12232:### 0.5 Ce que cela change pour les experts
12234:### 0.6 Non-objectifs
12268:## Assistant
12413:## Assistant
12492:## Assistant
12513:## Assistant
12551:## Assistant
12572:## Assistant
12619:## Assistant
12652:## Assistant
12763:## Assistant
12845:## Assistant
12918:## Assistant
12980:## Assistant
13016:## Assistant
13127:## Assistant
13395:## Assistant
13441:## 0. Décision structurante : un seul backend complet par API
13469:### 0.1 État actuel vs cible
13480:### 0.2 Où vivent les traits
13500:### 0.3 Migration incrémentale
13520:### 0.4 Contrat de compatibilité d'un backend
13540:### 0.5 Ce que cela change pour les experts
13551:## Assistant
13592:## Assistant
13689:## Assistant
13729:## Assistant
13753:## Assistant
13785:## Assistant
13834:## Assistant
13868:## Assistant
13891:## Assistant
13913:## Assistant
13995:## Assistant
14020:## Assistant
14083:## Assistant
14173:## Assistant
14206:## Fichiers créés
14213:## Ce que change ta remarque sur les backends
14224:## Points corrigés
14229:## État
14232:## Prochaines étapes possibles
14265:## Fichiers créés
14275:## Ta règle « un seul backend par API »
14290:## Corrections factuelles intégrées
14297:## État du dépôt
14308:## Prochaines étapes possibles
14319:## User
14325:## Assistant
14957:## Assistant
15432:## Assistant
15584:## Assistant
15618:## Memory
15649:## Mixture-of-experts offload
15667:## Coming
15673:### Runtime options
15888:## Assistant
16249:## Assistant
16510:## Assistant
17107:## Assistant
17550:## Assistant
17748:## Assistant
17760:### 1. Current state (facts)
17777:### 2. Limits / risks
17790:### 3. Potential improvements (prioritized)
17811:### 4. Expert profile
17830:### 5. Concrete audit plan (first 2-4 weeks)
17843:### 6. What to fix in docs
17917:## User
17923:## Assistant
18442:## Mixture-of-experts offload
18460:## Coming
18473:## Flywheel
18501:## References
18593:## Assistant
18626:### 4.4 Backends et matériel
18638:### P0.2 — Vulkan (L)
19064:## Assistant
19513:## Assistant
19768:## Assistant
19971:## Assistant
20183:## Assistant
20284:## Assistant
20377:## Assistant
20481:## Assistant
20736:## Assistant
20851:## Assistant
20947:## Assistant
21127:## Assistant
21150:### 2.3 P0 — Architecte runtime d'inférence
21185:### 2.4 P0/P1 — Architecte Rust systèmes/backend
21221:### 2.5 P0/P1 — Ingénieur quantification et exactitude numérique
21228:## Assistant
21307:## 1. La règle backend est bien en place (documents)
21314:## 2. Ce que fait le cache KV aujourd'hui
21320:## 3. Améliorations, par priorité
21326:## 4. Quel expert doit mener l'analyse
21333:## 5. Proposition
21378:## 0. Décision structurante : un seul backend complet par API
21413:## Assistant


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '7769,8000p' session-ad321db8-5ba0-40bf-b50d-4be596739cef.md"
}
```

## User

`Si tu veux, je peux transformer cette analyse en document docs/experts.md, en fiches de poste, ou en prompts d’agents spécialisés avec, pour chacun, son périmètre d’écriture, ses fichiers de contexte et ses commandes de mesure. ` : j'accepte cela et `Duplication backend	kinetic-models/src/cuda ≈ 4,5 k LOC vs vulkan ≈ 2,2 k LOC ; kinetic-cuda ≈ 8,5 k LOC vs kinetic-vulkan ≈ 5,9 k LOC. Ajouter HIP/Metal/SYCL va multiplier la maintenance. ` ne devrait avoir qu'un seul backend pour cuda, et un seul pour vulkan logiquement, reprenant toutes les fonctionnalités pour cuda d'une part et l'autre ceux pour vulkan. Les autres backend est pour le compatibilité mais effectivement, l'utilisateur n'en utilisera qu'un seul au final par rapport à sa carte et nous devons lui fournir les bases. 

---

## Assistant

_Thinking:_

The user accepts the offer: turn the analysis into docs/experts.md, job descriptions, or specialized agent prompts with scope, context files, measurement commands. They also make an architectural point: the backend duplication should be resolved so that there is only one backend for CUDA and one for Vulkan, each containing all functionality for its respective backend. Other backends are for compatibility, but the user will only use one depending on their GPU, and we must provide the basics.

So I need to:
1. Create `docs/experts.md` (probably a comprehensive document in French) with expert profiles, skills, scope, context files, measurement commands.
2. Create job descriptions (fiches de poste) — maybe a directory `docs/experts/` with individual files? Or include them in the same document. The user accepted "cela" — the three things: document, job descriptions, agent prompts. We should deliver all three, likely as files.
3. Incorporate the architectural correction: one backend implementation per API (one CUDA backend, one Vulkan backend), not duplicated between `kinetic-models` and `kinetic-cuda`/`kinetic-vulkan`. The user says "ne devrait avoir qu'un seul backend pour cuda, et un seul pour vulkan logiquement, reprenant toutes les fonctionnalités pour cuda d'une part et l'autre ceux pour vulkan." So the architecture target is: each GPU API has exactly one backend crate/module that owns everything for that API (driver, kernels, executor, model orchestration), and the generic layer is backend-agnostic. Other backends (HIP, Metal, SYCL) are compatibility targets, but the user uses only one based on their GPU, and we must provide the bases.

Need to think about how to structure this. The existing code has:
- `kinetic-cuda` crate: driver API + kernels + launchers (8.5k LOC)
- `kinetic-vulkan` crate: driver + kernels (5.9k LOC)
- `kinetic-models/src/cuda/` (4.5k LOC): CUDA executor (model orchestration)
- `kinetic-models/src/vulkan/` (2.2k LOC): Vulkan executor
So the duplication is between the two executors, not within the same API. The user's point: "il ne devrait y avoir qu'un seul backend pour CUDA, et un seul pour Vulkan" — i.e., don't have `kinetic-models/src/cuda` + `kinetic-cuda` as two separate CUDA implementations; consolidate all CUDA functionality into one backend (and same for Vulkan). This is a valid architectural direction: one crate per backend that owns driver, kernels, and executor, with a shared generic model/block layer above. The generic layer should express the model graph, and each backend implements the execution. That avoids duplication across backends and within an API.

Need to be careful: the user's statement can be interpreted as "one backend implementation per API, not a split between `kinetic-models/src/cuda` and `kinetic-cuda`." We should reflect this in the document:
- Target architecture: `kinetic-backend` trait/IR (generic) + `kinetic-cuda` (all CUDA: driver, kernels, executor) + `kinetic-vulkan` (all Vulkan: driver, kernels, executor) + `kinetic-cpu` (all CPU). `kinetic-models` holds architectures/blocks/speculative/warm, backend-agnostic. No `kinetic-models/src/cuda` or `/vulkan` long-term.
- One backend per API, complete. Other APIs (HIP, Metal, SYCL, WebGPU) are optional compatibility backends implementing the same trait; the user only loads one at runtime. Provide the base/trait and a reference implementation, not all at once.
- The generic layer must not know CUDA/Vulkan specifics. Backend capabilities (supported types, features, limits) are declared by the backend.
- The executor should be inside the backend crate, or a shared `kinetic-runtime` crate with backend-specific kernels. Need to decide. The user says "un seul backend pour cuda, et un seul pour vulkan logiquement, reprenant toutes les fonctionnalités". So the cleanest: each backend crate owns its executor. `kinetic-models` owns model definitions and block semantics, and calls a `Backend` trait. But the current design has model orchestration in `kinetic-models/src/cuda` because it needs to fuse layers and manage device buffers. We can propose a `kinetic-runtime` crate with generic graph/scheduling, and backend crates implement kernels; or backend crates own their executor. To honor "one backend per API", put the executor in the backend crate, and keep `kinetic-models` as the model description (tensors, blocks, forward semantics) with a `Backend` trait. But that may require a big refactor. We can propose an incremental path:
  1. Define `Backend`/`Device` capability trait in a new `kinetic-runtime` (or `kinetic-backend`) crate.
  2. Move `kinetic-models/src/cuda` into `kinetic-cuda` as `executor` module; same for Vulkan.
  3. Keep `kinetic-models` free of `cfg(feature = "cuda"/"vulkan")` and backend modules.
  4. Shared host-side model semantics stay in `kinetic-models`; backend executes a lowered plan.
  5. Add HIP/Metal/SYCL as separate crates implementing the same trait, only when needed.
This is a strong architectural recommendation.

Need to create the files. Let's decide the deliverable structure:

- `docs/experts.md` — main document: diagnosis, target architecture (including the one-backend-per-API correction), expert profiles, team composition, RACI, evaluation, measurement commands, agent prompts? Maybe too much for one file.
- `docs/experts/` directory with:
  - `README.md` (index)
  - `01-gpu-amd.md` (job description + agent prompt)
  - `02-runtime.md`
  - `03-rust-backend.md`
  - `04-quantization.md`
  - `05-performance-qa.md`
  - `06-model-integration.md`
  - `07-tokenizer.md`
  - `08-server-api.md`
  - `09-multimodal.md`
  - `10-build-release.md`
  - `AGENTS.md` or `agent-prompts.md` with prompts and scopes.
But the user said "docs/experts.md" singular. They accepted the offer "document docs/experts.md, fiches de poste, ou prompts d'agents". We can create `docs/experts.md` as the main document and a `docs/experts/` folder for the fiches and prompts. Or keep it all in one file with sections. Given the volume, a folder is cleaner. But the user specifically mentioned `docs/experts.md`; we can create that as the index/main analysis, and link to `docs/experts/*.md` for the fiches and prompts. That satisfies both.

Let's check the docs structure: `docs/src/` is the mdBook source. `docs/experts.md` would be outside the book. The user asked for `docs/experts.md`, so create it at `docs/experts.md`. The existing `docs/` has `good-first-issues.md`, `roadmap.md` outside `src/`. So `docs/experts.md` fits.

Maybe also create `docs/experts/` with:
- `README.md` (index)
- `gpu-amd.md`
- `runtime.md`
- `rust-backend.md`
- `quantization.md`
- `performance-qa.md`
- `model-integration.md`
- `tokenizer.md`
- `server-api.md`
- `multimodal.md`
- `build-release.md`
- `agent-prompts.md`

Each fiche includes:
- Mission
- Contexte projet (files, constraints)
- Compétences (must-have, nice-to-have)
- Périmètre d'écriture (paths)
- Fichiers de contexte à lire
- Commandes de mesure
- Livrables 30/60/90
- KPI / gates
- Interactions
- Profil recherché / signaux
- Prompt d'agent (for the agent version)

That's a lot of files. We can be efficient by creating a template and filling each. But the user asked for all three deliverables, so it's worth it.

Need to ensure the document is accurate and consistent with the repo. Let's plan the content.

### Main `docs/experts.md`

Sections:
1. Objet et portée.
2. Diagnostic (updated with the one-backend-per-API principle).
3. Architecture cible (the correction).
4. Profils d'experts (summary table + links to fiches).
5. Composition et séquencement.
6. RACI.
7. Évaluation et signaux.
8. Commandes de mesure communes.
9. Prompts d'agents (link).
10. Ce qu'il ne faut pas faire.

### Architecture correction

We need to articulate the target:

**Principe** : un seul backend complet par API graphique.
- `kinetic-cuda` : tout CUDA — driver, kernels, layouts, executor, graphes, mémoire, capabilities.
- `kinetic-vulkan` : tout Vulkan — driver, kernels WGSL/SPIR-V, executor, graphes, mémoire, capabilities.
- `kinetic-cpu` : tout CPU — kernels SIMD, thread pool, executor.
- `kinetic-models` : aucune dépendance à CUDA/Vulkan ; décrit les architectures, blocs, tenseurs, spéculatif, warm start, shortlist, convoy ; appelle une interface backend.
- `kinetic-runtime` (nouveau, optionnel) : traits `Backend`, `Device`, `Graph`, `Kernel`, `Capabilities`, `Memory` ; planification générique ; pas de code spécifique à une API.
- `kinetic-hip`, `kinetic-metal`, `kinetic-sycl`, `kinetic-webgpu` : backends de compatibilité, chacun complet pour son API, ajoutés seulement quand une cible le justifie. L'utilisateur charge un seul backend à l'exécution selon sa carte.

**Règle** : pas de module `kinetic-models/src/cuda` ni `kinetic-models/src/vulkan` à terme. Le code d'exécution vit dans le crate backend. `kinetic-models` ne contient que la sémantique du modèle et les invariants d'exactitude.

**Chemin de migration incrémental** :
1. Introduire `kinetic-runtime` avec les traits et les types de capabilities (pas de réécriture).
2. Déplacer `kinetic-models/src/cuda` vers `kinetic-cuda/src/executor` (ou `model`), en gardant les tests.
3. Déplacer `kinetic-models/src/vulkan` vers `kinetic-vulkan/src/executor`.
4. Faire de `kinetic-models` un crate sans features `cuda`/`vulkan` ; il dépend de `kinetic-runtime` uniquement.
5. Adapter `kinetic-server` pour choisir un backend via une factory (`BackendKind::Cuda`, `Vulkan`, `Cpu`) et ne charger que celui-là.
6. Ajouter HIP/Metal/SYCL derrière le même trait, sans toucher `kinetic-models`.

**Bénéfices** :
- Une seule implémentation par API : pas de divergence entre `kinetic-models/src/cuda` et `kinetic-cuda`.
- Les fonctionnalités CUDA (flash attention, MoE, graphs, layer split) vivent dans `kinetic-cuda`, pas éparpillées.
- Les fonctionnalités Vulkan (WGSL, coopmat, async) vivent dans `kinetic-vulkan`.
- `kinetic-models` reste testable sans GPU, avec un backend de test.
- Ajouter un backend = un crate + une factory, pas une modification du cœur.

**Contrainte** : ne pas casser les performances. Le déplacement doit être mécanique d'abord, puis les optimisations. Les tests d'identité doivent rester verts à chaque étape.

**Note sur "les autres backends"** : HIP/Metal/SYCL sont des cibles de compatibilité. On fournit le socle (trait, capabilities, tests de conformance, modèle de build), pas une implémentation complète tant qu'il n'y a pas de machine/CI pour la valider. C'est exactement la règle du `PLAN.md` : pas de travail invérifiable.

This addresses the user's point directly.

### Expert profiles

We can reuse the previous analysis but refine:
- Add "Périmètre d'écriture" and "Fichiers de contexte" and "Commandes de mesure" to each.
- For the Rust/backend architect, the mission now explicitly includes the one-backend-per-API consolidation.

### Measurement commands

Need to provide concrete commands. From the repo:
- Build: `cargo build --release --locked`
- Tests: `cargo test --release --locked --workspace`
- Clippy: `cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings`
- Bench: `target/release/kinetic bench MODEL --device vulkan --prompt 512 -r 2` (and `--json`).
- llama.cpp: `llama-bench -dev Vulkan0 -p 512 -n 128 -r 2`.
- Scripts: `python scripts/bench_protocol.py MODEL --llama-bin path/to/llama-bench`; `python scripts/bench_vs_llamacpp.py model.gguf --llama-dir ...`; `python scripts/conformance.py model.gguf --kinetic target/release/kinetic --llama ... --device vulkan`; `python scripts/server_abuse.py -- ...`.
- GPU idle check: `rocm-smi` or `radeontop`? The PLAN says check GPU idle. For AMD, `rocm-smi --showuse` and `rocm-smi --showtemp`. For Vulkan, maybe `vulkaninfo` to verify. Use `rocm-smi` since ROCm 7.2 is installed. Also `nvidia-smi` for NVIDIA (not local).
- Memory watchdog: `scripts/` doesn't have a watchdog. The skill has one, but not in repo. We can propose `scripts/` additions. For now, `rocm-smi --showmemuse`.
- Exactness: `cargo test --release -p kinetic-vulkan` (per PLAN), `cargo test --release -p kinetic-cuda` (on NVIDIA), `cargo test --release -p kinetic-cpu`.
- Conformance: `python scripts/conformance.py`.
- Needle test: no script? We can propose adding one; the PLAN mentions it. We can give a placeholder command: `python scripts/needle.py` (to create). But don't invent existing scripts. We can say "à créer" in the deliverables.
- Same answer: `python scripts/same_answer_demo.py --model ... --kinetic target/release/kinetic`.
- Server abuse: `python scripts/server_abuse.py --port 8091`.
- Bench protocol: `python scripts/bench_protocol.py`.
- `kinetic doctor` for device inventory.
- `kinetic info` for model info.

Let's check `kinetic bench` options to be accurate. `docs/src/cli.md` has `bench` section. Let's read it quickly to provide correct commands.

**Tool: read**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic/docs/src/cli.md",
  "offset": 224,
  "limit": 90
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/src/cli.md, lines 224-313
224: ## bench
225: 
226: ```sh
227: kinetic bench model.gguf
228: kinetic bench model.gguf --device cpu -t 6 -r 5 --json
229: ```
230: 
231: | Option | Meaning |
232: |---|---|
233: | `-r`, `--repetitions <N>` | runs of each test; the best is reported [default: 3] |
234: | `-t`, `--threads <N>` | CPU threads [default: all] |
235: | `--device <auto\|cpu\|cuda>` | the device to measure [default: auto] |
236: | `--gpus <LIST>` | CUDA GPUs to spread the layers over, as for `run` |
237: | `--json` | print the results as one JSON object |
238: | `--compare [LLAMA_BENCH]` | also run llama.cpp on the same file and print a card with both; see [bench --compare](#bench---compare) |
239: | `--off <LIST>` | turn speed technologies off, as for `run`; bench measures RAD-1 (`edit256`), Shortlist and Torque |
240: 
241: Three tests, each the best of `-r` runs:
242: 
243: - `pp512`: a 512-token prompt, as in `llama-bench`;
244: - `tg128`: 128 generated tokens, as in `llama-bench`, with speculative
245:   decoding off so every token costs one pass over the weights;
246: - `edit256`: 256 tokens of an answer that rewrites a short Rust file with
247:   one name changed, with speculative decoding on, the way an editing request
248:   runs. Its output is compared with plain decoding.
249: 
250: ```text
251: | model | size | params | backend | threads | test | t/s |
252: | --- | ---: | ---: | --- | ---: | ---: | ---: |
253: | SmolLM2-135M-Instruct-f16 | 0.25 GiB | 0.13 B | CPU | 4 | pp512 | ... |
254: | SmolLM2-135M-Instruct-f16 | 0.25 GiB | 0.13 B | CPU | 4 | tg128 | ... |
255: | SmolLM2-135M-Instruct-f16 | 0.25 GiB | 0.13 B | CPU | 4 | edit256 | ... |
256: 
257: on the CPU, best of 3, load 0.53 s
258: speed technologies on: RAD-1, Shortlist; Warm Start, Zero Wait and Convoy take no part
259: edit: drafted output identical to plain decoding: yes
260: peak memory 0.31 GiB
261: ```
262: 
263: To see what one technology gives, run `bench` twice, with and without
264: `--off` naming it, and compare the rows: `--off rad1` runs `edit256`
265: without drafting, `--off shortlist` and `--off torque` change `tg128` and
266: `edit256`. The line under the table says which ran; Warm Start, Zero Wait
267: and Convoy need repeated, typed or concurrent requests, which `bench`
268: never sends, so it says they take no part.
269: 
270: If the drafted output ever differs from plain decoding, `bench` says `NO`
271: and exits with 1; please [report it](https://github.com/Veloresearch/kinetic/issues/new?template=bug.yml).
272: `--json` prints:
273: 
274: ```json
275: {
276:   "model": "SmolLM2-135M-Instruct-f16.gguf",
277:   "backend": "cpu",
278:   "device": "the CPU",
279:   "threads": 4,


---

## Assistant

_Thinking:_

Interesting: `--device <auto|cpu|cuda>` not vulkan. The docs might be outdated or the CLI only exposes cpu/cuda. The PLAN mentions Vulkan bench. Let's check the actual CLI source for bench device options. Also check `kinetic run` options. Let's grep.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -rn \"device.*vulkan\\|Vulkan\" crates/kinetic-cli/src --include=*.rs | head -40"
}
```

crates/kinetic-cli/src/main.rs:479:    Vulkan,
crates/kinetic-cli/src/main.rs:1062:        kinetic_server::VulkanBackend::new(model, context).map(|gpu| {
crates/kinetic-cli/src/fit.rs:35:    /// The first weight type Vulkan has no kernel for, so the model runs on
crates/kinetic-cli/src/fit.rs:94:            // CUDA reports a nominal figure; Vulkan has none, so it is measured.
crates/kinetic-cli/src/fit.rs:125:/// The GPU models run on: CUDA before Vulkan, then the one with the most
crates/kinetic-cli/src/fit.rs:279:                // a Vulkan device cannot upload a weight type it has no kernel for
crates/kinetic-cli/src/fit.rs:285:                    Some(ty) => format!(", Vulkan has no kernel for {ty}"),
crates/kinetic-cli/src/system/gpu.rs:1://! The GPUs the CUDA driver and the Vulkan loader report.
crates/kinetic-cli/src/system/gpu.rs:36:/// The most free memory of any CUDA GPU, without asking Vulkan.
crates/kinetic-cli/src/system/gpu.rs:41:/// Read bandwidth of the Vulkan GPU named `name`, measured by a kernel on
crates/kinetic-cli/src/system/gpu.rs:42:/// that device: the only bandwidth a Vulkan driver answers with. `None`
crates/kinetic-cli/src/system/gpu.rs:43:/// when Vulkan is unavailable or the measurement fails.
crates/kinetic-cli/src/system/gpu.rs:51:    let device = kinetic_vulkan::Device::new(&physical).ok()?;
crates/kinetic-cli/src/system/gpu.rs:168:    let devices = kinetic_vulkan::Instance::new()
crates/kinetic-cli/src/system/gpu.rs:181:                    "Vulkan" => format!(
crates/kinetic-cli/src/system/gpu.rs:182:                        "{} {}, Vulkan {major}.{minor}.{patch}",
crates/kinetic-cli/src/system/gpu.rs:203:    Err("this kinetic was built without Vulkan".into())
crates/kinetic-cli/src/tech.rs:41:    /// Whether it does anything on `device` (`Cpu`, `Cuda` or `Vulkan`).
crates/kinetic-cli/src/tech.rs:47:            Tech::Torque => matches!(device, Device::Cuda | Device::Vulkan),
crates/kinetic-cli/src/tech.rs:53:        !(self == Tech::Torque && device == Device::Vulkan)
crates/kinetic-cli/src/tech.rs:96:        let s = status(&[Tech::Torque], Device::Vulkan, &Tech::ALL);
crates/kinetic-cli/src/bench.rs:125:        Some(Plain::Vulkan(plain)) => {
crates/kinetic-cli/src/bench.rs:128:                let gpu = kinetic_server::VulkanBackend::new(&model, context)?;
crates/kinetic-cli/src/bench.rs:228:    Vulkan(Box<kinetic_server::VulkanBackend<'a>>),
crates/kinetic-cli/src/bench.rs:265:        Device::Vulkan => Some(
crates/kinetic-cli/src/bench.rs:266:            kinetic_server::VulkanBackend::plain(model, context)
crates/kinetic-cli/src/bench.rs:267:                .map(|b| Plain::Vulkan(Box::new(b))),
crates/kinetic-cli/src/fix.rs:12:        "this build has no CUDA support; use --device vulkan or --device cpu, or rebuild with --features cuda"
crates/kinetic-cli/src/fix.rs:14:        "this build has no Vulkan support; use --device cuda or --device cpu, or rebuild with --features vulkan"
crates/kinetic-cli/src/fix.rs:58:            ("this kinetic was built without CUDA", "--device vulkan"),
crates/kinetic-cli/src/fix.rs:59:            ("this kinetic was built without Vulkan", "--device cuda"),
crates/kinetic-cli/src/gpu.rs:1://! Which GPU a model runs on: CUDA first, then Vulkan, else the CPU.
crates/kinetic-cli/src/gpu.rs:7:/// What `--device vulkan` runs through: Vulkan, or on macOS Metal.
crates/kinetic-cli/src/gpu.rs:11:pub const VULKAN: &str = "Vulkan";
crates/kinetic-cli/src/gpu.rs:14:const ORDER: [(Device, &str); 2] = [(Device::Cuda, "CUDA"), (Device::Vulkan, VULKAN)];
crates/kinetic-cli/src/receipt.rs:27:        Device::Vulkan => ReceiptDevice::new(format!("vulkan-{name}"), name),
crates/kinetic-cli/src/receipt.rs:126:        Device::Vulkan
crates/kinetic-cli/src/compare.rs:71:        Device::Vulkan => "vulkan",


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -rn \"enum Device\\|DeviceArg\\|device:.*String\\|--device\" crates/kinetic-cli/src --include=*.rs | head -40"
}
```

crates/kinetic-cli/src/main.rs:471:enum Device {
crates/kinetic-cli/src/tech.rs:59:pub fn status(off: &[Tech], device: Device, among: &[Tech]) -> String {
crates/kinetic-cli/src/compare/card.rs:16:    pub kinetic_device: String,
crates/kinetic-cli/src/fix.rs:10:        "try a smaller model or quantization, a shorter context such as -c 4096, or --device cpu"
crates/kinetic-cli/src/fix.rs:12:        "this build has no CUDA support; use --device vulkan or --device cpu, or rebuild with --features cuda"
crates/kinetic-cli/src/fix.rs:14:        "this build has no Vulkan support; use --device cuda or --device cpu, or rebuild with --features vulkan"
crates/kinetic-cli/src/fix.rs:16:        "install or update the GPU driver, or run on the processor with --device cpu"
crates/kinetic-cli/src/fix.rs:58:            ("this kinetic was built without CUDA", "--device vulkan"),
crates/kinetic-cli/src/fix.rs:59:            ("this kinetic was built without Vulkan", "--device cuda"),
crates/kinetic-cli/src/gpu.rs:7:/// What `--device vulkan` runs through: Vulkan, or on macOS Metal.
crates/kinetic-cli/src/llama_flags.rs:12:    ("-dev", "--device"),
crates/kinetic-cli/src/llama_flags.rs:77:        "Kinetic splits a model by layers only; choose the device with --device, and spread the layers over several CUDA GPUs with --gpus 0,1",
crates/kinetic-cli/src/llama_flags.rs:174:                "0" => out.push("--device=cpu".to_owned()),
crates/kinetic-cli/src/llama_flags.rs:293:            args(&["--device=cpu", "--device", "none"])
crates/kinetic-cli/src/compare.rs:76:        .args(["--json", "--device", device])


---

## Assistant

_Thinking:_

`--device vulkan` exists, docs cli.md is outdated. Fine. Use `--device vulkan`.

Now, let's look at the session from line 12131 onward where the architecture doc was produced, to understand what was already created. Also line 14206 "Fichiers créés". Let's read the relevant sections to see if the docs/experts.md was actually created and what the session says.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '12131,12300p' session-ad321db8-5ba0-40bf-b50d-4be596739cef.md"
}
```

## Assistant

_Thinking:_

Let's make targeted edits.

First, section 0 rewrite. I'll use edit with the exact text. Need to ensure no accidental issues.

Proposed section 0:

```
## 0. Décision structurante : un seul backend complet par API

Le dépôt a aujourd'hui **deux couches par API GPU** :
...
La cible est différente et plus simple :

> **Un seul backend complet par API GPU.** Tout CUDA vit dans `kinetic-cuda` :
> driver, noyaux, layouts, graphes, mémoire, exécution du modèle, capacités.
> Tout Vulkan vit dans `kinetic-vulkan`. Le CPU reste le chemin de référence :
> ses noyaux dans `kinetic-cpu`, son exécution générique dans `kinetic-models`.
> `kinetic-models` ne contient plus de code CUDA/Vulkan : il décrit les
> architectures, les blocs et les invariants, et expose une interface
> d'exécution **au niveau modèle** (charger, préfill, décoder, état de
> séquence, capacités), pas au niveau noyau.

Les autres API (HIP, Metal, SYCL, WebGPU) sont des **backends de
compatibilité** : chacun est complet pour son API, mais n'est ajouté que
lorsqu'une machine et une CI peuvent le valider. L'utilisateur n'en charge
qu'un seul à l'exécution, selon sa carte. On fournit le socle (trait,
capacités, tests de conformance, modèle de build), pas une implémentation
complète et invérifiable de chaque API.

### 0.1 État actuel vs cible

| Aujourd'hui | Cible |
|---|---|
| CUDA : `kinetic-cuda` + `kinetic-models/src/cuda` | `kinetic-cuda` contient tout CUDA |
| Vulkan : `kinetic-vulkan` + `kinetic-models/src/vulkan` | `kinetic-vulkan` contient tout Vulkan |
| `kinetic-models` a des features `cuda` / `vulkan` et des modules backend | `kinetic-models` sans feature backend, expose un trait `Runner` |
| `kinetic-server` a `cuda.rs`, `vulkan.rs`, `cpu.rs` | `kinetic-server` choisit un backend via une factory |
| Ajouter HIP = nouveau crate + nouveau module dans `kinetic-models` | Ajouter HIP = nouveau crate qui implémente `Runner` |
| Les tests modèle connaissent les backends | Les tests modèle utilisent le runner CPU ou un mock |

### 0.2 Où vivent les traits

Deux couches, pas une :

- **`kinetic-runtime`** (à créer) : découverte des devices, registre/factory
  de backends, capacités, budget mémoire, ordonnanceur, pool KV, métriques.
  Il ne connaît **pas** les types de modèle.
- **`kinetic-models`** : architectures, blocs, modèle générique, trait
  `Runner` au niveau modèle, runner CPU de référence.
- **`kinetic-cuda` / `kinetic-vulkan` / `kinetic-hip`** : implémentent
  `Runner` et possèdent leurs noyaux, graphes et politiques de mémoire.

Cette séparation évite un cycle de dépendances et évite de transformer
`kinetic-runtime` en compilateur de graphes. Le trait `Runner` reste petit :
charger, préfill, décoder, état de séquence, capacités. Chaque backend garde
la liberté de fusionner, grapher et ordonnancer ses noyaux comme il veut.

### 0.3 Migration incrémentale

1. Créer `kinetic-runtime` avec les types de devices, capacités, budget
   mémoire et factory. Aucun changement de comportement.
2. Extraire dans `kinetic-models` un trait `Runner` au niveau modèle et
   exposer les descripteurs nécessaires aux backends. Le runner CPU existant
   devient l'implémentation de référence.
3. Déplacer `kinetic-models/src/cuda` vers `kinetic-cuda/src/runner` et
   `kinetic-models/src/vulkan` vers `kinetic-vulkan/src/runner`. Ce n'est pas
   un simple `git mv` : il faut passer par le trait et les descripteurs
   exposés, et conserver les tests à chaque étape.
4. Retirer les features `cuda` / `vulkan` de `kinetic-models` ; il ne dépend
   plus des crates backend.
5. Ajouter une factory de backend dans `kinetic-server` / `kinetic-cli` :
   `auto | cpu | cuda | vulkan | hip`, un seul backend chargé.
6. Ajouter un backend de test (CPU ou mock) pour tester `kinetic-models`
   sans GPU.
7. Seulement ensuite, optimiser. Le refactor d'abord ; la performance vient
   après, avec les mêmes gates d'exactitude.

### 0.4 Contrat de compatibilité d'un backend

Un backend n'est « fourni » que s'il respecte ce contrat :

- **Découverte** : `probe()` liste les devices et leurs capacités réelles.
- **Capacités** : types de poids, formats KV, batch max, async, multi-device,
  mémoire disponible. Le serveur refuse clairement ce qui n'est pas supporté.
- **Conformance** : mêmes tokens que le runner CPU sur le corpus de contrôle,
  aux tolérances près, et mêmes gates d'identité.
- **Mesure** : `kinetic bench` et le protocole A/B fonctionnent sur ce
  backend, avec un seuil de régression.
- **Isolement** : un seul backend est chargé à l'exécution ; les autres ne
  coûtent ni mémoire ni temps de démarrage.
- **CI** : au moins un runner ou une machine de référence valide le backend
  avant qu'il ne soit activé par défaut.

Tant qu'un backend ne passe pas ce contrat, il reste **opt-in** et documenté
comme expérimental. C'est ainsi qu'on fournit les bases de HIP/Metal/SYCL
sans promettre une parité invérifiable.

### 0.5 Ce que cela change pour les experts
...
### 0.6 Non-objectifs
...
```

This is a solid improvement. But is `Runner` trait in `kinetic-models` the right place? Let's validate with actual code. `kinetic-models` has `Forward`, `Model`, `State`, `speculative::Target`, `warm::Runner`. There is already a `warm::Runner` trait. Naming `Runner` could conflict. Let's inspect `warm::Runner` and `Forward`.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n 'pub trait Runner' -A 60 crates/kinetic-models/src/warm.rs | head -90 && echo '--- Forward ---' && grep -n 'pub trait Forward' -A 60 crates/kinetic-models/src/model.rs | head -90"
}
```

grep: crates/kinetic-models/src/warm.rs: Aucun fichier ou dossier de ce nom
--- Forward ---


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -rn 'trait Runner\\|trait Target\\|trait Forward' crates/kinetic-models/src | head -20"
}
```

crates/kinetic-models/src/warm/mod.rs:42:pub trait Runner {
crates/kinetic-models/src/speculative/target.rs:6:pub trait Target {


---

## Assistant

_Thinking:_

`warm::Runner` exists, and `Forward` is likely a struct/trait elsewhere. Let's inspect.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '1,110p' crates/kinetic-models/src/warm/mod.rs && echo '--- Forward ---' && grep -rn 'pub trait Forward\\|pub struct Forward\\|enum Forward' crates/kinetic-models/src | head -20"
}
```

//! Warm starts: prompts that begin like an earlier one resume from that
//! one's exact state and run only their new tokens.
//!
//! Agents and chat clients resend the same system prompt, tools and history
//! with every request. [`Warm`] keeps the states those prompts left behind,
//! in memory under a byte budget and optionally on disk, and starts each
//! new prompt from the longest cached prefix it can return to exactly.
//! Attention caches cut back to any length; recurrent layers only to the
//! marks taken every few hundred tokens, where prompts part and where they
//! resume.
//!
//! A restored state is bit for bit the one its run left behind, so a warm
//! prefill gives exactly the logits of a cold one on models whose results
//! do not depend on how a prompt is split into batches, as the speculative
//! decoder already requires of every block. Otherwise it matches a cold run
//! with the same batch boundaries, and a single batch up to rounding;
//! [`Warm::aligned`] makes those boundaries the same by construction.



**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '14150,14340p' session-ad321db8-5ba0-40bf-b50d-4be596739cef.md"
}
```

docs/experts.md:72:  `reuses`, `run_ahead`, `settle_some`) au-dessus de `ModelRunner`.
docs/experts.md:75:`kinetic-runtime` en compilateur de graphes. Le trait `ModelRunner` reste
docs/experts.md:84:2. Extraire dans `kinetic-models` un trait `ModelRunner` au niveau modèle et
docs/experts.md:214:| Architecte Rust systèmes/backend | **P0/P1** | Rare | Un backend complet par API, dette, unsafe | workspace, `kinetic-runtime` |
docs/experts.md:282:`Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>` : un pool de backends, une
docs/experts.md:286:`Connection: close`.
docs/experts/fiches-de-poste.md:34:- La cible architecturale est **un seul backend complet par API** : tout
docs/experts/fiches-de-poste.md:83:- `crates/kinetic-models/src/runner/**` (à créer : trait `ModelRunner` et
docs/experts/fiches-de-poste.md:165:- `Engine.free` est un `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>` : une
docs/experts/fiches-de-poste.md:170:- `MAX_CONNECTIONS = 64`, `Connection: close`, pas de keep-alive.
docs/experts/fiches-de-poste.md:300:- Extraire `ModelRunner` dans `kinetic-models` : charger, préfill, décoder,
docs/experts/fiches-de-poste.md:303:  via `ModelRunner` ; ce n'est pas un simple `git mv`.
docs/experts/fiches-de-poste.md:309:- Documenter la règle « un backend complet par API » et son contrat de
docs/experts/prompts-agents.md:78:- docs/experts.md, section 0 (un seul backend complet par API).
docs/experts/prompts-agents.md:141:- docs/experts.md, section 0 (un seul backend complet par API).
docs/experts/prompts-agents.md:199:- docs/experts.md, section 0 (un seul backend complet par API).
docs/experts/prompts-agents.md:210:- crates/kinetic-models/src/runner/** (à créer : trait ModelRunner et runner CPU)
docs/experts/prompts-agents.md:225:   et du trait ModelRunner (charger, préfill, décoder, état).
docs/experts/prompts-agents.md:226:3. Extraire ModelRunner dans kinetic-models, avec le runner CPU comme


---

## Assistant

_Thinking:_

Good.

Now, I need to produce the final answer in French. It should be concise but cover:
1. What was created, with file paths and line counts.
2. The key architectural decision integrated: one complete backend per API, with details:
   - CUDA in `kinetic-cuda`, Vulkan in `kinetic-vulkan`, CPU reference in `kinetic-models`/`kinetic-cpu`.
   - `kinetic-runtime` for devices/capabilities/factory/scheduler/KV/metrics; `ModelRunner` in `kinetic-models`; `Backend` in server.
   - Migration steps, not a simple `git mv`.
   - Other backends (HIP/SYCL/WebGPU) as compatibility, opt-in until CI/conformance; Metal is a Vulkan path (MoltenVK).
   - User loads one backend at runtime.
3. What the docs contain (profiles, job descriptions, agent prompts).
4. Note no code changed, `PLAN.md` untouched.
5. Offer next steps: implement step 1 (skeleton `kinetic-runtime` + `ModelRunner`), write ADR, translate to English, create tracking issues.

Need to be careful not to overclaim that the docs are "published" in mdBook; note they are internal, outside `docs/src`, as stated. Maybe say "I placed them as internal docs; if you want them in the mdBook, I can move them under docs/src and add SUMMARY entries."

Let's also mention the corrected facts:
- `Engine.free` actual type.
- Metal via Vulkan.
- `conformance.py` only cpu/cuda.
Maybe not necessary, but useful to show rigor.

The final answer should be in French. Keep it structured with headings and bullet points. Desired oververbosity 5: moderate detail. Since the work is large, a summary table is appropriate.

Let's draft:

```
J'ai créé les trois documents et intégré ta règle « un backend complet par API » comme décision structurante.

## Fichiers créés
| Fichier | Contenu |
...
`docs/experts.md` : analyse, diagnostic chiffré, 10 profils, composition d'équipe, RACI, commandes de mesure, recommandation.
`docs/experts/fiches-de-poste.md` : 10 fiches ...
`docs/experts/prompts-agents.md` : 10 prompts ...

## Ce que change ta remarque sur les backends
...
- Aujourd'hui CUDA est éclaté entre kinetic-cuda et kinetic-models/src/cuda; Vulkan entre kinetic-vulkan et kinetic-models/src/vulkan.
- Cible: un seul crate complet par API GPU. `kinetic-cuda` = driver + noyaux + graphes + mémoire + exécution + capacités. Idem Vulkan.
- `kinetic-models` ne contient plus de code CUDA/Vulkan; il expose un trait `ModelRunner` au niveau modèle.
- `kinetic-runtime` (à créer) = devices, capacités, factory, scheduler, pool KV, métriques; pas de types de modèle.
- `kinetic-server` garde `Backend` (requête, reuses, run_ahead, settle_some).
- Les autres backends (HIP, SYCL, WebGPU) sont des backends de compatibilité, opt-in, avec un contrat: probe, capabilities, conformance, mesure, isolement, CI. Metal reste un chemin Vulkan (MoltenVK).
- Un seul backend chargé à l'exécution.
- Migration en 7 étapes; le déplacement des exécuteurs n'est pas un simple git mv, il faut extraire `ModelRunner` et exposer les descripteurs.

## Points corrigés
- `Engine.free` = `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>`.
- `conformance.py` ne connaît que cpu/cuda.
- Les docs sont internes, hors mdBook.

## État
Aucun code modifié; seuls les 3 docs ajoutés. `PLAN.md` inchangé.

## Prochaines étapes possibles
1. Squelette `kinetic-runtime` + trait `ModelRunner` (premier pas concret).
2. ADR `docs/adr/...` ou page `docs/src/backend-architecture.md`.
3. Traduction anglaise / intégration mdBook.
4. Issues GitHub à partir des livrables 30/60/90.
```

Need to mention "j'accepte cela" — yes, done.

Could also mention the recommended team: 4-5 people minimum, GPU AMD first, runtime, QA, numerical, Rust backend. The docs contain it. Maybe one sentence.

Let's ensure the final answer doesn't claim the docs are in `docs/src`. Say "docs internes". Good.

Also, maybe mention the line counts: 699, 1048, 657. Not necessary but shows substance. Could include.

One more thing: The user might expect the docs to be in English because the repo is English. We can offer translation. Mention "Les documents sont en français; je peux les traduire en anglais si tu veux les publier dans le mdBook." Good.

Let's also mention that I did not modify `PLAN.md`; `git status` shows only untracked docs. Good.

Now, before final, let's double-check that no code was accidentally modified by the earlier `PLAN.md` mtime. `git status` shows only docs. Good.

Let's consider whether the docs' `docs/experts.md` link to `docs/experts/...` works from the repo root or from the file. The link `[docs/experts/fiches-de-poste.md](experts/fiches-de-poste.md)` from `docs/experts.md` resolves to `docs/experts/fiches-de-poste.md`. Good. The fiches link `../experts.md` resolves to `docs/experts.md`. Good. The prompts link `../experts.md` and `fiches-de-poste.md`; from `docs/experts/prompts-agents.md`, `fiches-de-poste.md` resolves to same dir. Good.

Potential issue: In `docs/experts.md` section 7, the label is `docs/experts/fiches-de-poste.md` but the link is `experts/fiches-de-poste.md`; fine.

Potential issue: `docs/experts.md` says "docs/src/contributing.md" etc. Those exist. Good.

Now, final answer in French. Keep it clear and concise. Use a table for files. Mention key architectural decision. End with next steps and a question.

Let's write the final answer.

J’ai créé les trois documents et intégré ta remarque sur les backends comme **décision structurante** dans `docs/experts.md`.

## Fichiers créés

| Fichier | Contenu |
|---|---|
| `docs/experts.md` | Diagnostic chiffré, 10 profils d’experts, priorités P0/P1/P2, composition d’équipe, RACI, commandes de mesure, recommandation finale |
| `docs/experts/fiches-de-poste.md` | 10 fiches de poste détaillées : mission, contexte, compétences, périmètre d’écriture, fichiers de contexte, commandes, livrables 30/60/90, critères de succès, signaux d’alerte |
| `docs/experts/prompts-agents.md` | 10 prompts d’agents spécialisés avec contrat commun, garde-fous, boucle de travail, commandes et format de rapport |

Les documents sont en français, en dehors du mdBook publié (`docs/src/`). Je peux les traduire en anglais et les intégrer au livre si tu veux les publier.

## Ta règle « un seul backend par API »

Elle est maintenant le point de départ de `docs/experts.md`, avec une distinction importante :

- **CUDA** : tout dans `kinetic-cuda` — driver, noyaux, layouts, graphes, mémoire, exécution du modèle, capacités.
- **Vulkan** : tout dans `kinetic-vulkan`, selon le même principe.
- **CPU** : reste le chemin de référence ; ses noyaux dans `kinetic-cpu`, son exécution générique dans `kinetic-models`.
- **`kinetic-models`** : plus de code CUDA/Vulkan ; il expose un trait `ModelRunner` au niveau modèle (charger, préfill, décoder, état, capacités).
- **`kinetic-runtime`** (à créer) : devices, capacités, budget mémoire, factory de backends, ordonnanceur, pool KV, métriques. Il ne connaît pas les types de modèle.
- **`kinetic-server`** : garde le trait `Backend` actuel (requête, `reuses`, `run_ahead`, `settle_some`) au-dessus de `ModelRunner`.
- **HIP, SYCL, WebGPU** : backends de compatibilité, opt-in, avec un contrat précis — découverte, capacités, conformance, mesure, isolement, CI. Metal reste un chemin Vulkan via MoltenVK, pas un backend séparé.
- **Un seul backend chargé à l’exécution**, selon la carte.

La migration est décrite en 7 étapes. J’ai explicitement corrigé l’idée que le déplacement des exécuteurs serait un simple `git mv` : il faut d’abord extraire `ModelRunner` et exposer les descripteurs de modèle nécessaires, sinon `kinetic-cuda` et `kinetic-vulkan` ne peuvent pas lire les structures internes de `kinetic-models`.

## Corrections factuelles intégrées

- `Engine.free` est bien `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>`, pas `Mutex<Vec<Box<dyn Backend>>>`.
- `conformance.py` ne connaît aujourd’hui que `cpu` et `cuda` ; l’extension Vulkan est un livrable QA.
- Les chemins `ollama` sont des répertoires, pas des fichiers.
- Les docs sont marqués comme internes, hors du mdBook.

## État du dépôt

Aucun code modifié. `git status` ne montre que les nouveaux documents :

```text
?? docs/experts.md
?? docs/experts/
```

`PLAN.md` est inchangé.

## Prochaines étapes possibles

1. **Commencer le refactor** : créer le squelette `kinetic-runtime` + le trait `ModelRunner`, sans changer le comportement.
2. **Écrire un ADR** : `docs/adr/0001-one-backend-per-api.md` pour figer la décision.
3. **Créer les issues GitHub** à partir des livrables 30/60/90 des fiches.
4. **Traduire et publier** les docs dans le mdBook.

Si tu veux, je peux enchaîner sur le point 1 : une première PR minimale avec `kinetic-runtime` et `ModelRunner`, tests verts, sans toucher aux noyaux.

---

## User

tu as analyser ce qui se passe du coté du cache kv et les améliorations potentielles qu'il faudrait faire à ce niveau la et quel type d'expert devrait mener l'analyse profonde de l'existant et des solutions à apporter ? 

---

## Assistant

_Thinking:_

The user asks: "tu as analysé ce qui se passe du côté du cache KV et les améliorations potentielles qu'il faudrait faire à ce niveau-là et quel type d'expert devrait mener l'analyse profonde de l'existant et des solutions à apporter ?"

Translation: "Did you analyze what happens with the KV cache and the potential improvements that should be made at that level, and what type of expert should lead the deep analysis of the existing system and the solutions to bring?"

They want a deeper analysis of KV cache in the project, potential improvements, and the expert profile to lead it. We have some context: KV cache f16 only, contiguous per sequence, no paged KV, no quantization, rollback/journal for speculative decoding, warm cache, per-layer, convolution rings, delta-rule layers. Need to inspect the code more deeply to give a precise answer. We should analyze:
- `crates/kinetic-models/src/blocks/attention.rs` (KvCache struct, store, rollback, lend/borrow, windows, etc.)
- `crates/kinetic-vulkan/src/attention.rs` (KvCache f16, slots, ring?)
- `crates/kinetic-cuda` KV cache (KvType, maybe f16, q8?)
- `crates/kinetic-server` sequence, engine, slots, concurrency.
- `warm` cache (host-side prefix cache) and `.ktc`? Warm start.
- Speculative rollback, Convoy, per-layer.
- Memory budget, `memory.rs`, `per_layer.rs`.


---

## Assistant

_Thinking:_

The session file is a transcript of the previous session. The user now asks us to analyze `./docs/experts/` and the session file to see if all prompts for experts and missions were created. So we need to check completeness: all expert profiles mentioned in the session, all prompts, all missions, consistency.

We've read the two docs. There are 10 fiches and 10 prompts. But the session had earlier iterations with 11 or 12 profiles, then settled on 10. Need to check if any expert/mission from the session is missing from the final docs. The session's final analysis at lines 7357-7769 lists 10 profiles. The final `docs/experts.md` likely also 10. The user asks specifically "si tous les prompts pour les experts et missions ont bien été créés" — check that all prompts for experts and missions were created.

Let's analyze:
- Fiches: 10 (GPU AMD, runtime, Rust backend, quantization, performance/QA, model integration, tokenizers, server/API, multimodal, build/release).
- Prompts: 10, matching.
- Session mentioned an 11th "Technical writer / developer experience (optional)" and "Security engineer?" in an earlier draft. The final list has 10, no technical writer or security. Is that a gap? The final `docs/experts.md` might mention cross-cutting security. Let's check. The session at lines 6185 "Cross-cutting: sécurité" and 6453 profile 11 details. The final 10 profiles omit security and tech writer. The user asks if all prompts were created. We should flag missing: no dedicated prompt for security, no technical writer, no HIP-specific expert (HIP is folded into GPU AMD expert), no "model integration" split (architectures/tokenizers separate), no "server/API" split. Need to see if the final docs intentionally scoped 10.

Also check the "missions" — each fiche has a mission, each prompt has a mission. Are all missions from the session covered? The session's final "Composition d'équipe" lists 10 profiles. The docs have 10. But the earlier session had "Expert 11 — Build/release & bindings" and "Expert 10 — mesure/qualification". The final docs reorder: 5 performance, 6 model integration, 7 tokenizers, 8 server, 9 multimodal, 10 build. All covered.

Potential gaps:
1. **Security** — mentioned as cross-cutting, no prompt. If the user wants "all prompts", flag it.
2. **Technical writer / DevRel** — optional, not created. Flag as intentionally omitted.
3. **HIP/ROCm** — folded into Agent 1; no separate prompt. The session at line 5347 had "2. HIP/ROCm backend engineer (top priority locally, second wave)" then final merged into GPU AMD. The fiche 1 covers HIP. So not missing, but if the user expected a separate HIP expert, note it.
4. **KV cache deep-dive** — the session's last user question asks about KV cache analysis and which expert should lead it. The final docs do not have a dedicated KV cache expert or a dedicated mission section. The runtime architect fiche covers KV pool, paged KV, but there is no dedicated "KV cache" analysis/mission. The user's current question may be checking whether the KV cache analysis was included. We should highlight that the session ended with a KV cache question and the docs only partially cover it (runtime architect responsibilities: paged KV pool, quantization KV in GPU/quant experts; no dedicated deep-dive document/prompt). That's a significant gap.

Let's inspect the session around the KV cache question and answer to see if it produced a detailed analysis. Lines 21150 onwards mention runtime architect, Rust backend, quantization. Lines 21307 "1. La règle backend est bien en place... 2. Ce que fait le cache KV aujourd'hui... 3. Améliorations... 4. Quel expert doit mener l'analyse... 5. Proposition". Let's read that section.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '21120,21479p' session-ad321db8-5ba0-40bf-b50d-4be596739cef.md"
}
```

crates/kinetic-models/src/convoy.rs:324:        let convoy = Convoy::<usize, usize>::default();
crates/kinetic-models/src/model.rs:7:use crate::convoy::{Convoy, Seat};
crates/kinetic-models/src/model.rs:144:    pub(crate) convoy: Convoy<Lane, (State, Forward)>,


---

## Assistant

_Thinking:_

Convoy is only used by CUDA model, not Vulkan/CPU? `model.rs` has convoy field for CPU model; `cuda/mod.rs` has convoy. So CPU and CUDA can share passes across sequences; Vulkan doesn't. But the KV is per sequence; Convoy batches the work of multiple sequences into one pass over weights, not one KV. It helps throughput, not KV paging. This is relevant: "Convoy mutualise les passes, mais ne partage pas le KV et ne planifie pas token-level." The docs already say that. Good.

Now, let's think about the exact recommendation for a KV expert. The existing docs profile 2 is "Architecte runtime d'inférence" P0, very rare, mission "Ordonnanceur, KV, concurrence, TTFT, p99", scope `kinetic-runtime`, `kinetic-server`. The user asks "quel type d'expert devrait mener l'analyse profonde de l'existant et des solutions à apporter ?" We can say: this profile, but with a KV specialization. Maybe propose a new profile "Ingénieur KV cache et hiérarchie mémoire" as a specialization of profile 2, or make it a separate fiche. The docs already have 10 profiles; adding an 11th might be overkill. Better: refine profile 2's mission and skills to explicitly include KV cache architecture, paged attention, prefix sharing, offload, quantization. If the user wants a dedicated hire, create a "KV cache lead" variant.

Let me check `docs/experts.md` profile 2 details to reference accurately. Read lines 260-330 maybe.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '258,335p' docs/experts.md"
}
```


**KPI** : pp512/pp2048 vs llama Vulkan, tg128 sans régression, gates
bit-exact, GB/s et TFLOPS atteints, temps de démarrage.

### 2.3 P0 — Architecte runtime d'inférence

**Mission** : faire passer le serveur de « N backends indépendants » à un
ordonnanceur token-level qui tient la concurrence, le TTFT et le p99, sans
casser RAD-1, Warm Start ni Convoy.

**Compétences clés** :

- Continuous batching, paged KV, prefix caching, chunked prefill, admission
  control, fairness, priorité, annulation, backpressure.
- Décodage spéculatif : ordonnancement draft/verify, vérification d'arbre,
  rollback, réutilisation de KV.
- Concurrence Rust : threads/async, `Mutex`/`Condvar`, channels, atomics,
  `Send`/`Sync`, scoped threads, panic safety.
- Mémoire : dimensionnement du pool KV (N × contexte), fragmentation/défrag,
  mémoire unifiée, pinned host, paging des experts MoE, watchdog OOM.
- Graph capture/replay, multi-stream, recouvrement copies/calcul.
- Observabilité : TTFT, tok/s, p50/p99, cache hit, profondeur de file.

**Pourquoi ici** : `Engine.free` est un
`Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>` : un pool de backends, une
requête par backend ; `PREFILL_CHUNK = 256` existe mais le chunking est par
requête ; `Convoy` mutualise des passes mais ne planifie pas ; le KV est
contigu par séquence, non paginé ; `MAX_CONNECTIONS = 64` ; en-tête
`Connection: close`.

**Livrables 90 jours** :

1. Design doc + prototype d'ordonnanceur token-level.
2. Pool KV N × contexte ; chunked prefill ; annulation ; métriques par token.
3. Benchmarks c4/c16/c64 avec p99 et TTFT.
4. Préservation de l'exactitude et des invariants Warm Start/Convoy.

**KPI** : débit agrégé, p99, TTFT, cache hit, zéro régression d'exactitude.

### 2.4 P0/P1 — Architecte Rust systèmes/backend

**Mission** : réduire la duplication CUDA/Vulkan, faire de chaque API un
backend complet et préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.

**Compétences clés** :

- Rust avancé : lifetimes, traits, génériques vs `dyn`, zéro-copie,
  soundness `unsafe`, `Send`/`Sync`, `Arc`/`Rc`, mmap, allocators, layouts
  cache-friendly, monomorphisation.
- Architecture : ports/adapters, abstraction backend, graphe/dispatch,
  feature flags, compilation conditionnelle, FFI.
- Performance hôte : `perf`, flamegraph, heaptrack, criterion, LLVM
  optimization remarks, LTO/PGO/BOLT, démarrage.
- Build : workspaces Cargo, cross-compilation, toolchains CUDA/HIP, CI.
- Tests : unitaires, intégration, property-based, fuzz, tests déterministes.

**Pourquoi ici** : `kinetic-models/src/cuda` ≈ 4,5 k LOC vs `vulkan` ≈ 2,2 k ;
`kinetic-cuda` ≈ 8,5 k vs `kinetic-vulkan` ≈ 5,9 k ; ajouter HIP va dupliquer
encore. Les bons seams existent déjà (`Family`, `Target`, `Backend`,
`Projector`) ; il faut les préserver. La surface `unsafe` est concentrée aux
frontières SIMD/GPU/FFI ; un audit de soundness est nécessaire.

**Livrables 90 jours** :

1. Cartographie de la duplication ; proposition d'une couche partagée
   minimale (`kinetic-runtime`).
2. Refactor d'un chemin comme preuve (FFN/attention) sans changer le
   comportement.
3. Audit `unsafe` avec commentaires `SAFETY`.
4. Amélioration du démarrage et du temps de build.
5. API prête pour 3+ backends.

**KPI** : LOC dupliquées, temps de build, démarrage, overhead hôte par token,
nombre d'`unsafe` justifiés.

### 2.5 P0/P1 — Ingénieur quantification et exactitude numérique

**Mission** : étendre la quantification et garantir que chaque gain de


---

## Assistant

_Thinking:_

The docs already have the right profile. We can say: "C'est exactement le profil 2.3, mais l'analyse KV mérite d'être son premier chantier, avec un mandat explicite." Maybe the user wants a specific expert type, not just the profile number. We'll explain.

Now, let's consider adding a dedicated KV section to `docs/experts.md`. The user might appreciate it. The docs currently mention KV in several places but no deep analysis. We could add `docs/experts/kv-cache.md` with the analysis and a fiche. But the user asked a question, not "write a doc". However, the previous pattern: user asks analysis, assistant creates docs. The user then accepted docs. For KV, they ask "tu as analysé... et quel type d'expert..." — maybe they expect an answer, and possibly a doc. To be safe, I can answer in chat and ask if they want it formalized. But the instructions favor action; maybe create the doc proactively? Let's weigh:
- Creating a new doc is low risk (new file, no code).
- It would be consistent with the project's documentation style.
- But the user might not want more files; they might want a direct answer.
- The question is explicitly analytical; a chat answer is sufficient. The assistant can offer to add it.

Given the last user message is about the backend rule, adding a KV doc might be off-topic. I'll answer both in chat, and offer to formalize. No file creation unless asked.

Wait, the system says "You are an AI agent running in OpenCode... accomplish their goals using tools." The user asked a question; answering is enough. No need to edit.

Let me also verify the "q8_0 doubles context" claim: q8_0 is 34 bytes per 32 values = 1.0625 bytes/value; f16 = 2 bytes/value; ratio 1.882. So "about twice" is fine. For a 16 GiB card, q8_0 saves ~47% of KV, not exactly 50%. The docs say "about twice"; we can say "×1,88, ~2× en pratique".

Let's compute an example with the 27B model: llama-kinetic says 0.125 MiB/token f16. At 32k: 4096 MiB = 4 GiB. q8_0: 4 GiB × 34/64 = 2.125 GiB. q4_0 (18 bytes per 32 values? q4_0 block = 2 bytes scale + 16 bytes quants = 18 bytes/32 values = 0.5625 bytes/value; ratio vs f16 = 0.28125; 4 GiB × 0.28125 = 1.125 GiB). Good.

But is 0.125 MiB/token accurate for the 27B? It implies 131072 bytes/token. For f16, 65536 values/token; e.g., 64 layers × 2 × 8 heads × 128 dim = 131072 values? That's 64*2*8*128 = 131072 values, ×2 bytes = 262144 bytes = 0.25 MiB. Hmm, 0.125 MiB/token = 65536 values; e.g., 32 layers × 2 × 8 × 128 = 65536. For a 27B model with 64 layers, maybe n_head_kv=4, head_dim=128: 64*2*4*128 = 65536. Possible. We'll cite the doc's figure as an order of magnitude, not derive.

Actually, let's avoid specific model math that might be wrong; use the doc's "≈0.125 MiB/token, 1 GiB at 8k" and extrapolate: 32k ≈ 4 GiB per sequence; 4 slots ≈ 16 GiB. This is consistent. State "ordre de grandeur".

Now, the key architectural insight: **static full-context reservation is the root cause of poor KV memory efficiency and concurrency**. The code reserves `context` per layer per slot at load; `default_slots` max 4; Vulkan 1 slot. A 32k context on a 16 GiB card can consume the entire GPU with 4 slots. Paged/unified KV is the fix. This is the central recommendation.

Second insight: **Vulkan is the local priority, but it lacks q8_0 and multi-slot; CUDA has q8_0 but no paging**. So the KV roadmap should be backend-agnostic at the design level, with `kinetic-runtime` owning the KV pool/allocator and each backend implementing a `KvStorage`/attention interface. This ties into the backend consolidation rule: one complete backend per API, but the KV policy/allocator should be shared in `kinetic-runtime`, not duplicated. Good point: the KV allocator is runtime-level, not backend-level; backends implement storage/attention kernels. This aligns with the architecture decision.

Third insight: **shared KV (Gemma 4) is a concrete memory win**: CUDA copies the lender's cache per borrowing layer; zero-copy/reference would save `shared_layers × context × KV` memory. Vulkan doesn't implement sharing at all (it re-projects and stores duplicate KV). This is a good P1.

Fourth: **Warm Start is a strength but not a substitute for prefix caching**: it stores full states, exact, 1 GiB RAM/2 GiB disk; it doesn't share blocks across concurrent requests or partial prefixes. A block-level prefix cache would complement it. But be careful: Warm Start's exactness is a differentiator; don't replace it. Use block cache for cross-request sharing, Warm Start for exact resume.

Fifth: **rollback/speculative decoding** constrains KV: CPU journal copies old K/V; GPU requires ≤MAX_BATCH between checkpoint and rollback; paged KV would make rollback cheaper (truncate block table) and allow larger speculative windows. This is a nice synergy.

Sixth: **metrics**: no KV-specific metrics. Add `kv_bytes_allocated`, `kv_bytes_used`, `kv_fragmentation`, `kv_cache_hit`, `kv_evictions`, `cached_tokens`, per-request. This is a prerequisite for tuning.

Seventh: **quantization policy**: global `[KvType; 2]`; recommend per-layer/per-head policy:
- K q8_0, V f16 (K quantization usually cheaper in quality? Actually llama.cpp supports separate; V q8 may be more sensitive? Need evidence. We can say "mesurer K et V séparément; ne pas supposer que q8_0 K et V ont le même coût qualité").
- SWA layers can use q8_0/q4_0 more aggressively because they only keep a window; global layers f16.
- First/last layers f16; middle q8_0.
- Use KL divergence / perplexity on a calibration set, not just MSE.
- q8_0 exactness: outputs change; keep f16 for exactness tests; document.

Eighth: **offload**: host KV for cold blocks; pinned memory; PCIe 4.0 x16 ~25 GB/s; a 0.125 MiB/token KV means 1 GB/s per 8k tokens; offloading cold blocks can extend context but adds latency. Use only when memory pressure; prefetch. P2.

Ninth: **attention kernel**: paged KV requires gather; Vulkan storage buffer + descriptor indexing; CUDA block table. Flash attention with online softmax; GQA; SWA. For Vulkan, `gemv_many.wgsl` and attention shaders; need tiled attention. This is where the GPU kernel expert comes in.

Tenth: **multi-GPU**: KV follows layers; for layer split, each GPU holds its layers' KV; for tensor split, KV heads split. The current `--gpus` is layer split for dense; KV memory per GPU. P2.

Now, the expert profile. Let's write a detailed "fiche" in the answer:
- Title: **Staff/Principal Inference Runtime Engineer — KV cache & memory hierarchy** (ou "Architecte runtime d'inférence, spécialité KV").
- Mission: audit existing KV, design target, implement with backend teams, prove gains.
- Must-have:
  - 5+ years systems software (Rust/C++), 2+ years LLM serving.
  - Shipped paged KV / prefix caching / continuous batching in vLLM, SGLang, TensorRT-LLM, llama.cpp, or equivalent.
  - Deep GPU memory: allocators, fragmentation, UVA/HMM, pinned memory, PCIe/NVLink, Vulkan/CUDA/HIP memory models.
  - KV quantization: q8/q4/bf16/fp8, error analysis, long-context stability.
  - Scheduling: continuous batching, chunked prefill, preemption, fairness, TTFT/ITL/p99.
  - Rust: ownership, lifetimes, `Send`/`Sync`, `unsafe`, concurrency; able to read/modify `kinetic-models`, `kinetic-server`.
  - Measurement: ROCprof/RGP/Nsight, Vulkan timestamp queries, criterion, load tests.
- Nice-to-have: MLA/DSA, MoE KV, multi-GPU, Warm Start/prefix cache, llama.cpp internals.
- 30/60/90:
  - 30: audit + metrics + baseline; map every KV allocation; reproduce memory wall; design doc.
  - 60: unified pool + dynamic allocation prototype; q8_0 Vulkan; shared KV zero-copy; A/B.
  - 90: paged KV prototype + prefix sharing; scheduler integration; conformance; production gate.
- Interactions: GPU kernel expert (attention kernels), quantization/numerics (error bounds), server (scheduler/API), QA (conformance), perf/SRE (load tests).
- Why this profile: the bottleneck is not the attention math alone; it's the memory hierarchy and scheduling. A kernel-only expert will optimize the wrong thing; a generic ML engineer won't handle allocators/concurrency; a quant expert won't design paging.
- Hiring process: ask them to review `KvCache` and propose a paged design; whiteboard block table/COW/eviction; measure a q8_0 vs f16 long-context run; code review of `copy_lent` and `KvCache::checkpoint`.
- Could be a consultant for the audit, then full-time.

Should we mention "Architecte runtime d'inférence" already in docs? Yes: "C'est le profil 2.3 de `docs/experts.md`; je le spécialiserais explicitement KV." This ties back.

Now, the response format. The user writes in French; answer in French. Use headings, tables, bullet lists. Keep it clear but detailed. Maybe 2000-3000 words? Desired oververbosity 5; but this is a technical deep dive. Aim ~1500-2200 words. The user asked a broad question; thorough but not bloated.

Let me outline:

```
Oui — dans les documents, pas dans le code.

## 1. La règle backend est bien en place (documents)
- docs/experts.md §0 ...
- fiches 1 et 3, prompts 1 et 3
- migration en 7 étapes, contrat de compatibilité
- code inchangé: git status ...
- si tu veux, je peux enchaîner sur le squelette kinetic-runtime + ModelRunner.

## 2. Ce que fait le cache KV aujourd'hui
Table...
Points forts...
Limites...
Le problème central: réservation statique du contexte complet.

## 3. Améliorations, par priorité
P0 ...
P1 ...
P2 ...
Cross-cutting: metrics/conformance.

## 4. Quel expert doit mener l'analyse
Profile...
Skills...
30/60/90...
Why not...
Support team...

## 5. Proposition
- add docs/experts/kv-cache.md + fiche
- first concrete step: instrument + q8_0 Vulkan or unified pool ADR
```

Need to be careful: the user asked "as tu mis cela en place ?" — answer first. Then "tu as analysé le KV..." — answer. The transition can be: "Oui pour les documents. Et puisque tu enchaînes sur le KV, voici l'analyse..." But the KV question came before the backend question in the transcript. Maybe phrase: "Oui — et pour être précis sur les deux sujets..." Then cover both.

Actually, maybe the last user message is a duplicate of the earlier one due to the checkpoint, and the real pending question is the KV one. If I lead with the backend confirmation, it's fine; then the KV analysis. If the user only wanted the backend answer, the KV part is extra but relevant. If they only wanted KV, the backend part is a short confirmation. Good.

Let me double-check the exact docs sections for the backend rule to cite:
- `docs/experts.md` line 15: `## 0. Décision structurante : un seul backend complet par API`
- lines 26-... "Un seul backend complet par API GPU."
- `docs/experts/fiches-de-poste.md` Fiche 3 (line ~? "Architecte Rust systèmes/backend") and Fiche 1.
- `docs/experts/prompts-agents.md` Agent 3.
Let's verify headings.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n '^## Fiche\\|^## Agent' docs/experts/fiches-de-poste.md docs/experts/prompts-agents.md | head -40 && sed -n '15,45p' docs/experts.md"
}
```

docs/experts/fiches-de-poste.md:16:## Fiche 1 — Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)
docs/experts/fiches-de-poste.md:156:## Fiche 2 — Architecte runtime d'inférence
docs/experts/fiches-de-poste.md:279:## Fiche 3 — Architecte Rust systèmes/backend
docs/experts/fiches-de-poste.md:400:## Fiche 4 — Ingénieur quantification et exactitude numérique
docs/experts/fiches-de-poste.md:509:## Fiche 5 — Ingénieur performance / qualification
docs/experts/fiches-de-poste.md:613:## Fiche 6 — Ingénieur intégration des architectures
docs/experts/fiches-de-poste.md:709:## Fiche 7 — Ingénieur tokenizers / templates / tool calls
docs/experts/fiches-de-poste.md:798:## Fiche 8 — Ingénieur serveur / API / observabilité
docs/experts/fiches-de-poste.md:888:## Fiche 9 — Ingénieur multimodal
docs/experts/fiches-de-poste.md:967:## Fiche 10 — Ingénieur build/release/bindings
docs/experts/prompts-agents.md:67:## Agent 1 — GPU AMD (Vulkan/RDNA 4, puis HIP)
docs/experts/prompts-agents.md:130:## Agent 2 — Runtime d'inférence
docs/experts/prompts-agents.md:191:## Agent 3 — Rust systèmes/backend
docs/experts/prompts-agents.md:256:## Agent 4 — Quantification et exactitude numérique
docs/experts/prompts-agents.md:316:## Agent 5 — Performance et qualification
docs/experts/prompts-agents.md:371:## Agent 6 — Intégration des architectures
docs/experts/prompts-agents.md:425:## Agent 7 — Tokenizers, templates et tool calls
docs/experts/prompts-agents.md:473:## Agent 8 — Serveur, API et observabilité
docs/experts/prompts-agents.md:523:## Agent 9 — Multimodal
docs/experts/prompts-agents.md:570:## Agent 10 — Build, release et bindings
## 0. Décision structurante : un seul backend complet par API

Le dépôt a aujourd'hui **deux couches par API GPU** :

- CUDA est éclaté entre `kinetic-cuda` (driver, noyaux, lanceurs) et
  `kinetic-models/src/cuda` (orchestration du modèle) ;
- Vulkan est éclaté entre `kinetic-vulkan` (loader, shaders, noyaux) et
  `kinetic-models/src/vulkan` (orchestration du modèle).

La cible est différente et plus simple :

> **Un seul backend complet par API GPU.** Tout CUDA vit dans `kinetic-cuda` :
> driver, noyaux, layouts, graphes, mémoire, exécution du modèle, capacités.
> Tout Vulkan vit dans `kinetic-vulkan`. Le CPU reste le chemin de référence :
> ses noyaux dans `kinetic-cpu`, son exécution générique dans `kinetic-models`.
> `kinetic-models` ne contient plus de code CUDA/Vulkan : il décrit les
> architectures, les blocs et les invariants, et expose une interface
> d'exécution **au niveau modèle** (charger, préfill, décoder, état de
> séquence, capacités), pas au niveau noyau.

Consolider ne signifie pas copier CUDA dans Vulkan. Chaque backend doit
offrir les mêmes **classes** de fonctionnalités — graphes/replay, attention
flash, MoE groupé, KV quantifié, multi-device — avec les mécanismes de son
API. La parité se juge en fonctionnalités, en exactitude et en mesures, pas
en lignes de code.

Les autres API (HIP, SYCL, WebGPU) sont des **backends de compatibilité** :
chacun est complet pour son API, mais n'est ajouté que lorsqu'une machine et
une CI peuvent le valider. Metal reste un chemin du backend Vulkan
(MoltenVK), pas un backend séparé. L'utilisateur n'en charge qu'un seul à
l'exécution, selon sa carte. On fournit le socle (trait, capacités, tests de


---

## Assistant

_Thinking:_

Good.

Now let me think about whether to mention specific line numbers / file paths. Yes, precise references strengthen the analysis:
- `crates/kinetic-models/src/blocks/attention.rs`: CPU `KvCache` f32, journal, lent.
- `crates/kinetic-cuda/src/attention.rs`: `KvType`, `Store`, `with_types`, `copy_kv`.
- `crates/kinetic-vulkan/src/attention.rs`: f16 `DeviceBuffer<u32>`.
- `crates/kinetic-models/src/vulkan/layer.rs`: `KvCache::new(dev, &a.shape, positions, rows)`.
- `crates/kinetic-models/src/cuda/layer.rs`: `new_state`, `copy_lent`.
- `crates/kinetic-server/src/engine.rs`: `free`, one backend per request.
- `crates/kinetic-server/src/vulkan.rs`: single backend, no slot.
- `crates/kinetic-cli/src/main.rs`: `--cache-type-k/v`, `default_slots`, `MAX_DEFAULT_SLOTS`, `share`.
- `crates/kinetic-models/src/warm/`: exact state cache.
- `docs/src/devices.md`: q8_0 CUDA only, CPU/Vulkan ignore.
- `PLAN.md`: KV quantifié q8_0 P0.2.
- `llama-kinetic.md`: KV table, gaps.

Let's verify the `KvCache` in `kinetic-vulkan` uses `DeviceBuffer<u32>` and f16; yes. `KvCache::new` signature includes `positions, rows`; slots = window + rows. Good.

Potential issue: The CPU `KvCache` `k`/`v` are `Vec<f32>` and full context, but `state_bytes` uses `window` min. Good.

Potential issue: CUDA `KvCache` q8_0 `Store::save` returns f32 dequantized; `load` quantizes. The `q8_0_cache_saves_and_loads_unchanged` test tolerance 1e-2. Good.

Potential issue: Vulkan `KvCache` `positions` full context, `rows=PROMPT_BATCH`; for non-window, slots=positions; for window, slots=min(window+rows, positions). Good.

Potential issue: `state_bytes` for Vulkan includes `state.work.size_bytes()` + layers; `Work::size_bytes` includes KV? `LayerState::size_bytes` includes `KvCache::size_bytes`; yes. Good.

Potential issue: "Vulkan one slot" — `vulkan_backend` returns `vec![Box::new(gpu)]`; `Engine` slots=1. But maybe the server can create multiple `VulkanBackend`s via `--parallel`? The CLI ignores `parallel` for Vulkan. So one request at a time. This is a major gap for concurrency. Mention as P0 for server, but the KV expert should know.

Potential issue: `Engine` `MAX_CONNECTIONS=64` but one slot; HTTP connections queue. The server is thread-per-connection; 64 connections, one GPU request at a time on Vulkan. This is relevant to TTFT/p99. Mention.

Potential issue: `PREFILL_CHUNK=256` per request; with one Vulkan slot, a long prompt blocks all other requests. With CUDA Convoy, passes can be shared across slots, but a long prefill still occupies the device. Chunked prefill + scheduler needed.

Potential issue: `Convoy` only CUDA and CPU; Vulkan no Convoy. So Vulkan server single-request and no shared passes. Mention.

Potential issue: `Warm Start` is available on Vulkan? `vulkan_backend` calls `warm_cache(... Some("vulkan"))`, so yes, Warm Start works on Vulkan (docs/src/devices.md says "More on Vulkan: ... Warm Start" maybe not? It says "Coming: More on Vulkan: more architectures, speculative decoding, Warm Start and layer splits." Hmm, but `vulkan_backend` calls `warm_cache`; maybe Warm Start is supported? Let's check `warm_cache` function and `VulkanBackend::with_warm`; it is implemented. The docs "Coming" might be outdated or refer to something else. Not critical. We can say "Warm Start existe sur CPU/CUDA; le chemin Vulkan a `with_warm` mais la doc le liste encore comme à venir" — avoid overclaim. Better: "Warm Start est le cache de préfixe exact; il est branché sur CUDA et Vulkan via `with_warm` (à vérifier/valider côté Vulkan)." Hmm. Let's check `warm_cache` gating. It takes `Some("cuda")`/`Some("vulkan")` and likely checks device id. The `VulkanBackend::with_warm` exists. So it is implemented. The docs "Coming" may be stale. We can mention "Warm Start existe et est exact; il stocke l'état complet, pas des blocs." No need to claim backend coverage.

Potential issue: "KV cache is f16 on Vulkan" — yes. "No q8_0 on Vulkan" — yes, `CacheType` conversion is CUDA-only and `vulkan_backend` ignores `cache_type`. The CLI args are not gated by feature? They are always present, but `From<CacheType> for kinetic_cuda::KvType` is `#[cfg(feature = "cuda")]`; `cuda_backend` uses them. On Vulkan, ignored. Docs confirm.

Potential issue: "CPU f32" — yes. The docs table in `llama-kinetic.md` says CPU f16, but code is f32. Wait, the table says "Kinetic CPU | f16" for f32/f16 row. Is that accurate? The CPU `KvCache` stores `Vec<f32>`; `state_bytes` counts f32. So the table may be wrong or refers to something else. Let's verify: `blocks/attention.rs` `k: Vec<f32>`, `v: Vec<f32>`. Yes, CPU is f32. The `llama-kinetic.md` table says CPU f16, which is a factual error in that doc. We should not repeat it. We can say "CPU: f32 (référence exacte)" and maybe note the doc table is inaccurate if relevant. But the user didn't ask to correct `llama-kinetic.md`; we can mention "le tableau de `llama-kinetic.md` dit f16 pour le CPU, mais le code stocke des f32" as a finding. That's a useful audit insight. Let's double-check `KvCache` default and `store` methods to be sure. `k: Vec<f32>`, `v: Vec<f32>`. Yes. The `state_bytes` uses `size_of::<f32>()`. So CPU is f32. This is a discrepancy worth flagging: the parity table overstates CPU KV quantization. Good catch. The user asked for deep analysis; this is exactly the kind of finding.

Also, `llama-kinetic.md` table says "Kinetic Vulkan f16" (correct), "Kinetic CUDA f16, q8_0" (correct), "Kinetic HIP non" (correct). CPU f16 is wrong; it's f32. We can note "à corriger dans `llama-kinetic.md`".

Another discrepancy: `docs/src/devices.md` says "The CPU and Vulkan paths ignore it" for `--cache-type-k/v`; correct.

Another: `docs/src/devices.md` says "On CUDA, --cache-type-k q8_0 and --cache-type-v q8_0 ... 34 bytes where f16 takes 64" — correct.

Another: `PLAN.md` says "KV quantifié (q8_0 d'abord): 2× de contexte sur 16 Gio, absent aujourd'hui." But CUDA already has q8_0; the plan likely means Vulkan. It says "absent aujourd'hui" broadly; the code has CUDA q8_0. This is a discrepancy: PLAN P0.2 is Vulkan, and q8_0 exists on CUDA. The plan's wording may be outdated. We can note "q8_0 existe déjà côté CUDA; l'écart est Vulkan". This is relevant.

Let's verify `PLAN.md` context: P0.2 Vulkan; yes, the bullet "KV quantifié (q8_0 d'abord)" is under P0.2 Vulkan. So it means Vulkan. Fine, not a discrepancy. But the phrase "absent aujourd'hui" could be read globally; clarify.

Another finding: `CudaModel::state_bytes` only counts `state.work.size_bytes()`, which includes layer states (KV) and scratch. Good.

Another: `default_slots` uses `model.state_bytes(context)` (f32) for CPU; for GPU CUDA slots are allocated by actual memory, so no issue. But `fit::gpu_context` uses `share` to adjust context for q8_0; good. However, `default_slots` for CPU doesn't account for q8_0 because CPU ignores it; correct. For Vulkan, no slots.

Another: `MAX_DEFAULT_SLOTS=4` is a hard cap for CUDA/CPU; with paged KV, could be higher. Good.

Another: `SLOT_RESERVE=256 MiB` per CUDA slot; with many slots, this reserve adds up. Mention.

Another: `KvCache::with_types` rejects `slots <= window` for windowed caches; the cache needs `window + rows` slots. Good design; mention as strength.

Another: `KvCache::copy_kv` rejects non-F16 shared caches; so q8_0 + shared KV is unsupported. Mention.

Another: `KvCache::save/load` for q8_0 round-trips with quantization error; Warm Start exactness may be broken for q8_0? The test says q8_0 save/load unchanged within 1e-2, not bit-exact. Warm Start claims bit-for-bit exact state; if q8_0 is used, save/load dequantizes/quantizes and may not be bit-exact. Does Warm Start support q8_0? The warm cache uses `host_state` which calls `save_cache` and `KvCache::save`; for q8_0, `save` dequantizes to f32, and `load` re-quantizes. The resulting q8_0 bytes may differ due to rounding, so the state is not bit-for-bit. The docs say "A restored state is bit for bit the one its run left behind" — is that true for q8_0? The test `q8_0_cache_saves_and_loads_unchanged` asserts closeness 1e-2, not exact. This is a subtle but important finding


---

## Assistant

_Thinking:_

Interesting: The session is a transcript from a previous session. The last section we read is actually the previous assistant's analysis (the one we're supposed to maybe continue). The user's current question: "as tu analysé le cache kv..." — the transcript at 21120-21479 is the assistant's answer to that question. So the session file already contains the answer. Our job is to analyze `docs/experts/` and the session file to see if all expert prompts and missions were created.

We should not just reproduce the transcript; we need to audit the final docs against the session's analysis and identify gaps. The user asks "pour voir si tous les prompts pour les experts et missions ont bien été créés." So the deliverable is an audit: list what exists, what's missing, inconsistencies.

Let's carefully compare:
- Session final analysis (lines 7357-7769) lists 10 profiles:
  1. GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)
  2. Architecte runtime d'inférence
  3. Ingénieur quantification et exactitude numérique
  4. Architecte Rust systèmes / backend
  5. Ingénieur performance / qualification
  6. Ingénieur intégration des architectures
  7. Ingénieur tokenizers / templates / tool calls
  8. Ingénieur serveur / API / observabilité
  9. Ingénieur multimodal
  10. Ingénieur build/release/bindings

The docs have exactly these 10, in a slightly different order (Rust backend is #3 in fiches, quantization #4; session final had quantization #3, Rust #4; `docs/experts.md` has Rust #2.4, quantization #2.5, which matches session's final? Let's check the final table at 7413. It says:
- P0 — 1. GPU AMD
- P0 — 2. Runtime
- P0/P1 — 3. Quantification
- P0/P1 — 4. Rust
- P0 — 5. Performance
- P1 — 6. Model integration
- P1 — 7. Tokenizers
- P1 — 8. Server
- P2 — 9. Multimodal
- P2 — 10. Build

But `docs/experts.md` has 2.3 Runtime, 2.4 Rust, 2.5 Quantification. So the order differs, but all present. Is that an issue? Not really, but for consistency with the session's final recommendation, maybe reorder? The user asks if all were created, not order. We can note "ordre différent, contenu présent".

The `fiches-de-poste.md` order: 1 GPU, 2 Runtime, 3 Rust, 4 Quantization, 5 Performance, 6 Model, 7 Tokenizers, 8 Server, 9 Multimodal, 10 Build. So the fiches order matches `docs/experts.md`, but not the session's final table (which has Quantization #3, Rust #4). Minor.

Now, check each fiche against the session's "Profile X details" (lines 6298-6453) and the final "P0 — 1..." sections (7048 onward). Are all mission details present? The fiches are more detailed, so likely yes. But there may be missing "Interactions" or "Signaux d'alerte" for some. Let's compare:
- Fiche 1: mission, context, responsibilities, skills, appreciated, scope, context files, commands, 30/60/90, success, interactions, profile, warning signs. Complete.
- Fiche 2: same. Complete.
- Fiche 3: same. Complete.
- Fiche 4: same. Complete.
- Fiche 5: same. Complete.
- Fiche 6: same. Complete.
- Fiche 7: same. Complete.
- Fiche 8: same. Complete.
- Fiche 9: same. Complete.
- Fiche 10: same. Complete.

Prompts:
- Agent 1: role, mission, required reading, scope, forbidden, loop, commands, success, final report. Complete.
- Agent 2: same. Complete.
- Agent 3: same. Complete.
- Agent 4: same. Complete.
- Agent 5: same. Complete.
- Agent 6: same. Complete.
- Agent 7: same. Complete.
- Agent 8: same. Complete.
- Agent 9: same. Complete.
- Agent 10: same. Complete.

Common contract: truth/measurement, exactness, scope, process, stop. Complete.

Missing prompts for:
- Security expert (cross-cutting) — mentioned in session at 6185 and 6434? Let's inspect. The final 10 profiles omit security. The session's "Cross-cutting: sécurité" might be a section, not a profile. The user asks "tous les prompts pour les experts et missions" — if security was a cross-cutting concern, maybe no dedicated prompt needed. But the session's earlier "12. Security engineer?" and "Cross-cutting: sécurité" suggest it was considered. We should flag: no dedicated security prompt; security is only mentioned as cross-cutting in the earlier analysis, and not in the final docs. Is that a gap? For a local inference engine, security matters (server, model files, unsafe code). The docs' server/API fiche includes "Sécuriser les entrées" and the Rust backend fiche includes unsafe audit. But no security expert prompt. Flag as "optional gap".

- Technical writer / DevRel (optional) — session at 5431 "11. Technical writer / developer experience (optional)". Not in final docs. Flag as "intentionally omitted; no prompt".

- HIP/ROCm dedicated expert — session at 5347 "2. HIP/ROCm backend engineer (top priority locally, second wave)" but final merged into GPU AMD. The fiche 1 covers HIP. Not missing, but if the user expected a separate prompt, note it's merged.

- "Model integration engineer (architectures/tokenizers)" — session at 5388 combined; final split into 6 and 7. Both present.

- "Server/API & platform engineer" — session at 5399 combined; final split into 8 and 10. Both present.

- "Performance measurement / QA automation engineer" — present as 5.

- "Rust systems architect / performance engineer" — present as 3.

- "Quantization & numerical correctness" — present as 4.

- "Multimodal" — present as 9.

- "Build/release & bindings" — present as 10.

So the 10 final profiles are complete. The earlier 12-profile draft had 2 extra optional profiles not carried over. We should state that clearly.

Now, are there any **missions** in the session that are not in the docs? The session's final "Composition d'équipe recommandée" and "Séquencement" might contain missions like "cellule performance minimale (5 personnes)", "équipe complète (9–10)", "séquencement". The docs `docs/experts.md` section 4 likely covers composition and sequencing. Let's read that section to verify. Also the "Matrice chantier → pilote" and "Comment évaluer ces experts" and "Ce qu'il ne faut pas faire". The docs may have these. Let's inspect `docs/experts.md` sections 3-8.

**Tool: read**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md",
  "offset": 200,
  "limit": 200
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts.md, lines 200-399
200: - **P1 — Intégration des architectures** : couvrir les architectures
201:   manquantes par briques réutilisables, sans multiplier les `match`.
202: - **P1 — Tokenizers/templates/tool calls** : compatibilité des tokenizers
203:   et des templates.
204: - **P1 — Serveur/API/observabilité** : parité API, robustesse,
205:   observabilité.
206: - **P2 — Multimodal** : vision, audio, OCR.
207: - **P2 — Build/release/bindings** : rendre le produit installable et
208:   intégrable.
209: - **Transverse — Sécurité** : entrées, réseau, `unsafe`, chaîne
210:   d'approvisionnement. À traiter par le serveur, le backend Rust et la QA,
210:   pas par un poste dédié tant que la surface reste locale.

Ah, `docs/experts.md` has "Transverse — Sécurité" and says no dedicated post. So security is intentionally covered. Good. The fiches don't have a security fiche, but the docs explain why. We should note that.

Let's read the rest of `docs/experts.md` (lines 1-200 and 400-700) to fully audit.

**Tool: read**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md",
  "limit": 200
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts.md, lines 1-200
1: # Experts pour l'architecture et la performance de Kinetic
2: 
3: > **Statut** : analyse, 22 novembre 2026.
4: > **Snapshot** : `main` du fork `Angra974/kinetic`, complété par `PLAN.md` et
5: > `llama-kinetic.md`.
6: > **Portée** : le fork local (Ryzen 7 7800X3D, RX 9070 XT `gfx1201`, Vulkan
7: > RADV, ROCm 7.2), pas le plan CUDA de l'amont.
8: > **But** : déterminer les profils d'experts, leurs compétences, leur périmètre
9: > d'écriture, leurs fichiers de contexte et leurs commandes de mesure pour
10: > améliorer l'architecture et la performance de Kinetic.
11: > **Placement** : document interne, hors du mdBook publié (`docs/src/`).
12: 
13: ---
14: 
15: ## 0. Décision structurante : un seul backend complet par API
16: 
17: Le dépôt a aujourd'hui **deux couches par API GPU** :
18: 
19: - CUDA est éclaté entre `kinetic-cuda` (driver, noyaux, lanceurs) et
20:   `kinetic-models/src/cuda` (orchestration du modèle) ;
21: - Vulkan est éclaté entre `kinetic-vulkan` (loader, shaders, noyaux) et
22:   `kinetic-models/src/vulkan` (orchestration du modèle).
23: 
24: La cible est différente et plus simple :
25: 
26: > **Un seul backend complet par API GPU.** Tout CUDA vit dans `kinetic-cuda` :
27: > driver, noyaux, layouts, graphes, mémoire, exécution du modèle, capacités.
28: > Tout Vulkan vit dans `kinetic-vulkan`. Le CPU reste le chemin de référence :
29: > ses noyaux dans `kinetic-cpu`, son exécution générique dans `kinetic-models`.
30: > `kinetic-models` ne contient plus de code CUDA/Vulkan : il décrit les
31: > architectures, les blocs et les invariants, et expose une interface
32: > d'exécution **au niveau modèle** (charger, préfill, décoder, état de
33: > séquence, capacités), pas au niveau noyau.
34: 
35: Consolider ne signifie pas copier CUDA dans Vulkan. Chaque backend doit
36: offrir les mêmes **classes** de fonctionnalités — graphes/replay, attention
37: flash, MoE groupé, KV quantifié, multi-device — avec les mécanismes de son
38: API. La parité se juge en fonctionnalités, en exactitude et en mesures, pas
39: en lignes de code.
40: 
41: Les autres API (HIP, SYCL, WebGPU) sont des **backends de compatibilité** :
42: chacun est complet pour son API, mais n'est ajouté que lorsqu'une machine et
43: une CI peuvent le valider. Metal reste un chemin du backend Vulkan
44: (MoltenVK), pas un backend séparé. L'utilisateur n'en charge qu'un seul à
45: l'exécution, selon sa carte. On fournit le socle (trait, capacités, tests de
46: conformance, modèle de build), pas une implémentation complète et
47: invérifiable de chaque API.
48: 
49: ### 0.1 État actuel vs cible
50: 
51: | Aujourd'hui | Cible |
52: |---|---|
53: | CUDA : `kinetic-cuda` + `kinetic-models/src/cuda` | `kinetic-cuda` contient tout CUDA |
54: | Vulkan : `kinetic-vulkan` + `kinetic-models/src/vulkan` | `kinetic-vulkan` contient tout Vulkan |
55: | `kinetic-models` a des features `cuda` / `vulkan` et des modules backend | `kinetic-models` sans feature backend, expose un trait `ModelRunner` |
56: | `kinetic-server` a `cuda.rs`, `vulkan.rs`, `cpu.rs` | `kinetic-server` choisit un backend via une factory |
57: | Ajouter HIP = nouveau crate + nouveau module dans `kinetic-models` | Ajouter HIP = nouveau crate qui implémente `ModelRunner` |
58: | Les tests modèle connaissent les backends | Les tests modèle utilisent le runner CPU ou un mock |
59: 
60: ### 0.2 Où vivent les traits
61: 
62: Deux couches, pas une :
63: 
64: - **`kinetic-runtime`** (à créer) : découverte des devices, registre/factory
65:   de backends, capacités, budget mémoire, ordonnanceur, pool KV, métriques.
66:   Il ne connaît **pas** les types de modèle.
67: - **`kinetic-models`** : architectures, blocs, modèle générique, trait
68:   `ModelRunner` au niveau modèle, runner CPU de référence.
69: - **`kinetic-cuda` / `kinetic-vulkan` / `kinetic-hip`** : implémentent
70:   `ModelRunner` et possèdent leurs noyaux, graphes et politiques de mémoire.
71: - **`kinetic-server`** : garde le trait `Backend` actuel (requête,
72:   `reuses`, `run_ahead`, `settle_some`) au-dessus de `ModelRunner`.
73: 
74: Cette séparation évite un cycle de dépendances et évite de transformer
75: `kinetic-runtime` en compilateur de graphes. Le trait `ModelRunner` reste
76: petit : charger, préfill, décoder, état de séquence, capacités. Chaque
77: backend garde la liberté de fusionner, grapher et ordonnancer ses noyaux
78: comme il veut.
79: 
80: ### 0.3 Migration incrémentale
81: 
82: 1. Créer `kinetic-runtime` avec les types de devices, capacités, budget
83:    mémoire et factory. Aucun changement de comportement.
84: 2. Extraire dans `kinetic-models` un trait `ModelRunner` au niveau modèle et
85:    exposer les descripteurs nécessaires aux backends. Le runner CPU existant
86:    devient l'implémentation de référence.
87: 3. Déplacer `kinetic-models/src/cuda` vers `kinetic-cuda/src/runner` et
88:    `kinetic-models/src/vulkan` vers `kinetic-vulkan/src/runner`. Ce n'est pas
89:    un simple `git mv` : il faut passer par le trait et les descripteurs
90:    exposés, et conserver les tests à chaque étape.
91: 4. Retirer les features `cuda` / `vulkan` de `kinetic-models` ; il ne dépend
92:    plus des crates backend.
93: 5. Ajouter une factory de backend dans `kinetic-server` / `kinetic-cli` :
94:    `auto | cpu | cuda | vulkan | hip`, un seul backend chargé.
95: 6. Ajouter un backend de test (CPU ou mock) pour tester `kinetic-models`
96:    sans GPU.
97: 7. Seulement ensuite, optimiser. Le refactor d'abord ; la performance vient
98:    après, avec les mêmes gates d'exactitude.
99: 
100: ### 0.4 Contrat de compatibilité d'un backend
101: 
102: Un backend n'est « fourni » que s'il respecte ce contrat :
103: 
104: - **Découverte** : `probe()` liste les devices et leurs capacités réelles.
105: - **Capacités** : types de poids, formats KV, batch max, async, multi-device,
106:   mémoire disponible. Le serveur refuse clairement ce qui n'est pas supporté.
107: - **Conformance** : mêmes tokens que le runner CPU sur le corpus de contrôle,
108:   aux tolérances près, et mêmes gates d'identité.
109: - **Mesure** : `kinetic bench` et le protocole A/B fonctionnent sur ce
110:   backend, avec un seuil de régression.
111: - **Isolement** : un seul backend est chargé à l'exécution ; les autres ne
112:   coûtent ni mémoire ni temps de démarrage.
113: - **CI** : au moins un runner ou une machine de référence valide le backend
114:   avant qu'il ne soit activé par défaut.
115: 
116: Tant qu'un backend ne passe pas ce contrat, il reste **opt-in** et documenté
117: comme expérimental. C'est ainsi qu'on fournit les bases de HIP/SYCL/WebGPU
118: sans promettre une parité invérifiable.
119: 
120: ### 0.5 Ce que cela change pour les experts
121: 
122: - **GPU AMD** travaille dans `kinetic-vulkan` et `kinetic-hip`, pas dans
123:   `kinetic-models`.
124: - **Runtime** travaille dans `kinetic-runtime` et `kinetic-server`, pas dans
125:   les crates backend.
126: - **Rust backend** pilote la migration et garde les seams `Family`,
127:   `Target`, `Backend`, `Projector`.
128: - **Quantification** travaille dans `kinetic-quant` et les noyaux, avec une
129:   seule définition de référence.
130: - **Intégration modèles** travaille dans `kinetic-models/src/arch` et
131:   `blocks`, sans code backend.
132: - **QA** teste chaque backend séparément avec le même corpus et les mêmes
133:   gates.
134: 
135: ### 0.6 Non-objectifs
136: 
137: - Pas de compilateur de graphes généraliste (MLIR ou équivalent).
138: - Pas d'ABI de plugins dynamiques au départ : les features Cargo suffisent.
139: - Pas de noyau « universel » qui serait lent partout : chaque backend
140:   optimise son chemin.
141: - Pas de backend CUDA prioritaire sur cette machine : pas de GPU NVIDIA,
142:   donc pas de mesure locale.
143: 
144: ---
145: 
146: ## 1. Diagnostic
147: 
148: ### 1.1 Ce que le dépôt fait bien
149: 
150: - Architecture en couches claire : `kinetic-gguf` → `kinetic-quant` →
151:   `kinetic-cpu` → `kinetic-models` (`arch/`, `blocks/`) → exécuteurs →
152:   `kinetic-server` → CLI/bindings.
153: - Registre `Family` : une architecture = un fichier + une ligne, pas de
154:   `match` central.
155: - Contrat `Target` pour le spéculatif : RAD-1, rollback, arbre de
156:   vérification, exactitude.
157: - Culture de l'exactitude : référence f64, batch-invariance, égalité passe
158:   mono-token / passe prompt, tests de non-régression.
159: - Différenciateurs réels : RAD-1, Warm Start, Zero Wait, Shortlist, Convoy,
160:   Flywheel, `.ktc` signé.
161: - Discipline : zéro `TODO/FIXME`, doc comments, CI Linux/Windows/macOS,
162:   `clippy -D warnings`, ~517 tests.
163: 
164: ### 1.2 Les goulets mesurés
165: 
166: | Goulet | Preuve (22 nov. 2026) | Expert |
167: |---|---|---|
168: | Préfill Vulkan | 261,9 t/s contre 1154,2 pour llama.cpp Vulkan sur un dense 27B IQ3_XXS, soit **4,4×** | GPU AMD |
169: | MoE Vulkan | 55,7 contre 181,9 t/s en décodage, 489 contre 2637 t/s en préfill, soit **3,3× / 5,4×** | GPU AMD + runtime |
170: | Soumission synchrone | `vkWaitForFences` après chaque soumission, pas de timeline semaphores | GPU AMD |
171: | Cache de pipelines | `create_compute_pipelines(vk::PipelineCache::null(), …)` | GPU AMD |
172: | KV non quantifié | `KvCache` en f16, pas de q8_0/q4_0 | GPU AMD + quantification |
173: | Concurrence | `Engine.free` = un backend par requête ; `Convoy` mutualise des passes mais ne planifie pas | Runtime |
174: | Duplication backend | `kinetic-models/src/cuda` ≈ 4,5 k LOC + `kinetic-cuda` ≈ 8,5 k LOC ; `vulkan` ≈ 2,2 k + `kinetic-vulkan` ≈ 5,9 k | Rust backend |
175: | Quantification | 5 cibles contre ~30 ; pas d'imatrix, pas d'IQ encode, pas de NVFP4/FP8 runtime | Quantification |
176: | Couverture | 21/180 architectures, 13/68 prétokenizers, ~15/45 routes, 2/73 projecteurs | Modèles / tokenizers / serveur |
177: | Mesure | protocole excellent mais manuel ; pas de CI GPU, pas de `THEORY.md`, pas de ledger automatisé | QA/perf |
178: 
179: ### 1.3 Les goulets structurels
180: 
181: 1. **La duplication backend** : ajouter HIP/SYCL/WebGPU va multiplier les
182:    exécuteurs si `kinetic-models` continue de contenir du code CUDA/Vulkan.
183: 2. **Le runtime** : le serveur est conçu pour N backends indépendants, pas
184:    pour un ordonnanceur token-level avec KV paginé, chunked prefill global,
185:    priorité et annulation.
186: 3. **Le préfill** : `gemv_many.wgsl` n'est pas un GEMM tuilé ; `PROMPT_BATCH`
187:    vaut 128 et le chemin « many tokens » garde 32 tokens en vol. C'est le
188:    seul écart de performance mesuré sur la machine locale.
189: 4. **Le MoE** : le GEMV groupé lit les experts un slot à la fois ; le
190:    décodage est borné par les lectures d'experts en mémoire hôte, pas par le
191:    calcul GPU.
192: 5. **La quantification** : le runtime lit les formats courants, mais
193:    l'encodeur, l'imatrix, NVFP4/FP8 et le KV quantifié manquent.
194: 6. **La mesure** : les chiffres sont la crédibilité du projet ; ils doivent
195:    être automatisés, pas recopiés à la main.
196: 
197: ### 1.4 Conséquence
198: 
199: Ce n'est pas un problème de « framework ML ». C'est un problème de
200: **systèmes GPU, de runtime et de numérique**. Les experts à recruter sont
[Output truncated. Continue reading with offset: 201]

**Tool: read**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md",
  "offset": 400,
  "limit": 200
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts.md, lines 400-599
400: **Compétences clés** : zoo Transformer (MHA/GQA/MQA/MLA/DSA/sinks),
401: RoPE/iRoPE/partial/scaling, MoE top-k/shared/sigmoid, SSM/hybride (Mamba2,
402: RWKV, delta net), norm/activation, T5 encodeur-décodeur, têtes
403: embedding/rerank/TTS ; lecture GGUF ; mapping vers `blocks/` ; conformance
404: vs llama.cpp ; discipline « un fichier par famille ».
405: 
406: **Pourquoi ici** : 21/180 architectures ; les briques les plus rentables sont
407: MLA/DSA, les hybrides SSM et T5 ; les modèles locaux bloqués sont `qwen3vl`,
408: `deepseek`, `llama4`, `gemma2`, `phi3`, `command-r`, `dbrx`.
409: 
410: **Livrables** : lots d'architectures par brique réutilisable, tests de
411: conformance, documentation `models.md`.
412: 
413: **KPI** : modèles chargés, tokens identiques à llama.cpp, couverture.
414: 
415: ### 2.8 P1 — Ingénieur tokenizers / templates / tool calls
416: 
417: **Mission** : atteindre la compatibilité des tokenizers et des templates.
418: 
419: **Compétences clés** : BPE, SentencePiece, WordPiece, UGM, Tekken,
420: prétokenizers regex, byte-level, tokens spéciaux EOT/EOM/FIM ; templates
421: Jinja ; formats de tool calls ; JSON Schema/GBNF ; tests différentiels vs
422: `llama-tokenize` et `llama-server`.
423: 
424: **Pourquoi ici** : 13/68 prétokenizers, 55 manquants ; templates intégrés et
425: formats tool-call incomplets ; l'issue #5 (tool calls dans le raisonnement)
426: est typique.
427: 
428: **Livrables** : 15 prétokenizers prioritaires, UGM, split GGUF, templates
429: intégrés, formats tool-call.
430: 
431: **KPI** : ids identiques sur un corpus de contrôle, requêtes serveur
432: compatibles.
433: 
434: ### 2.9 P1 — Ingénieur serveur / API / observabilité
435: 
436: **Mission** : parité API, robustesse et observabilité.
437: 
438: **Compétences clés** : HTTP/1.1, SSE, keep-alive, chunked, timeouts,
439: backpressure, TLS, API keys, CORS, arrêt propre ; APIs OpenAI, llama-server,
440: Anthropic Messages, Ollama ; `n > 1`, `timings_per_token`, `/props`,
441: `/slots`, `/infill`, `/apply-template`, `/lora-adapters` ; Prometheus, logs
442: structurés, tracing, p99 ; sécurité (limites d'entrée, DoS, path traversal).
443: 
444: **Pourquoi ici** : ~15/45 routes, pas de keep-alive/TLS, 64 connexions
445: thread-per-connection, `docker stop` à améliorer, compatibilité clients
446: perfectible.
447: 
448: **Livrables** : routes manquantes, `n > 1`, TLS/API key, router multi-modèle,
449: métriques, arrêt propre.
450: 
451: **KPI** : conformité llama-server, overhead, stabilité sous charge.
452: 
453: ### 2.10 P2 — Ingénieur multimodal
454: 
455: **Mission** : vision, audio, OCR.
456: 
457: **Compétences clés** : ViT/SigLIP/CLIP, projecteurs (MLP/merger/Q-Former),
458: preprocessing image, vidéo, audio (mel, ASR/TTS), OCR, noyaux GPU vision.
459: 
460: **Pourquoi ici** : 2/73 projecteurs, pas d'audio/vidéo/OCR. Impact produit
461: fort mais chantier XL, hors chemin critique de performance.
462: 
463: **Livrables** : 10 projecteurs majeurs, crate `kinetic-audio`, pipeline OCR.
464: 
465: **KPI** : parité des embeddings vs `llama-mtmd`, qualité des réponses.
466: 
467: ### 2.11 P2 — Ingénieur build/release/bindings
468: 
469: **Mission** : rendre le produit installable et intégrable.
470: 
471: **Compétences clés** : Cargo, cross-compilation, toolchains CUDA/HIP, Docker,
472: GitHub Actions, signature de code, winget/scoop, maturin/PyO3, ABI C, semver,
473: mdBook.
474: 
475: **Pourquoi ici** : 10/200 fonctions C, bindings Python partiels, images
476: Docker et packaging à finaliser.
477: 
478: **Livrables** : ABI C complète, bindings générés, images Docker, installateurs,
479: CI release.
480: 
481: **KPI** : couverture ABI, temps/taille de build, taux d'installation réussie.
482: 
483: ---
484: 
485: ## 3. Composition et séquencement
486: 
487: ### 3.1 Cellule « performance » minimale (5 personnes)
488: 
489: | Rôle | Mission | Justification |
490: |---|---|---|
491: | GPU AMD | Vulkan d'abord, HIP ensuite | Le seul écart mesuré est le préfill/MoE |
492: | Runtime | Ordonnanceur, KV, concurrence | Le prochain plafond après le préfill |
493: | Rust backend | Abstraction backend, dette | Évite de dupliquer HIP/SYCL/WebGPU |
494: | Numérique | Quantification + gates | Protège le différenciateur exactitude |
495: | Performance/QA | CI GPU, ledger, A/B | Sans mesure, pas de gain prouvé |
496: 
497: ### 3.2 Équipe complète (9–10 personnes)
498: 
499: Ajouter : **intégration modèles**, **tokenizers/templates**, **serveur/API**,
500: **multimodal**, **build/release/bindings**.
501: 
502: ### 3.3 Phases
503: 
504: | Phase | Durée | Rôles | Objectif |
505: |---|---|---|---|
506: | 0. Mesure | 2–4 sem. | QA + GPU | Baseline, `THEORY.md`, CI GPU |
507: | 1. Préfill | 6–10 sem. | GPU + numérique + QA | GEMM tuilé, coopmat, async, pp ≥ 2× |
508: | 2. MoE/HIP | 8–12 sem. | GPU + runtime + Rust | MoE groupé, squelette HIP, KV q8 |
509: | 3. Concurrence | 6–10 sem. | Runtime + Rust + QA | Ordonnanceur, cN, TTFT |
510: | 4. Couverture | continu | Modèles + tokenizers + serveur | Architectures, prétokenizers, routes |
511: | 5. Produit | continu | Multimodal + build | Vision/audio, bindings |
512: 
513: ### 3.4 RACI simplifié
514: 
515: | Chantier | Pilote | Support | Porte de sortie |
516: |---|---|---|---|
517: | GEMM préfill Vulkan | GPU AMD | Numérique, QA | pp512 ≥ 2×, bit-exact |
518: | MoE Vulkan | GPU AMD | Runtime, Numérique | tg/pp, qualité |
519: | HIP/ROCm | GPU AMD/HIP | Rust backend | vs llama ROCm0 |
520: | Ordonnanceur/KV | Runtime | Rust backend, QA | cN, p99, TTFT |
521: | Quantification | Numérique | GPU/CPU kernel | Identité |
522: | Architectures | Intégration modèles | Numérique, QA | Conformance |
523: | Tokenizers | Tokenizers | Serveur | Ids identiques |
524: | API serveur | Serveur | QA | Conformance |
525: | CI perf | QA | Build/release | Régression |
526: | Bindings | Build/release | Serveur | Tests ABI |
527: 
528: ---
529: 
530: ## 4. Évaluation et recrutement
531: 
532: ### 4.1 Exercices
533: 
534: Éviter les entretiens théoriques. Donner un exercice tiré du dépôt, avec
535: mesure et gate d'exactitude.
536: 
537: | Rôle | Exercice |
538: |---|---|
539: | GPU AMD | Implémenter un GEMM tuilé Q4_K dans un nouveau shader ; le profiler avec RGP ; battre l'existant de ≥ 1,5× ; passer le test bit-exact. |
540: | Runtime | Concevoir un ordonnanceur continuous batching pour `Engine`/`Sequence` ; expliquer le rollback KV, l'annulation, la fairness ; identifier les impacts sur RAD-1/Warm Start. |
541: | Rust backend | Refactoriser un chemin dupliqué CUDA/Vulkan derrière une abstraction minimale, sans changer les tests ; expliquer `Send`/`Sync` et les `unsafe`. |
542: | Numérique | Implémenter un format de bloc de la référence f64 au noyau, avec tests différentiels et mesure. |
543: | Intégration modèles | Ajouter une famille à partir d'un GGUF inconnu, en un fichier + une ligne, et la comparer à llama.cpp. |
544: | Tokenizers | Faire passer un corpus de contrôle avec les mêmes ids que `llama-tokenize`, tokens spéciaux compris. |
545: | QA | Concevoir un A/B qui détecte une régression de 5 % avec un bruit de ±8 % ; calculer la taille d'échantillon et les contrôles. |
546: 
547: ### 4.2 Signaux d'alerte
548: 
549: - Uniquement PyTorch/TensorFlow, aucune notion de noyau ou de mémoire.
550: - CUDA uniquement, incapable de lire du WGSL/SPIR-V ou de raisonner RDNA.
551: - « Il suffit de réécrire en X » sans mesure ni migration incrémentale.
552: - Benchmarks sans contrôles, sans médiane, sans état machine.
553: - Aucune sensibilité à l'exactitude : « on verra plus tard ».
554: - Impossible de lire un profil GPU ou d'expliquer memory-bound vs
555:   compute-bound.
556: 
557: ### 4.3 Rareté et sourcing
558: 
559: | Profil | Rareté | Où chercher |
560: |---|---|---|
561: | GPU AMD | Très rare | HPC, ROCm, moteurs de jeu, compute Vulkan, équipes de noyaux ML |
562: | Runtime | Très rare | vLLM, SGLang, TensorRT-LLM, llama.cpp, ONNX Runtime |
563: | Rust backend | Rare | Systèmes, bases de données, moteurs de jeu, embedded |
564: | Numérique | Rare | Compression ML, co-design matériel/logiciel, calcul numérique |
565: | QA/perf | Rare | Performance engineering, SRE, équipes de benchmark |
566: | Intégration modèles | Moyenne | Communauté llama.cpp/transformers |
567: | Tokenizers | Moyenne | Infrastructure NLP |
568: | Serveur/API | Courante | Backend/API engineering |
569: | Multimodal | Rare | Vision/audio appliquée |
570: | Build/release | Moyenne | Release engineering, packaging |
571: 
572: ---
573: 
574: ## 5. Commandes de mesure communes
575: 
576: ### 5.1 Build et tests
577: 
578: ```sh
579: cargo build --release --locked -p kinetic-cli --features vulkan
580: cargo test --release --locked --workspace
581: cargo test --release --locked -p kinetic-vulkan
582: cargo test --release --locked -p kinetic-models
583: cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
584: ```
585: 
586: ### 5.2 Inventaire
587: 
588: ```sh
589: target/release/kinetic doctor
590: target/release/kinetic info MODEL.gguf
591: ```
592: 
593: ### 5.3 Bench Vulkan
594: 
595: ```sh
596: target/release/kinetic bench MODEL.gguf --device vulkan --json -r 6
597: llama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6
598: ```
599: 
[Output truncated. Continue reading with offset: 600]

---

## Assistant

_Thinking:_

We have a good view. Need to read the rest of `docs/experts.md` (600-699) for completeness and check sections 6-8.

**Tool: read**

**Input:**
```json
{
  "offset": 600,
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts.md, lines 600-699
600: `kinetic bench` mesure `pp512`, `tg128` et `edit256`. Pour un préfill plus
601: long, utiliser `--prompt 2048`. Les chiffres se comparent toujours sur le
602: même fichier, le même backend et le même état machine.
603: 
604: ### 5.4 Protocole et conformance
605: 
606: ```sh
607: python scripts/bench_protocol.py MODEL.gguf --llama-bin /path/to/llama-bench
608: python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
609:   --llama /path/to/llama.cpp/bin --device cpu
610: ```
611: 
612: `conformance.py` ne connaît aujourd'hui que `cpu` et `cuda`. L'extension à
613: `vulkan` est un livrable QA. En attendant, lancer
614: `kinetic serve --device vulkan` et comparer les tokens manuellement avec
615: `llama-server`.
616: 
617: ### 5.5 Serveur
618: 
619: ```sh
620: python scripts/llama_server_compat.py --model MODEL.gguf \
621:   --llama-server /path/to/llama-server --kinetic target/release/kinetic
622: python scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091
623: python scripts/same_answer_demo.py --model MODEL.gguf \
624:   --kinetic target/release/kinetic --kinetic-args "--device vulkan"
625: ```
626: 
627: ### 5.6 Vision
628: 
629: ```sh
630: cargo run --release -p kinetic-vision --example encode -- mmproj.gguf rainbow:512
631: llama-mtmd-debug --image rainbow -n 512 -m mmproj.gguf
632: ```
633: 
634: ### 5.7 Précautions
635: 
636: - Un seul job GPU à la fois ; vérifier la machine idle (`rocm-smi`,
637:   `radeontop`, aucun processus `kinetic`/`llama`).
638: - Médiane de ≥ 6 échantillons ; contrôles avant et après chaque candidat.
639: - Accepter seulement si le candidat bat les deux contrôles sur sa métrique et
640:   ne régresse rien de plus de ~3 %.
641: - Consigner chaque essai dans `PLAN.md` avec commande, commit, date, état
642:   machine et verdict.
643: - Ne jamais forcer un serveur en pleine passe : annuler les requêtes, laisser
644:   drainer, puis arrêter avec un timeout.
645: 
646: ---
647: 
648: ## 6. Ce qu'il ne faut pas faire
649: 
650: - **Ne pas recruter un « expert LLM » générique en premier.** Le goulot est
651:   GPU/runtime, pas la théorie des transformers.
652: - **Ne pas commencer par HIP.** Le décodage Vulkan est déjà à 95 % ; le
653:   préfill est à 23 %. Le GEMM Vulkan est le meilleur premier chantier.
654: - **Ne pas recruter un expert CUDA-only pour ce fork.** La machine n'a pas de
655:   GPU NVIDIA ; il ne pourrait rien mesurer. CUDA est utile en amont/CI, pas
656:   en P0 local.
657: - **Ne pas lancer un chantier multimodal ou bindings avant la performance.**
658:   C'est XL et hors chemin critique.
659: - **Ne pas introduire un compilateur de graphes généraliste.** Une couche de
660:   dispatch/exécution minimale suffit.
661: - **Ne pas casser la culture d'exactitude.** Un gain non prouvé, ou qui change
662:   la sortie, doit être rejeté.
663: 
664: ---
665: 
666: ## 7. Documents associés
667: 
668: - [`docs/experts/fiches-de-poste.md`](experts/fiches-de-poste.md) : fiches de
669:   poste détaillées pour les 10 profils.
670: - [`docs/experts/prompts-agents.md`](experts/prompts-agents.md) : prompts
671:   d'agents spécialisés, avec périmètre d'écriture, fichiers de contexte,
672:   commandes de mesure et garde-fous.
673: 
674: ---
675: 
676: ## 8. Recommandation finale
677: 
678: Si je devais n'embaucher que **quatre personnes** pour ce dépôt :
679: 
680: 1. **Ingénieur noyaux GPU AMD** — le plus gros gain mesuré, le plus rare.
681: 2. **Architecte runtime d'inférence** — le prochain plafond de scalabilité.
682: 3. **Ingénieur performance/qualification** — pour que les deux premiers
683:    prouvent leurs gains.
684: 4. **Ingénieur quantification/exactitude** — pour que ces gains restent
685:    conformes à la promesse produit.
686: 
687: Si un seul profil : **GPU AMD avec une forte sensibilité
688: quantification/exactitude**. C'est le seul qui attaque directement l'écart de
689: 4,4× en préfill et de 3,3–5,4× en MoE.
690: 
691: Si deux : ajouter **runtime**. Si trois : ajouter **QA/perf**. Si quatre :
692: ajouter **numérique**.
693: 
694: Le reste — architectures, tokenizers, serveur, multimodal, bindings — peut
695: être parallélisé, communautarisé ou recruté ensuite. Le projet a déjà les bons
696: seams (`Family`, `Target`, `Backend`, `Projector`) et une documentation
697: d'onboarding solide (`adding-a-family.md`, `contributing.md`) : un bon expert
698: doit pouvoir livrer une famille ou un noyau en quelques jours, pas en
699: quelques semaines.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '7357,7769p' session-ad321db8-5ba0-40bf-b50d-4be596739cef.md | grep -n '^### \\|^## \\|^| P' | head -60"
}
```

3:## Réponse courte
16:## 1. Ce que le dépôt montre
36:| Préfill | `gemv_many.wgsl` garde 32 tokens en vol, ce n’est pas un GEMM tuilé : **4,4×** derrière llama.cpp sur un dense 27B. |
55:## 2. Les profils d’experts
57:### Tableau de synthèse
59:| Profil | Priorité | Rareté | Ce qu’il débloque |
74:### P0 — 1. Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)
110:### P0 — 2. Architecte runtime d’inférence
143:### P0/P1 — 3. Ingénieur quantification et exactitude numérique
175:### P0/P1 — 4. Architecte Rust systèmes / backend
207:### P0 — 5. Ingénieur performance / qualification
238:### P1 — 6. Ingénieur intégration des architectures
255:### P1 — 7. Ingénieur tokenizers / templates / tool calls
271:### P1 — 8. Ingénieur serveur / API / observabilité
288:### P2 — 9. Ingénieur multimodal
300:### P2 — 10. Ingénieur build/release/bindings
312:## 3. Composition d’équipe recommandée
314:### Cellule « performance » minimale (5 personnes)
322:| Performance/QA | CI GPU, ledger, A/B | Sans mesure, pas de gain prouvé |
324:### Équipe complète (9–10 personnes)
328:### Séquencement
330:| Phase | Durée | Rôles | Objectif |
339:### Matrice chantier → pilote
356:## 4. Comment évaluer ces experts
381:## 5. Ce qu’il ne faut pas faire
392:## 6. Recommandation finale
413:## User


---

## Assistant

_Thinking:_

The docs are complete for the 10 profiles. Now we need to audit against the session for missing prompts/missions. Let's inspect the session's earlier lists and the final doc creation to identify what was intentionally omitted or possibly forgotten.

Key places:
- lines 5305-5434: "Project diagnosis" with 12 profiles (including technical writer, security).
- lines 6062-6298: 11 profiles (P0/P1/P2) + cross-cutting security.
- lines 6298-6453: "Profile X details" for 11 profiles.
- lines 7357-7769: final 10 profiles.
- lines 8443-8771: outline for docs/experts.md, fiches, prompts.
- lines 8771-8998: checks, writing.
- lines 9291 "Writing the files".
- lines 14206 "Fichiers créés".

Let's read the final "Fichiers créés" section to see what the previous assistant claimed. It might say 3 files. Also check if the docs were actually created in this session (they exist). The session transcript may have been generated by the previous assistant and then the docs were created in a later turn. Let's inspect lines 14206-14319.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '14190,14320p' session-ad321db8-5ba0-40bf-b50d-4be596739cef.md"
}
```


Need to be careful not to overclaim that the docs are "published" in mdBook; note they are internal, outside `docs/src`, as stated. Maybe say "I placed them as internal docs; if you want them in the mdBook, I can move them under docs/src and add SUMMARY entries."

Let's also mention the corrected facts:
- `Engine.free` actual type.
- Metal via Vulkan.
- `conformance.py` only cpu/cuda.
Maybe not necessary, but useful to show rigor.

The final answer should be in French. Keep it structured with headings and bullet points. Desired oververbosity 5: moderate detail. Since the work is large, a summary table is appropriate.

Let's draft:

```
J'ai créé les trois documents et intégré ta règle « un backend complet par API » comme décision structurante.

## Fichiers créés
| Fichier | Contenu |
...
`docs/experts.md` : analyse, diagnostic chiffré, 10 profils, composition d'équipe, RACI, commandes de mesure, recommandation.
`docs/experts/fiches-de-poste.md` : 10 fiches ...
`docs/experts/prompts-agents.md` : 10 prompts ...

## Ce que change ta remarque sur les backends
...
- Aujourd'hui CUDA est éclaté entre kinetic-cuda et kinetic-models/src/cuda; Vulkan entre kinetic-vulkan et kinetic-models/src/vulkan.
- Cible: un seul crate complet par API GPU. `kinetic-cuda` = driver + noyaux + graphes + mémoire + exécution + capacités. Idem Vulkan.
- `kinetic-models` ne contient plus de code CUDA/Vulkan; il expose un trait `ModelRunner` au niveau modèle.
- `kinetic-runtime` (à créer) = devices, capacités, factory, scheduler, pool KV, métriques; pas de types de modèle.
- `kinetic-server` garde `Backend` (requête, reuses, run_ahead, settle_some).
- Les autres backends (HIP, SYCL, WebGPU) sont des backends de compatibilité, opt-in, avec un contrat: probe, capabilities, conformance, mesure, isolement, CI. Metal reste un chemin Vulkan (MoltenVK).
- Un seul backend chargé à l'exécution.
- Migration en 7 étapes; le déplacement des exécuteurs n'est pas un simple git mv, il faut extraire `ModelRunner` et exposer les descripteurs.

## Points corrigés
- `Engine.free` = `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>`.
- `conformance.py` ne connaît que cpu/cuda.
- Les docs sont internes, hors mdBook.

## État
Aucun code modifié; seuls les 3 docs ajoutés. `PLAN.md` inchangé.

## Prochaines étapes possibles
1. Squelette `kinetic-runtime` + trait `ModelRunner` (premier pas concret).
2. ADR `docs/adr/...` ou page `docs/src/backend-architecture.md`.
3. Traduction anglaise / intégration mdBook.
4. Issues GitHub à partir des livrables 30/60/90.
```

Need to mention "j'accepte cela" — yes, done.

Could also mention the recommended team: 4-5 people minimum, GPU AMD first, runtime, QA, numerical, Rust backend. The docs contain it. Maybe one sentence.

Let's ensure the final answer doesn't claim the docs are in `docs/src`. Say "docs internes". Good.

Also, maybe mention the line counts: 699, 1048, 657. Not necessary but shows substance. Could include.

One more thing: The user might expect the docs to be in English because the repo is English. We can offer translation. Mention "Les documents sont en français; je peux les traduire en anglais si tu veux les publier dans le mdBook." Good.

Let's also mention that I did not modify `PLAN.md`; `git status` shows only untracked docs. Good.

Now, before final, let's double-check that no code was accidentally modified by the earlier `PLAN.md` mtime. `git status` shows only docs. Good.

Let's consider whether the docs' `docs/experts.md` link to `docs/experts/...` works from the repo root or from the file. The link `[docs/experts/fiches-de-poste.md](experts/fiches-de-poste.md)` from `docs/experts.md` resolves to `docs/experts/fiches-de-poste.md`. Good. The fiches link `../experts.md` resolves to `docs/experts.md`. Good. The prompts link `../experts.md` and `fiches-de-poste.md`; from `docs/experts/prompts-agents.md`, `fiches-de-poste.md` resolves to same dir. Good.

Potential issue: In `docs/experts.md` section 7, the label is `docs/experts/fiches-de-poste.md` but the link is `experts/fiches-de-poste.md`; fine.

Potential issue: `docs/experts.md` says "docs/src/contributing.md" etc. Those exist. Good.

Now, final answer in French. Keep it clear and concise. Use a table for files. Mention key architectural decision. End with next steps and a question.

Let's write the final answer.

J’ai créé les trois documents et intégré ta remarque sur les backends comme **décision structurante** dans `docs/experts.md`.

## Fichiers créés

| Fichier | Contenu |
|---|---|
| `docs/experts.md` | Diagnostic chiffré, 10 profils d’experts, priorités P0/P1/P2, composition d’équipe, RACI, commandes de mesure, recommandation finale |
| `docs/experts/fiches-de-poste.md` | 10 fiches de poste détaillées : mission, contexte, compétences, périmètre d’écriture, fichiers de contexte, commandes, livrables 30/60/90, critères de succès, signaux d’alerte |
| `docs/experts/prompts-agents.md` | 10 prompts d’agents spécialisés avec contrat commun, garde-fous, boucle de travail, commandes et format de rapport |

Les documents sont en français, en dehors du mdBook publié (`docs/src/`). Je peux les traduire en anglais et les intégrer au livre si tu veux les publier.

## Ta règle « un seul backend par API »

Elle est maintenant le point de départ de `docs/experts.md`, avec une distinction importante :

- **CUDA** : tout dans `kinetic-cuda` — driver, noyaux, layouts, graphes, mémoire, exécution du modèle, capacités.
- **Vulkan** : tout dans `kinetic-vulkan`, selon le même principe.
- **CPU** : reste le chemin de référence ; ses noyaux dans `kinetic-cpu`, son exécution générique dans `kinetic-models`.
- **`kinetic-models`** : plus de code CUDA/Vulkan ; il expose un trait `ModelRunner` au niveau modèle (charger, préfill, décoder, état, capacités).
- **`kinetic-runtime`** (à créer) : devices, capacités, budget mémoire, factory de backends, ordonnanceur, pool KV, métriques. Il ne connaît pas les types de modèle.
- **`kinetic-server`** : garde le trait `Backend` actuel (requête, `reuses`, `run_ahead`, `settle_some`) au-dessus de `ModelRunner`.
- **HIP, SYCL, WebGPU** : backends de compatibilité, opt-in, avec un contrat précis — découverte, capacités, conformance, mesure, isolement, CI. Metal reste un chemin Vulkan via MoltenVK, pas un backend séparé.
- **Un seul backend chargé à l’exécution**, selon la carte.

La migration est décrite en 7 étapes. J’ai explicitement corrigé l’idée que le déplacement des exécuteurs serait un simple `git mv` : il faut d’abord extraire `ModelRunner` et exposer les descripteurs de modèle nécessaires, sinon `kinetic-cuda` et `kinetic-vulkan` ne peuvent pas lire les structures internes de `kinetic-models`.

## Corrections factuelles intégrées

- `Engine.free` est bien `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>`, pas `Mutex<Vec<Box<dyn Backend>>>`.
- `conformance.py` ne connaît aujourd’hui que `cpu` et `cuda` ; l’extension Vulkan est un livrable QA.
- Les chemins `ollama` sont des répertoires, pas des fichiers.
- Les docs sont marqués comme internes, hors du mdBook.

## État du dépôt

Aucun code modifié. `git status` ne montre que les nouveaux documents :

```text
?? docs/experts.md
?? docs/experts/
```

`PLAN.md` est inchangé.

## Prochaines étapes possibles

1. **Commencer le refactor** : créer le squelette `kinetic-runtime` + le trait `ModelRunner`, sans changer le comportement.
2. **Écrire un ADR** : `docs/adr/0001-one-backend-per-api.md` pour figer la décision.
3. **Créer les issues GitHub** à partir des livrables 30/60/90 des fiches.
4. **Traduire et publier** les docs dans le mdBook.

Si tu veux, je peux enchaîner sur le point 1 : une première PR minimale avec `kinetic-runtime` et `ModelRunner`, tests verts, sans toucher aux noyaux.

---

## User



**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '8435,8790p' session-ad321db8-5ba0-40bf-b50d-4be596739cef.md"
}
```

49-        eprintln!("kinetic: running on the CPU; {}", skipped.join("; "));
50-    }
51-    Ok(None)
52-}


---

## Assistant

_Thinking:_

Good, `--device vulkan` works.

Now, let's plan the files and content. The user wants:
1. `docs/experts.md` — the analysis/document.
2. Fiches de poste.
3. Prompts d'agents spécialisés with scope, context files, measurement commands.

We should create a coherent set:
- `docs/experts.md` — main document (French), includes the updated architecture principle (one backend per API), expert summary, team composition, links to fiches.
- `docs/experts/README.md` — index of fiches and prompts.
- `docs/experts/gpu-amd.md`
- `docs/experts/runtime.md`
- `docs/experts/rust-backend.md`
- `docs/experts/quantization.md`
- `docs/experts/performance-qa.md`
- `docs/experts/model-integration.md`
- `docs/experts/tokenizer.md`
- `docs/experts/server-api.md`
- `docs/experts/multimodal.md`
- `docs/experts/build-release.md`
- `docs/experts/agent-prompts.md` — ready-to-use prompts for each agent, with scope/context/commands.

That's 12 files. Is that too much? The user accepted all three deliverables. We can make each fiche concise (~100-150 lines) and the prompts file ~300 lines. The main doc ~300 lines. Total ~2000 lines. That's a lot of writing but doable. We need to be efficient and accurate.

Maybe we can reduce to:
- `docs/experts.md` — main doc with all profiles, skills, scope, context, commands.
- `docs/experts/fiches/` — 10 job descriptions (one page each).
- `docs/experts/agent-prompts.md` — 10 prompts.
But that's still many files. Alternatively:
- `docs/experts.md` — everything (analysis + fiches + prompts) in one long document. The user asked for "document docs/experts.md, fiches de poste, ou prompts d'agents" — they may accept one file with all three sections. But they said "j'accepte cela" to all three. A single file with three parts is simpler and less redundant. However, job descriptions and agent prompts are distinct artifacts; a folder is more usable.

Let's choose a balanced structure:
- `docs/experts.md` — main analysis, target architecture, expert profiles summary, team composition, RACI, evaluation, common measurement commands. This is the document.
- `docs/experts/fiches-de-poste.md` — all 10 job descriptions in one file, with a template per role. This avoids 10 tiny files and keeps it maintainable.
- `docs/experts/prompts-agents.md` — all 10 agent prompts in one file, each with scope, context files, commands, guardrails.

This is 3 files, matches the three deliverables, and is easy to navigate. We can also add `docs/experts/README.md` as an index, but the main doc can serve as index. Maybe create `docs/experts/README.md` with links to the three files. That's 4 files. Good.

Let's decide on names in French or English. The repo docs are in English, but the user communicates in French and the PLAN/llama-kinetic docs are in French. Since the user asked in French, write the docs in French. File names can be English or French; use `docs/experts.md`, `docs/experts/fiches-de-poste.md`, `docs/experts/prompts-agents.md`. The main doc in French.

Need to ensure the docs are accurate and don't contradict the repo. Let's craft the content carefully.

### Main doc `docs/experts.md`

Outline:

```
# Experts pour l'architecture et la performance de Kinetic

> Statut : analyse, 22 nov. 2026. Snapshot : main ... 
> Portée : fork local Angra974/kinetic (RX 9070 XT, Vulkan/ROCm), pas le plan CUDA upstream.

## 0. Résumé
- 4 profils P0...
- Principe architectural : un backend complet par API.

## 1. Diagnostic
### 1.1 Ce que le dépôt fait bien
### 1.2 Les goulets mesurés
(table)
### 1.3 Les goulets structurels
(table)
### 1.4 Conséquence : quels experts, pas quels frameworks

## 2. Architecture cible : un backend complet par API
### 2.1 Principe
### 2.2 État actuel vs cible
(table)
### 2.3 Migration incrémentale
(steps)
### 2.4 Règle pour les backends de compatibilité
### 2.5 Ce que ça change pour les experts

## 3. Profils d'experts
### 3.1 Tableau de synthèse
### 3.2 P0 — GPU AMD
### 3.3 P0 — Runtime
### 3.4 P0/P1 — Rust backend
### 3.5 P0/P1 — Quantification/exactitude
### 3.6 P0 — Performance/QA
### 3.7 P1 — Intégration modèles
### 3.8 P1 — Tokenizers
### 3.9 P1 — Serveur/API
### 3.10 P2 — Multimodal
### 3.11 P2 — Build/release/bindings

For each: mission, skills, why, scope, context files, commands, deliverables, KPI.

## 4. Composition et séquencement
### 4.1 Cellule minimale
### 4.2 Équipe complète
### 4.3 Phases
### 4.4 RACI

## 5. Évaluation et recrutement
### 5.1 Exercices
### 5.2 Signaux d'alerte
### 5.3 Rareté et sourcing

## 6. Commandes de mesure communes
### 6.1 Build/test
### 6.2 Inventaire
### 6.3 Bench
### 6.4 Conformance/exactitude
### 6.5 Serveur
### 6.6 Précautions

## 7. Livrables
- fiches-de-poste.md
- prompts-agents.md

## 8. Ce qu'il ne faut pas faire
```

This is comprehensive. The per-profile sections can be shorter than the previous answer, with a link to the fiche for details. But the user wants the document to contain the analysis; we can include a compact profile table and then the detailed fiches in the separate file. To avoid duplication, the main doc can have:
- Summary table.
- For each P0: a half-page summary (mission, 5 skills, why, 3 deliverables, KPI).
- For P1/P2: 3-4 bullets.
- Link to fiche for full details.

Then `fiches-de-poste.md` has the full job descriptions with context, scope, evaluation.

Then `prompts-agents.md` has ready-to-use prompts.

This is a good separation of concerns.

### Architecture section details

Let's write the target architecture precisely.

**Current state:**

| Couche | CUDA | Vulkan | CPU |
|---|---|---|---|
| Driver/kernels | `kinetic-cuda` (8.5k) | `kinetic-vulkan` (5.9k) | `kinetic-cpu` (6.4k) |
| Exécution modèle | `kinetic-models/src/cuda` (4.5k) | `kinetic-models/src/vulkan` (2.2k) | `kinetic-models/src/model.rs` + `kinetic-server/src/cpu.rs` |
| Modèle/arch | `kinetic-models/src/arch`, `blocks` | same | same |
| Serveur | `kinetic-server/src/cuda.rs` | `kinetic-server/src/vulkan.rs` | `kinetic-server/src/cpu.rs` |

**Problem**: CUDA functionality is split between `kinetic-cuda` and `kinetic-models/src/cuda`; Vulkan between `kinetic-vulkan` and `kinetic-models/src/vulkan`. This is not "one backend per API"; it's two layers per API, and the model crate has backend-specific features. Adding HIP would create `kinetic-hip` + `kinetic-models/src/hip`.

**Target:**

```
kinetic-models        (arch, blocks, speculative, warm, shortlist, convoy; no backend feature)
        │
kinetic-runtime       (traits: Backend, Device, Graph, Kernel, Capabilities, Memory; scheduler-agnostic)
        │
├── kinetic-cpu       (all CPU: SIMD kernels, thread pool, executor)
├── kinetic-cuda      (all CUDA: driver, kernels, layouts, executor, graphs, layer split, MoE)
├── kinetic-vulkan    (all Vulkan: loader, WGSL/SPIR-V, executor, graphs, coopmat, async)
└── kinetic-hip       (compatibility, all HIP, added when a machine/CI can validate it)
```

**Rules:**
- `kinetic-models` has no `cfg(feature = "cuda"|"vulkan")`, no `cuda/` or `vulkan/` modules. It describes the model and the semantics; it calls the runtime traits.
- Each backend crate owns everything for its API: device discovery, memory, kernels, graph capture, executor, capability declaration, error mapping.
- `kinetic-server` selects exactly one backend at runtime (`auto|cpu|cuda|vulkan|hip`) and does not link the others' code paths into the hot path. Features can still compile them in, but the runtime uses one.
- Compatibility backends (HIP, Metal, SYCL, WebGPU) implement the same trait; they are added only when there is a machine/CI to validate them, per `PLAN.md`.
- The `Backend` trait must be rich enough to express: device capabilities, supported weight types, graph capture, async submission, memory placement (layer/expert/tensor), KV cache, sampling on device, and error handling. It must not leak CUDA/Vulkan types.
- Keep `Target` (speculative) and `Family` (architectures) as they are; they are good seams.

**Migration steps:**
1. Create `kinetic-runtime` with traits and capability types; no behavior change.
2. Move `kinetic-models/src/cuda` into `kinetic-cuda/src/executor` (mechanical move, keep tests).
3. Move `kinetic-models/src/vulkan` into `kinetic-vulkan/src/executor`.
4. Remove `cuda`/`vulkan` features from `kinetic-models`; `kinetic-models` depends only on `kinetic-runtime`.
5. Add a `BackendFactory` in `kinetic-server`/`kinetic-cli` that picks one backend; remove backend-specific branches from the engine.
6. Add a `TestBackend` (CPU or mock) to test `kinetic-models` without GPU.
7. Only then optimize: the refactor is mechanical first; perf work after.
8. Add HIP/Metal/SYCL as separate crates implementing `kinetic-runtime`; no changes to `kinetic-models`.

**Non-goals:**
- No full graph compiler / MLIR.
- No dynamic plugin ABI at first; compile-time features are fine.
- No backend-specific code in `kinetic-models`.
- No "universal kernel" that is slower on all backends; each backend optimizes its own path.

**Why this matters for experts:**
- The Rust/backend architect owns this migration.
- The GPU AMD expert works inside `kinetic-vulkan` and `kinetic-hip`, not in `kinetic-models`.
- The runtime architect works in `kinetic-runtime` and `kinetic-server`, not in backend crates.
- The quantization expert works in `kinetic-quant` and the kernel crates, with a shared reference.
- The model integration expert works only in `kinetic-models/src/arch` and `blocks`, with no backend code.

This is a clean separation of responsibilities.

### Expert profiles: scope and context

Let's define for each:

**GPU AMD**
- Scope: `crates/kinetic-vulkan/**`, `crates/kinetic-hip/**` (new), `crates/kinetic-runtime/src/{device,graph,kernel}.rs` (API proposals), `crates/kinetic-models/src/arch/*` read-only.
- Context: `PLAN.md` §4 P0.2/P0.3, `llama-kinetic.md` §7.2/§7.3/§7.10/§7.11, `crates/kinetic-vulkan/src/{commands,pipeline,gemv,attention,moe}.rs`, `shaders/*.wgsl`, `crates/kinetic-models/src/vulkan/*`, `docs/src/devices.md`.
- Commands: build with `--features vulkan`; `kinetic bench ... --device vulkan --json -r 6`; `llama-bench -dev Vulkan0 ...`; `cargo test --release -p kinetic-vulkan`; `RGP` capture; `rocm-smi`.
- KPI: pp512/pp2048, tg128, bit-exact, GB/s/TFLOPS, startup.
- 30/60/90.

**Runtime**
- Scope: `crates/kinetic-runtime/**` (new scheduler), `crates/kinetic-server/src/{engine,sequence,backend}.rs`, `crates/kinetic-models/src/{convoy,memory,per_layer}.rs`, `warm/`.
- Context: `PLAN.md` §4 P1.5/P1.6, `llama-kinetic.md` §7.6/§7.9, `engine.rs`, `sequence.rs`, `backend.rs`, `convoy.rs`, `warm/`, `server.rs`.
- Commands: `kinetic serve --parallel N`; a load generator (maybe `scripts/` doesn't have one; propose `oha`/`wrk` or a small Python script); `python scripts/same_answer_demo.py`; `python scripts/server_abuse.py`; `kinetic bench`.
- KPI: c1/c4/c16/c64, p99, TTFT, cache hit, exactness.

**Rust backend**
- Scope: workspace layout, `kinetic-runtime`, `kinetic-models` (remove backend modules), `kinetic-cuda`, `kinetic-vulkan`, build scripts, feature flags.
- Context: `Cargo.toml`, crate `lib.rs`, `docs/src/contributing.md`, `docs/src/building.md`, `kinetic-models/src/{lib,model}.rs`, `kinetic-cuda/src`, `kinetic-vulkan/src`.
- Commands: `cargo build --release --workspace --all-features`; `cargo test --release --workspace`; `cargo clippy ...`; `cargo tree`; `tokei`/`wc`; `perf`/flamegraph.
- KPI: duplication, build time, startup, unsafe audit, API stability.

**Quantization**
- Scope: `crates/kinetic-quant/**`, `crates/kinetic-gguf/**`, `crates/kinetic-cpu/src/kdot/**`, backend kernel tests, `docs/src/models.md`.
- Context: `docs/src/contributing.md` "Adding a weight format", `kinetic-quant/src/{types,dequant,encode}`, `kinetic-cpu/src/kdot/tests.rs`, `llama-kinetic.md` §7.7, `PLAN.md` P0.4.
- Commands: `cargo test --release -p kinetic-quant`, `cargo test --release -p kinetic-cpu`, `cargo test --release -p kinetic-vulkan`, `kinetic quantize`, `kinetic perplexity`, conformance.
- KPI: quality, speed, memory, identity.

**Performance/QA**
- Scope: `scripts/**`, `.github/workflows/**`, `PLAN.md`, `THEORY.md` (new), `ledger.jsonl` (new), `configs/` (new), `best-config.json` (new).
- Context: `PLAN.md` §2/§7, `llama-kinetic.md` §7.10-7.12, `docs/src/benchmarks.md`, `scripts/*.py`, CI workflows.
- Commands: all measurement commands; `rocm-smi`; `python scripts/bench_protocol.py`; `python scripts/conformance.py`; `python scripts/server_abuse.py`; `python scripts/same_answer_demo.py`.
- KPI: noise, false pos/neg, detection time, gate coverage.

**Model integration**
- Scope: `crates/kinetic-models/src/arch/**`, `blocks/**`, `docs/src/models.md`, `scripts/new_family.py`, tests.
- Context: `docs/src/adding-a-family.md`, `llama-kinetic.md` §4.1/§5.1/§6, `FAMILIES`, `arch/transformer.rs`, `blocks/`.
- Commands: `kinetic info`, `kinetic run --device cpu/vulkan`, `python scripts/conformance.py`, `cargo test --release -p kinetic-models`.
- KPI: load success, token match, coverage.

**Tokenizer**
- Scope: `crates/kinetic-tokenizer/**`, `crates/kinetic-server/src/{template,tool_calls,tool_grammar}.rs`, `docs/src/models.md`.
- Context: `llama-kinetic.md` §4.2/§5.3, `tokenizer` crate, `scripts/conformance.py`, `llama-tokenize`.
- Commands: `kinetic tokenize` (if exists? `llama.rs` has `/tokenize`; CLI maybe not). Use `cargo test -p kinetic-tokenizer`; a corpus comparison script. `llama-tokenize -m model.gguf -f corpus.txt`.
- KPI: identical ids, special tokens, tool calls.

**Server/API**
- Scope: `crates/kinetic-server/src/{server,http,api,llama,ollama,metrics,template,tool_calls}.rs`, `docs/src/server.md`, `docs/src/clients.md`.
- Context: `llama-kinetic.md` §4.5/§5.9, `PLAN.md` P0.1, `scripts/llama_server_compat.py`, `scripts/server_abuse.py`.
- Commands: `kinetic serve`, `curl`, `python scripts/llama_server_compat.py`, `python scripts/server_abuse.py`, `python scripts/same_answer_demo.py`.
- KPI: route coverage, conformance, overhead, stability.

**Multimodal**
- Scope: `crates/kinetic-vision/**`, `kinetic-audio` (new), `kinetic-models/src/input.rs`, `docs/src/models.md`.
- Context: `llama-kinetic.md` §4.10/§5.2, `docs/src/models.md` vision section, `kinetic-vision/src/lib.rs`, `preprocess.rs`, `ops.rs`.
- Commands: `cargo run --release -p kinetic-vision --example encode -- <mmproj> rainbow:<size>`, `llama-mtmd-debug`, `kinetic serve --mmproj`.
- KPI: embedding parity, answer quality.

**Build/release**
- Scope: `.github/workflows/**`, `docker/**`, `packaging/**`, `bindings/**`, `crates/kinetic-capi/**`, `docs/book.toml`.
- Context: `docs/src/building.md`, `docs/src/bindings.md`, CI workflows, `Cargo.toml`, `docker/`.
- Commands: `cargo build --release --locked --workspace --all-features`, `cargo test`, `maturin build`, `docker build`, `cargo install`.
- KPI: ABI coverage, build time/size, install success.

### Job descriptions

We can write a template:

```
# Fiche de poste — <titre>

**Statut** : P0/P1/P2
**Mission** : ...
**Rattachement** : ...
**Contexte** : ...
**Responsabilités** :
- ...
**Compétences requises** :
- ...
**Compétences appréciées** :
- ...
**Périmètre d'écriture** :
- ...
**Fichiers de contexte** :
- ...
**Commandes de mesure** :
- ...
**Livrables 30/60/90** :
- ...
**Critères de succès** :
- ...
**Interactions** :
- ...
**Profil / sourcing** :
- ...
**Signaux d'alerte** :
- ...
```

For each of the 10 roles. That's ~10 × 60 lines = 600 lines. Fine.

### Agent prompts

A template:

```
## Agent <nom>

**Rôle** : ...
**Périmètre d'écriture** : ...
**Lecture obligatoire** : ...
**Interdits** : ...
**Boucle de travail** :
1. Lire ...
2. Mesurer ...
3. Proposer ...
4. Modifier ...
5. Mesurer ...
6. Consigner ...
**Commandes** : ...
**Format de sortie** : ...
**Critère d'arrêt** : ...
```

Each prompt should be ready to paste into a subagent. Include:
- Role and mission.
- Context files to read first.
- Write scope (exact paths).
- Guardrails (one change at a time, exactness, no network, no copied code, don't touch other backends).
- Measurement protocol (control/candidate/control, median of ≥6, ledger).
- Output format (PLAN.md entry, config JSON, diff summary).
- Definition of done (KPI + gates).
- Escalation (when to stop and ask).

We can write 10 prompts. This is useful.

### Need to update PLAN.md?

The user didn't ask to modify PLAN.md. But the architecture correction is important; maybe add a note to `PLAN.md`? The user said "ne devrait avoir qu'un seul backend..." It's a design decision. We can include it in `docs/experts.md` and perhaps add a line to `PLAN.md` §4 P0.2/P0.3? But modifying PLAN.md unasked might be okay since it's a journal. However, the user asked for docs/experts, fiches, prompts. We should not modify PLAN.md unless necessary. We can mention in the final response that the architecture principle is documented in `docs/experts.md` and can be added to PLAN.md if they want. Maybe we can add a small "Décision d'architecture" note in `docs/experts.md` only.

### Check if `docs/experts/` directory exists

No. We'll create it.

### Language and style

Use French, consistent with the user. Technical terms in English where standard. Use tables, bullet lists, code blocks. Keep the docs precise and actionable.

### Before writing, let's inspect `scripts/llama_server_compat.py` and `llama_presets.py` to give accurate commands.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && for f in scripts/llama_server_compat.py scripts/llama_presets.py scripts/new_family.py; do echo \"--- $f\"; sed -n '1,45p' \"$f\"; done"


---

## Assistant

_Thinking:_

The session includes detailed analysis and drafting. Now, our task: audit final docs vs session. We have enough.

Let's systematically identify gaps/inconsistencies:

1. **10 profiles vs 12 earlier**:
   - Session early list (lines 5335-5434) had 12: GPU AMD, HIP/ROCm, Runtime, Rust, Quantization, Model integration, Server/API, Multimodal, Performance/QA, Build/release, Technical writer (optional), Security?.
   - Final list (lines 7357-7769) has 10: merged HIP into GPU AMD, merged Model integration/Tokenizers? Actually final splits Model integration and Tokenizers, and Server/API and Build/release. It omits technical writer and security. The docs have 10 and mention security as transverse.
   - So no missing profile from the final list. But if the user expected the earlier 12, two are missing: technical writer and dedicated security. The docs explicitly say security is transverse, not a post. Technical writer is not mentioned at all. Is that a gap? Maybe. We should flag: "Le profil 'Technical writer / DevRel' évoqué en début de session n'a pas été retenu dans la liste finale; aucun prompt. Si tu le veux, il manque." Also "Sécurité" is transverse but no fiche/prompt; the docs say covered by server/Rust/QA. If the user wants a dedicated security expert, it's missing.

2. **HIP/ROCm**:
   - Session early list had a separate "HIP/ROCm backend engineer (top priority locally, second wave)" (#2). Final merged into GPU AMD (#1) "puis HIP/ROCm". The fiche 1 covers HIP. No separate prompt. If the user expected a dedicated HIP prompt, it's missing, but it was intentionally merged. Flag as "fusionné, pas oublié".

3. **KV cache deep-dive**:
   - The session's last analysis (lines 21307-21479) produced a detailed KV analysis and recommended the runtime architect profile. The final docs include KV in the runtime fiche and GPU/quant fiches, but there is **no dedicated KV cache document or prompt**. If the user's goal is "all prompts for experts and missions", the KV mission is partially covered by Agent 2 (runtime) and Agent 1/4. But the deep analysis itself is not a separate mission/prompt. Flag as "partiellement couvert; pas de prompt dédié KV". This is probably the most important gap given the user's last question.

4. **Missions in the session not in the docs**:
   - "Cellule performance minimale (5 personnes)" -> docs section 3.1 yes.
   - "Équipe complète (9–10)" -> yes.
   - "Séquencement" -> yes.
   - "Matrice chantier → pilote" -> docs RACI yes (slightly different but covers).
   - "Comment évaluer ces experts" -> docs section 4.1 yes.
   - "Ce qu'il ne faut pas faire" -> docs section 6 yes.
   - "Recommandation finale" -> docs section 8 yes.
   - "Commandes de mesure communes" -> docs section 5 yes.
   - "Livrables" -> docs section 7 links, but no consolidated "Livrables" section? The session outline had "## 7. Livrables". The docs section 7 is "Documents associés", not "Livrables". The fiches have 30/60/90. Maybe a minor gap: no consolidated deliverables table. But not critical.
   - "RACI" -> yes.
   - "Phases" -> yes.
   - "Sourcing" -> yes.
   - "Exercices" -> yes.
   - "Signaux d'alerte" -> yes.

5. **Consistency of role order**:
   - Session final table: 1 GPU, 2 Runtime, 3 Quantization, 4 Rust, 5 Performance, 6 Model, 7 Tokenizers, 8 Server, 9 Multimodal, 10 Build.
   - Docs `experts.md`: 2.3 Runtime, 2.4 Rust, 2.5 Quantization. So Rust and Quantization swapped. Fiches: 3 Rust, 4 Quantization. This is not a missing item, but it's an inconsistency with the session's final recommendation. Could be intentional (Rust before quantization because foundation). Flag as minor.
   - The prompts follow the fiches order. Fine.

6. **Prompt completeness**:
   - Each prompt has role, mission, required reading, scope, forbidden, loop, commands, success, final report. Good.
   - But the prompts do **not** include the common contract inline; it's in a separate section at the top. That's fine, but if an agent is given only the prompt section, it might miss the common contract. The docs say "Tout agent Kinetic doit respecter les règles suivantes, même si son prompt spécifique ne les répète pas." Good.
   - The prompts do not include a "definition of done" with exact test commands for every role? They do.
   - The prompts do not include "stop conditions" explicitly; the common contract has "Arrêt". Good.
   - The prompts do not include "report format" for all? Each has "RAPPORT FINAL". Good.

7. **Missions in fiches**:
   - Each fiche has a "Mission" line. Good.
   - But some fiches' "Responsabilités" are broad and may overlap; not a gap.
   - Fiche 1's "Périmètre d'écriture" includes `crates/kinetic-models/src/runner/**` (to create) — but per the architecture, the runner should be in `kinetic-models` (trait) and executors in backend crates. The fiche says "crates/kinetic-models/src/runner/** (à créer : trait ModelRunner et runner CPU)" — that's consistent with the architecture (trait + CPU runner in kinetic-models). Good.
   - Fiche 1 also says `crates/kinetic-runtime/src/{device,capabilities,factory,memory}.rs` (propositions d'API uniquement). Good.
   - Fiche 3 says `crates/kinetic-models/src/{lib,model}.rs` (retrait des modules backend) and `crates/kinetic-cuda/**`, `crates/kinetic-vulkan/**` (accueil des exécuteurs). Good.
   - Fiche 3's "Périmètre d'écriture" includes `crates/kinetic-server/src/{backend,lib}.rs`, but the architecture also requires a factory in `kinetic-cli`; the prompt includes `kinetic-server` / `kinetic-cli`. Minor.
   - Fiche 2's scope includes `crates/kinetic-models/src/{convoy,memory,per_layer}.rs`, `warm/**`; good.
   - Fiche 4's scope includes `kinetic-cuda/tests/**` and `kinetic-vulkan/tests/**`; good.
   - Fiche 5's scope includes `scripts/**`, `.github/workflows/**`, `PLAN.md`, `THEORY.md`, `ledger.jsonl`, `configs/`, `best-config.json`; good.
   - Fiche 6's scope includes `arch/**`, `blocks/**`, `docs/src/models.md`, `scripts/new_family.py`; good.
   - Fiche 7's scope includes tokenizer, server template/tool_calls/tool_grammar; good.
   - Fiche 8's scope includes server files, docs; good.
   - Fiche 9's scope includes vision, audio, `input.rs`, docs; good.
   - Fiche 10's scope includes workflows, docker, packaging, bindings, capi, docs; good.

8. **Context files accuracy**:
   - Fiche 1: `crates/kinetic-models/src/vulkan/*` (to move to kinetic-vulkan). Good.
   - Fiche 1: `crates/kinetic-quant/src/{types,dequant}.rs` for formats. Good.
   - Fiche 2: `crates/kinetic-models/src/{convoy,memory,per_layer}.rs`. `per_layer.rs` exists? Let's check. `crates/kinetic-models/src/per_layer.rs` maybe. Yes, likely.
   - Fiche 3: `crates/kinetic-models/src/{cuda,vulkan}/**`; `crates/kinetic-cuda/src/**`, `kinetic-vulkan/src/**`; good.
   - Fiche 4: `crates/kinetic-quant/src/{types,dequant,encode}/**`; `encode` is a directory? Let's check. If `encode` is a module file, `encode/**` may be wrong. `crates/kinetic-quant/src/encode` likely a directory. We can check.
   - Fiche 5: `scripts/bench_protocol.py`, etc. exist. Good.
   - Fiche 6: `docs/src/adding-a-family.md` exists? Let's check. It was in the docs. Good.
   - Fiche 7: `crates/kinetic-server/src/{template,tool_calls}.rs`; `tool_grammar.rs` exists? The fiche scope includes it; likely. Good.
   - Fiche 8: `crates/kinetic-server/src/ollama/`; good.
   - Fiche 9: `crates/kinetic-vision/examples/encode.rs`; good.
   - Fiche 10: `crates/kinetic-capi/**`; good.

9. **Commands accuracy**:
   - Fiche 1: `cargo build --release --locked -p kinetic-cli --features vulkan`; good. `cargo test --release --locked -p kinetic-vulkan`; good. `cargo test --release --locked -p kinetic-models`; good. `kinetic bench MODEL.gguf --device vulkan --json -r 6`; good. `llama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6`; good. `RGP capture sur pp512`; okay.
   - Fiche 2: `kinetic serve ... --parallel 4`; good. `oha ...`; external tool, okay. `same_answer_demo.py`; good. `server_abuse.py`; good.
   - Fiche 3: `cargo bloat` requires cargo-bloat installed; okay. `perf record`; okay.
   - Fiche 4: `kinetic quantize MODEL.gguf --help`; does `quantize` exist? Let's check CLI. `kinetic quantize` likely exists. `kinetic perplexity` likely exists. Good.
   - Fiche 5: `python scripts/bench_protocol.py ...`; good. `python scripts/conformance.py ... --device cpu`; good. `python scripts/server_abuse.py -- target/release/kinetic serve ...`; good. `same_answer_demo.py`; good.
   - Fiche 6: `kinetic info`, `kinetic run`, conformance; good.
   - Fiche 7: `llama-tokenize -m MODEL.gguf -f corpus.txt`; good.
   - Fiche 8: `llama_server_compat.py`; good. `server_abuse.py`; good.
   - Fiche 9: `cargo run --release -p kinetic-vision --example encode -- mmproj.gguf rainbow:512`; good. `llama-mtmd-debug`; external.
   - Fiche 10: `maturin build`, `docker build`; good.

10. **Potential factual issues**:
    - `docs/experts.md` says "5 cibles contre ~30" quantization; likely from session. Fine.
    - "21/180 architectures", "13/68 prétokenizers", "~15/45 routes", "2/73 projecteurs"; from session. Fine.
    - "KV non quantifié: `KvCache` en f16, pas de q8_0/q4_0" — this is Vulkan-specific; CUDA has q8_0. The table row says "KV non quantifié | `KvCache` en f16, pas de q8_0/q4_0 | GPU AMD + quantification". If read globally, it's inaccurate because CUDA supports q8_0. The fiche 1 context says "KvCache est en f16, non quantifié" — in the Vulkan context, okay. But the table should say "KV Vulkan non quantifié" to avoid confusion. This is a factual precision issue. Flag it.
    - `docs/experts.md` says "Préfill Vulkan 261.9 vs 1154.2 ... 4.4×". 1154.2/261.9 = 4.407; good.
    - "MoE 55.7 vs 181.9 = 3.3×; 489 vs 2637 = 5.4×"; good.
    - "décodage Vulkan à 95% de llama.cpp (38.4 vs 40.1)" — from session; good.
    - "Le CPU et Vulkan ignorent `--cache-type-k/v`" — correct.
    - "CUDA q8_0" — correct.
    - "Metal via Vulkan" — correct.
    - "conformance.py only cpu/cuda" — correct.
    - "Engine.free" — correct.
    - "MAX_CONNECTIONS=64" — correct.
    - "PREFILL_CHUNK=256" — correct.
    - "PROMPT_BATCH=128" — correct.
    - "gemv_many.wgsl garde 32 tokens en vol" — correct.
    - "KvCache f16" — Vulkan; CPU f32. The table should specify Vulkan.
    - "5 cibles contre ~30" — from session; okay.

11. **Missing "missions" in prompts**:
    - Agent 1 mission: "combler l'écart de préfill Vulkan, débloquer le MoE, aligner Vulkan sur CUDA, porter sur HIP/ROCm." Good.
    - Agent 2 mission: "faire passer le serveur d'un modèle N backends à un ordonnanceur token-level..." Good.
    - Agent 3 mission: "réduire la duplication CUDA/Vulkan, faire de chaque API un backend complet, préparer HIP/SYCL/WebGPU..." Good.
    - Agent 4 mission: "étendre la quantification et garantir l'exactitude." Good.
    - Agent 5 mission: "rendre la mesure automatisée, reproductible et infalsifiable." Good.
    - Agent 6 mission: "couvrir les architectures manquantes par briques réutilisables." Good.
    - Agent 7 mission: "atteindre la compatibilité des tokenizers et templates." Good.
    - Agent 8 mission: "parité API, robustesse, observabilité." Good.
    - Agent 9 mission: "vision, audio, OCR." Good.
    - Agent 10 mission: "rendre le produit installable et intégrable." Good.

12. **Missing "KV cache" mission**:
    - The runtime agent's mission is broad; it includes KV. But no dedicated "KV cache" prompt. If the user wants a deep analysis, the runtime agent is the right lead, but the prompt should explicitly include the KV audit. The fiche 2 responsibilities include "Introduire un pool KV paginé..." and the prompt's loop includes "pool KV". So it's covered, but not as a standalone mission. We can recommend adding a dedicated section to Agent 2's prompt: "Premier chantier: audit KV, métriques, design paged KV, q8_0 Vulkan, shared KV zero-copy." This would make the KV mission explicit.

13. **Missing "security" prompt**:
    - Docs say security is transverse, not a dedicated post. The prompts for Agent 8 (server) and Agent 3 (Rust backend) mention security indirectly? Agent 8's fiche responsibilities include "Sécuriser les entrées"; prompt's "INTERDITS" doesn't mention security. Agent 3's prompt mentions unsafe audit. Could be enough. Flag as "pas de prompt dédié; couvert par 3 et 8".

14. **Missing "technical writer" prompt**:
    - Not in final docs. If the user wants all experts from the session, it's missing. But the final recommendation explicitly says 10 profiles and no tech writer. We can say "non retenu, donc pas un oubli si tu valides la liste à 10."

15. **"Missions" vs "fiches" count**:
    - 10 fiches, 10 prompts, 10 profiles. Good.
    - The session's "11. Technical writer" and "12. Security" were exploratory; not final. Good.

16. **Links and file names**:
    - `docs/experts.md` links to `experts/fiches-de-poste.md` and `experts/prompts-agents.md`; these exist. Good.
    - `docs/experts/fiches-de-poste.md` links to `../experts.md`; good.
    - `docs/experts/prompts-agents.md` links to `../experts.md` and `fiches-de-poste.md`; good.
    - No `README.md` in `docs/experts/`; not required.

17. **Potential duplication / drift**:
    - The main doc's profile sections and the fiches duplicate content. This is intentional (summary vs detailed), but it creates drift risk. Flag: "deux sources de vérité; la fiche est la référence, le doc principal doit rester un résumé." Good.
    - The prompts duplicate the fiches' scope/commands. Again intentional, but drift risk. Recommend a single source (e.g., generate prompts from fiches or add a check).

18. **"PLAN.md" entries**:
    - The prompts tell agents to write to `PLAN.md` and `ledger.jsonl`, but `ledger.jsonl` and `configs/` do not exist yet. The QA agent is tasked to create them. Other agents are told to write to them before they exist. This is a bootstrapping issue: Agent 1/2/3/4 prompts say "PLAN.md, ledger.jsonl, configs/" but the ledger doesn't exist. They can create it, but the QA agent owns it. Could cause conflicts. Flag: "les prompts supposent `ledger.jsonl` et `configs/`; ils n'existent pas encore. Prévoir que l'Agent 5 les crée en premier, ou que chaque agent crée une entrée si absent." This is a real operational gap.
    - `THEORY.md` doesn't exist; Agent 5 creates it. Good.
    - `best-config.json` doesn't exist; Agent 5 creates it. Good.
    - `docs/experts/` exists now. Good.

19. **"Write scope" conflicts**:
    - Agent 1 scope includes `PLAN.md`, `ledger.jsonl`, `configs/`; Agent 2 same; Agent 3 same; Agent 4 same; Agent 5 same. Multiple agents writing `PLAN.md` concurrently could conflict. The common contract says one change per trial, but no locking. Flag: "écriture concurrente dans PLAN.md/ledger; prévoir un format d'entrée et un merge policy, ou un seul agent QA qui consolide."
    - Agent 3 scope includes `crates/kinetic-models/src/{lib,model}.rs` and `crates/kinetic-cuda/**`, `kinetic-vulkan/**`; Agent 1 scope includes `crates/kinetic-vulkan/**`; Agent 4 includes `crates/kinetic-vulkan/tests/**`; overlaps. Need coordination. The docs' RACI helps but prompts don't mention file locks. Flag as "overlaps à coordonner".
    - Agent 6 scope `arch/**`, `blocks/**`; Agent 2 scope `convoy,memory,per_layer`, `warm/**`; no overlap. Good.
    - Agent 7 scope `server/src/{template,tool_calls,tool_grammar}.rs`; Agent 8 scope `server/src/{server,http,api,llama,metrics}.rs`, `ollama/`; no overlap except `template` maybe. Good.
    - Agent 9 scope `kinetic-models/src/input.rs`; Agent 6 scope `arch/**`, `blocks/**`; no overlap.
    - Agent 10 scope workflows/docker/packaging/bindings/capi; Agent 5 scope workflows; overlap on `.github/workflows/**`. Need coordination: QA owns CI GPU workflows, Build owns release workflows. Flag.

20. **"Backend rule" consistency in fiches/prompts**:
    - Fiche 1 says "La cible architecturale est un seul backend complet par API: tout Vulkan dans kinetic-vulkan, tout HIP dans kinetic-hip (à créer)." Good.
    - Fiche 3 says "Créer kinetic-runtime... Extraire ModelRunner... Déplacer les exécuteurs CUDA/Vulkan dans leurs crates backend respectifs..." Good.
    - Prompt 1 says "PÉRIMÈTRE D'ÉCRITURE: crates/kinetic-vulkan/**, crates/kinetic-hip/** (à créer), PLAN.md... Propositions d'API dans kinetic-runtime..." Good.
    - Prompt 3 says "PÉRIMÈTRE: Cargo.toml, kinetic-runtime, kinetic-models/src/runner, kinetic-models/src/{lib,model}, kinetic-cuda/**, kinetic-vulkan/**..." Good.
    - Prompt 2 says "crates/kinetic-runtime/** (à créer: scheduler, pool KV, métriques)" and "crates/kinetic-server..." Good.
    - The main doc's 0.5 says "GPU AMD travaille dans kinetic-vulkan et kinetic-hip, pas dans kinetic-models." But fiche 1's scope includes `crates/kinetic-models/src/runner/**` (to create) and `crates/kinetic-models/src/vulkan/*` (to move). That's a contradiction: the main doc says GPU AMD doesn't work in kinetic-models, but fiche 1 gives write scope in `kinetic-models/src/runner/**` and context `kinetic-models/src/vulkan/*`. The fiche says "à déplacer vers kinetic-vulkan" but the scope still includes `kinetic-models/src/runner/**` (trait + CPU runner), which is Rust backend's job, not GPU AMD's. This is an inconsistency. Flag: "Fiche 1 empiète sur kinetic-models (runner), alors que 0.5 dit que GPU AMD ne touche pas kinetic-models. À clarifier: le runner trait/CPU est piloté par Rust backend; GPU AMD propose seulement l'API côté backend."
    - Similarly, prompt 1 scope includes "Propositions d'API dans kinetic-runtime..." but 0.5 says GPU AMD works in kinetic-vulkan/hip, not runtime. Proposals are okay, but write scope says "uniquement, sans implémentation"; still a potential overlap. Flag as minor.
    - Fiche 3's scope includes `crates/kinetic-models/src/{lib,model}.rs` and `kinetic-models/src/runner/**`; good.
    - Main doc 0.5 says "Runtime travaille dans kinetic-runtime et kinetic-server, pas dans les crates backend." Fiche 2's scope includes `crates/kinetic-models/src/{convoy,memory,per_layer}.rs` and `warm/**`; these are model crate, not backend crates. The statement "pas dans les crates backend" is consistent (model crate is not backend). Good.
    - Main doc 0.5 says "Quantification travaille dans kinetic-quant et les noyaux" — fiche 4 scope includes `kinetic-cpu/src/kdot`, backend tests, docs; consistent.
    - Main doc 0.5 says "Intégration modèles travaille dans kinetic-models/src/arch et blocks, sans code backend" — fiche 6 scope includes `docs/src/models.md`, `scripts/new_family.py`; consistent.
    - Main doc 0.5 says "QA teste chaque backend séparément..." — fiche 5 scope includes scripts/CI; consistent.

21. **"One backend per API" vs `kinetic-models/src/runner`**:
    - The architecture says `kinetic-models` exposes `ModelRunner` trait; the CPU runner is in `kinetic-models`; backend crates implement it. This is consistent. But the phrase "tout CUDA vit dans kinetic-cuda" includes the CUDA implementation of `ModelRunner`; good. The trait is in `kinetic-models`, not CUDA-specific.
    - The fiche 1 scope includes `crates/kinetic-models/src/runner/**` (to create: trait `ModelRunner` and CPU runner). That's actually Rust backend's job (fiche 3 also includes it). Overlap. Flag.

22. **"kinetic-runtime" vs "kinetic-models" traits**:
    - Main doc says `kinetic-runtime` has "découverte des devices, registre/factory de backends, capacités, budget mémoire, ordonnanceur, pool KV, métriques" and "ne connaît pas les types de modèle." `kinetic-models` has `ModelRunner`. But a `ModelRunner` trait in `kinetic-models` that backend crates implement would require `kinetic-models` to depend on backend types? No, the trait is generic; backends depend on `kinetic-models` to implement it. That creates a dependency: `kinetic-cuda` -> `kinetic-models` (for trait) and `kinetic-models` -> `kinetic-runtime`? The doc says `kinetic-models` exposes `ModelRunner`; `kinetic-cuda`/`vulkan` implement it. That means backend crates depend on `kinetic-models`, which is okay (they need model descriptors). But it could create a cycle if `kinetic-models` depends on `kinetic-runtime` and `kinetic-runtime` depends on backends? The doc says `kinetic-runtime` has a factory/registry of backends, which implies it knows backend crates; if backend crates depend on `kinetic-models`, and `kinetic-models` depends on `kinetic-runtime`, and `kinetic-runtime` depends on backend crates, that's a cycle. This is a real architectural risk. Let's analyze:
      - Option A: `kinetic-runtime` defines traits; `kinetic-models` depends on `kinetic-runtime`; backend crates depend on `kinetic-models` + `kinetic-runtime`; `kinetic-server` depends on all backends and selects one. The factory lives in `kinetic-server`, not `kinetic-runtime`. The doc says "kinetic-runtime: ... registre/factory de backends" — if the factory is in runtime and runtime depends on backend crates, cycle. If the factory uses a registry populated by `kinetic-server`, no cycle. The doc's wording "registre/factory" is ambiguous. Flag: "préciser que la factory qui connaît les crates concrets vit dans kinetic-server/CLI, pas dans kinetic-runtime, sinon cycle de dépendances."
      - Option B: `kinetic-runtime` defines `Backend` trait and `ModelRunner` trait; `kinetic-models` depends on runtime; backends depend on models+runtime; server depends on backends. The factory in server. This is clean.
    - This is an important architectural detail. The docs say "kinetic-runtime ... registre/factory de backends" and "kinetic-server choisit un backend via une factory". If both, clarify. Flag as "ambiguïté à lever".
    - Similarly, `kinetic-models` exposes `ModelRunner`; `kinetic-runtime` has scheduler/pool KV. The scheduler needs to call `ModelRunner`; if `ModelRunner` is in `kinetic-models` and `kinetic-runtime` depends on `kinetic-models`, then `kinetic-models` cannot depend on `kinetic-runtime` (cycle). The doc says `kinetic-models` depends on `kinetic-runtime` (0.2: `kinetic-models` ... trait `ModelRunner`; `kinetic-runtime` ... scheduler, pool KV). It doesn't explicitly state the dependency direction. If `kinetic-runtime` owns scheduler/KV and calls `ModelRunner`, it must depend on `kinetic-models` or define the trait itself. If `kinetic-models` defines `ModelRunner`, then `kinetic-runtime` depends on `kinetic-models`; `kinetic-models` must not depend on `kinetic-runtime`. But `kinetic-models` needs device/capabilities types from runtime? Maybe not; it can use generic descriptors. This needs clarification. Flag: "la direction des dépendances entre kinetic-runtime et kinetic-models n'est pas explicite; risque de cycle. Proposer: runtime définit les traits de device/backend; models définit ModelRunner et dépend de runtime; runtime ne dépend pas de models; le scheduler vit dans server ou dans un crate au-dessus, pas dans runtime." This is a substantive architecture review finding. The user asked if all prompts/missions were created; this is beyond, but useful.
    - The docs say "kinetic-runtime ... ordonnanceur, pool KV" and "kinetic-models ... trait ModelRunner". If runtime has the scheduler and models has the runner trait, runtime must know the trait. It could define a generic `Sequence`/`Request` trait and let models implement it; but the doc says `ModelRunner` in models. This is a design ambiguity. Flag it.

23. **"One backend per API" vs "kinetic-runtime scheduler"**:
    - If each backend crate owns its executor and the scheduler is in runtime, the backend `ModelRunner` must expose token-level step/prefill/decode; the scheduler calls it. That's the design. Good, but dependency direction needs care.

24. **"CPU runner in kinetic-models" vs "one backend per API"**:
    - The doc says CPU reference runner in `kinetic-models`; that's an exception to "each backend owns its executor." Is CPU a backend? It says "Le CPU reste le chemin de référence: ses noyaux dans kinetic-cpu, son exécution générique dans kinetic-models." This is a deliberate exception. But if `kinetic-models` contains the CPU executor, it has backend-specific code (CPU) in the model crate, which contradicts "kinetic-models no backend code." The doc says "son exécution générique dans kinetic-models" — generic execution, not CPU-specific. The CPU kernels are in `kinetic-cpu`; the orchestration is generic. That's okay, but the trait/runner boundary needs clarity. Flag as "exception CPU à documenter: le runner CPU est dans kinetic-models comme référence, mais il ne doit pas contenir de code SIMD; les noyaux restent dans kinetic-cpu."

25. **"Metal via Vulkan"**:
    - The doc says Metal is a Vulkan path (MoltenVK). The repo's `kinetic-vulkan` has a `metal` feature that compiles WGSL to Metal shaders, not MoltenVK. Wait, the Cargo.toml comment says: "On macOS the executor drives the GPU through Metal itself, which needs no driver installed; see the `metal` feature of kinetic-vulkan." It says "drives the GPU through Metal itself", not MoltenVK. The docs `docs/src/devices.md` says "On macOS, drives the GPU through Metal itself instead of a Vulkan driver layered on it. The kernels are the same WGSL, written out as Metal shaders at build time, with GEMV kernels of its own from shaders/metal." So Metal is **not** MoltenVK; it's a native Metal path inside `kinetic-vulkan` (feature `metal`). The session's final docs say "Metal reste un chemin du backend Vulkan (MoltenVK), pas un backend séparé." This is factually wrong: it's native Metal, not MoltenVK. This is a significant factual error in `docs/experts.md` §0. We must flag it. The session earlier may have said "Metal is a Vulkan path" and then "MoltenVK" incorrectly. Let's verify `Cargo.toml` comments and `docs/src/devices.md`. The `Cargo.toml` comment: "On macOS the executor drives the GPU through Metal itself, which needs no driver installed; see the `metal` feature of kinetic-vulkan." The `docs/src/devices.md` section: "On macOS, drives the GPU through Metal itself instead of a Vulkan driver layered on it. The kernels are the same WGSL, written out as Metal shaders at build time, with GEMV kernels of its own from shaders/metal. No effect on other systems." So yes, native Metal, not MoltenVK. The docs `docs/experts.md` says "Metal reste un chemin du backend Vulkan (MoltenVK)". This is wrong. Also `docs/experts/fiches-de-poste.md` Fiche 1 says "Metal/SPIR-V croisé" as appreciated skill, not MoltenVK. The main doc error is significant because it affects the backend architecture: Metal is a target inside `kinetic-vulkan` with a `metal` feature, not a separate backend and not MoltenVK. We should correct it in the audit, and maybe edit the doc. The user asked to analyze if all prompts/missions were created, not to fix factual errors, but this is a quality issue worth flagging. We can offer to fix.

26. **"SYCL/WebGPU"**:
    - The session final mentions HIP/SYCL/WebGPU as compatibility. The repo has CUDA/Vulkan/CPU; no SYCL/WebGPU. Fine as future.
    - "Metal reste un chemin du backend Vulkan" is the only error; it should be "Metal est un chemin natif du crate `kinetic-vulkan` (feature `metal`), pas MoltenVK."

27. **"CUDA backend on local machine"**:
    - The doc says "Pas de backend CUDA prioritaire sur cette machine: pas de GPU NVIDIA, donc pas de mesure locale." Good.
    - But the session's earlier analysis says "CUDA is useful upstream/CI, not P0 local." Good.

28. **"HIP/ROCm"**:
    - The local machine has ROCm 7.2 and `gfx1201`. The docs say "HIP/ROCm, gfx1201" in fiche 1. Good.
    - The `docs/src/devices.md` says HIP is not supported; the fiche says "à créer." Good.

29. **"Quantization targets"**:
    - "5 cibles contre ~30" — from session. Fine.
    - "NVFP4 et F8_E4M3 absents" — from session. Fine.
    - "Pas d'imatrix, pas d'IQ encode" — from session. Fine.

30. **"Architecture coverage"**:
    - "21/180" — from session. Fine.
    - "13/68 prétokenizers" — from session. Fine.
    - "~15/45 routes" — from session. Fine.
    - "2/73 projecteurs" — from session. Fine.

31. **"KV cache"**:
    - The docs' KV row says "KvCache en f16, pas de q8_0/q4_0" — Vulkan-specific. The fiche 1 says "KvCache est en f16, non quantifié" in the Vulkan context. The fiche 4 says "KV f16 seulement" — but CUDA has q8_0. This is inaccurate globally. Fiche 4 context: "KV f16 seulement." If the quantization expert reads this, they might think no q8_0 exists. The CUDA q8_0 exists; the gap is Vulkan. Flag: "Fiche 4 dit 'KV f16 seulement' sans qualifier Vulkan; CUDA a déjà q8_0. À corriger pour éviter de refaire l'existant."
    - Fiche 1 context: "KvCache est en f16, non quantifié." It's under Vulkan context; okay, but could specify "Vulkan".
    - Main doc table: "KV non quantifié | KvCache en f16, pas de q8_0/q4_0" — same issue. Flag.

32. **"KV cache improvements"**:
    - The docs mention paged KV, q8_0/q4_0, prefix caching, pool N × context, but not the detailed KV analysis from the session's last answer (shared KV zero-copy, dynamic pool, block-level prefix sharing, offload, per-layer policy, metrics). The user's last question was specifically about KV. If the goal is "all missions", the KV mission is under-specified. Flag as the main content gap: add a dedicated KV section or a "Mission KV" to the runtime fiche/prompt. We can propose the detailed analysis we have.
    - The session's last answer (lines 21307-21479) has a rich KV analysis; the final docs only capture a subset. The user may want that incorporated. Flag: "l'analyse KV de la session n'a pas été reportée dans les docs; elle est résumée en une ligne dans le diagnostic et quelques responsabilités dans la fiche 2. Si tu veux que l'expert KV mène l'analyse, il manque une fiche/mission dédiée."

33. **"Missions" vs "prompts"**:
    - The prompts are agent prompts, not human job descriptions; the fiches are human job descriptions. Good.
    - The prompts don't include "durée" or "priorité"; the fiches do. Fine.
    - The prompts don't include "interactions"; the fiches do. Fine.

34. **"Common contract"**:
    - It says "Ne jamais annoncer un gain sans commande, commit, date et état machine." Good.
    - "Médiane de ≥ 6 échantillons" — good.
    - "Accepter seulement si le candidat bat les deux contrôles..." — good.
    - "Ne pas copier de code depuis llama.cpp" — good.
    - "Aucune connexion réseau, sauf `kinetic pull` demandé explicitement." — good.
    - "S'arrêter et demander si..." — good.
    - Missing: "Ne pas modifier les fichiers hors périmètre" is there.
    - Missing: "Un seul agent par rôle à la fois" is in the usage section. Good.
    - Missing: "Ne pas lancer de job GPU concurrent" — in the precautions. Good.

35. **"Agent 5"**:
    - It is tasked with `THEORY.md`, ledger, configs, A/B, conformance Vulkan, needle, soak, CI GPU. Good.
    - But the prompt's "PÉRIMÈTRE D'ÉCRITURE" includes `.github/workflows/**`, which overlaps Agent 10. Flag.
    - The prompt's "INTERDITS" says "Ne pas modifier les noyaux ou le runtime pour améliorer un chiffre." Good.
    - The prompt's "COMMANDES" includes `server_abuse.py -- target/release/kinetic serve ...`; good.

36. **"Agent 1"**:
    - It is tasked with "aligner Vulkan sur les classes de fonctionnalités du backend CUDA" — good.
    - It is tasked with "porter les noyaux sur HIP/ROCm" — good.
    - It is tasked with "KV q8_0" — good.
    - It is not tasked with "shared KV zero-copy" — that's runtime/GPU. Could be added. Flag.
    - It is not tasked with "paged KV attention kernel" — runtime + GPU. Could be added. Flag.

37. **"Agent 2"**:
    - It is tasked with "pool KV paginé" — good.
    - It is not explicitly tasked with "KV metrics" — it says "métriques par token" and "cache hit"; okay.
    - It is not explicitly tasked with "shared KV zero-copy" — could be added. Flag.
    - It is not explicitly tasked with "KV quantization policy" — that's quantization/GPU. Fine.

38. **"Agent 4"**:
    - It is tasked with "KV q8_0/q4_0" — good.
    - It is not tasked with "KV quantization error analysis / long-context" — could be added. Flag.

39. **"Agent 8"**:
    - It is tasked with "TLS/API key, CORS, arrêt propre" — good.
    - It is not tasked with "security audit" — the fiche says "Sécuriser les entrées"; okay.

40. **"Agent 10"**:
    - It is tasked with "CI release" — good.
    - It is not tasked with "reproducible builds" — the fiche says "build reproductible"; okay.

41. **"Docs status"**:
    - `docs/experts.md` says "Statut: analyse, 22 novembre 2026." Good.
    - It says "Snapshot: main du fork Angra974/kinetic" — is that the current repo? The user's repo is `mrnemesys/Devs/kinetic`, maybe fork. Fine.
    - It says "Portée: fork local (Ryzen 7 7800X3D, RX 9070 XT gfx1201, Vulkan RADV, ROCm 7.2), pas le plan CUDA de l'amont." Good.
    - It says "Placement: document interne, hors du mdBook publié." Good.

42. **"Docs language"**:
    - French, consistent with user. Good.
    - Some terms are English; fine.

43. **"Git status"**:
    - Only docs added; no code. Good.

44. **"Line counts"**:
    - `docs/experts.md` 699 lines; `fiches` 1048; `prompts` 657. Good.

45. **"All prompts created?"**:
    - 10 prompts for 10 profiles. Yes.
    - No prompt for security, technical writer, or a dedicated KV expert. The docs explain security as transverse and omit technical writer. KV is folded into runtime. If the user's definition of "all" is the final 10, yes; if it includes the exploratory 12, two missing. We should present both.

46. **"All missions created?"**:
    - Each fiche/prompt has a mission. Yes.
    - The session's "missions" like "cellule performance minimale", "équipe complète", "phases", "RACI" are in the main doc. Yes.
    - The session's "KV deep dive" mission is not a standalone mission. Flag.
    - The session's "security" mission is transverse. Flag.
    - The session's "technical writer" mission is absent. Flag.

47. **"Potential duplicate/conflicting missions"**:
    - Agent 1 and Agent 3 both may touch `kinetic-vulkan`; Agent 1 owns kernels, Agent 3 owns moving executors. Need coordination. The RACI says GPU AMD + Rust backend. Good, but prompts don't state "coordinate with Agent 3 before moving files." Flag.
    - Agent 5 and Agent 10 both own `.github/workflows/**`; need split. Flag.
    - Agent 1, 2, 3, 4, 5 all write `PLAN.md`/`ledger.jsonl`/`configs/`; need format/locking. Flag.
    - Agent 4 and Agent 1 both touch `kinetic-vulkan/tests/**`; Agent 4 owns tests, Agent 1 owns kernels. Need coordination. Flag.

48. **"Missing `docs/experts/README.md`"**:
    - Not required; main doc links. Fine.

49. **"Missing `docs/experts/` in mdBook SUMMARY"**:
    - The docs are internal, not in mdBook. Fine.

50. **"Missing update to `PLAN.md`"**:
    - The architecture decision is not in `PLAN.md`. The prompts reference `PLAN.md` §4 P0.2/P0.3, but the one-backend-per-API decision is only in `docs/experts.md`. If the team uses `PLAN.md` as the source of truth, the decision should be added there. Flag: "la décision structurante n'est pas dans PLAN.md; les prompts la lisent dans docs/experts.md, donc c'est okay, but consider adding a line to PLAN.md."
    - The user's previous message said "as tu mis cela en place ?" The answer: in the docs, yes; in `PLAN.md`/code, no. Flag.

51. **"Missing update to `docs/src/contributing.md`"**:
    - The one-backend-per-API rule should be in contributing docs for developers. The docs say "Documenter la règle..." as a responsibility of Rust backend. Not yet done. Flag as a follow-up.

52. **"Missing ADR"**:
    - The decision is significant; an ADR would help. The docs don't have one. Flag as a recommendation.

53. **"Prompt for KV expert"**:
    - If the user wants a deep KV analysis, the best profile is the runtime architect (fiche 2 / Agent 2), with support from GPU AMD (attention kernels) and quantization (KV formats). We can recommend adding a "Mission KV" section to Agent 2's prompt and a dedicated "KV cache" subsection in fiche 2. This is actionable.

54. **"Prompt for security"**:
    - If the user wants a dedicated security expert, create Agent 11. The docs currently say security is transverse. Flag as optional.

55. **"Prompt for technical writer"**:
    - If the user wants the earlier 11th profile, create Agent 11/12. Flag as optional.

56. **"Prompt for HIP"**:
    - If the user wants a separate HIP expert, split Agent 1 into Agent 1a (Vulkan) and Agent 1b (HIP). The final docs intentionally merge. Flag as optional.

57. **"Prompt for KV cache"**:
    - Same as above.

Now, how to respond? The user asks: "Anlayse ./docs/experts/ et ./session-...md pour voir si tout les prompts pour les experts et missions ont bien été crée."

We should answer with an audit report:
- **Verdict**: Yes for the 10 retained profiles; 10 fiches + 10 prompts, all with missions. No missing prompt among the final 10.
- **However**, there are gaps/inconsistencies:
  1. Two exploratory profiles not retained: technical writer, dedicated security. Security is documented as transverse; technical writer is absent.
  2. HIP/ROCm is merged into Agent 1, not a separate prompt. Intentional.
  3. KV cache deep-dive is not a standalone mission/prompt; it's folded into Agent 2, with bits in 1 and 4. Given the session's last question, this is the main gap.
  4. Factual error: Metal is native Metal in `kinetic-vulkan` (feature `metal`), not MoltenVK.
  5. KV "f16 only" is Vulkan-specific; CUDA already has q8_0. Fiche 4 and main table should say "Vulkan" to avoid redoing existing work.
  6. Architecture dependency ambiguity: `kinetic-runtime` factory/registry vs `kinetic-server` factory; risk of cycle between `kinetic-runtime` and `kinetic-models`. Clarify dependency direction.
  7. Write-scope overlaps/conflicts: Agent 1 vs 3 on `kinetic-vulkan`/`kinetic-models/src/runner`; Agent 5 vs 10 on `.github/workflows`; multiple agents writing `PLAN.md`/`ledger.jsonl`/`configs/`; Agent 4 vs 1 on Vulkan tests.
  8. Bootstrapping: `ledger.jsonl`, `configs/`, `THEORY.md`, `best-config.json` don't exist; prompts assume them. Agent 5 should create them first, or each agent should create an entry if absent.
  9. Minor order inconsistency: session final table has Quantization #3, Rust #4; docs have Rust #3, Quantization #4. Not a missing item.
  10. The main doc's profile summaries and fiches duplicate content; risk of drift. Keep fiches as source of truth.
  11. The main doc's "0.5 Ce que cela change pour les experts" says GPU AMD doesn't work in `kinetic-models`, but fiche 1 scope includes `kinetic-models/src/runner/**` and `kinetic-models/src/vulkan/*`; clarify.
  12. The prompts' "PÉRIMÈTRE D'ÉCRITURE" includes `PLAN.md`, `ledger.jsonl`, `configs/` for multiple agents; need a shared format and merge policy.
  13. The "security" transverse note is in `docs/experts.md` but not in the fiches/prompts; if security is a requirement, add a checklist to Agent 8/3/10.
  14. The "KV cache" analysis from the session is not fully captured; add a dedicated section/ficha.

We should prioritize the findings: "Bloquant / important / mineur". Then provide a table of coverage.

Let's build a coverage table:

| Profil (liste finale) | Fiche | Prompt | Mission | Statut |
|---|---|---|---|---|
| 1 GPU AMD | Fiche 1 | Agent 1 | Oui | Complet |
| 2 Runtime | Fiche 2 | Agent 2 | Oui | Complet, mais KV non détaillé |
| 3 Rust backend | Fiche 3 | Agent 3 | Oui | Complet |
| 4 Quantification | Fiche 4 | Agent 4 | Oui | Complet, mais "KV f16 seulement" à qualifier |
| 5 Performance/QA | Fiche 5 | Agent 5 | Oui | Complet |
| 6 Intégration modèles | Fiche 6 | Agent 6 | Oui | Complet |
| 7 Tokenizers | Fiche 7 | Agent 7 | Oui | Complet |
| 8 Serveur/API | Fiche 8 | Agent 8 | Oui | Complet |
| 9 Multimodal | Fiche 9 | Agent 9 | Oui | Complet |
| 10 Build/release | Fiche 10 | Agent 10 | Oui | Complet |

Then "Profils exploratoires non retenus":
| Profil | Session | Docs | Statut |
|---|---|---|---|
| HIP/ROCm dédié | line 5347 | Fusionné dans Fiche 1/Agent 1 | Volontaire |
| Technical writer/DevRel | line 5431 | Absent | Non retenu, pas de prompt |
| Sécurité | line 6185 | Transverse dans docs/experts.md §2 | Pas de fiche/prompt dédié |
| KV cache lead | last question | Partiel dans Fiche 2/Agent 2 | À ajouter si prioritaire |

Then "Missions de la session":
| Mission | Docs | Statut |
|---|---|---|
| Diagnostic | §1 | Oui |
| Architecture cible | §0 | Oui, avec erreur Metal |
| 10 profils | §2 + fiches | Oui |
| Composition 5/10 | §3 | Oui |
| Phases | §3.3 | Oui |
| RACI | §3.4 | Oui |
| Évaluation/exercices | §4 | Oui |
| Sourcing | §4.3 | Oui |
| Commandes | §5 | Oui |
| Livrables | Fiches 30/60/90 | Oui, mais pas de tableau consolidé |
| Ce qu'il ne faut pas faire | §6 | Oui |
| Recommandation finale | §8 | Oui |
| KV deep dive | Session last answer | Non intégré en tant que mission |

Then "Corrections recommandées" with priority:
- P0: Add KV mission/prompt (or extend Agent 2) — because the user asked.
- P0: Fix Metal/MoltenVK error.
- P1: Qualify "KV f16 only" as Vulkan; CUDA q8_0 exists.
- P1: Clarify runtime/models dependency direction and factory location.
- P1: Resolve write-scope overlaps (Agent 1/3, 5/10, ledger).
- P2: Decide on technical writer/security.
- P2: Reorder Rust/Quantization or add a note.
- P2: Add a consolidated deliverables table.

Then "Conclusion": The 10 final prompts/missions are created and consistent; no missing prompt among the 10. The gaps are the exploratory profiles, the KV deep-dive, and a few consistency issues. If the goal is "10 retained profiles", it's complete. If the goal is "everything discussed in the session", three things are missing: technical writer, dedicated security, KV lead.

Should we make edits? The user asked to analyze, not to fix. We can offer to fix. But maybe we should fix the obvious factual error and the KV qualification? The instruction says "Treat unfamiliar files or changes as potential user work and investigate before deleting or overwriting them." These docs were created by the previous assistant in this session; editing them is okay. But the user asked "analyse ... pour voir si tout ... ont bien été créés." An audit response is appropriate; editing without asking might be overstepping. We can offer to apply the fixes. However, the system says "accomplish their goals using tools"; the goal is to verify, not necessarily fix. We'll answer with the audit and ask if they want the corrections applied. That's safest.

But maybe the user expects us to actually create missing prompts. The phrasing "pour voir si tout les prompts ... ont bien été créé" = "to see if all the prompts ... were indeed created." It's a verification request. So an audit report is the right deliverable. We can include a "Ce qu'il manque" list and offer to create them.

Let's ensure we don't overstate: The session's final list has 10 profiles; the docs have 10. So "all prompts for the final list: yes." The exploratory 12-profile list had 2 extras; the final docs intentionally omitted them. We should distinguish "missing" vs "not retained."

Let's verify the exact line numbers in the session for the exploratory profiles to cite:
- 5335: "### 1. GPU compute kernel engineer — Vulkan/RDNA (top priority)"
- 5347: "### 2. HIP/ROCm backend engineer"
- 5353: "### 3. Inference runtime / scheduler architect"
- 5365: "### 4. Rust systems architect / performance engineer"
- 5377: "### 5. Quantization & numerical correctness engineer"
- 5388: "### 6. Model integration engineer (architectures/tokenizers)"
- 5399: "### 7. Server/API & platform engineer"
- 5408: "### 8. Multimodal engineer"
- 5414: "### 9. Performance measurement / QA automation engineer"
- 5425: "### 10. Build/release & bindings engineer"
- 5431: "### 11. Technical writer / developer experience (optional)"
- 5434: "### 12. Security engineer?"
- 6062: "## 1. Diagnostic synthétique"
- 6093: P0 Expert 1 GPU
- 6104: P0 Expert 2 Runtime
- 6115: P0/P1 Expert 3 Rust
- 6124: P1 Expert 4 Quantization
- 6133: P1 Expert 5 HIP/ROCm
- 6141: P1 Expert 6 Model integration
- 6150: P1/P2 Expert 7 Tokenizers
- 6157: P2 Expert 8 Server/API
- 6166: P2 Expert 9 Multimodal
- 6171: P2 Expert 10 Performance/QA
- 6180: P2 Expert 11 Build/release
- 6185: Cross-cutting: security
- 7357: final 10 profiles.
- 8443: outline for docs.
- 12131: architecture section.
- 21307: KV analysis.

We can cite these line numbers in the audit.

Let's verify the final docs' section 0 says "Metal reste un chemin du backend Vulkan (MoltenVK)" — yes, line 43-44. The repo's `Cargo.toml` comment and `docs/src/devices.md` say native Metal. This is a clear factual error. We should flag it prominently.

Let's verify `docs/src/devices.md` exact text for Metal:
- Lines 5207-5218: "## Apple GPUs ... On macOS, drives the GPU through Metal itself instead of a Vulkan driver layered on it. The kernels are the same WGSL, written out as Metal shaders at build time, with GEMV kernels of its own from shaders/metal. No effect on other systems." Good.
- `Cargo.toml` comment: "On macOS the executor drives the GPU through Metal itself, which needs no driver installed; see the `metal` feature of kinetic-vulkan." Good.
So the fix: "Metal est un chemin natif du crate `kinetic-vulkan` (feature `metal`, shaders Metal générés depuis WGSL), pas MoltenVK." This is a precise correction.

Let's verify the "KV f16 only" claim in fiche 4:
- Fiche 4 line 412: "KV f16 seulement." No qualification. CUDA has q8_0. This is misleading. The main doc line 172: "KV non quantifié | `KvCache` en f16, pas de q8_0/q4_0 | GPU AMD + quantification". The GPU AMD context is Vulkan, so it's okay there, but the table row is global. Fiche 4's context is global. Flag.

Let's verify the `KvCache` CPU f32 vs docs table:
- `llama-kinetic.md` table says CPU f16; code says f32. This is a pre-existing doc error, not created by the previous assistant. The new docs don't repeat the CPU f16 claim, so no issue in `docs/experts`. But the session's KV analysis may have repeated it? Let's check. The session at 21307 says "CPU: f32 (référence exacte)" maybe. We don't need to flag unless the new docs repeat it. They don't. Fine.

Let's verify the "KV cache" section in the session's final answer: it says "CPU: f32", "CUDA: f16 ou q8_0", "Vulkan: f16 uniquement", "pas de cache partagé". The new docs don't include this table. Flag: "l'analyse KV détaillée n'est pas dans les docs."

Let's verify the "KV improvements" priorities from the session: P0 metrics + unified pool + q8_0 Vulkan; P1 paged KV + prefix sharing + shared KV zero-copy; P2 offload + multi-GPU + per-layer policy. The new docs only mention paged pool, q8_0, shared KV in the fiches. Flag: "les priorités KV ne sont pas consolidées."

Let's verify the "KV expert" recommendation: the session says "C'est exactement le profil 2.3 (runtime architect), avec une spécialisation KV." The new docs have the runtime profile, but no explicit "KV lead" specialization. Flag.

Now, the final response structure:

```
## Verdict
Oui pour les 10 profils retenus : 10 fiches + 10 prompts, chacun avec mission, périmètre, contexte, commandes, livrables et rapport. Aucun prompt manquant dans la liste finale.

Non si on entend "tout ce qui a été discuté" : 3 sujets ne sont pas des prompts dédiés (HIP, sécurité, KV), et 1 profil exploratoire n'a pas été retenu (technical writer).

## Couverture
(table 10 profils)
(table profils exploratoires)
(table missions transverses)

## Ce qui manque ou doit être corrigé
### 1. KV cache : mission non formalisée (le plus important)
...
### 2. Erreur factuelle Metal/MoltenVK
...
### 3. "KV f16 seulement" à qualifier Vulkan
...
### 4. Ambiguïté d'architecture runtime/models/factory
...
### 5. Chevauchements de périmètre
...
### 6. Bootstrap ledger/configs
...
### 7. Profils exploratoires non retenus
...
### 8. Détails mineurs
...

## Recommandation
- Si la liste à 10 est la référence: complet, avec 4 corrections.
- Si tu veux "tout": ajouter Agent 11 KV (ou mission KV dans Agent 2), Agent 12 sécurité, Agent 13 tech writer.
- Prochaine action: je peux appliquer les corrections...
```

Let's make the coverage table concise but informative. Use ✅ / ⚠️ / ❌.

For each of the 10:
- Fiche, prompt, mission: all yes.
- Notes: KV, Metal, etc.

Let's include exact file paths and line numbers for the issues:
- `docs/experts.md:43` Metal/MoltenVK.
- `docs/experts.md:172` KV f16.
- `docs/experts/fiches-de-poste.md:412` KV f16 only.
- `docs/experts/fiches-de-poste.md:83` GPU AMD scope includes `kinetic-models/src/runner`.
- `docs/experts/fiches-de-poste.md:300` Rust backend scope includes runner.
- `docs/experts/prompts-agents.md:88` Agent 1 scope includes runtime proposals; `:210` Agent 3 runner.
- `docs/experts/prompts-agents.md:330` Agent 5 workflows; `:584` Agent 10 workflows.
- `docs/experts/prompts-agents.md:87` Agent 1 PLAN/ledger; etc.
- `docs/experts.md:64` runtime factory; `:71` server Backend; `:93` factory in server/CLI.
- `docs/experts.md:55` models feature; `:67` models ModelRunner.
- `docs/experts.md:26-33` one backend per API.
- `docs/experts.md:122-133` 0.5 roles.

Let's be precise: `docs/experts/fiches-de-poste.md` line 83 is `crates/kinetic-models/src/runner/**` (to create: trait `ModelRunner` and CPU runner). Yes.

`docs/experts/prompts-agents.md` line 88 is "Propositions d'API dans crates/kinetic-runtime..." and line 210 is "crates/kinetic-models/src/runner/**". Good.

`docs/experts/prompts-agents.md` line 330 is `- scripts/**`; line 331 `.github/workflows/**`; line 584 `.github/workflows/**` for Agent 10. Good.

`docs/experts.md` line 64: "découverte des devices, registre/factory de backends..." Good. Line 93: "Ajouter une factory de backend dans kinetic-server / kinetic-cli". Good. The ambiguity: runtime has a registry/factory, server also has a factory. Flag.

`docs/experts.md` line 55: "`kinetic-models` ... expose un trait `ModelRunner`"; line 67: "`kinetic-models` ... trait `ModelRunner`". The dependency direction is not stated. Flag.

`docs/experts.md` line 43-44: "Metal reste un chemin du backend Vulkan (MoltenVK)". Flag.

`docs/experts.md` line 172: "KV non quantifié | `KvCache` en f16, pas de q8_0/q4_0". Flag.

`docs/experts/fiches-de-poste.md` line 412: "KV f16 seulement." Flag.

`docs/experts/fiches-de-poste.md` line 31-33: "Le MoE 30B A3B ..."; okay.

`docs/experts/fiches-de-poste.md` line 83: GPU AMD scope includes `kinetic-models/src/runner`. Flag.

`docs/experts.md` line 122: "GPU AMD travaille dans kinetic-vulkan et kinetic-hip, pas dans kinetic-models." Contradiction with fiche 1. Flag.

`docs/experts.md` line 124: "Runtime travaille dans kinetic-runtime et kinetic-server, pas dans les crates backend." Fiche 2 scope includes `kinetic-models/src/{convoy,memory,per_layer}.rs` and `warm/**`; these are model crate, not backend. No contradiction. Good.

`docs/experts.md` line 126: "Rust backend pilote la migration..." Good.

`docs/experts.md` line 128: "Quantification travaille dans kinetic-quant et les noyaux..." Fiche 4 includes `kinetic-gguf`, `kinetic-cpu/src/kdot`, tests; "noyaux" covers CPU; okay.

`docs/experts.md` line 130: "Intégration modèles travaille dans kinetic-models/src/arch et blocks, sans code backend." Fiche 6 includes `scripts/new_family.py`; okay.

`docs/experts.md` line 132: "QA teste chaque backend séparément..." Good.

Now, should we mention the "session file" itself has a discrepancy: it contains the previous assistant's final answer claiming "les trois documents" were created, but the session file is a transcript and not a source of truth. The docs exist. Fine.

Let's also check if the docs were created in the session at lines 9291 "Writing the files". It might list the files. But we already see they exist. No need.

Potential issue: The session file's title is "Quels experts pour inference-optimisation" and the final docs are in French. Fine.

Let's think about the user's likely intent: They want to ensure the previous assistant didn't forget any expert/mission. They may be preparing to launch agents. The most useful answer is a checklist of what exists and what's missing, with actionable fixes. We should be direct: "Oui, les 10 prompts sont là. Mais il manque X, Y, Z."

Let's prioritize the KV gap because the user's last question in the session was about KV. The final docs mention KV but don't have a dedicated mission. The user may have expected a KV expert prompt. We should say: "Le KV n'a pas de prompt dédié; il est dilué dans Agent 1, 2 et 4. Si tu veux l'analyse profonde, il faut soit ajouter une section 'Mission KV' à Agent 2, soit créer un Agent 11 'KV cache & mémoire'." Provide the recommended profile and scope.

Let's also note the "technical writer" and "security" were exploratory, not final. This prevents false alarms.

Let's formulate the verdict carefully:
- "Pour la liste finale de 10 profils : oui, tout est créé."
- "Pour la liste exploratoire de 12 profils : non, il manque le technical writer et le security engineer en tant que prompts dédiés; la sécurité est traitée en transverse, et le technical writer a été abandonné."
- "Pour les missions : les 10 missions sont présentes; la mission KV n'est pas formalisée à part."

Let's include a "Checklist" with checkboxes:
```
- [x] 10 fiches de poste
- [x] 10 prompts d'agents
- [x] 10 missions
- [x] Contrat commun
- [x] Périmètres d'écriture
- [x] Fichiers de contexte
- [x] Commandes de mesure
- [x] Livrables 30/60/90
- [x] Critères de succès
- [x] Signaux d'alerte
- [ ] Mission KV dédiée
- [ ] Prompt sécurité dédié
- [ ] Prompt technical writer
- [ ] Tableau consolidé des livrables
```

This is clear.

Let's also mention the "missions" in the main doc: diagnostic, architecture, profiles, composition, phases, RACI, evaluation, sourcing, commands, non-goals, recommendation. All present. Good.

Let's prepare the final answer in French, with tables and bullet points. Keep it under ~2000 words. The user asked for an analysis; thorough but concise.

Before final, let's double-check the exact count of prompts and fiches:
- Fiches: 10 (Fiche 1..10). Yes.
- Prompts: 10 (Agent 1..10). Yes.
- Main doc profiles: 2.2..2.11 = 10. Yes.
- Common contract: 1. Yes.
- Usage section: 1. Yes.

Let's check if there is a prompt for "Agent 0" or "orchestrator" — no. The session didn't require one. The main doc's "Utilisation avec OpenCode" explains how to launch. Fine.

Let's check if there is a prompt for "QA" and "performance" combined — Agent 5. Good.

Let's check if there is a prompt for "server/API" and "observability" combined — Agent 8. Good.

Let's check if there is a prompt for "build/release" and "bindings" combined — Agent 10. Good.

Let's check if there is a prompt for "model integration" and "tokenizers" separate — Agent 6 and 7. Good.

Let's check if there is a prompt for "quantization" and "numerical correctness" combined — Agent 4. Good.

Let's check if there is a prompt for "Rust backend" and "runtime" separate — Agent 3 and 2. Good.

Let's check if there is a prompt for "GPU AMD" and "HIP" combined — Agent 1. Good.

So the final 10 are complete.

Now, let's write the audit. Use a professional tone. Avoid saying "I created" since we are auditing; say "Les documents contiennent..." and "J'ai vérifié...".

Possible opening:
"J'ai comparé les trois documents avec la session. Verdict : la liste finale de 10 profils est complète — 10 fiches, 10 prompts, 10 missions. En revanche, si 'tout' inclut les profils exploratoires et l'analyse KV de la fin de session, il manque trois choses."

Then the tables.

Let's draft the tables:

**Couverture des 10 profils retenus**

| # | Profil | Fiche | Prompt | Mission | État |
|---|---|---|---|---|---|
| 1 | GPU AMD (Vulkan/HIP) | Fiche 1 | Agent 1 | Oui | ✅ |
| 2 | Runtime d'inférence | Fiche 2 | Agent 2 | Oui | ✅ (KV à détailler) |
| 3 | Rust systèmes/backend | Fiche 3 | Agent 3 | Oui | ✅ |
| 4 | Quantification/exactitude | Fiche 4 | Agent 4 | Oui | ⚠️ "KV f16 seulement" à qualifier |
| 5 | Performance/QA | Fiche 5 | Agent 5 | Oui | ✅ |
| 6 | Intégration architectures | Fiche 6 | Agent 6 | Oui | ✅ |
| 7 | Tokenizers/templates | Fiche 7 | Agent 7 | Oui | ✅ |
| 8 | Serveur/API/observabilité | Fiche 8 | Agent 8 | Oui | ✅ |
| 9 | Multimodal | Fiche 9 | Agent 9 | Oui | ✅ |
| 10 | Build/release/bindings | Fiche 10 | Agent 10 | Oui | ✅ |

**Profils exploratoires non retenus**

| Profil | Où dans la session | Dans les docs | Verdict |
|---|---|---|---|
| HIP/ROCm dédié | ligne 5347 | Fusionné dans Fiche 1/Agent 1 | Volontaire, pas un oubli |
| Technical writer/DevRel | ligne 5431 | Absent | Non retenu; pas de prompt |
| Security engineer | ligne 6185 | Transverse dans `docs/experts.md` §2 | Pas de fiche/prompt dédié |
| KV cache lead | dernière question | Partiel dans Fiche 2/Agent 2 | À formaliser si prioritaire |

**Missions transverses**

| Mission | Document | État |
|---|---|---|
| Diagnostic | `docs/experts.md` §1 | ✅ |
| Architecture cible | §0 | ✅ (erreur Metal) |
| 10 profils | §2 + fiches | ✅ |
| Composition 5/10 | §3 | ✅ |
| Phases | §3.3 | ✅ |
| RACI | §3.4 | ✅ |
| Évaluation/exercices | §4 | ✅ |
| Sourcing | §4.3 | ✅ |
| Commandes | §5 | ✅ |
| Livrables 30/60/90 | fiches | ✅ (pas de tableau consolidé) |
| Non-objectifs | §6 | ✅ |
| Recommandation | §8 | ✅ |
| KV deep dive | session | ❌ pas de mission dédiée |

Then the findings:

**1. Le KV cache n'a pas de mission/prompt dédié (le plus important)**
- `docs/experts.md` line 172 reduces KV to one row.
- Fiche 2 responsibilities mention paged KV, but no audit plan/metrics/experiments.
- Fiche 4 says "KV f16 seulement" (false globally; CUDA q8_0).
- The session's last answer contains a detailed KV analysis (CPU f32, CUDA f16/q8_0, Vulkan f16, no shared cache, no paged KV, no metrics, improvements P0/P1/P2, expert profile). It is not in the docs.
- Recommendation: add a "Mission KV" section to Fiche 2/Agent 2 or create Fiche 11/Agent 11 "KV cache & mémoire". Lead: runtime architect with KV specialization; support: GPU AMD (attention kernels), quantization (formats/error), QA (metrics/load tests). 30/60/90.

**2. Erreur factuelle Metal/MoltenVK**
- `docs/experts.md:43` says "Metal reste un chemin du backend Vulkan (MoltenVK)".
- The repo says native Metal: `Cargo.toml` comment, `docs/src/devices.md` §Apple GPUs, `kinetic-vulkan` feature `metal`.
- Fix: "Metal est un chemin natif du crate `kinetic-vulkan` (feature `metal`, shaders Metal générés depuis WGSL), pas MoltenVK."

**3. "KV f16 seulement" à qualifier**
- `docs/experts/fiches-de-poste.md:412` and `docs/experts.md:172`.
- CUDA supports q8_0; the gap is Vulkan. Fix: "KV Vulkan f16 seulement; CUDA f16/q8_0; CPU f32."
- Otherwise Agent 4 may re-implement q8_0 CUDA.

**4. Ambiguïté runtime/models/factory**
- `docs/experts.md:64` says `kinetic-runtime` has registry/factory; `:93` says factory in server/CLI.
- `kinetic-models` defines `ModelRunner`; `kinetic-runtime` has scheduler/pool KV. The dependency direction is not explicit. Risk of cycle.
- Recommendation: define `kinetic-runtime` as traits + device/capabilities + memory; `kinetic-models` depends on runtime and defines `ModelRunner`; backend crates depend on models + runtime; `kinetic-server` owns the concrete factory and depends on all backends. Or put the factory in a separate `kinetic-backends` crate. Document it.

**5. Chevauchements de périmètre**
- Agent 1 vs Agent 3: `kinetic-vulkan/**`, `kinetic-models/src/runner/**`.
- Agent 5 vs Agent 10: `.github/workflows/**`.
- Agents 1-5: `PLAN.md`, `ledger.jsonl`, `configs/`.
- Agent 4 vs Agent 1: `kinetic-vulkan/tests/**`.
- Recommendation: add a "coordination" line to each prompt, and a lock/merge policy for PLAN/ledger; split CI workflows (QA vs release).

**6. Bootstrap ledger/configs**
- `ledger.jsonl`, `configs/`, `THEORY.md`, `best-config.json` don't exist.
- Agents 1-4 are told to write to them. Agent 5 owns them.
- Recommendation: Agent 5 creates the skeleton first, or each prompt says "if absent, create a minimal entry with the documented schema; do not restructure."

**7. Profils exploratoires**
- Technical writer and security: not retained. If the user wants "all", create them. Security can be a checklist in Agents 3/8/10 rather than a full-time role. Technical writer could be a P2 prompt for docs/onboarding.
- HIP: merged; if the user wants a dedicated HIP expert, split Agent 1.

**8. Détails mineurs**
- Order: session final table has Quantization #3, Rust #4; docs have Rust #3, Quantization #4. Not a missing item; align or add a note.
- No consolidated deliverables table; each fiche has 30/60/90. Add a summary table in `docs/experts.md` §7.
- Main doc and fiches duplicate content; risk of drift. Keep fiches as source of truth, or generate the summary.
- `docs/experts.md` §0.5 says GPU AMD doesn't work in `kinetic-models`, but Fiche 1 scope includes `kinetic-models/src/runner/**`; clarify ownership (Rust backend owns the trait/CPU runner; GPU AMD only proposes).
- The `docs/experts.md` line 172 says "KvCache en f16" but CPU is f32; specify backend.

Then the conclusion:
"Donc : oui pour les 10 profils retenus; non pour 'tout ce qui a été discuté'. Les 3 ajouts à faire si tu veux une couverture complète: (1) mission KV dédiée, (2) prompt sécurité (ou checklist), (3) prompt technical writer (optionnel). Les corrections P0: Metal, KV Vulkan vs CUDA, dépendances runtime/models, chevauchements."

Offer:
"Je peux appliquer ces corrections directement dans les docs et créer Fiche 11/Agent 11 KV + Agent 12 sécurité. Dis-moi si tu veux que je le fasse."

This is a solid audit.

Let's make sure the response doesn't claim the docs are "wrong" in a harsh way; be factual.

Let's also mention the exact file sizes/line counts to show completeness:
- `docs/experts.md`: 699 lines.
- `docs/experts/fiches-de-poste.md`: 1048 lines.
- `docs/experts/prompts-agents.md`: 657 lines.
This is a nice detail.

Let's include "10 fiches, 10 prompts, 10 missions" up front.

Now, let's consider whether to actually edit the docs. The user asked "analyse ... pour voir si tout ... ont bien été créés." We can answer without editing. But the system says "Prefer dedicated tools over shell commands" etc. We've used tools to read. Fine.

Could we use the `question` tool to ask if they want fixes? The instructions say use it to ask questions during execution. We can end with a question in the response instead; the `question` tool is for interactive choices. A simple offer is fine. Maybe use the `question` tool with options "Apply fixes", "Add KV prompt", "Just the audit"? That might be overkill. The user asked for an analysis; a textual offer is enough. But the harness may expect tool use for questions. Not necessary.

Let's ensure the final answer is in French and uses clear headings. No code changes, so no need to mention tests.

Let's write the final response now.

Wait, one more check: The user's current message says "Anlayse ./docs/experts/ et ./session-...md pour voir si tout les prompts pour les experts et missions ont bien été crée." They might want us to **verify** that the files contain all prompts, not to critique architecture. We should lead with the direct answer and keep the critique secondary. The direct answer: "Oui, les 10 prompts et 10 missions de la liste finale sont présents. Il manque les 2 profils exploratoires (tech writer, sécurité) et une mission KV dédiée." Then details.

Let's make the "Verdict" section crisp:
- "Liste finale (10 profils) : complet."
- "Liste exploratoire (12 profils) : 2 non retenus."
- "Missions : 10/10, mais KV non formalisé."
- "Cohérence : 6 points à corriger."

Then the tables and findings.

Let's also mention that the session's final answer itself says "les trois documents" were created, and they are indeed present; no missing file. Good.

Let's draft the final answer with moderate detail. Aim ~1800 words. The desired oververbosity is 5; this is a complex audit, so ~2000 words is acceptable.

Let's write.

## Verdict

**Oui pour la liste finale de 10 profils** : les 10 fiches de poste, les 10 prompts d’agents et les 10 missions existent bien.

| Document | Taille | Contenu |
|---|---:|---|
| `docs/experts.md` | 699 lignes | Diagnostic, architecture cible, 10 profils, composition d’équipe, RACI, commandes, recommandation |
| `docs/experts/fiches-de-poste.md` | 1 048 lignes | 10 fiches complètes : mission, contexte, compétences, périmètre, livrables 30/60/90, critères, signaux d’alerte |
| `docs/experts/prompts-agents.md` | 657 lignes | Contrat commun + 10 prompts prêts à l’emploi |

**Mais si « tout » inclut les profils exploratoires et l’analyse KV de la fin de session, il manque trois choses** : une mission KV dédiée, un prompt sécurité, et un prompt technical writer.

---

## 1. Couverture des 10 profils retenus

| # | Profil | Fiche | Prompt | Mission | État |
|---|---|---|---|---|---|
| 1 | GPU AMD Vulkan/HIP | Fiche 1 | Agent 1 | Oui | ✅ |
| 2 | Runtime d’inférence | Fiche 2 | Agent 2 | Oui | ✅ mais KV à détailler |
| 3 | Rust systèmes/backend | Fiche 3 | Agent 3 | Oui | ✅ |
| 4 | Quantification/exactitude | Fiche 4 | Agent 4 | Oui | ⚠️ « KV f16 seulement » à qualifier |
| 5 | Performance/QA | Fiche 5 | Agent 5 | Oui | ✅ |
| 6 | Intégration architectures | Fiche 6 | Agent 6 | Oui | ✅ |
| 7 | Tokenizers/templates | Fiche 7 | Agent 7 | Oui | ✅ |
| 8 | Serveur/API/observabilité | Fiche 8 | Agent 8 | Oui | ✅ |
| 9 | Multimodal | Fiche 9 | Agent 9 | Oui | ✅ |
| 10 | Build/release/bindings | Fiche 10 | Agent 10 | Oui | ✅ |

Chaque prompt contient bien : rôle, mission, lecture obligatoire, périmètre d’écriture, interdits, boucle de travail, commandes, critères de succès et format de rapport. Le contrat commun couvre la mesure, l’exactitude, le périmètre et les conditions d’arrêt.

---

## 2. Profils exploratoires non retenus

| Profil | Dans la session | Dans les docs | Verdict |
|---|---|---|---|
| HIP/ROCm dédié | ligne 5347 | Fusionné dans Fiche 1 / Agent 1 | Volontaire, pas un oubli |
| Technical writer / DevRel | ligne 5431 | Absent | Non retenu, aucun prompt |
| Security engineer | ligne 6185 | Transverse dans `docs/experts.md` §2 | Pas de fiche/prompt dédié |
| KV cache lead | dernière question | Partiel dans Fiche 2 / Agent 2 | **À formaliser si prioritaire** |

La sécurité est bien mentionnée comme transverse, couverte par le serveur, le backend Rust et la QA. Le technical writer a été abandonné dans la liste finale à 10. Ce ne sont donc pas des oublis si tu valides la liste à 10.

---

## 3. Les vrais manques / incohérences

### 3.1 Le KV cache n’a pas de mission dédiée — le point le plus important

C’est le sujet de la dernière question de la session, mais il est dilué :

- `docs/experts.md` ligne 172 le réduit à une ligne dans le tableau des goulets.
- Fiche 2 mentionne « pool KV paginé », mais sans audit de l’existant, sans métriques, sans protocole d’expérience.
- Fiche 4 dit « KV f16 seulement », ce qui est faux globalement : **CUDA a déjà q8_0** ; l’écart est côté Vulkan.
- L’analyse KV détaillée de la session (CPU f32, CUDA f16/q8_0, Vulkan f16, pas de cache partagé, pas de paged KV, priorités P0/P1/P2) n’est pas reportée dans les docs.

**Recommandation** : ajouter une section « Mission KV » dans Fiche 2 / Agent 2, ou créer **Fiche 11 / Agent 11 — KV cache et hiérarchie mémoire**. Profil pilote : architecte runtime spécialisé KV. Support : GPU AMD pour les noyaux d’attention paginée, quantification pour les formats et l’erreur, QA pour les métriques et les tests de charge.

### 3.2 Erreur factuelle : Metal n’est pas MoltenVK

`docs/experts.md` ligne 43 dit :

> « Metal reste un chemin du backend Vulkan (MoltenVK) »

C’est faux. Le dépôt indique que Metal est **natif** dans `kinetic-vulkan` :

- `Cargo.toml` : « On macOS the executor drives the GPU through Metal itself… see the `metal` feature of kinetic-vulkan. »
- `docs/src/devices.md` : « drives the GPU through Metal itself instead of a Vulkan driver layered on it… Metal shaders at build time. »

**Correction** : « Metal est un chemin natif du crate `kinetic-vulkan` (feature `metal`, shaders Metal générés depuis WGSL), pas MoltenVK. »

### 3.3 « KV f16 seulement » à qualifier

`docs/experts/fiches-de-poste.md` ligne 412 et `docs/experts.md` ligne 172.

Il faut écrire : **Vulkan f16 uniquement ; CUDA f16/q8_0 ; CPU f32**. Sinon l’Agent 4 peut croire que q8_0 n’existe nulle part et refaire l’existant CUDA.

### 3.4 Ambiguïté runtime / models / factory

- `docs/experts.md` ligne 64 dit que `kinetic-runtime` contient un « registre/factory de backends ».
- Ligne 93 dit que la factory est dans `kinetic-server` / `kinetic-cli`.
- `kinetic-models` définit `ModelRunner`, tandis que `kinetic-runtime` porte l’ordonnanceur et le pool KV. La direction des dépendances n’est pas explicite, avec un risque de cycle.

**À clarifier** : soit `kinetic-runtime` ne contient que les traits et les types de devices/capacités, et la factory concrète vit dans `kinetic-server` ; soit la factory vit dans un crate `kinetic-backends` au-dessus. Il faut aussi préciser si `kinetic-runtime` dépend de `kinetic-models` ou l’inverse.

### 3.5 Chevauchements de périmètre entre agents

| Chevauchement | Agents | Risque |
|---|---|---|
| `kinetic-vulkan/**` | Agent 1 et Agent 3 | Édition concurrente |
| `kinetic-models/src/runner/**` | Agent 1 et Agent 3 | Contradiction avec §0.5 |
| `.github/workflows/**` | Agent 5 et Agent 10 | CI perf vs CI release |
| `PLAN.md`, `ledger.jsonl`, `configs/` | Agents 1 à 5 | Conflits d’écriture |
| `kinetic-vulkan/tests/**` | Agent 1 et Agent 4 | Tests noyaux vs tests quant |

`docs/experts.md` §0.5 dit que le GPU AMD ne travaille pas dans `kinetic-models`, mais Fiche 1 lui donne `kinetic-models/src/runner/**` et `kinetic-models/src/vulkan/*`. Il faut trancher : le trait `ModelRunner` et le runner CPU appartiennent au Rust backend ; le GPU AMD ne fait que proposer l’API côté backend.

### 3.6 Bootstrap du ledger et des configs

Les Agents 1 à 4 sont censés écrire dans `PLAN.md`, `ledger.jsonl` et `configs/`, mais ces deux derniers n’existent pas encore. L’Agent 5 est chargé de les créer.

**Recommandation** : lancer Agent 5 en premier, ou ajouter dans chaque prompt : « si `ledger.jsonl`/`configs/` n’existent pas, créer une entrée minimale au format documenté, sans restructurer ».

### 3.7 Détails mineurs

- L’ordre diffère entre le tableau final de la session (Quantification #3, Rust #4) et les docs (Rust #3, Quantification #4). Ce n’est pas un manque, mais à aligner ou à assumer.
- Pas de tableau consolidé des livrables 30/60/90 : chaque fiche les a, mais `docs/experts.md` §7 ne les résume pas.
- Le doc principal et les fiches dupliquent les profils : risque de divergence. Garder les fiches comme source de vérité.
- Le tableau `docs/experts.md` ligne 172 dit « `KvCache` en f16 » sans préciser Vulkan ; le CPU est en f32.

---

## Conclusion

- **Liste finale à 10 profils** : tout est créé, cohérent et exploitable. Aucun prompt manquant.
- **Liste exploratoire à 12 profils** : 2 non retenus (technical writer, sécurité dédiée), 1 fusionné (HIP).
- **Missions** : 10/10, mais la mission KV n’est pas formalisée à part alors que c’est le dernier sujet analysé dans la session.

Si tu veux une couverture vraiment complète, je propose d’ajouter :

1. **Fiche 11 / Agent 11 — KV cache et mémoire** (pilote : runtime, support GPU/quant/QA).
2. **Agent 12 — Sécurité** ou, plus léger, une checklist sécurité dans les prompts 3, 8 et 10.
3. **Agent 13 — Technical writer** (optionnel, P2).

Je peux appliquer directement les corrections P0 (Metal, KV Vulkan vs CUDA, dépendances runtime/models, chevauchements) et créer les prompts manquants si tu veux.

---

## User

non, tu dois créer les missions et approfondir les missions si nécessaire mais pas travailler directemenbt sur le projet, on a déjà un agent en action qui s'occupe d'autres choses actuellement. 

---

## Assistant

_Thinking:_

The user says: no, you must create the missions and deepen the missions if necessary but not work directly on the project, we already have an agent in action handling other things currently.

Interpretation: They want me to create the missing missions/prompts (experts + missions), and deepen existing missions, but NOT touch the project code or interfere with the ongoing agent. So I should:
- Create the missing expert prompts/missions in docs/experts/ (documentation only, not code).
- Deepen missions where needed (KV cache, security, technical writer, HIP?).
- Avoid modifying code, PLAN.md, ledger, configs, or running builds/tests.
- Avoid touching files that the active agent may be working on? The user says "on a déjà un agent en action qui s'occupe d'autres choses actuellement" — another agent is working on other things. So we should not modify project code or interfere. Editing docs/experts/ should be safe, but we should be careful not to touch files outside docs/experts/. Maybe the active agent is working on the project code; we should only add/modify docs/experts files.

What exactly does "créer les missions" mean? The user wants missions for the experts, probably:
- Add a dedicated KV cache mission (Fiche 11 / Agent 11).
- Add a security mission (Fiche 12 / Agent 12 or transverse mission).
- Add a technical writer mission (Fiche 13 / Agent 13, optional).
- Deepen existing missions: KV in Agent 2, quantization, GPU, Rust backend, QA, etc.
- Possibly add a "mission" section to each fiche/prompt with concrete objectives, phases, deliverables, acceptance criteria, dependencies, risks, and handoffs. The existing fiches have mission lines and responsibilities, but the user wants "missions" perhaps as standalone mission briefs.

We need to decide the scope. The user says "tu dois créer les missions et approfondir les missions si nécessaire" — "you must create the missions and deepen the missions if necessary". So we should produce a set of mission documents/prompts. The existing docs already have 10 missions, but some are shallow or missing. We need to:
1. Create a dedicated `docs/experts/missions/` directory? Or add to existing files?
2. Add missing expert profiles/prompts: KV cache lead, security, technical writer, maybe HIP dedicated.
3. Deepen the missions of existing experts with concrete work packages, phases, dependencies, acceptance criteria, and handoffs.
4. Ensure consistency with the one-backend-per-API architecture and the current state.
5. Do not modify code or run project commands that could interfere. Only write docs.

Given the user's instruction, we should act, not just propose. We need to create the missions. Let's plan the deliverables.

### What to create

Option A: Create a new `docs/experts/missions.md` containing mission briefs for all 10+ experts, with detailed work packages. But the existing `fiches-de-poste.md` already has missions. The user says "créer les missions" — maybe they want a separate mission catalog that can be assigned to agents, with IDs, objectives, scope, dependencies, deliverables, acceptance criteria, and sequence. This would be more actionable than job descriptions.

Option B: Add `docs/experts/missions/` with one file per mission:
- `M01-prefill-vulkan.md`
- `M02-moe-vulkan.md`
- `M03-kv-cache.md`
- `M04-runtime-scheduler.md`
- `M05-backend-consolidation.md`
- `M06-quantization.md`
- `M07-qa-automation.md`
- `M08-model-coverage.md`
- `M09-tokenizers.md`
- `M10-server-api.md`
- `M11-multimodal.md`
- `M12-build-release.md`
- `M13-security.md`
- `M14-technical-writing.md`
- etc.
But that's a lot of files; maybe a single `missions.md` with a consistent template per mission is better.

Option C: Add a "Mission" section to each fiche and prompt, deepening the existing ones, plus add new fiches/prompts for KV, security, technical writer. This is more integrated but requires editing 3 large files and adding 3 profiles. The user says "créer les missions et approfondir les missions si nécessaire" — this suggests adding a missions layer, not necessarily new profiles.

Let's think about what's most useful for the user. They have an active agent working on the project. They want to prepare missions for other agents/experts. A mission catalog with:
- Mission ID
- Title
- Owner expert
- Priority
- Objective
- Context / current state
- Work packages (phases)
- Deliverables
- Acceptance criteria / gates
- Dependencies / handoffs
- Write scope
- Measurement commands
- Risks
- Duration
- Definition of done

This is exactly what a "mission" document should be. The existing fiches are job descriptions (who to hire); the missions are work packages (what to do). The user wants both, and the missions are missing. So create `docs/experts/missions.md` (or a `missions/` directory) with detailed mission briefs for each expert, including new KV, security, technical writing.

Maybe also create `docs/experts/missions/README.md` with an index and sequencing. But a single file is easier to maintain and read.

Let's decide on a structure:

```
docs/experts/
├── README.md                  (index, how to use)
├── experts.md                 (main analysis, already exists at docs/experts.md)
├── fiches-de-poste.md         (job descriptions, existing)
├── prompts-agents.md          (agent prompts, existing)
└── missions.md                (new: mission catalog)
```

Or `docs/experts/missions/` with one file per mission. Given the volume (14 missions × ~100 lines = 1400 lines), a single `missions.md` is large but manageable. A directory with one file per mission is more modular and easier for an agent to load only its mission. The prompts already say "Lis docs/experts/prompts-agents.md, section Agent X". If we add missions, we can have `docs/experts/missions/M01-...md` and reference them.

The user says "créer les missions" plural. A directory is appropriate. Let's create:
- `docs/experts/missions/README.md` — index, sequencing, dependency graph, status.
- `docs/experts/missions/M01-prefill-vulkan.md`
- `docs/experts/missions/M02-moe-vulkan.md`
- `docs/experts/missions/M03-kv-cache.md`
- `docs/experts/missions/M04-runtime-scheduler.md`
- `docs/experts/missions/M05-backend-consolidation.md`
- `docs/experts/missions/M06-quantization.md`
- `docs/experts/missions/M07-qa-automation.md`
- `docs/experts/missions/M08-model-coverage.md`
- `docs/experts/missions/M09-tokenizers.md`
- `docs/experts/missions/M10-server-api.md`
- `docs/experts/missions/M11-multimodal.md`
- `docs/experts/missions/M12-build-release.md`
- `docs/experts/missions/M13-security.md`
- `docs/experts/missions/M14-technical-writing.md`

That's 15 files. Is that too many? It's a lot of writing, but we can be efficient. Each mission ~80-150 lines. Total ~1500-2000 lines. The user asked for it. We have the token budget.

Alternatively, create a single `docs/experts/missions.md` with all missions, ~1200 lines. This is easier to navigate in one file and less file sprawl. But the user may want to assign one mission per agent; a single file is fine, they can copy the section. The existing `prompts-agents.md` is a single file with 10 prompts; consistent to have `missions.md` single file.

Let's go with a single `docs/experts/missions.md` plus a `docs/experts/README.md` index. But the user said "créer les missions" — a single file with all missions is a clear deliverable. We can also add a `docs/experts/missions/` directory if needed later. Let's choose single file for consistency and ease.

What about deepening existing missions? We can:
- Add a "Mission" section to each fiche? The fiches already have a mission line; we can leave them and put the detailed mission in `missions.md`.
- Add cross-references in `docs/experts.md` and `prompts-agents.md` to the mission IDs.
- Update the prompts to reference the mission file: "Ta mission détaillée est M03 dans docs/experts/missions.md." This makes the prompts actionable and avoids duplication.
- Add new experts for KV, security, technical writing? The user said "créer les missions", not necessarily new job descriptions. But if a mission has no owner, we need to assign it. We can assign:
  - KV mission: owner "Architecte runtime d'inférence" (Agent 2), with support GPU/quant/QA. No new expert profile needed, but a dedicated mission.
  - Security mission: owner "Serveur/API" + "Rust backend" + "Build/release" as a transverse mission, or create a "Security" mission with a lead from server/API. The user asked earlier if all experts were created; we flagged security as transverse. We can create a mission without a new expert, assigning it to the server/API expert with support.
  - Technical writing mission: owner "Build/release" or a new "Technical writer" profile. The user said "approfondir les missions si nécessaire" — maybe create a mission for documentation owned by the build/release expert, not a new hire. But earlier we flagged the technical writer as missing. The user said "non, tu dois créer les missions" — maybe they want the technical writer mission too. We can create a "Documentation & developer experience" mission owned by the build/release expert, with a note that it can be a dedicated hire if volume grows.
  - HIP/ROCm: a dedicated mission M02 (HIP skeleton) under GPU AMD, not a separate expert. Good.

So the mission catalog can cover all work without adding new expert profiles, except perhaps a "KV cache lead" specialization. The user wants missions, not necessarily new hires. Let's create missions with owners from the existing 10 experts, plus cross-cutting security/documentation missions.

Let's define the mission list with IDs, priority, owner, dependencies:

**P0 — Foundation / performance**
- M01 — Baseline, measurement harness, and ledger (owner: Performance/QA). This is a prerequisite for all. Creates `THEORY.md`, `ledger.jsonl`, `configs/`, A/B protocol, CI GPU. Duration 2-4 weeks.
- M02 — Vulkan prefill GEMM (owner: GPU AMD). Tiled GEMM, coopmat, split-K, async submission, pipeline cache. Target pp512 ≥ 2×, bit-exact. Depends on M01.
- M03 — Vulkan MoE (owner: GPU AMD). Grouped GEMM/GEMV, expert paging, async. Target decode/prefill parity. Depends on M01, M02.
- M04 — KV cache audit and redesign (owner: Runtime, support GPU/Quant/QA). Audit current KV, metrics, unified pool, paged KV, prefix sharing, shared KV zero-copy, q8_0 Vulkan, offload. This is the deep mission the user wants. Depends on M01.
- M05 — Runtime scheduler and concurrency (owner: Runtime). Token-level scheduler, continuous batching, chunked prefill, cancellation, fairness, p99/TTFT. Depends on M04 (KV pool) and M01.
- M06 — Backend consolidation (owner: Rust backend). Create `kinetic-runtime`, `ModelRunner`, move executors, remove backend features, factory, test backend. Depends on M01; coordinates with M02-M05.
- M07 — HIP/ROCm skeleton (owner: GPU AMD, support Rust backend). Port GEMV/attention, validate on gfx1201, compare to llama ROCm. Depends on M06 (backend trait) and M02 (Vulkan kernels).
- M08 — KV quantization and numerical gates (owner: Quantization, support Runtime/GPU). q8_0/q4_0 Vulkan, per-layer policy, quality measurement, exactness gates. Depends on M04.
- M09 — Quantization coverage (NVFP4, FP8, IQ encode, imatrix) (owner: Quantization). P1.
- M10 — Model coverage (architectures) (owner: Model integration). P1.
- M11 — Tokenizers/templates/tool calls (owner: Tokenizers). P1.
- M12 — Server/API parity and observability (owner: Server/API). P1.
- M13 — Multimodal (owner: Multimodal). P2.
- M14 — Build/release/bindings (owner: Build/release). P2.
- M15 — Security hardening (owner: Server/API + Rust backend + Build/release; transverse). P1/P2.
- M16 — Documentation and developer experience (owner: Build/release or technical writer). P2.

This is 16 missions. Maybe too many. We can consolidate:
- M01 Baseline & QA harness.
- M02 Vulkan prefill.
- M03 Vulkan MoE.
- M04 KV cache.
- M05 Runtime scheduler.
- M06 Backend consolidation.
- M07 HIP skeleton.
- M08 Quantization & KV formats.
- M09 Model coverage.
- M10 Tokenizers.
- M11 Server/API.
- M12 Multimodal.
- M13 Build/release.
- M14 Security.
- M15 Documentation.

15 missions. Good.

Each mission document should include:
- ID, title, priority, owner, support, status.
- Objective (one paragraph).
- Why now (context, measured gap).
- Scope (in/out).
- Work packages (WP1..WPn) with concrete tasks.
- Deliverables.
- Acceptance criteria / gates.
- Measurement protocol (commands, samples, controls).
- Dependencies and handoffs.
- Risks and mitigations.
- Duration / sequencing.
- Definition of done.
- Files to read / write scope.
- Reporting format.

This is a lot, but we can make each ~100 lines. 15 × 100 = 1500 lines. Fine.

We also need to "approfondir les missions si nécessaire" — the existing 10 missions are broad; the mission catalog will deepen them with work packages and acceptance criteria. We can also add a "Mission" section to each fiche? The mission catalog can be the deep version; the fiches remain job descriptions. We can add a line in each fiche: "Mission détaillée : `docs/experts/missions.md#m0x`." And in each prompt: "Ta mission détaillée est M0x." This ties everything together.

Let's plan the edits:
1. Create `docs/experts/missions.md` with the 15 missions, index, dependency graph, and sequencing.
2. Update `docs/experts.md`:
   - Add a link to `docs/experts/missions.md` in §7.
   - Fix the Metal/MoltenVK error.
   - Qualify "KV f16" as Vulkan; mention CUDA q8_0.
   - Clarify runtime/models dependency direction and factory location.
   - Add the KV mission and security/documentation missions to the profile summary? Maybe add a "Missions" section or a table linking profiles to mission IDs.
   - Add a note about the 10 profiles vs 15 missions.
3. Update `docs/experts/fiches-de-poste.md`:
   - Add "Mission détaillée : M0x" links in each fiche.
   - Fix "KV f16 seulement" to "KV Vulkan f16; CUDA f16/q8_0; CPU f32".
   - Clarify GPU AMD scope vs Rust backend (remove `kinetic-models/src/runner/**` from GPU AMD or mark as proposal only).
   - Add a new fiche for KV? The user said "créer les missions", not necessarily new fiches. But if KV is a mission owned by the runtime expert, no new fiche needed. However, the user earlier asked "quel type d'expert devrait mener l'analyse profonde" — we can add a "Spécialisation KV" subsection to Fiche 2, not a new fiche. That deepens the mission without adding a profile.
   - Add security/documentation missions as transverse, not new fiches. Or add them to Fiche 8/10 responsibilities. We can mention in the mission catalog.
4. Update `docs/experts/prompts-agents.md`:
   - Add "Mission détaillée : M0x" to each agent.
   - Add a "Mission KV" explicit first phase to Agent 2.
   - Add security guardrails to Agents 3, 8, 10.
   - Add documentation deliverables to Agent 10 or a new Agent 11? The user said create missions, not necessarily prompts. But if we create M15 Documentation, we need an owner; assign to Agent 10 (Build/release) with a note. Or create Agent 11 "Documentation & DX" to cover it. Hmm.
   - Add a "Agent 11 — KV cache" prompt? The user said "créer les missions" — a mission is enough, but a prompt for the KV lead would be useful. The KV mission is owned by Agent 2, so we can extend Agent 2's prompt with a "PREMIER CHANTIER: M04 KV" section. That avoids a new profile.
   - Add a "Agent 11 — Sécurité" prompt? Security is transverse; we can add a checklist to Agents 3/8/10 and a mission M14. No new prompt needed, but the user asked "all prompts for experts" earlier. If we add a security mission without a prompt, it might be seen as incomplete. We can create a short "Agent 11 — Sécurité" prompt, or add a "Security checklist" to the common contract. The latter is more efficient and applies to all agents. The mission M14 can be led by Agent 8 with support from 3 and 10. That's fine.
   - Add a "Agent 12 — Documentation" prompt? Or add documentation deliverables to Agent 10. The user said "approfondir les missions si nécessaire"; documentation is a mission, not necessarily a new expert. Assign to Agent 10 with support from all.

Let's think about the user's wording: "non, tu dois créer les missions et approfondir les missions si nécessaire mais pas travailler directemenbt sur le projet, on a déjà un agent en action qui s'occupe d'autres choses actuellement."

They explicitly say "créer les missions" — so the deliverable is missions. They don't necessarily ask for new expert profiles or prompts. They want the missions to be created and deepened. The existing docs have job descriptions and prompts, but the missions are shallow. So the core deliverable is a mission catalog. We should focus on that, and only update the existing docs for consistency and links. Avoid creating new experts unless a mission lacks an owner.

Let's create a robust `docs/experts/missions.md` with:
- Purpose and how to use.
- Mission status table.
- Dependency graph / sequencing.
- Detailed missions M01–M15.
- Cross-cutting rules (measurement, exactness, write scope, reporting).
- Mapping mission → expert → prompt.
- Definition of ready/done.

Then update the three existing docs with links and corrections, minimally.

Do we need to create separate mission files? A single `missions.md` is easier. But 15 missions × 100 lines = 1500 lines; a single file is fine. The existing docs are 700-1000 lines, so 1500 is consistent.

Let's draft the missions carefully, using the repo facts and the session analysis. We need to be accurate and not invent measurements. Use the known numbers:
- Vulkan dense 27B IQ3_XXS: pp512 261.9 vs llama.cpp 1154.2 t/s (4.4×); tg128 38.4 vs 40.1 (95%).
- MoE 30B A3B: decode 55.7 vs 181.9 (3.3×); prefill 489 vs 2637 (5.4×).
- KV: f16 Vulkan; CUDA q8_0; CPU f32; no paged KV; no shared KV; Warm Start exact state cache.
- Concurrency: one backend per request; Vulkan one slot; CUDA up to 4 slots default; `MAX_CONNECTIONS=64`; `Connection: close`.
- `PREFILL_CHUNK=256` per request; `PROMPT_BATCH=128`; `gemv_many.wgsl` 32 tokens in flight.
- `Commands::submit` waits on fences; no timeline semaphores; no pipeline cache.
- `KvCache` f16 Vulkan; `positions` full context; `rows=PROMPT_BATCH`; no paging.
- CUDA `KvCache` q8_0 save/load tolerance 1e-2; shared KV copies.
- `Warm` exact prefix cache: 1 GiB RAM/2 GiB disk; not block-level.
- `Engine.free` pool; `Convoy` only CPU/CUDA; Vulkan no Convoy.
- `MAX_DEFAULT_SLOTS=4`; `SLOT_RESERVE=256 MiB` per CUDA slot.
- `kinetic bench` measures pp512, tg128, edit256.
- Scripts: `bench_protocol.py`, `bench_vs_llamacpp.py`, `conformance.py` (cpu/cuda only), `server_abuse.py`, `same_answer_demo.py`, `llama_server_compat.py`.
- `PLAN.md` P0.2/P0.3, P1.5/P1.6, P0.4, P0.1.
- `docs/src/devices.md`: CUDA q8_0; CPU/Vulkan ignore cache type.
- `llama-kinetic.md` §7.2/7.3/7.10/7.11/7.6/7.9/7.7/7.10-7.12/4.1/5.1/6/4.2/5.3/4.5/5.9/4.10/5.2.

Let's ensure mission M04 KV includes the detailed analysis from the session:
- Current state table: CPU f32, CUDA f16/q8_0, Vulkan f16, contiguous per sequence, static reservation, no paging, no shared, Warm Start exact state cache.
- Problems: memory wall, no paging, no prefix sharing, duplicate KV for shared layers, quantization only CUDA, no metrics, rollback constraints.
- Work packages:
  1. Audit + metrics (kv_bytes_allocated/used, fragmentation, cache hit, evictions, per-request).
  2. Unified pool + dynamic allocation (budget, growth, eviction, fallback).
  3. Paged KV + block table + gather attention.
  4. Prefix sharing / COW / shared KV zero-copy.
  5. KV quantization: q8_0 Vulkan, q4_0, per-layer policy, K/V separate, quality gates.
  6. Offload cold blocks (P2), multi-GPU (P2).
  7. Speculative rollback integration.
- Acceptance: 32k context on 16 GiB with N slots; no OOM; conformance; q8_0 quality; p99/TTFT.
- Dependencies: M01, M05, M08.
- Owner: Runtime; support GPU AMD, Quantization, QA.
- 30/60/90.

Mission M05 scheduler:
- Current state, work packages, acceptance.
- Token-level scheduler, continuous batching, chunked prefill, cancellation, fairness, backpressure, metrics.
- Vulkan multi-slot prerequisite; M04 KV pool.
- Acceptance c4/c16/c64, p99, TTFT, exactness.

Mission M06 backend consolidation:
- Current state, target, work packages, acceptance.
- Create `kinetic-runtime`, `ModelRunner`, move executors, remove features, factory, test backend.
- Dependency direction clarification.
- Acceptance: `kinetic-models` no CUDA/Vulkan deps; tests green; build/startup no regression.

Mission M01 baseline/QA:
- Create `THEORY.md`, `ledger.jsonl`, `configs/`, A/B protocol, CI GPU, conformance Vulkan, needle, soak, watchdog.
- Acceptance: detect 5% regression with noise; every number reproducible.

Mission M02 prefill:
- Work packages: profile, tiled GEMM, coopmat, split-K, async submission, pipeline cache, KV q8? No, KV is M08.
- Acceptance: pp512/pp2048 ≥2×, tg128 no regression >3%, bit-exact.

Mission M03 MoE:
- Grouped GEMM/GEMV, expert paging, async, shared experts, quality.
- Acceptance: decode/prefill parity, no regression.

Mission M07 HIP:
- Skeleton, GEMV Q4_K/Q5_K/Q6_K/Q8_0, attention FP16, conformance vs llama ROCm0.
- Depends on M06 trait, M02 Vulkan kernels.
- Acceptance: `gfx1201` runs, tokens match, performance baseline.

Mission M08 quantization/KV formats:
- NVFP4, FP8 E4M3, IQ encode, imatrix, KV q8_0/q4_0, per-layer policy, quality.
- Depends on M04 for KV integration.
- Acceptance: f64 reference, quality measured, identity gates.

Mission M09 model coverage:
- Families, blocks, conformance.
- Acceptance: model loads/generates, tokens match, no central match.

Mission M10 tokenizers:
- Pretokenizers, UGM, special tokens, templates, tool calls.
- Acceptance: ids identical, tool calls valid.

Mission M11 server/API:
- Routes, keep-alive, TLS/API key, CORS, multi-model router, metrics, n>1, timings.
- Acceptance: clients work, no crash under abuse, metrics.

Mission M12 multimodal:
- Projectors, audio, OCR.
- Acceptance: embedding parity, no text regression.

Mission M13 build/release:
- C ABI, bindings, Docker, packaging, CI release, size/time.
- Acceptance: install on 3 OS, ABI tested, reproducible.

Mission M14 security:
- Threat model: local server, API keys, TLS, input limits, path traversal, model files, unsafe code, supply chain.
- Work packages: audit, fuzz, dependency audit, sandboxing, secrets, CI.
- Owner: Server/API + Rust backend + Build/release.
- Acceptance: no critical findings, abuse tests, unsafe audit, dependency audit.

Mission M15 documentation/DX:
- Update docs, onboarding, examples, migration guide, benchmarks, ADRs.
- Owner: Build/release or technical writer; support all.
- Acceptance: docs build, examples pass, onboarding <1 day.

Maybe also M16 "Warm Start / Convoy / speculative" — these are differentiators but not in the 10 profiles. The runtime mission covers Warm Start/Convoy. We can include a work package in M05 or a separate mission. The user wants all missions; maybe add M16 "Differentiators preservation" (RAD-1, Warm Start, Zero Wait, Shortlist, Convoy) as a cross-cutting QA mission. But that might be overkill. We can include it in M01/M05 acceptance criteria.

Let's keep 15 missions.

### Mission document template

Each mission:

```
## M04 — KV cache : audit, pool unifié et pagination

| Champ | Valeur |
|---|---|
| Priorité | P0 |
| Pilote | Architecte runtime d'inférence (Agent 2) |
| Support | GPU AMD (Agent 1), Quantification (Agent 4), QA (Agent 5) |
| Dépend de | M01 |
| Débloque | M05, M08 |
| Durée | 6–10 semaines |
| Statut | À lancer |

### Objectif
...

### État actuel
...

### Périmètre
**Inclus** : ...
**Exclu** : ...

### Work packages
#### WP1 — Audit et métriques (1–2 sem.)
...
#### WP2 — Pool unifié (2–3 sem.)
...
#### WP3 — Pagination (3–4 sem.)
...
#### WP4 — Partage de préfixe et shared KV (2–3 sem.)
...
#### WP5 — Quantification KV (2–3 sem.)
...
#### WP6 — Offload et multi-GPU (P2)
...

### Livrables
...

### Critères d'acceptation
...

### Protocole de mesure
...

### Dépendances et handoffs
...

### Risques
...

### Définition de terminé
...
```

This is detailed and actionable.

### Update existing docs

We need to be careful not to disrupt the active agent. The user says another agent is working on other things. Editing `docs/experts/` is likely safe, but we should avoid touching `PLAN.md`, code, scripts, or configs. We'll only edit docs/experts files. We should also avoid running builds/tests or commands that could interfere with the active agent. We can use `read`/`edit`/`write` only.

We need to make minimal edits to the existing docs:
- Add a link to `missions.md` in `docs/experts.md` §7.
- Fix the Metal/MoltenVK error.
- Qualify KV f16.
- Clarify runtime/models dependency direction? This is a deeper edit; we can add a note in §0.2 and §0.4. Let's do it carefully.
- Add mission IDs to the profile summary table? Maybe add a "Mission" column to §2.1 table. That requires editing the table. Let's inspect §2.1. It likely has a table with Profile, Priority, Rarity, What it unlocks. We can add a "Mission" column or a note. Adding a column to a Markdown table is doable but requires editing all rows. Maybe add a separate "Missions" table in `missions.md` mapping profiles to missions. Simpler.
- Add a note at the top: "Les missions détaillées sont dans `docs/experts/missions.md`."
- Fix `docs/experts/fiches-de-poste.md`:
  - Add "Mission détaillée : M0x" to each fiche header. This is 10 edits.
  - Fix Fiche 4 "KV f16 seulement".
  - Clarify Fiche 1 scope: remove `kinetic-models/src/runner/**` or mark it "proposition seulement; le trait et le runner CPU sont pilotés par le Rust backend". Since the fiche is a job description, better to remove the line and add a note. But editing a line is fine.
  - Add a "Spécialisation KV" subsection to Fiche 2? The mission catalog has M04; the fiche can link to it. Maybe add a bullet under "Responsabilités" or a "Mission détaillée" line. We can add a "Chantiers prioritaires" line with M04/M05.
- Fix `docs/experts/prompts-agents.md`:
  - Add "Mission détaillée : M0x" to each agent. 10 edits.
  - Add "PREMIER CHANTIER : M04" to Agent 2.
  - Add security checklist to common contract? Maybe add a "Sécurité" subsection to the common contract: "Ne pas exposer de secrets, valider les entrées, ne pas ajouter de dépendance réseau, ..." This is useful and applies to all.
  - Add a "Documentation" requirement to common contract: "Mettre à jour la doc affectée." This covers M15.
  - Add a note about `ledger.jsonl`/`configs/` bootstrap: "Si absents, créer une entrée minimale; ne pas restructurer." This addresses the gap.
  - Add coordination notes for overlapping scopes: "Agent 1 ne déplace pas les exécuteurs; Agent 3 le fait." "Agent 5 possède les workflows CI perf; Agent 10 les workflows release." This is important.
  - Add a "Mission KV" explicit section to Agent 2's prompt: "Commence par M04: audit KV, métriques, pool, pagination, q8_0 Vulkan, shared KV." This deepens the mission.

Do we need to add new prompts for security and documentation? The missions M14/M15 have owners from existing agents, so no new prompts. But the user asked earlier "all prompts for experts" — we can say the new missions are assigned to existing agents, so no new prompt is needed; if they want dedicated experts, we can add Agent 11/12. The user now says "créer les missions" — so missions are the priority. We can add a short "Agent 11 — Sécurité" and "Agent 12 — Documentation" only if necessary. Let's avoid adding new profiles; assign missions to existing owners. This keeps the expert list stable and respects the user's focus on missions.

But wait: the user said "tu dois créer les missions et approfondir les missions si nécessaire" — they might want the missions as separate artifacts, not necessarily new experts. Yes.

Let's also create `docs/experts/README.md` as an index:
- What each file is.
- How to use: read experts.md → choose fiche → read mission → use prompt.
- Status table.
- Rules: one agent per mission, one change per trial, no code outside scope.
This is useful and small.

### Mission sequencing

We need a dependency graph:
- M01 (baseline/QA) first.
- M02 (Vulkan prefill) can start after M01; M03 (MoE) after M02 or parallel with M01.
- M04 (KV) after M01; M05 (scheduler) after M04; M08 (KV quant) after M04.
- M06 (backend consolidation) after M01; can run in parallel with M02/M03 but must coordinate file ownership.
- M07 (HIP) after M06 and M02.
- M09-M13 can run in parallel after M01 (coverage/product).
- M14 security after M11/M13 or in parallel.
- M15 documentation continuous.

Critical path: M01 → M02 → M03 → M07; M01 → M04 → M05 → M08. M06 is a parallel foundation.

Let's create a Mermaid diagram? The repo uses Markdown; Mermaid may not render in all viewers. Use a simple ASCII/table. Maybe a table with "Phase / Missions / Durée / Porte de sortie". Good.

### Deepening the missions

We should include specific work packages with file paths and commands, but avoid claiming code changes. The missions are plans, not implementations.

For M04 KV, include a detailed "état actuel" table:
| Backend | Stockage | Quantification | Slots | Partage | Pagination |
| CPU | `Vec<f32>` par couche | f32 | N (mémoire) | non | non |
| CUDA | device buffer f16/q8_0 | f16/q8_0 | 1..4 | copie pour shared KV | non |
| Vulkan | device buffer f16 | f16 | 1 | non (re-projection) | non |
This is accurate based on our analysis. Need to be careful: CUDA slots default up to 4; `default_slots` max 4. Vulkan 1. CPU slots based on memory; `default_slots` up to 4? `default_slots` uses `MAX_DEFAULT_SLOTS=4`; CPU can have up to 4 default. Good.
- "Warm Start" is a host-side exact state cache, not a KV pool. Include.
- "Convoy" shares passes, not KV. Include.

M04 work packages:
1. Instrumentation: add counters, expose in `kinetic doctor`/server metrics, log per request.
2. Baseline: measure KV bytes/token, max context per slot, concurrency at 16 GiB, time-to-OOM.
3. Unified pool: one allocator per device, budget, dynamic growth, eviction, fallback to host, clear errors.
4. Paged KV: block table, block size, gather attention, block allocation/free, fragmentation metrics.
5. Prefix sharing: block-level hash, COW, shared KV zero-copy, eviction policy.
6. Quantization: q8_0 Vulkan, q4_0, per-layer policy, K/V separate, quality gates.
7. Speculative rollback: truncate block table, checkpoint, max draft window.
8. Offload/multi-GPU: host KV, pinned memory, prefetch, layer split/tensor split.
9. Acceptance: 32k context on 16 GiB with ≥4 slots, no OOM, exactness, p99/TTFT, q8_0 quality.

M05 scheduler:
- Current: one backend per request; `PREFILL_CHUNK=256` per request; `Convoy` shares passes; Vulkan one slot; `MAX_CONNECTIONS=64`; `Connection: close`.
- Work packages: request lifecycle, admission, priority, fairness, cancellation, backpressure, chunked prefill, prefill/decode overlap, KV rollback, metrics, multi-slot Vulkan.
- Acceptance: c4/c16/c64 throughput, p99, TTFT, same_answer, no leak.

M06 backend consolidation:
- Current table: CUDA split, Vulkan split, models features, server branches.
- Target: one backend per API, `kinetic-runtime` traits, `ModelRunner`, factory, test backend.
- Work packages: traits, move CUDA executor, move Vulkan executor, remove features, factory, test backend, CI.
- Acceptance: `kinetic-models` no CUDA/Vulkan deps; tests green; build/startup no regression; one backend loaded at runtime.
- Dependency direction: explicit proposal.

M01 QA:
- `THEORY.md`, ledger, configs, A/B, CI GPU, conformance Vulkan, needle, soak, watchdog.
- Acceptance: detect 5% with noise; every number has command/commit/date/machine.

M02 prefill:
- Profile, tiled GEMM, coopmat, split-K, async, pipeline cache, tests.
- Acceptance: pp512/pp2048 ≥2×, tg128 no regression >3%, bit-exact.

M03 MoE:
- Grouped GEMM/GEMV, expert paging, shared experts, async, tests.
- Acceptance: decode/prefill parity, no regression.

M07 HIP:
- Skeleton, GEMV, attention, conformance, CI.
- Acceptance: gfx1201 runs, tokens match, baseline.

M08 quantization:
- NVFP4, FP8, IQ, imatrix, KV q8/q4, quality.
- Acceptance: f64 ref, quality, identity.

M09-M15 as above.

### Update `docs/experts.md` corrections

Let's make the edits:

1. Line 43-44: replace "Metal reste un chemin du backend Vulkan (MoltenVK), pas un backend séparé." with "Metal est un chemin natif du crate `kinetic-vulkan` (feature `metal`, shaders Metal générés depuis WGSL), pas un backend séparé ; ce n'est pas MoltenVK."
2. Line 172: change "KV non quantifié | `KvCache` en f16, pas de q8_0/q4_0" to "KV Vulkan non quantifié | `KvCache` Vulkan en f16 ; CUDA a déjà f16/q8_0 ; CPU en f32 ; pas de paged KV". The table cell is long; use a concise wording: "KV Vulkan non quantifié | `KvCache` Vulkan f16 ; CUDA f16/q8_0 ; CPU f32 ; pas de paged KV".
3. Add a note after §0.2 about dependency direction and factory location. Let's craft:
```
**Direction des dépendances** : `kinetic-runtime` définit les traits de device, capacités et mémoire, mais **ne dépend pas** de `kinetic-models`. `kinetic-models` définit `ModelRunner` et dépend de `kinetic-runtime`. Les crates backend dépendent de `kinetic-models` + `kinetic-runtime`. La factory qui connaît les crates concrets vit dans `kinetic-server` (ou un crate `kinetic-backends` au-dessus), pas dans `kinetic-runtime`, pour éviter un cycle.
```
This resolves the ambiguity. But is this the best architecture? Let's think: If `kinetic-runtime` defines device/capabilities/memory traits, and `kinetic-models` defines `ModelRunner` and depends on runtime, then backend crates depend on models + runtime. `kinetic-server` depends on backends and models. The scheduler: where? The doc says runtime has scheduler/pool KV. If runtime has scheduler, it needs `ModelRunner`, which is in models, so runtime would depend on models → cycle. We need to resolve: either move `ModelRunner` to `kinetic-runtime`, or move scheduler to a higher crate (`kinetic-server` or `kinetic-scheduler`). The doc says `kinetic-models` defines `ModelRunner`; runtime has scheduler. That's a cycle if runtime depends on models. To avoid, put the scheduler in `kinetic-server` (or a new `kinetic-scheduler` crate above models), not in runtime. Or define `ModelRunner` in runtime and have models implement it; but the doc says models defines it. We need a consistent proposal.
Let's design cleanly:
- `kinetic-runtime`: device discovery, capabilities, memory, KV pool, **traits** `Backend`, `Device`, `KvPool`, and `ModelRunner`? The doc says `kinetic-models` defines `ModelRunner`. We can change that: put `ModelRunner` in `kinetic-runtime`, because it's an execution interface, not a model description. Then `kinetic-models` depends on runtime and implements `ModelRunner` for the CPU runner; backend crates depend on runtime + models (for descriptors) and implement `ModelRunner`. The scheduler in runtime can call `ModelRunner`. `kinetic-models` does not depend on runtime's scheduler, only on traits. No cycle: runtime ← models ← backends; runtime ← backends; server → backends + models + runtime. The factory in server.
- But the doc currently says `kinetic-models` defines `ModelRunner`. We can correct it to `kinetic-runtime` defines `ModelRunner`, and `kinetic-models` implements the CPU runner. This is cleaner.
- Alternatively, keep `ModelRunner` in models and put scheduler in server; runtime only device/memory/KV traits. But the doc says runtime has scheduler/pool KV. We can split: `kinetic-runtime` = devices/capabilities/memory/KV pool; `kinetic-server` = scheduler; `kinetic-models` = ModelRunner. Then no cycle: models → runtime; backends → models + runtime; server → all. This is also clean. The scheduler in server is okay because it orchestrates requests/backends. The doc's "runtime" name might be misleading; but we can say "scheduler in `kinetic-server` or a `kinetic-scheduler` crate above `kinetic-models`."
Which is better? The user's architecture says `kinetic-runtime` includes scheduler/pool KV. To honor that, put `ModelRunner` in runtime. Let's propose:
- `kinetic-runtime` defines: `Device`, `Capabilities`, `Memory`, `KvPool`, `ModelRunner` trait, `Scheduler`? The scheduler is generic over `ModelRunner`. It can live in runtime.
- `kinetic-models` defines model architectures/blocks and implements `ModelRunner` for the CPU reference; depends on runtime.
- Backend crates implement `ModelRunner`; depend on runtime + models (for model descriptors).
- `kinetic-server` owns the factory and depends on concrete backends; selects one.
- No cycle.
This is a clean correction. We should update §0.2 accordingly. The current doc says `kinetic-models` defines `ModelRunner`; we can change to `kinetic-runtime` defines `ModelRunner`, and `kinetic-models` provides the CPU implementation. Then §0.3 step 2 should say "Extraire dans `kinetic-runtime` un trait `ModelRunner`..." and step 3 move executors. The fiches/prompts mention `ModelRunner` in `kinetic-models`; we need to update them too. That's a lot of edits. Alternatively, keep `ModelRunner` in models and move scheduler to server, which requires fewer edits? The doc says runtime has scheduler; we can change to "runtime fournit les traits device/mémoire/KV; le scheduler vit dans `kinetic-server` (ou `kinetic-scheduler`) au-dessus de `ModelRunner`." This is also a significant edit but localized.
Which is more aligned with the existing code? The server already has `Engine` and `Sequence`; moving the scheduler into `kinetic-runtime` is a new crate. The `ModelRunner` trait in models is natural. The cleanest minimal change: keep `ModelRunner` in models; `kinetic-runtime` = devices/capabilities/memory/KV pool (no scheduler); scheduler = `kinetic-server` or a new `kinetic-scheduler` crate above models. But the user's architecture explicitly says `kinetic-runtime` has "ordonnanceur, pool KV, métriques." We can adjust: `kinetic-runtime` has the scheduler, and `ModelRunner` moves to runtime. This is a bigger conceptual change but more coherent.
Given the user wants missions, not a full architecture rewrite, we can add a "Point à trancher" note in `docs/experts.md` rather than rewrite everything:
```
**Point à trancher avant M06** : le doc place à la fois le scheduler dans `kinetic-runtime` et le trait `ModelRunner` dans `kinetic-models`. Pour éviter un cycle, choisir :
- A) `ModelRunner` dans `kinetic-runtime`, `kinetic-models` l'implémente pour le CPU ; le scheduler runtime appelle le trait. Recommandé.
- B) `ModelRunner` dans `kinetic-models`, scheduler dans `kinetic-server` (ou `kinetic-scheduler`), `kinetic-runtime` limité aux devices/capacités/mémoire/KV.
```
This flags the issue without forcing a rewrite. The mission M06 can include "trancher A ou B" as WP0. This is a good approach: the mission catalog deepens it. We don't need to edit §0.2 heavily; add a note. But the doc's current text says `kinetic-models` exposes `ModelRunner`; we can leave it and add the decision point. The mission M06 will resolve it.

4. Add a link to `missions.md` in §7.
5. Add a "Missions" note near §2.1 or §3: "Chaque profil est décliné en missions M01–M15 dans `missions.md`."
6. Maybe add a "Statut" table in `docs/experts.md`? Not necessary.

### Update fiches

Add "Mission détaillée : M0x" to each fiche. This is 10 edits. We can use `edit` with the header line. For example:
- Fiche 1 header: `## Fiche 1 — ...` → add a line after the header: `**Mission détaillée** : [M02](missions.md#m02--préfill-vulkan...) et [M03](...), [M07](...).`
But linking to anchors with accents is tricky. Use plain text: `**Missions détaillées** : M02 (préfill Vulkan), M03 (MoE), M07 (HIP). Voir `docs/experts/missions.md`.`
This avoids anchor issues. Good.
- Fiche 2: M04 (KV), M05 (scheduler).
- Fiche 3: M06 (backend consolidation).
- Fiche 4: M08 (quantification/KV formats), M09? Actually M08 is quantization; M09 is model coverage? Let's define IDs:
  - M01 Baseline/QA.
  - M02 Vulkan prefill.
  - M03 Vulkan MoE.
  - M04 KV cache.
  - M05 Runtime scheduler.
  - M06 Backend consolidation.
  - M07 HIP.
  - M08 Quantization & KV formats.
  - M09 Model coverage.
  - M10 Tokenizers.
  - M11 Server/API.
  - M12 Multimodal.
  - M13 Build/release.
  - M14 Security.
  - M15 Documentation/DX.
- Fiche 4: M08.
- Fiche 5: M01, and supports all.
- Fiche 6: M09.
- Fiche 7: M10.
- Fiche 8: M11, M14.
- Fiche 9: M12.
- Fiche 10: M13, M14, M15.
- Fiche 1: M02, M03, M07.
- Fiche 2: M04, M05.
- Fiche 3: M06.
Good.

Fix Fiche 4 line 412: "KV f16 seulement." → "KV Vulkan f16 seulement ; CUDA a déjà f16/q8_0 ; CPU en f32."
Fix Fiche 1 scope: remove `crates/kinetic-models/src/runner/**` or change to "propositions seulement". Let's edit:
```
- `crates/kinetic-models/src/runner/**` (à créer : trait `ModelRunner` et
  runner CPU)
```
to:
```
- `crates/kinetic-models/src/runner/**` : **lecture seule** ; le trait
  `ModelRunner` et le runner CPU sont pilotés par le Rust backend (Fiche 3).
```
This clarifies. But the fiche's scope is "Périmètre d'écriture"; adding a read-only item under it is odd. Better remove it and add a note:
```
- `crates/kinetic-runtime/src/{device,capabilities,factory,memory}.rs`
  (propositions d'API uniquement)
- `PLAN.md` ...
```
Remove the `kinetic-models/src/runner` line. The GPU AMD expert doesn't write there. Add a line under "Interactions": "**Rust backend** : le trait `ModelRunner` et le runner CPU sont de son périmètre ; le GPU AMD ne fait que proposer les capacités nécessaires." This is cleaner.
Also Fiche 1's context includes `crates/kinetic-models/src/vulkan/*` (to move); that's read-only context, fine.

Add to Fiche 2 a "Spécialisation KV" subsection or a line:
```
**Chantier prioritaire** : M04 (KV cache) avant M05 (scheduler). Voir
`docs/experts/missions.md`.
```
This deepens the mission.

Add to Fiche 8 "Sécurité" note:
```
**Mission transverse** : M14 (sécurité), avec le Rust backend et le
build/release.
```
Add to Fiche 10 "Documentation" note:
```
**Mission transverse** : M15 (documentation/DX).
```

### Update prompts

Add "Mission détaillée : M0x" to each agent. 10 edits. For Agent 2, add a "PREMIER CHANTIER : M04" block. For Agent 3, add a "DÉCISION M06" note about dependency direction. For Agent 5, add "M01 est ton premier livrable". For Agent 8, add "M14 sécurité". For Agent 10, add "M13 + M14 + M15". For Agent 1, add "M02/M03/M07". For Agent 4, add "M08". For Agent 6, "M09". For Agent 7, "M10". For Agent 9, "M12".

Add common contract sections:
- **Bootstrap** : "Si `ledger.jsonl` ou `configs/` n'existent pas, créer une entrée minimale au format documenté ; ne pas restructurer."
- **Sécurité** : "Ne pas exposer de secrets ; valider les entrées ; ne pas ajouter de dépendance réseau non nécessaire ; signaler toute surface d'attaque."
- **Documentation** : "Mettre à jour la doc affectée et `PLAN.md` ; toute nouvelle option/route doit être documentée."
- **Coordination** : "Un seul agent par fichier ; si un fichier est dans le périmètre d'un autre agent, demander avant d'écrire. Agent 1 ne déplace pas les exécuteurs (Agent 3). Agent 5 possède les workflows CI perf ; Agent 10 les workflows release."
- **Definition of done** : "Tests verts, mesure, ledger, doc, rapport."

Add to Agent 2 prompt:
```
PREMIER CHANTIER — M04 (KV cache)
- Auditer le KV actuel : CPU f32, CUDA f16/q8_0, Vulkan f16, un slot Vulkan,
  pas de paged KV, pas de partage de blocs, Warm Start = cache d'état exact.
- Instrumenter : octets alloués/utilisés, fragmentation, cache hit, évictions.
- Concevoir le pool unifié, puis la pagination, puis le partage de préfixe.
- Ne pas toucher aux noyaux d'attention : coordonner avec Agent 1.
- Livrer un design doc + prototype + mesures avant M05.
```
This is the deep KV mission.

Add to Agent 1 prompt:
```
MISSIONS : M02 (préfill), M03 (MoE), M07 (HIP).
Ne pas déplacer les exécuteurs de kinetic-models : c'est M06/Agent 3.
```
Add to Agent 3 prompt:
```
MISSION : M06 (consolidation backend).
DÉCISION À PRENDRE : direction des dépendances runtime/models et emplacement
de la factory. Voir M06 WP0. Ne pas modifier les noyaux.
```
Add to Agent 4 prompt:
```
MISSION : M08 (quantification et formats KV).
Le KV q8_0 existe déjà côté CUDA ; l'écart est Vulkan. Ne pas le réimplémenter.
```
Add to Agent 5 prompt:
```
MISSION : M01 (baseline, THEORY.md, ledger, CI GPU, conformance Vulkan).
C'est le prérequis de toutes les autres missions. Créer le squelette de
ledger/configs avant les autres agents.
```
Add to Agent 8 prompt:
```
MISSIONS : M11 (API) + M14 (sécurité).
```
Add to Agent 10 prompt:
```
MISSIONS : M13 (build/release) + M14 (sécurité, avec Agent 8) + M15 (doc/DX).
```
Add to Agent 6/7/9 prompts: mission IDs.

### Create `docs/experts/README.md`

A short index:
```
# Experts et missions Kinetic

## Ordre de lecture
1. docs/experts.md — analyse et architecture cible.
2. docs/experts/fiches-de-poste.md — qui recruter.
3. docs/experts/missions.md — quoi faire, dans quel ordre.
4. docs/experts/prompts-agents.md — comment lancer un agent.

## Règle
- Une mission = un agent = un périmètre.
- Ne pas lancer deux agents sur le même fichier.
- M01 d'abord.
```
This is useful.

### Mission file details

Let's write `docs/experts/missions.md` with 15 missions. We need to be careful with length and accuracy. Let's draft each mission with a consistent template but not overly long. Aim ~100-130 lines each; total ~1800 lines. That's a lot of writing but feasible.

We can use tables for mission summary and work packages, and bullet lists for deliverables/acceptance. Let's define a compact template:

```
## M01 — Baseline, théorie et harnais de mesure

| Champ | Valeur |
|---|---|
| Priorité | P0 |
| Pilote | Performance/QA (Agent 5) |
| Support | GPU AMD, Runtime |
| Dépend de | — |
| Débloque | Toutes les autres missions |
| Durée | 2–4 semaines |
| Statut | À lancer |

### Objectif
...

### État actuel
...

### Périmètre
- Inclus : ...
- Exclu : ...

### Work packages
| WP | Contenu | Sortie |
|---|---|---|
| WP1 | ... | ... |
...

### Livrables
...

### Critères d'acceptation
...

### Protocole de mesure
...

### Risques
...

### Definition of done
...
```

This is compact and scannable. We can use this template for all 15.

Let's write the content for each mission. Need to ensure no code changes, only plans.

#### M01 — Baseline, théorie et harnais de mesure

- Objective: make measurement automated, reproducible, falsifiable; create `THEORY.md`, `ledger.jsonl`, `configs/`, `best-config.json`; extend conformance to Vulkan; CI GPU; needle; soak; watchdog.
- Current: protocol exists in PLAN/llama-kinetic but manual; no CI GPU, no THEORY, no ledger.
- Work packages:
  1. `THEORY.md`: bandwidth/FLOPS/KV/TTFT ceilings for local machine.
  2. Ledger + configs schema; one JSONL entry per trial.
  3. A/B harness: control/candidate/control, median ≥6, noise, accept/reject.
  4. Vulkan conformance: extend `conformance.py` to `--device vulkan`; token comparison.
  5. Needle test 50%/100%; soak concurrent; memory watchdog.
  6. CI GPU self-hosted; regression threshold.
  7. Dashboard/report.
- Deliverables: files, scripts, CI, baseline report.
- Acceptance: detect 5% with noise ±8%; every number has command/commit/date/machine; negative results kept.
- Measurement: `kinetic bench`, `llama-bench`, `bench_protocol.py`, `conformance.py`, `server_abuse.py`, `same_answer_demo.py`.
- Risks: noisy GPU, thermal, driver; mitigate with idle checks, repeats, controls.
- DoD: M01 report + ledger populated + CI green.

#### M02 — Préfill Vulkan

- Objective: close 4.4× prefill gap; tiled GEMM, coopmat, split-K, async, pipeline cache.
- Current: `gemv_many.wgsl` 32 tokens in flight; not tiled GEMM; `Commands::submit` waits; no pipeline cache.
- Work packages:
  1. Profile RGP; roofline; identify memory-bound vs compute-bound.
  2. Tiled GEMM Q4_K/Q5_K/Q6_K/Q8_0/IQ3_XXS/IQ4_XS; bit-exact.
  3. Coopmat + split-K + LDS tiling.
  4. Async submission with timeline semaphores; dependency graph.
  5. Persistent pipeline cache; startup time.
  6. Tests: batch invariance, prompt/mono equality.
- Deliverables: shaders, tests, bench report.
- Acceptance: pp512/pp2048 ≥2× baseline, tg128 no regression >3%, `edit_identical: true`.
- Dependencies: M01.
- Risks: RDNA4 coopmat support, driver bugs, exactness.
- DoD: merged with tests + ledger.

#### M03 — MoE Vulkan

- Objective: grouped GEMM/GEMV, expert paging, shared experts; close 3.3× decode / 5.4× prefill gap.
- Current: `moe_gemv.wgsl` one expert slot at a time; host reads bound.
- Work packages:
  1. Profile expert routing, memory access.
  2. Grouped GEMV/GEMM for top-k experts; batch tokens by expert.
  3. Shared expert fusion; gate/up/down.
  4. Expert paging/prefetch; async copies.
  5. Tests vs CPU/llama; quality.
- Acceptance: decode/prefill parity, no regression dense.
- Dependencies: M01, M02.
- Risks: divergence, memory bandwidth, expert imbalance.
- DoD.

#### M04 — KV cache

- Objective: audit and redesign KV; unified pool, paged KV, prefix sharing, shared KV zero-copy, q8_0 Vulkan, metrics, offload.
- Current table.
- Work packages:
  1. Audit + metrics.
  2. Baseline memory wall: 32k context, 16 GiB, slots.
  3. Unified pool + dynamic allocation.
  4. Paged KV + block table + gather attention.
  5. Prefix sharing + COW + shared KV zero-copy.
  6. KV quantization: q8_0 Vulkan, q4_0, per-layer policy, K/V separate.
  7. Speculative rollback integration.
  8. Offload/multi-GPU (P2).
- Deliverables: design doc, metrics, prototype, tests, bench.
- Acceptance: 32k context on 16 GiB with ≥4 slots; no OOM; exactness; q8_0 quality; p99/TTFT.
- Dependencies: M01; M05/M08 depend on it.
- Risks: attention kernel changes, fragmentation, quality.
- DoD.

#### M05 — Runtime scheduler

- Objective: token-level scheduler, continuous batching, chunked prefill, cancellation, fairness, p99/TTFT; Vulkan multi-slot.
- Current: one backend per request; Vulkan one slot; `PREFILL_CHUNK=256` per request; `Convoy` shares passes; `MAX_CONNECTIONS=64`; `Connection: close`.
- Work packages:
  1. Map lifecycle: request, slot, KV, cancellation.
  2. Admission/priority/fairness/backpressure.
  3. Chunked prefill global; prefill/decode overlap.
  4. Multi-slot Vulkan backend; keep-alive.
  5. Metrics per token: TTFT, tok/s, p50/p99, cache hit, queue depth.
  6. Exactness under load: same_answer.
- Acceptance: c4/c16/c64 throughput up; p99/TTFT measured; no identity test broken; no leak.
- Dependencies: M01, M04.
- Risks: complexity, starvation, KV rollback.
- DoD.

#### M06 — Backend consolidation

- Objective: one complete backend per API; `kinetic-runtime` traits; `ModelRunner`; move executors; remove features; factory; test backend.
- Current table.
- Work packages:
  1. WP0: decide dependency direction and factory location (A/B).
  2. Create `kinetic-runtime` traits.
  3. Extract `ModelRunner`.
  4. Move CUDA executor.
  5. Move Vulkan executor.
  6. Remove features from models.
  7. Factory + test backend.
  8. CI/build/startup.
- Acceptance: models no CUDA/Vulkan deps; tests green; build/startup no regression; one backend loaded.
- Dependencies: M01; coordinates M02/M03/M04.
- Risks: big refactor, file overlaps.
- DoD.

#### M07 — HIP/ROCm skeleton

- Objective: `kinetic-hip` complete backend skeleton; GEMV Q4_K/Q5_K/Q6_K/Q8_0; attention FP16; conformance vs llama ROCm0 on gfx1201.
- Current: no HIP backend; ROCm 7.2 installed.
- Work packages:
  1. Skeleton: device, memory, capabilities, build.
  2. Port GEMV kernels.
  3. Attention FP16.
  4. Conformance and bench.
  5. CI/opt-in.
- Acceptance: model runs on gfx1201; tokens match; baseline; opt-in.
- Dependencies: M06, M02.
- Risks: ROCm support, kernel portability.
- DoD.

#### M08 — Quantization and KV formats

- Objective: extend quantization; NVFP4, FP8 E4M3, IQ encode, imatrix, KV q8_0/q4_0 Vulkan, per-layer policy, quality.
- Current: 5 targets vs ~30; no imatrix/IQ encode; NVFP4/FP8 absent; KV CUDA q8_0 only; CPU f32; Vulkan f16.
- Work packages:
  1. Audit gates/corpus.
  2. NVFP4/FP8 runtime.
  3. IQ encode + imatrix.
  4. KV q8_0/q4_0 Vulkan; per-layer policy; K/V separate.
  5. Quality: KL/perplexity; f64 reference.
  6. Tests/property/fuzz.
- Acceptance: each format has f64 ref + kernel; no identity regression; quality measured.
- Dependencies: M04 for KV integration; M02 for kernels.
- Risks: quality loss, exactness.
- DoD.

#### M09 — Model coverage

- Objective: cover missing architectures with reusable blocks; 21/180.
- Work packages: families, blocks, mapping, conformance, docs.
- Acceptance: loads/generates, tokens match, no central match.
- Dependencies: M01 (conformance).
- Risks: scope creep; prioritize local models.
- DoD.

#### M10 — Tokenizers/templates/tool calls

- Objective: 13/68 pretokenizers; UGM, special tokens, templates, tool calls.
- Work packages: corpus, pretokenizers, UGM, EOT/EOM/FIM, templates, tool calls, tests.
- Acceptance: ids identical, tool calls valid.
- Dependencies: M01.
- Risks: regex/copying, special tokens.
- DoD.

#### M11 — Server/API

- Objective: API parity, robustness, observability; ~15/45 routes; keep-alive/TLS; 64 connections; multi-model.
- Work packages: routes, keep-alive, TLS/API key, CORS, graceful shutdown, n>1, timings, metrics, client tests.
- Acceptance: clients work, no crash under abuse, metrics.
- Dependencies: M05 for slots; M10 for templates.
- Risks: compatibility, security.
- DoD.

#### M12 — Multimodal

- Objective: 2/73 projectors; vision, audio, OCR.
- Work packages: projectors, audio crate, OCR, parity.
- Acceptance: embeddings close, coherent answers, no text regression.
- Dependencies: M09 for architectures; M01.
- Risks: XL, preprocessing.
- DoD.

#### M13 — Build/release/bindings

- Objective: installable/integrable; 10/200 C functions; partial Python; Docker/packaging.
- Work packages: C ABI, bindings, Docker, installers, CI release, size/time, docs.
- Acceptance: install on 3 OS, ABI tested, reproducible.
- Dependencies: M06 for features; M14 security.
- Risks: cross-platform toolchains.
- DoD.

#### M14 — Security

- Objective: threat model, hardening, supply chain, unsafe audit.
- Work packages: threat model (local server, API keys, TLS, input limits, path traversal, model files), dependency audit, fuzz, unsafe audit, CI, secrets.
- Acceptance: no critical findings, abuse tests pass, unsafe audit, dependency audit.
- Dependencies: M11, M13.
- Risks: local-first threat model; don't over-engineer.
- DoD.

#### M15 — Documentation/DX

- Objective: docs build, onboarding, examples, migration guide, ADRs, benchmarks.
- Work packages: update docs, examples, migration guide, ADRs, contributing, onboarding test.
- Acceptance: docs build, examples pass, onboarding <1 day.
- Dependencies: all; continuous.
- Risks: drift; automate links.
- DoD.

### Mission index table

| ID | Mission | Priorité | Pilote | Support | Dépend de | Débloque | Durée |
...

### Sequencing

- Phase 0: M01.
- Phase 1: M02, M04, M06 (parallel, coordinated).
- Phase 2: M03, M05, M08, M07.
- Phase 3: M09–M13, M14, M15.
- Critical path: M01 → M02 → M03 → M07; M01 → M04 → M05 → M08.
- Note: M06 can run in parallel with M02/M04 but file ownership must be coordinated; M06 touches `kinetic-vulkan` and `kinetic-cuda` to move executors, while M02/M03 touch kernels. To avoid conflicts, sequence M06 before M02/M03? The user's priority is performance; M02/M03 are P0. The architecture consolidation is P0/P1. We can say: M06 WP0 (decision) first, then M06 moves one executor after M02/M03 stabilize, or use a branch/worktree. The mission should mention "worktree/branch" to avoid conflicts. But the user has an active agent; we should not create worktrees. The mission plan can recommend it.
- Maybe sequence: M01 → M06 (mechanical move) → M02/M03 (optimize in new location). But moving the executor before optimizing may add overhead. Alternatively, M02/M03 first, then M06. The docs' phases say "Refactor d'abord; performance après" in §0.3 step 7. But the docs' phases say "Phase 1: Préfill; Phase 2: MoE/HIP; Phase 3: Concurrency." The one-backend-per-API refactor is a prerequisite for HIP, but not for Vulkan prefill. We can propose:
  - M01 first.
  - M02/M03 (performance) and M04 (KV) in parallel with M06 WP0 design.
  - M06 moves executors after M02/M03 merge, or in a dedicated branch.
  - M07 HIP after M06.
This is pragmatic.
- The mission catalog should state "M06 ne doit pas bloquer M02/M03; faire M06 WP0 (design) puis déplacer après stabilisation."

### Update `docs/experts.md` with mission mapping

Add a section after §2 or in §3:
```
### 2.12 Mapping profils → missions
| Profil | Missions |
...
```
This is useful. We can add a compact table.

### Do not work directly on the project

We must avoid:
- Running cargo/build/test.
- Modifying code, scripts, PLAN.md, configs.
- Creating worktrees or moving sessions.
- Running commands that could interfere with the active agent.
We can use `read`/`grep`/`glob` and `edit`/`write` on docs/experts only. We already used shell commands to read; that's fine. For the new work, use `write`/`edit` on docs only.

We should mention in the final response: "Je n'ai touché qu'aux documents `docs/experts/` ; aucun code, `PLAN.md`, script ou config modifié. Je n'ai lancé aucun build/test."

### Plan of action

1. Create `docs/experts/missions.md` with 15 missions.
2. Create `docs/experts/README.md` index.
3. Edit `docs/experts.md`:
   - Add link to missions.
   - Fix Metal.
   - Fix KV row.
   - Add dependency-direction decision note.
   - Add mapping profiles → missions.
4. Edit `docs/experts/fiches-de-poste.md`:
   - Add mission links to each fiche (10 edits).
   - Fix KV f16 line.
   - Fix GPU AMD scope.
   - Add KV priority note to Fiche 2.
   - Add security/doc notes to Fiches 8/10.
5. Edit `docs/experts/prompts-agents.md`:
   - Add mission links to each agent (10 edits).
   - Add common contract sections: bootstrap, security, documentation, coordination, DoD.
   - Add M04 first phase to Agent 2.
   - Add M06 decision note to Agent 3.
   - Add M01 first to Agent 5.
   - Add M08 note to Agent 4.
   - Add M14 to Agent 8, M13/M14/M15 to Agent 10.
6. Verify with `grep`/`read` that all missions and links are consistent.
7. Final response summarizing what was created/deepened, with file paths and no code changes.

This is a substantial amount of writing. Let's start.

First, create `docs/experts/missions.md`. I need to write ~1500 lines. Let's compose carefully. I'll use a consistent template and keep each mission focused.

Let's draft the file content. I'll write it in one `write` call. Need to ensure Markdown is valid, tables align, and no typos.

Let's define the mission IDs and titles:

- M01 — Baseline, théorie et harnais de mesure
- M02 — Préfill Vulkan : GEMM tuilé et soumission asynchrone
- M03 — MoE Vulkan : GEMM groupé et pagination des experts
- M04 — KV cache : audit, pool unifié, pagination et partage
- M05 — Ordonnanceur runtime et concurrence
- M06 — Consolidation backend : un seul backend complet par API
- M07 — Backend HIP/ROCm : squelette et noyaux de base
- M08 — Quantification et formats KV
- M09 — Couverture des architectures
- M10 — Tokenizers, templates et tool calls
- M11 — Serveur, API et observabilité
- M12 — Multimodal : vision, audio, OCR
- M13 — Build, release et bindings
- M14 — Sécurité
- M15 — Documentation et expérience développeur

Mapping:
- Agent 1: M02, M03, M07.
- Agent 2: M04, M05.
- Agent 3: M06.
- Agent 4: M08.
- Agent 5: M01, supports all.
- Agent 6: M09.
- Agent 7: M10.
- Agent 8: M11, M14.
- Agent 9: M12.
- Agent 10: M13, M14, M15.

Mission dependencies:
- M01: none; enables all.
- M02: M01.
- M03: M01, M02.
- M04: M01.
- M05: M01, M04.
- M06: M01; coordinates M02/M03/M04.
- M07: M06, M02.
- M08: M01, M04 (KV integration), M02 (kernels).
- M09: M01.
- M10: M01.
- M11: M01, M05 (slots), M10 (templates).
- M12: M01, M09.
- M13: M06.
- M14: M11, M13.
- M15: continuous, all.

Mission phases:
- Phase 0: M01.
- Phase 1: M02, M04, M06 WP0.
- Phase 2: M03, M05, M06 execution, M08.
- Phase 3: M07, M09-M15.

Critical path:
- Performance: M01 → M02 → M03 → M07.
- Concurrency: M01 → M04 → M05 → M08.
- Architecture: M01 → M06 → M07.

Now, write each mission. Let's keep the "État actuel" accurate.

M01:
- Current: protocol in PLAN/llama-kinetic but manual; no CI GPU; no THEORY/ledger.
- Work packages.
- Acceptance.
- Commands.

M02:
- Current: pp512 261.9 vs 1154.2; 4.4×; `gemv_many.wgsl` 32 tokens; `Commands::submit` waits; no pipeline cache.
- Work packages.
- Acceptance.

M03:
- Current: MoE 30B A3B 55.7 vs 181.9 decode; 489 vs 2637 prefill; `moe_gemv.wgsl` one expert slot; host reads.
- Work packages.
- Acceptance.

M04:
- Current table:
  - CPU: `Vec<f32>` per layer, f32, N slots limited by memory, no paging, no sharing.
  - CUDA: f16/q8_0, 1..4 slots, shared KV copied, no paging.
  - Vulkan: f16, 1 slot, no sharing, no paging.
  - Warm Start: exact state cache, 1 GiB RAM/2 GiB disk, not block-level.
  - Convoy: shares passes, not KV.
- Work packages as above.
- Acceptance.

M05:
- Current: `Engine.free` pool; one backend per request; Vulkan one slot; `PREFILL_CHUNK=256` per request; `Convoy` only CPU/CUDA; `MAX_CONNECTIONS=64`; `Connection: close`.
- Work packages.
- Acceptance.

M06:
- Current: CUDA split `kinetic-cuda` + `kinetic-models/src/cuda`; Vulkan split; models features; server branches.
- Work packages.
- Acceptance.
- WP0 decision A/B.

M07:
- Current: no HIP; ROCm 7.2; gfx1201; `kinetic-vulkan` works.
- Work packages.
- Acceptance.

M08:
- Current: 5 targets vs ~30; no imatrix/IQ encode; NVFP4/FP8 absent; KV CUDA q8_0; Vulkan f16; CPU f32.
- Work packages.
- Acceptance.

M09:
- Current: 21/180; missing qwen3vl, deepseek, llama4, gemma2, phi3, command-r, dbrx; blocks MLA/DSA, SSM, T5.
- Work packages.
- Acceptance.

M10:
- Current: 13/68; UGM/RWKV/PLaMo/Tekken/Jais2 missing; special tokens; templates/tool calls.
- Work packages.
- Acceptance.

M11:
- Current: ~15/45 routes; no keep-alive/TLS; 64 connections; `docker stop`; client compat.
- Work packages.
- Acceptance.

M12:
- Current: 2/73 projectors; no audio/video/OCR; `kinetic-vision` contract.
- Work packages.
- Acceptance.

M13:
- Current: 10/200 C functions; partial Python; Docker/packaging; CI multi-OS.
- Work packages.
- Acceptance.

M14:
- Current: local-first server; no threat model; unsafe surface; dependencies; model files.
- Work packages.
- Acceptance.

M15:
- Current: docs good but drift; no migration guide/ADRs; examples.
- Work packages.
- Acceptance.

Let's write the file. I'll use `write` with the full content. It will be long. Let's ensure no accidental code changes.

Before writing, let's check if `docs/experts/missions.md` already exists. It doesn't. Good.

Let's compose the content in a structured way. I'll aim for clarity and conciseness per mission. Use `###` for sections inside each mission. Use tables for summary and work packages. Avoid overly long prose.

Let's start writing.

Compris. Je ne toucherai ni au code, ni à `PLAN.md`, ni aux scripts/configs, et je ne lancerai aucun build/test. Je crée uniquement la couche **missions** dans `docs/experts/`, en approfondissant les missions existantes et en ajoutant les missions manquantes (KV, sécurité, documentation).



**Tool: write**

**Input:**
```json
{}
```

The tool call was not executed because the output token limit (stop reason: length) was reached before you could complete the arguments. Break large payloads into smaller tool calls and reissue the call with complete arguments.

Input excerpt (first 65 of 65 characters):
{"path": "/home/mrnemesys/Devs/kinetic/docs/experts/missions.md"}

---

## Assistant

_Thinking:_

The write call failed because the content was too long for the output token limit. I need to split into smaller writes. Strategy: create the file with the header/index/first missions, then append missions in chunks using shell heredoc? But the user said not to work directly on the project; writing docs is fine. Using `write` with smaller content, then `edit` to append. Or use `shell` with `cat >>` heredoc, but that's a shell command and may be large; also the instructions prefer dedicated tools. We can use `write` for the first chunk, then `edit` with `oldString` at the end to append. Each chunk must be under the output token limit. The limit seems to be around 8k tokens? The failed call was very long (maybe 20k tokens). We can split into ~4-6 chunks.

Let's create the file in parts:
1. Header, index, usage, mapping, dependency graph, M01-M02.
2. M03-M05.
3. M06-M08.
4. M09-M11.
5. M12-M15, closing.

We need to use `write` for part 1, then `edit` to append. For appending, use `edit` with `oldString` being the last line of the file and `newString` that line + new content. This is reliable. Alternatively, use `shell` with `cat >> file <<'EOF'` for each chunk; but shell output may be noisy, and the instructions prefer dedicated tools. `edit` is fine.

Let's ensure each chunk is not too long. Aim ~300-400 lines per chunk.

Let's write part 1: header, index, mission map, sequencing, M01, M02. Then append.

I need to be mindful of the output token limit per tool call. The failed call had the full content; we don't know the limit exactly. Let's keep each call around 6k-8k tokens max. A 300-line chunk with tables is ~4k tokens. Good.

Let's start.

The file is too large for a single write. I’ll create it in sections.



**Tool: write**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/missions.md",
  "content": "# Missions des experts Kinetic\n\n> **Statut** : catalogue de missions, 22 novembre 2026.\n> **But** : transformer les 10 profils de `docs/experts.md` en missions\n> exécutables, avec objectif, périmètre, work packages, livrables, critères\n> d'acceptation, dépendances et protocole de mesure.\n>\n> **Règle** : une mission = un agent = un périmètre d'écriture. Ne pas lancer\n> deux agents sur les mêmes fichiers. M01 passe avant toutes les autres.\n\n---\n\n## 0. Comment utiliser ce document\n\n1. Lire `docs/experts.md` pour l'analyse et l'architecture cible.\n2. Choisir une mission ci-dessous.\n3. Lire la fiche de poste correspondante dans\n   `docs/experts/fiches-de-poste.md`.\n4. Donner à l'agent le prompt correspondant dans\n   `docs/experts/prompts-agents.md`, en y ajoutant l'ID de mission.\n5. L'agent commence par lire son périmètre, mesure la baseline, propose un\n   seul changement, mesure, consigne dans `PLAN.md` et le ledger, puis\n   rapporte.\n\n### 0.1 Règles communes\n\n- Un seul changement par essai ; une configuration par essai.\n- Baseline, candidat, puis contrôle ; médiane de ≥ 6 échantillons.\n- Accepter seulement si le candidat bat les deux contrôles et ne régresse\n  rien de plus de ~3 %.\n- Toute optimisation doit préserver l'égalité batch / mono-token et les\n  gates d'identité.\n- Consigner les échecs et les résultats négatifs.\n- Ne pas copier de code depuis llama.cpp ou un autre moteur.\n- Ne pas modifier les fichiers hors du périmètre de la mission.\n- Si `ledger.jsonl` ou `configs/` n'existent pas encore, créer une entrée\n  minimale au format documenté par M01 ; ne pas restructurer.\n\n### 0.2 Statuts\n\n| Statut | Sens |\n|---|---|\n| `à lancer` | Mission prête, aucun agent actif |\n| `en cours` | Un agent travaille dessus |\n| `bloquée` | Dépend d'une décision ou d'une mission |\n| `terminée` | Critères d'acceptation remplis et consignés |\n| `abandonnée` | Résultat négatif documenté |\n\n---\n\n## 1. Index des missions\n\n| ID | Mission | Priorité | Pilote | Support | Dépend de | Débloque | Durée |\n|---|---|---|---|---|---|---|---|\n| M01 | Baseline, théorie et harnais de mesure | P0 | Performance/QA | GPU, Runtime | — | Toutes | 2–4 sem. |\n| M02 | Préfill Vulkan : GEMM tuilé et soumission asynchrone | P0 | GPU AMD | Numérique, QA | M01 | M03, M07 | 6–10 sem. |\n| M03 | MoE Vulkan : GEMM groupé et pagination des experts | P0 | GPU AMD | Runtime, Numérique | M01, M02 | M07 | 6–8 sem. |\n| M04 | KV cache : audit, pool unifié, pagination et partage | P0 | Runtime | GPU, Numérique, QA | M01 | M05, M08 | 6–10 sem. |\n| M05 | Ordonnanceur runtime et concurrence | P0 | Runtime | Rust backend, QA | M01, M04 | M11 | 6–10 sem. |\n| M06 | Consolidation backend : un seul backend complet par API | P0/P1 | Rust backend | GPU, Runtime, QA | M01 | M07, M13 | 8–12 sem. |\n| M07 | Backend HIP/ROCm : squelette et noyaux de base | P1 | GPU AMD | Rust backend, QA | M06, M02 | — | 8–12 sem. |\n| M08 | Quantification et formats KV | P0/P1 | Numérique | GPU, Runtime | M01, M04, M02 | — | 8–12 sem. |\n| M09 | Couverture des architectures | P1 | Intégration modèles | Numérique, QA | M01 | M12 | continu |\n| M10 | Tokenizers, templates et tool calls | P1 | Tokenizers | Serveur, QA | M01 | M11 | 6–10 sem. |\n| M11 | Serveur, API et observabilité | P1 | Serveur/API | Runtime, Tokenizers, QA | M01, M05, M10 | M14 | 8–12 sem. |\n| M12 | Multimodal : vision, audio, OCR | P2 | Multimodal | Modèles, GPU, QA | M01, M09 | — | XL |\n| M13 | Build, release et bindings | P2 | Build/release | Serveur, Rust backend | M06 | M14 | 8–12 sem. |\n| M14 | Sécurité | P1/P2 | Serveur/API | Rust backend, Build/release | M11, M13 | — | 4–6 sem. |\n| M15 | Documentation et expérience développeur | P2 | Build/release | Tous | continu | — | continu |\n\n### 1.1 Séquencement\n\n| Phase | Durée | Missions | Porte de sortie |\n|---|---|---|---|\n| 0. Mesure | 2–4 sem. | M01 | Baseline, `THEORY.md`, ledger, CI GPU |\n| 1. Performance | 6–10 sem. | M02, M04, M06 WP0 | pp ≥ 2×, KV instrumenté, design backend validé |\n| 2. MoE et concurrence | 6–10 sem. | M03, M05, M06, M08 | MoE, cN, KV q8 Vulkan |\n| 3. HIP et couverture | continu | M07, M09–M13 | HIP opt-in, familles, API, produit |\n| 4. Consolidation | continu | M14, M15 | Sécurité, doc, release |\n\n### 1.2 Chemin critique\n\n- Performance : **M01 → M02 → M03 → M07**.\n- Concurrence : **M01 → M04 → M05 → M08**.\n- Architecture : **M01 → M06 → M07**.\n\nM06 peut démarrer en parallèle de M02/M04 pour le design (WP0), mais le\ndéplacement des exécuteurs doit se faire dans une branche ou un worktree\ndédié, après stabilisation des noyaux, pour éviter les conflits de fichiers.\n\n---\n\n## 2. Missions détaillées\n\n## M01 — Baseline, théorie et harnais de mesure\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0 |\n| Pilote | Performance/QA (Agent 5) |\n| Support | GPU AMD, Runtime |\n| Dépend de | — |\n| Débloque | Toutes les autres missions |\n| Durée | 2–4 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nRendre la mesure automatisée, reproductible et infalsifiable : plafonds\nthéoriques, baseline, ledger, configurations, protocole A/B, CI GPU,\nconformance Vulkan, tests de contexte long et soak.\n\n### État actuel\n\n- Le protocole existe dans `PLAN.md` et `llama-kinetic.md` §7.10, mais il est\n  manuel.\n- Pas de CI GPU, pas de `THEORY.md`, pas de `ledger.jsonl`, pas de\n  `configs/`, pas de `best-config.json`.\n- `scripts/bench_protocol.py`, `bench_vs_llamacpp.py`, `conformance.py`,\n  `server_abuse.py`, `same_answer_demo.py` existent.\n- `conformance.py` ne connaît que `cpu` et `cuda` ; l'extension Vulkan est à\n  faire.\n\n### Périmètre\n\n**Inclus** :\n\n- `THEORY.md` : bande passante, FLOPS, KV, TTFT, plafonds par modèle.\n- `ledger.jsonl` : une entrée par essai, avec commande, commit, date, état\n  machine, résultat, verdict.\n- `configs/` : configurations complètes et reproductibles.\n- `best-config.json` : meilleure configuration connue par modèle/backend.\n- Protocole A/B : contrôle, candidat, contrôle.\n- CI GPU self-hosted (Vulkan/ROCm) avec seuil de régression.\n- Extension de `conformance.py` à Vulkan.\n- Test needle 50 %/100 % de contexte.\n- Soak concurrent et watchdog mémoire.\n- Dashboard de régression et rapport type.\n\n**Exclu** :\n\n- Modifier les noyaux ou le runtime pour améliorer un chiffre.\n- Publier un chiffre sans commande, commit, date et état machine.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Écrire `THEORY.md` : bande passante mesurée, FLOPS, KV, TTFT, plafond par modèle | Document + formule de plafond |\n| WP2 | Définir le schéma `ledger.jsonl` et `configs/` ; créer le squelette | Fichiers + exemple d'entrée |\n| WP3 | Automatiser le protocole A/B avec contrôles avant/après | `scripts/ab_protocol.py` ou extension |\n| WP4 | Étendre `conformance.py` à `--device vulkan` | Corpus + rapport de conformance |\n| WP5 | Ajouter needle 50 %/100 %, soak concurrent, watchdog mémoire | Scripts + seuils |\n| WP6 | Ajouter une CI GPU avec seuil de régression | Workflow + artefacts |\n| WP7 | Dashboard et rapport type de régression | Rapport + historique |\n\n### Livrables\n\n- `THEORY.md`.\n- `ledger.jsonl` et `configs/` avec au moins 10 essais de baseline.\n- `best-config.json`.\n- `scripts/` : A/B, needle, soak, watchdog.\n- CI GPU opérationnelle.\n- Rapport de baseline pour le dense 27B et le MoE 30B A3B.\n\n### Critères d'acceptation\n\n- Un candidat à +5 % est distingué du bruit avec un protocole reproductible.\n- Une régression de 5 % est détectée automatiquement.\n- Chaque chiffre publié a une commande, un commit, une date et un état\n  machine.\n- Les résultats négatifs sont conservés.\n- `conformance.py` compare Vulkan au runner CPU sur le corpus de contrôle.\n\n### Protocole de mesure\n\n```sh\ntarget/release/kinetic doctor\ntarget/release/kinetic bench MODEL.gguf --device vulkan --json -r 6\nllama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6\npython scripts/bench_protocol.py MODEL.gguf --llama-bin /path/to/llama-bench\npython scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \\\n  --llama /path/to/llama.cpp/bin --device vulkan\npython scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091\npython scripts/same_answer_demo.py --model MODEL.gguf --kinetic target/release/kinetic\nrocm-smi --showuse --showtemp --showmemuse\n```\n\n### Dépendances et handoffs\n\n- Fournit le ledger et les configs à M02–M15.\n- Reçoit des mesures de M02/M03/M04/M05.\n- Valide les gates de M08/M11.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Bruit GPU/thermique | Machine idle, répétitions, contrôles, médiane |\n| Driver ou ROCm instable | Épingler version, redémarrer, documenter |\n| Mesure non représentative | Même fichier, même backend, même état machine |\n| Dérive du ledger | Schéma strict, une entrée par essai, revue |\n\n### Definition of done\n\n- `THEORY.md` relu et accepté.\n- Ledger rempli avec la baseline.\n- CI GPU verte.\n- `conformance.py` Vulkan fonctionne.\n- Rapport de baseline publié dans `PLAN.md`.\n\n---\n\n## M02 — Préfill Vulkan : GEMM tuilé et soumission asynchrone\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0 |\n| Pilote | GPU AMD (Agent 1) |\n| Support | Numérique (Agent 4), QA (Agent 5) |\n| Dépend de | M01 |\n| Débloque | M03, M07 |\n| Durée | 6–10 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nCombler l'écart de préfill Vulkan : passer de 261,9 t/s à ≥ 2× la baseline\nKinetic sur pp512/pp2048, sans régression > 3 % sur tg128, avec exactitude\nbit-exacte.\n\n### État actuel\n\n- Dense 27B IQ3_XXS : 261,9 t/s en préfill contre 1154,2 pour llama.cpp\n  Vulkan, soit 4,4×.\n- `gemv_many.wgsl` garde 32 tokens en vol ; ce n'est pas un GEMM tuilé.\n- `Commands::submit` soumet puis attend `vkWaitForFences` ; pas de timeline\n  semaphores.\n- Pas de `VkPipelineCache` persisté.\n- Le décodage dense est à 95 % de llama.cpp (38,4 contre 40,1 t/s).\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-vulkan/**` : shaders, pipelines, commandes, GEMV, attention.\n- Tests d'exactitude et de non-régression dans `kinetic-vulkan` et\n  `kinetic-models`.\n- `PLAN.md`, `ledger.jsonl`, `configs/` pour les essais.\n\n**Exclu** :\n\n- Déplacer les exécuteurs de `kinetic-models` (M06).\n- Modifier les architectures ou les blocs.\n- Modifier CUDA.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Profiler RGP sur pp512 ; roofline ; memory-bound vs compute-bound | Rapport de profil |\n| WP2 | GEMM tuilé Q4_K, Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ4_XS | Shader + tests bit-exact |\n| WP3 | Coopmat (`VK_KHR_cooperative_matrix`), split-K, tuilage LDS | Variantes mesurées |\n| WP4 | Soumission asynchrone avec timeline semaphores | Commandes + tests |\n| WP5 | `VkPipelineCache` persisté ; temps de démarrage | Cache + mesure |\n| WP6 | Tests batch-invariance et égalité passe prompt / mono-token | Tests verts |\n\n### Livrables\n\n- GEMM de préfill tuilé pour les formats prioritaires.\n- Variante coopmat et split-K.\n- Soumission asynchrone sans casser les dépendances.\n- Cache de pipelines persisté.\n- Rapport de bench avant/après avec RGP.\n\n### Critères d'acceptation\n\n- pp512/pp2048 ≥ 2× la baseline Kinetic.\n- tg128 sans régression > 3 %.\n- Tous les tests d'identité verts.\n- `kinetic bench` dit `edit_identical: true`.\n- Chaque essai est dans `PLAN.md` avec commande, commit, date et verdict.\n\n### Protocole de mesure\n\n```sh\ncargo build --release --locked -p kinetic-cli --features vulkan\ncargo test --release --locked -p kinetic-vulkan\ncargo test --release --locked -p kinetic-models\ntarget/release/kinetic bench MODEL.gguf --device vulkan --json -r 6\nllama-bench -m MODEL.gguf -dev Vulkan0 -p 512 -n 128 -r 6\nRGP capture sur pp512\nrocm-smi --showuse --showtemp --showmemuse\n```\n\n### Dépendances et handoffs\n\n- Reçoit le ledger et la baseline de M01.\n- Fournit les noyaux de base à M03 et M07.\n- Coordonne avec M06 : ne pas déplacer les exécuteurs pendant M02.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Coopmat mal supporté par RADV/RDNA 4 | Fallback tuilage classique, mesurer |\n| Exactitude cassée par le tuilage | Test bit-exact avant/après |\n| Régression décodage | Contrôle tg128 à chaque essai |\n| Conflit avec M06 | Branche dédiée, pas de déplacement de fichiers |\n\n### Definition of done\n\n- KPI atteint et consigné.\n- Tests verts.\n- Shaders documentés.\n- Ledger à jour.\n- Rapport final avec métrique, baseline, maintenant, plafond, méthode.\n"
}
```

Created file successfully: docs/experts/missions.md

---

## Assistant

_Thinking:_

Now append M03-M05. Use `edit` with oldString being the last line of the file and newString adding content. The last line is "- Rapport final avec métrique, baseline, maintenant, plafond, méthode." Let's append.

**Tool: edit**

**Input:**
```json
{
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/missions.md",
  "oldString": "- Rapport final avec métrique, baseline, maintenant, plafond, méthode.",
  "newString": "- Rapport final avec métrique, baseline, maintenant, plafond, méthode.\n\n---\n\n## M03 — MoE Vulkan : GEMM groupé et pagination des experts\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0 |\n| Pilote | GPU AMD (Agent 1) |\n| Support | Runtime (Agent 2), Numérique (Agent 4) |\n| Dépend de | M01, M02 |\n| Débloque | M07 |\n| Durée | 6–8 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nDébloquer le MoE Vulkan : passer de 55,7 à ≥ 181,9 t/s en décodage et de\n489 à ≥ 2637 t/s en préfill sur le MoE 30B A3B, sans régression dense.\n\n### État actuel\n\n- MoE 30B A3B : 55,7 t/s en décodage contre 181,9 pour llama.cpp, soit 3,3×.\n- Préfill : 489 contre 2637 t/s, soit 5,4×.\n- `moe_gemv.wgsl` lit les experts un slot à la fois ; pas de GEMM groupé.\n- Le décodage est borné par les lectures d'experts en mémoire hôte, pas par\n  le calcul GPU.\n- Les experts sont paginés côté hôte, mais le noyau ne regroupe pas les\n  tokens par expert.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-vulkan/src/moe.rs` et shaders MoE.\n- Tests de qualité et de non-régression MoE.\n- `PLAN.md`, `ledger.jsonl`, `configs/` pour les essais.\n\n**Exclu** :\n\n- Changer le format GGUF ou le routeur.\n- Modifier les architectures.\n- Déplacer les exécuteurs (M06).\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Profiler le routage et les accès experts ; identifier le goulot | Rapport RGP |\n| WP2 | GEMM/GEMV groupé : regrouper les tokens par expert | Shader + tests |\n| WP3 | Fusion gate/up/down et expert partagé | Variantes mesurées |\n| WP4 | Pagination/préchargement des experts ; copies asynchrones | Réduction des stalls |\n| WP5 | Tests de qualité vs CPU/llama.cpp ; non-régression dense | Tests + bench |\n\n### Livrables\n\n- GEMM groupé pour top-k experts.\n- Expert partagé fusionné.\n- Pagination/préchargement des experts.\n- Rapport de bench décodage/préfill.\n\n### Critères d'acceptation\n\n- Décodage MoE ≥ 181,9 t/s ou ≥ 3× la baseline Kinetic.\n- Préfill MoE ≥ 2637 t/s ou ≥ 5× la baseline Kinetic.\n- Pas de régression > 3 % sur le dense.\n- Qualité mesurée (KL/perplexité) et consignée.\n\n### Protocole de mesure\n\n```sh\ncargo test --release --locked -p kinetic-vulkan\ntarget/release/kinetic bench MOE.gguf --device vulkan --json -r 6\nllama-bench -m MOE.gguf -dev Vulkan0 -p 512 -n 128 -r 6\nrocm-smi --showuse --showtemp --showmemuse\n```\n\n### Dépendances et handoffs\n\n- Reçoit le GEMM de préfill de M02.\n- Fournit les noyaux MoE à M07.\n- Coordonne avec M04 pour le KV des experts si nécessaire.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Déséquilibre des experts | Regroupement dynamique, mesures par batch |\n| Bande passante mémoire | Préchargement, compression, layout |\n| Qualité | Gate KL/perplexité avant publication |\n| Conflit avec M06 | Branche dédiée |\n\n### Definition of done\n\n- KPI atteint et consigné.\n- Tests verts.\n- Rapport final avec métrique, baseline, maintenant, plafond, méthode.\n\n---\n\n## M04 — KV cache : audit, pool unifié, pagination et partage\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0 |\n| Pilote | Runtime (Agent 2) |\n| Support | GPU AMD (Agent 1), Numérique (Agent 4), QA (Agent 5) |\n| Dépend de | M01 |\n| Débloque | M05, M08 |\n| Durée | 6–10 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nComprendre et redesigner le cache KV : mesurer l'existant, unifier le pool,\nintroduire la pagination et le partage de préfixe, quantifier le KV côté\nVulkan, et lever le mur mémoire qui limite le contexte et la concurrence.\n\n### État actuel\n\n| Backend | Stockage | Quantification | Slots | Partage | Pagination |\n|---|---|---|---|---|---|\n| CPU | `Vec<f32>` par couche | f32 | N, limité par la mémoire | Non | Non |\n| CUDA | buffer device | f16 ou q8_0 | 1 à 4 | Copie pour shared KV | Non |\n| Vulkan | buffer device | f16 uniquement | 1 | Non (re-projection) | Non |\n| Warm Start | cache d'état hôte | exact | N/A | Préfixe exact, pas de blocs | Non |\n\n- Le KV est contigu par séquence et réservé pour le contexte complet.\n- `KvCache` Vulkan est en f16, non quantifié.\n- CUDA a déjà q8_0 ; CPU est en f32.\n- Pas de pool unifié, pas de block table, pas de partage de blocs entre\n  requêtes.\n- `Convoy` mutualise des passes, pas le KV.\n- `Warm Start` est un cache d'état exact (1 Gio RAM / 2 Gio disque), pas un\n  cache de blocs.\n- Le rollback spéculatif est contraint par la copie de l'ancien K/V côté CPU\n  et par `MAX_BATCH` côté GPU.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-runtime/**` : pool KV, block table, métriques.\n- `crates/kinetic-models/src/{convoy,memory,per_layer}.rs` et `warm/**`.\n- `crates/kinetic-server/src/{engine,sequence}.rs` pour l'intégration.\n- Tests et mesures dans `kinetic-models`, `kinetic-vulkan`, `kinetic-cuda`.\n- `PLAN.md`, `ledger.jsonl`, `configs/`.\n\n**Exclu** :\n\n- Modifier les noyaux d'attention sans coordination avec Agent 1.\n- Remplacer Warm Start.\n- Changer les contrats `Family`/`Target`.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Audit : cartographier chaque allocation KV, slot, cycle de vie | Cartographie + métriques |\n| WP2 | Instrumenter : octets alloués/utilisés, fragmentation, cache hit, évictions | Métriques exposées |\n| WP3 | Baseline : contexte max par slot, nombre de slots sur 16 Gio, time-to-OOM | Rapport de mur mémoire |\n| WP4 | Pool unifié : budget, croissance dynamique, éviction, fallback hôte | Design + prototype |\n| WP5 | Pagination : block table, taille de bloc, gather attention, fragmentation | Prototype + tests |\n| WP6 | Partage de préfixe : hash de blocs, COW, shared KV zéro-copie | Design + prototype |\n| WP7 | Quantification KV : q8_0 Vulkan, q4_0, politique par couche, K/V séparés | Formats + gates |\n| WP8 | Rollback spéculatif : troncature du block table, checkpoint | Tests d'exactitude |\n| WP9 | Offload froid et multi-GPU (P2) | Design + mesures |\n\n### Livrables\n\n- Design doc du pool KV et de la pagination.\n- Métriques KV par requête et par backend.\n- Prototype de pool unifié.\n- Prototype de pagination avec tests d'exactitude.\n- Design du partage de préfixe et du shared KV zéro-copie.\n- Rapport de bench mémoire/contexte/concurrence.\n\n### Critères d'acceptation\n\n- Un contexte de 32k tient sur 16 Gio avec ≥ 4 slots, sans OOM.\n- Les métriques KV sont exposées dans `kinetic doctor` ou le serveur.\n- La pagination ne change pas les tokens sur le corpus de contrôle.\n- q8_0 Vulkan double environ le contexte utile, avec qualité mesurée.\n- Le partage de préfixe réduit la mémoire des préfixes communs.\n- Aucune régression > 3 % sur les benches existants.\n\n### Protocole de mesure\n\n```sh\ntarget/release/kinetic doctor\ntarget/release/kinetic bench MODEL.gguf --device vulkan --json -r 6\ntarget/release/kinetic serve MODEL.gguf --device vulkan --parallel 4\npython scripts/same_answer_demo.py --model MODEL.gguf --kinetic target/release/kinetic\npython scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091\nrocm-smi --showmemuse --showuse --showtemp\n```\n\n### Dépendances et handoffs\n\n- Reçoit la baseline et le ledger de M01.\n- Fournit le pool KV à M05 et les formats KV à M08.\n- Coordonne avec Agent 1 pour les noyaux d'attention paginée.\n- Coordonne avec Agent 4 pour la quantification et les gates.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Complexité de la pagination | Prototype incrémental, tests d'abord |\n| Fragmentation | Métriques, défragmentation, taille de bloc |\n| Qualité q8_0/q4_0 | KL/perplexité, gate par format |\n| Rollback spéculatif | Tests d'égalité, checkpoint |\n| Conflit avec M05/M06 | Périmètre et branches séparés |\n\n### Definition of done\n\n- Design doc accepté.\n- Prototype mesuré.\n- Tests d'exactitude verts.\n- Métriques disponibles.\n- Ledger à jour.\n- Rapport final avec mémoire, contexte, concurrence, qualité.\n\n---\n\n## M05 — Ordonnanceur runtime et concurrence\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0 |\n| Pilote | Runtime (Agent 2) |\n| Support | Rust backend (Agent 3), QA (Agent 5) |\n| Dépend de | M01, M04 |\n| Débloque | M11 |\n| Durée | 6–10 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nFaire passer le serveur d'un modèle « N backends indépendants » à un\nordonnanceur token-level qui tient la concurrence, le TTFT et le p99, sans\ncasser RAD-1, Warm Start ni Convoy.\n\n### État actuel\n\n- `Engine.free` est un `Mutex<Vec<(usize, Box<dyn Backend + 'a>)>>` : une\n  requête par backend.\n- `PREFILL_CHUNK = 256` existe, mais le chunking est par requête.\n- `Convoy` mutualise des passes, mais ne planifie pas les requêtes.\n- Le KV est contigu par séquence, non paginé.\n- `MAX_CONNECTIONS = 64`, `Connection: close`, pas de keep-alive.\n- Vulkan n'a qu'un slot ; CUDA/CPU ont jusqu'à 4 slots par défaut.\n- Pas de métriques par token (TTFT, p50/p99, cache hit, queue depth).\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-runtime/**` : ordonnanceur, admission, priorités,\n  annulation, backpressure, métriques.\n- `crates/kinetic-server/src/{engine,sequence,backend,server}.rs`.\n- `crates/kinetic-models/src/{convoy,memory,per_layer}.rs`, `warm/**`.\n- Tests de charge et d'exactitude sous charge.\n\n**Exclu** :\n\n- Modifier les noyaux GPU.\n- Changer les contrats `Family`/`Target`/`Backend` sans décision.\n- Introduire une dépendance async lourde sans justification.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Cartographier le cycle de vie requête/slot/KV/annulation | Design doc |\n| WP2 | Admission, priorité, fairness, backpressure | Prototype |\n| WP3 | Chunked prefill global et recouvrement préfill/décodage | Prototype |\n| WP4 | Multi-slot Vulkan et keep-alive | Backend + tests |\n| WP5 | Métriques par token : TTFT, tok/s, p50/p99, cache hit, queue depth | Métriques |\n| WP6 | Exactitude sous charge : `same_answer_demo.py` | Rapport |\n| WP7 | Soak concurrent et watchdog mémoire | Tests |\n\n### Livrables\n\n- Design doc de l'ordonnanceur.\n- Prototype token-level.\n- Pool KV N × contexte intégré (avec M04).\n- Chunked prefill global.\n- Annulation et backpressure.\n- Métriques par token.\n- Benchmarks c1/c4/c16/c64.\n\n### Critères d'acceptation\n\n- Débit agrégé en hausse à c4/c16/c64.\n- p99 et TTFT mesurés et reproductibles.\n- `same_answer_demo.py` identique sous charge.\n- Aucun test d'identité cassé.\n- Aucune fuite mémoire ni cache non borné.\n\n### Protocole de mesure\n\n```sh\ntarget/release/kinetic serve MODEL.gguf --device vulkan --parallel 4\noha -n 64 -c 4 --json http://127.0.0.1:8080/v1/chat/completions ...\npython scripts/same_answer_demo.py --model MODEL.gguf \\\n  --kinetic target/release/kinetic --kinetic-args \"--device vulkan\"\npython scripts/server_abuse.py --port 8091\nrocm-smi --showuse --showtemp --showmemuse\n```\n\n### Dépendances et handoffs\n\n- Reçoit le pool KV de M04.\n- Fournit les slots et métriques à M11.\n- Coordonne avec M06 pour l'emplacement de l'ordonnanceur.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Complexité de l'ordonnanceur | Prototype incrémental, tests |\n| Famine ou priorité injuste | Fairness mesurée, quotas |\n| Rollback KV | Tests d'exactitude |\n| Vulkan mono-slot | Multi-slot avant cN |\n| Conflit avec M06 | Périmètre séparé |\n\n### Definition of done\n\n- Design doc accepté.\n- cN mesuré.\n- p99/TTFT documentés.\n- Tests d'exactitude verts.\n- Ledger à jour."
}
```

Edited docs/experts/missions.md (1 replacement)

---

## Assistant

_Thinking:_

Now append M06-M08. The last line is "- Ledger à jour." Let's use that as oldString. Ensure it's unique; there may be multiple "Ledger à jour." lines. Let's check: M02 DoD ends with "Rapport final..."; M03 ends "Rapport final..."; M04 ends "Rapport final..."; M05 ends "- Ledger à jour." Only one. Good.

**Tool: edit**

**Input:**
```json
{
  "newString": "- Ledger à jour.\n\n---\n\n## M06 — Consolidation backend : un seul backend complet par API\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0/P1 |\n| Pilote | Rust backend (Agent 3) |\n| Support | GPU AMD (Agent 1), Runtime (Agent 2), QA (Agent 5) |\n| Dépend de | M01 |\n| Débloque | M07, M13 |\n| Durée | 8–12 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nRéduire la duplication CUDA/Vulkan, faire de chaque API un backend complet\net préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.\n\n### État actuel\n\n| Aujourd'hui | Cible |\n|---|---|\n| CUDA : `kinetic-cuda` + `kinetic-models/src/cuda` | `kinetic-cuda` contient tout CUDA |\n| Vulkan : `kinetic-vulkan` + `kinetic-models/src/vulkan` | `kinetic-vulkan` contient tout Vulkan |\n| `kinetic-models` a des features `cuda`/`vulkan` | `kinetic-models` sans feature backend |\n| `kinetic-server` a `cuda.rs`, `vulkan.rs`, `cpu.rs` | factory, un seul backend chargé |\n| Ajouter HIP = crate + module dans `kinetic-models` | crate qui implémente `ModelRunner` |\n\n### WP0 — Décision à prendre avant tout code\n\nLe document `docs/experts.md` place à la fois l'ordonnanceur dans\n`kinetic-runtime` et le trait `ModelRunner` dans `kinetic-models`. Pour\néviter un cycle de dépendances, trancher :\n\n- **Option A (recommandée)** : `kinetic-runtime` définit `ModelRunner`,\n  `kinetic-models` implémente le runner CPU de référence, les backends\n  implémentent `ModelRunner`, l'ordonnanceur runtime appelle le trait. La\n  factory qui connaît les crates concrets vit dans `kinetic-server`.\n- **Option B** : `kinetic-models` définit `ModelRunner`, l'ordonnanceur vit\n  dans `kinetic-server` (ou `kinetic-scheduler`), `kinetic-runtime` se limite\n  aux devices, capacités, mémoire et pool KV.\n\nLa décision doit être écrite dans un ADR avant le déplacement des\nexécuteurs.\n\n### Périmètre\n\n**Inclus** :\n\n- `Cargo.toml`, `crates/*/Cargo.toml`.\n- `crates/kinetic-runtime/**` (à créer).\n- `crates/kinetic-models/src/{lib,model}.rs` et `src/runner/**`.\n- `crates/kinetic-cuda/**`, `crates/kinetic-vulkan/**`.\n- `crates/kinetic-server/src/{backend,lib}.rs`.\n- `docs/src/contributing.md`, `docs/src/building.md`.\n\n**Exclu** :\n\n- Modifier les noyaux ou les shaders.\n- Changer le comportement observable.\n- Introduire MLIR ou un compilateur de graphes.\n- Casser les seams `Family`/`Target`/`Backend`/`Projector`.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP0 | Trancher la direction des dépendances et l'emplacement de la factory | ADR |\n| WP1 | Créer `kinetic-runtime` : devices, capacités, mémoire, traits | Crate + tests |\n| WP2 | Extraire `ModelRunner` et le runner CPU de référence | Trait + tests |\n| WP3 | Déplacer l'exécuteur CUDA dans `kinetic-cuda/src/runner` | Déplacement + tests |\n| WP4 | Déplacer l'exécuteur Vulkan dans `kinetic-vulkan/src/runner` | Déplacement + tests |\n| WP5 | Retirer les features `cuda`/`vulkan` de `kinetic-models` | Build sans backend |\n| WP6 | Factory de backend dans `kinetic-server`/`kinetic-cli` | `auto/cpu/cuda/vulkan/hip` |\n| WP7 | Backend de test (CPU ou mock) | Tests sans GPU |\n| WP8 | Mesurer build, démarrage, overhead hôte | Rapport |\n\n### Livrables\n\n- `kinetic-runtime` avec les traits et types de base.\n- `ModelRunner` et runner CPU de référence.\n- Exécuteurs CUDA et Vulkan déplacés dans leurs crates.\n- `kinetic-models` sans feature backend.\n- Factory de backend.\n- Backend de test.\n- ADR et documentation de la règle « un backend complet par API ».\n\n### Critères d'acceptation\n\n- `kinetic-models` ne dépend plus de CUDA/Vulkan.\n- Un seul crate par API contient tout le code de cette API.\n- Les tests d'identité restent verts à chaque étape.\n- Le temps de build et le démarrage ne régressent pas.\n- Chaque `unsafe` a un commentaire `SAFETY` et un test associé.\n- Un seul backend est chargé à l'exécution.\n\n### Protocole de mesure\n\n```sh\ncargo build --release --locked --workspace --all-features\ncargo test --release --locked --workspace\ncargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings\ncargo tree -p kinetic-models\ncargo bloat --release -p kinetic-cli\nperf record -g target/release/kinetic ...\n```\n\n### Dépendances et handoffs\n\n- Reçoit la baseline de M01.\n- Coordonne avec M02/M03/M04 pour ne pas déplacer les fichiers en cours\n  d'optimisation.\n- Fournit le trait backend à M07.\n- Fournit les features stabilisées à M13.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Refactor massif | Déplacement mécanique d'abord, optimisation ensuite |\n| Conflit avec M02/M03/M04 | Branche/worktree dédié, coordination |\n| Cycle de dépendances | WP0 tranché avant code |\n| Régression de performance | Mesure build/démarrage/overhead à chaque étape |\n\n### Definition of done\n\n- ADR accepté.\n- `kinetic-models` sans backend.\n- Tests verts.\n- Build et démarrage non régressés.\n- Documentation à jour.\n\n---\n\n## M07 — Backend HIP/ROCm : squelette et noyaux de base\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P1 |\n| Pilote | GPU AMD (Agent 1) |\n| Support | Rust backend (Agent 3), QA (Agent 5) |\n| Dépend de | M06, M02 |\n| Débloque | — |\n| Durée | 8–12 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nPoser les bases d'un backend HIP complet pour `gfx1201`, avec les noyaux de\nbase et une conformance vs llama.cpp ROCm, sans dupliquer l'exécuteur.\n\n### État actuel\n\n- Pas de backend HIP ; ROCm 7.2 est installé sur la machine locale.\n- `kinetic-vulkan` fonctionne sur `gfx1201`.\n- Le décodage Vulkan est à 95 % de llama.cpp ; le préfill est le premier\n  chantier.\n- HIP est une cible de compatibilité, pas un prérequis du chemin critique.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-hip/**` (à créer).\n- Tests de conformance et de bench HIP.\n- `PLAN.md`, `ledger.jsonl`, `configs/`.\n\n**Exclu** :\n\n- Remplacer Vulkan.\n- Modifier `kinetic-models` ou les architectures.\n- Promettre une parité complète avant validation CI.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Squelette : device, mémoire, capacités, build `hipcc` | Crate qui compile |\n| WP2 | GEMV Q4_K, Q5_K, Q6_K, Q8_0 | Noyaux + tests |\n| WP3 | Attention FP16 | Noyau + tests |\n| WP4 | Conformance vs llama.cpp ROCm0 | Rapport de tokens |\n| WP5 | Bench et baseline `gfx1201` | Rapport de perf |\n| WP6 | CI/opt-in et documentation | Backend expérimental |\n\n### Livrables\n\n- `kinetic-hip` avec device, mémoire et capacités.\n- GEMV de base.\n- Attention FP16.\n- Rapport de conformance et de perf.\n- Documentation d'activation opt-in.\n\n### Critères d'acceptation\n\n- Un modèle se charge et génère sur `gfx1201`.\n- Tokens identiques au runner CPU sur le corpus de contrôle.\n- Baseline de perf consignée.\n- Backend opt-in, pas activé par défaut sans CI.\n\n### Protocole de mesure\n\n```sh\ncargo build --release --locked -p kinetic-cli --features hip\ncargo test --release --locked -p kinetic-hip\ntarget/release/kinetic bench MODEL.gguf --device hip --json -r 6\nllama-bench -m MODEL.gguf -dev ROCm0 -p 512 -n 128 -r 6\nrocm-smi --showuse --showtemp --showmemuse\n```\n\n### Dépendances et handoffs\n\n- Reçoit le trait backend de M06.\n- Reçoit les noyaux de base de M02.\n- Fournit une alternative à Vulkan pour les utilisateurs ROCm.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Support ROCm incomplet | Rester opt-in, documenter |\n| Portabilité des noyaux | Tests de conformance |\n| Pas de CI GPU | Machine locale de référence |\n| Duplication avec Vulkan | Partager les sources quand c'est possible, pas les exécuteurs |\n\n### Definition of done\n\n- Backend compile et tourne.\n- Conformance verte.\n- Baseline consignée.\n- Documentation à jour.\n\n---\n\n## M08 — Quantification et formats KV\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0/P1 |\n| Pilote | Numérique (Agent 4) |\n| Support | GPU AMD (Agent 1), Runtime (Agent 2) |\n| Dépend de | M01, M04, M02 |\n| Débloque | — |\n| Durée | 8–12 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nÉtendre la quantification et garantir que chaque gain de vitesse reste\nexact : NVFP4, FP8 E4M3, IQ encode, imatrix, KV q8_0/q4_0 Vulkan, politique\npar couche et qualité mesurée.\n\n### État actuel\n\n- 5 cibles de quantification contre ~30 dans llama.cpp.\n- Pas d'imatrix, pas d'IQ encode.\n- NVFP4 et F8_E4M3 absents du runtime.\n- KV : CUDA f16/q8_0 ; Vulkan f16 uniquement ; CPU f32.\n- `kinetic-quant` est la définition de référence ; `kdot/tests.rs` compare\n  chaque noyau.\n- L'exactitude est le différenciateur produit.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-quant/**`.\n- `crates/kinetic-gguf/**`.\n- `crates/kinetic-cpu/src/kdot/**`.\n- Tests de noyaux dans `kinetic-cuda/tests/**` et `kinetic-vulkan/tests/**`.\n- `docs/src/models.md`.\n- `PLAN.md`, `ledger.jsonl`, `configs/`.\n\n**Exclu** :\n\n- Modifier les architectures ou le runtime.\n- Accepter un noyau qui ne passe pas la référence f64.\n- Changer un format existant sans versionner.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Auditer les gates et le corpus de conformance | Inventaire |\n| WP2 | NVFP4 et FP8 E4M3 runtime | Noyaux + tests |\n| WP3 | IQ encode et imatrix | Encodeur + tests |\n| WP4 | KV q8_0 Vulkan, q4_0, politique par couche, K/V séparés | Formats + gates |\n| WP5 | Qualité : KL/perplexité, corpus de calibration | Rapport |\n| WP6 | Tests property/fuzz et conformance | Tests |\n\n### Livrables\n\n- NVFP4 et FP8 E4M3.\n- IQ encode et imatrix.\n- KV q8_0/q4_0 Vulkan.\n- Politique de quantification par tenseur/couche.\n- Rapport qualité/vitesse/mémoire.\n\n### Critères d'acceptation\n\n- Chaque nouveau format a une référence f64 et un noyau testé.\n- Aucune régression d'identité.\n- La qualité est mesurée, pas supposée.\n- Les chiffres sont reproductibles et consignés.\n- q8_0 Vulkan ne réimplémente pas q8_0 CUDA : il l'étend au backend Vulkan.\n\n### Protocole de mesure\n\n```sh\ncargo test --release --locked -p kinetic-quant\ncargo test --release --locked -p kinetic-cpu\ncargo test --release --locked -p kinetic-vulkan\ntarget/release/kinetic quantize MODEL.gguf --help\ntarget/release/kinetic perplexity MODEL.gguf\npython scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \\\n  --llama /path/to/llama.cpp/bin --device vulkan\n```\n\n### Dépendances et handoffs\n\n- Reçoit le pool KV de M04.\n- Reçoit les noyaux de M02.\n- Fournit les formats et gates à M01/M05.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Perte de qualité | KL/perplexité, gate par format |\n| Exactitude | Référence f64, tests différentiels |\n| Complexité IQ | Incrémental, corpus |\n| Conflit avec M02 | Coordination des shaders/tests |\n\n### Definition of done\n\n- Formats implémentés.\n- Gates verts.\n- Rapport qualité/vitesse.\n- Ledger à jour.",
  "oldString": "- Ledger à jour.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/missions.md"
}
```

Found 3 matches for oldString, but expected exactly one. Add more surrounding context to make oldString unique, or set replaceAll to true to replace every occurrence.

---

## Assistant

_Thinking:_

There are 3 "Ledger à jour." matches now (M05, and maybe M01? Let's see). We need to append at the end of the file. Use a more unique oldString: the last lines of M05:

```
### Definition of done

- Design doc accepté.
- cN mesuré.
- p99/TTFT documentés.
- Tests d'exactitude verts.
- Ledger à jour.
```

This should be unique. Use that as oldString and append.

**Tool: edit**

**Input:**
```json
{
  "newString": "### Definition of done\n\n- Design doc accepté.\n- cN mesuré.\n- p99/TTFT documentés.\n- Tests d'exactitude verts.\n- Ledger à jour.\n\n---\n\n## M06 — Consolidation backend : un seul backend complet par API\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0/P1 |\n| Pilote | Rust backend (Agent 3) |\n| Support | GPU AMD (Agent 1), Runtime (Agent 2), QA (Agent 5) |\n| Dépend de | M01 |\n| Débloque | M07, M13 |\n| Durée | 8–12 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nRéduire la duplication CUDA/Vulkan, faire de chaque API un backend complet\net préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.\n\n### État actuel\n\n| Aujourd'hui | Cible |\n|---|---|\n| CUDA : `kinetic-cuda` + `kinetic-models/src/cuda` | `kinetic-cuda` contient tout CUDA |\n| Vulkan : `kinetic-vulkan` + `kinetic-models/src/vulkan` | `kinetic-vulkan` contient tout Vulkan |\n| `kinetic-models` a des features `cuda`/`vulkan` | `kinetic-models` sans feature backend |\n| `kinetic-server` a `cuda.rs`, `vulkan.rs`, `cpu.rs` | factory, un seul backend chargé |\n| Ajouter HIP = crate + module dans `kinetic-models` | crate qui implémente `ModelRunner` |\n\n### WP0 — Décision à prendre avant tout code\n\nLe document `docs/experts.md` place à la fois l'ordonnanceur dans\n`kinetic-runtime` et le trait `ModelRunner` dans `kinetic-models`. Pour\néviter un cycle de dépendances, trancher :\n\n- **Option A (recommandée)** : `kinetic-runtime` définit `ModelRunner`,\n  `kinetic-models` implémente le runner CPU de référence, les backends\n  implémentent `ModelRunner`, l'ordonnanceur runtime appelle le trait. La\n  factory qui connaît les crates concrets vit dans `kinetic-server`.\n- **Option B** : `kinetic-models` définit `ModelRunner`, l'ordonnanceur vit\n  dans `kinetic-server` (ou `kinetic-scheduler`), `kinetic-runtime` se limite\n  aux devices, capacités, mémoire et pool KV.\n\nLa décision doit être écrite dans un ADR avant le déplacement des\nexécuteurs.\n\n### Périmètre\n\n**Inclus** :\n\n- `Cargo.toml`, `crates/*/Cargo.toml`.\n- `crates/kinetic-runtime/**` (à créer).\n- `crates/kinetic-models/src/{lib,model}.rs` et `src/runner/**`.\n- `crates/kinetic-cuda/**`, `crates/kinetic-vulkan/**`.\n- `crates/kinetic-server/src/{backend,lib}.rs`.\n- `docs/src/contributing.md`, `docs/src/building.md`.\n\n**Exclu** :\n\n- Modifier les noyaux ou les shaders.\n- Changer le comportement observable.\n- Introduire MLIR ou un compilateur de graphes.\n- Casser les seams `Family`/`Target`/`Backend`/`Projector`.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP0 | Trancher la direction des dépendances et l'emplacement de la factory | ADR |\n| WP1 | Créer `kinetic-runtime` : devices, capacités, mémoire, traits | Crate + tests |\n| WP2 | Extraire `ModelRunner` et le runner CPU de référence | Trait + tests |\n| WP3 | Déplacer l'exécuteur CUDA dans `kinetic-cuda/src/runner` | Déplacement + tests |\n| WP4 | Déplacer l'exécuteur Vulkan dans `kinetic-vulkan/src/runner` | Déplacement + tests |\n| WP5 | Retirer les features `cuda`/`vulkan` de `kinetic-models` | Build sans backend |\n| WP6 | Factory de backend dans `kinetic-server`/`kinetic-cli` | `auto/cpu/cuda/vulkan/hip` |\n| WP7 | Backend de test (CPU ou mock) | Tests sans GPU |\n| WP8 | Mesurer build, démarrage, overhead hôte | Rapport |\n\n### Livrables\n\n- `kinetic-runtime` avec les traits et types de base.\n- `ModelRunner` et runner CPU de référence.\n- Exécuteurs CUDA et Vulkan déplacés dans leurs crates.\n- `kinetic-models` sans feature backend.\n- Factory de backend.\n- Backend de test.\n- ADR et documentation de la règle « un backend complet par API ».\n\n### Critères d'acceptation\n\n- `kinetic-models` ne dépend plus de CUDA/Vulkan.\n- Un seul crate par API contient tout le code de cette API.\n- Les tests d'identité restent verts à chaque étape.\n- Le temps de build et le démarrage ne régressent pas.\n- Chaque `unsafe` a un commentaire `SAFETY` et un test associé.\n- Un seul backend est chargé à l'exécution.\n\n### Protocole de mesure\n\n```sh\ncargo build --release --locked --workspace --all-features\ncargo test --release --locked --workspace\ncargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings\ncargo tree -p kinetic-models\ncargo bloat --release -p kinetic-cli\nperf record -g target/release/kinetic ...\n```\n\n### Dépendances et handoffs\n\n- Reçoit la baseline de M01.\n- Coordonne avec M02/M03/M04 pour ne pas déplacer les fichiers en cours\n  d'optimisation.\n- Fournit le trait backend à M07.\n- Fournit les features stabilisées à M13.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Refactor massif | Déplacement mécanique d'abord, optimisation ensuite |\n| Conflit avec M02/M03/M04 | Branche/worktree dédié, coordination |\n| Cycle de dépendances | WP0 tranché avant code |\n| Régression de performance | Mesure build/démarrage/overhead à chaque étape |\n\n### Definition of done\n\n- ADR accepté.\n- `kinetic-models` sans backend.\n- Tests verts.\n- Build et démarrage non régressés.\n- Documentation à jour.\n\n---\n\n## M07 — Backend HIP/ROCm : squelette et noyaux de base\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P1 |\n| Pilote | GPU AMD (Agent 1) |\n| Support | Rust backend (Agent 3), QA (Agent 5) |\n| Dépend de | M06, M02 |\n| Débloque | — |\n| Durée | 8–12 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nPoser les bases d'un backend HIP complet pour `gfx1201`, avec les noyaux de\nbase et une conformance vs llama.cpp ROCm, sans dupliquer l'exécuteur.\n\n### État actuel\n\n- Pas de backend HIP ; ROCm 7.2 est installé sur la machine locale.\n- `kinetic-vulkan` fonctionne sur `gfx1201`.\n- Le décodage Vulkan est à 95 % de llama.cpp ; le préfill est le premier\n  chantier.\n- HIP est une cible de compatibilité, pas un prérequis du chemin critique.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-hip/**` (à créer).\n- Tests de conformance et de bench HIP.\n- `PLAN.md`, `ledger.jsonl`, `configs/`.\n\n**Exclu** :\n\n- Remplacer Vulkan.\n- Modifier `kinetic-models` ou les architectures.\n- Promettre une parité complète avant validation CI.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Squelette : device, mémoire, capacités, build `hipcc` | Crate qui compile |\n| WP2 | GEMV Q4_K, Q5_K, Q6_K, Q8_0 | Noyaux + tests |\n| WP3 | Attention FP16 | Noyau + tests |\n| WP4 | Conformance vs llama.cpp ROCm0 | Rapport de tokens |\n| WP5 | Bench et baseline `gfx1201` | Rapport de perf |\n| WP6 | CI/opt-in et documentation | Backend expérimental |\n\n### Livrables\n\n- `kinetic-hip` avec device, mémoire et capacités.\n- GEMV de base.\n- Attention FP16.\n- Rapport de conformance et de perf.\n- Documentation d'activation opt-in.\n\n### Critères d'acceptation\n\n- Un modèle se charge et génère sur `gfx1201`.\n- Tokens identiques au runner CPU sur le corpus de contrôle.\n- Baseline de perf consignée.\n- Backend opt-in, pas activé par défaut sans CI.\n\n### Protocole de mesure\n\n```sh\ncargo build --release --locked -p kinetic-cli --features hip\ncargo test --release --locked -p kinetic-hip\ntarget/release/kinetic bench MODEL.gguf --device hip --json -r 6\nllama-bench -m MODEL.gguf -dev ROCm0 -p 512 -n 128 -r 6\nrocm-smi --showuse --showtemp --showmemuse\n```\n\n### Dépendances et handoffs\n\n- Reçoit le trait backend de M06.\n- Reçoit les noyaux de base de M02.\n- Fournit une alternative à Vulkan pour les utilisateurs ROCm.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Support ROCm incomplet | Rester opt-in, documenter |\n| Portabilité des noyaux | Tests de conformance |\n| Pas de CI GPU | Machine locale de référence |\n| Duplication avec Vulkan | Partager les sources quand c'est possible, pas les exécuteurs |\n\n### Definition of done\n\n- Backend compile et tourne.\n- Conformance verte.\n- Baseline consignée.\n- Documentation à jour.\n\n---\n\n## M08 — Quantification et formats KV\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P0/P1 |\n| Pilote | Numérique (Agent 4) |\n| Support | GPU AMD (Agent 1), Runtime (Agent 2) |\n| Dépend de | M01, M04, M02 |\n| Débloque | — |\n| Durée | 8–12 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nÉtendre la quantification et garantir que chaque gain de vitesse reste\nexact : NVFP4, FP8 E4M3, IQ encode, imatrix, KV q8_0/q4_0 Vulkan, politique\npar couche et qualité mesurée.\n\n### État actuel\n\n- 5 cibles de quantification contre ~30 dans llama.cpp.\n- Pas d'imatrix, pas d'IQ encode.\n- NVFP4 et F8_E4M3 absents du runtime.\n- KV : CUDA f16/q8_0 ; Vulkan f16 uniquement ; CPU f32.\n- `kinetic-quant` est la définition de référence ; `kdot/tests.rs` compare\n  chaque noyau.\n- L'exactitude est le différenciateur produit.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-quant/**`.\n- `crates/kinetic-gguf/**`.\n- `crates/kinetic-cpu/src/kdot/**`.\n- Tests de noyaux dans `kinetic-cuda/tests/**` et `kinetic-vulkan/tests/**`.\n- `docs/src/models.md`.\n- `PLAN.md`, `ledger.jsonl`, `configs/`.\n\n**Exclu** :\n\n- Modifier les architectures ou le runtime.\n- Accepter un noyau qui ne passe pas la référence f64.\n- Changer un format existant sans versionner.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Auditer les gates et le corpus de conformance | Inventaire |\n| WP2 | NVFP4 et FP8 E4M3 runtime | Noyaux + tests |\n| WP3 | IQ encode et imatrix | Encodeur + tests |\n| WP4 | KV q8_0 Vulkan, q4_0, politique par couche, K/V séparés | Formats + gates |\n| WP5 | Qualité : KL/perplexité, corpus de calibration | Rapport |\n| WP6 | Tests property/fuzz et conformance | Tests |\n\n### Livrables\n\n- NVFP4 et FP8 E4M3.\n- IQ encode et imatrix.\n- KV q8_0/q4_0 Vulkan.\n- Politique de quantification par tenseur/couche.\n- Rapport qualité/vitesse/mémoire.\n\n### Critères d'acceptation\n\n- Chaque nouveau format a une référence f64 et un noyau testé.\n- Aucune régression d'identité.\n- La qualité est mesurée, pas supposée.\n- Les chiffres sont reproductibles et consignés.\n- q8_0 Vulkan ne réimplémente pas q8_0 CUDA : il l'étend au backend Vulkan.\n\n### Protocole de mesure\n\n```sh\ncargo test --release --locked -p kinetic-quant\ncargo test --release --locked -p kinetic-cpu\ncargo test --release --locked -p kinetic-vulkan\ntarget/release/kinetic quantize MODEL.gguf --help\ntarget/release/kinetic perplexity MODEL.gguf\npython scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \\\n  --llama /path/to/llama.cpp/bin --device vulkan\n```\n\n### Dépendances et handoffs\n\n- Reçoit le pool KV de M04.\n- Reçoit les noyaux de M02.\n- Fournit les formats et gates à M01/M05.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Perte de qualité | KL/perplexité, gate par format |\n| Exactitude | Référence f64, tests différentiels |\n| Complexité IQ | Incrémental, corpus |\n| Conflit avec M02 | Coordination des shaders/tests |\n\n### Definition of done\n\n- Formats implémentés.\n- Gates verts.\n- Rapport qualité/vitesse.\n- Ledger à jour.",
  "oldString": "### Definition of done\n\n- Design doc accepté.\n- cN mesuré.\n- p99/TTFT documentés.\n- Tests d'exactitude verts.\n- Ledger à jour.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/missions.md"
}
```

Edited docs/experts/missions.md (1 replacement)

---

## Assistant

_Thinking:_

Now append M09-M11. The last line is "- Ledger à jour." (M08). Need unique context: M08's DoD:

```
### Definition of done

- Formats implémentés.
- Gates verts.
- Rapport qualité/vitesse.
- Ledger à jour.
```

Use that. Append M09-M11.

**Tool: edit**

**Input:**
```json
{
  "newString": "### Definition of done\n\n- Formats implémentés.\n- Gates verts.\n- Rapport qualité/vitesse.\n- Ledger à jour.\n\n---\n\n## M09 — Couverture des architectures\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P1 |\n| Pilote | Intégration modèles (Agent 6) |\n| Support | Numérique (Agent 4), QA (Agent 5) |\n| Dépend de | M01 |\n| Débloque | M12 |\n| Durée | continu |\n| Statut | à lancer |\n\n### Objectif\n\nCouvrir les architectures manquantes par briques réutilisables, sans\nmultiplier les `match` ni dupliquer `transformer.rs`.\n\n### État actuel\n\n- 21/180 architectures.\n- Les briques rentables : MLA/DSA, hybrides SSM, T5, Gemma 2, GPT-OSS.\n- Les modèles locaux bloqués : `qwen3vl`, `deepseek`, `llama4`, `gemma2`,\n  `phi3`, `command-r`, `dbrx`.\n- Le registre `Family` et les blocs sont les bons points d'extension.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-models/src/arch/**`.\n- `crates/kinetic-models/src/blocks/**`.\n- `docs/src/models.md`.\n- `scripts/new_family.py` si nécessaire.\n\n**Exclu** :\n\n- Ajouter un `match` central.\n- Dupliquer `transformer.rs`.\n- Modifier les backends.\n- Copier du code llama.cpp.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Choisir une famille proche (phi3, gemma2, command-r) | Famille chargée |\n| WP2 | Mapper les tenseurs GGUF vers les blocs | Mapping + tests |\n| WP3 | Ajouter une ligne dans `FAMILIES` | Registre |\n| WP4 | Conformance vs llama.cpp | Rapport de tokens |\n| WP5 | Documenter `models.md` | Doc |\n| WP6 | Ajouter MLA/DSA, SSM, T5 par briques | Blocs réutilisables |\n\n### Livrables\n\n- Familles prioritaires chargées.\n- Blocs manquants (MLA, DSA, Mamba 1, RWKV, T5, iRoPE, sinks).\n- Tests de conformance.\n- Documentation à jour.\n\n### Critères d'acceptation\n\n- Le modèle se charge et génère.\n- Les tokens sont identiques à llama.cpp sur le corpus de contrôle.\n- Aucun `match` central ajouté.\n- La documentation est à jour.\n\n### Protocole de mesure\n\n```sh\ntarget/release/kinetic info MODEL.gguf\ntarget/release/kinetic run MODEL.gguf --device cpu \"Hello\"\npython scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \\\n  --llama /path/to/llama.cpp/bin --device cpu\ncargo test --release --locked -p kinetic-models\n```\n\n### Dépendances et handoffs\n\n- Reçoit la conformance de M01.\n- Fournit les architectures à M12.\n- Coordonne avec M10 pour les tokens spéciaux.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Scope creep | Prioriser les modèles locaux |\n| Variantes RoPE/norm | Tests de conformance |\n| Duplication | Un fichier par famille, blocs réutilisables |\n\n### Definition of done\n\n- Famille(s) chargée(s) et conforme(s).\n- Tests verts.\n- Doc à jour.\n\n---\n\n## M10 — Tokenizers, templates et tool calls\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P1 |\n| Pilote | Tokenizers (Agent 7) |\n| Support | Serveur (Agent 8), QA (Agent 5) |\n| Dépend de | M01 |\n| Débloque | M11 |\n| Durée | 6–10 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nAtteindre la compatibilité des tokenizers et des templates : prétokenizers,\nUGM, tokens spéciaux, templates intégrés et tool calls.\n\n### État actuel\n\n- 13/68 prétokenizers, 55 manquants.\n- UGM, RWKV, PLaMo, Tekken, Jais2, etc. manquants.\n- Tokens EOT/EOM/FIM et types spéciaux incomplets.\n- Templates intégrés et formats tool-call incomplets.\n- L'issue #5 (tool calls dans le raisonnement) est typique.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-tokenizer/**`.\n- `crates/kinetic-server/src/{template,tool_calls,tool_grammar}.rs`.\n- `docs/src/models.md`.\n\n**Exclu** :\n\n- Modifier les architectures ou les noyaux.\n- Changer le contrat `Tokenizer::from_gguf/encode/decode`.\n- Copier des regex sans NOTICE.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Corpus de contrôle et comparaison `llama-tokenize` | Corpus + rapport |\n| WP2 | 5 prétokenizers prioritaires | Prétokenizers + tests |\n| WP3 | UGM et variantes BPE | Implémentation + tests |\n| WP4 | EOT/EOM/FIM et tokens de contrôle | Tests |\n| WP5 | Templates intégrés par nom | Templates + tests |\n| WP6 | Formats tool-call manquants | Tests client |\n\n### Livrables\n\n- 15 prétokenizers.\n- UGM.\n- Tokens spéciaux.\n- Templates intégrés.\n- Formats tool-call.\n- Split GGUF si nécessaire.\n\n### Critères d'acceptation\n\n- Ids identiques sur le corpus.\n- Tokens spéciaux corrects.\n- Tool calls valides.\n- Aucune régression de template.\n\n### Protocole de mesure\n\n```sh\nllama-tokenize -m MODEL.gguf -f corpus.txt\ncargo test --release --locked -p kinetic-tokenizer\npython scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \\\n  --llama /path/to/llama.cpp/bin --device cpu\n```\n\n### Dépendances et handoffs\n\n- Reçoit le corpus de M01.\n- Fournit les templates et tool calls à M11.\n- Coordonne avec M09 pour les tokens de famille.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Regex complexes | Tests différentiels, NOTICE |\n| Tokens spéciaux | Corpus dédié |\n| Templates | Tests client réels |\n\n### Definition of done\n\n- Corpus vert.\n- Tool calls valides.\n- Doc à jour.\n\n---\n\n## M11 — Serveur, API et observabilité\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P1 |\n| Pilote | Serveur/API (Agent 8) |\n| Support | Runtime (Agent 2), Tokenizers (Agent 7), QA (Agent 5) |\n| Dépend de | M01, M05, M10 |\n| Débloque | M14 |\n| Durée | 8–12 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nAtteindre la parité API, la robustesse et l'observabilité : routes\nmanquantes, keep-alive, TLS/API key, multi-modèle, métriques et compatibilité\nclients.\n\n### État actuel\n\n- ~15/45 routes.\n- Pas de keep-alive/TLS.\n- 64 connexions thread-per-connection.\n- `docker stop` à améliorer.\n- Compatibilité clients perfectible.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-server/src/{server,http,api,llama,metrics}.rs`.\n- `crates/kinetic-server/src/ollama/`.\n- `docs/src/server.md`, `docs/src/clients.md`.\n\n**Exclu** :\n\n- Modifier les noyaux ou le runtime.\n- Casser les clients existants.\n- Ajouter une dépendance réseau non nécessaire.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Comparer routes/champs avec llama-server | Inventaire |\n| WP2 | Routes manquantes : `/props`, `/slots/:id`, `/infill`, `/apply-template`, `/lora-adapters`, `n > 1`, `timings_per_token` | Routes + tests |\n| WP3 | Keep-alive, TLS/API key, CORS | Configuration |\n| WP4 | Router multi-modèle | Design + tests |\n| WP5 | Métriques Prometheus, logs, tracing | Métriques |\n| WP6 | Arrêt propre et `docker stop` | Test |\n| WP7 | Compatibilité llama-server/OpenAI/Anthropic/Ollama | Rapport |\n| WP8 | Sécurité des entrées (limites, DoS, path traversal) | Tests + checklist |\n\n### Livrables\n\n- Routes manquantes.\n- Keep-alive/TLS/API key.\n- Router multi-modèle.\n- Métriques.\n- Arrêt propre.\n- Rapport de compatibilité.\n\n### Critères d'acceptation\n\n- Les clients existants fonctionnent sans modification.\n- Les champs inconnus sont ignorés ou refusés clairement.\n- Pas de crash sous `server_abuse.py`.\n- Métriques exploitables.\n- Pas de régression d'exactitude sous charge.\n\n### Protocole de mesure\n\n```sh\ntarget/release/kinetic serve MODEL.gguf --device vulkan --parallel 4\npython scripts/llama_server_compat.py --model MODEL.gguf \\\n  --llama-server /path/to/llama-server --kinetic target/release/kinetic\npython scripts/server_abuse.py --port 8091\npython scripts/same_answer_demo.py --model MODEL.gguf \\\n  --kinetic target/release/kinetic --kinetic-args \"--device vulkan\"\n```\n\n### Dépendances et handoffs\n\n- Reçoit les slots/métriques de M05.\n- Reçoit les templates de M10.\n- Fournit la surface à sécuriser à M14.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Compatibilité clients | Tests réels, corpus |\n| Sécurité | M14, limites d'entrée |\n| Concurrence | M05, multi-slot |\n| Scope routes | Prioriser les clients utilisés |\n\n### Definition of done\n\n- Routes prioritaires livrées.\n- Tests de compatibilité verts.\n- Métriques disponibles.\n- Doc à jour.",
  "oldString": "### Definition of done\n\n- Formats implémentés.\n- Gates verts.\n- Rapport qualité/vitesse.\n- Ledger à jour.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/missions.md"
}
```

Edited docs/experts/missions.md (1 replacement)

---

## Assistant

_Thinking:_

Now append M12-M15 and the closing mapping. Last line is "- Doc à jour." (M11). Need unique context: M11 DoD:

```
### Definition of done

- Routes prioritaires livrées.
- Tests de compatibilité verts.
- Métriques disponibles.
- Doc à jour.
```

Use that. Append M12-M15 plus a final "3. Mapping mission → agent" and "4. Règles de lancement". Let's write.

**Tool: edit**

**Input:**
```json
{
  "newString": "### Definition of done\n\n- Routes prioritaires livrées.\n- Tests de compatibilité verts.\n- Métriques disponibles.\n- Doc à jour.\n\n---\n\n## M12 — Multimodal : vision, audio, OCR\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P2 |\n| Pilote | Multimodal (Agent 9) |\n| Support | Modèles (Agent 6), GPU (Agent 1), QA (Agent 5) |\n| Dépend de | M01, M09 |\n| Débloque | — |\n| Durée | XL |\n| Statut | à lancer |\n\n### Objectif\n\nÉtendre la couverture multimodale : projecteurs vision, audio et OCR, avec\nparité des embeddings et sans régression texte.\n\n### État actuel\n\n- 2/73 projecteurs.\n- Pas d'audio/vidéo/OCR.\n- `kinetic-vision` a un contrat `Projector` propre.\n- Le multimodal est XL et hors chemin critique de performance.\n\n### Périmètre\n\n**Inclus** :\n\n- `crates/kinetic-vision/**`.\n- `crates/kinetic-audio/**` (à créer).\n- `crates/kinetic-models/src/input.rs`.\n- `docs/src/models.md`.\n\n**Exclu** :\n\n- Modifier le texte ou les backends.\n- Élargir le contrat `Projector` pour l'audio.\n- Copier du code llama.cpp.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | 3 projecteurs prioritaires | Projecteurs + tests |\n| WP2 | 10 projecteurs majeurs | Couverture |\n| WP3 | Crate `kinetic-audio` | Audio |\n| WP4 | Pipeline OCR | OCR |\n| WP5 | Parité `llama-mtmd-debug` | Rapport |\n| WP6 | Documentation | Doc |\n\n### Livrables\n\n- Projecteurs vision.\n- `kinetic-audio`.\n- Pipeline OCR.\n- Tests de parité.\n- Documentation.\n\n### Critères d'acceptation\n\n- Embeddings proches de llama.cpp.\n- Réponses cohérentes.\n- Pas de régression texte.\n\n### Protocole de mesure\n\n```sh\ncargo run --release -p kinetic-vision --example encode -- mmproj.gguf rainbow:512\nllama-mtmd-debug --image rainbow -n 512 -m mmproj.gguf\ntarget/release/kinetic serve MODEL.gguf --mmproj mmproj.gguf\n```\n\n### Dépendances et handoffs\n\n- Reçoit les architectures de M09.\n- Reçoit les noyaux GPU de M02.\n- Fournit la parité à M01.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Scope XL | Prioriser 3 puis 10 projecteurs |\n| Preprocessing | Tests de parité |\n| Audio/OCR | Traiter après la vision |\n\n### Definition of done\n\n- Projecteurs prioritaires livrés.\n- Parité mesurée.\n- Doc à jour.\n\n---\n\n## M13 — Build, release et bindings\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P2 |\n| Pilote | Build/release (Agent 10) |\n| Support | Serveur (Agent 8), Rust backend (Agent 3) |\n| Dépend de | M06 |\n| Débloque | M14 |\n| Durée | 8–12 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nRendre le produit installable et intégrable : ABI C, bindings, Docker,\npackaging, CI release, taille et temps de build.\n\n### État actuel\n\n- 10/200 fonctions C.\n- Bindings Python partiels.\n- Images Docker et packaging à finaliser.\n- CI multi-OS existante.\n\n### Périmètre\n\n**Inclus** :\n\n- `.github/workflows/**`.\n- `docker/**`, `packaging/**`.\n- `bindings/**`.\n- `crates/kinetic-capi/**`.\n- `docs/src/{building,bindings}.md`.\n\n**Exclu** :\n\n- Modifier les noyaux ou le runtime.\n- Casser l'ABI existante sans versionner.\n- Publier un artefact non testé.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Audit ABI et bindings | Inventaire |\n| WP2 | Compléter l'ABI C | ABI + tests |\n| WP3 | Bindings Python/Node/Go | Bindings + exemples |\n| WP4 | Docker, winget/scoop, archives | Artefacts |\n| WP5 | CI release | Workflow |\n| WP6 | Réduire taille et temps de build | Mesures |\n| WP7 | Documentation de build | Doc |\n\n### Livrables\n\n- ABI C complète.\n- Bindings générés.\n- Docker et installateurs.\n- CI release.\n- Documentation.\n\n### Critères d'acceptation\n\n- Installation réussie sur Windows/Linux/macOS.\n- ABI testée.\n- Build reproductible.\n- Documentation à jour.\n\n### Protocole de mesure\n\n```sh\ncargo build --release --locked --workspace --all-features\ncargo test --release --locked --workspace\ncd bindings/python && maturin build --release --locked\ndocker build -f docker/Dockerfile -t kinetic .\n```\n\n### Dépendances et handoffs\n\n- Reçoit les features stabilisées de M06.\n- Reçoit la surface serveur de M11.\n- Fournit les artefacts à M14.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Toolchains CUDA/HIP | CI dédiée, conteneurs |\n| Taille de build | LTO/PGO, features |\n| ABI instable | Semver, tests |\n\n### Definition of done\n\n- Artefacts testés.\n- CI release verte.\n- Doc à jour.\n\n---\n\n## M14 — Sécurité\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P1/P2 |\n| Pilote | Serveur/API (Agent 8) |\n| Support | Rust backend (Agent 3), Build/release (Agent 10) |\n| Dépend de | M11, M13 |\n| Débloque | — |\n| Durée | 4–6 semaines |\n| Statut | à lancer |\n\n### Objectif\n\nRéduire la surface d'attaque du produit local-first : serveur, fichiers de\nmodèle, `unsafe`, dépendances et chaîne d'approvisionnement.\n\n### État actuel\n\n- Serveur local avec 64 connexions, pas de TLS/API key.\n- Surface `unsafe` concentrée aux frontières SIMD/GPU/FFI.\n- Fichiers GGUF et mmproj lus depuis le disque.\n- Pas de threat model formel.\n- `server_abuse.py` existe mais ne couvre pas tout.\n\n### Périmètre\n\n**Inclus** :\n\n- `docs/src/security.md` (à créer).\n- Tests et scripts de sécurité.\n- Audit `unsafe` dans les crates backend.\n- Dépendances et workflows.\n\n**Exclu** :\n\n- Refonte cryptographique.\n- Promesse de sécurité cloud.\n- Modifier les noyaux sans coordination.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Threat model local-first | Document |\n| WP2 | Audit des entrées : HTTP, JSON, templates, chemins | Rapport |\n| WP3 | Limites de taille, timeouts, DoS, path traversal | Tests |\n| WP4 | Audit `unsafe` et `Send`/`Sync` | Rapport + SAFETY |\n| WP5 | Audit dépendances et supply chain | Rapport |\n| WP6 | Fuzzing des parseurs GGUF/tokenizer/template | Corpus |\n| WP7 | Checklist CI sécurité | Workflow |\n\n### Livrables\n\n- Threat model.\n- Rapport d'audit.\n- Tests d'abus.\n- Checklist CI.\n- Documentation sécurité.\n\n### Critères d'acceptation\n\n- Aucune vulnérabilité critique ouverte.\n- `server_abuse.py` ne crashe pas le serveur.\n- Chaque `unsafe` a un commentaire `SAFETY` et un test.\n- Les dépendances sont auditées.\n- Les secrets ne sont jamais loggés.\n\n### Protocole de mesure\n\n```sh\npython scripts/server_abuse.py --port 8091\ncargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings\ncargo audit\nrg -n \"unsafe\" crates/kinetic-cuda crates/kinetic-vulkan crates/kinetic-cpu\n```\n\n### Dépendances et handoffs\n\n- Reçoit la surface serveur de M11.\n- Reçoit les artefacts de M13.\n- Fournit les gates à M01.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Sur-ingénierie | Threat model local-first, prioriser |\n| Faux sentiment de sécurité | Tests, revue, audit |\n| Conflit avec les features | Checklist CI, pas de blocage |\n\n### Definition of done\n\n- Threat model accepté.\n- Audits terminés.\n- Tests verts.\n- Doc à jour.\n\n---\n\n## M15 — Documentation et expérience développeur\n\n| Champ | Valeur |\n|---|---|\n| Priorité | P2 |\n| Pilote | Build/release (Agent 10) |\n| Support | Tous |\n| Dépend de | continu |\n| Débloque | — |\n| Durée | continu |\n| Statut | à lancer |\n\n### Objectif\n\nGarder la documentation alignée avec le code, faciliter l'onboarding et\néviter la dérive entre les docs, les prompts et les missions.\n\n### État actuel\n\n- Documentation riche : `README.md`, `docs/src/`, `PLAN.md`,\n  `llama-kinetic.md`.\n- Risque de dérive entre `docs/experts.md`, les fiches, les prompts et les\n  missions.\n- Pas d'ADR pour les décisions structurantes.\n- Pas de guide de migration backend.\n\n### Périmètre\n\n**Inclus** :\n\n- `docs/**` (hors code).\n- `README.md`, `CONTRIBUTING.md`.\n- ADR.\n- Exemples et guides.\n\n**Exclu** :\n\n- Modifier le code ou les noyaux.\n- Réécrire l'historique.\n\n### Work packages\n\n| WP | Contenu | Sortie |\n|---|---|---|\n| WP1 | Index `docs/experts/README.md` | Index |\n| WP2 | ADR « un backend complet par API » | ADR |\n| WP3 | Guide de migration backend | Doc |\n| WP4 | Mettre à jour `docs/src/contributing.md` | Doc |\n| WP5 | Vérifier les liens et les commandes | CI doc |\n| WP6 | Onboarding test : une famille ou un noyau en < 1 jour | Rapport |\n\n### Livrables\n\n- Index des docs experts.\n- ADR.\n- Guide de migration.\n- Docs à jour.\n- Rapport d'onboarding.\n\n### Critères d'acceptation\n\n- `mdbook build` passe.\n- Les liens internes sont valides.\n- Les commandes documentées fonctionnent.\n- Un nouveau contributeur peut lancer un modèle et un bench en < 1 jour.\n\n### Protocole de mesure\n\n```sh\nmdbook build docs\nrg -n \"TODO|FIXME|à créer\" docs/experts\n```\n\n### Dépendances et handoffs\n\n- Reçoit les décisions de M06.\n- Reçoit les mesures de M01.\n- Met à jour les docs de toutes les missions.\n\n### Risques\n\n| Risque | Mitigation |\n|---|---|\n| Dérive | Revue à chaque mission |\n| Doc non testée | CI doc, exemples |\n| Surcharge | Prioriser les docs critiques |\n\n### Definition of done\n\n- Docs à jour.\n- Liens valides.\n- Onboarding testé.\n\n---\n\n## 3. Mapping mission → agent\n\n| Agent | Missions | Rôle principal |\n|---|---|---|\n| Agent 1 — GPU AMD | M02, M03, M07 | Noyaux Vulkan/HIP |\n| Agent 2 — Runtime | M04, M05 | KV, ordonnanceur, concurrence |\n| Agent 3 — Rust backend | M06 | Consolidation backend |\n| Agent 4 — Numérique | M08 | Quantification, KV, exactitude |\n| Agent 5 — Performance/QA | M01, support M02–M15 | Mesure, ledger, CI GPU |\n| Agent 6 — Intégration modèles | M09 | Architectures |\n| Agent 7 — Tokenizers | M10 | Tokenizers, templates, tool calls |\n| Agent 8 — Serveur/API | M11, M14 | API, observabilité, sécurité |\n| Agent 9 — Multimodal | M12 | Vision, audio, OCR |\n| Agent 10 — Build/release | M13, M14, M15 | Build, release, bindings, doc |\n\n---\n\n## 4. Règles de lancement\n\n1. **M01 d'abord.** Sans baseline, ledger et CI, les autres missions ne\n   peuvent pas prouver leurs gains.\n2. **Un agent par mission.** Ne pas mélanger GPU, runtime et modèles dans la\n   même session.\n3. **Un seul écrivain par fichier.** Si deux missions touchent le même\n   fichier, séquencer ou utiliser une branche/worktree dédié.\n4. **M06 WP0 avant le déplacement des exécuteurs.** Trancher la direction des\n   dépendances et l'emplacement de la factory.\n5. **Ne pas déplacer les exécuteurs pendant M02/M03/M04.** Le refactor\n   attend la stabilisation des noyaux, ou se fait dans une branche dédiée.\n6. **Le KV avant l'ordonnanceur.** M04 fournit le pool et les métriques dont\n   M05 a besoin.\n7. **La sécurité après l'API et le release.** M14 a besoin de M11 et M13.\n8. **La documentation en continu.** M15 ne bloque rien, mais doit suivre\n   chaque décision.\n\n---\n\n## 5. Ce qui n'est pas une mission\n\n- Un benchmark isolé sans contrôle ni ledger.\n- Une réécriture complète « pour faire propre ».\n- Un portage CUDA-only sur une machine sans NVIDIA.\n- Un backend HIP/Metal/SYCL complet sans machine ni CI pour le valider.\n- Une optimisation qui change la sortie sans gate d'identité.\n\n---\n\n## 6. Prochaine action\n\nCréer les fichiers de travail de M01 :\n\n- `THEORY.md`\n- `ledger.jsonl`\n- `configs/`\n- `best-config.json`\n\nPuis lancer M01 avant M02/M04/M06.",
  "oldString": "### Definition of done\n\n- Routes prioritaires livrées.\n- Tests de compatibilité verts.\n- Métriques disponibles.\n- Doc à jour.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/missions.md"
}
```

Edited docs/experts/missions.md (1 replacement)

---

## Assistant

_Thinking:_

Now let's verify the file structure and count. Use `grep` for mission headings and `wc -l`.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && wc -l docs/experts/missions.md && grep -n '^## M[0-9]' docs/experts/missions.md"
}
```

1738 docs/experts/missions.md
95:## M01 — Baseline, théorie et harnais de mesure
214:## M02 — Préfill Vulkan : GEMM tuilé et soumission asynchrone
321:## M03 — MoE Vulkan : GEMM groupé et pagination des experts
418:## M04 — KV cache : audit, pool unifié, pagination et partage
543:## M05 — Ordonnanceur runtime et concurrence
655:## M06 — Consolidation backend : un seul backend complet par API
788:## M07 — Backend HIP/ROCm : squelette et noyaux de base
887:## M08 — Quantification et formats KV
995:## M09 — Couverture des architectures
1093:## M10 — Tokenizers, templates et tool calls
1190:## M11 — Serveur, API et observabilité
1295:## M12 — Multimodal : vision, audio, OCR
1389:## M13 — Build, release et bindings
1487:## M14 — Sécurité
1587:## M15 — Documentation et expérience développeur


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n '^## \\|^### ' docs/experts/missions.md | head -80"
}
```

13:## 0. Comment utiliser ce document
25:### 0.1 Règles communes
39:### 0.2 Statuts
51:## 1. Index des missions
71:### 1.1 Séquencement
81:### 1.2 Chemin critique
93:## 2. Missions détaillées
95:## M01 — Baseline, théorie et harnais de mesure
107:### Objectif
113:### État actuel
124:### Périmètre
145:### Work packages
157:### Livrables
166:### Critères d'acceptation
175:### Protocole de mesure
189:### Dépendances et handoffs
195:### Risques
204:### Definition of done
214:## M02 — Préfill Vulkan : GEMM tuilé et soumission asynchrone
226:### Objectif
232:### État actuel
242:### Périmètre
257:### Work packages
268:### Livrables
276:### Critères d'acceptation
284:### Protocole de mesure
296:### Dépendances et handoffs
302:### Risques
311:### Definition of done
321:## M03 — MoE Vulkan : GEMM groupé et pagination des experts
333:### Objectif
338:### État actuel
348:### Périmètre
362:### Work packages
372:### Livrables
379:### Critères d'acceptation
386:### Protocole de mesure
395:### Dépendances et handoffs
401:### Risques
410:### Definition of done
418:## M04 — KV cache : audit, pool unifié, pagination et partage
430:### Objectif
436:### État actuel
456:### Périmètre
472:### Work packages
486:### Livrables
495:### Critères d'acceptation
504:### Protocole de mesure
515:### Dépendances et handoffs
522:### Risques
532:### Definition of done
543:## M05 — Ordonnanceur runtime et concurrence
555:### Objectif
561:### État actuel
572:### Périmètre
588:### Work packages
600:### Livrables
610:### Critères d'acceptation
618:### Protocole de mesure
629:### Dépendances et handoffs
635:### Risques
645:### Definition of done
655:## M06 — Consolidation backend : un seul backend complet par API
667:### Objectif
672:### État actuel
682:### WP0 — Décision à prendre avant tout code
699:### Périmètre
717:### Work packages
731:### Livrables
741:### Critères d'acceptation
750:### Protocole de mesure
761:### Dépendances et handoffs
769:### Risques
778:### Definition of done
788:## M07 — Backend HIP/ROCm : squelette et noyaux de base
800:### Objectif
805:### État actuel
813:### Périmètre
827:### Work packages
838:### Livrables


---

## Assistant

_Thinking:_

Good. 1738 lines, all 15 missions. Now create `docs/experts/README.md` index. Keep it concise.

**Tool: write**

**Input:**
```json
{
  "content": "# Experts et missions Kinetic\n\nIndex des documents de travail pour améliorer l'architecture et la\nperformance de Kinetic.\n\n## Ordre de lecture\n\n1. [`../experts.md`](../experts.md) — analyse, diagnostic, architecture\n   cible et profils.\n2. [`fiches-de-poste.md`](fiches-de-poste.md) — qui recruter, avec\n   compétences, périmètre, livrables et critères.\n3. [`missions.md`](missions.md) — quoi faire, dans quel ordre, avec work\n   packages, dépendances et gates.\n4. [`prompts-agents.md`](prompts-agents.md) — comment lancer un agent\n   spécialisé.\n\n## Règle de lancement\n\n- **M01 d'abord** : baseline, `THEORY.md`, ledger, CI GPU.\n- **Une mission = un agent = un périmètre d'écriture.**\n- **Un seul écrivain par fichier.** Si deux missions touchent le même\n  fichier, séquencer ou utiliser une branche/worktree dédié.\n- **M06 WP0 avant le déplacement des exécuteurs** : trancher la direction\n  des dépendances et l'emplacement de la factory.\n- **Ne pas déplacer les exécuteurs pendant M02/M03/M04.**\n- **Le KV (M04) avant l'ordonnanceur (M05).**\n- **La sécurité (M14) après l'API (M11) et le release (M13).**\n\n## Fichiers de travail à créer par M01\n\n- `THEORY.md`\n- `ledger.jsonl`\n- `configs/`\n- `best-config.json`\n\n## Ce qui n'est pas une mission\n\n- Un benchmark isolé sans contrôle ni ledger.\n- Une réécriture complète « pour faire propre ».\n- Un portage CUDA-only sur une machine sans NVIDIA.\n- Un backend HIP/Metal/SYCL complet sans machine ni CI pour le valider.\n- Une optimisation qui change la sortie sans gate d'identité.\n",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/README.md"
}
```

Created file successfully: docs/experts/README.md

---

## Assistant

_Thinking:_

Now update `docs/experts.md`:
1. Fix Metal/MoltenVK.
2. Fix KV row.
3. Add dependency-direction decision note.
4. Add missions link and mapping.

Let's make targeted edits.

First, Metal line:
```
Metal reste un chemin du backend Vulkan
(MoltenVK), pas un backend séparé.
```
Replace with native Metal text.

Second, KV row line 172:
```
| KV non quantifié | `KvCache` en f16, pas de q8_0/q4_0 | GPU AMD + quantification |
```
Replace with qualified text.

Third, add a note after §0.2 (after line 78 "comme il veut.") about dependency direction / decision point. Let's inspect the exact text around there to craft an edit.

Fourth, add missions link in §7 and a mapping table maybe after §2.11 or in §3. Let's add a short section "2.12 Mapping profils → missions" after §2.11 before `---` and `## 3`. We can use edit with the `## 3. Composition` anchor.

Let's do these edits.

**Tool: edit**

**Input:**
```json
{
  "newString": "Les autres API (HIP, SYCL, WebGPU) sont des **backends de compatibilité** :\nchacun est complet pour son API, mais n'est ajouté que lorsqu'une machine et\nune CI peuvent le valider. Metal est un chemin **natif** du crate\n`kinetic-vulkan` (feature `metal`, shaders Metal générés depuis WGSL), pas un\nbackend séparé et pas MoltenVK. L'utilisateur n'en charge qu'un seul à\nl'exécution, selon sa carte. On fournit le socle (trait, capacités, tests de",
  "oldString": "Les autres API (HIP, SYCL, WebGPU) sont des **backends de compatibilité** :\nchacun est complet pour son API, mais n'est ajouté que lorsqu'une machine et\nune CI peuvent le valider. Metal reste un chemin du backend Vulkan\n(MoltenVK), pas un backend séparé. L'utilisateur n'en charge qu'un seul à\nl'exécution, selon sa carte. On fournit le socle (trait, capacités, tests de",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Edited docs/experts.md (1 replacement)

**Tool: edit**

**Input:**
```json
{
  "newString": "| KV Vulkan non quantifié | `KvCache` Vulkan en f16 ; CUDA a déjà f16/q8_0 ; CPU en f32 ; pas de paged KV | GPU AMD + quantification |",
  "oldString": "| KV non quantifié | `KvCache` en f16, pas de q8_0/q4_0 | GPU AMD + quantification |",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Edited docs/experts.md (1 replacement)

---

## Assistant

_Thinking:_

Now add the dependency-direction note after §0.2. Let's inspect lines 60-80 to find exact text.

**Tool: read**

**Input:**
```json
{
  "limit": 25,
  "offset": 60,
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts.md, lines 60-84
60: 
61: ### 0.2 Où vivent les traits
62: 
63: Deux couches, pas une :
64: 
65: - **`kinetic-runtime`** (à créer) : découverte des devices, registre/factory
66:   de backends, capacités, budget mémoire, ordonnanceur, pool KV, métriques.
67:   Il ne connaît **pas** les types de modèle.
68: - **`kinetic-models`** : architectures, blocs, modèle générique, trait
69:   `ModelRunner` au niveau modèle, runner CPU de référence.
70: - **`kinetic-cuda` / `kinetic-vulkan` / `kinetic-hip`** : implémentent
71:   `ModelRunner` et possèdent leurs noyaux, graphes et politiques de mémoire.
72: - **`kinetic-server`** : garde le trait `Backend` actuel (requête,
73:   `reuses`, `run_ahead`, `settle_some`) au-dessus de `ModelRunner`.
74: 
75: Cette séparation évite un cycle de dépendances et évite de transformer
76: `kinetic-runtime` en compilateur de graphes. Le trait `ModelRunner` reste
77: petit : charger, préfill, décoder, état de séquence, capacités. Chaque
78: backend garde la liberté de fusionner, grapher et ordonnancer ses noyaux
79: comme il veut.
80: 
81: ### 0.3 Migration incrémentale
82: 
83: 1. Créer `kinetic-runtime` avec les types de devices, capacités, budget
84:    mémoire et factory. Aucun changement de comportement.
[Output truncated. Continue reading with offset: 85]

---

## Assistant

_Thinking:_

Add a note after line 79 about the ambiguity/decision. Let's insert:

```
**Point à trancher (M06 WP0)** : le document place à la fois l'ordonnanceur
dans `kinetic-runtime` et le trait `ModelRunner` dans `kinetic-models`. Pour
éviter un cycle, choisir :
- A) `ModelRunner` dans `kinetic-runtime`, `kinetic-models` implémente le
  runner CPU, les backends implémentent le trait ; la factory concrète vit
  dans `kinetic-server`. Recommandé.
- B) `ModelRunner` dans `kinetic-models`, ordonnanceur dans
  `kinetic-server` (ou `kinetic-scheduler`), `kinetic-runtime` limité aux
  devices/capacités/mémoire/KV.
```

This deepens the mission and flags the issue. Let's add it.

**Tool: edit**

**Input:**
```json
{
  "newString": "Cette séparation évite un cycle de dépendances et évite de transformer\n`kinetic-runtime` en compilateur de graphes. Le trait `ModelRunner` reste\npetit : charger, préfill, décoder, état de séquence, capacités. Chaque\nbackend garde la liberté de fusionner, grapher et ordonnancer ses noyaux\ncomme il veut.\n\n**Point à trancher avant M06 (WP0)** : ce document place à la fois\nl'ordonnanceur dans `kinetic-runtime` et le trait `ModelRunner` dans\n`kinetic-models`. Pour éviter un cycle de dépendances, choisir :\n\n- **Option A (recommandée)** : `kinetic-runtime` définit `ModelRunner`,\n  `kinetic-models` implémente le runner CPU de référence, les backends\n  implémentent le trait, l'ordonnanceur runtime appelle le trait. La factory\n  qui connaît les crates concrets vit dans `kinetic-server`.\n- **Option B** : `kinetic-models` définit `ModelRunner`, l'ordonnanceur vit\n  dans `kinetic-server` (ou un crate `kinetic-scheduler`), `kinetic-runtime`\n  se limite aux devices, capacités, mémoire et pool KV.\n\nLa décision doit être écrite dans un ADR avant le déplacement des exécuteurs.",
  "oldString": "Cette séparation évite un cycle de dépendances et évite de transformer\n`kinetic-runtime` en compilateur de graphes. Le trait `ModelRunner` reste\npetit : charger, préfill, décoder, état de séquence, capacités. Chaque\nbackend garde la liberté de fusionner, grapher et ordonnancer ses noyaux\ncomme il veut.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Edited docs/experts.md (1 replacement)

---

## Assistant

_Thinking:_

Now add mapping profiles → missions. We can insert before `## 3. Composition et séquencement`. Let's inspect around line 470-490.

**Tool: read**

**Input:**
```json
{
  "limit": 20,
  "offset": 475,
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts.md, lines 475-494
475: **Pourquoi ici** : 2/73 projecteurs, pas d'audio/vidéo/OCR. Impact produit
476: fort mais chantier XL, hors chemin critique de performance.
477: 
478: **Livrables** : 10 projecteurs majeurs, crate `kinetic-audio`, pipeline OCR.
479: 
480: **KPI** : parité des embeddings vs `llama-mtmd`, qualité des réponses.
481: 
482: ### 2.11 P2 — Ingénieur build/release/bindings
483: 
484: **Mission** : rendre le produit installable et intégrable.
485: 
486: **Compétences clés** : Cargo, cross-compilation, toolchains CUDA/HIP, Docker,
487: GitHub Actions, signature de code, winget/scoop, maturin/PyO3, ABI C, semver,
488: mdBook.
489: 
490: **Pourquoi ici** : 10/200 fonctions C, bindings Python partiels, images
491: Docker et packaging à finaliser.
492: 
493: **Livrables** : ABI C complète, bindings générés, images Docker, installateurs,
494: CI release.
[Output truncated. Continue reading with offset: 495]

---

## Assistant

_Thinking:_

Insert after line 495 (KPI) before `---` and `## 3`. Let's read lines 495-505.

**Tool: read**

**Input:**
```json
{
  "limit": 15,
  "offset": 495,
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts.md, lines 495-509
495: 
496: **KPI** : couverture ABI, temps/taille de build, taux d'installation réussie.
497: 
498: ---
499: 
500: ## 3. Composition et séquencement
501: 
502: ### 3.1 Cellule « performance » minimale (5 personnes)
503: 
504: | Rôle | Mission | Justification |
505: |---|---|---|
506: | GPU AMD | Vulkan d'abord, HIP ensuite | Le seul écart mesuré est le préfill/MoE |
507: | Runtime | Ordonnanceur, KV, concurrence | Le prochain plafond après le préfill |
508: | Rust backend | Abstraction backend, dette | Évite de dupliquer HIP/SYCL/WebGPU |
509: | Numérique | Quantification + gates | Protège le différenciateur exactitude |
[Output truncated. Continue reading with offset: 510]

---

## Assistant

_Thinking:_

Add a "2.12 Mapping profils → missions" section before the `---`. Use edit with the KPI line + `---` context. Let's insert.

**Tool: edit**

**Input:**
```json
{
  "newString": "**KPI** : couverture ABI, temps/taille de build, taux d'installation réussie.\n\n### 2.12 Mapping profils → missions\n\nLes missions détaillées sont dans\n[`docs/experts/missions.md`](experts/missions.md).\n\n| Profil | Missions | Rôle principal |\n|---|---|---|\n| GPU AMD | M02, M03, M07 | Noyaux Vulkan/HIP |\n| Runtime | M04, M05 | KV, ordonnanceur, concurrence |\n| Rust backend | M06 | Consolidation backend |\n| Numérique | M08 | Quantification, KV, exactitude |\n| Performance/QA | M01 + support | Mesure, ledger, CI GPU |\n| Intégration modèles | M09 | Architectures |\n| Tokenizers | M10 | Tokenizers, templates, tool calls |\n| Serveur/API | M11, M14 | API, observabilité, sécurité |\n| Multimodal | M12 | Vision, audio, OCR |\n| Build/release | M13, M14, M15 | Build, release, bindings, doc |\n\n---\n\n## 3. Composition et séquencement",
  "oldString": "**KPI** : couverture ABI, temps/taille de build, taux d'installation réussie.\n\n---\n\n## 3. Composition et séquencement",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Edited docs/experts.md (1 replacement)

---

## Assistant

_Thinking:_

Now add missions link in §7 Documents associés. Let's inspect lines around 680.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n '^## 7\\.' -A 12 docs/experts.md"
}
```

699:## 7. Documents associés
700-
701-- [`docs/experts/fiches-de-poste.md`](experts/fiches-de-poste.md) : fiches de
702-  poste détaillées pour les 10 profils.
703-- [`docs/experts/prompts-agents.md`](experts/prompts-agents.md) : prompts
704-  d'agents spécialisés, avec périmètre d'écriture, fichiers de contexte,
705-  commandes de mesure et garde-fous.
706-
707----
708-
709-## 8. Recommandation finale
710-
711-Si je devais n'embaucher que **quatre personnes** pour ce dépôt :


---

## Assistant

_Thinking:_

Add missions.md and README. Also fix the wording "10 profils" maybe keep. Let's edit.

**Tool: edit**

**Input:**
```json
{
  "newString": "## 7. Documents associés\n\n- [`docs/experts/README.md`](experts/README.md) : index et ordre de lecture.\n- [`docs/experts/fiches-de-poste.md`](experts/fiches-de-poste.md) : fiches de\n  poste détaillées pour les 10 profils.\n- [`docs/experts/missions.md`](experts/missions.md) : 15 missions\n  exécutables, avec work packages, dépendances, livrables et critères\n  d'acceptation.\n- [`docs/experts/prompts-agents.md`](experts/prompts-agents.md) : prompts\n  d'agents spécialisés, avec périmètre d'écriture, fichiers de contexte,\n  commandes de mesure et garde-fous.",
  "oldString": "## 7. Documents associés\n\n- [`docs/experts/fiches-de-poste.md`](experts/fiches-de-poste.md) : fiches de\n  poste détaillées pour les 10 profils.\n- [`docs/experts/prompts-agents.md`](experts/prompts-agents.md) : prompts\n  d'agents spécialisés, avec périmètre d'écriture, fichiers de contexte,\n  commandes de mesure et garde-fous.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Edited docs/experts.md (1 replacement)

---

## Assistant

_Thinking:_

Now update fiches-de-poste.md:
- Add mission links to each fiche header.
- Fix Fiche 4 KV line.
- Fix Fiche 1 scope (remove kinetic-models runner line or mark read-only).
- Add KV priority to Fiche 2.
- Add security/doc notes to Fiches 8/10.

Let's do the 10 mission links. Each fiche starts with `## Fiche N — ...`. We can add a line after the header. Use edits. Let's do them in parallel? The `edit` tool calls are independent but all target the same file; parallel edits could conflict because they read/write the same file. Better do sequential edits, or use one `edit` per header with unique context. We can do multiple edits in one `execute`? The `edit` tool isn't available in Code Mode; it's a direct tool. We can call multiple `edit` tools in parallel, but the file may be modified concurrently causing failures. Let's do them one at a time, or use `replaceAll` for common patterns? The headers are unique. We'll do sequential calls, maybe 2-3 at a time? The system says parallelize independent calls; but edits to the same file are not independent. Let's do sequential.

Fiche 1 header:
```
## Fiche 1 — Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)

**Priorité** : P0 — chemin critique.
```
Add mission line after header:
```
**Missions détaillées** : M02 (préfill Vulkan), M03 (MoE), M07 (HIP).
Voir [`missions.md`](missions.md).
```

Fiche 2:
```
**Missions détaillées** : M04 (KV cache), M05 (ordonnanceur).
Voir [`missions.md`](missions.md).
```

Fiche 3: M06.
Fiche 4: M08.
Fiche 5: M01 + support.
Fiche 6: M09.
Fiche 7: M10.
Fiche 8: M11, M14.
Fiche 9: M12.
Fiche 10: M13, M14, M15.

Let's do the edits. We need exact oldString including the header and the next line. Use the header plus blank line plus `**Priorité**` line. Let's inspect each header context quickly. We know from the read:
- Fiche 1: `## Fiche 1 ...` then blank then `**Priorité** : P0 — chemin critique.`
- Fiche 2: `**Priorité** : P0 — chemin critique.` same as Fiche 1. If oldString includes header, unique.
- Fiche 3: `**Priorité** : P0/P1 — fondation.`
- Fiche 4: `**Priorité** : P0/P1 — garde-fou.`
- Fiche 5: `**Priorité** : P0 — crédibilité.`
- Fiche 6: `**Priorité** : P1 — couverture.`
- Fiche 7: `**Priorité** : P1 — compatibilité.`
- Fiche 8: `**Priorité** : P1 — produit.`
- Fiche 9: `**Priorité** : P2 — expansion.`
- Fiche 10: `**Priorité** : P2 — distribution.`

Let's edit each.

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 1 — Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)\n\n**Missions détaillées** : M02 (préfill Vulkan), M03 (MoE), M07 (HIP).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P0 — chemin critique.",
  "oldString": "## Fiche 1 — Ingénieur noyaux GPU AMD (Vulkan/RDNA 4, puis HIP/ROCm)\n\n**Priorité** : P0 — chemin critique.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 2 — Architecte runtime d'inférence\n\n**Missions détaillées** : M04 (KV cache), M05 (ordonnanceur).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P0 — chemin critique.",
  "oldString": "## Fiche 2 — Architecte runtime d'inférence\n\n**Priorité** : P0 — chemin critique.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 3 — Architecte Rust systèmes/backend\n\n**Mission détaillée** : M06 (consolidation backend).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P0/P1 — fondation.",
  "oldString": "## Fiche 3 — Architecte Rust systèmes/backend\n\n**Priorité** : P0/P1 — fondation.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 4 — Ingénieur quantification et exactitude numérique\n\n**Mission détaillée** : M08 (quantification et formats KV).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P0/P1 — garde-fou.",
  "oldString": "## Fiche 4 — Ingénieur quantification et exactitude numérique\n\n**Priorité** : P0/P1 — garde-fou.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 5 — Ingénieur performance / qualification\n\n**Mission détaillée** : M01 (baseline, théorie, harnais de mesure), puis\nsupport des missions M02–M15.\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P0 — crédibilité.",
  "oldString": "## Fiche 5 — Ingénieur performance / qualification\n\n**Priorité** : P0 — crédibilité.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 6 — Ingénieur intégration des architectures\n\n**Mission détaillée** : M09 (couverture des architectures).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P1 — couverture.",
  "oldString": "## Fiche 6 — Ingénieur intégration des architectures\n\n**Priorité** : P1 — couverture.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 7 — Ingénieur tokenizers / templates / tool calls\n\n**Mission détaillée** : M10 (tokenizers, templates, tool calls).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P1 — compatibilité.",
  "oldString": "## Fiche 7 — Ingénieur tokenizers / templates / tool calls\n\n**Priorité** : P1 — compatibilité.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 8 — Ingénieur serveur / API / observabilité\n\n**Missions détaillées** : M11 (API/observabilité), M14 (sécurité, en\npilote).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P1 — produit.",
  "oldString": "## Fiche 8 — Ingénieur serveur / API / observabilité\n\n**Priorité** : P1 — produit.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 9 — Ingénieur multimodal\n\n**Mission détaillée** : M12 (vision, audio, OCR).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P2 — expansion.",
  "oldString": "## Fiche 9 — Ingénieur multimodal\n\n**Priorité** : P2 — expansion.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Fiche 10 — Ingénieur build/release/bindings\n\n**Missions détaillées** : M13 (build/release/bindings), M14 (sécurité, en\nsupport), M15 (documentation/DX).\nVoir [`missions.md`](missions.md).\n\n**Priorité** : P2 — distribution.",
  "oldString": "## Fiche 10 — Ingénieur build/release/bindings\n\n**Priorité** : P2 — distribution.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

_Thinking:_

Now fix Fiche 4 "KV f16 seulement." line. Let's locate it.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n 'KV f16 seulement\\|kinetic-models/src/runner' docs/experts/fiches-de-poste.md"
}
```

86:- `crates/kinetic-models/src/runner/**` (à créer : trait `ModelRunner` et
424:- KV f16 seulement.


---

## Assistant

_Thinking:_

Fix line 424:
`- KV f16 seulement.` → `- KV : Vulkan f16 uniquement ; CUDA f16/q8_0 ; CPU f32.`

Fix Fiche 1 scope lines 86-87. Let's read around 80-90.

**Tool: read**

**Input:**
```json
{
  "limit": 18,
  "offset": 77,
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md, lines 77-94
77: - Metal/SPIR-V croisé.
78: - Expérience de llama.cpp, ggml, vLLM ou TensorRT-LLM.
79: 
80: ### Périmètre d'écriture
81: 
82: - `crates/kinetic-vulkan/**`
83: - `crates/kinetic-hip/**` (à créer)
84: - `crates/kinetic-runtime/src/{device,capabilities,factory,memory}.rs`
85:   (propositions d'API uniquement)
86: - `crates/kinetic-models/src/runner/**` (à créer : trait `ModelRunner` et
87:   runner CPU)
88: - `PLAN.md` (entrées d'expériences), `ledger.jsonl`, `configs/`
89: 
90: ### Fichiers de contexte
91: 
92: - `PLAN.md` §4 P0.2 et P0.3.
93: - `llama-kinetic.md` §7.2, §7.3, §7.10, §7.11.
94: - `crates/kinetic-vulkan/src/{commands,pipeline,gemv,attention,moe}.rs`.
[Output truncated. Continue reading with offset: 95]

---

## Assistant

_Thinking:_

Remove the `kinetic-models/src/runner` scope line. Replace with a note under interactions or just remove. Let's remove lines 86-87. Then add a note under "Interactions" in Fiche 1. Let's inspect the interactions section.

**Tool: edit**

**Input:**
```json
{
  "newString": "- `crates/kinetic-vulkan/**`\n- `crates/kinetic-hip/**` (à créer)\n- `crates/kinetic-runtime/src/{device,capabilities,factory,memory}.rs`\n  (propositions d'API uniquement)\n- `PLAN.md` (entrées d'expériences), `ledger.jsonl`, `configs/`",
  "oldString": "- `crates/kinetic-vulkan/**`\n- `crates/kinetic-hip/**` (à créer)\n- `crates/kinetic-runtime/src/{device,capabilities,factory,memory}.rs`\n  (propositions d'API uniquement)\n- `crates/kinetic-models/src/runner/**` (à créer : trait `ModelRunner` et\n  runner CPU)\n- `PLAN.md` (entrées d'expériences), `ledger.jsonl`, `configs/`",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

**Tool: edit**

**Input:**
```json
{
  "newString": "- KV : Vulkan f16 uniquement ; CUDA f16/q8_0 ; CPU f32.",
  "oldString": "- KV f16 seulement.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

_Thinking:_

Now add a note in Fiche 1 interactions about the Rust backend owning the runner. Let's inspect the interactions section.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n '### Interactions' -A 10 docs/experts/fiches-de-poste.md | head -20"
}
```

134:### Interactions
135-
136-- **Runtime** : interfaces de graphe, soumission asynchrone, KV.
137-- **Numérique** : formats, tolérances, gates.
138-- **QA** : protocole A/B, CI GPU, ledger.
139-- **Rust backend** : migration des exécuteurs vers `kinetic-vulkan`.
140-
141-### Profil / sourcing
142-
143-HPC, ROCm, moteurs de jeu, compute Vulkan, équipes de noyaux ML. Un profil
144-CUDA-only n'est pas suffisant : il doit savoir lire du WGSL/SPIR-V et
--
261:### Interactions
262-
263-- **GPU AMD** : interfaces de graphe, KV, soumission.
264-- **Rust backend** : `kinetic-runtime`, découpage des responsabilités.
265-- **QA** : protocole de charge, p99, soak.
266-- **Serveur/API** : métriques, slots, keep-alive.
267-
268-### Profil / sourcing


---

## Assistant

_Thinking:_

Update line 139:
`- **Rust backend** : migration des exécuteurs vers kinetic-vulkan.` → `- **Rust backend** : il possède le trait ModelRunner et le runner CPU ; le GPU AMD ne fait que proposer les capacités nécessaires.`

Also add a note to Fiche 2 about KV first. Let's inspect Fiche 2 responsibilities. We can add after the mission line or under "Responsabilités". Let's add a line after the mission line:
`**Chantier prioritaire** : M04 (KV cache) avant M05 (ordonnanceur).` This is useful.

Let's edit.

**Tool: edit**

**Input:**
```json
{
  "newString": "- **Rust backend** : il possède le trait `ModelRunner` et le runner CPU ; le\n  GPU AMD ne fait que proposer les capacités nécessaires.",
  "oldString": "- **Rust backend** : migration des exécuteurs vers `kinetic-vulkan`.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

**Tool: edit**

**Input:**
```json
{
  "newString": "**Mission** : faire passer le serveur d'un modèle « N backends indépendants »\nà un ordonnanceur token-level qui tient la concurrence, le TTFT et le p99.\n\n**Chantier prioritaire** : M04 (KV cache) avant M05 (ordonnanceur). Le KV\nest le prochain plafond mémoire et le prérequis de la concurrence.",
  "oldString": "**Mission** : faire passer le serveur d'un modèle « N backends indépendants »\nà un ordonnanceur token-level qui tient la concurrence, le TTFT et le p99.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/fiches-de-poste.md"
}
```

Edited docs/experts/fiches-de-poste.md (1 replacement)

---

## Assistant

_Thinking:_

Now add security/doc notes to Fiches 8/10. Let's inspect their mission lines.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n '^## Fiche 8' -A 8 docs/experts/fiches-de-poste.md && grep -n '^## Fiche 10' -A 8 docs/experts/fiches-de-poste.md"
}
```

822:## Fiche 8 — Ingénieur serveur / API / observabilité
823-
824-**Missions détaillées** : M11 (API/observabilité), M14 (sécurité, en
825-pilote).
826-Voir [`missions.md`](missions.md).
827-
828-**Priorité** : P1 — produit.
829-**Rattachement** : équipe serveur.
830-**Mission** : parité API, robustesse et observabilité.
998:## Fiche 10 — Ingénieur build/release/bindings
999-
1000-**Missions détaillées** : M13 (build/release/bindings), M14 (sécurité, en
1001-support), M15 (documentation/DX).
1002-Voir [`missions.md`](missions.md).
1003-
1004-**Priorité** : P2 — distribution.
1005-**Rattachement** : équipe plateforme.
1006-**Mission** : rendre le produit installable et intégrable.


---

## Assistant

_Thinking:_

The mission lines already mention M14/M15. Good. No need for more.

Now update `docs/experts/prompts-agents.md`:
- Add mission links to each agent.
- Add common contract sections: bootstrap, security, documentation, coordination, DoD.
- Add M04 first phase to Agent 2.
- Add M06 decision note to Agent 3.
- Add M01 first to Agent 5.
- Add M08 note to Agent 4.
- Add M14 to Agent 8, M13/M14/M15 to Agent 10.

Let's first add common contract sections after "### Arrêt" (line 59-63) or before `---`. Let's inspect lines 55-66.

**Tool: read**

**Input:**
```json
{
  "limit": 20,
  "offset": 49,
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Read file /home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md, lines 49-68
49: ### Processus
50: 
51: - Lire les fichiers de contexte avant d'écrire.
52: - Écrire un plan court avant une modification non triviale.
53: - Laisser le dépôt dans un état compilable et testé.
54: - Mettre à jour `PLAN.md` avec : objectif, commande, résultat, verdict,
55:   raison.
56: - Produire un rapport final avec : métrique, baseline, maintenant, plafond,
57:   méthode de mesure, limites.
58: 
59: ### Arrêt
60: 
61: - S'arrêter et demander si : une modification architecturale est nécessaire,
62:   un test d'exactitude échoue sans explication, une mesure contredit le
63:   modèle, ou le périmètre d'écriture doit être élargi.
64: 
65: ---
66: 
67: ## Agent 1 — GPU AMD (Vulkan/RDNA 4, puis HIP)
68: 
[Output truncated. Continue reading with offset: 69]

---

## Assistant

_Thinking:_

Add sections after "### Arrêt" and before `---`:
- ### Bootstrap
- ### Sécurité
- ### Documentation
- ### Coordination
- ### Définition de terminé

Let's write concise content.

```
### Bootstrap

- Si `ledger.jsonl`, `configs/` ou `THEORY.md` n'existent pas, créer une
  entrée minimale au format documenté par M01 ; ne pas restructurer.
- M01 (Agent 5) crée le squelette ; les autres agents le complètent.

### Sécurité

- Ne pas exposer de secrets, de chemins privés ou de tokens dans les logs.
- Valider les entrées (taille, JSON, chemins, templates).
- Ne pas ajouter de dépendance réseau non nécessaire.
- Signaler toute surface d'attaque ou `unsafe` non justifié.

### Documentation

- Mettre à jour la doc affectée et `PLAN.md`.
- Toute nouvelle option, route ou format doit être documenté.
- M15 (Agent 10) suit les décisions ; les autres agents fournissent les
  faits.

### Coordination

- Un seul écrivain par fichier ; si un fichier est dans le périmètre d'un
  autre agent, demander avant d'écrire.
- Agent 1 ne déplace pas les exécuteurs : c'est M06/Agent 3.
- Agent 5 possède les workflows CI perf ; Agent 10 les workflows release.
- Agent 2 possède M04/M05 ; Agent 1 fournit les noyaux d'attention.
- Agent 4 possède les tests de quantification ; Agent 1 fournit les shaders.

### Définition de terminé

- Tests verts.
- Mesure baseline/candidat/contrôle.
- Entrée dans `PLAN.md` et le ledger.
- Documentation à jour.
- Rapport final avec limites et risques.
```

Let's add.

**Tool: edit**

**Input:**
```json
{
  "newString": "### Arrêt\n\n- S'arrêter et demander si : une modification architecturale est nécessaire,\n  un test d'exactitude échoue sans explication, une mesure contredit le\n  modèle, ou le périmètre d'écriture doit être élargi.\n\n### Bootstrap\n\n- Si `ledger.jsonl`, `configs/` ou `THEORY.md` n'existent pas, créer une\n  entrée minimale au format documenté par M01 ; ne pas restructurer.\n- M01 (Agent 5) crée le squelette ; les autres agents le complètent.\n\n### Sécurité\n\n- Ne pas exposer de secrets, de chemins privés ou de tokens dans les logs.\n- Valider les entrées (taille, JSON, chemins, templates).\n- Ne pas ajouter de dépendance réseau non nécessaire.\n- Signaler toute surface d'attaque ou `unsafe` non justifié.\n\n### Documentation\n\n- Mettre à jour la doc affectée et `PLAN.md`.\n- Toute nouvelle option, route ou format doit être documenté.\n- M15 (Agent 10) suit les décisions ; les autres agents fournissent les\n  faits.\n\n### Coordination\n\n- Un seul écrivain par fichier ; si un fichier est dans le périmètre d'un\n  autre agent, demander avant d'écrire.\n- Agent 1 ne déplace pas les exécuteurs : c'est M06/Agent 3.\n- Agent 5 possède les workflows CI perf ; Agent 10 les workflows release.\n- Agent 2 possède M04/M05 ; Agent 1 fournit les noyaux d'attention.\n- Agent 4 possède les tests de quantification ; Agent 1 fournit les shaders.\n\n### Définition de terminé\n\n- Tests verts.\n- Mesure baseline/candidat/contrôle.\n- Entrée dans `PLAN.md` et le ledger.\n- Documentation à jour.\n- Rapport final avec limites et risques.\n\n---",
  "oldString": "### Arrêt\n\n- S'arrêter et demander si : une modification architecturale est nécessaire,\n  un test d'exactitude échoue sans explication, une mesure contredit le\n  modèle, ou le périmètre d'écriture doit être élargi.\n\n---",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

_Thinking:_

Now add mission links to each agent header. Let's inspect headers and next lines. We can add a line after each header:
`**Mission détaillée** : M0x — ... Voir docs/experts/missions.md.`

Agent 1 header line 67 (now shifted). Let's use unique header strings and add a line. For each:

Agent 1:
```
## Agent 1 — GPU AMD (Vulkan/RDNA 4, puis HIP)

```text
```
Add before the code block:
`**Missions détaillées** : M02 (préfill), M03 (MoE), M07 (HIP). Voir ...`

Agent 2: M04/M05.
Agent 3: M06.
Agent 4: M08.
Agent 5: M01.
Agent 6: M09.
Agent 7: M10.
Agent 8: M11/M14.
Agent 9: M12.
Agent 10: M13/M14/M15.

Let's do sequential edits. The pattern for each is header + blank + ```text. We'll replace with header + blank + mission line + blank + ```text. Ensure exact. Let's check one header.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n '^## Agent' -A 3 docs/experts/prompts-agents.md"
}
```

104:## Agent 1 — GPU AMD (Vulkan/RDNA 4, puis HIP)
105-
106-```text
107-Tu es ingénieur noyaux GPU AMD pour Kinetic. Ta mission est de combler
--
167:## Agent 2 — Runtime d'inférence
168-
169-```text
170-Tu es architecte runtime d'inférence pour Kinetic. Ta mission est de faire
--
228:## Agent 3 — Rust systèmes/backend
229-
230-```text
231-Tu es architecte Rust systèmes/backend pour Kinetic. Ta mission est de
--
293:## Agent 4 — Quantification et exactitude numérique
294-
295-```text
296-Tu es ingénieur quantification et exactitude numérique pour Kinetic. Ta
--
353:## Agent 5 — Performance et qualification
354-
355-```text
356-Tu es ingénieur performance et qualification pour Kinetic. Ta mission est de
--
408:## Agent 6 — Intégration des architectures
409-
410-```text
411-Tu es ingénieur intégration des architectures pour Kinetic. Ta mission est
--
462:## Agent 7 — Tokenizers, templates et tool calls
463-
464-```text
465-Tu es ingénieur tokenizers/templates/tool calls pour Kinetic. Ta mission est
--
510:## Agent 8 — Serveur, API et observabilité
511-
512-```text
513-Tu es ingénieur serveur/API/observabilité pour Kinetic. Ta mission est la
--
560:## Agent 9 — Multimodal
561-
562-```text
563-Tu es ingénieur multimodal pour Kinetic. Ta mission est la vision, l'audio
--
607:## Agent 10 — Build, release et bindings
608-
609-```text
610-Tu es ingénieur build/release/bindings pour Kinetic. Ta mission est de


---

## Assistant

_Thinking:_

Let's edit each. Use header + blank + ```text. We need to escape backticks in the JSON string; the tool handles it. We'll use `\`\`\`text` in the string? In the tool call JSON, we can include backticks directly. Fine.

Let's do Agent 1-3 first, then 4-7, then 8-10. Sequential to avoid conflicts.

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 1 — GPU AMD (Vulkan/RDNA 4, puis HIP)\n\n**Missions détaillées** : M02 (préfill Vulkan), M03 (MoE), M07 (HIP).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 1 — GPU AMD (Vulkan/RDNA 4, puis HIP)\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 2 — Runtime d'inférence\n\n**Missions détaillées** : M04 (KV cache), M05 (ordonnanceur).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 2 — Runtime d'inférence\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 3 — Rust systèmes/backend\n\n**Mission détaillée** : M06 (consolidation backend).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 3 — Rust systèmes/backend\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 4 — Quantification et exactitude numérique\n\n**Mission détaillée** : M08 (quantification et formats KV).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 4 — Quantification et exactitude numérique\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 5 — Performance et qualification\n\n**Mission détaillée** : M01 (baseline, théorie, harnais de mesure), puis\nsupport des missions M02–M15.\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 5 — Performance et qualification\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 6 — Intégration des architectures\n\n**Mission détaillée** : M09 (couverture des architectures).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 6 — Intégration des architectures\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 7 — Tokenizers, templates et tool calls\n\n**Mission détaillée** : M10 (tokenizers, templates, tool calls).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 7 — Tokenizers, templates et tool calls\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 8 — Serveur, API et observabilité\n\n**Missions détaillées** : M11 (API/observabilité), M14 (sécurité, en\npilote).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 8 — Serveur, API et observabilité\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 9 — Multimodal\n\n**Mission détaillée** : M12 (vision, audio, OCR).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 9 — Multimodal\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

**Tool: edit**

**Input:**
```json
{
  "newString": "## Agent 10 — Build, release et bindings\n\n**Missions détaillées** : M13 (build/release/bindings), M14 (sécurité, en\nsupport), M15 (documentation/DX).\nVoir [`missions.md`](missions.md).\n\n```text",
  "oldString": "## Agent 10 — Build, release et bindings\n\n```text",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

_Thinking:_

Now deepen Agent 2 with M04 first phase. Let's inspect Agent 2's prompt around the loop.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '174,245p' docs/experts/prompts-agents.md"
}
```


```text
Tu es architecte runtime d'inférence pour Kinetic. Ta mission est de faire
passer le serveur d'un modèle « N backends indépendants » à un ordonnanceur
token-level qui tient la concurrence, le TTFT et le p99, sans casser RAD-1,
Warm Start ni Convoy.

LECTURE OBLIGATOIRE
- PLAN.md, sections 2, 3 et 4 (P1.5, P1.6).
- llama-kinetic.md, sections 7.6, 7.9, 7.10.
- docs/experts.md, section 0 (un seul backend complet par API).
- crates/kinetic-server/src/{engine,sequence,backend,server}.rs.
- crates/kinetic-models/src/{convoy,memory,per_layer}.rs.
- crates/kinetic-models/src/warm/**.
- docs/src/how-it-is-fast.md.

PÉRIMÈTRE D'ÉCRITURE
- crates/kinetic-runtime/** (à créer : scheduler, pool KV, métriques)
- crates/kinetic-server/src/{engine,sequence,backend,server}.rs
- crates/kinetic-models/src/{convoy,memory,per_layer}.rs
- crates/kinetic-models/src/warm/**
- PLAN.md, ledger.jsonl, configs/

INTERDITS
- Ne pas modifier les noyaux GPU.
- Ne pas changer les contrats Family/Target/Backend sans demande.
- Ne pas casser l'égalité batch / mono-token.
- Ne pas introduire de dépendance async lourde sans justification.

BOUCLE DE TRAVAIL
1. Cartographier le runtime actuel : slots, files, Convoy, Warm Start,
   Zero Wait, KV.
2. Mesurer c1/c4/c16/c64, TTFT, p50/p99, cache hit.
3. Identifier le prochain plafond (KV, ordonnancement, chunking, fairness).
4. Écrire un design court avec les invariants d'exactitude.
5. Implémenter un seul changement : pool KV, chunked prefill global,
   priorité, annulation, métriques.
6. Mesurer, contrôler, consigner.
7. Répéter.

COMMANDES
- target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
- oha -n 64 -c 4 --json http://127.0.0.1:8080/v1/chat/completions ...
- python scripts/same_answer_demo.py --model MODEL.gguf \
    --kinetic target/release/kinetic --kinetic-args "--device vulkan"
- python scripts/server_abuse.py --port 8091

CRITÈRES DE SUCCÈS
- Débit agrégé en hausse à c4/c16/c64.
- p99 et TTFT mesurés et reproductibles.
- same_answer_demo identique sous charge.
- Aucun test d'identité cassé, aucune fuite mémoire.

RAPPORT FINAL
Tableau : métrique | baseline | maintenant | plafond | méthode.
Puis : design retenu, invariants préservés, limites, prochains leviers.
```

---

## Agent 3 — Rust systèmes/backend

**Mission détaillée** : M06 (consolidation backend).
Voir [`missions.md`](missions.md).

```text
Tu es architecte Rust systèmes/backend pour Kinetic. Ta mission est de
réduire la duplication CUDA/Vulkan, de faire de chaque API un backend complet
et de préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.

LECTURE OBLIGATOIRE
- docs/experts.md, section 0 (un seul backend complet par API).


---

## Assistant

_Thinking:_

Add a "PREMIER CHANTIER — M04" section after "BOUCLE DE TRAVAIL" or before "COMMANDES". Let's insert after the loop (after "7. Répéter.") and before "COMMANDES":

```
PREMIER CHANTIER — M04 (KV cache)
- Auditer le KV actuel : CPU f32, CUDA f16/q8_0, Vulkan f16, un slot
  Vulkan, pas de paged KV, pas de partage de blocs, Warm Start = cache
  d'état exact.
- Instrumenter : octets alloués/utilisés, fragmentation, cache hit,
  évictions, par requête.
- Concevoir le pool unifié, puis la pagination, puis le partage de préfixe.
- Ne pas modifier les noyaux d'attention : coordonner avec Agent 1.
- Livrer un design doc + prototype + mesures avant M05.
```

This deepens the KV mission. Let's add.

**Tool: edit**

**Input:**
```json
{
  "newString": "BOUCLE DE TRAVAIL\n1. Cartographier le runtime actuel : slots, files, Convoy, Warm Start,\n   Zero Wait, KV.\n2. Mesurer c1/c4/c16/c64, TTFT, p50/p99, cache hit.\n3. Identifier le prochain plafond (KV, ordonnancement, chunking, fairness).\n4. Écrire un design court avec les invariants d'exactitude.\n5. Implémenter un seul changement : pool KV, chunked prefill global,\n   priorité, annulation, métriques.\n6. Mesurer, contrôler, consigner.\n7. Répéter.\n\nPREMIER CHANTIER — M04 (KV cache)\n- Auditer le KV actuel : CPU f32, CUDA f16/q8_0, Vulkan f16, un slot\n  Vulkan, pas de paged KV, pas de partage de blocs, Warm Start = cache\n  d'état exact.\n- Instrumenter : octets alloués/utilisés, fragmentation, cache hit,\n  évictions, par requête.\n- Concevoir le pool unifié, puis la pagination, puis le partage de préfixe.\n- Ne pas modifier les noyaux d'attention : coordonner avec Agent 1.\n- Livrer un design doc + prototype + mesures avant M05.\n\nCOMMANDES",
  "oldString": "BOUCLE DE TRAVAIL\n1. Cartographier le runtime actuel : slots, files, Convoy, Warm Start,\n   Zero Wait, KV.\n2. Mesurer c1/c4/c16/c64, TTFT, p50/p99, cache hit.\n3. Identifier le prochain plafond (KV, ordonnancement, chunking, fairness).\n4. Écrire un design court avec les invariants d'exactitude.\n5. Implémenter un seul changement : pool KV, chunked prefill global,\n   priorité, annulation, métriques.\n6. Mesurer, contrôler, consigner.\n7. Répéter.\n\nCOMMANDES",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

_Thinking:_

Now deepen Agent 3 with M06 decision note. Inspect Agent 3 loop.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '248,330p' docs/experts/prompts-agents.md"
}
```


```text
Tu es architecte Rust systèmes/backend pour Kinetic. Ta mission est de
réduire la duplication CUDA/Vulkan, de faire de chaque API un backend complet
et de préparer HIP/SYCL/WebGPU sans multiplier les exécuteurs.

LECTURE OBLIGATOIRE
- docs/experts.md, section 0 (un seul backend complet par API).
- docs/src/contributing.md, docs/src/building.md.
- crates/kinetic-models/src/{lib,model}.rs.
- crates/kinetic-models/src/{cuda,vulkan}/**.
- crates/kinetic-cuda/src/**, crates/kinetic-vulkan/src/**.
- crates/kinetic-server/src/{backend,engine}.rs.
- PLAN.md, section 3.

PÉRIMÈTRE D'ÉCRITURE
- Cargo.toml, crates/*/Cargo.toml
- crates/kinetic-runtime/** (à créer : devices, capacités, factory)
- crates/kinetic-models/src/runner/** (à créer : trait ModelRunner et runner CPU)
- crates/kinetic-models/src/{lib,model}.rs (retrait des modules backend)
- crates/kinetic-cuda/**, crates/kinetic-vulkan/** (accueil des exécuteurs)
- crates/kinetic-server/src/{backend,lib}.rs
- docs/src/contributing.md, docs/src/building.md

INTERDITS
- Ne pas modifier les noyaux ou les shaders.
- Ne pas changer le comportement observable.
- Ne pas introduire MLIR ou un compilateur de graphes.
- Ne pas casser les seams Family/Target/Backend/Projector.

BOUCLE DE TRAVAIL
1. Cartographier la duplication et les dépendances (cargo tree, wc -l).
2. Écrire le design de kinetic-runtime (devices, capacités, factory, erreurs)
   et du trait ModelRunner (charger, préfill, décoder, état).
3. Extraire ModelRunner dans kinetic-models, avec le runner CPU comme
   référence, tests verts.
4. Déplacer un exécuteur GPU derrière le trait, tests verts.
5. Mesurer build, démarrage, overhead hôte.
6. Répéter pour le second exécuteur.
7. Retirer les features backend de kinetic-models.
8. Ajouter la factory de backend et le backend de test.

COMMANDES
- cargo build --release --locked --workspace --all-features
- cargo test --release --locked --workspace
- cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
- cargo tree -p kinetic-models
- cargo bloat --release -p kinetic-cli
- perf record -g target/release/kinetic ...

CRITÈRES DE SUCCÈS
- kinetic-models ne dépend plus de CUDA/Vulkan.
- Un seul crate par API contient tout le code de cette API.
- Tests d'identité verts à chaque étape.
- Build et démarrage non régressés.
- Chaque unsafe a un commentaire SAFETY et un test.

RAPPORT FINAL
Tableau : duplication avant/après, build, démarrage, unsafe.
Puis : migration effectuée, ce qui reste, risques.
```

---

## Agent 4 — Quantification et exactitude numérique

**Mission détaillée** : M08 (quantification et formats KV).
Voir [`missions.md`](missions.md).

```text
Tu es ingénieur quantification et exactitude numérique pour Kinetic. Ta
mission est d'étendre la quantification et de garantir que chaque gain de
vitesse reste exact.

LECTURE OBLIGATOIRE
- docs/src/contributing.md, section « Adding a weight format or a kernel ».
- crates/kinetic-quant/src/{types,dequant,encode}/**.
- crates/kinetic-cpu/src/kdot/tests.rs.
- llama-kinetic.md, section 7.7.
- PLAN.md, P0.4.
- crates/kinetic-gguf/src/**.

PÉRIMÈTRE D'ÉCRITURE


---

## Assistant

_Thinking:_

Add a "DÉCISION M06 WP0" block before "BOUCLE DE TRAVAIL" or after "INTERDITS":

```
DÉCISION M06 WP0
- Trancher la direction des dépendances avant de déplacer les exécuteurs :
  A) ModelRunner dans kinetic-runtime, kinetic-models implémente le runner
  CPU, factory dans kinetic-server ; B) ModelRunner dans kinetic-models,
  ordonnanceur dans kinetic-server, kinetic-runtime limité aux devices/
  capacités/mémoire/KV.
- Écrire un ADR.
- Ne pas déplacer les exécuteurs pendant M02/M03/M04.
```

Let's add after "INTERDITS" and before "BOUCLE DE TRAVAIL".

**Tool: edit**

**Input:**
```json
{
  "newString": "INTERDITS\n- Ne pas modifier les noyaux ou les shaders.\n- Ne pas changer le comportement observable.\n- Ne pas introduire MLIR ou un compilateur de graphes.\n- Ne pas casser les seams Family/Target/Backend/Projector.\n\nDÉCISION M06 WP0\n- Trancher la direction des dépendances avant de déplacer les exécuteurs :\n  A) ModelRunner dans kinetic-runtime, kinetic-models implémente le runner\n  CPU, factory dans kinetic-server ; B) ModelRunner dans kinetic-models,\n  ordonnanceur dans kinetic-server, kinetic-runtime limité aux devices/\n  capacités/mémoire/KV.\n- Écrire un ADR.\n- Ne pas déplacer les exécuteurs pendant M02/M03/M04 : attendre la\n  stabilisation des noyaux ou travailler dans une branche dédiée.\n\nBOUCLE DE TRAVAIL",
  "oldString": "INTERDITS\n- Ne pas modifier les noyaux ou les shaders.\n- Ne pas changer le comportement observable.\n- Ne pas introduire MLIR ou un compilateur de graphes.\n- Ne pas casser les seams Family/Target/Backend/Projector.\n\nBOUCLE DE TRAVAIL",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

_Thinking:_

Now Agent 4 note: "Le KV q8_0 existe déjà côté CUDA ; l'écart est Vulkan. Ne pas le réimplémenter." Add after "INTERDITS" or before "BOUCLE". Let's inspect Agent 4 section.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '330,405p' docs/experts/prompts-agents.md"
}
```

vitesse reste exact.

LECTURE OBLIGATOIRE
- docs/src/contributing.md, section « Adding a weight format or a kernel ».
- crates/kinetic-quant/src/{types,dequant,encode}/**.
- crates/kinetic-cpu/src/kdot/tests.rs.
- llama-kinetic.md, section 7.7.
- PLAN.md, P0.4.
- crates/kinetic-gguf/src/**.

PÉRIMÈTRE D'ÉCRITURE
- crates/kinetic-quant/**
- crates/kinetic-gguf/**
- crates/kinetic-cpu/src/kdot/**
- Tests de noyaux dans crates/kinetic-cuda/tests/** et
  crates/kinetic-vulkan/tests/**
- docs/src/models.md
- PLAN.md, ledger.jsonl, configs/

INTERDITS
- Ne pas modifier les architectures ni le runtime.
- Ne pas accepter un noyau qui ne passe pas la référence f64.
- Ne pas changer un format existant sans versionner.
- Ne pas copier de tables sans NOTICE.

BOUCLE DE TRAVAIL
1. Auditer les gates existants et le corpus de conformance.
2. Choisir un format : NVFP4, FP8 E4M3, IQ encode, imatrix, KV q8_0.
3. Écrire la référence f64 et les tests différentiels.
4. Implémenter le noyau CPU, puis GPU si applicable.
5. Mesurer qualité (KL/perplexité), vitesse, mémoire.
6. Consigner et recommander.

COMMANDES
- cargo test --release --locked -p kinetic-quant
- cargo test --release --locked -p kinetic-cpu
- cargo test --release --locked -p kinetic-vulkan
- target/release/kinetic quantize MODEL.gguf --help
- target/release/kinetic perplexity MODEL.gguf
- python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
    --llama /path/to/llama.cpp/bin --device cpu

CRITÈRES DE SUCCÈS
- Chaque format a une référence f64 et un noyau testé.
- Aucune régression d'identité.
- Qualité mesurée, pas supposée.
- Chiffres reproductibles et consignés.

RAPPORT FINAL
Tableau : format | qualité | vitesse | mémoire | gate.
Puis : formats ajoutés, limites, risques.
```

---

## Agent 5 — Performance et qualification

**Mission détaillée** : M01 (baseline, théorie, harnais de mesure), puis
support des missions M02–M15.
Voir [`missions.md`](missions.md).

```text
Tu es ingénieur performance et qualification pour Kinetic. Ta mission est de
rendre la mesure automatisée, reproductible et infalsifiable.

LECTURE OBLIGATOIRE
- PLAN.md, sections 2 et 7.
- llama-kinetic.md, sections 7.10 à 7.12.
- docs/src/benchmarks.md.
- scripts/{bench_protocol,bench_vs_llamacpp,conformance,server_abuse,same_answer_demo}.py.
- .github/workflows/ci.yml.

PÉRIMÈTRE D'ÉCRITURE
- scripts/**
- .github/workflows/**
- PLAN.md, THEORY.md, ledger.jsonl, configs/, best-config.json


---

## Assistant

_Thinking:_

Add a note after "INTERDITS":
```
NOTE KV
- Le KV q8_0 existe déjà côté CUDA ; l'écart est Vulkan. Ne pas le
  réimplémenter côté CUDA.
- La politique KV par couche et les gates de qualité sont dans M04/M08.
```
Let's add.

**Tool: edit**

**Input:**
```json
{
  "newString": "INTERDITS\n- Ne pas modifier les architectures ni le runtime.\n- Ne pas accepter un noyau qui ne passe pas la référence f64.\n- Ne pas changer un format existant sans versionner.\n- Ne pas copier de tables sans NOTICE.\n\nNOTE KV\n- Le KV q8_0 existe déjà côté CUDA ; l'écart est Vulkan. Ne pas le\n  réimplémenter côté CUDA.\n- La politique KV par couche et les gates de qualité sont dans M04/M08.\n\nBOUCLE DE TRAVAIL",
  "oldString": "INTERDITS\n- Ne pas modifier les architectures ni le runtime.\n- Ne pas accepter un noyau qui ne passe pas la référence f64.\n- Ne pas changer un format existant sans versionner.\n- Ne pas copier de tables sans NOTICE.\n\nBOUCLE DE TRAVAIL",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

_Thinking:_

Now Agent 5: add "M01 est le prérequis" note. Inspect Agent 5 loop.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '405,475p' docs/experts/prompts-agents.md"
}
```

- .github/workflows/ci.yml.

PÉRIMÈTRE D'ÉCRITURE
- scripts/**
- .github/workflows/**
- PLAN.md, THEORY.md, ledger.jsonl, configs/, best-config.json
- docs/src/benchmarks.md

INTERDITS
- Ne pas modifier les noyaux ou le runtime pour « améliorer » un chiffre.
- Ne pas publier un chiffre sans commande, commit, date, état machine.
- Ne pas supprimer un résultat négatif.

BOUCLE DE TRAVAIL
1. Écrire THEORY.md : bande passante, FLOPS, KV, TTFT.
2. Créer le ledger et les configs.
3. Automatiser le protocole A/B (contrôle, candidat, contrôle).
4. Étendre conformance.py à Vulkan.
5. Ajouter un test needle 50 %/100 %.
6. Ajouter un soak concurrent et un watchdog mémoire.
7. Ajouter une CI GPU avec seuil de régression.

COMMANDES
- python scripts/bench_protocol.py MODEL.gguf --llama-bin /path/to/llama-bench
- python scripts/bench_vs_llamacpp.py MODEL.gguf --llama-dir /path/to/llama.cpp/bin
- python scripts/conformance.py MODEL.gguf --kinetic target/release/kinetic \
    --llama /path/to/llama.cpp/bin --device cpu
- python scripts/server_abuse.py -- target/release/kinetic serve MODEL.gguf -c 512 --port 8091
- python scripts/same_answer_demo.py --model MODEL.gguf --kinetic target/release/kinetic
- rocm-smi --showuse --showtemp --showmemuse

CRITÈRES DE SUCCÈS
- Un candidat à +5 % est distingué du bruit.
- Une régression est détectée automatiquement.
- Chaque chiffre publié est reproductible.
- Les résultats négatifs sont conservés.

RAPPORT FINAL
Tableau : métrique | baseline | maintenant | plafond | méthode.
Puis : protocole, gates, faux positifs/négatifs, limites.
```

---

## Agent 6 — Intégration des architectures

**Mission détaillée** : M09 (couverture des architectures).
Voir [`missions.md`](missions.md).

```text
Tu es ingénieur intégration des architectures pour Kinetic. Ta mission est
de couvrir les architectures manquantes par briques réutilisables.

LECTURE OBLIGATOIRE
- docs/src/adding-a-family.md.
- llama-kinetic.md, sections 4.1, 5.1, 6.
- crates/kinetic-models/src/arch/mod.rs (FAMILIES).
- crates/kinetic-models/src/arch/transformer.rs.
- crates/kinetic-models/src/blocks/**.
- scripts/new_family.py.

PÉRIMÈTRE D'ÉCRITURE
- crates/kinetic-models/src/arch/**
- crates/kinetic-models/src/blocks/**
- docs/src/models.md
- scripts/new_family.py (si nécessaire)

INTERDITS
- Ne pas ajouter de match central.
- Ne pas dupliquer transformer.rs.
- Ne pas modifier les backends.


---

## Assistant

_Thinking:_

Add after "INTERDITS" or before "BOUCLE":
```
PRIORITÉ M01
- Cette mission est le prérequis de M02–M15.
- Créer le squelette de ledger/configs avant les autres agents.
- Publier la baseline avant de lancer les optimisations.
```
Let's add after "INTERDITS".

**Tool: edit**

**Input:**
```json
{
  "newString": "INTERDITS\n- Ne pas modifier les noyaux ou le runtime pour « améliorer » un chiffre.\n- Ne pas publier un chiffre sans commande, commit, date, état machine.\n- Ne pas supprimer un résultat négatif.\n\nPRIORITÉ M01\n- Cette mission est le prérequis de M02–M15.\n- Créer le squelette de ledger/configs avant les autres agents.\n- Publier la baseline avant de lancer les optimisations.\n\nBOUCLE DE TRAVAIL",
  "oldString": "INTERDITS\n- Ne pas modifier les noyaux ou le runtime pour « améliorer » un chiffre.\n- Ne pas publier un chiffre sans commande, commit, date, état machine.\n- Ne pas supprimer un résultat négatif.\n\nBOUCLE DE TRAVAIL",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

_Thinking:_

Now Agent 8 and 10 security/doc notes. Inspect Agent 8.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '570,640p' docs/experts/prompts-agents.md && echo '---AGENT10---' && sed -n '660,730p' docs/experts/prompts-agents.md"
}
```

parité API, la robustesse et l'observabilité.

LECTURE OBLIGATOIRE
- llama-kinetic.md, sections 4.5, 5.9.
- PLAN.md, P0.1.
- scripts/{llama_server_compat,server_abuse}.py.
- docs/src/server.md, docs/src/clients.md.
- crates/kinetic-server/src/{server,http,api,llama,metrics}.rs,
  crates/kinetic-server/src/ollama/.

PÉRIMÈTRE D'ÉCRITURE
- crates/kinetic-server/src/{server,http,api,llama,metrics}.rs,
  crates/kinetic-server/src/ollama/
- docs/src/server.md, docs/src/clients.md

INTERDITS
- Ne pas modifier les noyaux ni le runtime.
- Ne pas casser les clients existants.
- Ne pas ajouter de dépendance réseau non nécessaire.

BOUCLE DE TRAVAIL
1. Comparer les routes et les champs avec llama-server.
2. Ajouter une route ou un champ.
3. Tester avec un client réel.
4. Tester server_abuse.
5. Documenter.

COMMANDES
- target/release/kinetic serve MODEL.gguf --device vulkan --parallel 4
- python scripts/llama_server_compat.py --model MODEL.gguf \
    --llama-server /path/to/llama-server --kinetic target/release/kinetic
- python scripts/server_abuse.py --port 8091

CRITÈRES DE SUCCÈS
- Clients existants fonctionnels.
- Champs inconnus ignorés ou refusés clairement.
- Pas de crash sous abuse.
- Métriques exploitables.

RAPPORT FINAL
Tableau : route | statut | test | limite.
Puis : ajouts, compatibilité, prochaines routes.
```

---

## Agent 9 — Multimodal

**Mission détaillée** : M12 (vision, audio, OCR).
Voir [`missions.md`](missions.md).

```text
Tu es ingénieur multimodal pour Kinetic. Ta mission est la vision, l'audio
et l'OCR.

LECTURE OBLIGATOIRE
- llama-kinetic.md, sections 4.10, 5.2.
- crates/kinetic-vision/src/{lib,ops,preprocess}.rs.
- crates/kinetic-vision/examples/encode.rs.
- docs/src/models.md.

PÉRIMÈTRE D'ÉCRITURE
- crates/kinetic-vision/**
- crates/kinetic-audio/** (à créer)
- crates/kinetic-models/src/input.rs
- docs/src/models.md

INTERDITS
- Ne pas modifier le texte ni les backends.
- Ne pas élargir le contrat Projector pour l'audio.
- Ne pas copier de code llama.cpp.
---AGENT10---
Tableau : projecteur | parité | vitesse | tests.
Puis : ajouts, limites, prochains projecteurs.
```

---

## Agent 10 — Build, release et bindings

**Missions détaillées** : M13 (build/release/bindings), M14 (sécurité, en
support), M15 (documentation/DX).
Voir [`missions.md`](missions.md).

```text
Tu es ingénieur build/release/bindings pour Kinetic. Ta mission est de
rendre le produit installable et intégrable.

LECTURE OBLIGATOIRE
- docs/src/building.md, docs/src/bindings.md.
- .github/workflows/*.yml.
- crates/kinetic-capi/**.
- docker/Dockerfile*, packaging/**.
- Cargo.toml, bindings/python/Cargo.toml.

PÉRIMÈTRE D'ÉCRITURE
- .github/workflows/**
- docker/**, packaging/**
- bindings/**
- crates/kinetic-capi/**
- docs/src/{building,bindings}.md

INTERDITS
- Ne pas modifier les noyaux ni le runtime.
- Ne pas casser l'ABI existante sans versionner.
- Ne pas publier un artefact non testé.

BOUCLE DE TRAVAIL
1. Auditer l'ABI et les bindings.
2. Construire sur les trois OS.
3. Tester les exemples C et Python.
4. Réduire la taille et le temps de build.
5. Documenter.

COMMANDES
- cargo build --release --locked --workspace --all-features
- cargo test --release --locked --workspace
- cd bindings/python && maturin build --release --locked
- docker build -f docker/Dockerfile -t kinetic .

CRITÈRES DE SUCCÈS
- Installation réussie sur Windows/Linux/macOS.
- ABI testée.
- Build reproductible.
- Documentation à jour.

RAPPORT FINAL
Tableau : artefact | build | tests | taille | limite.
Puis : ajouts, compatibilité, prochaines étapes.
```

---

## Utilisation avec OpenCode

- Chaque prompt peut être donné à un subagent `general` ou à un agent
  spécialisé.
- Donner à l'agent **un seul rôle** à la fois : ne pas mélanger GPU,
  runtime et modèles dans la même session.
- Toujours fournir le chemin du modèle et le backend cible.
- Demander un rapport au format tableau + verdict + prochaines étapes.
- Pour les modifications de code, demander un diff minimal et les tests
  exécutés.


---

## Assistant

_Thinking:_

Add to Agent 8 after INTERDITS:
```
SÉCURITÉ (M14)
- Vérifier les limites d'entrée, les chemins, les templates, les timeouts.
- Ne pas logger de secrets ni de tokens.
- Signaler les surfaces d'attaque ; M14 pilote l'audit.
```
Add to Agent 10 after INTERDITS:
```
SÉCURITÉ ET DOC (M14/M15)
- M14 : audit supply chain, artefacts signés, secrets.
- M15 : doc build/bindings, ADR, guide de migration.
- Agent 5 possède les workflows CI perf ; toi les workflows release.
```
Let's add.

**Tool: edit**

**Input:**
```json
{
  "newString": "INTERDITS\n- Ne pas modifier les noyaux ni le runtime.\n- Ne pas casser les clients existants.\n- Ne pas ajouter de dépendance réseau non nécessaire.\n\nSÉCURITÉ (M14)\n- Vérifier les limites d'entrée, les chemins, les templates, les timeouts.\n- Ne pas logger de secrets ni de tokens.\n- Signaler les surfaces d'attaque ; M14 pilote l'audit.\n\nBOUCLE DE TRAVAIL",
  "oldString": "INTERDITS\n- Ne pas modifier les noyaux ni le runtime.\n- Ne pas casser les clients existants.\n- Ne pas ajouter de dépendance réseau non nécessaire.\n\nBOUCLE DE TRAVAIL",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

**Tool: edit**

**Input:**
```json
{
  "newString": "INTERDITS\n- Ne pas modifier les noyaux ni le runtime.\n- Ne pas casser l'ABI existante sans versionner.\n- Ne pas publier un artefact non testé.\n\nSÉCURITÉ ET DOC (M14/M15)\n- M14 : audit supply chain, artefacts signés, secrets.\n- M15 : doc build/bindings, ADR, guide de migration.\n- Agent 5 possède les workflows CI perf ; toi les workflows release.\n\nBOUCLE DE TRAVAIL",
  "oldString": "INTERDITS\n- Ne pas modifier les noyaux ni le runtime.\n- Ne pas casser l'ABI existante sans versionner.\n- Ne pas publier un artefact non testé.\n\nBOUCLE DE TRAVAIL",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

_Thinking:_

Now update the "Utilisation avec OpenCode" section to mention missions and M01 first. Let's inspect lines around 730.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n '^## Utilisation avec OpenCode' -A 40 docs/experts/prompts-agents.md"
}
```

731:## Utilisation avec OpenCode
732-
733-- Chaque prompt peut être donné à un subagent `general` ou à un agent
734-  spécialisé.
735-- Donner à l'agent **un seul rôle** à la fois : ne pas mélanger GPU,
736-  runtime et modèles dans la même session.
737-- Toujours fournir le chemin du modèle et le backend cible.
738-- Demander un rapport au format tableau + verdict + prochaines étapes.
739-- Pour les modifications de code, demander un diff minimal et les tests
740-  exécutés.
741-- Pour les mesures, demander la commande exacte, la date, le commit et
742-  l'état machine.
743-
744-### Exemple de lancement
745-
746-```text
747-Lis docs/experts/prompts-agents.md, section « Agent 1 — GPU AMD ».
748-Tu es cet agent. Le modèle est MODEL.gguf, le backend est vulkan.
749-Commence par la baseline et le profil RGP, puis propose un seul levier.
750-Ne modifie rien hors de ton périmètre. Termine par le rapport demandé.
751-```
752-
753-### Exemple de garde-fou
754-
755-```text
756-Avant toute modification :
757-- vérifie que la machine est idle ;
758-- mesure la baseline ;
759-- écris ton hypothèse et le plafond calculé ;
760-- annonce le fichier exact que tu vas modifier.
761-
762-Après toute modification :
763-- lance les tests d'exactitude ;
764-- mesure le candidat puis le contrôle ;
765-- consigne dans PLAN.md ;
766-- donne le verdict avec la raison.
767-```


---

## Assistant

_Thinking:_

Add bullets:
- Lire la mission dans `docs/experts/missions.md` et donner son ID.
- M01 d'abord.
- Ne pas lancer deux agents sur les mêmes fichiers.

Update the example to include mission M02. Let's edit.

**Tool: edit**

**Input:**
```json
{
  "newString": "## Utilisation avec OpenCode\n\n- Chaque prompt peut être donné à un subagent `general` ou à un agent\n  spécialisé.\n- Toujours donner l'ID de mission (`M01`…`M15`) et lire\n  [`missions.md`](missions.md) avant de lancer l'agent.\n- **M01 d'abord** : baseline, `THEORY.md`, ledger, CI GPU.\n- Donner à l'agent **un seul rôle** à la fois : ne pas mélanger GPU,\n  runtime et modèles dans la même session.\n- Ne pas lancer deux agents sur les mêmes fichiers ; séquencer ou utiliser\n  une branche/worktree dédié.\n- Toujours fournir le chemin du modèle et le backend cible.\n- Demander un rapport au format tableau + verdict + prochaines étapes.\n- Pour les modifications de code, demander un diff minimal et les tests\n  exécutés.\n- Pour les mesures, demander la commande exacte, la date, le commit et\n  l'état machine.\n\n### Exemple de lancement\n\n```text\nLis docs/experts/missions.md, mission M02, et\ndocs/experts/prompts-agents.md, section « Agent 1 — GPU AMD ».\nTu es cet agent. Le modèle est MODEL.gguf, le backend est vulkan.\nCommence par la baseline et le profil RGP, puis propose un seul levier.\nNe modifie rien hors de ton périmètre. Termine par le rapport demandé.\n```",
  "oldString": "## Utilisation avec OpenCode\n\n- Chaque prompt peut être donné à un subagent `general` ou à un agent\n  spécialisé.\n- Donner à l'agent **un seul rôle** à la fois : ne pas mélanger GPU,\n  runtime et modèles dans la même session.\n- Toujours fournir le chemin du modèle et le backend cible.\n- Demander un rapport au format tableau + verdict + prochaines étapes.\n- Pour les modifications de code, demander un diff minimal et les tests\n  exécutés.\n- Pour les mesures, demander la commande exacte, la date, le commit et\n  l'état machine.\n\n### Exemple de lancement\n\n```text\nLis docs/experts/prompts-agents.md, section « Agent 1 — GPU AMD ».\nTu es cet agent. Le modèle est MODEL.gguf, le backend est vulkan.\nCommence par la baseline et le profil RGP, puis propose un seul levier.\nNe modifie rien hors de ton périmètre. Termine par le rapport demandé.\n```",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts/prompts-agents.md"
}
```

Edited docs/experts/prompts-agents.md (1 replacement)

---

## Assistant

_Thinking:_

Now we need to verify all edits and consistency. Use grep to check:
- `docs/experts.md` links to missions, Metal text, KV row, mapping table.
- `fiches-de-poste.md` mission lines for all 10, KV line.
- `prompts-agents.md` mission lines for all 10, M04/M06/M01/M08/M14/M15 notes.
- `missions.md` has 15 missions.
- No accidental issues (e.g., broken Markdown, duplicate "Mission détaillée" lines).

Let's run checks.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && echo '--- experts.md checks ---' && grep -n 'MoltenVK\\|Metal est\\|KV Vulkan non quantifié\\|Mapping profils\\|missions.md' docs/experts.md && echo '--- fiches mission lines ---' && grep -n 'Missions\\? détaillée' docs/experts/fiches-de-poste.md && echo '--- prompts mission lines ---' && grep -n 'Missions\\? détaillée' docs/experts/prompts-agents.md"
}
```

--- experts.md checks ---
43:une CI peuvent le valider. Metal est un chemin **natif** du crate
45:backend séparé et pas MoltenVK. L'utilisateur n'en charge qu'un seul à
187:| KV Vulkan non quantifié | `KvCache` Vulkan en f16 ; CUDA a déjà f16/q8_0 ; CPU en f32 ; pas de paged KV | GPU AMD + quantification |
498:### 2.12 Mapping profils → missions
501:[`docs/experts/missions.md`](experts/missions.md).
704:- [`docs/experts/missions.md`](experts/missions.md) : 15 missions
--- fiches mission lines ---
18:**Missions détaillées** : M02 (préfill Vulkan), M03 (MoE), M07 (HIP).
160:**Missions détaillées** : M04 (KV cache), M05 (ordonnanceur).
289:**Mission détaillée** : M06 (consolidation backend).
413:**Mission détaillée** : M08 (quantification et formats KV).
525:**Mission détaillée** : M01 (baseline, théorie, harnais de mesure), puis
633:**Mission détaillée** : M09 (couverture des architectures).
732:**Mission détaillée** : M10 (tokenizers, templates, tool calls).
824:**Missions détaillées** : M11 (API/observabilité), M14 (sécurité, en
918:**Mission détaillée** : M12 (vision, audio, OCR).
1000:**Missions détaillées** : M13 (build/release/bindings), M14 (sécurité, en
--- prompts mission lines ---
106:**Missions détaillées** : M02 (préfill Vulkan), M03 (MoE), M07 (HIP).
172:**Missions détaillées** : M04 (KV cache), M05 (ordonnanceur).
246:**Mission détaillée** : M06 (consolidation backend).
324:**Mission détaillée** : M08 (quantification et formats KV).
392:**Mission détaillée** : M01 (baseline, théorie, harnais de mesure), puis
456:**Mission détaillée** : M09 (couverture des architectures).
513:**Mission détaillée** : M10 (tokenizers, templates, tool calls).
564:**Missions détaillées** : M11 (API/observabilité), M14 (sécurité, en
623:**Mission détaillée** : M12 (vision, audio, OCR).
673:**Missions détaillées** : M13 (build/release/bindings), M14 (sécurité, en


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && echo '--- counts ---' && wc -l docs/experts.md docs/experts/README.md docs/experts/fiches-de-poste.md docs/experts/missions.md docs/experts/prompts-agents.md && echo '--- mission count ---' && grep -c '^## M[0-9][0-9]' docs/experts/missions.md && echo '--- agent count ---' && grep -c '^## Agent [0-9]' docs/experts/prompts-agents.md && echo '--- fiche count ---' && grep -c '^## Fiche [0-9]' docs/experts/fiches-de-poste.md"
}
```

--- counts ---
   736 docs/experts.md
    42 docs/experts/README.md
  1083 docs/experts/fiches-de-poste.md
  1738 docs/experts/missions.md
   773 docs/experts/prompts-agents.md
  4372 total
--- mission count ---
15
--- agent count ---
10
--- fiche count ---
10


---

## Assistant

_Thinking:_

Good. Now check for potential issues:
- `docs/experts.md` line 187 table row is long but fine.
- `docs/experts.md` line 43-45 Metal text good.
- `docs/experts/fiches-de-poste.md` line 424 KV fixed? Let's check.
- `docs/experts/prompts-agents.md` Agent 1 scope still includes `kinetic-models/src/runner/**`? The prompt's scope line 210 maybe still includes it. We need to check. The prompt says:
```
- crates/kinetic-models/src/runner/** (à créer : trait ModelRunner et runner CPU)
```
Wait, that was Agent 3, which is correct (Rust backend owns it). Agent 1's prompt scope did not include runner; it included `crates/kinetic-runtime/src/{device,capabilities,factory}.rs` proposals. Good.
- Fiche 1 scope removed runner; good.
- Agent 1 prompt scope includes `crates/kinetic-runtime/src/{device,capabilities,factory}.rs` proposals; main doc §0.5 says GPU AMD works in kinetic-vulkan/hip, not models. Proposals are okay but maybe a minor overlap. We can leave.

Check Fiche 4 KV line and Fiche 1 interactions.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n 'KV : Vulkan\\|KV f16 seulement\\|kinetic-models/src/runner' docs/experts/fiches-de-poste.md docs/experts/prompts-agents.md docs/experts.md && echo '--- Agent 1 scope ---' && sed -n '115,135p' docs/experts/prompts-agents.md && echo '--- Agent 3 decision ---' && sed -n '275,310p' docs/experts/prompts-agents.md"
}
```

docs/experts/fiches-de-poste.md:426:- KV : Vulkan f16 uniquement ; CUDA f16/q8_0 ; CPU f32.
docs/experts/prompts-agents.md:266:- crates/kinetic-models/src/runner/** (à créer : trait ModelRunner et runner CPU)
docs/experts.md:263:de pipelines ; KV f16 seulement ; `moe_gemv.wgsl` lit un expert à la fois.
docs/experts.md:365:d'IQ encode ; NVFP4 et F8_E4M3 absents du runtime ; KV f16 seulement.
--- Agent 1 scope ---
LECTURE OBLIGATOIRE
- PLAN.md, sections 2, 3 et 4 (P0.2, P0.3).
- llama-kinetic.md, sections 7.2, 7.3, 7.10, 7.11.
- docs/experts.md, section 0 (un seul backend complet par API).
- crates/kinetic-vulkan/src/{commands,pipeline,gemv,attention,moe}.rs.
- crates/kinetic-vulkan/shaders/*.wgsl.
- crates/kinetic-models/src/vulkan/* (à terme dans kinetic-vulkan).
- crates/kinetic-quant/src/{types,dequant}.rs.

PÉRIMÈTRE D'ÉCRITURE
- crates/kinetic-vulkan/**
- crates/kinetic-hip/** (à créer)
- PLAN.md, ledger.jsonl, configs/ (entrées d'expériences)
- Propositions d'API dans crates/kinetic-runtime/src/{device,capabilities,factory}.rs
  uniquement, sans implémentation.

INTERDITS
- Ne pas modifier kinetic-models/src/arch/** ni kinetic-models/src/blocks/**.
- Ne pas modifier kinetic-cuda/** sans demande.
- Ne pas changer la sémantique d'un noyau existant sans gate d'identité.
- Ne pas copier de code llama.cpp.
--- Agent 3 decision ---
- Ne pas introduire MLIR ou un compilateur de graphes.
- Ne pas casser les seams Family/Target/Backend/Projector.

DÉCISION M06 WP0
- Trancher la direction des dépendances avant de déplacer les exécuteurs :
  A) ModelRunner dans kinetic-runtime, kinetic-models implémente le runner
  CPU, factory dans kinetic-server ; B) ModelRunner dans kinetic-models,
  ordonnanceur dans kinetic-server, kinetic-runtime limité aux devices/
  capacités/mémoire/KV.
- Écrire un ADR.
- Ne pas déplacer les exécuteurs pendant M02/M03/M04 : attendre la
  stabilisation des noyaux ou travailler dans une branche dédiée.

BOUCLE DE TRAVAIL
1. Cartographier la duplication et les dépendances (cargo tree, wc -l).
2. Écrire le design de kinetic-runtime (devices, capacités, factory, erreurs)
   et du trait ModelRunner (charger, préfill, décoder, état).
3. Extraire ModelRunner dans kinetic-models, avec le runner CPU comme
   référence, tests verts.
4. Déplacer un exécuteur GPU derrière le trait, tests verts.
5. Mesurer build, démarrage, overhead hôte.
6. Répéter pour le second exécuteur.
7. Retirer les features backend de kinetic-models.
8. Ajouter la factory de backend et le backend de test.

COMMANDES
- cargo build --release --locked --workspace --all-features
- cargo test --release --locked --workspace
- cargo clippy --release --locked --workspace --all-targets --all-features -- -D warnings
- cargo tree -p kinetic-models
- cargo bloat --release -p kinetic-cli
- perf record -g target/release/kinetic ...

CRITÈRES DE SUCCÈS
- kinetic-models ne dépend plus de CUDA/Vulkan.
- Un seul crate par API contient tout le code de cette API.


---

## Assistant

_Thinking:_

There are two remaining "KV f16 seulement" in `docs/experts.md` lines 263 and 365. We need to qualify them. Let's inspect context.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && sed -n '255,270p' docs/experts.md && echo '---' && sed -n '358,370p' docs/experts.md"
}
```

- Roofline : mesurer la bande passante et les FLOPS réels, distinguer
  memory-bound et compute-bound avec des compteurs.
- Exactitude : accumulation f32, batch-invariance, égalité passe prompt /
  passe mono-token. Comprendre `Target` et RAD-1.
- Rust/ash/unsafe aux frontières des noyaux.

**Pourquoi ici** : `gemv_many.wgsl` n'est pas un GEMM tuilé ; `PROMPT_BATCH`
vaut 128 ; pas de coopmat ; `Commands::submit` attend le fence ; pas de cache
de pipelines ; KV f16 seulement ; `moe_gemv.wgsl` lit un expert à la fois.
C'est le seul écart de performance mesuré sur la RX 9070 XT.

**Livrables 90 jours** :

1. Baseline reproductible, roofline, profil RGP de pp512.
2. GEMM tuilé Q4_K/IQ, bit-exact, ≥ 1,5×.
3. Coopmat + split-K, ≥ 2× ; soumission asynchrone ; cache de pipelines.
---
  d'accumulation, fp16/bf16/fp8, arrondi stochastique.
- Encodeur `quantize`, politique par tenseur, requantification, split/merge.
- Tests différentiels, property-based, fuzzing, conformance vs llama.cpp.
- Sensibilité performance : dequant-in-registers, vectorisation, layouts
  compatibles noyaux.

**Pourquoi ici** : 5 cibles de quantification contre ~30 ; pas d'imatrix ni
d'IQ encode ; NVFP4 et F8_E4M3 absents du runtime ; KV f16 seulement.
L'exactitude est le différenciateur produit : un gain de vitesse qui change
la sortie n'est pas mergeable.

**Livrables 90 jours** :



---

## Assistant

_Thinking:_

Both are in `docs/experts.md` profile summaries:
- GPU AMD context: "KV f16 seulement" — under Vulkan context, but could be qualified.
- Quantization context: "KV f16 seulement" — global, should be qualified.

Fix both:
- GPU AMD: "KV Vulkan f16 seulement".
- Quantization: "KV : Vulkan f16 uniquement ; CUDA f16/q8_0 ; CPU f32."

Let's edit.

**Tool: edit**

**Input:**
```json
{
  "newString": "**Pourquoi ici** : `gemv_many.wgsl` n'est pas un GEMM tuilé ; `PROMPT_BATCH`\nvaut 128 ; pas de coopmat ; `Commands::submit` attend le fence ; pas de cache\nde pipelines ; KV Vulkan f16 seulement ; `moe_gemv.wgsl` lit un expert à la\nfois. C'est le seul écart de performance mesuré sur la RX 9070 XT.",
  "oldString": "**Pourquoi ici** : `gemv_many.wgsl` n'est pas un GEMM tuilé ; `PROMPT_BATCH`\nvaut 128 ; pas de coopmat ; `Commands::submit` attend le fence ; pas de cache\nde pipelines ; KV f16 seulement ; `moe_gemv.wgsl` lit un expert à la fois.\nC'est le seul écart de performance mesuré sur la RX 9070 XT.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Edited docs/experts.md (1 replacement)

**Tool: edit**

**Input:**
```json
{
  "newString": "**Pourquoi ici** : 5 cibles de quantification contre ~30 ; pas d'imatrix ni\nd'IQ encode ; NVFP4 et F8_E4M3 absents du runtime ; KV Vulkan f16 uniquement,\nCUDA f16/q8_0, CPU f32. L'exactitude est le différenciateur produit : un gain\nde vitesse qui change la sortie n'est pas mergeable.",
  "oldString": "**Pourquoi ici** : 5 cibles de quantification contre ~30 ; pas d'imatrix ni\nd'IQ encode ; NVFP4 et F8_E4M3 absents du runtime ; KV f16 seulement.\nL'exactitude est le différenciateur produit : un gain de vitesse qui change\nla sortie n'est pas mergeable.",
  "path": "/home/mrnemesys/Devs/kinetic/docs/experts.md"
}
```

Edited docs/experts.md (1 replacement)

---

## Assistant

_Thinking:_

Now check for any other "KV f16" or inconsistencies.

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/mrnemesys/Devs/kinetic && grep -n 'KV f16\\|KV non quantifié\\|MoltenVK' docs/experts.md docs/experts/fiches-de-poste.md docs/experts/prompts-agents.md docs/experts/missions.md"
}
```

docs/experts.md:45:backend séparé et pas MoltenVK. L'utilisateur n'en charge qu'un seul à

