# Benchmarks

Speed claims are only useful if you can check them on your own machine. This
page describes how Kinetic is measured against llama.cpp, how to repeat it,
and the results on our reference machines.

## Method

- **Same file, same threads.** Both engines run the same GGUF file. On the
  CPU both get the same thread count; on the GPU both run the whole model on
  the device.
- **The llama-bench tests.** `pp512` processes a 512-token prompt and
  `tg128` generates 128 tokens. `kinetic bench` runs the same two tests as
  `llama-bench`, with speculative decoding off, so every generated token
  costs one pass over the weights on both sides. Each number is the best of
  the repetitions.
- **An editing request.** Real use often repeats text: an assistant
  rewriting code, quoting a document, filling a template. The edit test sends
  one greedy chat request that renames a variable in a short Python file and
  asks for the whole file back, to `kinetic serve` and to `llama-server` with
  llama.cpp's own speculative decoding on (`--spec-type ngram-mod`, or
  `draft-mtp` for models with a prediction block). Speed is generated tokens
  over the time from the first to the last token, measured on the client the
  same way for both. Both answers are greedy, so they should be identical;
  the script reports whether they are.
- **Exactness is checked, not assumed.** `kinetic bench` also runs its own
  edit test and fails if drafted output differs from plain decoding.
- **A quiet machine.** No other programs using the CPU or the GPU, laptops
  on mains power with the performance power plan, best of three runs.

## Reproduce it

You need a llama.cpp build with `llama-bench` and `llama-server` and a
model file. On the CPU, one script runs everything and prints a markdown
table:

```sh
python scripts/bench_vs_llamacpp.py model.gguf --llama-dir path/to/llama.cpp/bin -t 6
```

| Option | Meaning |
|---|---|
| `--llama-dir` | folder with `llama-bench` and `llama-server` |
| `--kinetic` | the `kinetic` binary [default: `kinetic` on `PATH`] |
| `-t`, `--threads` | CPU threads for both [default: half the logical CPUs] |
| `-r`, `--repetitions` | runs of each plain test [default: 3] |
| `--max-tokens` | limit for the edit answer [default: 512] |
| `--llama-spec` | `llama-server --spec-type` [default: `ngram-mod`]; use `draft-mtp` for models with an MTP block |
| `--skip-edit` | only the two plain tests |

The script hides the GPUs from llama.cpp and passes `--device cpu` to
Kinetic, starts one server at a time on 127.0.0.1 and stops it afterwards.

On the GPU, run the two tools by hand:

```sh
kinetic bench model.gguf --device cuda -r 3
llama-bench -m model.gguf -p 512 -n 128 -r 3 -ngl 99 -fa on
```

## Results

Kinetic 0.1.0 against llama.cpp build b12001, tokens per second, higher is
better. Measured on a laptop with a Ryzen 5 6600H (6 cores, DDR5) and an RTX
3060 Laptop GPU (6 GB).

| Model | Device | Test | llama.cpp | Kinetic |
|---|---|---|---:|---:|
| Qwen3.5 4B Q4_K_M | RTX 3060 Laptop | pp512 | TBD | TBD |
| Qwen3.5 4B Q4_K_M | RTX 3060 Laptop | tg128 | TBD | TBD |
| Qwen3.5 4B Q4_K_M | RTX 3060 Laptop | edit | TBD | TBD |
| Qwen3.5 4B Q4_K_M | Ryzen 5 6600H, 6 threads | pp512 | TBD | TBD |
| Qwen3.5 4B Q4_K_M | Ryzen 5 6600H, 6 threads | tg128 | TBD | TBD |
| Qwen3.5 4B Q4_K_M | Ryzen 5 6600H, 6 threads | edit | TBD | TBD |
| LFM3 1.7B Q4_K_M | RTX 3060 Laptop | pp512 | TBD | TBD |
| LFM3 1.7B Q4_K_M | RTX 3060 Laptop | tg128 | TBD | TBD |
| LFM3 1.7B Q4_K_M | RTX 3060 Laptop | edit | TBD | TBD |
| LFM3 1.7B Q4_K_M | Ryzen 5 6600H, 6 threads | pp512 | TBD | TBD |
| LFM3 1.7B Q4_K_M | Ryzen 5 6600H, 6 threads | tg128 | TBD | TBD |
| LFM3 1.7B Q4_K_M | Ryzen 5 6600H, 6 threads | edit | TBD | TBD |
| Qwen3.5 35B-A3B UD-Q2_K_XL (12.2 GB) | RTX 3060 Laptop + system RAM | tg128 | TBD | TBD |

Prompt processing on the GPU (`pp512`) is an area where llama.cpp is ahead
today; it is on the [roadmap](roadmap.md).

## The RX 9070 XT baseline (M01, 22 November 2026)

Vulkan on an AMD RX 9070 XT (`gfx1201`, 16 GB), kinetic commit `f000853`,
llama.cpp build `19bc41dbb`, one GPU job at a time, page-cache warm loads,
six repetitions, best per run; the full record (command, commit, peak VRAM
and GTT) is in `ledger.jsonl` at the repository root, and `THEORY.md` reads
these numbers against the hardware ceiling. Kinetic runs `kinetic bench
--device vulkan -r 6`, llama runs `llama-bench -dev Vulkan0 -ngl 99 -p 512
-n 128 -r 6` (35B: `-ngl 31`).

| Model | pp512 llama | pp512 kinetic | tg128 llama | tg128 kinetic |
|---|---:|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 5062 | 1361 | 135,2 | **153,3** |
| L3-8B-Stheno IQ4_XS | 3945 | 913 | 121,5 | 118,7 |
| Qwen3.5-9B Q4_K_M | 3504 | 872 | 90,7 | **97,9** |
| gemma-4-12b Q5_K_M | 2419 | 516 | 60,2 | 59,8 |
| helcyon-gpt-4o Q5_K_M | 2517 | 536 | 66,5 | 65,4 |
| Qwen3.8-27B IQ3_XXS | 1167 | 264 | **40,7** | 38,7 |
| Qwen3-Coder-30B-A3B IQ3_XXS | 3522 | 484 | **201,5** | 55,4 |
| Qwen3.8-35B-A3B Q4_K_M | 751 | 251 | **51,6** | 31,2 |

`edit_identical` was true for all eight. Dense decode sits at the memory
bandwidth wall already (93–94 % of the measured 500 GB/s reference); the gap
against llama.cpp is prompt processing (4–7×) and MoE decode (3–7×), which is
the P0.2 many-token kernel work, not a hardware limit. Four local models do
not run yet: `qwen3vl`, PaddleOCR-VL, Unlimited-OCR (architectures not
supported) and Offrail-27B (folded weights not supported on Vulkan).

## Share your numbers

Results from other hardware are welcome. Open a
[performance issue](https://github.com/Veloresearch/kinetic/issues/new?template=performance.yml)
with the script's output, or the two commands above and their output, your
CPU, GPU, RAM and driver version, and the llama.cpp build you compared with.
