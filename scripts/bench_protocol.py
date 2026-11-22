#!/usr/bin/env python3
"""kinetic against llama.cpp, GPU, recorded the same way every time.

Runs `kinetic bench` and `llama-bench` on the same file in the same session,
for prompt sizes 512 and 2048 and 128 generated tokens, and appends one CSV
row per model and run to `--csv` (default `bench-results.csv`). The commit
of this tree is recorded with the numbers, so a later comparison never mixes
two revisions or two GPU states.

    python scripts/bench_protocol.py MODEL [MODEL ...] --llama-bin path/to/llama-bench
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

PROMPTS = (512, 2048)
GENERATE = 128


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def kinetic(binary, model, prompt, repetitions):
    out = run(
        [
            binary,
            "bench",
            model,
            "--json",
            "--prompt",
            str(prompt),
            "-r",
            str(repetitions),
        ]
    )
    if out.returncode:
        return None
    return json.loads(out.stdout)


def llama(binary, model, prompt, generate, repetitions):
    out = run(
        [
            binary,
            "-m",
            model,
            "-p",
            str(prompt),
            "-n",
            str(generate),
            "-r",
            str(repetitions),
        ]
    )
    if out.returncode:
        return None
    row = {}
    for line in out.stdout.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 7 and cells[-2] in (f"pp{prompt}", f"tg{generate}"):
            # "4262.90 ± 33.35": the mean comes first.
            row[cells[-2]] = float(re.match(r"[0-9.]+", cells[-1]).group())
    return row.get(f"pp{prompt}"), row.get(f"tg{generate}")


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("models", nargs="+")
    args.add_argument("--kinetic", default="target/release/kinetic")
    args.add_argument("--llama-bin", default="llama-bench")
    args.add_argument("--repetitions", type=int, default=5)
    args.add_argument("--csv", default="bench-results.csv")
    args = args.parse_args()

    commit = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip() or "?"
    csv = Path(args.csv)
    fresh = not csv.exists()
    with csv.open("a") as out:
        if fresh:
            out.write("when,commit,model,source,prompt,tokens_per_second\n")
        for model in args.models:
            name = Path(model).name if Path(model).exists() else model
            weights = None
            for prompt in PROMPTS:
                results = kinetic(args.kinetic, model, prompt, args.repetitions)
                if results is None:
                    print(f"{name}: kinetic bench failed", file=sys.stderr)
                    continue
                # kinetic names a model by its folder and quantization, so
                # llama.cpp is handed the file kinetic resolved it to.
                weights = results["model"]
                pp = results["prefill_tokens_per_second"]
                now = datetime.datetime.now().isoformat(timespec="seconds")
                out.write(f"{now},{commit},{name},kinetic,pp{prompt},{pp}\n")
                print(f"{name} kinetic pp{prompt}: {pp} t/s")
            if weights is None:
                continue
            for prompt in PROMPTS:
                pp, tg = llama(args.llama_bin, weights, prompt, GENERATE, 3)
                now = datetime.datetime.now().isoformat(timespec="seconds")
                if pp is not None:
                    out.write(f"{now},{commit},{name},llama,pp{prompt},{pp}\n")
                    print(f"{name} llama   pp{prompt}: {pp} t/s")
                if tg is not None and prompt == PROMPTS[0]:
                    out.write(f"{now},{commit},{name},llama,tg{GENERATE},{tg}\n")
                    print(f"{name} llama   tg{GENERATE}: {tg} t/s")
    print(f"rows appended to {csv}")


if __name__ == "__main__":
    main()
