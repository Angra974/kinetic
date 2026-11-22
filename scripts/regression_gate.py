#!/usr/bin/env python3
"""Regression gate: fail when a fresh bench regresses the recorded best.

Runs `kinetic bench` once (or takes its --json on stdin), compares the
prefill, decode and edit rates against `best-config.json`, and exits 1
when any of them falls more than `--threshold` (default 5 %) under the
recorded value. `--record` writes the new best instead, which is how
the file is created after a campaign.

    python scripts/regression_gate.py model.gguf --device vulkan
    python scripts/regression_gate.py --stdin < bench.json
    python scripts/regression_gate.py model.gguf --record
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BEST = "best-config.json"
METRICS = {
    "prefill": ("prefill_tokens_per_second", "pp"),
    "decode": ("decode_tokens_per_second", "tg"),
    "edit": ("edit_tokens_per_second", "edit"),
}


def load_best():
    if Path(BEST).exists():
        return json.loads(Path(BEST).read_text())
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--kinetic", default="target/release/kinetic")
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--record", action="store_true",
                    help="write the fresh numbers as the new best")
    ap.add_argument("--model-name", default=None,
                    help="key in best-config.json; defaults to the file stem")
    a = ap.parse_args()

    if a.stdin:
        res = json.load(sys.stdin)
    else:
        out = subprocess.run(
            [a.kinetic, "bench", a.model, "--device", a.device, "--json",
             "-r", str(a.repetitions)], capture_output=True, text=True)
        if out.returncode:
            sys.exit(f"bench failed: {out.stderr.strip()}")
        res = json.loads(out.stdout)

    name = a.model_name or (Path(a.model).stem if a.model else
                            Path(res.get("model", "?")).stem)
    best = load_best()
    failed = False
    if a.record:
        best[name] = {k: res[v] for k, (v, _) in METRICS.items()}
        best[name]["repetitions"] = a.repetitions
        best[name]["device"] = res.get("backend", a.device)
        Path(BEST).write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
        print(f"recorded {name}: {best[name]}")
        return

    ref = best.get(name)
    if not ref:
        print(f"no recorded best for {name}; nothing to compare")
        return
    for label, (field, _) in METRICS.items():
        if field not in res or not res.get(field):
            continue
        if label not in ref:
            continue
        drop = 1 - res[field] / ref[label]
        flag = "REGRESSION" if drop > a.threshold else "ok"
        failed |= drop > a.threshold
        print(f"{label}: {res[field]:.1f} vs best {ref[label]:.1f} "
              f"({drop:+.1%}) {flag}")
    if res.get("edit_identical") is False:
        print("edit_identical is false: output drift")
        failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
