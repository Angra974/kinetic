#!/usr/bin/env python3
"""Control → candidate → control A/B, gated, one ledger entry at the end.

Three arms run the same `kinetic bench` invocation, the candidate
differing from the controls by the flags and environment given. The
candidate is accepted only when:

- it beats **both** controls on the target metric;
- no other metric regresses more than 3 % against the weaker control;
- the two controls sit within --noise (default 5 %) of each other, else
  the run is marked inconclusive (machine not stable) and kept as such.

Each arm is one `kinetic bench -r R` call; its decode figure averages
the 128 tokens of a pass, the ≥ 6-sample population the protocol asks
for. Every value, accepted or not, goes to the ledger.

    python scripts/ab_protocol.py model.gguf \\
        --candidate-flag "-t 12" --metric tg128
"""

import argparse
import datetime
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

METRICS = {
    "pp": "prefill_tokens_per_second",
    "pp512": "prefill_tokens_per_second",
    "tg": "decode_tokens_per_second",
    "tg128": "decode_tokens_per_second",
    "edit": "edit_tokens_per_second",
}


def bench(binary, model, device, repetitions, extra_flags, env):
    cmd = [binary, "bench", model, "--device", device, "--json",
           "-r", str(repetitions)] + extra_flags
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode:
        return None, out.stderr.strip() or f"exit {out.returncode}"
    try:
        return json.loads(out.stdout), None
    except json.JSONDecodeError as e:
        return None, f"bad json: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--kinetic", default="target/release/kinetic")
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--repetitions", type=int, default=6)
    ap.add_argument("--candidate-flag", action="append", default=[],
                    help="extra bench flag for the candidate arm only, e.g. -t 12")
    ap.add_argument("--candidate-env", action="append", default=[],
                    help="VAR=VALUE set for the candidate arm only")
    ap.add_argument("--metric", choices=sorted(METRICS), default="tg128")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--regression", type=float, default=0.03)
    ap.add_argument("--ledger", default="ledger.jsonl")
    ap.add_argument("--skip-identity", action="store_true",
                    help="skip the edit_identical gate (never for a merge)")
    a = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    env = {k: v for k, v in (e.split("=", 1) for e in a.candidate_env)}

    arms = {}
    order = ["control_before", "candidate", "control_after"]
    for arm in order:
        flags = a.candidate_flag if arm == "candidate" else []
        e = env if arm == "candidate" else {}
        res, err = bench(a.kinetic, a.model, a.device, a.repetitions, flags,
                         {**e} if isinstance(e, dict) else e)
        if err:
            sys.exit(f"{arm} failed: {err}")
        arms[arm] = res
        print(f"{arm}: pp {res['prefill_tokens_per_second']:.1f} "
              f"tg {res['decode_tokens_per_second']:.1f} "
              f"edit {res['edit_tokens_per_second']:.1f} "
              f"edit_identical {res['edit_identical']}")

    metric = METRICS[a.metric]
    c1 = arms["control_before"]
    c2 = arms["control_after"]
    cand = arms["candidate"]
    controls = [c1[metric], c2[metric]]
    control_median = statistics.median(controls)
    value = cand[metric]
    spread = (max(controls) - min(controls)) / control_median

    verdict, reasons = None, []
    identity = all(x["edit_identical"] for x in arms.values())
    if not identity and not a.skip_identity:
        verdict, reasons = "rejected", ["edit_identical is false; the "
                                        "candidate changes the output"]
    elif spread > a.noise:
        verdict, reasons = "inconclusive", [
            f"controls drift {spread:.1%} apart (> {a.noise:.0%}); "
            "machine not stable"]
    else:
        beats = value > control_median * (1 + a.noise)
        worse = {
            name: min(c1[name], c2[name]) / cand[name] - 1
            for name in ("prefill_tokens_per_second",
                         "decode_tokens_per_second",
                         "edit_tokens_per_second")
            if name != metric and cand[name]
        }
        regressions = {k: v for k, v in worse.items() if v > a.regression}
        if not beats:
            reasons.append(f"candidate {value:.1f} does not beat both "
                           f"controls ({c1[metric]:.1f}, {c2[metric]:.1f}) "
                           f"on {a.metric}")
        for k, v in regressions.items():
            reasons.append(f"{k} regresses {v:.1%} (> {a.regression:.0%})")
        if beats and not regressions:
            verdict = "accepted"
            reasons = [f"candidate {value:.1f} beats both controls on "
                       f"{a.metric}, no other metric regresses"]
        else:
            verdict = verdict or "rejected"

    if verdict is None:
        verdict, reasons = "rejected", reasons or ["gates not met"]

    print(f"verdict: {verdict}")
    for r in reasons:
        print(f"  - {r}")

    entry = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "kind": "ab_test",
        "model": Path(a.model).stem,
        "file": a.model,
        "commit": commit,
        "tree_clean": subprocess.run(["git", "status", "--porcelain"],
                                     capture_output=True,
                                     text=True).stdout == "",
        "command": shlex.join([a.kinetic, "bench", a.model, "--device",
                               a.device, "--json", "-r",
                               str(a.repetitions)]) + " | candidate: " +
                   " ".join(a.candidate_flag) + " env:" + " ".join(a.candidate_env),
        "repetitions": a.repetitions,
        "metric": a.metric,
        "controls": {k: arms[k] for k in order},
        "verdict": verdict,
        "reasons": reasons,
    }
    with open(a.ledger, "a") as f:
        f.write(json.dumps(entry) + "\n")
    sys.exit(0 if verdict == "accepted" else 2)


if __name__ == "__main__":
    main()
