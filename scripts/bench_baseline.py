#!/usr/bin/env python3
"""M01 baseline sweep: one ledger entry per model, one GPU job at a time.

For every model given, runs `kinetic bench --device vulkan --json -r 6`,
samples VRAM and GTT use while it runs, then `llama-bench -dev Vulkan0
-p 512 -n 128 -r 6` for the comparison row, and appends one JSONL entry
per side to the ledger. Kinetic's reported numbers are best-of-r over
the repetitions; its decode rate itself averages the 128 tokens of a
pass, which is the ≥ 6-sample population the M01 protocol asks for.

Failures are recorded as negative entries, never dropped.
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

KINETIC = "target/release/kinetic"
LLAMA_BIN = "/home/mrnemesys/Devs/llama.cpp/build-vulkan/bin/llama-bench"


def run(cmd, timeout=3600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class MemorySampler:
    """Polls rocm-smi for peak VRAM and GTT use in a background thread."""

    USED = re.compile(r"GPU\[0\]\s*:.*Total Used Memory \(B\):\s*(\d+)")

    def __init__(self, interval=1.0):
        self.interval = interval
        self.peak_vram = 0
        self.peak_gtt = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _one(self, kind):
        out = run(["rocm-smi", "--showmeminfo", kind]).stdout
        used = [int(m) for m in self.USED.findall(out)]
        return used[0] if used else 0

    def _loop(self):
        while not self._stop.is_set():
            self.peak_vram = max(self.peak_vram, self._one("vram"))
            self.peak_gtt = max(self.peak_gtt, self._one("gtt"))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def machine_state():
    """Idle-level GPU temperature, clock and use at the start of a run."""
    out = run(["amd-smi", "monitor", "-P", "-t", "-g", "-u"]).stdout
    gpus = []
    for line in out.splitlines():
        if line.strip().startswith("0 ") or line.strip().startswith("1 "):
            gpus.append(" ".join(line.split()))
    return {"amd_smi": gpus}


def kinetic_bench(model, repetitions):
    out = run([KINETIC, "bench", model, "--device", "vulkan", "--json",
               "-r", str(repetitions)])
    if out.returncode != 0:
        return None, (out.stderr.strip() or f"exit {out.returncode}")
    try:
        return json.loads(out.stdout), None
    except json.JSONDecodeError as e:
        return None, f"bad json: {e}"


def llama_bench(model, repetitions, ngl):
    out = run([LLAMA_BIN, "-m", model, "-dev", "Vulkan0", "-ngl", str(ngl),
               "-p", "512", "-n", "128", "-r", str(repetitions)],
              timeout=7200)
    if out.returncode != 0:
        return None, (out.stderr.strip() or f"exit {out.returncode}")
    row = {}
    for line in out.stdout.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 7 and cells[-2] in ("pp512", "tg128"):
            m = re.match(r"([0-9.]+)\s*±\s*([0-9.]+)", cells[-1])
            if m:
                row[cells[-2]] = [float(m.group(1)), float(m.group(2))]
    return row or None, ("no rows parsed" if not row else None)


def entry(model, path, side, commit, machine, command, result, error,
          repetitions, mem=None, extra=None):
    e = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "kind": "baseline",
        "side": side,
        "model": model,
        "file": path,
        "commit": commit,
        "tree_clean": subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True).stdout == "",
        "command": command,
        "repetitions": repetitions,
        "machine": machine,
    }
    if mem:
        e["peak_memory_bytes"] = mem
    if result is not None:
        e["results"] = result
        e["verdict"] = "ok"
    else:
        e["verdict"] = "failed"
        e["error"] = error
    if extra:
        e.update(extra)
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--ledger", default="ledger.jsonl")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--repetitions", type=int, default=6)
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--llama-ngl", default="99")
    a = ap.parse_args()

    commit = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    machine = machine_state()
    ledger = Path(a.ledger)
    configs = Path(a.configs)
    configs.mkdir(exist_ok=True)

    for model, ngl in zip(a.models[::2], a.models[1::2]):
        name = Path(model).stem
        print(f"=== {name} ===", flush=True)
        with MemorySampler() as mem:
            res, err = kinetic_bench(model, a.repetitions)
        peak = {"vram": mem.peak_vram, "gtt": mem.peak_gtt}

        with ledger.open("a") as f:
            f.write(json.dumps(entry(
                name, model, "kinetic", commit, machine,
                " ".join([KINETIC, "bench", model, "--device", "vulkan",
                          "--json", "-r", str(a.repetitions)]),
                res, err, a.repetitions, peak)) + "\n")
        if res is not None:
            (configs / f"{name}.kinetic.json").write_text(
                json.dumps({"device": "vulkan", "repetitions": a.repetitions,
                            "threads": res.get("threads"),
                            "context": {"prefill_tokens": res.get("prefill_tokens"),
                                        "decode_tokens": res.get("decode_tokens"),
                                        "edit_tokens": res.get("edit_tokens")}},
                           indent=2) + "\n")
            print(f"  kinetic ok: pp {res['prefill_tokens_per_second']:.1f} "
                  f"tg {res['decode_tokens_per_second']:.1f} "
                  f"edit_identical {res['edit_identical']}", flush=True)
        else:
            print(f"  kinetic FAILED: {err}", flush=True)

        if a.skip_llama:
            continue
        with MemorySampler() as mem:
            res, err = llama_bench(model, a.repetitions, ngl)
        with ledger.open("a") as f:
            f.write(json.dumps(entry(
                name, model, "llama-bench", commit, machine,
                " ".join([LLAMA_BIN, "-m", model, "-dev", "Vulkan0",
                          "-ngl", ngl, "-p", "512", "-n", "128",
                          "-r", str(a.repetitions)]),
                res, err, a.repetitions,
                {"vram": mem.peak_vram, "gtt": mem.peak_gtt},
                extra={"ngl": ngl})) + "\n")
        if res is not None:
            (configs / f"{name}.llama.json").write_text(
                json.dumps({"device": "Vulkan0", "ngl": ngl,
                            "repetitions": a.repetitions}, indent=2) + "\n")
            print(f"  llama ok: {res}", flush=True)
        else:
            print(f"  llama FAILED: {err}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
