#!/usr/bin/env python3
"""Concurrent soak with a memory watchdog (M01 stability gate).

Starts `kinetic serve` with `--parallel N` (or reuses --base-url), fires
N concurrent requests that each generate `--max-tokens` tokens over a
long prompt, for `--rounds` rounds, and samples system memory and GPU
memory while it runs. The watchdog kills the server and fails the run
when MemAvailable falls below `--min-mem-mb` or the GPU's used VRAM
exceeds `--max-vram-mb`; a server that dies on its own fails the run
too. Reports wall time, tokens, aggregate and per-slot t/s, p50/p99 of
the request latencies, and the memory peaks.

    python scripts/soak_watchdog.py model.gguf --device vulkan \\
        --parallel 4 --context 8192 --max-tokens 512 --rounds 3
"""

import argparse
import datetime
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "Summarize the rules of the workshop in one sentence, then list them.\n"

VRAM_USED = re.compile(r"GPU\[0\]\s*:.*Total Used Memory \(B\):\s*(\d+)")


def post(url, body, timeout=1800):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(url, proc, seconds=300):
    end = time.time() + seconds
    while time.time() < end:
        if proc.poll() is not None:
            sys.exit(f"server exited early: {' '.join(proc.args)}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    proc.kill()
    sys.exit(f"server not ready after {seconds} s")


class Watchdog(threading.Thread):
    def __init__(self, proc, min_mem_mb, max_vram_mb):
        super().__init__(daemon=True)
        self.proc = proc
        self.min_mem_mb = min_mem_mb
        self.max_vram_mb = max_vram_mb
        self.stop = threading.Event()
        self.peak_vram = 0
        self.peak_min_mem = 1 << 60
        self.tripped = None

    def run(self):
        while not self.stop.is_set():
            try:
                with open("/proc/meminfo") as f:
                    m = {k: int(v.split()[0]) for k, v, *_ in
                         (l.split(":", 1) for l in f if ":" in l)}
                avail = m["MemAvailable"] // 1024
                self.peak_min_mem = min(self.peak_min_mem, avail)
                if avail < self.min_mem_mb:
                    self.tripped = f"MemAvailable {avail} MiB < {self.min_mem_mb} MiB"
                    self.proc.kill()
                    return
                out = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True, text=True).stdout
                used = [int(x) for x in VRAM_USED.findall(out)]
                if used:
                    self.peak_vram = max(self.peak_vram, used[0])
                    if self.max_vram_mb and used[0] > self.max_vram_mb * 2**20:
                        self.tripped = (f"VRAM {used[0] / 2**20:.0f} MiB > "
                                        f"{self.max_vram_mb} MiB")
                        self.proc.kill()
                        return
            except (KeyError, OSError, subprocess.SubprocessError):
                pass
            self.stop.wait(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--kinetic", default="target/release/kinetic")
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-mem-mb", type=int, default=2048,
                    help="watchdog: fail when MemAvailable falls under this")
    ap.add_argument("--max-vram-mb", type=int, default=0,
                    help="watchdog: fail when VRAM use exceeds this (0 = off)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--ledger", default="ledger.jsonl")
    a = ap.parse_args()

    filler = PROMPT * max(1, a.prompt_tokens * 4 // len(PROMPT))

    if a.base_url:
        base = a.base_url.rstrip("/")
        proc = None
    else:
        proc = subprocess.Popen(
            [a.kinetic, "serve", a.model, "--port", str(a.port),
             "--device", a.device, "--parallel", str(a.parallel),
             "-c", str(a.context)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{a.port}"
        wait_ready(base + "/health", proc)

    dog = Watchdog(proc, a.min_mem_mb, a.max_vram_mb)
    dog.start()
    t0 = time.time()
    latencies, tokens, errors = [], 0, []
    for r in range(a.rounds):
        print(f"round {r + 1}/{a.rounds} ...", flush=True)
        results = [None] * a.parallel

        def one(i):
            start = time.time()
            try:
                resp = post(base + "/v1/completions", {
                    "prompt": filler, "max_tokens": a.max_tokens,
                    "temperature": 0, "top_k": 1, "repeat_penalty": 1.0})
                usage = resp.get("usage") or {}
                results[i] = (time.time() - start,
                              usage.get("completion_tokens", 0))
            except Exception as e:
                errors.append(f"round {r + 1} slot {i}: {e}")

        threads = [threading.Thread(target=one, args=(i,))
                   for i in range(a.parallel)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if proc and proc.poll() is not None:
            errors.append(f"server exited: watchdog={dog.tripped}")
            break
        for lat, tok in results:
            if lat is not None:
                latencies.append(lat)
                tokens += tok
    wall = time.time() - t0
    dog.stop.set()
    dog.join(timeout=5)
    if proc:
        proc.kill()
        proc.wait()

    ok = not errors and latencies
    lat = sorted(latencies)
    summary = {
        "rounds": a.rounds,
        "parallel": a.parallel,
        "wall_s": round(wall, 1),
        "tokens": tokens,
        "aggregate_tokens_per_second": round(tokens / wall, 2) if wall else 0,
        "request_p50_s": round(statistics.median(lat), 2) if lat else None,
        "request_p99_s": round(lat[int(len(lat) * 0.99) - 1], 2) if lat else None,
        "peak_vram_mib": round(dog.peak_vram / 2**20) if dog.peak_vram else None,
        "min_mem_available_mib": dog.peak_min_mem,
        "watchdog_tripped": dog.tripped,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    entry = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "kind": "soak",
        "model": Path(a.model).stem if a.model else a.base_url,
        "commit": commit,
        "command": f"python scripts/soak_watchdog.py {' '.join(sys.argv[1:])}",
        "results": summary,
        "verdict": "ok" if ok and not dog.tripped else "failed",
    }
    with open(a.ledger, "a") as f:
        f.write(json.dumps(entry) + "\n")
    sys.exit(0 if ok and not dog.tripped else 1)


if __name__ == "__main__":
    main()
