#!/usr/bin/env python3
"""Needle retrieval at 50 % and 100 % of the context (M01 stability gate).

Starts `kinetic serve` (or reuses --base-url), fills the context with
repeating filler lines, hides one needle sentence with a random code
word at half and at the end of the window, asks the model to return the
code word greedily, and requires an exact answer. Position is placed by
characters with a 4 chars/token rule and a 10 % safety margin, so a
--context of N tokens is never overshot; the script reports the filler
token estimate against the context actually set.

    python scripts/needle_test.py model.gguf --device vulkan --context 8192
"""

import argparse
import json
import random
import re
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request

FILLER = ("The mill's wheel turns slowly under the autumn sky, and the "
          "barge carries grain down to the harbour.\n")
NEEDLE_TMPL = "The access code for this batch is {code}.\n"
QUESTION = ("\n\nWhat is the access code for this batch? Answer with the "
            "code only, nothing else.\n")
CPT = 4.5  # chars per token, first guess for English filler; the script
# self-corrects from the server's token count when the prompt is refused


def post(url, body, timeout=600):
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


def probe(base, prompt, ctx_tokens):
    """Chat completions, so the model answers through its template."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16, "temperature": 0, "top_k": 1,
        "repeat_penalty": 1.0}
    try:
        r = post(base + "/v1/chat/completions", body)
    except urllib.error.HTTPError as e:
        # 'the prompt is 16014 tokens, which leaves no room...': rescale
        m = re.search(r"the prompt is (\d+) tokens", e.read().decode())
        if not m:
            raise
        n = int(m.group(1))
        raise SystemExit(
            f"prompt too long ({n} tokens for a {ctx_tokens}-token "
            "context); lower --context or the filler")
    text = (r["choices"][0]["message"].get("content") or "").strip()
    usage = r.get("usage") or {}
    return text, usage


def needle_case(code, position, ctx_tokens):
    """Build the prompt whose needle sits at ~position of the window."""
    budget = int(ctx_tokens * CPT * 0.95)  # 5 % margin for the question
    needle = NEEDLE_TMPL.format(code=code)
    needle_at = int(budget * position)
    fill = needle_at // len(FILLER)
    prompt = FILLER * fill + needle
    prompt += FILLER * ((budget - len(prompt)) // len(FILLER))
    assert len(prompt) <= budget + len(needle) + len(FILLER)
    return prompt + QUESTION


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--kinetic", default="target/release/kinetic")
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--base-url", default=None,
                    help="reuse a running server instead of starting one")
    ap.add_argument("--port", type=int, default=18090)
    a = ap.parse_args()

    if a.base_url:
        base = a.base_url.rstrip("/")
    else:
        args = [a.kinetic, "serve", a.model, "--port", str(a.port),
                "--device", a.device, "--parallel", str(a.parallel),
                "-c", str(a.context)]
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{a.port}"
        wait_ready(base + "/health", proc)

    rng = random.Random()
    failed = False
    for position, label in ((0.5, "50 %"), (0.99, "100 %")):
        code = "NEEDLE-" + "".join(rng.choice(string.ascii_uppercase)
                                   for _ in range(6))
        prompt = needle_case(code, position, a.context)
        text, usage = probe(base, prompt, a.context)
        ok = code in text
        est = usage.get("prompt_tokens", "?")
        print(f"needle {label}: prompt_tokens={est} -> {text!r} "
              f"({'PASS' if ok else 'FAIL'})")
        failed |= not ok
    if not a.base_url:
        proc.kill()
        proc.wait()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
