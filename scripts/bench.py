#!/usr/bin/env python3
"""Generation throughput against a running vLLM server (stdlib only).

Sends N requests with C in flight at a time and reports per-request and aggregate tokens/s.
Usage: scripts/bench.py [--concurrency 1 8] [--requests 8] [--max-tokens 1024] [--temperature 0]
"""
import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser()
p.add_argument("--base-url", default=os.environ.get("BASE_URL") or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1")
p.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
p.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "nemotron-3.5-lightning"))
p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
p.add_argument("--requests", type=int, default=0, help="requests per level (default: 2 x concurrency, min 4)")
p.add_argument("--max-tokens", type=int, default=1024)
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--thinking", action="store_true", help="enable thinking (default off)")
a = p.parse_args()

PROMPTS = [
    "Write a Python function that parses German-formatted amounts like '1.234,56 €' into a Decimal, with unit tests.",
    "Write a TypeScript Zod schema and a parser for CSV bank statement lines (date;description;amount), with error handling.",
    "Explain how double-entry bookkeeping works, with a worked example of three transactions.",
    "Write a short essay on the trade-offs of running language models locally versus in the cloud.",
]


def post(path, body):
    req = urllib.request.Request(a.base_url + path, json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if a.api_key:
        req.add_header("Authorization", f"Bearer {a.api_key}")
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def one(i):
    t = time.time()
    r = post("/chat/completions", {
        "model": a.model, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": 0.95 if a.temperature else 1.0,
        "chat_template_kwargs": {"enable_thinking": a.thinking}})
    return r["usage"]["completion_tokens"], time.time() - t


def spec_counters():
    """Speculative decoding counters from /metrics (names differ between vLLM versions)."""
    try:
        req = urllib.request.Request(a.base_url.removesuffix("/v1") + "/metrics")
        if a.api_key:
            req.add_header("Authorization", f"Bearer {a.api_key}")
        text = urllib.request.urlopen(req, timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        vals = re.findall(rf"^vllm:spec_decode_{name}(?:_total)?(?:{{[^}}]*}})? ([0-9.e+]+)$", text, re.M)
        if vals:
            out[name] = sum(float(v) for v in vals)
    return out


print(f"model {a.model}, max_tokens {a.max_tokens}, temperature {a.temperature}, thinking {a.thinking}")
print("warmup ...")
for i in range(2):
    one(i)

print(f"\n{'conc':>4} {'reqs':>4} {'tok/s per req (median)':>22} {'aggregate tok/s':>16} {'avg tokens':>10} {'accept %':>8}")
for c in a.concurrency:
    n = a.requests or max(4, 2 * c)
    before = spec_counters()
    t = time.time()
    with ThreadPoolExecutor(c) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t
    after = spec_counters()
    drafted = after.get("num_draft_tokens", 0) - before.get("num_draft_tokens", 0)
    accepted = after.get("num_accepted_tokens", 0) - before.get("num_accepted_tokens", 0)
    acc = f"{100 * accepted / drafted:.0f}" if drafted else "-"
    toks = sum(x[0] for x in res)
    per = statistics.median(x[0] / x[1] for x in res)
    print(f"{c:>4} {n:>4} {per:>22.1f} {toks / wall:>16.1f} {toks / n:>10.0f} {acc:>8}")
