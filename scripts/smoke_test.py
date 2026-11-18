#!/usr/bin/env python3
"""Functional checks against a running vLLM OpenAI-compatible server (stdlib only).

Usage: scripts/smoke_test.py [--base-url http://127.0.0.1:8000/v1] [--api-key KEY] [--model NAME]
Defaults come from BASE_URL / API_KEY / SERVED_MODEL_NAME environment variables.
Exit code = number of failed checks.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base-url", default=os.environ.get("BASE_URL") or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1")
p.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
p.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "nemotron-3.5-lightning"))
a = p.parse_args()

PASS = FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"PASS: {msg}")


def bad(msg):
    global FAIL
    FAIL += 1
    print(f"FAIL: {msg}")


def call(path, body=None, timeout=600):
    req = urllib.request.Request(a.base_url + path, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if a.api_key:
        req.add_header("Authorization", f"Bearer {a.api_key}")
    data = json.dumps(body).encode() if body is not None else None
    t = time.time()
    with urllib.request.urlopen(req, data, timeout=timeout) as r:
        return json.load(r), time.time() - t


def chat(messages, thinking, **kw):
    body = {"model": a.model, "messages": messages, "temperature": 0, "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": thinking}, **kw}
    return call("/chat/completions", body)


def reasoning_of(msg):  # field was renamed reasoning_content -> reasoning in newer vLLM
    return msg.get("reasoning") or msg.get("reasoning_content") or ""


def show(r, dt):
    m = r["choices"][0]["message"]
    u = r.get("usage", {})
    print(f"  content: {(m.get('content') or '')[:300]!r}")
    print(f"  reasoning chars: {len(reasoning_of(m))}, completion tokens: {u.get('completion_tokens')}, "
          f"{dt:.1f}s, {u.get('completion_tokens', 0) / dt:.1f} tok/s, finish: {r['choices'][0]['finish_reason']}")
    return m


def step(title):
    print(f"\n=== {title}")


INVOICE_TEXT = """Muster Software GmbH · Hauptstr. 1 · 10115 Berlin
Rechnung Nr. 2026-1142    Rechnungsdatum: 03.11.2026
An: Example Consulting, Lindenweg 7, 80331 München
Pos  Beschreibung                         Menge  Einzelpreis   Gesamt
1    Cloud Hosting Paket M (November)        1      49,00 €    49,00 €
2    Zusätzlicher Speicher 100 GB            2       5,50 €    11,00 €
Nettobetrag                                                   60,00 €
zzgl. 19 % USt.                                               11,40 €
Rechnungsbetrag                                               71,40 €
Zahlbar bis 17.11.2026 per SEPA-Lastschrift."""

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "invoiceNumber": {"type": "string"},
        "invoiceDate": {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD)"},
        "netAmount": {"type": "number"},
        "vatRate": {"type": "number", "description": "percent, e.g. 19"},
        "vat": {"type": "number"},
        "totalAmount": {"type": "number"},
        "currency": {"type": "string"},
        "remark": {"type": "string"},
    },
    "required": ["vendor", "invoiceNumber", "invoiceDate", "netAmount", "vatRate", "vat", "totalAmount", "currency", "remark"],
    "additionalProperties": False,
}
EXPECTED = {"invoiceNumber": "2026-1142", "invoiceDate": "2026-11-03", "netAmount": 60.0, "vatRate": 19,
            "vat": 11.40, "totalAmount": 71.40}


def check_invoice(content, label):
    try:
        d = json.loads(content)
    except Exception as e:  # noqa: BLE001
        bad(f"{label}: content is not JSON ({e})")
        return
    wrong = {k: (d.get(k), v) for k, v in EXPECTED.items()
             if (abs(float(d.get(k, -1)) - v) > 0.001 if isinstance(v, float | int) else d.get(k) != v)}
    if "muster" not in str(d.get("vendor", "")).lower():
        wrong["vendor"] = (d.get("vendor"), "Muster Software GmbH")
    if wrong:
        bad(f"{label}: wrong fields {wrong}")
    else:
        ok(f"{label}: valid JSON, all fields correct")


try:
    step("1. /v1/models")
    r, _ = call("/models")
    ids = [m["id"] for m in r["data"]]
    print(f"  models: {ids}, max_model_len: {[m.get('max_model_len') for m in r['data']]}")
    ok("models") if a.model in ids else bad(f"{a.model} not in {ids}")

    q = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]

    step("2. chat, thinking off")
    m = show(*chat(q, False))
    ok("answer contains Paris, no reasoning") if "paris" in (m.get("content") or "").lower() and not reasoning_of(m) \
        else bad("thinking off")

    step("3. chat, thinking on (default)")
    m = show(*chat(q, True))
    ok("answer contains Paris, reasoning separated") if "paris" in (m.get("content") or "").lower() and reasoning_of(m) \
        else bad("thinking on: expected answer in content and reasoning in reasoning field")

    msgs = [{"role": "system", "content": "Extract the invoice data. Amounts as numbers. If something is unclear, say so in remark, else empty string."},
            {"role": "user", "content": INVOICE_TEXT}]
    rf = {"type": "json_schema", "json_schema": {"name": "invoice", "strict": True, "schema": INVOICE_SCHEMA}}

    step("4. structured output (json_schema, strict), thinking off")
    m = show(*chat(msgs, False, response_format=rf))
    check_invoice(m.get("content") or "", "json_schema/thinking off")

    step("5. structured output (json_schema, strict), thinking on")
    m = show(*chat(msgs, True, response_format=rf))
    check_invoice(m.get("content") or "", "json_schema/thinking on")
    ok("reasoning kept out of JSON content") if reasoning_of(m) else bad("no reasoning returned with thinking on")

    step("6. tool calling")
    tools = [{"type": "function", "function": {
        "name": "get_exchange_rate", "description": "Get the exchange rate between two currencies",
        "parameters": {"type": "object", "properties": {"base": {"type": "string"}, "quote": {"type": "string"}},
                       "required": ["base", "quote"]}}}]
    r, dt = chat([{"role": "user", "content": "How many US dollars is one euro today?"}], False, tools=tools,
                 tool_choice="auto")
    tc = r["choices"][0]["message"].get("tool_calls") or []
    print(f"  tool_calls: {[(t['function']['name'], t['function']['arguments']) for t in tc]}  ({dt:.1f}s)")
    if tc and tc[0]["function"]["name"] == "get_exchange_rate" and \
            {"base", "quote"} <= set(json.loads(tc[0]["function"]["arguments"])):
        ok("tool call")
    else:
        bad("tool call")

    step("7. authentication")
    if a.api_key:
        try:
            req = urllib.request.Request(a.base_url + "/models")
            urllib.request.urlopen(req, timeout=10)
            bad("request without API key was accepted")
        except urllib.error.HTTPError as e:
            ok(f"request without key rejected ({e.code})") if e.code == 401 else bad(f"unexpected {e.code}")
    else:
        print("  skipped (API_KEY empty)")

except urllib.error.URLError as e:
    bad(f"server not reachable at {a.base_url}: {e}")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(FAIL)
