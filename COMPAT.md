# Compatibility log

One row per tested combination. Host: MSI EdgeExpert, GB10 (sm_121), driver 580.178.04.

| Date | vLLM image | Model | Spec | Settings | Result | Notes |
|------|------------|-------|------|----------|--------|-------|

## Throughput (`scripts/bench.py`, temperature 0, thinking off, max_tokens 1024)

| Date | Spec | Concurrency | tok/s per request | aggregate tok/s | Acceptance |
|------|------|------------:|------------------:|----------------:|-----------:|

## Findings / gotchas

- **Network (2026-11-18):** on Wi-Fi the Spark got only ~0.1 MB/s from Hugging Face; image pull and
  weight download were aborted. Use a wired connection for `make pull` / `make models`
  (image ~10+ GB, model 52 safetensors shards + DSpark drafter ≈ 20 GB).

## Status / next steps (as of 2026-11-18)

Done: project files written and validated (`docker compose config`, script syntax), nothing run yet.
Nothing downloaded: `./cache` was removed, `vllm/vllm-openai:v0.27.1` is not on the host.

Next, on cable:

1. `make pull` – image first: `make models` runs the `hf` CLI from that image (no `hf` on the host)
2. `make models`
3. `make up && make logs` – first start takes several minutes; check the KV-cache / max-concurrency
   lines and that the flags in `serve.sh` are accepted by v0.27.1 (`make shell` → `vllm serve --help`)
4. `make smoke`, then `make bench`; record results in the tables above
5. Optional: compare `SPEC_METHOD=mtp` / `dflash` / `none`
