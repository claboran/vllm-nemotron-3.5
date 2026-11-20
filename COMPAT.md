# Compatibility log

One row per tested combination. Host: MSI EdgeExpert, GB10 (sm_121), driver 580.178.04.

| Date | vLLM image | Model | Spec | Settings | Result | Notes |
|------|------------|-------|------|----------|--------|-------|
| 2026-11-20 | `vllm/vllm-openai:v0.27.1` (torch 2.13.0+cu130, CUDA 13.0) | Nemotron 3.5 Lightning 30B-A3B NVFP4 | DSpark, 3 tokens | ctx 128k, mem 0.75, KV fp8, Marlin, max-num-seqs 16 | ✅ | smoke 8/8 (3 runs); startup ~4.5 min; 124.6 tok/s single, 334 tok/s aggregate @ 8; holds ~92 GB |
| 2026-11-20 | same | same | DSpark, 3 tokens | same, **mem 0.4** (new default) | ✅ | smoke 8/8; restart ~3 min (compile cache); 121.5–121.9 tok/s single, 320–324 @ 8, 470 @ 16; holds ~48 GB, no preemptions |

## Startup (first start, from log)

| Step | Time / size |
|------|-------------|
| main model weights | 115 s |
| DSpark drafter weights | 8 s |
| model memory (main + drafter) | 19.5 GiB |
| weights + non-torch memory | 22.0 GiB |
| peak activation | 0.74 GiB |
| CUDA graph capture | 16 s |
| KV cache (at `GPU_MEMORY_UTILIZATION=0.75`) | 68.5 GiB = 13.7M tokens = 104× concurrency at 128k |
| KV cache (at `GPU_MEMORY_UTILIZATION=0.4`) | 26.1 GiB = 5.2M tokens = 40× concurrency at 128k |
| total until `Application startup complete` | ~4.5 min first start, ~3 min with warm compile cache |

## Throughput (`scripts/bench.py`, temperature 0, thinking off, max_tokens 1024)

| Date | Spec | Concurrency | tok/s per request (median) | aggregate tok/s | Acceptance |
|------|------|------------:|---------------------------:|----------------:|-----------:|
| 2026-11-20 | DSpark, 3 | 1 | 124.6 | 120.0 | 61% |
| 2026-11-20 | DSpark, 3 | 8 | 44.9 | 333.8 | 60% |
| 2026-11-20 | DSpark, 3, mem 0.4 | 1 | 121.5 / 121.9 | 115.7 / 115.8 | 59% |
| 2026-11-20 | DSpark, 3, mem 0.4 | 8 | 44.0 / 43.4 | 320.6 / 323.8 | 59% |
| 2026-11-20 | DSpark, 3, mem 0.4 | 16 | 32.3 | 470.4 | 60% |

Rows without a `mem` note use 0.75. The first benchmark right after a restart came out ~8% lower
(114 / 297); repeats were back to normal → always run the benchmark twice after a restart.

Community reference (Classmethod, same image, 64k ctx): 124.2 single / 354.6 aggregate @ 8.

Smoke-test timings (single request, thinking on): ~130–140 tok/s.

## Findings / gotchas

- **0.75 over-provisions memory; 0.4 is the new default.** At 0.75 vLLM held ~92 GB (`nvidia-smi`), leaving
  ~20 GB available to the system, for 13.7M KV tokens. At 0.4 it holds ~48 GB (system: ~64 GB available),
  still 5.2M KV tokens, with the same smoke result and throughput within run-to-run variation. Hybrid Mamba:
  only a few layers keep a KV cache.
- **JSON-schema `description` fields are not shown to the model.** vLLM only uses the schema to constrain
  decoding. Format requirements (e.g. ISO dates) must be in the prompt, or enforced with `pattern` in the
  schema. Without it, the model returned `03.11.2026` instead of `2026-11-03` once (thinking on).
- **Temperature 0 is not fully deterministic** with thinking on: reasoning length varied between runs
  (5k vs 7k chars) with identical requests. Expected with batching / speculative decoding.
- **Tool calls:**
  - `tool_choice: "auto"` with **thinking off**: the model tends to answer from memory instead of calling the
    tool. With thinking on it calls the tool correctly.
  - `tool_choice: "required"` is broken in practice: vLLM constrains the output to an unbounded array of calls
    and the model repeats the same call until `max_tokens` (`finish_reason: length`). `parallel_tool_calls: false`
    only truncates the result to one call; the tokens are still generated.
  - A **named** `tool_choice` (`{"type":"function","function":{"name":…}}`) works cleanly with thinking off
    (one call, 37 tokens).
- **Warning `max_num_scheduled_tokens is set to 2016`** (speculative decoding reserves draft slots out of the
  default 2048 batched tokens). May slow down long-prompt prefill; try `EXTRA_ARGS=--max-num-batched-tokens 8192`.
- **`nemotron_v3` reasoning parser** is registered lazily: it does not show up in
  `ReasoningParserManager.reasoning_parsers` until used, but works.
- **Network:** downloads from Docker Hub, NGC (`nvcr.io`) and Hugging Face drop out sporadically on this
  network (TLS handshake / connect timeouts), on Wi-Fi and on cable. Retry loops get through; completed image
  layers and HF files are kept between attempts. NGC `nvcr.io/nvidia/vllm` was considered (26.10 = vLLM 0.31,
  CUDA 13.4) but not used: the Docker Hub image matches the version NVIDIA validated the Spark recipe with.

## Next steps

- Compare `SPEC_METHOD=none` / `mtp` / `dflash` against DSpark.
- Try `--max-num-batched-tokens 8192` for long prompts (prefill speed).
