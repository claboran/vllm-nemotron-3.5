# vllm-nemotron-3.5

Run **NVIDIA Nemotron 3.5 Lightning 30B-A3B** (NVFP4) with **vLLM** as an OpenAI-compatible API
on an NVIDIA **DGX Spark / GB10**, with Docker Compose. Includes a smoke test for the features
applications usually depend on (thinking on/off, strict JSON-schema output, tool calls) and a
small throughput benchmark.

Based on NVIDIA's DGX Spark recipe from the
[model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4), using the
pinned `vllm/vllm-openai:v0.27.1` image (no source build). Test results are in [COMPAT.md](COMPAT.md).

---

## Contents

- [The model](#the-model)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Using the API](#using-the-api)
- [Speculative decoding](#speculative-decoding)
- [Memory on GB10](#memory-on-gb10)
- [Why Marlin, not native FP4](#why-marlin-not-native-fp4)
- [Testing](#testing)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [License](#license)

---

## The model

| | |
|---|---|
| Architecture | hybrid Mamba / Transformer, Mixture of Experts: 30B total, ~3B active per token |
| Checkpoint | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) (4-bit NVFP4 weights) |
| Drafter | [`…-NVFP4-DSpark`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark), speculative-decoding draft model tuned for DGX Spark |
| Context | up to 1M tokens (this setup: 128k by default) |
| Languages | English, German, French, Spanish, Italian, Japanese, code |
| Thinking | on by default; switch off per request with `chat_template_kwargs.enable_thinking=false` |
| License | OpenMDW 1.1 (commercial use allowed) |

Only some layers are attention layers with a KV cache; the Mamba layers keep a small fixed-size state
per sequence. Long contexts are therefore cheap in memory.

## Requirements

| Component | Version / note |
|-----------|----------------|
| GPU | DGX Spark / GB10 (sm_121, 128 GB unified memory). Other GPUs: adapt the flags in `scripts/serve.sh` |
| NVIDIA driver | ≥ 580 (CUDA 13) |
| Docker + Compose v2 | with the NVIDIA Container Toolkit (`docker run --rm --gpus all ubuntu nvidia-smi` must work) |
| Disk | ~20 GB for the image, ~25 GB for weights and caches |
| Python 3 | on the host, for the test scripts (stdlib only, no packages) |

## Quick start

```bash
cp .env.example .env     # set RUN_UID/RUN_GID to `id -u` / `id -g`, adjust if needed
make pull                # pull vllm/vllm-openai:v0.27.1 (with retries)
make models              # download model + DSpark drafter into ./cache/huggingface
make up                  # start the server (first start takes several minutes)
make logs                # wait for "Application startup complete"
make smoke               # functional checks
make bench               # throughput at 1 and 8 parallel requests
```

The API is then at `http://127.0.0.1:8000/v1`. Other targets: `make status`, `make restart`,
`make down`, `make shell` (bash in the image, e.g. for `vllm serve --help`).

The first start is slow (weight loading, `torch.compile`, CUDA graph capture). The compile caches
are kept in `./cache`, so later starts are faster. The health check allows 20 minutes.

## Configuration

All settings are in `.env` (template: [`.env.example`](.env.example)). The `vllm serve` command
line is built from them by [`scripts/serve.sh`](scripts/serve.sh), which prints the final command
at the top of the log.

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_IMAGE` | `vllm/vllm-openai:v0.27.1` | pinned vLLM image |
| `RUN_UID` / `RUN_GID` | `1000` | user the container runs as (owner of `./cache`) |
| `CACHE_DIR` | `./cache` | weights (`huggingface/`) and compile caches |
| `HF_TOKEN` | *(empty)* | only for gated repositories |
| `BIND` / `PORT` | `127.0.0.1` / `8000` | host address and port; `0.0.0.0` for LAN access |
| `API_KEY` | *(empty)* | Bearer token; empty = no authentication |
| `MODEL` | `nvidia/…-NVFP4` | Hugging Face repo of the model |
| `SERVED_MODEL_NAME` | `nemotron-3.5-lightning` | model name clients send in `"model"` |
| `MAX_MODEL_LEN` | `131072` | max tokens per request (prompt + output) |
| `GPU_MEMORY_UTILIZATION` | `0.75` | share of total memory vLLM reserves, see [Memory](#memory-on-gb10) |
| `MAX_NUM_SEQS` | `16` | max parallel sequences |
| `KV_CACHE_DTYPE` | `fp8` | `fp8` or `auto` |
| `SPEC_METHOD` | `dspark` | `dspark`, `dflash`, `mtp` or `none` |
| `SPEC_TOKENS` | `3` | draft tokens per step |
| `EXTRA_ARGS` | *(empty)* | additional `vllm serve` flags, appended verbatim |

Fixed flags (from NVIDIA's DGX Spark recipe) in `serve.sh`: `--moe-backend marlin`,
`--mamba-backend flashinfer`, `--mamba-cache-mode align`, `--enable-prefix-caching`,
`--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder`, `--enable-auto-tool-choice`.

After changing `.env`: `make restart`.

## Using the API

Standard OpenAI API: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus `/health` and
`/metrics` (Prometheus). Add `-H "Authorization: Bearer $API_KEY"` if a key is set.

### Chat

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nemotron-3.5-lightning",
  "messages": [{"role": "user", "content": "Explain KV caching in two sentences."}],
  "max_tokens": 4096
}'
```

The model **thinks by default**. The thinking text is returned in a separate field of the message
(`reasoning`, older vLLM versions: `reasoning_content`), the answer in `content`. Give it enough
`max_tokens`, or switch thinking off per request:

```json
{ "messages": [...], "chat_template_kwargs": { "enable_thinking": false } }
```

NVIDIA recommends `temperature 1.0, top_p 0.95` for chat. For extraction tasks, `temperature 0`
gives reproducible output.

### Structured output (JSON schema)

vLLM enforces a JSON schema with constrained decoding, so `content` is always valid JSON for the schema:

```json
{
  "model": "nemotron-3.5-lightning",
  "messages": [{"role": "user", "content": "<invoice text>"}],
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false},
  "response_format": {
    "type": "json_schema",
    "json_schema": {"name": "invoice", "strict": true, "schema": { "type": "object", "...": "..." }}
  }
}
```

With thinking on, the schema only applies to the answer after the reasoning; `content` still holds
just the JSON (checked by the smoke test).

### Tool calls

`tools` / `tool_choice: "auto"` work as in the OpenAI API (parser `qwen3_coder`). For coding
agents, NVIDIA recommends `"chat_template_kwargs": {"force_nonempty_content": true}`.

### Python (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<API_KEY or anything>")
r = client.chat.completions.create(
    model="nemotron-3.5-lightning",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

## Speculative decoding

A small draft model proposes `SPEC_TOKENS` tokens, and the main model verifies them all in one
forward pass. Every accepted token saves a full decoding step. The output is the same as without
speculation; only the speed changes.

| `SPEC_METHOD` | Drafter | Note |
|---------------|---------|------|
| `dspark` (default) | separate checkpoint, tuned for DGX Spark | NVIDIA's recommendation for Spark |
| `dflash` | separate checkpoint (`make models-dflash`) | |
| `mtp` | the model's built-in multi-token-prediction heads | needs `moe_backend: triton` for the drafter, see below |
| `none` | – | baseline |

A community benchmark on DGX Spark (vLLM v0.27.1, 64k context, code prompts) measured:

| Method | 1 request (tok/s) | 8 parallel, aggregate (tok/s) |
|--------|------------------:|------------------------------:|
| none | 81 | 242 |
| MTP | 111 | 302 |
| DFlash | 95 | 269 |
| **DSpark** | **124** | **355** |

Source: [Classmethod – DSpark / DFlash / MTP on DGX Spark](https://dev.classmethod.jp/en/articles/nvidia-nemotron-3-5-lightning-dspark-dflash-mtp-dgx-spark/).
Our own numbers are in [COMPAT.md](COMPAT.md).

**About `moe_backend: triton` for MTP:** the MTP heads contain an MoE block. By default the
drafter inherits the main model's `--moe-backend marlin`, which the drafter does not support, so
vLLM fails at startup. `serve.sh` therefore sets the drafter's MoE kernel to `triton`. These are
the generic MoE kernels written in OpenAI Triton, a Python GPU-kernel language. It has nothing to
do with NVIDIA Triton Inference Server.

## Memory on GB10

vLLM allocates memory differently from llama.cpp. At startup it **reserves**
`GPU_MEMORY_UTILIZATION × total memory`: weights, activations and CUDA graphs, with **all the
rest used as KV cache** for parallel requests. On GB10, CPU and GPU share the same 128 GB
(about 121 GB visible), so this reservation is taken away from the OS and every other process.

| `GPU_MEMORY_UTILIZATION` | Reserved | Note |
|--------------------------|---------:|------|
| 0.85 (NVIDIA recipe) | ~103 GB | only if nothing else runs on the machine |
| **0.75 (default)** | ~90 GB | leaves room for the OS, IDE backend and tools |
| 0.45 | ~55 GB | enough for 128k context and a few parallel requests; use this next to other GPU services |

- vLLM refuses to start if less memory is free than requested. Stop other GPU services
  (e.g. llama.cpp containers) or lower the value.
- `MAX_MODEL_LEN` matters as well: vLLM requires that at least one sequence of full length fits
  into the KV cache. The model's default of 1M tokens forces a large reservation; 128k is plenty
  for most uses.
- The startup log tells you what you got: `GPU KV cache size: N tokens` and
  `Maximum concurrency for 131072 tokens per request: X`.

## Why Marlin, not native FP4

NVFP4 stores weights as 4-bit floats with one FP8 scale per 16 weights, about 4× smaller than BF16.

| Path | Arithmetic | Status on GB10 |
|------|-----------|----------------|
| native FP4 (W4A4) | weights *and* activations in FP4 on the FP4 tensor cores | vLLM's FP4 MoE kernels target datacenter Blackwell (sm_100); the model card marks this path "No" for GB10 |
| **Marlin (W4A16)** | 4-bit weights unpacked to BF16 in registers, BF16 tensor cores | used here (`--moe-backend marlin`) |

What this means in practice:

- **Token generation** is limited by memory bandwidth (GB10: ~273 GB/s). Weights are still read
  as 4-bit, so almost nothing is lost here.
- **Prompt processing / large batches** are compute-bound. Native FP4 would be faster; this is
  where the cost shows.
- **Quality:** W4A16 is, if anything, slightly more accurate than W4A4, since activations stay in 16 bits.

## Testing

```bash
make smoke    # or: scripts/smoke_test.py --base-url http://host:8000/v1 --api-key KEY
```

Checks: model listed · chat with thinking off (no reasoning) · thinking on (reasoning separated
from content) · strict `json_schema` extraction of a German invoice with thinking off and on (all
fields checked) · tool call · unauthenticated request rejected (if `API_KEY` is set). Exit code =
number of failures.

```bash
make bench                                          # concurrency 1 and 8, temperature 0
scripts/bench.py --concurrency 1 4 8 16 --max-tokens 2048
scripts/bench.py --temperature 1.0 --thinking
```

Reports median tokens/s per request, aggregate tokens/s and the draft acceptance rate (from
`/metrics`). Two warm-up requests are excluded. For clean numbers, nothing else should be running
on the GPU.

## Security

- By default the port is bound to **127.0.0.1**, so only clients on the Spark itself can connect
  (including an IDE remote backend running there).
- For LAN access set `BIND=0.0.0.0` **and** `API_KEY=$(openssl rand -hex 32)`. Ports published by
  Docker bypass `ufw` rules.
- The container runs as your user (not root) and only mounts `./cache` and `serve.sh` (read-only).

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Free memory on device … is less than desired GPU memory utilization` | other processes use the unified memory → stop them or lower `GPU_MEMORY_UTILIZATION` |
| `… max seq len … larger than the maximum number of tokens that can be stored in KV cache` | lower `MAX_MODEL_LEN` or raise `GPU_MEMORY_UTILIZATION` |
| error about `moe_backend` with `SPEC_METHOD=mtp` | the drafter needs `moe_backend: triton` (set by `serve.sh`) |
| container `unhealthy` during the first start | compile + CUDA graph capture can take many minutes; watch `make logs` |
| empty `content`, `finish_reason: length` | thinking used up `max_tokens` → raise it or disable thinking |
| permission errors in `./cache` | `RUN_UID`/`RUN_GID` in `.env` do not match the owner of `./cache` |
| Docker Hub TLS timeouts | `make pull` retries 5 times |

## Repository layout

```
.
├── compose.yaml          # vLLM service (GPU, port, caches, health check)
├── .env.example          # all settings, copy to .env
├── Makefile              # pull / models / up / down / logs / smoke / bench
├── scripts/
│   ├── serve.sh          # container entrypoint: builds `vllm serve …` from env
│   ├── smoke_test.py     # functional checks (stdlib only)
│   └── bench.py          # throughput benchmark (stdlib only)
├── COMPAT.md             # tested combinations and measured results
└── cache/                # (git-ignored) weights + compile caches
```

## License

The model is under the [OpenMDW License 1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4);
vLLM under Apache 2.0.
