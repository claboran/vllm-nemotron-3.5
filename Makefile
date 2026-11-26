# Convenience targets. Settings come from .env (copy .env.example).
-include .env
export
VLLM_IMAGE ?= vllm/vllm-openai:v0.27.1
CACHE_DIR  ?= ./cache
MODEL      ?= nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
DSPARK_MODEL ?= nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
DFLASH_MODEL ?= nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash
BENCH_ARGS ?= --concurrency 1 8
MON_COMPOSE := docker compose -f compose.monitoring.yaml --profile monitoring

HF_RUN = docker run --rm --user $(shell id -u):$(shell id -g) \
	-e HOME=/cache/home -e HF_HOME=/cache/huggingface -e HF_TOKEN \
	-v $(abspath $(CACHE_DIR)):/cache --entrypoint hf $(VLLM_IMAGE)

.PHONY: network pull cache-dir models models-dflash up down restart logs status smoke bench shell \
        up-monitoring down-monitoring logs-monitoring check-monitoring

network:  ## shared Docker network for Prometheus scraping (external in both compose files)
	@docker network inspect llm-metrics >/dev/null 2>&1 || docker network create llm-metrics

pull:  ## pull the pinned vLLM image (with retries)
	@for i in 1 2 3 4 5; do docker pull $(VLLM_IMAGE) && break; echo "retry $$i"; sleep 10; done

cache-dir:
	@mkdir -p $(CACHE_DIR)/home $(CACHE_DIR)/huggingface $(CACHE_DIR)/vllm $(CACHE_DIR)/triton $(CACHE_DIR)/xdg

models: cache-dir  ## download main model + DSpark drafter into CACHE_DIR
	$(HF_RUN) download $(MODEL)
	$(HF_RUN) download $(DSPARK_MODEL)

models-dflash: cache-dir  ## optional: DFlash drafter (only for SPEC_METHOD=dflash)
	$(HF_RUN) download $(DFLASH_MODEL)

up: cache-dir network
	docker compose up -d
down:
	docker compose down
restart: network
	docker compose up -d --force-recreate
logs:
	docker compose logs -f --tail 200
status:
	@docker compose ps
	@curl -fsS http://127.0.0.1:$${PORT:-8000}/health >/dev/null && echo "health: ok" || echo "health: not ready"

smoke:  ## functional checks against the running server
	python3 scripts/smoke_test.py

bench:  ## throughput at concurrency 1 and 8
	python3 scripts/bench.py $(BENCH_ARGS)

shell:  ## shell in the image (for vllm --help etc.)
	docker run --rm -it --gpus all --entrypoint bash $(VLLM_IMAGE)

# --- monitoring: Prometheus + Grafana (compose.monitoring.yaml, Grafana :3001, Prometheus 127.0.0.1:9091) ---
up-monitoring: network
	$(MON_COMPOSE) up -d
down-monitoring:
	$(MON_COMPOSE) down
logs-monitoring:
	$(MON_COMPOSE) logs -f --tail 100
check-monitoring:  ## Prometheus/Grafana up, vLLM scraped, dashboard provisioned
	scripts/monitoring-check.sh
