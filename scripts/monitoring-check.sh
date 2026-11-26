#!/usr/bin/env bash
# Checks the monitoring stack (compose.monitoring.yaml): Prometheus and Grafana are up, the vLLM
# server is scraped (if it is running), the vllm:* series the dashboard uses exist, and the
# dashboard is provisioned. Run after `make up-monitoring`.
# Usage: scripts/monitoring-check.sh            (reads ports/credentials from .env if present)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a
PROM="http://127.0.0.1:${PROMETHEUS_PORT:-9091}"
GRAFANA="http://127.0.0.1:${GRAFANA_PORT:-3001}"
GF_AUTH="${GRAFANA_ADMIN_USER:-admin}:${GRAFANA_ADMIN_PASSWORD:-}"
PASS=0; FAIL=0

ok()  { echo "PASS: $*"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $*"; FAIL=$((FAIL+1)); }
query() {  # PromQL -> JSON result list
  curl -fsS -G "$PROM/api/v1/query" --data-urlencode "query=$1" | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin)["data"]["result"]))'
}
count() {  # PromQL -> number of series
  query "$1" | python3 -c 'import sys,json; print(len(json.load(sys.stdin)))'
}

for _ in $(seq 1 15); do curl -fsS "$PROM/-/ready" >/dev/null 2>&1 && break; sleep 2; done
if curl -fsS "$PROM/-/ready" >/dev/null 2>&1; then ok "prometheus ready ($PROM)"; else bad "prometheus not reachable at $PROM"; fi
# Grafana needs ~30 s for database migrations on its first start
for _ in $(seq 1 30); do curl -fsS "$GRAFANA/api/health" >/dev/null 2>&1 && break; sleep 2; done
if curl -fsS "$GRAFANA/api/health" >/dev/null 2>&1; then ok "grafana healthy ($GRAFANA)"; else bad "grafana not reachable at $GRAFANA"; fi

target=$(curl -fsS "$PROM/api/v1/targets?scrapePool=vllm" 2>/dev/null | python3 -c '
import sys, json
for t in json.load(sys.stdin)["data"]["activeTargets"]:
    print(t["health"], t.get("lastError", ""))')
echo "  target vllm-nemotron:8000: ${target:-none}"
running=$(docker ps --filter name=^vllm-nemotron$ -q)
if [[ "$target" == up* ]]; then
  ok "vllm scraped"
  # metric names the dashboard relies on (counters/histograms appear once vLLM is ready)
  missing=()
  for m in num_requests_running num_requests_waiting kv_cache_usage_perc prompt_tokens_total \
           generation_tokens_total prefix_cache_queries_total time_to_first_token_seconds_bucket \
           inter_token_latency_seconds_bucket e2e_request_latency_seconds_bucket \
           request_queue_time_seconds_sum request_success_total num_preemptions_total; do
    [[ $(count "vllm:$m{job=\"vllm\"}") -ge 1 ]] || missing+=("$m")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then ok "all vllm:* series used by the dashboard present"
  else bad "missing vllm:* series: ${missing[*]} (metric names changed after a VLLM_IMAGE bump?)"; fi
  if [[ "${SPEC_METHOD:-dspark}" != none ]]; then
    if [[ $(count 'vllm:spec_decode_num_drafts_total{job="vllm"}') -ge 1 ]]; then ok "speculative-decoding series present"
    else bad "SPEC_METHOD=${SPEC_METHOD:-dspark} but no vllm:spec_decode_* series"; fi
  fi
elif [[ -n "$running" ]]; then
  bad "vllm-nemotron is running but not scraped (still loading? else not on network llm-metrics: make restart)"
else
  echo "skip: vllm-nemotron not running"
fi

if curl -fsS -u "$GF_AUTH" "$GRAFANA/api/dashboards/uid/vllm-nemotron" >/dev/null 2>&1; then
  ok "grafana dashboard 'vLLM Nemotron' provisioned"
else bad "grafana dashboard 'vllm-nemotron' missing (or GRAFANA_ADMIN_* wrong)"; fi

printf '\n=== result: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]]
