#!/usr/bin/env bash
# Validate the load generator against vLLM's own benchmark on ONE server instance (~10 GPU min).
# Run after preflight, before the pilot. A DISAGREE verdict means a client-side artefact:
# investigate before trusting any latency number from either tool.
set -euo pipefail
: "${PRICE:?set PRICE to the live on-demand usd/hr}"
. .venv/bin/activate
QPS=${QPS:-4}; DUR=${DUR:-140}; N=$(python -c "print(int($QPS*$DUR))")
OUT=results/crosscheck/$(date +%Y%m%d-%H%M); mkdir -p "$OUT"
MODEL=meta-llama/Llama-3.1-8B-Instruct

vllm serve "$MODEL" --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true' EXIT
until curl -fs localhost:8000/health >/dev/null; do sleep 3; kill -0 $SRV || { echo "server died"; exit 1; }; done

# 1) vLLM's own benchmark: same shape (512 in / 128 out, ignore_eos), Poisson at $QPS.
pow run xcheck-vllm-bench -- vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
  --model "$MODEL" --dataset-name random --random-input-len 512 --random-output-len 128 \
  --ignore-eos --request-rate "$QPS" --num-prompts "$N" --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,99 --save-result --result-dir "$OUT" --result-filename vllm_bench.json
# 2) tokbench on the same server instance.
pow run xcheck-tokbench -- python -m tokbench.runner configs/crosscheck.yaml --attach http://127.0.0.1:8000 \
  --server vllm --gpu nvml --usd-per-hr "$PRICE" --out "$OUT/tokbench"
python -m tokbench.crosscheck "$OUT"/tokbench/fp16-default__q4__r0.json "$OUT/vllm_bench.json" | tee "$OUT/verdict.txt"
