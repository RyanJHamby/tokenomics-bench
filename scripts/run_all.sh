#!/usr/bin/env bash
# Full sweep in dependency order. Every step is wrapped in `pow run` so raw output and
# hardware info are captured in results/. Raw per-cell JSON goes to results/raw/<stamp>-<block>/.
# Usage: PRICE=<usd/hr from the provider's live rate> scripts/run_all.sh [block ...]
set -euo pipefail
: "${PRICE:?set PRICE to the live on-demand usd/hr}"
. .venv/bin/activate
STAMP=$(date +%Y%m%d-%H%M)
[ $# -gt 0 ] && blocks=("$@") || blocks=(b1 b2 b3 b4 b5 b7)

run() { # name config
  pow run "$1" -- python -m tokbench.runner "$2" --server vllm --gpu nvml \
    --usd-per-hr "$PRICE" --out "results/raw/$STAMP-$1"
}

for b in "${blocks[@]}"; do
  case $b in
    b1) run b1 configs/b1_knee.yaml ;;
    b2)
      G=results/gates/$STAMP; mkdir -p "$G"
      for v in prefix-on-s90 prefix-off-s90; do
        pow run "gate-collect-$v" -- python -m tokbench.gates collect configs/b2_prefix_cache.yaml $v "$G/$v.json"
      done
      pow run gate-collect-noise -- python -m tokbench.gates collect configs/b2_prefix_cache.yaml prefix-on-s90 "$G/prefix-on-s90-rerun.json"
      pow run gate-prefix-compare -- python -m tokbench.gates compare "$G/prefix-on-s90.json" "$G/prefix-off-s90.json" \
        --noise "$G/prefix-on-s90.json" "$G/prefix-on-s90-rerun.json"
      run b2 configs/b2_prefix_cache.yaml ;;
    b3)
      for v in fp16 fp8 awq-int4; do
        pow run "gate-gsm8k-$v" -- python -m tokbench.gates gsm8k configs/b3_quantization.yaml $v data/gsm8k_test.jsonl --n 200
      done
      run b3 configs/b3_quantization.yaml ;;
    b4) run b4 configs/b4_cuda_graphs.yaml ;;
    b5) run b5 configs/b5_power_caps.yaml ;;
    b7)
      B1=$(ls -d results/raw/*-b1 | tail -1)
      python -m tokbench.overload "$B1" --variant fp16-default > "results/raw/b7-derived-$STAMP.yaml"
      run b7 "results/raw/b7-derived-$STAMP.yaml" ;;
    *) echo "unknown block $b"; exit 2 ;;
  esac
done
