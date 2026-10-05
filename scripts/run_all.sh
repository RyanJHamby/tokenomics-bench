#!/usr/bin/env bash
# Sweep in dependency order. Every step runs under `pow run` (raw output + hardware info
# saved in results/). Raw per-cell JSON lands in results/raw/<STAMP>-<block>/.
#
# Resumable: re-run with the SAME STAMP and finished cells are skipped:
#     STAMP=20261012-0900 PRICE=0.99 scripts/run_all.sh b2 b3
# Run inside tmux/screen: a dropped ssh session otherwise kills the sweep mid-cell.
set -euo pipefail
: "${PRICE:?set PRICE to the live on-demand usd/hr}"
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "${ALLOW_NO_MUX:-0}" != 1 ]; then
  echo "run inside tmux or screen (or ALLOW_NO_MUX=1)"; exit 2
fi
. .venv/bin/activate
STAMP=${STAMP:-$(date +%Y%m%d-%H%M)}
[ $# -gt 0 ] && blocks=("$@") || blocks=(b1 b2 b3 b4 b5 b7)
python -m tokbench.budget status

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
      # A failed gate must not abort the sweep; it is recorded and the block's speedups are
      # simply not claimable. Decide from the printed verdict.
      pow run gate-prefix-compare -- python -m tokbench.gates compare "$G/prefix-on-s90.json" "$G/prefix-off-s90.json" \
        --noise "$G/prefix-on-s90.json" "$G/prefix-on-s90-rerun.json" || echo "GATE FAILED: see results/"
      run b2 configs/b2_prefix_cache.yaml ;;
    b3)
      for v in fp16 fp8 awq-int4; do
        pow run "gate-gsm8k-$v" -- python -m tokbench.gates gsm8k configs/b3_quantization.yaml $v data/gsm8k_test.jsonl --n 200
      done
      run b3 configs/b3_quantization.yaml ;;
    b4) run b4 configs/b4_cuda_graphs.yaml ;;
    b5) run b5 configs/b5_power_caps.yaml ;;
    b7)
      : "${B1_DIR:?set B1_DIR to the COMPLETE b1 raw dir, e.g. results/raw/<STAMP>-b1}"
      python -m tokbench.overload "$B1_DIR" --variant fp16-default > "results/raw/b7-derived-$STAMP.yaml"
      run b7 "results/raw/b7-derived-$STAMP.yaml" ;;
    *) echo "unknown block $b"; exit 2 ;;
  esac
done
echo "DONE. STOP THE POD NOW, then record the invoice: python -m tokbench.budget add ..."
