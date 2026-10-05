#!/usr/bin/env bash
# Sweep in dependency order. Every step runs under `pow run` (raw output + hardware info
# saved in results/). Raw per-cell JSON lands in results/raw/<STAMP>-<block>/.
#
#   scripts/run_all.sh b1            # pilot + capacity (also the variance estimate)
#   python -m tokbench.pilot results/raw/<STAMP>-b1   # sizes later blocks
#   scripts/run_all.sh b2gen         # generate B2 arms from the pilot; COMMIT the result
#   scripts/run_all.sh b2 b3a b3b b4
#
# Resumable: re-run with the SAME STAMP and finished cells are skipped:
#     STAMP=20261012-0900 PRICE=0.99 scripts/run_all.sh b3b
# Run inside tmux/screen: a dropped ssh session otherwise kills the sweep mid-cell.
set -euo pipefail
: "${PRICE:?set PRICE to the live on-demand usd/hr}"
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "${ALLOW_NO_MUX:-0}" != 1 ]; then
  echo "run inside tmux or screen (or ALLOW_NO_MUX=1)"; exit 2
fi
. .venv/bin/activate
STAMP=${STAMP:-$(date +%Y%m%d-%H%M)}
[ $# -gt 0 ] && blocks=("$@") || { echo "usage: run_all.sh <block...>  (b1 b2gen b2 b3a b3b b4)"; exit 2; }
mkdir -p results/capacity
python -m tokbench.budget status

run() { # name config
  pow run "$1" -- python -m tokbench.runner "$2" --server vllm --gpu nvml \
    --usd-per-hr "$PRICE" --out "results/raw/$STAMP-$1"
}

for b in "${blocks[@]}"; do
  case $b in
    b1) run b1 configs/b1_capacity_pilot.yaml ;;
    b2gen)
      : "${B1_DIR:?set B1_DIR to the pilot raw dir, e.g. results/raw/<STAMP>-b1}"
      python -m tokbench.arms "$B1_DIR" --template configs/b2_power_clock.template.yaml > configs/b2_power_clock.yaml
      echo "Generated configs/b2_power_clock.yaml from the pilot. COMMIT IT (signed) before running b2;"
      echo "the arm levels are a pre-registered function of the pilot, and must be on record first." ;;
    b2)
      # refuse to run unless the generated arms are committed and unmodified
      git ls-files --error-unmatch configs/b2_power_clock.yaml >/dev/null 2>&1 \
        && git diff --quiet -- configs/b2_power_clock.yaml \
        || { echo "configs/b2_power_clock.yaml is missing, untracked or modified: run b2gen and commit it first"; exit 2; }
      run b2 configs/b2_power_clock.yaml ;;
    b3a) run b3a configs/b3a_quant_capacity.yaml ;;
    b3b)
      G=results/gates/$STAMP; mkdir -p "$G"
      for v in fp16-default fp8 fp8-kv8 awq-int4; do
        pow run "gate-gsm8k-$v" -- python -m tokbench.gates gsm8k configs/b3b_quant_fixed_load.yaml $v data/gsm8k_test.jsonl "$G/gsm8k-$v.json"
      done
      for v in fp8 fp8-kv8 awq-int4; do   # a failed gate is recorded, never aborts the sweep
        pow run "gate-gsm8k-cmp-$v" -- python -m tokbench.gates gsm8k-compare "$G/gsm8k-fp16-default.json" "$G/gsm8k-$v.json" || echo "QUALITY GATE: $v not non-inferior (see results/)"
      done
      run b3b configs/b3b_quant_fixed_load.yaml ;;
    b4) run b4 configs/b4_graph_modes.yaml ;;
    *) echo "unknown block $b"; exit 2 ;;
  esac
done
echo "DONE. STOP THE POD NOW, then record the invoice: python -m tokbench.budget add ..."
