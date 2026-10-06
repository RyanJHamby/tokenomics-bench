#!/usr/bin/env bash
# Sweep in dependency order. Every step runs under `pow run` (raw output + hardware info
# saved in results/). Raw per-cell JSON lands in results/raw/<STAMP>-<block>/.
#
#   scripts/run_all.sh b1            # pilot + capacity (also the variance estimate)
#   python -m tokbench.pilot results/raw/<STAMP>-b1   # sizes later blocks
#   scripts/run_all.sh b2gen         # generate B2 arms from the pilot; COMMIT the result
#   scripts/run_all.sh b2 b3a b3b b4          # core blocks (confirmatory)
#   scripts/run_all.sh b5                     # EXPLORATORY: overload + recovery
#   B1_DIR=... scripts/run_all.sh b6gen       # EXPLORATORY: MoE arms from the pilot; commit, then b6
#   scripts/setup_sglang.sh && scripts/run_all.sh b7   # EXPLORATORY, last: SGLang cross-check
# Run exploratory blocks only after the core blocks finish and only if budget remains.
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
[ $# -gt 0 ] && blocks=("$@") || { echo "usage: run_all.sh <block...>  (b1 b2gen b2 b3a b3b b4 b5 b6gen b6 b7)"; exit 2; }
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
    b3a) run b3a-fp8 configs/b3a_fp8_capacity.yaml; run b3a-extra configs/b3a_extra_capacity.yaml ;;
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
    b5) run b5 configs/b5_overload_recovery.yaml ;;
    b6gen)
      : "${B1_DIR:?set B1_DIR to the pilot raw dir}"
      python -m tokbench.arms "$B1_DIR" --template configs/b6_moe.template.yaml --locks 0.70 --no-caps > configs/b6_moe.yaml
      echo "Generated configs/b6_moe.yaml. COMMIT IT (signed) before running b6." ;;
    b6)
      git ls-files --error-unmatch configs/b6_moe.yaml >/dev/null 2>&1 \
        && git diff --quiet -- configs/b6_moe.yaml \
        || { echo "configs/b6_moe.yaml is missing, untracked or modified: run b6gen and commit it first"; exit 2; }
      run b6 configs/b6_moe.yaml ;;
    b7)
      [ -x .venv-sglang/bin/python ] || { echo "run scripts/setup_sglang.sh first"; exit 2; }
      SGLANG_PYTHON=$PWD/.venv-sglang/bin/python run b7 configs/b7_sglang_crosscheck.yaml ;;
    *) echo "unknown block $b"; exit 2 ;;
  esac
done
echo "DONE. STOP THE POD NOW, then record the invoice: python -m tokbench.budget add ..."
