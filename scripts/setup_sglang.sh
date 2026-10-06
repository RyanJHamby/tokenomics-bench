#!/usr/bin/env bash
# SGLang lives in its OWN venv: its pinned torch/flashinfer stack conflicts with vLLM's.
# Exploratory block B7 only. The flag names in configs/b7_sglang_crosscheck.yaml were NOT checked
# against a source tree when written; this script verifies each against the installed build and
# fails if any is missing, so you find out before paying for a launch.
set -euo pipefail
python3 -m venv .venv-sglang
. .venv-sglang/bin/activate
pip install -U pip
pip install "sglang[all]==0.5.10.post1"   # latest on PyPI at the time of writing; pin deliberately
help=$(python -m sglang.launch_server --help 2>&1)
fail=0
for f in model-path context-length mem-fraction-static dtype max-running-requests chunked-prefill-size disable-radix-cache revision enable-metrics host port; do
  echo "$help" | grep -q -- "--$f" && echo "ok    flag --$f" || { echo "FAIL  flag --$f missing in this SGLang"; fail=1; }
done
[ "$fail" = 0 ] && echo "SGLANG SETUP OK: now run scripts/run_all.sh b7 (only if budget remains)" || { echo "SGLANG FLAGS FAILED: fix configgen.py b7 args first"; exit 1; }
