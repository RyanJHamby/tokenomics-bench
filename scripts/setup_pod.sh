#!/usr/bin/env bash
# One-time setup on a fresh GPU pod (Ubuntu + NVIDIA driver + Python >= 3.11).
# Needs: HF_TOKEN exported (Llama 3.1 is gated), git signing key if you will commit from the pod.
set -euo pipefail
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
pip install -e .
mkdir -p data
[ -f data/gsm8k_test.jsonl ] || curl -fsSL \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
  -o data/gsm8k_test.jsonl
echo "setup done. next: scripts/preflight.sh"
