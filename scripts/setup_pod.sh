#!/usr/bin/env bash
# One-time setup on a fresh GPU pod (Ubuntu + NVIDIA driver + Python >= 3.11).
# Needs HF_TOKEN exported (Llama 3.1 is gated). Run inside tmux (see run_all.sh).
set -euo pipefail
GSM8K_SHA256=3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14

python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
pip install -e .

# pow is the proof-of-work wrapper every sweep step runs under; install it from proof-kit.
command -v pow >/dev/null || { echo "install pow first (RyanJHamby proof-kit): scripts need it"; exit 1; }

mkdir -p data
f=data/gsm8k_test.jsonl
if ! { [ -f "$f" ] && echo "$GSM8K_SHA256  $f" | sha256sum -c - >/dev/null 2>&1; }; then
  curl -fsSL \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
    -o "$f.tmp"
  echo "$GSM8K_SHA256  $f.tmp" | sha256sum -c -   # fails loudly on a partial/changed download
  mv "$f.tmp" "$f"
fi
echo "setup done. next: scripts/preflight.sh"
