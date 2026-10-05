#!/usr/bin/env bash
# Fail fast, before any billable sweep. Checks the things that have burned time before:
# wrong vLLM version, missing flags, no root for power caps, no model access, no FP8 support.
set -uo pipefail
fail=0
ok()  { echo "ok    $*"; }
bad() { echo "FAIL  $*"; fail=1; }

want=$(grep -E '^vllm==' requirements-gpu.txt | cut -d= -f3)
have=$(python -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || echo none)
[ "$have" = "$want" ] && ok "vllm $have" || bad "vllm is $have, pinned $want"

nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,power.default_limit \
  --format=csv,noheader && ok "nvidia-smi" || bad "nvidia-smi"

# Power cap round trip (needs root). Sets the current limit to itself.
cur=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
nvidia-smi -pl "$cur" >/dev/null 2>&1 && ok "power cap writable" || bad "cannot set power limit (need root?); B5 impossible"

help=$(vllm serve --help=all 2>&1 || vllm serve --help 2>&1)
for f in enable-prefix-caching enforce-eager quantization max-model-len gpu-memory-utilization; do
  echo "$help" | grep -q -- "--$f" && ok "flag --$f" || bad "flag --$f not found in this vLLM"
done
echo "$help" | grep -q -- "--no-enable-prefix-caching" && ok "flag --no-enable-prefix-caching" \
  || bad "--no-enable-prefix-caching missing: B2 'off' arm would silently stay on"

[ -n "${HF_TOKEN:-}" ] && ok "HF_TOKEN set" || bad "HF_TOKEN not set"
python - <<'PY' || fail=1
import os, sys
try:
    from huggingface_hub import model_info
    model_info("meta-llama/Llama-3.1-8B-Instruct", token=os.environ.get("HF_TOKEN"))
    print("ok    model access")
except Exception as e:
    print("FAIL  model access:", type(e).__name__); sys.exit(1)
PY

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
awk -v c="$cc" 'BEGIN{exit !(c>=8.9)}' && ok "compute capability $cc (FP8 ok)" \
  || bad "compute capability $cc: no FP8; drop fp8 variants"

df -h . | tail -1 | awk '{print "disk free: "$4}'
[ "$fail" = 0 ] && echo "PREFLIGHT PASSED" || { echo "PREFLIGHT FAILED"; exit 1; }
