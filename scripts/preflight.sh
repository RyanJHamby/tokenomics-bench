#!/usr/bin/env bash
# Fail fast, before any billable sweep. Each check exists because the failure it catches
# would waste GPU money or silently invalidate results.
set -uo pipefail
. .venv/bin/activate 2>/dev/null || true
fail=0
ok()   { echo "ok    $*"; }
bad()  { echo "FAIL  $*"; fail=1; }
warn() { echo "warn  $*"; }

command -v pow >/dev/null && ok "pow installed" || bad "pow not installed (run_all wraps every step in it)"

want=$(grep -E '^vllm==' requirements-gpu.txt | cut -d= -f3)
have=$(python -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || echo none)
[ "$have" = "$want" ] && ok "vllm $have" || bad "vllm is $have, pinned $want"

nvidia-smi --query-gpu=name,driver_version,memory.total,power.min_limit,power.max_limit,power.default_limit,power.limit \
  --format=csv,noheader && ok "nvidia-smi" || bad "nvidia-smi"
nvidia-smi -L | grep -qi "MIG" && bad "MIG device: measurements would be of a slice" || ok "no MIG"

# vllm 0.30.0 defaults to CUDA 13 wheels, which need a driver >= 580.
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${drv:-0}" -ge 580 ] && ok "driver $drv >= 580" || bad "driver $drv < 580: CUDA 13 wheels won't load; use a +cu129 build or another image"
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
  && ok "torch sees the GPU" || bad "torch.cuda.is_available() is false"

mkdir -p results/preflight
python -m tokbench.telemetry.probe | tee results/preflight/telemetry_probe.json | python -c 'import json,sys; d=json.load(sys.stdin); print("telemetry:", d.get("nvml"), "| dcgmi:", bool(d.get("dcgmi")), "| RAPL:", d.get("rapl_readable")); [print("   unsupported:", k) for k,v in d.get("fields",{}).items() if v!="ok"]'
python - <<'PY' && ok "nvml energy counter" || bad "nvidia-ml-py missing or no energy counter (falls back to sampled power: less accurate)"
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
PY

# Provider probe. Many containers forbid -pl / -lgc even as root; find out BEFORE paying
# for a power-cap sweep. A stale cap from a previous crashed run would also show here.
cur=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
def=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1)
[ "${cur%.*}" = "${def%.*}" ] && ok "power limit is the default ($def W)" \
  || bad "power limit $cur W != default $def W: stale cap from an earlier run; reset with nvidia-smi -pl $def"
nvidia-smi -pl "${cur%.*}" >/dev/null 2>&1 && ok "power cap (-pl) writable" \
  || { bad "cannot set power limit: power-cap block is impossible on this pod"; }
maxsm=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
if nvidia-smi -lgc "$maxsm,$maxsm" >/dev/null 2>&1; then
  nvidia-smi -rgc >/dev/null 2>&1; ok "clock lock (-lgc) writable"
else
  warn "cannot lock clocks (-lgc): the cap-vs-lock experiment is impossible on this pod"
fi

# Every power_cap_w in the configs must be inside this card's allowed range.
python - <<'PY' || fail=1
import glob, yaml, sys
from tokbench.power import power_limits
lim = power_limits(0)
bad = False
for f in sorted(glob.glob("configs/*.yaml")):
    for v in yaml.safe_load(open(f)).get("variants", []):
        w = v.get("power_cap_w")
        if w is not None and not lim["min"] <= w <= lim["max"]:
            print(f"FAIL  {f}: {v['name']} cap {w} W outside [{lim['min']}, {lim['max']}]"); bad = True
print(f"ok    power range [{lim['min']}, {lim['max']}] W, default {lim['default']} W" if not bad else "")
sys.exit(1 if bad else 0)
PY

help=$(vllm serve --help=all 2>&1 || vllm serve --help 2>&1)
for f in enable-prefix-caching enforce-eager quantization max-model-len gpu-memory-utilization kv-cache-dtype compilation-config dtype generation-config max-num-seqs max-num-batched-tokens seed attention-backend async-scheduling enable-chunked-prefill revision tokenizer-revision; do
  echo "$help" | grep -q -- "--$f" && ok "flag --$f" || bad "flag --$f not found in this vLLM"
done
echo "$help" | grep -q -- "--no-enable-prefix-caching" && ok "flag --no-enable-prefix-caching" \
  || bad "--no-enable-prefix-caching missing: B2 'off' arm would silently stay on"

# Every CUDA-graph mode named in the configs must be accepted by this vLLM build.
python - <<'PY' || fail=1
import glob, json, re, sys
try:
    from vllm.config import CompilationConfig
except Exception as e:
    print("FAIL  cannot import vllm.config:", type(e).__name__); sys.exit(1)
modes = set()
for f in glob.glob("configs/*.yaml"):
    modes |= set(re.findall(r'"cudagraph_mode": "(\w+)"', open(f).read()))
bad = []
for m in sorted(modes):
    try:
        CompilationConfig(**json.loads(json.dumps({"cudagraph_mode": m})))
    except Exception as e:
        bad.append((m, type(e).__name__))
print("ok    cudagraph modes accepted: " + ", ".join(sorted(modes)) if not bad else f"FAIL  rejected modes: {bad}")
sys.exit(1 if bad else 0)
PY

[ -n "${HF_TOKEN:-}" ] && ok "HF_TOKEN set" || bad "HF_TOKEN not set"
# Every (model, revision) the configs pin must still resolve, and the token must see the gated one.
python - <<'PY' || fail=1
import glob, os, sys, yaml
from huggingface_hub import model_info
pairs = set()
for f in glob.glob("configs/*.yaml"):
    cfg = yaml.safe_load(open(f))
    for v in cfg.get("variants", []):
        a = v.get("server_args", [])
        if "--revision" in a:
            pairs.add((v.get("model", cfg["model"]), a[a.index("--revision") + 1]))
bad = 0
for model, rev in sorted(pairs):
    try:
        info = model_info(model, revision=rev, token=os.environ.get("HF_TOKEN"))
        print(f"ok    {model}@{rev[:8]} resolves")
    except Exception as e:
        print(f"FAIL  {model}@{rev[:8]}: {type(e).__name__}"); bad = 1
sys.exit(bad)
PY

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
awk -v c="$cc" 'BEGIN{exit !(c>=8.9)}' && ok "compute capability $cc (FP8 ok)" \
  || bad "compute capability $cc: no FP8; drop fp8 variants"

[ -f data/gsm8k_test.jsonl ] && [ "$(wc -l < data/gsm8k_test.jsonl)" = 1319 ] && ok "gsm8k file (1319)" || bad "data/gsm8k_test.jsonl missing/incomplete"
python - <<'PY' || fail=1
import os
n = os.cpu_count() or 0
print(f"ok    {n} vCPUs" if n >= 8 else f"FAIL  {n} vCPUs: vLLM API server + client will contend; need >= 8")
raise SystemExit(0 if n >= 8 else 1)
PY
lsof -i :8000 >/dev/null 2>&1 && bad "port 8000 already in use" || ok "port 8000 free"
df -h . | tail -1 | awk '{print "disk free: "$4}'
python -m tokbench.budget status
[ "$fail" = 0 ] && echo "PREFLIGHT PASSED" || { echo "PREFLIGHT FAILED"; exit 1; }
