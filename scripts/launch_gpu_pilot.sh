#!/usr/bin/env bash
set -euo pipefail

source "${SMI_ENV_FILE:-/root/smi-env.sh}"

data_root="${SMI_DATA_ROOT:-/root/autodl-tmp/data}"
runs_root="${SMI_RUNS_ROOT:-/root/autodl-tmp/runs}"
logs_root="${SMI_TENSORBOARD_ROOT:-/root/tf-logs}"
run_name="${SMI_RUN_NAME:-rwkv7-smi-pilot-10m}"
mixture="${SMI_MIXTURE:-$data_root/smi-pilot-10m.parquet}"
manifest="${SMI_MANIFEST:-$mixture.manifest.json}"
prepare_pid_file="${SMI_PREPARE_PID_FILE:-$data_root/smi-pilot-10m.prepare.pid}"
output="$runs_root/$run_name"

if [[ -f "$prepare_pid_file" ]]; then
    prepare_pid="$(<"$prepare_pid_file")"
    while kill -0 "$prepare_pid" 2>/dev/null; do
        sleep 60
    done
fi

python - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
tokens = int(manifest.get("target_tokens", 0))
if not manifest.get("complete") or tokens < 10_000_000:
    raise SystemExit(f"mixture incomplete: {tokens} target tokens")
print(f"validated mixture: {tokens} tokens / {manifest['total_rows']} rows")
PY

if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 1
fi

if [[ "${SMI_DELETE_SOURCE_CACHE_AFTER_PREP:-0}" == "1" ]]; then
    source_cache="${SMI_PILOT_SOURCE_CACHE:?source cache path is required}"
    python - "$manifest" "$source_cache" <<'PY'
import json
import shutil
import sys
from pathlib import Path

data_root = Path("/root/autodl-tmp/data").resolve(strict=True)
target = Path(sys.argv[2]).resolve(strict=True)
with open(sys.argv[1], encoding="utf-8") as stream:
    recorded = Path(json.load(stream)["source_cache"]).resolve(strict=True)
if target != recorded:
    raise SystemExit(f"source cache differs from manifest: {target} != {recorded}")
if target == data_root or not target.is_relative_to(data_root):
    raise SystemExit(f"refusing unsafe source-cache deletion: {target}")
shutil.rmtree(target)
print(f"deleted source cache: {target}")
PY
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec smi-train-rwkv7 \
    --dataset parquet \
    --data-files "$mixture" \
    --messages-column messages_json \
    --dataset-num-proc 8 \
    --output "$output" \
    --max-length 2048 \
    --epochs 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 1 \
    --dtype bfloat16 \
    --wkv-implementation auto \
    --logging-dir "$logs_root/$run_name" \
    --logging-steps 10 \
    --run-name "$run_name" \
    --save-steps 500 \
    --save-total-limit 1
