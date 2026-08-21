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
