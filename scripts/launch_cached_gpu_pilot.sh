#!/usr/bin/env bash
set -euo pipefail

source "${SMI_ENV_FILE:-/root/smi-env.sh}"

data_root="${SMI_DATA_ROOT:-/root/autodl-tmp/data}"
source_cache="${SMI_PILOT_SOURCE_CACHE:-$data_root/smi-pilot-source-cache}"
cache_pid_file="${SMI_CACHE_PID_FILE:-$data_root/smi-pilot-source-cache.pid}"
mixture="${SMI_MIXTURE:-$data_root/smi-pilot-10m.parquet}"
prepare_pid_file="${SMI_PREPARE_PID_FILE:-$data_root/smi-pilot-10m.prepare.pid}"

cache_pid="$(<"$cache_pid_file")"
while kill -0 "$cache_pid" 2>/dev/null; do
    sleep 60
done

python - "$source_cache/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
if not manifest.get("complete") or not manifest.get("files"):
    raise SystemExit("pilot source cache is incomplete")
print(f"validated source cache: {len(manifest['files'])} files")
PY

if [[ -e "$mixture" || -e "$data_root/.smi-pilot-10m.parquet.incomplete" ]]; then
    echo "refusing to overwrite existing pilot mixture" >&2
    exit 1
fi

SMI_PILOT_SOURCE_CACHE="$source_cache" nohup smi-prepare-pilot \
    --output "$mixture" \
    --max-length 2048 \
    --max-serialized-chars 32768 \
    --shuffle-buffer 10000 \
    --row-group-size 1024 \
    --workers 16 \
    >"$data_root/smi-pilot-10m.prepare.log" 2>&1 &
prepare_pid=$!
echo "$prepare_pid" >"$prepare_pid_file"

export SMI_PILOT_SOURCE_CACHE="$source_cache"
export SMI_DELETE_SOURCE_CACHE_AFTER_PREP=1
exec "$(dirname "$0")/launch_gpu_pilot.sh"
