#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# One checkpoint export/evaluation only. User explicitly selects each run.
set -euo pipefail
source "${RELAX_ENV_SH:?}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
stage=${1:?smoke|pilot}; mode=${2:?common|off|on}; updates=${3:?0|1|16|32}; gpu=${4:?free physical GPU}
case "$stage:$mode:$updates" in pilot:common:0|pilot:off:16|pilot:on:16|pilot:off:32|pilot:on:32|smoke:off:1|smoke:on:1) ;; *) exit 2 ;; esac
: "${EXPERIMENT_ROOT:?}"
root=$EXPERIMENT_ROOT
model=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$root/manifest.json")
profile=eval; if [ "$stage" = smoke ]; then profile=eval-smoke; fi
output="$root/evaluations/$stage/$mode-u$updates"
[[ ! -e "$output" ]] || { echo "Result directory exists: $output" >&2; exit 1; }
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2
child=""
trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit 130' TERM INT
port=$((40000 + gpu))
python scripts/tools/sampling_replay_evaluate.py preflight --gpu "$gpu" --port "$port"
mkdir -p "$root/models" "$root/evidence"
exec 9>"$root/eval-gpu-$gpu.lock"
flock -n 9 || { echo 'This evaluation GPU is already reserved' >&2; exit 1; }
if [ "$updates" != 0 ]; then
    checkpoint="$root/$stage-$mode-s1234/checkpoints/iter_$(printf '%07d' "$((updates - 1))")"
    test -f "$checkpoint/.metadata"
    python -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; n=int(sys.argv[2]); assert len(rows)>=n and all(x["optimizer_successful"] for x in rows[:n])' "$root/$stage-$mode-s1234/parameter-updates.jsonl" "$updates"
    exported="$root/models/$stage-$mode-u$updates"
    [[ ! -e "$exported" ]] || { echo "Export exists; inspect before reuse: $exported" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/tools/convert_torch_dist_to_hf_bridge.py --input-dir "$checkpoint" \
        --origin-hf-dir "$model" --output-dir "$exported" --bf16 --copy-config --max-shard-size-mb 512 \
        > "$root/evidence/export-$stage-$mode-u$updates.log" 2>&1 &
    child=$!
    wait "$child"
    child=""
    model=$exported
fi
test -s "$root/evaluation-argv.json"
exec python scripts/tools/sampling_replay_evaluate.py run --training-argv "$root/evaluation-argv.json" \
    --model "$model" --data "$root/data/$profile.jsonl" --manifest "$root/$profile-manifest.json" \
    --output "$output" --gpu "$gpu" --port "$port" --samples-per-prompt 4 --concurrency 8 \
    --temperature 1 --top-p .95 --top-k 64 --max-response-len 4096 --max-context-len 8192 \
    --chat-template-kwargs '{"enable_thinking":false}' --fixed-sampling-seeds --repeats 1
