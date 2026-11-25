#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Explicit single submission; no loop, cleanup or automatic next stage.
set -euo pipefail
source "${RELAX_ENV_SH:?}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
stage=${1:?smoke|pilot|print}; mode=${2:?on|off}
case "$stage:$mode" in smoke:on|smoke:off|pilot:on|pilot:off|print:on|print:off) ;; *) exit 2 ;; esac
: "${EXPERIMENT_ROOT:?}"
profile=$stage; if [ "$stage" = print ]; then profile=pilot; fi
argv="$EXPERIMENT_ROOT/$profile-$mode-argv.json"
python scripts/tools/sampling_replay_exp08.py argv --root "$EXPERIMENT_ROOT" --stage "$profile" --mode "$mode" > "$argv"
if [ "$stage" = print ]; then cat "$argv"; exit; fi
: "${RAY_ADDRESS:?}" "${RAY_DASHBOARD:?}"
export RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_NO_WAIT=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
source scripts/entrypoint/ray-job.sh
exec 9>"$EXPERIMENT_ROOT/submit.lock"
flock -n 9 || { echo 'Another submission is in progress' >&2; exit 1; }
python scripts/tools/sampling_replay_exp08.py preflight --root "$EXPERIMENT_ROOT" --mode "$mode" --stage "$stage" --dashboard "$RAY_DASHBOARD"
readarray -d '' training_args < <(python -c 'import json,sys; [sys.stdout.write(x+"\0") for x in json.load(open(sys.argv[1]))]' "$argv")
run="$EXPERIMENT_ROOT/$stage-$mode-s1234"
ray job submit --no-wait --submission-id "sr-exp08-$stage-$mode-s1234" --address "$RAY_DASHBOARD" \
    --runtime-env-json="$RUNTIME_ENV_JSON" -- python -m relax.entrypoints.train "${training_args[@]}" > "$run/submit.log" 2>&1
cat "$run/submit.log"
