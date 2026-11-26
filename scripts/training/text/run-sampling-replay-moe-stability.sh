#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Explicitly launch ONE arm; never loop over methods or clear shared services.
set -euo pipefail
source "${RELAX_ENV_SH:?Export the existing env.sh path first}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
tier=${1:?smoke|baseline|pressure}
mode=${2:?off|on|r3|tis|sc}
root=${3:?prepared experiment root}
ray_tmp=${4:?new short private Ray temporary directory}
case "$tier" in smoke) subnet=40 ;; baseline) subnet=41 ;; pressure) subnet=42 ;; *) exit 2 ;; esac
case "$mode" in off) id=1 ;; on) id=2 ;; r3) id=3 ;; tis) id=4 ;; sc) id=5 ;; *) exit 2 ;; esac
instance="$root/$tier/$mode/ray"
python scripts/tools/sampling_replay_moe_stability.py argv --root "$root" --tier "$tier" --mode "$mode" >/dev/null
python scripts/tools/sampling_replay_isolated.py create --instance "$instance" --mode "$mode" \
    --gpus 0,1,2,3,4,5,6,7 --host-ip "127.96.$subnet.$id" --ray-tmpdir "$ray_tmp"
# The bridge listens only inside this owned namespace and only tunnels ClearML.
nohup python scripts/tools/sampling_replay_tracking_proxy.py --instance "$instance" \
    >"$root/$tier/$mode/tracking-proxy.log" 2>&1 </dev/null &
proxy_pid=$!
printf '%s\n' "$proxy_pid" >"$root/$tier/$mode/tracking-proxy.pid"
# Ready checks are bounded; none modifies or cleans existing services.
for ((attempt=0; attempt<60; attempt++)); do
    if ! kill -0 "$proxy_pid" 2>/dev/null; then
        cat "$root/$tier/$mode/tracking-proxy.log"
        exit 1
    fi
    if grep -q 'bridge ready' "$root/$tier/$mode/tracking-proxy.log" && \
        python scripts/tools/sampling_replay_isolated.py exec --instance "$instance" -- \
          curl --fail --silent --max-time 2 http://127.0.0.1:8265/api/version >/dev/null; then
        exec python scripts/tools/sampling_replay_isolated.py exec --instance "$instance" -- \
          python scripts/tools/sampling_replay_moe_stability.py submit \
            --root "$root" --tier "$tier" --mode "$mode" --instance "$instance"
    fi
    sleep 1
done
printf '%s\n' 'Private cluster/bridge not ready; inspect only this instance. No retry or cleanup performed.' >&2
exit 1
