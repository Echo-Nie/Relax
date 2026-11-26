#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Submit exactly one smoke into an explicitly created private eight-GPU namespace.
set -euo pipefail
source "${RELAX_ENV_SH:?Source the existing environment via RELAX_ENV_SH}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mode=${1:?off|on|r3|tis|sc}
root=${2:?prepared experiment root}
instance=${3:?private Ray instance directory}
case "$mode" in off|on|r3|tis|sc) ;; *) exit 2 ;; esac
exec python scripts/tools/sampling_replay_isolated.py exec --instance "$instance" -- \
    python scripts/tools/sampling_replay_moe.py submit --root "$root" --mode "$mode" --instance "$instance"
