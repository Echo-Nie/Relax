#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail
source "${RELAX_ENV_SH:?Export the existing env.sh path}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD:$PYTHONPATH"
exec python scripts/tools/sampling_replay_countdown.py start \
  --mode "${1:?off|on|tis|sc|sc-tis}" --root "${2:?experiment root}" \
  --ray-tmpdir "${3:?new short Ray directory}" --stage "${4:-formal}"
