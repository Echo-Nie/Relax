#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# One bounded Qwen3 B/C submission. No cleanup, retries or next experiment.
set -eo pipefail
source "${RELAX_ENV_SH:?Set RELAX_ENV_SH to the existing environment script}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mode=${1:?Usage: run-sampling-replay-p0.sh on|off}
case "$mode" in on|off) ;; *) exit 2 ;; esac
: "${MODEL_PATH:?}" "${EXPERIMENT_ROOT:?}" "${RAY_ADDRESS:?}" "${RAY_DASHBOARD:?}"
: "${P0_TOP_P:?Choose from completed pilot results}" "${P0_TOP_K:?Choose from completed pilot results}"
export RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_NO_WAIT=1 NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
source scripts/entrypoint/ray-job.sh
source scripts/models/qwen3-4B.sh
run_dir="$EXPERIMENT_ROOT/m1-$mode-s1234"
job_id="sr-p0-m1-$mode-s1234"
exec 9>"$EXPERIMENT_ROOT/submit.lock"
flock -n 9 || { echo 'Another submission is in progress' >&2; exit 1; }
python scripts/tools/sampling_replay_ab.py preflight --root "$EXPERIMENT_ROOT" --model "$MODEL_PATH" --dashboard "$RAY_DASHBOARD" --job "$job_id" --run "$run_dir"
replay=()
if [ "$mode" = on ]; then replay=(--use-rollout-sampling-replay); fi
command=(ray job submit --no-wait --submission-id "$job_id" --address "$RAY_DASHBOARD" \
    --runtime-env-json="$RUNTIME_ENV_JSON" -- python -m relax.entrypoints.train \
    --resource '{"actor":[1,1],"rollout":[1,1]}' --max-staleness 0 --num-data-storage-units 1 \
    --hf-checkpoint "$MODEL_PATH" --load "$MODEL_PATH" --megatron-to-hf-mode bridge "${MODEL_ARGS[@]}" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --prompt-data "$EXPERIMENT_ROOT/data/train.jsonl" --input-key prompt --label-key label \
    --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking":false}' \
    --rm-type dapo --reward-key score --num-rollout 6 --rollout-batch-size 2 --n-samples-per-prompt 4 \
    --rollout-max-response-len 4096 --rollout-max-context-len 8192 \
    --rollout-temperature 1 --rollout-top-k "$P0_TOP_K" --rollout-top-p "$P0_TOP_P" \
    --rollout-seed 1234 --rollout-shuffle --sglang-sampling-backend pytorch --sglang-random-seed 1234 "${replay[@]}" \
    --use-rollout-logprobs --get-mismatch-metrics \
    --custom-tis-function-path relax.backends.megatron.loss.vanilla_tis_function --tis-clip 1 --tis-clip-low 1 \
    --global-batch-size 8 --micro-batch-size 1 --advantage-estimator grpo \
    --eps-clip 0.2 --eps-clip-high 0.28 --kl-loss-coef 0 --entropy-coef 0 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.999 \
    --calculate-per-token-loss --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.2 --sglang-context-length 8256 \
    --sglang-enable-fp32-lm-head --sglang-max-running-requests 8 --sglang-disable-cuda-graph \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend flash --no-rope-fusion --seed 1234 \
    --eval-prompt-data heldout "$EXPERIMENT_ROOT/data/eval.jsonl" --eval-interval 6 \
    --n-samples-per-eval-prompt 2 --eval-temperature 1 --eval-top-k "$P0_TOP_K" --eval-top-p "$P0_TOP_P" \
    --eval-max-response-len 4096 --eval-max-context-len 8192 \
    --custom-megatron-before-train-step-hook-path scripts.training.text.sampling_replay_ab_hooks.before_train \
    --dump-details "$run_dir" --tb-project-name sampling-replay-p0 --tb-experiment-name "$job_id")
python -c 'import json,sys; print(json.dumps(sys.argv[1:],indent=2))' "${command[@]}" > "$run_dir/command.json"
"${command[@]}" > "$run_dir/submit.log" 2>&1
cat "$run_dir/submit.log"
