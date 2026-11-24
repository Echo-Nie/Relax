#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# One explicit submission only. No cleanup, loop, automatic retry or next experiment.
set -eo pipefail
source "${RELAX_ENV_SH:?Set RELAX_ENV_SH to the existing environment script}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
stage=${1:?Usage: run-sampling-replay-ab.sh smoke|formal on|off 1234|5678}
mode=${2:?Missing replay mode}
seed=${3:?Missing seed}
case "$stage:$mode:$seed" in
    smoke:on:1234|smoke:off:1234|formal:on:1234|formal:off:1234|formal:on:5678|formal:off:5678) ;;
    *) echo 'Unsupported experiment configuration' >&2; exit 2 ;;
esac
: "${MODEL_PATH:?}" "${EXPERIMENT_ROOT:?}" "${RAY_ADDRESS:?}" "${RAY_DASHBOARD:?}"
export RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_NO_WAIT=1 NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source scripts/entrypoint/ray-job.sh
run_dir="$EXPERIMENT_ROOT/$stage-$mode-s$seed"
job_id="sr-ab-$stage-$mode-s$seed"
if [ "${SAMPLING_REPLAY_PRINT_ONLY:-0}" != 1 ]; then
    exec 9>"$EXPERIMENT_ROOT/submit.lock"
    flock -n 9 || { echo "Another submission is in progress" >&2; exit 1; }
    python scripts/tools/sampling_replay_ab.py preflight --root "$EXPERIMENT_ROOT" --model "$MODEL_PATH" --dashboard "$RAY_DASHBOARD" --job "$job_id" --run "$run_dir"
fi
if [ "$stage" = smoke ]; then
    data="$EXPERIMENT_ROOT/data/smoke.jsonl"
    eval_data="$EXPERIMENT_ROOT/data/smoke-eval.jsonl"
    rounds=1; prompts=2; samples=8; batch=16; eval_interval=1
    extra=()
else
    data="$EXPERIMENT_ROOT/data/train.jsonl"
    eval_data="$EXPERIMENT_ROOT/data/eval.jsonl"
    rounds=32; prompts=8; samples=8; batch=64; eval_interval=8
    extra=(--save "$run_dir/checkpoints" --save-interval 16)
fi
replay=()
if [ "$mode" = on ]; then replay=(--use-rollout-sampling-replay); fi
# Capture/logprob semantics are the only ON/OFF treatment. Identical sampler and loss settings.
command=(ray job submit --no-wait --submission-id "$job_id" --address "$RAY_DASHBOARD" \
    --runtime-env-json="$RUNTIME_ENV_JSON" -- python -m relax.entrypoints.train \
    --resource '{"actor":[1,1],"rollout":[1,1]}' --max-staleness 0 --num-data-storage-units 1 \
    --hf-checkpoint "$MODEL_PATH" --load "$MODEL_PATH" --megatron-to-hf-mode bridge \
    --swiglu --num-layers 28 --hidden-size 1536 --ffn-hidden-size 8960 \
    --num-attention-heads 12 --group-query-attention --num-query-groups 2 \
    --use-rotary-position-embeddings --disable-bias-linear --add-qkv-bias \
    --normalization RMSNorm --norm-epsilon 1e-6 --rotary-base 1000000 --vocab-size 151936 --kv-channels 128 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --prompt-data "$data" --input-key prompt --label-key label --apply-chat-template \
    --rm-type dapo --reward-key score --num-rollout "$rounds" --rollout-batch-size "$prompts" \
    --n-samples-per-prompt "$samples" --rollout-max-response-len 1024 --rollout-max-context-len 2048 \
    --rollout-temperature 0.8 --rollout-top-k 32 --rollout-top-p 0.9 --rollout-seed "$seed" --rollout-shuffle \
    --sglang-sampling-backend pytorch --sglang-random-seed "$seed" "${replay[@]}" \
    --use-rollout-logprobs --get-mismatch-metrics \
    --custom-tis-function-path relax.backends.megatron.loss.vanilla_tis_function --tis-clip 1 --tis-clip-low 1 \
    --global-batch-size "$batch" --micro-batch-size 1 --advantage-estimator grpo \
    --eps-clip 0.2 --eps-clip-high 0.28 --kl-loss-coef 0 --entropy-coef 0 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.999 \
    --calculate-per-token-loss --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.2 --sglang-context-length 2112 \
    --sglang-enable-fp32-lm-head --sglang-max-running-requests 16 --sglang-disable-cuda-graph \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend flash --no-rope-fusion --seed "$seed" \
    --eval-prompt-data heldout "$eval_data" --eval-interval "$eval_interval" \
    --n-samples-per-eval-prompt 1 --eval-temperature 0 --eval-top-k 1 --eval-top-p 1 \
    --eval-max-response-len 1024 --eval-max-context-len 2048 \
    --custom-megatron-before-train-step-hook-path scripts.training.text.sampling_replay_ab_hooks.before_train \
    --dump-details "$run_dir" --tb-project-name sampling-replay-ab --tb-experiment-name "$job_id" "${extra[@]}")
if [ "${SAMPLING_REPLAY_PRINT_ONLY:-0}" = 1 ]; then
    python -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${command[@]}"
    exit 0
fi
"${command[@]}" > "$run_dir/submit.log" 2>&1
cat "$run_dir/submit.log"
