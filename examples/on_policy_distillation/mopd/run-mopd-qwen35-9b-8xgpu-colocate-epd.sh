#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# EPD variant of run-mopd-qwen35-9b-8xgpu-colocate.sh: the student's vision
# tower is split onto a dedicated GPU (SGLang --encoder-only), and the encoder
# additionally pre-computes the NEXT step's image embeddings while the current
# step is still running.
#
# Keep the non-EPD script as the A/B baseline; only the deltas below differ.
#
#   student : Qwen3.5-9B, actor(8) + rollout(5 = encoder 1 + LM 4) colocate
#   encoder : 1 GPU, TP1, vision tower only (~0.85 GiB), GPU-resident
#   teacher : 3 replicas x TP1 (was 2 x TP2) -- this is what frees the encoder GPU
#
# Constraint (enforced, unchanged): rollout_gpus + teacher_gpus == actor_gpus
#                                   (5 + 3 == 8), see relax/utils/opd/opd_utils.py.
# GPU map: see examples/on_policy_distillation/mopd/sglang-epd.yaml.
# Design notes: docs/draft/encoder-disaggregation-and-overlap-prefetch-plan.md
#
# Usage:
#   bash run-mopd-qwen35-9b-8xgpu-colocate-epd.sh

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export RELAX_OPD_PREEXPANDED_PATCH=1
# The OPD teacher needs a logprob at every response position, and
# logits_processor.py:471 materializes [N, vocab] (fp32) in one shot by default.
# N reaches tens of thousands, i.e. an 8-16 GiB single allocation -- exactly where
# the teacher OOM'd. Enabling this routes through
# process_input_logprobs_by_chunk(), capping the peak at
# SGLANG_LOGITS_PROCESSER_CHUNK_SIZE (default 2048, ~1 GiB at TP=2) and thereby
# decoupling logits memory from the prefill chunk size.
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
# EPD delta: the teacher drops from TP2 to TP1, so the logits gather is over the
# FULL vocab (248320) instead of half of it. 8192 x 248320 x 4B = 7.58 GiB vs
# 3.79 GiB at TP2 -- halve the chunk to hold the same peak.
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=4096
# ViT-output (embedding) cache. The default cap is 100 MiB (environ.py:701,
# injected at kv_cache_builder.py:248), but one image at --image-max-token-num
# 8192 is 8192 x 4096 x 2B = 64 MiB (measured mean ~39 MiB) and mmlongbench rows
# carry 2.8-3.6 images, so a single request evicts the whole cache and the LRU
# thrashes non-stop. 1024 MiB holds ~16-26 images, i.e. one in-flight wave.
#
# NOTE: this value must stay at 1024 here. On the language and teacher engines
# the cache holds GPU tensors in a module global that nothing frees on sleep, so
# raising it globally would withhold that much memory per card from the actor.
# The encoder's own (host-RAM) cache is sized separately, per-group, in
# sglang-epd.yaml.
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-1024}"
# Cross-step encoder prefetch. Set to 0 for the EPD-without-prefetch A/B arm.
export RELAX_ENCODER_PREFETCH="${RELAX_ENCODER_PREFETCH:-1}"
# Disable the EncoderBootstrapServer's periodic encoder eviction. That probe
# runs inside every *language_only* tokenizer manager (not in the encoder),
# allows only 2s, and drops an encoder after 3 consecutive misses. Under a real
# rollout wave the saturated encoder misses those probes, gets evicted, and
# every subsequent request silently falls back to the language engine's local
# ViT -- which is what killed the 15:30 run. We configure exactly one encoder
# statically via --encoder-urls, so eviction has nothing to fail over to and can
# only cause silent degradation.
# Trade-off: with the probe off, a genuinely dead encoder is no longer detected,
# and requests would block for SGLANG_ENCODER_RECV_TIMEOUT (180s) instead. For
# an experiment, failing loudly is preferable to silently un-disaggregating.
export SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL="${SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL:-0}"
# Step 0 always faces a COLD encoder cache: there is no step -1 to prefetch
# during, so all of that step's unique images need a ViT forward inside the
# rollout burst. The 180s default made step 0 time out ("Timed out waiting for
# image embeddings", 121x) before the cross-step prefetch could ever prove
# itself. Let the cold step run long instead, so step 0 (cold) vs step 1+ (warm,
# prefetched) isolates how much of the cost is actually the vision tower --
# the prefetch can only remove ViT, never the per-request HF preprocessing
# (encode_server runs it before the cache lookup) nor the embedding transfer.
export SGLANG_ENCODER_RECV_TIMEOUT="${SGLANG_ENCODER_RECV_TIMEOUT:-1800}"
# 🔴 Mandatory with --sglang-encoder-transfer-backend mooncake. Mooncake's
# TransferEngine picks its own P2P-handshake/RPC port, and its default range
# starts at 15000 -- the same base as Relax's engine-port allocator
# (find_available_port(15000)). Measured default pick: 15303.
#
# Relax hands out a port *number* and only binds it later, when sglang's uvicorn
# starts. Mooncake initialises inside the scheduler subprocess during that
# window, so it can and does take the engine's assigned HTTP port. When it wins,
# _wait_server_healthy polls mooncake's handshake socket forever: the encoder
# logs "readString: too large length from socket: 7018130145941931335" every 2s
# -- that integer is the ASCII bytes "GET /hea" read as a length prefix --
# SGLangEngine.init never returns, and the run hangs at
# "[engine-init-barrier:actor] ready N-1/N" with the engine answering /health
# on some *other* port the whole time. Seen 2026-09-03: rank 0 was assigned
# 15364 and mooncake bound it. It is a race, so it hits a different engine (or
# none) each run -- which is exactly why it must be pinned rather than retried.
#
# Both spellings are set deliberately: the mooncake binary exports MC_*_RPC_PORT
# and MC_*_PRC_PORT, and each was verified to move the port on its own.
export MC_MIN_RPC_PORT="${MC_MIN_RPC_PORT:-19000}"
export MC_MAX_RPC_PORT="${MC_MAX_RPC_PORT:-19999}"
export MC_MIN_PRC_PORT="${MC_MIN_PRC_PORT:-19000}"
export MC_MAX_PRC_PORT="${MC_MAX_PRC_PORT:-19999}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}RELAX_OPD_PREEXPANDED_PATCH,SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK,SGLANG_LOGITS_PROCESSER_CHUNK_SIZE,SGLANG_VLM_CACHE_SIZE_MB,RELAX_ENCODER_PREFETCH,SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL,SGLANG_ENCODER_RECV_TIMEOUT,MC_MIN_RPC_PORT,MC_MAX_RPC_PORT,MC_MIN_PRC_PORT,MC_MAX_PRC_PORT"

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/mopd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mopd-qwen35-9b-8xgpu-epd-${now}}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-/mnt/tidalfs-hssh01/dataset/sunhuo/models}"
DATA_DIR="${DATA_DIR:-/mnt/tidalfs-hssh01/dataset/sunhuo/datasets}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-9B}"
TEXT_TEACHER_MODEL_NAME="${TEXT_TEACHER_MODEL_NAME:-Qwen3.5-9B}"
VL_TEACHER_MODEL_NAME="${VL_TEACHER_MODEL_NAME:-Qwen3.5-9B}"
PROMPT_SET="${PROMPT_SET:-/mnt/tidal-alsh-share2/dataset/galapagos/ouwen/datasets/opensource/swift-vqa-distill-mmlongbench-quarantine/train.parquet}"
EVAL_SET="${EVAL_SET:-${DATA_DIR}/MOPD-vlonly/test_small.parquet}"

ACTOR_GPUS="${ACTOR_GPUS:-8}"
# EPD delta: the student LM keeps 4 GPUs, the encoder takes 1, and the teacher
# gives that GPU up by moving from 2 replicas x TP2 to 3 replicas x TP1.
# Replica count is throughput-neutral for a compute-bound teacher, and 3 spreads
# GRPO groups better across the group-affinity router (_pick_replica).
ENCODER_GPUS="${ENCODER_GPUS:-1}"
ROLLOUT_LM_GPUS="${ROLLOUT_LM_GPUS:-4}"
ROLLOUT_GPUS=$((ROLLOUT_LM_GPUS + ENCODER_GPUS))
TEACHER_GPUS="${TEACHER_GPUS:-3}"
TEACHER_NUM_GPUS_PER_ENGINE="${TEACHER_NUM_GPUS_PER_ENGINE:-1}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${STUDENT_MODEL_NAME}/"
   --megatron-to-hf-mode bridge
   # --save "${EXP_DIR}/save/mopd-${STUDENT_MODEL_NAME}/"
   # --save-interval 200
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --apply-chat-template

   --multimodal-keys '{"image":"images"}'
   # The default is 16384 per image and mmlongbench rows carry up to 5 images, so a
   # single sample reaches ~82K tokens. On the training side the full-vocab logits
   # are seq x 248320 x 4B / (TP4 x CP2) = 9.45 GiB, which OOMs -- and TP x CP
   # already spans all 8 GPUs, so no further sharding is available. 8192 caps the
   # worst sample at ~49K tokens (logits 5.74 GiB) and roughly halves the rollout
   # and teacher prefill volume as well.
   --image-max-token-num 8192

   --rm-type mopd
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256

   --log-passrate
   # --use-fault-tolerance
   --use-streaming-dataset
   # NOTE: deliberately NOT passing --rollout-shuffle, even though without it the
   # StreamingDataset prefetch thread never starts (set_index_order is only
   # reached via shuffle()) and every sample takes the serial synchronous path.
   # The 2026-09-01 baseline ran without it, and turning it on here would change
   # the data ORDER -- which that baseline identifies as the sole driver of
   # per-step time variance (16.4m..34.4m at a flat 24.1m mean). Enabling it in
   # only one arm would make multimodal_token_count differ step-by-step and
   # render the EPD comparison uninterpretable. Worth its own A/B, separately.
)

# PROMPT_SET here is 100% data_source=mmlongbench, so a table keyed on
# dapo-math-17k / multimodal-open-r1 raises KeyError for every sample
# (_pick_teacher_url, on_policy_distillation.py:127). Route that one key at the VL
# teacher, matching the 2026-08-21 runs. Restore the two-key table when switching
# back to the merged dapo-math-17k + multimodal-open-r1 prompt set.
TEACHER_ROUTES="{\"mmlongbench\":\"${MODEL_DIR}/${VL_TEACHER_MODEL_NAME}/\"}"

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-only-reward
   --opd-kl-coef 1.0
   --opd-teacher-key data_source
   --opd-teacher-routes "${TEACHER_ROUTES}"
   --teacher-num-gpus-per-engine "${TEACHER_NUM_GPUS_PER_ENGINE}"
   # The teacher is prefill-only, so its peak KV equals one in-flight request
   # (45K tokens is ~1.4 GiB; the largest 5-image rows ~2.6 GiB) -- measured
   # full token usage stays at 0.00-0.02. At TP1 the 9B weights take ~18.5 GiB of
   # the 0.4 x 97.8 = 39.1 GiB static budget, leaving a ~20 GiB KV pool: still
   # ~8x the demand, with ~58 GiB outside the budget for the logits gather peak.
   --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.4}"
   # A single multi-image request (~45K mm tokens) far exceeds the per-pass chunk
   # budget, so PrefillAdder admits exactly one sequence per iteration: the
   # chunked branch at schedule_policy.py:1101 drains rem_chunk_tokens and breaks
   # out immediately. 4096 is too small -- ~11 passes per request, each too small
   # a matmul for a 9B. 16384 is merely the max_prefill_tokens default; raising
   # both to 32768 cuts it to ~2 passes. This relies on
   # SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK above having decoupled the logits peak
   # from the chunk size.
   --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-32768}"
   --teacher-sglang-max-prefill-tokens "${TEACHER_MAX_PREFILL_TOKENS:-32768}"
   --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-16}"
   --teacher-sglang-disable-cuda-graph
   --opd-token-selection student_sampled
   --opd-log-prob-min-clamp -10.0
   # A multi-image teacher logprob re-prefills the whole multi-image context
   # (~45K tokens per request); 300s times out on the slow tail and
   # _raise_if_all_failed turns that into a hard error.
   --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-1200}"
   --opd-teacher-image-key images
   --use-rollout-logprobs
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-distributed-optimizer
   --accumulate-allreduce-grads-in-fp32
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXPERIMENT_NAME}"
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-prompt-data mopd-9b "${EVAL_SET}"
   --eval-global-batch-size 128
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

EPD_ARGS=(
   # Mandatory, and validated at startup: SGLang's --encoder-only server
   # has no weight-update endpoint, so the encoder's vision tower cannot be
   # re-synced after an optimizer step. With a trainable ViT the encoder would drift
   # from the policy after the very first update -- and because a language_only
   # engine still keeps its own ViT for local fallback, dispatched and fallback
   # requests would then disagree within the same policy version. Freezing also
   # makes the cross-step prefetch exact rather than one step stale.
   --freeze-vision-model
   # EPD topology: encoder group first (bundle 0), then the language engines.
   --sglang-config "${SCRIPT_DIR}/sglang-epd.yaml"
   # Encoder->language transport. Global on purpose: both sides must agree.
   #   zmq_to_scheduler (sglang default) -- encoder D2H copy, then pickle+TCP
   #     once per language TP rank, receiver does frombuffer+clone+H2D.
   #   mooncake -- the embedding stays on the GPU and is RDMA'd straight into
   #     the receiver's pre-registered GPU buffer. Needs IB (7x mlx5 here) plus
   #     nvidia_peermem, both present; no metadata server for plain transport
   #     (that is only required by --enable-mm-global-cache).
   # Upstream's mooncake path ignores the embedding cache entirely; mm_cache was
   # ported into encode_with_mooncake so caching survives the switch (verified
   # 20/20 hits by scripts/debug/epd_minimal_repro.sh).
   --sglang-encoder-transfer-backend "${ENCODER_TRANSFER_BACKEND:-mooncake}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 40960
)

SGLANG_ARGS=(
   # Kept even though sglang-epd.yaml sets TP per group: this value also feeds
   # the shared HTTP client concurrency (http_utils.py) and the weight-sync
   # engine_gpu_counts default, and would silently degrade to 1 if removed.
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   # The student is serialised the same way (the 8192 default is also far below a
   # single request's 45K mm tokens).
   --sglang-chunked-prefill-size 16384
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
)

RESOURCE_JSON="{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}"

python3 -m relax.entrypoints.train \
    --resource "${RESOURCE_JSON}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --offload \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${EPD_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "mopd-qwen35-9b-epd-${now}.log"
