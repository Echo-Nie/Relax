# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-response SGLang sampling support and behavior probability
boundary."""

import math
from argparse import Namespace
from typing import Any

import numpy as np

from relax.utils.sampling_mask import RolloutSamplingMask
from relax.utils.types import Sample


def validate_sampling_replay_args(args: Namespace) -> None:
    if not getattr(args, "use_rollout_sampling_replay", False):
        return
    if (
        getattr(args, "train_backend", "megatron") != "megatron"
        or getattr(args, "loss_type", "policy_loss") != "policy_loss"
    ):
        raise ValueError("sampling replay requires Megatron policy_loss")
    if getattr(args, "multimodal_keys", None):
        raise ValueError("sampling replay capture currently supports text-only rollout")
    if getattr(args, "use_opd", False):
        raise ValueError("sampling replay capture with OPD is not supported")
    for field in (
        "custom_generate_function_path",
        "custom_convert_samples_to_train_data_path",
        "rollout_external",
        "sglang_config",
        "rollout_engine_class_path",
    ):
        if getattr(args, field, None):
            raise ValueError(f"sampling replay does not support {field}")
    rollout = getattr(args, "rollout_function_path", None)
    standard_rollout = (
        "relax.agentic.rollout.generate_rollout"
        if getattr(args, "use_agentic_rollout", False)
        else "relax.engine.rollout.sglang_rollout.generate_rollout"
    )
    if rollout and rollout != standard_rollout:
        raise ValueError(f"sampling replay requires the standard rollout function: {standard_rollout}")
    if getattr(args, "use_slime_router", False) and "RadixTreeMiddleware" in getattr(
        args, "slime_router_middleware_paths", []
    ):
        raise ValueError("sampling replay does not support radix-tree trajectory rewriting")
    if getattr(args, "sglang_speculative_algorithm", None):
        raise ValueError("sampling replay does not support speculative decoding")
    if getattr(args, "sglang_sampling_backend", None) != "pytorch":
        raise ValueError("sampling replay requires --sglang-sampling-backend pytorch for native support capture")
    temperature = float(args.rollout_temperature)
    if not math.isfinite(temperature) or temperature < 1e-6:
        raise ValueError("sampling replay requires finite rollout_temperature >= 1e-6")
    if not isinstance(args.rollout_top_k, int) or args.rollout_top_k < 1:
        raise ValueError("sampling replay requires finite positive rollout_top_k (SGLang support capture limit)")
    if not 0 < args.rollout_top_p <= 1:
        raise ValueError("sampling replay requires 0 < rollout_top_p <= 1")


def configure_sampling_replay_request(args: Namespace, payload: dict[str, Any]) -> None:
    validate_sampling_replay_args(args)
    sampling = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        **payload["sampling_params"],
    }
    if sampling.get("temperature", args.rollout_temperature) != args.rollout_temperature:
        raise ValueError("sampling replay requires the same rollout_temperature on every request")
    if (
        sampling.get("top_k", args.rollout_top_k) != args.rollout_top_k
        or sampling.get("top_p", args.rollout_top_p) != args.rollout_top_p
    ):
        raise ValueError("sampling replay request top-p/top-k must match rollout configuration")
    if sampling.get("no_stop_trim", True) is not True:
        raise ValueError("sampling replay requires no_stop_trim to preserve token/support alignment")
    for key, default in (("frequency_penalty", 0), ("presence_penalty", 0), ("repetition_penalty", 1)):
        if sampling.get(key, default) != default:
            raise ValueError(f"sampling replay does not support {key}: actor logits would differ")
        sampling[key] = default
    if payload.get("custom_logit_processor") or sampling.get("logit_bias"):
        raise ValueError("sampling replay does not support custom logits processors or logit_bias")
    payload["return_logprob"] = True
    payload["return_sampling_mask"] = True
    payload["sampling_logprobs_mode"] = "support"
    payload["sampling_params"] = {**sampling, "no_stop_trim": True}


def sampling_replay_prefix(sample: Sample) -> RolloutSamplingMask:
    mask = sample.rollout_sampling_mask
    if mask is None:
        if sample.response_length:
            raise ValueError("sampling replay continuation is missing its prefix support")
        mask = RolloutSamplingMask.from_rows([])
    if len(mask) != sample.response_length or len(sample.rollout_log_probs or []) != sample.response_length:
        raise ValueError("sampling replay prefix support/logprobs must match response_length")
    return mask


def parse_sampling_replay_response(
    output: dict[str, Any], *, top_k: int
) -> tuple[list[int], list[float], RolloutSamplingMask]:
    meta = output["meta_info"]
    pairs = meta.get("output_token_logprobs")
    rows = meta.get("output_token_sampling_mask")
    probabilities = meta.get("output_token_sampling_logprobs")
    if not all(isinstance(values, list) for values in (pairs, rows, probabilities)):
        raise ValueError(
            f"sampling replay requires native support/logprobs for every token: {meta.get('finish_reason')}"
        )
    if any(not isinstance(pair, (list, tuple)) or len(pair) < 2 for pair in pairs):
        raise ValueError("sampling replay received malformed output_token_logprobs")
    tokens = [pair[1] for pair in pairs]
    n = len(tokens)
    if output.get("output_ids", tokens) != tokens or meta.get("completion_tokens", n) != n:
        raise ValueError("sampling replay output tokens and completion count disagree")
    if len(rows) != n or len(probabilities) != n or meta.get("output_token_sampling_mask_length", n) != n:
        raise ValueError("sampling replay support rows must match generated tokens")
    sampled_logprobs = []
    for token, row, values in zip(tokens, rows, probabilities, strict=True):
        ids, logprobs = np.asarray(row), np.asarray(values, dtype=np.float64)
        if ids.ndim != 1 or not 0 < len(ids) <= top_k or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("sampling replay requires nonempty integer support bounded by top_k")
        if (ids < 0).any() or (ids > np.iinfo(np.int32).max).any() or len(np.unique(ids)) != len(ids):
            raise ValueError("sampling replay support IDs must be unique nonnegative int32")
        if type(token) is not int or (matches := np.flatnonzero(ids == token)).size != 1:
            raise ValueError("sampling replay sampled token is absent from support")
        if logprobs.shape != ids.shape or not np.isfinite(logprobs).all() or (logprobs > 1e-6).any():
            raise ValueError("sampling replay requires per-support finite behavior logprobs")
        if not np.isclose(np.exp(logprobs).sum(), 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("sampling replay behavior probabilities must normalize on saved support")
        sampled_logprobs.append(float(logprobs[matches[0]]))
    return tokens, sampled_logprobs, RolloutSamplingMask.from_rows(rows)
