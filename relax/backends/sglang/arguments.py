# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse

from sglang.srt.server_args import ServerArgs

from relax.utils.http_utils import _wrap_ipv6
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _router_passthrough_skip_fields() -> set[str]:
    return {
        # Framework-managed addressing and topology.
        "host",
        "port",
        "worker_urls",
        "prefill",
        "decode",
        "prefill_urls",
        "decode_urls",
        "pd_disaggregation",
        "service_discovery",
        "selector",
        "service_discovery_port",
        "service_discovery_namespace",
        "prefill_selector",
        "decode_selector",
        "bootstrap_port_annotation",
        "prometheus_port",
        # Backward-compatible explicit flags handled below.
        "policy",
        "request_timeout_secs",
    }


def _add_prefixed_router_args(parser) -> None:
    from sglang_router.router_args import RouterArgs

    old_add_argument = argparse._ActionsContainer.add_argument
    skipped_args = _router_passthrough_skip_fields()

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        canonical_name = kwargs.get("dest")
        if not canonical_name:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    canonical_name = flag_name_candidate[2:].replace("-", "_")
                    break

        if canonical_name in skipped_args:
            return

        final_name_or_flags = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("--"):
                final_name_or_flags.append(f"--sglang-router-{item_flag[2:]}")
            else:
                final_name_or_flags.append(item_flag)

        final_kwargs = kwargs.copy()
        if canonical_name and not str(canonical_name).startswith("router_"):
            final_kwargs["dest"] = f"router_{canonical_name}"

        old_add_argument(*final_name_or_flags, **final_kwargs)

    argparse._ActionsContainer.add_argument = new_add_argument_wrapper
    try:
        RouterArgs.add_cli_args(parser, use_router_prefix=False, exclude_host_port=False)
    finally:
        argparse._ActionsContainer.add_argument = old_add_argument


def add_sglang_router_arguments(parser):
    """Add arguments to the parser for the SGLang router."""
    parser.add_argument(
        "--sglang-router-ip",
        type=str,
        default=None,
        help="IP address of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-port",
        type=int,
        default=None,
        help="Port of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-policy",
        type=str,
        default=None,
        help="Routing policy for the SGLang router (e.g., 'consistent_hashing', 'round_robin')",
    )
    parser.add_argument(
        "--sglang-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the SGLang router in seconds",
    )
    _add_prefixed_router_args(parser)
    return parser


def add_sglang_arguments(parser):
    """Add arguments to the parser for the SGLang server."""
    parser = add_sglang_router_arguments(parser)
    parser.set_defaults(router_balance_abs_threshold=10, router_balance_rel_threshold=1.2)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)

    # SGLang profiling arguments — triggers /start_profile and /stop_profile HTTP API
    # on all SGLang engines during rollout inference.
    # Can also be used standalone via: python tools/profile_rollout.py
    parser.add_argument(
        "--sglang-profile",
        action="store_true",
        default=False,
        help="Enable torch profiling on SGLang engines during rollout. Profile traces will be saved per rollout step.",
    )
    parser.add_argument(
        "--sglang-profile-output-dir",
        type=str,
        default=None,
        help=("Output directory for SGLang profile traces. Defaults to traces/<tb_experiment_name>/sglang_trace."),
    )
    parser.add_argument(
        "--sglang-profile-num-steps",
        type=int,
        default=3,
        help="Number of SGLang forward steps to profile per rollout. "
        "If -1, profiles the entire rollout step until stop_profile is called.",
    )
    parser.add_argument(
        "--sglang-profile-activities",
        type=str,
        nargs="+",
        default=["CPU", "GPU"],
        help="Activities to profile (e.g., CPU GPU).",
    )
    parser.add_argument(
        "--sglang-profile-by-stage",
        action="store_true",
        default=False,
        help="Profile by stage (prefill/decode) separately.",
    )
    parser.add_argument(
        "--sglang-profile-with-stack",
        action="store_true",
        default=False,
        help="Record call stack in profile traces.",
    )
    parser.add_argument(
        "--sglang-profile-record-shapes",
        action="store_true",
        default=False,
        help="Record tensor shapes in profile traces.",
    )
    parser.add_argument(
        "--sglang-profile-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "List of absolute rollout step IDs (0-indexed) at which to enable SGLang profiling. "
            "Takes precedence over --sglang-profile-step-start/end when set. "
            "Example: --sglang-profile-steps 3 10 50"
        ),
    )
    parser.add_argument(
        "--sglang-profile-step-start",
        type=int,
        default=None,
        help=(
            "Start of the rollout step range for SGLang profiling (inclusive, 0-indexed). "
            "Used together with --sglang-profile-step-end to specify a contiguous range. "
            "Ignored if --sglang-profile-steps is set."
        ),
    )
    parser.add_argument(
        "--sglang-profile-step-end",
        type=int,
        default=None,
        help=(
            "End of the rollout step range for SGLang profiling (inclusive, 0-indexed). "
            "Used together with --sglang-profile-step-start to specify a contiguous range. "
            "Ignored if --sglang-profile-steps is set. "
            "Example: --sglang-profile-step-start 2 --sglang-profile-step-end 4 profiles steps 2, 3, 4."
        ),
    )

    old_add_argument = parser.add_argument

    skipped_args = [
        "model_path",
        "config",
        "trust_remote_code",
        "random_seed",
        # memory
        "enable_memory_saver",
        # distributed
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
        "enable_return_routed_experts",
    ]

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        """Add arguments to the parser, ensuring that the server arguments are
        prefixed and skippable."""
        # Determine the canonical name for skip check (e.g., "model_path")
        canonical_name_for_skip_check = None
        if "dest" in kwargs:
            canonical_name_for_skip_check = kwargs["dest"]
        else:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    # Derive from first long flag: --foo-bar -> foo_bar
                    stem = flag_name_candidate[2:]
                    canonical_name_for_skip_check = stem.replace("-", "_")
                    break
            # If no long flag and no dest, skip logic might not catch it unless short flags imply a dest.

        if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
            return  # Skip this entire argument definition

        # If not skipped, proceed to prefix flags and dest
        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")  # "foo-bar" from "--foo-bar", or "f" from "-f"
                prefixed_item = f"--sglang-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                # Positional arguments or non-string items
                new_name_or_flags_list.append(item_flag)

        # Prepare kwargs for the actual add_argument call.
        # Make a copy to avoid modifying the original kwargs dict.
        final_kwargs = kwargs.copy()

        # If 'dest' is explicitly provided and is a string, prefix it.
        # This ensures the attribute on the args namespace becomes, e.g., args.sglang_dest_name.
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            # Avoid double prefixing if dest somehow already starts with sglang_
            if not original_dest.startswith("sglang_"):
                final_kwargs["dest"] = f"sglang_{original_dest}"
        # If 'dest' is not explicitly provided (or is None/not a string),
        # argparse will derive 'dest' from the (now prefixed) flag names.
        # E.g., if the first flag is "--sglang-foo-bar", argparse sets dest to "sglang_foo_bar".

        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument

    # PD disaggregation / multi-group config
    parser.add_argument(
        "--prefill-num-servers",
        type=int,
        default=None,
        help="Number of prefill servers for disaggregation.",
    )
    parser.add_argument(
        "--sglang-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for SGLang engine deployment. "
            "Defines engine_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict of "
            "ServerArgs field names that override the base --sglang-* CLI args. "
            "Placeholder groups reserve GPU slots without creating engines. "
            "Mutually exclusive with --prefill-num-servers."
        ),
    )
    parser.add_argument(
        "--sglang-external-model-package",
        type=str,
        default=None,
        help=(
            "Python package containing external SGLang model and processor for "
            "registration via SGLANG_EXTERNAL_MODEL_PACKAGE env var. "
            "The package should contain modules with EntryClass (model) and/or "
            "BaseMultimodalProcessor subclasses (processor). "
            "The multimodal architecture name is auto-derived from EntryClass. "
            "Example: relax.models.dots_ocr.sglang"
        ),
    )

    return parser


def validate_args(args):
    # Older SGLang versions stored these CLI aliases under their long names,
    # while newer versions use the short ServerArgs field names as argparse dests.
    # Keep both attributes available for user code, preferring the newer names
    # when a namespace happens to contain both.
    for current_name, legacy_name in (
        ("sglang_dp_size", "sglang_data_parallel_size"),
        ("sglang_pp_size", "sglang_pipeline_parallel_size"),
        ("sglang_ep_size", "sglang_expert_parallel_size"),
    ):
        value = getattr(args, current_name) if hasattr(args, current_name) else getattr(args, legacy_name)
        setattr(args, current_name, value)
        setattr(args, legacy_name, value)

    # Compute effective TP size considering PP size
    if args.sglang_pp_size > 1:
        assert args.rollout_num_gpus_per_engine % args.sglang_pp_size == 0, (
            f"rollout_num_gpus_per_engine ({args.rollout_num_gpus_per_engine}) must be divisible by "
            f"sglang_pipeline_parallel_size ({args.sglang_pp_size})"
        )
        args.sglang_tp_size = args.rollout_num_gpus_per_engine // args.sglang_pp_size
    else:
        args.sglang_tp_size = args.rollout_num_gpus_per_engine

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    if args.sglang_enable_dp_attention and not args.router_dp_aware:
        logger.warning(
            "sglang_enable_dp_attention=True requires router_dp_aware=True; "
            "overriding router_dp_aware=False with True."
        )
        args.router_dp_aware = True

    if getattr(args, "sglang_router_ip", None):
        args.sglang_router_ip = _wrap_ipv6(args.sglang_router_ip)

    # Mutual-exclusion checks for PD disaggregation / sglang-config.
    assert not (getattr(args, "prefill_num_servers", None) is not None and args.rollout_external), (
        "prefill_num_servers cannot be set when rollout_external is set."
    )

    assert not (getattr(args, "sglang_config", None) is not None and args.rollout_external), (
        "sglang_config cannot be set when rollout_external is set."
    )

    assert not (
        getattr(args, "sglang_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
    ), "sglang_config and prefill_num_servers are mutually exclusive. Use engine_groups in the YAML config instead."

    _validate_encoder_disaggregation(args)


def _validate_encoder_disaggregation(args) -> None:
    """Validate an ``worker_type: encoder`` topology from ``--sglang-config``.

    Encoder disaggregation moves the vision tower onto its own SGLang
    ``--encoder-only`` engines.  That server has no ``/update_weights_from_*``,
    no ``/release_memory_occupation`` and no cache-invalidation route, and its
    embedding cache is keyed purely by image content, so every constraint below
    exists to keep a *silently wrong* configuration from starting.
    """
    if getattr(args, "sglang_config", None) is None:
        return

    from relax.distributed.ray.rollout import SglangConfig

    config = SglangConfig.from_yaml(args.sglang_config)
    if not config.has_encoder_disaggregation:
        return

    for model in config.models:
        if not model.has_encoder_disaggregation:
            continue
        if not model.encoder_consumer_groups:
            raise ValueError(
                f"sglang_config model '{model.name}' has an 'encoder' group but no 'regular'/'prefill' group "
                "to consume its embeddings. An encoder alone serves no traffic."
            )
        encoder_engine_count = 0
        for group in model.encoder_groups:
            per_engine = group.num_gpus_per_engine or model.num_gpus_per_engine or args.rollout_num_gpus_per_engine
            encoder_engine_count += group.num_gpus // min(per_engine, args.num_gpus_per_node)
        if encoder_engine_count != 1:
            raise ValueError(
                f"sglang_config model '{model.name}' declares {encoder_engine_count} encoder engines; exactly 1 "
                "is supported. SGLang spreads multimodal items over encoders with a random shuffle "
                "(srt/disaggregation/encode_receiver.py::_assign_items_by_modality), which destroys the "
                "--enable-prefix-mm-cache hit rate that both EPD and the cross-step prefetch depend on."
            )

    # The encoder is loaded once from the checkpoint and never weight-synced, so
    # a trainable vision tower would make it drift from the policy after the
    # very first update -- and, because a language_only server still keeps its
    # own ViT for local fallback, dispatched and fallback requests would then
    # disagree on the same policy version.
    if not getattr(args, "freeze_vision_model", False):
        raise ValueError(
            "Encoder disaggregation (worker_type: encoder) currently requires --freeze-vision-model. "
            "The SGLang --encoder-only server exposes no weight-update endpoint, so its vision tower "
            "cannot be re-synced after an optimizer step; without freezing, the encoder would serve "
            "embeddings from a stale ViT while the language engines hold a fresh one. "
            "Pass --freeze-vision-model, or drop the encoder group."
        )

    # The OPD pre-expanded patch rewrites QwenVLImageProcessor.process_mm_data_async,
    # but a language_only engine bypasses that function entirely for dispatched
    # requests (embeddings come from the encoder, input_ids are rebuilt receiver
    # side).  Ordinary base64 rollout requests are unaffected -- they never carry
    # the opd_preexpanded_raw marker -- but the student-at-teacher-topk prefill does.
    if getattr(args, "use_opd", False) and getattr(args, "opd_log_prob_top_k", None):
        raise ValueError(
            "Encoder disaggregation is incompatible with OPD top-k student prefill "
            "(--opd-log-prob-top-k). Those requests carry image_data with "
            "format='opd_preexpanded_raw', which a language_only engine dispatches to the encoder, "
            "and the encoder does not understand that format. Disable one of the two."
        )


def sglang_parse_args():
    """Parse sglang server arguments independently using a separate
    ArgumentParser. Uses parse_known_args() to only consume sglang-related
    arguments from sys.argv, allowing the remaining arguments to be parsed by
    megatron separately.

    Returns:
        argparse.Namespace: Parsed sglang arguments (all attributes prefixed with sglang_).
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_sglang_arguments(parser)

    # Compute default sglang_tensor_parallel_size from CLI args
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    temp_parser.add_argument("--sglang-pp-size", type=int, default=1)
    temp_parser.add_argument("--sglang-pipeline-parallel-size", type=int, default=1)
    temp_args, _ = temp_parser.parse_known_args()
    pp_size = temp_args.sglang_pp_size if temp_args.sglang_pp_size != 1 else temp_args.sglang_pipeline_parallel_size
    sglang_tp_size = temp_args.rollout_num_gpus_per_engine // pp_size
    parser.set_defaults(sglang_tensor_parallel_size=sglang_tp_size)

    args, _ = parser.parse_known_args()
    return args
