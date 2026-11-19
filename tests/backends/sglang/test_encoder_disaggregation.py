# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for encoder disaggregation (EPD) and cross-step encoder
prefetch."""

import argparse
import asyncio
import dataclasses
import textwrap

import pytest
from sglang.srt.server_args import ServerArgs

from relax.backends.sglang.arguments import _validate_encoder_disaggregation
from relax.backends.sglang.encode_cache_only_patch import (
    PREFETCH_RID_PREFIX,
    is_prefetch_request,
)
from relax.backends.sglang.sglang_engine import _compute_server_args
from relax.distributed.ray.rollout import (
    EngineGroup,
    EngineGroupConfig,
    RolloutServer,
    SglangConfig,
)
from relax.engine.rollout.encoder_prefetch import (
    EncoderPrefetcher,
    _iter_group_images,
    is_encoder_prefetch_enabled,
)


EPD_YAML = textwrap.dedent(
    """
    sglang:
      - name: actor
        engine_groups:
          - worker_type: encoder
            num_gpus: 1
            num_gpus_per_engine: 1
            overrides:
              load_format: auto
            env:
              SGLANG_VLM_CACHE_SIZE_MB: "16384"
          - worker_type: regular
            num_gpus: 4
            num_gpus_per_engine: 2
    """
)


def _write_yaml(tmp_path, body: str) -> str:
    path = tmp_path / "sglang.yaml"
    path.write_text(body)
    return str(path)


def _validate_args(config_path, **overrides):
    base = dict(
        sglang_config=config_path,
        freeze_vision_model=True,
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        use_opd=False,
        opd_log_prob_top_k=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ----------------------------------------------------------------- config ---


def test_sglang_config_parses_encoder_group(tmp_path):
    config = SglangConfig.from_yaml(_write_yaml(tmp_path, EPD_YAML))
    model = config.models[0]

    assert config.has_encoder_disaggregation
    assert not config.has_pd_disaggregation
    # Encoder GPUs count toward the total, so --rollout-num-gpus must include them.
    assert config.total_num_gpus == 5
    assert [g.worker_type for g in model.encoder_groups] == ["encoder"]
    assert [g.worker_type for g in model.encoder_consumer_groups] == ["regular"]
    assert model.encoder_groups[0].env["SGLANG_VLM_CACHE_SIZE_MB"] == "16384"


def test_engine_group_config_rejects_unknown_worker_type():
    with pytest.raises(AssertionError, match="Invalid worker_type"):
        EngineGroupConfig(worker_type="vision", num_gpus=1)


# ------------------------------------------------------------- validation ---


def test_validate_encoder_disaggregation_accepts_valid_topology(tmp_path):
    _validate_encoder_disaggregation(_validate_args(_write_yaml(tmp_path, EPD_YAML)))


def test_validate_encoder_disaggregation_requires_frozen_vision_model(tmp_path):
    args = _validate_args(_write_yaml(tmp_path, EPD_YAML), freeze_vision_model=False)
    with pytest.raises(ValueError, match="requires --freeze-vision-model"):
        _validate_encoder_disaggregation(args)


def test_validate_encoder_disaggregation_rejects_opd_topk(tmp_path):
    args = _validate_args(_write_yaml(tmp_path, EPD_YAML), use_opd=True, opd_log_prob_top_k=20)
    with pytest.raises(ValueError, match="opd-log-prob-top-k"):
        _validate_encoder_disaggregation(args)


def test_validate_encoder_disaggregation_requires_a_consumer(tmp_path):
    body = textwrap.dedent(
        """
        sglang:
          - name: actor
            engine_groups:
              - worker_type: encoder
                num_gpus: 1
                num_gpus_per_engine: 1
        """
    )
    with pytest.raises(ValueError, match="no 'regular'/'prefill' group"):
        _validate_encoder_disaggregation(_validate_args(_write_yaml(tmp_path, body)))


def test_validate_encoder_disaggregation_rejects_multiple_encoders(tmp_path):
    """SGLang spreads mm items over encoders with a random shuffle, which
    destroys the embedding-cache hit rate the prefetch depends on."""
    body = textwrap.dedent(
        """
        sglang:
          - name: actor
            engine_groups:
              - worker_type: encoder
                num_gpus: 2
                num_gpus_per_engine: 1
              - worker_type: regular
                num_gpus: 3
                num_gpus_per_engine: 1
        """
    )
    with pytest.raises(ValueError, match="2 encoder engines; exactly 1"):
        _validate_encoder_disaggregation(_validate_args(_write_yaml(tmp_path, body)))


def test_validate_encoder_disaggregation_noop_without_config():
    _validate_encoder_disaggregation(_validate_args(None))


# ------------------------------------------------------------ server args ---


def _server_args_namespace(**overrides):
    values = {f"sglang_{f.name}": f.default for f in dataclasses.fields(ServerArgs)}
    values.update(
        hf_checkpoint="/tmp/model",
        sglang_hf_checkpoint=None,
        seed=42,
        rollout_num_gpus_per_engine=2,
        rollout_num_gpus=5,
        num_gpus_per_node=8,
        offload_rollout=True,
        use_rollout_routing_replay=False,
        fp16=False,
        lora_rank=0,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        colocate=True,
        sglang_load_format="dummy",
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        sglang_enable_draft_weights_cpu_backup=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _compute(worker_type, num_gpus_per_engine, sglang_overrides=None):
    kwargs, _ = _compute_server_args(
        _server_args_namespace(),
        rank=0,
        dist_init_addr="127.0.0.1:20000",
        nccl_port=20001,
        host="127.0.0.1",
        port=30000,
        base_gpu_id=0,
        worker_type=worker_type,
        num_gpus_per_engine=num_gpus_per_engine,
        sglang_overrides=sglang_overrides,
    )
    return kwargs


def test_compute_server_args_encoder_overrides_dummy_load_format():
    """The encoder never receives a weight sync, so it must not load dummy
    weights.

    Regression guard: a dummy-loaded encoder serves embeddings from a randomly
    initialised vision tower and reports no error anywhere.
    """
    kwargs = _compute("encoder", 1)

    assert kwargs["load_format"] == "auto"
    assert kwargs["encoder_only"] is True
    assert kwargs["enable_prefix_mm_cache"] is True
    assert kwargs["disaggregation_mode"] == "null"
    # encode_server has no torch_memory_saver integration.
    assert kwargs["enable_memory_saver"] is False
    assert kwargs["tp_size"] == 1


def test_compute_server_args_language_consumer_keeps_weight_sync_path():
    kwargs = _compute("regular", 2, {"language_only": True, "encoder_urls": ["http://enc:30000"]})

    assert kwargs["language_only"] is True
    assert kwargs["encoder_urls"] == ["http://enc:30000"]
    assert not kwargs.get("encoder_only")
    # Language engines are still weight-synced and still sleep under colocate.
    assert kwargs["load_format"] == "dummy"
    assert kwargs["enable_memory_saver"] is True
    assert kwargs["tp_size"] == 2


# -------------------------------------------------- weight-sync isolation ---


def _group(args, worker_type, n_engines, tp, gpu_offset):
    return EngineGroup(
        args=args,
        pg=None,
        all_engines=[f"{worker_type}{i}" for i in range(n_engines)],
        num_gpus_per_engine=tp,
        num_new_engines=0,
        worker_type=worker_type,
        gpu_offset=gpu_offset,
    )


def test_rollout_server_excludes_encoder_from_weight_sync_views():
    """world_size is ``1 + sum(engine_gpu_counts)`` over a single flat
    broadcast group, so an encoder counted here would deadlock every other
    member."""
    args = argparse.Namespace(num_gpus_per_node=8, rollout_num_gpus=5, rollout_num_gpus_per_engine=2)
    server = RolloutServer(engine_groups=[_group(args, "encoder", 1, 1, 0), _group(args, "regular", 2, 2, 1)])

    assert server.engines == ["regular0", "regular1"]
    assert server.encoder_engines == ["encoder0"]
    assert server.all_engines == ["encoder0", "regular0", "regular1"]
    assert server.engine_gpu_counts == [2, 2]
    assert len(server.engine_gpu_counts) == len(server.engines)
    # Language engines start after the encoder's bundle.
    assert server.engine_gpu_offsets == [1, 3]


class _RecordingEngine:
    """Stand-in for the Ray actor handle, recording ``.remote()`` calls."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        engine = self

        class _Method:
            @staticmethod
            def remote(*args, **kwargs):
                engine.calls.append(name)
                return f"ref:{name}"

        return _Method


def test_engine_group_spills_encoder_cache_instead_of_releasing_memory():
    """An ``--encoder-only`` server has no /release_memory_occupation, but its
    mooncake embedding cache is GPU-resident and idle for the whole training
    phase, so offload/onload must map to spill/restore instead of no-ops."""
    args = argparse.Namespace(num_gpus_per_node=8, rollout_num_gpus=5, rollout_num_gpus_per_engine=2)
    encoder = _group(args, "encoder", 1, 1, 0)
    engine = _RecordingEngine()
    encoder.all_engines = [engine]

    assert encoder.offload() == ["ref:spill_mm_cache"]
    assert encoder.onload() == ["ref:restore_mm_cache"]
    # A partial (weights-only) resume has no encoder analogue; the cache still
    # comes back, and restore_mm_cache is a no-op if it is already resident.
    assert encoder.onload(tags=["weights"]) == ["ref:restore_mm_cache"]
    assert engine.calls == ["spill_mm_cache", "restore_mm_cache", "restore_mm_cache"]


def test_engine_group_offload_unchanged_for_non_encoder():
    args = argparse.Namespace(num_gpus_per_node=8, rollout_num_gpus=5, rollout_num_gpus_per_engine=2)
    group = _group(args, "regular", 1, 2, 0)
    engine = _RecordingEngine()
    group.all_engines = [engine]

    group.offload()
    group.onload(tags=["weights"])
    assert engine.calls == ["release_memory_occupation", "resume_memory_occupation"]


# ------------------------------------------------------- cache-only patch ---


def test_encode_cache_only_patch_short_circuits_prefetch_rids():
    from relax.backends.sglang.encode_cache_only_patch import apply_patch

    apply_patch()
    from sglang.srt.disaggregation.encode_server import MMEncoder

    class FakeEncoder:
        def __init__(self):
            self.embedding_to_send = {}

    encoder = FakeEncoder()
    rid = f"{PREFETCH_RID_PREFIX}abc123"
    encoder.embedding_to_send[rid] = object()

    assert asyncio.run(MMEncoder.send_with_url(encoder, rid)) is None
    # Payload dropped instead of waiting out SGLANG_ENCODER_SEND_TIMEOUT.
    assert encoder.embedding_to_send == {}

    # Idempotent.
    assert apply_patch() is False


def test_is_prefetch_request():
    assert is_prefetch_request(f"{PREFETCH_RID_PREFIX}x")
    assert not is_prefetch_request("ordinary-rid")
    assert not is_prefetch_request(None)


# ------------------------------------------------------------- prefetcher ---


class _Sample:
    def __init__(self, multimodal_inputs):
        self.multimodal_inputs = multimodal_inputs


def test_iter_group_images_yields_once_per_group():
    """Samples in a group share one multimodal_inputs dict by reference, so the
    images must be encoded once per group rather than once per sample."""
    shared = {"images": ["a", "b"]}
    groups = [
        [_Sample(shared)] * 8,
        [_Sample({"images": ["c"]})] * 8,
        [_Sample({"images": []})] * 8,
        [_Sample(None)] * 8,
    ]

    assert list(_iter_group_images(groups)) == [["a", "b"], ["c"]]


def test_encoder_prefetch_disabled_without_encoder_urls():
    assert not is_encoder_prefetch_enabled(argparse.Namespace())
    assert is_encoder_prefetch_enabled(argparse.Namespace(rollout_encoder_urls=["http://enc:1"]))
    # submit() must be inert (and silent) on a non-EPD deployment.
    EncoderPrefetcher(argparse.Namespace()).submit(object())


def test_encoder_prefetch_payload_matches_sglang_encode_schema():
    calls = []

    class _Response:
        status_code = 200
        text = "ok"

    class _Session:
        def post(self, url, json=None, timeout=None):
            calls.append((url, json))
            return _Response()

    prefetcher = EncoderPrefetcher(argparse.Namespace(rollout_encoder_urls=["http://enc:30000"]))
    assert prefetcher._encode_one(_Session(), "http://enc:30000/", "B64", 600) is True

    url, payload = calls[0]
    assert url == "http://enc:30000/encode"
    assert payload["req_id"].startswith(PREFETCH_RID_PREFIX)
    assert payload["mm_items"] == ["B64"]
    assert payload["modality"] == "IMAGE"
    assert payload["num_parts"] == 1 and payload["part_idx"] == 0
    # None keeps the encoder on the send_with_url path, which the patch intercepts.
    assert payload["embedding_port"] is None


def test_encoder_prefetch_reports_failure_without_raising():
    class _Session:
        def post(self, url, json=None, timeout=None):
            raise ConnectionError("encoder down")

    prefetcher = EncoderPrefetcher(argparse.Namespace(rollout_encoder_urls=["http://enc:30000"]))
    assert prefetcher._encode_one(_Session(), "http://enc:30000", "B64", 1) is False
