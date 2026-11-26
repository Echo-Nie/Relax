# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Observe a fixed local expert-weight block; never alter optimizer updates.

Each rank records the first 65,536 flattened elements of one trainable expert
fc2 parameter. These are local samples, not whole-model gradient or delta
norms. A rank may legitimately observe no gradient/update when its expert is
not used; the experiment analyzer must verify effective learning across all
rank records.
"""

import hashlib
import json
import math
import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_SAMPLE_ELEMENTS = 65_536
_EXPERT_FC2 = re.compile(r"(?:^|\.)mlp\.experts\.(?:local_experts\.\d+\.)?linear_fc2\.weight\d*$")


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1)[:_SAMPLE_ELEMENTS].to(device="cpu", dtype=torch.float32).clone()


def _routing_replay_snapshot() -> dict[str, Any]:
    """Observe native consumption for the frozen 48-layer/full-recompute
    smoke."""
    from relax.utils.training.routing_replay import RoutingReplay

    layers = [
        {
            "layer": index,
            "recorded_batches": len(replay.top_indices_list),
            "forward_consumed": replay.forward_index,
            "backward_consumed": replay.backward_index,
        }
        for index, replay in enumerate(RoutingReplay.all_routing_replays)
    ]
    return {
        "expected_layers": 48,
        "layers": layers,
        "fully_consumed": len(layers) == 48
        and all(
            layer["recorded_batches"] > 0
            and layer["forward_consumed"] == layer["backward_consumed"] == layer["recorded_batches"]
            for layer in layers
        ),
    }


def install_observer(
    model: list,
    optimizer: Any,
    *,
    output: Path,
    rank: int,
    weight_decay: float,
    routing_replay: bool = False,
) -> None:
    """Wrap step once and preserve its original arguments and return value."""
    if getattr(optimizer, "sampling_replay_moe_observed", False):
        return
    selected = next(
        (
            (chunk_index, name, parameter)
            for chunk_index, chunk in enumerate(model)
            for name, parameter in chunk.named_parameters()
            if parameter.requires_grad and _EXPERT_FC2.search(name)
        ),
        None,
    )
    if selected is None:
        raise ValueError("MoE update observer requires a trainable mlp.experts expert fc2 weight")
    chunk_index, name, parameter = selected
    native_step = optimizer.step
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"parameter-updates-rank{rank}.jsonl"
    counter = 0

    def observed_step(*step_args: Any, **step_kwargs: Any) -> Any:
        nonlocal counter
        before = _snapshot(parameter)
        gradient = getattr(parameter, "main_grad", None)
        source = "main_grad"
        if gradient is None:
            gradient = parameter.grad
            source = "grad"
        sampled_gradient = _snapshot(gradient) if gradient is not None else None
        if not torch.isfinite(before).all() or (
            sampled_gradient is not None and not torch.isfinite(sampled_gradient).all()
        ):
            raise ValueError(f"MoE observer rank {rank}: non-finite sampled parameter/gradient before step")
        routing = _routing_replay_snapshot() if routing_replay else None
        if routing is not None and not routing["fully_consumed"]:
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "optimizer_step": counter,
                            "rank": rank,
                            "optimizer_called": False,
                            "routing_replay": routing,
                            "error": "native routing replay has missing or unconsumed layer batches",
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            raise ValueError(f"MoE observer rank {rank}: native routing replay not fully consumed; see {path}")
        result = native_step(*step_args, **step_kwargs)
        after = _snapshot(parameter)
        delta = after - before
        grad_norm = float(result[1]) if result[1] is not None else None
        row = {
            "optimizer_step": counter,
            "rank": rank,
            "model_chunk": chunk_index,
            "parameter": name,
            "parameter_shape": list(parameter.shape),
            "parameter_numel": parameter.numel(),
            "sample_offset": 0,
            "sample_elements": before.numel(),
            "observation_scope": "first 65536 elements of one local trainable expert fc2 parameter",
            "gradient_source": source if sampled_gradient is not None else None,
            "sampled_gradient_norm": float(sampled_gradient.norm()) if sampled_gradient is not None else 0.0,
            "sampled_gradient_nonzero": int(torch.count_nonzero(sampled_gradient))
            if sampled_gradient is not None
            else 0,
            "optimizer_grad_norm": grad_norm,
            "optimizer_successful": bool(result[0]),
            "parameter_finite_after": bool(torch.isfinite(after).all()),
            "changed_elements": int(torch.count_nonzero(delta)),
            "delta_l2": float(delta.norm()),
            "max_abs_delta": float(delta.abs().max()),
            "before_sha256": hashlib.sha256(before.numpy().tobytes()).hexdigest(),
            "after_sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest(),
            "weight_decay": weight_decay,
            "routing_replay": routing,
        }
        # Keep machine-readable evidence even when the native optimizer rejects a step.
        # None represents unavailable/non-finite values rather than invalid JSON NaN.
        finite = row["parameter_finite_after"] and all(
            value is None or math.isfinite(value)
            for value in (grad_norm, row["delta_l2"], row["max_abs_delta"], row["sampled_gradient_norm"])
        )
        for key in ("optimizer_grad_norm", "delta_l2", "max_abs_delta", "sampled_gradient_norm"):
            if row[key] is not None and not math.isfinite(row[key]):
                row[key] = None
        row["finite"] = finite
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        counter += 1
        if not finite or grad_norm is None or not result[0]:
            raise ValueError(f"MoE observer rank {rank}: optimizer failed or non-finite/unavailable norm; see {path}")
        return result

    optimizer.step = observed_step
    optimizer.sampling_replay_moe_observed = True


def before_train(args: Namespace, rollout_id: int, step_id: int, model: list, optimizer: Any, scheduler: Any) -> None:
    """Attach through the existing before-train-step experimental hook."""
    if args.rollout_result_dir and (Path(args.rollout_result_dir) / "quality-stop.json").exists():
        raise ValueError("Predeclared independent-evaluation quality collapse; preserve evidence")
    install_observer(
        model,
        optimizer,
        output=Path(os.environ.get("RELAX_MOE_AUDIT_DIR") or args.dump_details),
        rank=dist.get_rank(group=dist.group.WORLD),
        weight_decay=args.weight_decay,
        routing_replay=getattr(args, "use_rollout_routing_replay", False),
    )
