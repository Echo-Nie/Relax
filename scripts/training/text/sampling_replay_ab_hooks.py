# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Observe one real parameter around optimizer.step without changing its
result."""

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch


def before_train(args: Namespace, rollout_id: int, step_id: int, model: list, optimizer: Any, scheduler: Any) -> None:
    if getattr(optimizer, "sampling_replay_observed", False):
        return
    name, parameter = next(
        (name, parameter)
        for name, parameter in model[0].named_parameters()
        if name.endswith("decoder.layers.0.mlp.linear_fc2.weight")
    )
    native_step = optimizer.step
    counter = 0

    def observed_step(*step_args: Any, **step_kwargs: Any) -> tuple:
        nonlocal counter
        before = parameter.detach().float().cpu().clone()
        result = native_step(*step_args, **step_kwargs)
        after = parameter.detach().float().cpu()
        delta = after - before
        row = {
            "optimizer_step": counter,
            "parameter": name,
            "optimizer_successful": bool(result[0]),
            "changed_elements": int(torch.count_nonzero(delta)),
            "delta_l2": float(delta.norm()),
            "max_abs_delta": float(delta.abs().max()),
            "before_sha256": hashlib.sha256(before.numpy().tobytes()).hexdigest(),
            "after_sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest(),
            "weight_decay": args.weight_decay,
        }
        with (Path(args.dump_details) / "parameter-updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        counter += 1
        return result

    optimizer.step = observed_step
    optimizer.sampling_replay_observed = True
