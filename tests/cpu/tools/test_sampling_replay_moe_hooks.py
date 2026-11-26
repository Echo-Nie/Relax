# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU SGD updates; only MCore's optimizer return protocol is adapted."""

import json

import pytest
import torch

from scripts.training.text.sampling_replay_moe_hooks import install_observer


class ExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.linear_fc2 = torch.nn.Module()
        self.mlp.experts.linear_fc2.weight0 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    @property
    def weight(self):
        return self.mlp.experts.linear_fc2.weight0


class CoreOptimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.sgd = torch.optim.SGD([parameter], lr=0.25)

    def step(self):
        self.sgd.step()
        self.result = (True, float(self.parameter.grad.norm()), 0)
        return self.result


def read_row(tmp_path, rank=0):
    return json.loads((tmp_path / f"parameter-updates-rank{rank}.jsonl").read_text())


def test_moe_hooks_record_real_parameter_update_and_preserve_optimizer_result(tmp_path):
    model = ExpertModel()
    optimizer = CoreOptimizer(model.weight)
    install_observer([model], optimizer, output=tmp_path, rank=3, weight_decay=0)
    install_observer([model], optimizer, output=tmp_path, rank=3, weight_decay=0)
    model.weight.grad = torch.tensor([2.0, 0.0, -4.0])
    result = optimizer.step()
    torch.testing.assert_close(model.weight, torch.tensor([0.5, 2.0, 4.0]))
    assert result is optimizer.result
    assert result[0] and result[1] == pytest.approx(20**0.5) and result[2] == 0
    row = read_row(tmp_path, rank=3)
    assert row["changed_elements"] == 2
    assert row["sampled_gradient_norm"] == pytest.approx(20**0.5)
    assert row["delta_l2"] == pytest.approx(1.25**0.5)
    assert row["max_abs_delta"] == 1.0
    assert row["before_sha256"] != row["after_sha256"]
    assert row["sample_elements"] == row["parameter_numel"] == 3
    assert row["finite"]
    assert not (tmp_path / "parameter-updates-rank0.jsonl").exists()


def test_moe_hooks_zero_update_is_not_mistaken_for_learning(tmp_path):
    model = ExpertModel()
    optimizer = CoreOptimizer(model.weight)
    install_observer([model], optimizer, output=tmp_path, rank=0, weight_decay=0)
    model.weight.grad = torch.zeros(3)
    optimizer.step()
    row = read_row(tmp_path)
    assert row["optimizer_successful"]
    assert row["changed_elements"] == row["sampled_gradient_nonzero"] == 0
    assert row["before_sha256"] == row["after_sha256"]
    assert row["delta_l2"] == row["optimizer_grad_norm"] == 0


def test_moe_hooks_prefer_main_grad_and_limit_observation_to_documented_block(tmp_path):
    model = ExpertModel()
    model.mlp.experts.linear_fc2.weight0 = torch.nn.Parameter(torch.ones(65_540))
    model.weight.grad = torch.zeros_like(model.weight)
    model.weight.grad[-1] = 4
    model.weight.main_grad = torch.ones_like(model.weight)
    optimizer = CoreOptimizer(model.weight)
    install_observer([model], optimizer, output=tmp_path, rank=0, weight_decay=0)
    optimizer.step()
    row = read_row(tmp_path)
    assert model.weight[-1] == 0
    assert row["sample_elements"] == 65_536
    assert row["parameter_numel"] == 65_540
    assert row["gradient_source"] == "main_grad"
    assert row["sampled_gradient_norm"] == 256
    assert row["changed_elements"] == 0  # The real update is outside the sampled block.


def test_moe_hooks_reject_nonfinite_gradient_before_update_and_dense_only_models(tmp_path):
    model = ExpertModel()
    optimizer = CoreOptimizer(model.weight)
    install_observer([model], optimizer, output=tmp_path, rank=0, weight_decay=0)
    model.weight.grad = torch.tensor([float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError, match="non-finite sampled"):
        optimizer.step()
    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0, 3.0]))
    assert not (tmp_path / "parameter-updates-rank0.jsonl").exists()
    with pytest.raises(ValueError, match="trainable mlp.experts"):
        install_observer([torch.nn.Linear(3, 2)], object(), output=tmp_path, rank=0, weight_decay=0)


def test_moe_hooks_optimizer_rejection_is_recorded_and_fails(tmp_path):
    model = ExpertModel()

    class RejectedOptimizer:
        def step(self):
            return False, float("inf"), None

    optimizer = RejectedOptimizer()
    install_observer([model], optimizer, output=tmp_path, rank=0, weight_decay=0)
    with pytest.raises(ValueError, match="optimizer failed"):
        optimizer.step()
    row = read_row(tmp_path)
    assert row["optimizer_successful"] is False
    assert row["finite"] is False
    assert row["optimizer_grad_norm"] is None
    assert row["changed_elements"] == 0


@pytest.fixture
def native_routing_replays(monkeypatch):
    from relax.utils.training import routing_replay as routing

    # CPU-only test: isolate the pinned allocator/device boundary, not routing.
    empty_like = torch.empty_like

    def cpu_empty_like(tensor, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty_like(tensor, **kwargs)

    monkeypatch.setattr(torch, "empty_like", cpu_empty_like)
    monkeypatch.setattr(routing.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(routing.RoutingReplay, "all_routing_replays", [])
    expected = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32)
    replays = [routing.RoutingReplay() for _ in range(48)]
    for replay in replays:
        replay.record(expected)
        torch.testing.assert_close(replay.pop_forward(), expected.long())
        torch.testing.assert_close(replay.pop_backward(), expected.long())
    return routing.RoutingReplay.all_routing_replays


def test_moe_hooks_record_real_native_routing_consumption_without_reset(tmp_path, native_routing_replays):
    model = ExpertModel()
    optimizer = CoreOptimizer(model.weight)
    install_observer([model], optimizer, output=tmp_path, rank=0, weight_decay=0, routing_replay=True)
    model.weight.grad = torch.ones(3)
    optimizer.step()
    row = read_row(tmp_path)
    assert row["routing_replay"]["fully_consumed"]
    assert row["routing_replay"]["expected_layers"] == len(row["routing_replay"]["layers"]) == 48
    assert all(
        layer["recorded_batches"] == layer["forward_consumed"] == layer["backward_consumed"] == 1
        for layer in row["routing_replay"]["layers"]
    )
    assert row["changed_elements"] == 3
    assert all(
        replay.forward_index == replay.backward_index == len(replay.top_indices_list) == 1
        for replay in native_routing_replays
    )


@pytest.mark.parametrize("missing", ["layer", "forward", "backward"])
def test_moe_hooks_reject_unconsumed_native_routing_before_optimizer(tmp_path, native_routing_replays, missing):
    if missing == "layer":
        native_routing_replays.pop()
    elif missing == "forward":
        native_routing_replays[-1].clear_forward()
    else:
        native_routing_replays[-1].backward_index = 0
    model = ExpertModel()
    optimizer = CoreOptimizer(model.weight)
    install_observer([model], optimizer, output=tmp_path, rank=0, weight_decay=0, routing_replay=True)
    model.weight.grad = torch.ones(3)
    with pytest.raises(ValueError, match="routing replay not fully consumed"):
        optimizer.step()
    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0, 3.0]))
    assert not hasattr(optimizer, "result")
    row = read_row(tmp_path)
    assert row["optimizer_called"] is False
    assert row["routing_replay"]["fully_consumed"] is False
