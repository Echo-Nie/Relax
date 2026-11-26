# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

from relax.utils.metrics import service as metrics_service
from tests.cpu.core.controller_test_utils import load_controller_with_stubbed_dependencies


def test_metrics_deployment_receives_driver_runtime_environment(monkeypatch):
    module = load_controller_with_stubbed_dependencies("controller_metrics_runtime_test")
    controller = object.__new__(module.Controller)
    controller.config = SimpleNamespace(enable_affinity=False)
    controller.runtime_env = {"env_vars": {"HTTPS_PROXY": "http://127.0.0.1:18080"}}
    controller._health_manager = SimpleNamespace(status=None)
    options = Mock(wraps=metrics_service.MetricsService.options)
    monkeypatch.setattr(metrics_service.MetricsService, "options", options)
    monkeypatch.setattr(module.serve, "run", Mock())

    controller._deploy_metrics_service()

    assert options.call_args.kwargs["ray_actor_options"]["runtime_env"] == controller.runtime_env
