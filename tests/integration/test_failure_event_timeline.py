from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray

from relax.core.controller import Controller
from relax.utils.failure_events import FailureEvent, FailureEventStore


NAMESPACE = "relax_failure_events"


@pytest.fixture
def ray_runtime():
    started = not ray.is_initialized()
    if started:
        ray.init(address="auto")

    yield

    if started:
        ray.shutdown()


def create_store(name: str, capacity: int = 16):
    store_actor = ray.remote(num_cpus=0)(FailureEventStore)
    return store_actor.options(
        name=name,
        namespace=NAMESPACE,
        lifetime="detached",
        get_if_exists=True,
    ).remote(capacity)


def kill_store(name: str):
    try:
        store = ray.get_actor(name, namespace=NAMESPACE)
    except ValueError:
        return
    ray.kill(store)


def test_detached_store_survives_driver_reconnect(ray_runtime):
    name = "failure_event_test_reconnect"

    kill_store(name)

    store = create_store(name)
    ray.get(
        store.append.remote(
            FailureEvent(
                fault_id="fault-1",
                role="actor",
                phase="detected",
                occurred_at_ms=1,
            )
        )
    )

    ray.shutdown()
    ray.init(address="auto")

    reconnected = ray.get_actor(name, namespace=NAMESPACE)
    page = ray.get(reconnected.query.remote())

    assert [event["fault_id"] for event in page["events"]] == ["fault-1"]

    ray.kill(reconnected)


def test_different_store_names_do_not_share_history(ray_runtime):
    first_name = "failure_event_test_run_a"
    second_name = "failure_event_test_run_b"

    kill_store(first_name)
    kill_store(second_name)

    first = create_store(first_name)
    second = create_store(second_name)

    ray.get(
        first.append.remote(
            FailureEvent(
                fault_id="fault-a",
                role="actor",
                phase="detected",
                occurred_at_ms=1,
            )
        )
    )

    first_page = ray.get(first.query.remote())
    second_page = ray.get(second.query.remote())

    assert [event["fault_id"] for event in first_page["events"]] == ["fault-a"]
    assert second_page["events"] == []

    ray.kill(first)
    ray.kill(second)


def test_controller_shutdown_kills_failure_event_store(ray_runtime):
    name = "failure_event_test_shutdown"

    kill_store(name)
    store = create_store(name)

    controller = Controller.__new__(Controller)
    controller._failure_event_store = store
    controller._health_manager = SimpleNamespace(
        stop=Mock(),
    )
    controller.serve_dict = {}
    controller._teacher_manager = None
    controller._shutdown_agentic_rollout_services = Mock()
    controller._cleanup_s3_model_weights_after_init = Mock()
    controller.stop_health_check = Mock()

    controller.shutdown()

    with pytest.raises(ValueError):
        ray.get_actor(name, namespace=NAMESPACE)


def test_failure_event_reporting_does_not_change_recovery():
    controller = Controller.__new__(Controller)

    append = Mock(side_effect=RuntimeError("store unavailable"))
    controller._failure_event_store = SimpleNamespace(append=SimpleNamespace(remote=append))

    controller._emit_failure_event(
        fault_id="fault-1",
        role="actor",
        phase="detected",
        reason="reported_error",
        step=1,
    )

    append.assert_called_once()
