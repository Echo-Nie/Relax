"""Regression tests for Mini-SWE-Agent R2E reward scoring."""

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "examples/mini_swe_agent/agent_server.py"


def load_r2e_reward():
    """Load unchanged scoring functions without Agent dependencies."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    names = {
        "_parse_log_pytest",
        "_decolor_dict_keys",
        "_r2e_reward",
    }

    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]

    assert {node.name for node in nodes} == names

    namespace = {
        "json": json,
        "re": re,
        "Any": Any,
    }

    module = ast.Module(body=nodes, type_ignores=[])

    exec(compile(module, str(SOURCE), "exec"), namespace)

    return namespace["_r2e_reward"]


reward_fn = load_r2e_reward()


def score(expected, output):
    row = {"expected_output_json": json.dumps(expected)}
    return reward_fn(row, output)


HEADER = "short test summary info\n"

CASES = [
    (
        "valid_success",
        {"test_valid": "PASSED"},
        HEADER + "PASSED tests/test_calc.py::test_valid\n",
        1.0,
    ),
    (
        "valid_mismatch",
        {"test_valid": "PASSED"},
        HEADER + "FAILED tests/test_calc.py::test_valid - AssertionError\n",
        0.0,
    ),
    (
        "valid_expected_failed",
        {"test_valid": "FAILED"},
        HEADER + "FAILED tests/test_calc.py::test_valid - AssertionError\n",
        1.0,
    ),
    (
        "no_tests_empty_expected",
        {},
        "1 deselected in 0.00s\n",
        0.0,
    ),
    (
        "invalid_empty_test_id",
        {"test_valid": "PASSED"},
        HEADER + "PASSED\n",
        0.0,
    ),
    (
        "missing_expected_test",
        {"test_valid": "PASSED", "test_other": "PASSED"},
        HEADER + "PASSED tests/test_calc.py::test_valid\n",
        0.0,
    ),
]


@pytest.mark.parametrize(
    "name,expected,output,desired",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_r2e_reward_regression(name, expected, output, desired):
    assert score(expected, output) == desired
