from __future__ import annotations

from aragora.nomic.dev_coordination_verification import (
    _inferred_expected_tests_for_work_order,
    _pytest_command_targets,
    _verification_command_covers_expected,
)


def test_pytest_command_targets_ignores_options() -> None:
    targets = _pytest_command_targets(
        "python -m pytest tests/nomic/test_dev_coordination.py -k merge_gate --maxfail 1 -q"
    )

    assert targets == ["tests/nomic/test_dev_coordination.py"]


def test_verification_command_covers_directory_level_pytest_target() -> None:
    assert _verification_command_covers_expected(
        "python -m pytest tests/nomic -q",
        "python -m pytest tests/nomic/test_dev_coordination.py -q",
    )


def test_inferred_expected_tests_prefers_specific_pytest_targets() -> None:
    work_order = {
        "expected_tests": ["python -m pytest tests/nomic -q"],
        "metadata": {
            "acceptance_criteria": [
                "Run python -m pytest tests/nomic/test_dev_coordination.py -q before merge."
            ]
        },
    }

    assert _inferred_expected_tests_for_work_order(work_order) == [
        "python -m pytest tests/nomic/test_dev_coordination.py -q"
    ]
