from __future__ import annotations

import pytest

from scripts.profile_hot_paths import profile_function


def test_profile_function_rejects_zero_iterations() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        profile_function(lambda: None, iterations=0, name="noop")


def test_profile_function_rejects_boolean_iterations() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        profile_function(lambda: None, iterations=True, name="noop")


def test_profile_function_accepts_single_iteration() -> None:
    result = profile_function(lambda: None, iterations=1, name="noop")

    assert result.name == "noop"
    assert result.iterations == 1
    assert result.total_time >= 0.0
    assert result.avg_time >= 0.0
    assert result.min_time >= 0.0
    assert result.max_time >= 0.0
    assert result.std_dev == 0.0
