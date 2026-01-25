"""
Performance benchmarks for Aragora.

Run with:
    pytest tests/benchmarks/ --benchmark-only  # if pytest-benchmark installed
    pytest tests/benchmarks/ -v  # basic timing tests
"""

import pytest

pytestmark = pytest.mark.benchmark
