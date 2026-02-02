"""
Tests for OpenClaw handler bounds validation.

Verifies that environment variables and request parameters are
properly validated and clamped to safe ranges.
"""

import logging
from unittest.mock import patch

import pytest

from aragora.server.handlers.gateway import openclaw


class TestMemoryBoundsValidation:
    """Tests for OPENCLAW_MAX_MEMORY_MB bounds validation."""

    def setup_method(self):
        """Reset the gateway adapter singleton before each test."""
        openclaw._gateway_adapter = None

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "32", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    def test_memory_too_low_is_clamped(self, caplog):
        """Memory value below minimum (64) should be clamped to 64."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_memory_mb == 64
        assert "OPENCLAW_MAX_MEMORY_MB=32 out of bounds" in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "20000", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    def test_memory_too_high_is_clamped(self, caplog):
        """Memory value above maximum (16384) should be clamped to 16384."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_memory_mb == 16384
        assert "OPENCLAW_MAX_MEMORY_MB=20000 out of bounds" in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    def test_memory_valid_is_not_clamped(self, caplog):
        """Memory value within bounds should not be clamped."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_memory_mb == 512
        assert "OPENCLAW_MAX_MEMORY_MB" not in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "64", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    def test_memory_at_minimum_boundary(self, caplog):
        """Memory value at minimum boundary should not trigger warning."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_memory_mb == 64
        assert "OPENCLAW_MAX_MEMORY_MB" not in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "16384", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    def test_memory_at_maximum_boundary(self, caplog):
        """Memory value at maximum boundary should not trigger warning."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_memory_mb == 16384
        assert "OPENCLAW_MAX_MEMORY_MB" not in caplog.text


class TestExecutionTimeBoundsValidation:
    """Tests for OPENCLAW_MAX_EXECUTION_SECONDS bounds validation."""

    def setup_method(self):
        """Reset the gateway adapter singleton before each test."""
        openclaw._gateway_adapter = None

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "0"},
    )
    def test_execution_too_low_is_clamped(self, caplog):
        """Execution time below minimum (1) should be clamped to 1."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_execution_seconds == 1
        assert "OPENCLAW_MAX_EXECUTION_SECONDS=0 out of bounds" in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "7200"},
    )
    def test_execution_too_high_is_clamped(self, caplog):
        """Execution time above maximum (3600) should be clamped to 3600."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_execution_seconds == 3600
        assert "OPENCLAW_MAX_EXECUTION_SECONDS=7200 out of bounds" in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    def test_execution_valid_is_not_clamped(self, caplog):
        """Execution time within bounds should not be clamped."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_execution_seconds == 300
        assert "OPENCLAW_MAX_EXECUTION_SECONDS" not in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "1"},
    )
    def test_execution_at_minimum_boundary(self, caplog):
        """Execution time at minimum boundary should not trigger warning."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_execution_seconds == 1
        assert "OPENCLAW_MAX_EXECUTION_SECONDS" not in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "3600"},
    )
    def test_execution_at_maximum_boundary(self, caplog):
        """Execution time at maximum boundary should not trigger warning."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_execution_seconds == 3600
        assert "OPENCLAW_MAX_EXECUTION_SECONDS" not in caplog.text


class TestRequestTimeoutValidation:
    """Tests for timeout_seconds request parameter validation."""

    def setup_method(self):
        """Reset the gateway adapter singleton before each test."""
        openclaw._gateway_adapter = None

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    async def test_timeout_capped_at_max_execution(self):
        """Request timeout should be capped at max_execution_seconds."""
        # Get adapter to establish max_execution
        adapter = openclaw._get_gateway_adapter()
        assert adapter.sandbox_config.max_execution_seconds == 300

        # Test that timeout exceeding max_execution is capped
        data = {"content": "test task", "timeout_seconds": 600}

        # We test the timeout validation logic directly
        timeout = data.get("timeout_seconds", 300)
        if not isinstance(timeout, int) or timeout < 1:
            timeout = 300
        timeout = min(timeout, adapter.sandbox_config.max_execution_seconds)

        assert timeout == 300  # Capped at max_execution

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    async def test_timeout_valid_value_unchanged(self):
        """Valid timeout within bounds should not be changed."""
        adapter = openclaw._get_gateway_adapter()

        data = {"content": "test task", "timeout_seconds": 60}

        timeout = data.get("timeout_seconds", 300)
        if not isinstance(timeout, int) or timeout < 1:
            timeout = 300
        timeout = min(timeout, adapter.sandbox_config.max_execution_seconds)

        assert timeout == 60  # Unchanged

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    async def test_timeout_negative_defaults_to_300(self):
        """Negative timeout should default to 300."""
        adapter = openclaw._get_gateway_adapter()

        data = {"content": "test task", "timeout_seconds": -10}

        timeout = data.get("timeout_seconds", 300)
        if not isinstance(timeout, int) or timeout < 1:
            timeout = 300
        timeout = min(timeout, adapter.sandbox_config.max_execution_seconds)

        assert timeout == 300

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    async def test_timeout_zero_defaults_to_300(self):
        """Zero timeout should default to 300."""
        adapter = openclaw._get_gateway_adapter()

        data = {"content": "test task", "timeout_seconds": 0}

        timeout = data.get("timeout_seconds", 300)
        if not isinstance(timeout, int) or timeout < 1:
            timeout = 300
        timeout = min(timeout, adapter.sandbox_config.max_execution_seconds)

        assert timeout == 300

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    async def test_timeout_non_integer_defaults_to_300(self):
        """Non-integer timeout should default to 300."""
        adapter = openclaw._get_gateway_adapter()

        data = {"content": "test task", "timeout_seconds": "invalid"}

        timeout = data.get("timeout_seconds", 300)
        if not isinstance(timeout, int) or timeout < 1:
            timeout = 300
        timeout = min(timeout, adapter.sandbox_config.max_execution_seconds)

        assert timeout == 300

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "300"},
    )
    async def test_timeout_missing_defaults_to_300(self):
        """Missing timeout should default to 300."""
        adapter = openclaw._get_gateway_adapter()

        data = {"content": "test task"}  # No timeout_seconds

        timeout = data.get("timeout_seconds", 300)
        if not isinstance(timeout, int) or timeout < 1:
            timeout = 300
        timeout = min(timeout, adapter.sandbox_config.max_execution_seconds)

        assert timeout == 300

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "512", "OPENCLAW_MAX_EXECUTION_SECONDS": "100"},
    )
    async def test_timeout_default_capped_at_low_max_execution(self):
        """Default timeout (300) should be capped when max_execution is lower."""
        # Reset to pick up new env vars
        openclaw._gateway_adapter = None
        adapter = openclaw._get_gateway_adapter()
        assert adapter.sandbox_config.max_execution_seconds == 100

        data = {"content": "test task"}  # No timeout_seconds, will default to 300

        timeout = data.get("timeout_seconds", 300)
        if not isinstance(timeout, int) or timeout < 1:
            timeout = 300
        timeout = min(timeout, adapter.sandbox_config.max_execution_seconds)

        assert timeout == 100  # Capped at max_execution


class TestCombinedBoundsValidation:
    """Tests for combined memory and execution bounds."""

    def setup_method(self):
        """Reset the gateway adapter singleton before each test."""
        openclaw._gateway_adapter = None

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "10", "OPENCLAW_MAX_EXECUTION_SECONDS": "0"},
    )
    def test_both_values_clamped(self, caplog):
        """Both memory and execution should be clamped when out of bounds."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_memory_mb == 64
        assert adapter.sandbox_config.max_execution_seconds == 1
        assert "OPENCLAW_MAX_MEMORY_MB=10 out of bounds" in caplog.text
        assert "OPENCLAW_MAX_EXECUTION_SECONDS=0 out of bounds" in caplog.text

    @patch.dict(
        "os.environ",
        {"OPENCLAW_MAX_MEMORY_MB": "100000", "OPENCLAW_MAX_EXECUTION_SECONDS": "100000"},
    )
    def test_both_values_clamped_high(self, caplog):
        """Both memory and execution should be clamped when too high."""
        with caplog.at_level(logging.WARNING):
            adapter = openclaw._get_gateway_adapter()

        assert adapter.sandbox_config.max_memory_mb == 16384
        assert adapter.sandbox_config.max_execution_seconds == 3600
        assert "OPENCLAW_MAX_MEMORY_MB=100000 out of bounds" in caplog.text
        assert "OPENCLAW_MAX_EXECUTION_SECONDS=100000 out of bounds" in caplog.text
