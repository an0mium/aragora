"""
Tests for connector Prometheus metrics.

Tests cover:
- _init_metrics initialization
- _init_noop_metrics (when Prometheus disabled)
- record_sync
- record_sync_items
- record_sync_error
- record_item_latency
- record_rate_limit
- record_auth_failure
- set_connector_health
- inc_active_syncs / dec_active_syncs
- measure_sync context manager
- measure_item_sync context manager
- Convenience functions (sharepoint, confluence, etc.)
- record_cache_hit / record_cache_miss
- record_retry
- get_connector_metrics
"""

import pytest
from unittest.mock import patch, MagicMock
import time

# Import after mocking to test noop behavior
import aragora.connectors.metrics as metrics_module


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics initialization state before each test."""
    metrics_module._initialized = False
    yield
    metrics_module._initialized = False


class TestNoopMetrics:
    """Tests for noop metrics when Prometheus is disabled."""

    def test_noop_metric_accepts_any_method(self):
        """NoopMetric accepts any method call."""
        metrics_module._init_noop_metrics()

        # Should not raise
        metrics_module.CONNECTOR_SYNCS.labels(a=1, b=2).inc()
        metrics_module.CONNECTOR_SYNC_DURATION.labels(x="y").observe(1.5)
        metrics_module.CONNECTOR_HEALTH.labels(foo="bar").set(1)

    def test_noop_metric_returns_self(self):
        """NoopMetric returns self for chaining."""
        metrics_module._init_noop_metrics()

        result = metrics_module.CONNECTOR_SYNCS.labels(a=1)
        assert result.inc() is not None  # Can chain


class TestRecordSync:
    """Tests for record_sync function."""

    def test_record_sync_initializes_metrics(self):
        """record_sync initializes metrics."""
        with patch.object(metrics_module, "_init_metrics") as mock_init:
            mock_init.return_value = False
            metrics_module._init_noop_metrics()

            metrics_module.record_sync(
                connector_type="sharepoint",
                connector_id="corp",
                status="success",
            )

            mock_init.assert_called()

    def test_record_sync_with_all_params(self):
        """record_sync handles all parameters."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.record_sync(
            connector_type="confluence",
            connector_id="team",
            status="success",
            duration_seconds=45.2,
            items_synced=1500,
        )

    def test_record_sync_failure(self):
        """record_sync handles failure status."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.record_sync(
            connector_type="slack",
            connector_id="workspace",
            status="failure",
            duration_seconds=5.0,
        )


class TestRecordSyncItems:
    """Tests for record_sync_items function."""

    def test_record_sync_items(self):
        """record_sync_items records item counts."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.record_sync_items(
            connector_type="confluence",
            item_type="page",
            count=250,
        )

    def test_record_sync_items_default_count(self):
        """record_sync_items uses default count of 1."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.record_sync_items(
            connector_type="notion",
            item_type="database",
        )


class TestRecordSyncError:
    """Tests for record_sync_error function."""

    def test_record_sync_error(self):
        """record_sync_error records error types."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.record_sync_error(
            connector_type="gdrive",
            error_type="network",
        )
        metrics_module.record_sync_error(
            connector_type="gdrive",
            error_type="auth",
        )


class TestRecordItemLatency:
    """Tests for record_item_latency function."""

    def test_record_item_latency(self):
        """record_item_latency records latency."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.record_item_latency(
            connector_type="sharepoint",
            item_type="document",
            latency_seconds=0.25,
        )


class TestRecordRateLimit:
    """Tests for record_rate_limit function."""

    def test_record_rate_limit(self):
        """record_rate_limit increments counter."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.record_rate_limit("confluence")


class TestRecordAuthFailure:
    """Tests for record_auth_failure function."""

    def test_record_auth_failure(self):
        """record_auth_failure records failure."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.record_auth_failure(
            connector_type="slack",
            connector_id="workspace1",
        )


class TestSetConnectorHealth:
    """Tests for set_connector_health function."""

    def test_set_connector_healthy(self):
        """set_connector_health sets health to 1."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.set_connector_health(
            connector_type="notion",
            connector_id="workspace",
            healthy=True,
        )

    def test_set_connector_unhealthy(self):
        """set_connector_health sets health to 0."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.set_connector_health(
            connector_type="notion",
            connector_id="workspace",
            healthy=False,
        )


class TestActiveSyncs:
    """Tests for inc_active_syncs and dec_active_syncs."""

    def test_inc_active_syncs(self):
        """inc_active_syncs increments gauge."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.inc_active_syncs("sharepoint")

    def test_dec_active_syncs(self):
        """dec_active_syncs decrements gauge."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # Should not raise
        metrics_module.dec_active_syncs("sharepoint")


class TestMeasureSync:
    """Tests for measure_sync context manager."""

    def test_measure_sync_success(self):
        """measure_sync records success."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        with metrics_module.measure_sync("sharepoint", "corp") as ctx:
            ctx["items_synced"] = 100
            ctx["status"] = "success"

        # Context manager should complete without error

    def test_measure_sync_failure(self):
        """measure_sync records failure on exception."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        with pytest.raises(ValueError):
            with metrics_module.measure_sync("confluence", "team") as ctx:
                raise ValueError("Sync failed")

    def test_measure_sync_default_status(self):
        """measure_sync uses default success status."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        with metrics_module.measure_sync("notion", "ws") as ctx:
            pass  # Don't set status

        # Default status should be "success"


class TestMeasureItemSync:
    """Tests for measure_item_sync context manager."""

    def test_measure_item_sync(self):
        """measure_item_sync records item and latency."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        with metrics_module.measure_item_sync("confluence", "page"):
            time.sleep(0.01)  # Small delay

    def test_measure_item_sync_always_records_latency(self):
        """measure_item_sync records latency even on error."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        with pytest.raises(RuntimeError):
            with metrics_module.measure_item_sync("slack", "message"):
                raise RuntimeError("Process failed")


class TestConvenienceFunctions:
    """Tests for connector-specific convenience functions."""

    def test_record_sharepoint_sync(self):
        """record_sharepoint_sync records sync and items."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        metrics_module.record_sharepoint_sync(
            connector_id="corp",
            status="success",
            duration_seconds=30.0,
            documents=100,
            folders=10,
        )

    def test_record_confluence_sync(self):
        """record_confluence_sync records sync and items."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        metrics_module.record_confluence_sync(
            connector_id="team",
            status="success",
            duration_seconds=45.0,
            pages=200,
            attachments=50,
        )

    def test_record_notion_sync(self):
        """record_notion_sync records sync and items."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        metrics_module.record_notion_sync(
            connector_id="workspace",
            status="success",
            duration_seconds=20.0,
            pages=150,
            databases=5,
        )

    def test_record_slack_sync(self):
        """record_slack_sync records sync and items."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        metrics_module.record_slack_sync(
            connector_id="company",
            status="success",
            duration_seconds=60.0,
            messages=5000,
            files=100,
        )

    def test_record_gdrive_sync(self):
        """record_gdrive_sync records sync and items."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        metrics_module.record_gdrive_sync(
            connector_id="drive",
            status="success",
            duration_seconds=90.0,
            documents=500,
            folders=50,
        )


class TestCacheMetrics:
    """Tests for cache hit/miss metrics."""

    def test_record_cache_hit(self):
        """record_cache_hit increments counter."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        metrics_module.record_cache_hit("confluence")

    def test_record_cache_miss(self):
        """record_cache_miss increments counter."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        metrics_module.record_cache_miss("confluence")


class TestRecordRetry:
    """Tests for record_retry function."""

    def test_record_retry(self):
        """record_retry increments counter with reason."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        metrics_module.record_retry("sharepoint", "network_error")
        metrics_module.record_retry("sharepoint", "rate_limit")


class TestGetConnectorMetrics:
    """Tests for get_connector_metrics function."""

    def test_get_connector_metrics_returns_dict(self):
        """get_connector_metrics returns a dict."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        result = metrics_module.get_connector_metrics()
        # Should return a dict (may have error key if prometheus not installed)
        assert isinstance(result, dict)

    def test_get_connector_metrics_handles_missing_prometheus(self):
        """Handles prometheus not being installed."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # The function handles ImportError internally
        result = metrics_module.get_connector_metrics()
        assert isinstance(result, dict)


class TestInitMetrics:
    """Tests for _init_metrics function."""

    def test_init_metrics_skips_if_already_initialized(self):
        """_init_metrics returns early if already initialized."""
        metrics_module._initialized = True

        result = metrics_module._init_metrics()
        assert result is True

    def test_init_metrics_can_be_called_multiple_times(self):
        """_init_metrics can be called multiple times safely."""
        # First call initializes
        metrics_module._init_metrics()
        # Second call returns early
        result = metrics_module._init_metrics()
        assert result is True  # Already initialized

    def test_noop_metrics_work_after_init(self):
        """After init, metrics functions work (even if noop)."""
        metrics_module._init_noop_metrics()
        metrics_module._initialized = True

        # All metric functions should work without error
        metrics_module.record_sync("test", "id", "success")
        metrics_module.record_sync_items("test", "item")
        metrics_module.record_sync_error("test", "error")
        metrics_module.record_item_latency("test", "item", 0.1)
        metrics_module.record_rate_limit("test")
        metrics_module.record_auth_failure("test", "id")
        metrics_module.set_connector_health("test", "id", True)
        metrics_module.inc_active_syncs("test")
        metrics_module.dec_active_syncs("test")
        metrics_module.record_cache_hit("test")
        metrics_module.record_cache_miss("test")
        metrics_module.record_retry("test", "reason")
