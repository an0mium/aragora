"""
Tests for Web3 provider module.

Tests cover:
- RPCHealth tracking (success, failure, cooldown, is_healthy, scaling)
- Web3Provider construction (from_config, from_env, __post_init__)
- Chain management (add_chain, get_config, multi-chain)
- Web3 instance caching and failover
- Health status reporting
- RPC failure recording and failover triggering
- Connection checking
- _check_web3 and _require_web3 utilities
- Module __all__ exports
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.blockchain.config import ChainConfig
from aragora.blockchain.provider import RPCHealth, Web3Provider

# Check if web3 is available
try:
    import web3 as _web3_mod

    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False

requires_web3 = pytest.mark.skipif(not HAS_WEB3, reason="web3 package required")


class TestRPCHealth:
    """Tests for RPCHealth tracking."""

    def test_create_health(self):
        health = RPCHealth(url="https://eth.rpc")
        assert health.url == "https://eth.rpc"
        assert health.consecutive_failures == 0
        assert health.last_failure_time == 0.0
        assert health.last_success_time == 0.0
        assert health.total_requests == 0
        assert health.total_failures == 0

    def test_is_healthy_default(self):
        health = RPCHealth(url="https://eth.rpc")
        assert health.is_healthy is True

    def test_record_success(self):
        health = RPCHealth(url="https://eth.rpc")
        health.record_success()
        assert health.consecutive_failures == 0
        assert health.total_requests == 1
        assert health.last_success_time > 0

    def test_record_failure(self):
        health = RPCHealth(url="https://eth.rpc")
        health.record_failure()
        assert health.consecutive_failures == 1
        assert health.total_requests == 1
        assert health.total_failures == 1
        assert health.last_failure_time > 0

    def test_consecutive_failures(self):
        health = RPCHealth(url="https://eth.rpc")
        health.record_failure()
        health.record_failure()
        health.record_failure()
        assert health.consecutive_failures == 3
        assert health.total_failures == 3

    def test_success_resets_consecutive_failures(self):
        health = RPCHealth(url="https://eth.rpc")
        health.record_failure()
        health.record_failure()
        assert health.consecutive_failures == 2
        health.record_success()
        assert health.consecutive_failures == 0
        assert health.total_failures == 2  # Total doesn't reset

    def test_is_healthy_with_failures_but_cooldown_passed(self):
        health = RPCHealth(url="https://eth.rpc")
        health.consecutive_failures = 1
        health.last_failure_time = time.monotonic() - 120  # 2 minutes ago
        assert health.is_healthy is True

    def test_is_healthy_false_within_cooldown(self):
        health = RPCHealth(url="https://eth.rpc")
        health.consecutive_failures = 1
        health.last_failure_time = time.monotonic()  # Just now
        assert health.is_healthy is False

    def test_cooldown_scales_with_failures(self):
        """Cooldown scales: 60s per failure, capped at 300s."""
        health = RPCHealth(url="https://eth.rpc")
        for _ in range(5):
            health.record_failure()
        # 5 failures => 300s cooldown
        health.last_failure_time = time.monotonic() - 250
        assert health.is_healthy is False  # 250 < 300

        health.last_failure_time = time.monotonic() - 310
        assert health.is_healthy is True  # 310 > 300

    def test_cooldown_capped_at_300(self):
        """Cooldown never exceeds 300s even with many failures."""
        health = RPCHealth(url="https://eth.rpc")
        for _ in range(20):
            health.record_failure()
        # 20 * 60 = 1200, but capped at 300
        health.last_failure_time = time.monotonic() - 310
        assert health.is_healthy is True

    def test_mixed_success_failure(self):
        """Mixed success and failure tracking."""
        health = RPCHealth(url="https://eth.rpc")
        health.record_success()
        health.record_failure()
        health.record_success()
        assert health.total_requests == 3
        assert health.total_failures == 1
        assert health.consecutive_failures == 0


class TestWeb3Provider:
    """Tests for Web3Provider class."""

    def test_create_provider_empty(self):
        provider = Web3Provider()
        assert provider.configs == {}
        assert provider.default_chain_id == 1

    def test_create_provider_with_config(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config}, default_chain_id=1)
        assert 1 in provider.configs
        assert provider.default_chain_id == 1

    def test_from_config(self):
        config = ChainConfig(chain_id=137, rpc_url="https://polygon.rpc")
        provider = Web3Provider.from_config(config)
        assert provider.default_chain_id == 137
        assert 137 in provider.configs

    def test_from_env(self):
        env = {"ERC8004_RPC_URL": "https://custom.rpc", "ERC8004_CHAIN_ID": "1"}
        with patch.dict("os.environ", env, clear=False):
            provider = Web3Provider.from_env()
            assert provider.default_chain_id == 1

    @patch("aragora.blockchain.provider.get_chain_config")
    def test_from_env_with_override_chain_id(self, mock_get_config):
        """from_env passes chain_id to get_chain_config."""
        config = ChainConfig(chain_id=42, rpc_url="https://rpc.test")
        mock_get_config.return_value = config
        provider = Web3Provider.from_env(chain_id=42)
        mock_get_config.assert_called_once_with(42)
        assert provider.default_chain_id == 42

    @patch("aragora.blockchain.provider.get_chain_config")
    def test_from_env_default_chain(self, mock_get_config):
        """from_env with no chain_id passes None."""
        config = ChainConfig(chain_id=1, rpc_url="https://rpc.test")
        mock_get_config.return_value = config
        Web3Provider.from_env()
        mock_get_config.assert_called_once_with(None)

    def test_add_chain(self):
        provider = Web3Provider()
        config = ChainConfig(chain_id=137, rpc_url="https://polygon.rpc")
        provider.add_chain(config)
        assert 137 in provider.configs
        assert provider.configs[137].rpc_url == "https://polygon.rpc"

    def test_add_chain_health_tracking(self):
        """add_chain initializes health tracking for all RPCs including fallbacks."""
        provider = Web3Provider()
        config = ChainConfig(
            chain_id=42161,
            rpc_url="https://arb.rpc",
            fallback_rpc_urls=["https://arb-fb.rpc"],
        )
        provider.add_chain(config)
        assert "https://arb.rpc" in provider._rpc_health
        assert "https://arb-fb.rpc" in provider._rpc_health

    def test_add_chain_does_not_override_active(self):
        """add_chain does not override an already-active RPC for existing chain."""
        provider = Web3Provider()
        c1 = ChainConfig(chain_id=10, rpc_url="https://first.rpc")
        provider.add_chain(c1)
        c2 = ChainConfig(chain_id=10, rpc_url="https://second.rpc")
        provider.add_chain(c2)
        assert provider._active_rpc[10] == "https://first.rpc"

    def test_get_config(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config}, default_chain_id=1)
        result = provider.get_config(chain_id=1)
        assert result.chain_id == 1
        assert result.rpc_url == "https://eth.rpc"

    def test_get_config_default(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config}, default_chain_id=1)
        result = provider.get_config()  # No chain_id
        assert result.chain_id == 1

    def test_get_config_unknown_chain(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config})
        with pytest.raises(ValueError, match="No configuration for chain"):
            provider.get_config(chain_id=999)

    def test_record_success(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config})
        provider.record_success(chain_id=1)
        health = provider._rpc_health.get("https://eth.rpc")
        assert health is not None
        assert health.total_requests == 1
        assert health.consecutive_failures == 0

    def test_record_failure(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config})
        provider.record_failure(chain_id=1)
        health = provider._rpc_health.get("https://eth.rpc")
        assert health is not None
        assert health.consecutive_failures == 1

    def test_record_success_unknown_chain(self):
        """record_success is a no-op for unconfigured chains."""
        provider = Web3Provider()
        # Should not raise
        provider.record_success(999)

    def test_record_failure_unknown_chain(self):
        """record_failure is a no-op for unconfigured chains."""
        provider = Web3Provider()
        # Should not raise
        provider.record_failure(999)

    def test_record_success_specific_chain(self):
        """record_success can target a specific chain."""
        provider = Web3Provider()
        c1 = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        c2 = ChainConfig(chain_id=137, rpc_url="https://polygon.rpc")
        provider.add_chain(c1)
        provider.add_chain(c2)
        provider.record_success(chain_id=137)
        assert provider._rpc_health["https://eth.rpc"].total_requests == 0
        assert provider._rpc_health["https://polygon.rpc"].total_requests == 1

    def test_is_connected_returns_false_without_web3(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config})
        with patch("aragora.blockchain.provider._check_web3", return_value=False):
            assert provider.is_connected(chain_id=1) is False

    def test_is_connected_delegates_to_web3(self):
        """is_connected calls w3.is_connected()."""
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config})
        mock_w3 = MagicMock()
        mock_w3.is_connected.return_value = True

        with patch.object(provider, "get_web3", return_value=mock_w3):
            assert provider.is_connected() is True
            mock_w3.is_connected.assert_called_once()

    def test_is_connected_returns_false_when_disconnected(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        provider = Web3Provider(configs={1: config})
        mock_w3 = MagicMock()
        mock_w3.is_connected.return_value = False
        provider._web3_instances["https://eth.rpc"] = mock_w3

        with patch("aragora.blockchain.provider._require_web3"):
            assert provider.is_connected() is False

    def test_is_connected_returns_false_on_exception(self):
        """is_connected returns False when get_web3 raises."""
        provider = Web3Provider()
        assert provider.is_connected() is False

    def test_get_health_status(self):
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://primary.rpc",
            fallback_rpc_urls=["https://backup.rpc"],
        )
        provider = Web3Provider(configs={1: config})
        provider.record_success(chain_id=1)
        provider.record_failure(chain_id=1)

        status = provider.get_health_status()
        assert "https://primary.rpc" in status
        assert status["https://primary.rpc"]["total_requests"] == 2

    def test_get_health_status_empty(self):
        """Empty provider returns empty health status."""
        provider = Web3Provider()
        assert provider.get_health_status() == {}

    def test_get_health_status_format(self):
        """Health status contains all expected keys per RPC."""
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://rpc.test",
            fallback_rpc_urls=["https://fb.test"],
        )
        provider = Web3Provider.from_config(config)
        status = provider.get_health_status()
        for url, info in status.items():
            assert "healthy" in info
            assert "consecutive_failures" in info
            assert "total_requests" in info
            assert "total_failures" in info

    def test_rpc_health_initialized_for_all_urls(self):
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://primary.rpc",
            fallback_rpc_urls=["https://backup1.rpc", "https://backup2.rpc"],
        )
        provider = Web3Provider(configs={1: config})
        assert "https://primary.rpc" in provider._rpc_health
        assert "https://backup1.rpc" in provider._rpc_health
        assert "https://backup2.rpc" in provider._rpc_health

    def test_active_rpc_initialized(self):
        config = ChainConfig(chain_id=1, rpc_url="https://primary.rpc")
        provider = Web3Provider(configs={1: config})
        assert provider._active_rpc[1] == "https://primary.rpc"

    def test_multi_chain_provider(self):
        config1 = ChainConfig(chain_id=1, rpc_url="https://eth.rpc")
        config2 = ChainConfig(chain_id=137, rpc_url="https://polygon.rpc")
        provider = Web3Provider(configs={1: config1, 137: config2}, default_chain_id=1)
        assert len(provider.configs) == 2
        assert provider.get_config(1).rpc_url == "https://eth.rpc"
        assert provider.get_config(137).rpc_url == "https://polygon.rpc"


class TestWeb3ProviderGetWeb3:
    """Tests for get_web3 method.

    These tests mock _require_web3 and the Web3 global so they run
    regardless of whether the web3 package is installed.
    """

    @patch("aragora.blockchain.provider._require_web3")
    def test_get_web3_returns_cached_instance(self, _mock_require):
        """get_web3 returns cached instances for the same RPC URL."""
        config = ChainConfig(chain_id=1, rpc_url="https://rpc.test")
        provider = Web3Provider.from_config(config)
        mock_w3 = MagicMock()
        provider._web3_instances["https://rpc.test"] = mock_w3
        # Ensure the Web3 global is set so the method doesn't try to import
        import aragora.blockchain.provider as _pmod

        _orig = _pmod.Web3
        _pmod.Web3 = MagicMock()
        try:
            assert provider.get_web3() is mock_w3
        finally:
            _pmod.Web3 = _orig

    @patch("aragora.blockchain.provider._require_web3")
    def test_get_web3_missing_chain_raises(self, _mock_require):
        """get_web3 raises ValueError for unconfigured chain."""
        import aragora.blockchain.provider as _pmod

        _orig = _pmod.Web3
        _pmod.Web3 = MagicMock()
        try:
            provider = Web3Provider()
            with pytest.raises(ValueError, match="No configuration for chain"):
                provider.get_web3(999)
        finally:
            _pmod.Web3 = _orig

    @patch("aragora.blockchain.provider._require_web3")
    def test_get_web3_failover_on_unhealthy(self, _mock_require):
        """get_web3 fails over to healthy fallback when primary is unhealthy."""
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://primary.rpc",
            fallback_rpc_urls=["https://fallback.rpc"],
        )
        provider = Web3Provider.from_config(config)

        # Make primary unhealthy
        provider._rpc_health["https://primary.rpc"].consecutive_failures = 3
        provider._rpc_health["https://primary.rpc"].last_failure_time = time.monotonic()

        # Put cached instance for fallback
        mock_w3 = MagicMock()
        provider._web3_instances["https://fallback.rpc"] = mock_w3

        import aragora.blockchain.provider as _pmod

        _orig = _pmod.Web3
        _pmod.Web3 = MagicMock()
        try:
            result = provider.get_web3()
            assert result is mock_w3
            assert provider._active_rpc[1] == "https://fallback.rpc"
        finally:
            _pmod.Web3 = _orig

    @patch("aragora.blockchain.provider._require_web3")
    def test_get_web3_no_healthy_rpc_uses_active(self, _mock_require):
        """get_web3 uses current active RPC when no healthy alternatives exist."""
        config = ChainConfig(chain_id=1, rpc_url="https://rpc.test")
        provider = Web3Provider.from_config(config)

        # Make primary unhealthy
        provider._rpc_health["https://rpc.test"].consecutive_failures = 3
        provider._rpc_health["https://rpc.test"].last_failure_time = time.monotonic()

        mock_w3 = MagicMock()
        provider._web3_instances["https://rpc.test"] = mock_w3

        import aragora.blockchain.provider as _pmod

        _orig = _pmod.Web3
        _pmod.Web3 = MagicMock()
        try:
            result = provider.get_web3()
            assert result is mock_w3
        finally:
            _pmod.Web3 = _orig


class TestWeb3ProviderFailover:
    """Tests for Web3Provider failover behavior."""

    def test_failover_on_record_failure(self):
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://primary.rpc",
            fallback_rpc_urls=["https://backup.rpc"],
        )
        provider = Web3Provider(configs={1: config})

        # Mark primary as unhealthy
        provider._rpc_health["https://primary.rpc"].consecutive_failures = 5
        provider._rpc_health["https://primary.rpc"].last_failure_time = time.monotonic()

        # Record another failure should trigger failover
        provider.record_failure(chain_id=1)

        # Active RPC should change to backup
        assert provider._active_rpc[1] == "https://backup.rpc"

    def test_no_failover_without_alternatives(self):
        """record_failure does not failover when there are no healthy alternatives."""
        config = ChainConfig(chain_id=1, rpc_url="https://only.rpc")
        provider = Web3Provider(configs={1: config})
        provider.record_failure(chain_id=1)
        # Only one RPC available and now unhealthy
        assert provider._active_rpc[1] == "https://only.rpc"

    def test_find_healthy_rpc(self):
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://primary.rpc",
            fallback_rpc_urls=["https://backup1.rpc", "https://backup2.rpc"],
        )
        provider = Web3Provider(configs={1: config})

        # Mark primary as unhealthy
        provider._rpc_health["https://primary.rpc"].consecutive_failures = 5
        provider._rpc_health["https://primary.rpc"].last_failure_time = time.monotonic()

        healthy = provider._find_healthy_rpc(config)
        assert healthy in ["https://backup1.rpc", "https://backup2.rpc"]

    def test_no_healthy_rpc_returns_none(self):
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://primary.rpc",
            fallback_rpc_urls=["https://backup.rpc"],
        )
        provider = Web3Provider(configs={1: config})

        # Mark all as unhealthy
        now = time.monotonic()
        for url in ["https://primary.rpc", "https://backup.rpc"]:
            provider._rpc_health[url].consecutive_failures = 10
            provider._rpc_health[url].last_failure_time = now

        healthy = provider._find_healthy_rpc(config)
        assert healthy is None

    def test_find_healthy_rpc_with_unknown_url(self):
        """Returns URL that has no health record (assumed healthy)."""
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://primary.rpc",
            fallback_rpc_urls=["https://new.rpc"],
        )
        provider = Web3Provider.from_config(config)
        # Remove health record for fallback
        del provider._rpc_health["https://new.rpc"]
        # Make primary unhealthy
        provider._rpc_health["https://primary.rpc"].consecutive_failures = 1
        provider._rpc_health["https://primary.rpc"].last_failure_time = time.monotonic()
        result = provider._find_healthy_rpc(config)
        assert result == "https://new.rpc"


class TestWeb3Check:
    """Tests for _check_web3 and _require_web3."""

    def test_require_web3_raises_when_unavailable(self):
        import aragora.blockchain.provider as pmod

        original = pmod._web3_available
        try:
            pmod._web3_available = False
            with pytest.raises(ImportError, match="web3 is required"):
                pmod._require_web3()
        finally:
            pmod._web3_available = original

    def test_check_web3_caches_result(self):
        import aragora.blockchain.provider as pmod

        original = pmod._web3_available
        try:
            pmod._web3_available = True
            assert pmod._check_web3() is True
        finally:
            pmod._web3_available = original


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_exports_rpc_health(self):
        from aragora.blockchain.provider import __all__

        assert "RPCHealth" in __all__

    def test_exports_web3_provider(self):
        from aragora.blockchain.provider import __all__

        assert "Web3Provider" in __all__

    def test_all_count(self):
        from aragora.blockchain.provider import __all__

        assert len(__all__) == 2
