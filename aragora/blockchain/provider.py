"""
Web3 provider with multi-chain support, failover, and circuit breaker.

Manages connections to Ethereum-compatible networks for ERC-8004 interactions.
Wraps the web3.py library with resilience patterns matching Aragora conventions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aragora.blockchain.config import ChainConfig, get_chain_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy web3 import to keep blockchain optional
_web3_available: bool | None = None
Web3: Any | None = None


def _check_web3() -> bool:
    """Check if web3 is available."""
    global _web3_available
    if Web3 is not None:
        _web3_available = True
        return True
    if _web3_available is None:
        try:
            import web3  # noqa: F401

            _web3_available = True
        except ImportError:
            _web3_available = False
    return _web3_available


def _require_web3() -> None:
    """Raise ImportError if web3 is not installed."""
    if not _check_web3():
        raise ImportError(
            "web3 is required for blockchain integration. "
            "Install with: pip install aragora[blockchain]"
        )


@dataclass
class RPCHealth:
    """Health tracking for an RPC endpoint."""

    url: str
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0

    @property
    def is_healthy(self) -> bool:
        """Whether this RPC endpoint is considered healthy."""
        if self.consecutive_failures == 0:
            return True
        # Allow retry after cooldown (60s per consecutive failure, max 300s)
        cooldown = min(self.consecutive_failures * 60, 300)
        return (time.monotonic() - self.last_failure_time) > cooldown

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_success_time = time.monotonic()
        self.total_requests += 1

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        self.total_requests += 1
        self.total_failures += 1


@dataclass
class Web3Provider:
    """Multi-chain Web3 provider with failover and health tracking.

    Manages Web3 instances for one or more chains, automatically failing over
    to backup RPC endpoints when the primary becomes unavailable.

    Usage:
        provider = Web3Provider.from_env()
        w3 = provider.get_web3()
        block = w3.eth.block_number

    Attributes:
        configs: Chain configurations keyed by chain_id.
        default_chain_id: Default chain to use when not specified.
    """

    configs: dict[int, ChainConfig] = field(default_factory=dict)
    default_chain_id: int = 1
    _web3_instances: dict[str, Any] = field(default_factory=dict, repr=False)
    _rpc_health: dict[str, RPCHealth] = field(default_factory=dict, repr=False)
    _active_rpc: dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Initialize RPC health tracking."""
        for config in self.configs.values():
            for url in config.all_rpc_urls:
                if url not in self._rpc_health:
                    self._rpc_health[url] = RPCHealth(url=url)
            if config.chain_id not in self._active_rpc:
                self._active_rpc[config.chain_id] = config.rpc_url

    @classmethod
    def from_env(cls, chain_id: int | None = None) -> Web3Provider:
        """Create a provider from environment variables.

        Args:
            chain_id: Override chain ID. If None, reads from ERC8004_CHAIN_ID.

        Returns:
            Configured Web3Provider instance.
        """
        config = get_chain_config(chain_id)
        return cls(
            configs={config.chain_id: config},
            default_chain_id=config.chain_id,
        )

    @classmethod
    def from_config(cls, config: ChainConfig) -> Web3Provider:
        """Create a provider from a ChainConfig.

        Args:
            config: Chain configuration.

        Returns:
            Configured Web3Provider instance.
        """
        return cls(
            configs={config.chain_id: config},
            default_chain_id=config.chain_id,
        )

    def add_chain(self, config: ChainConfig) -> None:
        """Add a chain configuration.

        Args:
            config: Chain configuration to add.
        """
        self.configs[config.chain_id] = config
        for url in config.all_rpc_urls:
            if url not in self._rpc_health:
                self._rpc_health[url] = RPCHealth(url=url)
        if config.chain_id not in self._active_rpc:
            self._active_rpc[config.chain_id] = config.rpc_url

    def get_web3(self, chain_id: int | None = None) -> Any:
        """Get a Web3 instance for the specified chain.

        Automatically fails over to backup RPC endpoints if the active one
        is unhealthy.

        Args:
            chain_id: Chain ID. Defaults to default_chain_id.

        Returns:
            Web3 instance connected to the chain.

        Raises:
            ImportError: If web3 is not installed.
            ValueError: If no configuration exists for the chain.
            ConnectionError: If no healthy RPC endpoints are available.
        """
        _require_web3()
        global Web3
        if Web3 is None:
            from web3 import Web3 as Web3Class

            Web3 = Web3Class

        cid = chain_id or self.default_chain_id
        config = self.configs.get(cid)
        if not config:
            raise ValueError(f"No configuration for chain {cid}")

        # Find a healthy RPC URL
        active_url = self._active_rpc.get(cid, config.rpc_url)
        health = self._rpc_health.get(active_url)

        if health and not health.is_healthy:
            # Try failover
            new_url = self._find_healthy_rpc(config)
            if new_url:
                active_url = new_url
                self._active_rpc[cid] = new_url
                logger.info(f"Failover to RPC: {new_url} for chain {cid}")
            else:
                logger.warning(f"No healthy RPCs for chain {cid}, using {active_url}")

        # Return cached or create new instance
        if active_url in self._web3_instances:
            return self._web3_instances[active_url]

        w3 = Web3(Web3.HTTPProvider(active_url))
        self._web3_instances[active_url] = w3
        return w3

    def get_config(self, chain_id: int | None = None) -> ChainConfig:
        """Get the chain configuration.

        Args:
            chain_id: Chain ID. Defaults to default_chain_id.

        Returns:
            ChainConfig for the specified chain.

        Raises:
            ValueError: If no configuration exists.
        """
        cid = chain_id or self.default_chain_id
        config = self.configs.get(cid)
        if not config:
            raise ValueError(f"No configuration for chain {cid}")
        return config

    def record_success(self, chain_id: int | None = None) -> None:
        """Record a successful RPC call for health tracking."""
        cid = chain_id or self.default_chain_id
        url = self._active_rpc.get(cid)
        if url and url in self._rpc_health:
            self._rpc_health[url].record_success()

    def record_failure(self, chain_id: int | None = None) -> None:
        """Record a failed RPC call and attempt failover if needed."""
        cid = chain_id or self.default_chain_id
        url = self._active_rpc.get(cid)
        if url and url in self._rpc_health:
            self._rpc_health[url].record_failure()

        config = self.configs.get(cid)
        if config:
            new_url = self._find_healthy_rpc(config)
            if new_url and new_url != url:
                self._active_rpc[cid] = new_url
                logger.info(f"RPC failover: {url} -> {new_url} for chain {cid}")

    def is_connected(self, chain_id: int | None = None) -> bool:
        """Check if connected to the chain.

        Args:
            chain_id: Chain ID to check.

        Returns:
            True if connected.
        """
        try:
            w3 = self.get_web3(chain_id)
            return w3.is_connected()
        except (RuntimeError, ConnectionError, ValueError, OSError) as exc:
            logger.debug("Chain connection check failed: %s", exc)
            return False

    def get_health_status(self) -> dict[str, Any]:
        """Get health status for all RPC endpoints.

        Returns:
            Dictionary with health info per RPC URL.
        """
        return {
            url: {
                "healthy": health.is_healthy,
                "consecutive_failures": health.consecutive_failures,
                "total_requests": health.total_requests,
                "total_failures": health.total_failures,
            }
            for url, health in self._rpc_health.items()
        }

    def _find_healthy_rpc(self, config: ChainConfig) -> str | None:
        """Find a healthy RPC URL from the config's URL list."""
        for url in config.all_rpc_urls:
            health = self._rpc_health.get(url)
            if health is None or health.is_healthy:
                return url
        return None


__all__ = [
    "RPCHealth",
    "Web3Provider",
]
