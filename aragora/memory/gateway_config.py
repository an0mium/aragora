"""Configuration for the Unified Memory Gateway."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryGatewayConfig:
    """Configuration for MemoryGateway."""

    enabled: bool = False
    query_timeout_seconds: float = 15.0
    dedup_threshold: float = 0.95
    default_sources: list[str] | None = None  # None = all available
    enable_retention_gate: bool = False
    parallel_queries: bool = True
