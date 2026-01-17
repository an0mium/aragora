"""
Feature stability markers for honest epistemics.

This module provides stability markers for Aragora features, enabling:
1. Honest communication about feature maturity
2. Clear distinction between stable and experimental features
3. Programmatic detection of feature stability levels

The hacker aesthetic is about honesty, not chaos. These markers let us
ship experimental things with clear "this is experimental" flags - not
as legal disclaimers but as honest epistemics.

Usage:
    from aragora.config.stability import Stability, stability_marker

    @stability_marker(Stability.EXPERIMENTAL)
    async def nomic_self_modify():
        '''This feature is experimental - may change or break.'''
        ...

    @stability_marker(Stability.STABLE)
    async def run_debate():
        '''This is a stable, tested feature.'''
        ...

    # Check stability at runtime
    stability = get_feature_stability("nomic_self_modify")
    if stability == Stability.EXPERIMENTAL:
        logger.warning("Using experimental feature: nomic_self_modify")
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class Stability(Enum):
    """Feature stability levels for honest epistemics.

    These aren't legal disclaimers - they're honest communication about
    what works reliably and what's still being figured out.
    """

    STABLE = "stable"
    """Works reliably. Tested extensively. API unlikely to change."""

    EXPERIMENTAL = "experimental"
    """Works sometimes. Here's what we know. Here's what we don't."""

    PREVIEW = "preview"
    """Might break. Early access. Expect changes."""

    DEPRECATED = "deprecated"
    """Being phased out. Use alternative instead."""


@dataclass
class FeatureStabilityInfo:
    """Information about a feature's stability."""

    name: str
    stability: Stability
    description: str = ""
    alternative: str | None = None  # For deprecated features
    since_version: str | None = None  # When this stability was assigned


# Registry of feature stability markers
_stability_registry: dict[str, FeatureStabilityInfo] = {}


def stability_marker(
    stability: Stability,
    description: str = "",
    alternative: str | None = None,
    since_version: str | None = None,
) -> Callable[[F], F]:
    """Decorator to mark a function's stability level.

    Args:
        stability: The stability level of this feature
        description: Optional description of what works/doesn't
        alternative: For deprecated features, what to use instead
        since_version: When this stability level was assigned

    Example:
        @stability_marker(Stability.EXPERIMENTAL, description="May produce false positives")
        def detect_sycophancy(response: str) -> bool:
            ...
    """

    def decorator(func: F) -> F:
        func_name = func.__qualname__

        # Register this feature
        _stability_registry[func_name] = FeatureStabilityInfo(
            name=func_name,
            stability=stability,
            description=description,
            alternative=alternative,
            since_version=since_version,
        )

        # Store stability on the function itself
        setattr(func, "_stability", stability)
        setattr(func, "_stability_info", _stability_registry[func_name])

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Log warning for non-stable features
            if stability == Stability.EXPERIMENTAL:
                logger.debug(f"Using experimental feature: {func_name}")
            elif stability == Stability.PREVIEW:
                logger.info(f"Using preview feature: {func_name}")
            elif stability == Stability.DEPRECATED:
                logger.warning(
                    f"Using deprecated feature: {func_name}. "
                    f"Use {alternative or 'alternative'} instead."
                )
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def get_feature_stability(feature_name: str) -> Stability | None:
    """Get the stability level of a registered feature.

    Args:
        feature_name: The function's qualified name

    Returns:
        Stability level or None if not registered
    """
    info = _stability_registry.get(feature_name)
    return info.stability if info else None


def get_feature_info(feature_name: str) -> FeatureStabilityInfo | None:
    """Get full stability information for a registered feature."""
    return _stability_registry.get(feature_name)


def list_features_by_stability(stability: Stability) -> list[FeatureStabilityInfo]:
    """List all features with a given stability level."""
    return [info for info in _stability_registry.values() if info.stability == stability]


def list_all_features() -> dict[str, FeatureStabilityInfo]:
    """Get all registered features with their stability info."""
    return dict(_stability_registry)


# Convenience constants for common feature areas
STABLE_FEATURES = [
    "debate.run",
    "consensus.detect",
    "elo.calculate",
    "memory.retrieve",
]

EXPERIMENTAL_FEATURES = [
    "nomic.self_modify",
    "trickster.intervene",
    "counterfactual.branch",
    "evidence.validate",
]

PREVIEW_FEATURES = [
    "belief.propagate",
    "calibration.adjust",
]


def get_stability_badge(stability: Stability) -> str:
    """Get a display badge for a stability level.

    Returns ASCII art badge for terminal/log display.
    """
    badges = {
        Stability.STABLE: "[STABLE]",
        Stability.EXPERIMENTAL: "[EXPERIMENTAL]",
        Stability.PREVIEW: "[PREVIEW]",
        Stability.DEPRECATED: "[DEPRECATED]",
    }
    return badges.get(stability, "[UNKNOWN]")


def get_stability_color(stability: Stability) -> str:
    """Get a color name for UI display of stability level."""
    colors = {
        Stability.STABLE: "green",
        Stability.EXPERIMENTAL: "yellow",
        Stability.PREVIEW: "orange",
        Stability.DEPRECATED: "red",
    }
    return colors.get(stability, "gray")


__all__ = [
    "Stability",
    "FeatureStabilityInfo",
    "stability_marker",
    "get_feature_stability",
    "get_feature_info",
    "list_features_by_stability",
    "list_all_features",
    "get_stability_badge",
    "get_stability_color",
    "STABLE_FEATURES",
    "EXPERIMENTAL_FEATURES",
    "PREVIEW_FEATURES",
]
