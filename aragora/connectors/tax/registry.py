"""
Tax Connector Registry - Map jurisdictions to connector classes.
"""

from __future__ import annotations

from collections.abc import Callable

from aragora.connectors.accounting.irs import IRSConnector
from aragora.connectors.tax.generic import GenericTaxConnector


class TaxConnectorRegistry:
    """Registry for tax connectors by jurisdiction."""

    def __init__(self) -> None:
        self._registry: dict[str, Callable[[], object]] = {
            "us": IRSConnector,
            "usa": IRSConnector,
            "united_states": IRSConnector,
            "irs": IRSConnector,
        }

    def register(self, jurisdiction: str, factory: Callable[[], object]) -> None:
        self._registry[jurisdiction.lower()] = factory

    def resolve(self, jurisdiction: str) -> object:
        key = (jurisdiction or "us").lower()
        factory = self._registry.get(key)
        if factory:
            return factory()
        # Fall back to generic connector for non-US jurisdictions.
        return GenericTaxConnector(jurisdiction=jurisdiction or "UNKNOWN")


_REGISTRY = TaxConnectorRegistry()


def resolve_tax_connector(jurisdiction: str) -> object:
    """Resolve a tax connector instance for the given jurisdiction."""
    return _REGISTRY.resolve(jurisdiction)
