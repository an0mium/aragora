"""
Tax Connectors.

Provides a generic tax connector for non-US jurisdictions and a registry
for mapping jurisdiction codes to connector instances.
"""

from aragora.connectors.tax.generic import GenericTaxConnector
from aragora.connectors.tax.registry import TaxConnectorRegistry, resolve_tax_connector

__all__ = [
    "GenericTaxConnector",
    "TaxConnectorRegistry",
    "resolve_tax_connector",
]
