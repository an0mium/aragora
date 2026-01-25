"""
EHR (Electronic Health Record) Adapters.

Vendor-specific adapters for major EHR systems:
- Epic (SMART on FHIR + Epic-specific extensions)
- Cerner (SMART on FHIR + Millennium API patterns)

All adapters extend the base FHIR connector with vendor-specific
authentication, endpoints, and data transformations.
"""

from aragora.connectors.enterprise.healthcare.ehr.base import (
    EHRAdapter,
    EHRVendor,
    EHRCapability,
    EHRConnectionConfig,
)
from aragora.connectors.enterprise.healthcare.ehr.epic import EpicAdapter
from aragora.connectors.enterprise.healthcare.ehr.cerner import CernerAdapter

__all__ = [
    # Base
    "EHRAdapter",
    "EHRVendor",
    "EHRCapability",
    "EHRConnectionConfig",
    # Vendors
    "EpicAdapter",
    "CernerAdapter",
]
