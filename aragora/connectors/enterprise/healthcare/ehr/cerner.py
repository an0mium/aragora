"""
Cerner EHR Adapter.

SMART on FHIR integration with Cerner-specific patterns:
- Cerner Millennium authentication flows
- PowerChart extensions (when available)
- Bulk data export support
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from aragora.connectors.enterprise.healthcare.ehr.base import (
    EHRAdapter,
    EHRCapability,
    EHRConnectionConfig,
    EHRVendor,
)

logger = logging.getLogger(__name__)


class CernerAdapter(EHRAdapter):
    """
    Cerner EHR adapter with SMART on FHIR support.

    Features:
    - SMART on FHIR R4 compliance
    - Cerner Millennium conventions
    - Bulk data export (where enabled)
    """

    vendor = EHRVendor.CERNER
    capabilities: Set[EHRCapability] = {
        EHRCapability.SMART_ON_FHIR,
        EHRCapability.FHIR_R4,
        EHRCapability.BACKEND_SERVICES,
        EHRCapability.BULK_DATA_EXPORT,
        EHRCapability.CERNER_MILLENNIUM,
        EHRCapability.CERNER_POWERCHART,
    }

    CERNER_SCOPES = [
        "system/*.read",
        "patient/*.read",
        "openid",
        "profile",
        "launch/patient",
        "launch/encounter",
        "offline_access",
    ]

    def __init__(self, config: EHRConnectionConfig):
        if not config.scopes:
            config.scopes = self.CERNER_SCOPES

        super().__init__(config)

    async def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Get patient resource by ID.

        Args:
            patient_id: Cerner FHIR patient ID

        Returns:
            FHIR Patient resource
        """
        patient = await self._request("GET", f"/Patient/{patient_id}")
        logger.debug("Retrieved Cerner patient: %s", patient_id)
        return patient

    async def search_patients(
        self,
        family: Optional[str] = None,
        given: Optional[str] = None,
        birthdate: Optional[str] = None,
        identifier: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Search for patients in Cerner.

        Args:
            family: Family name
            given: Given name
            birthdate: Birth date (YYYY-MM-DD)
            identifier: Any identifier

        Returns:
            List of matching Patient resources
        """
        params: Dict[str, Any] = {}
        if family:
            params["family"] = family
        if given:
            params["given"] = given
        if birthdate:
            params["birthdate"] = birthdate
        if identifier:
            params["identifier"] = identifier

        bundle = await self._request("GET", "/Patient", params=params)
        entries = bundle.get("entry", [])
        patients = [entry.get("resource", {}) for entry in entries]
        logger.debug("Found %d Cerner patients matching search", len(patients))
        return patients

    async def get_patient_records(
        self,
        patient_id: str,
        resource_types: Optional[List[str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Get all clinical records for a patient.

        Uses $everything when available; falls back to pagination if needed.
        """
        bundle = await self._request(
            "GET",
            f"/Patient/{patient_id}/$everything",
        )

        type_filter = set(resource_types) if resource_types else None

        while True:
            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                resource_type = resource.get("resourceType")
                if type_filter is None or resource_type in type_filter:
                    yield resource

            next_link = None
            for link in bundle.get("link", []):
                if link.get("relation") == "next":
                    next_link = link.get("url")
                    break

            if not next_link:
                break

            bundle = await self._request("GET", next_link)
