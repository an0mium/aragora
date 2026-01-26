"""
Epic EHR Adapter.

SMART on FHIR integration with Epic-specific extensions:
- Epic App Orchard authentication
- MyChart patient portal integration
- Care Everywhere health information exchange
- Epic-specific resource extensions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from aragora.connectors.enterprise.healthcare.ehr.base import (
    EHRAdapter,
    EHRCapability,
    EHRConnectionConfig,
    EHRVendor,
)

logger = logging.getLogger(__name__)


# Epic-specific FHIR resource types and extensions
EPIC_RESOURCE_EXTENSIONS = {
    "Patient": [
        "http://open.epic.com/FHIR/StructureDefinition/extension/patient-mychart-status",
        "http://open.epic.com/FHIR/StructureDefinition/extension/patient-recordtype",
    ],
    "Encounter": [
        "http://open.epic.com/FHIR/StructureDefinition/extension/encounter-facility",
        "http://open.epic.com/FHIR/StructureDefinition/extension/encounter-department",
    ],
    "Appointment": [
        "http://open.epic.com/FHIR/StructureDefinition/extension/appointment-type",
    ],
}


# Epic-specific operation URLs
EPIC_OPERATIONS = {
    "patient-match": "/$match",
    "document-search": "/DocumentReference/$docref",
    "patient-everything": "/Patient/{id}/$everything",
    "care-everywhere": "/CareEverywhere/$query",
}


@dataclass
class EpicPatientContext:
    """Epic patient launch context."""

    patient_id: str
    fhir_id: str
    mychart_status: Optional[str] = None
    mrn: Optional[str] = None
    encounter_id: Optional[str] = None


class EpicAdapter(EHRAdapter):
    """
    Epic EHR adapter with SMART on FHIR and Epic-specific extensions.

    Features:
    - SMART on FHIR R4 compliance
    - Epic App Orchard authentication
    - MyChart patient context
    - Care Everywhere integration
    - Epic-specific resource extensions

    Note: Requires Epic App Orchard registration and appropriate scopes.
    """

    vendor = EHRVendor.EPIC
    capabilities: Set[EHRCapability] = {
        EHRCapability.SMART_ON_FHIR,
        EHRCapability.FHIR_R4,
        EHRCapability.BACKEND_SERVICES,
        EHRCapability.BULK_DATA_EXPORT,
        EHRCapability.CDS_HOOKS,
        EHRCapability.DOCUMENT_REFERENCES,
        EHRCapability.EPIC_MYCHART,
        EHRCapability.EPIC_CARE_EVERYWHERE,
    }

    # Epic-specific scopes
    EPIC_SCOPES = [
        "patient/*.read",
        "patient/Patient.read",
        "patient/Observation.read",
        "patient/Condition.read",
        "patient/MedicationRequest.read",
        "patient/AllergyIntolerance.read",
        "patient/Procedure.read",
        "patient/Immunization.read",
        "patient/DiagnosticReport.read",
        "patient/DocumentReference.read",
        "patient/Encounter.read",
        "launch/patient",
        "online_access",
    ]

    def __init__(self, config: EHRConnectionConfig):
        # Ensure Epic-specific scopes are included
        if not config.scopes:
            config.scopes = self.EPIC_SCOPES

        super().__init__(config)
        self._patient_context: Optional[EpicPatientContext] = None

    async def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Get patient resource by ID.

        Args:
            patient_id: Epic FHIR patient ID

        Returns:
            FHIR Patient resource with Epic extensions
        """
        patient = await self._request("GET", f"/Patient/{patient_id}")

        # Extract Epic-specific extensions
        mychart_status = self._extract_extension(
            patient,
            "http://open.epic.com/FHIR/StructureDefinition/extension/patient-mychart-status",
        )
        if mychart_status:
            patient["_epicMyChartStatus"] = mychart_status

        logger.debug(f"Retrieved Epic patient: {patient_id}")
        return patient

    async def search_patients(
        self,
        family: Optional[str] = None,
        given: Optional[str] = None,
        birthdate: Optional[str] = None,
        identifier: Optional[str] = None,
        mrn: Optional[str] = None,
        ssn_last4: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Search for patients in Epic.

        Args:
            family: Family name
            given: Given name
            birthdate: Birth date (YYYY-MM-DD)
            identifier: Any identifier
            mrn: Epic MRN (uses Epic-specific search)
            ssn_last4: Last 4 digits of SSN

        Returns:
            List of matching Patient resources
        """
        params = {}

        if family:
            params["family"] = family
        if given:
            params["given"] = given
        if birthdate:
            params["birthdate"] = birthdate
        if identifier:
            params["identifier"] = identifier
        if mrn:
            # Epic-specific MRN search
            params["identifier"] = f"urn:oid:1.2.840.114350.1.13.0.1.7.5.737384.0|{mrn}"

        # SSN search (Epic-specific)
        if ssn_last4:
            params["identifier"] = f"http://hl7.org/fhir/sid/us-ssn|****{ssn_last4}"

        bundle = await self._request("GET", "/Patient", params=params)
        entries = bundle.get("entry", [])

        patients = [entry.get("resource", {}) for entry in entries]
        logger.debug(f"Found {len(patients)} Epic patients matching search")

        return patients

    async def patient_match(
        self,
        family: str,
        given: str,
        birthdate: str,
        gender: Optional[str] = None,
        phone: Optional[str] = None,
        address_line: Optional[str] = None,
        address_city: Optional[str] = None,
        address_state: Optional[str] = None,
        address_postalcode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use Epic's $match operation for probabilistic patient matching.

        This is more robust than simple search for patient identification.

        Args:
            family: Family name (required)
            given: Given name (required)
            birthdate: Birth date YYYY-MM-DD (required)
            gender: Gender (male, female, other, unknown)
            phone: Phone number
            address_line: Street address
            address_city: City
            address_state: State
            address_postalcode: Postal code

        Returns:
            List of matched Patient resources with confidence scores
        """
        # Build Parameters resource for $match
        parameters = {
            "resourceType": "Parameters",
            "parameter": [
                {
                    "name": "resource",
                    "resource": {
                        "resourceType": "Patient",
                        "name": [
                            {
                                "family": family,
                                "given": [given],
                            }
                        ],
                        "birthDate": birthdate,
                    },
                },
                {
                    "name": "onlyCertainMatches",
                    "valueBoolean": False,
                },
            ],
        }

        patient_resource: dict = parameters["parameter"][0]["resource"]  # type: ignore[index]

        if gender:
            patient_resource["gender"] = gender

        if phone:
            patient_resource["telecom"] = [{"system": "phone", "value": phone}]

        if any([address_line, address_city, address_state, address_postalcode]):
            address: dict = {}
            if address_line:
                address["line"] = [address_line]
            if address_city:
                address["city"] = address_city
            if address_state:
                address["state"] = address_state
            if address_postalcode:
                address["postalCode"] = address_postalcode
            patient_resource["address"] = [address]

        result = await self._request(
            "POST",
            "/Patient/$match",
            json=parameters,
        )

        entries = result.get("entry", [])
        matches = []

        for entry in entries:
            patient = entry.get("resource", {})
            # Extract match confidence from search.score extension
            search = entry.get("search", {})
            score = search.get("score", 0)
            patient["_matchScore"] = score
            matches.append(patient)

        # Sort by match score descending
        matches.sort(key=lambda p: p.get("_matchScore", 0), reverse=True)

        logger.info(f"Patient $match found {len(matches)} potential matches")
        return matches

    async def get_patient_records(  # type: ignore[override]
        self,
        patient_id: str,
        resource_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Get all clinical records for a patient using $everything operation.

        Args:
            patient_id: Epic FHIR patient ID
            resource_types: Filter to specific resource types
            start_date: Start of date range (YYYY-MM-DD)
            end_date: End of date range (YYYY-MM-DD)

        Yields:
            FHIR resources for the patient
        """
        params = {}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date

        # Use $everything operation
        bundle = await self._request(
            "GET",
            f"/Patient/{patient_id}/$everything",
            params=params,
        )

        # Filter by resource type if specified
        type_filter = set(resource_types) if resource_types else None

        while True:
            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                resource_type = resource.get("resourceType")

                if type_filter is None or resource_type in type_filter:
                    yield resource

            # Check for pagination
            next_link = None
            for link in bundle.get("link", []):
                if link.get("relation") == "next":
                    next_link = link.get("url")
                    break

            if not next_link:
                break

            # Fetch next page
            bundle = await self._request("GET", next_link)

    async def get_documents(
        self,
        patient_id: str,
        category: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        include_content: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get documents for a patient using Epic's $docref operation.

        Args:
            patient_id: Epic FHIR patient ID
            category: Document category code
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            include_content: If True, fetch document content

        Returns:
            List of DocumentReference resources
        """
        params: Dict[str, Any] = {
            "patient": patient_id,
        }

        if category:
            params["category"] = category
        if date_from:
            params["date"] = f"ge{date_from}"
        if date_to:
            if "date" in params:
                params["date"] = [params["date"], f"le{date_to}"]
            else:
                params["date"] = f"le{date_to}"

        bundle = await self._request("GET", "/DocumentReference", params=params)
        documents = [entry.get("resource", {}) for entry in bundle.get("entry", [])]

        # Fetch document content if requested
        if include_content:
            for doc in documents:
                await self._fetch_document_content(doc)

        logger.debug(f"Retrieved {len(documents)} documents for patient {patient_id}")
        return documents

    async def _fetch_document_content(self, doc: Dict[str, Any]) -> None:
        """Fetch and attach document content."""
        for content in doc.get("content", []):
            attachment = content.get("attachment", {})
            url = attachment.get("url")

            if url and not attachment.get("data"):
                try:
                    # Fetch document binary
                    access_token = await self._ensure_authenticated()
                    response = await self._http_client.get(
                        url,
                        headers={"Authorization": f"Bearer {access_token}"},
                    )

                    if response.status_code == 200:
                        import base64

                        attachment["data"] = base64.b64encode(response.content).decode()

                except Exception as e:
                    logger.warning(f"Failed to fetch document content: {e}")

    async def get_appointments(
        self,
        patient_id: str,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get appointments for a patient.

        Args:
            patient_id: Epic FHIR patient ID
            status: Appointment status filter
            date_from: Start date
            date_to: End date

        Returns:
            List of Appointment resources
        """
        params: Dict[str, Any] = {
            "patient": patient_id,
        }

        if status:
            params["status"] = status
        if date_from:
            params["date"] = f"ge{date_from}"
        if date_to:
            if "date" in params:
                params["date"] = [params["date"], f"le{date_to}"]
            else:
                params["date"] = f"le{date_to}"

        bundle = await self._request("GET", "/Appointment", params=params)
        appointments = [entry.get("resource", {}) for entry in bundle.get("entry", [])]

        # Extract Epic-specific appointment type extension
        for apt in appointments:
            apt_type = self._extract_extension(
                apt,
                "http://open.epic.com/FHIR/StructureDefinition/extension/appointment-type",
            )
            if apt_type:
                apt["_epicAppointmentType"] = apt_type

        logger.debug(f"Retrieved {len(appointments)} appointments for patient {patient_id}")
        return appointments

    async def query_care_everywhere(
        self,
        patient_id: str,
        include_documents: bool = True,
    ) -> Dict[str, Any]:
        """
        Query Care Everywhere network for external records.

        Care Everywhere is Epic's health information exchange network.

        Args:
            patient_id: Epic FHIR patient ID
            include_documents: Include external documents

        Returns:
            External records from Care Everywhere network
        """
        # Note: Care Everywhere requires specific Epic configuration
        # and may not be available at all installations

        params = {
            "patient": patient_id,
            "_include": "DocumentReference" if include_documents else None,
        }

        try:
            result = await self._request(
                "GET",
                "/CareEverywhere",  # Epic-specific endpoint
                params={k: v for k, v in params.items() if v},
            )
            logger.info(f"Care Everywhere query completed for patient {patient_id}")
            return result
        except Exception as e:
            logger.warning(f"Care Everywhere query failed: {e}")
            return {"error": str(e), "available": False}

    async def get_mychart_status(self, patient_id: str) -> Dict[str, Any]:
        """
        Get MyChart activation status for a patient.

        Args:
            patient_id: Epic FHIR patient ID

        Returns:
            MyChart status information
        """
        patient = await self.get_patient(patient_id)

        status = {
            "patient_id": patient_id,
            "mychart_active": False,
            "activation_date": None,
            "last_login": None,
        }

        # Extract MyChart extension
        for ext in patient.get("extension", []):
            if "mychart-status" in ext.get("url", ""):
                status["mychart_active"] = ext.get("valueBoolean", False)
            elif "mychart-activation-date" in ext.get("url", ""):
                status["activation_date"] = ext.get("valueDateTime")
            elif "mychart-last-login" in ext.get("url", ""):
                status["last_login"] = ext.get("valueDateTime")

        return status

    def _extract_extension(
        self,
        resource: Dict[str, Any],
        extension_url: str,
    ) -> Optional[Any]:
        """Extract value from a FHIR extension."""
        for ext in resource.get("extension", []):
            if ext.get("url") == extension_url:
                # Return the first value found
                for key in ext:
                    if key.startswith("value"):
                        return ext[key]
        return None

    async def bulk_export(
        self,
        resource_types: Optional[List[str]] = None,
        since: Optional[str] = None,
        output_format: str = "application/fhir+ndjson",
    ) -> Dict[str, Any]:
        """
        Initiate FHIR Bulk Data Export.

        Args:
            resource_types: Resource types to export
            since: Only include resources modified since this date
            output_format: Output format (default: ndjson)

        Returns:
            Export job status with content-location for polling
        """
        params = {
            "_outputFormat": output_format,
        }

        if resource_types:
            params["_type"] = ",".join(resource_types)
        if since:
            params["_since"] = since

        # Kick off export - Epic returns 202 Accepted
        if not self._http_client:
            raise RuntimeError("Not connected")

        access_token = await self._ensure_authenticated()
        response = await self._http_client.get(
            f"{self.config.base_url}/$export",
            params=params,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/fhir+json",
                "Prefer": "respond-async",
            },
        )

        if response.status_code == 202:
            content_location = response.headers.get("Content-Location")
            logger.info(f"Bulk export initiated. Poll: {content_location}")
            return {
                "status": "in-progress",
                "poll_url": content_location,
            }
        else:
            raise RuntimeError(f"Bulk export failed: {response.status_code}")

    async def check_bulk_export_status(self, poll_url: str) -> Dict[str, Any]:
        """
        Check status of a bulk export job.

        Args:
            poll_url: Content-Location URL from export initiation

        Returns:
            Export status with output files when complete
        """
        if not self._http_client:
            raise RuntimeError("Not connected")

        access_token = await self._ensure_authenticated()
        response = await self._http_client.get(
            poll_url,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code == 202:
            # Still processing
            progress = response.headers.get("X-Progress", "unknown")
            return {
                "status": "in-progress",
                "progress": progress,
            }
        elif response.status_code == 200:
            # Complete
            return {
                "status": "complete",
                **response.json(),
            }
        else:
            return {
                "status": "error",
                "error": response.text,
            }
