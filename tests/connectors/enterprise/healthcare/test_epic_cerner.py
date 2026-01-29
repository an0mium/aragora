"""
Tests for Epic and Cerner EHR adapters.

Covers:
- EpicAdapter: patient ops, search, $match, documents, appointments,
  Care Everywhere, MyChart, bulk export, error handling
- CernerAdapter: patient ops, search, patient records pagination, error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from aragora.connectors.enterprise.healthcare.ehr.base import (
    EHRConnectionConfig,
    EHRVendor,
)
from aragora.connectors.enterprise.healthcare.ehr.epic import EpicAdapter
from aragora.connectors.enterprise.healthcare.ehr.cerner import CernerAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _epic_config() -> EHRConnectionConfig:
    return EHRConnectionConfig(
        vendor=EHRVendor.EPIC,
        base_url="https://fhir.epic.example.com/R4",
        organization_id="org-epic-001",
        client_id="epic-client-id",
        client_secret="epic-client-secret",
    )


def _cerner_config() -> EHRConnectionConfig:
    return EHRConnectionConfig(
        vendor=EHRVendor.CERNER,
        base_url="https://fhir.cerner.example.com/R4",
        organization_id="org-cerner-001",
        client_id="cerner-client-id",
        client_secret="cerner-client-secret",
    )


@pytest.fixture
def epic_adapter():
    return EpicAdapter(_epic_config())


@pytest.fixture
def cerner_adapter():
    return CernerAdapter(_cerner_config())


# Helpers
def _auth_patch(adapter):
    return patch.object(
        adapter, "_ensure_authenticated", new_callable=AsyncMock, return_value="mock-token"
    )


_AUTH_PATCH_EPIC = _auth_patch
_AUTH_PATCH_CERNER = _AUTH_PATCH_EPIC  # same signature

SAMPLE_PATIENT = {
    "resourceType": "Patient",
    "id": "patient-123",
    "name": [{"family": "Smith", "given": ["John"]}],
    "birthDate": "1990-01-15",
    "gender": "male",
    "extension": [],
}

SAMPLE_PATIENT_WITH_MYCHART = {
    **SAMPLE_PATIENT,
    "extension": [
        {
            "url": "http://open.epic.com/FHIR/StructureDefinition/extension/patient-mychart-status",
            "valueBoolean": True,
        },
        {
            "url": "http://open.epic.com/FHIR/StructureDefinition/extension/patient-mychart-activation-date",
            "valueDateTime": "2023-06-01T00:00:00Z",
        },
        {
            "url": "http://open.epic.com/FHIR/StructureDefinition/extension/patient-mychart-last-login",
            "valueDateTime": "2024-01-10T14:30:00Z",
        },
    ],
}

SAMPLE_BUNDLE = {
    "resourceType": "Bundle",
    "type": "searchset",
    "entry": [
        {"resource": {"resourceType": "Patient", "id": "p1", "name": [{"family": "Doe"}]}},
        {"resource": {"resourceType": "Patient", "id": "p2", "name": [{"family": "Roe"}]}},
    ],
}

SAMPLE_MATCH_BUNDLE = {
    "resourceType": "Bundle",
    "type": "searchset",
    "entry": [
        {
            "resource": {"resourceType": "Patient", "id": "m1"},
            "search": {"score": 0.95},
        },
        {
            "resource": {"resourceType": "Patient", "id": "m2"},
            "search": {"score": 0.72},
        },
    ],
}

SAMPLE_DOCUMENT_BUNDLE = {
    "resourceType": "Bundle",
    "entry": [
        {
            "resource": {
                "resourceType": "DocumentReference",
                "id": "doc-1",
                "content": [{"attachment": {"url": "https://example.com/doc1"}}],
            }
        },
    ],
}

SAMPLE_APPOINTMENT_BUNDLE = {
    "resourceType": "Bundle",
    "entry": [
        {
            "resource": {
                "resourceType": "Appointment",
                "id": "apt-1",
                "status": "booked",
                "extension": [
                    {
                        "url": "http://open.epic.com/FHIR/StructureDefinition/extension/appointment-type",
                        "valueString": "Office Visit",
                    }
                ],
            }
        },
    ],
}


# ===================================================================
# Epic Adapter Tests
# ===================================================================


class TestEpicAdapterGetPatient:
    """Tests for EpicAdapter.get_patient()."""

    @pytest.mark.asyncio
    async def test_get_patient_returns_fhir_resource(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter, "_request", new_callable=AsyncMock, return_value=SAMPLE_PATIENT
            ):
                result = await epic_adapter.get_patient("patient-123")
                assert result["resourceType"] == "Patient"
                assert result["id"] == "patient-123"

    @pytest.mark.asyncio
    async def test_get_patient_calls_correct_path(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value=SAMPLE_PATIENT)
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.get_patient("patient-xyz")
                mock_req.assert_awaited_once_with("GET", "/Patient/patient-xyz")

    @pytest.mark.asyncio
    async def test_get_patient_extracts_mychart_extension(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter,
                "_request",
                new_callable=AsyncMock,
                return_value=SAMPLE_PATIENT_WITH_MYCHART,
            ):
                result = await epic_adapter.get_patient("patient-123")
                assert result.get("_epicMyChartStatus") is True


class TestEpicAdapterSearchPatients:
    """Tests for EpicAdapter.search_patients()."""

    @pytest.mark.asyncio
    async def test_search_patients_returns_list(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter, "_request", new_callable=AsyncMock, return_value=SAMPLE_BUNDLE
            ):
                results = await epic_adapter.search_patients(family="Doe")
                assert len(results) == 2
                assert results[0]["id"] == "p1"

    @pytest.mark.asyncio
    async def test_search_patients_passes_params(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value={"resourceType": "Bundle", "entry": []})
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.search_patients(
                    family="Smith", given="Jane", birthdate="1985-03-20"
                )
                _, kwargs = mock_req.call_args
                assert kwargs["params"]["family"] == "Smith"
                assert kwargs["params"]["given"] == "Jane"
                assert kwargs["params"]["birthdate"] == "1985-03-20"

    @pytest.mark.asyncio
    async def test_search_patients_by_mrn(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value={"resourceType": "Bundle", "entry": []})
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.search_patients(mrn="MRN12345")
                _, kwargs = mock_req.call_args
                assert "MRN12345" in kwargs["params"]["identifier"]

    @pytest.mark.asyncio
    async def test_search_patients_empty_bundle(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter,
                "_request",
                new_callable=AsyncMock,
                return_value={"resourceType": "Bundle", "entry": []},
            ):
                results = await epic_adapter.search_patients(family="Nobody")
                assert results == []


class TestEpicAdapterPatientMatch:
    """Tests for EpicAdapter.patient_match()."""

    @pytest.mark.asyncio
    async def test_patient_match_returns_sorted_by_score(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter, "_request", new_callable=AsyncMock, return_value=SAMPLE_MATCH_BUNDLE
            ):
                matches = await epic_adapter.patient_match(
                    family="Smith", given="John", birthdate="1990-01-15"
                )
                assert len(matches) == 2
                assert matches[0]["_matchScore"] == 0.95
                assert matches[1]["_matchScore"] == 0.72

    @pytest.mark.asyncio
    async def test_patient_match_posts_parameters_resource(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value={"entry": []})
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.patient_match(
                    family="Smith", given="John", birthdate="1990-01-15", gender="male"
                )
                mock_req.assert_awaited_once()
                args, kwargs = mock_req.call_args
                assert args[0] == "POST"
                assert args[1] == "/Patient/$match"
                body = kwargs["json"]
                assert body["resourceType"] == "Parameters"
                patient_resource = body["parameter"][0]["resource"]
                assert patient_resource["gender"] == "male"

    @pytest.mark.asyncio
    async def test_patient_match_with_address(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value={"entry": []})
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.patient_match(
                    family="Doe",
                    given="Jane",
                    birthdate="1985-03-20",
                    address_city="Boston",
                    address_state="MA",
                )
                body = mock_req.call_args.kwargs["json"]
                patient_resource = body["parameter"][0]["resource"]
                assert patient_resource["address"][0]["city"] == "Boston"
                assert patient_resource["address"][0]["state"] == "MA"


class TestEpicAdapterGetDocuments:
    """Tests for EpicAdapter.get_documents()."""

    @pytest.mark.asyncio
    async def test_get_documents_returns_list(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter,
                "_request",
                new_callable=AsyncMock,
                return_value=SAMPLE_DOCUMENT_BUNDLE,
            ):
                docs = await epic_adapter.get_documents("patient-123")
                assert len(docs) == 1
                assert docs[0]["resourceType"] == "DocumentReference"

    @pytest.mark.asyncio
    async def test_get_documents_passes_category_param(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value={"resourceType": "Bundle", "entry": []})
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.get_documents("patient-123", category="clinical-note")
                _, kwargs = mock_req.call_args
                assert kwargs["params"]["category"] == "clinical-note"

    @pytest.mark.asyncio
    async def test_get_documents_date_range(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value={"resourceType": "Bundle", "entry": []})
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.get_documents(
                    "patient-123", date_from="2024-01-01", date_to="2024-06-30"
                )
                _, kwargs = mock_req.call_args
                date_param = kwargs["params"]["date"]
                assert "ge2024-01-01" in date_param
                assert "le2024-06-30" in date_param


class TestEpicAdapterGetAppointments:
    """Tests for EpicAdapter.get_appointments()."""

    @pytest.mark.asyncio
    async def test_get_appointments_returns_list(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter,
                "_request",
                new_callable=AsyncMock,
                return_value=SAMPLE_APPOINTMENT_BUNDLE,
            ):
                apts = await epic_adapter.get_appointments("patient-123")
                assert len(apts) == 1
                assert apts[0]["resourceType"] == "Appointment"
                assert apts[0]["_epicAppointmentType"] == "Office Visit"

    @pytest.mark.asyncio
    async def test_get_appointments_with_status_filter(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value={"resourceType": "Bundle", "entry": []})
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.get_appointments("patient-123", status="booked")
                _, kwargs = mock_req.call_args
                assert kwargs["params"]["status"] == "booked"

    @pytest.mark.asyncio
    async def test_get_appointments_date_range(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            mock_req = AsyncMock(return_value={"resourceType": "Bundle", "entry": []})
            with patch.object(epic_adapter, "_request", mock_req):
                await epic_adapter.get_appointments(
                    "patient-123", date_from="2024-01-01", date_to="2024-12-31"
                )
                _, kwargs = mock_req.call_args
                date_param = kwargs["params"]["date"]
                assert "ge2024-01-01" in date_param
                assert "le2024-12-31" in date_param


class TestEpicAdapterCareEverywhere:
    """Tests for EpicAdapter.query_care_everywhere()."""

    @pytest.mark.asyncio
    async def test_care_everywhere_returns_result(self, epic_adapter):
        care_result = {"resourceType": "Bundle", "total": 3}
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter, "_request", new_callable=AsyncMock, return_value=care_result
            ):
                result = await epic_adapter.query_care_everywhere("patient-123")
                assert result["total"] == 3

    @pytest.mark.asyncio
    async def test_care_everywhere_handles_failure_gracefully(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter,
                "_request",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Network error"),
            ):
                result = await epic_adapter.query_care_everywhere("patient-123")
                assert result["available"] is False
                assert "error" in result


class TestEpicAdapterMyChart:
    """Tests for EpicAdapter.get_mychart_status()."""

    @pytest.mark.asyncio
    async def test_mychart_status_active(self, epic_adapter):
        patient_with_mychart = {
            **SAMPLE_PATIENT,
            "extension": [
                {
                    "url": "http://open.epic.com/FHIR/StructureDefinition/extension/patient-mychart-status",
                    "valueBoolean": True,
                },
            ],
        }
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter,
                "_request",
                new_callable=AsyncMock,
                return_value=patient_with_mychart,
            ):
                status = await epic_adapter.get_mychart_status("patient-123")
                assert status["patient_id"] == "patient-123"
                assert status["mychart_active"] is True

    @pytest.mark.asyncio
    async def test_mychart_status_inactive_no_extension(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter, "_request", new_callable=AsyncMock, return_value=SAMPLE_PATIENT
            ):
                status = await epic_adapter.get_mychart_status("patient-123")
                assert status["mychart_active"] is False


class TestEpicAdapterBulkExport:
    """Tests for EpicAdapter.bulk_export() and check_bulk_export_status()."""

    @pytest.mark.asyncio
    async def test_bulk_export_initiation(self, epic_adapter):
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {"Content-Location": "https://fhir.epic.example.com/export/job-1"}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        epic_adapter._http_client = mock_client

        with _AUTH_PATCH_EPIC(epic_adapter):
            result = await epic_adapter.bulk_export(resource_types=["Patient", "Observation"])
            assert result["status"] == "in-progress"
            assert result["poll_url"] == "https://fhir.epic.example.com/export/job-1"

    @pytest.mark.asyncio
    async def test_bulk_export_failure(self, epic_adapter):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        epic_adapter._http_client = mock_client

        with _AUTH_PATCH_EPIC(epic_adapter):
            with pytest.raises(RuntimeError, match="Bulk export failed"):
                await epic_adapter.bulk_export()

    @pytest.mark.asyncio
    async def test_check_bulk_export_status_in_progress(self, epic_adapter):
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {"X-Progress": "50%"}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        epic_adapter._http_client = mock_client

        with _AUTH_PATCH_EPIC(epic_adapter):
            result = await epic_adapter.check_bulk_export_status("https://example.com/poll")
            assert result["status"] == "in-progress"
            assert result["progress"] == "50%"

    @pytest.mark.asyncio
    async def test_check_bulk_export_status_complete(self, epic_adapter):
        export_result = {
            "output": [
                {"type": "Patient", "url": "https://example.com/patient.ndjson"},
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = export_result

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        epic_adapter._http_client = mock_client

        with _AUTH_PATCH_EPIC(epic_adapter):
            result = await epic_adapter.check_bulk_export_status("https://example.com/poll")
            assert result["status"] == "complete"
            assert len(result["output"]) == 1

    @pytest.mark.asyncio
    async def test_check_bulk_export_status_error(self, epic_adapter):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        epic_adapter._http_client = mock_client

        with _AUTH_PATCH_EPIC(epic_adapter):
            result = await epic_adapter.check_bulk_export_status("https://example.com/poll")
            assert result["status"] == "error"


class TestEpicAdapterErrorHandling:
    """Tests for Epic adapter error handling."""

    @pytest.mark.asyncio
    async def test_request_raises_on_fhir_error(self, epic_adapter):
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter,
                "_request",
                new_callable=AsyncMock,
                side_effect=RuntimeError("FHIR OperationOutcome: resource not found"),
            ):
                with pytest.raises(RuntimeError, match="OperationOutcome"):
                    await epic_adapter.get_patient("nonexistent")

    @pytest.mark.asyncio
    async def test_search_with_no_entries_key(self, epic_adapter):
        """Bundle without 'entry' key should return empty list."""
        with _AUTH_PATCH_EPIC(epic_adapter):
            with patch.object(
                epic_adapter,
                "_request",
                new_callable=AsyncMock,
                return_value={"resourceType": "Bundle"},
            ):
                results = await epic_adapter.search_patients(family="Ghost")
                assert results == []

    @pytest.mark.asyncio
    async def test_bulk_export_not_connected_raises(self, epic_adapter):
        """bulk_export raises RuntimeError when _http_client is None."""
        epic_adapter._http_client = None
        with _AUTH_PATCH_EPIC(epic_adapter):
            with pytest.raises(RuntimeError, match="Not connected"):
                await epic_adapter.bulk_export()


# ===================================================================
# Cerner Adapter Tests
# ===================================================================


class TestCernerAdapterGetPatient:
    """Tests for CernerAdapter.get_patient()."""

    @pytest.mark.asyncio
    async def test_get_patient_returns_fhir_resource(self, cerner_adapter):
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter, "_request", new_callable=AsyncMock, return_value=SAMPLE_PATIENT
            ):
                result = await cerner_adapter.get_patient("patient-123")
                assert result["resourceType"] == "Patient"
                assert result["id"] == "patient-123"

    @pytest.mark.asyncio
    async def test_get_patient_calls_correct_path(self, cerner_adapter):
        with _AUTH_PATCH_CERNER(cerner_adapter):
            mock_req = AsyncMock(return_value=SAMPLE_PATIENT)
            with patch.object(cerner_adapter, "_request", mock_req):
                await cerner_adapter.get_patient("cerner-pat-456")
                mock_req.assert_awaited_once_with("GET", "/Patient/cerner-pat-456")


class TestCernerAdapterSearchPatients:
    """Tests for CernerAdapter.search_patients()."""

    @pytest.mark.asyncio
    async def test_search_patients_returns_list(self, cerner_adapter):
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter, "_request", new_callable=AsyncMock, return_value=SAMPLE_BUNDLE
            ):
                results = await cerner_adapter.search_patients(family="Doe")
                assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_patients_passes_all_params(self, cerner_adapter):
        with _AUTH_PATCH_CERNER(cerner_adapter):
            mock_req = AsyncMock(return_value={"resourceType": "Bundle", "entry": []})
            with patch.object(cerner_adapter, "_request", mock_req):
                await cerner_adapter.search_patients(
                    family="Smith", given="Anna", birthdate="1992-05-10", identifier="ID999"
                )
                _, kwargs = mock_req.call_args
                assert kwargs["params"]["family"] == "Smith"
                assert kwargs["params"]["given"] == "Anna"
                assert kwargs["params"]["birthdate"] == "1992-05-10"
                assert kwargs["params"]["identifier"] == "ID999"

    @pytest.mark.asyncio
    async def test_search_patients_empty_result(self, cerner_adapter):
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter,
                "_request",
                new_callable=AsyncMock,
                return_value={"resourceType": "Bundle", "entry": []},
            ):
                results = await cerner_adapter.search_patients(family="Nobody")
                assert results == []

    @pytest.mark.asyncio
    async def test_search_patients_missing_entry_key(self, cerner_adapter):
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter,
                "_request",
                new_callable=AsyncMock,
                return_value={"resourceType": "Bundle"},
            ):
                results = await cerner_adapter.search_patients(family="Ghost")
                assert results == []


class TestCernerAdapterGetPatientRecords:
    """Tests for CernerAdapter.get_patient_records() async iterator."""

    @pytest.mark.asyncio
    async def test_get_patient_records_yields_resources(self, cerner_adapter):
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Condition", "id": "c1"}},
                {"resource": {"resourceType": "Observation", "id": "o1"}},
            ],
            "link": [],
        }
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter, "_request", new_callable=AsyncMock, return_value=bundle
            ):
                records = []
                async for rec in cerner_adapter.get_patient_records("patient-123"):
                    records.append(rec)
                assert len(records) == 2
                assert records[0]["resourceType"] == "Condition"
                assert records[1]["resourceType"] == "Observation"

    @pytest.mark.asyncio
    async def test_get_patient_records_filters_by_resource_type(self, cerner_adapter):
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Condition", "id": "c1"}},
                {"resource": {"resourceType": "Observation", "id": "o1"}},
                {"resource": {"resourceType": "Condition", "id": "c2"}},
            ],
            "link": [],
        }
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter, "_request", new_callable=AsyncMock, return_value=bundle
            ):
                records = []
                async for rec in cerner_adapter.get_patient_records(
                    "patient-123", resource_types=["Condition"]
                ):
                    records.append(rec)
                assert len(records) == 2
                assert all(r["resourceType"] == "Condition" for r in records)

    @pytest.mark.asyncio
    async def test_get_patient_records_pagination(self, cerner_adapter):
        page1 = {
            "resourceType": "Bundle",
            "entry": [{"resource": {"resourceType": "Observation", "id": "o1"}}],
            "link": [{"relation": "next", "url": "https://fhir.cerner.example.com/R4/next-page"}],
        }
        page2 = {
            "resourceType": "Bundle",
            "entry": [{"resource": {"resourceType": "Observation", "id": "o2"}}],
            "link": [],
        }

        with _AUTH_PATCH_CERNER(cerner_adapter):
            mock_req = AsyncMock(side_effect=[page1, page2])
            with patch.object(cerner_adapter, "_request", mock_req):
                records = []
                async for rec in cerner_adapter.get_patient_records("patient-123"):
                    records.append(rec)
                assert len(records) == 2
                assert records[0]["id"] == "o1"
                assert records[1]["id"] == "o2"
                assert mock_req.await_count == 2

    @pytest.mark.asyncio
    async def test_get_patient_records_empty_bundle(self, cerner_adapter):
        bundle = {"resourceType": "Bundle", "entry": [], "link": []}
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter, "_request", new_callable=AsyncMock, return_value=bundle
            ):
                records = []
                async for rec in cerner_adapter.get_patient_records("patient-123"):
                    records.append(rec)
                assert records == []


class TestCernerAdapterErrorHandling:
    """Tests for Cerner adapter error handling."""

    @pytest.mark.asyncio
    async def test_request_raises_on_error(self, cerner_adapter):
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter,
                "_request",
                new_callable=AsyncMock,
                side_effect=RuntimeError("FHIR OperationOutcome: forbidden"),
            ):
                with pytest.raises(RuntimeError, match="forbidden"):
                    await cerner_adapter.get_patient("no-access")

    @pytest.mark.asyncio
    async def test_search_raises_on_network_error(self, cerner_adapter):
        with _AUTH_PATCH_CERNER(cerner_adapter):
            with patch.object(
                cerner_adapter,
                "_request",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Connection refused"),
            ):
                with pytest.raises(ConnectionError):
                    await cerner_adapter.search_patients(family="Doe")
