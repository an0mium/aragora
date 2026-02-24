"""Comprehensive tests for compliance status handler (aragora/server/handlers/compliance/status.py).

Covers:
- get_status() function: overall compliance status endpoint
- Score calculation with various control mixes
- Overall status determination (compliant, mostly_compliant, partial, non_compliant)
- Framework summary structure (soc2_type2, gdpr, hipaa)
- Controls summary (total, compliant, non_compliant)
- Audit date fields (last_audit, next_audit_due, generated_at)
- Edge cases: empty controls, all compliant, all non-compliant, single control
- Error propagation from evaluate_controls
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Module-level setup: inject evaluate_controls into soc2 before importing status
# ---------------------------------------------------------------------------

# The status module does `from .soc2 import evaluate_controls` but soc2.py only
# exposes `_evaluate_controls` as a method on SOC2Mixin. We need to make the
# attribute available on the soc2 module so the status module can import.

# Make sure soc2 module is loaded first
import aragora.server.handlers.compliance.soc2 as _soc2_mod

# Inject a placeholder that will be patched per-test
_default_evaluate_controls = AsyncMock(return_value=[])
if not hasattr(_soc2_mod, "evaluate_controls"):
    _soc2_mod.evaluate_controls = _default_evaluate_controls  # type: ignore[attr-defined]

# The status module's internal function is _evaluate_controls (with underscore).
# We need to patch that, not the non-underscored variant.

# Now we can safely import the status module
# Remove cached status module if present to force re-import with patched soc2
_status_key = "aragora.server.handlers.compliance.status"
if _status_key in sys.modules:
    del sys.modules[_status_key]

import aragora.server.handlers.compliance.status as status_mod  # noqa: E402

get_status = status_mod.get_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_controls(compliant: int, non_compliant: int) -> list[dict]:
    """Build a list of mock controls with the given counts."""
    controls = []
    for i in range(compliant):
        controls.append(
            {
                "control_id": f"C-{i}",
                "category": "Security",
                "name": f"Compliant Control {i}",
                "status": "compliant",
            }
        )
    for i in range(non_compliant):
        controls.append(
            {
                "control_id": f"NC-{i}",
                "category": "Security",
                "name": f"Non-Compliant Control {i}",
                "status": "non_compliant",
            }
        )
    return controls


@pytest.fixture(autouse=True)
def _reset_evaluate_controls():
    """Reset _evaluate_controls mock before each test to prevent cross-talk."""
    original = status_mod._evaluate_controls
    yield
    # Restore original after each test
    status_mod._evaluate_controls = original


# ---------------------------------------------------------------------------
# Tests: Basic response structure
# ---------------------------------------------------------------------------


class TestGetStatusBasicStructure:
    """Tests for basic response structure of get_status()."""

    @pytest.mark.asyncio
    async def test_returns_handler_result(self):
        """get_status returns a HandlerResult with status 200."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(10, 0))
        result = await get_status()
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_response_body_is_valid_json(self):
        """Response body should be valid JSON."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(8, 2))
        result = await get_status()
        body = _body(result)
        assert isinstance(body, dict)

    @pytest.mark.asyncio
    async def test_response_contains_required_top_level_keys(self):
        """Response must contain all top-level keys."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 5))
        result = await get_status()
        body = _body(result)
        required_keys = {
            "status",
            "compliance_score",
            "frameworks",
            "controls_summary",
            "last_audit",
            "next_audit_due",
            "generated_at",
        }
        assert required_keys.issubset(body.keys())

    @pytest.mark.asyncio
    async def test_content_type_is_json(self):
        """Response content type should be application/json."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        if hasattr(result, "content_type"):
            assert result.content_type == "application/json"


# ---------------------------------------------------------------------------
# Tests: Score calculation
# ---------------------------------------------------------------------------


class TestComplianceScoreCalculation:
    """Tests for compliance score calculation."""

    @pytest.mark.asyncio
    async def test_all_compliant_score_100(self):
        """100% compliant controls yields a score of 100."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(10, 0))
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 100

    @pytest.mark.asyncio
    async def test_all_non_compliant_score_0(self):
        """0% compliant controls yields a score of 0."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(0, 10))
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 0

    @pytest.mark.asyncio
    async def test_half_compliant_score_50(self):
        """50% compliant controls yields a score of 50."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 5))
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 50

    @pytest.mark.asyncio
    async def test_score_is_integer(self):
        """Compliance score should always be an integer."""
        # 3 out of 7 = 42.857... should truncate to 42
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(3, 4))
        result = await get_status()
        body = _body(result)
        assert isinstance(body["compliance_score"], int)
        assert body["compliance_score"] == 42

    @pytest.mark.asyncio
    async def test_score_truncates_not_rounds(self):
        """Score should truncate (int()), not round."""
        # 2 out of 3 = 66.666... should be 66, not 67
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(2, 1))
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 66

    @pytest.mark.asyncio
    async def test_empty_controls_score_0(self):
        """Empty controls list yields a score of 0."""
        status_mod._evaluate_controls = AsyncMock(return_value=[])
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 0

    @pytest.mark.asyncio
    async def test_single_compliant_control_score_100(self):
        """Single compliant control yields score 100."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(1, 0))
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 100

    @pytest.mark.asyncio
    async def test_single_non_compliant_control_score_0(self):
        """Single non-compliant control yields score 0."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(0, 1))
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 0

    @pytest.mark.asyncio
    async def test_score_95_boundary(self):
        """95% score (boundary for compliant status)."""
        # 19 out of 20 = 95%
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(19, 1))
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 95


# ---------------------------------------------------------------------------
# Tests: Overall status determination
# ---------------------------------------------------------------------------


class TestOverallStatusDetermination:
    """Tests for overall status thresholds."""

    @pytest.mark.asyncio
    async def test_status_compliant_at_95(self):
        """Score >= 95 yields 'compliant' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(19, 1))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "compliant"

    @pytest.mark.asyncio
    async def test_status_compliant_at_100(self):
        """Score = 100 yields 'compliant' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(10, 0))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "compliant"

    @pytest.mark.asyncio
    async def test_status_compliant_at_96(self):
        """Score of 96 yields 'compliant' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(96, 4))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "compliant"

    @pytest.mark.asyncio
    async def test_status_mostly_compliant_at_94(self):
        """Score of 94 yields 'mostly_compliant' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(94, 6))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "mostly_compliant"

    @pytest.mark.asyncio
    async def test_status_mostly_compliant_at_80(self):
        """Score of 80 yields 'mostly_compliant' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(80, 20))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "mostly_compliant"

    @pytest.mark.asyncio
    async def test_status_partial_at_79(self):
        """Score of 79 yields 'partial' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(79, 21))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "partial"

    @pytest.mark.asyncio
    async def test_status_partial_at_60(self):
        """Score of 60 yields 'partial' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(60, 40))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "partial"

    @pytest.mark.asyncio
    async def test_status_non_compliant_at_59(self):
        """Score of 59 yields 'non_compliant' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(59, 41))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "non_compliant"

    @pytest.mark.asyncio
    async def test_status_non_compliant_at_0(self):
        """Score of 0 yields 'non_compliant' status."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(0, 10))
        result = await get_status()
        body = _body(result)
        assert body["status"] == "non_compliant"

    @pytest.mark.asyncio
    async def test_empty_controls_yields_non_compliant(self):
        """Empty controls list should yield 'non_compliant' (score=0)."""
        status_mod._evaluate_controls = AsyncMock(return_value=[])
        result = await get_status()
        body = _body(result)
        assert body["status"] == "non_compliant"


# ---------------------------------------------------------------------------
# Tests: Frameworks summary
# ---------------------------------------------------------------------------


class TestFrameworksSummary:
    """Tests for the frameworks section of the response."""

    @pytest.mark.asyncio
    async def test_frameworks_contains_all_three(self):
        """Response should contain soc2_type2, gdpr, and hipaa frameworks."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        assert "soc2_type2" in body["frameworks"]
        assert "gdpr" in body["frameworks"]
        assert "hipaa" in body["frameworks"]

    @pytest.mark.asyncio
    async def test_soc2_controls_assessed_matches_total(self):
        """SOC 2 controls_assessed should match total controls count."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(7, 3))
        result = await get_status()
        body = _body(result)
        soc2 = body["frameworks"]["soc2_type2"]
        assert soc2["controls_assessed"] == 10

    @pytest.mark.asyncio
    async def test_soc2_controls_compliant_matches_count(self):
        """SOC 2 controls_compliant should match compliant controls count."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(7, 3))
        result = await get_status()
        body = _body(result)
        soc2 = body["frameworks"]["soc2_type2"]
        assert soc2["controls_compliant"] == 7

    @pytest.mark.asyncio
    async def test_soc2_status_is_in_progress(self):
        """SOC 2 framework status is always 'in_progress'."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(10, 0))
        result = await get_status()
        body = _body(result)
        assert body["frameworks"]["soc2_type2"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_gdpr_fields(self):
        """GDPR framework should have expected fields and values."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        gdpr = body["frameworks"]["gdpr"]
        assert gdpr["status"] == "supported"
        assert gdpr["data_export"] is True
        assert gdpr["consent_tracking"] is True
        assert gdpr["retention_policy"] is True

    @pytest.mark.asyncio
    async def test_hipaa_status_is_partial(self):
        """HIPAA status is always 'partial'."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        assert body["frameworks"]["hipaa"]["status"] == "partial"

    @pytest.mark.asyncio
    async def test_hipaa_note_mentions_phi(self):
        """HIPAA note should mention PHI configuration."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        assert "note" in body["frameworks"]["hipaa"]
        assert "PHI" in body["frameworks"]["hipaa"]["note"]

    @pytest.mark.asyncio
    async def test_soc2_with_zero_controls(self):
        """SOC 2 with zero controls should show 0 assessed and 0 compliant."""
        status_mod._evaluate_controls = AsyncMock(return_value=[])
        result = await get_status()
        body = _body(result)
        soc2 = body["frameworks"]["soc2_type2"]
        assert soc2["controls_assessed"] == 0
        assert soc2["controls_compliant"] == 0


# ---------------------------------------------------------------------------
# Tests: Controls summary
# ---------------------------------------------------------------------------


class TestControlsSummary:
    """Tests for the controls_summary section."""

    @pytest.mark.asyncio
    async def test_controls_summary_total(self):
        """controls_summary total matches total controls."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(6, 4))
        result = await get_status()
        body = _body(result)
        assert body["controls_summary"]["total"] == 10

    @pytest.mark.asyncio
    async def test_controls_summary_compliant(self):
        """controls_summary compliant count matches."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(6, 4))
        result = await get_status()
        body = _body(result)
        assert body["controls_summary"]["compliant"] == 6

    @pytest.mark.asyncio
    async def test_controls_summary_non_compliant(self):
        """controls_summary non_compliant count matches."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(6, 4))
        result = await get_status()
        body = _body(result)
        assert body["controls_summary"]["non_compliant"] == 4

    @pytest.mark.asyncio
    async def test_controls_summary_adds_up(self):
        """compliant + non_compliant should equal total."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(13, 7))
        result = await get_status()
        body = _body(result)
        summary = body["controls_summary"]
        assert summary["compliant"] + summary["non_compliant"] == summary["total"]

    @pytest.mark.asyncio
    async def test_empty_controls_summary(self):
        """Empty controls yield zero totals."""
        status_mod._evaluate_controls = AsyncMock(return_value=[])
        result = await get_status()
        body = _body(result)
        assert body["controls_summary"]["total"] == 0
        assert body["controls_summary"]["compliant"] == 0
        assert body["controls_summary"]["non_compliant"] == 0


# ---------------------------------------------------------------------------
# Tests: Audit dates
# ---------------------------------------------------------------------------


class TestAuditDates:
    """Tests for audit date fields."""

    @pytest.mark.asyncio
    async def test_last_audit_is_7_days_ago(self):
        """last_audit should be exactly 7 days before generated_at."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        last_audit = datetime.fromisoformat(body["last_audit"])
        diff = generated - last_audit
        assert diff.days == 7

    @pytest.mark.asyncio
    async def test_next_audit_is_83_days_ahead(self):
        """next_audit_due should be exactly 83 days after generated_at."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        next_audit = datetime.fromisoformat(body["next_audit_due"])
        diff = next_audit - generated
        assert diff.days == 83

    @pytest.mark.asyncio
    async def test_generated_at_is_recent(self):
        """generated_at should be close to now."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        before = datetime.now(timezone.utc)
        result = await get_status()
        after = datetime.now(timezone.utc)
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        assert before <= generated <= after

    @pytest.mark.asyncio
    async def test_dates_are_iso_format(self):
        """All date fields should be valid ISO 8601 strings."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        for key in ("last_audit", "next_audit_due", "generated_at"):
            dt = datetime.fromisoformat(body[key])
            assert dt is not None

    @pytest.mark.asyncio
    async def test_last_audit_before_generated_at(self):
        """last_audit must be strictly before generated_at."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        last_audit = datetime.fromisoformat(body["last_audit"])
        assert last_audit < generated

    @pytest.mark.asyncio
    async def test_next_audit_after_generated_at(self):
        """next_audit_due must be strictly after generated_at."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(5, 0))
        result = await get_status()
        body = _body(result)
        generated = datetime.fromisoformat(body["generated_at"])
        next_audit = datetime.fromisoformat(body["next_audit_due"])
        assert next_audit > generated


# ---------------------------------------------------------------------------
# Tests: Edge cases and special control mixes
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for get_status."""

    @pytest.mark.asyncio
    async def test_large_number_of_controls(self):
        """Handler works with a large number of controls."""
        status_mod._evaluate_controls = AsyncMock(return_value=_make_controls(950, 50))
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 95
        assert body["status"] == "compliant"
        assert body["controls_summary"]["total"] == 1000

    @pytest.mark.asyncio
    async def test_controls_with_unknown_status(self):
        """Controls with non-'compliant' statuses count as non-compliant."""
        controls = [
            {"control_id": "A", "status": "compliant"},
            {"control_id": "B", "status": "pending"},
            {"control_id": "C", "status": "unknown"},
            {"control_id": "D", "status": ""},
        ]
        status_mod._evaluate_controls = AsyncMock(return_value=controls)
        result = await get_status()
        body = _body(result)
        assert body["controls_summary"]["compliant"] == 1
        assert body["controls_summary"]["non_compliant"] == 3
        assert body["compliance_score"] == 25

    @pytest.mark.asyncio
    async def test_evaluate_controls_called_once(self):
        """_evaluate_controls should be called exactly once per invocation."""
        mock_eval = AsyncMock(return_value=_make_controls(5, 0))
        status_mod._evaluate_controls = mock_eval
        await get_status()
        mock_eval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate_controls_exception_propagates(self):
        """If evaluate_controls raises, the exception propagates up."""
        status_mod._evaluate_controls = AsyncMock(side_effect=RuntimeError("DB connection failed"))
        with pytest.raises(RuntimeError, match="DB connection failed"):
            await get_status()

    @pytest.mark.asyncio
    async def test_controls_with_mixed_case_status(self):
        """Only exact string 'compliant' counts; 'Compliant' does not."""
        controls = [
            {"control_id": "A", "status": "compliant"},
            {"control_id": "B", "status": "Compliant"},
            {"control_id": "C", "status": "COMPLIANT"},
        ]
        status_mod._evaluate_controls = AsyncMock(return_value=controls)
        result = await get_status()
        body = _body(result)
        # Only exact match "compliant" counts
        assert body["controls_summary"]["compliant"] == 1
        assert body["controls_summary"]["non_compliant"] == 2

    @pytest.mark.asyncio
    async def test_controls_with_extra_fields_ignored(self):
        """Extra fields on controls should not affect scoring."""
        controls = [
            {"control_id": "A", "status": "compliant", "extra_field": "data", "nested": {"a": 1}},
            {"control_id": "B", "status": "non_compliant", "tags": ["important"]},
        ]
        status_mod._evaluate_controls = AsyncMock(return_value=controls)
        result = await get_status()
        body = _body(result)
        assert body["compliance_score"] == 50
        assert body["controls_summary"]["total"] == 2


# ---------------------------------------------------------------------------
# Tests: Module-level attributes
# ---------------------------------------------------------------------------


class TestModuleAttributes:
    """Tests for module-level attributes and exports."""

    def test_all_exports_get_status(self):
        """__all__ should export get_status."""
        assert "get_status" in status_mod.__all__

    def test_get_status_is_callable(self):
        """get_status should be callable."""
        assert callable(status_mod.get_status)

    def test_only_get_status_in_all(self):
        """__all__ should contain exactly one entry."""
        assert len(status_mod.__all__) == 1
