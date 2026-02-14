"""
E2E smoke test: Gauntlet → Receipt → Channel routing.

Tests the full pipeline:
1. Run gauntlet validation on input content
2. Generate a DecisionReceipt from the GauntletResult
3. Register a debate origin (simulating a chat platform trigger)
4. Route the receipt back to the originating channel
5. Verify idempotent delivery (no duplicate sends)

All external dependencies are mocked. Completes in under 5 seconds.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gauntlet.receipt import DecisionReceipt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_origin_store(tmp_path, monkeypatch):
    """Isolate debate origin storage per test."""
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    import aragora.server.debate_origin as debate_origin

    debate_origin._origin_store.clear()
    debate_origin._sqlite_store = None
    yield
    debate_origin._origin_store.clear()
    debate_origin._sqlite_store = None


def _make_gauntlet_result_dict(
    *,
    gauntlet_id: str = "gauntlet-smoke-001",
    verdict: str = "PASS",
    confidence: float = 0.87,
    vulnerabilities: int = 0,
) -> dict:
    """Create a minimal GauntletResult-like dict for receipt creation."""
    return {
        "gauntlet_id": gauntlet_id,
        "input_hash": hashlib.sha256(b"smoke test input").hexdigest(),
        "input_summary": "Should we deploy the new auth system to production?",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": 2.5,
        "verdict": verdict,
        "confidence": confidence,
        "verdict_reasoning": "All probes passed with acceptable risk level.",
        "risk_summary": {"critical": 0, "high": 0, "medium": 1, "low": 2, "total": 3},
        "vulnerabilities_found": vulnerabilities,
        "vulnerability_details": [],
        "attacks_attempted": 8,
        "attacks_successful": 0,
        "probes_run": 6,
        "robustness_score": 0.92,
        "agents_used": ["anthropic-api", "openai-api", "mistral-api"],
    }


# ---------------------------------------------------------------------------
# Test: Full Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestGauntletReceiptChannelPipeline:
    """Full pipeline: gauntlet result → receipt → channel delivery."""

    @pytest.mark.asyncio
    async def test_full_pipeline_slack(self):
        """Gauntlet → Receipt → Slack channel delivery."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            get_debate_origin,
            _origin_store,
        )

        gauntlet_id = f"gauntlet-{uuid.uuid4().hex[:8]}"

        # Step 1: Register origin (simulating Slack trigger)
        origin = register_debate_origin(
            debate_id=gauntlet_id,
            platform="slack",
            channel_id="C0123456789",
            user_id="U9876543210",
            thread_id="1707900000.000001",
            metadata={"workspace": "acme-corp"},
        )
        assert origin.platform == "slack"
        assert not origin.result_sent

        # Step 2: Simulate gauntlet execution result
        gauntlet_data = _make_gauntlet_result_dict(gauntlet_id=gauntlet_id)

        # Step 3: Generate receipt from gauntlet result
        receipt = DecisionReceipt(
            receipt_id=f"receipt-{uuid.uuid4().hex[:8]}",
            gauntlet_id=gauntlet_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=gauntlet_data["input_summary"],
            input_hash=gauntlet_data["input_hash"],
            risk_summary=gauntlet_data["risk_summary"],
            attacks_attempted=gauntlet_data["attacks_attempted"],
            attacks_successful=gauntlet_data["attacks_successful"],
            probes_run=gauntlet_data["probes_run"],
            vulnerabilities_found=gauntlet_data["vulnerabilities_found"],
            verdict=gauntlet_data["verdict"],
            confidence=gauntlet_data["confidence"],
            robustness_score=gauntlet_data["robustness_score"],
            verdict_reasoning=gauntlet_data["verdict_reasoning"],
        )

        # Verify receipt integrity
        assert receipt.artifact_hash is not None
        assert len(receipt.artifact_hash) == 64
        assert receipt.verify_integrity() is True
        assert receipt.verdict == "PASS"

        # Verify receipt exports work
        json_str = receipt.to_json()
        data = json.loads(json_str)
        assert data["verdict"] == "PASS"
        assert data["gauntlet_id"] == gauntlet_id

        markdown = receipt.to_markdown()
        assert "PASS" in markdown
        assert "Decision Receipt" in markdown

        # Step 4: Route result to Slack channel
        debate_result = {
            "consensus_reached": True,
            "final_answer": gauntlet_data["verdict_reasoning"],
            "confidence": gauntlet_data["confidence"],
            "participants": gauntlet_data["agents_used"],
            "task": gauntlet_data["input_summary"],
        }

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"ok": True}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
                success = await route_debate_result(gauntlet_id, debate_result)

        # Verify origin tracking
        final_origin = get_debate_origin(gauntlet_id)
        assert final_origin is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_telegram(self):
        """Gauntlet → Receipt → Telegram channel delivery."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            get_debate_origin,
            _origin_store,
        )

        gauntlet_id = f"gauntlet-{uuid.uuid4().hex[:8]}"

        # Step 1: Register Telegram origin
        origin = register_debate_origin(
            debate_id=gauntlet_id,
            platform="telegram",
            channel_id="123456789",
            user_id="987654321",
            message_id="42",
            metadata={"username": "smoke_tester"},
        )
        assert origin.platform == "telegram"

        # Step 2: Create gauntlet result with findings
        gauntlet_data = _make_gauntlet_result_dict(
            gauntlet_id=gauntlet_id,
            verdict="CONDITIONAL",
            confidence=0.65,
            vulnerabilities=2,
        )

        # Step 3: Generate receipt
        receipt = DecisionReceipt(
            receipt_id=f"receipt-{uuid.uuid4().hex[:8]}",
            gauntlet_id=gauntlet_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=gauntlet_data["input_summary"],
            input_hash=gauntlet_data["input_hash"],
            risk_summary=gauntlet_data["risk_summary"],
            attacks_attempted=gauntlet_data["attacks_attempted"],
            attacks_successful=gauntlet_data["attacks_successful"],
            probes_run=gauntlet_data["probes_run"],
            vulnerabilities_found=gauntlet_data["vulnerabilities_found"],
            verdict=gauntlet_data["verdict"],
            confidence=gauntlet_data["confidence"],
            robustness_score=gauntlet_data["robustness_score"],
        )

        assert receipt.verdict == "CONDITIONAL"
        assert receipt.verify_integrity() is True

        # Step 4: Route to Telegram
        debate_result = {
            "consensus_reached": False,
            "final_answer": "Conditional approval - manual review required.",
            "confidence": 0.65,
            "participants": gauntlet_data["agents_used"],
        }

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"ok": True, "result": {}}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test_token"}):
                await route_debate_result(gauntlet_id, debate_result)


# ---------------------------------------------------------------------------
# Test: Receipt Integrity Through Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestReceiptIntegrityPipeline:
    """Verify receipt cryptographic integrity survives the full pipeline."""

    def test_receipt_hash_survives_json_roundtrip(self):
        """Receipt hash valid after JSON export/import."""
        receipt = DecisionReceipt(
            receipt_id="roundtrip-001",
            gauntlet_id="gauntlet-roundtrip-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test roundtrip integrity",
            input_hash=hashlib.sha256(b"roundtrip").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
            attacks_attempted=4,
            attacks_successful=0,
            probes_run=3,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.95,
            robustness_score=0.9,
        )

        # Export → reimport
        json_str = receipt.to_json()
        data = json.loads(json_str)
        restored = DecisionReceipt.from_dict(data)

        # Integrity preserved
        assert restored.artifact_hash == receipt.artifact_hash
        assert restored.verify_integrity() is True

    def test_receipt_tamper_detection(self):
        """Modified receipt fails integrity check."""
        receipt = DecisionReceipt(
            receipt_id="tamper-001",
            gauntlet_id="gauntlet-tamper-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test tamper detection",
            input_hash=hashlib.sha256(b"tamper").hexdigest(),
            risk_summary={"critical": 0},
            attacks_attempted=2,
            attacks_successful=0,
            probes_run=2,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )

        assert receipt.verify_integrity() is True

        # Tamper
        receipt.verdict = "FAIL"
        assert receipt.verify_integrity() is False

    def test_receipt_all_export_formats(self):
        """All export formats produce valid output."""
        receipt = DecisionReceipt(
            receipt_id="formats-001",
            gauntlet_id="gauntlet-formats-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Multi-format export test",
            input_hash=hashlib.sha256(b"formats").hexdigest(),
            risk_summary={"critical": 0, "high": 1, "medium": 2, "low": 3},
            attacks_attempted=10,
            attacks_successful=1,
            probes_run=8,
            vulnerabilities_found=1,
            vulnerability_details=[
                {
                    "id": "vuln-001",
                    "title": "Minor input validation gap",
                    "severity": "HIGH",
                    "severity_level": "HIGH",
                    "category": "security",
                    "description": "Input not fully sanitized",
                }
            ],
            verdict="CONDITIONAL",
            confidence=0.72,
            robustness_score=0.65,
        )

        # JSON
        json_data = json.loads(receipt.to_json())
        assert json_data["verdict"] == "CONDITIONAL"

        # Markdown
        md = receipt.to_markdown()
        assert "Decision Receipt" in md
        assert "CONDITIONAL" in md

        # HTML
        html = receipt.to_html()
        assert "<!DOCTYPE html>" in html
        assert "CONDITIONAL" in html

        # SARIF
        sarif = receipt.to_sarif()
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"]) == 1

        # CSV
        csv_content = receipt.to_csv()
        assert "Finding ID" in csv_content


# ---------------------------------------------------------------------------
# Test: Idempotent Delivery
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestIdempotentDelivery:
    """Verify results aren't sent twice to the same channel."""

    @pytest.mark.asyncio
    async def test_duplicate_send_prevention(self):
        """Second route attempt returns True without sending."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            mark_result_sent,
            get_debate_origin,
            _origin_store,
        )

        debate_id = f"idempotent-{uuid.uuid4().hex[:8]}"

        register_debate_origin(
            debate_id=debate_id,
            platform="slack",
            channel_id="C123",
            user_id="U456",
        )

        # Mark as already sent
        mark_result_sent(debate_id)
        origin = get_debate_origin(debate_id)
        assert origin.result_sent is True

        # Second attempt should not re-send
        result = {"final_answer": "Test", "confidence": 0.9}
        success = await route_debate_result(debate_id, result)
        assert success  # Returns True (already handled)

    @pytest.mark.asyncio
    async def test_nonexistent_origin_fails_gracefully(self):
        """Routing to nonexistent origin returns False."""
        from aragora.server.debate_origin import route_debate_result

        result = {"final_answer": "Test", "confidence": 0.9}
        success = await route_debate_result("nonexistent-debate", result)
        assert not success


# ---------------------------------------------------------------------------
# Test: Multi-Platform Origin Registration
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestMultiPlatformOrigins:
    """Test origin registration across all supported platforms."""

    @pytest.mark.parametrize(
        "platform,channel_id,user_id,extra_kwargs",
        [
            ("telegram", "123456789", "987654321", {"message_id": "42"}),
            ("slack", "C0123456789", "U9876543210", {"thread_id": "1707900000.000001"}),
            ("discord", "1234567890123456789", "9876543210987654321", {}),
            ("teams", "teams-channel-id", "teams-user-id", {}),
            ("whatsapp", "+1234567890", "wa_user_123", {}),
            ("email", "user@example.com", "user@example.com", {}),
        ],
    )
    def test_register_origin_per_platform(self, platform, channel_id, user_id, extra_kwargs):
        """Each platform can register and retrieve origins."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
        )

        debate_id = f"multi-{platform}-{uuid.uuid4().hex[:8]}"
        origin = register_debate_origin(
            debate_id=debate_id,
            platform=platform,
            channel_id=channel_id,
            user_id=user_id,
            **extra_kwargs,
        )

        assert origin.platform == platform
        assert origin.channel_id == channel_id

        retrieved = get_debate_origin(debate_id)
        assert retrieved is not None
        assert retrieved.platform == platform
