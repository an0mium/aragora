"""Tests for the adversarial demo handler.

Validates:
1. Demo endpoint returns valid response with topic
2. Demo includes opposing positions from agents
3. Demo includes consensus result
4. Demo includes decision receipt with checksum
5. Demo works in offline mode with mock agents
6. Returns 400 for missing topic
7. Calibration impact is included when include_calibration=True
8. Status endpoint returns demo results by ID
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.demo.adversarial_demo import (
    _compute_calibration_impact,
    _compute_consensus,
    _demo_store,
    _generate_receipt,
    _is_demo_mode,
    _run_demo_debate,
    _select_agents,
    _sha256,
    handle_adversarial_demo,
    handle_demo_status,
    register_routes,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_demo_store():
    """Clear the in-memory demo store between tests."""
    _demo_store.clear()
    yield
    _demo_store.clear()


def _make_request(body: dict | None = None, *, match_info: dict | None = None):
    """Build a mock aiohttp request with optional JSON body and match_info."""
    request = MagicMock()

    if body is not None:
        request.json = AsyncMock(return_value=body)
    else:
        request.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))

    request.match_info = match_info or {}
    return request


async def _parse_json_response(response) -> dict:
    """Parse JSON from an aiohttp web.Response."""
    return json.loads(response.body)


# ============================================================================
# 1. Demo endpoint returns valid response with topic
# ============================================================================


class TestAdversarialDemoEndpoint:
    """Core happy-path tests for POST /api/v1/demo/adversarial."""

    @pytest.mark.asyncio
    async def test_valid_topic_returns_completed(self):
        """Submitting a valid topic returns status=completed and a demo_id."""
        request = _make_request({"topic": "Should we use microservices or monolith?"})
        response = await handle_adversarial_demo(request)

        assert response.status == 200
        data = await _parse_json_response(response)

        assert data["status"] == "completed"
        assert data["demo_id"].startswith("demo_")
        assert "debate" in data
        assert "receipt" in data
        assert data["debate"]["topic"] == "Should we use microservices or monolith?"

    @pytest.mark.asyncio
    async def test_default_agent_count_is_three(self):
        """Without agent_count, three agents participate."""
        request = _make_request({"topic": "Test topic"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        positions = data["debate"]["positions"]
        assert len(positions) == 3

    @pytest.mark.asyncio
    async def test_custom_agent_count(self):
        """Agent count is respected within clamped bounds."""
        request = _make_request({"topic": "Scale", "agent_count": 5})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert len(data["debate"]["positions"]) == 5

    @pytest.mark.asyncio
    async def test_agent_count_clamped_lower(self):
        """Agent count below 2 is clamped to 2."""
        request = _make_request({"topic": "Scale", "agent_count": 0})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert len(data["debate"]["positions"]) == 2

    @pytest.mark.asyncio
    async def test_agent_count_clamped_upper(self):
        """Agent count above 6 is clamped to 6."""
        request = _make_request({"topic": "Scale", "agent_count": 99})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert len(data["debate"]["positions"]) == 6

    @pytest.mark.asyncio
    async def test_custom_round_count(self):
        """Rounds parameter controls the number of debate rounds."""
        request = _make_request({"topic": "Rounds test", "rounds": 4})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert len(data["debate"]["rounds"]) == 4

    @pytest.mark.asyncio
    async def test_elapsed_seconds_present(self):
        """Response includes elapsed execution time."""
        request = _make_request({"topic": "Timing"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert "elapsed_seconds" in data
        assert isinstance(data["elapsed_seconds"], float)
        assert data["elapsed_seconds"] >= 0


# ============================================================================
# 2. Demo includes opposing positions from agents
# ============================================================================


class TestOpposingPositions:
    """Positions must include both pro and con viewpoints."""

    @pytest.mark.asyncio
    async def test_positions_include_pro_and_con(self):
        """At least one pro and one con agent participate."""
        request = _make_request({"topic": "Architecture choice"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        positions = data["debate"]["positions"]
        biases = {v["bias"] for v in positions.values()}

        assert "pro" in biases, "No pro-bias agent in positions"
        assert "con" in biases, "No con-bias agent in positions"

    @pytest.mark.asyncio
    async def test_positions_have_required_fields(self):
        """Each position entry has position text, confidence, and calibration weight."""
        request = _make_request({"topic": "Field check"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        for name, pos in data["debate"]["positions"].items():
            assert "position" in pos, f"Missing 'position' for {name}"
            assert "confidence" in pos, f"Missing 'confidence' for {name}"
            assert "calibration_weight" in pos, f"Missing 'calibration_weight' for {name}"
            assert isinstance(pos["confidence"], float)
            assert 0 <= pos["confidence"] <= 1
            assert pos["calibration_weight"] > 0

    @pytest.mark.asyncio
    async def test_rounds_contain_proposals_and_critiques(self):
        """Each debate round has proposals and critiques."""
        request = _make_request({"topic": "Round structure", "rounds": 2})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        for rnd in data["debate"]["rounds"]:
            assert "proposals" in rnd
            assert "critiques" in rnd
            assert len(rnd["proposals"]) > 0
            assert len(rnd["critiques"]) > 0

    @pytest.mark.asyncio
    async def test_critiques_reference_other_agents(self):
        """Critiques target a different agent than the critic."""
        request = _make_request({"topic": "Cross-critique"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        for rnd in data["debate"]["rounds"]:
            for critique in rnd["critiques"]:
                assert critique["critic"] != critique["target"]


# ============================================================================
# 3. Demo includes consensus result
# ============================================================================


class TestConsensusResult:
    """Consensus output must have required structure."""

    @pytest.mark.asyncio
    async def test_consensus_has_required_keys(self):
        """Consensus includes reached, confidence, threshold_used, winner."""
        request = _make_request({"topic": "Consensus check"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        consensus = data["debate"]["consensus"]

        assert "reached" in consensus
        assert "confidence" in consensus
        assert "threshold_used" in consensus
        assert "winner" in consensus
        assert isinstance(consensus["reached"], bool)
        assert isinstance(consensus["confidence"], float)
        assert isinstance(consensus["threshold_used"], float)

    @pytest.mark.asyncio
    async def test_consensus_threshold_between_bounds(self):
        """Adaptive threshold stays within [0.45, 0.65]."""
        request = _make_request({"topic": "Threshold bounds"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        threshold = data["debate"]["consensus"]["threshold_used"]
        assert 0.45 <= threshold <= 0.65

    @pytest.mark.asyncio
    async def test_synthesis_mentions_topic(self):
        """The synthesis text should reference the original topic."""
        topic = "container orchestration strategy"
        request = _make_request({"topic": topic})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert topic in data["debate"]["synthesis"]


# ============================================================================
# 4. Demo includes decision receipt with checksum
# ============================================================================


class TestDecisionReceipt:
    """Receipt must contain receipt_id, checksum, decision, evidence_chain."""

    @pytest.mark.asyncio
    async def test_receipt_has_required_fields(self):
        """Receipt includes receipt_id, checksum, decision, evidence_chain."""
        request = _make_request({"topic": "Receipt test"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        receipt = data["receipt"]

        assert receipt["receipt_id"].startswith("rcpt_")
        assert isinstance(receipt["checksum"], str)
        assert len(receipt["checksum"]) == 64  # SHA-256 hex length
        assert isinstance(receipt["decision"], str)
        assert isinstance(receipt["evidence_chain"], list)
        assert len(receipt["evidence_chain"]) > 0
        assert "timestamp" in receipt

    @pytest.mark.asyncio
    async def test_evidence_chain_contains_position_events(self):
        """Evidence chain includes a 'position' event per agent."""
        request = _make_request({"topic": "Evidence chain", "agent_count": 3})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        chain = data["receipt"]["evidence_chain"]
        position_events = [e for e in chain if e["event_type"] == "position"]
        assert len(position_events) == 3

    @pytest.mark.asyncio
    async def test_evidence_chain_contains_consensus_event(self):
        """Evidence chain includes a final 'consensus' event."""
        request = _make_request({"topic": "Consensus event"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        chain = data["receipt"]["evidence_chain"]
        consensus_events = [e for e in chain if e["event_type"] == "consensus"]
        assert len(consensus_events) == 1

    @pytest.mark.asyncio
    async def test_evidence_entries_have_hashes(self):
        """Each evidence entry has a non-empty evidence_hash."""
        request = _make_request({"topic": "Hash check"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        for entry in data["receipt"]["evidence_chain"]:
            assert "evidence_hash" in entry
            assert len(entry["evidence_hash"]) == 64

    @pytest.mark.asyncio
    async def test_checksum_is_valid_sha256(self):
        """Receipt checksum is a valid hex SHA-256 digest."""
        request = _make_request({"topic": "Checksum validity"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        checksum = data["receipt"]["checksum"]
        # SHA-256 produces 64 hex chars
        assert len(checksum) == 64
        int(checksum, 16)  # Should not raise


# ============================================================================
# 5. Demo works in offline mode with mock agents
# ============================================================================


class TestOfflineMode:
    """Handler functions correctly with ARAGORA_OFFLINE or DEMO_MODE."""

    @pytest.mark.asyncio
    async def test_offline_mode_detected_with_aragora_offline(self):
        """_is_demo_mode returns True when ARAGORA_OFFLINE is set."""
        with patch.dict("os.environ", {"ARAGORA_OFFLINE": "1"}):
            assert _is_demo_mode() is True

    @pytest.mark.asyncio
    async def test_offline_mode_detected_with_demo_mode(self):
        """_is_demo_mode returns True when DEMO_MODE is set."""
        with patch.dict("os.environ", {"DEMO_MODE": "true"}):
            assert _is_demo_mode() is True

    @pytest.mark.asyncio
    async def test_offline_mode_false_without_env(self):
        """_is_demo_mode returns False with no env vars."""
        with patch.dict("os.environ", {}, clear=True):
            assert _is_demo_mode() is False

    @pytest.mark.asyncio
    async def test_demo_runs_in_offline_mode(self):
        """Full demo executes successfully with ARAGORA_OFFLINE set."""
        with patch.dict("os.environ", {"ARAGORA_OFFLINE": "1"}):
            request = _make_request({"topic": "Offline test"})
            response = await handle_adversarial_demo(request)

            assert response.status == 200
            data = await _parse_json_response(response)
            assert data["status"] == "completed"
            assert len(data["debate"]["positions"]) == 3

    @pytest.mark.asyncio
    async def test_demo_runs_in_demo_mode(self):
        """Full demo executes successfully with DEMO_MODE set."""
        with patch.dict("os.environ", {"DEMO_MODE": "1"}):
            request = _make_request({"topic": "Demo mode test"})
            response = await handle_adversarial_demo(request)

            assert response.status == 200
            data = await _parse_json_response(response)
            assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_deterministic_agents_for_same_topic(self):
        """Same topic produces the same agent selection (deterministic seed)."""
        agents_a = _select_agents(3, "Determinism test topic")
        agents_b = _select_agents(3, "Determinism test topic")

        names_a = [a["name"] for a in agents_a]
        names_b = [a["name"] for a in agents_b]
        assert names_a == names_b

    @pytest.mark.asyncio
    async def test_different_topics_may_produce_different_agents(self):
        """Different topics can produce different agent selections."""
        agents_a = _select_agents(3, "Topic Alpha")
        agents_b = _select_agents(3, "Topic Zeta completely unrelated")

        names_a = [a["name"] for a in agents_a]
        names_b = [a["name"] for a in agents_b]
        # Not guaranteed different, but very likely with different seeds
        # We just verify the function runs without error
        assert len(names_a) == 3
        assert len(names_b) == 3


# ============================================================================
# 6. Returns 400 for missing topic
# ============================================================================


class TestInputValidation:
    """Request validation and error handling."""

    @pytest.mark.asyncio
    async def test_missing_topic_returns_400(self):
        """Request without 'topic' key returns 400."""
        request = _make_request({"agent_count": 3})
        response = await handle_adversarial_demo(request)

        assert response.status == 400
        data = await _parse_json_response(response)
        assert "error" in data
        assert "topic" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_topic_returns_400(self):
        """Empty string topic returns 400."""
        request = _make_request({"topic": ""})
        response = await handle_adversarial_demo(request)

        assert response.status == 400
        data = await _parse_json_response(response)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_whitespace_only_topic_returns_400(self):
        """Whitespace-only topic returns 400."""
        request = _make_request({"topic": "   "})
        response = await handle_adversarial_demo(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        """Non-JSON request body returns 400."""
        request = _make_request(None)  # Triggers JSONDecodeError
        response = await handle_adversarial_demo(request)

        assert response.status == 400
        data = await _parse_json_response(response)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_non_dict_body_returns_400(self):
        """JSON body that is not an object returns 400."""
        request = MagicMock()
        request.json = AsyncMock(return_value=["not", "a", "dict"])
        response = await handle_adversarial_demo(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_non_string_topic_returns_400(self):
        """Numeric topic returns 400."""
        request = _make_request({"topic": 42})
        response = await handle_adversarial_demo(request)

        assert response.status == 400


# ============================================================================
# 7. Calibration impact included when include_calibration=True
# ============================================================================


class TestCalibrationImpact:
    """Calibration impact analysis in demo response."""

    @pytest.mark.asyncio
    async def test_calibration_included_by_default(self):
        """include_calibration defaults to True."""
        request = _make_request({"topic": "Default calibration"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert "calibration_impact" in data

    @pytest.mark.asyncio
    async def test_calibration_included_when_explicit_true(self):
        """Explicit include_calibration=true includes the section."""
        request = _make_request(
            {
                "topic": "Explicit calibration",
                "include_calibration": True,
            }
        )
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert "calibration_impact" in data

    @pytest.mark.asyncio
    async def test_calibration_excluded_when_false(self):
        """include_calibration=false omits calibration_impact."""
        request = _make_request(
            {
                "topic": "No calibration",
                "include_calibration": False,
            }
        )
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert "calibration_impact" not in data

    @pytest.mark.asyncio
    async def test_calibration_impact_has_required_fields(self):
        """Calibration impact includes threshold_adjustment, weight range, explanation."""
        request = _make_request({"topic": "Calibration fields"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        cal = data["calibration_impact"]

        assert "threshold_adjustment" in cal
        assert "vote_weight_range" in cal
        assert "average_calibration" in cal
        assert "explanation" in cal

        assert isinstance(cal["vote_weight_range"], list)
        assert len(cal["vote_weight_range"]) == 2
        assert cal["vote_weight_range"][0] <= cal["vote_weight_range"][1]

    @pytest.mark.asyncio
    async def test_calibration_explanation_is_descriptive(self):
        """Explanation mentions calibration percentages and thresholds."""
        request = _make_request({"topic": "Calibration explanation"})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        explanation = data["calibration_impact"]["explanation"]
        assert "calibration" in explanation.lower()
        assert "threshold" in explanation.lower()


# ============================================================================
# 8. Status endpoint returns demo results by ID
# ============================================================================


class TestDemoStatusEndpoint:
    """GET /api/v1/demo/adversarial/status/{demo_id}."""

    @pytest.mark.asyncio
    async def test_status_returns_stored_demo(self):
        """Status endpoint returns a previously-run demo by ID."""
        # First, run a demo
        run_request = _make_request({"topic": "Status lookup"})
        run_response = await handle_adversarial_demo(run_request)
        run_data = await _parse_json_response(run_response)
        demo_id = run_data["demo_id"]

        # Now query status
        status_request = _make_request(match_info={"demo_id": demo_id})
        status_response = await handle_demo_status(status_request)

        assert status_response.status == 200
        status_data = await _parse_json_response(status_response)
        assert status_data["demo_id"] == demo_id
        assert status_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_status_not_found(self):
        """Unknown demo_id returns 404."""
        request = _make_request(match_info={"demo_id": "demo_nonexistent"})
        response = await handle_demo_status(request)

        assert response.status == 404
        data = await _parse_json_response(response)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_status_missing_demo_id(self):
        """Empty demo_id returns 400."""
        request = _make_request(match_info={"demo_id": ""})
        response = await handle_demo_status(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_status_preserves_full_response(self):
        """Stored demo contains debate, receipt, and calibration_impact."""
        run_request = _make_request({"topic": "Full payload check"})
        run_response = await handle_adversarial_demo(run_request)
        run_data = await _parse_json_response(run_response)
        demo_id = run_data["demo_id"]

        status_request = _make_request(match_info={"demo_id": demo_id})
        status_response = await handle_demo_status(status_request)
        status_data = await _parse_json_response(status_response)

        assert "debate" in status_data
        assert "receipt" in status_data
        assert "calibration_impact" in status_data


# ============================================================================
# Unit tests for internal helpers
# ============================================================================


class TestInternalHelpers:
    """Unit tests for helper functions."""

    def test_sha256_deterministic(self):
        """Same input produces the same hash."""
        assert _sha256("hello") == _sha256("hello")
        assert _sha256("hello") != _sha256("world")

    def test_sha256_length(self):
        """SHA-256 hex digest is 64 characters."""
        assert len(_sha256("test")) == 64

    def test_select_agents_guarantees_diversity(self):
        """_select_agents always includes at least one pro and one con."""
        for count in (2, 3, 4, 5, 6):
            agents = _select_agents(count, f"diversity-{count}")
            biases = {a["bias"] for a in agents}
            assert "pro" in biases, f"No pro agent for count={count}"
            assert "con" in biases, f"No con agent for count={count}"

    def test_select_agents_assigns_calibration_weights(self):
        """Every selected agent has a calibration_weight."""
        agents = _select_agents(3, "weight check")
        for agent in agents:
            assert "calibration_weight" in agent
            assert agent["calibration_weight"] > 0

    @pytest.mark.asyncio
    async def test_run_demo_debate_returns_complete_result(self):
        """_run_demo_debate returns all required top-level keys."""
        result = await _run_demo_debate(
            demo_id="demo_test",
            topic="Integration test",
            agent_count=3,
            rounds=2,
            include_calibration=True,
        )

        assert result["demo_id"] == "demo_test"
        assert result["status"] == "completed"
        assert "debate" in result
        assert "receipt" in result
        assert "calibration_impact" in result

    def test_register_routes_adds_endpoints(self):
        """register_routes adds both POST and GET routes."""
        app = MagicMock()
        register_routes(app)

        calls = app.router.add_post.call_args_list + app.router.add_get.call_args_list
        paths = [call[0][0] for call in calls]
        assert "/api/v1/demo/adversarial" in paths
        assert "/api/v1/demo/adversarial/status/{demo_id}" in paths


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_long_topic_handled(self):
        """Extremely long topic string does not crash the handler."""
        long_topic = "Should we " + "really " * 500 + "do this?"
        request = _make_request({"topic": long_topic})
        response = await handle_adversarial_demo(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_special_characters_in_topic(self):
        """Topics with special characters are handled safely."""
        request = _make_request({"topic": "What about <script>alert('xss')</script>?"})
        response = await handle_adversarial_demo(request)

        assert response.status == 200
        data = await _parse_json_response(response)
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_single_round_debate(self):
        """A single-round debate still produces valid output."""
        request = _make_request({"topic": "Quick debate", "rounds": 1})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert len(data["debate"]["rounds"]) == 1
        assert data["debate"]["consensus"]["reached"] is not None

    @pytest.mark.asyncio
    async def test_two_agent_debate(self):
        """Minimum agent count (2) produces a valid debate."""
        request = _make_request({"topic": "Minimal agents", "agent_count": 2})
        response = await handle_adversarial_demo(request)

        data = await _parse_json_response(response)
        assert len(data["debate"]["positions"]) == 2
        biases = {v["bias"] for v in data["debate"]["positions"].values()}
        assert "pro" in biases
        assert "con" in biases

    @pytest.mark.asyncio
    async def test_demo_store_bounded(self):
        """Store evicts old entries when capacity is exceeded."""
        from aragora.server.handlers.demo.adversarial_demo import (
            _MAX_STORED_DEMOS,
            _store_demo,
        )

        for i in range(_MAX_STORED_DEMOS + 5):
            _store_demo(f"demo_{i}", {"id": i})

        assert len(_demo_store) <= _MAX_STORED_DEMOS
