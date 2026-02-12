"""
Tests for Debate API Hardening.

Covers:
1. OpenAPI request schema completeness for all debate endpoints
2. Cost estimation logic
3. WebSocket resume token generation, validation, and replay
"""

import time

import pytest
from unittest.mock import patch

# ============================================================================
# Schema Tests
# ============================================================================


class TestDebateRequestSchemas:
    """Verify all debate-related request schemas are defined."""

    def _get_schemas(self):
        from aragora.server.openapi.schemas.debate_requests import DEBATE_REQUEST_SCHEMAS

        return DEBATE_REQUEST_SCHEMAS

    def test_debate_join_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateJoinRequest" in schemas
        props = schemas["DebateJoinRequest"]["properties"]
        assert "role" in props
        assert props["role"]["enum"] == ["observer", "participant", "moderator"]

    def test_debate_vote_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateVoteRequest" in schemas
        props = schemas["DebateVoteRequest"]["properties"]
        assert "position" in props
        assert "intensity" in props
        assert props["intensity"]["minimum"] == 1
        assert props["intensity"]["maximum"] == 10

    def test_debate_suggestion_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateSuggestionRequest" in schemas
        assert schemas["DebateSuggestionRequest"]["required"] == ["content"]
        props = schemas["DebateSuggestionRequest"]["properties"]
        assert "type" in props
        assert "argument" in props["type"]["enum"]

    def test_debate_update_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateUpdateRequest" in schemas
        props = schemas["DebateUpdateRequest"]["properties"]
        assert "rounds" in props
        assert "consensus" in props

    def test_debate_fork_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateForkRequest" in schemas
        assert schemas["DebateForkRequest"]["required"] == ["branch_point"]

    def test_debate_broadcast_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateBroadcastRequest" in schemas
        props = schemas["DebateBroadcastRequest"]["properties"]
        assert "format" in props
        assert "voices" in props

    def test_debate_clone_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateCloneRequest" in schemas
        props = schemas["DebateCloneRequest"]["properties"]
        assert "preserveAgents" in props
        assert "preserveContext" in props

    def test_debate_followup_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateFollowupRequest" in schemas
        props = schemas["DebateFollowupRequest"]["properties"]
        assert "cruxId" in props

    def test_debate_evidence_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateEvidenceRequest" in schemas
        assert schemas["DebateEvidenceRequest"]["required"] == ["evidence"]

    def test_debate_verify_claim_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateVerifyClaimRequest" in schemas
        assert schemas["DebateVerifyClaimRequest"]["required"] == ["claim_id"]

    def test_debate_user_input_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateUserInputRequest" in schemas
        assert schemas["DebateUserInputRequest"]["required"] == ["input"]

    def test_debate_counterfactual_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateCounterfactualRequest" in schemas
        assert schemas["DebateCounterfactualRequest"]["required"] == ["condition"]

    def test_debate_message_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateMessageRequest" in schemas
        assert schemas["DebateMessageRequest"]["required"] == ["content"]

    def test_debate_batch_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateBatchRequest" in schemas
        assert schemas["DebateBatchRequest"]["required"] == ["requests"]

    def test_intervention_inject_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateInjectArgumentRequest" in schemas
        assert schemas["DebateInjectArgumentRequest"]["required"] == ["content"]
        props = schemas["DebateInjectArgumentRequest"]["properties"]
        assert "type" in props
        assert props["type"]["enum"] == ["argument", "follow_up"]

    def test_intervention_weights_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateUpdateWeightsRequest" in schemas
        assert schemas["DebateUpdateWeightsRequest"]["required"] == ["agent", "weight"]
        props = schemas["DebateUpdateWeightsRequest"]["properties"]
        assert props["weight"]["minimum"] == 0.0
        assert props["weight"]["maximum"] == 2.0

    def test_intervention_threshold_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateUpdateThresholdRequest" in schemas
        assert schemas["DebateUpdateThresholdRequest"]["required"] == ["threshold"]
        props = schemas["DebateUpdateThresholdRequest"]["properties"]
        assert props["threshold"]["minimum"] == 0.5
        assert props["threshold"]["maximum"] == 1.0

    def test_cost_estimate_request_schema(self):
        schemas = self._get_schemas()
        assert "DebateCostEstimateRequest" in schemas
        props = schemas["DebateCostEstimateRequest"]["properties"]
        assert "num_agents" in props
        assert "num_rounds" in props
        assert "model_types" in props

    def test_cost_estimate_response_schema(self):
        schemas = self._get_schemas()
        assert "DebateCostEstimateResponse" in schemas
        assert "total_estimated_cost_usd" in schemas["DebateCostEstimateResponse"]["required"]

    def test_websocket_resume_token_schema(self):
        schemas = self._get_schemas()
        assert "WebSocketResumeToken" in schemas
        assert "resume_token" in schemas["WebSocketResumeToken"]["required"]

    def test_schemas_registered_in_common(self):
        """Verify new schemas are available via the unified COMMON_SCHEMAS."""
        from aragora.server.openapi.schemas import COMMON_SCHEMAS

        for name in [
            "DebateJoinRequest",
            "DebateVoteRequest",
            "DebateSuggestionRequest",
            "DebateUpdateRequest",
            "DebateCostEstimateResponse",
            "WebSocketResumeToken",
        ]:
            assert name in COMMON_SCHEMAS, f"{name} missing from COMMON_SCHEMAS"

    def test_total_new_request_schemas_count(self):
        """Verify we have 20 new debate request schemas."""
        schemas = self._get_schemas()
        assert len(schemas) >= 20


class TestDebateHardeningEndpoints:
    """Verify debate hardening endpoint definitions reference typed schemas."""

    def _get_endpoints(self):
        from aragora.server.openapi.endpoints.debate_hardening import DEBATE_HARDENING_ENDPOINTS

        return DEBATE_HARDENING_ENDPOINTS

    def test_join_endpoint_exists(self):
        eps = self._get_endpoints()
        assert "/api/v1/debates/{id}/join" in eps
        post = eps["/api/v1/debates/{id}/join"]["post"]
        assert post["operationId"] == "joinDebateV1"
        schema = post["requestBody"]["content"]["application/json"]["schema"]
        assert schema["$ref"] == "#/components/schemas/DebateJoinRequest"

    def test_vote_endpoint_exists(self):
        eps = self._get_endpoints()
        assert "/api/v1/debates/{id}/vote" in eps
        post = eps["/api/v1/debates/{id}/vote"]["post"]
        assert post["requestBody"]["required"] is True
        schema = post["requestBody"]["content"]["application/json"]["schema"]
        assert schema["$ref"] == "#/components/schemas/DebateVoteRequest"

    def test_suggest_endpoint_exists(self):
        eps = self._get_endpoints()
        assert "/api/v1/debates/{id}/suggest" in eps
        schema = eps["/api/v1/debates/{id}/suggest"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        assert schema["$ref"] == "#/components/schemas/DebateSuggestionRequest"

    def test_update_endpoint_exists(self):
        eps = self._get_endpoints()
        assert "/api/v1/debates/{id}/update" in eps
        put = eps["/api/v1/debates/{id}/update"]["put"]
        assert put["operationId"] == "updateDebateV1"

    def test_intervention_pause_endpoint(self):
        eps = self._get_endpoints()
        assert "/api/debates/{debate_id}/intervention/pause" in eps

    def test_intervention_resume_endpoint(self):
        eps = self._get_endpoints()
        assert "/api/debates/{debate_id}/intervention/resume" in eps

    def test_intervention_inject_endpoint(self):
        eps = self._get_endpoints()
        path = "/api/debates/{debate_id}/intervention/inject"
        assert path in eps
        schema = eps[path]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert schema["$ref"] == "#/components/schemas/DebateInjectArgumentRequest"

    def test_intervention_weights_endpoint(self):
        eps = self._get_endpoints()
        path = "/api/debates/{debate_id}/intervention/weights"
        assert path in eps
        schema = eps[path]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert schema["$ref"] == "#/components/schemas/DebateUpdateWeightsRequest"

    def test_intervention_threshold_endpoint(self):
        eps = self._get_endpoints()
        path = "/api/debates/{debate_id}/intervention/threshold"
        assert path in eps

    def test_intervention_state_endpoint(self):
        eps = self._get_endpoints()
        assert "/api/debates/{debate_id}/intervention/state" in eps

    def test_intervention_log_endpoint(self):
        eps = self._get_endpoints()
        assert "/api/debates/{debate_id}/intervention/log" in eps

    def test_cost_estimation_endpoint(self):
        eps = self._get_endpoints()
        assert "/api/v1/debates/estimate-cost" in eps
        get = eps["/api/v1/debates/estimate-cost"]["get"]
        assert get["operationId"] == "estimateDebateCost"
        assert any(p["name"] == "num_agents" for p in get["parameters"])
        assert any(p["name"] == "num_rounds" for p in get["parameters"])
        assert any(p["name"] == "model_types" for p in get["parameters"])

    def test_endpoints_registered_in_all_endpoints(self):
        """New endpoints should be in the ALL_ENDPOINTS dict."""
        from aragora.server.openapi.endpoints import ALL_ENDPOINTS

        for path in [
            "/api/v1/debates/{id}/join",
            "/api/v1/debates/{id}/vote",
            "/api/v1/debates/{id}/suggest",
            "/api/v1/debates/estimate-cost",
        ]:
            assert path in ALL_ENDPOINTS, f"{path} missing from ALL_ENDPOINTS"

    def test_endpoint_count(self):
        """Should have 12 new endpoints."""
        eps = self._get_endpoints()
        assert len(eps) >= 12


# ============================================================================
# Cost Estimation Tests
# ============================================================================


class TestCostEstimation:
    """Test the debate cost estimation logic."""

    def test_estimate_default_params(self):
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        result = estimate_debate_cost()
        assert result["num_agents"] == 3
        assert result["num_rounds"] == 9
        assert result["total_estimated_cost_usd"] > 0
        assert len(result["breakdown_by_model"]) == 3

    def test_estimate_single_agent(self):
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        result = estimate_debate_cost(num_agents=1, num_rounds=1)
        assert result["num_agents"] == 1
        assert len(result["breakdown_by_model"]) == 1
        assert result["total_estimated_cost_usd"] > 0

    def test_estimate_with_specific_models(self):
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        result = estimate_debate_cost(
            num_agents=2,
            num_rounds=5,
            model_types=["claude-opus-4", "gpt-4o"],
        )
        assert len(result["breakdown_by_model"]) == 2
        models = [b["model"] for b in result["breakdown_by_model"]]
        assert "claude-opus-4" in models
        assert "gpt-4o" in models

    def test_estimate_model_round_robin(self):
        """When fewer models than agents, models are assigned round-robin."""
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        result = estimate_debate_cost(
            num_agents=4,
            num_rounds=3,
            model_types=["claude-sonnet-4", "gpt-4o"],
        )
        assert len(result["breakdown_by_model"]) == 4
        models = [b["model"] for b in result["breakdown_by_model"]]
        assert models == ["claude-sonnet-4", "gpt-4o", "claude-sonnet-4", "gpt-4o"]

    def test_estimate_unknown_model_uses_default(self):
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        result = estimate_debate_cost(
            num_agents=1,
            num_rounds=1,
            model_types=["unknown-model-xyz"],
        )
        assert result["breakdown_by_model"][0]["provider"] == "openrouter"
        assert result["total_estimated_cost_usd"] > 0

    def test_estimate_assumptions_included(self):
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        result = estimate_debate_cost()
        assert "assumptions" in result
        assert result["assumptions"]["includes_system_prompt"] is True
        assert result["assumptions"]["avg_input_tokens_per_round"] > 0

    def test_estimate_cost_increases_with_rounds(self):
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        cost_3 = estimate_debate_cost(num_rounds=3)["total_estimated_cost_usd"]
        cost_9 = estimate_debate_cost(num_rounds=9)["total_estimated_cost_usd"]
        assert cost_9 > cost_3

    def test_estimate_cost_increases_with_agents(self):
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        cost_2 = estimate_debate_cost(num_agents=2)["total_estimated_cost_usd"]
        cost_5 = estimate_debate_cost(num_agents=5)["total_estimated_cost_usd"]
        assert cost_5 > cost_2

    def test_handle_estimate_cost_valid(self):
        from aragora.server.handlers.debates.cost_estimation import handle_estimate_cost

        result = handle_estimate_cost(num_agents=3, num_rounds=5)
        assert result[1] == 200  # status code
        body = result[0]
        assert body["total_estimated_cost_usd"] > 0

    def test_handle_estimate_cost_invalid_agents(self):
        from aragora.server.handlers.debates.cost_estimation import handle_estimate_cost

        result = handle_estimate_cost(num_agents=0, num_rounds=5)
        assert result[1] == 400

    def test_handle_estimate_cost_invalid_rounds(self):
        from aragora.server.handlers.debates.cost_estimation import handle_estimate_cost

        result = handle_estimate_cost(num_agents=3, num_rounds=20)
        assert result[1] == 400

    def test_handle_estimate_cost_with_model_string(self):
        from aragora.server.handlers.debates.cost_estimation import handle_estimate_cost

        result = handle_estimate_cost(
            num_agents=2,
            num_rounds=3,
            model_types_str="claude-sonnet-4,gpt-4o",
        )
        assert result[1] == 200
        body = result[0]
        assert len(body["breakdown_by_model"]) == 2

    def test_opus_more_expensive_than_sonnet(self):
        """Verify pricing hierarchy: opus > sonnet."""
        from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

        opus = estimate_debate_cost(num_agents=1, num_rounds=3, model_types=["claude-opus-4"])
        sonnet = estimate_debate_cost(num_agents=1, num_rounds=3, model_types=["claude-sonnet-4"])
        assert opus["total_estimated_cost_usd"] > sonnet["total_estimated_cost_usd"]


# ============================================================================
# WebSocket Resume Token Tests
# ============================================================================


class TestResumeToken:
    """Test WebSocket resume token generation, validation, and replay."""

    def _make_manager(self):
        from aragora.server.stream.resume_token import ResumeTokenManager

        return ResumeTokenManager(secret="test-secret")

    def test_generate_and_validate_token(self):
        mgr = self._make_manager()
        token = mgr.generate_token("deb_123", 42)
        result = mgr.validate_token(token)
        assert result is not None
        debate_id, last_seq = result
        assert debate_id == "deb_123"
        assert last_seq == 42

    def test_invalid_token_rejected(self):
        mgr = self._make_manager()
        assert mgr.validate_token("garbage") is None
        assert mgr.validate_token("") is None
        assert mgr.validate_token("foo.bar") is None

    def test_tampered_token_rejected(self):
        mgr = self._make_manager()
        token = mgr.generate_token("deb_123", 42)
        parts = token.rsplit(".", 1)
        tampered = parts[0] + "." + parts[1][::-1]
        assert mgr.validate_token(tampered) is None

    def test_expired_token_rejected(self):
        from aragora.server.stream import resume_token as rt_module

        mgr = self._make_manager()
        token = mgr.generate_token("deb_123", 10)

        with patch.object(rt_module.time, "time", return_value=time.time() + 600):
            assert mgr.validate_token(token) is None

    def test_buffer_and_replay(self):
        mgr = self._make_manager()
        mgr.buffer_event("deb_1", 1, {"type": "debate_start", "seq": 1})
        mgr.buffer_event("deb_1", 2, {"type": "round_start", "seq": 2})
        mgr.buffer_event("deb_1", 3, {"type": "agent_message", "seq": 3})

        # Client was at seq=1, should get events 2 and 3
        token = mgr.generate_token("deb_1", 1)
        events = mgr.get_replay_events(token)
        assert events is not None
        assert len(events) == 2
        assert events[0]["seq"] == 2
        assert events[1]["seq"] == 3

    def test_replay_returns_none_for_invalid_token(self):
        mgr = self._make_manager()
        assert mgr.get_replay_events("invalid.token") is None

    def test_buffer_bounded_size(self):
        from aragora.server.stream.resume_token import EventReplayBuffer

        buf = EventReplayBuffer("deb_1", max_size=5)
        for i in range(10):
            buf.append(i, {"seq": i})
        assert len(buf) == 5  # Only last 5 kept

    def test_buffer_evicts_expired_events(self):
        from aragora.server.stream.resume_token import EventReplayBuffer

        buf = EventReplayBuffer("deb_1", ttl_seconds=1)
        buf.append(1, {"seq": 1})
        time.sleep(1.1)
        buf.append(2, {"seq": 2})

        events = buf.get_events_after(0)
        assert len(events) == 1
        assert events[0]["seq"] == 2

    def test_inject_resume_token(self):
        from aragora.server.stream.resume_token import (
            inject_resume_token,
            get_resume_token_manager,
        )

        event = {"type": "agent_message", "seq": 5, "data": {}}
        result = inject_resume_token(event, "deb_100")
        assert "resume_token" in result
        mgr = get_resume_token_manager()
        parsed = mgr.validate_token(result["resume_token"])
        assert parsed is not None
        assert parsed[0] == "deb_100"
        assert parsed[1] == 5

    def test_inject_resume_token_skips_no_seq(self):
        from aragora.server.stream.resume_token import inject_resume_token

        event = {"type": "heartbeat", "data": {}}
        result = inject_resume_token(event, "deb_1")
        assert "resume_token" not in result

    def test_cleanup_expired_buffers(self):
        mgr = self._make_manager()
        mgr.buffer_event("deb_old", 1, {"seq": 1})

        buf = mgr.get_or_create_buffer("deb_old")
        with buf._lock:
            for evt in buf._buffer:
                evt.timestamp = time.time() - 1000

        removed = mgr.cleanup_expired()
        assert removed == 1

    def test_remove_buffer(self):
        mgr = self._make_manager()
        mgr.buffer_event("deb_x", 1, {"seq": 1})
        mgr.remove_buffer("deb_x")
        buf = mgr.get_or_create_buffer("deb_x")
        assert len(buf) == 0

    def test_different_secrets_reject_each_other(self):
        from aragora.server.stream.resume_token import ResumeTokenManager

        mgr1 = ResumeTokenManager(secret="secret-a")
        mgr2 = ResumeTokenManager(secret="secret-b")
        token = mgr1.generate_token("deb_1", 10)
        assert mgr2.validate_token(token) is None

    def test_replay_empty_buffer(self):
        mgr = self._make_manager()
        token = mgr.generate_token("deb_empty", 0)
        events = mgr.get_replay_events(token)
        assert events is not None
        assert len(events) == 0

    def test_buffer_get_events_only_after_seq(self):
        """Events at or before last_seq should not be returned."""
        mgr = self._make_manager()
        for i in range(1, 6):
            mgr.buffer_event("deb_seq", i, {"seq": i, "data": f"event_{i}"})

        token = mgr.generate_token("deb_seq", 3)
        events = mgr.get_replay_events(token)
        assert events is not None
        assert len(events) == 2
        assert events[0]["seq"] == 4
        assert events[1]["seq"] == 5
