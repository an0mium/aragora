"""Tests for the crux operations handler mixin (ODR-4 / #8227).

Covers the honest-absence contract of GET /api/v1/debates/{id}/cruxes:
- 404 when the debate does not exist anywhere
- ODR-style absent marker (never fabrication) when crux mode was not enabled
- skip reason surfaced verbatim when crux finder was requested but skipped
- present crux map when the crux_finder consensus mode recorded one
- route wiring (suffix table, ID-only dispatch, can_handle)
"""

from __future__ import annotations

import json

from aragora.server.handlers.debates.cruxes import CruxOperationsMixin

CRUX_ITEM = {
    "claim_id": "claim_1",
    "statement": "The cache invalidation strategy is sound",
    "author": "claude",
    "crux_score": 0.81,
    "influence_score": 0.7,
    "disagreement_score": 0.9,
    "uncertainty_score": 0.6,
    "centrality_score": 0.5,
    "affected_claims": ["claim_2"],
    "contesting_agents": ["claude", "codex"],
    "resolution_impact": 0.4,
}


def _crux_debate_record() -> dict:
    """Stored debate artifact dict as written after a crux_finder run."""
    return {
        "id": "debate-crux-1",
        "consensus_proof": {
            "final_claim": "__CRUX_MAP__: no verdict by design; see CruxReceipt.cruxes",
            "consensus_reached": False,
            "metadata": {
                "consensus_mode": "crux_finder",
                "cruxes": [CRUX_ITEM],
                "crux_count": 1,
                "convergence_barrier": 0.42,
                "counterfactuals": [{"claim_id": "claim_1", "likelihood": 0.6}],
                "recommended_focus": ["claim_1"],
            },
        },
    }


class _Storage:
    def __init__(self, debates: dict | None = None):
        self._debates = debates or {}

    def get_debate(self, debate_id: str):
        return self._debates.get(debate_id)


class MockCruxHandler(CruxOperationsMixin):
    def __init__(self, storage=None, nomic_dir=None):
        self._storage = storage
        self._nomic_dir = nomic_dir
        self.ctx = {}

    def get_storage(self):
        return self._storage

    def get_nomic_dir(self):
        return self._nomic_dir


class TestGetCruxesNotFound:
    def test_404_when_debate_missing_everywhere(self, tmp_path):
        handler = MockCruxHandler(storage=_Storage(), nomic_dir=tmp_path)
        result = handler._get_cruxes("missing-debate")
        assert result.status_code == 404
        body = json.loads(result.body)
        assert "not found" in body["error"].lower()

    def test_404_when_no_storage_and_no_trace(self):
        handler = MockCruxHandler(storage=None, nomic_dir=None)
        result = handler._get_cruxes("missing-debate")
        assert result.status_code == 404


class TestGetCruxesHonestAbsence:
    def test_absent_marker_when_crux_mode_not_enabled(self):
        """A normal (non-crux) debate must yield an explicit absent marker."""
        record = {
            "id": "debate-1",
            "consensus_proof": {"final_claim": "Use Redis", "metadata": {}},
        }
        handler = MockCruxHandler(storage=_Storage({"debate-1": record}))
        result = handler._get_cruxes("debate-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "debate-1"
        assert body["cruxes"]["status"] == "absent"
        assert body["cruxes"]["reason"]
        assert body["crux_count"] == 0
        # Honest absence means NO fabricated items key.
        assert "items" not in body["cruxes"]

    def test_absent_marker_when_no_consensus_proof(self):
        record = {"id": "debate-2", "task": "decide something"}
        handler = MockCruxHandler(storage=_Storage({"debate-2": record}))
        result = handler._get_cruxes("debate-2")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["cruxes"]["status"] == "absent"
        assert "crux mode was not enabled" in body["cruxes"]["reason"]

    def test_skip_reason_surfaced_verbatim(self):
        """crux_finder requested but skipped -> reason includes the recorded cause."""
        record = {
            "id": "debate-3",
            "metadata": {"crux_finder_skipped_reason": "no_belief_network"},
        }
        handler = MockCruxHandler(storage=_Storage({"debate-3": record}))
        result = handler._get_cruxes("debate-3")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["cruxes"]["status"] == "absent"
        assert "no_belief_network" in body["cruxes"]["reason"]

    def test_empty_crux_list_is_absent_not_fabricated_present(self):
        """Crux finder ran but found nothing -> absent marker, not empty present."""
        record = _crux_debate_record()
        record["consensus_proof"]["metadata"]["cruxes"] = []
        record["consensus_proof"]["metadata"]["crux_count"] = 0
        handler = MockCruxHandler(storage=_Storage({"debate-crux-1": record}))
        result = handler._get_cruxes("debate-crux-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["cruxes"]["status"] == "absent"
        assert "no cruxes" in body["cruxes"]["reason"]


class TestGetCruxesPresent:
    def test_present_crux_map_returned(self):
        handler = MockCruxHandler(storage=_Storage({"debate-crux-1": _crux_debate_record()}))
        result = handler._get_cruxes("debate-crux-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "debate-crux-1"
        assert body["cruxes"]["status"] == "present"
        items = body["cruxes"]["items"]
        assert len(items) == 1
        assert items[0]["statement"] == CRUX_ITEM["statement"]
        assert items[0]["contesting_agents"] == ["claude", "codex"]
        assert body["crux_count"] == 1
        assert body["convergence_barrier"] == 0.42
        assert body["recommended_focus"] == ["claim_1"]
        assert body["consensus_mode"] == "crux_finder"
        assert body["counterfactuals"] == [{"claim_id": "claim_1", "likelihood": 0.6}]

    def test_present_crux_map_from_nested_result_dict(self):
        record = {"id": "debate-crux-2", "result": _crux_debate_record()}
        handler = MockCruxHandler(storage=_Storage({"debate-crux-2": record}))
        result = handler._get_cruxes("debate-crux-2")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["cruxes"]["status"] == "present"

    def test_storage_error_falls_back_then_404(self, tmp_path):
        class BrokenStorage:
            def get_debate(self, debate_id):
                raise RuntimeError("db down")

        handler = MockCruxHandler(storage=BrokenStorage(), nomic_dir=tmp_path)
        result = handler._get_cruxes("debate-x")
        assert result.status_code == 404


class TestRouteWiring:
    def test_suffix_route_registered(self):
        from aragora.server.handlers.debates.routing import ID_ONLY_METHODS, SUFFIX_ROUTES

        entries = [e for e in SUFFIX_ROUTES if e[0] == "/cruxes"]
        assert entries, "/cruxes suffix route missing"
        suffix, method_name, needs_id, extra = entries[0]
        assert method_name == "_get_cruxes"
        assert needs_id is True
        assert extra is None
        assert "_get_cruxes" in ID_ONLY_METHODS

    def test_routes_list_contains_cruxes_pattern(self):
        from aragora.server.handlers.debates.routing import ROUTES

        assert "/api/v1/debates/*/cruxes" in ROUTES

    def test_handler_class_composes_mixin(self):
        from aragora.server.handlers.debates import DebatesHandler

        assert hasattr(DebatesHandler, "_get_cruxes")
        handler = DebatesHandler(ctx={})
        assert handler.can_handle("/api/v1/debates/abc/cruxes")
