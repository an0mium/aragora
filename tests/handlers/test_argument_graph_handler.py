"""Tests for argument graph REST API in AnalysisOperationsMixin."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class MockDebateResult:
    """Mock debate result for graph building."""

    def __init__(self, task="test debate", messages=None, critiques=None):
        self.task = task
        self.messages = messages or []
        self.critiques = critiques or []
        self.rounds = []


class MockMessage:
    def __init__(self, agent, content, role="proposer", round_num=1):
        self.agent = agent
        self.content = content
        self.role = role
        self.round = round_num


class MockCritique:
    def __init__(self, agent, target, severity=0.5, reasoning="", round_num=1):
        self.agent = agent
        self.target = target
        self.severity = severity
        self.reasoning = reasoning
        self.round = round_num


class MockTrace:
    def __init__(self, result):
        self._result = result

    def to_debate_result(self):
        return self._result

    @classmethod
    def load(cls, path):
        return cls(
            MockDebateResult(
                task="Test debate topic",
                messages=[
                    MockMessage("claude", "I propose we use microservices", "proposer", 1),
                    MockMessage("gpt4", "I suggest a monolith instead", "proposer", 1),
                    MockMessage("claude", "Microservices scale better", "revision", 2),
                ],
                critiques=[
                    MockCritique("gpt4", "claude", 0.8, "Too complex for small teams"),
                ],
            )
        )


class EmptyTrace:
    @classmethod
    def load(cls, path):
        return cls()

    def to_debate_result(self):
        return MockDebateResult(task="empty debate")


@pytest.fixture
def mixin():
    """Create a mock object that behaves like AnalysisOperationsMixin."""
    from aragora.server.handlers.debates.analysis import AnalysisOperationsMixin

    class MockHandler(AnalysisOperationsMixin):
        def __init__(self):
            self.ctx = {}

        def get_nomic_dir(self):
            return self.ctx.get("nomic_dir")

        def get_storage(self):
            return None

    return MockHandler()


def _setup_trace_dir(tmp_path, debate_id="debate-1"):
    """Create trace directory with stub file."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir(exist_ok=True)
    (traces_dir / f"{debate_id}.json").write_text("{}")
    return tmp_path


class TestArgumentGraph:
    """Tests for GET /api/v1/debates/{debate_id}/argument-graph."""

    def test_json_graph_has_nodes_and_edges(self, mixin, tmp_path):
        """JSON graph output includes nodes and edges."""
        mixin.ctx["nomic_dir"] = _setup_trace_dir(tmp_path)

        with patch("aragora.debate.traces.DebateTrace", MockTrace):
            result = mixin._get_argument_graph("debate-1", "json")

        body = result[0]
        assert body["debate_id"] == "debate-1"
        assert body["format"] == "json"
        graph = body["graph"]
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 3  # 3 messages

    def test_mermaid_format_works(self, mixin, tmp_path):
        """Mermaid format returns string diagram."""
        mixin.ctx["nomic_dir"] = _setup_trace_dir(tmp_path)

        with patch("aragora.debate.traces.DebateTrace", MockTrace):
            result = mixin._get_argument_graph("debate-1", "mermaid")

        body = result[0]
        assert body["format"] == "mermaid"
        assert isinstance(body["graph"], str)
        assert "graph TD" in body["graph"]

    def test_empty_debate_returns_empty_graph(self, mixin, tmp_path):
        """Debate with no messages returns empty graph."""
        mixin.ctx["nomic_dir"] = _setup_trace_dir(tmp_path, "debate-empty")

        with patch("aragora.debate.traces.DebateTrace", EmptyTrace):
            result = mixin._get_argument_graph("debate-empty", "json")

        body = result[0]
        graph = body["graph"]
        assert graph["nodes"] == []
        assert graph["edges"] == []

    def test_node_types_correct(self, mixin, tmp_path):
        """Nodes have valid type values."""
        mixin.ctx["nomic_dir"] = _setup_trace_dir(tmp_path, "debate-typed")

        with patch("aragora.debate.traces.DebateTrace", MockTrace):
            result = mixin._get_argument_graph("debate-typed", "json")

        body = result[0]
        graph = body["graph"]
        valid_types = {
            "proposal",
            "critique",
            "evidence",
            "concession",
            "rebuttal",
            "vote",
            "consensus",
        }
        for node in graph["nodes"]:
            assert node["node_type"] in valid_types

    def test_missing_debate_returns_404(self, mixin, tmp_path):
        """Non-existent debate returns 404."""
        mixin.ctx["nomic_dir"] = tmp_path
        (tmp_path / "traces").mkdir(exist_ok=True)

        result = mixin._get_argument_graph("nonexistent-debate", "json")
        assert result[1] == 404

    def test_no_nomic_dir_returns_503(self, mixin):
        """Missing nomic dir returns 503."""
        mixin.ctx["nomic_dir"] = None
        result = mixin._get_argument_graph("debate-1", "json")
        assert result[1] == 503

    def test_graph_metadata_present(self, mixin, tmp_path):
        """Graph JSON includes metadata section."""
        mixin.ctx["nomic_dir"] = _setup_trace_dir(tmp_path, "debate-meta")

        with patch("aragora.debate.traces.DebateTrace", MockTrace):
            result = mixin._get_argument_graph("debate-meta", "json")

        body = result[0]
        graph = body["graph"]
        assert "metadata" in graph
        assert "node_count" in graph["metadata"]
        assert "edge_count" in graph["metadata"]

    def test_graph_includes_critique_edges(self, mixin, tmp_path):
        """Edges from critiques appear in the graph."""
        mixin.ctx["nomic_dir"] = _setup_trace_dir(tmp_path, "debate-edges")

        with patch("aragora.debate.traces.DebateTrace", MockTrace):
            result = mixin._get_argument_graph("debate-edges", "json")

        body = result[0]
        graph = body["graph"]
        # There should be at least one edge from the critique
        assert len(graph["edges"]) > 0
