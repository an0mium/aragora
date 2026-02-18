"""Tests for semantic idea clustering (GoalExtractor.cluster_ideas_semantically).

Covers: Jaccard similarity, agglomerative clustering, cluster naming,
full integration with canvas data, and edge cases.
"""

from __future__ import annotations

import pytest

from aragora.goals.extractor import (
    GoalExtractor,
    _jaccard_similarity,
    _name_cluster,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    return GoalExtractor()


def _make_canvas(ideas: list[str], edges: list[dict] | None = None) -> dict:
    """Build a minimal idea canvas from a list of idea strings."""
    nodes = []
    for i, idea in enumerate(ideas):
        nodes.append({
            "id": f"idea-{i}",
            "label": idea[:80],
            "data": {
                "idea_type": "concept",
                "full_content": idea,
            },
        })
    return {"nodes": nodes, "edges": edges or []}


# ===========================================================================
# TestJaccardSimilarity
# ===========================================================================


class TestJaccardSimilarity:
    def test_identical_sets(self):
        tokens = frozenset({"database", "performance", "query"})
        assert _jaccard_similarity(tokens, tokens) == 1.0

    def test_completely_different(self):
        a = frozenset({"database", "query"})
        b = frozenset({"frontend", "react"})
        assert _jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset({"database", "query", "performance"})
        b = frozenset({"database", "query", "index"})
        # intersection: {database, query} = 2
        # union: {database, query, performance, index} = 4
        assert _jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_both_empty(self):
        assert _jaccard_similarity(frozenset(), frozenset()) == 0.0

    def test_one_empty(self):
        a = frozenset({"database"})
        assert _jaccard_similarity(a, frozenset()) == 0.0
        assert _jaccard_similarity(frozenset(), a) == 0.0

    def test_single_element_match(self):
        a = frozenset({"performance"})
        b = frozenset({"performance"})
        assert _jaccard_similarity(a, b) == 1.0

    def test_single_element_no_match(self):
        a = frozenset({"performance"})
        b = frozenset({"security"})
        assert _jaccard_similarity(a, b) == 0.0


# ===========================================================================
# TestTokenize
# ===========================================================================


class TestTokenize:
    def test_basic_tokenization(self):
        result = _tokenize("optimize database queries")
        assert "optimize" in result
        assert "database" in result
        assert "queries" in result

    def test_stopword_removal(self):
        result = _tokenize("the database is slow and the server is down")
        assert "the" not in result
        assert "is" not in result
        assert "and" not in result
        assert "database" in result
        assert "slow" in result
        assert "server" in result

    def test_case_insensitive(self):
        result = _tokenize("Database PERFORMANCE Query")
        assert "database" in result
        assert "performance" in result
        assert "query" in result

    def test_punctuation_splitting(self):
        result = _tokenize("database, query; performance!")
        assert "database" in result
        assert "query" in result
        assert "performance" in result

    def test_empty_text(self):
        result = _tokenize("")
        assert result == frozenset()

    def test_only_stopwords(self):
        result = _tokenize("the and is are to of")
        assert result == frozenset()

    def test_short_tokens_removed(self):
        """Single-character tokens should be removed."""
        result = _tokenize("a b c database x y z")
        assert "database" in result
        # 'a' is also a stop word, but single chars should be dropped anyway
        assert "b" not in result
        assert "c" not in result

    def test_returns_frozenset(self):
        result = _tokenize("hello world")
        assert isinstance(result, frozenset)


# ===========================================================================
# TestClusterNaming
# ===========================================================================


class TestClusterNaming:
    def test_common_terms_extraction(self):
        sets = [
            frozenset({"database", "query", "performance"}),
            frozenset({"database", "query", "index"}),
            frozenset({"database", "optimization", "cache"}),
        ]
        name = _name_cluster(sets)
        # "database" appears 3 times, "query" appears 2 times
        assert "database" in name

    def test_max_three_terms(self):
        sets = [
            frozenset({"a", "b", "c", "d", "e"}),
            frozenset({"a", "b", "c", "d", "f"}),
        ]
        name = _name_cluster(sets)
        parts = name.split(" / ")
        assert len(parts) <= 3

    def test_empty_token_sets(self):
        name = _name_cluster([frozenset()])
        assert name == "Unnamed cluster"

    def test_single_term_cluster(self):
        sets = [frozenset({"performance"})]
        name = _name_cluster(sets)
        assert "performance" in name

    def test_deduplication_in_naming(self):
        """Each term appears at most once in the cluster name."""
        sets = [
            frozenset({"api", "endpoint"}),
            frozenset({"api", "service"}),
        ]
        name = _name_cluster(sets)
        # api should appear exactly once despite appearing in both sets
        assert name.count("api") == 1


# ===========================================================================
# TestAgglomerativeClustering
# ===========================================================================


class TestAgglomerativeClustering:
    def test_similar_ideas_form_cluster(self, extractor):
        canvas = _make_canvas([
            "optimize database query performance",
            "improve database query speed",
            "frontend react component design",
        ])
        result = extractor.cluster_ideas_semantically(canvas, similarity_threshold=0.2)
        cluster_nodes = [
            n for n in result["nodes"] if n.get("data", {}).get("idea_type") == "cluster"
        ]
        # The two DB ideas should cluster; frontend is different
        assert len(cluster_nodes) >= 1

    def test_all_similar_one_big_cluster(self, extractor):
        canvas = _make_canvas([
            "optimize database query performance tuning",
            "improve database query performance speed",
            "enhance database query performance optimization",
        ])
        result = extractor.cluster_ideas_semantically(canvas, similarity_threshold=0.1)
        cluster_nodes = [
            n for n in result["nodes"] if n.get("data", {}).get("idea_type") == "cluster"
        ]
        assert len(cluster_nodes) == 1
        # Should have all 3 members
        member_edges = [
            e for e in result["edges"] if e["type"] == "member_of"
        ]
        assert len(member_edges) == 3

    def test_all_different_no_clusters(self, extractor):
        canvas = _make_canvas([
            "optimize database queries",
            "design frontend components",
            "configure network security",
        ])
        result = extractor.cluster_ideas_semantically(canvas, similarity_threshold=0.8)
        cluster_nodes = [
            n for n in result["nodes"] if n.get("data", {}).get("idea_type") == "cluster"
        ]
        # Very high threshold should prevent clustering
        assert len(cluster_nodes) == 0

    def test_threshold_behavior(self, extractor):
        """Higher threshold means fewer/smaller clusters."""
        canvas = _make_canvas([
            "database query optimization for performance",
            "database index optimization for speed",
            "frontend design patterns for react",
            "frontend component architecture for react",
        ])
        low_thresh = extractor.cluster_ideas_semantically(
            canvas, similarity_threshold=0.1
        )
        high_thresh = extractor.cluster_ideas_semantically(
            canvas, similarity_threshold=0.5
        )
        low_clusters = [
            n for n in low_thresh["nodes"]
            if n.get("data", {}).get("idea_type") == "cluster"
        ]
        high_clusters = [
            n for n in high_thresh["nodes"]
            if n.get("data", {}).get("idea_type") == "cluster"
        ]
        assert len(low_clusters) >= len(high_clusters)

    def test_min_cluster_size(self, extractor):
        """min_cluster_size=3 should require at least 3 members."""
        canvas = _make_canvas([
            "database query performance optimization",
            "database query speed improvement",
            "frontend react design",
        ])
        result = extractor.cluster_ideas_semantically(
            canvas, similarity_threshold=0.2, min_cluster_size=3
        )
        cluster_nodes = [
            n for n in result["nodes"]
            if n.get("data", {}).get("idea_type") == "cluster"
        ]
        # Only 2 DB ideas, so no cluster with min_cluster_size=3
        assert len(cluster_nodes) == 0


# ===========================================================================
# TestClusterIdeasSemantically (full integration)
# ===========================================================================


class TestClusterIdeasSemantically:
    def test_original_nodes_preserved(self, extractor):
        canvas = _make_canvas(["idea one", "idea two", "idea three"])
        result = extractor.cluster_ideas_semantically(canvas)
        original_ids = {f"idea-{i}" for i in range(3)}
        result_ids = {n["id"] for n in result["nodes"]}
        assert original_ids.issubset(result_ids)

    def test_original_not_mutated(self, extractor):
        canvas = _make_canvas(["database queries", "database performance"])
        original_node_count = len(canvas["nodes"])
        result = extractor.cluster_ideas_semantically(canvas)
        # Original should not be mutated
        assert len(canvas["nodes"]) == original_node_count
        # Result may have more nodes (clusters added)
        assert len(result["nodes"]) >= original_node_count

    def test_cluster_nodes_have_correct_data(self, extractor):
        canvas = _make_canvas([
            "optimize database query performance",
            "improve database query speed",
        ])
        result = extractor.cluster_ideas_semantically(canvas, similarity_threshold=0.1)
        cluster_nodes = [
            n for n in result["nodes"]
            if n.get("data", {}).get("idea_type") == "cluster"
        ]
        if cluster_nodes:
            cn = cluster_nodes[0]
            assert cn["data"]["auto_generated"] is True
            assert cn["data"]["member_count"] >= 2
            assert cn["id"].startswith("cluster-")

    def test_membership_edges_created(self, extractor):
        canvas = _make_canvas([
            "optimize database query performance",
            "improve database query speed",
        ])
        result = extractor.cluster_ideas_semantically(canvas, similarity_threshold=0.1)
        member_edges = [e for e in result["edges"] if e["type"] == "member_of"]
        if member_edges:
            # All membership edges should point to a cluster node
            cluster_ids = {
                n["id"]
                for n in result["nodes"]
                if n.get("data", {}).get("idea_type") == "cluster"
            }
            for edge in member_edges:
                assert edge["target"] in cluster_ids

    def test_empty_canvas(self, extractor):
        result = extractor.cluster_ideas_semantically({"nodes": [], "edges": []})
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_preserves_existing_edges(self, extractor):
        canvas = _make_canvas(["database queries", "database performance"])
        canvas["edges"] = [{"source": "idea-0", "target": "idea-1", "type": "support"}]
        result = extractor.cluster_ideas_semantically(canvas, similarity_threshold=0.1)
        # Original edge should still be there
        support_edges = [e for e in result["edges"] if e["type"] == "support"]
        assert len(support_edges) == 1


# ===========================================================================
# TestClusteringEdgeCases
# ===========================================================================


class TestClusteringEdgeCases:
    def test_single_idea(self, extractor):
        canvas = _make_canvas(["single idea about databases"])
        result = extractor.cluster_ideas_semantically(canvas)
        cluster_nodes = [
            n for n in result["nodes"]
            if n.get("data", {}).get("idea_type") == "cluster"
        ]
        assert len(cluster_nodes) == 0  # Can't cluster a single idea

    def test_two_identical_ideas(self, extractor):
        canvas = _make_canvas([
            "optimize database performance",
            "optimize database performance",
        ])
        result = extractor.cluster_ideas_semantically(canvas, similarity_threshold=0.1)
        cluster_nodes = [
            n for n in result["nodes"]
            if n.get("data", {}).get("idea_type") == "cluster"
        ]
        assert len(cluster_nodes) == 1

    def test_missing_nodes_key(self, extractor):
        result = extractor.cluster_ideas_semantically({})
        assert result.get("nodes", []) == []

    def test_nodes_without_label(self, extractor):
        """Nodes missing labels should still work (empty tokens)."""
        canvas = {"nodes": [
            {"id": "n1", "data": {"full_content": "database performance"}},
            {"id": "n2", "data": {"full_content": "database optimization"}},
        ], "edges": []}
        result = extractor.cluster_ideas_semantically(canvas, similarity_threshold=0.1)
        # Should not crash
        assert len(result["nodes"]) >= 2
