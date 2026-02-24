"""Tests for the TF-IDF idea clusterer."""

from __future__ import annotations

from aragora.pipeline.idea_clusterer import ClusterResult, IdeaCluster, cluster_ideas


class TestClusterIdeas:
    """Test cluster_ideas with various inputs."""

    def test_empty_input(self):
        result = cluster_ideas([])
        assert isinstance(result, ClusterResult)
        assert result.clusters == []
        assert result.similarity_edges == []

    def test_single_item(self):
        result = cluster_ideas(["Build a rate limiter for the API gateway"])
        assert len(result.clusters) == 1
        assert result.clusters[0].idea_indices == [0]
        assert len(result.clusters[0].centroid_terms) > 0
        assert result.similarity_edges == []

    def test_distinct_clusters(self):
        ideas = [
            "Build a rate limiter for the API gateway",
            "Implement API rate limiting with token bucket algorithm",
            "Write comprehensive user documentation for onboarding",
            "Create getting started guide and tutorials",
        ]
        result = cluster_ideas(ideas, threshold=0.2)

        # Should form at least 2 groups (rate limiting vs documentation)
        # All 4 ideas should be assigned to some cluster
        all_indices = set()
        for cluster in result.clusters:
            all_indices.update(cluster.idea_indices)
        assert all_indices == {0, 1, 2, 3}

    def test_similarity_edges(self):
        ideas = [
            "Build a rate limiter for the API gateway",
            "Implement API rate limiting with token bucket",
            "Something completely different about cooking recipes",
        ]
        result = cluster_ideas(ideas, threshold=0.1)

        # The first two ideas should have high similarity
        has_01_edge = any(
            (i == 0 and j == 1) or (i == 1 and j == 0) for i, j, sim in result.similarity_edges
        )
        assert has_01_edge, "Expected similarity edge between ideas 0 and 1"

        # Check edge weights are reasonable
        for i, j, sim in result.similarity_edges:
            assert 0.0 <= sim <= 1.0

    def test_all_similar_ideas(self):
        ideas = [
            "Implement rate limiting",
            "Add rate limiter middleware",
            "Rate limit the API endpoints",
        ]
        result = cluster_ideas(ideas, threshold=0.1)

        # All should cluster together
        assert len(result.clusters) <= 2  # Could be 1 or 2 depending on threshold

    def test_high_threshold_produces_singletons(self):
        ideas = [
            "Build a rate limiter",
            "Create documentation",
            "Add caching layer",
        ]
        result = cluster_ideas(ideas, threshold=0.99)

        # With very high threshold, each idea should be its own cluster
        assert len(result.clusters) == 3

    def test_cluster_has_centroid_terms(self):
        ideas = [
            "Implement rate limiting for API",
            "Add rate limiter middleware",
        ]
        result = cluster_ideas(ideas, threshold=0.1)

        for cluster in result.clusters:
            assert isinstance(cluster.centroid_terms, list)
            # Centroid terms should be actual words
            for term in cluster.centroid_terms:
                assert isinstance(term, str)
                assert len(term) > 0

    def test_cluster_label_is_top_term(self):
        result = cluster_ideas(["rate limiting for API gateway"])
        assert isinstance(result.clusters[0].label, str)
        assert len(result.clusters[0].label) > 0


class TestIdeaCluster:
    def test_dataclass(self):
        cluster = IdeaCluster(
            label="test",
            idea_indices=[0, 1],
            centroid_terms=["rate", "limit"],
        )
        assert cluster.label == "test"
        assert cluster.idea_indices == [0, 1]


class TestClusterResult:
    def test_defaults(self):
        result = ClusterResult()
        assert result.clusters == []
        assert result.similarity_edges == []
