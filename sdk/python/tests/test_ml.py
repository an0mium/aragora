"""Tests for ML (Machine Learning) namespace API."""

from __future__ import annotations

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Routing Tests
# =========================================================================


class TestMLRoute:
    """Tests for ML-based agent routing."""

    def test_route_basic(self, client: AragoraClient, mock_request) -> None:
        """Route agents for a task with basic parameters."""
        mock_request.return_value = {
            "selected_agents": ["claude", "gpt-4", "gemini"],
            "task_type": "coding",
            "confidence": 0.92,
            "reasoning": "Selected agents with strong coding capabilities",
        }

        result = client.ml.route(
            task="Implement a caching layer",
            available_agents=["claude", "gpt-4", "gemini", "mistral", "llama"],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/route",
            json={
                "task": "Implement a caching layer",
                "available_agents": ["claude", "gpt-4", "gemini", "mistral", "llama"],
                "team_size": 3,
            },
        )
        assert result["selected_agents"] == ["claude", "gpt-4", "gemini"]
        assert result["task_type"] == "coding"
        assert result["confidence"] == 0.92

    def test_route_custom_team_size(self, client: AragoraClient, mock_request) -> None:
        """Route agents with a custom team size."""
        mock_request.return_value = {
            "selected_agents": ["claude", "gpt-4", "gemini", "mistral", "codex"],
            "confidence": 0.88,
        }

        client.ml.route(
            task="Complex multi-disciplinary task",
            available_agents=["claude", "gpt-4", "gemini", "mistral", "codex", "llama"],
            team_size=5,
        )

        call_args = mock_request.call_args
        assert call_args[1]["json"]["team_size"] == 5

    def test_route_with_constraints(self, client: AragoraClient, mock_request) -> None:
        """Route agents with constraints."""
        mock_request.return_value = {
            "selected_agents": ["codex", "claude"],
            "task_type": "coding",
        }

        client.ml.route(
            task="Implement database migrations",
            available_agents=["claude", "gpt-4", "codex", "gemini"],
            team_size=2,
            constraints={"require_code": True, "prefer_fast": False},
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["constraints"] == {"require_code": True, "prefer_fast": False}
        assert call_json["team_size"] == 2

    def test_route_without_constraints(self, client: AragoraClient, mock_request) -> None:
        """Route agents without constraints (constraints not in payload)."""
        mock_request.return_value = {"selected_agents": ["claude"]}

        client.ml.route(
            task="Simple reasoning task",
            available_agents=["claude", "gpt-4"],
            team_size=1,
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert "constraints" not in call_json


# =========================================================================
# Scoring Tests
# =========================================================================


class TestMLScore:
    """Tests for response quality scoring."""

    def test_score_text_only(self, client: AragoraClient, mock_request) -> None:
        """Score response quality with text only."""
        mock_request.return_value = {
            "overall": 0.87,
            "coherence": 0.92,
            "completeness": 0.85,
            "relevance": 0.88,
            "clarity": 0.83,
        }

        result = client.ml.score(text="The caching layer should use Redis for fast lookups.")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/score",
            json={"text": "The caching layer should use Redis for fast lookups."},
        )
        assert result["overall"] == 0.87
        assert result["coherence"] == 0.92

    def test_score_with_context(self, client: AragoraClient, mock_request) -> None:
        """Score response quality with context for relevance scoring."""
        mock_request.return_value = {
            "overall": 0.91,
            "relevance": 0.95,
        }

        client.ml.score(
            text="Use Redis with TTL-based expiration for the cache.",
            context="Design a caching strategy for a high-traffic API",
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["text"] == "Use Redis with TTL-based expiration for the cache."
        assert call_json["context"] == "Design a caching strategy for a high-traffic API"

    def test_score_without_context(self, client: AragoraClient, mock_request) -> None:
        """Score response without context (context not in payload)."""
        mock_request.return_value = {"overall": 0.75}

        client.ml.score(text="Some response text")

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert "context" not in call_json


class TestMLScoreBatch:
    """Tests for batch response scoring."""

    def test_score_batch_texts_only(self, client: AragoraClient, mock_request) -> None:
        """Score multiple responses without contexts."""
        mock_request.return_value = {
            "scores": [
                {"overall": 0.85, "coherence": 0.88},
                {"overall": 0.72, "coherence": 0.75},
                {"overall": 0.91, "coherence": 0.93},
            ]
        }

        result = client.ml.score_batch(
            texts=[
                "First response text",
                "Second response text",
                "Third response text",
            ]
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/score-batch",
            json={
                "texts": [
                    "First response text",
                    "Second response text",
                    "Third response text",
                ]
            },
        )
        assert len(result["scores"]) == 3
        assert result["scores"][0]["overall"] == 0.85

    def test_score_batch_with_contexts(self, client: AragoraClient, mock_request) -> None:
        """Score multiple responses with corresponding contexts."""
        mock_request.return_value = {
            "scores": [
                {"overall": 0.90, "relevance": 0.95},
                {"overall": 0.88, "relevance": 0.92},
            ]
        }

        client.ml.score_batch(
            texts=["Response 1", "Response 2"],
            contexts=["Context 1", "Context 2"],
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["texts"] == ["Response 1", "Response 2"]
        assert call_json["contexts"] == ["Context 1", "Context 2"]

    def test_score_batch_without_contexts(self, client: AragoraClient, mock_request) -> None:
        """Score batch without contexts (contexts not in payload)."""
        mock_request.return_value = {"scores": []}

        client.ml.score_batch(texts=["Text 1", "Text 2"])

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert "contexts" not in call_json


# =========================================================================
# Consensus Prediction Tests
# =========================================================================


class TestMLPredictConsensus:
    """Tests for consensus likelihood prediction."""

    def test_predict_consensus_basic(self, client: AragoraClient, mock_request) -> None:
        """Predict consensus with basic parameters."""
        mock_request.return_value = {
            "probability": 0.78,
            "confidence": 0.85,
            "convergence_trend": "increasing",
            "estimated_rounds_remaining": 2,
        }

        result = client.ml.predict_consensus(
            responses=[
                ("claude", "We should use Redis for caching."),
                ("gpt-4", "Redis is a good choice for caching."),
                ("gemini", "Consider Memcached as an alternative."),
            ]
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/consensus",
            json={
                "responses": [
                    ["claude", "We should use Redis for caching."],
                    ["gpt-4", "Redis is a good choice for caching."],
                    ["gemini", "Consider Memcached as an alternative."],
                ],
                "current_round": 1,
                "total_rounds": 3,
            },
        )
        assert result["probability"] == 0.78
        assert result["convergence_trend"] == "increasing"

    def test_predict_consensus_with_context(self, client: AragoraClient, mock_request) -> None:
        """Predict consensus with task context."""
        mock_request.return_value = {
            "probability": 0.92,
            "confidence": 0.88,
        }

        client.ml.predict_consensus(
            responses=[
                ("claude", "Position A"),
                ("gpt-4", "Position A with minor variation"),
            ],
            context="Decide on caching strategy for API",
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["context"] == "Decide on caching strategy for API"

    def test_predict_consensus_custom_rounds(self, client: AragoraClient, mock_request) -> None:
        """Predict consensus with custom round settings."""
        mock_request.return_value = {"probability": 0.65}

        client.ml.predict_consensus(
            responses=[("claude", "Response")],
            current_round=3,
            total_rounds=5,
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["current_round"] == 3
        assert call_json["total_rounds"] == 5

    def test_predict_consensus_without_context(self, client: AragoraClient, mock_request) -> None:
        """Predict consensus without context (context not in payload)."""
        mock_request.return_value = {"probability": 0.50}

        client.ml.predict_consensus(
            responses=[("agent1", "Text")],
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert "context" not in call_json


# =========================================================================
# Embeddings Tests
# =========================================================================


class TestMLEmbed:
    """Tests for text embedding generation."""

    def test_embed_single_text(self, client: AragoraClient, mock_request) -> None:
        """Generate embedding for a single text."""
        mock_request.return_value = {
            "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "dimension": 5,
        }

        result = client.ml.embed(text="The caching layer improves performance.")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/embed",
            json={"text": "The caching layer improves performance."},
        )
        assert len(result["embeddings"]) == 1
        assert result["dimension"] == 5

    def test_embed_multiple_texts(self, client: AragoraClient, mock_request) -> None:
        """Generate embeddings for multiple texts."""
        mock_request.return_value = {
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            "dimension": 3,
        }

        result = client.ml.embed(
            texts=[
                "First document",
                "Second document",
                "Third document",
            ]
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/embed",
            json={
                "texts": [
                    "First document",
                    "Second document",
                    "Third document",
                ]
            },
        )
        assert len(result["embeddings"]) == 3

    def test_embed_both_text_and_texts(self, client: AragoraClient, mock_request) -> None:
        """Generate embeddings with both text and texts provided."""
        mock_request.return_value = {
            "embeddings": [[0.1], [0.2], [0.3]],
            "dimension": 1,
        }

        client.ml.embed(
            text="Single text",
            texts=["Text 1", "Text 2"],
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["text"] == "Single text"
        assert call_json["texts"] == ["Text 1", "Text 2"]

    def test_embed_empty_request(self, client: AragoraClient, mock_request) -> None:
        """Generate embeddings with no text provided (edge case)."""
        mock_request.return_value = {"embeddings": [], "dimension": 0}

        client.ml.embed()

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/embed",
            json={},
        )


class TestMLSearch:
    """Tests for semantic search."""

    def test_search_basic(self, client: AragoraClient, mock_request) -> None:
        """Semantic search with basic parameters."""
        mock_request.return_value = {
            "results": [
                {"text": "Redis caching guide", "score": 0.95, "index": 0},
                {"text": "Caching best practices", "score": 0.88, "index": 2},
            ]
        }

        result = client.ml.search(
            query="caching strategies",
            documents=[
                "Redis caching guide",
                "Database indexing tips",
                "Caching best practices",
            ],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/search",
            json={
                "query": "caching strategies",
                "documents": [
                    "Redis caching guide",
                    "Database indexing tips",
                    "Caching best practices",
                ],
                "top_k": 5,
                "threshold": 0.0,
            },
        )
        assert len(result["results"]) == 2
        assert result["results"][0]["score"] == 0.95

    def test_search_custom_top_k(self, client: AragoraClient, mock_request) -> None:
        """Semantic search with custom top_k."""
        mock_request.return_value = {"results": [{"text": "Doc 1", "score": 0.99}]}

        client.ml.search(
            query="query",
            documents=["Doc 1", "Doc 2", "Doc 3"],
            top_k=1,
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["top_k"] == 1

    def test_search_with_threshold(self, client: AragoraClient, mock_request) -> None:
        """Semantic search with minimum similarity threshold."""
        mock_request.return_value = {"results": []}

        client.ml.search(
            query="specific query",
            documents=["Doc 1", "Doc 2"],
            threshold=0.8,
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["threshold"] == 0.8

    def test_search_all_parameters(self, client: AragoraClient, mock_request) -> None:
        """Semantic search with all parameters."""
        mock_request.return_value = {
            "results": [
                {"text": "Best match", "score": 0.92, "index": 5},
            ]
        }

        client.ml.search(
            query="advanced search",
            documents=["Doc " + str(i) for i in range(10)],
            top_k=3,
            threshold=0.7,
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["query"] == "advanced search"
        assert len(call_json["documents"]) == 10
        assert call_json["top_k"] == 3
        assert call_json["threshold"] == 0.7


# =========================================================================
# Training Data Export Tests
# =========================================================================


class TestMLExportTraining:
    """Tests for training data export."""

    def test_export_training_default_format(self, client: AragoraClient, mock_request) -> None:
        """Export training data with default JSON format."""
        mock_request.return_value = {
            "examples": 2,
            "data": [
                {"prompt": "task1", "chosen": "consensus1", "rejected": "rejected1"},
                {"prompt": "task2", "chosen": "consensus2", "rejected": "rejected2"},
            ],
        }

        result = client.ml.export_training(
            debates=[
                {"task": "task1", "consensus": "consensus1", "rejected": "rejected1"},
                {"task": "task2", "consensus": "consensus2", "rejected": "rejected2"},
            ]
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/ml/export-training",
            json={
                "debates": [
                    {"task": "task1", "consensus": "consensus1", "rejected": "rejected1"},
                    {"task": "task2", "consensus": "consensus2", "rejected": "rejected2"},
                ],
                "format": "json",
            },
        )
        assert result["examples"] == 2

    def test_export_training_jsonl_format(self, client: AragoraClient, mock_request) -> None:
        """Export training data in JSONL format."""
        mock_request.return_value = {
            "examples": 1,
            "data": '{"prompt":"task","chosen":"consensus","rejected":"rejected"}\n',
        }

        client.ml.export_training(
            debates=[{"task": "task", "consensus": "consensus", "rejected": "rejected"}],
            format="jsonl",
        )

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["format"] == "jsonl"

    def test_export_training_empty_debates(self, client: AragoraClient, mock_request) -> None:
        """Export with empty debates list."""
        mock_request.return_value = {"examples": 0, "data": []}

        result = client.ml.export_training(debates=[])

        call_args = mock_request.call_args
        call_json = call_args[1]["json"]
        assert call_json["debates"] == []
        assert result["examples"] == 0


# =========================================================================
# Models & Stats Tests
# =========================================================================


class TestMLListModels:
    """Tests for listing ML models."""

    def test_list_models(self, client: AragoraClient, mock_request) -> None:
        """List available ML models and capabilities."""
        mock_request.return_value = {
            "capabilities": {
                "routing": True,
                "scoring": True,
                "consensus_prediction": True,
                "embeddings": True,
                "semantic_search": True,
            },
            "models": {
                "embeddings": "text-embedding-3-small",
                "routing": "custom-routing-v2",
                "scoring": "quality-scorer-v1",
            },
        }

        result = client.ml.list_models()

        mock_request.assert_called_once_with("GET", "/api/v1/ml/models")
        assert result["capabilities"]["routing"] is True
        assert result["models"]["embeddings"] == "text-embedding-3-small"


class TestMLGetStats:
    """Tests for getting ML module statistics."""

    def test_get_stats(self, client: AragoraClient, mock_request) -> None:
        """Get ML module statistics."""
        mock_request.return_value = {
            "routing": {
                "total_requests": 1500,
                "accuracy": 0.89,
                "avg_latency_ms": 45,
            },
            "consensus_calibration": {
                "samples": 500,
                "brier_score": 0.12,
                "calibration_error": 0.05,
            },
        }

        result = client.ml.get_stats()

        mock_request.assert_called_once_with("GET", "/api/v1/ml/stats")
        assert result["routing"]["total_requests"] == 1500
        assert result["routing"]["accuracy"] == 0.89
        assert result["consensus_calibration"]["brier_score"] == 0.12


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncMLRoute:
    """Async tests for ML-based agent routing."""

    @pytest.mark.asyncio
    async def test_async_route_basic(self, mock_async_request) -> None:
        """Route agents asynchronously."""
        mock_async_request.return_value = {
            "selected_agents": ["claude", "gpt-4"],
            "confidence": 0.90,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.route(
                task="Async routing task",
                available_agents=["claude", "gpt-4", "gemini"],
                team_size=2,
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/ml/route",
                json={
                    "task": "Async routing task",
                    "available_agents": ["claude", "gpt-4", "gemini"],
                    "team_size": 2,
                },
            )
            assert result["selected_agents"] == ["claude", "gpt-4"]

    @pytest.mark.asyncio
    async def test_async_route_with_constraints(self, mock_async_request) -> None:
        """Route agents asynchronously with constraints."""
        mock_async_request.return_value = {"selected_agents": ["codex"]}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.ml.route(
                task="Coding task",
                available_agents=["claude", "codex"],
                constraints={"require_code": True},
            )

            call_args = mock_async_request.call_args
            call_json = call_args[1]["json"]
            assert call_json["constraints"] == {"require_code": True}


class TestAsyncMLScore:
    """Async tests for response scoring."""

    @pytest.mark.asyncio
    async def test_async_score(self, mock_async_request) -> None:
        """Score response asynchronously."""
        mock_async_request.return_value = {"overall": 0.88}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.score(text="Async response text")

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/ml/score",
                json={"text": "Async response text"},
            )
            assert result["overall"] == 0.88

    @pytest.mark.asyncio
    async def test_async_score_with_context(self, mock_async_request) -> None:
        """Score response asynchronously with context."""
        mock_async_request.return_value = {"overall": 0.92, "relevance": 0.95}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.score(
                text="Response text",
                context="Task context",
            )

            call_args = mock_async_request.call_args
            call_json = call_args[1]["json"]
            assert call_json["context"] == "Task context"
            assert result["relevance"] == 0.95


class TestAsyncMLScoreBatch:
    """Async tests for batch scoring."""

    @pytest.mark.asyncio
    async def test_async_score_batch(self, mock_async_request) -> None:
        """Score batch asynchronously."""
        mock_async_request.return_value = {"scores": [{"overall": 0.80}, {"overall": 0.85}]}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.score_batch(texts=["Text 1", "Text 2"])

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/ml/score-batch",
                json={"texts": ["Text 1", "Text 2"]},
            )
            assert len(result["scores"]) == 2

    @pytest.mark.asyncio
    async def test_async_score_batch_with_contexts(self, mock_async_request) -> None:
        """Score batch asynchronously with contexts."""
        mock_async_request.return_value = {"scores": [{"overall": 0.90}]}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.ml.score_batch(
                texts=["Response"],
                contexts=["Context"],
            )

            call_args = mock_async_request.call_args
            call_json = call_args[1]["json"]
            assert call_json["contexts"] == ["Context"]


class TestAsyncMLPredictConsensus:
    """Async tests for consensus prediction."""

    @pytest.mark.asyncio
    async def test_async_predict_consensus(self, mock_async_request) -> None:
        """Predict consensus asynchronously."""
        mock_async_request.return_value = {
            "probability": 0.85,
            "convergence_trend": "stable",
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.predict_consensus(
                responses=[
                    ("claude", "Position A"),
                    ("gpt-4", "Position A variant"),
                ]
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/ml/consensus",
                json={
                    "responses": [
                        ["claude", "Position A"],
                        ["gpt-4", "Position A variant"],
                    ],
                    "current_round": 1,
                    "total_rounds": 3,
                },
            )
            assert result["probability"] == 0.85

    @pytest.mark.asyncio
    async def test_async_predict_consensus_with_context(self, mock_async_request) -> None:
        """Predict consensus asynchronously with context."""
        mock_async_request.return_value = {"probability": 0.70}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.ml.predict_consensus(
                responses=[("agent", "response")],
                context="Task context",
                current_round=2,
                total_rounds=4,
            )

            call_args = mock_async_request.call_args
            call_json = call_args[1]["json"]
            assert call_json["context"] == "Task context"
            assert call_json["current_round"] == 2
            assert call_json["total_rounds"] == 4


class TestAsyncMLEmbed:
    """Async tests for text embeddings."""

    @pytest.mark.asyncio
    async def test_async_embed_single(self, mock_async_request) -> None:
        """Generate embedding asynchronously for single text."""
        mock_async_request.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "dimension": 3,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.embed(text="Single text")

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/ml/embed",
                json={"text": "Single text"},
            )
            assert result["dimension"] == 3

    @pytest.mark.asyncio
    async def test_async_embed_multiple(self, mock_async_request) -> None:
        """Generate embeddings asynchronously for multiple texts."""
        mock_async_request.return_value = {
            "embeddings": [[0.1], [0.2]],
            "dimension": 1,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.embed(texts=["Text 1", "Text 2"])

            call_args = mock_async_request.call_args
            call_json = call_args[1]["json"]
            assert call_json["texts"] == ["Text 1", "Text 2"]
            assert len(result["embeddings"]) == 2


class TestAsyncMLSearch:
    """Async tests for semantic search."""

    @pytest.mark.asyncio
    async def test_async_search(self, mock_async_request) -> None:
        """Semantic search asynchronously."""
        mock_async_request.return_value = {
            "results": [{"text": "Match", "score": 0.90, "index": 0}]
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.search(
                query="search query",
                documents=["Match", "No match"],
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/ml/search",
                json={
                    "query": "search query",
                    "documents": ["Match", "No match"],
                    "top_k": 5,
                    "threshold": 0.0,
                },
            )
            assert result["results"][0]["score"] == 0.90

    @pytest.mark.asyncio
    async def test_async_search_with_params(self, mock_async_request) -> None:
        """Semantic search asynchronously with custom parameters."""
        mock_async_request.return_value = {"results": []}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.ml.search(
                query="query",
                documents=["Doc 1"],
                top_k=10,
                threshold=0.5,
            )

            call_args = mock_async_request.call_args
            call_json = call_args[1]["json"]
            assert call_json["top_k"] == 10
            assert call_json["threshold"] == 0.5


class TestAsyncMLExportTraining:
    """Async tests for training data export."""

    @pytest.mark.asyncio
    async def test_async_export_training(self, mock_async_request) -> None:
        """Export training data asynchronously."""
        mock_async_request.return_value = {"examples": 1, "data": []}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.export_training(
                debates=[{"task": "t", "consensus": "c", "rejected": "r"}]
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/ml/export-training",
                json={
                    "debates": [{"task": "t", "consensus": "c", "rejected": "r"}],
                    "format": "json",
                },
            )
            assert result["examples"] == 1

    @pytest.mark.asyncio
    async def test_async_export_training_jsonl(self, mock_async_request) -> None:
        """Export training data asynchronously in JSONL format."""
        mock_async_request.return_value = {"examples": 0, "data": ""}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.ml.export_training(debates=[], format="jsonl")

            call_args = mock_async_request.call_args
            call_json = call_args[1]["json"]
            assert call_json["format"] == "jsonl"


class TestAsyncMLListModels:
    """Async tests for listing ML models."""

    @pytest.mark.asyncio
    async def test_async_list_models(self, mock_async_request) -> None:
        """List ML models asynchronously."""
        mock_async_request.return_value = {
            "capabilities": {"routing": True},
            "models": {"embeddings": "model-v1"},
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.list_models()

            mock_async_request.assert_called_once_with("GET", "/api/v1/ml/models")
            assert result["capabilities"]["routing"] is True


class TestAsyncMLGetStats:
    """Async tests for getting ML stats."""

    @pytest.mark.asyncio
    async def test_async_get_stats(self, mock_async_request) -> None:
        """Get ML stats asynchronously."""
        mock_async_request.return_value = {
            "routing": {"total_requests": 1000},
            "consensus_calibration": {"samples": 250},
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.ml.get_stats()

            mock_async_request.assert_called_once_with("GET", "/api/v1/ml/stats")
            assert result["routing"]["total_requests"] == 1000
            assert result["consensus_calibration"]["samples"] == 250
