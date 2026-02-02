"""
ML (Machine Learning) Namespace API.

Provides methods for ML capabilities:
- Agent routing recommendations
- Response quality scoring
- Consensus prediction
- Text embeddings and semantic search
- Training data export

Endpoints:
    POST /api/v1/ml/route          - Get ML-based agent routing
    POST /api/v1/ml/score          - Score response quality
    POST /api/v1/ml/score-batch    - Score multiple responses
    POST /api/v1/ml/consensus      - Predict consensus likelihood
    POST /api/v1/ml/embed          - Generate text embeddings
    POST /api/v1/ml/search         - Semantic search
    POST /api/v1/ml/export-training - Export debate data for training
    GET  /api/v1/ml/models         - List available ML models
    GET  /api/v1/ml/stats          - Get ML module statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ExportFormat = Literal["json", "jsonl"]


class MLAPI:
    """
    Synchronous ML API.

    Provides methods for ML-based agent routing, quality scoring,
    consensus prediction, and embeddings.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Get agent routing recommendation
        >>> routing = client.ml.route(
        ...     task="Implement a caching layer",
        ...     available_agents=["claude", "gpt-4", "codex"],
        ...     team_size=3,
        ... )
        >>> print(routing["selected_agents"])
        >>> # Score response quality
        >>> score = client.ml.score(text="The caching layer should use Redis...")
        >>> print(f"Quality: {score['overall']}")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Routing
    # =========================================================================

    def route(
        self,
        task: str,
        available_agents: list[str],
        team_size: int = 3,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get ML-based agent routing for a task.

        Args:
            task: The task description.
            available_agents: List of available agent names.
            team_size: Number of agents to select.
            constraints: Optional constraints (e.g., {"require_code": True}).

        Returns:
            Dict with selected_agents, task_type, confidence, reasoning.
        """
        data: dict[str, Any] = {
            "task": task,
            "available_agents": available_agents,
            "team_size": team_size,
        }
        if constraints:
            data["constraints"] = constraints

        return self._client.request("POST", "/api/v1/ml/route", json=data)

    # =========================================================================
    # Scoring
    # =========================================================================

    def score(
        self,
        text: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Score response quality.

        Args:
            text: The response text to score.
            context: Optional task context for relevance scoring.

        Returns:
            Dict with overall, coherence, completeness, relevance, clarity scores.
        """
        data: dict[str, Any] = {"text": text}
        if context:
            data["context"] = context

        return self._client.request("POST", "/api/v1/ml/score", json=data)

    def score_batch(
        self,
        texts: list[str],
        contexts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Score multiple responses in batch.

        Args:
            texts: List of response texts (max 100).
            contexts: Optional list of contexts (same length as texts).

        Returns:
            Dict with scores array.
        """
        data: dict[str, Any] = {"texts": texts}
        if contexts:
            data["contexts"] = contexts

        return self._client.request("POST", "/api/v1/ml/score-batch", json=data)

    # =========================================================================
    # Consensus Prediction
    # =========================================================================

    def predict_consensus(
        self,
        responses: list[tuple[str, str]],
        context: str | None = None,
        current_round: int = 1,
        total_rounds: int = 3,
    ) -> dict[str, Any]:
        """
        Predict consensus likelihood.

        Args:
            responses: List of (agent_name, response_text) tuples.
            context: Task context.
            current_round: Current debate round.
            total_rounds: Total planned rounds.

        Returns:
            Dict with probability, confidence, convergence_trend, etc.
        """
        data: dict[str, Any] = {
            "responses": [[agent, text] for agent, text in responses],
            "current_round": current_round,
            "total_rounds": total_rounds,
        }
        if context:
            data["context"] = context

        return self._client.request("POST", "/api/v1/ml/consensus", json=data)

    # =========================================================================
    # Embeddings
    # =========================================================================

    def embed(
        self,
        text: str | None = None,
        texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate text embeddings.

        Args:
            text: Single text to embed.
            texts: List of texts to embed (max 100).

        Returns:
            Dict with embeddings array and dimension.
        """
        data: dict[str, Any] = {}
        if text:
            data["text"] = text
        if texts:
            data["texts"] = texts

        return self._client.request("POST", "/api/v1/ml/embed", json=data)

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """
        Semantic search over documents.

        Args:
            query: Search query.
            documents: List of documents to search (max 1000).
            top_k: Number of results to return.
            threshold: Minimum similarity threshold.

        Returns:
            Dict with results array (text, score, index).
        """
        return self._client.request(
            "POST",
            "/api/v1/ml/search",
            json={
                "query": query,
                "documents": documents,
                "top_k": top_k,
                "threshold": threshold,
            },
        )

    # =========================================================================
    # Training Data Export
    # =========================================================================

    def export_training(
        self,
        debates: list[dict[str, Any]],
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export debate data for training.

        Requires ml:train permission.

        Args:
            debates: List of debate data dicts with task, consensus, rejected.
            format: Output format ("json" or "jsonl").

        Returns:
            Dict with examples count and data.
        """
        return self._client.request(
            "POST",
            "/api/v1/ml/export-training",
            json={"debates": debates, "format": format},
        )

    # =========================================================================
    # Models & Stats
    # =========================================================================

    def list_models(self) -> dict[str, Any]:
        """
        List available ML models and capabilities.

        Returns:
            Dict with capabilities and models info.
        """
        return self._client.request("GET", "/api/v1/ml/models")

    def get_stats(self) -> dict[str, Any]:
        """
        Get ML module statistics.

        Returns:
            Dict with routing, consensus calibration stats.
        """
        return self._client.request("GET", "/api/v1/ml/stats")


class AsyncMLAPI:
    """Asynchronous ML API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Routing
    # =========================================================================

    async def route(
        self,
        task: str,
        available_agents: list[str],
        team_size: int = 3,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get ML-based agent routing for a task."""
        data: dict[str, Any] = {
            "task": task,
            "available_agents": available_agents,
            "team_size": team_size,
        }
        if constraints:
            data["constraints"] = constraints

        return await self._client.request("POST", "/api/v1/ml/route", json=data)

    # =========================================================================
    # Scoring
    # =========================================================================

    async def score(
        self,
        text: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Score response quality."""
        data: dict[str, Any] = {"text": text}
        if context:
            data["context"] = context

        return await self._client.request("POST", "/api/v1/ml/score", json=data)

    async def score_batch(
        self,
        texts: list[str],
        contexts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Score multiple responses in batch."""
        data: dict[str, Any] = {"texts": texts}
        if contexts:
            data["contexts"] = contexts

        return await self._client.request("POST", "/api/v1/ml/score-batch", json=data)

    # =========================================================================
    # Consensus Prediction
    # =========================================================================

    async def predict_consensus(
        self,
        responses: list[tuple[str, str]],
        context: str | None = None,
        current_round: int = 1,
        total_rounds: int = 3,
    ) -> dict[str, Any]:
        """Predict consensus likelihood."""
        data: dict[str, Any] = {
            "responses": [[agent, text] for agent, text in responses],
            "current_round": current_round,
            "total_rounds": total_rounds,
        }
        if context:
            data["context"] = context

        return await self._client.request("POST", "/api/v1/ml/consensus", json=data)

    # =========================================================================
    # Embeddings
    # =========================================================================

    async def embed(
        self,
        text: str | None = None,
        texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate text embeddings."""
        data: dict[str, Any] = {}
        if text:
            data["text"] = text
        if texts:
            data["texts"] = texts

        return await self._client.request("POST", "/api/v1/ml/embed", json=data)

    async def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Semantic search over documents."""
        return await self._client.request(
            "POST",
            "/api/v1/ml/search",
            json={
                "query": query,
                "documents": documents,
                "top_k": top_k,
                "threshold": threshold,
            },
        )

    # =========================================================================
    # Training Data Export
    # =========================================================================

    async def export_training(
        self,
        debates: list[dict[str, Any]],
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """Export debate data for training."""
        return await self._client.request(
            "POST",
            "/api/v1/ml/export-training",
            json={"debates": debates, "format": format},
        )

    # =========================================================================
    # Models & Stats
    # =========================================================================

    async def list_models(self) -> dict[str, Any]:
        """List available ML models and capabilities."""
        return await self._client.request("GET", "/api/v1/ml/models")

    async def get_stats(self) -> dict[str, Any]:
        """Get ML module statistics."""
        return await self._client.request("GET", "/api/v1/ml/stats")
