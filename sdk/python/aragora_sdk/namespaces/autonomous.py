"""
Autonomous Namespace API

Provides methods for autonomous agent operations:
- Manage autonomous approvals
- Configure triggers
- Monitor autonomous executions

Learning endpoints (AutonomousLearningHandler):
- GET  /api/v2/learning/sessions         - List training sessions
- POST /api/v2/learning/sessions         - Start new training session
- GET  /api/v2/learning/sessions/:id     - Get session details
- GET  /api/v2/learning/metrics          - Get learning metrics
- GET  /api/v2/learning/metrics/:type    - Get specific metric
- POST /api/v2/learning/feedback         - Submit learning feedback
- GET  /api/v2/learning/patterns         - List detected patterns
- GET  /api/v2/learning/knowledge        - Get extracted knowledge
- GET  /api/v2/learning/recommendations  - Get learning recommendations
- GET  /api/v2/learning/performance      - Get model performance stats
- POST /api/v2/learning/calibrate        - Trigger calibration

Monitoring endpoints (MonitoringHandler):
- POST /api/v1/autonomous/monitoring/record          - Record metric value
- GET  /api/v1/autonomous/monitoring/trends           - Get all trends
- GET  /api/v1/autonomous/monitoring/trends/:name     - Get trend for metric
- GET  /api/v1/autonomous/monitoring/anomalies        - Get recent anomalies
- GET  /api/v1/autonomous/monitoring/baseline/:name   - Get baseline stats
- GET  /api/v1/autonomous/monitoring/circuit-breaker  - Get circuit breaker status
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list  # Preserve builtin list for type annotations


class AutonomousAPI:
    """Synchronous Autonomous API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Learning - Sessions
    # =========================================================================

    def list_sessions(
        self,
        *,
        status: str | None = None,
        mode: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List training sessions.

        GET /api/v2/learning/sessions

        Args:
            status: Filter by session status (pending, running, completed, failed, cancelled)
            mode: Filter by learning mode (supervised, reinforcement, self_supervised, transfer, federated)
            limit: Maximum sessions to return (1-100)
            offset: Pagination offset

        Returns:
            Dict with sessions array and pagination info
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if mode:
            params["mode"] = mode
        return self._client.request("GET", "/api/v2/learning/sessions", params=params)

    def create_session(
        self,
        name: str,
        *,
        mode: str = "supervised",
        total_epochs: int = 100,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Start a new training session.

        POST /api/v2/learning/sessions

        Args:
            name: Session name
            mode: Learning mode (supervised, reinforcement, self_supervised, transfer, federated)
            total_epochs: Total training epochs
            config: Optional session configuration

        Returns:
            Dict with created session and confirmation message
        """
        data: dict[str, Any] = {"name": name, "mode": mode, "total_epochs": total_epochs}
        if config:
            data["config"] = config
        return self._client.request("POST", "/api/v2/learning/sessions", json=data)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """
        Get training session details.

        GET /api/v2/learning/sessions/:session_id

        Args:
            session_id: Session identifier

        Returns:
            Session details including status, metrics, and progress
        """
        return self._client.request("GET", f"/api/v2/learning/sessions/{session_id}")

    # =========================================================================
    # Learning - Metrics
    # =========================================================================

    def get_learning_metrics(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Get learning metrics.

        GET /api/v2/learning/metrics

        Args:
            session_id: Filter by session
            agent_id: Filter by agent
            limit: Maximum metrics to return (1-1000)

        Returns:
            Dict with metrics array and count
        """
        params: dict[str, Any] = {"limit": limit}
        if session_id:
            params["session_id"] = session_id
        if agent_id:
            params["agent_id"] = agent_id
        return self._client.request("GET", "/api/v2/learning/metrics", params=params)

    def get_learning_metric_by_type(self, metric_type: str) -> dict[str, Any]:
        """
        Get metrics of a specific type with aggregates.

        GET /api/v2/learning/metrics/:metric_type

        Args:
            metric_type: Type of metric (accuracy, loss, precision, recall, f1_score, calibration, convergence)

        Returns:
            Dict with metric type, count, average, min, max, and recent values
        """
        return self._client.request("GET", f"/api/v2/learning/metrics/{metric_type}")

    # =========================================================================
    # Learning - Feedback
    # =========================================================================

    def submit_learning_feedback(
        self,
        *,
        feedback_type: str = "neutral",
        target_type: str,
        target_id: str,
        comment: str = "",
        rating: int | None = None,
    ) -> dict[str, Any]:
        """
        Submit feedback on learning outcomes.

        POST /api/v2/learning/feedback

        Args:
            feedback_type: Type of feedback (positive, negative, neutral, correction)
            target_type: Target type (session, pattern, knowledge)
            target_id: Target identifier
            comment: Feedback comment
            rating: Optional 1-5 rating

        Returns:
            Dict with submitted feedback and confirmation
        """
        data: dict[str, Any] = {
            "feedback_type": feedback_type,
            "target_type": target_type,
            "target_id": target_id,
            "comment": comment,
        }
        if rating is not None:
            data["rating"] = rating
        return self._client.request("POST", "/api/v2/learning/feedback", json=data)

    # =========================================================================
    # Learning - Patterns
    # =========================================================================

    def list_learning_patterns(
        self,
        *,
        pattern_type: str | None = None,
        validated: bool | None = None,
        min_confidence: float = 0.5,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        List detected learning patterns.

        GET /api/v2/learning/patterns

        Args:
            pattern_type: Filter by type (consensus, disagreement, agent_preference, topic_cluster, temporal, cross_debate)
            validated: Filter by validation status
            min_confidence: Minimum confidence threshold
            limit: Maximum patterns to return

        Returns:
            Dict with patterns array and count
        """
        params: dict[str, Any] = {"min_confidence": min_confidence, "limit": limit}
        if pattern_type:
            params["pattern_type"] = pattern_type
        if validated is not None:
            params["validated"] = str(validated).lower()
        return self._client.request("GET", "/api/v2/learning/patterns", params=params)

    # =========================================================================
    # Learning - Knowledge
    # =========================================================================

    def get_learning_knowledge(
        self,
        *,
        verified: bool | None = None,
        source_type: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get extracted knowledge.

        GET /api/v2/learning/knowledge

        Args:
            verified: Filter by verification status
            source_type: Filter by source type
            limit: Maximum items to return

        Returns:
            Dict with knowledge array and count
        """
        params: dict[str, Any] = {"limit": limit}
        if verified is not None:
            params["verified"] = str(verified).lower()
        if source_type:
            params["source_type"] = source_type
        return self._client.request("GET", "/api/v2/learning/knowledge", params=params)

    # =========================================================================
    # Learning - Recommendations & Performance
    # =========================================================================

    def get_learning_recommendations(self, *, limit: int = 10) -> dict[str, Any]:
        """
        Get learning recommendations.

        GET /api/v2/learning/recommendations

        Args:
            limit: Maximum recommendations to return (1-50)

        Returns:
            Dict with recommendations array and count
        """
        return self._client.request(
            "GET", "/api/v2/learning/recommendations", params={"limit": limit}
        )

    def get_learning_performance(self) -> dict[str, Any]:
        """
        Get model performance statistics.

        GET /api/v2/learning/performance

        Returns:
            Dict with performance stats (sessions, accuracy, loss, epochs, etc.)
        """
        return self._client.request("GET", "/api/v2/learning/performance")

    def calibrate_learning(
        self,
        *,
        agent_ids: _List[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Trigger model calibration.

        POST /api/v2/learning/calibrate

        Args:
            agent_ids: Optional list of agent IDs to calibrate
            force: Force re-calibration

        Returns:
            Dict with calibration ID and metric
        """
        data: dict[str, Any] = {"force": force}
        if agent_ids:
            data["agent_ids"] = agent_ids
        return self._client.request("POST", "/api/v2/learning/calibrate", json=data)

    # =========================================================================
    # Monitoring
    # =========================================================================

    def record_monitoring_metric(
        self,
        metric_name: str,
        value: float,
    ) -> dict[str, Any]:
        """
        Record a metric value for trend and anomaly detection.

        POST /api/v1/autonomous/monitoring/record

        Args:
            metric_name: Name of the metric (alphanumeric, underscores, hyphens, dots)
            value: Current metric value (must be a finite number)

        Returns:
            Dict with success status, value, and anomaly info if detected
        """
        return self._client.request(
            "POST",
            "/api/v1/autonomous/monitoring/record",
            json={"metric_name": metric_name, "value": value},
        )

    def get_monitoring_trends(self) -> dict[str, Any]:
        """
        Get trends for all monitored metrics.

        GET /api/v1/autonomous/monitoring/trends

        Returns:
            Dict with trends map and count
        """
        return self._client.request("GET", "/api/v1/autonomous/monitoring/trends")

    def get_monitoring_trend(
        self,
        metric_name: str,
        *,
        period_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Get trend for a specific metric.

        GET /api/v1/autonomous/monitoring/trends/:metric_name

        Args:
            metric_name: Name of the metric to analyze
            period_seconds: Time period to analyze in seconds (0-86400)

        Returns:
            Dict with trend data (direction, values, confidence)
        """
        params: dict[str, Any] = {}
        if period_seconds is not None:
            params["period_seconds"] = period_seconds
        return self._client.request(
            "GET",
            f"/api/v1/autonomous/monitoring/trends/{metric_name}",
            params=params or None,
        )

    def get_monitoring_anomalies(
        self,
        *,
        hours: int = 24,
        metric_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Get recent anomalies.

        GET /api/v1/autonomous/monitoring/anomalies

        Args:
            hours: Hours to look back (1-720, default 24)
            metric_name: Optional filter by metric name

        Returns:
            Dict with anomalies array and count
        """
        params: dict[str, Any] = {"hours": hours}
        if metric_name:
            params["metric_name"] = metric_name
        return self._client.request(
            "GET", "/api/v1/autonomous/monitoring/anomalies", params=params
        )

    def list_monitoring_baselines(self) -> dict[str, Any]:
        """
        List all monitoring baselines.

        GET /api/v1/autonomous/monitoring/baseline

        Returns:
            Dict with all baseline statistics across tracked metrics.
        """
        return self._client.request("GET", "/api/v1/autonomous/monitoring/baseline")

    def get_monitoring_baseline(self, metric_name: str) -> dict[str, Any]:
        """
        Get baseline statistics for a metric.

        GET /api/v1/autonomous/monitoring/baseline/:metric_name

        Args:
            metric_name: Name of the metric

        Returns:
            Dict with baseline stats (mean, stdev, min, max, median)
        """
        return self._client.request(
            "GET", f"/api/v1/autonomous/monitoring/baseline/{metric_name}"
        )

    def get_monitoring_circuit_breaker(self) -> dict[str, Any]:
        """
        Get monitoring circuit breaker status.

        GET /api/v1/autonomous/monitoring/circuit-breaker

        Returns:
            Dict with circuit breaker state, failure counts, and thresholds
        """
        return self._client.request("GET", "/api/v1/autonomous/monitoring/circuit-breaker")


class AsyncAutonomousAPI:
    """Asynchronous Autonomous API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Learning - Sessions
    # =========================================================================

    async def list_sessions(
        self,
        *,
        status: str | None = None,
        mode: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List training sessions. GET /api/v2/learning/sessions"""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if mode:
            params["mode"] = mode
        return await self._client.request("GET", "/api/v2/learning/sessions", params=params)

    async def create_session(
        self,
        name: str,
        *,
        mode: str = "supervised",
        total_epochs: int = 100,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new training session. POST /api/v2/learning/sessions"""
        data: dict[str, Any] = {"name": name, "mode": mode, "total_epochs": total_epochs}
        if config:
            data["config"] = config
        return await self._client.request("POST", "/api/v2/learning/sessions", json=data)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get training session details. GET /api/v2/learning/sessions/:session_id"""
        return await self._client.request("GET", f"/api/v2/learning/sessions/{session_id}")

    # =========================================================================
    # Learning - Metrics
    # =========================================================================

    async def get_learning_metrics(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get learning metrics. GET /api/v2/learning/metrics"""
        params: dict[str, Any] = {"limit": limit}
        if session_id:
            params["session_id"] = session_id
        if agent_id:
            params["agent_id"] = agent_id
        return await self._client.request("GET", "/api/v2/learning/metrics", params=params)

    async def get_learning_metric_by_type(self, metric_type: str) -> dict[str, Any]:
        """Get metrics of a specific type. GET /api/v2/learning/metrics/:metric_type"""
        return await self._client.request("GET", f"/api/v2/learning/metrics/{metric_type}")

    # =========================================================================
    # Learning - Feedback
    # =========================================================================

    async def submit_learning_feedback(
        self,
        *,
        feedback_type: str = "neutral",
        target_type: str,
        target_id: str,
        comment: str = "",
        rating: int | None = None,
    ) -> dict[str, Any]:
        """Submit feedback on learning outcomes. POST /api/v2/learning/feedback"""
        data: dict[str, Any] = {
            "feedback_type": feedback_type,
            "target_type": target_type,
            "target_id": target_id,
            "comment": comment,
        }
        if rating is not None:
            data["rating"] = rating
        return await self._client.request("POST", "/api/v2/learning/feedback", json=data)

    # =========================================================================
    # Learning - Patterns
    # =========================================================================

    async def list_learning_patterns(
        self,
        *,
        pattern_type: str | None = None,
        validated: bool | None = None,
        min_confidence: float = 0.5,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List detected learning patterns. GET /api/v2/learning/patterns"""
        params: dict[str, Any] = {"min_confidence": min_confidence, "limit": limit}
        if pattern_type:
            params["pattern_type"] = pattern_type
        if validated is not None:
            params["validated"] = str(validated).lower()
        return await self._client.request("GET", "/api/v2/learning/patterns", params=params)

    # =========================================================================
    # Learning - Knowledge
    # =========================================================================

    async def get_learning_knowledge(
        self,
        *,
        verified: bool | None = None,
        source_type: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get extracted knowledge. GET /api/v2/learning/knowledge"""
        params: dict[str, Any] = {"limit": limit}
        if verified is not None:
            params["verified"] = str(verified).lower()
        if source_type:
            params["source_type"] = source_type
        return await self._client.request("GET", "/api/v2/learning/knowledge", params=params)

    # =========================================================================
    # Learning - Recommendations & Performance
    # =========================================================================

    async def get_learning_recommendations(self, *, limit: int = 10) -> dict[str, Any]:
        """Get learning recommendations. GET /api/v2/learning/recommendations"""
        return await self._client.request(
            "GET", "/api/v2/learning/recommendations", params={"limit": limit}
        )

    async def get_learning_performance(self) -> dict[str, Any]:
        """Get model performance statistics. GET /api/v2/learning/performance"""
        return await self._client.request("GET", "/api/v2/learning/performance")

    async def calibrate_learning(
        self,
        *,
        agent_ids: _List[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Trigger model calibration. POST /api/v2/learning/calibrate"""
        data: dict[str, Any] = {"force": force}
        if agent_ids:
            data["agent_ids"] = agent_ids
        return await self._client.request("POST", "/api/v2/learning/calibrate", json=data)

    # =========================================================================
    # Monitoring
    # =========================================================================

    async def record_monitoring_metric(
        self,
        metric_name: str,
        value: float,
    ) -> dict[str, Any]:
        """Record a metric value. POST /api/v1/autonomous/monitoring/record"""
        return await self._client.request(
            "POST",
            "/api/v1/autonomous/monitoring/record",
            json={"metric_name": metric_name, "value": value},
        )

    async def get_monitoring_trends(self) -> dict[str, Any]:
        """Get trends for all monitored metrics. GET /api/v1/autonomous/monitoring/trends"""
        return await self._client.request("GET", "/api/v1/autonomous/monitoring/trends")

    async def get_monitoring_trend(
        self,
        metric_name: str,
        *,
        period_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Get trend for a specific metric. GET /api/v1/autonomous/monitoring/trends/:metric_name"""
        params: dict[str, Any] = {}
        if period_seconds is not None:
            params["period_seconds"] = period_seconds
        return await self._client.request(
            "GET",
            f"/api/v1/autonomous/monitoring/trends/{metric_name}",
            params=params or None,
        )

    async def get_monitoring_anomalies(
        self,
        *,
        hours: int = 24,
        metric_name: str | None = None,
    ) -> dict[str, Any]:
        """Get recent anomalies. GET /api/v1/autonomous/monitoring/anomalies"""
        params: dict[str, Any] = {"hours": hours}
        if metric_name:
            params["metric_name"] = metric_name
        return await self._client.request(
            "GET", "/api/v1/autonomous/monitoring/anomalies", params=params
        )

    async def list_monitoring_baselines(self) -> dict[str, Any]:
        """List all monitoring baselines. GET /api/v1/autonomous/monitoring/baseline"""
        return await self._client.request("GET", "/api/v1/autonomous/monitoring/baseline")

    async def get_monitoring_baseline(self, metric_name: str) -> dict[str, Any]:
        """Get baseline statistics. GET /api/v1/autonomous/monitoring/baseline/:metric_name"""
        return await self._client.request(
            "GET", f"/api/v1/autonomous/monitoring/baseline/{metric_name}"
        )

    async def get_monitoring_circuit_breaker(self) -> dict[str, Any]:
        """Get circuit breaker status. GET /api/v1/autonomous/monitoring/circuit-breaker"""
        return await self._client.request(
            "GET", "/api/v1/autonomous/monitoring/circuit-breaker"
        )

