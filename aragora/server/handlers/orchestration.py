"""
Unified Orchestration Handler for Aragora Control Plane.

Provides a single API that embodies "Control plane for multi-agent
vetted decisionmaking across org knowledge and channels":

- Accept knowledge context sources (Slack channel, Confluence page, etc.)
- Auto-fetch content as vetted decisionmaking context
- Run vetted decisionmaking with specified or auto-selected agent team
- Route results to output channels
- Return receipt with full provenance

Endpoints:
    POST /api/v1/orchestration/deliberate     - Unified vetted decisionmaking endpoint
    GET  /api/v1/orchestration/status/:id     - Get vetted decisionmaking status
    GET  /api/v1/orchestration/templates      - List available templates
    POST /api/v1/orchestration/deliberate/sync - Synchronous vetted decisionmaking
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, cast

from aragora.config import MAX_ROUNDS

if TYPE_CHECKING:
    pass

# =============================================================================
# Type Protocols for External Connectors
# =============================================================================


class ConfluenceConnectorProtocol(Protocol):
    """Protocol for Confluence connector with page content fetching."""

    async def get_page_content(self, page_id: str) -> str | None:
        """Fetch content from a Confluence page."""
        ...


class GitHubConnectorProtocol(Protocol):
    """Protocol for GitHub connector with PR/issue content fetching."""

    async def get_pr_content(self, owner: str, repo: str, number: int) -> str | None:
        """Fetch content from a GitHub PR."""
        ...

    async def get_issue_content(self, owner: str, repo: str, number: int) -> str | None:
        """Fetch content from a GitHub issue."""
        ...


class JiraConnectorProtocol(Protocol):
    """Protocol for Jira connector with issue fetching."""

    async def get_issue(self, issue_key: str) -> Optional[dict[str, Any]]:
        """Fetch a Jira issue."""
        ...


class EmailSenderProtocol(Protocol):
    """Protocol for email sending function."""

    async def __call__(self, to: str, subject: str, body: str) -> None:
        """Send an email."""
        ...


class KnowledgeMoundProtocol(Protocol):
    """Protocol for Knowledge Mound search interface."""

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search the knowledge mound."""
        ...


# Type alias for recommend_agents function
RecommendAgentsFunc = Callable[[str], "Any"]
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.http_utils import run_async

logger = logging.getLogger(__name__)

# =============================================================================
# Data Models
# =============================================================================


class TeamStrategy(Enum):
    """Strategy for selecting the agent team."""

    SPECIFIED = "specified"  # Use explicitly provided agents
    BEST_FOR_DOMAIN = "best_for_domain"  # Auto-select based on domain
    DIVERSE = "diverse"  # Maximize agent diversity
    FAST = "fast"  # Optimize for speed
    RANDOM = "random"  # Random selection


class OutputFormat(Enum):
    """Format for vetted decisionmaking output."""

    STANDARD = "standard"  # Default JSON response
    DECISION_RECEIPT = "decision_receipt"  # Full audit receipt
    SUMMARY = "summary"  # Condensed summary
    GITHUB_REVIEW = "github_review"  # GitHub PR review format
    SLACK_MESSAGE = "slack_message"  # Slack-formatted message


@dataclass
class KnowledgeContextSource:
    """A source of knowledge context for vetted decisionmaking."""

    source_type: str  # slack, confluence, github, document, etc.
    source_id: str  # Channel ID, page ID, PR number, etc.
    lookback_minutes: int = 60
    max_items: int = 50

    @classmethod
    def from_string(cls, source_str: str) -> "KnowledgeContextSource":
        """Parse from 'type:id' format."""
        if ":" not in source_str:
            return cls(source_type="document", source_id=source_str)
        parts = source_str.split(":", 1)
        return cls(source_type=parts[0], source_id=parts[1])


@dataclass
class OutputChannel:
    """A channel to route vetted decisionmaking results to."""

    channel_type: str  # slack, teams, discord, telegram, email, webhook
    channel_id: str  # Channel ID, email address, webhook URL
    thread_id: str | None = None  # Optional thread/conversation ID

    @classmethod
    def from_string(cls, channel_str: str) -> "OutputChannel":
        """Parse from 'type:id' or 'type:id:thread_id' format.

        Special handling for URLs (webhook:https://...) to avoid splitting on
        the protocol colon.
        """
        if ":" not in channel_str:
            return cls(channel_type="webhook", channel_id=channel_str)

        # Split on first colon only to get type
        parts = channel_str.split(":", 1)
        channel_type = parts[0].lower()

        # For webhooks with URLs, the rest is the channel_id
        if channel_type == "webhook" or parts[1].startswith("http"):
            return cls(channel_type=channel_type, channel_id=parts[1])

        # For other types, check for thread_id (type:id:thread_id)
        remaining = parts[1]
        if ":" in remaining:
            id_parts = remaining.split(":", 1)
            return cls(channel_type=channel_type, channel_id=id_parts[0], thread_id=id_parts[1])

        return cls(channel_type=channel_type, channel_id=remaining)


@dataclass
class OrchestrationRequest:
    """Request for a unified orchestration vetted decisionmaking session."""

    question: str
    knowledge_sources: list[KnowledgeContextSource] = field(default_factory=list)
    workspaces: list[str] = field(default_factory=list)
    team_strategy: TeamStrategy = TeamStrategy.BEST_FOR_DOMAIN
    agents: list[str] = field(default_factory=list)
    output_channels: list[OutputChannel] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.STANDARD
    require_consensus: bool = True
    priority: str = "normal"
    max_rounds: int = MAX_ROUNDS
    timeout_seconds: float = 300.0
    template: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Assigned by system
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestrationRequest":
        """Create from request payload."""
        # Parse knowledge sources
        knowledge_sources = []
        for source in data.get("knowledge_sources", []):
            if isinstance(source, str):
                knowledge_sources.append(KnowledgeContextSource.from_string(source))
            elif isinstance(source, dict):
                knowledge_sources.append(
                    KnowledgeContextSource(
                        source_type=source.get("type", "document"),
                        source_id=source.get("id", ""),
                        lookback_minutes=source.get("lookback_minutes", 60),
                        max_items=source.get("max_items", 50),
                    )
                )

        # Also support nested knowledge_context format
        if "knowledge_context" in data:
            ctx = data["knowledge_context"]
            for source in ctx.get("sources", []):
                if isinstance(source, str):
                    knowledge_sources.append(KnowledgeContextSource.from_string(source))

        # Parse output channels
        output_channels = []
        for channel in data.get("output_channels", []):
            if isinstance(channel, str):
                output_channels.append(OutputChannel.from_string(channel))
            elif isinstance(channel, dict):
                output_channels.append(
                    OutputChannel(
                        channel_type=channel.get("type", "webhook"),
                        channel_id=channel.get("id", ""),
                        thread_id=channel.get("thread_id"),
                    )
                )

        # Parse team strategy
        strategy_str = data.get("team_strategy", "best_for_domain")
        try:
            team_strategy = TeamStrategy(strategy_str)
        except ValueError:
            team_strategy = TeamStrategy.BEST_FOR_DOMAIN

        # Parse output format
        format_str = data.get("output_format", "standard")
        try:
            output_format = OutputFormat(format_str)
        except ValueError:
            output_format = OutputFormat.STANDARD

        return cls(
            question=data.get("question", ""),
            knowledge_sources=knowledge_sources,
            workspaces=data.get(
                "workspaces", data.get("knowledge_context", {}).get("workspaces", [])
            ),
            team_strategy=team_strategy,
            agents=data.get("agents", []),
            output_channels=output_channels,
            output_format=output_format,
            require_consensus=data.get("require_consensus", True),
            priority=data.get("priority", "normal"),
            max_rounds=data.get("max_rounds", MAX_ROUNDS),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            template=data.get("template"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OrchestrationResult:
    """Result of a unified orchestration vetted decisionmaking session."""

    request_id: str
    success: bool
    consensus_reached: bool = False
    final_answer: str | None = None
    confidence: float | None = None
    agents_participated: list[str] = field(default_factory=list)
    rounds_completed: int = 0
    duration_seconds: float = 0.0
    knowledge_context_used: list[str] = field(default_factory=list)
    channels_notified: list[str] = field(default_factory=list)
    receipt_id: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "consensus_reached": self.consensus_reached,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "agents_participated": self.agents_participated,
            "rounds_completed": self.rounds_completed,
            "duration_seconds": self.duration_seconds,
            "knowledge_context_used": self.knowledge_context_used,
            "channels_notified": self.channels_notified,
            "receipt_id": self.receipt_id,
            "error": self.error,
            "created_at": self.created_at,
        }


# =============================================================================
# Deliberation Templates - Import from templates module
# =============================================================================

# Import from the dedicated templates module for extensibility
# Pre-declare types for fallback case
DeliberationTemplate: Any = None
TEMPLATES: dict[str, Any] = {}
_list_templates: Any = None
_get_template: Any = None

try:
    from aragora.deliberation.templates import (
        DeliberationTemplate,
        BUILTIN_TEMPLATES,
        list_templates as _list_templates,
        get_template as _get_template,
    )

    # Use the expanded template set from the templates module
    TEMPLATES = BUILTIN_TEMPLATES
except ImportError:
    # Fallback if templates module not available
    @dataclass
    class _FallbackDeliberationTemplate:
        """Pre-built vetted decisionmaking pattern (fallback)."""

        name: str
        description: str
        default_agents: list[str] = field(default_factory=list)
        default_knowledge_sources: list[str] = field(default_factory=list)
        output_format: Any = OutputFormat.STANDARD
        consensus_threshold: float = 0.7
        max_rounds: int = MAX_ROUNDS
        personas: list[str] = field(default_factory=list)

        def to_dict(self) -> dict[str, Any]:
            """Convert to dictionary."""
            output_fmt = self.output_format
            output_value = output_fmt.value if hasattr(output_fmt, "value") else str(output_fmt)
            return {
                "name": self.name,
                "description": self.description,
                "default_agents": self.default_agents,
                "default_knowledge_sources": self.default_knowledge_sources,
                "output_format": output_value,
                "consensus_threshold": self.consensus_threshold,
                "max_rounds": self.max_rounds,
                "personas": self.personas,
            }

    DeliberationTemplate = _FallbackDeliberationTemplate

    # Fallback templates
    TEMPLATES = {
        "code_review": _FallbackDeliberationTemplate(
            name="code_review",
            description="Multi-agent code review with security focus",
            default_agents=["anthropic-api", "openai-api", "codestral"],
            default_knowledge_sources=["github:pr"],
            output_format=OutputFormat.GITHUB_REVIEW,
            consensus_threshold=0.7,
            max_rounds=3,
            personas=["security", "performance", "maintainability"],
        ),
        "quick_decision": _FallbackDeliberationTemplate(
            name="quick_decision",
            description="Fast decision with minimal agents",
            default_agents=["anthropic-api", "openai-api"],
            output_format=OutputFormat.SUMMARY,
            consensus_threshold=0.5,
            max_rounds=2,
        ),
    }

    def _list_templates(**kwargs: Any) -> list[Any]:
        return list(TEMPLATES.values())

    def _get_template(name: str) -> Any:
        return TEMPLATES.get(name)

# =============================================================================
# In-Memory State (Production would use Redis/DB)
# =============================================================================

_orchestration_requests: dict[str, OrchestrationRequest] = {}
_orchestration_results: dict[str, OrchestrationResult] = {}

# =============================================================================
# Handler Implementation
# =============================================================================


class OrchestrationHandler(SecureHandler):
    """
    Unified orchestration handler for the Aragora control plane.

    This handler provides the primary API for the "Control plane for
    multi-agent vetted decisionmaking across org knowledge and channels" positioning.

    RBAC Permissions:
    - orchestration:read - View templates and deliberation status
    - orchestration:execute - Run deliberations
    """

    RESOURCE_TYPE = "orchestration"

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/v1/orchestration/")

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route GET requests."""
        # Require authentication for all orchestration operations
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        # Check read permission
        try:
            self.check_permission(auth_context, "orchestration:read")
        except ForbiddenError:
            logger.warning(f"Orchestration read denied for user {auth_context.user_id}")
            return error_response("Permission denied: orchestration:read", 403)

        if path == "/api/v1/orchestration/templates":
            return self._get_templates(query_params)

        if path.startswith("/api/v1/orchestration/status/"):
            request_id = path.split("/")[-1]
            return self._get_status(request_id)

        return None

    async def handle_post(
        self,
        path: str,
        data: dict[str, Any],
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Route POST requests."""
        # Require authentication for all orchestration operations
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        # Execute operations require execute permission
        try:
            self.check_permission(auth_context, "orchestration:execute")
        except ForbiddenError:
            logger.warning(f"Orchestration execute denied for user {auth_context.user_id}")
            return error_response("Permission denied: orchestration:execute", 403)

        if path == "/api/v1/orchestration/deliberate":
            return self._handle_deliberate(data, handler, sync=False)

        if path == "/api/v1/orchestration/deliberate/sync":
            return self._handle_deliberate(data, handler, sync=True)

        return None

    # =========================================================================
    # Endpoint Handlers
    # =========================================================================

    def _get_templates(self, query_params: dict[str, Any]) -> HandlerResult:
        """
        GET /api/v1/orchestration/templates

        Returns list of available vetted decisionmaking templates.
        """
        templates = [t.to_dict() for t in TEMPLATES.values()]
        return json_response(
            {
                "templates": templates,
                "count": len(templates),
            }
        )

    def _get_status(self, request_id: str) -> HandlerResult:
        """
        GET /api/v1/orchestration/status/:id

        Returns status of a vetted decisionmaking request.
        """
        if request_id in _orchestration_results:
            result = _orchestration_results[request_id]
            return json_response(
                {
                    "request_id": request_id,
                    "status": "completed" if result.success else "failed",
                    "result": result.to_dict(),
                }
            )

        if request_id in _orchestration_requests:
            return json_response(
                {
                    "request_id": request_id,
                    "status": "in_progress",
                    "result": None,
                }
            )

        return error_response("Request not found", 404)

    @rate_limit(requests_per_minute=30)
    def _handle_deliberate(
        self, data: dict[str, Any], handler: Any, sync: bool = False
    ) -> HandlerResult:
        """
        POST /api/v1/orchestration/deliberate
        POST /api/v1/orchestration/deliberate/sync

        Unified vetted decisionmaking endpoint that:
        1. Fetches context from knowledge sources
        2. Selects agent team based on strategy
        3. Runs vetted decisionmaking
        4. Routes results to output channels
        5. Returns receipt with provenance
        """
        try:
            # Parse request
            request = OrchestrationRequest.from_dict(data)

            # Validate
            if not request.question:
                return error_response("Question is required", 400)

            # Apply template if specified
            if request.template and request.template in TEMPLATES:
                template = TEMPLATES[request.template]
                if not request.agents:
                    request.agents = template.default_agents
                if not request.knowledge_sources:
                    for src in template.default_knowledge_sources:
                        request.knowledge_sources.append(KnowledgeContextSource.from_string(src))
                request.output_format = cast(OutputFormat, template.output_format)
                request.max_rounds = template.max_rounds

            # Store request
            _orchestration_requests[request.request_id] = request

            if sync:
                # Synchronous execution
                result = run_async(self._execute_deliberation(request))
                _orchestration_results[request.request_id] = result
                return json_response(result.to_dict())
            else:
                # Async execution - queue and return immediately
                asyncio.create_task(self._execute_and_store(request))
                return json_response(
                    {
                        "request_id": request.request_id,
                        "status": "queued",
                        "message": "Deliberation queued. Check status at /api/v1/orchestration/status/{request_id}",
                    },
                    status=202,
                )

        except Exception as e:
            logger.exception(f"Orchestration error: {e}")
            return error_response(f"Orchestration failed: {str(e)}", 500)

    # =========================================================================
    # Core Orchestration Logic
    # =========================================================================

    async def _execute_and_store(self, request: OrchestrationRequest) -> None:
        """Execute vetted decisionmaking and store result."""
        try:
            result = await self._execute_deliberation(request)
            _orchestration_results[request.request_id] = result
        except Exception as e:
            logger.exception(f"Async vetted decisionmaking failed: {e}")
            _orchestration_results[request.request_id] = OrchestrationResult(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )
        finally:
            # Clean up request from in-progress
            _orchestration_requests.pop(request.request_id, None)

    async def _execute_deliberation(self, request: OrchestrationRequest) -> OrchestrationResult:
        """
        Execute a unified orchestration vetted decisionmaking session.

        This is the core method that ties together:
        - Knowledge context fetching
        - Agent team selection
        - Deliberation execution
        - Output channel routing
        """
        import time

        start_time = time.time()
        knowledge_context_used: list[str] = []
        channels_notified: list[str] = []

        try:
            # Step 1: Fetch knowledge context
            context_parts: list[str] = []
            for source in request.knowledge_sources:
                try:
                    context = await self._fetch_knowledge_context(source)
                    if context:
                        context_parts.append(
                            f"[{source.source_type}:{source.source_id}]\n{context}"
                        )
                        knowledge_context_used.append(f"{source.source_type}:{source.source_id}")
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch context from {source.source_type}:{source.source_id}: {e}"
                    )

            # Combine context with question
            full_context = "\n\n".join(context_parts) if context_parts else None

            # Step 2: Select agent team
            agents = await self._select_agent_team(request)

            # Step 3: Run vetted decisionmaking
            from typing import cast as typing_cast
            from aragora.control_plane.deliberation import DeliberationManager
            from aragora.control_plane.coordinator import ControlPlaneCoordinator

            # coordinator may be None if not configured in server context
            raw_coordinator = self.ctx.get("control_plane_coordinator")
            coordinator: ControlPlaneCoordinator | None = (
                typing_cast(ControlPlaneCoordinator, raw_coordinator)
                if raw_coordinator is not None
                else None
            )
            manager = DeliberationManager(coordinator=coordinator)

            if coordinator:
                # Use control plane task scheduling
                task_id = await manager.submit_deliberation(
                    question=request.question,
                    context=full_context,
                    agents=agents,
                    priority=request.priority,
                    timeout_seconds=request.timeout_seconds,
                    max_rounds=request.max_rounds,
                    consensus_required=request.require_consensus,
                    metadata={
                        "orchestration_request_id": request.request_id,
                        "knowledge_sources": knowledge_context_used,
                        "output_channels": [
                            f"{c.channel_type}:{c.channel_id}" for c in request.output_channels
                        ],
                    },
                )

                # Wait for result
                outcome = await manager.wait_for_outcome(
                    task_id, timeout=request.timeout_seconds + 30
                )

                if outcome:
                    # Build result from outcome
                    result = OrchestrationResult(
                        request_id=request.request_id,
                        success=outcome.success,
                        consensus_reached=outcome.consensus_reached,
                        final_answer=outcome.winning_position,
                        confidence=outcome.consensus_confidence,
                        agents_participated=agents,
                        rounds_completed=0,  # Would come from outcome
                        duration_seconds=time.time() - start_time,
                        knowledge_context_used=knowledge_context_used,
                        receipt_id=task_id,
                    )
                else:
                    result = OrchestrationResult(
                        request_id=request.request_id,
                        success=False,
                        error="Deliberation timed out",
                        duration_seconds=time.time() - start_time,
                        knowledge_context_used=knowledge_context_used,
                    )
            else:
                # Fallback: Direct vetted decisionmaking without control plane
                from aragora.core.decision import DecisionRequest, get_decision_router, DecisionType

                decision_request = DecisionRequest(
                    content=request.question,
                    decision_type=DecisionType.DEBATE,
                )

                router = get_decision_router()
                decision_result = await router.route(decision_request)

                result = OrchestrationResult(
                    request_id=request.request_id,
                    success=decision_result.success,
                    consensus_reached=getattr(decision_result, "consensus_reached", False),
                    final_answer=getattr(decision_result, "final_answer", None),
                    confidence=getattr(decision_result, "confidence", None),
                    agents_participated=agents,
                    rounds_completed=getattr(decision_result, "rounds", 0),
                    duration_seconds=time.time() - start_time,
                    knowledge_context_used=knowledge_context_used,
                )

            # Step 4: Route results to output channels
            if result.success:
                for channel in request.output_channels:
                    try:
                        await self._route_to_channel(channel, result, request)
                        channels_notified.append(f"{channel.channel_type}:{channel.channel_id}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to route to {channel.channel_type}:{channel.channel_id}: {e}"
                        )

            result.channels_notified = channels_notified
            return result

        except Exception as e:
            logger.exception(f"Deliberation execution failed: {e}")
            return OrchestrationResult(
                request_id=request.request_id,
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
                knowledge_context_used=knowledge_context_used,
            )

    async def _fetch_knowledge_context(self, source: KnowledgeContextSource) -> str | None:
        """Fetch context from a knowledge source."""
        source_type = source.source_type.lower()

        # Chat platform channels (Slack, Teams, Discord, Telegram, WhatsApp)
        chat_platforms = {"slack", "teams", "discord", "telegram", "whatsapp", "google_chat"}
        if source_type in chat_platforms:
            return await self._fetch_chat_context(source_type, source)

        # Confluence page
        if source_type == "confluence":
            return await self._fetch_confluence_context(source)

        # GitHub PR/issue
        if source_type == "github":
            return await self._fetch_github_context(source)

        # Document from knowledge mound
        if source_type in ("document", "doc", "km"):
            return await self._fetch_document_context(source)

        # Jira issue
        if source_type == "jira":
            return await self._fetch_jira_context(source)

        logger.warning(f"Unknown knowledge source type: {source_type}")
        return None

    async def _fetch_chat_context(
        self, platform: str, source: KnowledgeContextSource
    ) -> str | None:
        """
        Fetch context from any chat platform using the registry.

        Supports: Slack, Teams, Discord, Telegram, WhatsApp, Google Chat.
        """
        try:
            from aragora.connectors.chat.registry import get_connector

            connector = get_connector(platform)
            if connector is None:
                logger.warning(f"No {platform} connector available")
                return None

            ctx = await connector.fetch_context(
                channel_id=source.source_id,
                lookback_minutes=source.lookback_minutes,
                max_messages=source.max_items,
            )
            if ctx and ctx.messages:
                return ctx.to_context_string(max_messages=source.max_items)

            return None
        except Exception as e:
            logger.warning(f"Failed to fetch {platform} context: {e}")
        return None

    async def _fetch_confluence_context(self, source: KnowledgeContextSource) -> str | None:
        """Fetch content from a Confluence page.

        Note: ConfluenceConnector requires base_url configuration. This method
        expects the connector to be configured via environment or context.
        The source_id could be a page_id like "confluence-12345" or just "12345".
        """
        try:
            import os
            from aragora.connectors.enterprise.collaboration.confluence import ConfluenceConnector

            # Get base_url from context or environment
            ctx_url = self.ctx.get("confluence_url")
            confluence_url: str = (
                str(ctx_url) if ctx_url else os.environ.get("CONFLUENCE_BASE_URL", "")
            )
            if not confluence_url:
                logger.warning("Confluence base URL not configured")
                return None

            connector = ConfluenceConnector(base_url=confluence_url)
            # Use the fetch method which exists on the connector
            # source_id could be page_id or space_key/page_title
            page_id = source.source_id
            if not page_id.startswith("confluence-"):
                page_id = f"confluence-{page_id}"
            evidence = await connector.fetch(page_id)
            if evidence:
                return str(getattr(evidence, "content", ""))
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch Confluence context: {e}")
        return None

    async def _fetch_github_context(self, source: KnowledgeContextSource) -> str | None:
        """Fetch content from a GitHub PR or issue.

        Uses the GitHubConnector's search method to retrieve content.
        source_id format: owner/repo/pr/123 or owner/repo/issue/123
        """
        try:
            from aragora.connectors.github import GitHubConnector

            # source_id format: owner/repo/pr/123 or owner/repo/issue/123
            parts = source.source_id.split("/")
            if len(parts) >= 4:
                owner, repo, item_type, number = parts[0], parts[1], parts[2], parts[3]
                full_repo = f"{owner}/{repo}"
                connector = GitHubConnector(repo=full_repo)

                # Use search to find the specific PR/issue
                search_type = "prs" if item_type == "pr" else "issues"
                results = await connector.search(
                    query=f"#{number}",
                    limit=1,
                    search_type=search_type,
                )
                if results:
                    # Evidence objects have content attribute
                    return str(getattr(results[0], "content", ""))
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch GitHub context: {e}")
        return None

    async def _fetch_document_context(self, source: KnowledgeContextSource) -> str | None:
        """Fetch content from knowledge mound."""
        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound()
            if mound:
                # Query for relevant knowledge using the KnowledgeMound API
                results = await mound.query(source.source_id, limit=source.max_items)
                if results and hasattr(results, "items"):
                    # QueryResult has an items attribute with KnowledgeItem objects
                    return "\n\n".join(
                        str(getattr(item, "content", ""))
                        for item in results.items
                        if getattr(item, "content", None)
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch document context: {e}")
        return None

    async def _fetch_jira_context(self, source: KnowledgeContextSource) -> str | None:
        """Fetch content from a Jira issue.

        Note: JiraConnector requires base_url configuration. This method
        expects the connector to be configured via environment or context.
        The source_id should be a Jira issue key like "PROJ-123".
        """
        try:
            import os
            from aragora.connectors.enterprise.collaboration.jira import JiraConnector

            # Get base_url from context or environment
            ctx_url = self.ctx.get("jira_url")
            jira_url: str = str(ctx_url) if ctx_url else os.environ.get("JIRA_BASE_URL", "")
            if not jira_url:
                logger.warning("Jira base URL not configured")
                return None

            connector = JiraConnector(base_url=jira_url)
            # Use the fetch method which exists on the connector
            issue_key = source.source_id
            if not issue_key.startswith("jira-"):
                issue_key = f"jira-{issue_key}"
            evidence = await connector.fetch(issue_key)
            if evidence:
                return str(getattr(evidence, "content", ""))
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch Jira context: {e}")
        return None

    async def _select_agent_team(self, request: OrchestrationRequest) -> list[str]:
        """Select agent team based on strategy."""
        # If agents explicitly specified, use them
        if request.agents:
            return request.agents

        # Default agent pool
        default_agents = ["anthropic-api", "openai-api", "gemini", "mistral"]

        if request.team_strategy == TeamStrategy.SPECIFIED:
            return request.agents or default_agents[:2]

        if request.team_strategy == TeamStrategy.FAST:
            return default_agents[:2]

        if request.team_strategy == TeamStrategy.DIVERSE:
            return default_agents

        if request.team_strategy == TeamStrategy.RANDOM:
            import random

            return random.sample(default_agents, min(3, len(default_agents)))

        # BEST_FOR_DOMAIN - use routing handler if available
        try:
            import aragora.server.handlers.routing as routing_module

            recommend_fn = getattr(routing_module, "recommend_agents", None)
            if recommend_fn is not None:
                recommended = await recommend_fn(request.question)
                if recommended:
                    return recommended[:4]
        except (ImportError, AttributeError) as e:
            logger.debug(f"Routing handler not available: {e}")
        except Exception as e:
            logger.warning(f"Agent routing failed: {e}")

        return default_agents[:3]

    async def _route_to_channel(
        self,
        channel: OutputChannel,
        result: OrchestrationResult,
        request: OrchestrationRequest,
    ) -> None:
        """Route vetted decisionmaking result to an output channel."""
        channel_type = channel.channel_type.lower()

        # Format message based on output format
        message = self._format_result_for_channel(result, request)

        if channel_type == "slack":
            await self._send_to_slack(channel, message)
        elif channel_type == "teams":
            await self._send_to_teams(channel, message)
        elif channel_type == "discord":
            await self._send_to_discord(channel, message)
        elif channel_type == "telegram":
            await self._send_to_telegram(channel, message)
        elif channel_type == "email":
            await self._send_to_email(channel, message, request)
        elif channel_type == "webhook":
            await self._send_to_webhook(channel, result)
        else:
            logger.warning(f"Unknown channel type: {channel_type}")

    def _format_result_for_channel(
        self, result: OrchestrationResult, request: OrchestrationRequest
    ) -> str:
        """Format result for channel delivery."""
        if request.output_format == OutputFormat.SUMMARY:
            return f"**Deliberation Complete**\n\n{result.final_answer or 'No conclusion reached.'}"

        consensus_status = "Consensus reached" if result.consensus_reached else "No consensus"
        confidence_str = f" ({result.confidence:.0%} confidence)" if result.confidence else ""

        return (
            f"**Deliberation Result**\n\n"
            f"**Question:** {request.question[:200]}...\n\n"
            f"**Status:** {consensus_status}{confidence_str}\n\n"
            f"**Answer:**\n{result.final_answer or 'No conclusion reached.'}\n\n"
            f"**Agents:** {', '.join(result.agents_participated)}\n"
            f"**Duration:** {result.duration_seconds:.1f}s\n"
            f"**Request ID:** `{result.request_id}`"
        )

    async def _send_to_slack(self, channel: OutputChannel, message: str) -> None:
        """Send result to Slack."""
        try:
            from aragora.connectors.chat.registry import get_connector

            connector = get_connector("slack")
            if connector:
                await connector.send_message(
                    channel.channel_id,
                    message,
                    thread_ts=channel.thread_id,
                )
        except Exception as e:
            logger.warning(f"Failed to send to Slack: {e}")

    async def _send_to_teams(self, channel: OutputChannel, message: str) -> None:
        """Send result to Microsoft Teams."""
        try:
            from aragora.connectors.chat.registry import get_connector

            connector = get_connector("teams")
            if connector:
                await connector.send_message(channel.channel_id, message)
        except Exception as e:
            logger.warning(f"Failed to send to Teams: {e}")

    async def _send_to_discord(self, channel: OutputChannel, message: str) -> None:
        """Send result to Discord."""
        try:
            from aragora.connectors.chat.registry import get_connector

            connector = get_connector("discord")
            if connector:
                await connector.send_message(channel.channel_id, message)
        except Exception as e:
            logger.warning(f"Failed to send to Discord: {e}")

    async def _send_to_telegram(self, channel: OutputChannel, message: str) -> None:
        """Send result to Telegram."""
        try:
            from aragora.connectors.chat.registry import get_connector

            connector = get_connector("telegram")
            if connector:
                await connector.send_message(channel.channel_id, message)
        except Exception as e:
            logger.warning(f"Failed to send to Telegram: {e}")

    async def _send_to_email(
        self, channel: OutputChannel, message: str, request: OrchestrationRequest
    ) -> None:
        """Send result via email."""
        try:
            import aragora.connectors.email as email_module

            send_fn = getattr(email_module, "send_email", None)
            if send_fn is not None:
                await send_fn(
                    to=channel.channel_id,
                    subject=f"Deliberation Result: {request.question[:50]}...",
                    body=message,
                )
        except Exception as e:
            logger.warning(f"Failed to send email: {e}")

    async def _send_to_webhook(self, channel: OutputChannel, result: OrchestrationResult) -> None:
        """Send result to webhook."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    channel.channel_id,
                    json=result.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status >= 400:
                        logger.warning(f"Webhook returned {response.status}")
        except Exception as e:
            logger.warning(f"Failed to send to webhook: {e}")


# =============================================================================
# Module Exports
# =============================================================================

handler = OrchestrationHandler({})

__all__ = [
    "OrchestrationHandler",
    "OrchestrationRequest",
    "OrchestrationResult",
    "DeliberationTemplate",
    "KnowledgeContextSource",
    "OutputChannel",
    "TeamStrategy",
    "OutputFormat",
    "TEMPLATES",
    "handler",
]
