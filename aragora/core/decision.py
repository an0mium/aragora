"""
Unified Decision Request Schema and Router.

Provides a single entry point for all decision-making requests across
Aragora, normalizing inputs from HTTP, WebSocket, chat, voice, and email
channels before routing to the appropriate decision engine.

Usage:
    from aragora.core.decision import DecisionRequest, DecisionRouter

    # Create a unified request
    request = DecisionRequest(
        content="Should we use microservices?",
        decision_type=DecisionType.DEBATE,
        source=InputSource.SLACK,
        response_channel=ResponseChannel(platform="slack", channel_id="C123"),
    )

    # Route to appropriate engine
    router = DecisionRouter()
    result = await router.route(request)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

# Lazy import for tracing to avoid circular imports
_tracing_imported = False
_trace_decision = None
_trace_decision_engine = None
_trace_response_delivery = None


def _import_tracing():
    """Lazy import tracing utilities."""
    global _tracing_imported, _trace_decision, _trace_decision_engine, _trace_response_delivery
    if _tracing_imported:
        return
    try:
        from aragora.observability.tracing import (
            trace_decision,
            trace_decision_engine,
            trace_response_delivery,
        )
        _trace_decision = trace_decision
        _trace_decision_engine = trace_decision_engine
        _trace_response_delivery = trace_response_delivery
    except ImportError:
        pass
    _tracing_imported = True


# Lazy import for caching
_cache_imported = False
_decision_cache = None


def _import_cache():
    """Lazy import cache utilities."""
    global _cache_imported, _decision_cache
    if _cache_imported:
        return
    try:
        from aragora.core.decision_cache import get_decision_cache
        _decision_cache = get_decision_cache()
    except ImportError:
        pass
    _cache_imported = True


# Lazy import for metrics
_metrics_imported = False
_record_decision_request = None
_record_decision_result = None
_record_decision_error = None
_record_decision_cache_hit = None
_record_decision_dedup_hit = None


def _import_metrics():
    """Lazy import metrics utilities."""
    global _metrics_imported, _record_decision_request, _record_decision_result
    global _record_decision_error, _record_decision_cache_hit, _record_decision_dedup_hit
    if _metrics_imported:
        return
    try:
        from aragora.observability.decision_metrics import (
            record_decision_request,
            record_decision_result,
            record_decision_error,
            record_decision_cache_hit,
            record_decision_dedup_hit,
        )
        _record_decision_request = record_decision_request
        _record_decision_result = record_decision_result
        _record_decision_error = record_decision_error
        _record_decision_cache_hit = record_decision_cache_hit
        _record_decision_dedup_hit = record_decision_dedup_hit
    except ImportError:
        pass
    _metrics_imported = True


logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Types of decision-making processes available."""

    DEBATE = "debate"  # Multi-agent debate with consensus
    WORKFLOW = "workflow"  # DAG-based workflow execution
    GAUNTLET = "gauntlet"  # Adversarial validation pipeline
    QUICK = "quick"  # Fast single-agent response
    AUTO = "auto"  # Auto-detect based on content


class InputSource(str, Enum):
    """Source channel for the decision request."""

    # Chat platforms
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    GOOGLE_CHAT = "google_chat"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"

    # Direct interfaces
    HTTP_API = "http_api"
    WEBSOCKET = "websocket"
    CLI = "cli"

    # Voice
    VOICE = "voice"
    VOICE_SLACK = "voice_slack"
    VOICE_TELEGRAM = "voice_telegram"
    VOICE_WHATSAPP = "voice_whatsapp"

    # Email
    EMAIL = "email"
    GMAIL = "gmail"

    # Cloud storage
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    SHAREPOINT = "sharepoint"
    DROPBOX = "dropbox"
    S3 = "s3"

    # Enterprise integrations
    JIRA = "jira"
    GITHUB = "github"
    SERVICENOW = "servicenow"
    CONFLUENCE = "confluence"
    NOTION = "notion"

    # Event streams
    KAFKA = "kafka"
    RABBITMQ = "rabbitmq"
    WEBHOOK = "webhook"

    # Internal
    WORKFLOW = "workflow"  # Triggered by another workflow
    SCHEDULED = "scheduled"  # Scheduled task
    INTERNAL = "internal"  # System-generated


class Priority(str, Enum):
    """Request priority levels."""

    CRITICAL = "critical"  # Immediate processing
    HIGH = "high"  # Fast-track queue
    NORMAL = "normal"  # Standard processing
    LOW = "low"  # Background processing
    BATCH = "batch"  # Batch with similar requests


class ResponseFormat(str, Enum):
    """Format for the response delivery."""

    FULL = "full"  # Complete response with reasoning
    SUMMARY = "summary"  # Condensed summary
    NOTIFICATION = "notification"  # Brief notification
    VOICE = "voice"  # Audio/TTS response
    VOICE_WITH_TEXT = "voice_with_text"  # Both audio and text


@dataclass
class ResponseChannel:
    """
    Where to send the decision result.

    Supports multiple output channels for the same request.
    """

    platform: str  # slack, discord, http, websocket, email, webhook, voice
    channel_id: Optional[str] = None  # Slack channel, Discord channel, etc.
    user_id: Optional[str] = None  # Direct message to user
    thread_id: Optional[str] = None  # Reply in thread
    webhook_url: Optional[str] = None  # Webhook callback URL
    email_address: Optional[str] = None  # Email recipient
    response_format: str = "full"  # full, summary, notification, voice
    include_reasoning: bool = True  # Include chain-of-thought

    # Voice/TTS settings
    voice_enabled: bool = False  # Whether to include voice response
    voice_id: str = "narrator"  # TTS voice/speaker identifier
    voice_only: bool = False  # If True, only send audio (no text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "webhook_url": self.webhook_url,
            "email_address": self.email_address,
            "response_format": self.response_format,
            "include_reasoning": self.include_reasoning,
            "voice_enabled": self.voice_enabled,
            "voice_id": self.voice_id,
            "voice_only": self.voice_only,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseChannel":
        """Create from dictionary."""
        return cls(
            platform=data.get("platform", "http"),
            channel_id=data.get("channel_id"),
            user_id=data.get("user_id"),
            thread_id=data.get("thread_id"),
            webhook_url=data.get("webhook_url"),
            email_address=data.get("email_address"),
            response_format=data.get("response_format", "full"),
            include_reasoning=data.get("include_reasoning", True),
            voice_enabled=data.get("voice_enabled", False),
            voice_id=data.get("voice_id", "narrator"),
            voice_only=data.get("voice_only", False),
        )


@dataclass
class RequestContext:
    """
    Context and metadata for the decision request.

    Provides audit trail and correlation capabilities.
    """

    # Correlation
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_request_id: Optional[str] = None  # If spawned from another request
    session_id: Optional[str] = None  # User session

    # User info
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    user_roles: List[str] = field(default_factory=list)

    # Tenant/workspace
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None  # Hard deadline for response

    # Additional context
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "parent_request_id": self.parent_request_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "user_roles": self.user_roles,
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestContext":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        deadline = data.get("deadline")
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)

        return cls(
            correlation_id=data.get("correlation_id", str(uuid.uuid4())),
            parent_request_id=data.get("parent_request_id"),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            user_name=data.get("user_name"),
            user_email=data.get("user_email"),
            user_roles=data.get("user_roles", []),
            tenant_id=data.get("tenant_id"),
            workspace_id=data.get("workspace_id"),
            created_at=created_at or datetime.utcnow(),
            deadline=deadline,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DecisionConfig:
    """
    Configuration for the decision-making process.

    Unified configuration that maps to specific engine configs.
    """

    # Common settings
    timeout_seconds: int = 300
    max_agents: int = 3
    agents: List[str] = field(default_factory=lambda: ["anthropic-api", "openai-api"])

    # Debate-specific (mapped to DebateProtocol)
    rounds: int = 3
    consensus: str = "majority"  # majority, unanimous, judge, weighted
    enable_calibration: bool = True
    early_stopping: bool = True

    # Workflow-specific (mapped to WorkflowConfig)
    workflow_id: Optional[str] = None
    workflow_inputs: Dict[str, Any] = field(default_factory=dict)
    stop_on_failure: bool = True
    enable_checkpointing: bool = True

    # Gauntlet-specific (mapped to GauntletConfig)
    enable_adversarial: bool = False
    enable_formal_verification: bool = False
    robustness_threshold: float = 0.6
    attack_categories: List[str] = field(default_factory=list)

    # Knowledge integration
    use_knowledge_mound: bool = True
    include_evidence: bool = True
    evidence_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_agents": self.max_agents,
            "agents": self.agents,
            "rounds": self.rounds,
            "consensus": self.consensus,
            "enable_calibration": self.enable_calibration,
            "early_stopping": self.early_stopping,
            "workflow_id": self.workflow_id,
            "workflow_inputs": self.workflow_inputs,
            "stop_on_failure": self.stop_on_failure,
            "enable_checkpointing": self.enable_checkpointing,
            "enable_adversarial": self.enable_adversarial,
            "enable_formal_verification": self.enable_formal_verification,
            "robustness_threshold": self.robustness_threshold,
            "attack_categories": self.attack_categories,
            "use_knowledge_mound": self.use_knowledge_mound,
            "include_evidence": self.include_evidence,
            "evidence_sources": self.evidence_sources,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionConfig":
        """Create from dictionary."""
        return cls(
            timeout_seconds=data.get("timeout_seconds", 300),
            max_agents=data.get("max_agents", 3),
            agents=data.get("agents", ["anthropic-api", "openai-api"]),
            rounds=data.get("rounds", 3),
            consensus=data.get("consensus", "majority"),
            enable_calibration=data.get("enable_calibration", True),
            early_stopping=data.get("early_stopping", True),
            workflow_id=data.get("workflow_id"),
            workflow_inputs=data.get("workflow_inputs", {}),
            stop_on_failure=data.get("stop_on_failure", True),
            enable_checkpointing=data.get("enable_checkpointing", True),
            enable_adversarial=data.get("enable_adversarial", False),
            enable_formal_verification=data.get("enable_formal_verification", False),
            robustness_threshold=data.get("robustness_threshold", 0.6),
            attack_categories=data.get("attack_categories", []),
            use_knowledge_mound=data.get("use_knowledge_mound", True),
            include_evidence=data.get("include_evidence", True),
            evidence_sources=data.get("evidence_sources", []),
        )


@dataclass
class DecisionRequest:
    """
    Unified decision request that normalizes input from all channels.

    This is the canonical request format for Aragora's decision-making
    capabilities, accepting input from HTTP, WebSocket, chat platforms,
    voice, email, and enterprise integrations.
    """

    # Core content
    content: str  # The question, topic, or input text
    decision_type: DecisionType = DecisionType.AUTO

    # Source and routing
    source: InputSource = InputSource.HTTP_API
    response_channels: List[ResponseChannel] = field(default_factory=list)

    # Request context
    context: RequestContext = field(default_factory=RequestContext)

    # Configuration
    config: DecisionConfig = field(default_factory=DecisionConfig)

    # Priority
    priority: Priority = Priority.NORMAL

    # Additional input data
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Request ID
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Validate and normalize the request."""
        # Ensure content is not empty
        if not self.content or not self.content.strip():
            raise ValueError("Decision request content cannot be empty")

        # Ensure at least one response channel
        if not self.response_channels:
            self.response_channels = [
                ResponseChannel(platform=self.source.value)
            ]

        # Auto-detect decision type if needed
        if self.decision_type == DecisionType.AUTO:
            self.decision_type = self._detect_decision_type()

    def _detect_decision_type(self) -> DecisionType:
        """Auto-detect the appropriate decision type based on content."""
        content_lower = self.content.lower()

        # Check for workflow triggers
        if self.config.workflow_id:
            return DecisionType.WORKFLOW

        # Check for gauntlet triggers
        gauntlet_keywords = ["validate", "stress test", "probe", "attack", "security"]
        if any(kw in content_lower for kw in gauntlet_keywords):
            return DecisionType.GAUNTLET

        # Check for quick response triggers
        quick_keywords = ["quick", "fast", "simple", "brief"]
        if any(kw in content_lower for kw in quick_keywords):
            return DecisionType.QUICK

        # Default to debate for complex questions
        return DecisionType.DEBATE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "content": self.content,
            "decision_type": self.decision_type.value,
            "source": self.source.value,
            "response_channels": [rc.to_dict() for rc in self.response_channels],
            "context": self.context.to_dict(),
            "config": self.config.to_dict(),
            "priority": self.priority.value,
            "attachments": self.attachments,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRequest":
        """Create from dictionary."""
        return cls(
            request_id=data.get("request_id", str(uuid.uuid4())),
            content=data.get("content", ""),
            decision_type=DecisionType(data.get("decision_type", "auto")),
            source=InputSource(data.get("source", "http_api")),
            response_channels=[
                ResponseChannel.from_dict(rc)
                for rc in data.get("response_channels", [])
            ],
            context=RequestContext.from_dict(data.get("context", {})),
            config=DecisionConfig.from_dict(data.get("config", {})),
            priority=Priority(data.get("priority", "normal")),
            attachments=data.get("attachments", []),
            evidence=data.get("evidence", []),
        )

    @classmethod
    def from_chat_message(
        cls,
        message: str,
        platform: str,
        channel_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs,
    ) -> "DecisionRequest":
        """Create from a chat platform message."""
        source = InputSource(platform.lower())

        response_channel = ResponseChannel(
            platform=platform,
            channel_id=channel_id,
            user_id=user_id,
            thread_id=thread_id,
        )

        context = RequestContext(
            user_id=user_id,
            metadata=kwargs,
        )

        return cls(
            content=message,
            source=source,
            response_channels=[response_channel],
            context=context,
        )

    @classmethod
    def from_http(
        cls,
        body: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> "DecisionRequest":
        """Create from HTTP API request body."""
        # Extract correlation ID from headers if present
        correlation_id = None
        if headers:
            correlation_id = headers.get("X-Correlation-ID") or headers.get("X-Request-ID")

        # Handle both new unified format and legacy debate format
        if "content" in body:
            request = cls.from_dict(body)
        else:
            # Legacy format: {question, agents, rounds, consensus, ...}
            request = cls(
                content=body.get("question") or body.get("task") or body.get("input_text", ""),
                decision_type=DecisionType(body.get("decision_type", "debate")),
                source=InputSource.HTTP_API,
                config=DecisionConfig(
                    agents=body.get("agents", ["anthropic-api", "openai-api"]),
                    rounds=body.get("rounds", 3),
                    consensus=body.get("consensus", "majority"),
                    timeout_seconds=body.get("timeout", 300),
                ),
            )

        if correlation_id:
            request.context.correlation_id = correlation_id

        return request

    @classmethod
    def from_voice(
        cls,
        transcription: str,
        platform: str,
        channel_id: str,
        audio_duration: Optional[float] = None,
        voice_response: bool = True,
        voice_id: str = "narrator",
        **kwargs,
    ) -> "DecisionRequest":
        """Create from a voice message transcription."""
        source_map = {
            "slack": InputSource.VOICE_SLACK,
            "telegram": InputSource.VOICE_TELEGRAM,
            "whatsapp": InputSource.VOICE_WHATSAPP,
        }
        source = source_map.get(platform.lower(), InputSource.VOICE)

        context = RequestContext(
            metadata={
                "audio_duration": audio_duration,
                "transcription_source": "whisper",
                **kwargs,
            }
        )

        # Voice input defaults to voice+text response
        response_channel = ResponseChannel(
            platform=platform,
            channel_id=channel_id,
            response_format="voice_with_text" if voice_response else "full",
            voice_enabled=voice_response,
            voice_id=voice_id,
        )

        return cls(
            content=transcription,
            source=source,
            response_channels=[response_channel],
            context=context,
        )

    @classmethod
    def from_document(
        cls,
        content: str,
        source_platform: str,
        document_id: str,
        document_title: Optional[str] = None,
        document_url: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> "DecisionRequest":
        """Create from a document source (Google Drive, SharePoint, etc.)."""
        source_map = {
            "google_drive": InputSource.GOOGLE_DRIVE,
            "gdrive": InputSource.GOOGLE_DRIVE,
            "onedrive": InputSource.ONEDRIVE,
            "sharepoint": InputSource.SHAREPOINT,
            "dropbox": InputSource.DROPBOX,
            "s3": InputSource.S3,
            "confluence": InputSource.CONFLUENCE,
            "notion": InputSource.NOTION,
        }
        source = source_map.get(source_platform.lower(), InputSource.INTERNAL)

        context = RequestContext(
            user_id=user_id,
            metadata={
                "document_id": document_id,
                "document_title": document_title,
                "document_url": document_url,
                "source_platform": source_platform,
                **kwargs,
            }
        )

        # For document sources, typically respond via webhook or the originating system
        response_channel = ResponseChannel(
            platform=source_platform,
            webhook_url=kwargs.get("webhook_url"),
            response_format="full",
        )

        return cls(
            content=content,
            source=source,
            response_channels=[response_channel],
            context=context,
        )


@dataclass
class DecisionResult:
    """
    Unified decision result returned by all decision engines.
    """

    request_id: str
    decision_type: DecisionType

    # Core result
    answer: str
    confidence: float  # 0-1
    consensus_reached: bool

    # Detailed results
    reasoning: Optional[str] = None
    evidence_used: List[Dict[str, Any]] = field(default_factory=list)
    agent_contributions: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    duration_seconds: float = 0.0
    completed_at: datetime = field(default_factory=datetime.utcnow)

    # Engine-specific results
    debate_result: Optional[Any] = None  # DebateResult
    workflow_result: Optional[Any] = None  # WorkflowResult
    gauntlet_result: Optional[Any] = None  # GauntletResult

    # Status
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "decision_type": self.decision_type.value,
            "answer": self.answer,
            "confidence": self.confidence,
            "consensus_reached": self.consensus_reached,
            "reasoning": self.reasoning,
            "evidence_used": self.evidence_used,
            "agent_contributions": self.agent_contributions,
            "duration_seconds": self.duration_seconds,
            "completed_at": self.completed_at.isoformat(),
            "success": self.success,
            "error": self.error,
        }


class DecisionRouter:
    """
    Routes decision requests to the appropriate engine.

    Provides a unified entry point for all decision-making requests,
    handling routing, validation, and response delivery.
    """

    def __init__(
        self,
        debate_engine: Optional[Any] = None,
        workflow_engine: Optional[Any] = None,
        gauntlet_engine: Optional[Any] = None,
        enable_voice_responses: bool = True,
        enable_caching: bool = True,
        enable_deduplication: bool = True,
        cache_ttl_seconds: float = 3600.0,
    ):
        """
        Initialize the router.

        Args:
            debate_engine: Arena instance or factory
            workflow_engine: WorkflowEngine instance or factory
            gauntlet_engine: GauntletOrchestrator instance or factory
            enable_voice_responses: Whether to enable TTS for voice responses
            enable_caching: Whether to cache decision results
            enable_deduplication: Whether to deduplicate concurrent identical requests
            cache_ttl_seconds: How long to cache results (default 1 hour)
        """
        self._debate_engine = debate_engine
        self._workflow_engine = workflow_engine
        self._gauntlet_engine = gauntlet_engine
        self._response_handlers: Dict[str, Callable] = {}
        self._enable_voice_responses = enable_voice_responses
        self._tts_bridge: Optional[Any] = None

        # Cache configuration
        self._enable_caching = enable_caching
        self._enable_deduplication = enable_deduplication
        self._cache_ttl_seconds = cache_ttl_seconds

    def register_response_handler(
        self,
        platform: str,
        handler: Callable[[DecisionResult, ResponseChannel], None],
    ) -> None:
        """Register a handler for delivering responses to a platform."""
        self._response_handlers[platform.lower()] = handler

    async def route(self, request: DecisionRequest) -> DecisionResult:
        """
        Route a decision request to the appropriate engine.

        Args:
            request: Unified decision request

        Returns:
            DecisionResult with the outcome
        """
        # Initialize tracing, caching, and metrics if available
        _import_tracing()
        _import_cache()
        _import_metrics()

        logger.info(
            f"Routing decision request {request.request_id} "
            f"(type={request.decision_type.value}, source={request.source.value})"
        )

        start_time = datetime.utcnow()

        # Record incoming request metric
        if _record_decision_request:
            _record_decision_request(
                decision_type=request.decision_type.value,
                source=request.source.value,
                priority=request.priority.value if request.priority else "normal",
            )

        # Check cache first
        cache_hit = False
        dedup_hit = False
        if self._enable_caching and _decision_cache:
            cached_result = await _decision_cache.get(request)
            if cached_result:
                logger.info(f"Cache hit for request {request.request_id}")
                cache_hit = True
                # Record cache hit metric
                if _record_decision_cache_hit:
                    _record_decision_cache_hit(request.decision_type.value)
                # Record result metric for cached response
                if _record_decision_result:
                    _record_decision_result(
                        decision_type=request.decision_type.value,
                        source=request.source.value,
                        success=cached_result.success,
                        confidence=cached_result.confidence,
                        duration_seconds=0.001,  # Near-instant for cache hit
                        consensus_reached=cached_result.consensus_reached,
                        cache_hit=True,
                    )
                # Still deliver responses for cached results
                await self._deliver_responses(request, cached_result)
                return cached_result

        # Check for in-flight deduplication
        if self._enable_deduplication and _decision_cache:
            if await _decision_cache.is_in_flight(request):
                logger.info(f"Waiting for in-flight request {request.request_id}")
                dedup_result = await _decision_cache.wait_for_result(request)
                if dedup_result:
                    dedup_hit = True
                    # Record dedup hit metric
                    if _record_decision_dedup_hit:
                        _record_decision_dedup_hit(request.decision_type.value)
                    # Record result metric for dedup response
                    if _record_decision_result:
                        _record_decision_result(
                            decision_type=request.decision_type.value,
                            source=request.source.value,
                            success=dedup_result.success,
                            confidence=dedup_result.confidence,
                            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                            consensus_reached=dedup_result.consensus_reached,
                            dedup_hit=True,
                        )
                    await self._deliver_responses(request, dedup_result)
                    return dedup_result

        # Mark as in-flight for deduplication
        if self._enable_deduplication and _decision_cache:
            await _decision_cache.mark_in_flight(request)

        # Create tracing span if available
        span = None
        span_ctx = None
        if _trace_decision_engine:
            try:
                from aragora.observability.tracing import trace_decision_routing
                span_ctx = trace_decision_routing(
                    request_id=request.request_id,
                    decision_type=request.decision_type.value,
                    source=request.source.value,
                    priority=request.priority.value if request.priority else "normal",
                )
                span = span_ctx.__enter__()
                span.set_attribute("decision.content_length", len(request.content))
                span.set_attribute("decision.cache_hit", cache_hit)
                if request.config and request.config.agents:
                    span.set_attribute("decision.agent_count", len(request.config.agents))
            except Exception as e:
                logger.debug(f"Tracing not available: {e}")

        try:
            if request.decision_type == DecisionType.DEBATE:
                result = await self._route_to_debate(request)
            elif request.decision_type == DecisionType.WORKFLOW:
                result = await self._route_to_workflow(request)
            elif request.decision_type == DecisionType.GAUNTLET:
                result = await self._route_to_gauntlet(request)
            elif request.decision_type == DecisionType.QUICK:
                result = await self._route_to_quick(request)
            else:
                raise ValueError(f"Unknown decision type: {request.decision_type}")

            # Calculate duration
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

            # Record result in span
            if span:
                span.set_attribute("decision.duration_seconds", result.duration_seconds)
                span.set_attribute("decision.confidence", result.confidence)
                span.set_attribute("decision.consensus_reached", result.consensus_reached)
                span.set_attribute("decision.success", result.success)

            # Record successful result metric
            if _record_decision_result:
                agent_count = len(request.config.agents) if request.config.agents else 0
                _record_decision_result(
                    decision_type=request.decision_type.value,
                    source=request.source.value,
                    success=result.success,
                    confidence=result.confidence,
                    duration_seconds=result.duration_seconds,
                    consensus_reached=result.consensus_reached,
                    cache_hit=False,
                    dedup_hit=False,
                    agent_count=agent_count,
                )

            # Cache the result
            if self._enable_caching and _decision_cache and result.success:
                await _decision_cache.set(request, result, ttl_seconds=self._cache_ttl_seconds)

            # Complete in-flight for deduplication
            if self._enable_deduplication and _decision_cache:
                await _decision_cache.complete_in_flight(request, result=result)

            # Deliver responses
            await self._deliver_responses(request, result)

            return result

        except Exception as e:
            logger.error(f"Decision routing failed: {e}", exc_info=True)
            if span:
                span.set_attribute("decision.error", str(e)[:200])
                try:
                    span.record_exception(e)
                except Exception:
                    pass

            error_duration = (datetime.utcnow() - start_time).total_seconds()

            # Record error metrics
            error_type = type(e).__name__
            if _record_decision_error:
                _record_decision_error(
                    decision_type=request.decision_type.value,
                    error_type=error_type,
                )

            if _record_decision_result:
                _record_decision_result(
                    decision_type=request.decision_type.value,
                    source=request.source.value,
                    success=False,
                    confidence=0.0,
                    duration_seconds=error_duration,
                    consensus_reached=False,
                    error_type=error_type,
                )

            error_result = DecisionResult(
                request_id=request.request_id,
                decision_type=request.decision_type,
                answer="",
                confidence=0.0,
                consensus_reached=False,
                success=False,
                error=str(e),
                duration_seconds=error_duration,
            )

            # Complete in-flight with error
            if self._enable_deduplication and _decision_cache:
                await _decision_cache.complete_in_flight(request, error=e)

            return error_result
        finally:
            # Close span context
            if span_ctx:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass

            # Clean up in-flight status after a delay
            if self._enable_deduplication and _decision_cache:
                try:
                    await _decision_cache.clear_in_flight(request)
                except Exception:
                    pass

    async def _route_to_debate(self, request: DecisionRequest) -> DecisionResult:
        """Route to debate engine."""
        # Create engine span if tracing available
        span = None
        span_ctx = None
        if _trace_decision_engine:
            try:
                span_ctx = _trace_decision_engine("debate", request.request_id)
                span = span_ctx.__enter__()
            except Exception:
                pass

        try:
            if self._debate_engine is None:
                # Lazy load
                from aragora.debate import Arena
                self._debate_engine = Arena

            # Convert to debate format
            from aragora.core_types import Environment
            from aragora.debate.protocol import DebateProtocol

            env = Environment(task=request.content)
            protocol = DebateProtocol(
                rounds=request.config.rounds,
                consensus=request.config.consensus,
                enable_calibration=request.config.enable_calibration,
                early_stopping=request.config.early_stopping,
                timeout_seconds=request.config.timeout_seconds,
            )

            if span:
                span.set_attribute("debate.rounds", request.config.rounds)
                span.set_attribute("debate.agents", ",".join(request.config.agents or []))

            # Create arena and run
            arena = self._debate_engine(
                environment=env,
                protocol=protocol,
                agent_names=request.config.agents,
            )

            debate_result = await arena.run()

            if span:
                span.set_attribute("debate.consensus_reached", debate_result.consensus_reached)
                if hasattr(debate_result, "rounds_used"):
                    span.set_attribute("debate.rounds_used", debate_result.rounds_used)

            return DecisionResult(
                request_id=request.request_id,
                decision_type=DecisionType.DEBATE,
                answer=debate_result.final_answer or "",
                confidence=debate_result.confidence if hasattr(debate_result, "confidence") else 0.8,
                consensus_reached=debate_result.consensus_reached,
                reasoning=debate_result.summary if hasattr(debate_result, "summary") else None,
                debate_result=debate_result,
            )
        finally:
            if span_ctx:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    async def _route_to_workflow(self, request: DecisionRequest) -> DecisionResult:
        """Route to workflow engine."""
        span = None
        span_ctx = None
        if _trace_decision_engine:
            try:
                span_ctx = _trace_decision_engine("workflow", request.request_id)
                span = span_ctx.__enter__()
            except Exception:
                pass

        try:
            if self._workflow_engine is None:
                from aragora.workflow.engine import get_workflow_engine
                self._workflow_engine = get_workflow_engine()

            # Get or create workflow definition
            workflow_id = request.config.workflow_id
            if not workflow_id:
                raise ValueError("Workflow ID required for workflow decision type")

            if span:
                span.set_attribute("workflow.id", workflow_id)

            # Load workflow definition
            definition = await self._workflow_engine.get_definition(workflow_id)
            if not definition:
                raise ValueError(f"Workflow not found: {workflow_id}")

            # Execute
            workflow_result = await self._workflow_engine.execute(
                definition=definition,
                inputs={
                    "content": request.content,
                    **request.config.workflow_inputs,
                },
            )

            if span:
                span.set_attribute("workflow.success", workflow_result.success)

            # Extract answer from workflow outputs
            answer = workflow_result.outputs.get("answer") or workflow_result.outputs.get("result") or ""

            return DecisionResult(
                request_id=request.request_id,
                decision_type=DecisionType.WORKFLOW,
                answer=str(answer),
                confidence=0.9 if workflow_result.success else 0.0,
                consensus_reached=workflow_result.success,
                workflow_result=workflow_result,
            )
        finally:
            if span_ctx:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    async def _route_to_gauntlet(self, request: DecisionRequest) -> DecisionResult:
        """Route to gauntlet engine."""
        span = None
        span_ctx = None
        if _trace_decision_engine:
            try:
                span_ctx = _trace_decision_engine("gauntlet", request.request_id)
                span = span_ctx.__enter__()
            except Exception:
                pass

        try:
            if self._gauntlet_engine is None:
                from aragora.gauntlet.orchestrator import GauntletOrchestrator
                self._gauntlet_engine = GauntletOrchestrator()

            from aragora.gauntlet.config import GauntletConfig

            config = GauntletConfig(
                agents=request.config.agents,
                enable_adversarial_probing=request.config.enable_adversarial,
                enable_formal_verification=request.config.enable_formal_verification,
                robustness_threshold=request.config.robustness_threshold,
                timeout_seconds=request.config.timeout_seconds,
            )

            if span:
                span.set_attribute("gauntlet.adversarial", request.config.enable_adversarial)
                span.set_attribute("gauntlet.formal_verification", request.config.enable_formal_verification)

            gauntlet_result = await self._gauntlet_engine.run(
                input_text=request.content,
                config=config,
            )

            if span:
                span.set_attribute("gauntlet.passed", gauntlet_result.passed)
                span.set_attribute("gauntlet.confidence", gauntlet_result.confidence)

            return DecisionResult(
                request_id=request.request_id,
                decision_type=DecisionType.GAUNTLET,
                answer=gauntlet_result.verdict_summary or "",
                confidence=gauntlet_result.confidence,
                consensus_reached=gauntlet_result.passed,
                gauntlet_result=gauntlet_result,
            )
        finally:
            if span_ctx:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    async def _route_to_quick(self, request: DecisionRequest) -> DecisionResult:
        """Route to quick single-agent response."""
        span = None
        span_ctx = None
        if _trace_decision_engine:
            try:
                span_ctx = _trace_decision_engine("quick", request.request_id)
                span = span_ctx.__enter__()
            except Exception:
                pass

        # Use first configured agent for quick response
        agent_name = request.config.agents[0] if request.config.agents else "anthropic-api"

        if span:
            span.set_attribute("quick.agent", agent_name)

        try:
            from aragora.agents import get_agent
            agent = get_agent(agent_name)

            response = await agent.generate(request.content)

            if span:
                span.set_attribute("quick.response_length", len(response))

            return DecisionResult(
                request_id=request.request_id,
                decision_type=DecisionType.QUICK,
                answer=response,
                confidence=0.7,  # Single agent = moderate confidence
                consensus_reached=True,
                agent_contributions=[{"agent": agent_name, "response": response}],
            )
        except Exception as e:
            logger.error(f"Quick response failed: {e}")
            if span:
                span.set_attribute("quick.error", str(e)[:200])
            return DecisionResult(
                request_id=request.request_id,
                decision_type=DecisionType.QUICK,
                answer="",
                confidence=0.0,
                consensus_reached=False,
                success=False,
                error=str(e),
            )
        finally:
            if span_ctx:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass

    def _get_tts_bridge(self) -> Optional[Any]:
        """Lazy-load TTS bridge for voice responses."""
        if self._tts_bridge is not None:
            return self._tts_bridge

        if not self._enable_voice_responses:
            return None

        try:
            from aragora.connectors.chat.tts_bridge import get_tts_bridge

            self._tts_bridge = get_tts_bridge()
            logger.info("TTS bridge initialized for voice responses")
            return self._tts_bridge
        except ImportError as e:
            logger.warning(f"TTS bridge not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize TTS bridge: {e}")
            return None

    async def synthesize_voice_response(
        self,
        text: str,
        voice_id: str = "narrator",
        context: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Synthesize text to audio for voice response.

        Args:
            text: Text to synthesize
            voice_id: Voice/speaker identifier
            context: Optional context hint for voice selection

        Returns:
            Audio bytes if successful, None otherwise
        """
        tts = self._get_tts_bridge()
        if tts is None:
            logger.warning("TTS not available for voice response")
            return None

        try:
            audio_path = await tts.synthesize(
                text=text,
                voice=voice_id,
                context=context,
            )
            if audio_path and audio_path.exists():
                audio_bytes = audio_path.read_bytes()
                # Clean up temp file
                try:
                    audio_path.unlink()
                except Exception:
                    pass
                return audio_bytes
        except Exception as e:
            logger.error(f"Voice synthesis failed: {e}")

        return None

    async def _deliver_responses(
        self,
        request: DecisionRequest,
        result: DecisionResult,
    ) -> None:
        """Deliver result to all response channels."""
        for channel in request.response_channels:
            try:
                # Handle voice responses
                if channel.voice_enabled or channel.response_format in ("voice", "voice_with_text"):
                    await self._deliver_voice_response(result, channel)

                    # If voice_only, skip text delivery
                    if channel.voice_only:
                        continue

                # Deliver text response via platform handler
                handler = self._response_handlers.get(channel.platform)
                if handler:
                    await handler(result, channel)

            except Exception as e:
                logger.error(f"Failed to deliver to {channel.platform}: {e}")

    async def _deliver_voice_response(
        self,
        result: DecisionResult,
        channel: ResponseChannel,
    ) -> Optional[bytes]:
        """Deliver voice response for a channel."""
        # Format text for voice (more concise than full text)
        if channel.response_format == "notification":
            voice_text = f"Decision complete. {result.answer[:200]}"
        elif channel.response_format == "summary":
            voice_text = result.answer[:500]
        else:
            voice_text = result.answer[:1000]

        # Add confidence if applicable
        if result.consensus_reached:
            voice_text = f"Consensus reached with {result.confidence:.0%} confidence. {voice_text}"

        # Synthesize audio
        audio_bytes = await self.synthesize_voice_response(
            text=voice_text,
            voice_id=channel.voice_id,
            context=result.decision_type.value,
        )

        if audio_bytes:
            # If there's a voice-specific handler, use it
            voice_handler = self._response_handlers.get(f"{channel.platform}_voice")
            if voice_handler:
                try:
                    await voice_handler(result, channel, audio_bytes)
                except Exception as e:
                    logger.error(f"Voice delivery failed for {channel.platform}: {e}")

        return audio_bytes


# Singleton router instance
_router: Optional[DecisionRouter] = None


def get_decision_router() -> DecisionRouter:
    """Get or create the global decision router."""
    global _router
    if _router is None:
        _router = DecisionRouter()
    return _router


def reset_decision_router() -> None:
    """Reset the global decision router (for testing)."""
    global _router
    _router = None
