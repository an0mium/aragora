"""
Gmail Q&A query handler.

Enables natural language queries over email content using
RLM compression and KnowledgeMound search.

Endpoints:
- POST /api/gmail/query - Text Q&A over inbox
- POST /api/gmail/query/voice - Voice input Q&A
- GET /api/gmail/query/stream - Streaming Q&A response
- GET /api/gmail/inbox/priority - Get prioritized inbox
- POST /api/gmail/inbox/feedback - Record interaction feedback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aragora.server.validation.query_params import safe_query_int

from ..base import (
    HandlerResult,
    error_response,
    json_response,
)
from ..secure import ForbiddenError, SecureHandler, UnauthorizedError
from .gmail_ingest import get_user_state

logger = logging.getLogger(__name__)

# Gmail permissions
GMAIL_READ_PERMISSION = "gmail:read"
GMAIL_WRITE_PERMISSION = "gmail:write"


@dataclass
class QueryResponse:
    """Response from email Q&A."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    query: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "query": self.query,
        }


class GmailQueryHandler(SecureHandler):
    """Handler for Gmail Q&A and priority inbox endpoints.

    Requires authentication and gmail:read/gmail:write permissions.
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/gmail/query",
        "/api/v1/gmail/query/voice",
        "/api/v1/gmail/query/stream",
        "/api/v1/gmail/inbox/priority",
        "/api/v1/gmail/inbox/feedback",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the path."""
        return path.startswith("/api/v1/gmail/query") or path.startswith("/api/v1/gmail/inbox/")

    async def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Route GET requests."""
        # RBAC: Require authentication and gmail:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, GMAIL_READ_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        user_id = query_params.get("user_id", "default")

        if path == "/api/v1/gmail/inbox/priority":
            return await self._get_priority_inbox(user_id, query_params)

        if path == "/api/v1/gmail/query/stream":
            # Streaming would need WebSocket - return regular response
            return await self._handle_query(user_id, {"question": query_params.get("q", "")})

        return error_response("Not found", 404)

    async def handle_post(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Route POST requests."""
        # RBAC: Require authentication and gmail:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, GMAIL_READ_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        # Read JSON body from request
        body = self.read_json_body(handler)
        if body is None:
            body = {}

        user_id = body.get("user_id", "default")

        if path == "/api/v1/gmail/query":
            return await self._handle_query(user_id, body)

        if path == "/api/v1/gmail/query/voice":
            return await self._handle_voice_query(user_id, body, handler)

        if path == "/api/v1/gmail/inbox/feedback":
            return await self._record_feedback(user_id, body)

        return error_response("Not found", 404)

    async def _handle_query(self, user_id: str, body: dict[str, Any]) -> HandlerResult:
        """Handle natural language query over email content."""
        state = get_user_state(user_id)

        if not state or not getattr(state, "refresh_token", None):
            return error_response("Not connected - please authenticate first", 401)

        question = body.get("question", body.get("q", ""))
        if not question:
            return error_response("Question is required", 400)

        limit = body.get("limit", 10)

        try:
            response = await self._run_query(user_id, state, question, limit)
            return json_response(response.to_dict())

        except Exception as e:
            logger.error(f"[GmailQuery] Query failed: {e}")
            return error_response(f"Query failed: {e}", 500)

    async def _run_query(
        self,
        user_id: str,
        state: Any,
        question: str,
        limit: int,
    ) -> QueryResponse:
        """Execute Q&A query against email content."""
        from aragora.connectors.enterprise.communication.gmail import GmailConnector

        # Create connector with user's tokens
        connector = GmailConnector(max_results=limit * 2)
        setattr(connector, "_access_token", getattr(state, "access_token", None))
        setattr(connector, "_refresh_token", getattr(state, "refresh_token", None))
        setattr(connector, "_token_expiry", getattr(state, "token_expiry", None))

        # Search for relevant emails
        results = await connector.search(query=question, limit=limit)

        if not results:
            return QueryResponse(
                answer="I couldn't find any emails matching your query.",
                sources=[],
                confidence=0.0,
                query=question,
            )

        # Fetch full content for top results using batch method to avoid N+1 queries
        emails_content = []
        sources = []

        # Extract message IDs and fetch in batch
        message_ids = [r.id.replace("gmail-", "") for r in results[:5]]
        messages = await connector.get_messages(message_ids)

        for msg in messages:
            emails_content.append(
                f"From: {msg.from_address}\n"
                f"Subject: {msg.subject}\n"
                f"Date: {msg.date.isoformat() if msg.date else 'Unknown'}\n"
                f"Content: {msg.body_text[:1000] if msg.body_text else msg.snippet}\n"
            )

            sources.append(
                {
                    "id": msg.id,
                    "subject": msg.subject,
                    "from": msg.from_address,
                    "date": msg.date.isoformat() if msg.date else None,
                    "url": f"https://mail.google.com/mail/u/0/#inbox/{msg.id}",
                }
            )

        if not emails_content:
            return QueryResponse(
                answer="Found some emails but couldn't retrieve their content.",
                sources=sources,
                confidence=0.3,
                query=question,
            )

        # Try to use RLM for compressed analysis
        answer = await self._generate_answer(question, emails_content)

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=0.8 if answer else 0.5,
            query=question,
        )

    async def _generate_answer(
        self,
        question: str,
        emails_content: list[str],
    ) -> str:
        """Generate answer using RLM or fallback."""
        context = "\n---\n".join(emails_content)

        # Try RLM first
        try:
            from aragora.rlm.streaming import StreamingRLMQuery

            rlm = StreamingRLMQuery()

            # Compress context if too long
            if len(context) > 4000 and hasattr(rlm, "compress"):
                compressed = await rlm.compress(
                    content=context,
                    target_tokens=3000,
                )
                context = compressed or context[:4000]

            # Generate answer
            if not hasattr(rlm, "query"):
                raise AttributeError("RLM does not have query method")
            answer = await rlm.query(
                context=context,
                question=question,
                system_prompt=(
                    "You are an email assistant. Answer the user's question based only "
                    "on the provided email content. Be concise and specific. "
                    "If the emails don't contain the answer, say so."
                ),
                max_tokens=500,
            )

            if answer:
                return answer

        except ImportError:
            logger.debug("[GmailQuery] RLM not available, using fallback")
        except Exception as e:
            logger.warning(f"[GmailQuery] RLM failed: {e}")

        # Fallback: Try using an LLM directly
        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

            agent = AnthropicAPIAgent()
            if not hasattr(agent, "respond"):
                raise AttributeError("Agent does not have respond method")
            response = await agent.respond(
                f"Based on these emails:\n\n{context[:3000]}\n\n"
                f"Answer this question: {question}\n\n"
                "Be concise and specific. If the emails don't contain the answer, say so."
            )

            return response.content if response else self._simple_answer(question, emails_content)

        except Exception as e:
            logger.debug(f"[GmailQuery] LLM fallback failed: {e}")
            return self._simple_answer(question, emails_content)

    def _simple_answer(self, question: str, emails_content: list[str]) -> str:
        """Generate simple answer without LLM."""
        # Extract key info
        email_count = len(emails_content)

        if "how many" in question.lower():
            return f"I found {email_count} emails matching your query."

        if "from" in question.lower() or "who" in question.lower():
            senders = set()
            for content in emails_content:
                if "From:" in content:
                    from_line = content.split("From:")[1].split("\n")[0].strip()
                    senders.add(from_line)
            if senders:
                return f"Found emails from: {', '.join(list(senders)[:5])}"

        if "about" in question.lower() or "what" in question.lower():
            subjects = []
            for content in emails_content:
                if "Subject:" in content:
                    subj = content.split("Subject:")[1].split("\n")[0].strip()
                    subjects.append(subj)
            if subjects:
                return f"Found {email_count} emails about: {'; '.join(subjects[:3])}"

        return f"I found {email_count} relevant emails. Check the sources below for details."

    async def _handle_voice_query(
        self,
        user_id: str,
        body: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Handle voice input query."""
        state = get_user_state(user_id)

        if not state or not getattr(state, "refresh_token", None):
            return error_response("Not connected - please authenticate first", 401)

        # Get audio data
        audio_data = body.get("audio")
        audio_url = body.get("audio_url")

        if not audio_data and not audio_url:
            return error_response("Audio data or URL is required", 400)

        try:
            # Transcribe audio
            if audio_data:
                import base64

                audio_bytes = base64.b64decode(audio_data)
            else:
                # Fetch from URL
                from aragora.server.http_client_pool import get_http_pool

                pool = get_http_pool()
                async with pool.get_session("google") as client:
                    resp = await client.get(audio_url)
                    audio_bytes = resp.content

            # Transcribe
            question = await self._transcribe(audio_bytes)

            if not question:
                return error_response("Could not transcribe audio", 400)

            # Run query
            response = await self._run_query(user_id, state, question, body.get("limit", 10))

            result = response.to_dict()
            result["transcription"] = question

            return json_response(result)

        except Exception as e:
            logger.error(f"[GmailQuery] Voice query failed: {e}")
            return error_response(f"Voice query failed: {e}", 500)

    async def _transcribe(self, audio_bytes: bytes) -> str | None:
        """Transcribe audio using Whisper."""
        try:
            from aragora.connectors.whisper import WhisperConnector

            whisper = WhisperConnector()
            result = await whisper.transcribe(audio_bytes)

            return result.text if result else None

        except ImportError:
            logger.warning("[GmailQuery] Whisper not available")
            return None
        except Exception as e:
            logger.error(f"[GmailQuery] Transcription failed: {e}")
            return None

    async def _get_priority_inbox(
        self,
        user_id: str,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Get prioritized inbox list."""
        state = get_user_state(user_id)

        if not state or not getattr(state, "refresh_token", None):
            return error_response("Not connected - please authenticate first", 401)

        limit = safe_query_int(query_params, "limit", default=20, min_val=1, max_val=100)
        query = query_params.get("query", "is:unread OR newer_than:7d")

        try:
            emails = await self._get_prioritized_emails(user_id, state, query, limit)

            return json_response(
                {
                    "emails": emails,
                    "count": len(emails),
                    "user_id": user_id,
                }
            )

        except Exception as e:
            logger.error(f"[GmailQuery] Priority inbox failed: {e}")
            return error_response(f"Failed to get priority inbox: {e}", 500)

    async def _get_prioritized_emails(
        self,
        user_id: str,
        state: Any,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get emails with priority scores."""
        from aragora.analysis.email_priority import EmailPriorityAnalyzer
        from aragora.connectors.enterprise.communication.gmail import GmailConnector

        # Create connector
        connector = GmailConnector(max_results=limit * 2)
        setattr(connector, "_access_token", getattr(state, "access_token", None))
        setattr(connector, "_refresh_token", getattr(state, "refresh_token", None))
        setattr(connector, "_token_expiry", getattr(state, "token_expiry", None))

        # Get message IDs
        message_ids, _ = await connector.list_messages(query=query, max_results=limit * 2)

        # Create priority analyzer
        analyzer = EmailPriorityAnalyzer(user_id=user_id)

        # Batch fetch messages to avoid N+1 queries
        messages = await connector.get_messages(
            message_ids[: limit * 2],
            format="metadata",
        )

        # Prepare email data for batch scoring
        email_data = [
            {
                "id": msg.id,
                "subject": msg.subject,
                "from_address": msg.from_address,
                "snippet": msg.snippet,
                "labels": msg.labels,
                "is_read": msg.is_read,
                "is_starred": msg.is_starred,
            }
            for msg in messages
        ]

        # Batch score emails (loads preferences once, scores in parallel)
        # This reduces 40+ queries to 2-3 queries per request
        scores = await analyzer.score_batch(email_data)

        # Build response combining message data with scores
        emails = []
        for msg, score in zip(messages, scores):
            emails.append(
                {
                    "id": msg.id,
                    "thread_id": msg.thread_id,
                    "subject": msg.subject,
                    "from": msg.from_address,
                    "snippet": msg.snippet,
                    "date": msg.date.isoformat() if msg.date else None,
                    "labels": msg.labels,
                    "is_read": msg.is_read,
                    "is_starred": msg.is_starred,
                    "priority_score": score.score,
                    "priority_reason": score.reason,
                    "url": f"https://mail.google.com/mail/u/0/#inbox/{msg.id}",
                }
            )

        # Sort by priority score
        emails.sort(
            key=lambda x: float(x.get("priority_score", 0) or 0),
            reverse=True,
        )

        return emails[:limit]

    async def _record_feedback(self, user_id: str, body: dict[str, Any]) -> HandlerResult:
        """Record user feedback on email interaction."""
        email_id = body.get("email_id")
        action = body.get("action")
        from_address = body.get("from_address", "")
        subject = body.get("subject", "")
        labels = body.get("labels", [])

        if not email_id or not action:
            return error_response("email_id and action are required", 400)

        valid_actions = ["opened", "replied", "starred", "archived", "deleted", "snoozed"]
        if action not in valid_actions:
            return error_response(f"Invalid action. Must be one of: {valid_actions}", 400)

        try:
            from aragora.analysis.email_priority import EmailFeedbackLearner

            learner = EmailFeedbackLearner(user_id=user_id)
            success = await learner.record_interaction(
                email_id=email_id,
                action=action,
                from_address=from_address,
                subject=subject,
                labels=labels,
            )

            return json_response(
                {
                    "success": success,
                    "email_id": email_id,
                    "action": action,
                }
            )

        except Exception as e:
            logger.error(f"[GmailQuery] Feedback recording failed: {e}")
            return error_response(f"Failed to record feedback: {e}", 500)


# Export for handler registration
__all__ = ["GmailQueryHandler"]
