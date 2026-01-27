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

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import (
    HandlerResult,
    error_response,
    json_response,
)
from ..secure import SecureHandler, UnauthorizedError, ForbiddenError
from .gmail_ingest import get_user_state

logger = logging.getLogger(__name__)

# Gmail permissions
GMAIL_READ_PERMISSION = "gmail:read"
GMAIL_WRITE_PERMISSION = "gmail:write"


@dataclass
class QueryResponse:
    """Response from email Q&A."""

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    query: str = ""

    def to_dict(self) -> Dict[str, Any]:
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
        query_params: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
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
            return self._get_priority_inbox(user_id, query_params)

        if path == "/api/v1/gmail/query/stream":
            # Streaming would need WebSocket - return regular response
            return self._handle_query(user_id, {"question": query_params.get("q", "")})

        return error_response("Not found", 404)

    async def handle_post(
        self,
        path: str,
        body: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route POST requests."""
        # RBAC: Require authentication and gmail:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, GMAIL_READ_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        user_id = body.get("user_id", "default")

        if path == "/api/v1/gmail/query":
            return self._handle_query(user_id, body)

        if path == "/api/v1/gmail/query/voice":
            return self._handle_voice_query(user_id, body, handler)

        if path == "/api/v1/gmail/inbox/feedback":
            return self._record_feedback(user_id, body)

        return error_response("Not found", 404)

    def _handle_query(self, user_id: str, body: Dict[str, Any]) -> HandlerResult:
        """Handle natural language query over email content."""
        state = get_user_state(user_id)

        if not state or not state.refresh_token:  # type: ignore[union-attr,attr-defined]
            return error_response("Not connected - please authenticate first", 401)

        question = body.get("question", body.get("q", ""))
        if not question:
            return error_response("Question is required", 400)

        limit = body.get("limit", 10)

        try:
            # Run query
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                response = loop.run_until_complete(self._run_query(user_id, state, question, limit))
            finally:
                loop.close()

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
        connector._access_token = state.access_token
        connector._refresh_token = state.refresh_token
        connector._token_expiry = state.token_expiry

        # Search for relevant emails
        results = await connector.search(query=question, limit=limit)

        if not results:
            return QueryResponse(
                answer="I couldn't find any emails matching your query.",
                sources=[],
                confidence=0.0,
                query=question,
            )

        # Fetch full content for top results
        emails_content = []
        sources = []

        for r in results[:5]:
            try:
                msg_id = r.id.replace("gmail-", "")
                msg = await connector.get_message(msg_id)

                emails_content.append(
                    f"From: {msg.from_address}\n"
                    f"Subject: {msg.subject}\n"
                    f"Date: {msg.date.isoformat() if msg.date else 'Unknown'}\n"
                    f"Content: {msg.body_text[:1000] if msg.body_text else msg.snippet}\n"
                )

                sources.append(
                    {
                        "id": msg_id,
                        "subject": msg.subject,
                        "from": msg.from_address,
                        "date": msg.date.isoformat() if msg.date else None,
                        "url": f"https://mail.google.com/mail/u/0/#inbox/{msg_id}",
                    }
                )

            except Exception as e:
                logger.warning(f"[GmailQuery] Failed to fetch message: {e}")

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
        emails_content: List[str],
    ) -> str:
        """Generate answer using RLM or fallback."""
        context = "\n---\n".join(emails_content)

        # Try RLM first
        try:
            from aragora.rlm.streaming import StreamingRLMQuery

            rlm = StreamingRLMQuery()  # type: ignore[call-arg]

            # Compress context if too long
            if len(context) > 4000:
                compressed = await rlm.compress(  # type: ignore[attr-defined]
                    content=context,
                    target_tokens=3000,
                )
                context = compressed or context[:4000]

            # Generate answer
            answer = await rlm.query(  # type: ignore[attr-defined]
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
            response = await agent.respond(  # type: ignore[attr-defined]
                f"Based on these emails:\n\n{context[:3000]}\n\n"
                f"Answer this question: {question}\n\n"
                "Be concise and specific. If the emails don't contain the answer, say so."
            )

            return response.content if response else self._simple_answer(question, emails_content)

        except Exception as e:
            logger.debug(f"[GmailQuery] LLM fallback failed: {e}")
            return self._simple_answer(question, emails_content)

    def _simple_answer(self, question: str, emails_content: List[str]) -> str:
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

    def _handle_voice_query(
        self,
        user_id: str,
        body: Dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Handle voice input query."""
        state = get_user_state(user_id)

        if not state or not state.refresh_token:  # type: ignore[union-attr,attr-defined]
            return error_response("Not connected - please authenticate first", 401)

        # Get audio data
        audio_data = body.get("audio")
        audio_url = body.get("audio_url")

        if not audio_data and not audio_url:
            return error_response("Audio data or URL is required", 400)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Transcribe audio
                if audio_data:
                    import base64

                    audio_bytes = base64.b64decode(audio_data)
                else:
                    # Fetch from URL
                    import httpx

                    async def fetch_audio():
                        async with httpx.AsyncClient() as client:
                            resp = await client.get(audio_url)
                            return resp.content

                    audio_bytes = loop.run_until_complete(fetch_audio())

                # Transcribe
                question = loop.run_until_complete(self._transcribe(audio_bytes))

                if not question:
                    return error_response("Could not transcribe audio", 400)

                # Run query
                response = loop.run_until_complete(
                    self._run_query(user_id, state, question, body.get("limit", 10))
                )

            finally:
                loop.close()

            result = response.to_dict()
            result["transcription"] = question

            return json_response(result)

        except Exception as e:
            logger.error(f"[GmailQuery] Voice query failed: {e}")
            return error_response(f"Voice query failed: {e}", 500)

    async def _transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio using Whisper."""
        try:
            from aragora.connectors.whisper import WhisperConnector

            whisper = WhisperConnector()
            result = await whisper.transcribe(audio_bytes)  # type: ignore[call-arg]

            return result.text if result else None

        except ImportError:
            logger.warning("[GmailQuery] Whisper not available")
            return None
        except Exception as e:
            logger.error(f"[GmailQuery] Transcription failed: {e}")
            return None

    def _get_priority_inbox(
        self,
        user_id: str,
        query_params: Dict[str, Any],
    ) -> HandlerResult:
        """Get prioritized inbox list."""
        state = get_user_state(user_id)

        if not state or not state.refresh_token:  # type: ignore[union-attr,attr-defined]
            return error_response("Not connected - please authenticate first", 401)

        limit = int(query_params.get("limit", 20))
        query = query_params.get("query", "is:unread OR newer_than:7d")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                emails = loop.run_until_complete(
                    self._get_prioritized_emails(user_id, state, query, limit)
                )
            finally:
                loop.close()

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
    ) -> List[Dict[str, Any]]:
        """Get emails with priority scores."""
        from aragora.connectors.enterprise.communication.gmail import GmailConnector
        from aragora.analysis.email_priority import EmailPriorityAnalyzer

        # Create connector
        connector = GmailConnector(max_results=limit * 2)
        connector._access_token = state.access_token
        connector._refresh_token = state.refresh_token
        connector._token_expiry = state.token_expiry

        # Get message IDs
        message_ids, _ = await connector.list_messages(query=query, max_results=limit * 2)

        # Create priority analyzer
        analyzer = EmailPriorityAnalyzer(user_id=user_id)

        # Fetch and score emails
        emails = []
        for msg_id in message_ids[: limit * 2]:
            try:
                msg = await connector.get_message(msg_id, format="metadata")

                # Score the email
                score = await analyzer.score_email(
                    email_id=msg.id,
                    subject=msg.subject,
                    from_address=msg.from_address,
                    snippet=msg.snippet,
                    labels=msg.labels,
                    is_read=msg.is_read,
                    is_starred=msg.is_starred,
                )

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

            except Exception as e:
                logger.warning(f"[GmailQuery] Failed to process message {msg_id}: {e}")

        # Sort by priority score
        emails.sort(key=lambda x: x["priority_score"], reverse=True)  # type: ignore[arg-type,return-value]

        return emails[:limit]

    def _record_feedback(self, user_id: str, body: Dict[str, Any]) -> HandlerResult:
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                from aragora.analysis.email_priority import EmailFeedbackLearner

                learner = EmailFeedbackLearner(user_id=user_id)
                success = loop.run_until_complete(
                    learner.record_interaction(
                        email_id=email_id,
                        action=action,
                        from_address=from_address,
                        subject=subject,
                        labels=labels,
                    )
                )
            finally:
                loop.close()

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
