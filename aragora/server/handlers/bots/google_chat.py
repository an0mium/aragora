"""
Google Chat Bot webhook handler.

Handles Google Chat App webhook events for bidirectional chat.

Endpoints:
- POST /api/bots/google-chat/webhook - Handle Google Chat events
- GET  /api/bots/google-chat/status - Get bot status

Environment Variables:
- GOOGLE_CHAT_CREDENTIALS - Service account JSON or path to file
- GOOGLE_CHAT_PROJECT_ID - Google Cloud project ID (required for audience validation)

Security (Phase 3.1):
- Bearer token validation uses a layered verification strategy:
  1. Primary: PyJWT-based JWKS verification via jwt_verify.JWTVerifier
  2. Fallback: google-auth id_token.verify_oauth2_token
  3. Last resort: HTTP tokeninfo endpoint check
- Token verification results are cached (5-minute TTL) to avoid
  hammering Google's endpoints on every webhook call.
- Fails closed: rejects requests when verification cannot be performed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Coroutine, Optional, Protocol, cast

from aragora.audit.unified import audit_data

if TYPE_CHECKING:
    pass


class VoteRecordingStore(Protocol):
    """Protocol for stores that can record votes.

    This is a forward-looking interface - ConsensusStore with record_vote
    is planned but not yet implemented. The code gracefully handles ImportError.
    """

    def record_vote(
        self,
        debate_id: str,
        user_id: str,
        vote: str,
        source: str,
    ) -> None:
        """Record a user vote on a debate outcome."""
        ...


from aragora.config import DEFAULT_AGENTS, DEFAULT_CONSENSUS, DEFAULT_ROUNDS
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.bots.base import BotHandlerMixin
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Environment variables - None defaults make misconfiguration explicit
GOOGLE_CHAT_CREDENTIALS = os.environ.get("GOOGLE_CHAT_CREDENTIALS")
GOOGLE_CHAT_PROJECT_ID = os.environ.get("GOOGLE_CHAT_PROJECT_ID")

# Log warnings at module load time for missing secrets
if not GOOGLE_CHAT_CREDENTIALS:
    logger.warning("GOOGLE_CHAT_CREDENTIALS not configured - Google Chat bot disabled")


# =============================================================================
# Bearer Token Verification (Phase 3.1)
# =============================================================================

# Cache for token verification results.
# Maps token_hash -> (is_valid, expiry_time)
# Uses SHA-256 hashes of tokens as keys to avoid storing raw tokens in memory.
_token_cache: dict[str, tuple[bool, float]] = {}
_token_cache_lock = threading.Lock()

# Cache TTL: 5 minutes for valid tokens, 60 seconds for invalid tokens.
# Valid tokens are safe to cache longer since they have their own expiry.
# Invalid tokens use a shorter TTL so retries with corrected tokens succeed quickly.
_TOKEN_CACHE_VALID_TTL = 300  # 5 minutes
_TOKEN_CACHE_INVALID_TTL = 60  # 1 minute

# Maximum cache size to prevent unbounded memory growth
_TOKEN_CACHE_MAX_SIZE = 1000


def _token_cache_key(token: str) -> str:
    """Generate a cache key from a token without storing the raw token.

    Uses SHA-256 to create a fixed-size key that cannot be reversed
    to recover the original token.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _get_cached_result(token: str) -> bool | None:
    """Check if a token verification result is cached and still valid.

    Args:
        token: The raw Bearer token.

    Returns:
        True/False if cached and not expired, None if not cached or expired.
    """
    key = _token_cache_key(token)
    with _token_cache_lock:
        entry = _token_cache.get(key)
        if entry is None:
            return None
        is_valid, expiry = entry
        if time.monotonic() > expiry:
            # Expired entry -- remove it
            _token_cache.pop(key, None)
            return None
        return is_valid


def _set_cached_result(token: str, is_valid: bool) -> None:
    """Cache a token verification result with appropriate TTL.

    Args:
        token: The raw Bearer token.
        is_valid: Whether the token was verified successfully.
    """
    key = _token_cache_key(token)
    ttl = _TOKEN_CACHE_VALID_TTL if is_valid else _TOKEN_CACHE_INVALID_TTL
    expiry = time.monotonic() + ttl

    with _token_cache_lock:
        # Evict expired entries if cache is getting large
        if len(_token_cache) >= _TOKEN_CACHE_MAX_SIZE:
            now = time.monotonic()
            expired_keys = [k for k, (_, exp) in _token_cache.items() if now > exp]
            for k in expired_keys:
                del _token_cache[k]
            # If still too large after eviction, clear oldest half
            if len(_token_cache) >= _TOKEN_CACHE_MAX_SIZE:
                sorted_entries = sorted(_token_cache.items(), key=lambda item: item[1][1])
                for k, _ in sorted_entries[: len(sorted_entries) // 2]:
                    del _token_cache[k]

        _token_cache[key] = (is_valid, expiry)


def clear_token_cache() -> None:
    """Clear the token verification cache.

    Useful for testing and when credentials are rotated.
    """
    with _token_cache_lock:
        _token_cache.clear()


def _verify_token_via_jwt_verifier(token: str) -> bool | None:
    """Verify token using the JWTVerifier from connectors.chat.jwt_verify.

    This is the preferred verification method. It uses PyJWT with Google's
    JWKS endpoint to cryptographically verify the token signature, expiry,
    issuer, and audience claims.

    Args:
        token: The raw JWT token (without 'Bearer ' prefix).

    Returns:
        True if valid, False if invalid, None if verifier is unavailable.
    """
    try:
        from aragora.connectors.chat.jwt_verify import verify_google_chat_webhook

        # verify_google_chat_webhook expects the full "Bearer <token>" header
        result = verify_google_chat_webhook(
            f"Bearer {token}",
            project_id=GOOGLE_CHAT_PROJECT_ID,
        )
        return result
    except ImportError:
        logger.debug("jwt_verify module not available, trying next verification method")
        return None
    except (RuntimeError, OSError) as e:
        logger.warning(f"JWTVerifier error: {e}, trying next verification method")
        return None


def _verify_token_via_google_auth(token: str) -> bool | None:
    """Verify token using google-auth library's id_token verifier.

    Falls back to this when PyJWT-based verification is unavailable.
    Uses Google's public keys to verify token signature and claims.

    Args:
        token: The raw JWT token (without 'Bearer ' prefix).

    Returns:
        True if valid, False if invalid, None if google-auth is unavailable.
    """
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        expected_audience = GOOGLE_CHAT_PROJECT_ID
        if not expected_audience:
            logger.warning(
                "GOOGLE_CHAT_PROJECT_ID not configured -- cannot validate audience claim"
            )
            # In production, audience validation is mandatory
            is_production = os.environ.get("ARAGORA_ENV", "development").lower() in (
                "production",
                "prod",
            )
            if is_production:
                logger.error(
                    "SECURITY: Rejecting token -- audience validation required "
                    "in production. Set GOOGLE_CHAT_PROJECT_ID."
                )
                return False
            # In development, proceed without audience validation (with warning)

        id_info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            audience=expected_audience,
        )

        # Verify the token is from Google Chat's service account
        email = id_info.get("email", "")
        if email and not email.endswith("@system.gserviceaccount.com"):
            logger.warning(f"Google Chat token from unexpected issuer email: {email}")
            return False

        return True

    except ImportError:
        logger.debug("google-auth library not installed, trying next verification method")
        return None
    except ValueError as e:
        logger.warning(f"google-auth token verification failed: {e}")
        return False
    except (RuntimeError, OSError) as e:
        logger.warning(f"google-auth verification error: {e}")
        return None


def _verify_token_via_tokeninfo(token: str) -> bool | None:
    """Verify token via Google's OAuth2 tokeninfo HTTP endpoint.

    This is the last-resort verification method when neither PyJWT nor
    google-auth are available. It makes an HTTP call to Google's tokeninfo
    endpoint to validate the token.

    Args:
        token: The raw JWT token (without 'Bearer ' prefix).

    Returns:
        True if valid, False if invalid, None if HTTP request fails.
    """
    try:
        import urllib.error
        import urllib.request

        url = f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "Aragora/1.0")

        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                logger.warning(f"Google tokeninfo returned status {resp.status}")
                return False

            data = json.loads(resp.read().decode("utf-8"))

            # Validate audience if project ID is configured
            if GOOGLE_CHAT_PROJECT_ID:
                aud = data.get("aud", "")
                if aud != GOOGLE_CHAT_PROJECT_ID:
                    logger.warning(
                        f"Token audience mismatch: expected {GOOGLE_CHAT_PROJECT_ID}, got {aud}"
                    )
                    return False

            # Validate issuer email
            email = data.get("email", "")
            if email and not email.endswith("@system.gserviceaccount.com"):
                logger.warning(f"Token from unexpected email: {email}")
                return False

            # Check token is not expired (tokeninfo may still return
            # data for recently expired tokens)
            exp = data.get("exp")
            if exp:
                try:
                    if int(exp) < int(time.time()):
                        logger.warning("Token is expired per tokeninfo response")
                        return False
                except (ValueError, TypeError):
                    pass

            return True

    except urllib.error.HTTPError as e:
        logger.warning(f"Google tokeninfo HTTP error: {e.code}")
        return False
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning(f"Google tokeninfo request failed: {e}")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Google tokeninfo response parse error: {e}")
        return False


def _verify_google_chat_token(auth_header: str) -> bool:
    """Verify Google Chat Bearer token from the Authorization header.

    Uses a layered verification strategy with caching:
    1. Check token cache (avoids repeated verification of the same token)
    2. Try PyJWT-based JWKS verification (cryptographic, preferred)
    3. Try google-auth id_token verification (cryptographic, fallback)
    4. Try HTTP tokeninfo endpoint (network-based, last resort)

    Security policy: Fails closed. If no verification method is available
    or all methods return inconclusive results, the token is rejected.

    Args:
        auth_header: The full Authorization header value
                     (e.g., "Bearer <token>").

    Returns:
        True if the token is verified as valid, False otherwise.
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Missing or malformed Authorization header")
        return False

    token = auth_header[7:].strip()  # Remove "Bearer " prefix
    if not token:
        logger.warning("Empty Bearer token")
        return False

    # 1. Check cache first
    cached = _get_cached_result(token)
    if cached is not None:
        logger.debug(f"Token verification cache {'hit (valid)' if cached else 'hit (invalid)'}")
        return cached

    # 2. Try PyJWT-based JWKS verification (preferred)
    result = _verify_token_via_jwt_verifier(token)
    if result is not None:
        _set_cached_result(token, result)
        return result

    # 3. Try google-auth library verification
    result = _verify_token_via_google_auth(token)
    if result is not None:
        _set_cached_result(token, result)
        return result

    # 4. Try HTTP tokeninfo endpoint (last resort)
    result = _verify_token_via_tokeninfo(token)
    if result is not None:
        _set_cached_result(token, result)
        return result

    # All verification methods failed or were unavailable -- fail closed
    logger.error(
        "SECURITY: All token verification methods unavailable. "
        "Install PyJWT (pip install pyjwt[crypto]) or "
        "google-auth (pip install google-auth) for token verification. "
        "Rejecting request."
    )
    return False


# Input validation limits
MAX_TOPIC_LENGTH = 500
MAX_STATEMENT_LENGTH = 1000


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed: {exc}", exc_info=exc)


def create_tracked_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_exception(t, name))
    return task


# Cache for Google Chat connector
_google_chat_connector: Any | None = None


def get_google_chat_connector() -> Any | None:
    """Get or create the Google Chat connector singleton."""
    global _google_chat_connector
    if _google_chat_connector is None:
        if not GOOGLE_CHAT_CREDENTIALS:
            logger.debug("Google Chat connector disabled (no GOOGLE_CHAT_CREDENTIALS)")
            return None
        try:
            from aragora.connectors.chat.google_chat import GoogleChatConnector

            _google_chat_connector = GoogleChatConnector(
                credentials_json=GOOGLE_CHAT_CREDENTIALS,
                project_id=GOOGLE_CHAT_PROJECT_ID,
            )
            logger.info("Google Chat connector initialized")
        except ImportError as e:
            logger.warning(f"Google Chat connector module not available: {e}")
            return None
        except (RuntimeError, ValueError, OSError) as e:
            logger.exception(f"Error initializing Google Chat connector: {e}")
            return None
    return _google_chat_connector


class GoogleChatHandler(BotHandlerMixin, SecureHandler):
    """Handler for Google Chat App webhook endpoints.

    Uses BotHandlerMixin for shared auth/status patterns.

    RBAC Protected:
    - bots.read - required for status endpoint

    Note: Webhook endpoints are authenticated via Google's bearer token,
    not RBAC, since they are called by Google servers directly.
    """

    # BotHandlerMixin configuration
    bot_platform = "google_chat"

    ROUTES = [
        "/api/v1/bots/google-chat/webhook",
        "/api/v1/bots/google-chat/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def _is_bot_enabled(self) -> bool:
        """Check if Google Chat bot is configured."""
        connector = get_google_chat_connector()
        return connector is not None

    def _build_status_response(
        self, extra_status: Optional[dict[str, Any]] = None
    ) -> HandlerResult:
        """Build Google Chat-specific status response."""
        status = {
            "platform": self.bot_platform,
            "enabled": self._is_bot_enabled(),
            "credentials_configured": bool(GOOGLE_CHAT_CREDENTIALS),
            "project_id_configured": bool(GOOGLE_CHAT_PROJECT_ID),
        }
        if extra_status:
            status.update(extra_status)
        return json_response(status)

    @rate_limit(requests_per_minute=60)
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route Google Chat GET requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/google-chat/status":
            # Use BotHandlerMixin's RBAC-protected status handler
            return await self.handle_status_request(handler)

        return None

    @rate_limit(requests_per_minute=120)
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests (webhook events)."""
        if path == "/api/v1/bots/google-chat/webhook":
            return self._handle_webhook(handler)

        return None

    def _handle_webhook(self, handler: Any) -> HandlerResult:
        """Handle Google Chat webhook events.

        Google Chat sends events to this endpoint for:
        - Messages (regular and slash commands)
        - Card button clicks
        - Bot added/removed from space

        Authentication: Google Chat sends a bearer token in the Authorization header.
        We verify the token is a valid Google-issued JWT with the correct audience.
        """
        # Require credentials to be configured
        if not GOOGLE_CHAT_CREDENTIALS:
            logger.warning("GOOGLE_CHAT_CREDENTIALS not configured - rejecting webhook request")
            return json_response(
                {"error": "Google Chat credentials not configured"},
                status=503,
            )

        # Verify bearer token (Google Chat authentication - Phase 3.1)
        auth_header = handler.headers.get("Authorization", "")
        if not _verify_google_chat_token(auth_header):
            logger.warning("Google Chat webhook authentication failed")
            self._audit_webhook_auth_failure("bearer_token", "invalid")
            return error_response("Invalid authorization", 401)

        try:
            # Read and parse body
            body = self._read_request_body(handler)
            event, err = self._parse_json_body(body, "Google Chat event")
            if err:
                return err

            event_type = event.get("type", "")
            logger.debug(f"Google Chat event: {event_type}")

            # Route by event type
            if event_type == "MESSAGE":
                return self._handle_message(event)
            elif event_type == "CARD_CLICKED":
                return self._handle_card_click(event)
            elif event_type == "ADDED_TO_SPACE":
                return self._handle_added_to_space(event)
            elif event_type == "REMOVED_FROM_SPACE":
                return self._handle_removed_from_space(event)
            else:
                logger.debug(f"Unhandled Google Chat event type: {event_type}")
                return json_response({})

        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.exception(f"Google Chat webhook error: {e}")
            return json_response({"text": f"Error: {str(e)[:100]}"})

    def _handle_message(self, event: dict[str, Any]) -> HandlerResult:
        """Handle incoming Google Chat message."""
        message = event.get("message", {})
        space = event.get("space", {})
        user = event.get("user", {})

        space_name = space.get("name", "")
        space_type = space.get("type", "")
        user_name = user.get("displayName", "Unknown")
        text = message.get("text", "")

        logger.info(f"Google Chat message from {user_name}: {text[:50]}...")

        # Check for slash command
        slash_command = message.get("slashCommand")
        if slash_command:
            command_name = slash_command.get("commandName", "").lstrip("/")
            args = message.get("argumentText", "")
            return self._handle_slash_command(command_name, args, space_name, user, event)

        # Check if in DM - treat as implicit debate request
        if space_type == "DM" and text.strip():
            return self._start_debate_from_message(text, space_name, user, event)

        # Check for @mention
        mentions = message.get("annotations", [])
        for mention in mentions:
            if (
                mention.get("type") == "USER_MENTION"
                and mention.get("userMention", {}).get("type") == "BOT"
            ):
                # Bot was mentioned - extract text after mention
                mention_text = text
                if mention_text.strip():
                    return self._start_debate_from_message(mention_text, space_name, user, event)

        # Regular message in group - acknowledge
        return json_response({})

    def _handle_slash_command(
        self,
        command: str,
        args: str,
        space_name: str,
        user: dict[str, Any],
        event: dict[str, Any],
    ) -> HandlerResult:
        """Handle Google Chat slash command."""
        command = command.lower()
        user_name = user.get("displayName", "Unknown")

        logger.info(f"Google Chat command from {user_name}: /{command} {args[:50]}...")

        if command == "debate":
            return self._cmd_debate(args, space_name, user, event)
        elif command == "gauntlet":
            return self._cmd_gauntlet(args, space_name, user, event)
        elif command == "status":
            return self._cmd_status(space_name)
        elif command == "help":
            return self._cmd_help()
        elif command == "agents":
            return self._cmd_agents()
        else:
            return self._card_response(
                title="Unknown Command",
                body=f"Unknown command: /{command}\n\nUse /help for available commands.",
            )

    def _handle_card_click(self, event: dict[str, Any]) -> HandlerResult:
        """Handle Google Chat card button click."""
        action = event.get("action", {})
        function_name = action.get("actionMethodName", "")
        params = {p.get("key"): p.get("value") for p in action.get("parameters", [])}

        user = event.get("user", {})
        user_id = user.get("name", "").split("/")[-1]
        space = event.get("space", {})
        space_name = space.get("name", "")

        logger.info(f"Google Chat card click: {function_name} from {user_id}")

        # Route actions
        if function_name == "vote_agree":
            return self._handle_vote(params.get("debate_id", ""), "agree", user_id, space_name)
        elif function_name == "vote_disagree":
            return self._handle_vote(params.get("debate_id", ""), "disagree", user_id, space_name)
        elif function_name == "view_details":
            return self._handle_view_details(params.get("debate_id", ""))

        return json_response({})

    def _handle_vote(
        self, debate_id: str, vote_option: str, user_id: str, space_name: str
    ) -> HandlerResult:
        """Handle vote button click."""
        if not debate_id:
            return self._card_response(body="Error: No debate ID provided")

        logger.info(f"Vote from {user_id} on {debate_id}: {vote_option}")

        try:
            # ConsensusStore with record_vote is a planned feature.
            # This import will fail until it's implemented.
            from aragora.memory.consensus import ConsensusStore

            store: VoteRecordingStore = cast(VoteRecordingStore, ConsensusStore())
            store.record_vote(
                debate_id=debate_id,
                user_id=f"gchat:{user_id}",
                vote=vote_option,
                source="google_chat",
            )

            audit_data(
                user_id=f"gchat:{user_id}",
                resource_type="debate_vote",
                resource_id=debate_id,
                action="create",
                vote_option=vote_option,
                platform="google_chat",
            )

            emoji = "\u2705" if vote_option == "agree" else "\u274c"
            return self._card_response(
                body=f"{emoji} Your vote for '{vote_option}' has been recorded!"
            )

        except ImportError:
            logger.warning("ConsensusStore not available")
            return self._card_response(body="Vote acknowledged")
        except (RuntimeError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"Vote recording failed: {e}")
            return self._card_response(body="Error recording vote")

    def _handle_view_details(self, debate_id: str) -> HandlerResult:
        """Handle view details button click."""
        if not debate_id:
            return self._card_response(body="Error: No debate ID provided")

        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            debate = db.get(debate_id) if db else None

            if not debate:
                return self._card_response(body=f"Debate {debate_id} not found")

            task = debate.get("task", "Unknown topic")
            final_answer = debate.get("final_answer", "No conclusion")
            consensus = debate.get("consensus_reached", False)
            confidence = debate.get("confidence", 0)
            rounds_used = debate.get("rounds_used", 0)

            return self._card_response(
                title="Debate Details",
                body=f"*Topic:* {task[:200]}\n\n"
                f"*Consensus:* {'Yes' if consensus else 'No'}\n"
                f"*Confidence:* {confidence:.1%}\n"
                f"*Rounds:* {rounds_used}\n\n"
                f"*Conclusion:*\n{final_answer[:500]}...",
            )

        except (ImportError, RuntimeError, KeyError, AttributeError) as e:
            logger.error(f"View details failed: {e}")
            return self._card_response(body="Error fetching debate details")

    def _handle_added_to_space(self, event: dict[str, Any]) -> HandlerResult:
        """Handle bot added to space."""
        space = event.get("space", {})
        space.get("displayName", "this space")

        return self._card_response(
            title="Welcome to Aragora!",
            body="I'm a control plane for multi-agent vetted decisionmaking.\n\n"
            "I orchestrate 15+ AI models (Claude, GPT, Gemini, Grok and more) "
            "to debate and deliver defensible decisions.\n\n"
            "*Commands:*\n"
            "/debate <topic> - Start a multi-agent vetted decisionmaking\n"
            "/gauntlet <statement> - Run adversarial validation\n"
            "/status - Check system status\n"
            "/help - Show available commands\n\n"
            "Or just @mention me with a question!",
        )

    def _handle_removed_from_space(self, event: dict[str, Any]) -> HandlerResult:
        """Handle bot removed from space."""
        space = event.get("space", {})
        logger.info(f"Bot removed from space: {space.get('name')}")
        return json_response({})

    # ==========================================================================
    # Slash Command Implementations
    # ==========================================================================

    def _cmd_help(self) -> HandlerResult:
        """Show help message."""
        return self._card_response(
            title="Aragora Commands",
            body="*/debate <topic>* - Start a multi-agent debate\n"
            "*/gauntlet <statement>* - Run adversarial stress-test\n"
            "*/status* - Check system status\n"
            "*/agents* - List available agents\n"
            "*/help* - Show this message\n\n"
            "*Examples:*\n"
            "/debate Should AI be regulated?\n"
            "/gauntlet We should migrate to microservices",
        )

    def _cmd_status(self, space_name: str) -> HandlerResult:
        """Show system status."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            return self._card_response(
                title="Aragora System Status",
                fields=[
                    ("Status", "Online"),
                    ("Agents", str(len(agents))),
                    ("Models", "Claude, GPT, Gemini, Grok, Mistral, DeepSeek"),
                ],
            )

        except ImportError:
            return self._card_response(
                title="Aragora System Status",
                fields=[("Status", "Online"), ("Agents", "Available")],
            )

    def _cmd_agents(self) -> HandlerResult:
        """List available agents."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            if not agents:
                return self._card_response(body="No agents registered yet.")

            agents = sorted(agents, key=lambda a: getattr(a, "elo", 1500), reverse=True)

            lines = []
            for i, agent in enumerate(agents[:10]):
                name = getattr(agent, "name", "Unknown")
                elo = getattr(agent, "elo", 1500)
                medal = ["\U0001f947", "\U0001f948", "\U0001f949"][i] if i < 3 else f"{i + 1}."
                lines.append(f"{medal} *{name}* - ELO: {elo:.0f}")

            return self._card_response(
                title="Top Agents by ELO",
                body="\n".join(lines),
            )

        except ImportError:
            return self._card_response(body="Agents service temporarily unavailable")

    def _cmd_debate(
        self,
        topic: str,
        space_name: str,
        user: dict[str, Any],
        event: dict[str, Any],
    ) -> HandlerResult:
        """Start a debate."""
        if not topic.strip():
            return self._card_response(
                body="Please provide a topic.\n\nExample: /debate Should AI be regulated?"
            )

        topic = topic.strip().strip("\"'")

        if len(topic) < 10:
            return self._card_response(body="Topic is too short. Please provide more detail.")

        if len(topic) > MAX_TOPIC_LENGTH:
            return self._card_response(
                body=f"Topic is too long. Please limit to {MAX_TOPIC_LENGTH} characters."
            )

        user_name = user.get("displayName", "Unknown")
        user_id = user.get("name", "").split("/")[-1]

        # Queue debate asynchronously
        create_tracked_task(
            self._run_debate_async(topic, space_name, user_id),
            name=f"gchat-debate-{topic[:30]}",
        )

        return self._card_response(
            title="Starting Debate",
            body=f"*Topic:* {topic}\n\n_Processing... I'll post the result when complete._",
            context=f"Requested by {user_name}",
        )

    def _cmd_gauntlet(
        self,
        statement: str,
        space_name: str,
        user: dict[str, Any],
        event: dict[str, Any],
    ) -> HandlerResult:
        """Run gauntlet validation."""
        if not statement.strip():
            return self._card_response(
                body="Please provide a statement to stress-test.\n\n"
                "Example: /gauntlet We should migrate to microservices"
            )

        statement = statement.strip().strip("\"'")

        if len(statement) < 10:
            return self._card_response(body="Statement is too short. Please provide more detail.")

        if len(statement) > MAX_STATEMENT_LENGTH:
            return self._card_response(
                body=f"Statement is too long. Please limit to {MAX_STATEMENT_LENGTH} characters."
            )

        user_name = user.get("displayName", "Unknown")
        user_id = user.get("name", "").split("/")[-1]

        # Queue gauntlet asynchronously
        create_tracked_task(
            self._run_gauntlet_async(statement, space_name, user_id),
            name=f"gchat-gauntlet-{statement[:30]}",
        )

        return self._card_response(
            title="Running Gauntlet Stress-Test",
            body=f"*Statement:* {statement[:200]}{'...' if len(statement) > 200 else ''}\n\n"
            "_Running adversarial validation..._",
            context=f"Requested by {user_name}",
        )

    def _start_debate_from_message(
        self,
        text: str,
        space_name: str,
        user: dict[str, Any],
        event: dict[str, Any],
    ) -> HandlerResult:
        """Start a debate from a regular message or @mention."""
        # Clean up the text (remove @mentions)
        import re

        clean_text = re.sub(r"<users/[^>]+>", "", text).strip()

        if not clean_text or len(clean_text) < 10:
            return self._card_response(
                body="I need more context. Try asking a specific question or use /debate <topic>."
            )

        return self._cmd_debate(clean_text, space_name, user, event)

    # ==========================================================================
    # Async Operations
    # ==========================================================================

    async def _run_debate_async(
        self,
        topic: str,
        space_name: str,
        user_id: str,
    ) -> None:
        """Run debate asynchronously via DecisionRouter and post result.

        Uses the unified DecisionRouter for:
        - Deduplication across channels
        - Response caching
        - RBAC enforcement
        - Consistent routing

        Falls back to direct Arena execution if DecisionRouter unavailable.
        """
        connector = get_google_chat_connector()
        if not connector:
            logger.warning("No Google Chat connector available")
            return

        # Try DecisionRouter first (preferred)
        try:
            from aragora.core.decision import (
                DecisionRequest,
                DecisionType,
                InputSource,
                ResponseChannel,
                RequestContext,
                get_decision_router,
            )
            from aragora.server.debate_origin import register_debate_origin
            import uuid

            debate_id = str(uuid.uuid4())

            # Register origin for result routing
            register_debate_origin(
                debate_id=debate_id,
                platform="google_chat",
                channel_id=space_name,
                user_id=user_id,
                metadata={"topic": topic},
            )

            request = DecisionRequest(
                content=topic,
                decision_type=DecisionType.DEBATE,
                source=InputSource.GOOGLE_CHAT,
                response_channels=[
                    ResponseChannel(
                        platform="google_chat",
                        channel_id=space_name,
                        user_id=user_id,
                    )
                ],
                context=RequestContext(
                    user_id=f"gchat:{user_id}",
                    metadata={"space_name": space_name},
                ),
            )

            router = get_decision_router()
            result = await router.route(request)
            if result and result.success:
                logger.info(f"DecisionRouter completed debate for Google Chat: {result.request_id}")
                return  # Response handler will post the result
            else:
                logger.warning("DecisionRouter returned unsuccessful result, falling back")

        except ImportError:
            logger.debug("DecisionRouter not available, falling back to direct execution")
        except (RuntimeError, ValueError, KeyError, AttributeError) as e:
            logger.error(f"DecisionRouter failed: {e}, falling back to direct execution")

        # Fallback to direct Arena execution
        try:
            from aragora import Arena, DebateProtocol, Environment
            from aragora.agents import get_agents_by_names

            env = Environment(task=f"Debate: {topic}")
            default_agents = [a.strip() for a in DEFAULT_AGENTS.split(",") if a.strip()]
            agents = get_agents_by_names(default_agents)
            protocol = DebateProtocol(
                rounds=DEFAULT_ROUNDS,
                consensus=DEFAULT_CONSENSUS,
                convergence_detection=False,
                early_stopping=False,
            )

            if not agents:
                await connector.send_message(
                    space_name,
                    "Failed to start debate: No agents available",
                )
                return

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            # Build result card
            consensus_emoji = "\u2705" if result.consensus_reached else "\u274c"
            confidence_bar = "\u2588" * int(result.confidence * 5) + "\u2591" * (
                5 - int(result.confidence * 5)
            )

            sections = (
                connector.format_blocks(
                    title=f"{consensus_emoji} Debate Complete",
                    body=f"*Topic:* {topic[:100]}...\n\n"
                    f"*Conclusion:*\n{result.final_answer[:400] if result.final_answer else 'No conclusion'}...",
                    fields=[
                        ("Consensus", "Yes" if result.consensus_reached else "No"),
                        ("Confidence", f"{confidence_bar} {result.confidence:.0%}"),
                        ("Rounds", str(result.rounds_used)),
                        ("Agents", str(len(agents))),
                    ],
                    actions=[
                        connector.connectors.chat.models.MessageButton(
                            text="\U0001f44d Agree",
                            action_id="vote_agree",
                            value=result.id,
                        ),
                        connector.connectors.chat.models.MessageButton(
                            text="\U0001f44e Disagree",
                            action_id="vote_disagree",
                            value=result.id,
                        ),
                    ],
                )
                if hasattr(connector, "format_blocks")
                else []
            )

            await connector.send_message(
                space_name,
                f"Debate complete: {topic[:50]}...",
                blocks=sections,
            )

        except (RuntimeError, ImportError, ValueError, AttributeError, OSError) as e:
            logger.error(f"Async debate failed: {e}", exc_info=True)
            if connector:
                await connector.send_message(
                    space_name,
                    f"Debate failed: {str(e)[:100]}",
                )

    async def _run_gauntlet_async(
        self,
        statement: str,
        space_name: str,
        user_id: str,
    ) -> None:
        """Run gauntlet asynchronously and post result."""
        connector = get_google_chat_connector()
        if not connector:
            logger.warning("No Google Chat connector available")
            return

        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("google_chat_handler") as client:
                resp = await client.post(
                    "http://localhost:8080/api/v1/gauntlet/run",
                    json={
                        "statement": statement,
                        "intensity": "medium",
                        "metadata": {
                            "source": "google_chat",
                            "space_name": space_name,
                            "user_id": user_id,
                        },
                    },
                    timeout=120.0,
                )
                data = resp.json()

                if resp.status_code != 200:
                    await connector.send_message(
                        space_name,
                        f"Gauntlet failed: {data.get('error', 'Unknown error')}",
                    )
                    return

                score = data.get("score", 0)
                passed = data.get("passed", False)
                vulnerabilities = data.get("vulnerabilities", [])

                status_emoji = "\u2705" if passed else "\u274c"
                score_bar = "\u2588" * int(score * 5) + "\u2591" * (5 - int(score * 5))

                body = f"*Statement:* {statement[:200]}{'...' if len(statement) > 200 else ''}\n\n"
                body += f"*Score:* {score_bar} {score:.0%}\n"
                body += f"*Status:* {'Passed' if passed else 'Failed'}\n"
                body += f"*Issues Found:* {len(vulnerabilities)}\n"

                if vulnerabilities:
                    body += "\n*Issues:*\n"
                    for v in vulnerabilities[:3]:
                        body += f"\u2022 {v.get('description', 'Unknown')[:80]}\n"
                    if len(vulnerabilities) > 3:
                        body += f"_...and {len(vulnerabilities) - 3} more_"

                await connector.send_message(
                    space_name,
                    "Gauntlet complete",
                    blocks=(
                        connector.format_blocks(
                            title=f"{status_emoji} Gauntlet Results",
                            body=body,
                        )
                        if hasattr(connector, "format_blocks")
                        else None
                    ),
                )

        except (RuntimeError, ImportError, ValueError, AttributeError, OSError) as e:
            logger.error(f"Async gauntlet failed: {e}", exc_info=True)
            if connector:
                await connector.send_message(
                    space_name,
                    f"Gauntlet failed: {str(e)[:100]}",
                )

    # ==========================================================================
    # Response Helpers
    # ==========================================================================

    def _card_response(
        self,
        title: str | None = None,
        body: str | None = None,
        fields: list[tuple[str, str] | None] = None,
        context: str | None = None,
        actions: list[dict] | None = None,
    ) -> HandlerResult:
        """Create a Google Chat Card v2 response."""
        sections: list[dict] = []

        # Header
        if title:
            sections.append({"header": title})

        # Body text
        if body:
            sections.append({"widgets": [{"textParagraph": {"text": body}}]})

        # Fields as decorated text
        if fields:
            widgets = []
            for label, value in fields:
                widgets.append(
                    {
                        "decoratedText": {
                            "topLabel": label,
                            "text": value,
                        }
                    }
                )
            sections.append({"widgets": widgets})

        # Context footer
        if context:
            sections.append({"widgets": [{"textParagraph": {"text": f"<i>{context}</i>"}}]})

        # Action buttons
        if actions:
            sections.append({"widgets": [{"buttonList": {"buttons": actions}}]})

        # Build card response
        response: dict[str, Any] = {}

        if sections:
            response["cardsV2"] = [
                {
                    "cardId": "aragora_response",
                    "card": {"sections": sections},
                }
            ]
        elif body:
            response["text"] = body

        return json_response(response)


# Export handler factory
_google_chat_handler: Optional["GoogleChatHandler"] = None


def get_google_chat_handler(server_context: dict[str, Any] | None = None) -> "GoogleChatHandler":
    """Get or create the Google Chat handler instance."""
    global _google_chat_handler
    if _google_chat_handler is None:
        if server_context is None:
            server_context = {}
        _google_chat_handler = GoogleChatHandler(server_context)
    return _google_chat_handler


__all__ = [
    "GoogleChatHandler",
    "get_google_chat_handler",
    "get_google_chat_connector",
    "clear_token_cache",
]
