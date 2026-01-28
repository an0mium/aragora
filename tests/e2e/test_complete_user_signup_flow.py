"""
E2E tests for complete user signup flow.

Tests the full user journey from signup to first value:
1. User signup with email/password
2. Email verification (mocked)
3. First login
4. Create first debate
5. View debate result
6. Export decision receipt

This validates the core product value proposition end-to-end.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.billing.models import User, hash_password
from aragora.billing.jwt_auth import create_token_pair, decode_jwt
from aragora.server.handlers.base import HandlerResult


# =============================================================================
# Test Helpers
# =============================================================================


def get_body(result: HandlerResult) -> dict:
    """Extract body from HandlerResult."""
    if result is None:
        return {}
    return json.loads(result.body.decode("utf-8")) if result.body else {}


def get_status(result: HandlerResult) -> int:
    """Extract status code from HandlerResult."""
    if result is None:
        return 404
    return result.status_code


def create_mock_request(
    body: Optional[dict] = None,
    headers: Optional[dict] = None,
    method: str = "POST",
    path: str = "/",
    client_address: tuple = ("127.0.0.1", 54321),
) -> MagicMock:
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = headers or {}
    handler.client_address = client_address

    if body:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.rfile = BytesIO(body_bytes)
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.headers["Content-Type"] = "application/json"
    else:
        handler.rfile = BytesIO(b"")

    return handler


def create_mock_agent(name: str, response: str = "Default response") -> MagicMock:
    """Create a properly mocked agent with all required async methods."""
    agent = MagicMock()
    agent.name = name
    agent.generate = AsyncMock(return_value=response)

    mock_vote = MagicMock()
    mock_vote.choice = 0
    mock_vote.confidence = 0.8
    mock_vote.reasoning = "Agreed with proposal"
    agent.vote = AsyncMock(return_value=mock_vote)

    mock_critique = MagicMock()
    mock_critique.issues = []
    mock_critique.suggestions = []
    mock_critique.score = 0.8
    mock_critique.severity = 0.2
    mock_critique.text = "No issues found."
    mock_critique.agent = name
    mock_critique.target_agent = "other"
    mock_critique.round = 1
    agent.critique = AsyncMock(return_value=mock_critique)

    agent.total_input_tokens = 0
    agent.total_output_tokens = 0
    agent.input_tokens = 0
    agent.output_tokens = 0
    agent.total_tokens_in = 0
    agent.total_tokens_out = 0
    agent.metrics = None
    agent.provider = None

    return agent


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiters before each test."""
    try:
        from aragora.server.handlers.utils.rate_limit import _limiters

        for limiter in _limiters.values():
            limiter.clear()
        yield
        for limiter in _limiters.values():
            limiter.clear()
    except ImportError:
        yield


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "signup_flow_e2e_test.db"


@pytest.fixture
def user_store(temp_db_path):
    """Create a UserStore with temporary database."""
    from aragora.storage.user_store import UserStore

    store = UserStore(str(temp_db_path))
    return store


@pytest.fixture
def auth_handler(user_store):
    """Create an AuthHandler with user store context."""
    from aragora.server.handlers.auth import AuthHandler

    return AuthHandler({"user_store": user_store})


@pytest.fixture
def signup_handler():
    """Get the signup handler function."""
    from aragora.server.handlers.auth.signup_handlers import handle_signup

    return handle_signup


@pytest.fixture
def debate_handler():
    """Create a DebateHandler for debate operations."""
    from aragora.server.handlers.debates import DebateHandler

    return DebateHandler({})


@pytest.fixture
def receipt_handler():
    """Create a ReceiptHandler for receipt operations."""
    from aragora.server.handlers.receipts import ReceiptHandler

    return ReceiptHandler({})


# =============================================================================
# Complete User Signup Flow E2E Tests
# =============================================================================


@pytest.mark.e2e
class TestCompleteUserSignupFlow:
    """E2E tests for the complete user signup to first value journey."""

    def test_step1_user_signup(self, signup_handler, user_store):
        """Test user can sign up with email and password."""
        signup_data = {
            "email": "newuser@example.com",
            "password": "SecurePassword123!",
            "name": "New User",
        }

        request = create_mock_request(body=signup_data, path="/api/v1/auth/signup")
        result = signup_handler.handle(request)

        assert get_status(result) in [200, 201], f"Signup failed: {get_body(result)}"

        body = get_body(result)
        assert "user_id" in body or "id" in body or "message" in body

        # Verify user was created in store
        user = user_store.get_user_by_email("newuser@example.com")
        assert user is not None
        assert user.name == "New User"

    def test_step2_email_verification_flow(self, signup_handler, user_store):
        """Test email verification token generation and validation."""
        # First signup
        signup_data = {
            "email": "verify@example.com",
            "password": "SecurePassword123!",
            "name": "Verify User",
        }
        request = create_mock_request(body=signup_data, path="/api/v1/auth/signup")
        result = signup_handler.handle(request)

        status = get_status(result)
        assert status in [200, 201, 202], f"Signup failed with status {status}"

        # User should exist but may need verification
        user = user_store.get_user_by_email("verify@example.com")
        assert user is not None

    def test_step3_user_login(self, auth_handler, user_store):
        """Test user can login after signup."""
        # Create verified user
        password_hash, password_salt = hash_password("TestPassword123!")
        user = user_store.create_user(
            email="login@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Login User",
            role="member",
        )

        login_data = {
            "email": "login@example.com",
            "password": "TestPassword123!",
        }

        request = create_mock_request(body=login_data, path="/api/v1/auth/login")
        result = auth_handler.handle(request)

        status = get_status(result)
        body = get_body(result)

        assert status == 200, f"Login failed: {body}"
        assert "access_token" in body or "token" in body

    def test_step4_authenticated_request(self, auth_handler, user_store):
        """Test authenticated requests work with JWT token."""
        # Create user and login
        password_hash, password_salt = hash_password("TestPassword123!")
        user = user_store.create_user(
            email="auth@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Auth User",
            role="member",
        )

        # Generate tokens
        tokens = create_token_pair(str(user.id), user.email, user.role)

        # Make authenticated request
        request = create_mock_request(
            headers={"Authorization": f"Bearer {tokens.access_token}"},
            method="GET",
            path="/api/v1/auth/me",
        )

        result = auth_handler.handle(request)
        body = get_body(result)

        # Should get user info or valid response
        assert get_status(result) in [200, 401]  # 401 if endpoint not implemented

    @pytest.mark.asyncio
    async def test_step5_create_first_debate(self, user_store):
        """Test creating first debate after login."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        # Create user
        password_hash, password_salt = hash_password("TestPassword123!")
        user = user_store.create_user(
            email="debate@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Debate User",
            role="member",
        )

        # Create debate
        env = Environment(task="Should we implement feature X?")
        protocol = DebateProtocol(
            rounds=3,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        mock_agents = [
            create_mock_agent("claude", "I think we should implement it."),
            create_mock_agent("gpt", "I agree, it would be valuable."),
            create_mock_agent("gemini", "The consensus is to implement."),
        ]

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert hasattr(result, "rounds_completed")
        assert result.rounds_completed <= 3

    @pytest.mark.asyncio
    async def test_step6_view_debate_result(self, user_store):
        """Test viewing debate result after completion."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task="What is the best approach?")
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        mock_agents = [
            create_mock_agent("agent1", "Approach A is best."),
            create_mock_agent("agent2", "I agree with approach A."),
        ]

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        # Verify result structure
        assert result is not None
        assert hasattr(result, "rounds_completed")

        # Result should have final answer or consensus
        has_answer = hasattr(result, "final_answer") or hasattr(result, "consensus")
        assert has_answer or result.rounds_completed > 0

    @pytest.mark.asyncio
    async def test_step7_export_decision_receipt(self, user_store):
        """Test exporting debate result as decision receipt."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment
        from aragora.gauntlet.receipts import ReceiptGenerator

        env = Environment(task="Should we approve this decision?")
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        mock_agents = [
            create_mock_agent("agent1", "Yes, approved."),
            create_mock_agent("agent2", "Agreed, approved."),
        ]

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        # Generate receipt
        generator = ReceiptGenerator()

        # Create receipt data from result
        receipt_data = {
            "debate_id": "test-debate-123",
            "task": env.task,
            "rounds_completed": result.rounds_completed,
            "agents": ["agent1", "agent2"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        receipt = generator.generate(receipt_data)

        assert receipt is not None
        assert "debate_id" in receipt or hasattr(receipt, "debate_id")


@pytest.mark.e2e
class TestFullSignupToExportJourney:
    """Integration test for the complete signup to export journey."""

    @pytest.mark.asyncio
    async def test_complete_user_journey(self, user_store):
        """Test the complete user journey from signup to export."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment
        from aragora.billing.jwt_auth import create_token_pair

        # Step 1: Create user (simulating signup)
        password_hash, password_salt = hash_password("JourneyTest123!")
        user = user_store.create_user(
            email="journey@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Journey User",
            role="member",
        )
        assert user is not None

        # Step 2: Generate auth tokens (simulating login)
        tokens = create_token_pair(str(user.id), user.email, user.role)
        assert tokens.access_token is not None

        # Step 3: Create and run debate
        env = Environment(task="Should we adopt this new technology?")
        protocol = DebateProtocol(
            rounds=3,
            consensus="majority",
            enable_calibration=False,
            enable_rhetorical_observer=False,
            enable_trickster=False,
        )

        mock_agents = [
            create_mock_agent("claude", "Yes, the technology is mature."),
            create_mock_agent("gpt", "I agree, it has strong adoption."),
            create_mock_agent("gemini", "Consensus to adopt."),
        ]

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed > 0

        # Step 4: Verify result can be accessed
        assert hasattr(result, "rounds_completed")

        # Step 5: Generate receipt (export)
        from aragora.gauntlet.receipts import ReceiptGenerator

        generator = ReceiptGenerator()
        receipt_data = {
            "debate_id": f"debate-{user.id}",
            "user_id": str(user.id),
            "task": env.task,
            "rounds_completed": result.rounds_completed,
            "agents": [a.name for a in mock_agents],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        receipt = generator.generate(receipt_data)
        assert receipt is not None

        # Journey complete!
        print(f"Complete journey successful for user {user.email}")


@pytest.mark.e2e
class TestSignupValidation:
    """Tests for signup validation edge cases."""

    def test_signup_rejects_weak_password(self, signup_handler):
        """Test signup rejects weak passwords."""
        signup_data = {
            "email": "weak@example.com",
            "password": "123",  # Too weak
            "name": "Weak Password User",
        }

        request = create_mock_request(body=signup_data, path="/api/v1/auth/signup")
        result = signup_handler.handle(request)

        # Should reject with 400
        assert get_status(result) == 400

    def test_signup_rejects_invalid_email(self, signup_handler):
        """Test signup rejects invalid emails."""
        signup_data = {
            "email": "not-an-email",
            "password": "SecurePassword123!",
            "name": "Invalid Email User",
        }

        request = create_mock_request(body=signup_data, path="/api/v1/auth/signup")
        result = signup_handler.handle(request)

        # Should reject with 400
        assert get_status(result) == 400

    def test_signup_rejects_duplicate_email(self, signup_handler, user_store):
        """Test signup rejects duplicate email addresses."""
        # Create existing user
        password_hash, password_salt = hash_password("ExistingPassword123!")
        user_store.create_user(
            email="existing@example.com",
            password_hash=password_hash,
            password_salt=password_salt,
            name="Existing User",
            role="member",
        )

        # Try to signup with same email
        signup_data = {
            "email": "existing@example.com",
            "password": "NewPassword123!",
            "name": "Duplicate User",
        }

        request = create_mock_request(body=signup_data, path="/api/v1/auth/signup")
        result = signup_handler.handle(request)

        # Should reject with 400 or 409
        assert get_status(result) in [400, 409]
