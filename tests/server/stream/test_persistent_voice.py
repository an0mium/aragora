"""
Tests for PersistentVoiceManager and PersistentVoiceSession.

Tests cover:
- Session creation and lifecycle
- Reconnection token generation and validation
- Heartbeat handling and timeout detection
- Session cleanup and expiry
- Concurrent session handling
- Error recovery scenarios
- State persistence across reconnections
- Callbacks for session events
- Statistics and monitoring
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.stream.persistent_voice import (
    PersistentVoiceManager,
    PersistentVoiceSession,
    get_persistent_voice_manager,
    VOICE_HEARTBEAT_TIMEOUT_SECONDS,
    VOICE_MAX_SESSION_SECONDS,
    VOICE_RECONNECT_WINDOW_SECONDS,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def manager():
    """Create a fresh PersistentVoiceManager for each test."""
    return PersistentVoiceManager()


@pytest.fixture
def mock_session_store():
    """Create a mock session store."""
    store = MagicMock()
    store.set_voice_session = MagicMock()
    store.get_voice_session = MagicMock(return_value=None)
    store.delete_voice_session = MagicMock()
    return store


@pytest.fixture
def sample_session():
    """Create a sample PersistentVoiceSession."""
    return PersistentVoiceSession(
        session_id="voice_abc123",
        user_id="user_456",
        debate_id="debate_789",
        is_persistent=True,
        audio_format="pcm_16khz",
    )


# ===========================================================================
# PersistentVoiceSession Tests
# ===========================================================================


class TestPersistentVoiceSession:
    """Tests for PersistentVoiceSession dataclass."""

    def test_session_initialization_defaults(self):
        """Test session initializes with correct defaults."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
        )

        assert session.session_id == "voice_test"
        assert session.user_id == "user_123"
        assert session.debate_id is None
        assert session.is_persistent is True
        assert session.audio_format == "pcm_16khz"
        assert session.state == "active"
        assert session.reconnect_token is None
        assert session.reconnect_expires_at is None
        assert session.audio_buffer == b""
        assert session.transcript_buffer == ""
        assert session.metadata == {}

    def test_session_initialization_with_all_params(self):
        """Test session initializes with all parameters."""
        now = time.time()
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            debate_id="debate_456",
            is_persistent=False,
            audio_format="opus",
            state="disconnected",
            reconnect_token="token_abc",
            reconnect_expires_at=now + 300,
            last_heartbeat=now,
            created_at=now,
            expires_at=now + 3600,
            metadata={"key": "value"},
            audio_buffer=b"\x00\x01",
            transcript_buffer="hello",
        )

        assert session.debate_id == "debate_456"
        assert session.is_persistent is False
        assert session.audio_format == "opus"
        assert session.state == "disconnected"
        assert session.reconnect_token == "token_abc"
        assert session.metadata == {"key": "value"}
        assert session.audio_buffer == b"\x00\x01"
        assert session.transcript_buffer == "hello"

    def test_is_active_true(self):
        """Test is_active returns True for active, non-expired session."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="active",
            expires_at=time.time() + 3600,
        )
        assert session.is_active is True

    def test_is_active_false_wrong_state(self):
        """Test is_active returns False for non-active state."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="disconnected",
            expires_at=time.time() + 3600,
        )
        assert session.is_active is False

    def test_is_active_false_expired(self):
        """Test is_active returns False for expired session."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="active",
            expires_at=time.time() - 1,  # Expired
        )
        assert session.is_active is False

    def test_is_disconnected_true(self):
        """Test is_disconnected returns True for reconnectable disconnected session."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="disconnected",
            is_persistent=True,
            reconnect_expires_at=time.time() + 300,
        )
        assert session.is_disconnected is True

    def test_is_disconnected_false_wrong_state(self):
        """Test is_disconnected returns False for active session."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="active",
            is_persistent=True,
            reconnect_expires_at=time.time() + 300,
        )
        assert session.is_disconnected is False

    def test_is_disconnected_false_cannot_reconnect(self):
        """Test is_disconnected returns False when can't reconnect."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="disconnected",
            is_persistent=False,  # Not persistent
        )
        assert session.is_disconnected is False

    def test_can_reconnect_true(self):
        """Test can_reconnect returns True for valid reconnect window."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            is_persistent=True,
            reconnect_expires_at=time.time() + 300,
        )
        assert session.can_reconnect is True

    def test_can_reconnect_false_not_persistent(self):
        """Test can_reconnect returns False for non-persistent session."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            is_persistent=False,
            reconnect_expires_at=time.time() + 300,
        )
        assert session.can_reconnect is False

    def test_can_reconnect_false_no_token(self):
        """Test can_reconnect returns False when no reconnect token set."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            is_persistent=True,
            reconnect_expires_at=None,
        )
        assert session.can_reconnect is False

    def test_can_reconnect_false_expired(self):
        """Test can_reconnect returns False when window expired."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            is_persistent=True,
            reconnect_expires_at=time.time() - 1,  # Expired
        )
        assert session.can_reconnect is False

    def test_time_until_expiry(self):
        """Test time_until_expiry calculation."""
        future = time.time() + 3600
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            expires_at=future,
        )

        remaining = session.time_until_expiry
        assert 3599 <= remaining <= 3600

    def test_time_until_expiry_zero_when_expired(self):
        """Test time_until_expiry returns 0 for expired session."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            expires_at=time.time() - 100,
        )
        assert session.time_until_expiry == 0

    def test_time_until_reconnect_expiry(self):
        """Test time_until_reconnect_expiry calculation."""
        future = time.time() + 300
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            reconnect_expires_at=future,
        )

        remaining = session.time_until_reconnect_expiry
        assert 299 <= remaining <= 300

    def test_time_until_reconnect_expiry_zero_when_none(self):
        """Test time_until_reconnect_expiry returns 0 when not set."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            reconnect_expires_at=None,
        )
        assert session.time_until_reconnect_expiry == 0

    def test_time_until_reconnect_expiry_zero_when_expired(self):
        """Test time_until_reconnect_expiry returns 0 when expired."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            reconnect_expires_at=time.time() - 100,
        )
        assert session.time_until_reconnect_expiry == 0

    def test_touch_updates_heartbeat(self):
        """Test touch() updates last_heartbeat timestamp."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
        )
        original = session.last_heartbeat
        time.sleep(0.01)

        session.touch()

        assert session.last_heartbeat > original

    def test_extend_updates_expiry(self):
        """Test extend() updates expiration timestamp."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            expires_at=time.time() + 100,
        )

        session.extend(7200)  # Extend by 2 hours

        # Should be around 2 hours from now
        expected = time.time() + 7200
        assert abs(session.expires_at - expected) < 1

    def test_to_dict_serialization(self):
        """Test to_dict() serializes session correctly."""
        now = time.time()
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            debate_id="debate_456",
            is_persistent=True,
            audio_format="opus",
            state="active",
            created_at=now,
            expires_at=now + 3600,
            metadata={"key": "value"},
        )

        data = session.to_dict()

        assert data["session_id"] == "voice_test"
        assert data["user_id"] == "user_123"
        assert data["debate_id"] == "debate_456"
        assert data["is_persistent"] is True
        assert data["audio_format"] == "opus"
        assert data["state"] == "active"
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data
        assert "expires_at" in data
        assert "can_reconnect" in data
        assert "time_until_expiry" in data
        assert "time_until_reconnect_expiry" in data


# ===========================================================================
# PersistentVoiceManager Lifecycle Tests
# ===========================================================================


class TestManagerLifecycle:
    """Tests for PersistentVoiceManager start/stop lifecycle."""

    def test_initialization(self, manager):
        """Test manager initializes with correct defaults."""
        assert manager._sessions == {}
        assert manager._reconnect_tokens == {}
        assert manager._heartbeat_task is None
        assert manager._cleanup_task is None
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_start_creates_background_tasks(self, manager):
        """Test start() creates background tasks."""
        await manager.start()

        try:
            assert manager._running is True
            assert manager._heartbeat_task is not None
            assert manager._cleanup_task is not None
            assert not manager._heartbeat_task.done()
            assert not manager._cleanup_task.done()
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, manager):
        """Test start() is idempotent."""
        await manager.start()

        try:
            heartbeat_task = manager._heartbeat_task
            cleanup_task = manager._cleanup_task

            # Call start again
            await manager.start()

            # Should be same tasks
            assert manager._heartbeat_task is heartbeat_task
            assert manager._cleanup_task is cleanup_task
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, manager):
        """Test stop() cancels background tasks."""
        await manager.start()
        heartbeat_task = manager._heartbeat_task
        cleanup_task = manager._cleanup_task

        await manager.stop()

        assert manager._running is False
        assert heartbeat_task.cancelled() or heartbeat_task.done()
        assert cleanup_task.cancelled() or cleanup_task.done()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, manager):
        """Test stop() handles case where start was never called."""
        # Should not raise
        await manager.stop()
        assert manager._running is False


# ===========================================================================
# Session Management Tests
# ===========================================================================


class TestSessionManagement:
    """Tests for session creation, retrieval, and termination."""

    @pytest.mark.asyncio
    async def test_create_session_basic(self, manager):
        """Test basic session creation."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            session = await manager.create_session(
                user_id="user_123",
                debate_id="debate_456",
            )

        assert session.session_id.startswith("voice_")
        assert session.user_id == "user_123"
        assert session.debate_id == "debate_456"
        assert session.is_persistent is True
        assert session.state == "active"
        assert session.session_id in manager._sessions

    @pytest.mark.asyncio
    async def test_create_session_non_persistent(self, manager):
        """Test creating non-persistent session."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            session = await manager.create_session(
                user_id="user_123",
                persistent=False,
            )

        assert session.is_persistent is False

    @pytest.mark.asyncio
    async def test_create_session_custom_ttl(self, manager):
        """Test creating session with custom TTL."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            session = await manager.create_session(
                user_id="user_123",
                ttl_hours=1,  # 1 hour
            )

        # Should expire in about 1 hour
        expected = time.time() + 3600
        assert abs(session.expires_at - expected) < 1

    @pytest.mark.asyncio
    async def test_create_session_custom_audio_format(self, manager):
        """Test creating session with custom audio format."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            session = await manager.create_session(
                user_id="user_123",
                audio_format="opus",
            )

        assert session.audio_format == "opus"

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(self, manager):
        """Test creating session with metadata."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            session = await manager.create_session(
                user_id="user_123",
                metadata={"device": "mobile", "language": "en"},
            )

        assert session.metadata == {"device": "mobile", "language": "en"}

    @pytest.mark.asyncio
    async def test_get_session_exists(self, manager, sample_session):
        """Test getting existing session."""
        manager._sessions[sample_session.session_id] = sample_session

        with patch.object(manager, "_load_session", new_callable=AsyncMock):
            result = await manager.get_session(sample_session.session_id)

        assert result is sample_session

    @pytest.mark.asyncio
    async def test_get_session_loads_from_store(self, manager):
        """Test get_session loads from persistent store if not in memory."""
        stored_session = PersistentVoiceSession(
            session_id="voice_stored",
            user_id="user_123",
        )

        with patch.object(
            manager, "_load_session", new_callable=AsyncMock, return_value=stored_session
        ):
            result = await manager.get_session("voice_stored")

        assert result is stored_session
        assert "voice_stored" in manager._sessions

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, manager):
        """Test getting non-existent session."""
        with patch.object(manager, "_load_session", new_callable=AsyncMock, return_value=None):
            result = await manager.get_session("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, manager):
        """Test getting all sessions for a user."""
        session1 = PersistentVoiceSession(session_id="voice_1", user_id="user_123")
        session2 = PersistentVoiceSession(session_id="voice_2", user_id="user_456")
        session3 = PersistentVoiceSession(session_id="voice_3", user_id="user_123")

        manager._sessions = {
            "voice_1": session1,
            "voice_2": session2,
            "voice_3": session3,
        }

        sessions = await manager.get_user_sessions("user_123")

        assert len(sessions) == 2
        assert session1 in sessions
        assert session3 in sessions
        assert session2 not in sessions

    @pytest.mark.asyncio
    async def test_get_user_sessions_empty(self, manager):
        """Test getting sessions for user with no sessions."""
        sessions = await manager.get_user_sessions("nonexistent_user")
        assert sessions == []

    @pytest.mark.asyncio
    async def test_terminate_session_exists(self, manager, sample_session):
        """Test terminating existing session."""
        manager._sessions[sample_session.session_id] = sample_session
        sample_session.reconnect_token = "token_123"
        manager._reconnect_tokens["token_123"] = sample_session.session_id

        with patch.object(manager, "_remove_session", new_callable=AsyncMock):
            result = await manager.terminate_session(sample_session.session_id)

        assert result is True
        assert sample_session.session_id not in manager._sessions
        assert sample_session.state == "expired"
        assert "token_123" not in manager._reconnect_tokens

    @pytest.mark.asyncio
    async def test_terminate_session_not_exists(self, manager):
        """Test terminating non-existent session."""
        with patch.object(manager, "_remove_session", new_callable=AsyncMock):
            result = await manager.terminate_session("nonexistent")

        assert result is False


# ===========================================================================
# Heartbeat Tests
# ===========================================================================


class TestHeartbeat:
    """Tests for heartbeat handling and timeout detection."""

    @pytest.mark.asyncio
    async def test_heartbeat_success(self, manager, sample_session):
        """Test successful heartbeat updates timestamp."""
        manager._sessions[sample_session.session_id] = sample_session
        original_heartbeat = sample_session.last_heartbeat

        with patch.object(manager, "_load_session", new_callable=AsyncMock):
            time.sleep(0.01)
            result = await manager.heartbeat(sample_session.session_id)

        assert result is True
        assert sample_session.last_heartbeat > original_heartbeat

    @pytest.mark.asyncio
    async def test_heartbeat_session_not_found(self, manager):
        """Test heartbeat for non-existent session."""
        with patch.object(manager, "_load_session", new_callable=AsyncMock, return_value=None):
            result = await manager.heartbeat("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_detects_timeout(self, manager, sample_session):
        """Test heartbeat monitor detects timed out sessions."""
        manager._running = True
        manager._sessions[sample_session.session_id] = sample_session

        # Set heartbeat to way in the past
        sample_session.last_heartbeat = time.time() - VOICE_HEARTBEAT_TIMEOUT_SECONDS - 10

        callback_called = []

        def timeout_callback(session):
            callback_called.append(session.session_id)

        manager._on_heartbeat_timeout = timeout_callback

        call_count = [0]

        async def mock_sleep_func(seconds):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()
            # First call returns normally to allow loop body to execute

        mock_disconnect = AsyncMock()

        with patch.object(manager, "handle_disconnect", mock_disconnect):
            with patch("asyncio.sleep", mock_sleep_func):
                try:
                    await manager._heartbeat_monitor()
                except asyncio.CancelledError:
                    pass

            assert sample_session.session_id in callback_called
            mock_disconnect.assert_called_once_with(sample_session)

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_ignores_disconnected(self, manager, sample_session):
        """Test heartbeat monitor ignores already disconnected sessions."""
        manager._running = True
        sample_session.state = "disconnected"
        sample_session.last_heartbeat = time.time() - VOICE_HEARTBEAT_TIMEOUT_SECONDS - 10
        manager._sessions[sample_session.session_id] = sample_session

        call_count = [0]

        async def mock_sleep_func(seconds):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch.object(manager, "handle_disconnect", new_callable=AsyncMock) as mock_disconnect:
            with patch("asyncio.sleep", mock_sleep_func):
                try:
                    await manager._heartbeat_monitor()
                except asyncio.CancelledError:
                    pass

        # Should not be called since session is already disconnected
        mock_disconnect.assert_not_called()


# ===========================================================================
# Disconnect and Reconnect Tests
# ===========================================================================


class TestDisconnectAndReconnect:
    """Tests for disconnect handling and reconnection."""

    @pytest.mark.asyncio
    async def test_handle_disconnect_persistent(self, manager, sample_session):
        """Test disconnect generates reconnect token for persistent session."""
        manager._sessions[sample_session.session_id] = sample_session

        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            token = await manager.handle_disconnect(sample_session)

        assert token is not None
        assert sample_session.state == "disconnected"
        assert sample_session.reconnect_token == token
        assert sample_session.reconnect_expires_at is not None
        assert token in manager._reconnect_tokens
        assert manager._reconnect_tokens[token] == sample_session.session_id

    @pytest.mark.asyncio
    async def test_handle_disconnect_non_persistent(self, manager):
        """Test disconnect expires non-persistent session immediately."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            is_persistent=False,
        )
        manager._sessions[session.session_id] = session

        token = await manager.handle_disconnect(session)

        assert token is None
        assert session.state == "expired"

    @pytest.mark.asyncio
    async def test_reconnect_success(self, manager, sample_session):
        """Test successful reconnection."""
        sample_session.state = "disconnected"
        sample_session.reconnect_token = "token_abc"
        sample_session.reconnect_expires_at = time.time() + 300

        manager._sessions[sample_session.session_id] = sample_session
        manager._reconnect_tokens["token_abc"] = sample_session.session_id

        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            with patch.object(manager, "_load_session", new_callable=AsyncMock):
                result = await manager.reconnect("token_abc")

        assert result is sample_session
        assert sample_session.state == "active"
        assert sample_session.reconnect_token is None
        assert sample_session.reconnect_expires_at is None
        assert "token_abc" not in manager._reconnect_tokens

    @pytest.mark.asyncio
    async def test_reconnect_invalid_token(self, manager):
        """Test reconnect with invalid token."""
        result = await manager.reconnect("invalid_token")
        assert result is None

    @pytest.mark.asyncio
    async def test_reconnect_session_not_found(self, manager):
        """Test reconnect when session is gone."""
        manager._reconnect_tokens["token_abc"] = "nonexistent_session"

        with patch.object(manager, "_load_session", new_callable=AsyncMock, return_value=None):
            result = await manager.reconnect("token_abc")

        assert result is None

    @pytest.mark.asyncio
    async def test_reconnect_window_expired(self, manager, sample_session):
        """Test reconnect when window has expired."""
        sample_session.state = "disconnected"
        sample_session.reconnect_token = "token_abc"
        sample_session.reconnect_expires_at = time.time() - 1  # Expired

        manager._sessions[sample_session.session_id] = sample_session
        manager._reconnect_tokens["token_abc"] = sample_session.session_id

        with patch.object(manager, "_load_session", new_callable=AsyncMock):
            result = await manager.reconnect("token_abc")

        assert result is None


# ===========================================================================
# Cleanup Tests
# ===========================================================================


class TestCleanup:
    """Tests for session cleanup and expiry."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, manager):
        """Test cleanup removes expired sessions."""
        manager._running = True

        expired_session = PersistentVoiceSession(
            session_id="voice_expired",
            user_id="user_123",
            expires_at=time.time() - 100,  # Expired
        )
        active_session = PersistentVoiceSession(
            session_id="voice_active",
            user_id="user_456",
            expires_at=time.time() + 3600,
        )

        manager._sessions = {
            "voice_expired": expired_session,
            "voice_active": active_session,
        }

        callback_called = []

        def expiry_callback(session):
            callback_called.append(session.session_id)

        manager._on_session_expired = expiry_callback

        call_count = [0]

        async def mock_sleep_func(seconds):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", mock_sleep_func):
            try:
                await manager._cleanup_expired()
            except asyncio.CancelledError:
                pass

        assert "voice_expired" not in manager._sessions
        assert "voice_active" in manager._sessions
        assert "voice_expired" in callback_called
        assert expired_session.state == "expired"

    @pytest.mark.asyncio
    async def test_cleanup_reconnect_expired_sessions(self, manager):
        """Test cleanup removes sessions with expired reconnect window."""
        manager._running = True

        reconnect_expired = PersistentVoiceSession(
            session_id="voice_reconnect_expired",
            user_id="user_123",
            state="disconnected",
            reconnect_token="token_abc",
            reconnect_expires_at=time.time() - 100,  # Expired
            expires_at=time.time() + 3600,  # Session not expired
        )

        manager._sessions = {"voice_reconnect_expired": reconnect_expired}
        manager._reconnect_tokens = {"token_abc": "voice_reconnect_expired"}

        call_count = [0]

        async def mock_sleep_func(seconds):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", mock_sleep_func):
            try:
                await manager._cleanup_expired()
            except asyncio.CancelledError:
                pass

        assert "voice_reconnect_expired" not in manager._sessions
        assert "token_abc" not in manager._reconnect_tokens
        assert reconnect_expired.state == "expired"

    @pytest.mark.asyncio
    async def test_cleanup_stale_reconnect_tokens(self, manager):
        """Test cleanup removes stale reconnect tokens."""
        manager._running = True

        # Token pointing to non-existent session
        manager._reconnect_tokens = {"stale_token": "nonexistent_session"}
        manager._sessions = {}

        call_count = [0]

        async def mock_sleep_func(seconds):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", mock_sleep_func):
            try:
                await manager._cleanup_expired()
            except asyncio.CancelledError:
                pass

        assert "stale_token" not in manager._reconnect_tokens


# ===========================================================================
# Persistence Tests
# ===========================================================================


class TestPersistence:
    """Tests for session state persistence."""

    @pytest.mark.asyncio
    async def test_persist_session_success(self, manager, sample_session):
        """Test persisting session to store."""
        import sys

        mock_voice_session_class = MagicMock()
        mock_store = MagicMock()
        mock_get_store = MagicMock(return_value=mock_store)

        mock_module = MagicMock()
        mock_module.VoiceSession = mock_voice_session_class
        mock_module.get_session_store = mock_get_store

        with patch.dict(sys.modules, {"aragora.server.session_store": mock_module}):
            await manager._persist_session(sample_session)

        mock_store.set_voice_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_session_import_error(self, manager, sample_session):
        """Test persist handles ImportError gracefully."""
        import sys

        # Remove the module from cache to trigger ImportError
        with patch.dict(sys.modules, {"aragora.server.session_store": None}):
            # Should not raise
            await manager._persist_session(sample_session)

    @pytest.mark.asyncio
    async def test_persist_session_store_error(self, manager, sample_session):
        """Test persist handles store errors gracefully."""
        import sys

        mock_store = MagicMock()
        mock_store.set_voice_session.side_effect = ValueError("Store error")
        mock_get_store = MagicMock(return_value=mock_store)

        mock_module = MagicMock()
        mock_module.VoiceSession = MagicMock()
        mock_module.get_session_store = mock_get_store

        with patch.dict(sys.modules, {"aragora.server.session_store": mock_module}):
            # Should not raise
            await manager._persist_session(sample_session)

    @pytest.mark.asyncio
    async def test_load_session_success(self, manager):
        """Test loading session from store."""
        import sys

        mock_voice_session = MagicMock()
        mock_voice_session.session_id = "voice_stored"
        mock_voice_session.user_id = "user_123"
        mock_voice_session.debate_id = None
        mock_voice_session.is_persistent = True
        mock_voice_session.audio_format = "pcm_16khz"
        mock_voice_session.reconnect_token = None
        mock_voice_session.reconnect_expires_at = None
        mock_voice_session.last_heartbeat = time.time()
        mock_voice_session.created_at = time.time()
        mock_voice_session.expires_at = time.time() + 3600
        mock_voice_session.metadata = {}

        mock_store = MagicMock()
        mock_store.get_voice_session.return_value = mock_voice_session
        mock_get_store = MagicMock(return_value=mock_store)

        mock_module = MagicMock()
        mock_module.get_session_store = mock_get_store

        with patch.dict(sys.modules, {"aragora.server.session_store": mock_module}):
            result = await manager._load_session("voice_stored")

        assert result is not None
        assert result.session_id == "voice_stored"
        assert result.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_load_session_not_found(self, manager):
        """Test loading non-existent session."""
        import sys

        mock_store = MagicMock()
        mock_store.get_voice_session.return_value = None
        mock_get_store = MagicMock(return_value=mock_store)

        mock_module = MagicMock()
        mock_module.get_session_store = mock_get_store

        with patch.dict(sys.modules, {"aragora.server.session_store": mock_module}):
            result = await manager._load_session("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_load_session_import_error(self, manager):
        """Test load handles ImportError gracefully."""
        import sys

        with patch.dict(sys.modules, {"aragora.server.session_store": None}):
            result = await manager._load_session("voice_test")

        assert result is None

    @pytest.mark.asyncio
    async def test_load_session_with_reconnect_token(self, manager):
        """Test loading session with reconnect token sets state to disconnected."""
        import sys

        mock_voice_session = MagicMock()
        mock_voice_session.session_id = "voice_stored"
        mock_voice_session.user_id = "user_123"
        mock_voice_session.debate_id = None
        mock_voice_session.is_persistent = True
        mock_voice_session.audio_format = "pcm_16khz"
        mock_voice_session.reconnect_token = "token_abc"  # Has reconnect token
        mock_voice_session.reconnect_expires_at = time.time() + 300
        mock_voice_session.last_heartbeat = time.time()
        mock_voice_session.created_at = time.time()
        mock_voice_session.expires_at = time.time() + 3600
        mock_voice_session.metadata = {}

        mock_store = MagicMock()
        mock_store.get_voice_session.return_value = mock_voice_session
        mock_get_store = MagicMock(return_value=mock_store)

        mock_module = MagicMock()
        mock_module.get_session_store = mock_get_store

        with patch.dict(sys.modules, {"aragora.server.session_store": mock_module}):
            result = await manager._load_session("voice_stored")

        assert result.state == "disconnected"

    @pytest.mark.asyncio
    async def test_remove_session_success(self, manager):
        """Test removing session from store."""
        import sys

        mock_store = MagicMock()
        mock_get_store = MagicMock(return_value=mock_store)

        mock_module = MagicMock()
        mock_module.get_session_store = mock_get_store

        with patch.dict(sys.modules, {"aragora.server.session_store": mock_module}):
            await manager._remove_session("voice_test")

        mock_store.delete_voice_session.assert_called_once_with("voice_test")

    @pytest.mark.asyncio
    async def test_remove_session_import_error(self, manager):
        """Test remove handles ImportError gracefully."""
        import sys

        with patch.dict(sys.modules, {"aragora.server.session_store": None}):
            # Should not raise
            await manager._remove_session("voice_test")


# ===========================================================================
# Callback Tests
# ===========================================================================


class TestCallbacks:
    """Tests for session event callbacks."""

    def test_on_session_expired_callback_setter(self, manager):
        """Test setting session expired callback."""
        callback = MagicMock()
        manager.on_session_expired(callback)
        assert manager._on_session_expired is callback

    def test_on_heartbeat_timeout_callback_setter(self, manager):
        """Test setting heartbeat timeout callback."""
        callback = MagicMock()
        manager.on_heartbeat_timeout(callback)
        assert manager._on_heartbeat_timeout is callback


# ===========================================================================
# Statistics Tests
# ===========================================================================


class TestStatistics:
    """Tests for manager statistics."""

    def test_get_stats_empty(self, manager):
        """Test get_stats with no sessions."""
        stats = manager.get_stats()

        assert stats["total_sessions"] == 0
        assert stats["active_sessions"] == 0
        assert stats["disconnected_sessions"] == 0
        assert stats["pending_reconnects"] == 0
        assert stats["running"] is False

    def test_get_stats_with_sessions(self, manager):
        """Test get_stats with various sessions."""
        active_session = PersistentVoiceSession(
            session_id="voice_1", user_id="user_1", state="active"
        )
        disconnected_session = PersistentVoiceSession(
            session_id="voice_2", user_id="user_2", state="disconnected"
        )
        expired_session = PersistentVoiceSession(
            session_id="voice_3", user_id="user_3", state="expired"
        )

        manager._sessions = {
            "voice_1": active_session,
            "voice_2": disconnected_session,
            "voice_3": expired_session,
        }
        manager._reconnect_tokens = {"token_1": "voice_2", "token_2": "voice_3"}
        manager._running = True

        stats = manager.get_stats()

        assert stats["total_sessions"] == 3
        assert stats["active_sessions"] == 1
        assert stats["disconnected_sessions"] == 1
        assert stats["pending_reconnects"] == 2
        assert stats["running"] is True


# ===========================================================================
# Global Instance Tests
# ===========================================================================


class TestGlobalInstance:
    """Tests for global manager instance."""

    def test_get_persistent_voice_manager_singleton(self):
        """Test get_persistent_voice_manager returns singleton."""
        import aragora.server.stream.persistent_voice as pv

        # Reset global
        original = pv._manager
        pv._manager = None

        try:
            manager1 = get_persistent_voice_manager()
            manager2 = get_persistent_voice_manager()

            assert manager1 is manager2
        finally:
            pv._manager = original


# ===========================================================================
# Concurrent Session Tests
# ===========================================================================


class TestConcurrentSessions:
    """Tests for concurrent session handling."""

    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self, manager):
        """Test creating multiple sessions concurrently."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            sessions = await asyncio.gather(
                *[manager.create_session(user_id=f"user_{i}") for i in range(10)]
            )

        assert len(sessions) == 10
        assert len(manager._sessions) == 10

        # All session IDs should be unique
        session_ids = [s.session_id for s in sessions]
        assert len(set(session_ids)) == 10

    @pytest.mark.asyncio
    async def test_concurrent_heartbeats(self, manager):
        """Test handling concurrent heartbeats."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            with patch.object(manager, "_load_session", new_callable=AsyncMock):
                sessions = await asyncio.gather(
                    *[manager.create_session(user_id=f"user_{i}") for i in range(5)]
                )

        # Send heartbeats concurrently
        results = await asyncio.gather(*[manager.heartbeat(s.session_id) for s in sessions])

        assert all(results)

    @pytest.mark.asyncio
    async def test_concurrent_reconnect_attempts(self, manager):
        """Test handling concurrent reconnect attempts with same token."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="disconnected",
            is_persistent=True,
            reconnect_token="token_abc",
            reconnect_expires_at=time.time() + 300,
        )
        manager._sessions[session.session_id] = session
        manager._reconnect_tokens["token_abc"] = session.session_id

        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            with patch.object(manager, "_load_session", new_callable=AsyncMock):
                # Only one should succeed
                results = await asyncio.gather(*[manager.reconnect("token_abc") for _ in range(3)])

        # At least one should get the session, others get None (token removed)
        successful = [r for r in results if r is not None]
        assert len(successful) >= 1


# ===========================================================================
# Error Recovery Tests
# ===========================================================================


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_error_handling(self, manager):
        """Test heartbeat monitor handles errors gracefully."""
        manager._running = True
        manager._sessions = {"voice_1": MagicMock()}

        iteration_count = [0]

        async def controlled_sleep(seconds):
            iteration_count[0] += 1
            if iteration_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", controlled_sleep):
            with patch.object(
                manager,
                "handle_disconnect",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Test error"),
            ):
                try:
                    await manager._heartbeat_monitor()
                except asyncio.CancelledError:
                    pass

        # Should have completed at least one iteration despite error
        assert iteration_count[0] >= 1

    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self, manager):
        """Test cleanup handles errors gracefully."""
        manager._running = True

        # Create a session that will cause issues
        bad_session = MagicMock()
        bad_session.state = "active"
        bad_session.expires_at = time.time() - 100  # Expired
        manager._sessions = {"voice_bad": bad_session}

        iteration_count = [0]

        async def controlled_sleep(seconds):
            iteration_count[0] += 1
            if iteration_count[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", controlled_sleep):
            try:
                await manager._cleanup_expired()
            except asyncio.CancelledError:
                pass

        # Should have handled the error gracefully
        assert iteration_count[0] >= 1


# ===========================================================================
# State Persistence Across Reconnections Tests
# ===========================================================================


class TestStatePersistenceAcrossReconnections:
    """Tests for state persistence across reconnections."""

    @pytest.mark.asyncio
    async def test_audio_buffer_preserved_across_disconnect(self, manager):
        """Test audio buffer is preserved during disconnect."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            session = await manager.create_session(user_id="user_123")

        session.audio_buffer = b"\x00\x01\x02\x03" * 100
        session.transcript_buffer = "Previous transcription"

        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            token = await manager.handle_disconnect(session)

        # Buffers should be preserved
        assert session.audio_buffer == b"\x00\x01\x02\x03" * 100
        assert session.transcript_buffer == "Previous transcription"

    @pytest.mark.asyncio
    async def test_metadata_preserved_across_reconnect(self, manager):
        """Test metadata is preserved across reconnect."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="disconnected",
            is_persistent=True,
            reconnect_token="token_abc",
            reconnect_expires_at=time.time() + 300,
            metadata={"device": "mobile", "custom_data": 123},
        )
        manager._sessions[session.session_id] = session
        manager._reconnect_tokens["token_abc"] = session.session_id

        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            with patch.object(manager, "_load_session", new_callable=AsyncMock):
                reconnected = await manager.reconnect("token_abc")

        assert reconnected.metadata == {"device": "mobile", "custom_data": 123}

    @pytest.mark.asyncio
    async def test_session_expiry_not_extended_on_reconnect(self, manager):
        """Test session expiry is NOT automatically extended on reconnect."""
        original_expiry = time.time() + 1800  # 30 minutes remaining

        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            state="disconnected",
            is_persistent=True,
            reconnect_token="token_abc",
            reconnect_expires_at=time.time() + 300,
            expires_at=original_expiry,
        )
        manager._sessions[session.session_id] = session
        manager._reconnect_tokens["token_abc"] = session.session_id

        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            with patch.object(manager, "_load_session", new_callable=AsyncMock):
                reconnected = await manager.reconnect("token_abc")

        # Expiry should remain the same (not extended)
        assert reconnected.expires_at == original_expiry


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_create_session_empty_user_id(self, manager):
        """Test creating session with empty user ID."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            session = await manager.create_session(user_id="")

        assert session.user_id == ""

    @pytest.mark.asyncio
    async def test_terminate_same_session_twice(self, manager, sample_session):
        """Test terminating same session twice."""
        manager._sessions[sample_session.session_id] = sample_session

        with patch.object(manager, "_remove_session", new_callable=AsyncMock):
            result1 = await manager.terminate_session(sample_session.session_id)
            result2 = await manager.terminate_session(sample_session.session_id)

        assert result1 is True
        assert result2 is False

    @pytest.mark.asyncio
    async def test_heartbeat_after_session_expired(self, manager):
        """Test heartbeat for expired session still works."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            expires_at=time.time() - 100,  # Expired
        )
        manager._sessions[session.session_id] = session

        with patch.object(manager, "_load_session", new_callable=AsyncMock):
            result = await manager.heartbeat(session.session_id)

        # Heartbeat still works even for expired session
        assert result is True

    def test_session_zero_ttl(self):
        """Test session with zero TTL."""
        now = time.time()
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
            expires_at=now,
        )

        # Should be expired immediately
        assert session.time_until_expiry == 0
        assert session.is_active is False

    @pytest.mark.asyncio
    async def test_create_session_very_long_ttl(self, manager):
        """Test creating session with very long TTL."""
        with patch.object(manager, "_persist_session", new_callable=AsyncMock):
            session = await manager.create_session(
                user_id="user_123",
                ttl_hours=8760,  # 1 year
            )

        # Should expire in about 1 year
        expected = time.time() + (8760 * 3600)
        assert abs(session.expires_at - expected) < 1

    def test_session_state_transitions(self):
        """Test all valid session state transitions."""
        session = PersistentVoiceSession(
            session_id="voice_test",
            user_id="user_123",
        )

        # active -> disconnected
        assert session.state == "active"
        session.state = "disconnected"
        assert session.state == "disconnected"

        # disconnected -> reconnecting
        session.state = "reconnecting"
        assert session.state == "reconnecting"

        # reconnecting -> active
        session.state = "active"
        assert session.state == "active"

        # active -> expired
        session.state = "expired"
        assert session.state == "expired"
