"""Tests for the SharingHandler class."""

import json
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict


# -----------------------------------------------------------------------------
# DebateVisibility Tests
# -----------------------------------------------------------------------------


class TestDebateVisibility:
    """Test DebateVisibility enum."""

    def test_visibility_values(self):
        """Visibility enum has expected values."""
        from aragora.server.handlers.sharing import DebateVisibility

        assert DebateVisibility.PRIVATE.value == "private"
        assert DebateVisibility.TEAM.value == "team"
        assert DebateVisibility.PUBLIC.value == "public"

    def test_visibility_from_string(self):
        """Can create visibility from string."""
        from aragora.server.handlers.sharing import DebateVisibility

        assert DebateVisibility("private") == DebateVisibility.PRIVATE
        assert DebateVisibility("team") == DebateVisibility.TEAM
        assert DebateVisibility("public") == DebateVisibility.PUBLIC

    def test_invalid_visibility_raises(self):
        """Invalid visibility string raises ValueError."""
        from aragora.server.handlers.sharing import DebateVisibility

        with pytest.raises(ValueError):
            DebateVisibility("invalid")


# -----------------------------------------------------------------------------
# ShareSettings Tests
# -----------------------------------------------------------------------------


class TestShareSettings:
    """Test ShareSettings dataclass."""

    def test_default_values(self):
        """ShareSettings has correct defaults."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(debate_id="test-123")

        assert settings.debate_id == "test-123"
        assert settings.visibility == DebateVisibility.PRIVATE
        assert settings.share_token is None
        assert settings.expires_at is None
        assert settings.allow_comments is False
        assert settings.allow_forking is False
        assert settings.view_count == 0
        assert settings.owner_id is None
        assert settings.org_id is None

    def test_to_dict(self):
        """to_dict returns serializable dictionary."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            visibility=DebateVisibility.PUBLIC,
            share_token="abc123",
            allow_comments=True,
            view_count=42,
        )

        result = settings.to_dict()

        assert result["debate_id"] == "debate-1"
        assert result["visibility"] == "public"
        assert result["share_token"] == "abc123"
        assert result["share_url"] == "/api/shared/abc123"
        assert result["allow_comments"] is True
        assert result["view_count"] == 42
        assert result["is_expired"] is False

    def test_to_dict_no_token(self):
        """to_dict returns None share_url when no token."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(debate_id="debate-1")
        result = settings.to_dict()

        assert result["share_token"] is None
        assert result["share_url"] is None

    def test_from_dict(self):
        """from_dict creates ShareSettings from dictionary."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        data = {
            "debate_id": "debate-1",
            "visibility": "team",
            "share_token": "token123",
            "allow_comments": True,
            "allow_forking": True,
            "view_count": 10,
            "owner_id": "user-1",
            "org_id": "org-1",
        }

        settings = ShareSettings.from_dict(data)

        assert settings.debate_id == "debate-1"
        assert settings.visibility == DebateVisibility.TEAM
        assert settings.share_token == "token123"
        assert settings.allow_comments is True
        assert settings.allow_forking is True
        assert settings.view_count == 10
        assert settings.owner_id == "user-1"
        assert settings.org_id == "org-1"

    def test_from_dict_defaults(self):
        """from_dict uses defaults for missing fields."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        data = {"debate_id": "debate-1"}
        settings = ShareSettings.from_dict(data)

        assert settings.visibility == DebateVisibility.PRIVATE
        assert settings.share_token is None
        assert settings.allow_comments is False

    def test_is_expired_no_expiration(self):
        """is_expired returns False when no expiration set."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(debate_id="test", expires_at=None)
        assert settings.is_expired is False

    def test_is_expired_future(self):
        """is_expired returns False for future expiration."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(
            debate_id="test",
            expires_at=time.time() + 3600  # 1 hour from now
        )
        assert settings.is_expired is False

    def test_is_expired_past(self):
        """is_expired returns True for past expiration."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(
            debate_id="test",
            expires_at=time.time() - 3600  # 1 hour ago
        )
        assert settings.is_expired is True


# -----------------------------------------------------------------------------
# ShareStore Tests
# -----------------------------------------------------------------------------


class TestShareStore:
    """Test ShareStore in-memory storage."""

    @pytest.fixture
    def store(self):
        """Create fresh ShareStore instance."""
        from aragora.server.handlers.sharing import ShareStore
        return ShareStore()

    @pytest.fixture
    def sample_settings(self):
        """Create sample ShareSettings."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility
        return ShareSettings(
            debate_id="debate-1",
            visibility=DebateVisibility.PUBLIC,
            share_token="token123",
            owner_id="user-1",
        )

    def test_get_nonexistent(self, store):
        """get returns None for nonexistent debate."""
        assert store.get("nonexistent") is None

    def test_save_and_get(self, store, sample_settings):
        """Can save and retrieve settings."""
        store.save(sample_settings)

        retrieved = store.get("debate-1")
        assert retrieved is not None
        assert retrieved.debate_id == "debate-1"
        assert retrieved.share_token == "token123"

    def test_get_by_token(self, store, sample_settings):
        """Can retrieve settings by share token."""
        store.save(sample_settings)

        retrieved = store.get_by_token("token123")
        assert retrieved is not None
        assert retrieved.debate_id == "debate-1"

    def test_get_by_token_nonexistent(self, store):
        """get_by_token returns None for nonexistent token."""
        assert store.get_by_token("nonexistent") is None

    def test_delete(self, store, sample_settings):
        """Can delete settings."""
        store.save(sample_settings)
        assert store.get("debate-1") is not None

        result = store.delete("debate-1")

        assert result is True
        assert store.get("debate-1") is None
        assert store.get_by_token("token123") is None

    def test_delete_nonexistent(self, store):
        """delete returns False for nonexistent debate."""
        result = store.delete("nonexistent")
        assert result is False

    def test_revoke_token(self, store, sample_settings):
        """Can revoke share token."""
        store.save(sample_settings)

        result = store.revoke_token("debate-1")

        assert result is True
        settings = store.get("debate-1")
        assert settings.share_token is None
        assert store.get_by_token("token123") is None

    def test_revoke_token_no_token(self, store):
        """revoke_token returns False when no token exists."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(debate_id="debate-1")
        store.save(settings)

        result = store.revoke_token("debate-1")
        assert result is False

    def test_revoke_token_nonexistent(self, store):
        """revoke_token returns False for nonexistent debate."""
        result = store.revoke_token("nonexistent")
        assert result is False

    def test_increment_view_count(self, store, sample_settings):
        """Can increment view count."""
        sample_settings.view_count = 5
        store.save(sample_settings)

        store.increment_view_count("debate-1")

        settings = store.get("debate-1")
        assert settings.view_count == 6

    def test_increment_view_count_nonexistent(self, store):
        """increment_view_count does nothing for nonexistent debate."""
        # Should not raise
        store.increment_view_count("nonexistent")


# -----------------------------------------------------------------------------
# get_share_store Tests
# -----------------------------------------------------------------------------


class TestGetShareStore:
    """Test get_share_store factory function."""

    def test_returns_store_instance(self):
        """get_share_store returns a store instance."""
        from aragora.server.handlers import sharing

        # Reset global
        sharing._share_store = None

        with patch.object(sharing, "_share_store", None):
            # Mock the import to fail, triggering fallback
            with patch.dict("sys.modules", {"aragora.storage.share_store": None}):
                store = sharing.get_share_store()
                assert store is not None

    def test_singleton_behavior(self):
        """get_share_store returns same instance on subsequent calls."""
        from aragora.server.handlers import sharing
        from aragora.server.handlers.sharing import ShareStore

        # Reset and set a known store
        test_store = ShareStore()
        sharing._share_store = test_store

        store1 = sharing.get_share_store()
        store2 = sharing.get_share_store()

        assert store1 is store2
        assert store1 is test_store


# -----------------------------------------------------------------------------
# SharingHandler Routing Tests
# -----------------------------------------------------------------------------


class TestSharingHandlerRouting:
    """Test route matching for SharingHandler."""

    @pytest.fixture
    def handler(self):
        """Create SharingHandler instance."""
        from aragora.server.handlers.sharing import SharingHandler
        return SharingHandler()

    def test_routes_defined(self, handler):
        """Handler has expected routes."""
        assert "/api/debates/*/share" in handler.ROUTES
        assert "/api/debates/*/share/revoke" in handler.ROUTES
        assert "/api/shared/*" in handler.ROUTES

    def test_auth_required_endpoints(self, handler):
        """Auth required endpoints defined."""
        assert "/share" in handler.AUTH_REQUIRED_ENDPOINTS
        assert "/share/revoke" in handler.AUTH_REQUIRED_ENDPOINTS


# -----------------------------------------------------------------------------
# SharingHandler Extract Debate ID Tests
# -----------------------------------------------------------------------------


class TestExtractDebateId:
    """Test debate ID extraction from paths."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler
        return SharingHandler()

    def test_extract_from_share_path(self, handler):
        """Extracts ID from /api/debates/{id}/share."""
        debate_id, err = handler._extract_debate_id("/api/debates/abc123/share")
        assert debate_id == "abc123"
        assert err is None

    def test_extract_from_revoke_path(self, handler):
        """Extracts ID from /api/debates/{id}/share/revoke."""
        debate_id, err = handler._extract_debate_id("/api/debates/xyz789/share/revoke")
        assert debate_id == "xyz789"
        assert err is None

    def test_extract_with_uuid(self, handler):
        """Extracts UUID-style ID."""
        debate_id, err = handler._extract_debate_id(
            "/api/debates/550e8400-e29b-41d4-a716-446655440000/share"
        )
        assert debate_id == "550e8400-e29b-41d4-a716-446655440000"
        assert err is None

    def test_extract_invalid_path(self, handler):
        """Returns error for invalid path."""
        debate_id, err = handler._extract_debate_id("/api/other/path")
        assert debate_id is None
        assert err is not None


# -----------------------------------------------------------------------------
# SharingHandler GET Tests
# -----------------------------------------------------------------------------


class TestSharingHandlerGet:
    """Test GET request handling."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler, ShareStore
        handler = SharingHandler()
        handler._store = ShareStore()
        return handler

    @pytest.fixture
    def mock_user(self):
        user = Mock()
        user.id = "user-1"
        user.org_id = "org-1"
        return user

    def test_get_share_settings_requires_auth(self, handler):
        """GET /api/debates/{id}/share requires authentication."""
        mock_http = Mock()

        with patch.object(handler, "get_current_user", return_value=None):
            result = handler._get_share_settings("debate-1", mock_http)

        assert result.status_code == 401

    def test_get_share_settings_creates_default(self, handler, mock_user):
        """Creates default settings if none exist."""
        mock_http = Mock()

        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._get_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "debate-1"
        assert data["visibility"] == "private"

    def test_get_share_settings_returns_existing(self, handler, mock_user):
        """Returns existing settings."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            visibility=DebateVisibility.PUBLIC,
            share_token="token123",
            owner_id="user-1",
            view_count=42,
        )
        handler._store.save(settings)

        mock_http = Mock()
        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._get_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["visibility"] == "public"
        assert data["view_count"] == 42

    def test_get_share_settings_forbidden_for_other_owner(self, handler, mock_user):
        """Returns 403 if user doesn't own the debate."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(
            debate_id="debate-1",
            owner_id="other-user",  # Different owner
            org_id="other-org",
        )
        handler._store.save(settings)

        mock_http = Mock()
        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._get_share_settings("debate-1", mock_http)

        assert result.status_code == 403


# -----------------------------------------------------------------------------
# SharingHandler Shared Debate Access Tests
# -----------------------------------------------------------------------------


class TestGetSharedDebate:
    """Test shared debate access via token."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler, ShareStore
        handler = SharingHandler()
        handler._store = ShareStore()
        return handler

    def test_shared_debate_not_found(self, handler):
        """Returns 404 for nonexistent token."""
        result = handler._get_shared_debate("nonexistent", {})

        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data["error"].lower()

    def test_shared_debate_expired(self, handler):
        """Returns 410 for expired share link."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            visibility=DebateVisibility.PUBLIC,
            share_token="token123",
            expires_at=time.time() - 3600,  # Expired
        )
        handler._store.save(settings)

        result = handler._get_shared_debate("token123", {})

        assert result.status_code == 410
        data = json.loads(result.body)
        assert "expired" in data["error"].lower()

    def test_shared_debate_not_public(self, handler):
        """Returns 403 for non-public debate."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            visibility=DebateVisibility.PRIVATE,
            share_token="token123",
        )
        handler._store.save(settings)

        result = handler._get_shared_debate("token123", {})

        assert result.status_code == 403

    def test_shared_debate_success(self, handler):
        """Returns debate data for valid share link."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            visibility=DebateVisibility.PUBLIC,
            share_token="token123",
            allow_comments=True,
        )
        handler._store.save(settings)

        mock_debate = {"id": "debate-1", "title": "Test Debate"}
        with patch.object(handler, "_get_debate_data", return_value=mock_debate):
            result = handler._get_shared_debate("token123", {})

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate"]["id"] == "debate-1"
        assert data["sharing"]["allow_comments"] is True

    def test_shared_debate_increments_view_count(self, handler):
        """Accessing shared debate increments view count."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            visibility=DebateVisibility.PUBLIC,
            share_token="token123",
            view_count=5,
        )
        handler._store.save(settings)

        mock_debate = {"id": "debate-1"}
        with patch.object(handler, "_get_debate_data", return_value=mock_debate):
            handler._get_shared_debate("token123", {})

        updated = handler._store.get("debate-1")
        assert updated.view_count == 6


# -----------------------------------------------------------------------------
# SharingHandler POST Tests
# -----------------------------------------------------------------------------


class TestSharingHandlerPost:
    """Test POST request handling."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler, ShareStore
        handler = SharingHandler()
        handler._store = ShareStore()
        return handler

    @pytest.fixture
    def mock_user(self):
        user = Mock()
        user.id = "user-1"
        user.org_id = "org-1"
        return user

    def test_update_requires_auth(self, handler):
        """Update requires authentication."""
        mock_http = Mock()
        mock_http.headers = {"Content-Length": "2"}
        mock_http.rfile.read.return_value = b'{}'

        with patch.object(handler, "get_current_user", return_value=None):
            result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 401

    def test_update_requires_json_body(self, handler, mock_user):
        """Update requires valid JSON body."""
        mock_http = Mock()

        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=None):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 400

    def test_update_visibility_to_public(self, handler, mock_user):
        """Can update visibility to public."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(debate_id="debate-1", owner_id="user-1")
        handler._store.save(settings)

        mock_http = Mock()
        body = {"visibility": "public"}

        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=body):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["settings"]["visibility"] == "public"
        assert data["settings"]["share_token"] is not None

    def test_update_visibility_to_private_revokes_token(self, handler, mock_user):
        """Updating to private revokes share token."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            owner_id="user-1",
            visibility=DebateVisibility.PUBLIC,
            share_token="token123",
        )
        handler._store.save(settings)

        mock_http = Mock()
        body = {"visibility": "private"}

        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=body):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["settings"]["visibility"] == "private"
        assert data["settings"]["share_token"] is None

    def test_update_invalid_visibility(self, handler, mock_user):
        """Returns 400 for invalid visibility value."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(debate_id="debate-1", owner_id="user-1")
        handler._store.save(settings)

        mock_http = Mock()
        body = {"visibility": "invalid"}

        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=body):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 400

    def test_update_expiration(self, handler, mock_user):
        """Can set expiration time."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(debate_id="debate-1", owner_id="user-1")
        handler._store.save(settings)

        mock_http = Mock()
        body = {"expires_in_hours": 24}

        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=body):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["settings"]["expires_at"] is not None

    def test_update_clear_expiration(self, handler, mock_user):
        """Can clear expiration with zero or negative hours."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(
            debate_id="debate-1",
            owner_id="user-1",
            expires_at=time.time() + 3600,
        )
        handler._store.save(settings)

        mock_http = Mock()
        body = {"expires_in_hours": 0}

        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=body):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["settings"]["expires_at"] is None

    def test_update_allow_comments(self, handler, mock_user):
        """Can update allow_comments setting."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(debate_id="debate-1", owner_id="user-1")
        handler._store.save(settings)

        mock_http = Mock()
        body = {"allow_comments": True}

        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=body):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["settings"]["allow_comments"] is True

    def test_update_forbidden_for_other_owner(self, handler, mock_user):
        """Returns 403 if user doesn't own the debate."""
        from aragora.server.handlers.sharing import ShareSettings

        settings = ShareSettings(debate_id="debate-1", owner_id="other-user")
        handler._store.save(settings)

        mock_http = Mock()
        body = {"visibility": "public"}

        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=body):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 403


# -----------------------------------------------------------------------------
# SharingHandler Revoke Tests
# -----------------------------------------------------------------------------


class TestRevokeShare:
    """Test share revocation."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler, ShareStore
        handler = SharingHandler()
        handler._store = ShareStore()
        return handler

    @pytest.fixture
    def mock_user(self):
        user = Mock()
        user.id = "user-1"
        user.org_id = "org-1"
        return user

    def test_revoke_requires_auth(self, handler):
        """Revoke requires authentication."""
        mock_http = Mock()

        with patch.object(handler, "get_current_user", return_value=None):
            result = handler._revoke_share("debate-1", mock_http)

        assert result.status_code == 401

    def test_revoke_not_found(self, handler, mock_user):
        """Returns 404 when no settings exist."""
        mock_http = Mock()

        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._revoke_share("nonexistent", mock_http)

        assert result.status_code == 404

    def test_revoke_forbidden_for_other_owner(self, handler, mock_user):
        """Returns 403 if user doesn't own the debate."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            owner_id="other-user",
            visibility=DebateVisibility.PUBLIC,
            share_token="token123",
        )
        handler._store.save(settings)

        mock_http = Mock()
        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._revoke_share("debate-1", mock_http)

        assert result.status_code == 403

    def test_revoke_success(self, handler, mock_user):
        """Successfully revokes share links."""
        from aragora.server.handlers.sharing import ShareSettings, DebateVisibility

        settings = ShareSettings(
            debate_id="debate-1",
            owner_id="user-1",
            visibility=DebateVisibility.PUBLIC,
            share_token="token123",
        )
        handler._store.save(settings)

        mock_http = Mock()
        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._revoke_share("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True

        # Verify settings updated
        updated = handler._store.get("debate-1")
        assert updated.visibility == DebateVisibility.PRIVATE
        assert updated.share_token is None


# -----------------------------------------------------------------------------
# SharingHandler Token Generation Tests
# -----------------------------------------------------------------------------


class TestGenerateShareToken:
    """Test share token generation."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler
        return SharingHandler()

    def test_generates_unique_tokens(self, handler):
        """Generates unique tokens for same debate."""
        token1 = handler._generate_share_token("debate-1")
        token2 = handler._generate_share_token("debate-1")

        assert token1 != token2

    def test_token_is_url_safe(self, handler):
        """Token is URL-safe string."""
        token = handler._generate_share_token("debate-1")

        # URL-safe base64 characters
        import re
        assert re.match(r'^[A-Za-z0-9_-]+$', token)

    def test_token_length(self, handler):
        """Token has reasonable length."""
        token = handler._generate_share_token("debate-1")

        # secrets.token_urlsafe(16) produces ~22 chars
        assert 15 <= len(token) <= 30


# -----------------------------------------------------------------------------
# SharingHandler handle() Method Tests
# -----------------------------------------------------------------------------


class TestHandleMethod:
    """Test the main handle() routing method."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler, ShareStore
        handler = SharingHandler()
        handler._store = ShareStore()
        return handler

    def test_handle_shared_debate_route(self, handler):
        """Routes /api/shared/{token} to _get_shared_debate."""
        mock_http = Mock()

        with patch.object(handler, "_get_shared_debate") as mock_get:
            mock_get.return_value = Mock(status_code=200, body="{}")
            handler.handle("/api/shared/token123", {}, mock_http)
            mock_get.assert_called_once_with("token123", {})

    def test_handle_share_settings_route(self, handler):
        """Routes /api/debates/{id}/share to _get_share_settings."""
        mock_http = Mock()

        with patch.object(handler, "_get_share_settings") as mock_get:
            mock_get.return_value = Mock(status_code=200, body="{}")
            handler.handle("/api/debates/debate-1/share", {}, mock_http)
            mock_get.assert_called_once_with("debate-1", mock_http)

    def test_handle_returns_none_for_unknown_route(self, handler):
        """Returns None for unknown routes."""
        mock_http = Mock()

        result = handler.handle("/api/unknown", {}, mock_http)
        assert result is None


# -----------------------------------------------------------------------------
# SharingHandler handle_post() Method Tests
# -----------------------------------------------------------------------------


class TestHandlePostMethod:
    """Test the handle_post() routing method."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler, ShareStore
        handler = SharingHandler()
        handler._store = ShareStore()
        return handler

    def test_handle_post_revoke_route(self, handler):
        """Routes /api/debates/{id}/share/revoke to _revoke_share."""
        mock_http = Mock()

        with patch.object(handler, "_revoke_share") as mock_revoke:
            mock_revoke.return_value = Mock(status_code=200, body="{}")
            handler.handle_post("/api/debates/debate-1/share/revoke", {}, mock_http)
            mock_revoke.assert_called_once_with("debate-1", mock_http)

    def test_handle_post_update_route(self, handler):
        """Routes /api/debates/{id}/share to _update_share_settings."""
        mock_http = Mock()

        with patch.object(handler, "_update_share_settings") as mock_update:
            mock_update.return_value = Mock(status_code=200, body="{}")
            handler.handle_post("/api/debates/debate-1/share", {}, mock_http)
            mock_update.assert_called_once_with("debate-1", mock_http)

    def test_handle_post_returns_none_for_unknown_route(self, handler):
        """Returns None for unknown routes."""
        mock_http = Mock()

        result = handler.handle_post("/api/unknown", {}, mock_http)
        assert result is None


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestSharingIntegration:
    """Integration tests for sharing workflow."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.sharing import SharingHandler, ShareStore
        handler = SharingHandler()
        handler._store = ShareStore()
        return handler

    @pytest.fixture
    def mock_user(self):
        user = Mock()
        user.id = "user-1"
        user.org_id = "org-1"
        return user

    def test_full_sharing_workflow(self, handler, mock_user):
        """Test complete sharing workflow."""
        mock_http = Mock()

        # 1. Get initial settings (creates default)
        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._get_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["visibility"] == "private"

        # 2. Make public
        body = {"visibility": "public", "allow_comments": True}
        with patch.object(handler, "get_current_user", return_value=mock_user):
            with patch.object(handler, "read_json_body", return_value=body):
                result = handler._update_share_settings("debate-1", mock_http)

        assert result.status_code == 200
        data = json.loads(result.body)
        token = data["settings"]["share_token"]
        assert token is not None

        # 3. Access via shared link (no auth needed)
        mock_debate = {"id": "debate-1", "title": "Test"}
        with patch.object(handler, "_get_debate_data", return_value=mock_debate):
            result = handler._get_shared_debate(token, {})

        assert result.status_code == 200

        # 4. Revoke share
        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler._revoke_share("debate-1", mock_http)

        assert result.status_code == 200

        # 5. Shared link no longer works
        result = handler._get_shared_debate(token, {})
        assert result.status_code == 404
