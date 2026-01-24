"""
Tests for TeamsUserIdentityBridge - Azure AD to Aragora user mapping.

Tests cover:
- TeamsUserInfo dataclass
- User resolution from Azure AD
- User sync from Teams to Aragora
- Linking/unlinking Teams identities
- Activity user info extraction
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.chat.teams_identity import (
    TeamsUserIdentityBridge,
    TeamsUserInfo,
    get_teams_identity_bridge,
    TEAMS_PROVIDER,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def identity_bridge():
    """Create a TeamsUserIdentityBridge instance for testing."""
    bridge = TeamsUserIdentityBridge()
    # Reset lazy-loaded repos
    bridge._external_identity_repo = None
    bridge._user_repo = None
    return bridge


@pytest.fixture
def sample_teams_user():
    """Create a sample TeamsUserInfo."""
    return TeamsUserInfo(
        aad_object_id="aad-object-id-123",
        tenant_id="tenant-id-456",
        display_name="Test User",
        email="test@example.com",
        user_principal_name="test@example.com",
        given_name="Test",
        surname="User",
        job_title="Engineer",
        department="Engineering",
    )


@pytest.fixture
def sample_activity():
    """Create a sample Bot Framework activity."""
    return {
        "type": "message",
        "id": "activity-123",
        "from": {
            "id": "29:user-id-123",
            "name": "Test User",
            "aadObjectId": "aad-object-id-456",
        },
        "conversation": {
            "id": "conv-id-789",
            "tenantId": "tenant-id-abc",
        },
        "channelData": {
            "tenant": {"id": "tenant-id-abc"},
            "channel": {"id": "channel-id-xyz"},
        },
    }


@pytest.fixture
def mock_external_identity_repo():
    """Create a mock external identity repository."""
    repo = MagicMock()
    return repo


@pytest.fixture
def mock_user_repo():
    """Create a mock user repository."""
    repo = MagicMock()
    return repo


# ===========================================================================
# TeamsUserInfo Tests
# ===========================================================================


class TestTeamsUserInfo:
    """Tests for TeamsUserInfo dataclass."""

    def test_to_dict(self, sample_teams_user):
        """Test to_dict includes all fields."""
        result = sample_teams_user.to_dict()

        assert result["aad_object_id"] == "aad-object-id-123"
        assert result["tenant_id"] == "tenant-id-456"
        assert result["display_name"] == "Test User"
        assert result["email"] == "test@example.com"
        assert result["user_principal_name"] == "test@example.com"
        assert result["given_name"] == "Test"
        assert result["surname"] == "User"
        assert result["job_title"] == "Engineer"
        assert result["department"] == "Engineering"

    def test_to_dict_with_none_fields(self):
        """Test to_dict handles None fields."""
        user = TeamsUserInfo(
            aad_object_id="aad-123",
            tenant_id="tenant-456",
        )

        result = user.to_dict()

        assert result["aad_object_id"] == "aad-123"
        assert result["display_name"] is None
        assert result["email"] is None

    def test_minimal_init(self):
        """Test minimal initialization with required fields only."""
        user = TeamsUserInfo(
            aad_object_id="aad-123",
            tenant_id="tenant-456",
        )

        assert user.aad_object_id == "aad-123"
        assert user.tenant_id == "tenant-456"
        assert user.display_name is None


# ===========================================================================
# TeamsUserIdentityBridge Initialization Tests
# ===========================================================================


class TestTeamsUserIdentityBridgeInit:
    """Tests for TeamsUserIdentityBridge initialization."""

    def test_init_lazy_loads_repos(self, identity_bridge):
        """Test repositories are lazy-loaded."""
        assert identity_bridge._external_identity_repo is None
        assert identity_bridge._user_repo is None

    def test_get_external_identity_repo_caches(self, identity_bridge):
        """Test external identity repo is cached after first load."""
        mock_repo = MagicMock()

        # Patch where it's imported from (inside _get_external_identity_repo)
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.repositories.external_identity": MagicMock(
                    get_external_identity_repository=MagicMock(return_value=mock_repo)
                )
            },
        ):
            repo1 = identity_bridge._get_external_identity_repo()
            repo2 = identity_bridge._get_external_identity_repo()

        assert repo1 is repo2
        assert repo1 is mock_repo


# ===========================================================================
# Resolve User Tests
# ===========================================================================


class TestTeamsUserIdentityBridgeResolveUser:
    """Tests for resolve_user method."""

    @pytest.mark.asyncio
    async def test_resolve_user_found(self, identity_bridge, mock_external_identity_repo):
        """Test resolving an existing user."""
        mock_identity = MagicMock()
        mock_identity.id = "identity-123"
        mock_identity.user_id = "aragora-user-456"
        mock_identity.email = "test@example.com"
        mock_identity.display_name = "Test User"
        mock_identity.raw_claims = {"custom": "claim"}

        mock_external_identity_repo.get_by_external_id.return_value = mock_identity
        mock_external_identity_repo.update_last_seen = MagicMock()

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            with patch.object(identity_bridge, "_get_user_repo", return_value=None):
                result = await identity_bridge.resolve_user(
                    aad_object_id="aad-123",
                    tenant_id="tenant-456",
                )

        assert result is not None
        assert result.id == "aragora-user-456"
        assert result.email == "test@example.com"
        mock_external_identity_repo.get_by_external_id.assert_called_once_with(
            provider=TEAMS_PROVIDER,
            external_id="aad-123",
            tenant_id="tenant-456",
        )
        mock_external_identity_repo.update_last_seen.assert_called_once_with("identity-123")

    @pytest.mark.asyncio
    async def test_resolve_user_not_found(self, identity_bridge, mock_external_identity_repo):
        """Test returns None when user not found."""
        mock_external_identity_repo.get_by_external_id.return_value = None

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.resolve_user(
                aad_object_id="unknown-aad",
                tenant_id="tenant-456",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_user_enriches_from_user_repo(
        self, identity_bridge, mock_external_identity_repo, mock_user_repo
    ):
        """Test enriches user data from user repository."""
        mock_identity = MagicMock()
        mock_identity.id = "identity-123"
        mock_identity.user_id = "aragora-user-456"
        mock_identity.email = None
        mock_identity.display_name = None
        mock_identity.raw_claims = {}

        mock_external_identity_repo.get_by_external_id.return_value = mock_identity

        mock_stored_user = MagicMock()
        mock_stored_user.email = "enriched@example.com"
        mock_stored_user.name = "Enriched User"
        mock_stored_user.roles = ["admin"]
        mock_user_repo.get.return_value = mock_stored_user

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            with patch.object(identity_bridge, "_get_user_repo", return_value=mock_user_repo):
                result = await identity_bridge.resolve_user(
                    aad_object_id="aad-123",
                    tenant_id="tenant-456",
                )

        assert result.email == "enriched@example.com"
        assert result.name == "Enriched User"
        assert result.roles == ["admin"]


# ===========================================================================
# Sync User from Teams Tests
# ===========================================================================


class TestTeamsUserIdentityBridgeSyncUser:
    """Tests for sync_user_from_teams method."""

    @pytest.mark.asyncio
    async def test_sync_user_updates_existing(
        self, identity_bridge, sample_teams_user, mock_external_identity_repo
    ):
        """Test updates existing identity mapping."""
        mock_existing = MagicMock()
        mock_existing.user_id = "existing-user-id"
        mock_existing.email = "old@example.com"
        mock_existing.display_name = "Old Name"

        mock_external_identity_repo.get_by_external_id.return_value = mock_existing
        mock_external_identity_repo.update = MagicMock()

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.sync_user_from_teams(sample_teams_user)

        assert result is not None
        assert result.id == "existing-user-id"
        mock_external_identity_repo.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_user_creates_new(
        self, identity_bridge, sample_teams_user, mock_external_identity_repo
    ):
        """Test creates new user when not found."""
        mock_external_identity_repo.get_by_external_id.return_value = None
        mock_external_identity_repo.link_or_update = MagicMock()

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            with patch.object(
                identity_bridge,
                "_find_or_create_user",
                new_callable=AsyncMock,
                return_value="new-user-id",
            ):
                result = await identity_bridge.sync_user_from_teams(
                    sample_teams_user, create_if_missing=True
                )

        assert result is not None
        assert result.id == "new-user-id"
        mock_external_identity_repo.link_or_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_user_returns_none_when_creation_disabled(
        self, identity_bridge, sample_teams_user, mock_external_identity_repo
    ):
        """Test returns None when user not found and creation disabled."""
        mock_external_identity_repo.get_by_external_id.return_value = None

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            with patch.object(
                identity_bridge,
                "_find_or_create_user",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await identity_bridge.sync_user_from_teams(
                    sample_teams_user, create_if_missing=False
                )

        assert result is None

    @pytest.mark.asyncio
    async def test_sync_user_uses_provided_aragora_user_id(
        self, identity_bridge, sample_teams_user, mock_external_identity_repo
    ):
        """Test uses provided Aragora user ID when specified."""
        mock_external_identity_repo.get_by_external_id.return_value = None
        mock_external_identity_repo.link_or_update = MagicMock()

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.sync_user_from_teams(
                sample_teams_user, aragora_user_id="specific-user-id"
            )

        assert result is not None
        assert result.id == "specific-user-id"


# ===========================================================================
# Find or Create User Tests
# ===========================================================================


class TestTeamsUserIdentityBridgeFindOrCreate:
    """Tests for _find_or_create_user method."""

    @pytest.mark.asyncio
    async def test_find_by_email(self, identity_bridge, sample_teams_user, mock_user_repo):
        """Test finds user by email."""
        mock_existing_user = MagicMock()
        mock_existing_user.id = "existing-by-email"
        mock_user_repo.get_by_email.return_value = mock_existing_user

        with patch.object(identity_bridge, "_get_user_repo", return_value=mock_user_repo):
            result = await identity_bridge._find_or_create_user(
                sample_teams_user, create_if_missing=True
            )

        assert result == "existing-by-email"
        mock_user_repo.get_by_email.assert_called_once_with("test@example.com")

    @pytest.mark.asyncio
    async def test_creates_new_user(self, identity_bridge, sample_teams_user, mock_user_repo):
        """Test creates new user when not found by email."""
        mock_user_repo.get_by_email.return_value = None
        mock_user_repo.create = MagicMock()

        with patch.object(identity_bridge, "_get_user_repo", return_value=mock_user_repo):
            result = await identity_bridge._find_or_create_user(
                sample_teams_user, create_if_missing=True
            )

        assert result is not None
        assert result.startswith("teams-")
        mock_user_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_creation_disabled(
        self, identity_bridge, sample_teams_user, mock_user_repo
    ):
        """Test returns None when creation disabled and user not found."""
        mock_user_repo.get_by_email.return_value = None

        with patch.object(identity_bridge, "_get_user_repo", return_value=mock_user_repo):
            result = await identity_bridge._find_or_create_user(
                sample_teams_user, create_if_missing=False
            )

        assert result is None


# ===========================================================================
# Get User by AAD ID Tests
# ===========================================================================


class TestTeamsUserIdentityBridgeGetByAadId:
    """Tests for get_user_by_aad_id method."""

    @pytest.mark.asyncio
    async def test_get_user_found(self, identity_bridge, mock_external_identity_repo):
        """Test returns user ID when found."""
        mock_identity = MagicMock()
        mock_identity.user_id = "aragora-user-123"
        mock_external_identity_repo.get_by_external_id.return_value = mock_identity

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.get_user_by_aad_id("aad-123")

        assert result == "aragora-user-123"

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, identity_bridge, mock_external_identity_repo):
        """Test returns None when not found."""
        mock_external_identity_repo.get_by_external_id.return_value = None

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.get_user_by_aad_id("unknown-aad")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_with_tenant_filter(self, identity_bridge, mock_external_identity_repo):
        """Test passes tenant_id filter when provided."""
        mock_external_identity_repo.get_by_external_id.return_value = None

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            await identity_bridge.get_user_by_aad_id("aad-123", tenant_id="specific-tenant")

        mock_external_identity_repo.get_by_external_id.assert_called_once_with(
            provider=TEAMS_PROVIDER,
            external_id="aad-123",
            tenant_id="specific-tenant",
        )


# ===========================================================================
# Link/Unlink Teams User Tests
# ===========================================================================


class TestTeamsUserIdentityBridgeLinking:
    """Tests for link and unlink methods."""

    @pytest.mark.asyncio
    async def test_link_teams_user_success(self, identity_bridge, mock_external_identity_repo):
        """Test successfully links Teams user."""
        mock_external_identity_repo.link_or_update = MagicMock()

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.link_teams_user(
                aragora_user_id="aragora-123",
                aad_object_id="aad-456",
                tenant_id="tenant-789",
                email="user@example.com",
                display_name="Test User",
            )

        assert result is True
        mock_external_identity_repo.link_or_update.assert_called_once_with(
            user_id="aragora-123",
            provider=TEAMS_PROVIDER,
            external_id="aad-456",
            tenant_id="tenant-789",
            email="user@example.com",
            display_name="Test User",
        )

    @pytest.mark.asyncio
    async def test_link_teams_user_failure(self, identity_bridge, mock_external_identity_repo):
        """Test returns False on link failure."""
        mock_external_identity_repo.link_or_update = MagicMock(side_effect=Exception("DB error"))

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.link_teams_user(
                aragora_user_id="aragora-123",
                aad_object_id="aad-456",
                tenant_id="tenant-789",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_unlink_teams_user_success(self, identity_bridge, mock_external_identity_repo):
        """Test successfully unlinks Teams user."""
        mock_identity = MagicMock()
        mock_identity.id = "identity-123"
        mock_external_identity_repo.get_by_external_id.return_value = mock_identity
        mock_external_identity_repo.deactivate.return_value = True

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.unlink_teams_user(
                aad_object_id="aad-456",
                tenant_id="tenant-789",
            )

        assert result is True
        mock_external_identity_repo.deactivate.assert_called_once_with("identity-123")

    @pytest.mark.asyncio
    async def test_unlink_teams_user_not_found(self, identity_bridge, mock_external_identity_repo):
        """Test returns False when user not linked."""
        mock_external_identity_repo.get_by_external_id.return_value = None

        with patch.object(
            identity_bridge,
            "_get_external_identity_repo",
            return_value=mock_external_identity_repo,
        ):
            result = await identity_bridge.unlink_teams_user(
                aad_object_id="unknown-aad",
                tenant_id="tenant-789",
            )

        assert result is False


# ===========================================================================
# Extract User Info from Activity Tests
# ===========================================================================


class TestTeamsUserIdentityBridgeExtractActivity:
    """Tests for extract_user_info_from_activity method."""

    def test_extract_user_info_complete(self, identity_bridge, sample_activity):
        """Test extracts all available user info."""
        result = identity_bridge.extract_user_info_from_activity(sample_activity)

        assert result is not None
        assert result.aad_object_id == "aad-object-id-456"
        assert result.tenant_id == "tenant-id-abc"
        assert result.display_name == "Test User"

    def test_extract_user_info_missing_aad_id(self, identity_bridge):
        """Test returns None when aadObjectId is missing."""
        activity = {
            "from": {"id": "user-id", "name": "Test User"},
            "conversation": {"tenantId": "tenant-123"},
        }

        result = identity_bridge.extract_user_info_from_activity(activity)

        assert result is None

    def test_extract_user_info_uses_conversation_tenant(self, identity_bridge):
        """Test extracts tenant ID from conversation."""
        activity = {
            "from": {"id": "user-id", "aadObjectId": "aad-123"},
            "conversation": {"tenantId": "conv-tenant-id"},
            "channelData": {},
        }

        result = identity_bridge.extract_user_info_from_activity(activity)

        assert result.tenant_id == "conv-tenant-id"

    def test_extract_user_info_uses_channel_data_tenant(self, identity_bridge):
        """Test extracts tenant ID from channelData when not in conversation."""
        activity = {
            "from": {"id": "user-id", "aadObjectId": "aad-123"},
            "conversation": {},
            "channelData": {"tenant": {"id": "channel-tenant-id"}},
        }

        result = identity_bridge.extract_user_info_from_activity(activity)

        assert result.tenant_id == "channel-tenant-id"

    def test_extract_user_info_empty_tenant(self, identity_bridge):
        """Test handles missing tenant ID."""
        activity = {
            "from": {"id": "user-id", "aadObjectId": "aad-123"},
            "conversation": {},
            "channelData": {},
        }

        result = identity_bridge.extract_user_info_from_activity(activity)

        assert result.tenant_id == ""


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestTeamsUserIdentityBridgeSingleton:
    """Tests for singleton pattern."""

    def test_get_teams_identity_bridge_returns_singleton(self):
        """Test get_teams_identity_bridge returns the same instance."""
        import aragora.connectors.chat.teams_identity as module

        # Reset singleton
        module._bridge = None

        bridge1 = get_teams_identity_bridge()
        bridge2 = get_teams_identity_bridge()

        assert bridge1 is bridge2

        # Cleanup
        module._bridge = None
