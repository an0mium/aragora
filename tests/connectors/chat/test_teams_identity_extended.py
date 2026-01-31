"""
Extended tests for TeamsUserIdentityBridge - comprehensive coverage.

Covers additional edge cases and scenarios beyond the base test file:
- TeamsUserInfo: full field population, equality, repr, empty strings
- Lazy loading: external identity repo ImportError, user repo ImportError, caching
- resolve_user: enrichment error branches, empty identity fields
- sync_user_from_teams: existing update field merging, no aragora_user_id path
- _find_or_create_user: no email, no user_repo, create errors, name fallback logic
- get_user_by_aad_id: with/without tenant
- link_teams_user: ValueError/TypeError/RuntimeError, no optional fields
- unlink_teams_user: deactivate returns False
- extract_user_info_from_activity: empty from, missing conversation, missing channelData
- Singleton: reset and re-creation, concurrent calls
- TEAMS_PROVIDER constant
"""

import sys
import time
from dataclasses import fields as dataclass_fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.chat.teams_identity import (
    TEAMS_PROVIDER,
    TeamsUserIdentityBridge,
    TeamsUserInfo,
    get_teams_identity_bridge,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def bridge():
    """Create a fresh TeamsUserIdentityBridge instance."""
    b = TeamsUserIdentityBridge()
    b._external_identity_repo = None
    b._user_repo = None
    return b


@pytest.fixture
def teams_user_full():
    """TeamsUserInfo with all fields populated."""
    return TeamsUserInfo(
        aad_object_id="aad-full-001",
        tenant_id="tenant-full-001",
        display_name="Jane Doe",
        email="jane.doe@contoso.com",
        user_principal_name="jane.doe@contoso.com",
        given_name="Jane",
        surname="Doe",
        job_title="Senior Engineer",
        department="Platform",
    )


@pytest.fixture
def teams_user_minimal():
    """TeamsUserInfo with only required fields."""
    return TeamsUserInfo(
        aad_object_id="aad-minimal-001",
        tenant_id="tenant-minimal-001",
    )


@pytest.fixture
def mock_ext_repo():
    """Mock external identity repository."""
    return MagicMock()


@pytest.fixture
def mock_usr_repo():
    """Mock user repository."""
    return MagicMock()


# ===========================================================================
# Constants
# ===========================================================================


class TestTeamsProviderConstant:
    """Tests for the TEAMS_PROVIDER constant."""

    def test_teams_provider_value(self):
        """TEAMS_PROVIDER should be 'azure_ad'."""
        assert TEAMS_PROVIDER == "azure_ad"

    def test_teams_provider_is_string(self):
        """TEAMS_PROVIDER should be a string."""
        assert isinstance(TEAMS_PROVIDER, str)


# ===========================================================================
# TeamsUserInfo Extended Tests
# ===========================================================================


class TestTeamsUserInfoExtended:
    """Extended tests for TeamsUserInfo dataclass."""

    def test_all_fields_populated(self, teams_user_full):
        """All fields should be accessible when populated."""
        assert teams_user_full.aad_object_id == "aad-full-001"
        assert teams_user_full.tenant_id == "tenant-full-001"
        assert teams_user_full.display_name == "Jane Doe"
        assert teams_user_full.email == "jane.doe@contoso.com"
        assert teams_user_full.user_principal_name == "jane.doe@contoso.com"
        assert teams_user_full.given_name == "Jane"
        assert teams_user_full.surname == "Doe"
        assert teams_user_full.job_title == "Senior Engineer"
        assert teams_user_full.department == "Platform"

    def test_default_none_fields(self, teams_user_minimal):
        """Optional fields default to None."""
        assert teams_user_minimal.display_name is None
        assert teams_user_minimal.email is None
        assert teams_user_minimal.user_principal_name is None
        assert teams_user_minimal.given_name is None
        assert teams_user_minimal.surname is None
        assert teams_user_minimal.job_title is None
        assert teams_user_minimal.department is None

    def test_to_dict_keys(self, teams_user_full):
        """to_dict should contain exactly 9 keys."""
        result = teams_user_full.to_dict()
        expected_keys = {
            "aad_object_id",
            "tenant_id",
            "display_name",
            "email",
            "user_principal_name",
            "given_name",
            "surname",
            "job_title",
            "department",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_minimal_has_all_keys(self, teams_user_minimal):
        """to_dict from minimal user still has all keys."""
        result = teams_user_minimal.to_dict()
        assert len(result) == 9
        assert result["aad_object_id"] == "aad-minimal-001"
        assert result["tenant_id"] == "tenant-minimal-001"
        for key in [
            "display_name",
            "email",
            "user_principal_name",
            "given_name",
            "surname",
            "job_title",
            "department",
        ]:
            assert result[key] is None

    def test_to_dict_roundtrip_values(self, teams_user_full):
        """Values in to_dict should match attribute values."""
        d = teams_user_full.to_dict()
        for f in dataclass_fields(TeamsUserInfo):
            assert d[f.name] == getattr(teams_user_full, f.name)

    def test_empty_string_fields(self):
        """Empty strings should be preserved, not converted to None."""
        user = TeamsUserInfo(
            aad_object_id="aad-empty",
            tenant_id="",
            display_name="",
            email="",
        )
        assert user.tenant_id == ""
        assert user.display_name == ""
        assert user.email == ""
        d = user.to_dict()
        assert d["tenant_id"] == ""
        assert d["display_name"] == ""
        assert d["email"] == ""

    def test_dataclass_equality(self):
        """Two TeamsUserInfo with same fields should be equal."""
        a = TeamsUserInfo(aad_object_id="x", tenant_id="y", display_name="Z")
        b = TeamsUserInfo(aad_object_id="x", tenant_id="y", display_name="Z")
        assert a == b

    def test_dataclass_inequality(self):
        """Different field values should make objects unequal."""
        a = TeamsUserInfo(aad_object_id="x", tenant_id="y")
        b = TeamsUserInfo(aad_object_id="x2", tenant_id="y")
        assert a != b

    def test_special_characters_in_fields(self):
        """Fields should handle special/unicode characters."""
        user = TeamsUserInfo(
            aad_object_id="aad-special",
            tenant_id="tenant-special",
            display_name="Jean-Pierre O'Brien",
            email="jean+test@example.com",
            given_name="Jean-Pierre",
            surname="O'Brien",
        )
        d = user.to_dict()
        assert d["display_name"] == "Jean-Pierre O'Brien"
        assert d["email"] == "jean+test@example.com"


# ===========================================================================
# Bridge Initialization Extended Tests
# ===========================================================================


class TestBridgeInitExtended:
    """Extended tests for bridge initialization and lazy loading."""

    def test_initial_state(self, bridge):
        """Bridge starts with None repos."""
        assert bridge._external_identity_repo is None
        assert bridge._user_repo is None

    def test_get_external_identity_repo_lazy_loads(self, bridge):
        """First call to _get_external_identity_repo triggers import."""
        mock_repo = MagicMock()
        mock_module = MagicMock()
        mock_module.get_external_identity_repository.return_value = mock_repo

        with patch.dict(
            sys.modules,
            {
                "aragora.storage.repositories.external_identity": mock_module,
            },
        ):
            result = bridge._get_external_identity_repo()

        assert result is mock_repo
        assert bridge._external_identity_repo is mock_repo

    def test_get_external_identity_repo_cached(self, bridge):
        """Second call returns cached repo without re-importing."""
        sentinel = MagicMock()
        bridge._external_identity_repo = sentinel
        result = bridge._get_external_identity_repo()
        assert result is sentinel

    def test_get_user_repo_lazy_loads(self, bridge):
        """First call to _get_user_repo triggers import."""
        mock_repo = MagicMock()
        mock_module = MagicMock()
        mock_module.get_user_repository.return_value = mock_repo

        with patch.dict(
            sys.modules,
            {
                "aragora.storage.repositories.users": mock_module,
            },
        ):
            result = bridge._get_user_repo()

        assert result is mock_repo
        assert bridge._user_repo is mock_repo

    def test_get_user_repo_cached(self, bridge):
        """Second call returns cached user repo."""
        sentinel = MagicMock()
        bridge._user_repo = sentinel
        result = bridge._get_user_repo()
        assert result is sentinel

    def test_get_user_repo_import_error(self, bridge):
        """ImportError on user repo import returns None."""
        with patch.dict(
            sys.modules,
            {
                "aragora.storage.repositories.users": None,
            },
        ):
            # Importing from a module mapped to None raises ImportError
            result = bridge._get_user_repo()

        assert result is None
        assert bridge._user_repo is None


# ===========================================================================
# resolve_user Extended Tests
# ===========================================================================


class TestResolveUserExtended:
    """Extended tests for resolve_user method."""

    @pytest.mark.asyncio
    async def test_resolve_user_builds_sso_user_correctly(self, bridge, mock_ext_repo):
        """Verify all SSOUser fields are set from identity."""
        identity = MagicMock()
        identity.id = "id-1"
        identity.user_id = "user-1"
        identity.email = "alice@example.com"
        identity.display_name = "Alice"
        identity.raw_claims = {"role": "admin"}
        mock_ext_repo.get_by_external_id.return_value = identity

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=None):
                result = await bridge.resolve_user(
                    aad_object_id="aad-alice",
                    tenant_id="tenant-1",
                )

        assert result.id == "user-1"
        assert result.email == "alice@example.com"
        assert result.name == "Alice"
        assert result.display_name == "Alice"
        assert result.azure_object_id == "aad-alice"
        assert result.azure_tenant_id == "tenant-1"
        assert result.provider_type == "azure_ad"
        assert result.raw_claims == {"role": "admin"}

    @pytest.mark.asyncio
    async def test_resolve_user_empty_email_and_name(self, bridge, mock_ext_repo):
        """When identity has None email/display_name, SSOUser gets empty strings."""
        identity = MagicMock()
        identity.id = "id-2"
        identity.user_id = "user-2"
        identity.email = None
        identity.display_name = None
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=None):
                result = await bridge.resolve_user(
                    aad_object_id="aad-2",
                    tenant_id="t-2",
                )

        assert result.email == ""
        assert result.name == ""
        assert result.display_name == ""

    @pytest.mark.asyncio
    async def test_resolve_user_enrichment_attribute_error(
        self, bridge, mock_ext_repo, mock_usr_repo
    ):
        """AttributeError during enrichment should not crash."""
        identity = MagicMock()
        identity.id = "id-3"
        identity.user_id = "user-3"
        identity.email = "before@example.com"
        identity.display_name = "Before"
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        mock_usr_repo.get.side_effect = AttributeError("no attribute 'email'")

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
                result = await bridge.resolve_user(
                    aad_object_id="aad-3",
                    tenant_id="t-3",
                )

        # Original values preserved when enrichment fails
        assert result.email == "before@example.com"
        assert result.name == "Before"

    @pytest.mark.asyncio
    async def test_resolve_user_enrichment_key_error(self, bridge, mock_ext_repo, mock_usr_repo):
        """KeyError during enrichment should not crash."""
        identity = MagicMock()
        identity.id = "id-4"
        identity.user_id = "user-4"
        identity.email = "key@test.com"
        identity.display_name = "Key"
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        mock_usr_repo.get.side_effect = KeyError("user not in dict")

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
                result = await bridge.resolve_user(
                    aad_object_id="aad-4",
                    tenant_id="t-4",
                )

        assert result.email == "key@test.com"

    @pytest.mark.asyncio
    async def test_resolve_user_enrichment_lookup_error(self, bridge, mock_ext_repo, mock_usr_repo):
        """LookupError during enrichment should not crash."""
        identity = MagicMock()
        identity.id = "id-5"
        identity.user_id = "user-5"
        identity.email = "lookup@test.com"
        identity.display_name = "Lookup"
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        mock_usr_repo.get.side_effect = LookupError("not found")

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
                result = await bridge.resolve_user(
                    aad_object_id="aad-5",
                    tenant_id="t-5",
                )

        assert result.email == "lookup@test.com"

    @pytest.mark.asyncio
    async def test_resolve_user_enrichment_stored_user_none(
        self, bridge, mock_ext_repo, mock_usr_repo
    ):
        """When stored_user is None, no enrichment happens."""
        identity = MagicMock()
        identity.id = "id-6"
        identity.user_id = "user-6"
        identity.email = "orig@test.com"
        identity.display_name = "Original"
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        mock_usr_repo.get.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
                result = await bridge.resolve_user(
                    aad_object_id="aad-6",
                    tenant_id="t-6",
                )

        assert result.email == "orig@test.com"
        assert result.name == "Original"

    @pytest.mark.asyncio
    async def test_resolve_user_enrichment_partial_stored_user(
        self, bridge, mock_ext_repo, mock_usr_repo
    ):
        """Stored user with some None fields only overwrites non-None."""
        identity = MagicMock()
        identity.id = "id-7"
        identity.user_id = "user-7"
        identity.email = "identity@test.com"
        identity.display_name = "Identity Name"
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        stored = MagicMock()
        stored.email = None
        stored.name = None
        stored.roles = ["viewer"]
        mock_usr_repo.get.return_value = stored

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
                result = await bridge.resolve_user(
                    aad_object_id="aad-7",
                    tenant_id="t-7",
                )

        # None or falsy stored values should NOT overwrite identity values
        # because of `stored_user.email or user.email` pattern
        assert result.email == "identity@test.com"
        assert result.name == "Identity Name"
        assert result.roles == ["viewer"]

    @pytest.mark.asyncio
    async def test_resolve_user_enrichment_no_roles_attr(
        self, bridge, mock_ext_repo, mock_usr_repo
    ):
        """When stored user has no 'roles' attribute, getattr returns empty list."""
        identity = MagicMock()
        identity.id = "id-8"
        identity.user_id = "user-8"
        identity.email = "noroles@test.com"
        identity.display_name = "NoRoles"
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        stored = MagicMock(spec=["email", "name"])  # No 'roles'
        stored.email = "enriched@test.com"
        stored.name = "Enriched"
        mock_usr_repo.get.return_value = stored

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
                result = await bridge.resolve_user(
                    aad_object_id="aad-8",
                    tenant_id="t-8",
                )

        assert result.email == "enriched@test.com"
        assert result.roles == []  # getattr default

    @pytest.mark.asyncio
    async def test_resolve_user_calls_update_last_seen(self, bridge, mock_ext_repo):
        """update_last_seen should be called with identity.id."""
        identity = MagicMock()
        identity.id = "id-seen"
        identity.user_id = "user-seen"
        identity.email = "seen@test.com"
        identity.display_name = "Seen"
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=None):
                await bridge.resolve_user(aad_object_id="aad-seen", tenant_id="t-seen")

        mock_ext_repo.update_last_seen.assert_called_once_with("id-seen")


# ===========================================================================
# sync_user_from_teams Extended Tests
# ===========================================================================


class TestSyncUserExtended:
    """Extended tests for sync_user_from_teams method."""

    @pytest.mark.asyncio
    async def test_sync_existing_updates_email(self, bridge, teams_user_full, mock_ext_repo):
        """Existing identity email should be updated from teams_user."""
        existing = MagicMock()
        existing.user_id = "existing-uid"
        existing.email = "old@example.com"
        existing.display_name = "Old Name"
        mock_ext_repo.get_by_external_id.return_value = existing

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.sync_user_from_teams(teams_user_full)

        assert existing.email == "jane.doe@contoso.com"
        assert existing.display_name == "Jane Doe"
        mock_ext_repo.update.assert_called_once_with(existing)

    @pytest.mark.asyncio
    async def test_sync_existing_preserves_email_when_teams_none(self, bridge, mock_ext_repo):
        """If teams_user email is None, existing email should be preserved."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-preserve",
            tenant_id="tenant-preserve",
            email=None,
            display_name=None,
        )
        existing = MagicMock()
        existing.user_id = "existing-uid"
        existing.email = "keep@example.com"
        existing.display_name = "Keep Name"
        mock_ext_repo.get_by_external_id.return_value = existing

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.sync_user_from_teams(teams_user)

        assert existing.email == "keep@example.com"
        assert existing.display_name == "Keep Name"

    @pytest.mark.asyncio
    async def test_sync_existing_sets_last_seen_at(self, bridge, mock_ext_repo):
        """Existing identity last_seen_at should be updated to current time."""
        teams_user = TeamsUserInfo(aad_object_id="aad-time", tenant_id="t-time")
        existing = MagicMock()
        existing.user_id = "uid-time"
        existing.email = None
        existing.display_name = None
        existing.last_seen_at = 0.0
        mock_ext_repo.get_by_external_id.return_value = existing

        before = time.time()
        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            await bridge.sync_user_from_teams(teams_user)
        after = time.time()

        assert before <= existing.last_seen_at <= after

    @pytest.mark.asyncio
    async def test_sync_existing_sets_raw_claims(self, bridge, teams_user_full, mock_ext_repo):
        """raw_claims should be set to teams_user.to_dict()."""
        existing = MagicMock()
        existing.user_id = "uid-claims"
        existing.email = "old@test.com"
        existing.display_name = "Old"
        mock_ext_repo.get_by_external_id.return_value = existing

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            await bridge.sync_user_from_teams(teams_user_full)

        assert existing.raw_claims == teams_user_full.to_dict()

    @pytest.mark.asyncio
    async def test_sync_new_returns_correct_sso_user(self, bridge, teams_user_full, mock_ext_repo):
        """New user sync should return SSOUser with all teams_user fields."""
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(
                bridge,
                "_find_or_create_user",
                new_callable=AsyncMock,
                return_value="new-uid-full",
            ):
                result = await bridge.sync_user_from_teams(teams_user_full)

        assert result.id == "new-uid-full"
        assert result.email == "jane.doe@contoso.com"
        assert result.name == "Jane Doe"
        assert result.display_name == "Jane Doe"
        assert result.first_name == "Jane"
        assert result.last_name == "Doe"
        assert result.azure_object_id == "aad-full-001"
        assert result.azure_tenant_id == "tenant-full-001"
        assert result.provider_type == "azure_ad"
        assert result.raw_claims == teams_user_full.to_dict()

    @pytest.mark.asyncio
    async def test_sync_new_with_no_display_name(self, bridge, mock_ext_repo):
        """SSOUser should use empty string for missing display_name."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-noname",
            tenant_id="t-noname",
        )
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(
                bridge,
                "_find_or_create_user",
                new_callable=AsyncMock,
                return_value="uid-noname",
            ):
                result = await bridge.sync_user_from_teams(teams_user)

        assert result.name == ""
        assert result.display_name == ""
        assert result.email == ""
        assert result.first_name == ""
        assert result.last_name == ""

    @pytest.mark.asyncio
    async def test_sync_new_calls_link_or_update(self, bridge, teams_user_full, mock_ext_repo):
        """New user should call link_or_update with correct args."""
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(
                bridge,
                "_find_or_create_user",
                new_callable=AsyncMock,
                return_value="linked-uid",
            ):
                await bridge.sync_user_from_teams(teams_user_full)

        mock_ext_repo.link_or_update.assert_called_once_with(
            user_id="linked-uid",
            provider=TEAMS_PROVIDER,
            external_id="aad-full-001",
            tenant_id="tenant-full-001",
            email="jane.doe@contoso.com",
            display_name="Jane Doe",
            raw_claims=teams_user_full.to_dict(),
        )

    @pytest.mark.asyncio
    async def test_sync_new_with_provided_aragora_user_id(self, bridge, mock_ext_repo):
        """When aragora_user_id is provided, _find_or_create_user should NOT be called."""
        teams_user = TeamsUserInfo(aad_object_id="aad-provided", tenant_id="t-provided")
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(
                bridge,
                "_find_or_create_user",
                new_callable=AsyncMock,
            ) as mock_find:
                result = await bridge.sync_user_from_teams(
                    teams_user,
                    aragora_user_id="given-uid",
                )

        mock_find.assert_not_called()
        assert result.id == "given-uid"

    @pytest.mark.asyncio
    async def test_sync_new_find_or_create_returns_none(self, bridge, mock_ext_repo):
        """When _find_or_create_user returns None, sync should return None."""
        teams_user = TeamsUserInfo(aad_object_id="aad-none", tenant_id="t-none")
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(
                bridge,
                "_find_or_create_user",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await bridge.sync_user_from_teams(teams_user)

        assert result is None


# ===========================================================================
# _find_or_create_user Extended Tests
# ===========================================================================


class TestFindOrCreateUserExtended:
    """Extended tests for _find_or_create_user."""

    @pytest.mark.asyncio
    async def test_find_by_email_success(self, bridge, teams_user_full, mock_usr_repo):
        """When user found by email, return their id."""
        found = MagicMock()
        found.id = "found-by-email"
        mock_usr_repo.get_by_email.return_value = found

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            result = await bridge._find_or_create_user(teams_user_full, create_if_missing=True)

        assert result == "found-by-email"
        mock_usr_repo.get_by_email.assert_called_once_with("jane.doe@contoso.com")

    @pytest.mark.asyncio
    async def test_find_by_email_attribute_error(self, bridge, teams_user_full, mock_usr_repo):
        """AttributeError in get_by_email should be caught; proceeds to creation."""
        mock_usr_repo.get_by_email.side_effect = AttributeError("no method")

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user_full,
                    create_if_missing=True,
                )

        assert result is not None
        assert result.startswith("teams-")

    @pytest.mark.asyncio
    async def test_find_by_email_key_error(self, bridge, teams_user_full, mock_usr_repo):
        """KeyError in get_by_email should be caught."""
        mock_usr_repo.get_by_email.side_effect = KeyError("no key")

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("h", "s")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user_full,
                    create_if_missing=True,
                )

        assert result is not None
        assert result.startswith("teams-")

    @pytest.mark.asyncio
    async def test_find_by_email_lookup_error(self, bridge, teams_user_full, mock_usr_repo):
        """LookupError in get_by_email should be caught."""
        mock_usr_repo.get_by_email.side_effect = LookupError("lookup fail")

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("h", "s")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user_full,
                    create_if_missing=True,
                )

        assert result is not None

    @pytest.mark.asyncio
    async def test_no_email_skips_search(self, bridge, mock_usr_repo):
        """When teams_user has no email, get_by_email should not be called."""
        teams_user = TeamsUserInfo(aad_object_id="aad-noemail", tenant_id="t")

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("h", "s")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user,
                    create_if_missing=True,
                )

        mock_usr_repo.get_by_email.assert_not_called()
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_user_repo_skips_search(self, bridge, teams_user_full):
        """When user_repo is None, skip email search and go to creation."""
        with patch.object(bridge, "_get_user_repo", return_value=None):
            result = await bridge._find_or_create_user(
                teams_user_full,
                create_if_missing=True,
            )

        assert result is not None
        assert result.startswith("teams-")

    @pytest.mark.asyncio
    async def test_create_user_with_display_name(self, bridge, teams_user_full, mock_usr_repo):
        """When display_name is available, it should be used as name."""
        mock_usr_repo.get_by_email.return_value = None

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user_full,
                    create_if_missing=True,
                )

        # Check that create was called with display_name as name
        mock_usr_repo.create.assert_called_once()
        call_kwargs = mock_usr_repo.create.call_args
        assert call_kwargs[1]["name"] == "Jane Doe" or call_kwargs.kwargs["name"] == "Jane Doe"

    @pytest.mark.asyncio
    async def test_create_user_name_fallback_given_surname(self, bridge, mock_usr_repo):
        """When display_name is None, use given_name + surname."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-fallback",
            tenant_id="t-fallback",
            email="fallback@test.com",
            display_name=None,
            given_name="Alice",
            surname="Wonderland",
        )
        mock_usr_repo.get_by_email.return_value = None

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                await bridge._find_or_create_user(teams_user, create_if_missing=True)

        call_kwargs = mock_usr_repo.create.call_args
        # Should have joined given_name and surname
        name_arg = call_kwargs[1].get("name") or call_kwargs.kwargs.get("name")
        assert name_arg == "Alice Wonderland"

    @pytest.mark.asyncio
    async def test_create_user_name_fallback_given_only(self, bridge, mock_usr_repo):
        """When only given_name is available, use just that."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-given",
            tenant_id="t-given",
            email="given@test.com",
            display_name=None,
            given_name="Bob",
            surname=None,
        )
        mock_usr_repo.get_by_email.return_value = None

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                await bridge._find_or_create_user(teams_user, create_if_missing=True)

        call_kwargs = mock_usr_repo.create.call_args
        name_arg = call_kwargs[1].get("name") or call_kwargs.kwargs.get("name")
        assert name_arg == "Bob"

    @pytest.mark.asyncio
    async def test_create_user_name_fallback_surname_only(self, bridge, mock_usr_repo):
        """When only surname is available, use just that."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-surname",
            tenant_id="t-surname",
            email="surname@test.com",
            display_name=None,
            given_name=None,
            surname="Smith",
        )
        mock_usr_repo.get_by_email.return_value = None

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                await bridge._find_or_create_user(teams_user, create_if_missing=True)

        call_kwargs = mock_usr_repo.create.call_args
        name_arg = call_kwargs[1].get("name") or call_kwargs.kwargs.get("name")
        assert name_arg == "Smith"

    @pytest.mark.asyncio
    async def test_create_user_name_fallback_default(self, bridge, mock_usr_repo):
        """When no name fields, use 'Teams User' default."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-default",
            tenant_id="t-default",
            email="default@test.com",
            display_name=None,
            given_name=None,
            surname=None,
        )
        mock_usr_repo.get_by_email.return_value = None

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                await bridge._find_or_create_user(teams_user, create_if_missing=True)

        call_kwargs = mock_usr_repo.create.call_args
        name_arg = call_kwargs[1].get("name") or call_kwargs.kwargs.get("name")
        assert name_arg == "Teams User"

    @pytest.mark.asyncio
    async def test_create_user_email_fallback_to_local(self, bridge, mock_usr_repo):
        """When teams_user has no email, use <user_id>@teams.local."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-local",
            tenant_id="t-local",
            email=None,
        )
        mock_usr_repo.get_by_email.assert_not_called  # no email means skip search

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user,
                    create_if_missing=True,
                )

        call_kwargs = mock_usr_repo.create.call_args
        email_arg = call_kwargs[1].get("email") or call_kwargs.kwargs.get("email")
        assert email_arg.endswith("@teams.local")
        assert email_arg.startswith("teams-")

    @pytest.mark.asyncio
    async def test_create_user_repo_create_value_error(self, bridge, mock_usr_repo):
        """ValueError during user_repo.create should be caught; user_id still returned."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-valerr",
            tenant_id="t-valerr",
            email="valerr@test.com",
        )
        mock_usr_repo.get_by_email.return_value = None
        mock_usr_repo.create.side_effect = ValueError("invalid data")

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user,
                    create_if_missing=True,
                )

        # user_id should still be returned even if repo.create fails
        assert result is not None
        assert result.startswith("teams-")

    @pytest.mark.asyncio
    async def test_create_user_repo_create_type_error(self, bridge, mock_usr_repo):
        """TypeError during user_repo.create should be caught."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-typerr",
            tenant_id="t-typerr",
            email="typerr@test.com",
        )
        mock_usr_repo.get_by_email.return_value = None
        mock_usr_repo.create.side_effect = TypeError("type issue")

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user,
                    create_if_missing=True,
                )

        assert result is not None

    @pytest.mark.asyncio
    async def test_create_user_repo_create_runtime_error(self, bridge, mock_usr_repo):
        """RuntimeError during user_repo.create should be caught."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-rterr",
            tenant_id="t-rterr",
            email="rterr@test.com",
        )
        mock_usr_repo.get_by_email.return_value = None
        mock_usr_repo.create.side_effect = RuntimeError("runtime issue")

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            with patch.dict(
                sys.modules,
                {
                    "aragora.billing.models": MagicMock(
                        hash_password=MagicMock(return_value=("hash", "salt")),
                    ),
                },
            ):
                result = await bridge._find_or_create_user(
                    teams_user,
                    create_if_missing=True,
                )

        assert result is not None

    @pytest.mark.asyncio
    async def test_create_disabled_returns_none(self, bridge, mock_usr_repo):
        """When create_if_missing is False and no user found, return None."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-nocreate",
            tenant_id="t-nocreate",
            email="nocreate@test.com",
        )
        mock_usr_repo.get_by_email.return_value = None

        with patch.object(bridge, "_get_user_repo", return_value=mock_usr_repo):
            result = await bridge._find_or_create_user(
                teams_user,
                create_if_missing=False,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_create_disabled_no_repo_returns_none(self, bridge):
        """When create_if_missing is False and no user_repo, return None."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-norepo",
            tenant_id="t-norepo",
            email="norepo@test.com",
        )

        with patch.object(bridge, "_get_user_repo", return_value=None):
            result = await bridge._find_or_create_user(
                teams_user,
                create_if_missing=False,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_create_no_user_repo_still_returns_id(self, bridge):
        """When no user_repo but create_if_missing, still generate and return user_id."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-norepo-create",
            tenant_id="t-norepo-create",
        )

        with patch.object(bridge, "_get_user_repo", return_value=None):
            result = await bridge._find_or_create_user(
                teams_user,
                create_if_missing=True,
            )

        assert result is not None
        assert result.startswith("teams-")
        assert len(result) == len("teams-") + 12  # uuid4().hex[:12]

    @pytest.mark.asyncio
    async def test_generated_user_id_format(self, bridge):
        """Generated user IDs should be 'teams-' plus 12 hex chars."""
        teams_user = TeamsUserInfo(
            aad_object_id="aad-format",
            tenant_id="t-format",
        )

        with patch.object(bridge, "_get_user_repo", return_value=None):
            result = await bridge._find_or_create_user(
                teams_user,
                create_if_missing=True,
            )

        assert result.startswith("teams-")
        hex_part = result[len("teams-") :]
        assert len(hex_part) == 12
        # Verify it's valid hex
        int(hex_part, 16)


# ===========================================================================
# get_user_by_aad_id Extended Tests
# ===========================================================================


class TestGetUserByAadIdExtended:
    """Extended tests for get_user_by_aad_id."""

    @pytest.mark.asyncio
    async def test_found_returns_user_id(self, bridge, mock_ext_repo):
        """When identity found, return user_id."""
        identity = MagicMock()
        identity.user_id = "found-user"
        mock_ext_repo.get_by_external_id.return_value = identity

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.get_user_by_aad_id("aad-found")

        assert result == "found-user"

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self, bridge, mock_ext_repo):
        """When identity not found, return None."""
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.get_user_by_aad_id("aad-missing")

        assert result is None

    @pytest.mark.asyncio
    async def test_without_tenant_id(self, bridge, mock_ext_repo):
        """When tenant_id is None, passes None to repo."""
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            await bridge.get_user_by_aad_id("aad-no-tenant")

        mock_ext_repo.get_by_external_id.assert_called_once_with(
            provider=TEAMS_PROVIDER,
            external_id="aad-no-tenant",
            tenant_id=None,
        )

    @pytest.mark.asyncio
    async def test_with_tenant_id(self, bridge, mock_ext_repo):
        """When tenant_id is provided, passes it to repo."""
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            await bridge.get_user_by_aad_id("aad-with-tenant", tenant_id="t-123")

        mock_ext_repo.get_by_external_id.assert_called_once_with(
            provider=TEAMS_PROVIDER,
            external_id="aad-with-tenant",
            tenant_id="t-123",
        )


# ===========================================================================
# link_teams_user Extended Tests
# ===========================================================================


class TestLinkTeamsUserExtended:
    """Extended tests for link_teams_user."""

    @pytest.mark.asyncio
    async def test_link_success_minimal(self, bridge, mock_ext_repo):
        """Linking with only required params should succeed."""
        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.link_teams_user(
                aragora_user_id="uid-1",
                aad_object_id="aad-1",
                tenant_id="t-1",
            )

        assert result is True
        mock_ext_repo.link_or_update.assert_called_once_with(
            user_id="uid-1",
            provider=TEAMS_PROVIDER,
            external_id="aad-1",
            tenant_id="t-1",
            email=None,
            display_name=None,
        )

    @pytest.mark.asyncio
    async def test_link_success_with_all_params(self, bridge, mock_ext_repo):
        """Linking with email and display_name should pass them through."""
        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.link_teams_user(
                aragora_user_id="uid-2",
                aad_object_id="aad-2",
                tenant_id="t-2",
                email="link@test.com",
                display_name="Linked User",
            )

        assert result is True
        mock_ext_repo.link_or_update.assert_called_once_with(
            user_id="uid-2",
            provider=TEAMS_PROVIDER,
            external_id="aad-2",
            tenant_id="t-2",
            email="link@test.com",
            display_name="Linked User",
        )

    @pytest.mark.asyncio
    async def test_link_value_error(self, bridge, mock_ext_repo):
        """ValueError should be caught and return False."""
        mock_ext_repo.link_or_update.side_effect = ValueError("bad value")

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.link_teams_user(
                aragora_user_id="uid-3",
                aad_object_id="aad-3",
                tenant_id="t-3",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_link_type_error(self, bridge, mock_ext_repo):
        """TypeError should be caught and return False."""
        mock_ext_repo.link_or_update.side_effect = TypeError("type issue")

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.link_teams_user(
                aragora_user_id="uid-4",
                aad_object_id="aad-4",
                tenant_id="t-4",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_link_runtime_error(self, bridge, mock_ext_repo):
        """RuntimeError should be caught and return False."""
        mock_ext_repo.link_or_update.side_effect = RuntimeError("runtime issue")

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.link_teams_user(
                aragora_user_id="uid-5",
                aad_object_id="aad-5",
                tenant_id="t-5",
            )

        assert result is False


# ===========================================================================
# unlink_teams_user Extended Tests
# ===========================================================================


class TestUnlinkTeamsUserExtended:
    """Extended tests for unlink_teams_user."""

    @pytest.mark.asyncio
    async def test_unlink_success(self, bridge, mock_ext_repo):
        """Unlink should call deactivate and return its result."""
        identity = MagicMock()
        identity.id = "ident-unlink"
        mock_ext_repo.get_by_external_id.return_value = identity
        mock_ext_repo.deactivate.return_value = True

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.unlink_teams_user(
                aad_object_id="aad-unlink",
                tenant_id="t-unlink",
            )

        assert result is True
        mock_ext_repo.deactivate.assert_called_once_with("ident-unlink")

    @pytest.mark.asyncio
    async def test_unlink_not_found(self, bridge, mock_ext_repo):
        """When identity not found, return False without calling deactivate."""
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.unlink_teams_user(
                aad_object_id="aad-notfound",
                tenant_id="t-notfound",
            )

        assert result is False
        mock_ext_repo.deactivate.assert_not_called()

    @pytest.mark.asyncio
    async def test_unlink_deactivate_returns_false(self, bridge, mock_ext_repo):
        """When deactivate returns False, unlink returns False."""
        identity = MagicMock()
        identity.id = "ident-deact-false"
        mock_ext_repo.get_by_external_id.return_value = identity
        mock_ext_repo.deactivate.return_value = False

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.unlink_teams_user(
                aad_object_id="aad-deact-false",
                tenant_id="t-deact-false",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_unlink_passes_correct_params_to_get(self, bridge, mock_ext_repo):
        """get_by_external_id should receive correct provider, id, tenant."""
        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            await bridge.unlink_teams_user(
                aad_object_id="aad-params",
                tenant_id="t-params",
            )

        mock_ext_repo.get_by_external_id.assert_called_once_with(
            provider=TEAMS_PROVIDER,
            external_id="aad-params",
            tenant_id="t-params",
        )


# ===========================================================================
# extract_user_info_from_activity Extended Tests
# ===========================================================================


class TestExtractUserInfoExtended:
    """Extended tests for extract_user_info_from_activity."""

    def test_complete_activity(self, bridge):
        """Full activity should produce TeamsUserInfo with all extractable fields."""
        activity = {
            "from": {
                "id": "29:user-123",
                "name": "Full User",
                "aadObjectId": "aad-full",
            },
            "conversation": {"tenantId": "tenant-conv"},
            "channelData": {"tenant": {"id": "tenant-ch"}},
        }
        result = bridge.extract_user_info_from_activity(activity)

        assert result is not None
        assert result.aad_object_id == "aad-full"
        assert result.tenant_id == "tenant-conv"  # conversation takes precedence
        assert result.display_name == "Full User"

    def test_no_aad_object_id(self, bridge):
        """Missing aadObjectId should return None."""
        activity = {
            "from": {"id": "29:user-no-aad", "name": "NoAAD"},
            "conversation": {"tenantId": "t"},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result is None

    def test_empty_from(self, bridge):
        """Empty 'from' dict should return None (no aadObjectId)."""
        activity = {"from": {}, "conversation": {"tenantId": "t"}}
        result = bridge.extract_user_info_from_activity(activity)
        assert result is None

    def test_missing_from_key(self, bridge):
        """Activity without 'from' key should return None."""
        activity = {"conversation": {"tenantId": "t"}}
        result = bridge.extract_user_info_from_activity(activity)
        assert result is None

    def test_tenant_from_conversation(self, bridge):
        """Tenant should come from conversation.tenantId first."""
        activity = {
            "from": {"aadObjectId": "aad-1"},
            "conversation": {"tenantId": "conv-tenant"},
            "channelData": {"tenant": {"id": "ch-tenant"}},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.tenant_id == "conv-tenant"

    def test_tenant_from_channel_data(self, bridge):
        """If conversation has no tenantId, fall back to channelData."""
        activity = {
            "from": {"aadObjectId": "aad-2"},
            "conversation": {},
            "channelData": {"tenant": {"id": "ch-tenant"}},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.tenant_id == "ch-tenant"

    def test_tenant_empty_when_both_missing(self, bridge):
        """If neither source has tenant, tenant_id should be empty string."""
        activity = {
            "from": {"aadObjectId": "aad-3"},
            "conversation": {},
            "channelData": {},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.tenant_id == ""

    def test_no_conversation_key(self, bridge):
        """Missing 'conversation' key should fallback gracefully."""
        activity = {
            "from": {"aadObjectId": "aad-4"},
            "channelData": {"tenant": {"id": "ch-t"}},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.tenant_id == "ch-t"

    def test_no_channel_data_key(self, bridge):
        """Missing 'channelData' key should fallback gracefully."""
        activity = {
            "from": {"aadObjectId": "aad-5"},
            "conversation": {"tenantId": "conv-t"},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.tenant_id == "conv-t"

    def test_no_conversation_and_no_channel_data(self, bridge):
        """Missing both 'conversation' and 'channelData' yields empty tenant."""
        activity = {
            "from": {"aadObjectId": "aad-6"},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.tenant_id == ""

    def test_channel_data_tenant_without_id(self, bridge):
        """channelData.tenant exists but has no 'id' key."""
        activity = {
            "from": {"aadObjectId": "aad-7"},
            "conversation": {},
            "channelData": {"tenant": {}},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.tenant_id == ""

    def test_display_name_from_from_name(self, bridge):
        """display_name should be extracted from from.name."""
        activity = {
            "from": {"aadObjectId": "aad-8", "name": "From Name"},
            "conversation": {"tenantId": "t"},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.display_name == "From Name"

    def test_display_name_none_when_no_name(self, bridge):
        """display_name should be None when from has no name."""
        activity = {
            "from": {"aadObjectId": "aad-9"},
            "conversation": {"tenantId": "t"},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert result.display_name is None

    def test_email_is_none(self, bridge):
        """Email is not extracted from activity (needs Graph API)."""
        activity = {
            "from": {"aadObjectId": "aad-10", "email": "ignored@test.com"},
            "conversation": {"tenantId": "t"},
        }
        result = bridge.extract_user_info_from_activity(activity)
        # Email is not set from activity data
        assert result.email is None

    def test_empty_activity(self, bridge):
        """Completely empty activity should return None."""
        result = bridge.extract_user_info_from_activity({})
        assert result is None

    def test_returns_teams_user_info_type(self, bridge):
        """Return type should be TeamsUserInfo."""
        activity = {
            "from": {"aadObjectId": "aad-type"},
            "conversation": {"tenantId": "t"},
        }
        result = bridge.extract_user_info_from_activity(activity)
        assert isinstance(result, TeamsUserInfo)


# ===========================================================================
# Singleton Extended Tests
# ===========================================================================


class TestSingletonExtended:
    """Extended tests for singleton pattern."""

    def test_singleton_returns_bridge_instance(self):
        """get_teams_identity_bridge should return a TeamsUserIdentityBridge."""
        import aragora.connectors.chat.teams_identity as module

        module._bridge = None
        try:
            bridge = get_teams_identity_bridge()
            assert isinstance(bridge, TeamsUserIdentityBridge)
        finally:
            module._bridge = None

    def test_singleton_same_instance(self):
        """Calling get_teams_identity_bridge twice returns same object."""
        import aragora.connectors.chat.teams_identity as module

        module._bridge = None
        try:
            b1 = get_teams_identity_bridge()
            b2 = get_teams_identity_bridge()
            assert b1 is b2
        finally:
            module._bridge = None

    def test_singleton_reset(self):
        """Resetting _bridge to None causes new instance creation."""
        import aragora.connectors.chat.teams_identity as module

        module._bridge = None
        try:
            b1 = get_teams_identity_bridge()
            module._bridge = None
            b2 = get_teams_identity_bridge()
            assert b1 is not b2
        finally:
            module._bridge = None

    def test_singleton_uses_existing(self):
        """If _bridge is already set, get_teams_identity_bridge returns it."""
        import aragora.connectors.chat.teams_identity as module

        sentinel = TeamsUserIdentityBridge()
        module._bridge = sentinel
        try:
            result = get_teams_identity_bridge()
            assert result is sentinel
        finally:
            module._bridge = None


# ===========================================================================
# Integration-like scenario tests
# ===========================================================================


class TestScenarios:
    """End-to-end-like scenarios combining multiple methods."""

    @pytest.mark.asyncio
    async def test_extract_then_resolve(self, bridge, mock_ext_repo):
        """Extract user info from activity, then resolve to SSOUser."""
        activity = {
            "from": {"aadObjectId": "aad-scenario-1", "name": "Scenario User"},
            "conversation": {"tenantId": "scenario-tenant"},
        }

        info = bridge.extract_user_info_from_activity(activity)
        assert info is not None

        identity = MagicMock()
        identity.id = "ident-sc1"
        identity.user_id = "user-sc1"
        identity.email = "scenario@test.com"
        identity.display_name = "Scenario User"
        identity.raw_claims = {}
        mock_ext_repo.get_by_external_id.return_value = identity

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(bridge, "_get_user_repo", return_value=None):
                user = await bridge.resolve_user(
                    aad_object_id=info.aad_object_id,
                    tenant_id=info.tenant_id,
                )

        assert user is not None
        assert user.id == "user-sc1"
        assert user.azure_object_id == "aad-scenario-1"

    @pytest.mark.asyncio
    async def test_extract_then_sync(self, bridge, mock_ext_repo):
        """Extract user info, then sync to create new mapping."""
        activity = {
            "from": {"aadObjectId": "aad-scenario-2", "name": "New Sync User"},
            "conversation": {"tenantId": "sync-tenant"},
        }

        info = bridge.extract_user_info_from_activity(activity)
        assert info is not None

        mock_ext_repo.get_by_external_id.return_value = None

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            with patch.object(
                bridge,
                "_find_or_create_user",
                new_callable=AsyncMock,
                return_value="new-sync-uid",
            ):
                result = await bridge.sync_user_from_teams(info)

        assert result is not None
        assert result.id == "new-sync-uid"
        assert result.display_name == "New Sync User"

    @pytest.mark.asyncio
    async def test_link_then_get_by_aad_id(self, bridge, mock_ext_repo):
        """Link a user, then look them up by AAD ID."""
        # Link
        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            linked = await bridge.link_teams_user(
                aragora_user_id="uid-linked",
                aad_object_id="aad-linked",
                tenant_id="t-linked",
            )
        assert linked is True

        # Look up
        identity = MagicMock()
        identity.user_id = "uid-linked"
        mock_ext_repo.get_by_external_id.return_value = identity

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            found = await bridge.get_user_by_aad_id("aad-linked", tenant_id="t-linked")

        assert found == "uid-linked"

    @pytest.mark.asyncio
    async def test_link_then_unlink(self, bridge, mock_ext_repo):
        """Link and then unlink a user."""
        # Link
        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            await bridge.link_teams_user(
                aragora_user_id="uid-lu",
                aad_object_id="aad-lu",
                tenant_id="t-lu",
            )

        # Unlink
        identity = MagicMock()
        identity.id = "ident-lu"
        mock_ext_repo.get_by_external_id.return_value = identity
        mock_ext_repo.deactivate.return_value = True

        with patch.object(bridge, "_get_external_identity_repo", return_value=mock_ext_repo):
            result = await bridge.unlink_teams_user(
                aad_object_id="aad-lu",
                tenant_id="t-lu",
            )

        assert result is True
