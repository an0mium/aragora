"""Tests for SCIM 2.0 namespace API.

Tests both synchronous (SCIMAPI) and asynchronous (AsyncSCIMAPI) classes
for SCIM 2.0 compliant user and group provisioning (RFC 7643/7644).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# User Operations - Sync
# =========================================================================


class TestSCIMListUsers:
    """Tests for listing SCIM users."""

    def test_list_users_defaults(self) -> None:
        """List users with default pagination parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 2,
                "startIndex": 1,
                "itemsPerPage": 2,
                "Resources": [
                    {"id": "user-001", "userName": "alice@example.com"},
                    {"id": "user-002", "userName": "bob@example.com"},
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.list_users()

            mock_request.assert_called_once_with(
                "GET",
                "/scim/v2/Users",
                params={"startIndex": 1, "count": 100},
            )
            assert result["totalResults"] == 2
            assert len(result["Resources"]) == 2
            client.close()

    def test_list_users_custom_pagination(self) -> None:
        """List users with custom pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 50,
                "startIndex": 11,
                "itemsPerPage": 10,
                "Resources": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.list_users(start_index=11, count=10)

            mock_request.assert_called_once_with(
                "GET",
                "/scim/v2/Users",
                params={"startIndex": 11, "count": 10},
            )
            assert result["startIndex"] == 11
            client.close()

    def test_list_users_with_filter(self) -> None:
        """List users with SCIM filter expression."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 1,
                "Resources": [
                    {"id": "user-001", "userName": "john@example.com"},
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.list_users(filter='userName eq "john@example.com"')

            mock_request.assert_called_once_with(
                "GET",
                "/scim/v2/Users",
                params={
                    "startIndex": 1,
                    "count": 100,
                    "filter": 'userName eq "john@example.com"',
                },
            )
            assert result["totalResults"] == 1
            assert result["Resources"][0]["userName"] == "john@example.com"
            client.close()

    def test_list_users_with_all_params(self) -> None:
        """List users with all parameters specified."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 5,
                "Resources": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.scim.list_users(
                start_index=5,
                count=25,
                filter="active eq true",
            )

            mock_request.assert_called_once_with(
                "GET",
                "/scim/v2/Users",
                params={
                    "startIndex": 5,
                    "count": 25,
                    "filter": "active eq true",
                },
            )
            client.close()

    def test_list_users_empty_results(self) -> None:
        """List users returns empty when no users match."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 0,
                "Resources": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.list_users(filter='userName eq "nonexistent@example.com"')

            assert result["totalResults"] == 0
            assert result["Resources"] == []
            client.close()


class TestSCIMGetUser:
    """Tests for getting a single SCIM user."""

    def test_get_user(self) -> None:
        """Get a user by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "userName": "alice@example.com",
                "name": {
                    "givenName": "Alice",
                    "familyName": "Smith",
                    "formatted": "Alice Smith",
                },
                "emails": [
                    {"value": "alice@example.com", "primary": True, "type": "work"},
                ],
                "active": True,
                "meta": {
                    "resourceType": "User",
                    "created": "2025-01-01T12:00:00Z",
                    "lastModified": "2025-01-15T09:30:00Z",
                },
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.get_user("user-001")

            mock_request.assert_called_once_with("GET", "/scim/v2/Users/user-001")
            assert result["id"] == "user-001"
            assert result["userName"] == "alice@example.com"
            assert result["name"]["givenName"] == "Alice"
            assert result["active"] is True
            client.close()

    def test_get_user_with_enterprise_schema(self) -> None:
        """Get a user with enterprise extension schema."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": [
                    "urn:ietf:params:scim:schemas:core:2.0:User",
                    "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User",
                ],
                "id": "user-002",
                "userName": "bob@example.com",
                "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User": {
                    "employeeNumber": "EMP-12345",
                    "department": "Engineering",
                    "manager": {"value": "user-001"},
                },
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.get_user("user-002")

            mock_request.assert_called_once_with("GET", "/scim/v2/Users/user-002")
            enterprise = result["urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"]
            assert enterprise["employeeNumber"] == "EMP-12345"
            assert enterprise["department"] == "Engineering"
            client.close()


class TestSCIMCreateUser:
    """Tests for creating SCIM users."""

    def test_create_user_minimal(self) -> None:
        """Create a user with minimal required fields."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-new-001",
                "userName": "newuser@example.com",
                "meta": {
                    "resourceType": "User",
                    "created": "2025-01-20T10:00:00Z",
                    "location": "https://api.aragora.ai/scim/v2/Users/user-new-001",
                },
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            user_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "userName": "newuser@example.com",
            }
            result = client.scim.create_user(user_data)

            mock_request.assert_called_once_with(
                "POST",
                "/scim/v2/Users",
                json=user_data,
            )
            assert result["id"] == "user-new-001"
            assert result["userName"] == "newuser@example.com"
            client.close()

    def test_create_user_full(self) -> None:
        """Create a user with all common fields."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-new-002",
                "userName": "jane@example.com",
                "name": {
                    "givenName": "Jane",
                    "familyName": "Doe",
                },
                "emails": [{"value": "jane@example.com", "primary": True}],
                "active": True,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            user_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "userName": "jane@example.com",
                "name": {
                    "givenName": "Jane",
                    "familyName": "Doe",
                },
                "emails": [
                    {"value": "jane@example.com", "primary": True, "type": "work"},
                ],
                "active": True,
            }
            result = client.scim.create_user(user_data)

            mock_request.assert_called_once_with(
                "POST",
                "/scim/v2/Users",
                json=user_data,
            )
            assert result["name"]["givenName"] == "Jane"
            assert result["emails"][0]["value"] == "jane@example.com"
            client.close()

    def test_create_user_with_external_id(self) -> None:
        """Create a user with externalId for IdP correlation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-new-003",
                "externalId": "okta-user-abc123",
                "userName": "external@example.com",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            user_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "externalId": "okta-user-abc123",
                "userName": "external@example.com",
            }
            result = client.scim.create_user(user_data)

            mock_request.assert_called_once_with(
                "POST",
                "/scim/v2/Users",
                json=user_data,
            )
            assert result["externalId"] == "okta-user-abc123"
            client.close()


class TestSCIMReplaceUser:
    """Tests for replacing (full update) SCIM users."""

    def test_replace_user(self) -> None:
        """Replace a user with full resource."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "userName": "alice.updated@example.com",
                "name": {
                    "givenName": "Alice",
                    "familyName": "Johnson",
                },
                "active": True,
                "meta": {
                    "lastModified": "2025-01-21T14:00:00Z",
                },
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            user_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "userName": "alice.updated@example.com",
                "name": {
                    "givenName": "Alice",
                    "familyName": "Johnson",
                },
                "emails": [{"value": "alice.updated@example.com", "primary": True}],
                "active": True,
            }
            result = client.scim.replace_user("user-001", user_data)

            mock_request.assert_called_once_with(
                "PUT",
                "/scim/v2/Users/user-001",
                json=user_data,
            )
            assert result["userName"] == "alice.updated@example.com"
            assert result["name"]["familyName"] == "Johnson"
            client.close()

    def test_replace_user_deactivate(self) -> None:
        """Replace user to deactivate account."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-002",
                "userName": "bob@example.com",
                "active": False,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            user_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "userName": "bob@example.com",
                "name": {"givenName": "Bob", "familyName": "Smith"},
                "active": False,
            }
            result = client.scim.replace_user("user-002", user_data)

            mock_request.assert_called_once_with(
                "PUT",
                "/scim/v2/Users/user-002",
                json=user_data,
            )
            assert result["active"] is False
            client.close()


class TestSCIMPatchUser:
    """Tests for partially updating SCIM users."""

    def test_patch_user_add_operation(self) -> None:
        """Patch user with add operation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "userName": "alice@example.com",
                "phoneNumbers": [{"value": "+1-555-1234", "type": "work"}],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            patch_ops = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "add",
                        "path": "phoneNumbers",
                        "value": [{"value": "+1-555-1234", "type": "work"}],
                    },
                ],
            }
            result = client.scim.patch_user("user-001", patch_ops)

            mock_request.assert_called_once_with(
                "PATCH",
                "/scim/v2/Users/user-001",
                json=patch_ops,
            )
            assert result["phoneNumbers"][0]["value"] == "+1-555-1234"
            client.close()

    def test_patch_user_replace_operation(self) -> None:
        """Patch user with replace operation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "active": False,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            patch_ops = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {"op": "replace", "path": "active", "value": False},
                ],
            }
            result = client.scim.patch_user("user-001", patch_ops)

            mock_request.assert_called_once_with(
                "PATCH",
                "/scim/v2/Users/user-001",
                json=patch_ops,
            )
            assert result["active"] is False
            client.close()

    def test_patch_user_remove_operation(self) -> None:
        """Patch user with remove operation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "userName": "alice@example.com",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            patch_ops = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {"op": "remove", "path": "phoneNumbers"},
                ],
            }
            result = client.scim.patch_user("user-001", patch_ops)

            mock_request.assert_called_once_with(
                "PATCH",
                "/scim/v2/Users/user-001",
                json=patch_ops,
            )
            assert "phoneNumbers" not in result
            client.close()

    def test_patch_user_multiple_operations(self) -> None:
        """Patch user with multiple operations in a single request."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "userName": "alice.new@example.com",
                "active": True,
                "name": {"givenName": "Alicia", "familyName": "Smith"},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            patch_ops = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {"op": "replace", "path": "userName", "value": "alice.new@example.com"},
                    {"op": "replace", "path": "name.givenName", "value": "Alicia"},
                    {"op": "replace", "path": "active", "value": True},
                ],
            }
            result = client.scim.patch_user("user-001", patch_ops)

            mock_request.assert_called_once_with(
                "PATCH",
                "/scim/v2/Users/user-001",
                json=patch_ops,
            )
            assert result["userName"] == "alice.new@example.com"
            assert result["name"]["givenName"] == "Alicia"
            client.close()


class TestSCIMDeleteUser:
    """Tests for deleting SCIM users."""

    def test_delete_user(self) -> None:
        """Delete a user (204 No Content)."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = None

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.delete_user("user-001")

            mock_request.assert_called_once_with("DELETE", "/scim/v2/Users/user-001")
            assert result is None
            client.close()

    def test_delete_user_returns_empty(self) -> None:
        """Delete user returns empty dict when server sends empty response."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.delete_user("user-002")

            mock_request.assert_called_once_with("DELETE", "/scim/v2/Users/user-002")
            assert result == {}
            client.close()


# =========================================================================
# Group Operations - Sync
# =========================================================================


class TestSCIMListGroups:
    """Tests for listing SCIM groups."""

    def test_list_groups_defaults(self) -> None:
        """List groups with default pagination parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 3,
                "startIndex": 1,
                "itemsPerPage": 3,
                "Resources": [
                    {"id": "group-001", "displayName": "Engineering"},
                    {"id": "group-002", "displayName": "Marketing"},
                    {"id": "group-003", "displayName": "Sales"},
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.list_groups()

            mock_request.assert_called_once_with(
                "GET",
                "/scim/v2/Groups",
                params={"startIndex": 1, "count": 100},
            )
            assert result["totalResults"] == 3
            assert len(result["Resources"]) == 3
            client.close()

    def test_list_groups_custom_pagination(self) -> None:
        """List groups with custom pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 100,
                "startIndex": 21,
                "itemsPerPage": 20,
                "Resources": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.list_groups(start_index=21, count=20)

            mock_request.assert_called_once_with(
                "GET",
                "/scim/v2/Groups",
                params={"startIndex": 21, "count": 20},
            )
            assert result["startIndex"] == 21
            client.close()

    def test_list_groups_with_filter(self) -> None:
        """List groups with SCIM filter expression."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 1,
                "Resources": [
                    {"id": "group-001", "displayName": "Engineering"},
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.list_groups(filter='displayName eq "Engineering"')

            mock_request.assert_called_once_with(
                "GET",
                "/scim/v2/Groups",
                params={
                    "startIndex": 1,
                    "count": 100,
                    "filter": 'displayName eq "Engineering"',
                },
            )
            assert result["totalResults"] == 1
            assert result["Resources"][0]["displayName"] == "Engineering"
            client.close()

    def test_list_groups_empty_results(self) -> None:
        """List groups returns empty when no groups match."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 0,
                "Resources": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.list_groups(filter='displayName eq "NonExistent"')

            assert result["totalResults"] == 0
            assert result["Resources"] == []
            client.close()


class TestSCIMGetGroup:
    """Tests for getting a single SCIM group."""

    def test_get_group(self) -> None:
        """Get a group by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "Engineering",
                "members": [
                    {"value": "user-001", "display": "Alice Smith"},
                    {"value": "user-002", "display": "Bob Jones"},
                ],
                "meta": {
                    "resourceType": "Group",
                    "created": "2025-01-01T12:00:00Z",
                    "lastModified": "2025-01-15T09:30:00Z",
                },
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.get_group("group-001")

            mock_request.assert_called_once_with("GET", "/scim/v2/Groups/group-001")
            assert result["id"] == "group-001"
            assert result["displayName"] == "Engineering"
            assert len(result["members"]) == 2
            client.close()

    def test_get_group_empty_members(self) -> None:
        """Get a group with no members."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-empty",
                "displayName": "New Team",
                "members": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.get_group("group-empty")

            mock_request.assert_called_once_with("GET", "/scim/v2/Groups/group-empty")
            assert result["members"] == []
            client.close()


class TestSCIMCreateGroup:
    """Tests for creating SCIM groups."""

    def test_create_group_minimal(self) -> None:
        """Create a group with minimal required fields."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-new-001",
                "displayName": "New Team",
                "meta": {
                    "resourceType": "Group",
                    "created": "2025-01-20T10:00:00Z",
                    "location": "https://api.aragora.ai/scim/v2/Groups/group-new-001",
                },
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            group_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "displayName": "New Team",
            }
            result = client.scim.create_group(group_data)

            mock_request.assert_called_once_with(
                "POST",
                "/scim/v2/Groups",
                json=group_data,
            )
            assert result["id"] == "group-new-001"
            assert result["displayName"] == "New Team"
            client.close()

    def test_create_group_with_members(self) -> None:
        """Create a group with initial members."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-new-002",
                "displayName": "DevOps",
                "members": [
                    {"value": "user-001", "display": "Alice Smith"},
                    {"value": "user-003", "display": "Carol White"},
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            group_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "displayName": "DevOps",
                "members": [
                    {"value": "user-001"},
                    {"value": "user-003"},
                ],
            }
            result = client.scim.create_group(group_data)

            mock_request.assert_called_once_with(
                "POST",
                "/scim/v2/Groups",
                json=group_data,
            )
            assert len(result["members"]) == 2
            client.close()

    def test_create_group_with_external_id(self) -> None:
        """Create a group with externalId for IdP correlation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-new-003",
                "externalId": "azure-group-xyz789",
                "displayName": "External Group",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            group_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "externalId": "azure-group-xyz789",
                "displayName": "External Group",
            }
            result = client.scim.create_group(group_data)

            mock_request.assert_called_once_with(
                "POST",
                "/scim/v2/Groups",
                json=group_data,
            )
            assert result["externalId"] == "azure-group-xyz789"
            client.close()


class TestSCIMReplaceGroup:
    """Tests for replacing (full update) SCIM groups."""

    def test_replace_group(self) -> None:
        """Replace a group with full resource."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "Engineering Team",
                "members": [
                    {"value": "user-001", "display": "Alice Smith"},
                ],
                "meta": {
                    "lastModified": "2025-01-21T14:00:00Z",
                },
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            group_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "displayName": "Engineering Team",
                "members": [
                    {"value": "user-001"},
                ],
            }
            result = client.scim.replace_group("group-001", group_data)

            mock_request.assert_called_once_with(
                "PUT",
                "/scim/v2/Groups/group-001",
                json=group_data,
            )
            assert result["displayName"] == "Engineering Team"
            assert len(result["members"]) == 1
            client.close()

    def test_replace_group_clear_members(self) -> None:
        """Replace group to remove all members."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-002",
                "displayName": "Empty Group",
                "members": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            group_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "displayName": "Empty Group",
                "members": [],
            }
            result = client.scim.replace_group("group-002", group_data)

            mock_request.assert_called_once_with(
                "PUT",
                "/scim/v2/Groups/group-002",
                json=group_data,
            )
            assert result["members"] == []
            client.close()


class TestSCIMPatchGroup:
    """Tests for partially updating SCIM groups."""

    def test_patch_group_add_member(self) -> None:
        """Patch group to add a member."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "Engineering",
                "members": [
                    {"value": "user-001", "display": "Alice Smith"},
                    {"value": "user-005", "display": "New Member"},
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            patch_ops = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "add",
                        "path": "members",
                        "value": [{"value": "user-005"}],
                    },
                ],
            }
            result = client.scim.patch_group("group-001", patch_ops)

            mock_request.assert_called_once_with(
                "PATCH",
                "/scim/v2/Groups/group-001",
                json=patch_ops,
            )
            assert len(result["members"]) == 2
            client.close()

    def test_patch_group_remove_member(self) -> None:
        """Patch group to remove a member."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "Engineering",
                "members": [
                    {"value": "user-001", "display": "Alice Smith"},
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            patch_ops = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "remove",
                        "path": 'members[value eq "user-002"]',
                    },
                ],
            }
            result = client.scim.patch_group("group-001", patch_ops)

            mock_request.assert_called_once_with(
                "PATCH",
                "/scim/v2/Groups/group-001",
                json=patch_ops,
            )
            assert len(result["members"]) == 1
            client.close()

    def test_patch_group_replace_display_name(self) -> None:
        """Patch group to change displayName."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "Platform Engineering",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            patch_ops = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "replace",
                        "path": "displayName",
                        "value": "Platform Engineering",
                    },
                ],
            }
            result = client.scim.patch_group("group-001", patch_ops)

            mock_request.assert_called_once_with(
                "PATCH",
                "/scim/v2/Groups/group-001",
                json=patch_ops,
            )
            assert result["displayName"] == "Platform Engineering"
            client.close()

    def test_patch_group_multiple_operations(self) -> None:
        """Patch group with multiple operations."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "SRE",
                "members": [
                    {"value": "user-001"},
                    {"value": "user-010"},
                ],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            patch_ops = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {"op": "replace", "path": "displayName", "value": "SRE"},
                    {"op": "add", "path": "members", "value": [{"value": "user-010"}]},
                    {"op": "remove", "path": 'members[value eq "user-005"]'},
                ],
            }
            result = client.scim.patch_group("group-001", patch_ops)

            mock_request.assert_called_once_with(
                "PATCH",
                "/scim/v2/Groups/group-001",
                json=patch_ops,
            )
            assert result["displayName"] == "SRE"
            client.close()


class TestSCIMDeleteGroup:
    """Tests for deleting SCIM groups."""

    def test_delete_group(self) -> None:
        """Delete a group (204 No Content)."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = None

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.delete_group("group-001")

            mock_request.assert_called_once_with("DELETE", "/scim/v2/Groups/group-001")
            assert result is None
            client.close()

    def test_delete_group_returns_empty(self) -> None:
        """Delete group returns empty dict when server sends empty response."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.scim.delete_group("group-002")

            mock_request.assert_called_once_with("DELETE", "/scim/v2/Groups/group-002")
            assert result == {}
            client.close()


# =========================================================================
# Async User Operations
# =========================================================================


class TestAsyncSCIMUsers:
    """Tests for async SCIM user operations."""

    @pytest.mark.asyncio
    async def test_async_list_users_defaults(self) -> None:
        """List users asynchronously with defaults."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 1,
                "Resources": [{"id": "user-001", "userName": "alice@example.com"}],
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.scim.list_users()

                mock_request.assert_called_once_with(
                    "GET",
                    "/scim/v2/Users",
                    params={"startIndex": 1, "count": 100},
                )
                assert result["totalResults"] == 1

    @pytest.mark.asyncio
    async def test_async_list_users_with_filter(self) -> None:
        """List users asynchronously with filter."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 1,
                "Resources": [{"id": "user-001", "userName": "john@example.com"}],
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.scim.list_users(
                    start_index=5,
                    count=50,
                    filter="active eq true",
                )

                call_kwargs = mock_request.call_args[1]
                assert call_kwargs["params"]["startIndex"] == 5
                assert call_kwargs["params"]["count"] == 50
                assert call_kwargs["params"]["filter"] == "active eq true"

    @pytest.mark.asyncio
    async def test_async_get_user(self) -> None:
        """Get user asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "userName": "alice@example.com",
                "active": True,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.scim.get_user("user-001")

                mock_request.assert_called_once_with("GET", "/scim/v2/Users/user-001")
                assert result["userName"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_async_create_user(self) -> None:
        """Create user asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-new-001",
                "userName": "newuser@example.com",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                user_data = {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                    "userName": "newuser@example.com",
                    "name": {"givenName": "New", "familyName": "User"},
                }
                result = await client.scim.create_user(user_data)

                mock_request.assert_called_once_with(
                    "POST",
                    "/scim/v2/Users",
                    json=user_data,
                )
                assert result["id"] == "user-new-001"

    @pytest.mark.asyncio
    async def test_async_replace_user(self) -> None:
        """Replace user asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "userName": "alice.updated@example.com",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                user_data = {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                    "userName": "alice.updated@example.com",
                    "active": True,
                }
                result = await client.scim.replace_user("user-001", user_data)

                mock_request.assert_called_once_with(
                    "PUT",
                    "/scim/v2/Users/user-001",
                    json=user_data,
                )
                assert result["userName"] == "alice.updated@example.com"

    @pytest.mark.asyncio
    async def test_async_patch_user(self) -> None:
        """Patch user asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "id": "user-001",
                "active": False,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                patch_ops = {
                    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                    "Operations": [
                        {"op": "replace", "path": "active", "value": False},
                    ],
                }
                result = await client.scim.patch_user("user-001", patch_ops)

                mock_request.assert_called_once_with(
                    "PATCH",
                    "/scim/v2/Users/user-001",
                    json=patch_ops,
                )
                assert result["active"] is False

    @pytest.mark.asyncio
    async def test_async_delete_user(self) -> None:
        """Delete user asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = None

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.scim.delete_user("user-001")

                mock_request.assert_called_once_with("DELETE", "/scim/v2/Users/user-001")
                assert result is None


# =========================================================================
# Async Group Operations
# =========================================================================


class TestAsyncSCIMGroups:
    """Tests for async SCIM group operations."""

    @pytest.mark.asyncio
    async def test_async_list_groups_defaults(self) -> None:
        """List groups asynchronously with defaults."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 2,
                "Resources": [
                    {"id": "group-001", "displayName": "Engineering"},
                    {"id": "group-002", "displayName": "Marketing"},
                ],
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.scim.list_groups()

                mock_request.assert_called_once_with(
                    "GET",
                    "/scim/v2/Groups",
                    params={"startIndex": 1, "count": 100},
                )
                assert result["totalResults"] == 2

    @pytest.mark.asyncio
    async def test_async_list_groups_with_filter(self) -> None:
        """List groups asynchronously with filter."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
                "totalResults": 1,
                "Resources": [{"id": "group-001", "displayName": "Engineering"}],
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.scim.list_groups(
                    start_index=1,
                    count=25,
                    filter='displayName sw "Eng"',
                )

                call_kwargs = mock_request.call_args[1]
                assert call_kwargs["params"]["filter"] == 'displayName sw "Eng"'

    @pytest.mark.asyncio
    async def test_async_get_group(self) -> None:
        """Get group asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "Engineering",
                "members": [{"value": "user-001"}],
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.scim.get_group("group-001")

                mock_request.assert_called_once_with("GET", "/scim/v2/Groups/group-001")
                assert result["displayName"] == "Engineering"

    @pytest.mark.asyncio
    async def test_async_create_group(self) -> None:
        """Create group asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-new-001",
                "displayName": "New Team",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                group_data = {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                    "displayName": "New Team",
                }
                result = await client.scim.create_group(group_data)

                mock_request.assert_called_once_with(
                    "POST",
                    "/scim/v2/Groups",
                    json=group_data,
                )
                assert result["displayName"] == "New Team"

    @pytest.mark.asyncio
    async def test_async_replace_group(self) -> None:
        """Replace group asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "Engineering Team",
                "members": [],
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                group_data = {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                    "displayName": "Engineering Team",
                    "members": [],
                }
                result = await client.scim.replace_group("group-001", group_data)

                mock_request.assert_called_once_with(
                    "PUT",
                    "/scim/v2/Groups/group-001",
                    json=group_data,
                )
                assert result["displayName"] == "Engineering Team"

    @pytest.mark.asyncio
    async def test_async_patch_group(self) -> None:
        """Patch group asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "id": "group-001",
                "displayName": "Engineering",
                "members": [
                    {"value": "user-001"},
                    {"value": "user-005"},
                ],
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                patch_ops = {
                    "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                    "Operations": [
                        {"op": "add", "path": "members", "value": [{"value": "user-005"}]},
                    ],
                }
                result = await client.scim.patch_group("group-001", patch_ops)

                mock_request.assert_called_once_with(
                    "PATCH",
                    "/scim/v2/Groups/group-001",
                    json=patch_ops,
                )
                assert len(result["members"]) == 2

    @pytest.mark.asyncio
    async def test_async_delete_group(self) -> None:
        """Delete group asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = None

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.scim.delete_group("group-001")

                mock_request.assert_called_once_with("DELETE", "/scim/v2/Groups/group-001")
                assert result is None


# =========================================================================
# Fixture-based Tests
# =========================================================================


class TestSCIMWithFixtures:
    """Tests using shared fixtures from conftest.py."""

    def test_list_users_with_fixture(self, client: AragoraClient, mock_request) -> None:
        """Test list_users using fixtures."""
        mock_request.return_value = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": 0,
            "Resources": [],
        }

        result = client.scim.list_users()

        mock_request.assert_called_once_with(
            "GET",
            "/scim/v2/Users",
            params={"startIndex": 1, "count": 100},
        )
        assert result["totalResults"] == 0

    def test_get_user_with_fixture(self, client: AragoraClient, mock_request) -> None:
        """Test get_user using fixtures."""
        mock_request.return_value = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "id": "user-fixture",
            "userName": "fixture@example.com",
        }

        result = client.scim.get_user("user-fixture")

        mock_request.assert_called_once_with("GET", "/scim/v2/Users/user-fixture")
        assert result["id"] == "user-fixture"

    def test_list_groups_with_fixture(self, client: AragoraClient, mock_request) -> None:
        """Test list_groups using fixtures."""
        mock_request.return_value = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": 1,
            "Resources": [{"id": "group-fixture", "displayName": "Test Group"}],
        }

        result = client.scim.list_groups()

        mock_request.assert_called_once_with(
            "GET",
            "/scim/v2/Groups",
            params={"startIndex": 1, "count": 100},
        )
        assert result["totalResults"] == 1


class TestAsyncSCIMWithFixtures:
    """Async tests using shared fixtures."""

    @pytest.mark.asyncio
    async def test_async_list_users_with_fixture(self, mock_async_request) -> None:
        """Test async list_users using fixtures."""
        mock_async_request.return_value = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": 5,
            "Resources": [],
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.scim.list_users()

            assert result["totalResults"] == 5

    @pytest.mark.asyncio
    async def test_async_create_group_with_fixture(self, mock_async_request) -> None:
        """Test async create_group using fixtures."""
        mock_async_request.return_value = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
            "id": "group-async-fixture",
            "displayName": "Async Test Group",
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            group_data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "displayName": "Async Test Group",
            }
            result = await client.scim.create_group(group_data)

            assert result["displayName"] == "Async Test Group"
