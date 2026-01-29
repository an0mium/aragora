"""
Tests for SCIM 2.0 Implementation.

Tests cover:
- User schema parsing and serialization
- Group schema parsing and serialization
- Filter parsing and matching
- Server CRUD operations
- Enterprise extension support
- Patch operations
"""

import pytest
from datetime import datetime, timezone

from aragora.auth.scim import (
    SCIMConfig,
    SCIMServer,
    SCIMUser,
    SCIMGroup,
    SCIMGroupMember,
    SCIMName,
    SCIMEmail,
    SCIMPhoneNumber,
    SCIMAddress,
    SCIMEnterprise,
    SCIMMeta,
    SCIMError,
    SCIMListResponse,
    SCIMPatchOperation,
    SCIMPatchRequest,
    SCIMFilterParser,
    SCIMFilter,
)
from aragora.auth.scim.schemas import (
    SCHEMA_USER,
    SCHEMA_GROUP,
    SCHEMA_ENTERPRISE_USER,
    SCIMPatchOp,
)


# =============================================================================
# User Schema Tests
# =============================================================================


class TestSCIMUser:
    """Tests for SCIMUser schema."""

    def test_user_from_dict_minimal(self):
        """Should create user from minimal dict."""
        data = {
            "schemas": [SCHEMA_USER],
            "userName": "john@example.com",
        }

        user = SCIMUser.from_dict(data)

        assert user.user_name == "john@example.com"
        assert user.active is True
        assert SCHEMA_USER in user.schemas

    def test_user_from_dict_full(self):
        """Should create user from full dict."""
        data = {
            "schemas": [SCHEMA_USER],
            "id": "user-123",
            "externalId": "ext-456",
            "userName": "john@example.com",
            "name": {
                "formatted": "John Doe",
                "familyName": "Doe",
                "givenName": "John",
            },
            "displayName": "John D.",
            "active": True,
            "emails": [
                {"value": "john@example.com", "type": "work", "primary": True},
                {"value": "john.personal@gmail.com", "type": "home"},
            ],
            "phoneNumbers": [
                {"value": "+1-555-1234", "type": "work", "primary": True},
            ],
            "roles": [{"value": "admin"}, {"value": "user"}],
        }

        user = SCIMUser.from_dict(data)

        assert user.id == "user-123"
        assert user.external_id == "ext-456"
        assert user.user_name == "john@example.com"
        assert user.name.family_name == "Doe"
        assert user.name.given_name == "John"
        assert user.display_name == "John D."
        assert len(user.emails) == 2
        assert user.emails[0].primary is True
        assert len(user.phone_numbers) == 1
        assert len(user.roles) == 2
        assert "admin" in user.roles

    def test_user_to_dict(self):
        """Should serialize user to dict."""
        user = SCIMUser(
            id="user-123",
            schemas=[SCHEMA_USER],
            user_name="jane@example.com",
            name=SCIMName(given_name="Jane", family_name="Smith"),
            display_name="Jane S.",
            active=True,
            emails=[SCIMEmail(value="jane@example.com", type="work", primary=True)],
        )

        data = user.to_dict()

        assert data["id"] == "user-123"
        assert data["userName"] == "jane@example.com"
        assert data["name"]["givenName"] == "Jane"
        assert data["displayName"] == "Jane S."
        assert data["active"] is True
        assert len(data["emails"]) == 1

    def test_user_password_not_in_output(self):
        """Password should never be in serialized output."""
        user = SCIMUser(
            id="user-123",
            schemas=[SCHEMA_USER],
            user_name="john@example.com",
            password="secret123",
        )

        data = user.to_dict()

        assert "password" not in data

    def test_user_with_enterprise_extension(self):
        """Should handle enterprise extension."""
        data = {
            "schemas": [SCHEMA_USER, SCHEMA_ENTERPRISE_USER],
            "userName": "enterprise@example.com",
            SCHEMA_ENTERPRISE_USER: {
                "employeeNumber": "EMP-123",
                "department": "Engineering",
                "organization": "Aragora",
            },
        }

        user = SCIMUser.from_dict(data)

        assert user.enterprise is not None
        assert user.enterprise.employee_number == "EMP-123"
        assert user.enterprise.department == "Engineering"
        assert SCHEMA_ENTERPRISE_USER in user.schemas

    def test_get_primary_email(self):
        """Should return primary email."""
        user = SCIMUser(
            id="user-123",
            schemas=[SCHEMA_USER],
            user_name="john@example.com",
            emails=[
                SCIMEmail(value="secondary@example.com", type="home"),
                SCIMEmail(value="primary@example.com", type="work", primary=True),
            ],
        )

        assert user.get_primary_email() == "primary@example.com"

    def test_get_primary_email_fallback(self):
        """Should return first email if no primary."""
        user = SCIMUser(
            id="user-123",
            schemas=[SCHEMA_USER],
            user_name="john@example.com",
            emails=[
                SCIMEmail(value="first@example.com", type="work"),
                SCIMEmail(value="second@example.com", type="home"),
            ],
        )

        assert user.get_primary_email() == "first@example.com"


# =============================================================================
# Group Schema Tests
# =============================================================================


class TestSCIMGroup:
    """Tests for SCIMGroup schema."""

    def test_group_from_dict(self):
        """Should create group from dict."""
        data = {
            "schemas": [SCHEMA_GROUP],
            "id": "group-123",
            "displayName": "Engineering",
            "members": [
                {"value": "user-1", "display": "John Doe"},
                {"value": "user-2", "display": "Jane Smith"},
            ],
        }

        group = SCIMGroup.from_dict(data)

        assert group.id == "group-123"
        assert group.display_name == "Engineering"
        assert len(group.members) == 2
        assert group.members[0].value == "user-1"

    def test_group_to_dict(self):
        """Should serialize group to dict."""
        group = SCIMGroup(
            id="group-456",
            schemas=[SCHEMA_GROUP],
            display_name="Sales",
            members=[
                SCIMGroupMember(value="user-1", display="Bob"),
            ],
        )

        data = group.to_dict()

        assert data["id"] == "group-456"
        assert data["displayName"] == "Sales"
        assert len(data["members"]) == 1


# =============================================================================
# Filter Tests
# =============================================================================


class TestSCIMFilter:
    """Tests for SCIM filter parsing and matching."""

    @pytest.fixture
    def parser(self):
        """Create filter parser."""
        return SCIMFilterParser()

    def test_parse_eq_string(self, parser):
        """Should parse equality filter with string."""
        filter = parser.parse('userName eq "john@example.com"')

        assert filter is not None
        assert filter.attribute == "userName"
        assert filter.operator.value == "eq"
        assert filter.value == "john@example.com"

    def test_parse_eq_boolean(self, parser):
        """Should parse equality filter with boolean."""
        filter = parser.parse("active eq true")

        assert filter is not None
        assert filter.attribute == "active"
        assert filter.value is True

    def test_parse_sw(self, parser):
        """Should parse starts with filter."""
        filter = parser.parse('name.familyName sw "J"')

        assert filter is not None
        assert filter.attribute == "name.familyName"
        assert filter.operator.value == "sw"
        assert filter.value == "J"

    def test_parse_co(self, parser):
        """Should parse contains filter."""
        filter = parser.parse('emails.value co "@example.com"')

        assert filter is not None
        assert filter.operator.value == "co"

    def test_parse_pr(self, parser):
        """Should parse presence filter."""
        filter = parser.parse("title pr")

        assert filter is not None
        assert filter.attribute == "title"
        assert filter.operator.value == "pr"
        assert filter.value is None

    def test_filter_matches_eq(self, parser):
        """Should match equality filter."""
        filter = parser.parse('userName eq "john@example.com"')
        resource = {"userName": "john@example.com", "active": True}

        assert filter.matches(resource) is True

    def test_filter_matches_eq_case_insensitive(self, parser):
        """Should match equality case-insensitively."""
        filter = parser.parse('userName eq "JOHN@example.com"')
        resource = {"userName": "john@example.com"}

        assert filter.matches(resource) is True

    def test_filter_matches_sw(self, parser):
        """Should match starts with filter."""
        filter = parser.parse('displayName sw "John"')
        resource = {"displayName": "John Doe"}

        assert filter.matches(resource) is True

    def test_filter_matches_co(self, parser):
        """Should match contains filter."""
        filter = parser.parse('userName co "example"')
        resource = {"userName": "john@example.com"}

        assert filter.matches(resource) is True

    def test_filter_matches_pr(self, parser):
        """Should match presence filter."""
        filter = parser.parse("title pr")

        assert filter.matches({"title": "Engineer"}) is True
        assert filter.matches({"title": ""}) is False
        assert filter.matches({}) is False

    def test_filter_matches_boolean(self, parser):
        """Should match boolean filter."""
        filter = parser.parse("active eq true")

        assert filter.matches({"active": True}) is True
        assert filter.matches({"active": False}) is False

    def test_filter_not_matches(self, parser):
        """Should not match when filter doesn't apply."""
        filter = parser.parse('userName eq "other@example.com"')
        resource = {"userName": "john@example.com"}

        assert filter.matches(resource) is False

    def test_parse_and(self, parser):
        """Should parse AND expression."""
        filter = parser.parse('userName eq "john" and active eq true')

        assert filter is not None
        # Compound filter

    def test_parse_or(self, parser):
        """Should parse OR expression."""
        filter = parser.parse('userName eq "john" or userName eq "jane"')

        assert filter is not None


# =============================================================================
# Server Tests
# =============================================================================


class TestSCIMServer:
    """Tests for SCIM server operations."""

    @pytest.fixture
    def server(self):
        """Create SCIM server with in-memory stores."""
        config = SCIMConfig(bearer_token="test-token")
        return SCIMServer(config)

    @pytest.mark.asyncio
    async def test_create_user(self, server):
        """Should create a new user."""
        user_data = {
            "schemas": [SCHEMA_USER],
            "userName": "newuser@example.com",
            "name": {
                "givenName": "New",
                "familyName": "User",
            },
            "emails": [
                {"value": "newuser@example.com", "type": "work", "primary": True},
            ],
        }

        result, status = await server.create_user(user_data)

        assert status == 201
        assert result["userName"] == "newuser@example.com"
        assert "id" in result
        assert result["active"] is True

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self, server):
        """Should reject duplicate username."""
        user_data = {
            "schemas": [SCHEMA_USER],
            "userName": "duplicate@example.com",
        }

        # Create first user
        await server.create_user(user_data)

        # Try to create duplicate
        result, status = await server.create_user(user_data)

        assert status == 409
        assert "uniqueness" in result.get("scimType", "").lower()

    @pytest.mark.asyncio
    async def test_get_user(self, server):
        """Should get user by ID."""
        # Create user first
        user_data = {"schemas": [SCHEMA_USER], "userName": "getuser@example.com"}
        create_result, _ = await server.create_user(user_data)
        user_id = create_result["id"]

        # Get user
        result, status = await server.get_user(user_id)

        assert status == 200
        assert result["id"] == user_id
        assert result["userName"] == "getuser@example.com"

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, server):
        """Should return 404 for nonexistent user."""
        result, status = await server.get_user("nonexistent-id")

        assert status == 404

    @pytest.mark.asyncio
    async def test_replace_user(self, server):
        """Should replace user via PUT."""
        # Create user
        user_data = {"schemas": [SCHEMA_USER], "userName": "original@example.com"}
        create_result, _ = await server.create_user(user_data)
        user_id = create_result["id"]

        # Replace user
        new_data = {
            "schemas": [SCHEMA_USER],
            "userName": "updated@example.com",
            "displayName": "Updated User",
        }
        result, status = await server.replace_user(user_id, new_data)

        assert status == 200
        assert result["userName"] == "updated@example.com"
        assert result["displayName"] == "Updated User"

    @pytest.mark.asyncio
    async def test_patch_user_replace_active(self, server):
        """Should patch user active status."""
        # Create user
        user_data = {"schemas": [SCHEMA_USER], "userName": "patchuser@example.com"}
        create_result, _ = await server.create_user(user_data)
        user_id = create_result["id"]

        # Patch user
        patch_data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {"op": "replace", "path": "active", "value": False},
            ],
        }
        result, status = await server.patch_user(user_id, patch_data)

        assert status == 200
        assert result["active"] is False

    @pytest.mark.asyncio
    async def test_delete_user_soft(self, server):
        """Should soft delete user (mark inactive)."""
        # Create user
        user_data = {"schemas": [SCHEMA_USER], "userName": "deleteuser@example.com"}
        create_result, _ = await server.create_user(user_data)
        user_id = create_result["id"]

        # Delete user
        result, status = await server.delete_user(user_id)

        assert status == 204

        # User should still exist but be inactive
        get_result, get_status = await server.get_user(user_id)
        assert get_status == 200
        assert get_result["active"] is False

    @pytest.mark.asyncio
    async def test_list_users(self, server):
        """Should list users with pagination."""
        # Create multiple users
        for i in range(5):
            await server.create_user(
                {
                    "schemas": [SCHEMA_USER],
                    "userName": f"listuser{i}@example.com",
                }
            )

        # List users
        result = await server.list_users(start_index=1, count=3)

        assert result["totalResults"] == 5
        assert result["startIndex"] == 1
        assert len(result["Resources"]) == 3

    @pytest.mark.asyncio
    async def test_list_users_with_filter(self, server):
        """Should filter users."""
        # Create users
        await server.create_user(
            {
                "schemas": [SCHEMA_USER],
                "userName": "findme@example.com",
                "displayName": "Find Me",
            }
        )
        await server.create_user(
            {
                "schemas": [SCHEMA_USER],
                "userName": "other@example.com",
                "displayName": "Other User",
            }
        )

        # Filter by username
        result = await server.list_users(filter_expr='userName eq "findme@example.com"')

        assert result["totalResults"] == 1
        assert result["Resources"][0]["userName"] == "findme@example.com"


# =============================================================================
# Group Server Tests
# =============================================================================


class TestSCIMServerGroups:
    """Tests for SCIM group operations."""

    @pytest.fixture
    def server(self):
        """Create SCIM server with in-memory stores."""
        config = SCIMConfig(bearer_token="test-token", sync_groups=True)
        return SCIMServer(config)

    @pytest.mark.asyncio
    async def test_create_group(self, server):
        """Should create a new group."""
        group_data = {
            "schemas": [SCHEMA_GROUP],
            "displayName": "Engineering",
        }

        result, status = await server.create_group(group_data)

        assert status == 201
        assert result["displayName"] == "Engineering"
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_group_with_members(self, server):
        """Should create group with initial members."""
        # Create user first
        user_result, _ = await server.create_user(
            {
                "schemas": [SCHEMA_USER],
                "userName": "member@example.com",
            }
        )
        user_id = user_result["id"]

        # Create group with member
        group_data = {
            "schemas": [SCHEMA_GROUP],
            "displayName": "Team",
            "members": [
                {"value": user_id, "display": "Member User"},
            ],
        }

        result, status = await server.create_group(group_data)

        assert status == 201
        assert len(result["members"]) == 1
        assert result["members"][0]["value"] == user_id

    @pytest.mark.asyncio
    async def test_get_group(self, server):
        """Should get group by ID."""
        # Create group
        create_result, _ = await server.create_group(
            {
                "schemas": [SCHEMA_GROUP],
                "displayName": "GetGroup",
            }
        )
        group_id = create_result["id"]

        # Get group
        result, status = await server.get_group(group_id)

        assert status == 200
        assert result["id"] == group_id

    @pytest.mark.asyncio
    async def test_patch_group_add_member(self, server):
        """Should add member via PATCH."""
        # Create user
        user_result, _ = await server.create_user(
            {
                "schemas": [SCHEMA_USER],
                "userName": "newmember@example.com",
            }
        )
        user_id = user_result["id"]

        # Create empty group
        group_result, _ = await server.create_group(
            {
                "schemas": [SCHEMA_GROUP],
                "displayName": "PatchGroup",
            }
        )
        group_id = group_result["id"]

        # Patch to add member
        patch_data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "add",
                    "path": "members",
                    "value": [{"value": user_id, "display": "New Member"}],
                },
            ],
        }

        result, status = await server.patch_group(group_id, patch_data)

        assert status == 200
        assert len(result["members"]) == 1

    @pytest.mark.asyncio
    async def test_delete_group(self, server):
        """Should delete group."""
        # Create group
        create_result, _ = await server.create_group(
            {
                "schemas": [SCHEMA_GROUP],
                "displayName": "DeleteGroup",
            }
        )
        group_id = create_result["id"]

        # Delete group
        result, status = await server.delete_group(group_id)

        assert status == 204

        # Verify deleted
        get_result, get_status = await server.get_group(group_id)
        assert get_status == 404


# =============================================================================
# Meta Tests
# =============================================================================


class TestSCIMMeta:
    """Tests for SCIM metadata."""

    def test_meta_to_dict(self):
        """Should serialize meta to dict."""
        meta = SCIMMeta(
            resource_type="User",
            created=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            last_modified=datetime(2024, 1, 16, 14, 45, 0, tzinfo=timezone.utc),
            location="https://example.com/scim/v2/Users/123",
        )

        data = meta.to_dict()

        assert data["resourceType"] == "User"
        assert "2024-01-15" in data["created"]
        assert "2024-01-16" in data["lastModified"]
        assert data["location"] == "https://example.com/scim/v2/Users/123"

    def test_meta_from_dict(self):
        """Should create meta from dict."""
        data = {
            "resourceType": "Group",
            "created": "2024-01-15T10:30:00+00:00",
            "lastModified": "2024-01-16T14:45:00+00:00",
        }

        meta = SCIMMeta.from_dict(data)

        assert meta.resource_type == "Group"
        assert meta.created.year == 2024


# =============================================================================
# Error Tests
# =============================================================================


class TestSCIMError:
    """Tests for SCIM error responses."""

    def test_error_to_dict(self):
        """Should serialize error to dict."""
        from aragora.auth.scim.schemas import SCIMErrorType

        error = SCIMError(
            status=400,
            detail="Invalid value for userName",
            scim_type=SCIMErrorType.INVALID_VALUE,
        )

        data = error.to_dict()

        assert data["status"] == "400"
        assert data["detail"] == "Invalid value for userName"
        assert data["scimType"] == "invalidValue"
        assert "urn:ietf:params:scim:api:messages:2.0:Error" in data["schemas"]


# =============================================================================
# List Response Tests
# =============================================================================


class TestSCIMListResponse:
    """Tests for SCIM list responses."""

    def test_list_response_to_dict(self):
        """Should serialize list response to dict."""
        users = [
            SCIMUser(id="1", schemas=[SCHEMA_USER], user_name="user1@example.com"),
            SCIMUser(id="2", schemas=[SCHEMA_USER], user_name="user2@example.com"),
        ]

        response = SCIMListResponse(
            total_results=10,
            resources=users,
            start_index=1,
            items_per_page=2,
        )

        data = response.to_dict()

        assert data["totalResults"] == 10
        assert data["startIndex"] == 1
        assert data["itemsPerPage"] == 2
        assert len(data["Resources"]) == 2


# =============================================================================
# Patch Operation Tests
# =============================================================================


class TestSCIMPatchOperation:
    """Tests for SCIM PATCH operations."""

    def test_patch_operation_from_dict(self):
        """Should create patch operation from dict."""
        data = {
            "op": "replace",
            "path": "active",
            "value": False,
        }

        op = SCIMPatchOperation.from_dict(data)

        assert op.op == SCIMPatchOp.REPLACE
        assert op.path == "active"
        assert op.value is False

    def test_patch_request_from_dict(self):
        """Should create patch request from dict."""
        data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {"op": "add", "path": "emails", "value": [{"value": "new@example.com"}]},
                {"op": "remove", "path": "nickName"},
            ],
        }

        request = SCIMPatchRequest.from_dict(data)

        assert len(request.operations) == 2
        assert request.operations[0].op == SCIMPatchOp.ADD
        assert request.operations[1].op == SCIMPatchOp.REMOVE


# =============================================================================
# Address and Phone Tests
# =============================================================================


class TestSCIMAddress:
    """Tests for SCIM address component."""

    def test_address_to_dict(self):
        """Should serialize address to dict."""
        address = SCIMAddress(
            street_address="123 Main St",
            locality="San Francisco",
            region="CA",
            postal_code="94102",
            country="USA",
            type="work",
            primary=True,
        )

        data = address.to_dict()

        assert data["streetAddress"] == "123 Main St"
        assert data["locality"] == "San Francisco"
        assert data["region"] == "CA"
        assert data["postalCode"] == "94102"
        assert data["country"] == "USA"
        assert data["type"] == "work"
        assert data["primary"] is True


class TestSCIMPhoneNumber:
    """Tests for SCIM phone number component."""

    def test_phone_to_dict(self):
        """Should serialize phone to dict."""
        phone = SCIMPhoneNumber(
            value="+1-555-1234",
            type="mobile",
            primary=True,
        )

        data = phone.to_dict()

        assert data["value"] == "+1-555-1234"
        assert data["type"] == "mobile"
        assert data["primary"] is True


# =============================================================================
# Enterprise Extension Tests
# =============================================================================


class TestSCIMEnterprise:
    """Tests for enterprise user extension."""

    def test_enterprise_to_dict(self):
        """Should serialize enterprise extension to dict."""
        enterprise = SCIMEnterprise(
            employee_number="EMP-12345",
            department="Engineering",
            organization="Aragora Inc.",
            division="Product",
            cost_center="CC-100",
            manager={"value": "mgr-123", "$ref": "../Users/mgr-123"},
        )

        data = enterprise.to_dict()

        assert data["employeeNumber"] == "EMP-12345"
        assert data["department"] == "Engineering"
        assert data["organization"] == "Aragora Inc."
        assert data["division"] == "Product"
        assert data["costCenter"] == "CC-100"
        assert data["manager"]["value"] == "mgr-123"

    def test_enterprise_from_dict(self):
        """Should create enterprise extension from dict."""
        data = {
            "employeeNumber": "EMP-999",
            "department": "Sales",
        }

        enterprise = SCIMEnterprise.from_dict(data)

        assert enterprise.employee_number == "EMP-999"
        assert enterprise.department == "Sales"
