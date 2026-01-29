"""
SCIM 2.0 HTTP Server.

Implements RFC 7644 (SCIM Protocol) HTTP endpoints:
- GET /Users - List users with filtering and pagination
- POST /Users - Create user
- GET /Users/{id} - Get user by ID
- PUT /Users/{id} - Replace user
- PATCH /Users/{id} - Partial update user
- DELETE /Users/{id} - Delete user
- GET /Groups - List groups
- POST /Groups - Create group
- GET /Groups/{id} - Get group by ID
- PUT /Groups/{id} - Replace group
- PATCH /Groups/{id} - Partial update group
- DELETE /Groups/{id} - Delete group
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from .filters import SCIMFilterParser
from .schemas import (
    SCIMError,
    SCIMErrorType,
    SCIMGroup,
    SCIMGroupMember,
    SCIMListResponse,
    SCIMMeta,
    SCIMPatchOp,
    SCIMPatchRequest,
    SCIMUser,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SCIMConfig:
    """SCIM server configuration."""

    # Authentication
    bearer_token: str = ""  # Required for production
    tenant_id: str | None = None  # For multi-tenant deployments

    # Pagination
    default_page_size: int = 100
    max_page_size: int = 1000

    # Base URL for resource locations
    base_url: str = ""

    # Feature flags
    allow_password_sync: bool = False  # Set passwords via SCIM
    soft_delete: bool = True  # Mark inactive vs hard delete
    sync_groups: bool = True  # Enable group provisioning

    # Rate limiting
    requests_per_minute: int = 100


# =============================================================================
# User Store Interface
# =============================================================================


class SCIMUserStore(Protocol):
    """Interface for SCIM user storage backend."""

    async def create_user(self, user: SCIMUser) -> SCIMUser:
        """Create a new user."""
        ...

    async def get_user(self, user_id: str) -> SCIMUser | None:
        """Get a user by ID."""
        ...

    async def get_user_by_username(self, username: str) -> SCIMUser | None:
        """Get a user by username."""
        ...

    async def get_user_by_external_id(self, external_id: str) -> SCIMUser | None:
        """Get a user by external ID."""
        ...

    async def update_user(self, user_id: str, user: SCIMUser) -> SCIMUser | None:
        """Update an existing user."""
        ...

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        ...

    async def list_users(
        self,
        start_index: int = 1,
        count: int = 100,
        filter_expr: str | None = None,
    ) -> tuple[list[SCIMUser], int]:
        """List users with pagination and filtering."""
        ...


class SCIMGroupStore(Protocol):
    """Interface for SCIM group storage backend."""

    async def create_group(self, group: SCIMGroup) -> SCIMGroup:
        """Create a new group."""
        ...

    async def get_group(self, group_id: str) -> SCIMGroup | None:
        """Get a group by ID."""
        ...

    async def get_group_by_display_name(self, display_name: str) -> SCIMGroup | None:
        """Get a group by display name."""
        ...

    async def update_group(self, group_id: str, group: SCIMGroup) -> SCIMGroup | None:
        """Update an existing group."""
        ...

    async def delete_group(self, group_id: str) -> bool:
        """Delete a group."""
        ...

    async def list_groups(
        self,
        start_index: int = 1,
        count: int = 100,
        filter_expr: str | None = None,
    ) -> tuple[list[SCIMGroup], int]:
        """List groups with pagination and filtering."""
        ...

    async def add_member(self, group_id: str, member: SCIMGroupMember) -> bool:
        """Add a member to a group."""
        ...

    async def remove_member(self, group_id: str, member_id: str) -> bool:
        """Remove a member from a group."""
        ...


# =============================================================================
# In-Memory Store (for testing)
# =============================================================================


class InMemoryUserStore:
    """In-memory SCIM user store for testing."""

    def __init__(self) -> None:
        self._users: dict[str, SCIMUser] = {}
        self._filter_parser = SCIMFilterParser()

    async def create_user(self, user: SCIMUser) -> SCIMUser:
        """Create a new user."""
        if not user.id:
            user.id = str(uuid4())
        user.meta = SCIMMeta(
            resource_type="User",
            created=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc),
        )
        self._users[user.id] = user
        return user

    async def get_user(self, user_id: str) -> SCIMUser | None:
        """Get a user by ID."""
        return self._users.get(user_id)

    async def get_user_by_username(self, username: str) -> SCIMUser | None:
        """Get a user by username."""
        for user in self._users.values():
            if user.user_name.lower() == username.lower():
                return user
        return None

    async def get_user_by_external_id(self, external_id: str) -> SCIMUser | None:
        """Get a user by external ID."""
        for user in self._users.values():
            if user.external_id == external_id:
                return user
        return None

    async def update_user(self, user_id: str, user: SCIMUser) -> SCIMUser | None:
        """Update an existing user."""
        if user_id not in self._users:
            return None
        user.id = user_id
        if user.meta:
            user.meta.last_modified = datetime.now(timezone.utc)
        else:
            existing = self._users[user_id]
            if existing.meta:
                user.meta = existing.meta
                user.meta.last_modified = datetime.now(timezone.utc)
        self._users[user_id] = user
        return user

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False

    async def list_users(
        self,
        start_index: int = 1,
        count: int = 100,
        filter_expr: str | None = None,
    ) -> tuple[list[SCIMUser], int]:
        """List users with pagination and filtering."""
        users = list(self._users.values())

        # Apply filter
        if filter_expr:
            parsed_filter = self._filter_parser.parse(filter_expr)
            if parsed_filter:
                users = [u for u in users if parsed_filter.matches(u.to_dict())]

        total = len(users)

        # Apply pagination (1-indexed)
        start = max(0, start_index - 1)
        end = start + count
        users = users[start:end]

        return users, total


class InMemoryGroupStore:
    """In-memory SCIM group store for testing."""

    def __init__(self) -> None:
        self._groups: dict[str, SCIMGroup] = {}
        self._filter_parser = SCIMFilterParser()

    async def create_group(self, group: SCIMGroup) -> SCIMGroup:
        """Create a new group."""
        if not group.id:
            group.id = str(uuid4())
        group.meta = SCIMMeta(
            resource_type="Group",
            created=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc),
        )
        self._groups[group.id] = group
        return group

    async def get_group(self, group_id: str) -> SCIMGroup | None:
        """Get a group by ID."""
        return self._groups.get(group_id)

    async def get_group_by_display_name(self, display_name: str) -> SCIMGroup | None:
        """Get a group by display name."""
        for group in self._groups.values():
            if group.display_name.lower() == display_name.lower():
                return group
        return None

    async def update_group(self, group_id: str, group: SCIMGroup) -> SCIMGroup | None:
        """Update an existing group."""
        if group_id not in self._groups:
            return None
        group.id = group_id
        if group.meta:
            group.meta.last_modified = datetime.now(timezone.utc)
        else:
            existing = self._groups[group_id]
            if existing.meta:
                group.meta = existing.meta
                group.meta.last_modified = datetime.now(timezone.utc)
        self._groups[group_id] = group
        return group

    async def delete_group(self, group_id: str) -> bool:
        """Delete a group."""
        if group_id in self._groups:
            del self._groups[group_id]
            return True
        return False

    async def list_groups(
        self,
        start_index: int = 1,
        count: int = 100,
        filter_expr: str | None = None,
    ) -> tuple[list[SCIMGroup], int]:
        """List groups with pagination and filtering."""
        groups = list(self._groups.values())

        # Apply filter
        if filter_expr:
            parsed_filter = self._filter_parser.parse(filter_expr)
            if parsed_filter:
                groups = [g for g in groups if parsed_filter.matches(g.to_dict())]

        total = len(groups)

        # Apply pagination (1-indexed)
        start = max(0, start_index - 1)
        end = start + count
        groups = groups[start:end]

        return groups, total

    async def add_member(self, group_id: str, member: SCIMGroupMember) -> bool:
        """Add a member to a group."""
        group = self._groups.get(group_id)
        if not group:
            return False
        # Check if already a member
        for m in group.members:
            if m.value == member.value:
                return True  # Already a member
        group.members.append(member)
        return True

    async def remove_member(self, group_id: str, member_id: str) -> bool:
        """Remove a member from a group."""
        group = self._groups.get(group_id)
        if not group:
            return False
        original_len = len(group.members)
        group.members = [m for m in group.members if m.value != member_id]
        return len(group.members) < original_len


# =============================================================================
# SCIM Server
# =============================================================================


class SCIMServer:
    """
    SCIM 2.0 Protocol Server.

    Implements user and group provisioning endpoints per RFC 7644.

    Usage with FastAPI:
        from fastapi import FastAPI
        from aragora.auth.scim import SCIMServer, SCIMConfig

        app = FastAPI()
        scim = SCIMServer(SCIMConfig(bearer_token="secret"))
        app.include_router(scim.router, prefix="/scim/v2")

    Usage standalone:
        server = SCIMServer(config)
        result = await server.create_user(user_data)
    """

    def __init__(
        self,
        config: SCIMConfig,
        user_store: SCIMUserStore | None = None,
        group_store: SCIMGroupStore | None = None,
    ) -> None:
        """
        Initialize the SCIM server.

        Args:
            config: SCIM server configuration
            user_store: Storage backend for users (defaults to in-memory)
            group_store: Storage backend for groups (defaults to in-memory)
        """
        self.config = config
        self.user_store = user_store or InMemoryUserStore()
        self.group_store = group_store or InMemoryGroupStore()
        self._router = None

    @property
    def router(self):
        """Get the FastAPI router for SCIM endpoints."""
        if self._router is None:
            self._router = self._create_router()
        return self._router

    def _create_router(self):
        """Create FastAPI router with SCIM endpoints."""
        try:
            from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
            from fastapi.responses import JSONResponse
        except ImportError:
            raise ImportError(
                "FastAPI is required for SCIM router. Install with: pip install fastapi"
            )

        router = APIRouter(tags=["SCIM"])

        async def verify_bearer_token(authorization: str = Header(None)) -> None:
            """Verify the Bearer token."""
            if not self.config.bearer_token:
                return  # No auth configured
            if not authorization:
                raise HTTPException(status_code=401, detail="Authorization header required")
            if not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Bearer token required")
            token = authorization[7:]
            if token != self.config.bearer_token:
                raise HTTPException(status_code=401, detail="Invalid bearer token")

        # User endpoints
        @router.get("/Users")
        async def list_users(
            startIndex: int = Query(1, ge=1),
            count: int = Query(100, ge=1, le=1000),
            filter: str = Query(None),
            _: None = Depends(verify_bearer_token),
        ):
            result = await self.list_users(startIndex, count, filter)
            return JSONResponse(content=result, media_type="application/scim+json")

        @router.post("/Users")
        async def create_user(
            request: Request,
            _: None = Depends(verify_bearer_token),
        ):
            body = await request.json()
            result, status = await self.create_user(body)
            return JSONResponse(
                content=result, status_code=status, media_type="application/scim+json"
            )

        @router.get("/Users/{user_id}")
        async def get_user(
            user_id: str,
            _: None = Depends(verify_bearer_token),
        ):
            result, status = await self.get_user(user_id)
            return JSONResponse(
                content=result, status_code=status, media_type="application/scim+json"
            )

        @router.put("/Users/{user_id}")
        async def replace_user(
            user_id: str,
            request: Request,
            _: None = Depends(verify_bearer_token),
        ):
            body = await request.json()
            result, status = await self.replace_user(user_id, body)
            return JSONResponse(
                content=result, status_code=status, media_type="application/scim+json"
            )

        @router.patch("/Users/{user_id}")
        async def patch_user(
            user_id: str,
            request: Request,
            _: None = Depends(verify_bearer_token),
        ):
            body = await request.json()
            result, status = await self.patch_user(user_id, body)
            return JSONResponse(
                content=result, status_code=status, media_type="application/scim+json"
            )

        @router.delete("/Users/{user_id}")
        async def delete_user(
            user_id: str,
            _: None = Depends(verify_bearer_token),
        ):
            result, status = await self.delete_user(user_id)
            if status == 204:
                return JSONResponse(content=None, status_code=204)
            return JSONResponse(
                content=result, status_code=status, media_type="application/scim+json"
            )

        # Group endpoints
        if self.config.sync_groups:

            @router.get("/Groups")
            async def list_groups(
                startIndex: int = Query(1, ge=1),
                count: int = Query(100, ge=1, le=1000),
                filter: str = Query(None),
                _: None = Depends(verify_bearer_token),
            ):
                result = await self.list_groups(startIndex, count, filter)
                return JSONResponse(content=result, media_type="application/scim+json")

            @router.post("/Groups")
            async def create_group(
                request: Request,
                _: None = Depends(verify_bearer_token),
            ):
                body = await request.json()
                result, status = await self.create_group(body)
                return JSONResponse(
                    content=result, status_code=status, media_type="application/scim+json"
                )

            @router.get("/Groups/{group_id}")
            async def get_group(
                group_id: str,
                _: None = Depends(verify_bearer_token),
            ):
                result, status = await self.get_group(group_id)
                return JSONResponse(
                    content=result, status_code=status, media_type="application/scim+json"
                )

            @router.put("/Groups/{group_id}")
            async def replace_group(
                group_id: str,
                request: Request,
                _: None = Depends(verify_bearer_token),
            ):
                body = await request.json()
                result, status = await self.replace_group(group_id, body)
                return JSONResponse(
                    content=result, status_code=status, media_type="application/scim+json"
                )

            @router.patch("/Groups/{group_id}")
            async def patch_group(
                group_id: str,
                request: Request,
                _: None = Depends(verify_bearer_token),
            ):
                body = await request.json()
                result, status = await self.patch_group(group_id, body)
                return JSONResponse(
                    content=result, status_code=status, media_type="application/scim+json"
                )

            @router.delete("/Groups/{group_id}")
            async def delete_group(
                group_id: str,
                _: None = Depends(verify_bearer_token),
            ):
                result, status = await self.delete_group(group_id)
                if status == 204:
                    return JSONResponse(content=None, status_code=204)
                return JSONResponse(
                    content=result, status_code=status, media_type="application/scim+json"
                )

        return router

    # =========================================================================
    # User Operations
    # =========================================================================

    async def list_users(
        self,
        start_index: int = 1,
        count: int = 100,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """List users with pagination and filtering."""
        count = min(count, self.config.max_page_size)

        users, total = await self.user_store.list_users(start_index, count, filter_expr)

        response = SCIMListResponse(
            total_results=total,
            resources=users,
            start_index=start_index,
            items_per_page=len(users),
        )
        return response.to_dict()

    async def create_user(self, data: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Create a new user."""
        try:
            user = SCIMUser.from_dict(data)

            # Check for existing user
            existing = await self.user_store.get_user_by_username(user.user_name)
            if existing:
                error = SCIMError(
                    status=409,
                    detail=f"User with userName '{user.user_name}' already exists",
                    scim_type=SCIMErrorType.UNIQUENESS,
                )
                return error.to_dict(), 409

            # Check external ID uniqueness if provided
            if user.external_id:
                existing = await self.user_store.get_user_by_external_id(user.external_id)
                if existing:
                    error = SCIMError(
                        status=409,
                        detail=f"User with externalId '{user.external_id}' already exists",
                        scim_type=SCIMErrorType.UNIQUENESS,
                    )
                    return error.to_dict(), 409

            # Clear password if not allowed
            if not self.config.allow_password_sync:
                user.password = None

            created_user = await self.user_store.create_user(user)

            logger.info("SCIM: Created user %s (%s)", created_user.id, created_user.user_name)
            return created_user.to_dict(), 201

        except Exception as e:
            logger.error("SCIM: Error creating user: %s", e)
            error = SCIMError(
                status=400,
                detail=str(e),
                scim_type=SCIMErrorType.INVALID_VALUE,
            )
            return error.to_dict(), 400

    async def get_user(self, user_id: str) -> tuple[dict[str, Any], int]:
        """Get a user by ID."""
        user = await self.user_store.get_user(user_id)
        if not user:
            error = SCIMError(
                status=404,
                detail=f"User {user_id} not found",
            )
            return error.to_dict(), 404
        return user.to_dict(), 200

    async def replace_user(
        self,
        user_id: str,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """Replace a user (PUT)."""
        existing = await self.user_store.get_user(user_id)
        if not existing:
            error = SCIMError(
                status=404,
                detail=f"User {user_id} not found",
            )
            return error.to_dict(), 404

        try:
            user = SCIMUser.from_dict(data)
            user.id = user_id

            # Check username uniqueness if changed
            if user.user_name != existing.user_name:
                other = await self.user_store.get_user_by_username(user.user_name)
                if other and other.id != user_id:
                    error = SCIMError(
                        status=409,
                        detail=f"User with userName '{user.user_name}' already exists",
                        scim_type=SCIMErrorType.UNIQUENESS,
                    )
                    return error.to_dict(), 409

            # Clear password if not allowed
            if not self.config.allow_password_sync:
                user.password = None

            updated_user = await self.user_store.update_user(user_id, user)
            if not updated_user:
                error = SCIMError(status=500, detail="Failed to update user")
                return error.to_dict(), 500

            logger.info("SCIM: Replaced user %s", user_id)
            return updated_user.to_dict(), 200

        except Exception as e:
            logger.error("SCIM: Error replacing user %s: %s", user_id, e)
            error = SCIMError(
                status=400,
                detail=str(e),
                scim_type=SCIMErrorType.INVALID_VALUE,
            )
            return error.to_dict(), 400

    async def patch_user(
        self,
        user_id: str,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """Partially update a user (PATCH)."""
        existing = await self.user_store.get_user(user_id)
        if not existing:
            error = SCIMError(
                status=404,
                detail=f"User {user_id} not found",
            )
            return error.to_dict(), 404

        try:
            patch_request = SCIMPatchRequest.from_dict(data)

            # Apply patch operations
            user_dict = existing.to_dict()
            for op in patch_request.operations:
                user_dict = self._apply_patch_operation(user_dict, op)

            # Reconstruct user
            user = SCIMUser.from_dict(user_dict)
            user.id = user_id

            # Clear password if not allowed
            if not self.config.allow_password_sync:
                user.password = None

            updated_user = await self.user_store.update_user(user_id, user)
            if not updated_user:
                error = SCIMError(status=500, detail="Failed to update user")
                return error.to_dict(), 500

            logger.info("SCIM: Patched user %s", user_id)
            return updated_user.to_dict(), 200

        except Exception as e:
            logger.error("SCIM: Error patching user %s: %s", user_id, e)
            error = SCIMError(
                status=400,
                detail=str(e),
                scim_type=SCIMErrorType.INVALID_VALUE,
            )
            return error.to_dict(), 400

    async def delete_user(self, user_id: str) -> tuple[dict[str, Any] | None, int]:
        """Delete a user."""
        existing = await self.user_store.get_user(user_id)
        if not existing:
            error = SCIMError(
                status=404,
                detail=f"User {user_id} not found",
            )
            return error.to_dict(), 404

        if self.config.soft_delete:
            # Soft delete - mark as inactive
            existing.active = False
            await self.user_store.update_user(user_id, existing)
            logger.info("SCIM: Soft-deleted user %s", user_id)
        else:
            # Hard delete
            await self.user_store.delete_user(user_id)
            logger.info("SCIM: Deleted user %s", user_id)

        return None, 204

    # =========================================================================
    # Group Operations
    # =========================================================================

    async def list_groups(
        self,
        start_index: int = 1,
        count: int = 100,
        filter_expr: str | None = None,
    ) -> dict[str, Any]:
        """List groups with pagination and filtering."""
        count = min(count, self.config.max_page_size)

        groups, total = await self.group_store.list_groups(start_index, count, filter_expr)

        response = SCIMListResponse(
            total_results=total,
            resources=groups,
            start_index=start_index,
            items_per_page=len(groups),
        )
        return response.to_dict()

    async def create_group(self, data: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Create a new group."""
        try:
            group = SCIMGroup.from_dict(data)

            # Check for existing group
            existing = await self.group_store.get_group_by_display_name(group.display_name)
            if existing:
                error = SCIMError(
                    status=409,
                    detail=f"Group with displayName '{group.display_name}' already exists",
                    scim_type=SCIMErrorType.UNIQUENESS,
                )
                return error.to_dict(), 409

            created_group = await self.group_store.create_group(group)

            logger.info("SCIM: Created group %s (%s)", created_group.id, created_group.display_name)
            return created_group.to_dict(), 201

        except Exception as e:
            logger.error("SCIM: Error creating group: %s", e)
            error = SCIMError(
                status=400,
                detail=str(e),
                scim_type=SCIMErrorType.INVALID_VALUE,
            )
            return error.to_dict(), 400

    async def get_group(self, group_id: str) -> tuple[dict[str, Any], int]:
        """Get a group by ID."""
        group = await self.group_store.get_group(group_id)
        if not group:
            error = SCIMError(
                status=404,
                detail=f"Group {group_id} not found",
            )
            return error.to_dict(), 404
        return group.to_dict(), 200

    async def replace_group(
        self,
        group_id: str,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """Replace a group (PUT)."""
        existing = await self.group_store.get_group(group_id)
        if not existing:
            error = SCIMError(
                status=404,
                detail=f"Group {group_id} not found",
            )
            return error.to_dict(), 404

        try:
            group = SCIMGroup.from_dict(data)
            group.id = group_id

            updated_group = await self.group_store.update_group(group_id, group)
            if not updated_group:
                error = SCIMError(status=500, detail="Failed to update group")
                return error.to_dict(), 500

            logger.info("SCIM: Replaced group %s", group_id)
            return updated_group.to_dict(), 200

        except Exception as e:
            logger.error("SCIM: Error replacing group %s: %s", group_id, e)
            error = SCIMError(
                status=400,
                detail=str(e),
                scim_type=SCIMErrorType.INVALID_VALUE,
            )
            return error.to_dict(), 400

    async def patch_group(
        self,
        group_id: str,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """Partially update a group (PATCH)."""
        existing = await self.group_store.get_group(group_id)
        if not existing:
            error = SCIMError(
                status=404,
                detail=f"Group {group_id} not found",
            )
            return error.to_dict(), 404

        try:
            patch_request = SCIMPatchRequest.from_dict(data)

            # Handle member operations specially
            for op in patch_request.operations:
                if op.path == "members" or (
                    not op.path and isinstance(op.value, dict) and "members" in op.value
                ):
                    if op.op == SCIMPatchOp.ADD:
                        members = op.value if isinstance(op.value, list) else [op.value]
                        for member_data in members:
                            if isinstance(member_data, dict):
                                member = SCIMGroupMember.from_dict(member_data)
                                await self.group_store.add_member(group_id, member)
                    elif op.op == SCIMPatchOp.REMOVE:
                        if isinstance(op.value, list):
                            for member_data in op.value:
                                if isinstance(member_data, dict):
                                    await self.group_store.remove_member(
                                        group_id, member_data["value"]
                                    )
                        elif op.path and "members[" in op.path:
                            # Parse member ID from path like members[value eq "user-id"]
                            import re

                            match = re.search(r'value eq "([^"]+)"', op.path)
                            if match:
                                await self.group_store.remove_member(group_id, match.group(1))

            # Refresh group
            updated_group = await self.group_store.get_group(group_id)
            if not updated_group:
                error = SCIMError(status=500, detail="Failed to update group")
                return error.to_dict(), 500

            logger.info("SCIM: Patched group %s", group_id)
            return updated_group.to_dict(), 200

        except Exception as e:
            logger.error("SCIM: Error patching group %s: %s", group_id, e)
            error = SCIMError(
                status=400,
                detail=str(e),
                scim_type=SCIMErrorType.INVALID_VALUE,
            )
            return error.to_dict(), 400

    async def delete_group(self, group_id: str) -> tuple[dict[str, Any] | None, int]:
        """Delete a group."""
        existing = await self.group_store.get_group(group_id)
        if not existing:
            error = SCIMError(
                status=404,
                detail=f"Group {group_id} not found",
            )
            return error.to_dict(), 404

        await self.group_store.delete_group(group_id)
        logger.info("SCIM: Deleted group %s", group_id)

        return None, 204

    # =========================================================================
    # Helpers
    # =========================================================================

    def _apply_patch_operation(
        self,
        resource: dict[str, Any],
        operation: Any,
    ) -> dict[str, Any]:
        """Apply a PATCH operation to a resource."""
        from .schemas import SCIMPatchOp, SCIMPatchOperation

        if not isinstance(operation, SCIMPatchOperation):
            return resource

        op = operation.op
        path = operation.path
        value = operation.value

        if op == SCIMPatchOp.ADD:
            if path:
                self._set_path_value(resource, path, value, merge=True)
            elif isinstance(value, dict):
                for k, v in value.items():
                    resource[k] = v

        elif op == SCIMPatchOp.REPLACE:
            if path:
                self._set_path_value(resource, path, value, merge=False)
            elif isinstance(value, dict):
                for k, v in value.items():
                    resource[k] = v

        elif op == SCIMPatchOp.REMOVE:
            if path:
                self._remove_path(resource, path)

        return resource

    def _set_path_value(
        self,
        resource: dict[str, Any],
        path: str,
        value: Any,
        merge: bool = False,
    ) -> None:
        """Set a value at a path in a resource."""
        parts = path.split(".")
        current = resource

        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        final_key = parts[-1]
        if merge and isinstance(current.get(final_key), list) and isinstance(value, list):
            current[final_key].extend(value)
        else:
            current[final_key] = value

    def _remove_path(self, resource: dict[str, Any], path: str) -> None:
        """Remove a value at a path in a resource."""
        parts = path.split(".")
        current = resource

        for part in parts[:-1]:
            if part not in current:
                return
            current = current[part]

        final_key = parts[-1]
        if final_key in current:
            del current[final_key]
