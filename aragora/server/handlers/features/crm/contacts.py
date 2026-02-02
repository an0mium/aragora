"""Contacts operations mixin for CRM handlers.

This module provides a lightweight stub implementation so the CRM handler
can be imported in minimal environments without optional connector deps.
"""

from __future__ import annotations

from typing import Any


class ContactOperationsMixin:
    """Stub mixin for CRM contact operations.

    The full implementation lives in the CRM feature package. This stub keeps
    imports stable for tests that only validate handler wiring.
    """

    def _contacts_unavailable(self: "Any") -> Any:
        return self._error_response(503, "CRM contacts are not available")

    async def _list_all_contacts(self: "Any", request: Any) -> Any:  # pragma: no cover
        return self._contacts_unavailable()

    async def _list_platform_contacts(
        self: "Any", request: Any, platform: str
    ) -> Any:  # pragma: no cover
        return self._contacts_unavailable()

    async def _get_contact(self: "Any", request: Any, contact_id: str) -> Any:  # pragma: no cover
        return self._contacts_unavailable()

    async def _create_contact(self: "Any", request: Any, platform: str) -> Any:  # pragma: no cover
        return self._contacts_unavailable()

    async def _update_contact(
        self: "Any", request: Any, platform: str, contact_id: str
    ) -> Any:  # pragma: no cover
        return self._contacts_unavailable()

    async def _delete_contact(
        self: "Any", request: Any, platform: str, contact_id: str
    ) -> Any:  # pragma: no cover
        return self._contacts_unavailable()
