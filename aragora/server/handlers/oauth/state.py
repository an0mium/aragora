"""
OAuth state management.

Handles CSRF state tokens for OAuth flows. Uses Redis in production
for multi-instance support, falls back to in-memory storage in development.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, MutableMapping
from typing import Any

from aragora.server.oauth_state_store import (
    FallbackOAuthStateStore,
    OAuthState,
    get_oauth_state_store,
)
from aragora.server.oauth_state_store import (
    validate_oauth_state as _validate_state_internal,
)

logger = logging.getLogger(__name__)


class _OAuthStatesView(MutableMapping[str, dict[str, Any]]):
    """Compatibility view over OAuth state storage."""

    def __init__(self, store: FallbackOAuthStateStore) -> None:
        self._store = store

    @property
    def _states(self) -> dict[str, OAuthState]:
        return self._store._memory_store._states

    def __getitem__(self, key: str) -> dict[str, Any]:
        value = self._states[key]
        if isinstance(value, OAuthState):
            return value.to_dict()
        if isinstance(value, dict):
            return value
        return {"value": value}

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, OAuthState):
            self._states[key] = value
            return
        if isinstance(value, dict):
            self._states[key] = OAuthState.from_dict(value)
            return
        self._states[key] = value

    def __delitem__(self, key: str) -> None:
        del self._states[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def values(self):
        return [self[k] for k in list(self._states.keys())]

    def items(self):
        return [(k, self[k]) for k in list(self._states.keys())]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._states:
            return self[key]
        return default


# Initialize state store
_state_store = get_oauth_state_store()
_OAUTH_STATES: _OAuthStatesView | dict[str, Any] = {}
try:
    _OAUTH_STATES = _OAuthStatesView(_state_store)
except AttributeError:
    pass  # Keep empty dict fallback

# Legacy constants for backward compatibility
_STATE_TTL_SECONDS = 600  # 10 minutes
MAX_OAUTH_STATES = 10000  # Prevent memory exhaustion


def _validate_state(state: str) -> dict[str, Any] | None:
    """Validate and consume OAuth state token.

    Uses Redis in production for multi-instance support,
    falls back to in-memory storage in development.
    """
    return _validate_state_internal(state)


def _cleanup_expired_states() -> int:
    """Backward-compatible cleanup helper for in-memory states."""
    if isinstance(_OAUTH_STATES, _OAuthStatesView):
        return _OAUTH_STATES._store._memory_store.cleanup_expired()
    return 0
