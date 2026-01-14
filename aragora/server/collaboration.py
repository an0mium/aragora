"""
Real-Time Collaboration Support for Aragora.

Enables multiple humans to participate simultaneously in debates with:
- Session management for shared debate spaces
- Participant presence tracking
- Role-based permissions
- Real-time participant awareness
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class ParticipantRole(str, Enum):
    """Roles a participant can have in a collaborative session."""

    VIEWER = "viewer"  # Can watch, cannot interact
    VOTER = "voter"  # Can watch and vote
    CONTRIBUTOR = "contributor"  # Can vote and suggest
    MODERATOR = "moderator"  # Can moderate suggestions, kick users


class SessionState(str, Enum):
    """State of a collaboration session."""

    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    ARCHIVED = "archived"


@dataclass
class Participant:
    """A participant in a collaborative session."""

    user_id: str
    session_id: str
    role: ParticipantRole = ParticipantRole.VIEWER
    display_name: str = ""
    avatar_url: str = ""
    joined_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    is_online: bool = True
    metadata: dict = field(default_factory=dict)

    def can_vote(self) -> bool:
        """Check if participant can vote."""
        return self.role in (
            ParticipantRole.VOTER,
            ParticipantRole.CONTRIBUTOR,
            ParticipantRole.MODERATOR,
        )

    def can_suggest(self) -> bool:
        """Check if participant can make suggestions."""
        return self.role in (ParticipantRole.CONTRIBUTOR, ParticipantRole.MODERATOR)

    def can_moderate(self) -> bool:
        """Check if participant has moderation powers."""
        return self.role == ParticipantRole.MODERATOR

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role.value,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "joined_at": self.joined_at,
            "last_active": self.last_active,
            "is_online": self.is_online,
        }


@dataclass
class CollaborationSession:
    """A collaborative debate session."""

    session_id: str
    debate_id: str
    created_by: str  # user_id of creator
    created_at: float = field(default_factory=time.time)
    state: SessionState = SessionState.ACTIVE
    is_public: bool = False
    max_participants: int = 50
    org_id: str = ""
    title: str = ""
    description: str = ""
    expires_at: Optional[float] = None
    allow_anonymous: bool = False
    require_approval: bool = False
    metadata: dict = field(default_factory=dict)

    # Managed by SessionManager
    participants: dict[str, Participant] = field(default_factory=dict)
    pending_approvals: list[str] = field(default_factory=list)

    @property
    def participant_count(self) -> int:
        """Number of current participants."""
        return len(self.participants)

    @property
    def online_count(self) -> int:
        """Number of online participants."""
        return sum(1 for p in self.participants.values() if p.is_online)

    @property
    def is_full(self) -> bool:
        """Check if session is at capacity."""
        return self.participant_count >= self.max_participants

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self, include_participants: bool = True) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "session_id": self.session_id,
            "debate_id": self.debate_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "state": self.state.value,
            "is_public": self.is_public,
            "max_participants": self.max_participants,
            "org_id": self.org_id,
            "title": self.title,
            "description": self.description,
            "expires_at": self.expires_at,
            "allow_anonymous": self.allow_anonymous,
            "require_approval": self.require_approval,
            "participant_count": self.participant_count,
            "online_count": self.online_count,
        }
        if include_participants:
            result["participants"] = [p.to_dict() for p in self.participants.values()]
        return result


class CollaborationEventType(str, Enum):
    """Event types for collaboration."""

    SESSION_CREATED = "session_created"
    SESSION_CLOSED = "session_closed"
    SESSION_UPDATED = "session_updated"
    PARTICIPANT_JOINED = "participant_joined"
    PARTICIPANT_LEFT = "participant_left"
    PARTICIPANT_UPDATED = "participant_updated"
    PRESENCE_UPDATE = "presence_update"
    TYPING_START = "typing_start"
    TYPING_END = "typing_end"
    ROLE_CHANGED = "role_changed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"


@dataclass
class CollaborationEvent:
    """An event in the collaboration system."""

    type: CollaborationEventType
    session_id: str
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "data": self.data,
        }


class SessionManager:
    """
    Manages collaborative debate sessions.

    Thread-safe implementation with LRU eviction and TTL cleanup.
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        session_ttl: float = 86400.0,  # 24 hours
        presence_timeout: float = 300.0,  # 5 minutes
        cleanup_interval: float = 300.0,  # 5 minutes
    ):
        self._sessions: OrderedDict[str, CollaborationSession] = OrderedDict()
        self._debate_sessions: dict[str, set[str]] = {}  # debate_id -> session_ids
        self._user_sessions: dict[str, set[str]] = {}  # user_id -> session_ids
        self._lock = threading.RLock()
        self._event_handlers: list[Callable[[CollaborationEvent], None]] = []

        self.max_sessions = max_sessions
        self.session_ttl = session_ttl
        self.presence_timeout = presence_timeout
        self.cleanup_interval = cleanup_interval

        self._cleanup_task: Optional[asyncio.Task] = None
        self._access_count = 0
        self._cleanup_threshold = 100

    def add_event_handler(self, handler: Callable[[CollaborationEvent], None]) -> None:
        """Add a handler for collaboration events."""
        self._event_handlers.append(handler)

    def remove_event_handler(self, handler: Callable[[CollaborationEvent], None]) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    def _emit_event(self, event: CollaborationEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")

    def _generate_session_id(self) -> str:
        """Generate a secure session ID."""
        return f"collab-{secrets.token_urlsafe(16)}"

    def _maybe_cleanup(self) -> None:
        """Perform cleanup if threshold reached."""
        self._access_count += 1
        if self._access_count >= self._cleanup_threshold:
            self._access_count = 0
            self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        """Remove expired sessions and offline participants."""
        now = time.time()
        expired_sessions = []

        with self._lock:
            for session_id, session in self._sessions.items():
                # Check session expiration
                if session.is_expired:
                    expired_sessions.append(session_id)
                    continue

                # Check TTL
                if now - session.created_at > self.session_ttl:
                    expired_sessions.append(session_id)
                    continue

                # Update presence for inactive participants
                for user_id, participant in session.participants.items():
                    if (
                        participant.is_online
                        and (now - participant.last_active) > self.presence_timeout
                    ):
                        participant.is_online = False
                        self._emit_event(
                            CollaborationEvent(
                                type=CollaborationEventType.PRESENCE_UPDATE,
                                session_id=session_id,
                                user_id=user_id,
                                data={"is_online": False},
                            )
                        )

            # Remove expired sessions
            for session_id in expired_sessions:
                self._remove_session_internal(session_id)

    def _remove_session_internal(self, session_id: str) -> None:
        """Internal method to remove a session (caller holds lock)."""
        if session_id not in self._sessions:
            return

        session = self._sessions[session_id]

        # Remove from debate index
        if session.debate_id in self._debate_sessions:
            self._debate_sessions[session.debate_id].discard(session_id)
            if not self._debate_sessions[session.debate_id]:
                del self._debate_sessions[session.debate_id]

        # Remove from user index
        for user_id in session.participants:
            if user_id in self._user_sessions:
                self._user_sessions[user_id].discard(session_id)
                if not self._user_sessions[user_id]:
                    del self._user_sessions[user_id]

        del self._sessions[session_id]

    def create_session(
        self,
        debate_id: str,
        created_by: str,
        *,
        title: str = "",
        description: str = "",
        is_public: bool = False,
        max_participants: int = 50,
        org_id: str = "",
        expires_in: Optional[float] = None,
        allow_anonymous: bool = False,
        require_approval: bool = False,
    ) -> CollaborationSession:
        """
        Create a new collaboration session for a debate.

        Args:
            debate_id: The debate this session is for
            created_by: User ID of the creator
            title: Optional session title
            description: Optional session description
            is_public: Whether session is publicly joinable
            max_participants: Maximum number of participants
            org_id: Organization ID for multi-tenant filtering
            expires_in: Optional expiration time in seconds
            allow_anonymous: Whether anonymous users can join
            require_approval: Whether joining requires approval

        Returns:
            The created CollaborationSession
        """
        self._maybe_cleanup()

        with self._lock:
            # Enforce max sessions (LRU eviction)
            while len(self._sessions) >= self.max_sessions:
                oldest_id = next(iter(self._sessions))
                self._remove_session_internal(oldest_id)

            session_id = self._generate_session_id()
            now = time.time()

            session = CollaborationSession(
                session_id=session_id,
                debate_id=debate_id,
                created_by=created_by,
                created_at=now,
                title=title or f"Debate Session {session_id[:8]}",
                description=description,
                is_public=is_public,
                max_participants=max_participants,
                org_id=org_id,
                expires_at=now + expires_in if expires_in else None,
                allow_anonymous=allow_anonymous,
                require_approval=require_approval,
            )

            # Add creator as moderator
            creator_participant = Participant(
                user_id=created_by,
                session_id=session_id,
                role=ParticipantRole.MODERATOR,
                joined_at=now,
                last_active=now,
            )
            session.participants[created_by] = creator_participant

            # Store session
            self._sessions[session_id] = session

            # Update indexes
            if debate_id not in self._debate_sessions:
                self._debate_sessions[debate_id] = set()
            self._debate_sessions[debate_id].add(session_id)

            if created_by not in self._user_sessions:
                self._user_sessions[created_by] = set()
            self._user_sessions[created_by].add(session_id)

            # Emit event
            self._emit_event(
                CollaborationEvent(
                    type=CollaborationEventType.SESSION_CREATED,
                    session_id=session_id,
                    user_id=created_by,
                    data=session.to_dict(include_participants=False),
                )
            )

            logger.info(f"Created collaboration session {session_id} for debate {debate_id}")
            return session

    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a session by ID."""
        self._maybe_cleanup()
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # Move to end (LRU)
                self._sessions.move_to_end(session_id)
            return session

    def get_sessions_for_debate(self, debate_id: str) -> list[CollaborationSession]:
        """Get all active sessions for a debate."""
        with self._lock:
            session_ids = self._debate_sessions.get(debate_id, set())
            return [
                self._sessions[sid]
                for sid in session_ids
                if sid in self._sessions and self._sessions[sid].state == SessionState.ACTIVE
            ]

    def get_sessions_for_user(self, user_id: str) -> list[CollaborationSession]:
        """Get all sessions a user is participating in."""
        with self._lock:
            session_ids = self._user_sessions.get(user_id, set())
            return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    def join_session(
        self,
        session_id: str,
        user_id: str,
        *,
        role: ParticipantRole = ParticipantRole.VOTER,
        display_name: str = "",
        avatar_url: str = "",
    ) -> tuple[bool, str, Optional[Participant]]:
        """
        Join a collaboration session.

        Args:
            session_id: Session to join
            user_id: User joining
            role: Role to assign (may be downgraded based on session settings)
            display_name: Display name for the participant
            avatar_url: Avatar URL for the participant

        Returns:
            Tuple of (success, message, participant or None)
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False, "Session not found", None

            if session.state != SessionState.ACTIVE:
                return False, f"Session is {session.state.value}", None

            if session.is_expired:
                return False, "Session has expired", None

            if session.is_full:
                return False, "Session is full", None

            # Check if already joined
            if user_id in session.participants:
                participant = session.participants[user_id]
                participant.is_online = True
                participant.last_active = time.time()
                return True, "Already in session", participant

            # Check approval requirement
            if session.require_approval and user_id != session.created_by:
                if user_id not in session.pending_approvals:
                    session.pending_approvals.append(user_id)
                    self._emit_event(
                        CollaborationEvent(
                            type=CollaborationEventType.APPROVAL_REQUESTED,
                            session_id=session_id,
                            user_id=user_id,
                            data={"display_name": display_name},
                        )
                    )
                return False, "Approval required - request pending", None

            # Cap role at CONTRIBUTOR for non-moderator joins
            if role == ParticipantRole.MODERATOR and user_id != session.created_by:
                role = ParticipantRole.CONTRIBUTOR

            now = time.time()
            participant = Participant(
                user_id=user_id,
                session_id=session_id,
                role=role,
                display_name=display_name,
                avatar_url=avatar_url,
                joined_at=now,
                last_active=now,
            )

            session.participants[user_id] = participant

            # Update user index
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session_id)

            # Emit event
            self._emit_event(
                CollaborationEvent(
                    type=CollaborationEventType.PARTICIPANT_JOINED,
                    session_id=session_id,
                    user_id=user_id,
                    data=participant.to_dict(),
                )
            )

            logger.info(f"User {user_id} joined session {session_id}")
            return True, "Joined successfully", participant

    def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a collaboration session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            if user_id not in session.participants:
                return False

            del session.participants[user_id]

            # Update user index
            if user_id in self._user_sessions:
                self._user_sessions[user_id].discard(session_id)
                if not self._user_sessions[user_id]:
                    del self._user_sessions[user_id]

            # Emit event
            self._emit_event(
                CollaborationEvent(
                    type=CollaborationEventType.PARTICIPANT_LEFT,
                    session_id=session_id,
                    user_id=user_id,
                )
            )

            logger.info(f"User {user_id} left session {session_id}")
            return True

    def update_presence(
        self,
        session_id: str,
        user_id: str,
        is_online: bool = True,
    ) -> bool:
        """Update participant presence."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            participant = session.participants.get(user_id)
            if not participant:
                return False

            participant.is_online = is_online
            participant.last_active = time.time()

            self._emit_event(
                CollaborationEvent(
                    type=CollaborationEventType.PRESENCE_UPDATE,
                    session_id=session_id,
                    user_id=user_id,
                    data={"is_online": is_online},
                )
            )

            return True

    def set_typing(
        self,
        session_id: str,
        user_id: str,
        is_typing: bool,
        typing_context: str = "",  # e.g., "vote", "suggestion"
    ) -> bool:
        """Set typing indicator for a participant."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            participant = session.participants.get(user_id)
            if not participant:
                return False

            participant.last_active = time.time()

            event_type = (
                CollaborationEventType.TYPING_START
                if is_typing
                else CollaborationEventType.TYPING_END
            )
            self._emit_event(
                CollaborationEvent(
                    type=event_type,
                    session_id=session_id,
                    user_id=user_id,
                    data={"context": typing_context},
                )
            )

            return True

    def change_role(
        self,
        session_id: str,
        target_user_id: str,
        new_role: ParticipantRole,
        changed_by: str,
    ) -> tuple[bool, str]:
        """
        Change a participant's role.

        Only moderators can change roles.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False, "Session not found"

            changer = session.participants.get(changed_by)
            if not changer or not changer.can_moderate():
                return False, "Permission denied"

            target = session.participants.get(target_user_id)
            if not target:
                return False, "Target user not in session"

            # Cannot change creator's role
            if target_user_id == session.created_by:
                return False, "Cannot change session creator's role"

            old_role = target.role
            target.role = new_role

            self._emit_event(
                CollaborationEvent(
                    type=CollaborationEventType.ROLE_CHANGED,
                    session_id=session_id,
                    user_id=target_user_id,
                    data={
                        "old_role": old_role.value,
                        "new_role": new_role.value,
                        "changed_by": changed_by,
                    },
                )
            )

            return True, f"Role changed from {old_role.value} to {new_role.value}"

    def approve_join(
        self,
        session_id: str,
        user_id: str,
        approved_by: str,
        approved: bool = True,
        role: ParticipantRole = ParticipantRole.VOTER,
    ) -> tuple[bool, str]:
        """Approve or deny a join request."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False, "Session not found"

            approver = session.participants.get(approved_by)
            if not approver or not approver.can_moderate():
                return False, "Permission denied"

            if user_id not in session.pending_approvals:
                return False, "No pending approval for this user"

            session.pending_approvals.remove(user_id)

            if approved:
                self._emit_event(
                    CollaborationEvent(
                        type=CollaborationEventType.APPROVAL_GRANTED,
                        session_id=session_id,
                        user_id=user_id,
                        data={"approved_by": approved_by, "role": role.value},
                    )
                )
                # Auto-join after approval
                return self.join_session(session_id, user_id, role=role)[:2]
            else:
                self._emit_event(
                    CollaborationEvent(
                        type=CollaborationEventType.APPROVAL_DENIED,
                        session_id=session_id,
                        user_id=user_id,
                        data={"denied_by": approved_by},
                    )
                )
                return True, "Join request denied"

    def close_session(self, session_id: str, closed_by: str) -> bool:
        """Close a collaboration session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            closer = session.participants.get(closed_by)
            if not closer or not closer.can_moderate():
                return False

            session.state = SessionState.CLOSED

            self._emit_event(
                CollaborationEvent(
                    type=CollaborationEventType.SESSION_CLOSED,
                    session_id=session_id,
                    user_id=closed_by,
                )
            )

            logger.info(f"Session {session_id} closed by {closed_by}")
            return True

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            active_sessions = sum(
                1 for s in self._sessions.values() if s.state == SessionState.ACTIVE
            )
            total_participants = sum(s.participant_count for s in self._sessions.values())
            online_participants = sum(s.online_count for s in self._sessions.values())

            return {
                "total_sessions": len(self._sessions),
                "active_sessions": active_sessions,
                "total_participants": total_participants,
                "online_participants": online_participants,
                "debates_with_sessions": len(self._debate_sessions),
                "users_with_sessions": len(self._user_sessions),
            }


# Global singleton instance
_session_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    with _manager_lock:
        if _session_manager is None:
            _session_manager = SessionManager()
        return _session_manager


__all__ = [
    "ParticipantRole",
    "SessionState",
    "Participant",
    "CollaborationSession",
    "CollaborationEventType",
    "CollaborationEvent",
    "SessionManager",
    "get_session_manager",
]
