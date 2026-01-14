"""
HTTP Handlers for Real-Time Collaboration.

Provides REST endpoints for session management, participant tracking,
and collaboration features.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.collaboration import (
    CollaborationSession,
    ParticipantRole,
    SessionManager,
    get_session_manager,
)

logger = logging.getLogger(__name__)


class CollaborationHandlers:
    """
    HTTP handlers for collaboration features.

    Endpoints:
        POST /api/collaboration/sessions - Create session
        GET  /api/collaboration/sessions/{id} - Get session
        GET  /api/collaboration/sessions - List sessions
        POST /api/collaboration/sessions/{id}/join - Join session
        POST /api/collaboration/sessions/{id}/leave - Leave session
        POST /api/collaboration/sessions/{id}/presence - Update presence
        POST /api/collaboration/sessions/{id}/typing - Set typing indicator
        POST /api/collaboration/sessions/{id}/role - Change participant role
        POST /api/collaboration/sessions/{id}/approve - Approve/deny join
        POST /api/collaboration/sessions/{id}/close - Close session
        GET  /api/collaboration/stats - Get statistics
    """

    def __init__(self, manager: SessionManager | None = None):
        self.manager = manager or get_session_manager()

    async def create_session(
        self,
        debate_id: str,
        user_id: str,
        *,
        title: str = "",
        description: str = "",
        is_public: bool = False,
        max_participants: int = 50,
        org_id: str = "",
        expires_in: float | None = None,
        allow_anonymous: bool = False,
        require_approval: bool = False,
    ) -> dict[str, Any]:
        """
        Create a new collaboration session for a debate.

        POST /api/collaboration/sessions
        Body: {
            "debate_id": str,
            "title": str (optional),
            "description": str (optional),
            "is_public": bool (default: false),
            "max_participants": int (default: 50),
            "expires_in": float (seconds, optional),
            "allow_anonymous": bool (default: false),
            "require_approval": bool (default: false)
        }
        """
        if not debate_id:
            return {"error": "debate_id is required"}
        if not user_id:
            return {"error": "user_id is required"}

        try:
            session = self.manager.create_session(
                debate_id=debate_id,
                created_by=user_id,
                title=title,
                description=description,
                is_public=is_public,
                max_participants=max_participants,
                org_id=org_id,
                expires_in=expires_in,
                allow_anonymous=allow_anonymous,
                require_approval=require_approval,
            )
            return {
                "success": True,
                "session": session.to_dict(),
            }
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return {"error": f"Failed to create session: {e}"}

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """
        Get a collaboration session by ID.

        GET /api/collaboration/sessions/{session_id}
        """
        if not session_id:
            return {"error": "session_id is required"}

        session = self.manager.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        return {"session": session.to_dict()}

    async def list_sessions(
        self,
        debate_id: str = "",
        user_id: str = "",
        include_closed: bool = False,
    ) -> dict[str, Any]:
        """
        List collaboration sessions.

        GET /api/collaboration/sessions
        Query params:
            debate_id: Filter by debate (optional)
            user_id: Filter by participant (optional)
            include_closed: Include closed sessions (default: false)
        """
        sessions: list[CollaborationSession] = []

        if debate_id:
            sessions = self.manager.get_sessions_for_debate(debate_id)
        elif user_id:
            sessions = self.manager.get_sessions_for_user(user_id)
        else:
            # Return all active sessions (paginated in production)
            sessions = list(self.manager._sessions.values())

        # Filter closed sessions if not requested
        if not include_closed:
            sessions = [s for s in sessions if s.state.value != "closed"]

        return {
            "sessions": [s.to_dict(include_participants=False) for s in sessions],
            "count": len(sessions),
        }

    async def join_session(
        self,
        session_id: str,
        user_id: str,
        *,
        role: str = "voter",
        display_name: str = "",
        avatar_url: str = "",
    ) -> dict[str, Any]:
        """
        Join a collaboration session.

        POST /api/collaboration/sessions/{session_id}/join
        Body: {
            "role": str (viewer/voter/contributor, default: voter),
            "display_name": str (optional),
            "avatar_url": str (optional)
        }
        """
        if not session_id:
            return {"error": "session_id is required"}
        if not user_id:
            return {"error": "user_id is required"}

        try:
            participant_role = ParticipantRole(role)
        except ValueError:
            return {"error": f"Invalid role: {role}"}

        success, message, participant = self.manager.join_session(
            session_id=session_id,
            user_id=user_id,
            role=participant_role,
            display_name=display_name,
            avatar_url=avatar_url,
        )

        return {
            "success": success,
            "message": message,
            "participant": participant.to_dict() if participant else None,
        }

    async def leave_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Leave a collaboration session.

        POST /api/collaboration/sessions/{session_id}/leave
        """
        if not session_id:
            return {"error": "session_id is required"}
        if not user_id:
            return {"error": "user_id is required"}

        success = self.manager.leave_session(session_id, user_id)
        return {
            "success": success,
            "message": "Left session" if success else "Failed to leave session",
        }

    async def update_presence(
        self,
        session_id: str,
        user_id: str,
        is_online: bool = True,
    ) -> dict[str, Any]:
        """
        Update participant presence.

        POST /api/collaboration/sessions/{session_id}/presence
        Body: {"is_online": bool}
        """
        if not session_id:
            return {"error": "session_id is required"}
        if not user_id:
            return {"error": "user_id is required"}

        success = self.manager.update_presence(session_id, user_id, is_online)
        return {
            "success": success,
            "message": "Presence updated" if success else "Failed to update presence",
        }

    async def set_typing(
        self,
        session_id: str,
        user_id: str,
        is_typing: bool = True,
        context: str = "",
    ) -> dict[str, Any]:
        """
        Set typing indicator.

        POST /api/collaboration/sessions/{session_id}/typing
        Body: {"is_typing": bool, "context": str (e.g., "vote", "suggestion")}
        """
        if not session_id:
            return {"error": "session_id is required"}
        if not user_id:
            return {"error": "user_id is required"}

        success = self.manager.set_typing(session_id, user_id, is_typing, context)
        return {
            "success": success,
            "message": "Typing indicator updated" if success else "Failed to update",
        }

    async def change_role(
        self,
        session_id: str,
        target_user_id: str,
        new_role: str,
        changed_by: str,
    ) -> dict[str, Any]:
        """
        Change a participant's role.

        POST /api/collaboration/sessions/{session_id}/role
        Body: {"target_user_id": str, "new_role": str}
        """
        if not session_id:
            return {"error": "session_id is required"}
        if not target_user_id:
            return {"error": "target_user_id is required"}
        if not changed_by:
            return {"error": "Moderator user_id required"}

        try:
            participant_role = ParticipantRole(new_role)
        except ValueError:
            return {"error": f"Invalid role: {new_role}"}

        success, message = self.manager.change_role(
            session_id=session_id,
            target_user_id=target_user_id,
            new_role=participant_role,
            changed_by=changed_by,
        )

        return {"success": success, "message": message}

    async def approve_join(
        self,
        session_id: str,
        user_id: str,
        approved_by: str,
        approved: bool = True,
        role: str = "voter",
    ) -> dict[str, Any]:
        """
        Approve or deny a join request.

        POST /api/collaboration/sessions/{session_id}/approve
        Body: {"user_id": str, "approved": bool, "role": str (optional)}
        """
        if not session_id:
            return {"error": "session_id is required"}
        if not user_id:
            return {"error": "user_id is required"}
        if not approved_by:
            return {"error": "Moderator user_id required"}

        try:
            participant_role = ParticipantRole(role)
        except ValueError:
            participant_role = ParticipantRole.VOTER

        success, message = self.manager.approve_join(
            session_id=session_id,
            user_id=user_id,
            approved_by=approved_by,
            approved=approved,
            role=participant_role,
        )

        return {"success": success, "message": message}

    async def close_session(
        self,
        session_id: str,
        closed_by: str,
    ) -> dict[str, Any]:
        """
        Close a collaboration session.

        POST /api/collaboration/sessions/{session_id}/close
        """
        if not session_id:
            return {"error": "session_id is required"}
        if not closed_by:
            return {"error": "Moderator user_id required"}

        success = self.manager.close_session(session_id, closed_by)
        return {
            "success": success,
            "message": "Session closed" if success else "Failed to close session",
        }

    async def get_stats(self) -> dict[str, Any]:
        """
        Get collaboration statistics.

        GET /api/collaboration/stats
        """
        return self.manager.get_stats()

    async def get_participants(self, session_id: str) -> dict[str, Any]:
        """
        Get participants for a session.

        GET /api/collaboration/sessions/{session_id}/participants
        """
        if not session_id:
            return {"error": "session_id is required"}

        session = self.manager.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        return {
            "participants": [p.to_dict() for p in session.participants.values()],
            "count": session.participant_count,
            "online_count": session.online_count,
        }


# Singleton handler instance
_handlers: CollaborationHandlers | None = None


def get_collaboration_handlers() -> CollaborationHandlers:
    """Get the global collaboration handlers instance."""
    global _handlers
    if _handlers is None:
        _handlers = CollaborationHandlers()
    return _handlers


__all__ = [
    "CollaborationHandlers",
    "get_collaboration_handlers",
]
