"""
Team Inbox WebSocket Events and Collaboration.

Real-time events for team inbox collaboration:
- Message assignment/unassignment
- User presence (viewing, typing)
- @mentions and notifications
- Status updates
- Internal notes/comments
- Activity feed

Events:
- team_inbox_message_assigned: Message was assigned to team member
- team_inbox_message_unassigned: Message was unassigned
- team_inbox_status_changed: Message status changed
- team_inbox_user_viewing: User is viewing a message
- team_inbox_user_typing: User is typing a response
- team_inbox_mention: User was @mentioned
- team_inbox_note_added: Internal note added
- team_inbox_comment: Comment on message thread
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TeamInboxEventType(str, Enum):
    """Types of team inbox collaboration events."""

    # Assignment events
    MESSAGE_ASSIGNED = "team_inbox_message_assigned"
    MESSAGE_UNASSIGNED = "team_inbox_message_unassigned"
    MESSAGE_REASSIGNED = "team_inbox_message_reassigned"

    # Status events
    STATUS_CHANGED = "team_inbox_status_changed"
    PRIORITY_CHANGED = "team_inbox_priority_changed"

    # Presence events
    USER_VIEWING = "team_inbox_user_viewing"
    USER_LEFT = "team_inbox_user_left"
    USER_TYPING = "team_inbox_user_typing"
    USER_STOPPED_TYPING = "team_inbox_user_stopped_typing"

    # Collaboration events
    MENTION = "team_inbox_mention"
    NOTE_ADDED = "team_inbox_note_added"
    COMMENT_ADDED = "team_inbox_comment_added"
    TAG_ADDED = "team_inbox_tag_added"
    TAG_REMOVED = "team_inbox_tag_removed"

    # Activity events
    ACTIVITY = "team_inbox_activity"
    NOTIFICATION = "team_inbox_notification"


@dataclass
class TeamMember:
    """A team member in a shared inbox."""

    user_id: str
    email: str
    name: str
    role: str = "member"  # admin, member, viewer
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    avatar_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "userId": self.user_id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "joinedAt": self.joined_at.isoformat(),
            "avatarUrl": self.avatar_url,
        }


@dataclass
class Mention:
    """An @mention in a message or note."""

    id: str
    mentioned_user_id: str
    mentioned_by_user_id: str
    message_id: str
    inbox_id: str
    context: str  # The text around the mention
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "mentionedUserId": self.mentioned_user_id,
            "mentionedByUserId": self.mentioned_by_user_id,
            "messageId": self.message_id,
            "inboxId": self.inbox_id,
            "context": self.context,
            "createdAt": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledgedAt": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class InternalNote:
    """An internal note on a message (not visible to customer)."""

    id: str
    message_id: str
    inbox_id: str
    author_id: str
    author_name: str
    content: str
    mentions: List[str] = field(default_factory=list)  # List of user IDs mentioned
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    is_pinned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "messageId": self.message_id,
            "inboxId": self.inbox_id,
            "authorId": self.author_id,
            "authorName": self.author_name,
            "content": self.content,
            "mentions": self.mentions,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "isPinned": self.is_pinned,
        }


@dataclass
class TeamInboxEvent:
    """A team inbox collaboration event."""

    type: TeamInboxEventType
    inbox_id: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    message_id: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value if isinstance(self.type, TeamInboxEventType) else self.type,
            "inboxId": self.inbox_id,
            "userId": self.user_id,
            "messageId": self.message_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class TeamInboxEmitter:
    """
    Emitter for team inbox WebSocket events.

    Manages inbox subscriptions and broadcasts collaboration events
    to all team members viewing the inbox.
    """

    # Regex pattern for @mentions
    MENTION_PATTERN = re.compile(r"@(\w+(?:\.\w+)?(?:@[\w.-]+)?)")

    def __init__(self):
        # Map inbox_id -> set of websocket connections
        self._inbox_subscriptions: Dict[str, Set[Any]] = {}
        # Map message_id -> set of user_ids currently viewing
        self._message_viewers: Dict[str, Set[str]] = {}
        # Map message_id -> set of user_ids currently typing
        self._message_typers: Dict[str, Set[str]] = {}
        # In-memory storage for notes and mentions
        self._notes: Dict[str, List[InternalNote]] = {}  # message_id -> notes
        self._mentions: Dict[str, List[Mention]] = {}  # user_id -> mentions
        self._team_members: Dict[str, Dict[str, TeamMember]] = {}  # inbox_id -> {user_id: member}
        self._lock = asyncio.Lock()
        self._event_callbacks: List[Callable[[TeamInboxEvent], None]] = []

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def subscribe_to_inbox(self, inbox_id: str, websocket: Any) -> None:
        """Subscribe a WebSocket to inbox events."""
        async with self._lock:
            if inbox_id not in self._inbox_subscriptions:
                self._inbox_subscriptions[inbox_id] = set()
            self._inbox_subscriptions[inbox_id].add(websocket)
            logger.debug(f"[TeamInbox] Subscribed to inbox {inbox_id}")

    async def unsubscribe_from_inbox(self, inbox_id: str, websocket: Any) -> None:
        """Unsubscribe from inbox events."""
        async with self._lock:
            if inbox_id in self._inbox_subscriptions:
                self._inbox_subscriptions[inbox_id].discard(websocket)
                if not self._inbox_subscriptions[inbox_id]:
                    del self._inbox_subscriptions[inbox_id]

    async def emit(self, event: TeamInboxEvent) -> int:
        """Broadcast event to all subscribers of the inbox."""
        inbox_id = event.inbox_id
        sent_count = 0

        # Call registered callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"[TeamInbox] Callback error: {e}")

        async with self._lock:
            clients = self._inbox_subscriptions.get(inbox_id, set()).copy()

        if not clients:
            return 0

        message = event.to_json()
        dead_clients = []

        for websocket in clients:
            try:
                await websocket.send(message)
                sent_count += 1
            except Exception as e:
                logger.debug(f"[TeamInbox] Failed to send to client: {e}")
                dead_clients.append(websocket)

        # Cleanup dead connections
        if dead_clients:
            async with self._lock:
                if inbox_id in self._inbox_subscriptions:
                    for ws in dead_clients:
                        self._inbox_subscriptions[inbox_id].discard(ws)

        return sent_count

    # =========================================================================
    # Team Member Management
    # =========================================================================

    async def add_team_member(
        self,
        inbox_id: str,
        user_id: str,
        email: str,
        name: str,
        role: str = "member",
    ) -> TeamMember:
        """Add a team member to an inbox."""
        member = TeamMember(
            user_id=user_id,
            email=email,
            name=name,
            role=role,
        )
        async with self._lock:
            if inbox_id not in self._team_members:
                self._team_members[inbox_id] = {}
            self._team_members[inbox_id][user_id] = member
        return member

    async def get_team_members(self, inbox_id: str) -> List[TeamMember]:
        """Get all team members for an inbox."""
        async with self._lock:
            members = self._team_members.get(inbox_id, {})
            return list(members.values())

    async def remove_team_member(self, inbox_id: str, user_id: str) -> bool:
        """Remove a team member from an inbox."""
        async with self._lock:
            if inbox_id in self._team_members:
                if user_id in self._team_members[inbox_id]:
                    del self._team_members[inbox_id][user_id]
                    return True
        return False

    # =========================================================================
    # Assignment Events
    # =========================================================================

    async def emit_message_assigned(
        self,
        inbox_id: str,
        message_id: str,
        assigned_to_user_id: str,
        assigned_by_user_id: str,
        assigned_to_name: str,
        assigned_by_name: str,
    ) -> int:
        """Emit message assignment event."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_ASSIGNED,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=assigned_by_user_id,
            data={
                "assignedToUserId": assigned_to_user_id,
                "assignedToName": assigned_to_name,
                "assignedByUserId": assigned_by_user_id,
                "assignedByName": assigned_by_name,
            },
        )
        return await self.emit(event)

    async def emit_message_unassigned(
        self,
        inbox_id: str,
        message_id: str,
        unassigned_by_user_id: str,
        unassigned_by_name: str,
        previous_assignee_id: Optional[str] = None,
    ) -> int:
        """Emit message unassignment event."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.MESSAGE_UNASSIGNED,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=unassigned_by_user_id,
            data={
                "unassignedByUserId": unassigned_by_user_id,
                "unassignedByName": unassigned_by_name,
                "previousAssigneeId": previous_assignee_id,
            },
        )
        return await self.emit(event)

    # =========================================================================
    # Status Events
    # =========================================================================

    async def emit_status_changed(
        self,
        inbox_id: str,
        message_id: str,
        old_status: str,
        new_status: str,
        changed_by_user_id: str,
        changed_by_name: str,
    ) -> int:
        """Emit status change event."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.STATUS_CHANGED,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=changed_by_user_id,
            data={
                "oldStatus": old_status,
                "newStatus": new_status,
                "changedByUserId": changed_by_user_id,
                "changedByName": changed_by_name,
            },
        )
        return await self.emit(event)

    # =========================================================================
    # Presence Events
    # =========================================================================

    async def emit_user_viewing(
        self,
        inbox_id: str,
        message_id: str,
        user_id: str,
        user_name: str,
    ) -> int:
        """Emit user started viewing message event."""
        # Track viewer
        async with self._lock:
            if message_id not in self._message_viewers:
                self._message_viewers[message_id] = set()
            self._message_viewers[message_id].add(user_id)

        event = TeamInboxEvent(
            type=TeamInboxEventType.USER_VIEWING,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=user_id,
            data={
                "userId": user_id,
                "userName": user_name,
                "viewers": list(self._message_viewers.get(message_id, set())),
            },
        )
        return await self.emit(event)

    async def emit_user_left(
        self,
        inbox_id: str,
        message_id: str,
        user_id: str,
        user_name: str,
    ) -> int:
        """Emit user stopped viewing message event."""
        # Remove viewer
        async with self._lock:
            if message_id in self._message_viewers:
                self._message_viewers[message_id].discard(user_id)
            # Also remove from typers
            if message_id in self._message_typers:
                self._message_typers[message_id].discard(user_id)

        event = TeamInboxEvent(
            type=TeamInboxEventType.USER_LEFT,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=user_id,
            data={
                "userId": user_id,
                "userName": user_name,
                "viewers": list(self._message_viewers.get(message_id, set())),
            },
        )
        return await self.emit(event)

    async def emit_user_typing(
        self,
        inbox_id: str,
        message_id: str,
        user_id: str,
        user_name: str,
    ) -> int:
        """Emit user is typing event."""
        async with self._lock:
            if message_id not in self._message_typers:
                self._message_typers[message_id] = set()
            self._message_typers[message_id].add(user_id)

        event = TeamInboxEvent(
            type=TeamInboxEventType.USER_TYPING,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=user_id,
            data={
                "userId": user_id,
                "userName": user_name,
                "typers": list(self._message_typers.get(message_id, set())),
            },
        )
        return await self.emit(event)

    async def emit_user_stopped_typing(
        self,
        inbox_id: str,
        message_id: str,
        user_id: str,
        user_name: str,
    ) -> int:
        """Emit user stopped typing event."""
        async with self._lock:
            if message_id in self._message_typers:
                self._message_typers[message_id].discard(user_id)

        event = TeamInboxEvent(
            type=TeamInboxEventType.USER_STOPPED_TYPING,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=user_id,
            data={
                "userId": user_id,
                "userName": user_name,
                "typers": list(self._message_typers.get(message_id, set())),
            },
        )
        return await self.emit(event)

    async def get_message_viewers(self, message_id: str) -> List[str]:
        """Get list of users currently viewing a message."""
        async with self._lock:
            return list(self._message_viewers.get(message_id, set()))

    # =========================================================================
    # Mentions
    # =========================================================================

    def extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        matches = self.MENTION_PATTERN.findall(text)
        return list(set(matches))

    async def create_mention(
        self,
        mentioned_user_id: str,
        mentioned_by_user_id: str,
        message_id: str,
        inbox_id: str,
        context: str,
    ) -> Mention:
        """Create a new mention and emit event."""
        mention = Mention(
            id=str(uuid.uuid4()),
            mentioned_user_id=mentioned_user_id,
            mentioned_by_user_id=mentioned_by_user_id,
            message_id=message_id,
            inbox_id=inbox_id,
            context=context,
        )

        async with self._lock:
            if mentioned_user_id not in self._mentions:
                self._mentions[mentioned_user_id] = []
            self._mentions[mentioned_user_id].append(mention)

        # Emit mention event
        event = TeamInboxEvent(
            type=TeamInboxEventType.MENTION,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=mentioned_by_user_id,
            data=mention.to_dict(),
        )
        await self.emit(event)

        return mention

    async def get_mentions_for_user(
        self,
        user_id: str,
        unacknowledged_only: bool = False,
    ) -> List[Mention]:
        """Get mentions for a user."""
        async with self._lock:
            mentions = self._mentions.get(user_id, []).copy()

        if unacknowledged_only:
            mentions = [m for m in mentions if not m.acknowledged]

        return mentions

    async def acknowledge_mention(self, user_id: str, mention_id: str) -> bool:
        """Mark a mention as acknowledged."""
        async with self._lock:
            mentions = self._mentions.get(user_id, [])
            for mention in mentions:
                if mention.id == mention_id:
                    mention.acknowledged = True
                    mention.acknowledged_at = datetime.now(timezone.utc)
                    return True
        return False

    # =========================================================================
    # Internal Notes
    # =========================================================================

    async def add_note(
        self,
        inbox_id: str,
        message_id: str,
        author_id: str,
        author_name: str,
        content: str,
    ) -> InternalNote:
        """Add an internal note to a message."""
        # Extract mentions from content
        mentioned_usernames = self.extract_mentions(content)

        note = InternalNote(
            id=str(uuid.uuid4()),
            message_id=message_id,
            inbox_id=inbox_id,
            author_id=author_id,
            author_name=author_name,
            content=content,
            mentions=mentioned_usernames,  # Store usernames, resolve to IDs later
        )

        async with self._lock:
            if message_id not in self._notes:
                self._notes[message_id] = []
            self._notes[message_id].append(note)

        # Emit note added event
        event = TeamInboxEvent(
            type=TeamInboxEventType.NOTE_ADDED,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=author_id,
            data=note.to_dict(),
        )
        await self.emit(event)

        return note

    async def get_notes(self, message_id: str) -> List[InternalNote]:
        """Get all internal notes for a message."""
        async with self._lock:
            return self._notes.get(message_id, []).copy()

    # =========================================================================
    # Activity Feed
    # =========================================================================

    async def emit_activity(
        self,
        inbox_id: str,
        activity_type: str,
        description: str,
        user_id: str,
        user_name: str,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Emit a generic activity event for the feed."""
        event = TeamInboxEvent(
            type=TeamInboxEventType.ACTIVITY,
            inbox_id=inbox_id,
            message_id=message_id,
            user_id=user_id,
            data={
                "activityType": activity_type,
                "description": description,
                "userId": user_id,
                "userName": user_name,
                "metadata": metadata or {},
            },
        )
        return await self.emit(event)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_callback(self, callback: Callable[[TeamInboxEvent], None]) -> None:
        """Add a callback for all events."""
        self._event_callbacks.append(callback)

    def remove_callback(self, callback: Callable[[TeamInboxEvent], None]) -> None:
        """Remove a callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)


# Global emitter instance
_team_inbox_emitter: Optional[TeamInboxEmitter] = None


def get_team_inbox_emitter() -> TeamInboxEmitter:
    """Get the global team inbox emitter instance."""
    global _team_inbox_emitter
    if _team_inbox_emitter is None:
        _team_inbox_emitter = TeamInboxEmitter()
    return _team_inbox_emitter


# Convenience functions
async def emit_message_assigned(inbox_id: str, message_id: str, **kwargs) -> int:
    """Emit message assignment event."""
    return await get_team_inbox_emitter().emit_message_assigned(inbox_id, message_id, **kwargs)


async def emit_message_unassigned(inbox_id: str, message_id: str, **kwargs) -> int:
    """Emit message unassignment event."""
    return await get_team_inbox_emitter().emit_message_unassigned(inbox_id, message_id, **kwargs)


async def emit_status_changed(inbox_id: str, message_id: str, **kwargs) -> int:
    """Emit status change event."""
    return await get_team_inbox_emitter().emit_status_changed(inbox_id, message_id, **kwargs)


async def emit_user_viewing(inbox_id: str, message_id: str, **kwargs) -> int:
    """Emit user viewing event."""
    return await get_team_inbox_emitter().emit_user_viewing(inbox_id, message_id, **kwargs)


async def emit_user_typing(inbox_id: str, message_id: str, **kwargs) -> int:
    """Emit user typing event."""
    return await get_team_inbox_emitter().emit_user_typing(inbox_id, message_id, **kwargs)


__all__ = [
    # Classes
    "TeamInboxEmitter",
    "TeamInboxEvent",
    "TeamInboxEventType",
    "TeamMember",
    "Mention",
    "InternalNote",
    # Factory
    "get_team_inbox_emitter",
    # Convenience functions
    "emit_message_assigned",
    "emit_message_unassigned",
    "emit_status_changed",
    "emit_user_viewing",
    "emit_user_typing",
]
