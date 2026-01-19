"""Enterprise communication connectors - Gmail, Outlook, etc."""

from .gmail import GmailConnector
from .models import (
    EmailMessage,
    EmailThread,
    EmailAttachment,
    GmailSyncState,
    GmailLabel,
)

__all__ = [
    "GmailConnector",
    "EmailMessage",
    "EmailThread",
    "EmailAttachment",
    "GmailSyncState",
    "GmailLabel",
]
