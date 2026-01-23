"""Enterprise communication connectors - Gmail, Outlook, etc."""

from .gmail import GmailConnector
from .outlook import OutlookConnector
from .models import (
    EmailMessage,
    EmailThread,
    EmailAttachment,
    GmailSyncState,
    GmailLabel,
    OutlookSyncState,
    OutlookFolder,
)

__all__ = [
    "GmailConnector",
    "OutlookConnector",
    "EmailMessage",
    "EmailThread",
    "EmailAttachment",
    "GmailSyncState",
    "GmailLabel",
    "OutlookSyncState",
    "OutlookFolder",
]
