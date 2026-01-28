"""
Slack Events API package.

Handles Slack event subscriptions:
- App mentions: Respond when the app is mentioned
- Message events: Handle direct messages

Mixins:

    from aragora.server.handlers.social.slack.events import EventsMixin

    class MyHandler(EventsMixin):
        pass
"""

from .handlers import EventsMixin

__all__ = ["EventsMixin"]
