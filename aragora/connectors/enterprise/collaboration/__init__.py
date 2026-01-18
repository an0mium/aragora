"""
Collaboration Platform Connectors.

Provides integration with collaboration and communication platforms:
- Atlassian Confluence (wiki/documentation)
- Notion (workspaces and databases)
- Slack (channels and messages)
"""

from aragora.connectors.enterprise.collaboration.confluence import ConfluenceConnector
from aragora.connectors.enterprise.collaboration.notion import NotionConnector
from aragora.connectors.enterprise.collaboration.slack import SlackConnector

__all__ = [
    "ConfluenceConnector",
    "NotionConnector",
    "SlackConnector",
]
