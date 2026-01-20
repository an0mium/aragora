"""
Collaboration Platform Connectors.

Provides integration with collaboration and communication platforms:
- Atlassian Confluence (wiki/documentation)
- Atlassian Jira (issue tracking)
- Notion (workspaces and databases)
- Slack (channels and messages)
"""

from aragora.connectors.enterprise.collaboration.confluence import ConfluenceConnector
from aragora.connectors.enterprise.collaboration.jira import JiraConnector
from aragora.connectors.enterprise.collaboration.notion import NotionConnector
from aragora.connectors.enterprise.collaboration.slack import SlackConnector

__all__ = [
    "ConfluenceConnector",
    "JiraConnector",
    "NotionConnector",
    "SlackConnector",
]
