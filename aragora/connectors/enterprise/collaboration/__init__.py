"""
Collaboration Platform Connectors.

Provides integration with collaboration and communication platforms:
- Asana (project and task management)
- Atlassian Confluence (wiki/documentation)
- Atlassian Jira (issue tracking)
- Microsoft Teams (teams, channels, messages)
- Notion (workspaces and databases)
- Slack (channels and messages)
"""

from aragora.connectors.enterprise.collaboration.asana import AsanaConnector
from aragora.connectors.enterprise.collaboration.confluence import ConfluenceConnector
from aragora.connectors.enterprise.collaboration.jira import JiraConnector
from aragora.connectors.enterprise.collaboration.notion import NotionConnector
from aragora.connectors.enterprise.collaboration.slack import SlackConnector
from aragora.connectors.enterprise.collaboration.teams import TeamsEnterpriseConnector

__all__ = [
    "AsanaConnector",
    "ConfluenceConnector",
    "JiraConnector",
    "NotionConnector",
    "SlackConnector",
    "TeamsEnterpriseConnector",
]
