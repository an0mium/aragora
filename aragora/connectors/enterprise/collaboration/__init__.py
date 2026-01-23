"""
Collaboration Platform Connectors.

Provides integration with collaboration and communication platforms:
- Asana (project and task management)
- Atlassian Confluence (wiki/documentation)
- Atlassian Jira (issue tracking)
- Microsoft Teams (teams, channels, messages)
- Monday.com (work management)
- Notion (workspaces and databases)
- Slack (channels and messages)
"""

from aragora.connectors.enterprise.collaboration.asana import AsanaConnector
from aragora.connectors.enterprise.collaboration.confluence import ConfluenceConnector
from aragora.connectors.enterprise.collaboration.jira import JiraConnector
from aragora.connectors.enterprise.collaboration.monday import MondayConnector
from aragora.connectors.enterprise.collaboration.notion import NotionConnector
from aragora.connectors.enterprise.collaboration.slack import SlackConnector
from aragora.connectors.enterprise.collaboration.teams import TeamsEnterpriseConnector
from aragora.connectors.enterprise.collaboration.linear import (
    LinearConnector,
    LinearCredentials,
    LinearIssue,
    LinearTeam,
    LinearUser,
    IssueState,
    Label,
    Project,
    Cycle,
    Comment as LinearComment,
    LinearError,
    IssuePriority,
    IssueStateType,
    get_mock_issue as get_mock_linear_issue,
    get_mock_team as get_mock_linear_team,
)

__all__ = [
    "AsanaConnector",
    "ConfluenceConnector",
    "JiraConnector",
    "LinearConnector",
    "LinearCredentials",
    "LinearIssue",
    "LinearTeam",
    "LinearUser",
    "IssueState",
    "Label",
    "Project",
    "Cycle",
    "LinearComment",
    "LinearError",
    "IssuePriority",
    "IssueStateType",
    "get_mock_linear_issue",
    "get_mock_linear_team",
    "MondayConnector",
    "NotionConnector",
    "SlackConnector",
    "TeamsEnterpriseConnector",
]
