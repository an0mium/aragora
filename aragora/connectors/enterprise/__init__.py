"""
Enterprise Connectors for the Multi-Agent Control Plane.

Provides crawling and syncing from enterprise data sources:
- Git repositories (GitHub, GitLab, Bitbucket)
- Cloud documents (S3, SharePoint, Google Drive, Google Sheets)
- Databases (PostgreSQL, MongoDB, Snowflake)
- Collaboration platforms (Confluence, Notion, Slack, Jira)
- CRM platforms (Salesforce)
- ITSM platforms (ServiceNow)
- Healthcare systems (FHIR)

All connectors support:
- Incremental sync with cursor/token persistence
- Credential management (env, Vault, AWS Secrets Manager)
- Knowledge Mound ingestion
- Multi-tenant isolation
"""

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncState,
    SyncResult,
    SyncItem,
    CredentialProvider,
    EnvCredentialProvider,
)
from aragora.connectors.enterprise.git import GitHubEnterpriseConnector
from aragora.connectors.enterprise.documents import (
    S3Connector,
    SharePointConnector,
    GoogleDriveConnector,
    OneDriveConnector,
    DropboxConnector,
)
from aragora.connectors.enterprise.database import (
    PostgreSQLConnector,
    MongoDBConnector,
    SnowflakeConnector,
)
from aragora.connectors.enterprise.sync import SyncScheduler, SyncJob, SyncSchedule, SyncHistory
from aragora.connectors.enterprise.healthcare import FHIRConnector, PHIRedactor, FHIRAuditLogger
from aragora.connectors.enterprise.collaboration import (
    ConfluenceConnector,
    NotionConnector,
    SlackConnector,
    JiraConnector,
    TeamsEnterpriseConnector,
)
from aragora.connectors.enterprise.itsm import (
    ServiceNowConnector,
)
from aragora.connectors.enterprise.crm import (
    SalesforceConnector,
)

__all__ = [
    # Base classes
    "EnterpriseConnector",
    "SyncState",
    "SyncResult",
    "SyncItem",
    "CredentialProvider",
    "EnvCredentialProvider",
    # Git connectors
    "GitHubEnterpriseConnector",
    # Document connectors
    "S3Connector",
    "SharePointConnector",
    "GoogleDriveConnector",
    "OneDriveConnector",
    "DropboxConnector",
    # Database connectors
    "PostgreSQLConnector",
    "MongoDBConnector",
    "SnowflakeConnector",
    # Collaboration connectors
    "ConfluenceConnector",
    "NotionConnector",
    "SlackConnector",
    "JiraConnector",
    "TeamsEnterpriseConnector",
    # ITSM connectors
    "ServiceNowConnector",
    # CRM connectors
    "SalesforceConnector",
    # Sync scheduler
    "SyncScheduler",
    "SyncJob",
    "SyncSchedule",
    "SyncHistory",
    # Healthcare connectors
    "FHIRConnector",
    "PHIRedactor",
    "FHIRAuditLogger",
]
