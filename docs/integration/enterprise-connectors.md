# Enterprise Connectors Guide

This guide covers enterprise connectors for integrating external data sources with the Aragora platform.

## Overview

Enterprise connectors enable:
- **Document Sources**: S3, Google Drive, SharePoint
- **Collaboration**: Slack, Confluence, Notion
- **Databases**: PostgreSQL, MongoDB
- **Git**: GitHub repositories
- **Healthcare**: FHIR-compliant systems

## Quick Start

### Configuring a Connector

```python
from aragora.connectors.enterprise import (
    ConnectorRegistry,
    S3Connector,
    ConnectorConfig,
)

# Create connector configuration
config = {
    "bucket": "my-documents",
    "region": "us-west-2",
    "prefix": "legal/contracts/",
}

# Create and register connector
connector = S3Connector(
    connector_id="legal-docs-s3",
    name="Legal Documents",
    config=config,
)

registry = ConnectorRegistry()
registry.register(connector)

# Sync documents
results = await connector.sync()
```

### Using Sync Store

```python
from aragora.connectors.enterprise import SyncStore, ConnectorConfig

store = SyncStore(database_url="sqlite:///connectors.db")
await store.initialize()

# Save connector configuration
await store.save_connector(ConnectorConfig(
    id="github-main",
    connector_type="github",
    name="Main Repository",
    config={"repo": "org/repo", "branch": "main"},
))

# Get sync history
history = await store.get_sync_history("github-main", limit=10)
```

## Document Connectors

### Amazon S3

```python
from aragora.connectors.enterprise.documents import S3Connector

connector = S3Connector(
    connector_id="s3-docs",
    name="S3 Documents",
    config={
        "bucket": "company-documents",
        "region": "us-east-1",
        "prefix": "contracts/",
        "access_key_id": "AKIAXXXX",  # Or use IAM role
        "secret_access_key": "...",
    },
)

# Sync all documents
results = await connector.sync()

# Sync with filters
results = await connector.sync(
    filters={
        "modified_after": "2024-01-01",
        "file_types": [".pdf", ".docx"],
    },
)
```

### Google Drive

```python
from aragora.connectors.enterprise.documents import GDriveConnector

connector = GDriveConnector(
    connector_id="gdrive-legal",
    name="Legal Drive",
    config={
        "folder_id": "1ABC...",
        "service_account_key": {...},  # Service account JSON
        "include_shared": True,
    },
)

# Authenticate
await connector.authenticate()

# Sync
results = await connector.sync()
```

### SharePoint

```python
from aragora.connectors.enterprise.documents import SharePointConnector

connector = SharePointConnector(
    connector_id="sharepoint-hr",
    name="HR Documents",
    config={
        "site_url": "https://company.sharepoint.com/sites/HR",
        "client_id": "...",
        "client_secret": "...",
        "tenant_id": "...",
        "library": "Documents",
    },
)
```

## Collaboration Connectors

### Slack

```python
from aragora.connectors.enterprise.collaboration import SlackConnector

connector = SlackConnector(
    connector_id="slack-eng",
    name="Engineering Slack",
    config={
        "bot_token": "xoxb-...",
        "channels": ["engineering", "incidents"],
        "sync_threads": True,
        "min_reactions": 3,  # Only sync popular messages
    },
)

# Sync channel history
results = await connector.sync()

# Search messages
messages = await connector.search("deployment issue")
```

### Confluence

```python
from aragora.connectors.enterprise.collaboration import ConfluenceConnector

connector = ConfluenceConnector(
    connector_id="confluence-docs",
    name="Engineering Wiki",
    config={
        "url": "https://company.atlassian.net/wiki",
        "email": "bot@company.com",
        "api_token": "...",
        "spaces": ["ENG", "OPS"],
    },
)

# Sync all pages
results = await connector.sync()

# Sync specific space
results = await connector.sync(filters={"space": "ENG"})
```

### Notion

```python
from aragora.connectors.enterprise.collaboration import NotionConnector

connector = NotionConnector(
    connector_id="notion-wiki",
    name="Company Wiki",
    config={
        "api_key": "secret_...",
        "database_ids": ["db-id-1", "db-id-2"],
        "page_ids": ["page-id-1"],
    },
)
```

## Database Connectors

### PostgreSQL

```python
from aragora.connectors.enterprise.database import PostgresConnector

connector = PostgresConnector(
    connector_id="postgres-main",
    name="Main Database",
    config={
        "host": "db.company.com",
        "port": 5432,
        "database": "production",
        "user": "reader",
        "password": "...",
        "tables": ["customers", "orders"],
        "query": """
            SELECT id, content, metadata
            FROM knowledge_base
            WHERE updated_at > %(last_sync)s
        """,
    },
)
```

### MongoDB

```python
from aragora.connectors.enterprise.database import MongoDBConnector

connector = MongoDBConnector(
    connector_id="mongo-docs",
    name="Document Store",
    config={
        "connection_string": "mongodb://...",
        "database": "documents",
        "collections": ["contracts", "policies"],
        "query": {"status": "active"},
    },
)
```

## Git Connectors

### GitHub

```python
from aragora.connectors.enterprise.git import GitHubConnector

connector = GitHubConnector(
    connector_id="github-main",
    name="Main Repository",
    config={
        "token": "ghp_...",
        "owner": "company",
        "repo": "main-repo",
        "branch": "main",
        "paths": ["docs/", "README.md"],
        "file_types": [".md", ".txt", ".rst"],
    },
)

# Sync repository content
results = await connector.sync()

# Get specific file
content = await connector.fetch("docs/architecture.md")
```

## Healthcare Connectors

### FHIR

```python
from aragora.connectors.enterprise.healthcare import FHIRConnector

connector = FHIRConnector(
    connector_id="fhir-ehr",
    name="EHR System",
    config={
        "base_url": "https://fhir.hospital.com",
        "auth_type": "oauth2",
        "client_id": "...",
        "client_secret": "...",
        "resources": ["Patient", "Observation", "Condition"],
    },
)

# Search patients
patients = await connector.search("Patient", {"name": "John"})

# Get resource
patient = await connector.fetch("Patient/123")
```

## Sync Scheduling

### Configuring Schedules

```python
from aragora.connectors.enterprise.sync import SyncScheduler, SyncSchedule

scheduler = SyncScheduler(store=sync_store)

# Schedule periodic sync
schedule = SyncSchedule(
    connector_id="s3-docs",
    schedule_type="interval",
    interval_hours=4,
)
await scheduler.add_schedule(schedule)

# Schedule cron-based sync
schedule = SyncSchedule(
    connector_id="github-main",
    schedule_type="cron",
    cron_expression="0 */2 * * *",  # Every 2 hours
)
await scheduler.add_schedule(schedule)

# Start scheduler
await scheduler.start()
```

### Managing Schedules

```python
# List schedules
schedules = await scheduler.list_schedules()

# Pause schedule
await scheduler.pause("s3-docs")

# Resume schedule
await scheduler.resume("s3-docs")

# Trigger immediate sync
await scheduler.trigger_sync("s3-docs")
```

## Multi-Tenant Support

```python
connector = S3Connector(
    connector_id="s3-tenant-123",
    name="Tenant Documents",
    config={...},
    tenant_id="tenant-123",
)

# All operations scoped to tenant
results = await connector.sync()

# Query scoped to tenant
docs = await registry.search(
    query="contract",
    tenant_id="tenant-123",
)
```

## Error Handling

### Retry Configuration

```python
connector = S3Connector(
    connector_id="s3-docs",
    name="Documents",
    config={...},
    retry_config={
        "max_retries": 3,
        "retry_delay": 5.0,
        "backoff_multiplier": 2.0,
    },
)
```

### Sync Callbacks

```python
async def on_sync_complete(connector_id: str, result: SyncResult):
    if result.failed_count > 0:
        await notify_admin(f"Sync had {result.failed_count} failures")

connector.on_sync_complete(on_sync_complete)
```

## Metrics

Connectors expose metrics:

- `aragora_connector_sync_total` - Total syncs
- `aragora_connector_sync_items` - Items synced
- `aragora_connector_sync_duration` - Sync duration
- `aragora_connector_errors` - Error count

## Security

### Credential Management

```python
from aragora.connectors.enterprise import CredentialStore

store = CredentialStore(encryption_key="...")

# Store credentials
await store.set("s3-docs", {
    "access_key_id": "...",
    "secret_access_key": "...",
})

# Retrieve credentials
creds = await store.get("s3-docs")
```

### Audit Logging

All sync operations are logged:

```python
# Logs include:
# - Connector ID
# - Sync start/end time
# - Items synced/failed
# - User who triggered (if manual)
```

## API Reference

Core modules:
- `aragora/connectors/enterprise/base.py` - Base connector class
- `aragora/connectors/enterprise/sync_store.py` - Sync state storage
- `aragora/connectors/enterprise/sync/scheduler.py` - Sync scheduling
- `aragora/connectors/enterprise/documents/` - Document connectors
- `aragora/connectors/enterprise/collaboration/` - Collaboration connectors
- `aragora/connectors/enterprise/database/` - Database connectors
- `aragora/connectors/enterprise/git/` - Git connectors
- `aragora/connectors/enterprise/healthcare/` - Healthcare connectors
