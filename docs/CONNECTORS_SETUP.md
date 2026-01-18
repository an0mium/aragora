# Enterprise Connectors Setup Guide

This guide covers the setup and configuration of Aragora's enterprise connectors for integrating external data sources with the Knowledge Mound.

## Overview

Enterprise connectors enable incremental syncing from various data sources:

| Category | Connectors | Features |
|----------|------------|----------|
| **Git** | GitHub Enterprise | PRs, Issues, Discussions, Code |
| **Documents** | S3, SharePoint, Google Drive | File sync, Office documents |
| **Databases** | PostgreSQL, MongoDB | Table sync, LISTEN/NOTIFY |
| **Collaboration** | Confluence, Notion, Slack | Pages, Workspaces, Channels |
| **Healthcare** | FHIR | HL7 FHIR resources, PHI redaction |

All connectors support:
- Incremental sync with cursor/token persistence
- Credential management (environment, Vault, AWS Secrets Manager)
- Multi-tenant isolation
- Webhook support for real-time updates

## Quick Start

```python
from aragora.connectors.enterprise import (
    GitHubEnterpriseConnector,
    S3Connector,
    PostgreSQLConnector,
    SyncScheduler,
)

# Create a connector
github = GitHubEnterpriseConnector(
    organization="my-org",
    tenant_id="workspace-1",
)

# Run a sync
result = await github.sync()
print(f"Synced {result.items_synced} items")

# Schedule periodic syncs
scheduler = SyncScheduler()
scheduler.add_connector(github, interval_hours=1)
await scheduler.start()
```

## Credential Management

### Environment Variables (Default)

The default `EnvCredentialProvider` reads credentials from environment variables:

```bash
# GitHub
export ARAGORA_GITHUB_TOKEN="ghp_..."

# Google Drive
export ARAGORA_GDRIVE_CLIENT_ID="..."
export ARAGORA_GDRIVE_CLIENT_SECRET="..."
export ARAGORA_GDRIVE_REFRESH_TOKEN="..."

# PostgreSQL
export ARAGORA_POSTGRES_USER="..."
export ARAGORA_POSTGRES_PASSWORD="..."

# S3
export ARAGORA_AWS_ACCESS_KEY_ID="..."
export ARAGORA_AWS_SECRET_ACCESS_KEY="..."

# SharePoint
export ARAGORA_SHAREPOINT_CLIENT_ID="..."
export ARAGORA_SHAREPOINT_CLIENT_SECRET="..."
export ARAGORA_SHAREPOINT_TENANT_ID="..."
```

### Custom Credential Providers

Implement the `CredentialProvider` protocol for custom backends:

```python
from aragora.connectors.enterprise import CredentialProvider

class VaultCredentialProvider:
    def __init__(self, vault_addr: str, vault_token: str):
        self.vault_addr = vault_addr
        self.vault_token = vault_token

    async def get_credential(self, key: str) -> Optional[str]:
        # Fetch from HashiCorp Vault
        async with aiohttp.ClientSession() as session:
            resp = await session.get(
                f"{self.vault_addr}/v1/secret/data/aragora/{key}",
                headers={"X-Vault-Token": self.vault_token}
            )
            data = await resp.json()
            return data.get("data", {}).get("data", {}).get("value")

    async def set_credential(self, key: str, value: str) -> None:
        # Store in Vault
        pass

# Use with connectors
vault = VaultCredentialProvider(
    vault_addr="https://vault.example.com",
    vault_token=os.environ["VAULT_TOKEN"],
)

github = GitHubEnterpriseConnector(
    organization="my-org",
    credentials=vault,
)
```

## Git Connectors

### GitHub Enterprise

```python
from aragora.connectors.enterprise import GitHubEnterpriseConnector

github = GitHubEnterpriseConnector(
    organization="my-org",
    tenant_id="workspace-1",
    # Optional: specify repos (default: all accessible)
    repositories=["repo1", "repo2"],
    # Sync options
    sync_code=True,           # Sync source files
    sync_issues=True,         # Sync issues
    sync_prs=True,            # Sync pull requests
    sync_discussions=True,    # Sync discussions
    sync_wikis=False,         # Sync wiki pages
    # File filters
    include_patterns=["*.py", "*.ts", "*.md"],
    exclude_patterns=["node_modules/**", "*.min.js"],
    # For GitHub Enterprise Server
    base_url="https://github.mycompany.com/api/v3",
)

# Full sync (ignores cursor, syncs everything)
result = await github.sync(full_sync=True)

# Incremental sync (uses saved cursor)
result = await github.sync()
```

**Environment Variables:**
```bash
ARAGORA_GITHUB_TOKEN=ghp_xxxxxxxxxxxx
# Optional for GitHub Enterprise Server
ARAGORA_GITHUB_BASE_URL=https://github.mycompany.com/api/v3
```

### Webhook Setup

Register a webhook in GitHub for real-time updates:

1. Go to your organization/repo settings
2. Add webhook URL: `https://your-aragora-instance/api/webhooks/github`
3. Content type: `application/json`
4. Events: Push, Pull Request, Issues, Discussion
5. Set secret and configure in environment:

```bash
ARAGORA_GITHUB_WEBHOOK_SECRET=your-webhook-secret
```

## Document Connectors

### Google Drive

```python
from aragora.connectors.enterprise import GoogleDriveConnector

gdrive = GoogleDriveConnector(
    tenant_id="workspace-1",
    # Folder to sync (ID or "root" for entire drive)
    root_folder_id="root",
    # Include shared drives
    include_shared_drives=True,
    # Specific shared drives by name or ID
    shared_drives=["Engineering Docs", "Design Assets"],
    # File type filters
    include_mimes=["application/vnd.google-apps.document", "text/plain"],
    exclude_mimes=["image/png"],
    # Path patterns
    include_paths=["Engineering/**", "Design/**"],
    exclude_paths=["Archive/**"],
)

result = await gdrive.sync()
```

**OAuth2 Setup:**

1. Create OAuth2 credentials in Google Cloud Console
2. Enable Drive API
3. Add authorized redirect URI: `http://localhost:8080/oauth/callback`
4. Run OAuth flow to get refresh token:

```python
from aragora.connectors.enterprise.documents.gdrive import get_oauth_url, exchange_code

# Get authorization URL
url = get_oauth_url(
    client_id="your-client-id",
    redirect_uri="http://localhost:8080/oauth/callback",
)
print(f"Visit: {url}")

# After user authorizes, exchange code for tokens
tokens = await exchange_code(
    code="authorization-code-from-callback",
    client_id="your-client-id",
    client_secret="your-client-secret",
    redirect_uri="http://localhost:8080/oauth/callback",
)
print(f"Refresh token: {tokens['refresh_token']}")
```

**Environment Variables:**
```bash
ARAGORA_GDRIVE_CLIENT_ID=your-client-id
ARAGORA_GDRIVE_CLIENT_SECRET=your-client-secret
ARAGORA_GDRIVE_REFRESH_TOKEN=your-refresh-token
```

### S3

```python
from aragora.connectors.enterprise import S3Connector

s3 = S3Connector(
    bucket="my-documents-bucket",
    prefix="engineering/",
    tenant_id="workspace-1",
    # AWS configuration
    region="us-east-1",
    endpoint_url=None,  # Use for MinIO/LocalStack
    # File filters
    include_patterns=["*.pdf", "*.docx", "*.md"],
    exclude_patterns=["*.tmp", "archive/**"],
    # Sync options
    sync_metadata=True,   # Index S3 object metadata
    max_file_size_mb=50,  # Skip files larger than 50MB
)

result = await s3.sync()
```

**Environment Variables:**
```bash
ARAGORA_AWS_ACCESS_KEY_ID=AKIA...
ARAGORA_AWS_SECRET_ACCESS_KEY=...
ARAGORA_AWS_REGION=us-east-1
# Optional for non-AWS S3-compatible storage
ARAGORA_S3_ENDPOINT_URL=http://localhost:9000
```

### SharePoint

```python
from aragora.connectors.enterprise import SharePointConnector

sharepoint = SharePointConnector(
    tenant_id="workspace-1",
    site_url="https://mycompany.sharepoint.com/sites/Engineering",
    # Specific document libraries (default: all)
    libraries=["Documents", "Shared Documents"],
    # Folder paths within libraries
    include_paths=["Projects/**", "Specs/**"],
    exclude_paths=["Archive/**"],
)

result = await sharepoint.sync()
```

**Azure AD Setup:**

1. Register an app in Azure Active Directory
2. Add API permissions: `Sites.Read.All`, `Files.Read.All`
3. Create a client secret

**Environment Variables:**
```bash
ARAGORA_SHAREPOINT_CLIENT_ID=your-app-id
ARAGORA_SHAREPOINT_CLIENT_SECRET=your-client-secret
ARAGORA_SHAREPOINT_TENANT_ID=your-azure-tenant-id
```

## Database Connectors

### PostgreSQL

```python
from aragora.connectors.enterprise import PostgreSQLConnector

postgres = PostgreSQLConnector(
    host="localhost",
    port=5432,
    database="myapp",
    schema="public",
    tenant_id="workspace-1",
    # Tables to sync
    tables=["documents", "articles", "faq"],
    # Column for incremental sync
    timestamp_column="updated_at",
    primary_key_column="id",
    # Columns to include in content
    content_columns=["title", "body", "summary"],
    # Real-time updates via LISTEN/NOTIFY
    notify_channel="document_changes",
    # Connection pool
    pool_size=5,
)

result = await postgres.sync()
```

**LISTEN/NOTIFY Setup:**

Create a trigger for real-time updates:

```sql
-- Create notification function
CREATE OR REPLACE FUNCTION notify_document_changes()
RETURNS TRIGGER AS $$
BEGIN
  PERFORM pg_notify(
    'document_changes',
    json_build_object(
      'table', TG_TABLE_NAME,
      'operation', TG_OP,
      'id', COALESCE(NEW.id, OLD.id)
    )::text
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER documents_notify_trigger
AFTER INSERT OR UPDATE OR DELETE ON documents
FOR EACH ROW EXECUTE FUNCTION notify_document_changes();
```

**Environment Variables:**
```bash
ARAGORA_POSTGRES_USER=myuser
ARAGORA_POSTGRES_PASSWORD=mypassword
```

### MongoDB

```python
from aragora.connectors.enterprise import MongoDBConnector

mongo = MongoDBConnector(
    uri="mongodb://localhost:27017",
    database="myapp",
    tenant_id="workspace-1",
    # Collections to sync
    collections=["documents", "articles"],
    # Field for incremental sync
    timestamp_field="updatedAt",
    # Fields to include in content
    content_fields=["title", "body", "tags"],
    # Watch for real-time updates (requires replica set)
    enable_change_streams=True,
)

result = await mongo.sync()
```

**Environment Variables:**
```bash
ARAGORA_MONGODB_URI=mongodb://user:pass@host:27017/dbname?authSource=admin
```

## Collaboration Connectors

### Confluence

```python
from aragora.connectors.enterprise import ConfluenceConnector

confluence = ConfluenceConnector(
    base_url="https://mycompany.atlassian.net/wiki",
    tenant_id="workspace-1",
    # Spaces to sync (default: all accessible)
    spaces=["ENG", "DESIGN", "PRODUCT"],
    # Include archived pages
    include_archived=False,
    # Include page attachments
    include_attachments=True,
    # Page label filters
    include_labels=["documentation", "public"],
    exclude_labels=["draft", "internal"],
)

result = await confluence.sync()
```

**Environment Variables:**
```bash
ARAGORA_CONFLUENCE_EMAIL=user@company.com
ARAGORA_CONFLUENCE_API_TOKEN=your-api-token
```

### Notion

```python
from aragora.connectors.enterprise import NotionConnector

notion = NotionConnector(
    tenant_id="workspace-1",
    # Database IDs to sync
    databases=["abc123", "def456"],
    # Page IDs to sync (with children)
    root_pages=["page123"],
    # Include database items
    include_database_items=True,
)

result = await notion.sync()
```

**Environment Variables:**
```bash
ARAGORA_NOTION_TOKEN=secret_xxxxx
```

### Slack

```python
from aragora.connectors.enterprise import SlackConnector

slack = SlackConnector(
    tenant_id="workspace-1",
    # Channels to sync
    channels=["engineering", "product", "general"],
    # Channel patterns (regex)
    channel_patterns=["^team-.*", "^project-.*"],
    # Sync options
    include_threads=True,
    include_files=False,  # File attachments
    # Time range
    days_back=90,
)

result = await slack.sync()
```

**Slack App Setup:**

1. Create a Slack app at api.slack.com
2. Add OAuth scopes: `channels:history`, `channels:read`, `users:read`
3. Install to workspace

**Environment Variables:**
```bash
ARAGORA_SLACK_BOT_TOKEN=xoxb-...
ARAGORA_SLACK_USER_TOKEN=xoxp-...  # Optional, for user context
```

## Healthcare Connectors

### FHIR

```python
from aragora.connectors.enterprise import FHIRConnector, PHIRedactor

# Create PHI redactor for HIPAA compliance
redactor = PHIRedactor(
    redact_names=True,
    redact_mrn=True,
    redact_ssn=True,
    redact_dates=True,  # Shift to relative dates
    redact_addresses=True,
    redact_phone=True,
    redact_email=True,
)

fhir = FHIRConnector(
    base_url="https://fhir.myorg.com/r4",
    tenant_id="workspace-1",
    # Resource types to sync
    resource_types=["Patient", "Condition", "Observation", "DiagnosticReport"],
    # PHI redaction
    phi_redactor=redactor,
    # Audit logging
    audit_enabled=True,
    audit_log_path="/var/log/aragora/fhir-audit.log",
    # Incremental sync
    use_since_parameter=True,
)

result = await fhir.sync()
```

**Environment Variables:**
```bash
ARAGORA_FHIR_CLIENT_ID=your-client-id
ARAGORA_FHIR_CLIENT_SECRET=your-client-secret
# Or for SMART on FHIR
ARAGORA_FHIR_ACCESS_TOKEN=your-token
```

## Sync Scheduling

### Basic Scheduling

```python
from aragora.connectors.enterprise import SyncScheduler, SyncSchedule

scheduler = SyncScheduler()

# Add connectors with schedules
scheduler.add_connector(
    github,
    schedule=SyncSchedule(
        interval_hours=1,
        start_time="08:00",  # Start at 8 AM
        end_time="20:00",    # Stop at 8 PM
        timezone="America/New_York",
    ),
)

scheduler.add_connector(
    postgres,
    schedule=SyncSchedule(
        interval_minutes=5,  # Sync every 5 minutes
    ),
)

# Start scheduler
await scheduler.start()

# Check sync history
history = scheduler.get_history(connector_id=github.connector_id, limit=10)
for entry in history:
    print(f"{entry.started_at}: {entry.items_synced} items, {entry.status}")

# Stop scheduler
await scheduler.stop()
```

### Cron-Based Scheduling

```python
scheduler.add_connector(
    confluence,
    schedule=SyncSchedule(
        cron="0 2 * * *",  # Daily at 2 AM
        timezone="UTC",
    ),
)
```

## Multi-Tenant Isolation

Each connector operates within a tenant boundary:

```python
# Workspace 1 connector
github_ws1 = GitHubEnterpriseConnector(
    organization="my-org",
    tenant_id="workspace-1",
)

# Workspace 2 connector (isolated data)
github_ws2 = GitHubEnterpriseConnector(
    organization="my-org",
    tenant_id="workspace-2",
)

# Data is stored separately in Knowledge Mound
await github_ws1.sync()  # -> workspace-1 namespace
await github_ws2.sync()  # -> workspace-2 namespace
```

## Monitoring and Debugging

### Sync Status

```python
# Get current status
status = await github.get_sync_status()
print(f"Status: {status['status']}")
print(f"Last sync: {status['last_sync_at']}")
print(f"Items synced: {status['items_synced']}")
print(f"Errors: {status['errors']}")
```

### Progress Callbacks

```python
def on_progress(synced: int, total: int):
    print(f"Progress: {synced}/{total}")

def on_item(item):
    print(f"Synced: {item.title}")

github.on_progress(on_progress)
github.on_item_synced(on_item)

await github.sync()
```

### Logging

```python
import logging

# Enable connector logging
logging.getLogger("aragora.connectors.enterprise").setLevel(logging.DEBUG)

# Enable specific connector
logging.getLogger("aragora.connectors.enterprise.git").setLevel(logging.DEBUG)
```

## Error Handling

### Retry Logic

Connectors automatically retry failed items. Configure retry behavior:

```python
result = await github.sync(
    max_retries=3,
    retry_delay=5,  # seconds between retries
)

if not result.success:
    print(f"Failed items: {result.items_failed}")
    for error in result.errors:
        print(f"  - {error}")
```

### Cancellation

```python
import asyncio

async def sync_with_timeout():
    task = asyncio.create_task(github.sync())

    try:
        result = await asyncio.wait_for(task, timeout=3600)
    except asyncio.TimeoutError:
        github.cancel_sync()
        print("Sync cancelled due to timeout")
```

## State Persistence

Sync state is persisted to enable incremental syncs:

```python
# Default location: ~/.aragora/sync_state/
github = GitHubEnterpriseConnector(
    organization="my-org",
    # Custom state directory
    state_dir=Path("/var/lib/aragora/sync_state"),
)

# View saved state
state = github.load_state()
print(f"Cursor: {state.cursor}")
print(f"Last item: {state.last_item_id}")

# Reset state for full re-sync
result = await github.sync(full_sync=True)
```

## Best Practices

### Security

1. **Use secret managers** for production credentials (Vault, AWS Secrets Manager)
2. **Rotate tokens** regularly
3. **Limit permissions** to minimum required (read-only when possible)
4. **Enable audit logging** for compliance
5. **Redact PHI** when syncing healthcare data

### Performance

1. **Use incremental sync** instead of full sync when possible
2. **Filter content** at the source (use include/exclude patterns)
3. **Adjust batch sizes** based on memory constraints
4. **Schedule off-peak** for large syncs
5. **Monitor sync duration** and adjust intervals

### Reliability

1. **Set reasonable timeouts** for long-running syncs
2. **Monitor for errors** and alert on failures
3. **Test with small datasets** before full sync
4. **Keep state backups** for disaster recovery

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Auth errors | Verify credentials and permissions |
| Rate limiting | Reduce sync frequency or use exponential backoff |
| Timeout errors | Increase timeout or reduce batch size |
| Missing data | Check include/exclude patterns |
| Stale data | Verify cursor is advancing |

### Debug Mode

```python
import os

# Enable debug mode
os.environ["ARAGORA_DEBUG"] = "1"

# Run sync with verbose logging
result = await github.sync()
```

### Reset Sync State

```python
import os
from pathlib import Path

# Remove state file to force full re-sync
state_path = Path.home() / ".aragora" / "sync_state" / f"{github.connector_id}_{github.tenant_id}.json"
if state_path.exists():
    state_path.unlink()

result = await github.sync()  # Will do full sync
```

## Related Documentation

- [Control Plane Guide](./CONTROL_PLANE_GUIDE.md)
- [Knowledge Mound Architecture](./ADR/014-knowledge-mound-architecture.md)
- [API Reference](./API_REFERENCE.md)
