# Enterprise Connectors Guide

Business system integrations for enterprise deployments.

## Overview

Enterprise connectors integrate Aragora with critical business systems for data ingestion, workflow automation, and bidirectional communication.

---

## Collaboration Connectors

### Slack Connector

Team communication and notifications.

```python
from aragora.connectors.enterprise.collaboration.slack import SlackConnector

connector = SlackConnector(
    bot_token="xoxb-xxx",  # or SLACK_BOT_TOKEN env var
    signing_secret="xxx",  # For webhook verification
)

# Send message
await connector.send_message(
    channel="#debates",
    text="Decision reached: Approve proposal",
    blocks=[...],  # Rich formatting
)

# Fetch channel history
messages = await connector.get_channel_history(
    channel="C123456",
    limit=100,
)

# Create thread
await connector.reply_in_thread(
    channel="#debates",
    thread_ts="1234567890.123456",
    text="Here's the detailed analysis...",
)
```

**Environment Variables:**
```bash
SLACK_BOT_TOKEN=xoxb-xxx
SLACK_SIGNING_SECRET=xxx
SLACK_APP_TOKEN=xapp-xxx  # For Socket Mode
```

### Microsoft Teams Connector

Teams integration for messaging and channels.

```python
from aragora.connectors.enterprise.collaboration.teams import TeamsConnector

connector = TeamsConnector(
    client_id="xxx",
    client_secret="xxx",
    tenant_id="xxx",
)

# Send message
await connector.send_message(
    team_id="xxx",
    channel_id="xxx",
    content="Decision reached!",
)

# Create adaptive card
await connector.send_card(
    team_id="xxx",
    channel_id="xxx",
    card={
        "type": "AdaptiveCard",
        "body": [...],
    },
)
```

### Jira Connector

Issue tracking and project management.

```python
from aragora.connectors.enterprise.collaboration.jira import JiraConnector

connector = JiraConnector(
    url="https://company.atlassian.net",
    email="user@company.com",
    api_token="xxx",
)

# Create issue from decision
issue = await connector.create_issue(
    project="PROJ",
    summary="Implement feature from debate decision",
    description="Based on debate #123...",
    issue_type="Task",
)

# Search issues
issues = await connector.search(
    jql="project = PROJ AND status = Open",
)

# Add comment
await connector.add_comment(
    issue_key="PROJ-123",
    body="Debate conclusion: ...",
)
```

### Confluence Connector

Documentation and knowledge management.

```python
from aragora.connectors.enterprise.collaboration.confluence import ConfluenceConnector

connector = ConfluenceConnector(
    url="https://company.atlassian.net/wiki",
    email="user@company.com",
    api_token="xxx",
)

# Create page from debate
page = await connector.create_page(
    space="DOC",
    title="Decision Record: Feature X",
    body="<h2>Context</h2><p>...</p>",
    parent_id="123456",
)

# Search content
results = await connector.search(
    cql='space = DOC AND text ~ "authentication"',
)
```

### Linear Connector

Modern issue tracking.

```python
from aragora.connectors.enterprise.collaboration.linear import LinearConnector

connector = LinearConnector(api_key="lin_xxx")

# Create issue
issue = await connector.create_issue(
    team_id="xxx",
    title="Implement decision",
    description="...",
    priority=2,
)
```

### Notion Connector

Workspace and database integration.

```python
from aragora.connectors.enterprise.collaboration.notion import NotionConnector

connector = NotionConnector(api_key="secret_xxx")

# Query database
results = await connector.query_database(
    database_id="xxx",
    filter={"property": "Status", "select": {"equals": "Active"}},
)

# Create page
page = await connector.create_page(
    parent_id="xxx",
    properties={"Name": {"title": [{"text": {"content": "Decision"}}]}},
)
```

### Monday Connector

Work management platform.

```python
from aragora.connectors.enterprise.collaboration.monday import MondayConnector

connector = MondayConnector(api_key="xxx")

# Create item
item = await connector.create_item(
    board_id="xxx",
    item_name="New decision to implement",
    column_values={"status": "Working on it"},
)
```

### Asana Connector

Task management.

```python
from aragora.connectors.enterprise.collaboration.asana import AsanaConnector

connector = AsanaConnector(access_token="xxx")

# Create task
task = await connector.create_task(
    project_gid="xxx",
    name="Implement debate decision",
    notes="...",
)
```

---

## Document Storage Connectors

### Google Drive Connector

```python
from aragora.connectors.enterprise.documents.gdrive import GoogleDriveConnector

connector = GoogleDriveConnector(
    credentials_file="/path/to/credentials.json",
)

# List files
files = await connector.list_files(
    folder_id="xxx",
    mime_types=["application/pdf", "application/vnd.google-apps.document"],
)

# Download file
content = await connector.download_file(file_id="xxx")

# Upload file
file = await connector.upload_file(
    name="decision_record.pdf",
    content=pdf_bytes,
    folder_id="xxx",
)
```

### SharePoint Connector

```python
from aragora.connectors.enterprise.documents.sharepoint import SharePointConnector

connector = SharePointConnector(
    client_id="xxx",
    client_secret="xxx",
    tenant_id="xxx",
    site_url="https://company.sharepoint.com/sites/team",
)

# List documents
docs = await connector.list_documents(
    library="Documents",
    folder_path="/Decisions",
)

# Download document
content = await connector.download_file(
    library="Documents",
    file_path="/Decisions/Q1_Report.docx",
)
```

### OneDrive Connector

```python
from aragora.connectors.enterprise.documents.onedrive import OneDriveConnector

connector = OneDriveConnector(
    client_id="xxx",
    client_secret="xxx",
)

# List files
files = await connector.list_files(path="/Documents")

# Upload file
await connector.upload_file(
    path="/Documents/decision.pdf",
    content=pdf_bytes,
)
```

### Dropbox Connector

```python
from aragora.connectors.enterprise.documents.dropbox import DropboxConnector

connector = DropboxConnector(access_token="xxx")

# List files
files = await connector.list_files(path="/Team/Decisions")

# Download file
content = await connector.download(path="/Team/Decisions/report.pdf")
```

### S3 Connector

```python
from aragora.connectors.enterprise.documents.s3 import S3Connector

connector = S3Connector(
    aws_access_key_id="xxx",
    aws_secret_access_key="xxx",
    region="us-east-1",
)

# List objects
objects = await connector.list_objects(
    bucket="aragora-data",
    prefix="evidence/",
)

# Download object
content = await connector.get_object(
    bucket="aragora-data",
    key="evidence/document.pdf",
)
```

---

## Database Connectors

### PostgreSQL Connector

```python
from aragora.connectors.enterprise.database.postgres import PostgresConnector

connector = PostgresConnector(
    host="localhost",
    port=5432,
    database="aragora",
    user="user",
    password="xxx",
)

# Execute query
results = await connector.query(
    sql="SELECT * FROM decisions WHERE created_at > $1",
    params=["2024-01-01"],
)

# Get schema
schema = await connector.get_schema(tables=["decisions", "debates"])
```

### MongoDB Connector

```python
from aragora.connectors.enterprise.database.mongodb import MongoDBConnector

connector = MongoDBConnector(
    uri="mongodb://user:pass@host:27017",
    database="aragora",
)

# Find documents
docs = await connector.find(
    collection="decisions",
    filter={"status": "approved"},
    limit=100,
)

# Aggregate
results = await connector.aggregate(
    collection="debates",
    pipeline=[
        {"$match": {"created_at": {"$gte": "2024-01-01"}}},
        {"$group": {"_id": "$topic", "count": {"$sum": 1}}},
    ],
)
```

### Snowflake Connector

```python
from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

connector = SnowflakeConnector(
    account="xxx",
    user="xxx",
    password="xxx",
    warehouse="COMPUTE_WH",
    database="ARAGORA",
    schema="PUBLIC",
)

# Execute query
results = await connector.query(
    sql="SELECT * FROM decisions LIMIT 100"
)
```

---

## Healthcare Connectors

### FHIR Connector

HL7 FHIR healthcare data standard.

```python
from aragora.connectors.enterprise.healthcare.fhir import FHIRConnector

connector = FHIRConnector(
    base_url="https://fhir.example.com",
    client_id="xxx",
    client_secret="xxx",
)

# Search patients
patients = await connector.search(
    resource_type="Patient",
    params={"name": "Smith", "birthdate": "1990"},
)

# Get resource
patient = await connector.get(
    resource_type="Patient",
    resource_id="123",
)

# Bundle transaction
result = await connector.transaction(
    entries=[
        {"resource": {...}, "request": {"method": "POST", "url": "Patient"}},
    ],
)
```

---

## Streaming Connectors

### Kafka Connector

```python
from aragora.connectors.enterprise.streaming.kafka import KafkaConnector

connector = KafkaConnector(
    bootstrap_servers="localhost:9092",
    group_id="aragora-consumer",
)

# Consume messages
async for message in connector.consume(topics=["events"]):
    process_message(message)

# Produce message
await connector.produce(
    topic="decisions",
    value={"decision_id": "123", "status": "approved"},
)
```

### RabbitMQ Connector

```python
from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQConnector

connector = RabbitMQConnector(
    url="amqp://user:pass@localhost:5672",
)

# Consume messages
async for message in connector.consume(queue="debates"):
    await process_message(message)
    await message.ack()

# Publish message
await connector.publish(
    exchange="aragora",
    routing_key="decisions",
    message={"decision_id": "123"},
)
```

---

## CRM Connectors

### Salesforce Connector

```python
from aragora.connectors.enterprise.crm.salesforce import SalesforceConnector

connector = SalesforceConnector(
    username="user@company.com",
    password="xxx",
    security_token="xxx",
)

# Query records
accounts = await connector.query(
    soql="SELECT Id, Name FROM Account WHERE CreatedDate > 2024-01-01T00:00:00Z"
)

# Create record
opportunity = await connector.create(
    sobject="Opportunity",
    data={
        "Name": "New Deal",
        "StageName": "Prospecting",
        "CloseDate": "2024-12-31",
    },
)
```

---

## ITSM Connectors

### ServiceNow Connector

```python
from aragora.connectors.enterprise.itsm.servicenow import ServiceNowConnector

connector = ServiceNowConnector(
    instance="company",
    username="user",
    password="xxx",
)

# Create incident
incident = await connector.create_incident(
    short_description="System issue",
    description="Detailed description...",
    urgency=2,
    impact=2,
)

# Query records
records = await connector.query(
    table="incident",
    query="state=1^priority=1",
)
```

---

## Authentication Patterns

### OAuth 2.0 Flow

```python
from aragora.connectors.credentials import OAuth2Provider

provider = OAuth2Provider(
    client_id="xxx",
    client_secret="xxx",
    authorize_url="https://provider.com/oauth/authorize",
    token_url="https://provider.com/oauth/token",
)

# Get authorization URL
auth_url = provider.get_authorization_url(
    redirect_uri="https://aragora.example.com/callback",
    scope=["read", "write"],
)

# Exchange code for token
tokens = await provider.exchange_code(
    code="xxx",
    redirect_uri="https://aragora.example.com/callback",
)

# Refresh token
new_tokens = await provider.refresh(refresh_token=tokens.refresh_token)
```

### Service Account

```python
from aragora.connectors.credentials import ServiceAccountProvider

provider = ServiceAccountProvider(
    credentials_file="/path/to/service-account.json",
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)

token = await provider.get_access_token()
```

---

## See Also

- [Connector Integration Index](../CONNECTOR_INTEGRATION_INDEX.md) - Master connector list
- [Evidence Connectors Guide](EVIDENCE_CONNECTORS.md) - Evidence sources
- [Operational Connectors Guide](OPERATIONAL_CONNECTORS.md) - Operations tools
