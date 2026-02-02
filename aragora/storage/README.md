# Aragora Storage Module

The storage module provides persistent storage backends for the Aragora platform.
It supports multiple backends (SQLite, PostgreSQL, Redis) with a unified interface.

## Architecture

```
storage/
├── Base Classes
│   ├── base_database.py      # BaseDatabase abstract class
│   ├── base_store.py         # SQLiteStore base implementation
│   └── postgres_store.py     # PostgreSQL base implementation
│
├── Backend Configuration
│   ├── backends.py           # DatabaseBackend enum and factory
│   ├── connection_factory.py # Connection pooling
│   ├── connection_router.py  # Read replica routing
│   └── redis_ha.py           # Redis high availability config
│
├── Core Stores
│   ├── user_store/           # User management (SQLite + Postgres)
│   ├── organization_store.py # Organization management
│   ├── audit_store.py        # Audit logging
│   └── audit_trail_store.py  # Detailed audit trails
│
├── Feature Stores
│   ├── gmail_token_store.py      # Gmail OAuth tokens
│   ├── integration_store.py      # Integration configs
│   ├── webhook_store.py          # Webhook idempotency
│   ├── notification_config_store.py  # Notification preferences
│   └── ...
│
├── Domain Stores
│   ├── gauntlet_run_store.py     # Gauntlet execution history
│   ├── receipt_store.py          # Decision receipts
│   ├── decision_result_store.py  # Decision outcomes
│   └── provenance_store.py       # Claim provenance
│
└── Utilities
    ├── schema.py             # Schema management
    ├── encrypted_fields.py   # Field-level encryption
    ├── fts_utils.py          # Full-text search utilities
    └── utils.py              # Common utilities
```

## Backend Support

| Backend    | Use Case                    | Configuration                |
|------------|-----------------------------|-----------------------------|
| SQLite     | Development, single-node    | `DATABASE_BACKEND=sqlite`   |
| PostgreSQL | Production, multi-node      | `DATABASE_BACKEND=postgresql` |
| Redis      | Caching, session state      | `REDIS_URL=redis://...`     |

### Backend Selection

```python
from aragora.storage import get_database_backend, DatabaseBackend

# Get current backend
backend = get_database_backend()
print(f"Using: {backend.name}")

# Check PostgreSQL availability
from aragora.storage import POSTGRESQL_AVAILABLE
if POSTGRESQL_AVAILABLE:
    print("PostgreSQL driver installed")
```

## Store Patterns

All stores follow a consistent pattern with three implementations:

1. **InMemory** - For testing and development
2. **SQLite** - For local/single-node deployment
3. **Redis** - For distributed/cached access

### Store Access

```python
from aragora.storage import (
    get_integration_store,
    get_webhook_store,
    get_gmail_token_store,
)

# Get the configured store (auto-selects based on environment)
store = get_integration_store()

# Store operations
await store.set("my-integration", config)
config = await store.get("my-integration")
await store.delete("my-integration")
```

### Custom Store Configuration

```python
from aragora.storage import (
    SQLiteIntegrationStore,
    set_integration_store,
    reset_integration_store,
)

# Use SQLite explicitly
store = SQLiteIntegrationStore("/path/to/db.sqlite")
set_integration_store(store)

# Reset to default
reset_integration_store()
```

## Core Stores Reference

### UserStore
User account management with password hashing and OAuth.

```python
from aragora.storage import UserStore

store = UserStore()
user = store.get_user_by_email("user@example.com")
store.create_user("user@example.com", "hashed_password")
```

### OrganizationStore
Multi-tenant organization management.

```python
from aragora.storage import OrganizationStore

store = OrganizationStore()
org = store.get_organization(org_id)
store.add_member(org_id, user_id, role="member")
```

### AuditStore
Audit event logging with retention policies.

```python
from aragora.storage import AuditStore

store = AuditStore()
store.log_event(
    action="user.login",
    actor_id="user-123",
    resource_type="session",
    details={"ip": "192.168.1.1"}
)
```

## Feature Stores Reference

### IntegrationStore
External integration configuration (Slack, GitHub, etc.).

```python
from aragora.storage import get_integration_store, IntegrationConfig

store = get_integration_store()
config = IntegrationConfig(
    integration_type="slack",
    credentials={"token": "xoxb-..."},
    settings={"channel": "#general"}
)
await store.set("workspace-123", config)
```

### WebhookStore
Webhook delivery tracking and idempotency.

```python
from aragora.storage import get_webhook_store

store = get_webhook_store()
# Check if webhook was already delivered
if await store.is_delivered(webhook_id):
    return  # Skip duplicate
await store.mark_delivered(webhook_id)
```

### GmailTokenStore
Gmail OAuth token management with refresh handling.

```python
from aragora.storage import get_gmail_token_store, GmailUserState

store = get_gmail_token_store()
state = GmailUserState(
    user_id="user-123",
    access_token="ya29...",
    refresh_token="1//...",
    expires_at=datetime.now() + timedelta(hours=1)
)
await store.set(state.user_id, state)
```

## Connection Routing

For read-heavy workloads, use the connection router:

```python
from aragora.storage import (
    initialize_connection_router,
    get_connection_router,
    RouterConfig,
    ReplicaConfig,
)

# Configure primary + replicas
config = RouterConfig(
    primary_dsn="postgresql://primary:5432/aragora",
    replicas=[
        ReplicaConfig(dsn="postgresql://replica1:5432/aragora", weight=50),
        ReplicaConfig(dsn="postgresql://replica2:5432/aragora", weight=50),
    ],
)
initialize_connection_router(config)

# Get router
router = get_connection_router()

# Reads go to replicas, writes go to primary
async with router.read_connection() as conn:
    result = await conn.fetch("SELECT * FROM users")

async with router.write_connection() as conn:
    await conn.execute("INSERT INTO users ...")
```

## Redis High Availability

For Redis cluster or sentinel mode:

```python
from aragora.storage import (
    RedisMode,
    RedisHAConfig,
    get_ha_redis_client,
)

# Sentinel mode
config = RedisHAConfig(
    mode=RedisMode.SENTINEL,
    sentinels=[("sentinel1", 26379), ("sentinel2", 26379)],
    master_name="mymaster",
)

# Cluster mode
config = RedisHAConfig(
    mode=RedisMode.CLUSTER,
    cluster_nodes=[("node1", 6379), ("node2", 6379)],
)

client = get_ha_redis_client(config)
```

## Schema Management

Use the schema utilities for migrations:

```python
from aragora.storage.schema import (
    run_migrations,
    check_schema_version,
    upgrade_schema,
)

# Check current version
version = check_schema_version(connection)

# Run pending migrations
run_migrations(connection, target_version=42)
```

## Encrypted Fields

For sensitive data storage:

```python
from aragora.storage.encrypted_fields import (
    EncryptedField,
    get_encryption_key,
)

# Encrypt before storage
field = EncryptedField(key=get_encryption_key())
encrypted = field.encrypt("sensitive-data")

# Decrypt on read
decrypted = field.decrypt(encrypted)
```

## Testing

All stores support in-memory backends for testing:

```python
import pytest
from aragora.storage import (
    InMemoryIntegrationStore,
    set_integration_store,
    reset_integration_store,
)

@pytest.fixture
def integration_store():
    store = InMemoryIntegrationStore()
    set_integration_store(store)
    yield store
    reset_integration_store()

async def test_integration(integration_store):
    await integration_store.set("test", config)
    result = await integration_store.get("test")
    assert result == config
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_BACKEND` | Backend type (sqlite/postgresql) | `sqlite` |
| `DATABASE_URL` | PostgreSQL connection URL | - |
| `SQLITE_PATH` | SQLite database path | `aragora.db` |
| `REDIS_URL` | Redis connection URL | - |
| `REDIS_MODE` | Redis mode (standalone/sentinel/cluster) | `standalone` |
| `ENCRYPTION_KEY` | Key for encrypted fields | - |

## Module Files (59 total)

### Base Infrastructure
- `base_database.py` - Abstract database interface
- `base_store.py` - SQLite base implementation
- `postgres_store.py` - PostgreSQL base implementation
- `backends.py` - Backend enum and factory
- `connection_factory.py` - Connection pooling
- `connection_router.py` - Read replica routing
- `factory.py` - Store factory utilities
- `schema.py` - Schema versioning and migrations

### Core Stores
- `user_store/` - User management (subpackage)
- `organization_store.py` - Organization management
- `audit_store.py` - Audit event logging
- `audit_trail_store.py` - Detailed audit trails

### Integration Stores
- `gmail_token_store.py` - Gmail OAuth tokens
- `integration_store.py` - External integrations
- `slack_workspace_store.py` - Slack workspace data
- `slack_debate_store.py` - Slack debate context
- `teams_workspace_store.py` - Microsoft Teams workspaces
- `teams_tenant_store.py` - Teams tenant data
- `discord_guild_store.py` - Discord guild data
- `plaid_credential_store.py` - Plaid banking credentials

### Feature Stores
- `webhook_store.py` - Webhook idempotency
- `webhook_config_store.py` - Webhook configurations
- `notification_config_store.py` - Notification preferences
- `channel_subscription_store.py` - Channel subscriptions
- `unified_inbox_store.py` - Unified inbox state

### Business Stores
- `gauntlet_run_store.py` - Gauntlet execution history
- `receipt_store.py` - Decision receipts
- `receipt_share_store.py` - Shared receipts
- `decision_result_store.py` - Decision outcomes
- `provenance_store.py` - Claim provenance
- `approval_request_store.py` - Approval workflows

### Financial Stores
- `expense_store.py` - Expense tracking
- `invoice_store.py` - Invoice management
- `ar_invoice_store.py` - Accounts receivable
- `marketplace_store.py` - Marketplace listings

### Utility Stores
- `job_queue_store.py` - Background job queue
- `snooze_store.py` - Snooze/remind later
- `followup_store.py` - Follow-up tracking
- `email_store.py` - Email metadata
- `finding_workflow_store.py` - Security findings
- `federation_registry_store.py` - Federation config
- `share_store.py` - Share links
- `password_reset_store.py` - Password reset tokens
- `impersonation_store.py` - Admin impersonation
- `inbox_activity_store.py` - Inbox activity tracking
- `token_blacklist_store.py` - JWT blacklist

### Utilities
- `encrypted_fields.py` - Field-level encryption
- `fts_utils.py` - Full-text search
- `utils.py` - Common utilities
- `redis_utils.py` - Redis helpers
- `redis_ha.py` - Redis HA configuration
- `repositories/` - Repository pattern implementations
- `governance/` - Data governance utilities
