# Admin Console Guide

The Aragora Admin Console provides system administrators with tools for managing users, organizations, personas, audit logs, and training data exports.

## Overview

The Admin Console is accessible at `/admin` and provides:
- **System Health** - Server status, component health, circuit breakers
- **Organizations** - Manage tenants and subscription tiers
- **Users** - User accounts, roles, and access control
- **Personas** - Agent persona configuration and traits
- **Audit Logs** - Security and compliance event tracking
- **Revenue** - Billing metrics and subscription analytics
- **Training Export** - Export debate data for ML fine-tuning

## Access Requirements

| Role | Access Level |
|------|--------------|
| `admin` | Full access to all admin features |
| `owner` | Organization-scoped admin access |
| `member` | Read-only access to some panels |
| `viewer` | No admin access |

## Admin Pages

### System Health (`/admin`)

Monitor server health, component status, and agent availability.

**Features:**
- Real-time health status (healthy/degraded/unhealthy)
- Component latency monitoring (database, agents, memory, websocket)
- Circuit breaker states for each agent
- Rate limit status across endpoints
- Recent error log viewer
- Metrics dashboard

**API Endpoints:**
```
GET /api/health              - Overall health status
GET /api/system/circuit-breakers - Agent circuit breaker states
GET /api/system/errors       - Recent error logs
GET /api/system/rate-limits  - Rate limit status
```

### Organizations (`/admin/organizations`)

Manage multi-tenant organizations and subscription tiers.

**Features:**
- List all organizations with pagination
- Filter by subscription tier (free, starter, professional, enterprise)
- View debate usage and billing status
- Stripe integration status

**Tiers:**
| Tier | Monthly Debates | Features |
|------|-----------------|----------|
| Free | 10 | Basic debates |
| Starter | 100 | Priority agents |
| Professional | 500 | All agents, analytics |
| Enterprise | Unlimited | Custom integrations |
| Enterprise+ | Unlimited | Dedicated support |

**API Endpoints:**
```
GET /api/admin/organizations - List organizations
GET /api/admin/organizations/:id - Get organization details
```

### Users (`/admin/users`)

Manage user accounts and access control.

**Features:**
- List all users with pagination
- Filter by role (owner, admin, member, viewer)
- Filter active/inactive users
- Activate/deactivate user accounts
- View email verification status
- Track last login times

**API Endpoints:**
```
GET  /api/admin/users           - List users
POST /api/admin/users/:id/activate   - Activate user
POST /api/admin/users/:id/deactivate - Deactivate user
```

### Personas (`/admin/personas`)

View and manage agent persona configurations.

**Features:**
- Grid and list view modes
- Search personas by name or traits
- View persona details including:
  - Description
  - Traits (analytical, creative, etc.)
  - Expertise areas
  - Creation/update timestamps
- Detail panel with full persona information

**API Endpoints:**
```
GET /api/personas           - List all personas
GET /api/agent/:name/persona - Get specific agent persona
```

### Audit Logs (`/admin/audit`)

Security and compliance event tracking with full audit trail.

**Features:**
- Real-time event feed with pagination
- Filter by:
  - Category (auth, data, admin, system)
  - Outcome (success, failure, error)
  - Date range
  - Search query
- Event detail panel showing:
  - Full event ID and timestamp
  - Actor information
  - Resource type and ID
  - IP address
  - Detailed JSON payload
  - Integrity hash
- Export functionality:
  - JSON format
  - CSV format
  - SOC2 compliance format
- Integrity verification

**Event Categories:**
| Category | Events |
|----------|--------|
| `auth` | login, logout, token_refresh, mfa_verify |
| `data` | create_debate, update_debate, delete_debate |
| `admin` | user_activate, user_deactivate, org_update |
| `system` | server_start, config_change, error |

**API Endpoints:**
```
GET  /api/audit/events      - Query audit events
GET  /api/audit/stats       - Get audit statistics
POST /api/audit/export      - Export audit logs
POST /api/audit/verify      - Verify log integrity
```

### Revenue (`/admin/revenue`)

Billing metrics and subscription analytics.

**Features:**
- Monthly Recurring Revenue (MRR) breakdown by tier
- Organization count per tier
- Total revenue tracking
- Admin statistics

**API Endpoints:**
```
GET /api/admin/revenue - Get revenue metrics
GET /api/admin/stats   - Get admin statistics
```

### Training Export (`/admin/training`)

Export debate data for ML model fine-tuning.

**Features:**
- Three export formats:
  - **SFT** (Supervised Fine-Tuning): Task-response pairs
  - **DPO** (Direct Preference Optimization): Chosen/rejected pairs
  - **Gauntlet**: Adversarial findings for safety training
- Configurable parameters per format
- JSON and JSONL output formats
- Download functionality
- Export history and statistics

**SFT Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_confidence` | Minimum confidence threshold | 0.7 |
| `min_success_rate` | Minimum agent success rate | 0.6 |
| `limit` | Maximum records to export | 1000 |
| `include_critiques` | Include critique data | true |
| `include_patterns` | Include learned patterns | true |
| `include_debates` | Include full debate context | true |

**DPO Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_confidence_diff` | Minimum confidence difference | 0.1 |
| `limit` | Maximum pairs to export | 500 |

**Gauntlet Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `persona` | Compliance persona (all, gdpr, hipaa, ai_act) | all |
| `min_severity` | Minimum finding severity | 0.5 |
| `limit` | Maximum findings to export | 500 |

**API Endpoints:**
```
GET  /api/training/stats           - Get training data statistics
GET  /api/training/formats         - Get available export formats
GET  /api/training/export/sft      - Export SFT data
GET  /api/training/export/dpo      - Export DPO data
GET  /api/training/export/gauntlet - Export Gauntlet data
```

## React Components

The admin console uses three main React components:

### PersonaEditor

```typescript
import { PersonaEditor } from '@/components/admin/PersonaEditor';

<PersonaEditor apiBase="/api" />
```

**Props:**
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `apiBase` | string | `/api` | API base URL |

### AuditLogViewer

```typescript
import { AuditLogViewer } from '@/components/admin/AuditLogViewer';

<AuditLogViewer apiBase="/api" />
```

**Props:**
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `apiBase` | string | `/api` | API base URL |

### TrainingExportPanel

```typescript
import { TrainingExportPanel } from '@/components/admin/TrainingExportPanel';

<TrainingExportPanel apiBase="/api" />
```

**Props:**
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `apiBase` | string | `/api` | API base URL |

## Authentication

Admin endpoints require authentication via JWT tokens with admin role claims.

```typescript
// Example: Fetch with auth header
const response = await fetch('/api/admin/users', {
  headers: {
    'Authorization': `Bearer ${accessToken}`,
  },
});
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ARAGORA_API_TOKEN` | API authentication token | Yes |
| `STRIPE_SECRET_KEY` | Stripe API key for billing | For revenue |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook validation | For revenue |

## Best Practices

1. **Regular Audit Review** - Check audit logs weekly for security anomalies
2. **User Cleanup** - Deactivate unused accounts monthly
3. **Export Backups** - Export audit logs monthly for compliance
4. **Integrity Checks** - Run integrity verification before audits
5. **Rate Limit Monitoring** - Watch for rate limit exhaustion patterns

## Troubleshooting

### Common Issues

**"Admin access required" error:**
- Ensure user has `admin` or `owner` role
- Check JWT token is valid and not expired
- Verify token includes role claims

**Empty audit logs:**
- Check date range filter
- Verify audit logging is enabled in server config
- Check category/outcome filters

**Training export fails:**
- Verify sufficient debate data exists
- Check exporter availability in stats
- Reduce limit parameter if memory issues

## See Also

- [API Reference](API_REFERENCE.md) - Full API documentation
- [Authentication Guide](AUTH.md) - Auth configuration
- [Runbook](RUNBOOK.md) - Operational procedures
- [Evidence System](EVIDENCE.md) - Evidence connectors
