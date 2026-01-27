# Handler Test Coverage Matrix

Auto-generated report of handler test coverage.

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Handlers | 256 |
| Handlers with Tests | 116 (45.3%) |
| Handlers without Tests | 140 (54.7%) |

### Coverage Levels

| Level | Count | Description |
|-------|-------|-------------|
| Comprehensive | 35 | 30+ test methods, 600+ lines |
| Moderate | 55 | 15-29 test methods, 300-599 lines |
| Minimal | 23 | 5-14 test methods, 100-299 lines |
| Stub | 3 | <5 test methods |
| None | 140 | No test file |

## Coverage by Category

| Category | Total | Tested | Coverage | Priority |
|----------|-------|--------|----------|----------|
| Core Debate | 13 | 2 | 15% | HIGH |
| Authentication & Security | 5 | 0 | 0% | HIGH |
| Features | 42 | 23 | 55% | CRITICAL |
| Knowledge Management | 25 | 4 | 16% | CRITICAL |
| Social & Communication | 25 | 11 | 44% | MEDIUM |
| Admin & Operations | 9 | 4 | 44% | MEDIUM |
| Utilities & Infrastructure | 137 | 72 | 53% | LOW |

## Admin & Operations

| Handler | Test Status | Coverage | Test Methods | Priority |
|---------|-------------|----------|--------------|----------|
| `admin` | ✓ | Comprehensive | 33 | LOW |
| `billing` | ✓ | Comprehensive | 31 | LOW |
| `dashboard` | ✓ | Comprehensive | 36 | LOW |
| `health` | ✓ | Moderate | 56 | LOW |
| `cache` | ✗ | None | - | HIGH |
| `credits` | ✗ | None | - | HIGH |
| `health_utils` | ✗ | None | - | HIGH |
| `security` | ✗ | None | - | HIGH |
| `system` | ✗ | None | - | HIGH |

## Authentication & Security

| Handler | Test Status | Coverage | Test Methods | Priority |
|---------|-------------|----------|--------------|----------|
| `handler` | ✗ | None | - | HIGH |
| `signup_handlers` | ✗ | None | - | HIGH |
| `sso_handlers` | ✗ | None | - | HIGH |
| `store` | ✗ | None | - | HIGH |
| `validation` | ✗ | None | - | HIGH |

## Core Debate

| Handler | Test Status | Coverage | Test Methods | Priority |
|---------|-------------|----------|--------------|----------|
| `batch` | ✓ | Minimal | 13 | MEDIUM |
| `decision` | ✓ | Minimal | 13 | MEDIUM |
| `analysis` | ✗ | None | - | HIGH |
| `deliberations` | ✗ | None | - | HIGH |
| `explain` | ✗ | None | - | HIGH |
| `export` | ✗ | None | - | HIGH |
| `fork` | ✗ | None | - | HIGH |
| `graph_debates` | ✗ | None | - | HIGH |
| `handler` | ✗ | None | - | HIGH |
| `intervention` | ✗ | Missing tests (handler exists) | - | HIGH |
| `matrix_debates` | ✗ | None | - | HIGH |
| `response_formatting` | ✗ | None | - | HIGH |
| `search` | ✗ | None | - | HIGH |

## Features

| Handler | Test Status | Coverage | Test Methods | Priority |
|---------|-------------|----------|--------------|----------|
| `advertising` | ✓ | Minimal | 7 | MEDIUM |
| `analytics_platforms` | ✓ | Stub | 4 | MEDIUM |
| `audio` | ✓ | Minimal | 6 | MEDIUM |
| `codebase_audit` | ✓ | Moderate | 33 | LOW |
| `connectors` | ✓ | Moderate | 33 | LOW |
| `control_plane` | ✓ | Comprehensive | 34 | LOW |
| `crm` | ✓ | Minimal | 9 | MEDIUM |
| `cross_platform_analytics` | ✓ | Comprehensive | 47 | LOW |
| `documents` | ✓ | Minimal | 12 | MEDIUM |
| `ecommerce` | ✓ | Stub | 4 | MEDIUM |
| `email_webhooks` | ✓ | Moderate | 27 | LOW |
| `evidence` | ✓ | Moderate | 34 | LOW |
| `finding_workflow` | ✓ | Moderate | 21 | LOW |
| `gmail_labels` | ✓ | Minimal | 7 | MEDIUM |
| `integrations` | ✓ | Moderate | 26 | LOW |
| `marketplace` | ✓ | Moderate | 30 | LOW |
| `outlook` | ✓ | Minimal | 5 | MEDIUM |
| `pulse` | ✓ | Comprehensive | 47 | LOW |
| `reconciliation` | ✓ | Moderate | 27 | LOW |
| `rlm` | ✓ | Comprehensive | 52 | LOW |
| `support` | ✓ | Stub | 4 | MEDIUM |
| `transcription` | ✓ | Minimal | 19 | MEDIUM |
| `unified_inbox` | ✓ | Moderate | 25 | LOW |
| `audit_sessions` | ✗ | None | - | HIGH |
| `broadcast` | ✗ | None | - | HIGH |
| `cloud_storage` | ✗ | None | - | HIGH |
| `devops` | ✗ | None | - | HIGH |
| `document_query` | ✗ | None | - | HIGH |
| `documents_batch` | ✗ | None | - | HIGH |
| `evidence_enrichment` | ✗ | None | - | HIGH |
| `features` | ✗ | None | - | HIGH |
| `folder_upload` | ✗ | None | - | HIGH |
| `gmail_ingest` | ✗ | None | - | HIGH |
| `gmail_query` | ✗ | None | - | HIGH |
| `gmail_threads` | ✗ | None | - | HIGH |
| `legal` | ✗ | None | - | HIGH |
| `plugins` | ✗ | None | - | HIGH |
| `provenance` | ✗ | None | - | HIGH |
| `routing_rules` | ✗ | None | - | HIGH |
| `scheduler` | ✗ | None | - | HIGH |
| `smart_upload` | ✗ | None | - | HIGH |
| `speech` | ✗ | None | - | HIGH |

## Knowledge Management

| Handler | Test Status | Coverage | Test Methods | Priority |
|---------|-------------|----------|--------------|----------|
| `analytics` | ✓ | Moderate | 30 | LOW |
| `analytics` | ✓ | Moderate | 30 | LOW |
| `checkpoints` | ✓ | Moderate | 22 | LOW |
| `dashboard` | ✓ | Comprehensive | 36 | LOW |
| `base_mixin` | ✗ | None | - | HIGH |
| `confidence_decay` | ✗ | None | - | HIGH |
| `contradiction` | ✗ | None | - | HIGH |
| `culture` | ✗ | None | - | HIGH |
| `curation` | ✗ | None | - | HIGH |
| `dedup` | ✗ | Missing tests (handler exists) | - | HIGH |
| `export` | ✗ | None | - | HIGH |
| `extraction` | ✗ | None | - | HIGH |
| `federation` | ✗ | Missing tests (handler exists) | - | HIGH |
| `global_knowledge` | ✗ | None | - | HIGH |
| `governance` | ✗ | None | - | HIGH |
| `graph` | ✗ | None | - | HIGH |
| `handler` | ✗ | None | - | HIGH |
| `nodes` | ✗ | None | - | HIGH |
| `pruning` | ✗ | None | - | HIGH |
| `relationships` | ✗ | None | - | HIGH |
| `sharing` | ✗ | None | - | HIGH |
| `sharing_notifications` | ✗ | None | - | HIGH |
| `staleness` | ✗ | None | - | HIGH |
| `sync` | ✗ | None | - | HIGH |
| `visibility` | ✗ | None | - | HIGH |

## Social & Communication

| Handler | Test Status | Coverage | Test Methods | Priority |
|---------|-------------|----------|--------------|----------|
| `discord` | ✓ | Minimal | 14 | MEDIUM |
| `discord_oauth` | ✓ | Moderate | 28 | LOW |
| `email_webhook` | ✓ | Moderate | 18 | LOW |
| `slack_oauth` | ✓ | Moderate | 29 | LOW |
| `teams_oauth` | ✓ | Moderate | 28 | LOW |
| `telegram` | ✓ | Comprehensive | 47 | LOW |
| `telegram` | ✓ | Comprehensive | 47 | LOW |
| `tts_helper` | ✓ | Comprehensive | 40 | LOW |
| `whatsapp` | ✓ | Comprehensive | 48 | LOW |
| `whatsapp` | ✓ | Comprehensive | 48 | LOW |
| `zoom` | ✓ | Moderate | 19 | LOW |
| `channel_health` | ✗ | None | - | HIGH |
| `chat_events` | ✗ | None | - | HIGH |
| `collaboration` | ✗ | None | - | HIGH |
| `google_chat` | ✗ | None | - | HIGH |
| `notifications` | ✗ | None | - | HIGH |
| `relationship` | ✗ | None | - | HIGH |
| `router` | ✗ | None | - | HIGH |
| `sharing` | ✗ | None | - | HIGH |
| `slack` | ✗ | None | - | HIGH |
| `slack` | ✗ | None | - | HIGH |
| `social_media` | ✗ | None | - | HIGH |
| `teams` | ✗ | None | - | HIGH |
| `teams` | ✗ | None | - | HIGH |
| `telemetry` | ✗ | None | - | HIGH |

## Utilities & Infrastructure

| Handler | Test Status | Coverage | Test Methods | Priority |
|---------|-------------|----------|--------------|----------|
| `accounting` | ✓ | Comprehensive | 47 | LOW |
| `agents` | ✓ | Moderate | 46 | LOW |
| `analytics` | ✓ | Moderate | 30 | LOW |
| `analytics_dashboard` | ✓ | Moderate | 52 | LOW |
| `analytics_metrics` | ✓ | Moderate | 24 | LOW |
| `audit_export` | ✓ | Minimal | 26 | MEDIUM |
| `auditing` | ✓ | Moderate | 40 | LOW |
| `auth` | ✓ | Comprehensive | 63 | LOW |
| `backup_handler` | ✓ | Moderate | 15 | LOW |
| `belief` | ✓ | Moderate | 26 | LOW |
| `checkpoints` | ✓ | Moderate | 22 | LOW |
| `compliance_handler` | ✓ | Minimal | 18 | MEDIUM |
| `connectors` | ✓ | Moderate | 33 | LOW |
| `consensus` | ✓ | Moderate | 18 | LOW |
| `control_plane` | ✓ | Comprehensive | 34 | LOW |
| `costs` | ✓ | Comprehensive | 44 | LOW |
| `critique` | ✓ | Moderate | 19 | LOW |
| `dashboard` | ✓ | Comprehensive | 36 | LOW |
| `dependency_analysis` | ✓ | Moderate | 16 | LOW |
| `dr_handler` | ✓ | Minimal | 12 | MEDIUM |
| `email` | ✓ | Moderate | 20 | LOW |
| `email_services` | ✓ | Moderate | 20 | LOW |
| `evaluation` | ✓ | Minimal | 11 | MEDIUM |
| `exceptions` | ✓ | Moderate | 57 | LOW |
| `expenses` | ✓ | Comprehensive | 38 | LOW |
| `explainability` | ✓ | Moderate | 21 | LOW |
| `external_integrations` | ✓ | Minimal | 13 | MEDIUM |
| `gallery` | ✓ | Minimal | 10 | MEDIUM |
| `gauntlet` | ✓ | Comprehensive | 32 | LOW |
| `gauntlet_v1` | ✓ | Moderate | 30 | LOW |
| `genesis` | ✓ | Comprehensive | 56 | LOW |
| `inbox_command` | ✓ | Moderate | 22 | LOW |
| `integrations` | ✓ | Moderate | 26 | LOW |
| `intelligence` | ✓ | Moderate | 26 | LOW |
| `introspection` | ✓ | Moderate | 27 | LOW |
| `invoices` | ✓ | Comprehensive | 54 | LOW |
| `knowledge` | ✓ | Comprehensive | 60 | LOW |
| `marketplace` | ✓ | Moderate | 30 | LOW |
| `memory` | ✓ | Comprehensive | 34 | LOW |
| `metrics` | ✓ | Moderate | 33 | LOW |
| `metrics` | ✓ | Moderate | 33 | LOW |
| `ml` | ✓ | Moderate | 31 | LOW |
| `nomic` | ✓ | Moderate | 24 | LOW |
| `oauth` | ✓ | Moderate | 34 | LOW |
| `onboarding` | ✓ | Moderate | 32 | LOW |
| `orchestration` | ✓ | Moderate | 23 | LOW |
| `organizations` | ✓ | Comprehensive | 54 | LOW |
| `payments` | ✓ | Moderate | 29 | LOW |
| `policy` | ✓ | Minimal | 26 | MEDIUM |
| `pr_review` | ✓ | Moderate | 23 | LOW |
| `privacy` | ✓ | Comprehensive | 40 | LOW |
| `queue` | ✓ | Comprehensive | 33 | LOW |
| `receipts` | ✓ | Comprehensive | 36 | LOW |
| `repository` | ✓ | Moderate | 18 | LOW |
| `rlm` | ✓ | Comprehensive | 52 | LOW |
| `routing` | ✓ | Moderate | 20 | LOW |
| `routing` | ✓ | Moderate | 20 | LOW |
| `secure` | ✓ | Comprehensive | 38 | LOW |
| `selection` | ✓ | Moderate | 26 | LOW |
| `shared_inbox` | ✓ | Comprehensive | 67 | LOW |
| `sme_usage_dashboard` | ✓ | Moderate | 27 | LOW |
| `sso` | ✓ | Minimal | 36 | MEDIUM |
| `status_page` | ✓ | Minimal | 20 | MEDIUM |
| `template_marketplace` | ✓ | Comprehensive | 37 | LOW |
| `training` | ✓ | Minimal | 27 | MEDIUM |
| `transcription` | ✓ | Minimal | 19 | MEDIUM |
| `uncertainty` | ✓ | Minimal | 13 | MEDIUM |
| `usage_metering` | ✓ | Minimal | 9 | MEDIUM |
| `webhooks` | ✓ | Comprehensive | 53 | LOW |
| `workflow_templates` | ✓ | Moderate | 23 | LOW |
| `workflows` | ✓ | Comprehensive | 70 | LOW |
| `workspace` | ✓ | Comprehensive | 94 | LOW |
| `a2a` | ✗ | None | - | HIGH |
| `ab_testing` | ✗ | None | - | HIGH |
| `action_items` | ✗ | None | - | HIGH |
| `alerts` | ✗ | None | - | HIGH |
| `ap_automation` | ✗ | None | - | HIGH |
| `approvals` | ✗ | None | - | HIGH |
| `ar_automation` | ✗ | None | - | HIGH |
| `audit_bridge` | ✗ | None | - | HIGH |
| `audit_trail` | ✗ | None | - | HIGH |
| `breakpoints` | ✗ | None | - | HIGH |
| `budgets` | ✗ | None | - | HIGH |
| `calibration` | ✗ | None | - | HIGH |
| `code_review` | ✗ | None | - | HIGH |
| `composite` | ✗ | None | - | HIGH |
| `config` | ✗ | None | - | HIGH |
| `coordinator` | ✗ | None | - | HIGH |
| `cross_pollination` | ✗ | None | - | HIGH |
| `database` | ✗ | None | - | HIGH |
| `decorators` | ✗ | None | - | HIGH |
| `docs` | ✗ | None | - | HIGH |
| `email_actions` | ✗ | None | - | HIGH |
| `email_debate` | ✗ | None | - | HIGH |
| `explainability_store` | ✗ | None | - | HIGH |
| `facts` | ✗ | None | - | HIGH |
| `formal_verification` | ✗ | None | - | HIGH |
| `formatters` | ✗ | None | - | HIGH |
| `handler` | ✗ | None | - | HIGH |
| `handler` | ✗ | None | - | HIGH |
| `handler` | ✗ | None | - | HIGH |
| `handler` | ✗ | None | - | HIGH |
| `insights` | ✗ | None | - | HIGH |
| `interface` | ✗ | None | - | HIGH |
| `knowledge_chat` | ✗ | None | - | HIGH |
| `laboratory` | ✗ | None | - | HIGH |
| `leaderboard` | ✗ | None | - | HIGH |
| `learning` | ✗ | None | - | HIGH |
| `learning` | ✗ | None | - | HIGH |
| `memory_analytics` | ✗ | None | - | HIGH |
| `moments` | ✗ | None | - | HIGH |
| `monitoring` | ✗ | None | - | HIGH |
| `oauth_wizard` | ✗ | None | - | HIGH |
| `openapi_decorator` | ✗ | None | - | HIGH |
| `params` | ✗ | None | - | HIGH |
| `partner` | ✗ | None | - | HIGH |
| `persona` | ✗ | None | - | HIGH |
| `probes` | ✗ | None | - | HIGH |
| `query` | ✗ | None | - | HIGH |
| `quick_scan` | ✗ | None | - | HIGH |
| `rate_limit` | ✗ | None | - | HIGH |
| `replays` | ✗ | None | - | HIGH |
| `responses` | ✗ | None | - | HIGH |
| `reviews` | ✗ | None | - | HIGH |
| `safe_data` | ✗ | None | - | HIGH |
| `search` | ✗ | None | - | HIGH |
| `security` | ✗ | None | - | HIGH |
| `slo` | ✗ | None | - | HIGH |
| `team_inbox` | ✗ | None | - | HIGH |
| `threat_intel` | ✗ | None | - | HIGH |
| `tournaments` | ✗ | None | - | HIGH |
| `tracking` | ✗ | None | - | HIGH |
| `triggers` | ✗ | None | - | HIGH |
| `url_security` | ✗ | None | - | HIGH |
| `utilities` | ✗ | None | - | HIGH |
| `verification` | ✗ | None | - | HIGH |
| `verticals` | ✗ | None | - | HIGH |

## Critical Gaps (No Tests)

Handlers without any test coverage that should be prioritized:

### Admin & Operations

- `aragora/server/handlers/admin/cache.py`
- `aragora/server/handlers/admin/credits.py`
- `aragora/server/handlers/admin/health_utils.py`
- `aragora/server/handlers/admin/security.py`
- `aragora/server/handlers/admin/system.py`

### Authentication & Security

- `aragora/server/handlers/auth/handler.py`
- `aragora/server/handlers/auth/signup_handlers.py`
- `aragora/server/handlers/auth/sso_handlers.py`
- `aragora/server/handlers/auth/store.py`
- `aragora/server/handlers/auth/validation.py`

### Core Debate

- `aragora/server/handlers/debates/analysis.py`
- `aragora/server/handlers/deliberations.py`
- `aragora/server/handlers/decisions/explain.py`
- `aragora/server/handlers/debates/export.py`
- `aragora/server/handlers/debates/fork.py`
- `aragora/server/handlers/debates/graph_debates.py`
- `aragora/server/handlers/debates/handler.py`
- `aragora/server/handlers/debates/intervention.py`
- `aragora/server/handlers/debates/matrix_debates.py`
- `aragora/server/handlers/debates/response_formatting.py`
- `aragora/server/handlers/debates/search.py`

### Features

- `aragora/server/handlers/features/audit_sessions.py`
- `aragora/server/handlers/features/broadcast.py`
- `aragora/server/handlers/features/cloud_storage.py`
- `aragora/server/handlers/features/devops.py`
- `aragora/server/handlers/features/document_query.py`
- `aragora/server/handlers/features/documents_batch.py`
- `aragora/server/handlers/features/evidence_enrichment.py`
- `aragora/server/handlers/features/features.py`
- `aragora/server/handlers/features/folder_upload.py`
- `aragora/server/handlers/features/gmail_ingest.py`
- `aragora/server/handlers/features/gmail_query.py`
- `aragora/server/handlers/features/gmail_threads.py`
- `aragora/server/handlers/features/legal.py`
- `aragora/server/handlers/features/plugins.py`
- `aragora/server/handlers/features/provenance.py`
- `aragora/server/handlers/features/routing_rules.py`
- `aragora/server/handlers/features/scheduler.py`
- `aragora/server/handlers/features/smart_upload.py`
- `aragora/server/handlers/features/speech.py`

### Knowledge Management

- `aragora/server/handlers/knowledge_base/mound/base_mixin.py`
- `aragora/server/handlers/knowledge_base/mound/confidence_decay.py`
- `aragora/server/handlers/knowledge_base/mound/contradiction.py`
- `aragora/server/handlers/knowledge_base/mound/culture.py`
- `aragora/server/handlers/knowledge_base/mound/curation.py`
- `aragora/server/handlers/knowledge_base/mound/dedup.py`
- `aragora/server/handlers/knowledge_base/mound/export.py`
- `aragora/server/handlers/knowledge_base/mound/extraction.py`
- `aragora/server/handlers/knowledge_base/mound/federation.py`
- `aragora/server/handlers/knowledge_base/mound/global_knowledge.py`
- `aragora/server/handlers/knowledge_base/mound/governance.py`
- `aragora/server/handlers/knowledge_base/mound/graph.py`
- `aragora/server/handlers/knowledge_base/mound/handler.py`
- `aragora/server/handlers/knowledge_base/mound/nodes.py`
- `aragora/server/handlers/knowledge_base/mound/pruning.py`
- `aragora/server/handlers/knowledge_base/mound/relationships.py`
- `aragora/server/handlers/knowledge_base/mound/sharing.py`
- `aragora/server/handlers/knowledge/sharing_notifications.py`
- `aragora/server/handlers/knowledge_base/mound/staleness.py`
- `aragora/server/handlers/knowledge_base/mound/sync.py`
- `aragora/server/handlers/knowledge_base/mound/visibility.py`

### Social & Communication

- `aragora/server/handlers/social/channel_health.py`
- `aragora/server/handlers/social/chat_events.py`
- `aragora/server/handlers/social/collaboration.py`
- `aragora/server/handlers/bots/google_chat.py`
- `aragora/server/handlers/social/notifications.py`
- `aragora/server/handlers/social/relationship.py`
- `aragora/server/handlers/chat/router.py`
- `aragora/server/handlers/social/sharing.py`
- `aragora/server/handlers/bots/slack.py`
- `aragora/server/handlers/social/slack.py`
- `aragora/server/handlers/social/social_media.py`
- `aragora/server/handlers/bots/teams.py`
- `aragora/server/handlers/social/teams.py`
- `aragora/server/handlers/social/telemetry.py`

### Utilities & Infrastructure

- `aragora/server/handlers/a2a.py`
- `aragora/server/handlers/evolution/ab_testing.py`
- `aragora/server/handlers/inbox/action_items.py`
- `aragora/server/handlers/autonomous/alerts.py`
- `aragora/server/handlers/ap_automation.py`
- `aragora/server/handlers/autonomous/approvals.py`
- `aragora/server/handlers/ar_automation.py`
- `aragora/server/handlers/github/audit_bridge.py`
- `aragora/server/handlers/audit_trail.py`
- `aragora/server/handlers/breakpoints.py`
- `aragora/server/handlers/budgets.py`
- `aragora/server/handlers/agents/calibration.py`
- `aragora/server/handlers/code_review.py`
- `aragora/server/handlers/composite.py`
- `aragora/server/handlers/agents/config.py`
- `aragora/server/handlers/memory/coordinator.py`
- `aragora/server/handlers/cross_pollination.py`
- `aragora/server/handlers/utils/database.py`
- `aragora/server/handlers/utils/decorators.py`
- `aragora/server/handlers/docs.py`
- `aragora/server/handlers/inbox/email_actions.py`
- `aragora/server/handlers/email_debate.py`
- `aragora/server/handlers/explainability_store.py`
- `aragora/server/handlers/knowledge_base/facts.py`
- `aragora/server/handlers/verification/formal_verification.py`
- `aragora/server/handlers/metrics/formatters.py`
- `aragora/server/handlers/metrics/handler.py`
- `aragora/server/handlers/voice/handler.py`
- `aragora/server/handlers/evolution/handler.py`
- `aragora/server/handlers/knowledge_base/handler.py`
- `aragora/server/handlers/memory/insights.py`
- `aragora/server/handlers/interface.py`
- `aragora/server/handlers/knowledge_chat.py`
- `aragora/server/handlers/laboratory.py`
- `aragora/server/handlers/agents/leaderboard.py`
- `aragora/server/handlers/memory/learning.py`
- `aragora/server/handlers/autonomous/learning.py`
- `aragora/server/handlers/memory/memory_analytics.py`
- `aragora/server/handlers/moments.py`
- `aragora/server/handlers/autonomous/monitoring.py`
- `aragora/server/handlers/oauth_wizard.py`
- `aragora/server/handlers/openapi_decorator.py`
- `aragora/server/handlers/utils/params.py`
- `aragora/server/handlers/partner.py`
- `aragora/server/handlers/persona.py`
- `aragora/server/handlers/agents/probes.py`
- `aragora/server/handlers/knowledge_base/query.py`
- `aragora/server/handlers/codebase/quick_scan.py`
- `aragora/server/handlers/utils/rate_limit.py`
- `aragora/server/handlers/replays.py`
- `aragora/server/handlers/utils/responses.py`
- `aragora/server/handlers/reviews.py`
- `aragora/server/handlers/utils/safe_data.py`
- `aragora/server/handlers/knowledge_base/search.py`
- `aragora/server/handlers/codebase/security.py`
- `aragora/server/handlers/slo.py`
- `aragora/server/handlers/inbox/team_inbox.py`
- `aragora/server/handlers/threat_intel.py`
- `aragora/server/handlers/tournaments.py`
- `aragora/server/handlers/metrics/tracking.py`
- `aragora/server/handlers/autonomous/triggers.py`
- `aragora/server/handlers/utils/url_security.py`
- `aragora/server/handlers/utilities.py`
- `aragora/server/handlers/verification/verification.py`
- `aragora/server/handlers/verticals.py`
