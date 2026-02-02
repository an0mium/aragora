# Services Module

Centralized service layer providing domain-specific business logic and cross-cutting concerns.

## Overview

The services module provides:

| Category | Services |
|----------|----------|
| **Service Infrastructure** | `ServiceRegistry`, factories, lifecycle management |
| **Email Intelligence** | `EmailPrioritizer`, `SpamClassifier`, `ActionItemExtractor` |
| **Cross-Channel** | `CrossChannelContextService`, `MultiInboxManager` |
| **Financial** | `InvoiceProcessor`, `ExpenseTracker`, `APAutomation`, `ARAutomation` |
| **Security** | `ThreatIntelligenceService`, `PIIRedactor` |
| **Analytics** | `SenderHistoryService`, `UsageMetering`, `InboxSLAMonitor` |

## Quick Start

```python
from aragora.services import (
    ServiceRegistry,
    get_service,
    register_service,
    EmailPrioritizer,
    CrossChannelContextService,
)

# Register a service
register_service(EmailPrioritizer, prioritizer_instance)

# Get a service
prioritizer = get_service(EmailPrioritizer)

# Check if registered
if has_service(EmailPrioritizer):
    # Service available
    pass
```

## Service Registry

Thread-safe singleton registry replacing scattered global instances.

### Registration

```python
from aragora.services.registry import (
    ServiceRegistry,
    ServiceScope,
    ServiceNotFoundError,
)

registry = ServiceRegistry.get()

# Direct registration
registry.register(TierManager, tier_manager_instance)

# Factory registration (lazy initialization)
registry.register_factory(
    TierManager,
    lambda: TierManager(config=load_config()),
)

# Resolve a service
tier_manager = registry.resolve(TierManager)

# With default fallback
tier_manager = registry.resolve(TierManager, default=TierManager())
```

### Lifecycle Management

```python
# List all registered services
services = registry.list_services()
print(services)  # ['TierManager', 'EmailPrioritizer', ...]

# Get statistics
stats = registry.stats()
print(f"Total services: {stats.total_services}")
print(f"Initialized: {stats.initialized_count}")
print(f"Pending (factory-only): {stats.pending_count}")

# Graceful shutdown (calls close/shutdown/cleanup methods)
hooks_called = registry.shutdown()

# Reset for testing
ServiceRegistry.reset()
```

### Cache Service Markers

Register multiple cache instances with type safety:

```python
from aragora.services import (
    MethodCacheService,
    QueryCacheService,
    EmbeddingCacheService,
    HandlerCacheService,
)

# Register different cache types
register_service(MethodCacheService, method_cache)
register_service(QueryCacheService, query_cache)

# Resolve by type
method_cache = get_service(MethodCacheService)
query_cache = get_service(QueryCacheService)
```

## Email Prioritization

3-tier intelligent email scoring with cross-channel context.

### Architecture

| Tier | Latency | Method | Use Case |
|------|---------|--------|----------|
| Tier 1 | <200ms | Rule-based | Clear cases (VIP, spam, newsletters) |
| Tier 2 | <500ms | Single-agent | Ambiguous cases needing light analysis |
| Tier 3 | Async | Multi-agent debate | Complex prioritization decisions |

### Basic Usage

```python
from aragora.services.email_prioritization import (
    EmailPrioritizer,
    EmailPriorityResult,
    EmailPriority,
    EmailPrioritizationConfig,
)

config = EmailPrioritizationConfig(
    tier_1_confidence_threshold=0.7,
    tier_2_confidence_threshold=0.6,
    vip_domains={"important-client.com"},
    internal_domains={"company.com"},
)

prioritizer = EmailPrioritizer(
    gmail_connector=gmail,
    knowledge_mound=mound,
    config=config,
)

# Score a single email
result = await prioritizer.score_email(email)
print(f"Priority: {result.priority.name}")
print(f"Confidence: {result.confidence}")
print(f"Tier used: {result.tier_used}")
print(f"Rationale: {result.rationale}")

# Batch process inbox
ranked_emails = await prioritizer.rank_inbox(emails)
for result in ranked_emails:
    print(f"{result.email_id}: {result.priority.name}")
```

### Priority Levels

| Priority | Value | Description |
|----------|-------|-------------|
| `CRITICAL` | 1 | Immediate attention required |
| `HIGH` | 2 | Important, respond today |
| `MEDIUM` | 3 | Standard priority |
| `LOW` | 4 | Review when time allows |
| `DEFER` | 5 | Archive or auto-file |
| `BLOCKED` | 6 | Blocked sender, auto-archive |

### Scoring Components

```python
result = await prioritizer.score_email(email)

# Detailed breakdown
print(f"Sender score: {result.sender_score}")
print(f"Content urgency: {result.content_urgency_score}")
print(f"Context relevance: {result.context_relevance_score}")
print(f"Time sensitivity: {result.time_sensitivity_score}")

# Spam detection
print(f"Spam score: {result.spam_score}")
print(f"Spam category: {result.spam_category}")

# Cross-channel boosts
print(f"Slack boost: {result.slack_activity_boost}")
print(f"Calendar boost: {result.calendar_relevance_boost}")

# Recommendations
print(f"Suggested labels: {result.suggested_labels}")
print(f"Auto-archive: {result.auto_archive}")
```

### Convenience Function

```python
from aragora.services.email_prioritization import prioritize_inbox

# Quick inbox prioritization
results = await prioritize_inbox(
    emails=inbox_emails,
    gmail_connector=gmail,
)
```

## Cross-Channel Context

Unified signals from Slack, Google Drive, Calendar, and GitHub.

### Basic Usage

```python
from aragora.services.cross_channel_context import (
    CrossChannelContextService,
    ChannelContext,
    create_context_service,
)

service = CrossChannelContextService(
    slack_connector=slack,
    gmail_connector=gmail,
    drive_connector=drive,
    calendar_connector=calendar,
)

# Get user's unified context
context = await service.get_user_context("user@company.com")
print(f"Overall activity: {context.overall_activity_score}")
print(f"Is busy: {context.is_likely_busy}")
print(f"Active projects: {context.active_projects}")

# Get context relevant to an email
email_context = await service.get_email_context(email_message)
```

### Channel Signals

**Slack Activity:**
```python
slack = context.slack
print(f"Online: {slack.is_online}")
print(f"Active channels: {slack.active_channels}")
print(f"Recent mentions: {slack.recent_mentions}")
print(f"Urgent threads: {slack.urgent_threads}")
```

**Calendar:**
```python
calendar = context.calendar
print(f"Upcoming meetings: {len(calendar.upcoming_meetings)}")
print(f"Meeting density: {calendar.meeting_density_score}")  # 0=free, 1=busy
print(f"Next free slot: {calendar.next_free_slot}")
```

**Drive Activity:**
```python
drive = context.drive
print(f"Recently edited: {drive.recently_edited_files}")
print(f"Recently shared: {drive.shared_with_me_recent}")
```

### Email Context Boost

```python
from aragora.services.cross_channel_context import EmailContextBoost

# Get priority boosts for email based on cross-channel activity
boost = await service.get_email_boost(email_message)
print(f"Slack boost: {boost.slack_boost}")
print(f"Calendar boost: {boost.calendar_boost}")
print(f"Drive boost: {boost.drive_boost}")
print(f"Total boost: {boost.total_boost}")
```

## Spam Classification

Multi-layer spam detection with phishing analysis.

```python
from aragora.services.spam_classifier import (
    SpamClassifier,
    SpamClassifierConfig,
    SpamCategory,
    classify_email_spam,
)

classifier = SpamClassifier(
    config=SpamClassifierConfig(
        phishing_sensitivity=0.8,
        promotional_threshold=0.6,
    )
)

result = await classifier.classify(email)
print(f"Category: {result.category}")  # ham, spam, phishing, promotional
print(f"Confidence: {result.confidence}")
print(f"Indicators: {result.indicators}")

# Quick function
result = await classify_email_spam(email)
```

### Categories

| Category | Description |
|----------|-------------|
| `HAM` | Legitimate email |
| `SPAM` | Unsolicited bulk email |
| `PHISHING` | Credential theft attempt |
| `PROMOTIONAL` | Marketing/newsletters |
| `SUSPICIOUS` | Requires manual review |

## Multi-Inbox Manager

Unified management across email accounts.

```python
from aragora.services.multi_inbox_manager import (
    MultiInboxManager,
    InboxAccount,
    AccountType,
    create_multi_inbox_manager,
)

manager = MultiInboxManager()

# Add accounts
await manager.add_account(InboxAccount(
    account_id="work",
    email="user@company.com",
    account_type=AccountType.GMAIL,
    connector=work_gmail,
))

await manager.add_account(InboxAccount(
    account_id="personal",
    email="user@gmail.com",
    account_type=AccountType.GMAIL,
    connector=personal_gmail,
))

# Unified inbox
unified_emails = await manager.get_unified_inbox(limit=50)

# Cross-account sender profile
sender_profile = await manager.get_sender_profile("contact@example.com")
print(f"Total emails: {sender_profile.total_emails}")
print(f"Accounts seen: {sender_profile.accounts_seen}")
```

## Sender History

Track sender reputation and interaction history.

```python
from aragora.services.sender_history import (
    SenderHistoryService,
    SenderStats,
    SenderReputation,
    create_sender_history_service,
)

service = create_sender_history_service(store=store)

# Get sender stats
stats = await service.get_sender_stats("contact@example.com")
print(f"Total emails: {stats.total_emails}")
print(f"Response rate: {stats.response_rate}")
print(f"Avg response time: {stats.avg_response_time_hours}h")

# Get reputation
reputation = await service.get_reputation("contact@example.com")
print(f"Score: {reputation.score}")
print(f"Classification: {reputation.classification}")  # vip, trusted, unknown, suspicious
```

## Usage Metering

Track API and feature usage for billing.

```python
from aragora.services.usage_metering import UsageMeteringService

metering = UsageMeteringService(store=store)

# Record usage
await metering.record(
    tenant_id="tenant_123",
    feature="email_prioritization",
    units=10,
)

# Get usage report
report = await metering.get_usage(
    tenant_id="tenant_123",
    start_date=start,
    end_date=end,
)
print(f"Total units: {report.total_units}")
print(f"By feature: {report.by_feature}")
```

## Financial Services

### Invoice Processing

```python
from aragora.services.invoice_processor import InvoiceProcessor

processor = InvoiceProcessor(
    knowledge_mound=mound,
    extraction_model="claude-3-sonnet",
)

# Extract invoice data from email
invoice = await processor.process_email(email)
print(f"Vendor: {invoice.vendor}")
print(f"Amount: {invoice.amount}")
print(f"Due date: {invoice.due_date}")
print(f"Line items: {invoice.line_items}")

# Match to PO
match = await processor.match_to_po(invoice)
```

### Expense Tracking

```python
from aragora.services.expense_tracker import ExpenseTracker

tracker = ExpenseTracker(store=store)

# Track expense from email
expense = await tracker.track_from_email(email)
print(f"Category: {expense.category}")
print(f"Amount: {expense.amount}")
print(f"Merchant: {expense.merchant}")

# Get expense report
report = await tracker.get_report(
    user_id="user_123",
    start_date=start,
    end_date=end,
)
```

### AP/AR Automation

```python
from aragora.services.ap_automation import APAutomationService
from aragora.services.ar_automation import ARAutomationService

# Accounts Payable
ap = APAutomationService(store=store)
await ap.process_invoice(invoice)
await ap.schedule_payment(invoice_id, payment_date)

# Accounts Receivable
ar = ARAutomationService(store=store)
await ar.send_invoice(customer_id, invoice_data)
await ar.record_payment(invoice_id, payment)
overdue = await ar.get_overdue_invoices()
```

## Security Services

### PII Redaction

```python
from aragora.services.pii_redactor import PIIRedactor

redactor = PIIRedactor()

# Redact PII from text
redacted = redactor.redact("John's SSN is 123-45-6789")
print(redacted)  # "[NAME]'s SSN is [SSN]"

# Detect PII without redacting
pii_found = redactor.detect(text)
for pii in pii_found:
    print(f"Type: {pii.type}, Value: {pii.value}, Position: {pii.position}")
```

## Email Actions

Automated email actions based on rules.

```python
from aragora.services.email_actions import EmailActionService

service = EmailActionService(gmail=gmail)

# Define action rules
await service.add_rule({
    "name": "Auto-label invoices",
    "condition": {"subject_contains": "invoice"},
    "action": {"add_label": "Finance/Invoices"},
})

# Process email
actions_taken = await service.process_email(email)
```

## Follow-up Tracker

Track emails requiring follow-up.

```python
from aragora.services.followup_tracker import FollowupTracker

tracker = FollowupTracker(store=store)

# Add follow-up
await tracker.add_followup(
    email_id=email.id,
    due_date=due_date,
    note="Waiting for contract approval",
)

# Get due follow-ups
due = await tracker.get_due_followups(user_id)
for item in due:
    print(f"Email: {item.email_id}, Due: {item.due_date}")
```

## Meeting Detection

Extract meeting requests from emails.

```python
from aragora.services.meeting_detector import MeetingDetector

detector = MeetingDetector()

# Detect meeting in email
meeting = await detector.detect(email)
if meeting:
    print(f"Proposed time: {meeting.proposed_time}")
    print(f"Duration: {meeting.duration}")
    print(f"Attendees: {meeting.attendees}")
    print(f"Location: {meeting.location}")
```

## Action Item Extraction

Extract action items from email threads.

```python
from aragora.services.action_item_extractor import ActionItemExtractor

extractor = ActionItemExtractor()

# Extract from email thread
actions = await extractor.extract(email_thread)
for action in actions:
    print(f"Task: {action.description}")
    print(f"Assignee: {action.assignee}")
    print(f"Due: {action.due_date}")
    print(f"Priority: {action.priority}")
```

## Inbox SLA Monitor

Track response time SLAs.

```python
from aragora.services.inbox_sla_monitor import InboxSLAMonitor

monitor = InboxSLAMonitor(store=store)

# Define SLAs
await monitor.add_sla({
    "name": "VIP Response",
    "sender_pattern": "*@vip-client.com",
    "response_time_hours": 4,
})

# Check SLA status
status = await monitor.check_slas(user_id)
for sla in status.breached:
    print(f"Breached: {sla.email_id}, Overdue by: {sla.overdue_hours}h")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_EMAIL_TIER1_THRESHOLD` | Tier 1 confidence threshold | `0.7` |
| `ARAGORA_EMAIL_TIER2_THRESHOLD` | Tier 2 confidence threshold | `0.6` |
| `ARAGORA_SPAM_PHISHING_SENSITIVITY` | Phishing detection sensitivity | `0.8` |
| `ARAGORA_THREAT_INTEL_ENABLED` | Enable threat intelligence | `true` |

## API Reference

### Module Exports

```python
from aragora.services import (
    # Registry
    ServiceRegistry,
    ServiceNotFoundError,
    ServiceScope,
    ServiceDescriptor,
    RegistryStats,
    get_service,
    register_service,
    has_service,

    # Cache markers
    MethodCacheService,
    QueryCacheService,
    EmbeddingCacheService,
    HandlerCacheService,
    EmbeddingProviderService,

    # Email Prioritization
    EmailPrioritizer,
    EmailPriorityResult,
    EmailPriority,
    EmailPrioritizationConfig,
    prioritize_inbox,

    # Cross-Channel
    CrossChannelContextService,
    ChannelContext,
    EmailContextBoost,
    SlackActivitySignal,
    create_context_service,

    # Sender History
    SenderHistoryService,
    SenderStats,
    SenderReputation,
    create_sender_history_service,

    # Cost Optimization
    CostOptimizedPrioritizer,
    CostConfig,
    UsageStats,
    create_cost_optimized_prioritizer,

    # Multi-Inbox
    MultiInboxManager,
    InboxAccount,
    UnifiedEmail,
    CrossAccountSenderProfile,
    AccountType,
    create_multi_inbox_manager,

    # Threat Intelligence
    ThreatIntelligenceService,
    ThreatIntelConfig,
    ThreatResult,
    ThreatType,
    ThreatSeverity,
    ThreatSource,
    IPReputationResult,
    FileHashResult,
    check_threat,

    # Spam Classification
    SpamClassifier,
    SpamClassifierConfig,
    SpamClassificationResult,
    SpamCategory,
    classify_email_spam,
)
```

## See Also

- [Privacy Module](../privacy/README.md) - Data protection services
- [Security Module](../security/README.md) - Encryption and threat protection
- [Connectors](../connectors/README.md) - External service integrations
- [Knowledge Mound](../knowledge/mound/README.md) - Knowledge storage backend
