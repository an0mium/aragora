# Deprecated: Aragora Implementation Roadmap

This roadmap is historical and no longer maintained. For current capability
status and planned work, use:
- `docs/FEATURES.md`
- `docs/CONTROL_PLANE_GUIDE.md`
- `docs/COMMERCIAL_POSITIONING.md`

**Control Plane for Multi-Agent Knowledge Work**

*Target: SMEs in Accounting, Legal, Healthcare, and Software verticals*

---

## Executive Summary

This roadmap transforms Aragora from a multi-agent debate platform into a complete **control plane for knowledge work** that helps SMEs:
- Process unruly inboxes and surface urgent, relevant messages
- Crawl, understand, and audit codebases for bugs and security issues
- Implement features using multi-agent flows
- Integrate with vertical-specific tools (QuickBooks, legal CLM, healthcare EHR)

---

## Current State (v2.1.11)

| Component | Status | Key Stats |
|-----------|--------|-----------|
| Email Prioritization | Production-ready | 869 lines, 3-tier scoring |
| Triage Rules UI | Complete | 700 lines, rule builder |
| Code Intelligence | Solid | 1740 lines, 5 languages |
| Security Scanner | Solid | 1025 lines, 50+ patterns |
| Control Plane | Foundation | 13 modules |
| Connectors | Extensive | 45+ (missing QBO) |
| Test Coverage | Comprehensive | 43,974+ tests |

---

## Phase 1: Quick Wins (2-3 Weeks)

### 1.1 Unified Inbox Command Center
**Goal**: Single view for email triage with quick actions

**Files to Create**:
```
aragora/live/src/app/(app)/command-center/page.tsx
aragora/live/src/components/inbox/CommandCenter.tsx
aragora/live/src/components/inbox/QuickActionsBar.tsx
aragora/live/src/components/inbox/SenderInsightsPanel.tsx
aragora/live/src/components/inbox/DailyDigestWidget.tsx
aragora/server/handlers/inbox_command.py
```

**Features**:
- Combined PriorityInboxList + TriageRulesPanel view
- Quick Actions bar: archive defer, snooze low, respond to critical
- Sender insights panel (VIP status, history, response patterns)
- Daily digest summary widget
- Keyboard shortcuts for power users

**Integration Points**:
- `/api/email/inbox` - Existing endpoint
- `EmailPrioritizer.record_user_action()` - Learning loop
- `CostOptimizedPrioritizer` - Budget-aware processing

**Success Metrics**:
- Time to zero inbox reduced by 50%
- User actions per email session increased

---

### 1.2 One-Click Security Scan Wizard
**Goal**: Guided security audit with executive summary

**Files to Create**:
```
aragora/live/src/app/(app)/security-scan/page.tsx
aragora/live/src/components/codebase/SecurityScanWizard.tsx
aragora/live/src/components/codebase/ScanProgressView.tsx
aragora/live/src/components/codebase/FindingsSummary.tsx
aragora/live/src/components/codebase/ReportExporter.tsx
aragora/server/handlers/codebase/quick_scan.py
```

**Features**:
- 3-step wizard: Connect repo -> Scan -> Review
- Real-time scan progress with streaming updates
- Executive summary with risk score
- Detailed findings grouped by severity
- PDF/HTML export for stakeholders
- "Fix with AI" button for each finding

**Integration Points**:
- `SecurityScanner` - Pattern-based detection
- `DependencyScanner` - CVE lookup
- `CodeIntelligence` - AST analysis
- `RepositoryCrawler` - File access

**Success Metrics**:
- Time from repo URL to report < 5 minutes
- Critical findings detection rate > 95%

---

### 1.3 Cost Visibility Dashboard
**Goal**: Transparent usage and budget tracking

**Files to Create**:
```
aragora/live/src/components/control-plane/CostDashboard.tsx
aragora/live/src/components/control-plane/UsageChart.tsx
aragora/live/src/components/control-plane/BudgetAlerts.tsx
aragora/live/src/components/control-plane/FeatureBreakdown.tsx
aragora/server/handlers/admin/cost_summary.py
```

**Features**:
- Daily/monthly spend visualization
- Budget alerts and projections
- Per-feature breakdown (inbox, code analysis, debates)
- Tier distribution chart
- Export for accounting

**Integration Points**:
- `CostOptimizedPrioritizer.get_usage_stats()`
- `aragora/billing/cost_tracker.py`
- `aragora/tenancy/quotas.py`

---

### 1.4 Vertical Quick-Start Templates
**Goal**: One-click workflow deployment

**Files to Create**:
```
aragora/live/src/components/workflow-builder/QuickStartGallery.tsx
aragora/live/src/components/workflow-builder/TemplateCard.tsx
aragora/live/src/components/workflow-builder/DeploymentWizard.tsx
aragora/workflow/quick_deploy.py
```

**Templates to Surface**:
- Financial Audit (SOX compliance)
- Code Security Review
- Contract Analysis
- Email Triage Setup

---

## Phase 2: Core Differentiators (4-6 Weeks)

### 2.1 QuickBooks Online Connector
**Goal**: Full accounting data integration

**Files to Create**:
```
aragora/connectors/enterprise/accounting/__init__.py
aragora/connectors/enterprise/accounting/quickbooks.py
aragora/connectors/enterprise/accounting/models.py
aragora/connectors/enterprise/accounting/sync.py
aragora/live/src/components/connectors/QuickBooksSetup.tsx
tests/connectors/enterprise/test_quickbooks.py
```

**Capabilities**:
```python
class QuickBooksConnector(BaseConnector):
    # Read operations
    async def get_invoices(since: datetime) -> List[Invoice]
    async def get_expenses(since: datetime) -> List[Expense]
    async def get_profit_loss(start: date, end: date) -> Report
    async def get_balance_sheet(as_of: date) -> Report
    async def get_accounts() -> List[Account]

    # Analysis operations
    async def detect_anomalies(transactions: List) -> List[Anomaly]
    async def suggest_categorization(expense: Expense) -> Category
    async def reconcile_accounts(bank_data: List) -> ReconciliationResult

    # Write operations (Phase 2+)
    async def create_invoice(invoice: InvoiceCreate) -> Invoice
    async def categorize_expense(id: str, category: str) -> Expense
```

**OAuth Flow**:
1. User clicks "Connect QuickBooks"
2. Redirect to Intuit OAuth
3. Callback stores tokens
4. Initial sync of last 90 days
5. Incremental sync every hour

**Integration Points**:
- `AccountingSpecialist` for GAAP/SOX guidance
- `FinancialAuditWorkflow` for anomaly review
- Knowledge Mound for financial context

---

### 2.2 Multi-Agent Code Review Workflow
**Goal**: AI-powered PR reviews with consensus

**Files to Create**:
```
aragora/workflows/presets/code_review_debate.py
aragora/agents/code_review_agents.py
aragora/live/src/components/codebase/CodeReviewPanel.tsx
aragora/live/src/components/codebase/ReviewConversation.tsx
aragora/server/handlers/codebase/review.py
```

**Agent Roles**:
```python
REVIEW_AGENTS = [
    SecurityAuditor(
        focus=["vulnerabilities", "input validation", "auth"],
        uses=[SecurityScanner, DependencyScanner]
    ),
    PerformanceCritic(
        focus=["complexity", "memory", "algorithms"],
        uses=[CodeIntelligence, CallGraph]
    ),
    StyleReviewer(
        focus=["patterns", "conventions", "documentation"],
        uses=[CodeIntelligence]
    ),
    ArchitectureAdvisor(
        focus=["coupling", "cohesion", "dependencies"],
        uses=[CallGraph, DependencyAnalyzer]
    )
]
```

**Workflow**:
1. PR webhook triggers review
2. CodeIntelligence extracts diff context
3. Each agent reviews independently
4. Debate synthesizes findings
5. Post consolidated review to PR
6. Track resolution status

---

### 2.3 Inbox Action Execution
**Goal**: Rules that actually execute

**Files to Create**:
```
aragora/services/inbox_actions.py
aragora/live/src/components/inbox/ActionConfirmDialog.tsx
aragora/live/src/components/inbox/ActionHistoryPanel.tsx
aragora/server/handlers/inbox_actions.py
```

**Supported Actions**:
```python
class InboxActionExecutor:
    async def execute(action: RuleAction, email: Email) -> Result:
        match action.type:
            case "label": return await self.gmail.add_label(...)
            case "archive": return await self.gmail.archive(...)
            case "forward": return await self.gmail.forward(...)
            case "assign": return await self.create_task(...)
            case "escalate": return await self.notify_team(...)
            case "reply_draft": return await self.create_draft(...)
            case "schedule_followup": return await self.create_reminder(...)
            case "notify_slack": return await self.slack.post(...)
```

**Safety Features**:
- Confirmation dialog for destructive actions
- Undo window (30 seconds)
- Action history log
- Rate limiting per rule

---

### 2.4 Workspace Budget Controls
**Goal**: Enforce spending limits

**Files to Create**:
```
aragora/control_plane/budget_enforcer.py
aragora/live/src/components/control-plane/BudgetManager.tsx
aragora/live/src/components/control-plane/BudgetAlertConfig.tsx
aragora/server/handlers/admin/budgets.py
```

**Budget Model**:
```python
@dataclass
class WorkspaceBudget:
    monthly_limit_usd: float
    daily_limit_usd: float
    alert_threshold_percent: float = 80.0
    hard_stop_enabled: bool = False

    # Per-feature budgets
    inbox_budget_usd: Optional[float] = None
    code_analysis_budget_usd: Optional[float] = None
    debate_budget_usd: Optional[float] = None

    # Rollover
    rollover_enabled: bool = False
    rollover_max_percent: float = 20.0
```

**Enforcement**:
- Soft limit: Degrade to Tier 1 processing
- Hard limit: Block new requests with friendly message
- Alert channels: Email, Slack, in-app

---

## Phase 3: Full Vertical Support (8+ Weeks)

### 3.1 Accounting Vertical Enhancement
- Xero connector (second accounting platform)
- Monthly close workflow template
- Automated reconciliation with bank feeds
- Expense pattern anomaly detection
- Financial ratio dashboard
- Integration with payroll (Gusto, ADP)

### 3.2 Legal Vertical Enhancement
- Clio connector (practice management)
- DocuSign connector (e-signatures)
- Contract clause extraction and risk scoring
- Deadline tracking from document analysis
- Client matter organization
- Billing time entry automation

### 3.3 Healthcare Vertical Enhancement
- EHR bridge (Epic, Cerner, Athena patterns)
- Enhanced PHI detection and redaction
- Clinical documentation assistance
- Appointment scheduling from email
- Prior authorization workflow
- HIPAA audit logging

### 3.4 Unified SME Dashboard
- Widget-based customizable layout
- Vertical-specific widgets
- Cross-vertical insights
- One-click access to all features
- Mobile-responsive design

---

## Competitive Positioning

| Capability | Aragora | CrewAI | AutoGen | LangGraph |
|------------|---------|--------|---------|-----------|
| Multi-agent debate consensus | ✅ | ❌ | Partial | ❌ |
| 3-tier cost optimization | ✅ | ❌ | ❌ | ❌ |
| Vertical specialists | ✅ | ❌ | ❌ | ❌ |
| Cross-channel context | ✅ | ❌ | ❌ | ❌ |
| Security scanning | ✅ | ❌ | ❌ | ❌ |
| SME-focused UI | ✅ | ❌ | ❌ | ❌ |
| QuickBooks integration | Phase 2 | ❌ | ❌ | ❌ |

---

## Success Metrics

### Phase 1
- Inbox zero time reduced 50%
- Security scan to report < 5 minutes
- Template deployment < 2 minutes
- Cost visibility adoption > 80%

### Phase 2
- QBO connected workspaces > 100
- Code reviews with AI consensus > 500/month
- Action automation saves 2+ hours/week
- Budget compliance > 95%

### Phase 3
- Vertical-specific workflow adoption
- Cross-vertical insights generated
- SME satisfaction score > 4.5/5

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| QBO API rate limits | Implement smart caching and batching |
| HIPAA compliance | Use existing PHI detection + audit logging |
| Budget overruns | Graceful degradation to Tier 1 |
| Adoption friction | One-click templates + onboarding wizard |

---

## Timeline

```
Week 1-2:  Phase 1.1 (Inbox Command Center)
Week 2-3:  Phase 1.2 (Security Scan Wizard)
Week 3:    Phase 1.3 (Cost Dashboard) + 1.4 (Templates)
Week 4-5:  Phase 2.1 (QuickBooks Connector)
Week 6:    Phase 2.2 (Code Review Workflow)
Week 7:    Phase 2.3 (Inbox Actions) + 2.4 (Budget Controls)
Week 8+:   Phase 3 (Vertical Enhancements)
```

---

*Last updated: 2026-01-22*
*Status: Implementation in progress*
