---
title: Vetted Decisionmaking Templates
description: Vetted Decisionmaking Templates
---

# Vetted Decisionmaking Templates

Aragora ships with built-in vetted decisionmaking templates for common enterprise
workflows (code review, security audit, compliance, inbox triage). The template
API and module names retain the `deliberation` namespace for compatibility:
`aragora/deliberation/templates/`. Templates provide defaults for agent selection,
knowledge sources, output formats, and consensus thresholds.

## Key Types

```python
from aragora.deliberation.templates import (
    DeliberationTemplate,
    TemplateCategory,
    TeamStrategy,
    OutputFormat,
)
```

### TemplateCategory
`code`, `legal`, `finance`, `healthcare`, `compliance`, `academic`, `general`

### TeamStrategy
`specified`, `best_for_domain`, `diverse`, `fast`, `random`

### OutputFormat
`standard`, `decision_receipt`, `summary`, `github_review`, `slack_message`,
`jira_comment`, `confluence_page`, `email`, `compliance_report`

## Built-In Templates

| Template | Category | Output | Notes |
|----------|----------|--------|-------|
| `code_review` | code | github_review | PR review with security/perf/readability personas |
| `security_audit` | code | decision_receipt | OWASP + compliance personas |
| `architecture_decision` | code | decision_receipt | Trade-off analysis |
| `contract_review` | legal | decision_receipt | Contract risk analysis |
| `due_diligence` | legal | decision_receipt | M&A / investment diligence |
| `financial_audit` | finance | compliance_report | Financial statement audit |
| `risk_assessment` | finance | decision_receipt | Enterprise risk review |
| `hipaa_compliance` | healthcare | compliance_report | HIPAA controls review |
| `clinical_review` | healthcare | decision_receipt | Evidence-based guidance |
| `compliance_check` | compliance | compliance_report | Framework gap check |
| `soc2_audit` | compliance | compliance_report | SOC 2 readiness |
| `gdpr_assessment` | compliance | compliance_report | GDPR assessment |
| `citation_verification` | academic | standard | Citation validation |
| `peer_review` | academic | decision_receipt | Academic peer review |
| `quick_decision` | general | summary | Fast, low-consensus path |
| `research_analysis` | general | decision_receipt | Research synthesis |
| `brainstorm` | general | standard | Divergent ideation |
| `email_prioritization` | general | summary | Inbox priority triage |
| `inbox_triage` | general | standard | Batch categorization |
| `meeting_prep` | general | summary | Meeting prep across sources |

## Usage

### Load a template

```python
from aragora.deliberation.templates import get_template

template = get_template("security_audit")
request = template.merge_with_request({
    "question": "Assess auth architecture risk",
    "knowledge_sources": ["github:owner/repo", "confluence"],
})
```

### List templates

```python
from aragora.deliberation.templates import list_templates

templates = list_templates(category="compliance")
```

## Customize

Templates are plain data structures. You can override any field by passing
values in your request (agents, consensus threshold, output format, etc.).
