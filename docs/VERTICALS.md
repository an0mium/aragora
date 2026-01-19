# Vertical Specialists

Domain-specific AI agents with specialized prompts, tools, compliance checking, and model configurations.

## Overview

The Vertical Specialist System provides domain-specific AI agents that extend the base `APIAgent` with:

- **Domain-Specific System Prompts** - Jinja2 templates tailored to each vertical's expertise
- **Specialized Tools** - Domain tools like code search, legal case lookup, or PubMed search
- **Compliance Checking** - Automated checks against frameworks like OWASP, HIPAA, SOX, GDPR
- **Fine-Tuned Model Support** - Optional HuggingFace models and LoRA adapters

## Available Verticals

| Vertical | ID | Expertise Areas | Compliance Frameworks |
|----------|----|-----------------|-----------------------|
| Software Engineering | `software` | Code Review, Security Analysis, Architecture Design | OWASP, CWE |
| Legal | `legal` | Contract Analysis, Regulatory Compliance, Risk Assessment | GDPR, CCPA, HIPAA |
| Healthcare | `healthcare` | Clinical Documentation, Medical Research, HIPAA Compliance | HIPAA, HITECH, FDA 21 CFR Part 11 |
| Accounting | `accounting` | Financial Statement Analysis, Audit & Assurance, SOX Compliance | SOX, GAAP, PCAOB |
| Research | `research` | Research Methodology, Statistical Analysis, Literature Review | IRB, CONSORT, PRISMA |

---

## Quick Start

### Python SDK

```python
from aragora.verticals import VerticalRegistry

# Create a vertical specialist
software_agent = VerticalRegistry.create_specialist(
    "software",
    name="code-reviewer-1",
    model="claude-sonnet-4",
)

# Generate a response
response = await software_agent.respond(
    task="Review this code for security vulnerabilities",
    prompt_context={"code": code_to_review}
)

# Perform code review
review_results = await software_agent.review_code(
    code=code_to_review,
    language="python",
    focus_areas=["security", "quality"]
)
```

### REST API

```bash
# List available verticals
curl http://localhost:8080/api/verticals

# Get vertical configuration
curl http://localhost:8080/api/verticals/software

# Suggest best vertical for a task
curl "http://localhost:8080/api/verticals/suggest?task=review+this+contract"

# Create a specialist agent
curl -X POST http://localhost:8080/api/verticals/software/agent \
  -H "Content-Type: application/json" \
  -d '{"name": "reviewer-1", "model": "claude-sonnet-4"}'

# Run a vertical-specific debate
curl -X POST http://localhost:8080/api/verticals/legal/debate \
  -H "Content-Type: application/json" \
  -d '{"topic": "Review this NDA for risks", "rounds": 3}'
```

---

## Vertical Details

### Software Engineering Specialist

Expert in software development, code review, security analysis, and architecture design.

**Expertise Areas:**
- Code Review
- Security Analysis
- Architecture Design
- Performance Optimization
- Testing Strategy
- API Design
- Database Design
- DevOps & CI/CD
- Technical Documentation

**Tools:**
| Tool | Description |
|------|-------------|
| `code_search` | Search codebase for patterns or symbols |
| `security_scan` | Run security analysis on code |
| `dependency_check` | Check for vulnerable dependencies |
| `github_lookup` | Look up GitHub issues or PRs |

**Compliance Frameworks:**
- **OWASP Top 10 (2021)** - A01-A10 vulnerability checks
- **CWE** - Common Weakness Enumeration (CWE-20, CWE-78, CWE-79, CWE-89, CWE-200, CWE-502)

**Security Patterns Detected:**
- SQL Injection
- Command Injection
- Cross-Site Scripting (XSS)
- Hardcoded Secrets

**Usage Example:**
```python
software_specialist = VerticalRegistry.create_specialist("software", name="reviewer")

# Review code for security issues
review = await software_specialist.review_code(
    code=source_code,
    language="python",
    focus_areas=["security", "quality", "performance"]
)

# Check compliance
violations = await software_specialist.check_compliance(source_code)
```

---

### Legal Specialist

Expert in contract analysis, compliance review, and regulatory matters.

**Expertise Areas:**
- Contract Analysis
- Regulatory Compliance
- Risk Assessment
- Legal Research
- Document Review
- Due Diligence
- Privacy Law
- Intellectual Property
- Employment Law

**Tools:**
| Tool | Description |
|------|-------------|
| `case_search` | Search legal case databases (Westlaw integration) |
| `statute_lookup` | Look up statutes and regulations |
| `contract_compare` | Compare contract versions |

**Compliance Frameworks:**
- **GDPR** - Data processing, consent, rights, transfers, breach notification
- **CCPA** - Disclosure, opt-out, deletion, nondiscrimination
- **HIPAA** - Privacy, security, breach notification (enforced)

**Clause Patterns Detected:**
- Indemnification clauses
- Limitation of liability
- Termination provisions
- Confidentiality clauses
- Intellectual property terms

**Usage Example:**
```python
legal_specialist = VerticalRegistry.create_specialist("legal", name="contract-reviewer")

# Analyze a contract
analysis = await legal_specialist.analyze_contract(
    contract_text=contract_content,
    focus_areas=["indemnification", "termination", "confidentiality"]
)

# Check for GDPR compliance
violations = await legal_specialist.check_compliance(
    contract_content,
    framework="GDPR"
)
```

---

### Healthcare Specialist

Expert in clinical analysis, medical research, and health informatics.

**Expertise Areas:**
- Clinical Documentation
- Medical Research
- Health Informatics
- HIPAA Compliance
- Drug Interactions
- Clinical Trials
- Patient Safety
- Healthcare Analytics
- Medical Coding

**Tools:**
| Tool | Description |
|------|-------------|
| `pubmed_search` | Search PubMed for medical literature |
| `drug_lookup` | Look up drug information and interactions |
| `icd_lookup` | Look up ICD-10 codes |
| `clinical_guidelines` | Search clinical practice guidelines |

**Compliance Frameworks:**
- **HIPAA** - Privacy Rule, Security Rule, Breach Notification, Minimum Necessary (enforced)
- **HITECH** - Breach notification, EHR incentives, enforcement (enforced)
- **FDA 21 CFR Part 11** - Electronic records, signatures, audit trails

**PHI Detection:**
The healthcare specialist automatically detects Protected Health Information (PHI) using HIPAA Safe Harbor patterns:
- Names, dates, phone numbers, email addresses
- Social Security Numbers, Medical Record Numbers
- Physical addresses

**Usage Example:**
```python
healthcare_specialist = VerticalRegistry.create_specialist("healthcare", name="clinical-reviewer")

# Analyze a clinical document
analysis = await healthcare_specialist.analyze_clinical_document(
    document_text=clinical_note,
    document_type="clinical_note"
)

# Check de-identification status
deidentification = await healthcare_specialist.check_deidentification(document_content)
```

---

### Accounting & Finance Specialist

Expert in financial analysis, audit, compliance, and accounting standards.

**Expertise Areas:**
- Financial Statement Analysis
- Audit & Assurance
- SOX Compliance
- Tax Planning
- Internal Controls
- Revenue Recognition
- Cost Accounting
- Financial Reporting
- Regulatory Compliance

**Tools:**
| Tool | Description |
|------|-------------|
| `sec_filings` | Search SEC filings and documents |
| `gaap_lookup` | Look up GAAP accounting standards |
| `ratio_calculator` | Calculate financial ratios |
| `tax_reference` | Look up tax regulations and rates |

**Compliance Frameworks:**
- **SOX** - Section 302, 404, 802, 906 (enforced)
- **GAAP** - Revenue recognition, fair value, leases, disclosure (enforced)
- **PCAOB** - AS 2201, AS 3101, AS 2110

**Financial Ratios Supported:**
- Current Ratio, Quick Ratio
- Debt to Equity, Return on Equity
- Gross Margin, Net Margin

**Fraud Indicators Detected:**
- Management override, circumvention
- Unusual transactions or adjustments
- Significant related party transactions
- Aggressive accounting practices

**Usage Example:**
```python
accounting_specialist = VerticalRegistry.create_specialist("accounting", name="auditor")

# Analyze financial statements
analysis = await accounting_specialist.analyze_financial_statement(
    statement_text=financial_data,
    statement_type="income_statement"
)

# Review internal controls
control_review = await accounting_specialist.review_internal_controls(
    control_description=control_documentation
)

# Calculate financial ratios
ratios = await accounting_specialist.invoke_tool("ratio_calculator", {
    "current_assets": 1000000,
    "current_liabilities": 500000,
    "revenue": 5000000,
    "net_income": 750000
})
```

---

### Research Specialist

Expert in research methodology, literature analysis, and scientific writing.

**Expertise Areas:**
- Literature Review
- Research Methodology
- Statistical Analysis
- Scientific Writing
- Peer Review
- Research Ethics
- Data Analysis
- Citation Analysis
- Meta-Analysis

**Tools:**
| Tool | Description |
|------|-------------|
| `arxiv_search` | Search arXiv for preprints |
| `pubmed_search` | Search PubMed for medical literature |
| `semantic_scholar` | Search Semantic Scholar for papers |
| `citation_check` | Verify citations and check for retractions |

**Compliance Frameworks:**
- **IRB** - Informed consent, minimal risk, privacy, vulnerable populations (enforced)
- **CONSORT (2010)** - Randomization, blinding, outcomes, sample size, flow diagram
- **PRISMA (2020)** - Search strategy, selection, synthesis, bias assessment

**Methodology Patterns Detected:**
- Study designs (RCT, cohort, case-control, meta-analysis)
- Statistical methods (t-test, ANOVA, chi-square, regression)
- Sampling approaches (random, convenience, stratified)
- Bias indicators (selection, confirmation, publication)

**Citation Styles Detected:**
- APA, MLA, Chicago
- DOI extraction

**Usage Example:**
```python
research_specialist = VerticalRegistry.create_specialist("research", name="methodology-reviewer")

# Analyze research methodology
methodology = await research_specialist.analyze_methodology(paper_text)

# Analyze citations
citations = await research_specialist.analyze_citations(paper_text)

# Check IRB compliance
violations = await research_specialist.check_compliance(
    paper_text,
    framework="IRB"
)
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/verticals` | List all available verticals |
| GET | `/api/verticals/:id` | Get vertical configuration |
| GET | `/api/verticals/:id/tools` | Get tools for a vertical |
| GET | `/api/verticals/:id/compliance` | Get compliance frameworks |
| GET | `/api/verticals/suggest?task=...` | Suggest best vertical for task |
| POST | `/api/verticals/:id/agent` | Create specialist agent instance |
| POST | `/api/verticals/:id/debate` | Run vertical-specific debate |

### Request/Response Examples

**List Verticals:**
```bash
GET /api/verticals

Response:
{
  "verticals": [
    {
      "vertical_id": "software",
      "display_name": "Software Engineering Specialist",
      "description": "Expert in software development...",
      "expertise_areas": ["Code Review", "Security Analysis", ...],
      "tools": ["code_search", "security_scan", ...],
      "compliance_frameworks": ["OWASP", "CWE"],
      "default_model": "claude-sonnet-4"
    },
    ...
  ],
  "total": 5
}
```

**Create Debate with Specialist:**
```bash
POST /api/verticals/legal/debate
Content-Type: application/json

{
  "topic": "Review this SaaS agreement for risks",
  "rounds": 3,
  "consensus": "weighted",
  "model": "claude-sonnet-4",
  "additional_agents": [
    {"type": "anthropic-api", "role": "critic"},
    {"type": "openai-api", "role": "synthesizer"}
  ]
}

Response:
{
  "debate_id": "dbt_abc123",
  "vertical_id": "legal",
  "topic": "Review this SaaS agreement for risks",
  "consensus_reached": true,
  "final_answer": "...",
  "confidence": 0.87,
  "participants": ["legal-specialist", "anthropic-critic", "openai-synthesizer"]
}
```

---

## Configuration

### VerticalConfig Schema

```python
@dataclass
class VerticalConfig:
    vertical_id: str          # Unique identifier
    display_name: str         # Human-readable name
    description: str          # Description of capabilities

    domain_keywords: List[str]     # Keywords for auto-detection
    expertise_areas: List[str]     # Areas of expertise

    system_prompt_template: str    # Jinja2 template for system prompt

    tools: List[ToolConfig]                  # Domain tools
    compliance_frameworks: List[ComplianceConfig]  # Compliance checks

    model_config: ModelConfig      # Model settings

    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = []
```

### Loading from YAML

```yaml
# verticals/custom_vertical.yaml
vertical_id: custom
display_name: Custom Vertical
description: Custom domain specialist

domain_keywords:
  - custom
  - domain

expertise_areas:
  - Area 1
  - Area 2

system_prompt_template: |
  You are a custom specialist with expertise in:
  {% for area in expertise_areas %}
  - {{ area }}
  {% endfor %}

tools:
  - name: custom_tool
    description: Custom tool description
    enabled: true
    connector_type: custom

compliance_frameworks:
  - framework: CUSTOM
    version: "1.0"
    level: warning
    rules: ["rule1", "rule2"]

model_config:
  primary_model: claude-sonnet-4
  temperature: 0.3
```

```python
config = VerticalConfig.from_yaml("verticals/custom_vertical.yaml")
```

---

## Compliance Levels

| Level | Behavior |
|-------|----------|
| `advisory` | Log suggestions only |
| `warning` | Warn on violations (default) |
| `enforced` | Block output on violations |

```python
from aragora.verticals.config import ComplianceLevel

# Check if blocking is required
violations = await specialist.check_compliance(content)
should_block = specialist.should_block_on_compliance(violations)
```

---

## Creating Custom Verticals

### 1. Define Configuration

```python
from aragora.verticals.config import (
    VerticalConfig, ToolConfig, ComplianceConfig,
    ComplianceLevel, ModelConfig
)

CUSTOM_CONFIG = VerticalConfig(
    vertical_id="custom",
    display_name="Custom Specialist",
    description="Expert in custom domain",
    domain_keywords=["custom", "domain"],
    expertise_areas=["Area 1", "Area 2"],
    system_prompt_template="...",
    tools=[
        ToolConfig(name="custom_tool", description="..."),
    ],
    compliance_frameworks=[
        ComplianceConfig(
            framework="CUSTOM",
            level=ComplianceLevel.WARNING,
            rules=["rule1"],
        ),
    ],
    model_config=ModelConfig(
        primary_model="claude-sonnet-4",
        temperature=0.3,
    ),
)
```

### 2. Implement Specialist Class

```python
from aragora.verticals.base import VerticalSpecialistAgent
from aragora.verticals.registry import VerticalRegistry

@VerticalRegistry.register(
    "custom",
    config=CUSTOM_CONFIG,
    description="Custom domain specialist",
)
class CustomSpecialist(VerticalSpecialistAgent):

    async def _execute_tool(self, tool, parameters):
        if tool.name == "custom_tool":
            return await self._custom_tool(parameters)
        return {"error": f"Unknown tool: {tool.name}"}

    async def _custom_tool(self, parameters):
        # Implement tool logic
        return {"result": "..."}

    async def _check_framework_compliance(self, content, framework):
        if framework.framework == "CUSTOM":
            return await self._check_custom_compliance(content, framework)
        return []

    async def _generate_response(self, task, system_prompt, context=None, **kwargs):
        # Generate response using the model
        return Message(
            role="assistant",
            content="...",
            agent=self.name,
        )
```

### 3. Use the Custom Vertical

```python
# Import triggers registration
from my_verticals.custom import CustomSpecialist

# Create specialist instance
agent = VerticalRegistry.create_specialist(
    "custom",
    name="custom-agent-1",
    model="claude-sonnet-4",
)
```

---

## Best Practices

1. **Choose the Right Vertical** - Use `VerticalRegistry.get_for_task(description)` to auto-select the best vertical
2. **Configure Compliance Appropriately** - Use `enforced` for critical regulations, `warning` for best practices
3. **Audit Tool Calls** - Use `agent.get_tool_call_history()` for compliance auditing
4. **Add Disclaimers** - Each specialist includes appropriate disclaimers (e.g., "not legal advice")
5. **Combine with General Agents** - Use specialists alongside general agents in debates for diverse perspectives

---

## See Also

- [API Reference](API_REFERENCE.md) - Full API documentation
- [Agents Guide](AGENTS.md) - General agent configuration
- [Compliance Presets](COMPLIANCE_PRESETS.md) - Built-in compliance configurations
- [Knowledge Mound](KNOWLEDGE_MOUND.md) - Knowledge base integration
