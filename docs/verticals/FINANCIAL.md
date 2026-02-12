# Aragora for Financial Services

**Audit-Grade Decision Integrity for Risk, Compliance, and Investment Workflows**

---

## The Problem

Financial services decisions are subject to extraordinary scrutiny. A credit committee approving a loan, an investment team evaluating an acquisition, a compliance officer assessing SOX controls -- every material decision may eventually be examined by regulators, auditors, or litigants. The question is never just "what did you decide?" but "how did you decide, what alternatives did you consider, and can you prove it?"

Traditional AI tools in financial services produce recommendations without adversarial challenge. A single model scores a credit application or flags a transaction, and the reasoning behind that score is opaque. When a regulator asks why a decision was made, the answer is a black box.

Aragora brings adversarial multi-agent debate to financial decision-making. Multiple independent AI models analyze, challenge, and refine each other's reasoning before producing a consensus recommendation backed by a cryptographic Decision Receipt. Every argument, every counterargument, every piece of evidence cited is documented in an immutable audit trail suitable for regulatory examination.

---

## How Aragora Applies to Financial Services

### Risk Assessment and Underwriting

Aragora's debate engine turns risk assessment from a single-model score into a documented deliberation:

- **Credit decisions**: Multiple models independently assess creditworthiness, then debate their conclusions. One model may focus on debt-to-income ratios while another examines industry-specific default patterns, and a third stress-tests the assessment under adverse scenarios.
- **Underwriting**: For insurance or lending underwriting, agents analyze risk factors from different angles, challenging each other's assumptions about risk concentration, correlation, and tail events.
- **Fraud investigation**: When a transaction is flagged, agents debate the evidence for and against fraud, ensuring that legitimate transactions are not blocked by a single model's false positive.

### Audit-Grade Decision Trails

Every Aragora debate produces a Decision Receipt with:

- **SHA-256 cryptographic integrity hash**: Proves the receipt has not been modified after generation
- **Full argument chain**: Every agent's position, every critique, every revision -- documented with timestamps
- **Dissenting views**: When agents disagree, the dissent is explicitly recorded. Regulators see not just the decision, but the full range of considerations
- **Evidence citations**: Specific regulations, standards, and data points referenced by each agent
- **Export formats**: Markdown, HTML, SARIF, and CSV for integration with existing audit systems

### SOX Compliance Integration

Aragora includes a pre-built SOX compliance workflow and evaluation profile:

- **`compliance_sox` weight profile**: Prioritizes accuracy (25%) and completeness (25%) with evidence (15%) -- designed for control assessments where missing a material weakness is unacceptable
- **SOC 2 audit template**: Five-agent debate covering security, availability, processing integrity, confidentiality, and privacy trust service criteria
- **Financial audit workflow template**: Multi-step workflow with preliminary analytical review, internal control testing, parallel substantive testing, findings synthesis, and materiality assessment
- **Human checkpoints**: Material findings automatically route to partner-level review; significant deficiencies route to manager review

---

## Use Cases

### 1. Credit Decision Review

**Scenario**: A lending team is reviewing a commercial loan application for $2M. The borrower's financials show strong revenue growth but increasing leverage.

**Without Aragora**: A single credit scoring model produces a number. An analyst reviews the file. The decision is documented in a brief memo.

**With Aragora**: Three independent models analyze the application. Model A focuses on cash flow coverage ratios and finds adequate coverage. Model B examines the leverage trend and flags that debt-to-EBITDA has increased from 2.5x to 3.8x over 18 months. Model C stress-tests the loan under a 20% revenue decline scenario and finds the borrower would breach coverage covenants. The resulting Decision Receipt documents all three perspectives, the consensus that the loan should be approved with modified covenants, and the specific conditions recommended.

**CLI**:
```bash
aragora ask "Evaluate this commercial loan application: Revenue $12M growing 15% YoY, \
  EBITDA $3.2M, total debt $12.2M (debt/EBITDA 3.8x, up from 2.5x 18 months ago), \
  requesting $2M term loan for equipment. Assess credit risk, appropriate covenants, \
  and stress scenario performance." \
  --vertical accounting \
  --rounds 5 \
  --consensus unanimous
```

### 2. Investment Committee Review

**Scenario**: An investment committee is evaluating a private equity acquisition target. The target has strong EBITDA but customer concentration risk and pending litigation.

**Without Aragora**: Each committee member reviews the CIM independently, discusses in a meeting, and votes. The reasoning is captured in brief meeting minutes.

**With Aragora**: The due diligence data is submitted to Aragora. Agents perform parallel analysis of corporate documents, financials, material contracts, IP, litigation, and compliance. The debate synthesizes findings, flags the customer concentration (top 3 customers = 68% of revenue) and litigation exposure ($4.2M contingent liability) as deal risks. The Decision Receipt becomes part of the investment committee's permanent record, documenting exactly why the deal was approved at an adjusted valuation.

**CLI**:
```bash
aragora ask "Evaluate acquisition target: SaaS company, $8M ARR, 85% gross margins, \
  but top 3 customers represent 68% of revenue. Pending IP litigation with \
  estimated $4.2M contingent liability. Current ask: 6x ARR ($48M). Assess \
  deal risk, fair valuation range, and required deal protections." \
  --vertical accounting \
  --enable-verticals \
  --rounds 5
```

### 3. Fraud Investigation

**Scenario**: The fraud detection system has flagged a series of wire transfers totaling $1.8M as potentially suspicious. The patterns match both legitimate vendor payments and known layering schemes.

**With Aragora**: Models debate the evidence. Agent A argues the transfer pattern matches the company's regular vendor payment schedule based on historical data. Agent B identifies that the receiving accounts were opened within the last 90 days and the transfer amounts are structured just below reporting thresholds. Agent C notes that the beneficiary entities share a registered agent with a previously flagged shell company. The consensus recommendation to escalate is documented with specific BSA/AML citations and the evidence chain.

### 4. M&A Due Diligence

Aragora includes a pre-built Legal Due Diligence workflow that runs parallel review streams:

- **Corporate document review**: Charter, bylaws, board minutes, shareholder agreements
- **Financial review**: Audited statements, tax filings, debt obligations, off-balance-sheet items
- **Material contracts review**: Change of control provisions, assignment restrictions, termination rights
- **IP review**: Patent registrations, trademarks, licensing agreements, IP litigation
- **Litigation review**: Pending cases, threatened claims, judgment liens, settlement history
- **Compliance review**: Regulatory licenses, compliance history, pending investigations

All streams feed into a multi-agent synthesis debate that produces a unified risk matrix and recommendation.

### 5. SOX Control Assessment

**CLI**:
```bash
aragora ask "Assess SOX Section 404 internal controls over financial reporting \
  for Q4 2025. Focus on: revenue recognition (ASC 606 compliance), accounts \
  receivable valuation (allowance methodology), and treasury operations \
  (cash management and wire transfer controls)." \
  --vertical accounting \
  --enable-verticals \
  --rounds 5 \
  --consensus unanimous
```

---

## Financial Evaluation Framework

### Weight Profiles

Aragora's financial weight profiles are calibrated for audit-grade accuracy:

| Dimension | `financial_audit` | `financial_risk` | `compliance_sox` | General |
|-----------|------------------:|------------------:|-----------------:|--------:|
| Accuracy | **30%** | 20% | **25%** | 15% |
| Completeness | **20%** | 15% | **25%** | 15% |
| Reasoning | 15% | **20%** | 10% | 25% |
| Evidence | 15% | 10% | 15% | 15% |
| Relevance | 10% | 15% | 10% | 15% |
| Safety | 5% | 5% | 10% | 5% |
| Clarity | 5% | 10% | 5% | 10% |
| Creativity | **0%** | 5% | **0%** | 0% |

Key design decisions:
- **Accuracy is 30% for financial audit** -- the highest of any profile. Material misstatements must be caught.
- **Completeness is 25% for SOX** -- a missed control deficiency can mean a material weakness finding.
- **Creativity is zero for audit and SOX profiles**. Financial reporting must follow established standards, not novel interpretations.
- **Financial risk allows 5% creativity** -- because risk assessment sometimes requires scenario imagination, but tightly bounded.

### Financial-Specific Rubrics

Agent contributions are evaluated against financial-domain criteria:

**Accuracy Rubric** -- "Are financial figures, calculations, and regulatory citations correct?"
- Score 1: Material financial errors or misstatements
- Score 3: Generally accurate with minor computational gaps
- Score 5: Audit-grade accuracy with verified calculations

**Completeness Rubric** -- "Does the analysis cover all required financial controls and standards?"
- Score 1: Missing critical SOX controls or financial areas
- Score 3: Covers main financial areas adequately
- Score 5: Complete coverage of all controls, standards, and risk areas

**Evidence Rubric** -- "Are conclusions supported by financial data, precedent, or regulation?"
- Score 1: No supporting financial evidence
- Score 3: Some financial data but not comprehensive
- Score 5: Audit-trail quality evidence with cross-referenced data

### Agent Team Composition

Financial debates use specialized personas from the accounting vertical:

| Persona | Role | Focus |
|---------|------|-------|
| `financial_auditor` | Primary analyst | GAAP/IFRS compliance, material misstatement detection |
| `internal_auditor` | Control testing | SOX controls, segregation of duties, process evaluation |
| `forensic_accountant` | Fraud detection | Irregularity identification, transaction tracing |
| `tax_specialist` | Tax compliance | Tax provision accuracy, transfer pricing, regulatory filing |
| `compliance_officer` | Regulatory alignment | SOX, SEC, FINRA requirements |

### Multi-Model Consensus: Preventing Single-Model Bias

Single-model financial analysis is structurally flawed. If one model has a systematic bias -- overweighting recent data, underestimating tail risk, or consistently optimistic on revenue growth assumptions -- every decision it touches carries that bias.

Aragora addresses this by requiring consensus across models from different providers:

- **Provider diversity**: Default financial debate teams include models from at least two different providers (e.g., Anthropic and OpenAI), ensuring architecturally distinct reasoning paths
- **Role assignment**: Agents are assigned adversarial roles (proposer, critic, synthesizer), so at least one model is structurally incentivized to find weaknesses in the analysis
- **Calibration tracking**: Each agent's historical accuracy is tracked through Aragora's ELO rating system, and vote weights are adjusted based on demonstrated performance in the financial domain
- **Hollow consensus detection**: Aragora's Trickster system detects when models are superficially agreeing without genuine independent analysis, flagging potentially unreliable consensus

---

## Regulatory Alignment

### SOX (Sarbanes-Oxley Act)

- **Section 302**: Decision Receipts document management assertions and the evidence supporting them
- **Section 404**: Pre-built control assessment workflows cover all five COSO framework components
- **Audit trail**: Immutable, timestamped logs of every decision step meet SOX record retention requirements

### SEC / FINRA

- **Model risk management**: Multi-model adversarial debate inherently addresses SR 11-7 (model risk management) requirements by ensuring no single model drives material decisions
- **Books and records**: Decision Receipts are exportable in formats compatible with SEC Rule 17a-4 retention requirements

### GAAP / IFRS

- Financial weight profiles and rubrics are calibrated to flag deviations from GAAP and IFRS standards
- Revenue recognition (ASC 606), lease accounting (ASC 842), and credit loss (CECL/ASC 326) are covered in the financial audit workflow template

### Basel III / IV (Banking)

- Risk assessment profiles support stress testing and scenario analysis debates
- Multi-model consensus aligns with regulatory expectations for model validation and challenge processes

---

## Example: Risk Assessment Debate Flow

```
1. INPUT
   Risk analyst submits: "Assess credit risk for $5M revolving facility
   to regional retail chain. 12 locations, $45M revenue, 3.2% net margins,
   seasonal cash flow volatility. Existing $8M term loan at 2.1x DSCR."

2. ROUND 1 - PROPOSAL
   Agent A (financial_auditor): "Moderate risk. DSCR of 2.1x provides
   adequate coverage. Revenue base of $45M supports the facility size.
   Recommend approval with standard covenants: minimum 1.5x DSCR,
   maximum 3.5x leverage, quarterly financial reporting."

3. ROUND 2 - CRITIQUE
   Agent B (forensic_accountant): "Disagree on risk level. Retail sector
   faces structural headwinds. 3.2% net margins are thin -- a 10% revenue
   decline (plausible in recession) would compress margins to break-even.
   Combined $13M in debt obligations on $45M revenue = 0.29x debt/revenue.
   Seasonal cash flow volatility means covenant compliance is not guaranteed
   in Q1 and Q3 (historically weak quarters). Risk is ELEVATED."

4. ROUND 3 - COUNTER
   Agent C (internal_auditor): "Agent B raises valid concerns. Adding:
   the revolving facility creates liquidity risk if drawn during seasonal
   troughs and the term loan covenant is tested simultaneously. Recommend
   conditional approval: reduce facility to $3.5M, add borrowing base
   tied to inventory + receivables, monthly (not quarterly) reporting
   during seasonal troughs, cross-default provision with term loan."

5. ROUND 4 - SYNTHESIS
   Agent A revises: "Accept elevated risk designation. Concur with
   $3.5M reduced facility and borrowing base structure. This balances
   the borrower's working capital needs with appropriate risk controls."

6. ROUND 5 - FINAL VOTE
   Consensus: CONDITIONAL APPROVAL (82% confidence)
   - Reduce facility from $5M to $3.5M
   - Add borrowing base (inventory + receivables)
   - Monthly reporting during Q1 and Q3
   - Minimum DSCR covenant: 1.5x tested quarterly
   - Cross-default with existing term loan

7. DECISION RECEIPT
   - Receipt ID: FA-2026-0212-004
   - Agents consulted: 3
   - Rounds completed: 5
   - Dissenting views: 1 (initial risk level disagreement, resolved)
   - Evidence chain: DSCR calculation, margin sensitivity analysis,
     seasonal cash flow pattern, retail sector benchmarks
   - Integrity hash: SHA-256
```

---

## Financial Audit Workflow

Aragora includes a complete financial statement audit workflow template:

```
Preliminary Analytical Review
         |
         v
Internal Control Testing (multi-agent debate)
         |
         v
Parallel Substantive Testing
   |         |         |          |
Revenue   Expense    Asset    Liability
Testing   Testing   Testing   Testing
   |         |         |          |
   +----+----+----+----+
         |
         v
Findings Synthesis (multi-agent debate)
         |
         v
Materiality Assessment
    /        |        \
Material  Significant   No Material
Finding   Deficiency    Issues
    |         |           |
Partner   Manager      Clean
Review    Review       Opinion
    \         |         /
     +----+----+----+
          |
          v
     Generate Audit Report
          |
          v
     Archive (Knowledge Mound)
```

Each substantive testing stream (revenue, expense, asset, liability) runs in parallel for efficiency. The findings synthesis step uses multi-agent debate to assess materiality and identify patterns across streams.

---

## Decision Receipt Format

```json
{
  "receipt_id": "FA-2026-0212-004",
  "timestamp": "2026-02-12T16:45:00Z",
  "schema_version": "1.0",
  "profile": "financial_audit",
  "verdict": {
    "consensus_reached": true,
    "confidence": 0.82,
    "final_answer": "CONDITIONAL APPROVAL: Reduce facility to $3.5M..."
  },
  "audit_trail": {
    "agents_consulted": 3,
    "rounds_completed": 5,
    "votes_cast": 3,
    "dissenting_views_count": 1,
    "agent_summaries": [
      {"agent": "financial_auditor", "role": "proposer", "content_preview": "..."},
      {"agent": "forensic_accountant", "role": "critic", "content_preview": "..."},
      {"agent": "internal_auditor", "role": "synthesizer", "content_preview": "..."}
    ]
  },
  "integrity": {
    "artifact_hash": "c4a8f2e7d..."
  }
}
```

Receipts are exportable as Markdown, HTML, SARIF, or CSV for integration with GRC platforms, audit management systems, and regulatory filing tools.

---

## Getting Started

### 1. Install Aragora

```bash
pip install aragora
```

### 2. Configure API Keys

For multi-model consensus in financial analysis, configure at least two providers:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### 3. Run a Financial Risk Assessment

```bash
aragora ask "Evaluate the credit risk of extending a $10M revolving credit \
  facility to a mid-market manufacturing company. Revenue $85M, EBITDA $12M, \
  existing debt $30M, DSCR 1.8x. Customer concentration: top customer = 22% \
  of revenue. Assess risk level and recommended covenant structure." \
  --vertical accounting \
  --rounds 5 \
  --consensus unanimous
```

### 4. Run a SOX Compliance Assessment

```bash
aragora ask "Assess internal controls over financial reporting for the \
  accounts payable process. Evaluate: segregation of duties between \
  invoice approval and payment processing, three-way match controls, \
  vendor master file change management, and duplicate payment detection." \
  --vertical accounting \
  --enable-verticals \
  --rounds 5
```

### 5. Run an Investment Committee Review

```bash
aragora ask "Evaluate acquisition target for investment committee: \
  B2B SaaS platform, $15M ARR, 120% net revenue retention, but \
  -$3M free cash flow and 18-month runway. Ask: $75M (5x ARR). \
  Assess valuation, key risks, and deal structure recommendations." \
  --vertical accounting \
  --rounds 5 \
  --decision-integrity
```

The `--decision-integrity` flag produces a full decision integrity package including the Decision Receipt and an implementation plan.

### 6. Programmatic Usage

```python
from aragora import Arena, Environment, DebateProtocol

env = Environment(
    task="Assess materiality of $2.3M revenue restatement for Q3 2025",
    context="Total revenue: $180M. Restatement due to ASC 606 timing error.",
)

protocol = DebateProtocol(
    rounds=5,
    consensus="unanimous",
    weight_profile="financial_audit",
)

arena = Arena(env, agents, protocol)
result = await arena.run()

# Export receipt for audit file
from aragora.gauntlet.receipt import receipt_to_markdown
markdown = receipt_to_markdown(result.receipt)
```

### 7. Use Pre-Built Workflow Templates

```python
from aragora.workflow.engine import WorkflowEngine

engine = WorkflowEngine()

# Load the financial audit template
result = await engine.run_template(
    "template_accounting_financial_audit",
    inputs={
        "financial_statements": financial_data,
        "period": "Q4 2025",
        "audit_type": "external",
        "materiality_threshold": 500000,
    },
)
```

---

## Frequently Asked Questions

**How does multi-model consensus prevent single-model bias in financial analysis?**
When one model systematically overestimates growth rates or underweights tail risk, the adversarial debate structure forces other models to challenge those assumptions. The resulting consensus reflects a range of analytical perspectives rather than one model's biases. Aragora's calibration system also tracks each model's historical accuracy in financial domains, down-weighting models that consistently produce poor financial analysis.

**Can Decision Receipts be submitted to regulators?**
Decision Receipts are designed for regulatory examination. They include cryptographic integrity verification (SHA-256), full audit trails, evidence citations, and dissenting views. The specific format requirements vary by regulator -- Aragora exports to Markdown, HTML, SARIF, and CSV to accommodate different filing systems.

**How does Aragora handle material non-public information (MNPI)?**
Aragora processes data locally within your infrastructure. No debate content is shared with LLM providers beyond the standard API call (which is subject to provider data processing agreements). For maximum sensitivity, use locally hosted models in `--local` mode.

**Does Aragora replace financial analysts or auditors?**
No. Aragora augments human decision-makers by ensuring that AI-assisted analysis is adversarially challenged, comprehensively documented, and audit-ready. The human decision-maker remains accountable, but now has a documented, multi-perspective analysis to inform their judgment.

**What is the latency for a financial debate?**
A typical 5-round debate with 3 agents completes in 30-90 seconds depending on the complexity of the analysis and the LLM providers used. Parallel substantive testing workflows take longer but can run concurrently.
