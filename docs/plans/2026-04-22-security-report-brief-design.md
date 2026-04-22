# SecurityReportBrief Design

Last updated: 2026-04-22
Status: design draft — implementation waits for Mode 3 PDB dogfood to validate the underlying pattern.

Extends: [docs/plans/2026-04-19-pr-intelligence-brief.md](2026-04-19-pr-intelligence-brief.md), [docs/plans/2026-04-20-pdb-brief-generation-mode3-design.md](2026-04-20-pdb-brief-generation-mode3-design.md).

## Problem

Security researchers increasingly use AI to generate vulnerability reports on open-source projects. Daniel Stenberg (curl maintainer, 2026-04-06): *"Over the last few months, we have stopped getting AI slop security reports. They're gone. Instead, we get an ever-increasing amount of really good security reports, almost all done with the help of AI... they're being submitted faster than ever before and are imposing a growing workload on maintainers."*

Three-phase arc observed:
1. **AI slop** (2023-2024) — low-quality reports, easy to dismiss
2. **AI plausible slop** (2025) — looked right, wasted maintainer time
3. **AI high-quality reports** (2026+) — genuine signal, but too much of it to verify at human pace

Phase 3 is a verification bottleneck, not a discovery bottleneck. The constraint is: "how fast can a human settle a claim as real/false/out-of-scope?"

## Thesis

Security-report triage is structurally identical to PR review: a claim arrives from a lower-trust source, humans must settle it (approve the fix / accept the CVE / reject as dup), and adversarial cross-checking among models converts "plausible-looking text" into "signed brief that reduces the settlement decision to low-effort sign-off."

Aragora already ships the Mode 3 PDB pipeline for PR review. The SecurityReportBrief is a **sibling artifact** of `ReviewBrief` — same panel, different prompts, different verdict space, different input normalization.

## Non-goals

- **Not a vulnerability scanner.** ZeroPath / CodeQL / Semgrep / OSS-Fuzz find issues. SecurityReportBrief adjudicates claims (regardless of who produced them — human, LLM, scanner).
- **Not a triage auto-closer.** The brief informs a human settlement. Closing issues without a maintainer's sign-off is out of scope.
- **Not a CVE authority.** We produce an opinion with dissent preserved. Authoritative classification stays with the CVE numbering authority / maintainer.
- **Not a PoC executor.** We reason about the PoC as structured input; we don't run it. (Sandboxed execution is a future variant.)

## Architecture reuse vs fork

| Mode 3 PDB primitive | Reuse in SecurityReportBrief |
|---|---|
| `aragora/pdb/storage.py` state machine | **Reuse** — same `absent/queued/running/ready/failed/stale` lifecycle; different path prefix `.aragora/security-report/briefs/` |
| `aragora/pdb/protocol.py::run_protocol_b` executor | **Reuse** — findings → critique → synthesis is identical shape |
| `aragora/pdb/budget.py` | **Reuse** — same budget reservation/release; different per-brief default cap (security reports tend to be shorter than full PRs) |
| `aragora/pdb/panel_config.py` + `pdb_panel.yaml` | **Fork** — distinct panel roster tuned for security reasoning; keep same schema |
| `aragora/pdb/prompts.py` | **Fork** — new prompt set for findings/critique/synthesis |
| `aragora/pdb/input_loader.py` | **Fork** — `load_security_report_input` from GitHub security advisory / HackerOne / custom intake instead of PR packet |
| `aragora/pdb/worker.py` | **Reuse with different work-type** — same bounded concurrency, dedupe key becomes `(repo, advisory_id, commit_or_sha)` |
| `aragora/server/handlers/review_queue_brief.py` endpoints | **Sibling module** — `aragora/server/handlers/security_report_brief.py` mirrors the pattern |
| UI (`BriefPanel` + `ApproveDecisionModal`) | **Sibling components** — `SecurityReportPanel` + `ReportDecisionModal`. Different verdict classes surfaced. |

**Key insight:** the entire Mode 3 pipeline is extractable. Mode 3 is the reference instance; future variants (security report, compliance review, license audit) are configurations of the same primitives. This is worth refactoring explicitly in a Phase-B pass: pull `absent|queued|running|ready|failed|stale`, the executor, the worker, and the budget into `aragora/brief_engine/` with Mode 3 as its first consumer.

## Input schema: `SecurityReportInput`

```python
@dataclass(frozen=True, slots=True)
class SecurityReportInput:
    # Claim identity
    advisory_id: str              # GHSA-xxxx / HackerOne report ID / internal ID
    source: str                   # "github-advisory" | "hackerone" | "custom"
    reporter: str                 # username or anonymous
    submitted_at: datetime
    
    # Repository context
    repo: str                     # "synaptent/aragora"
    affected_refs: tuple[str, ...]  # e.g., ("main", "v2.7.0", "v2.8.0")
    affected_paths: tuple[str, ...]  # file paths claimed to be affected
    
    # The claim itself
    title: str
    summary: str                  # "Buffer overflow in X when Y"
    severity_claimed: str         # reporter's self-assessment: "critical/high/medium/low/info"
    cwe_claimed: tuple[str, ...]  # e.g., ("CWE-121", "CWE-787") — optional
    cve_proposed: str | None      # if reporter suggested one
    
    # The PoC
    poc_description: str
    poc_code: str | None          # sanitized — strip secrets, tokens
    poc_environment: dict[str, str]  # platform/version needed to reproduce
    
    # Supporting evidence
    stack_traces: tuple[str, ...]
    references: tuple[str, ...]   # links to docs/RFCs/related CVEs
    
    # Brief budget + panel
    policy: ReviewPolicy
    panel_id: str                 # default: "security_report_default"
```

**Invariants:**
- `poc_code` is pre-sanitized on the caller side — never send raw secrets/tokens to the panel.
- `affected_paths` are file paths that exist in `affected_refs` at the time of intake (loader validates).
- `severity_claimed` is for context only; the panel's synthesis produces its own severity in the verdict.

## Output schema: `SecurityReportBrief`

```python
@dataclass(frozen=True, slots=True)
class SecurityReportBrief:
    # Identity
    advisory_id: str
    repo: str
    head_sha: str
    packet_sha: str
    generated_at: datetime
    
    # Verdict
    verdict: SecurityReportVerdict  # see below
    top_line: str                   # one-sentence summary
    severity_assessed: str          # "critical/high/medium/low/info/none"
    cwe_assessed: tuple[str, ...]
    
    # Role-structured findings (mirror ReviewBrief)
    reproducibility: str   # "likely reproducible" / "cannot reproduce from given PoC" / ...
    exploitability: str    # "pre-auth RCE" / "requires local access" / "DoS only" / ...
    compliance: str        # "violates RFC 5321 §4.1.2" / "within spec" / "no applicable RFC" / ...
    fix_soundness: str     # if a fix was proposed: "fix is correct / incomplete / wrong direction"
    skeptic: str           # devil's-advocate reading — "what makes this NOT a vuln?"
    
    # Dissent preservation (identical to ReviewBrief)
    role_findings: tuple[RoleFinding, ...]
    dissent: tuple[DissentingView, ...]
    
    # Metadata
    overall_confidence: int          # 1-5
    disagreement_score: float        # 0.0-1.0
    total_cost_usd: float
    total_wall_clock_ms: int
    agent_roster: tuple[str, ...]
```

## Verdict space

Four primary verdicts, mirror the `Recommendation` enum shape from `aragora/review/protocol.py`:

| Verdict | Meaning | Suggested next action |
|---|---|---|
| `confirmed_bug` | All core slots agree: reproducible, exploitable (even if only DoS), outside current design intent | Assign severity; produce fix; coordinate CVE disclosure |
| `likely_bug_needs_repro` | Reasoning suggests real; PoC insufficient to verify | Ask reporter for minimal reproducer; hold pending |
| `false_positive` | Panel finds the claim doesn't hold — either misreading, pattern-match without context, or already-fixed | Close with explanation; do not shame reporter |
| `out_of_scope` | Real issue but outside project's security model (e.g., physical access, compromised system) | Document in security policy; close |

Plus one fallback: `needs_human_attention` — when confidence is low or dissent is high, return this regardless of which direction the majority leaned. Human decides.

## Panel roster (initial)

Mirror `pdb_panel.yaml` structure:

```yaml
default_panel: security_report_default
slots:
  claude_security:     # core
    review_role: exploit_reasoner
    lens: core
    family: claude
    candidates: [claude, anthropic-api]
    required: true
  gpt_security:        # core
    review_role: rfc_compliance_reviewer
    lens: core
    family: gpt
    candidates: [codex, openai-api, openai]
    required: true
  gemini_skeptic:      # heterodox
    review_role: skeptic
    lens: heterodox
    family: gemini
    candidates: [gemini-cli, gemini]
    required: false
  grok_reproducibility:  # heterodox
    review_role: reproducibility_analyst
    lens: heterodox
    family: grok
    candidates: [grok-cli, grok]
    required: false
  mistral_regulatory:  # regulatory
    review_role: compliance_advisor  # GDPR / export controls / dual-use concerns
    lens: regulatory
    family: mistral
    candidates: [mistral-api, mistral]
    required: false
panels:
  security_report_default:
    findings_slots: [claude_security, gpt_security, gemini_skeptic, grok_reproducibility, mistral_regulatory]
    critique_slots: same_as_findings
    synthesizer_slot: claude_security
```

**Rationale for differences from PDB panel:**
- Only 5 slots (vs 8 for PDB PR review) — security reports have less context to reason about; more slots diminish returns.
- Skeptic role is elevated (heterodox) because false-positive rate for AI-generated security reports is historically ~50-80%.
- No `logic_reviewer` / `security_reviewer` / `maintainability_reviewer` / `skeptic` mapping from PDB — those prompts don't fit. Fresh role names: `exploit_reasoner`, `rfc_compliance_reviewer`, `reproducibility_analyst`, `compliance_advisor`.

## Prompt structure sketches

**Findings prompt** (each slot receives independently):

```
You are a {review_role} reviewing a security claim for {repo}.
The claim is:
  Title: {title}
  Severity claimed: {severity_claimed}
  Affected paths: {affected_paths}

PoC description: {poc_description}
PoC code (if provided, already sanitized):
  {poc_code}

Your task:
1. Read the relevant source code at {affected_paths} in the {head_sha} tree.
2. Answer (as structured JSON):
   - reproducibility: likely_yes / likely_no / need_more_info, with one-paragraph reason
   - exploitability: pre_auth_critical / post_auth_high / dos_only / local_only / none / unclear, with reason
   - rfc_or_spec_violation: name the RFC/spec and clause, or "none_applicable"
   - out_of_scope: yes / no, with reason if yes
   - confidence: 1-5
3. Be honest. If the PoC is insufficient, say so.
```

**Critique prompt** (each slot receives peer findings):

```
You are a {review_role}. You already produced findings for the security claim above. 
Your peers produced these findings:
  {peer_findings_structured}

1. For each peer finding, note if you agree / disagree / need more evidence.
2. Identify contested points (e.g., peer says "DoS only", you assessed "pre-auth RCE"). 
3. Preserve your dissent — do not collapse to the majority just to agree.
```

**Synthesis prompt** (synthesizer slot):

```
You are producing a final SecurityReportBrief from {N} panel findings + critiques.

Required output (JSON):
- verdict: confirmed_bug / likely_bug_needs_repro / false_positive / out_of_scope / needs_human_attention
- top_line: one sentence
- severity_assessed: critical / high / medium / low / info / none
- cwe_assessed: list of CWE IDs
- reproducibility, exploitability, compliance, fix_soundness, skeptic: prose sections, each <=150 words
- dissent: list of per-slot disagreements that didn't collapse

CRITICAL: preserve dissent. If even one slot (especially the skeptic) disagrees
with the majority verdict, emit needs_human_attention OR list the disagreement 
in the dissent array. Never flatten to a single view.
```

## Budget considerations

- Per-brief default cap: **$4.00** (half of PDB PR review; less context per claim).
- Per-day cap: configurable per project; default **$100.00**.
- Degrade rules: identical to PDB — optional slots may drop if budget tight; core slots are non-negotiable.
- Fail-closed: identical — if both core slots plus synthesis can't be funded, return `budget_exceeded`.

## Differentiation vs ZeroPath / existing scanners

| Dimension | Scanners (ZeroPath, CodeQL, Semgrep, OSS-Fuzz) | SecurityReportBrief |
|---|---|---|
| Role | **Discover** vulnerabilities in static/dynamic analysis | **Adjudicate** already-claimed vulnerabilities |
| Input | Source code | A claim + source code + PoC |
| Output | Finding (potential issue) | Verdict with preserved dissent (is the finding real?) |
| Consumer | AppSec engineer / dev | Maintainer triaging their inbox |
| False-positive handling | Each scanner has its own suppression | Panel critique is specifically designed to surface FPs |
| Audit trail | Report output | Cryptographic receipt + role-structured reasoning |

**The two are complementary.** A workflow might look like: scanner finds 170 issues in curl (ZeroPath's case) → researcher files 20 of them as reports → maintainer runs each report through SecurityReportBrief → brief reduces triage from "re-read all the code" to "sign off or push back." As scanners improve, their output volume grows; SecurityReportBrief is the load-bearing piece that lets the human-in-the-loop scale.

## Rollout plan (post-Mode 3 dogfood)

### Phase 0 (now)
- Land design doc (this file).
- Wait for Mode 3 dogfood findings to inform prompt quality + panel config defaults.

### Phase 1 — `aragora/brief_engine/` extraction
- Lift Mode 3's storage/executor/worker/budget into `aragora/brief_engine/`.
- Make Mode 3 PDB its first consumer (no behavior change; refactor only).
- Acceptance: Mode 3 tests all pass unchanged.

### Phase 2 — SecurityReportBrief minimum-viable
- Fork the PDB panel config + prompts into `aragora/security_report/`.
- `SecurityReportInput` + `SecurityReportBrief` dataclasses.
- Executor uses `brief_engine.run_protocol_b` with new prompts.
- One end-to-end test with mocked panels produces a brief from a canned input.

### Phase 3 — input loaders
- GitHub security advisory loader (reads from the Advisory JSON API).
- Manual intake CLI: `aragora security-report add --repo X --poc-file Y.md`.
- No HackerOne / Bugcrowd integration yet; manual intake covers 90% of OSS cases.

### Phase 4 — endpoints + UI
- Handler: `aragora/server/handlers/security_report_brief.py` mirroring `review_queue_brief.py`.
- UI: `aragora/live/src/app/(app)/security-reports/page.tsx` with a mirror of the review queue.

### Phase 5 — dogfood
- Pick 10 real public CVE-adjacent security reports (e.g., from curl's public tracker, or historical Aragora advisories if any).
- Run each through SecurityReportBrief.
- Measure: verdict accuracy vs. ground truth, false-positive rate, maintainer time saved.

## Open questions

- **PoC execution:** do we ever run PoC code in a sandbox? Answer: not in this design. Future variant, separate ADR.
- **Cross-project panels:** should a brief for a curl report use curl-expertise-tuned prompts, or generic ones? Proposal: start generic; measure; specialize if accuracy is a clear win.
- **Severity scoring:** do we produce CVSS v3 vectors, or just qualitative severity? Proposal: qualitative first (critical/high/medium/low); CVSS is a follow-up — too many models hallucinate scores.
- **Report chaining:** if a report is flagged `likely_bug_needs_repro` and the reporter submits a better PoC, do we re-run the full panel or just the reproducibility slot? Proposal: full re-run; it's cheap vs. the wall-clock savings of single-slot updates.
- **Privacy:** vulnerability details are embargoed before public disclosure. Does Aragora's brief-generation leak anything to OpenRouter / third-party hosted models? Answer: must use only first-party-API models (Anthropic/OpenAI direct) for embargoed reports; document clearly.

## Success criteria for this design

- Clear enough that an engineer could scaffold Phase 2 in a weekend.
- Honest about what's reusable vs what must fork (not a lazy "just add another pipeline").
- Non-goals section prevents scope creep (no auto-close, no PoC execution, no CVE authority).
- Explicit dependency: **do not implement until Mode 3 dogfood findings are incorporated.** The PDB's lessons on prompt quality, dissent preservation, and synthesis accuracy matter here.

## What this unlocks

Assuming Mode 3 proves the pattern works: SecurityReportBrief extends the same adversarial-verification wedge from "should we merge this PR?" to "is this vulnerability claim real?" Both are verification-bottleneck problems.

If both work, the primitives generalize to:
- Compliance-review triage (is this config compliant with SOC 2?)
- License-audit claims (is this dependency's license compatible?)
- Governance decisions (should we adopt this RFC?)

Each is a distinct brief variant with its own panel + prompts but the same engine. The Mode 3 PDB pipeline is the reference; SecurityReportBrief is the second instance; the `brief_engine/` refactor (Phase 1 above) is what makes the 3rd, 4th, 5th variants tractable.
