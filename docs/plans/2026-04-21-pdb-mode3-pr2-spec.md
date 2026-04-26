# PDB Mode 3 PR2 Spec
Last updated: 2026-04-21
Status: implementation-ready docs slice for `#6306` PR2

Extends:
- [docs/plans/2026-04-19-pr-intelligence-brief.md](2026-04-19-pr-intelligence-brief.md)
- [docs/plans/2026-04-19-pr-intelligence-brief-addendum.md](2026-04-19-pr-intelligence-brief-addendum.md)
- [docs/plans/2026-04-20-pr-review-execution-path.md](2026-04-20-pr-review-execution-path.md)

Grounded in landed foundations:
- `#6355` packet scaffold in `aragora/swarm/pr_review_protocol.py`
- `#6353` receipt schema in `aragora/review/receipt.py`
- `#6359` review policy / budget schema in `aragora/review/policy.py`
- `#6368` provider slot resolver in `aragora/review/provider_slots.py`

`#6369` is still PR1. This spec exists so PR2 can start the moment PR1 merges.

## Purpose
Define the exact boundary for the first real Mode 3 execution slice:
- YAML-driven panel config
- validated panel loading
- per-brief and per-day budget enforcement
- Protocol B execution: findings round -> critique round -> one synthesis pass
- prompt templates
- one stable executor input for later PR3 callers

## Boundary
PR2 owns exactly these files:

| File | Responsibility |
| --- | --- |
| `aragora/config/pdb_panel.yaml` | default panel roster + budget caps |
| `aragora/pdb/panel_config.py` | load + validate YAML into typed config |
| `aragora/pdb/budget.py` | budget reservation and breach handling |
| `aragora/pdb/protocol.py` | Protocol B executor only |
| `aragora/pdb/prompts.py` | prompt templates for findings/critique/synthesis |

PR2 may add tests for those modules.

PR2 is explicitly out of scope for:
- HTTP routes
- worker queues
- UI
- review-queue build integration
- webhook or batch automation
- Protocol C
- dual synthesis
- storage/state-machine work already owned by `#6369`

## Guardrails
1. No silent downgrade to `metadata_heuristic`.
2. Dissent must survive as structured data, not only one summary string.
3. PR2 must build on `ProviderSlotDefinition` and `ProviderSlotResolver`, not fork them.
4. PR2 defines transport-neutral execution contracts only; PR3 wires callers.

## `aragora/config/pdb_panel.yaml`
Introduce one committed default config file. It is configuration, not a secret store.

Required shape:

```yaml
version: 1
default_panel: protocol_b_default
default_prompt_set: protocol_b_v1
budgets:
  per_brief_usd: 8.0
  per_day_usd: 200.0
  reserve_for_manual_escalation_usd: 40.0
slots:
  claude_core:
    review_role: logic_reviewer
    lens: core
    family: claude
    candidates: [claude, anthropic-api]
    required: true
  gpt_core:
    review_role: security_reviewer
    lens: core
    family: gpt
    candidates: [codex, openai-api, openai]
    required: true
  gemini_heterodox:
    review_role: maintainability_reviewer
    lens: heterodox
    family: gemini
    candidates: [gemini-cli, gemini]
    required: false
  grok_heterodox:
    review_role: skeptic
    lens: heterodox
    family: grok
    candidates: [grok-cli, grok]
    required: false
  deepseek_heterodox:
    review_role: skeptic
    lens: heterodox
    family: deepseek
    candidates: [deepseek-cli, deepseek]
    required: false
  kimi_heterodox:
    review_role: skeptic
    lens: heterodox
    family: kimi
    candidates: [kimi]
    required: false
  qwen_heterodox:
    review_role: skeptic
    lens: heterodox
    family: qwen
    candidates: [qwen-cli, qwen]
    required: false
  mistral_regulatory:
    review_role: skeptic
    lens: regulatory
    family: mistral
    candidates: [mistral-api, mistral]
    required: false
panels:
  protocol_b_default:
    findings_slots: [claude_core, gpt_core, gemini_heterodox, grok_heterodox, deepseek_heterodox, kimi_heterodox, qwen_heterodox, mistral_regulatory]
    critique_slots: same_as_findings
    synthesizer_slot: claude_core
prompt_sets:
  protocol_b_v1:
    findings_prompt: protocol_b_findings
    critique_prompt: protocol_b_critique
    synthesis_prompt: protocol_b_synthesis
```

Validation rules:
- `version == 1`
- `default_panel` exists in `panels`
- `synthesizer_slot` exists in `slots`
- every slot defines `review_role`, `lens`, `family`, `candidates`
- `findings_slots` includes both core slots
- selected panel includes at least one non-core lens
- `same_as_findings` is allowed only for `critique_slots`
- `required: true` means hard fail if unresolved
- `required: false` means degrade only if minimum safe roster still holds

## `aragora/pdb/panel_config.py`
Owns YAML loading and validation.

Required dataclasses:
- `PDBBudgetConfig`
- `PDBPanelSlot`
- `PDBPanelDefinition`
- `PDBPromptSet`
- `PDBPanelConfig`

Required functions:
- `load_panel_config(path: Path | None = None) -> PDBPanelConfig`
- `validate_panel_config(raw: Mapping[str, Any]) -> PDBPanelConfig`
- `provider_slot_definitions(config: PDBPanelConfig, panel_id: str) -> tuple[ProviderSlotDefinition, ...]`

Required behavior:
- load `aragora/config/pdb_panel.yaml`
- validate slot references with exact field-path errors
- project slot records into landed `ProviderSlotDefinition`
- preserve per-slot `required` metadata beside the generic resolver output

## `aragora/pdb/budget.py`
Owns Protocol B budget reservation and denial logic on top of `ReviewPolicy`.

Required types:
- `PDBBudgetDecision`
- `PDBBudgetReservation`
- `PDBBudgetStatus`

Required statuses:
- `allowed`
- `budget_exceeded`
- `budget_degraded`

Required behavior:
- estimate full configured-panel spend before execution
- reserve spend before findings round
- release unused reserve after synthesis
- enforce per-brief and per-day caps
- record the active roster budget actually funded

Budget rules:
- if both core slots and one synthesis pass fit in budget, execution may run
- if optional non-core slots do not fit, execution may degrade
- if both core slots plus synthesis do not fit, return `budget_exceeded`
- PR2 must not emit a fake real-review packet from a hidden cheaper roster

At breach:
- `PRReviewProtocolPacket.status` must not remain `metadata_heuristic`
- executor returns explicit denial or degraded status
- `recommendation_class` defaults to `needs_human_attention`
- caller may still show a heuristic packet only if it remains labeled heuristic
  and is paired with the denial result

## `aragora/pdb/prompts.py`
Owns prompt templates only.

Required prompt families:
- findings round
- critique round
- synthesis

Prompt requirements:
- bind `repo`, `pr_number`, `base_sha`, `head_sha`
- request structured findings, not essays
- preserve `core`, `heterodox`, and `regulatory` lens identity
- frame the regulatory lens as a European/regulatory perspective, not a compliance oracle
- instruct synthesis to preserve disagreement instead of flattening it

## `aragora/pdb/protocol.py`
Owns the bounded Protocol B executor only.

Required entry point:
- `run_protocol_b(input: PDBExecutionInput) -> PDBExecutionResult`

Required stages:
1. load and validate panel config
2. resolve configured slots through `ProviderSlotResolver`
3. evaluate and reserve budget
4. run findings round across the active roster
5. run critique round with peer findings context
6. run one synthesis pass through the configured synthesizer slot
7. return packet-facing and brief-facing structured output

PR2 must not:
- write storage state
- enqueue jobs
- call HTTP routes
- run Protocol C
- run secondary synthesis

## Normalized executor input
PR2 must define one transport-neutral payload:

```python
@dataclass(frozen=True, slots=True)
class PDBExecutionInput:
    binding: PRReviewBinding
    packet: PRReviewProtocolPacket
    packet_sha: str
    pr_title: str
    pr_body: str
    labels: tuple[str, ...]
    changed_files: tuple[str, ...]
    diff_excerpt: str
    validation_summary: Mapping[str, Any]
    panel_id: str
    policy: ReviewPolicy
```

Rules:
- no file handles, request objects, CLI parser objects, or git processes
- `diff_excerpt` is prepared text, not a live repository handle
- `packet.status` may start as `metadata_heuristic`; it changes only if a real execution attempt occurs
- PR3 must construct this payload and nothing richer

## Dissent preservation
PR2 must preserve disagreement at three layers:

1. reviewer outputs
   - one record per slot
   - recommendation
   - confidence
   - top findings
   - contested finding ids

2. `PRReviewProtocolPacket`
   - `dissent_summary` is scan-oriented only
   - `dissenting_views` is the lossless machine-readable array
   - each entry includes `slot_id`, `lens`, `recommendation`, `reason`, `contested_finding_ids`

3. brief/receipt projection
   - `ReviewBrief.dissent` contains one entry per dissenting slot
   - `BriefReceipt.brief.dissent` therefore preserves per-dissenter identity
   - supporting dissent evidence goes in `BriefReceipt.evidence_refs`

## Degrade vs fail-closed
Degrade only when:
- optional non-core slot is unavailable
- optional non-core slot is priced out by budget
- optional non-core critique response times out after findings succeeded

Fail closed when:
- any required core slot is unavailable
- synthesizer slot is unavailable
- budget cannot fund both core slots plus synthesis
- final panel would collapse below two model families

When degraded, output must name:
- missing slots
- final active roster
- degraded execution status

## Acceptance criteria
PR2 is done when:
- panel config loads from YAML and validates against the landed slot resolver
- Protocol B executes with mocked providers over `PDBExecutionInput`
- budget denial yields explicit `budget_exceeded` output without silent fallback
- degraded execution preserves reduced-roster and dissent details
- executor output is transport-neutral enough for PR3 wiring without changing these contracts

## Follow-on boundary
After this spec lands and `#6369` merges:
- PR2 may implement panel config + budget + Protocol B execution
- PR3 may later own transport and artifact wiring
- PR4 remains out of scope until PR2 lands cleanly
