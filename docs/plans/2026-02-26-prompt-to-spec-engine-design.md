# Prompt-to-Spec Engine Design

> **Date**: 2026-02-26
> **Status**: Approved for implementation
> **Approach**: Orchestration layer over existing components (Approach A)

## Vision

Take a vague user prompt ("make onboarding better"), decompose it, ask clarifying questions, research context from multiple sources, generate a professional specification, validate it through adversarial debate, and hand off to execution — all through a stateful API that can be rendered as a web Canvas, CLI, or Obsidian plugin.

## Architecture

### 7-Stage State Machine

```
INTAKE -> DECOMPOSE -> INTERROGATE -> RESEARCH -> SPEC -> VALIDATE -> HANDOFF
```

- **INTAKE**: Raw prompt string -> EnrichedBrainDump (via BrainDumpParser)
- **DECOMPOSE**: EnrichedBrainDump -> PromptIntent (intent type, domains, ambiguities, scope estimate)
- **INTERROGATE**: PromptIntent -> list[ClarifyingQuestion] -> user answers -> RefinedIntent
- **RESEARCH**: RefinedIntent -> ResearchReport (KM precedents, codebase scan, Obsidian notes, web results)
- **SPEC**: RefinedIntent + ResearchReport -> SwarmSpec (enhanced with research evidence)
- **VALIDATE**: SwarmSpec -> Arena debate (if confidence < threshold) -> ValidatedSpec
- **HANDOFF**: ValidatedSpec -> PrioritizedGoals -> IdeaToExecutionPipeline

### API Surface

- `POST /api/v1/prompt-engine/sessions` — Create session (returns session_id)
- `GET /api/v1/prompt-engine/sessions/{id}` — Get session state
- `DELETE /api/v1/prompt-engine/sessions/{id}` — Cancel session
- `WS /ws/prompt-engine/{session_id}` — Real-time interaction

### WebSocket Protocol

Client -> Server:
```json
{"type": "answer", "question_id": "q1", "answer": "option_a"}
{"type": "approve_spec"}
{"type": "skip_validation"}
{"type": "override_sources", "sources": ["km", "codebase", "web"]}
```

Server -> Client:
```json
{"type": "stage_transition", "from": "intake", "to": "decompose", "state": {}}
{"type": "question", "question": {"id": "q1", "question": "...", "options": []}}
{"type": "research_progress", "source": "km", "status": "complete", "results": 3}
{"type": "spec_ready", "spec": {}, "confidence": 0.85}
{"type": "validation_started", "debate_id": "..."}
{"type": "validation_result", "verdict": "approved", "confidence": 0.91}
{"type": "handoff_complete", "pipeline_id": "..."}
{"type": "error", "stage": "research", "message": "..."}
```

## Data Model

### Core Types (aragora/prompt_engine/types.py)

```python
class EngineStage(str, Enum):
    INTAKE = "intake"
    DECOMPOSE = "decompose"
    INTERROGATE = "interrogate"
    RESEARCH = "research"
    SPEC = "spec"
    VALIDATE = "validate"
    HANDOFF = "handoff"

class UserProfile(str, Enum):
    FOUNDER = "founder"      # Quick interrogation, auto-execute at 0.8, show code
    CTO = "cto"              # Thorough, auto-execute at 0.9, show code
    BUSINESS = "business"    # Thorough, auto-execute at 0.95, hide code
    TEAM = "team"            # Exhaustive, always require approval, show code

class ResearchSource(str, Enum):
    KNOWLEDGE_MOUND = "km"
    CODEBASE = "codebase"
    OBSIDIAN = "obsidian"
    WEB = "web"

@dataclass
class PromptIntent:
    raw_prompt: str
    intent_type: str           # feature | improvement | investigation | fix | strategic
    domains: list[str]
    ambiguities: list[str]
    assumptions: list[str]
    scope_estimate: str        # small | medium | large | epic
    enriched_dump: EnrichedBrainDump | None

@dataclass
class ClarifyingQuestion:
    id: str
    question: str
    why_it_matters: str
    options: list[QuestionOption]
    default_option: str | None
    impact: str                # high | medium | low

@dataclass
class QuestionOption:
    label: str
    description: str
    tradeoff: str

@dataclass
class RefinedIntent:
    intent: PromptIntent
    answers: dict[str, str]
    confidence: float          # 0.0-1.0

@dataclass
class ResearchReport:
    km_precedents: list[dict]
    codebase_context: list[dict]
    obsidian_notes: list[dict]
    web_results: list[dict]
    sources_used: list[ResearchSource]

@dataclass
class SessionState:
    session_id: str
    stage: EngineStage
    profile: UserProfile
    research_sources: list[ResearchSource]
    raw_prompt: str
    intent: PromptIntent | None
    questions: list[ClarifyingQuestion]
    answers: dict[str, str]
    refined_intent: RefinedIntent | None
    research: ResearchReport | None
    spec: SwarmSpec | None
    validation_result: DebateResult | None
    pipeline_id: str | None
    created_at: datetime
    updated_at: datetime
    provenance_hash: str
```

### User Profile Defaults

```python
PROFILE_DEFAULTS = {
    "founder": {
        "interrogation_depth": "quick",        # 3-5 questions
        "auto_execute_threshold": 0.8,
        "require_approval": False,
        "show_code": True,
        "autonomy_level": "propose_and_approve",
    },
    "cto": {
        "interrogation_depth": "thorough",     # 8-12 questions
        "auto_execute_threshold": 0.9,
        "require_approval": True,
        "show_code": True,
        "autonomy_level": "propose_and_approve",
    },
    "business": {
        "interrogation_depth": "thorough",     # 8-12 questions
        "auto_execute_threshold": 0.95,
        "require_approval": True,
        "show_code": False,
        "autonomy_level": "human_guided",
    },
    "team": {
        "interrogation_depth": "exhaustive",   # 15-20 questions
        "auto_execute_threshold": 1.0,         # Always require approval
        "require_approval": True,
        "show_code": True,
        "autonomy_level": "metrics_driven",
    },
}
```

## Components

### Backend (5 new files)

**aragora/prompt_engine/engine.py** — `PromptToSpecEngine`
- Stateful orchestrator driving sessions through 7 stages
- Each method returns updated `SessionState` (testable without WebSocket)
- Methods: `start_session()`, `answer_question()`, `run_research()`, `generate_spec()`, `validate_spec()`, `handoff()`
- Private methods: `_decompose()` (LLM call for intent classification), `_generate_questions()` (LLM call constrained by profile depth)

**aragora/prompt_engine/researcher.py** — `MultiSourceResearcher`
- Fans out research to enabled sources via `asyncio.gather()`
- Sources: KM (via PipelineKMBridge), Codebase (via assessment engine), Obsidian (via ObsidianAdapter), Web (via web search)
- Each source fails independently (graceful degradation)
- Emits `research_progress` events via SpectatorStream

**aragora/prompt_engine/spec_builder.py** — `SpecBuilder`
- Takes RefinedIntent + ResearchReport -> SwarmSpec
- Single LLM call with structured prompt including intent, evidence, profile constraints, codebase conventions
- Output is existing `SwarmSpec` dataclass (no new types)

**aragora/prompt_engine/types.py** — All dataclasses and enums above

**aragora/prompt_engine/__init__.py** — Public exports

### API Handler

**aragora/server/handlers/prompt_engine/handler.py**
- REST: session CRUD endpoints
- WebSocket: thin dispatcher calling engine methods, streaming state transitions

### Frontend (1 page + 7 components + 1 hook)

**aragora/live/src/app/(app)/prompt/page.tsx** — Main page

**Components (aragora/live/src/components/prompt-engine/):**
- `PromptInput.tsx` — Large textarea with real-time intent preview
- `QuestionCard.tsx` — Single clarifying question with option buttons
- `RefinementFlow.tsx` — Sequential question flow with progress indicator
- `ResearchGrid.tsx` — Source cards with progress spinners and result counts
- `SpecEditor.tsx` — Structured spec document, inline editable sections
- `StageNav.tsx` — Left sidebar stage navigation with status indicators
- `ProvenanceBar.tsx` — Bottom bar showing provenance chain

**Hook:**
- `usePromptEngine.ts` — WebSocket connection, session state management

**Reused components:** LiveDebateStream (validate stage), PipelineCanvas (execute stage), Scanlines/CRTVignette (visual consistency), useRightSidebar (context panel)

## Confidence-Gated Validation

Specs are auto-assessed for complexity. The confidence gate determines whether to run adversarial debate:

- `spec.estimated_complexity == "high"` OR `refined_intent.confidence < profile.auto_execute_threshold` -> run 5-agent debate (Devil's Advocate, Scope Creep Detector, Security Reviewer, UX Advocate, Tech Debt Auditor)
- Otherwise -> skip debate, proceed directly to handoff

Users can always manually trigger or skip validation regardless of the gate.

## Error Handling

- **Stage-level recovery**: State preserved at last successful stage. User can retry or skip.
- **Source-level degradation**: Research continues if individual sources fail.
- **LLM fallback**: AirlockProxy -> OpenRouter fallback on quota/timeout.
- **Session timeout**: 24-hour inactivity expiry. State persisted for resume.
- **Provenance integrity**: SHA-256 hash chain on every state transition.

## Testing Strategy

### Backend (tests/prompt_engine/)
- `test_engine.py` — State machine transitions, session lifecycle (mock LLM)
- `test_decomposer.py` — Intent classification from various prompt styles (parameterized)
- `test_interrogator_api.py` — Question generation, answer processing
- `test_researcher.py` — Multi-source fan-out, graceful degradation, parallel execution
- `test_spec_builder.py` — Spec assembly, SwarmSpec field population
- `test_profiles.py` — Profile defaults applied correctly per persona
- `test_confidence_gate.py` — Debate skip/trigger logic
- `test_session_persistence.py` — Serialize/restore/continue
- `test_integration.py` — End-to-end: raw prompt -> pipeline_id (mock LLM)

### Frontend (aragora/live/__tests__/prompt-engine/)
- `PromptInput.test.tsx` — Real-time intent preview rendering
- `QuestionCard.test.tsx` — Option selection, answer submission
- `RefinementFlow.test.tsx` — Sequential question progression
- `usePromptEngine.test.ts` — WebSocket message handling, state updates

## Existing Components Reused

| Component | Used In |
|-----------|---------|
| BrainDumpParser | INTAKE stage |
| SwarmInterrogator | INTERROGATE stage (enhanced for API mode) |
| PipelineKMBridge | RESEARCH stage (KM source) |
| ObsidianAdapter | RESEARCH stage (Obsidian source) |
| Arena | VALIDATE stage (5-agent spec debate) |
| IdeaToExecutionPipeline | HANDOFF stage |
| SpectatorStream | All stages (real-time events) |
| SwarmSpec | SPEC stage output format |
| AirlockProxy | LLM fallback handling |

## Estimated Scope

- Backend: ~400-500 lines across 5 files
- Frontend: ~600-800 lines across 9 files
- Tests: ~800-1000 lines across 13 files
- Total: ~1800-2300 lines
