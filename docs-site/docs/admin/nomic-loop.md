---
title: Nomic Loop Documentation
description: Nomic Loop Documentation
---

# Nomic Loop Documentation

The Nomic Loop is aragora's autonomous self-improvement system—a society of AI agents that debates and implements improvements to its own codebase.

Note: The nomic loop is experimental. Run it in a sandbox and review changes before auto-commit.

## Overview

The name "nomic" comes from the game of Nomic, where players modify the rules of the game as part of gameplay. Similarly, the nomic loop allows AI agents to propose, debate, and implement changes to aragora itself.

## Architecture

The nomic loop is a 6-phase cycle with two implementations:

### State Machine (Recommended)

Event-driven, robust, checkpoint-resumable. Uses `NomicStateMachine` with phase handlers.

### Legacy Integration

Phase-based, integrated with aragora features. Uses `NomicLoop` class for backward compatibility.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NOMIC LOOP                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Phase 1: CONTEXT                                                  │
│   ├── Claude (Claude Code) ──┐                                      │
│   ├── Codex (Codex CLI)      ├── Explore codebase, read docs        │
│   ├── Gemini (Kilo Code)     │                                      │
│   └── Grok (Kilo Code) ──────┘                                      │
│                                                                      │
│   Phase 2: DEBATE                                                   │
│   ├── Agents propose improvements based on context                  │
│   ├── Arena orchestrates critique rounds                            │
│   └── Agents vote on proposals (weighted by Elo rating)             │
│                                                                      │
│   Phase 3: DESIGN                                                   │
│   ├── Generate implementation design                                │
│   ├── Identify affected files                                       │
│   └── Safety review (auto-approve or human review)                  │
│                                                                      │
│   Phase 4: IMPLEMENT                                                │
│   ├── Generate code from design                                     │
│   ├── Validate syntax and dangerous patterns                        │
│   └── Write files with backup                                       │
│                                                                      │
│   Phase 5: VERIFY                                                   │
│   ├── Syntax check (python -m py_compile)                           │
│   ├── Import check (python -c "import aragora")                     │
│   └── Test suite (pytest)                                           │
│                                                                      │
│   Phase 6: COMMIT (if all checks pass)                              │
│   └── Auto-commit with detailed message                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Running the Nomic Loop

### Basic Usage

```bash
# Run with streaming dashboard
NOMIC_AUTO_COMMIT=1 python scripts/run_nomic_with_stream.py run --cycles 24 --auto

# Run without streaming
NOMIC_AUTO_COMMIT=1 python scripts/nomic_loop.py --cycles 10 --auto
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cycles N` | Number of improvement cycles | 10 |
| `--auto` | Auto-commit successful changes (requires `NOMIC_AUTO_COMMIT=1`) | False |
| `--human-approval` | Require human approval for commits | False |
| `--skip-genesis` | Skip codebase analysis | False |

### Safety Gates (Environment)

| Variable | Description | Default |
|----------|-------------|---------|
| `NOMIC_AUTO_COMMIT` | Allow `--auto` commits (must be `1` to enable) | `0` |
| `NOMIC_AUTO_CONTINUE` | Auto-continue non-interactive runs | `1` |
| `ARAGORA_ENABLE_FORKING` | Enable parallel forked debates in the nomic loop | `0` |

### Approval Workflow

The nomic loop can require approvals for sensitive changes via
`aragora/nomic/approval.py`. The default policy classifies files into
`info`, `review`, or `critical` based on path patterns and enforces votes or
timeouts before changes proceed.

### Live Dashboard

The streaming version provides real-time visibility:

- **HTTP Dashboard**: `http://localhost:8080`
- **WebSocket Stream**: `ws://localhost:8765/ws`
- **Public View**: `https://aragora.ai`

## Cycle Phases

### Phase 1: Context

All agents explore the codebase using their respective CLI tools:

- **Claude**: Uses Claude Code for deep analysis
- **Codex**: Uses OpenAI Codex CLI
- **Gemini**: Uses Google's Kilo Code
- **Grok**: Uses xAI's Grok CLI (via Kilo Code)

Each agent reads key files and identifies potential improvements. Implemented by `ContextPhase` class.

### Phase 2: Debate

The Arena orchestrates a structured debate cycle:

1. **Proposal**: Agents submit structured proposals
2. **Critique**: Each agent critiques other proposals (severity scores 0-1)
3. **Voting**: Agents vote on proposals (weighted by Elo rating)
4. **Consensus**: Check if threshold is reached

```json
{
  "title": "Add semantic caching for debate context",
  "description": "Implement vector-based caching to reduce redundant LLM calls",
  "impact": "HIGH",
  "complexity": "MEDIUM",
  "files_affected": ["aragora/debate/orchestrator.py", "aragora/memory/cache.py"]
}
```

Implemented by `DebatePhase` class with `consensus_threshold` configuration.

### Phase 3: Design

Generate implementation design from the winning proposal:

1. **Generate Design**: Create detailed implementation plan
2. **Identify Files**: Determine affected files
3. **Safety Review**: Check for protected files and dangerous patterns
4. **Approval**: Auto-approve low-risk designs or flag for human review

Implemented by `DesignPhase` class with `auto_approve_threshold` configuration.

### Phase 4: Implement

Code generation from the approved design:

1. **Generate Code**: Agent produces implementation
2. **Validate Syntax**: Check for syntax errors
3. **Check Patterns**: Detect dangerous code patterns
4. **Write Files**: Create backup and write changes

```bash
# Example: Claude implements
claude-code --task "Implement the approved proposal" --context proposal.json
```

Implemented by `ImplementPhase` class with backup/rollback support.

### Test Generation

The nomic loop can generate tests from specifications using the TDD helpers in
`aragora/nomic/test_generator.py`, producing unit and edge-case tests alongside
implementation steps.

### Phase 5: Verify

Three-stage verification:

1. **Syntax Check**: `python -m py_compile` on all modified files
2. **Import Check**: `python -c "import aragora"` ensures no broken imports
3. **Test Suite**: `pytest tests/` must pass (currently 34,400+ tests)

Implemented by `VerifyPhase` class. Triggers rollback on failure.

### Phase 6: Commit

If all checks pass:

```bash
git add -A
git commit -m "feat(nomic): <title>

<description>

Proposed by: <agent>
Approved by: <voters>
Cycle: N

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

Implemented by `CommitPhase` class.

## Python API

### Using NomicLoop (Legacy)

```python
from aragora.nomic import NomicLoop

# Create loop with agents
loop = NomicLoop(
    aragora_path="/path/to/project",
    agents=agents,
    max_cycles=10,
    auto_commit=True,
)

# Run cycles
await loop.run()

# Or run a single cycle
result = await loop.run_cycle()
```

### Using NomicStateMachine (Recommended)

```python
from aragora.nomic import (
    NomicStateMachine,
    create_nomic_state_machine,
    create_handlers,
    NomicState,
)

# Create with factory
state_machine = create_nomic_state_machine(
    aragora_path="/path/to/project",
    agents=agents,
)

# Or create manually with handlers
handlers = create_handlers(aragora_path, agents)
state_machine = NomicStateMachine(handlers=handlers)

# Run the loop
await state_machine.run()
```

### Phase Classes

Each phase can be used independently:

```python
from aragora.nomic.phases import (
    ContextPhase,
    DebatePhase,
    DesignPhase,
    ImplementPhase,
    VerifyPhase,
    CommitPhase,
)

# Example: Run debate phase
debate = DebatePhase(
    aragora_path=path,
    claude_agent=claude,
    codex_agent=codex,
    consensus_threshold=0.66,
)
result = await debate.run(context="Improve error handling")

# Example: Run design phase
design = DesignPhase(
    aragora_path=path,
    claude_agent=claude,
    protected_files=["CLAUDE.md", "core.py"],
    auto_approve_threshold=0.5,
)
result = await design.run(proposal=winning_proposal)
```

### Safety Gates

```python
from aragora.nomic.gates import (
    is_protected_file,
    check_change_volume,
    check_dangerous_patterns,
    check_all_gates,
    GateConfig,
)

# Check if file is protected
if is_protected_file("CLAUDE.md"):
    print("Cannot modify protected file")

# Check change volume
result = check_change_volume(
    files_changed=15,
    max_files=20,
    lines_added=500,
    lines_removed=100,
    max_lines=1000,
)

# Check all gates at once
config = GateConfig(max_files=10, max_lines=500)
result = check_all_gates({
    "files": ["a.py", "b.py"],
    "code": "import os\nos.system('echo hello')",
    "estimated_duration": 300,
})
```

## State Management

### State File

Located at `.nomic/nomic_state.json`:

```json
{
  "phase": "verify",
  "stage": "complete",
  "cycle": 3,
  "all_passed": true,
  "features_integrated": {
    "Phase 1": ["ContinuumMemory", "ReplayRecorder", ...],
    "Phase 2": ["ConsensusMemory", "InsightExtractor"],
    ...
  }
}
```

### Checkpoints

The CheckpointManager saves state for crash recovery:

```python
checkpoint.save("debate-123", {
    "phase": "debate",
    "round": 2,
    "messages": [...],
    "votes": [...]
})
```

Resume after crash:
```bash
python scripts/run_nomic_with_stream.py run --resume
```

### Backups

Before each cycle, critical files are backed up:

```
.nomic/backups/
├── backup_cycle_1_20260103_120000/
│   ├── scripts/nomic_loop.py
│   ├── aragora/core.py
│   └── ...
└── backup_cycle_2_20260103_130000/
    └── ...
```

## Agent Specialization

Each agent has a persona that evolves based on success:

| Agent | Role | Specialization |
|-------|------|----------------|
| **Claude** | Visionary | Architecture, design patterns |
| **Codex** | Engineer | Implementation, performance |
| **Gemini** | Strategist | Product vision, viral growth |
| **Grok** | Lateral Thinker | Creative solutions, edge cases |

Personas are managed by PersonaManager and evolve via PersonaLaboratory.

### Structured Thinking Protocols

Each agent uses a specialized thinking protocol to ensure high-quality proposals:

#### Claude (Visionary Architect)
```
1. EXPLORE: Understand current state - read files, trace code paths
2. PLAN: Design approach before implementing - consider alternatives
3. REASON: Show thinking step-by-step - explain tradeoffs
4. PROPOSE: Make concrete, actionable proposals with clear impact
```
Uses Claude Code's Explore/Plan modes for deep codebase understanding.

#### Codex (Pragmatic Engineer)
```
1. TRACE: Follow code paths to understand dependencies and data flow
2. ANALYZE: Identify patterns, anti-patterns, and improvement opportunities
3. DESIGN: Consider multiple implementation approaches with pros/cons
4. VALIDATE: Think about edge cases, tests, and failure modes
```
Shows reasoning chains: "I observed X → which implies Y → so we should Z"

#### Gemini (Product Strategist)
```
1. EXPLORE: Understand current state - what exists, what's missing
2. ENVISION: Imagine the ideal outcome - what would success look like
3. REASON: Show thinking step-by-step - explain tradeoffs
4. PROPOSE: Make concrete, actionable proposals with clear impact
```
Focuses on viral growth potential and developer excitement.

#### Grok (Lateral Synthesizer)
```
1. DIVERGE: Generate multiple unconventional perspectives
2. CONNECT: Find surprising links between disparate ideas
3. SYNTHESIZE: Combine insights into novel, coherent proposals
4. GROUND: Anchor creative ideas in practical implementation
```
Shows lateral thinking: "Others see X, but what if Y..."

## Integrated Features

The nomic loop integrates all 30+ features:

### Memory Systems
- ContinuumMemory (multi-timescale)
- MemoryStream (per-agent)
- ConsensusMemory (topic tracking)
- SemanticRetriever (embeddings)

### Evolution
- PersonaManager (traits)
- PromptEvolver (system prompts)
- PersonaLaboratory (A/B testing)
- EloSystem (skill tracking)

### Verification
- ProofExecutor (code verification)
- FormalVerificationManager (Z3)
- ReliabilityScorer (confidence)

### Resilience
- CheckpointManager (crash recovery)
- BreakpointManager (human intervention)
- DebateTracer (audit logs)

## Monitoring

### Logs

Main log at `.nomic/nomic_loop.log`:

```
=== NOMIC LOOP STARTED: 2026-01-03T13:52:49 ===
[13:52:50] NOMIC CYCLE 1
[13:52:50]   [replay] Recording cycle 1
[13:52:50]   [persona] claude-visionary: balanced
[13:52:54]   claude (Claude Code): exploring codebase...
```

### Metrics

Track via the dashboard or logs:

- Consensus rate per cycle
- Average rounds to convergence
- Test pass rate
- Feature implementation success rate

## Troubleshooting

### Loop Stuck in Phase

Check if an agent CLI tool is hung:
```bash
ps aux | grep -E "claude|codex|kilo"
```

Kill and resume:
```bash
pkill -f "codex exec"
python scripts/run_nomic_with_stream.py run --resume
```

### Tests Failing

1. Check the specific failure in `.nomic/nomic_loop.log`
2. Rollback to last backup if needed
3. Run tests manually: `pytest tests/ -v`

### Checkpoint Corruption

Reset state:
```bash
rm .nomic/nomic_state.json
rm .nomic/checkpoints.db
```

## Best Practices

1. **Run with `--auto`** for autonomous operation
2. **Monitor the live dashboard** for real-time visibility
3. **Review commits** even with auto-commit (can revert)
4. **Set cycle limits** appropriate for the task
5. **Use human-approval** for critical codebases
