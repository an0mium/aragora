# Nomic Loop

The Nomic Loop is Aragora's autonomous self-improvement cycle. Agents debate
what to improve, design solutions, implement code, and verify changes -- all
with safety guardrails to prevent regressions.

## Loop Overview

```mermaid
flowchart TD
    Start([Trigger: Goal or Schedule]) --> P0

    subgraph P0["Phase 0 -- Context Gathering"]
        Scan["Scan codebase state"]
        Status["Check feature status"]
        Metrics["Gather test/coverage metrics"]
        Scan --> Status --> Metrics
    end

    P0 --> P1

    subgraph P1["Phase 1 -- Multi-Agent Debate"]
        Propose["Agents propose improvements"]
        Critique["Cross-agent critique"]
        Vote["Consensus vote on priorities"]
        Propose --> Critique --> Vote
    end

    P1 --> P2

    subgraph P2["Phase 2 -- Design"]
        Arch["Architecture planning"]
        Decompose["Task decomposition"]
        Branch["Branch coordination"]
        Arch --> Decompose --> Branch
    end

    P2 --> P3

    subgraph P3["Phase 3 -- Implementation"]
        Backup["Create backup"]
        Checksum["Record protected file checksums"]
        Code["Generate code (Codex / Claude)"]
        Backup --> Checksum --> Code
    end

    P3 --> P4

    subgraph P4["Phase 4 -- Verification"]
        Syntax["Syntax check"]
        Tests["Run test suite"]
        IntCheck["Integration checks"]
        ChecksumVerify["Verify protected files unchanged"]
        Syntax --> Tests --> IntCheck --> ChecksumVerify
    end

    P4 --> Decision{All checks pass?}
    Decision -- Yes --> Commit["Commit changes"]
    Decision -- No --> Rollback["Rollback to backup"]
    Rollback --> P1
    Commit --> Start
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant MP as MetaPlanner
    participant TD as TaskDecomposer
    participant BC as BranchCoordinator
    participant Ag as Agents
    participant VF as Verifier
    participant Git as Git

    O->>O: Phase 0 -- Gather context
    O->>MP: Phase 1 -- Debate improvements
    MP->>Ag: Distribute proposals
    Ag-->>MP: Proposals + critiques + votes
    MP-->>O: Prioritized improvement plan

    O->>TD: Phase 2 -- Decompose tasks
    TD-->>O: Subtask list
    O->>BC: Create feature branches
    BC-->>O: Branch ready

    O->>Git: Phase 3 -- Backup + checksum
    O->>Ag: Generate implementation
    Ag-->>O: Code changes

    O->>VF: Phase 4 -- Verify
    VF->>VF: Syntax check
    VF->>VF: Run tests
    VF->>VF: Verify checksums
    VF-->>O: Pass / Fail

    alt All checks pass
        O->>Git: Commit changes
    else Any check fails
        O->>Git: Rollback to backup
        O->>O: Re-enter Phase 1
    end
```

## Safety Features

```mermaid
flowchart LR
    subgraph Before["Pre-Implementation"]
        Backup["Automatic backup"]
        Checksums["Protected file checksums"]
        Approval["Human approval gate (optional)"]
    end

    subgraph During["During Implementation"]
        Protected["Protected file guard"]
        BranchIsolation["Feature branch isolation"]
        Incremental["Incremental changes"]
    end

    subgraph After["Post-Implementation"]
        SyntaxCheck["Syntax validation"]
        TestSuite["Full test suite (45,100+ tests)"]
        ChecksumVerify["Checksum verification"]
        Rollback["Automatic rollback on failure"]
    end

    Before --> During --> After
```

| Safety Mechanism | When | What It Protects |
|------------------|------|------------------|
| Automatic backup | Before Phase 3 | Full codebase state |
| Protected file checksums | Before and after Phase 3 | `CLAUDE.md`, `core.py`, `__init__.py`, `nomic_loop.py` |
| Human approval gate | Between phases (optional) | High-risk changes |
| Feature branch isolation | Phase 3 | Main branch stability |
| Syntax validation | Phase 4 | Basic correctness |
| Test suite execution | Phase 4 | Functional correctness |
| Automatic rollback | On Phase 4 failure | Reverts all changes |

## CLI Entry Points

```bash
# Full autonomous run with approval gates
python scripts/self_develop.py --goal "Improve test coverage" --require-approval

# Dry run -- preview decomposition only
python scripts/self_develop.py --goal "Refactor dashboard" --dry-run

# Debate-based decomposition for abstract goals
python scripts/self_develop.py --goal "Maximize SME utility" --dry-run --debate

# Staged execution -- run individual phases
python scripts/nomic_staged.py debate
python scripts/nomic_staged.py design
python scripts/nomic_staged.py implement
python scripts/nomic_staged.py verify
python scripts/nomic_staged.py commit

# Streaming loop with multiple cycles
python scripts/run_nomic_with_stream.py run --cycles 3
```

## Key Components

| Component | Path | Role |
|-----------|------|------|
| Nomic Loop | `scripts/nomic_loop.py` | Core loop runner (protected) |
| MetaPlanner | `aragora/nomic/meta_planner.py` | Debate-driven goal prioritization |
| TaskDecomposer | `aragora/nomic/task_decomposer.py` | Break goals into subtasks |
| BranchCoordinator | `aragora/nomic/branch_coordinator.py` | Parallel branch management |
| AutonomousOrchestrator | `aragora/nomic/autonomous_orchestrator.py` | End-to-end orchestration |
| Staged Runner | `scripts/nomic_staged.py` | Phase-by-phase execution |
| Self-Develop CLI | `scripts/self_develop.py` | Goal-driven entry point |

## Track-Based Execution

The self-improvement system supports parallel execution across focus tracks:

```bash
python scripts/self_develop.py --goal "Improve platform" --tracks sme developer qa --max-parallel 2
```

Each track operates on its own feature branch, coordinated by `BranchCoordinator`,
and merged only after all verification checks pass.
