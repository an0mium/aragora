# ADR-001: Debate Orchestration Pattern

## Status
Accepted

## Context
Aragora needed a flexible system for orchestrating multi-agent debates where heterogeneous AI agents discuss, critique, and improve responses. The core challenge was designing a pattern that:
- Supports multiple debate protocols (adversarial, collaborative, Socratic)
- Handles agent failures gracefully
- Enables extensibility for new phase types
- Maintains consensus detection and convergence tracking

## Decision
We adopted the **Arena-Phase pattern** with the following structure:

### Arena (Orchestrator)
- `aragora/debate/orchestrator.py` - Main `Arena` class
- Coordinates phases, manages state, handles consensus detection
- Uses ELO-based team selection (`team_selector.py`)
- Integrates memory systems via `memory_manager.py`

### Phase Protocol
- Each debate phase implements a common interface
- Phases are extracted to `aragora/debate/phases/`:
  - `proposal_phase.py` - Initial proposals
  - `critique_phase.py` - Cross-agent critique
  - `voting_phase.py` - Agent voting
  - `synthesis_phase.py` - Consensus synthesis
  - `consensus_verification.py` - Final verification

### Execution Flow
```
Arena.run()
  -> PhaseExecutor.execute_phase()
     -> For each agent: agent.generate() / agent.critique() / agent.vote()
  -> ConsensusDetector.check_consensus()
  -> ConvergenceTracker.update()
```

### Agent Fallback
- `aragora/agents/fallback.py` - OpenRouter fallback on quota errors
- `aragora/resilience.py` - CircuitBreaker for failure handling
- `aragora/agents/airlock.py` - AirlockProxy for agent resilience

## Consequences
**Positive:**
- Clean separation between orchestration and agent logic
- Easy to add new phases without modifying core
- Robust error handling with fallback mechanisms
- ELO-based selection improves team quality over time

**Negative:**
- Complex state management across phases
- Debate module grew large (38.5K lines) - potential decomposition needed
- Phase dependencies require careful ordering

## References
- `aragora/debate/orchestrator.py` - Arena implementation
- `aragora/debate/phases/` - Phase implementations
- `docs/DEBATE_PHASES.md` - Phase documentation
