# Nomic Loop Live Fire Test

**Date:** 2026-03-02
**Status:** Approved
**Goal:** Run the self-improvement loop end-to-end on a real improvement goal and produce a real autonomous commit.

## Background

The Nomic Loop implementation spans ~67K LOC across `scripts/nomic_loop.py` and `aragora/nomic/`. All 6 phases (context, debate, design, implement, verify, commit) are wired with real LLM calls, real git operations, and real subprocess execution. However, the full loop has never been demonstrated end-to-end in a live environment with real API keys producing a real commit.

This test proves the self-improvement thesis by executing the loop for real.

## Test Scenario

**Goal:** A small, concrete, verifiable improvement that exercises all phases.

Candidate goals (in priority order):
1. "Add type annotations to the 3 functions with the most callers in aragora/nomic/meta_planner.py"
2. "Improve error messages in aragora/nomic/task_decomposer.py to include actionable context"
3. "Add missing docstrings to public methods in aragora/nomic/branch_coordinator.py"

Selection criteria: (a) small enough for one cycle, (b) verifiable by verify phase (tests pass, types check), (c) safe (no protected files, no breaking changes), (d) targets the nomic module itself (self-referential improvement).

## Execution Plan

### Phase 0: Setup (~5 min)
- Create branch `nomic/live-fire-001` from main
- Verify API keys: `ANTHROPIC_API_KEY` (required), `OPENAI_API_KEY` (optional fallback)
- Set cost cap: `--max-cost 10`
- Capture baseline: current test count, mypy status, file checksums

### Phase 1: Dry Run (~2 min)
```bash
python scripts/self_develop.py --goal "<selected goal>" --dry-run
```
Verify the decomposition looks reasonable before spending API credits.

### Phase 2: Live Execution (~10-30 min)
```bash
python scripts/nomic_loop.py --cycles 1 --agents claude --max-cost 10 2>&1 | tee nomic_live_fire.log
```
Or via self_develop.py:
```bash
python scripts/self_develop.py --goal "<selected goal>" --require-approval 2>&1 | tee nomic_live_fire.log
```
Monitor each phase transition. Record timing.

### Phase 3: Diagnosis (variable)
- If it completes: inspect the commit diff, verify changes are correct
- If it fails: identify which phase broke, document the failure, fix the wiring, retry

### Phase 4: Validation
- Run `pytest tests/nomic/ -x -q` to confirm no regressions
- Run `python -c "import ast; ast.parse(open('<changed_file>').read())"` for syntax
- Run `mypy <changed_file>` for type safety
- Manual review of the diff for quality

### Phase 5: Document
- Save execution log to `docs/plans/nomic_live_fire_001_log.txt`
- Record success/failure, timing, cost, and changes made
- If successful: the commit on `nomic/live-fire-001` is the proof artifact

## Success Criteria

1. The loop runs through all phases without manual intervention
2. A real commit is produced on the branch
3. The commit contains meaningful, correct code changes (not empty or trivial)
4. Tests still pass after the changes
5. Total API cost < $10

## Risk Mitigation

- **Branch isolation:** All work on `nomic/live-fire-001`, never on main
- **Cost cap:** `--max-cost 10` prevents runaway API spend
- **Protected files:** `CLAUDE.md`, `__init__.py`, `.env`, `nomic_loop.py` cannot be modified
- **Verify phase:** Catches broken code before commit
- **Rollback:** `git reset --hard` if changes are unacceptable

## Expected Failure Modes

1. **API quota exhaustion:** Mitigated by cost cap and OpenRouter fallback
2. **Phase timeout:** Debate rounds may take longer than expected with real LLMs
3. **Verify phase false positive:** May reject valid changes due to pre-existing test issues
4. **Scope creep:** LLM may attempt changes beyond the stated goal — design phase safety review catches this
5. **Import/dependency issues:** nomic_loop.py may have import errors in fresh environment

## What This Proves

If successful, this demonstrates:
- Aragora can autonomously identify, plan, implement, verify, and commit code improvements
- The adversarial debate phase produces actionable improvement proposals
- The safety rails (protected files, verify phase, cost cap) work in practice
- The self-improvement thesis is not aspirational — it's operational
