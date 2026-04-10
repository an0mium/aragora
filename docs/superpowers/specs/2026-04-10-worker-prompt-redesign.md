# Worker Prompt Redesign Spec

> **Problem**: Boss loop workers have a 6% overnight completion rate. Workers hallucinate file paths, fail simple tasks, and produce low-quality code. The root cause is the prompt structure — it treats workers as constrained executors instead of intelligent collaborators.

> **Evidence**: The same models (Claude Code, Codex) succeed at 90%+ rates when given full context and freedom (see RingRift repo workflows). The difference is prompt design, not model capability.

## Current Prompt Problems

1. **Context starvation**: Workers get 200-line truncated snippets. For a 500-line file, they miss the bottom 300 lines which often contain the functions they need to modify.

2. **Exploration prohibition**: "NEVER spend more than 2 minutes reading/exploring" (removed but the spirit persists in constraints). Workers rush to write code before understanding the problem.

3. **Boilerplate overload**: 60% of the prompt is commit instructions, staging rules, and stop conditions. The agent already knows how git works.

4. **Hard scope boundaries**: "Only modify files in this scope. If the fix genuinely requires other files, stop and report that blocker." This causes workers to make incorrect, scope-bounded fixes instead of correct cross-file fixes.

5. **No test context**: Workers are told "run pytest tests/foo.py" but never shown the actual test file. They can't understand what the test expects.

6. **No architectural context**: Workers don't know how their target file fits into the system. They make local fixes that break callers.

## Redesigned Prompt Structure

### Principles (from RingRift success patterns)

- **90% context, 10% instructions** (inverted from current 40/60)
- **Trust the agent** — it knows git, it knows how to code, it knows how to test
- **Full file contents** for the target + test file (up to 500 lines each)
- **Encourage exploration** — "read what you need" not "stay in scope"
- **Explain the goal** in plain English, not issue-body markdown
- **Show prior failures** as learning material, not warnings

### New Prompt Template

```
# {title}

## What needs to happen

{plain_english_goal}

## The code you're working with

{full_file_contents_of_target_files}

## The test that validates your change

{full_test_file_contents}

## How this code fits in

{caller_context — who imports/calls the functions you're modifying}
{dependency_context — what this module imports and uses}

## Prior attempts (if any)

{repair_journal — what was tried, why it failed, what to try differently}

## Validation

Run: {validation_command}
Expected: All tests pass.

## When you're done

Commit your changes with `git add <files> && git commit -m 'fix: ...'`.
```

That's it. No scope restrictions. No time pressure. No 15-line stop condition block. No "REMINDER" screaming about commits.

### Key Changes

| Aspect | Current | Redesigned |
|--------|---------|-----------|
| File content | First 200 lines, 5 files max | Full content, target + test (2 files) |
| Scope enforcement | Hard boundary with blocker instruction | None — trust the agent to be focused |
| Instructions | 12 rules about committing/staging | Single line: "commit when done" |
| Exploration | Implicitly discouraged | Encouraged: "read what you need" |
| Role framing | "One managed lane in a supervised swarm" | Removed — just describe the task |
| Test context | File path only | Full test file content |
| Caller context | 5 grep results, symbol-only | Full import chain + usage examples |
| Time pressure | "NEVER spend more than 2 minutes" | None |

### Context Enrichment Improvements

1. **Full file content** (not 200-line truncation) for files under 500 lines
2. **Include the test file** — if `tests/swarm/test_boss_loop.py` is the validation target, include its content (or the relevant test class)
3. **Show the import chain** — what does the target file import? What imports it?
4. **Include CLAUDE.md** from the target's directory — it has local conventions
5. **Show recent git log** for the target file — what changed recently?

### What to Remove

- The entire role statement ("You are one Aragora-managed CLI worker lane...")
- All commit/staging discipline blocks (agents know git)
- File scope as hard boundary (soft guidance is fine)
- The "Decision boundary" blocker instruction
- "REMINDER" blocks
- Lease/receipt details (the harness handles this, agent doesn't need to know)
- Agent-specific discipline blocks (Codex early-commit, Claude how-to-work)

### What to Keep

- Repair journal (but reframed as learning material)
- Validation command (one line, not a section)
- Acceptance criteria (if specific and meaningful)

## Implementation

### Phase 1: Enrich context (immediate)

Expand `_enrich_task_context()`:
- Remove 200-line truncation for files under 500 lines
- Add test file content (read the validation target)
- Add directory CLAUDE.md if present
- Add `git log --oneline -5 -- {file}` for recent changes

### Phase 2: Slim the prompt (immediate)

Rewrite `_build_prompt()`:
- Remove role statement, scope boundary, discipline blocks, stop conditions
- Reduce to: goal + context + validation + commit instruction
- Total template should be ~10 lines of instructions, rest is code content

### Phase 3: Remove hard scope (requires testing)

- Change "Only modify files in this scope" to soft guidance
- Let the agent use its judgment about which files need changing
- The verification gate (tests passing) is the real boundary, not file lists

### Phase 4: Measure (compare overnight completion rates)

- Run a 20-tick canary with the new prompt
- Compare: files_changed, elapsed time, completion rate
- Success criteria: >50% completion rate (up from 6%)

## Risk Assessment

**Risk**: Workers without scope constraints might make broad, breaking changes.
**Mitigation**: The verification gate (tests must pass) catches this. If a worker breaks tests, it fails — same as today. The difference is workers can now fix things properly instead of making scope-bounded hacks.

**Risk**: Full file content makes prompts too long.
**Mitigation**: Focus on 2 files (target + test), not 5 truncated files. A 500-line Python file is ~15K tokens. Two files = ~30K tokens of context, well within the 200K token window.

**Risk**: Removing commit discipline causes workers to exit without committing.
**Mitigation**: Keep a single line: "Commit when done." The agent knows what committing means. The 15-line discipline block didn't prevent the problem (workers still exited without commits overnight).
