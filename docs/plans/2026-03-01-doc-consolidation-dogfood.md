# Doc Consolidation & Self-Improvement Dogfood Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate aragoradocs by deleting duplicates and migrating unique docs into the main repo, then dogfood the self-improvement pipelines using canonical goals as input, and evaluate results.

**Architecture:** Three sequential phases. Phase 1 is file operations + git commit. Phase 2 runs three pipeline paths (self_develop dry-run, nomic debate, aragora ask). Phase 3 analyzes results and produces improvement plan.

**Tech Stack:** bash (file ops), python (pipeline scripts), git

---

### Task 1: Delete duplicate files from aragoradocs

**Files:**
- Delete: `~/Development/aragoradocs/ARAGORA_COMMERCIAL_OVERVIEW.md`
- Delete: `~/Development/aragoradocs/ARAGORA_FEATURE_DISCOVERY.md`
- Delete: `~/Development/aragoradocs/ARAGORA_BUSINESS_SUMMARY.md`
- Delete: `~/Development/aragoradocs/ARAGORA_COMPREHENSIVE_REPORT.md`
- Delete: `~/Development/aragoradocs/ARAGORA_ELEVATOR_PITCH.md`
- Delete: `~/Development/aragoradocs/ARAGORA_WHY_ARAGORA.md`
- Delete: `~/Development/aragoradocs/ARAGORA_COMPARISON_MATRIX.md`
- Delete: `~/Development/aragoradocs/ARAGORA_COMMERCIAL_POSITIONING.md`
- Delete: `~/Development/aragoradocs/ARAGORA_EXECUTION_PROGRAM_2026Q2_Q4.md`
- Delete: `~/Development/aragoradocs/ARAGORA_STRATEGIC_ANALYSIS.md`

**Step 1:** Delete the 10 duplicate markdown files.

**Step 2:** Delete the stale `ARAGORA_PRICING.md` (keep `ARAGORA_PRICING_PAGE.md` for merge).

**Step 3:** Verify remaining files are only: CANONICAL_GOALS, HONEST_ASSESSMENT, OMNIVOROUS_ROADMAP, SME_STARTER_PACK, PRICING_PAGE (to migrate), plus external marketing files.

---

### Task 2: Migrate unique docs into main repo

**Files:**
- Copy: `~/Development/aragoradocs/ARAGORA_CANONICAL_GOALS.md` -> `docs/CANONICAL_GOALS.md`
- Copy: `~/Development/aragoradocs/ARAGORA_HONEST_ASSESSMENT.md` -> `docs/HONEST_ASSESSMENT.md`
- Copy: `~/Development/aragoradocs/ARAGORA_OMNIVOROUS_ROADMAP.md` -> `docs/OMNIVOROUS_ROADMAP.md`
- Copy: `~/Development/aragoradocs/ARAGORA_SME_STARTER_PACK.md` -> `docs/guides/SME_STARTER_PACK.md`
- Copy: `~/Development/aragoradocs/ARAGORA_PRICING_PAGE.md` -> overwrite `docs/PRICING.md` (if newer/more complete)

**Step 1:** Copy files to main repo.

**Step 2:** Delete the migrated files from aragoradocs.

**Step 3:** Verify the remaining aragoradocs files are only marketing/external materials.

**Step 4:** Commit migrated docs to main repo.

---

### Task 3: Dogfood self_develop.py --dry-run

**Step 1:** Run self_develop.py with a goal from the canonical goals doc.

```bash
cd /Users/armand/Development/aragora
python scripts/self_develop.py \
  --goal "Wire the Nomic Loop end-to-end by replacing stub methods in the orchestration loop, achieving autonomous self-improvement cycles as described in docs/CANONICAL_GOALS.md Pillar 5" \
  --dry-run
```

**Step 2:** Capture and evaluate the task decomposition output.

---

### Task 4: Dogfood nomic_staged.py debate

**Step 1:** Run a staged debate.

```bash
cd /Users/armand/Development/aragora
python scripts/nomic_staged.py debate
```

**Step 2:** Capture debate output and evaluate quality.

---

### Task 5: Dogfood aragora ask with quality gate

**Step 1:** Run a full debate with quality gate using canonical goals as context.

```bash
cd /Users/armand/Development/aragora
python -m aragora ask \
  "Using the canonical goals in docs/CANONICAL_GOALS.md as source of truth, identify the single highest-impact improvement that would advance Pillar 5 (Self-Repair and Self-Improvement). Produce a ranked task list with specific file paths, acceptance criteria, and rollback triggers." \
  --agents anthropic-api,openai-api \
  --rounds 2 \
  --consensus hybrid \
  --local \
  --timeout 600 \
  --quality-fail-open \
  --context "$(head -200 docs/CANONICAL_GOALS.md)"
```

**Step 2:** Parse quality line from output.

**Step 3:** Evaluate against quality gate thresholds.

---

### Task 6: Evaluate results and produce improvement plan

Analyze all three dogfood runs. Identify:
- What worked well
- Where pipelines fell short
- Specific improvements needed

Produce improvement plan saved to docs/plans/.
