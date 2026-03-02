# Doc Consolidation & Self-Improvement Pipeline Dogfooding

**Date:** 2026-03-01
**Status:** Approved

## Overview

Three-phase task: (1) consolidate aragoradocs by deleting duplicates, migrating unique files into the main repo, and merging pricing docs; (2) dogfood the self-improvement pipelines using the canonical goals doc as input; (3) evaluate results and develop improvement plans.

## Phase 1: Doc Consolidation

### Delete from aragoradocs (duplicates)

| aragoradocs File | Already In Main Repo |
|---|---|
| ARAGORA_COMMERCIAL_OVERVIEW.md | docs/COMMERCIAL_OVERVIEW.md |
| ARAGORA_FEATURE_DISCOVERY.md | docs/FEATURE_DISCOVERY.md |
| ARAGORA_BUSINESS_SUMMARY.md | docs/ARAGORA_BUSINESS_SUMMARY.md |
| ARAGORA_COMPREHENSIVE_REPORT.md | docs/investor/ARAGORA_COMPREHENSIVE_REPORT.md |
| ARAGORA_ELEVATOR_PITCH.md | docs/investor/ARAGORA_ELEVATOR_PITCH.md |
| ARAGORA_WHY_ARAGORA.md | docs/WHY_ARAGORA.md |
| ARAGORA_COMPARISON_MATRIX.md | docs/COMPARISON_MATRIX.md |
| ARAGORA_COMMERCIAL_POSITIONING.md | docs/status/COMMERCIAL_POSITIONING.md |
| ARAGORA_EXECUTION_PROGRAM_2026Q2_Q4.md | docs/status/EXECUTION_PROGRAM_2026Q2_Q4.md |
| ARAGORA_STRATEGIC_ANALYSIS.md | docs/STRATEGIC_ANALYSIS.md |

### Migrate to main repo

| Source | Destination |
|---|---|
| ARAGORA_CANONICAL_GOALS.md | docs/CANONICAL_GOALS.md |
| ARAGORA_HONEST_ASSESSMENT.md | docs/HONEST_ASSESSMENT.md |
| ARAGORA_OMNIVOROUS_ROADMAP.md | docs/OMNIVOROUS_ROADMAP.md |
| ARAGORA_SME_STARTER_PACK.md | docs/guides/SME_STARTER_PACK.md |

### Merge

ARAGORA_PRICING_PAGE.md (detailed) becomes docs/PRICING.md. Delete both PRICING files from aragoradocs.

### Keep in aragoradocs (external marketing)

- Aragora_Enterprise_Pilot_Prospects.md
- Aragora_Outbound_Email_Campaign.md
- Aragora_Updated_Description.md
- aragora_report_pack.md
- Binary files (docx, pptx)

## Phase 2: Dogfood Self-Improvement Pipeline

### Test 1: self_develop.py --dry-run
Feed a goal from canonical goals. Evaluate task decomposition quality.

### Test 2: nomic_staged.py debate
Run a debate on "what single improvement would most benefit aragora" with canonical goals as context.

### Test 3: aragora ask with quality gate
Full debate with quality gate using migrated docs as context.

## Phase 3: Evaluate & Plan

Analyze dogfood results against quality gate thresholds. Produce concrete improvement plan.

## Success Criteria

- aragoradocs reduced from 24 to ~8 files (marketing/external only)
- 4 unique docs committed to main repo
- At least one dogfood pipeline run produces quality-gate-passing output
- Improvement plan addresses specific pipeline weaknesses
