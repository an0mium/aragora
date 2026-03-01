# Dogfood Findings: Output Quality and Process Reliability

Date: 2026-02-28

## Scope
- Benchmark report: `docs/plans/dogfood_timeout_gate_report_2026-02-28_timeout1200_api_roster.json`
- Reclassified benchmark report: `docs/plans/dogfood_timeout_gate_report_2026-02-28_timeout1200_api_roster_reclassified.json`
- Live gate validation: fail-closed run after parser fix (EXIT=1 on invalid JSON payload)

## Measured Results
- Agent roster benchmarked: `anthropic-api,openai-api,gemini,grok`
- Strict timeout: `1200s` (`subprocess timeout 1320s`)
- Runs: `3`
- Successes: `3/3` (no strict timeout)
- Durations: `531.77s`, `536.79s`, `433.66s`
- Quality verdicts (deterministic contract): `good`, `needs_work`, `needs_work`
- Good-run rate: `33%` (1/3)
- Average quality score: `8.79/10`
- Reclassified runtime blockers: `0` blocker runs, `3` warning-only runs (`ResourceWarning`/unclosed transport noise)

## Quality Defects Observed
- Missing/invalid `JSON Payload` section (2/3 runs)
- Missing explicit quantitative thresholds in `Gate Criteria` (1/3 runs)
- Empty required section content (`Suggested Subtasks`) in one run
- ResourceWarning noise from unclosed sockets/transports in stderr (all runs)

## Root Cause Found
The post-consensus quality gate was phrase-sensitive and only activated for narrow task text patterns (`"output sections ..."`).
Prompts like `"these sections as markdown headers"` could bypass contract derivation, causing silent skip of quality repair/fail-closed behavior.

## Implemented Today
1. **Broadened output contract parsing** in `aragora/debate/output_quality.py`:
- Added support for:
  - `these sections as markdown headers: ...`
  - `sections as markdown headers: ...`
  - `required sections: ...`
- Added fallback inference from known required headings when embedded in free-form prose.

2. **Added parser regression tests** in `tests/debate/test_output_quality.py`:
- `test_derive_output_contract_from_task_markdown_headers_phrase`
- `test_derive_output_contract_from_task_fallback_known_headings`

3. **Added explicit contract sources in CLI**:
- `--output-contract-file` (JSON contract, highest precedence)
- `--required-sections` (comma-separated heading contract)

4. **Improved quality observability** in `aragora/cli/commands/debate.py`:
- Emits explicit signal when no contract is derived:
  - `[quality] skipped=no_contract reason=no_explicit_output_contract_detected`
- Prevents silent quality-gate non-activation.

5. **Added explicit contract override + fail-fast guard**:
- If `--quality-fail-closed` is set and no contract is derivable/provided, command exits early with code `2` and clear guidance.

6. **Added deterministic JSON finalizer pass**:
- Final quality pipeline now applies `finalize_json_payload(...)` before final verdict, ensuring valid JSON blocks are injected/repaired when required by contract.

7. **Added stderr blocker-vs-warning classification tooling**:
- New classifier module: `aragora/debate/runtime_blockers.py`
- New report reclassifier script: `scripts/reclassify_dogfood_report.py`
- Reclassified report confirms warning-only noise does not count as runtime blockers.

8. **Validated fail-closed behavior live**:
- Real run now fails closed for contract violations:
  - `Debate failed quality gate: Post-consensus quality gate failed after upgrade loops: JSON Payload is invalid or missing ...`

## Recommended Next Improvements (Priority Order)
1. **Schema-level contract support**
- Extend `--output-contract-file` to support optional key/type constraints per JSON payload field (not just heading-level requirements).

2. **Wire blocker classifier into the benchmark harness by default**
- Make `classify_stderr_signals(...)` the default for future dogfood report generation so warning-only noise never inflates blocker counts.

3. **Add throughput gates alongside quality gates**
- Suggested ops gate: p95 run duration <= 10 minutes for 4-model, 2-round debates.
- Use this to detect regressions where quality remains acceptable but throughput collapses.

4. **Roster-specific timeout policy**
- Keep `1200s` for 4-model API roster with full contract.
- Use higher timeout only for CLI full-latest roster if subprocess instrumentation confirms active provider I/O.

5. **Re-run a 5-run quality gate with explicit contract file**
- Target: `good_run_rate >= 0.8`, `runtime_blockers_zero == true` (post-classification), `p95_duration_seconds <= 600`.

## Proposed Acceptance Gates for Next Dogfood Cycle
- `good_run_rate >= 0.8` over 5 runs
- `all_runs_quality_line_present == true` OR explicit `skipped=no_contract`
- `runtime_blockers_zero == true` (excluding classified warning-only noise)
- `p95_duration_seconds <= 600` for API roster

## Additional Findings (Timeout 1800 Analysis)
- `--timeout 1800` is already the CLI default (`aragora/cli/parser.py`).
- Two concrete latency risks were identified that can push runs beyond practical budgets:

1. **Quality-upgrade timeout budgeting bug (fixed)**
- In `aragora/cli/commands/debate.py`, per-provider upgrade attempts used a fixed formula:
  - `per_attempt_timeout = max(120, min(600, int(debate_timeout * 0.4)))`
- At `--timeout 1800`, this gave `600s` per provider attempt; with 4+ providers and multiple loops, potential upgrade time could exceed remaining debate wall-clock budget.
- **Fix implemented**: per-attempt timeout is now budget-aware, derived from remaining global wall-clock budget, capped to `30..180s`, and stops when budget is exhausted.

2. **Over-broad web-search auto-triggering (fixed)**
- `Anthropic`, `OpenAI`, and `Gemini` adapters previously triggered web mode on generic keywords like `latest/current/recent`.
- The dogfood context contains `current`, which could unintentionally activate web tools and increase latency.
- **Fix implemented**: refined patterns to require stronger phrases (e.g., `latest news`, `current prices`, `recent updates/articles`) while preserving URL/GitHub/website/news/article triggers.

## Validation
- Updated targeted web-search detection tests and quality-path CLI test passed:
  - `tests/agents/api_agents/test_anthropic.py::TestAnthropicWebSearchDetection::test_detects_current_info_keywords`
  - `tests/agents/api_agents/test_gemini.py::TestGeminiWebSearchDetection::test_detects_current_info_keywords`
  - `tests/agents/api_agents/test_openai.py::TestOpenAIWebSearchDetection::test_detects_current_info_keywords`
  - `tests/agents/api_agents/test_gemini.py::TestGeminiWebSearchIndicators::test_case_insensitive_matching`
  - `tests/cli/test_offline_golden_path.py::test_cmd_ask_upgrades_output_to_good`
  - `tests/test_cli_main.py::TestCommandHandlers::test_cmd_ask_runs_debate`

## Interpretation: Is >1800s a bug?
- **Sometimes expected**: long multi-model debates with large context + strict quality upgrades can legitimately run long.
- **Sometimes a bug/config issue**: if runtime is dominated by non-budget-aware repair loops or accidental web-search activation.
- With the two fixes above, time >1800s is now more likely to reflect genuine provider latency/queueing rather than local orchestration bugs.

## Timeout-Doubled Validation Run (3600s)
- Command timeout: `--timeout 3600` (strict wall-clock in CLI + subprocess guard)
- Result: `exit_code=0`, `timed_out=false`
- Duration: `814.92s` (~13m35s)
- Runtime blockers: none
- Agent/debate timeout events: `0`
- Quality upgrade failures: `0`
- Final quality: `verdict=good`, `score=10.0`, `loops=2`, `upgraded=true`

Interpretation:
- With doubled timeout and fixes applied, this workload completed cleanly well below 1800s.
- This indicates previous timeout pressure was driven more by timeout-budgeting/web-trigger behavior than an unavoidable >1800s baseline requirement for this scenario.
