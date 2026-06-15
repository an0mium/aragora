# Hardening Burn-Down Report (2026-02-12)

Scope: checklist item 2 (CI stabilization lanes) from `docs/status/HARDENING_SPRINT_2W_CHECKLIST.md`.

## Environment Check

- Python runtime: `pyenv` -> `3.11.11` (repo `.python-version` honored)
- Command path verified via `python --version`, `pyenv which python`

## Targeted Lane Results (Green)

Executed:

```bash
python -m pytest tests/reasoning/test_claim_check.py -q
python -m pytest tests/verification/test_hilbert.py -q
python -m pytest tests/debate/test_stability_detector.py -q
python -m pytest tests/knowledge/test_lara_router.py -q
python -m pytest tests/scripts/test_nomic_sica.py tests/nomic/test_handlers.py -q
```

Result:
- `tests/reasoning/test_claim_check.py`: 3 passed
- `tests/verification/test_hilbert.py`: 1 passed
- `tests/debate/test_stability_detector.py`: 4 passed
- `tests/knowledge/test_lara_router.py`: 5 passed
- `tests/scripts/test_nomic_sica.py tests/nomic/test_handlers.py`: 3 passed (2 deprecation warnings)

## Broader Lane Snapshot

### `tests/knowledge -q`

Result:
- 15 failed, 5736 passed, 25 skipped, 1 xfailed, 4 rerun

Primary failure clusters:
1. External embedding dependency instability
- `Gemini Embedding` API key invalid/expired (`API_KEY_INVALID`) causing downstream retrieval assertions to fail.
- Affects multiple `KnowledgeMound` integration/facade tests.

2. API call-shape mismatch
- `tests/knowledge/test_mound_ops.py::TestKnowledgeMoundOperations::test_fetch_knowledge_context_success`
- Expected `query_semantic(query=...)`, actual positional call.

3. Type robustness issue in unified store sorting
- `TypeError: '<' not supported between instances of 'str' and 'float'`
- `aragora/knowledge/unified/unified_store.py` sorting on `importance` with mixed types.

### `tests/nomic -q`

Result:
- 27 failed, 2590 passed, 1 error, 14 rerun

Primary failure clusters:
1. Commit phase behavior/API drift
- multiple failures in `tests/nomic/phases/test_commit_phase.py`
- includes approval gate behavior and `ApprovalRequired` typing mismatch.

2. Scope limiter correctness
- protected-file detection not enforcing `is_implementable=False`
- zero-division at `max_files=0` in `scope_limiter.py`.

3. Context phase expectations drift
- prompt/content shape changed vs tests (`_build_explore_prompt`, context output composition).

4. TestFixer API and serialization regressions
- `TestFixerOrchestrator.__init__` missing `event_emitter` kwarg expected by tests.
- JSON serialization failing on `AsyncMock` in runner artifacts (`pid`/mock objects).

### `tests/debate --maxfail=1`

Initial blocker fixed:
- `TestGatherClaudeWebSearch::test_returns_none_for_empty_result`

Second blocker fixed:
- `TestCompressWithRlm::test_returns_content_under_threshold`

Current next blocker:
- `tests/debate/test_context_gatherer.py::TestGetContinuumContext::test_includes_glacial_insights`
- Status at failure point: 1 failed, 2954 passed, 2 rerun.

## Fixes Applied During Burn-Down

1. `context_gatherer` compression dispatch bug
- File: `aragora/debate/context_gatherer/sources.py`
- Fix: `SourceGatheringMixin._compress_with_rlm()` now delegates to `CompressionMixin` via `super()` instead of shadowing with a no-op stub.
- Impact: restores expected RLM compression behavior in tests.

2. Backward-compatibility patch target support
- Ensured `aragora.debate.context_gatherer` exposes `asyncio` where legacy tests patch `asyncio.wait_for`.

## Next 3 Fixes (Priority Queue)

1. Debate lane: glacial insights inclusion path
- Fix `ContextGatherer.get_continuum_context()` to include "Long-term patterns" section when glacial insights exist.

2. Knowledge lane: decouple tests from live Gemini embedding dependency
- Inject deterministic local/mock embedding provider in test config defaults.
- Ensure store/get path does not fail hard when embedding backend is unavailable.

3. Nomic lane: testfixer runner JSON serialization hardening
- sanitize mock/non-serializable process metadata before writing run artifacts.

---

## Update (Later 2026-02-12)

### Regression clusters closed

Fixed and revalidated:
- `aragora/debate/convergence/cache.py`
  - max-cache eviction now honors patched limits in tests.
- `aragora/nomic/gates.py`
  - `ApprovalRequired` accepts legacy string gate types.
- `aragora/nomic/phases/commit.py`
  - gate fallback/decline compatibility and robust short-hash parsing.
  - runtime `NOMIC_AUTO_COMMIT` env check.
- `aragora/nomic/phases/scope_limiter.py`
  - protected-file detection for non-`.py` files.
  - no override of protected-file failures.
  - zero-division fix for `max_files=0`.
- `aragora/nomic/meta_planner.py`
  - `Track.SECURITY` present as required.

### Lane validation

Executed:

```bash
python -m pytest --lf tests/debate tests/knowledge tests/nomic -q -p no:randomly --tb=short
python -m pytest tests/scripts/test_nomic_sica.py \
  tests/nomic/test_handlers.py \
  tests/nomic/test_sica_settings.py \
  tests/reasoning/test_claim_check.py \
  tests/verification/test_hilbert.py \
  tests/debate/test_stability_detector.py \
  tests/knowledge/test_lara_router.py \
  tests/nomic/phases/test_commit_phase.py \
  tests/nomic/phases/test_scope_limiter.py \
  tests/nomic/test_meta_planner.py \
  tests/debate/test_convergence_cache.py \
  -q -p no:randomly --tb=short
```

Results:
- `--lf` lane: `12 passed, 0 failed`
- hardening suite: `189 passed, 0 failed`

### Bench baselines captured

Executed:

```bash
python scripts/benchmarks/bench_lara_routing.py
python scripts/benchmarks/bench_adaptive_stopping.py
```

Results:
- LaRA routing: mean `0.0073 ms`, median `0.0038 ms`, distribution `graph=66 rlm=73 semantic=60 keyword=1`.
- Adaptive stopping: mean stability `0.3367`, mean latency `0.0016 ms`.

### Rollout operations guide

Added staged rollout runbook:
- `docs/status/SICA_GRAPHRAG_ROLLOUT_RUNBOOK.md`
- linked from `docs/status/NEXT_STEPS.md`
