# Ralph Hardening & Decision Integrity Kernel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the Ralph autonomous loop with file-scope enforcement and canonical PR protocol, harden RLM integration, and begin Decision Integrity Kernel unification with provider routing.

**Architecture:** Five initiatives building on the LLM scope classifier (PR #1020). Scope enforcement and PR protocol harden the existing swarm control plane. RLM hardening fills production gaps. Kernel unification and provider routing are the strategic M1 priorities.

**Tech Stack:** Python 3.11, pytest, asyncio, Prometheus metrics, YAML state persistence, gh CLI

---

## Chunk 1: Scope Enforcement Hardening (#840)

**Context:** File-scope enforcement already exists as a 2-layer system (prompt guidance + supervisor validation with LLM adjudication). Issue #840 is about enforcing file-scope *ownership* on agent lanes — making scope constraints mandatory rather than advisory, and improving the LLM adjudication quality.

### Task 1: Make file_scope mandatory for supervised work orders

**Files:**
- Modify: `aragora/swarm/supervisor.py:804-853` (`_build_supervised_work_orders`)
- Modify: `aragora/swarm/spec.py:147-168` (`dispatch_bounds`)
- Test: `tests/swarm/test_file_scope_enforcement.py`

- [x] **Step 1: Write the failing test**

```python
# tests/swarm/test_file_scope_enforcement.py
class TestMandatoryFileScope:
    def test_work_order_without_scope_gets_spec_hints(self):
        """Work orders with empty file_scope should inherit spec hints."""
        spec = SwarmSpec(goal="fix bug", file_scope_hints=["aragora/debate/**"])
        wo = {"title": "fix consensus", "file_scope": []}
        result = _ensure_work_order_scope(wo, spec)
        assert result["file_scope"] == ["aragora/debate/**"]

    def test_work_order_without_scope_or_hints_gets_inferred(self):
        """Work orders with no scope and no hints get scope inferred from task."""
        spec = SwarmSpec(goal="fix debate consensus", file_scope_hints=[])
        wo = {"title": "fix consensus detection", "file_scope": []}
        result = _ensure_work_order_scope(wo, spec)
        assert len(result["file_scope"]) > 0  # LLM or keyword inference
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/swarm/test_file_scope_enforcement.py::TestMandatoryFileScope -v`
Expected: FAIL with `NameError: _ensure_work_order_scope` not defined

- [x] **Step 3: Implement _ensure_work_order_scope in supervisor.py**

Add a method that:
1. If work_order has file_scope → keep it
2. Else if spec has file_scope_hints → copy hints to work_order
3. Else → try LLM `infer_spec_fields()` for scope hints, fall back to keyword heuristics from task title
4. Log warning if scope remains empty (advisory, not blocking — for backward compat)

- [x] **Step 4: Wire into _build_supervised_work_orders post-processing**

After the existing work order construction loop, call `_ensure_work_order_scope(wo, spec)` for each work order.

- [x] **Step 5: Run tests to verify they pass**

Run: `pytest tests/swarm/test_file_scope_enforcement.py -v`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add aragora/swarm/supervisor.py tests/swarm/test_file_scope_enforcement.py
git commit -m "feat(swarm): make file_scope mandatory for supervised work orders (#840)"
```

### Task 2: Improve LLM scope adjudication with richer context

**Files:**
- Modify: `aragora/ralph/llm_classifier.py:185-221` (`adjudicate_scope`)
- Test: `tests/ralph/test_llm_classifier.py`

- [x] **Step 1: Write failing test for improved adjudication**

```python
class TestImprovedScopeAdjudication:
    @pytest.mark.asyncio
    async def test_adjudicate_scope_considers_dependency_files(self):
        """Scope adjudication should recognize dependency-related files as justified."""
        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value='{"justified": ["package.json", "package-lock.json"], "rejected": [], "reasoning": "dependency files"}')
        with patch("aragora.agents.base.create_agent", return_value=mock_agent):
            classifier = LLMBlockerClassifier()
            verdict = await classifier.adjudicate_scope(
                task_description="Update React to v19",
                declared_scope=["src/components/**"],
                changed_paths=["src/components/App.tsx", "package.json", "package-lock.json"],
                violations=[{"path": "package.json"}, {"path": "package-lock.json"}],
            )
            assert "package.json" in verdict.justified_paths
```

- [x] **Step 2: Run test, verify it passes with current implementation**

If it already passes (the LLM mock returns the right answer), add a test for the prompt quality instead — verify the prompt includes dependency-file heuristics.

- [x] **Step 3: Enhance the scope adjudication prompt**

Update `adjudicate_scope()` to include in the prompt:
- Common justified patterns: dependency files (`package.json`, `requirements.txt`, `Cargo.lock`), test files, `__init__.py`, config files
- Task context: what the work order was trying to achieve
- Repository structure hints (top-level dirs) for better reasoning

- [x] **Step 4: Run all scope tests**

Run: `pytest tests/ralph/test_llm_classifier.py tests/swarm/test_file_scope_enforcement.py -v`
Expected: All pass

- [x] **Step 5: Commit**

```bash
git add aragora/ralph/llm_classifier.py tests/ralph/test_llm_classifier.py
git commit -m "feat(ralph): improve LLM scope adjudication with dependency-aware prompts (#840)"
```

---

## Chunk 2: Canonical PR Protocol (#841)

**Context:** Ralph tracks one repair PR via `active_merge_target` dict in `SupervisorState`. Swarm supervisor doesn't track PR state at all. When multiple agents create PRs for the same branch, there's no coordination.

### Task 3: Create PullRequestRegistry data model

**Files:**
- Create: `aragora/swarm/pr_registry.py`
- Test: `tests/swarm/test_pr_registry.py`

- [x] **Step 1: Write failing tests for PR registry**

```python
class TestPullRequestRegistry:
    def test_register_pr(self, tmp_path):
        registry = PullRequestRegistry(state_dir=tmp_path)
        registry.register("fix/scope", "https://github.com/org/repo/pull/42", creator="worker-1")
        entry = registry.get("fix/scope")
        assert entry is not None
        assert entry["pr_url"] == "https://github.com/org/repo/pull/42"

    def test_supersede_pr(self, tmp_path):
        registry = PullRequestRegistry(state_dir=tmp_path)
        registry.register("fix/scope", "https://github.com/org/repo/pull/42", creator="worker-1")
        registry.supersede("fix/scope", "https://github.com/org/repo/pull/43", reason="newer implementation")
        entry = registry.get("fix/scope")
        assert entry["pr_url"] == "https://github.com/org/repo/pull/43"
        assert entry["superseded"] == [{"pr_url": "https://github.com/org/repo/pull/42", "reason": "newer implementation"}]

    def test_list_active_prs(self, tmp_path):
        registry = PullRequestRegistry(state_dir=tmp_path)
        registry.register("fix/a", "url-1", creator="w1")
        registry.register("fix/b", "url-2", creator="w2")
        registry.close("fix/a", outcome="merged")
        active = registry.list_active()
        assert len(active) == 1
        assert active[0]["branch"] == "fix/b"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/swarm/test_pr_registry.py -v`
Expected: FAIL with `ImportError`

- [x] **Step 3: Implement PullRequestRegistry**

```python
# aragora/swarm/pr_registry.py
@dataclass
class PREntry:
    branch: str
    pr_url: str
    creator: str
    created_at: str
    status: str  # "active", "merged", "closed", "superseded"
    superseded: list[dict]
    gate_snapshot: dict | None

class PullRequestRegistry:
    def __init__(self, state_dir: Path | None = None): ...
    def register(self, branch, pr_url, creator, **kwargs): ...
    def supersede(self, branch, new_pr_url, reason=""): ...
    def close(self, branch, outcome): ...
    def get(self, branch) -> dict | None: ...
    def list_active(self) -> list[dict]: ...
```

YAML-backed persistence to `state_dir/pr_registry.yaml`.

- [x] **Step 4: Run tests**

Run: `pytest tests/swarm/test_pr_registry.py -v`
Expected: All pass

- [x] **Step 5: Commit**

```bash
git add aragora/swarm/pr_registry.py tests/swarm/test_pr_registry.py
git commit -m "feat(swarm): add PullRequestRegistry for canonical PR tracking (#841)"
```

### Task 4: Wire PR registry into swarm supervisor

**Files:**
- Modify: `aragora/swarm/supervisor.py` (result application, merge gate)
- Modify: `aragora/ralph/supervisor.py` (repair PR creation)
- Test: `tests/swarm/test_supervisor.py`

- [x] **Step 1: Write failing test for PR registration on worker completion**

```python
def test_worker_pr_registered_on_completion(supervisor_fixtures):
    """When a worker creates a PR, it should be registered in the PR registry."""
    sup, store, state = supervisor_fixtures
    # ... setup worker with PR result
    # After _apply_worker_result:
    registry = sup._get_pr_registry()
    entry = registry.get("fix/scope")
    assert entry is not None
```

- [x] **Step 2: Implement PR registration hooks**

In `_apply_worker_result()`, after successful completion:
- If worker result includes PR URL → register in `PullRequestRegistry`
- If branch already has an active PR → supersede old one

In Ralph's `_create_repair_pr()`:
- Register new PR in registry
- If previous repair PR exists → supersede it with reason

- [x] **Step 3: Run tests**

Run: `pytest tests/swarm/test_supervisor.py tests/swarm/test_pr_registry.py -v`

- [x] **Step 4: Commit**

```bash
git add aragora/swarm/supervisor.py aragora/ralph/supervisor.py tests/swarm/test_supervisor.py
git commit -m "feat(swarm): wire PR registry into supervisor result flow (#841)"
```

---

## Chunk 3: RLM Integration Hardening (#1008)

**Context:** RLM has 184 exports and works for REPL-based context access. Missing: streaming export, circuit breaker, fallback chain, timeout enforcement.

### Task 5: Export streaming RLM adapter and add fallback chain

**Files:**
- Modify: `aragora/rlm/__init__.py` (add streaming exports)
- Modify: `aragora/rlm/factory.py` (add fallback chain)
- Test: `tests/rlm/test_rlm_hardening.py`

- [x] **Step 1: Write tests for streaming export and fallback**

```python
class TestRLMHardening:
    def test_streaming_adapter_importable(self):
        from aragora.rlm import StreamingRLMAdapter
        assert StreamingRLMAdapter is not None

    def test_fallback_chain_on_unavailable(self):
        """When real RLM unavailable, factory should return compressor fallback."""
        from aragora.rlm import get_rlm
        rlm = get_rlm(fallback=True)
        assert rlm is not None

    def test_timeout_enforcement(self):
        from aragora.rlm import RLMTimeoutError
        assert RLMTimeoutError is not None
```

- [x] **Step 2: Export missing types from __init__.py**

Add to `aragora/rlm/__init__.py`:
- `StreamingRLMAdapter` from `streaming.py`
- `RLMCircuitBreaker` (new or from existing)
- `RLMTimeoutError` (ensure exported)

- [x] **Step 3: Add fallback chain to factory**

In `aragora/rlm/factory.py`, update `get_rlm()`:
- If `fallback=True` and real RLM unavailable → return `HierarchicalCompressor` wrapper
- Log warning when falling back

- [x] **Step 4: Run tests**

Run: `pytest tests/rlm/test_rlm_hardening.py -v`

- [x] **Step 5: Commit**

```bash
git add aragora/rlm/__init__.py aragora/rlm/factory.py tests/rlm/test_rlm_hardening.py
git commit -m "feat(rlm): export streaming adapter and add fallback chain (#1008)"
```

### Task 6: Add circuit breaker and timeout to RLM queries

**Files:**
- Create: `aragora/rlm/resilience.py`
- Modify: `aragora/rlm/factory.py`
- Test: `tests/rlm/test_rlm_resilience.py`

- [x] **Step 1: Write failing tests**

```python
class TestRLMResilience:
    def test_circuit_breaker_opens_after_failures(self):
        cb = RLMCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open

    def test_timeout_decorator_raises(self):
        @rlm_timeout(seconds=0.01)
        def slow_fn():
            import time; time.sleep(1)
        with pytest.raises(RLMTimeoutError):
            slow_fn()
```

- [x] **Step 2: Implement resilience module**

```python
# aragora/rlm/resilience.py
class RLMCircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60): ...
    def record_failure(self): ...
    def record_success(self): ...
    @property
    def is_open(self) -> bool: ...

def rlm_timeout(seconds: float):
    """Decorator that raises RLMTimeoutError after seconds."""
    ...
```

- [x] **Step 3: Wire into factory**

`get_rlm()` should use circuit breaker state to short-circuit when RLM is consistently failing.

- [x] **Step 4: Run tests**

Run: `pytest tests/rlm/ -v`

- [x] **Step 5: Commit**

```bash
git add aragora/rlm/resilience.py aragora/rlm/factory.py tests/rlm/test_rlm_resilience.py
git commit -m "feat(rlm): add circuit breaker and timeout enforcement (#1008)"
```

---

## Chunk 4: Decision Integrity Kernel — Pre-Debate Context Bridge (#811)

**Context:** Decision Integrity currently captures context AFTER debate. Pipeline runs stages independently. The kernel needs to be unified so context flows seamlessly: pre-load relevant memory/knowledge BEFORE debate, and feed debate outcomes into the next pipeline stage.

### Task 7: Add pre-debate context preloader

**Files:**
- Modify: `aragora/pipeline/decision_integrity.py`
- Test: `tests/pipeline/test_decision_integrity.py`

- [x] **Step 1: Write failing test**

```python
class TestPreDebateContextPreloader:
    def test_preload_returns_context_dict(self):
        preloader = DecisionContextPreloader()
        ctx = preloader.preload(task="Design a rate limiter", domain="software")
        assert "precedents" in ctx
        assert "relevant_knowledge" in ctx
        assert "agent_calibration" in ctx
```

- [x] **Step 2: Implement DecisionContextPreloader**

```python
class DecisionContextPreloader:
    """Loads relevant context BEFORE debate starts."""
    def preload(self, task: str, domain: str | None = None) -> dict:
        # 1. Query KnowledgeMound for precedents
        # 2. Query ContinuumMemory for relevant cross-debate patterns
        # 3. Load agent calibration scores for domain
        # 4. Return structured context dict
```

- [x] **Step 3: Wire into Arena as optional pre-debate hook**

Add `context_preloader` parameter to Arena/ArenaConfig. If set, call before first round.

- [x] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_decision_integrity.py -v`

- [x] **Step 5: Commit**

```bash
git add aragora/pipeline/decision_integrity.py tests/pipeline/test_decision_integrity.py
git commit -m "feat(pipeline): add pre-debate context preloader for kernel unification (#811)"
```

### Task 8: Add debate outcome → pipeline stage bridge

**Files:**
- Create: `aragora/pipeline/debate_bridge.py`
- Test: `tests/pipeline/test_debate_bridge.py`

- [x] **Step 1: Write failing test**

```python
class TestDebateOutcomeBridge:
    def test_debate_result_populates_workflow_stage(self):
        bridge = DebateOutcomeBridge()
        debate_result = {"consensus": True, "winner": "agent-1", "dissent": ["agent-2"], ...}
        workflow_hints = bridge.extract_workflow_hints(debate_result)
        assert "recommended_agents" in workflow_hints
        assert "risk_factors" in workflow_hints
        assert "dissent_summary" in workflow_hints
```

- [x] **Step 2: Implement DebateOutcomeBridge**

Extracts from debate result:
- Recommended agents for execution (based on debate performance)
- Risk factors from dissent
- Acceptance criteria from consensus claims
- Cost estimate from provider routing data

- [x] **Step 3: Wire into IdeaToExecutionPipeline stage transitions**

After Stage 2 (Goals debate) → auto-populate Stage 3 (Workflow) with debate-derived hints.

- [x] **Step 4: Run tests**

Run: `pytest tests/pipeline/test_debate_bridge.py -v`

- [x] **Step 5: Commit**

```bash
git add aragora/pipeline/debate_bridge.py tests/pipeline/test_debate_bridge.py
git commit -m "feat(pipeline): add debate outcome to pipeline stage bridge (#811)"
```

---

## Chunk 5: Provider Routing Integration (#813)

**Context:** ProviderRouter selects provider names based on cost/quality metrics. TeamSelector selects agent objects based on ELO/calibration. They're disconnected. Integration means ProviderRouter output informs TeamSelector input, and debate outcomes feed back into provider metrics.

### Task 9: Bridge ProviderRouter → TeamSelector

**Files:**
- Modify: `aragora/routing/provider_router.py`
- Modify: `aragora/debate/team_selector.py`
- Test: `tests/debate/test_provider_routing_integration.py`

- [x] **Step 1: Write failing test**

```python
class TestProviderRoutingIntegration:
    def test_team_selector_accepts_provider_hints(self):
        """TeamSelector should use provider_hints to weight agent selection."""
        selector = TeamSelector(agents=[...])
        hints = {"claude-sonnet-4": 0.9, "gpt-5.5": 0.7, "deepseek-r1": 0.5}
        team = selector.select(task="code review", num_agents=3, provider_hints=hints)
        # Agents backed by higher-scoring providers should be preferred
        assert len(team) == 3
```

- [x] **Step 2: Add provider_hints parameter to TeamSelector.select()**

New optional parameter that applies a multiplicative bonus to agents whose backing provider appears in the hints dict. This augments existing ELO/calibration scoring without replacing it.

- [x] **Step 3: Add post-debate metric recording**

After Arena.run() completes, call:
```python
for agent in debate.agents:
    provider_router.record_outcome(agent.provider, quality=score, cost=cost)
```

- [x] **Step 4: Run tests**

Run: `pytest tests/debate/test_provider_routing_integration.py -v`

- [x] **Step 5: Commit**

```bash
git add aragora/routing/provider_router.py aragora/debate/team_selector.py tests/debate/test_provider_routing_integration.py
git commit -m "feat(routing): bridge ProviderRouter into TeamSelector for cost-aware agent selection (#813)"
```

### Task 10: Add budget-aware debate configuration

**Files:**
- Modify: `aragora/debate/arena_config.py`
- Modify: `aragora/routing/provider_router.py`
- Test: `tests/debate/test_budget_aware_debate.py`

- [x] **Step 1: Write failing test**

```python
class TestBudgetAwareDebate:
    def test_arena_config_accepts_provider_budget(self):
        config = ArenaConfig(provider_budget=5.0)
        assert config.provider_budget == 5.0

    def test_provider_router_filters_by_budget(self):
        router = ProviderRouter()
        providers = router.select_providers_for_debate(
            num_agents=3, budget=2.0
        )
        # All selected providers should have per-debate cost <= budget/num_agents
        for p in providers:
            assert p.estimated_cost <= 2.0 / 3
```

- [x] **Step 2: Add provider_budget to ArenaConfig**

- [x] **Step 3: Wire budget into Arena startup**

Before debate, if `provider_budget` is set:
1. Call `ProviderRouter.select_providers_for_debate(budget=config.provider_budget)`
2. Pass result as `provider_hints` to `TeamSelector.select()`

- [x] **Step 4: Run tests**

Run: `pytest tests/debate/test_budget_aware_debate.py -v`

- [x] **Step 5: Commit**

```bash
git add aragora/debate/arena_config.py aragora/routing/provider_router.py tests/debate/test_budget_aware_debate.py
git commit -m "feat(debate): add budget-aware provider selection in Arena (#813)"
```

---

## Execution Sequence

| Order | Task | Issue | Est. | Dependencies |
|-------|------|-------|------|--------------|
| 1 | Make file_scope mandatory | #840 | Small | None |
| 2 | Improve LLM scope adjudication | #840 | Small | Task 1 |
| 3 | Create PullRequestRegistry | #841 | Medium | None |
| 4 | Wire PR registry into supervisor | #841 | Medium | Task 3 |
| 5 | RLM streaming + fallback exports | #1008 | Small | None |
| 6 | RLM circuit breaker + timeout | #1008 | Small | Task 5 |
| 7 | Pre-debate context preloader | #811 | Medium | None |
| 8 | Debate outcome → pipeline bridge | #811 | Medium | Task 7 |
| 9 | ProviderRouter → TeamSelector | #813 | Medium | None |
| 10 | Budget-aware debate config | #813 | Small | Task 9 |

**Parallelizable groups:**
- Tasks 1-2 (#840) can run in parallel with Tasks 3-4 (#841) and Tasks 5-6 (#1008)
- Tasks 7-8 (#811) can run in parallel with Tasks 9-10 (#813)
- Group 2 depends on Group 1 completing (kernel work builds on hardened swarm)
