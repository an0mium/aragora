"""Tests for aragora.nomic.work_mix (epic #9039, issue #9040)."""

from __future__ import annotations

from aragora.nomic.work_mix import (
    BudgetVerdict,
    MixBudget,
    WorkClass,
    classify_path,
    classify_paths,
    compute_mix,
    evaluate_budget,
)


class TestClassifyPath:
    def test_product_core_runtime(self):
        assert classify_path("aragora/debate/orchestrator.py") is WorkClass.PRODUCT_CORE
        assert classify_path("aragora/gauntlet/receipts.py") is WorkClass.PRODUCT_CORE
        assert classify_path("sdk/python/client.py") is WorkClass.PRODUCT_CORE

    def test_substrate_loop_machinery(self):
        assert classify_path("scripts/merge_executor.py") is WorkClass.SUBSTRATE
        assert classify_path("aragora/swarm/boss_loop.py") is WorkClass.SUBSTRATE
        assert classify_path("aragora/nomic/meta_planner.py") is WorkClass.SUBSTRATE
        assert classify_path(".github/workflows/ci.yml") is WorkClass.SUBSTRATE
        assert classify_path("docs/plans/2026-07-08-plan.md") is WorkClass.SUBSTRATE
        assert classify_path("docs/runbooks/main-red.md") is WorkClass.SUBSTRATE

    def test_docs_site_is_product_proof(self):
        # published user-facing docs are product output (#9047 openai [P2])
        assert classify_path("docs-site/docs/api/reference.md") is WorkClass.PRODUCT_PROOF

    def test_product_proof_artifacts(self):
        assert classify_path("docs/artifacts/dogfood.md") is WorkClass.PRODUCT_PROOF
        assert classify_path("docs/specs/OPEN_DECISION_RECEIPT.md") is WorkClass.PRODUCT_PROOF
        assert classify_path("aragora-verify/verify.py") is WorkClass.PRODUCT_PROOF
        assert classify_path("benchmarks/ten_domain.py") is WorkClass.PRODUCT_PROOF

    def test_maintenance(self):
        assert classify_path("tests/nomic/test_x.py") is WorkClass.MAINTENANCE
        assert classify_path("pyproject.toml") is WorkClass.MAINTENANCE
        assert classify_path("docs/README_misc.md") is WorkClass.MAINTENANCE
        assert classify_path("totally/unknown/file.txt") is WorkClass.MAINTENANCE

    def test_longest_prefix_wins(self):
        # docs/ is maintenance but docs/specs/ is product-proof
        assert classify_path("docs/specs/x.md") is WorkClass.PRODUCT_PROOF
        # aragora/ is product-core but aragora/swarm/ is substrate
        assert classify_path("aragora/swarm/x.py") is WorkClass.SUBSTRATE


class TestClassifyPaths:
    def test_pure_product_pr(self):
        work = classify_paths(["aragora/debate/a.py", "aragora/server/b.py"], identifier="1")
        assert work.work_class is WorkClass.PRODUCT_CORE
        assert not work.exempt

    def test_substrate_half_or_more_is_substrate(self):
        # 2 of 4 paths substrate -> substrate wins (anti-laundering)
        work = classify_paths(
            [
                "scripts/settle_pr.py",
                "aragora/swarm/gate.py",
                "aragora/debate/a.py",
                "aragora/server/b.py",
            ]
        )
        assert work.work_class is WorkClass.SUBSTRATE

    def test_minority_substrate_does_not_dominate(self):
        work = classify_paths(
            [
                "scripts/settle_pr.py",
                "aragora/debate/a.py",
                "aragora/server/b.py",
                "aragora/gauntlet/c.py",
            ]
        )
        assert work.work_class is WorkClass.PRODUCT_CORE

    def test_exempt_by_label(self):
        work = classify_paths(["aragora/auth/x.py"], labels=["Security"])
        assert work.exempt

    def test_exempt_by_title_prefix(self):
        assert classify_paths(["scripts/x.py"], title="Revert broken gate").exempt
        assert classify_paths(["tests/a.py"], title="fix(main-red): shard").exempt
        assert not classify_paths(["scripts/x.py"], title="feat: new gate").exempt

    def test_security_title_alone_does_not_exempt(self):
        # Free-text titles must not confer security exemption — that requires
        # a label (gaming vector: prefix a substrate PR with "security:").
        assert not classify_paths(["scripts/x.py"], title="security: harden gate").exempt
        assert not classify_paths(["scripts/x.py"], title="fix(security): races").exempt
        assert classify_paths(["scripts/x.py"], labels=["security"]).exempt

    def test_line_weights_defeat_trivial_padding(self):
        # 4 substantive substrate files vs 5 one-line product edits: by file
        # count product would win (5 > 4, substrate 44% < 50%); by line
        # weight the substrate work dominates and the PR stays substrate.
        paths = [f"scripts/gate_{i}.py" for i in range(4)] + [
            f"aragora/debate/pad_{i}.py" for i in range(5)
        ]
        weights = dict.fromkeys(paths[:4], 200) | dict.fromkeys(paths[4:], 1)
        assert classify_paths(paths, weights=weights).work_class is WorkClass.SUBSTRATE
        # Without weights, the padding attack still flips it — documents why
        # the snapshot feeder must pass line weights.
        assert classify_paths(paths).work_class is WorkClass.PRODUCT_CORE

    def test_zero_weight_paths_count_as_one(self):
        work = classify_paths(["aragora/debate/a.py"], weights={"aragora/debate/a.py": 0})
        assert work.work_class is WorkClass.PRODUCT_CORE

    def test_empty_paths_maintenance(self):
        work = classify_paths([])
        assert work.work_class is WorkClass.MAINTENANCE
        assert work.rationale == "no changed paths"

    def test_llm_classifier_of_record_records_disagreement(self):
        work = classify_paths(
            ["scripts/x.py"],
            llm_classifier=lambda paths, title: WorkClass.PRODUCT_PROOF,
        )
        assert work.work_class is WorkClass.PRODUCT_PROOF
        assert work.cross_check_disagreement is not None
        assert "substrate" in work.cross_check_disagreement

    def test_llm_classifier_none_falls_back_to_baseline(self):
        work = classify_paths(["scripts/x.py"], llm_classifier=lambda paths, title: None)
        assert work.work_class is WorkClass.SUBSTRATE
        assert work.cross_check_disagreement is None


class TestMixAndBudget:
    @staticmethod
    def _work(cls: WorkClass, exempt: bool = False):
        return classify_paths(
            {
                WorkClass.PRODUCT_CORE: ["aragora/debate/a.py"],
                WorkClass.PRODUCT_PROOF: ["docs/artifacts/a.md"],
                WorkClass.SUBSTRATE: ["scripts/a.py"],
                WorkClass.MAINTENANCE: ["tests/a.py"],
            }[cls],
            labels=["security"] if exempt else [],
        )

    def test_compute_mix_shares(self):
        works = [self._work(WorkClass.PRODUCT_CORE)] * 3 + [self._work(WorkClass.SUBSTRATE)]
        mix = compute_mix(works)
        assert mix.total == 4
        assert mix.product_share == 0.75
        assert mix.substrate_share == 0.25

    def test_exempt_excluded(self):
        works = [self._work(WorkClass.PRODUCT_CORE), self._work(WorkClass.SUBSTRATE, exempt=True)]
        mix = compute_mix(works)
        assert mix.total == 1
        assert mix.substrate_share == 0.0

    def test_budget_ok(self):
        works = [self._work(WorkClass.PRODUCT_CORE)] * 3 + [self._work(WorkClass.SUBSTRATE)]
        verdict = evaluate_budget(compute_mix(works))
        assert isinstance(verdict, BudgetVerdict)
        assert verdict.ok
        assert not verdict.freeze_recommended

    def test_substrate_breach_recommends_freeze(self):
        works = [self._work(WorkClass.SUBSTRATE)] * 2 + [self._work(WorkClass.PRODUCT_CORE)]
        verdict = evaluate_budget(compute_mix(works))
        assert verdict.substrate_breach
        assert verdict.freeze_recommended
        assert any("substrate share" in r for r in verdict.reasons)

    def test_product_shortfall_alone_no_freeze(self):
        works = [self._work(WorkClass.MAINTENANCE)] * 3 + [self._work(WorkClass.PRODUCT_CORE)]
        verdict = evaluate_budget(compute_mix(works))
        assert verdict.product_shortfall
        assert not verdict.substrate_breach
        assert not verdict.freeze_recommended
        assert not verdict.ok

    def test_empty_window_is_ok(self):
        verdict = evaluate_budget(compute_mix([]))
        assert verdict.ok
        assert not verdict.freeze_recommended

    def test_custom_budget(self):
        works = [self._work(WorkClass.SUBSTRATE)] + [self._work(WorkClass.PRODUCT_CORE)] * 9
        verdict = evaluate_budget(
            compute_mix(works), MixBudget(product_min=0.95, substrate_max=0.05)
        )
        assert verdict.substrate_breach
        assert verdict.product_shortfall
