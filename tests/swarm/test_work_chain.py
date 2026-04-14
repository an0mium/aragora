"""Tests for work chain DAG validation and topological ordering."""

from __future__ import annotations

import pytest

from aragora.swarm.work_chain import (
    ChainStatus,
    ChainStep,
    CycleDetectedError,
    ExecutionWave,
    StepStatus,
    WorkChain,
    build_chain_from_work_orders,
)


class TestChainStep:
    def test_step_generates_id_when_empty(self) -> None:
        step = ChainStep(step_id="", work_order_id="wo-1", title="Step 1")
        assert step.step_id  # auto-generated
        assert len(step.step_id) == 12

    def test_step_preserves_explicit_id(self) -> None:
        step = ChainStep(step_id="my-step", work_order_id="wo-1", title="Step 1")
        assert step.step_id == "my-step"

    def test_step_deduplicates_depends_on(self) -> None:
        step = ChainStep(
            step_id="s1",
            work_order_id="wo-1",
            title="Step 1",
            depends_on=["a", "b", "a", "c", "b"],
        )
        assert step.depends_on == ["a", "b", "c"]

    def test_step_strips_empty_depends_on(self) -> None:
        step = ChainStep(
            step_id="s1",
            work_order_id="wo-1",
            title="Step 1",
            depends_on=["a", "", None, "b"],  # type: ignore[list-item]
        )
        assert step.depends_on == ["a", "b"]

    def test_step_to_dict(self) -> None:
        step = ChainStep(
            step_id="s1",
            work_order_id="wo-1",
            title="Step 1",
            depends_on=["s0"],
            file_scope=["foo.py"],
        )
        d = step.to_dict()
        assert d["step_id"] == "s1"
        assert d["depends_on"] == ["s0"]
        assert d["status"] == "pending"


class TestWorkChainValidation:
    def test_linear_chain_validates(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="Step A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="Step B", depends_on=["a"]),
            ChainStep(step_id="c", work_order_id="wo-c", title="Step C", depends_on=["b"]),
        ]
        chain = WorkChain(chain_id="test", title="Linear chain", steps=steps)
        assert chain.topological_order == ["a", "b", "c"]
        assert chain.num_waves == 3

    def test_diamond_chain_validates(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="Root"),
            ChainStep(step_id="b", work_order_id="wo-b", title="Left", depends_on=["a"]),
            ChainStep(step_id="c", work_order_id="wo-c", title="Right", depends_on=["a"]),
            ChainStep(step_id="d", work_order_id="wo-d", title="Join", depends_on=["b", "c"]),
        ]
        chain = WorkChain(chain_id="test", title="Diamond", steps=steps)
        topo = chain.topological_order
        assert topo.index("a") < topo.index("b")
        assert topo.index("a") < topo.index("c")
        assert topo.index("b") < topo.index("d")
        assert topo.index("c") < topo.index("d")

    def test_parallel_roots_form_single_wave(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B"),
            ChainStep(step_id="c", work_order_id="wo-c", title="C"),
        ]
        chain = WorkChain(chain_id="test", title="Parallel", steps=steps)
        assert chain.num_waves == 1
        assert sorted(chain.waves[0].step_ids) == ["a", "b", "c"]

    def test_cycle_detected(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A", depends_on=["c"]),
            ChainStep(step_id="b", work_order_id="wo-b", title="B", depends_on=["a"]),
            ChainStep(step_id="c", work_order_id="wo-c", title="C", depends_on=["b"]),
        ]
        with pytest.raises(CycleDetectedError) as exc_info:
            WorkChain(chain_id="test", title="Cycle", steps=steps)
        assert set(exc_info.value.involved_steps) == {"a", "b", "c"}

    def test_self_cycle_detected(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A", depends_on=["a"]),
        ]
        with pytest.raises(CycleDetectedError):
            WorkChain(chain_id="test", title="Self-cycle", steps=steps)

    def test_dangling_dependency_raises(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A", depends_on=["missing"]),
        ]
        with pytest.raises(ValueError, match="unknown step 'missing'"):
            WorkChain(chain_id="test", title="Dangling", steps=steps)

    def test_empty_chain_validates(self) -> None:
        chain = WorkChain(chain_id="test", title="Empty")
        assert chain.topological_order == []
        assert chain.waves == []
        assert chain.num_waves == 0


class TestExecutionWaves:
    def test_diamond_has_three_waves(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="Root"),
            ChainStep(step_id="b", work_order_id="wo-b", title="Left", depends_on=["a"]),
            ChainStep(step_id="c", work_order_id="wo-c", title="Right", depends_on=["a"]),
            ChainStep(step_id="d", work_order_id="wo-d", title="Join", depends_on=["b", "c"]),
        ]
        chain = WorkChain(chain_id="test", title="Diamond", steps=steps)
        assert chain.num_waves == 3
        assert chain.waves[0].step_ids == ["a"]
        assert sorted(chain.waves[1].step_ids) == ["b", "c"]
        assert chain.waves[2].step_ids == ["d"]

    def test_wave_to_dict(self) -> None:
        wave = ExecutionWave(wave_index=0, step_ids=["a", "b"])
        d = wave.to_dict()
        assert d["wave_index"] == 0
        assert d["step_ids"] == ["a", "b"]


class TestReadySteps:
    def test_all_roots_are_ready_initially(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B"),
            ChainStep(step_id="c", work_order_id="wo-c", title="C", depends_on=["a"]),
        ]
        chain = WorkChain(chain_id="test", title="Ready test", steps=steps)
        ready = chain.ready_steps()
        ready_ids = [step.step_id for step in ready]
        assert sorted(ready_ids) == ["a", "b"]

    def test_completing_predecessor_makes_successor_ready(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B", depends_on=["a"]),
        ]
        chain = WorkChain(chain_id="test", title="Sequence", steps=steps)
        assert len(chain.ready_steps()) == 1
        assert chain.ready_steps()[0].step_id == "a"

        chain.mark_step("a", StepStatus.COMPLETED)
        ready = chain.ready_steps()
        assert len(ready) == 1
        assert ready[0].step_id == "b"


class TestChainStatus:
    def test_all_completed_makes_chain_completed(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B"),
        ]
        chain = WorkChain(chain_id="test", title="Test", steps=steps)
        chain.mark_step("a", StepStatus.COMPLETED)
        chain.mark_step("b", StepStatus.COMPLETED)
        assert chain.status == ChainStatus.COMPLETED

    def test_any_failed_makes_chain_failed(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B"),
        ]
        chain = WorkChain(chain_id="test", title="Test", steps=steps)
        chain.mark_step("a", StepStatus.FAILED)
        assert chain.status == ChainStatus.FAILED

    def test_skipped_counts_as_done(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B"),
        ]
        chain = WorkChain(chain_id="test", title="Test", steps=steps)
        chain.mark_step("a", StepStatus.COMPLETED)
        chain.mark_step("b", StepStatus.SKIPPED)
        assert chain.status == ChainStatus.COMPLETED

    def test_mark_unknown_step_raises(self) -> None:
        chain = WorkChain(
            chain_id="test",
            title="Test",
            steps=[
                ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ],
        )
        with pytest.raises(KeyError, match="Unknown step"):
            chain.mark_step("nonexistent", StepStatus.COMPLETED)


class TestFingerprint:
    def test_same_structure_same_fingerprint(self) -> None:
        def make_chain() -> WorkChain:
            return WorkChain(
                chain_id="different-id",
                title="Test chain",
                steps=[
                    ChainStep(step_id="a", work_order_id="wo-a", title="A"),
                    ChainStep(step_id="b", work_order_id="wo-b", title="B", depends_on=["a"]),
                ],
            )

        assert make_chain().fingerprint() == make_chain().fingerprint()

    def test_different_structure_different_fingerprint(self) -> None:
        chain_a = WorkChain(
            chain_id="test",
            title="Test",
            steps=[
                ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ],
        )
        chain_b = WorkChain(
            chain_id="test",
            title="Test",
            steps=[
                ChainStep(step_id="a", work_order_id="wo-a", title="A"),
                ChainStep(step_id="b", work_order_id="wo-b", title="B"),
            ],
        )
        assert chain_a.fingerprint() != chain_b.fingerprint()


class TestBuildFromWorkOrders:
    def test_builds_chain_from_work_order_dicts(self) -> None:
        work_orders = [
            {"work_order_id": "wo-1", "title": "Create module", "file_scope": ["mod.py"]},
            {
                "work_order_id": "wo-2",
                "title": "Add tests",
                "dependency_ids": ["wo-1"],
                "file_scope": ["test_mod.py"],
            },
        ]
        chain = build_chain_from_work_orders("Test chain", work_orders)
        assert len(chain.steps) == 2
        assert chain.topological_order[0] == "wo-1"
        assert chain.topological_order[1] == "wo-2"

    def test_skips_empty_work_order_ids(self) -> None:
        work_orders = [
            {"work_order_id": "", "title": "Empty"},
            {"work_order_id": "wo-1", "title": "Real"},
        ]
        chain = build_chain_from_work_orders("Test", work_orders)
        assert len(chain.steps) == 1

    def test_ignores_dangling_dependency_ids(self) -> None:
        work_orders = [
            {
                "work_order_id": "wo-1",
                "title": "Step",
                "dependency_ids": ["nonexistent"],
            },
        ]
        chain = build_chain_from_work_orders("Test", work_orders)
        assert len(chain.steps) == 1
        assert chain.steps[0].depends_on == []


class TestComplexDAGPatterns:
    def test_wide_fan_out(self) -> None:
        """1 root with 10 parallel children."""
        children = [
            ChainStep(
                step_id=f"c{i}",
                work_order_id=f"wo-c{i}",
                title=f"Child {i}",
                depends_on=["root"],
            )
            for i in range(10)
        ]
        steps = [
            ChainStep(step_id="root", work_order_id="wo-root", title="Root"),
            *children,
        ]
        chain = WorkChain(chain_id="test", title="Wide fan-out", steps=steps)
        assert chain.num_waves == 2
        assert chain.waves[0].step_ids == ["root"]
        assert sorted(chain.waves[1].step_ids) == [f"c{i}" for i in range(10)]
        assert chain.topological_order[0] == "root"

    def test_deep_chain(self) -> None:
        """10 sequential steps forming a long chain."""
        steps = [
            ChainStep(
                step_id=f"s{i}",
                work_order_id=f"wo-s{i}",
                title=f"Step {i}",
                depends_on=[f"s{i - 1}"] if i > 0 else [],
            )
            for i in range(10)
        ]
        chain = WorkChain(chain_id="test", title="Deep chain", steps=steps)
        assert chain.topological_order == [f"s{i}" for i in range(10)]
        assert chain.num_waves == 10
        for i, wave in enumerate(chain.waves):
            assert wave.step_ids == [f"s{i}"]

    def test_mixed_diamond_multiple_join_points(self) -> None:
        """Diamond DAG with two join points: d joins b+c, g joins e+f."""
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="Root"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B", depends_on=["a"]),
            ChainStep(step_id="c", work_order_id="wo-c", title="C", depends_on=["a"]),
            ChainStep(step_id="d", work_order_id="wo-d", title="Join1", depends_on=["b", "c"]),
            ChainStep(step_id="e", work_order_id="wo-e", title="E", depends_on=["d"]),
            ChainStep(step_id="f", work_order_id="wo-f", title="F", depends_on=["d"]),
            ChainStep(step_id="g", work_order_id="wo-g", title="Join2", depends_on=["e", "f"]),
        ]
        chain = WorkChain(chain_id="test", title="Multi-diamond", steps=steps)
        topo = chain.topological_order
        # Verify ordering constraints
        assert topo.index("a") < topo.index("b")
        assert topo.index("a") < topo.index("c")
        assert topo.index("b") < topo.index("d")
        assert topo.index("c") < topo.index("d")
        assert topo.index("d") < topo.index("e")
        assert topo.index("d") < topo.index("f")
        assert topo.index("e") < topo.index("g")
        assert topo.index("f") < topo.index("g")
        # Waves: a | b,c | d | e,f | g
        assert chain.num_waves == 5
        assert chain.waves[0].step_ids == ["a"]
        assert sorted(chain.waves[1].step_ids) == ["b", "c"]
        assert chain.waves[2].step_ids == ["d"]
        assert sorted(chain.waves[3].step_ids) == ["e", "f"]
        assert chain.waves[4].step_ids == ["g"]

    def test_all_steps_completed_chain_status(self) -> None:
        """Chain with all steps completed should have COMPLETED status."""
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B", depends_on=["a"]),
            ChainStep(step_id="c", work_order_id="wo-c", title="C", depends_on=["a"]),
            ChainStep(step_id="d", work_order_id="wo-d", title="D", depends_on=["b", "c"]),
        ]
        chain = WorkChain(chain_id="test", title="Full completion", steps=steps)
        assert chain.status == ChainStatus.PENDING

        for step_id in chain.topological_order:
            chain.mark_step(step_id, StepStatus.COMPLETED)

        assert chain.status == ChainStatus.COMPLETED
        assert chain.ready_steps() == []


class TestBuildChainCircularDeps:
    def test_self_referencing_dependency_ids_raises(self) -> None:
        """build_chain_from_work_orders passes through self-references which cause a cycle."""
        work_orders = [
            {
                "work_order_id": "wo-1",
                "title": "Step 1",
                "dependency_ids": ["wo-1"],  # self-reference
            },
            {
                "work_order_id": "wo-2",
                "title": "Step 2",
                "dependency_ids": ["wo-1"],
            },
        ]
        with pytest.raises(CycleDetectedError):
            build_chain_from_work_orders("Circular test", work_orders)

    def test_mutual_circular_dependency_ids_raises(self) -> None:
        """Mutual circular deps that survive filtering should raise CycleDetectedError."""
        work_orders = [
            {
                "work_order_id": "wo-1",
                "title": "Step 1",
                "dependency_ids": ["wo-2"],
            },
            {
                "work_order_id": "wo-2",
                "title": "Step 2",
                "dependency_ids": ["wo-1"],
            },
        ]
        with pytest.raises(CycleDetectedError):
            build_chain_from_work_orders("Mutual circular", work_orders)


class TestToDict:
    def test_chain_roundtrip_serialization(self) -> None:
        steps = [
            ChainStep(step_id="a", work_order_id="wo-a", title="A"),
            ChainStep(step_id="b", work_order_id="wo-b", title="B", depends_on=["a"]),
        ]
        chain = WorkChain(chain_id="test-chain", title="Serialize test", steps=steps)
        d = chain.to_dict()
        assert d["chain_id"] == "test-chain"
        assert len(d["steps"]) == 2
        assert d["topological_order"] == ["a", "b"]
        assert len(d["waves"]) == 2
        assert isinstance(d["fingerprint"], str)
