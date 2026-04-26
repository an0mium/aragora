"""Tests for :mod:`aragora.pdb.worker`.

The worker's contract is: dedupe by ``(repo, pr_number, head_sha)``,
cap concurrency via a semaphore, transition storage records
``queued → running → ready | failed`` in order, and treat cancellation
as best-effort + failed-record-recording.

We don't exercise the real executor in these tests. ``run_protocol_b``
is monkey-patched to a controllable stub — the PR 2 test suite already
exercises the real pipeline.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.pdb import budget as budget_mod
from aragora.pdb import storage
from aragora.pdb import worker as worker_mod
from aragora.pdb.brief_state import BriefLifecycleState
from aragora.pdb.protocol import (
    PDBExecutionInput,
    PDBExecutionResult,
    PDBExecutionStatus,
)
from aragora.pdb.worker import (
    AlreadyRunningError,
    BriefGenerationWorker,
    JobKey,
    JobRequest,
    get_worker,
    reset_worker,
    set_worker,
)
from aragora.review.policy import ReviewPolicy
from aragora.review.provider_slots import ProviderSlotAvailabilitySummary
from aragora.swarm.pr_review_protocol import PRReviewBinding, PRReviewProtocolPacket


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_storage_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh_worker_singleton():
    reset_worker()
    yield
    reset_worker()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binding() -> PRReviewBinding:
    return PRReviewBinding(
        repo="synaptent/aragora",
        pr_number=4242,
        base_sha="base0000",
        head_sha="deadbeefcafe",
    )


def _make_input(pr_number: int = 4242, head_sha: str = "deadbeefcafe") -> PDBExecutionInput:
    binding = PRReviewBinding(
        repo="synaptent/aragora",
        pr_number=pr_number,
        base_sha="base0000",
        head_sha=head_sha,
    )
    packet = PRReviewProtocolPacket(
        protocol_version="pr_review_protocol.v1",
        status="metadata_heuristic",
        binding=binding,
        review_roles=[],
        provider_slots=[],
        availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
        recommendation_class="needs_human_attention",
        recommendation_reason="stub",
        confidence=0.5,
        confidence_basis="metadata_heuristic",
        dissent_summary="",
        dissenting_views=[],
        validation_summary={},
        top_findings=[],
        cost_estimate={},
    )
    return PDBExecutionInput(
        binding=binding,
        packet=packet,
        packet_sha="",
        pr_title="Tighten rate limiter",
        pr_body="",
        labels=(),
        changed_files=(),
        diff_excerpt="",
        validation_summary={},
        panel_id="p",
        policy=ReviewPolicy(),
    )


def _make_key(pr_number: int = 4242, head_sha: str = "deadbeefcafe") -> JobKey:
    return JobKey(repo="synaptent/aragora", pr_number=pr_number, head_sha=head_sha)


def _make_success_result(
    pr_number: int = 4242, head_sha: str = "deadbeefcafe"
) -> PDBExecutionResult:
    # Build a minimal success result with a brief payload whose
    # ``to_dict`` shape survives a storage round-trip.
    from aragora.pdb.budget import PDBBudgetDecision, PDBBudgetStatus

    class _FakeBrief:
        def to_dict(self) -> dict[str, Any]:
            return {
                "pr_number": pr_number,
                "head_sha": head_sha,
                "verdict": "approve_candidate",
                "confidence": 4,
            }

    packet = PRReviewProtocolPacket(
        protocol_version="pr_review_protocol.v1",
        status="panel_executed",
        binding=PRReviewBinding(
            repo="synaptent/aragora",
            pr_number=pr_number,
            base_sha="base0000",
            head_sha=head_sha,
        ),
        review_roles=[],
        provider_slots=[],
        availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
        recommendation_class="approve_candidate",
        recommendation_reason="stub",
        confidence=0.8,
        confidence_basis="panel_executed",
        dissent_summary="",
        dissenting_views=[],
        validation_summary={},
        top_findings=[],
        cost_estimate={},
    )
    return PDBExecutionResult(
        status=PDBExecutionStatus.SUCCESS,
        packet=packet,
        brief=_FakeBrief(),
        budget_decision=PDBBudgetDecision(
            status=PDBBudgetStatus.ALLOWED,
            total_estimated_usd=0.0,
            per_brief_cap_usd=10.0,
            per_day_cap_usd=100.0,
            per_day_spent_before_usd=0.0,
            per_day_remaining_before_usd=100.0,
            funded_slots=(),
            dropped_slots=(),
            slot_estimates_usd={},
            synthesis_estimate_usd=0.0,
            reason="allowed",
            binding_cap=None,
        ),
        active_roster=(),
        missing_slots=(),
        degrade_reasons=(),
        failure_reason=None,
        findings_by_slot={},
        critiques_by_slot={},
        synthesis=None,
        resolutions=(),
        availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
        actual_cost_usd=1.23,
    )


class _DummyInvoker:
    """Placeholder — run_protocol_b is monkey-patched, so this is unused."""


def _seed_queued(pr_number: int, head_sha: str) -> None:
    storage.queue_generation(pr_number, head_sha, ("slot_a", "slot_b"))


# ---------------------------------------------------------------------------
# Construction / startup
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_rejects_nonpositive_concurrency(self):
        with pytest.raises(ValueError):
            BriefGenerationWorker(max_concurrent=0)

    def test_start_stop_idempotent(self):
        w = BriefGenerationWorker(max_concurrent=1)
        w.start()
        w.start()  # second call is a no-op
        assert w._thread is not None
        w.stop()
        assert w._thread is None

    def test_get_worker_creates_singleton(self):
        w1 = get_worker()
        w2 = get_worker()
        assert w1 is w2

    def test_set_worker_replaces_singleton(self):
        w1 = get_worker()
        w1.start()
        replacement = BriefGenerationWorker(max_concurrent=2)
        set_worker(replacement)
        assert get_worker() is replacement


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_submit_transitions_queued_running_ready(self, monkeypatch):
        _seed_queued(4242, "deadbeefcafe")

        result = _make_success_result()

        def _fake_run(input_: PDBExecutionInput, *, invoker) -> PDBExecutionResult:
            return result

        monkeypatch.setattr(worker_mod, "run_protocol_b", _fake_run)

        worker = BriefGenerationWorker(max_concurrent=2)
        set_worker(worker)
        request = JobRequest(
            key=_make_key(),
            input=_make_input(),
            invoker_factory=lambda: _DummyInvoker(),
        )
        fut = worker.submit(request)
        fut.result(timeout=5)

        assert storage.get_state(4242, "deadbeefcafe") == BriefLifecycleState.READY
        payload = storage.load_ready_brief(4242, "deadbeefcafe")
        assert payload is not None
        assert payload["verdict"] == "approve_candidate"


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_concurrent_submit_same_key_returns_409(self, monkeypatch):
        _seed_queued(4242, "deadbeefcafe")

        release_event = threading.Event()
        call_count = {"n": 0}

        def _blocking_run(input_: PDBExecutionInput, *, invoker) -> PDBExecutionResult:
            call_count["n"] += 1
            release_event.wait(timeout=5)
            return _make_success_result()

        monkeypatch.setattr(worker_mod, "run_protocol_b", _blocking_run)

        worker = BriefGenerationWorker(max_concurrent=4)
        set_worker(worker)
        req = JobRequest(
            key=_make_key(),
            input=_make_input(),
            invoker_factory=lambda: _DummyInvoker(),
        )

        first = worker.submit(req)
        # Wait briefly for the task to actually start and claim the dedupe slot.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not worker.is_active(req.key):
            time.sleep(0.01)
        assert worker.is_active(req.key)

        with pytest.raises(AlreadyRunningError):
            worker.submit(req)

        release_event.set()
        first.result(timeout=5)

        assert call_count["n"] == 1  # only one executor invocation


# ---------------------------------------------------------------------------
# Concurrency limit
# ---------------------------------------------------------------------------


class TestConcurrencyLimit:
    def test_semaphore_caps_parallel_work(self, monkeypatch):
        # Seed queued records for 3 distinct keys.
        keys = [_make_key(pr_number=i, head_sha=f"sha{i:012d}") for i in (1, 2, 3)]
        for key in keys:
            _seed_queued(key.pr_number, key.head_sha)

        in_flight = 0
        max_in_flight = 0
        lock = threading.Lock()
        release_event = threading.Event()

        def _slow_run(input_, *, invoker) -> PDBExecutionResult:
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            try:
                release_event.wait(timeout=5)
                return _make_success_result(
                    pr_number=input_.binding.pr_number,
                    head_sha=input_.binding.head_sha,
                )
            finally:
                with lock:
                    in_flight -= 1

        monkeypatch.setattr(worker_mod, "run_protocol_b", _slow_run)

        worker = BriefGenerationWorker(max_concurrent=2)
        set_worker(worker)
        futs: list[Future[PDBExecutionResult]] = []
        for key in keys:
            req = JobRequest(
                key=key,
                input=_make_input(pr_number=key.pr_number, head_sha=key.head_sha),
                invoker_factory=lambda: _DummyInvoker(),
            )
            futs.append(worker.submit(req))

        # Let the first two start, verify we're capped at 2 in flight.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and max_in_flight < 2:
            time.sleep(0.01)

        assert max_in_flight == 2, f"semaphore cap breached: max_in_flight={max_in_flight}"

        release_event.set()
        for fut in futs:
            fut.result(timeout=5)

        assert max_in_flight == 2  # final check after all drain


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_queued_and_running_records_failed(self, monkeypatch):
        _seed_queued(4242, "deadbeefcafe")

        started = threading.Event()
        release = threading.Event()

        def _blocking_run(input_, *, invoker) -> PDBExecutionResult:
            started.set()
            release.wait(timeout=5)
            return _make_success_result()

        monkeypatch.setattr(worker_mod, "run_protocol_b", _blocking_run)

        worker = BriefGenerationWorker(max_concurrent=1)
        set_worker(worker)
        req = JobRequest(
            key=_make_key(),
            input=_make_input(),
            invoker_factory=lambda: _DummyInvoker(),
        )
        fut = worker.submit(req)
        # Wait for the task to actually claim the semaphore and begin
        # the executor call.
        assert started.wait(timeout=2)

        cancelled = worker.cancel(req.key)
        assert cancelled is True

        # Unblock the executor so the task can clean up.
        release.set()
        from concurrent.futures import CancelledError as _CfCancelled

        with pytest.raises((asyncio.CancelledError, _CfCancelled)):
            fut.result(timeout=5)

        # Give the worker loop a moment to finalize the failed record
        # after the cancel propagates.
        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and storage.get_state(4242, "deadbeefcafe") != BriefLifecycleState.FAILED
        ):
            time.sleep(0.01)

        state = storage.get_state(4242, "deadbeefcafe")
        assert state == BriefLifecycleState.FAILED

    def test_cancel_missing_key_returns_false(self):
        worker = BriefGenerationWorker(max_concurrent=1)
        worker.start()
        try:
            assert worker.cancel(_make_key(pr_number=99, head_sha="x" * 12)) is False
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestExecutorErrors:
    def test_exception_records_failed(self, monkeypatch):
        _seed_queued(4242, "deadbeefcafe")

        def _raise(input_, *, invoker):
            raise RuntimeError("boom")

        monkeypatch.setattr(worker_mod, "run_protocol_b", _raise)

        worker = BriefGenerationWorker(max_concurrent=1)
        set_worker(worker)
        req = JobRequest(
            key=_make_key(),
            input=_make_input(),
            invoker_factory=lambda: _DummyInvoker(),
        )
        fut = worker.submit(req)
        fut.result(timeout=5)

        state = storage.get_state(4242, "deadbeefcafe")
        assert state == BriefLifecycleState.FAILED

    def test_budget_exceeded_recorded_as_failed(self, monkeypatch):
        _seed_queued(4242, "deadbeefcafe")

        def _budget_exceeded(input_, *, invoker):
            from aragora.pdb.budget import PDBBudgetDecision, PDBBudgetStatus

            packet = PRReviewProtocolPacket(
                protocol_version="pr_review_protocol.v1",
                status="budget_exceeded",
                binding=input_.binding,
                review_roles=[],
                provider_slots=[],
                availability_summary=ProviderSlotAvailabilitySummary(
                    total_slots=0, resolved_slots=0
                ),
                recommendation_class="needs_human_attention",
                recommendation_reason="budget",
                confidence=0.0,
                confidence_basis="budget_exceeded",
                dissent_summary="",
                dissenting_views=[],
                validation_summary={},
                top_findings=[],
                cost_estimate={},
            )
            return PDBExecutionResult(
                status=PDBExecutionStatus.BUDGET_EXCEEDED,
                packet=packet,
                brief=None,
                budget_decision=PDBBudgetDecision(
                    status=PDBBudgetStatus.BUDGET_EXCEEDED,
                    total_estimated_usd=42.0,
                    per_brief_cap_usd=8.0,
                    per_day_cap_usd=200.0,
                    per_day_spent_before_usd=200.0,
                    per_day_remaining_before_usd=0.0,
                    funded_slots=(),
                    dropped_slots=(),
                    slot_estimates_usd={},
                    synthesis_estimate_usd=0.0,
                    reason="budget_exhausted_before_phase=findings",
                    binding_cap="per_day",
                ),
                active_roster=(),
                missing_slots=(),
                degrade_reasons=(),
                failure_reason="budget_exhausted_before_phase=findings",
                findings_by_slot={},
                critiques_by_slot={},
                synthesis=None,
                resolutions=(),
                availability_summary=ProviderSlotAvailabilitySummary(
                    total_slots=0, resolved_slots=0
                ),
                actual_cost_usd=0.0,
            )

        monkeypatch.setattr(worker_mod, "run_protocol_b", _budget_exceeded)
        worker = BriefGenerationWorker(max_concurrent=1)
        set_worker(worker)
        req = JobRequest(
            key=_make_key(),
            input=_make_input(),
            invoker_factory=lambda: _DummyInvoker(),
        )
        fut = worker.submit(req)
        fut.result(timeout=5)

        state = storage.get_state(4242, "deadbeefcafe")
        assert state == BriefLifecycleState.FAILED
