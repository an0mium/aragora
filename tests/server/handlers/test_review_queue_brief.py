"""Tests for the Mode 3 brief-generation HTTP endpoints."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.pdb import storage, worker as worker_mod
from aragora.pdb.brief_state import BriefLifecycleState
from aragora.pdb.input_loader import (
    InputLoaderError,
    InputLoaderErrorReason,
    LoadedExecutionInput,
)
from aragora.pdb.protocol import (
    PDBExecutionInput,
    PDBExecutionResult,
    PDBExecutionStatus,
)
from aragora.pdb.worker import BriefGenerationWorker, set_worker, reset_worker
from aragora.review.policy import ReviewPolicy
from aragora.review.provider_slots import ProviderSlotAvailabilitySummary
from aragora.server.handlers import review_queue as rq
from aragora.server.handlers import review_queue_brief as rqb
from aragora.swarm.pr_review_protocol import PRReviewBinding, PRReviewProtocolPacket


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    rq._review_queue_limiter._buckets.clear()


@pytest.fixture(autouse=True)
def _enable_flag(monkeypatch):
    monkeypatch.setenv("ARAGORA_PDB_BRIEF_GENERATION_ENABLED", "1")


@pytest.fixture(autouse=True)
def _fresh_worker():
    reset_worker()
    yield
    reset_worker()


@pytest.fixture(autouse=True)
def _clear_overrides():
    rqb.set_test_invoker_factory(None)
    rqb.set_test_input_loader(None)
    yield
    rqb.set_test_invoker_factory(None)
    rqb.set_test_input_loader(None)


@pytest.fixture
def handler():
    return rq.ReviewQueueHandler(ctx={})


# ---------------------------------------------------------------------------
# Mock request handler
# ---------------------------------------------------------------------------


def _mock_http_handler(method: str = "POST", body: bytes = b"") -> MagicMock:
    h = MagicMock()
    h.command = method
    h.client_address = ("127.0.0.1", 11111)
    h.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
        "Authorization": "Bearer test",
        "Host": "localhost:8080",
    }
    h.rfile = MagicMock()
    h.rfile.read.return_value = body
    return h


def _parse(result) -> dict[str, Any]:
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# Authentication stub
# ---------------------------------------------------------------------------


@pytest.fixture
def _authed(monkeypatch):
    # Bypass token auth — test handler always treats requests as authenticated.
    class _StubUser:
        user_id = "test-user"

    def _fake_require_auth(self, handler):
        return _StubUser(), None

    monkeypatch.setattr(rq.ReviewQueueHandler, "require_auth_or_error", _fake_require_auth)


# ---------------------------------------------------------------------------
# Input-loader + invoker stubs
# ---------------------------------------------------------------------------


def _make_loaded_input(
    *,
    pr_number: int = 4242,
    head_sha: str = "deadbeefcafe0011",
    repo: str = "synaptent/aragora",
) -> LoadedExecutionInput:
    binding = PRReviewBinding(
        repo=repo, pr_number=pr_number, base_sha="base000011", head_sha=head_sha
    )
    packet = PRReviewProtocolPacket(
        protocol_version="pr_review_protocol.v1",
        status="metadata_heuristic",
        binding=binding,
        review_roles=[],
        provider_slots=[],
        availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
        recommendation_class="needs_human_attention",
        recommendation_reason="test",
        confidence=0.5,
        confidence_basis="metadata_heuristic",
        dissent_summary="",
        dissenting_views=[],
        validation_summary={},
        top_findings=[],
        cost_estimate={},
    )
    exec_input = PDBExecutionInput(
        binding=binding,
        packet=packet,
        packet_sha="",
        pr_title="Test",
        pr_body="",
        labels=(),
        changed_files=(),
        diff_excerpt="",
        validation_summary={},
        panel_id="protocol_b_default",
        policy=ReviewPolicy(),
    )
    return LoadedExecutionInput(
        input=exec_input,
        head_sha=head_sha,
        base_sha=binding.base_sha,
        panel_models=("slot_a", "slot_b"),
        repo=repo,
    )


def _make_ready_result(
    *, pr_number: int, head_sha: str, repo: str = "synaptent/aragora"
) -> PDBExecutionResult:
    from aragora.pdb.budget import PDBBudgetDecision, PDBBudgetStatus

    class _FakeBrief:
        def to_dict(self) -> dict[str, Any]:
            return {
                "pr_number": pr_number,
                "head_sha": head_sha,
                "verdict": "approve_candidate",
                "confidence": 4,
            }

    return PDBExecutionResult(
        status=PDBExecutionStatus.SUCCESS,
        packet=PRReviewProtocolPacket(
            protocol_version="pr_review_protocol.v1",
            status="panel_executed",
            binding=PRReviewBinding(
                repo=repo, pr_number=pr_number, base_sha="b", head_sha=head_sha
            ),
            review_roles=[],
            provider_slots=[],
            availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
            recommendation_class="approve_candidate",
            recommendation_reason="",
            confidence=0.8,
            confidence_basis="panel_executed",
            dissent_summary="",
            dissenting_views=[],
            validation_summary={},
            top_findings=[],
            cost_estimate={},
        ),
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


def _install_loader(head_sha: str = "deadbeefcafe0011", pr_number: int = 4242) -> None:
    def _loader(pr: int, repo, policy):
        return _make_loaded_input(pr_number=pr, head_sha=head_sha)

    rqb.set_test_input_loader(_loader)


def _install_fast_invoker(monkeypatch) -> None:
    def _run(input_, *, invoker):
        return _make_ready_result(
            pr_number=input_.binding.pr_number, head_sha=input_.binding.head_sha
        )

    monkeypatch.setattr(worker_mod, "run_protocol_b", _run)
    rqb.set_test_invoker_factory(lambda: object())


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


class TestFeatureFlagOff:
    def test_generate_returns_503(self, handler, _authed, monkeypatch):
        monkeypatch.delenv("ARAGORA_PDB_BRIEF_GENERATION_ENABLED", raising=False)
        mock_h = _mock_http_handler("POST")
        result = handler.handle_post("/api/v1/review-queue/prs/42/brief/generate", {}, mock_h)
        assert result.status_code == 503

    def test_state_returns_503(self, handler, monkeypatch):
        monkeypatch.delenv("ARAGORA_PDB_BRIEF_GENERATION_ENABLED", raising=False)
        mock_h = _mock_http_handler("GET")
        result = handler.handle("/api/v1/review-queue/prs/42/brief/state", {}, mock_h)
        assert result.status_code == 503

    def test_delete_returns_503(self, handler, _authed, monkeypatch):
        monkeypatch.delenv("ARAGORA_PDB_BRIEF_GENERATION_ENABLED", raising=False)
        mock_h = _mock_http_handler("DELETE")
        result = handler.handle_delete("/api/v1/review-queue/prs/42/brief/generate", {}, mock_h)
        assert result.status_code == 503

    def test_get_brief_and_list_still_work_when_flag_off(self, handler, monkeypatch):
        monkeypatch.delenv("ARAGORA_PDB_BRIEF_GENERATION_ENABLED", raising=False)
        # List PRs: gh unavailable is degraded but not 503.
        with pytest.MonkeyPatch.context() as m:
            m.setattr(rq, "_run_gh", lambda *a, **k: (127, "", "no gh"))
            mock_h = _mock_http_handler("GET")
            result = handler.handle("/api/v1/review-queue/prs", {}, mock_h)
        assert result.status_code == 200
        # GET brief: returns 404 when none on disk (flag-independent).
        mock_h = _mock_http_handler("GET")
        result = handler.handle("/api/v1/review-queue/prs/42/brief", {}, mock_h)
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Full state machine flow
# ---------------------------------------------------------------------------


class TestFullFlow:
    def test_generate_then_state_becomes_ready(self, handler, _authed, monkeypatch):
        _install_loader(head_sha="deadbeefcafe0011", pr_number=4242)
        _install_fast_invoker(monkeypatch)
        worker = BriefGenerationWorker(max_concurrent=2)
        set_worker(worker)

        mock_h = _mock_http_handler("POST", body=b"{}")
        result = handler.handle_post("/api/v1/review-queue/prs/4242/brief/generate", {}, mock_h)
        assert result.status_code == 202
        data = _parse(result)
        assert data["state"] == "queued"
        assert data["head_sha"] == "deadbeefcafe0011"
        assert data["estimated_completion_seconds"] > 0

        # Poll /brief/state until ready.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            mock_h_state = _mock_http_handler("GET")
            state_result = handler.handle(
                "/api/v1/review-queue/prs/4242/brief/state", {}, mock_h_state
            )
            payload = _parse(state_result)
            if payload["state"] == "ready":
                break
            time.sleep(0.05)

        final = _parse(state_result)
        assert final["state"] == "ready"
        # Final GET /brief returns the brief payload.
        mock_h_brief = _mock_http_handler("GET")
        brief_result = handler.handle("/api/v1/review-queue/prs/4242/brief", {}, mock_h_brief)
        assert brief_result.status_code == 200
        brief_payload = _parse(brief_result)["brief"]
        assert brief_payload["verdict"] == "approve_candidate"


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_409_when_already_running(self, handler, _authed, monkeypatch):
        _install_loader(head_sha="aaaa0000", pr_number=100)
        import threading

        release = threading.Event()

        def _slow_run(input_, *, invoker):
            release.wait(timeout=5)
            return _make_ready_result(
                pr_number=input_.binding.pr_number, head_sha=input_.binding.head_sha
            )

        monkeypatch.setattr(worker_mod, "run_protocol_b", _slow_run)
        rqb.set_test_invoker_factory(lambda: object())

        worker = BriefGenerationWorker(max_concurrent=2)
        set_worker(worker)

        first = handler.handle_post(
            "/api/v1/review-queue/prs/100/brief/generate",
            {},
            _mock_http_handler("POST", body=b"{}"),
        )
        assert first.status_code == 202

        # Wait for the state to reflect queued on disk.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            s = storage.get_state(100, "aaaa0000")
            if s in (BriefLifecycleState.QUEUED, BriefLifecycleState.RUNNING):
                break
            time.sleep(0.01)

        second = handler.handle_post(
            "/api/v1/review-queue/prs/100/brief/generate",
            {},
            _mock_http_handler("POST", body=b"{}"),
        )
        assert second.status_code == 409
        body2 = _parse(second)
        assert body2["state"] in ("queued", "running")

        release.set()
        # Let the first complete.
        deadline = time.monotonic() + 5
        while (
            time.monotonic() < deadline
            and storage.get_state(100, "aaaa0000") != BriefLifecycleState.READY
        ):
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    def test_delete_cancels_running_work(self, handler, _authed, monkeypatch):
        _install_loader(head_sha="bbbb1111", pr_number=77)
        import threading

        started = threading.Event()
        release = threading.Event()

        def _blocking(input_, *, invoker):
            started.set()
            release.wait(timeout=5)
            return _make_ready_result(
                pr_number=input_.binding.pr_number, head_sha=input_.binding.head_sha
            )

        monkeypatch.setattr(worker_mod, "run_protocol_b", _blocking)
        rqb.set_test_invoker_factory(lambda: object())

        worker = BriefGenerationWorker(max_concurrent=1)
        set_worker(worker)

        result = handler.handle_post(
            "/api/v1/review-queue/prs/77/brief/generate",
            {},
            _mock_http_handler("POST", body=b"{}"),
        )
        assert result.status_code == 202
        assert started.wait(timeout=2)

        del_result = handler.handle_delete(
            "/api/v1/review-queue/prs/77/brief/generate",
            {},
            _mock_http_handler("DELETE"),
        )
        assert del_result.status_code == 200
        body = _parse(del_result)
        assert body["cancelled"] is True
        assert body["worker_cancelled"] is True

        release.set()
        # Eventually we should see failed.
        deadline = time.monotonic() + 3
        while (
            time.monotonic() < deadline
            and storage.get_state(77, "bbbb1111") != BriefLifecycleState.FAILED
        ):
            time.sleep(0.02)
        assert storage.get_state(77, "bbbb1111") == BriefLifecycleState.FAILED

    def test_delete_no_active_returns_200_with_state(self, handler, _authed, monkeypatch):
        _install_loader(head_sha="eeee3333", pr_number=5)
        worker = BriefGenerationWorker(max_concurrent=1)
        set_worker(worker)
        del_result = handler.handle_delete(
            "/api/v1/review-queue/prs/5/brief/generate",
            {},
            _mock_http_handler("DELETE"),
        )
        assert del_result.status_code == 200
        body = _parse(del_result)
        assert body["cancelled"] is False
        assert body["state"] == "absent"


# ---------------------------------------------------------------------------
# Stale invalidation
# ---------------------------------------------------------------------------


class TestStaleInvalidation:
    def test_old_ready_brief_moved_on_new_head(self, handler, _authed, monkeypatch, tmp_path):
        # Pre-seed a ready brief with an old head SHA.
        briefs_root = tmp_path / "briefs"
        briefs_root.mkdir(parents=True, exist_ok=True)
        old_file = briefs_root / "pr-321-oldsha000000.json"
        old_file.write_text(
            json.dumps(
                {
                    "pr_number": 321,
                    "head_sha": "oldsha00000011223344",
                    "verdict": "approve_candidate",
                    "confidence": 4,
                }
            ),
            encoding="utf-8",
        )

        _install_loader(head_sha="newsha99999900", pr_number=321)
        _install_fast_invoker(monkeypatch)
        worker = BriefGenerationWorker(max_concurrent=1)
        set_worker(worker)

        result = handler.handle_post(
            "/api/v1/review-queue/prs/321/brief/generate",
            {},
            _mock_http_handler("POST", body=b"{}"),
        )
        assert result.status_code == 202

        # Old brief has been moved to invalidated/
        invalidated = briefs_root / "invalidated" / "pr-321-oldsha000000.json"
        assert invalidated.exists()
        assert not old_file.exists()


# ---------------------------------------------------------------------------
# Input loader errors surfacing as HTTP codes
# ---------------------------------------------------------------------------


class TestLoaderErrors:
    def test_pr_not_found_returns_404(self, handler, _authed):
        def _raise(pr, repo, policy):
            raise InputLoaderError(InputLoaderErrorReason.PR_NOT_FOUND, "no such PR")

        rqb.set_test_input_loader(_raise)
        result = handler.handle_post(
            "/api/v1/review-queue/prs/999/brief/generate",
            {},
            _mock_http_handler("POST", body=b"{}"),
        )
        assert result.status_code == 404

    def test_gh_missing_returns_503(self, handler, _authed):
        def _raise(pr, repo, policy):
            raise InputLoaderError(InputLoaderErrorReason.GH_MISSING, "")

        rqb.set_test_input_loader(_raise)
        result = handler.handle_post(
            "/api/v1/review-queue/prs/999/brief/generate",
            {},
            _mock_http_handler("POST", body=b"{}"),
        )
        assert result.status_code == 503

    def test_auth_returns_403(self, handler, _authed):
        def _raise(pr, repo, policy):
            raise InputLoaderError(InputLoaderErrorReason.GH_AUTHENTICATION, "")

        rqb.set_test_input_loader(_raise)
        result = handler.handle_post(
            "/api/v1/review-queue/prs/999/brief/generate",
            {},
            _mock_http_handler("POST", body=b"{}"),
        )
        assert result.status_code == 403
