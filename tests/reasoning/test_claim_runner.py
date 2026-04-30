"""Tests for aragora.reasoning.claim_runner (DIC-14)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from aragora.reasoning.claim_runner import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_EVIDENCE_CHARS,
    ClaimContext,
    ClaimReport,
    ClaimResult,
    ClaimRunner,
    ClaimVerdict,
    ExecutableClaim,
    _normalise_predicate_result,
    _run_one,
    _truncate_evidence,
)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _bool_pred(value: bool):
    def _p(_ctx: ClaimContext) -> bool:
        return value

    return _p


def _tuple_pred(value: bool, evidence: str):
    def _p(_ctx: ClaimContext) -> tuple[bool, str]:
        return value, evidence

    return _p


def _slow_pred(seconds: float):
    def _p(_ctx: ClaimContext) -> bool:
        time.sleep(seconds)
        return True

    return _p


async def _async_pred(value: bool):
    return value


# --------------------------------------------------------------------- #
# Module-level surface                                                  #
# --------------------------------------------------------------------- #


class TestModuleExports:
    def test_default_timeout_positive(self) -> None:
        assert DEFAULT_TIMEOUT_SECONDS > 0

    def test_evidence_cap_positive(self) -> None:
        assert MAX_EVIDENCE_CHARS > 100

    def test_verdict_values(self) -> None:
        assert ClaimVerdict.PASS.value == "pass"
        assert ClaimVerdict.FAIL.value == "fail"
        assert ClaimVerdict.TIMEOUT.value == "timeout"
        assert ClaimVerdict.ERROR.value == "error"


# --------------------------------------------------------------------- #
# Helper functions                                                      #
# --------------------------------------------------------------------- #


class TestTruncateEvidence:
    def test_under_limit_passthrough(self) -> None:
        assert _truncate_evidence("short") == "short"

    def test_over_limit_inserts_sentinel(self) -> None:
        # Use a clearly-oversized input so sentinel overhead can't
        # accidentally make the output longer than the input.
        s = "x" * (MAX_EVIDENCE_CHARS * 2)
        out = _truncate_evidence(s)
        assert "[TRUNCATED:" in out
        assert len(out) < len(s)


class TestNormalisePredicateResult:
    def test_bare_bool_true(self) -> None:
        assert _normalise_predicate_result(True) == (True, "")

    def test_bare_bool_false(self) -> None:
        assert _normalise_predicate_result(False) == (False, "")

    def test_tuple_passes_through(self) -> None:
        assert _normalise_predicate_result((True, "ok")) == (True, "ok")

    def test_tuple_with_long_evidence_truncated(self) -> None:
        big = "x" * (MAX_EVIDENCE_CHARS + 50)
        passed, ev = _normalise_predicate_result((True, big))
        assert passed is True
        assert "[TRUNCATED:" in ev

    def test_rejects_non_bool(self) -> None:
        with pytest.raises(TypeError):
            _normalise_predicate_result("not a bool")  # type: ignore[arg-type]

    def test_rejects_three_tuple(self) -> None:
        with pytest.raises(ValueError):
            _normalise_predicate_result((True, "a", "b"))  # type: ignore[arg-type]

    def test_rejects_tuple_with_non_bool(self) -> None:
        with pytest.raises(TypeError):
            _normalise_predicate_result((1, "x"))  # type: ignore[arg-type]

    def test_rejects_tuple_with_non_str_evidence(self) -> None:
        with pytest.raises(TypeError):
            _normalise_predicate_result((True, 42))  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# _run_one                                                              #
# --------------------------------------------------------------------- #


class TestRunOne:
    @pytest.mark.asyncio
    async def test_passing_sync_predicate(self) -> None:
        claim = ExecutableClaim("ok", _bool_pred(True))
        result = await _run_one(claim, {}, default_timeout=5.0)
        assert result.verdict == ClaimVerdict.PASS
        assert result.error is None
        assert result.evidence == ""

    @pytest.mark.asyncio
    async def test_failing_sync_predicate(self) -> None:
        claim = ExecutableClaim("nope", _bool_pred(False))
        result = await _run_one(claim, {}, default_timeout=5.0)
        assert result.verdict == ClaimVerdict.FAIL
        assert result.passed() is False

    @pytest.mark.asyncio
    async def test_evidence_carried_through(self) -> None:
        claim = ExecutableClaim("withevid", _tuple_pred(True, "p99=87ms"))
        result = await _run_one(claim, {}, default_timeout=5.0)
        assert result.verdict == ClaimVerdict.PASS
        assert result.evidence == "p99=87ms"

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_verdict(self) -> None:
        claim = ExecutableClaim("slow", _slow_pred(2.0), timeout_seconds=0.05)
        result = await _run_one(claim, {}, default_timeout=10.0)
        assert result.verdict == ClaimVerdict.TIMEOUT
        assert result.error is not None
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_exception_returns_error_verdict(self) -> None:
        def boom(_ctx):
            raise RuntimeError("kaboom")

        claim = ExecutableClaim("boom", boom)
        result = await _run_one(claim, {}, default_timeout=5.0)
        assert result.verdict == ClaimVerdict.ERROR
        assert result.error is not None
        assert "RuntimeError" in result.error

    @pytest.mark.asyncio
    async def test_async_predicate_supported(self) -> None:
        async def predicate(_ctx):
            return True

        claim = ExecutableClaim("async-pass", predicate)
        result = await _run_one(claim, {}, default_timeout=5.0)
        assert result.verdict == ClaimVerdict.PASS

    @pytest.mark.asyncio
    async def test_async_predicate_with_evidence(self) -> None:
        async def predicate(_ctx):
            return False, "something went wrong"

        claim = ExecutableClaim("async-fail", predicate)
        result = await _run_one(claim, {}, default_timeout=5.0)
        assert result.verdict == ClaimVerdict.FAIL
        assert result.evidence == "something went wrong"

    @pytest.mark.asyncio
    async def test_per_claim_timeout_overrides_default(self) -> None:
        # Default timeout is 10s but the claim caps itself at 50ms.
        claim = ExecutableClaim("explicit-to", _slow_pred(1.0), timeout_seconds=0.05)
        result = await _run_one(claim, {}, default_timeout=10.0)
        assert result.verdict == ClaimVerdict.TIMEOUT

    @pytest.mark.asyncio
    async def test_sync_timeout_is_cooperative_not_force(self) -> None:
        """Round-30e Phase H finding from codex review: sync predicate
        timeouts are caller-side only. The caller sees TIMEOUT
        promptly, but the worker thread keeps running in the
        background because Python threads aren't externally
        cancellable. This test pins that contract."""

        def slow(_ctx):
            # Sleep longer than the timeout. Python threads can't be
            # force-killed so this thread runs to completion off-loop.
            time.sleep(0.3)
            return True

        claim = ExecutableClaim("slow-sync", slow, timeout_seconds=0.05)
        t0 = time.monotonic()
        result = await _run_one(claim, {}, default_timeout=2.0)
        caller_elapsed = time.monotonic() - t0
        # Caller sees TIMEOUT promptly...
        assert result.verdict == ClaimVerdict.TIMEOUT
        # ...without waiting for the underlying thread.
        assert caller_elapsed < 0.2, f"caller waited {caller_elapsed:.3f}s; expected <0.2s"


# --------------------------------------------------------------------- #
# ClaimRunner.run                                                       #
# --------------------------------------------------------------------- #


class TestClaimRunner:
    @pytest.mark.asyncio
    async def test_runs_multiple_claims_in_parallel(self) -> None:
        runner = ClaimRunner(default_timeout_seconds=5.0)
        # Each predicate sleeps 0.2s. If serialised, total > 0.6s. If
        # parallel, total ~0.2s. We assert "well under serial".
        claims = [ExecutableClaim(f"c{i}", _slow_pred(0.2)) for i in range(3)]
        t0 = time.monotonic()
        report = await runner.run(claims, {})
        elapsed = time.monotonic() - t0
        assert report.all_passed
        assert elapsed < 0.5, f"expected parallel execution; got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_empty_claims_returns_empty_report(self) -> None:
        runner = ClaimRunner()
        report = await runner.run([], {})
        assert report.total == 0
        assert report.all_passed is False  # ``all_passed`` requires total > 0

    @pytest.mark.asyncio
    async def test_one_failure_does_not_cascade(self) -> None:
        def boom(_ctx):
            raise RuntimeError("nope")

        claims = [
            ExecutableClaim("good", _bool_pred(True)),
            ExecutableClaim("bad", boom),
            ExecutableClaim("good2", _bool_pred(True)),
        ]
        report = await ClaimRunner().run(claims, {})
        assert report.total == 3
        assert report.passed == 2
        assert report.errored == 1

    @pytest.mark.asyncio
    async def test_report_aggregates_correctly(self) -> None:
        claims = [
            ExecutableClaim("p1", _bool_pred(True)),
            ExecutableClaim("p2", _bool_pred(True)),
            ExecutableClaim("f1", _bool_pred(False)),
            ExecutableClaim("to1", _slow_pred(1.0), timeout_seconds=0.05),
        ]
        report = await ClaimRunner(default_timeout_seconds=2.0).run(claims, {})
        assert report.total == 4
        assert report.passed == 2
        assert report.failed == 1
        assert report.timed_out == 1
        assert report.errored == 0
        assert report.all_passed is False

    @pytest.mark.asyncio
    async def test_max_concurrency_bounds_parallelism(self) -> None:
        # With max_concurrency=1 and three 100ms claims, serial >=300ms.
        runner = ClaimRunner(default_timeout_seconds=5.0, max_concurrency=1)
        claims = [ExecutableClaim(f"c{i}", _slow_pred(0.1)) for i in range(3)]
        t0 = time.monotonic()
        await runner.run(claims, {})
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.25, f"expected serialised execution; got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_context_dict_passed_through(self) -> None:
        seen: dict[str, object] = {}

        def capture(ctx):
            seen.update(ctx)
            return True

        report = await ClaimRunner().run([ExecutableClaim("cap", capture)], {"x": 42, "y": "z"})
        assert report.passed == 1
        assert seen == {"x": 42, "y": "z"}

    @pytest.mark.asyncio
    async def test_rejects_duplicate_claim_names(self) -> None:
        claims = [
            ExecutableClaim("dup", _bool_pred(True)),
            ExecutableClaim("dup", _bool_pred(True)),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            await ClaimRunner().run(claims, {})

    @pytest.mark.asyncio
    async def test_rejects_non_callable_predicate(self) -> None:
        bad = ExecutableClaim("bad", "not-callable")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            await ClaimRunner().run([bad], {})

    @pytest.mark.asyncio
    async def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            await ClaimRunner().run([ExecutableClaim("", _bool_pred(True))], {})


# --------------------------------------------------------------------- #
# Report serialisation                                                  #
# --------------------------------------------------------------------- #


class TestReportSerialisation:
    @pytest.mark.asyncio
    async def test_report_to_json_round_trips(self) -> None:
        claims = [
            ExecutableClaim("p", _tuple_pred(True, "ev")),
            ExecutableClaim("f", _bool_pred(False)),
        ]
        report = await ClaimRunner().run(claims, {})
        data = report.to_json()
        # Must be JSON-serialisable verbatim.
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        assert decoded["total"] == 2
        assert decoded["passed"] == 1
        assert decoded["failed"] == 1
        assert any(r["verdict"] == "pass" for r in decoded["results"])
        assert any(r["verdict"] == "fail" for r in decoded["results"])

    def test_result_to_json_includes_verdict_string(self) -> None:
        r = ClaimResult(
            name="x",
            verdict=ClaimVerdict.PASS,
            evidence="e",
            elapsed_seconds=0.1,
            started_at="2026-04-30T00:00:00+00:00",
            finished_at="2026-04-30T00:00:01+00:00",
        )
        d = r.to_json()
        assert d["verdict"] == "pass"

    def test_empty_report_all_passed_false(self) -> None:
        # Sanity property: ``all_passed`` requires at least one claim.
        r = ClaimReport()
        assert r.all_passed is False
        assert r.total == 0
