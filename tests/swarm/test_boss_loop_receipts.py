"""Focused tests for BossLoop operational receipt emission."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from aragora.receipts.lane import LaneCompletionReceipt, validate_receipt
from aragora.swarm.boss_loop import BossLoop, BossLoopConfig, BossStopReason, RunnerFreshnessResult

UTC = timezone.utc


def _fresh_result(
    *, fresh: bool = True, blocked_reason: str | None = None
) -> RunnerFreshnessResult:
    return RunnerFreshnessResult(
        fresh=fresh,
        runner_ids=["codex-runner-1"] if fresh else [],
        checked_at=datetime.now(UTC).isoformat(),
        blocked_reason=blocked_reason,
    )


class TestBossLoopOperationalReceipts:
    def test_run_emits_terminal_receipt(self) -> None:
        feed = MagicMock()
        feed.fetch.return_value = []

        loop = BossLoop(
            config=BossLoopConfig(max_iterations=1, iteration_interval_seconds=0.0),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        with patch("aragora.receipts.provenance.emit_operational_receipt") as emit_receipt:
            result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value
        emit_receipt.assert_called_once()
        kwargs = emit_receipt.call_args.kwargs
        assert kwargs["source"] == "boss_loop"
        assert kwargs["action"] == "run_completed"
        assert kwargs["inputs"]["run_id"] == result.run_id
        assert kwargs["outputs"]["stop_reason"] == BossStopReason.NO_SUITABLE_ISSUE.value

    def test_receipt_failures_do_not_block_run_completion(self) -> None:
        feed = MagicMock()
        feed.fetch.return_value = []

        loop = BossLoop(
            config=BossLoopConfig(max_iterations=1, iteration_interval_seconds=0.0),
            issue_feed=feed,
            freshness_checker=lambda **kw: _fresh_result(fresh=True),
        )

        with patch(
            "aragora.receipts.provenance.emit_operational_receipt",
            side_effect=RuntimeError("receipt store unavailable"),
        ):
            result = asyncio.run(loop.run())

        assert result.stop_reason == BossStopReason.NO_SUITABLE_ISSUE.value


class TestLaneCompletionReceiptSchema:
    def test_receipt_roundtrip(self) -> None:
        receipt = LaneCompletionReceipt(
            task_id="task-42",
            lease_id="lease-abc",
            agent_id="claude-worker-1",
            base_sha="aaa111",
            head_sha="bbb222",
            changed_files=["aragora/swarm/boss_loop.py"],
            validations_run=[{"name": "pytest", "passed": True}],
            outcome="pass",
            risks=["large diff"],
            pr_url="https://github.com/org/repo/pull/99",
            pr_number=99,
            branch="feat/lane-42",
            duration_seconds=12.5,
        )
        d = receipt.to_dict()

        assert d["task_id"] == "task-42"
        assert d["lease_id"] == "lease-abc"
        assert d["agent_id"] == "claude-worker-1"
        assert d["base_sha"] == "aaa111"
        assert d["head_sha"] == "bbb222"
        assert d["changed_files"] == ["aragora/swarm/boss_loop.py"]
        assert d["validations_run"] == [{"name": "pytest", "passed": True}]
        assert d["outcome"] == "pass"
        assert d["risks"] == ["large diff"]
        assert d["pr_url"] == "https://github.com/org/repo/pull/99"
        assert d["pr_number"] == 99
        assert d["branch"] == "feat/lane-42"
        assert "content_hash" in d
        assert len(d["content_hash"]) == 64  # SHA-256 hex

        restored = LaneCompletionReceipt.from_dict(d)
        assert restored.task_id == receipt.task_id
        assert restored.lease_id == receipt.lease_id
        assert restored.outcome == receipt.outcome

    def test_content_hash_deterministic(self) -> None:
        r1 = LaneCompletionReceipt(task_id="t1", lease_id="l1", agent_id="a1", outcome="pass")
        r2 = LaneCompletionReceipt(
            task_id="t1",
            lease_id="l1",
            agent_id="a1",
            receipt_id=r1.receipt_id,
            outcome="pass",
        )
        assert r1.content_hash() == r2.content_hash()

    def test_validate_receipt_valid(self) -> None:
        receipt = LaneCompletionReceipt(task_id="t1", lease_id="l1", agent_id="a1", outcome="pass")
        errors = validate_receipt(receipt)
        assert errors == []

    def test_validate_receipt_missing_fields(self) -> None:
        errors = validate_receipt(
            {
                "task_id": "",
                "lease_id": None,
                "agent_id": "a1",
                "outcome": "pass",
            }
        )
        assert any("task_id" in e for e in errors)
        assert any("lease_id" in e for e in errors)

    def test_validate_receipt_invalid_outcome(self) -> None:
        errors = validate_receipt(
            {
                "task_id": "t1",
                "lease_id": "l1",
                "agent_id": "a1",
                "outcome": "banana",
            }
        )
        assert any("invalid outcome" in e for e in errors)

    def test_validate_receipt_from_dict(self) -> None:
        data = {
            "task_id": "t1",
            "lease_id": "l1",
            "agent_id": "a1",
            "outcome": "blocked",
            "extra_field": "ignored",
        }
        receipt = LaneCompletionReceipt.from_dict(data)
        assert receipt.task_id == "t1"
        assert receipt.outcome == "blocked"
        assert validate_receipt(receipt) == []

    def test_missing_receipt_detectable(self) -> None:
        """A None or empty dict is flagged as malformed."""
        errors = validate_receipt({})
        assert len(errors) >= 4  # at least task_id, lease_id, agent_id, outcome
