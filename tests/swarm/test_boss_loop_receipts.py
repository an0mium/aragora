"""Focused tests for BossLoop operational receipt emission."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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
