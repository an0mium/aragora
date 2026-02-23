"""Tests for shutdown drain sequence."""

import asyncio

import pytest

from aragora.server.shutdown_sequence import ShutdownPhase, ShutdownSequence


class TestShutdownDrain:
    @pytest.mark.asyncio
    async def test_shutdown_waits_for_active_requests(self):
        """Drain phase completes before proceeding."""
        order = []

        async def drain():
            await asyncio.sleep(0.1)
            order.append("drain")

        async def cleanup():
            order.append("cleanup")

        seq = ShutdownSequence()
        seq.add_phase(ShutdownPhase(name="drain", execute=drain, timeout=5.0, critical=True))
        seq.add_phase(ShutdownPhase(name="cleanup", execute=cleanup, timeout=5.0))
        result = await seq.execute_all()
        assert order == ["drain", "cleanup"]
        assert "drain" in result["completed"]
        assert "cleanup" in result["completed"]

    @pytest.mark.asyncio
    async def test_shutdown_times_out_stuck_drain(self):
        """Stuck drain phase doesn't block shutdown."""

        async def stuck():
            await asyncio.sleep(100)  # Would block forever

        async def after():
            pass

        seq = ShutdownSequence()
        seq.add_phase(ShutdownPhase(name="stuck_drain", execute=stuck, timeout=0.1))
        seq.add_phase(ShutdownPhase(name="after", execute=after, timeout=1.0))
        result = await seq.execute_all(overall_timeout=2.0)
        assert "stuck_drain" in result["failed"]
        assert "after" in result["completed"]

    @pytest.mark.asyncio
    async def test_shutdown_phases_execute_in_order(self):
        """Sequential phase ordering preserved."""
        order = []
        seq = ShutdownSequence()
        for i in range(4):

            async def phase_fn(idx=i):
                order.append(idx)

            seq.add_phase(ShutdownPhase(name=f"phase_{i}", execute=phase_fn, timeout=5.0))

        await seq.execute_all()
        assert order == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_shutdown_result_captures_failures(self):
        """Failed phases appear in result dict."""

        async def good():
            pass

        async def bad():
            raise RuntimeError("boom")

        async def also_good():
            pass

        seq = ShutdownSequence()
        seq.add_phase(ShutdownPhase(name="good", execute=good))
        seq.add_phase(ShutdownPhase(name="bad", execute=bad))
        seq.add_phase(ShutdownPhase(name="also_good", execute=also_good))
        result = await seq.execute_all()
        assert "good" in result["completed"]
        assert "bad" in result["failed"]
        assert "also_good" in result["completed"]
        assert isinstance(result["elapsed"], float)
