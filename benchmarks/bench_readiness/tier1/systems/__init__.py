"""Systems under test.

Keep provider-specific imports lazy so the harness can be imported and tested
without requiring every runtime dependency to be installed.
"""

from benchmarks.bench_readiness.tier1.systems.base import SystemOutput


def run_solo_opus(*args, **kwargs):
    from benchmarks.bench_readiness.tier1.systems.solo_opus import run_solo_opus as _impl

    return _impl(*args, **kwargs)


def run_aragora_debate(*args, **kwargs):
    from benchmarks.bench_readiness.tier1.systems.aragora_debate import (
        run_aragora_debate as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = ["SystemOutput", "run_solo_opus", "run_aragora_debate"]
