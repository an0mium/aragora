from __future__ import annotations

import importlib
import os
import warnings
from pathlib import Path


def _reload_seed_demo():
    return importlib.reload(importlib.import_module("scripts.seed_demo"))


def test_seed_demo_defaults_repo_local_data_dir(monkeypatch) -> None:
    monkeypatch.delenv("ARAGORA_DATA_DIR", raising=False)
    monkeypatch.delenv("ARAGORA_NOMIC_DIR", raising=False)

    seed_demo = _reload_seed_demo()

    expected = Path(seed_demo.__file__).resolve().parent.parent / ".nomic"
    assert os.environ["ARAGORA_DATA_DIR"] == str(expected)
    assert seed_demo._default_demo_data_dir() == expected


def test_seed_demo_preserves_explicit_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ARAGORA_NOMIC_DIR", raising=False)

    _reload_seed_demo()

    assert os.environ["ARAGORA_DATA_DIR"] == str(tmp_path)


def test_seed_analytics_avoids_event_loop_deprecation(monkeypatch, tmp_path: Path) -> None:
    seed_demo = importlib.import_module("scripts.seed_demo")
    monkeypatch.setattr(seed_demo, "_data_dir", lambda: tmp_path)

    class FakeAnalytics:
        def __init__(self) -> None:
            self.debates = 0
            self.activities = 0
            self.elo_updates = 0

        async def record_debate(self, **kwargs) -> None:
            self.debates += 1

        async def record_agent_activity(self, **kwargs) -> None:
            self.activities += 1

        async def record_elo_update(self, **kwargs) -> None:
            self.elo_updates += 1

    fake = FakeAnalytics()
    monkeypatch.setattr(
        "aragora.analytics.debate_analytics.get_debate_analytics",
        lambda db_path: fake,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        count = seed_demo.seed_analytics(clear=False)

    assert count == 30
    assert fake.debates == 30
    assert fake.activities > 0
    assert fake.elo_updates == 80
