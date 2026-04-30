"""Tests for ``scripts/publisher_freshness_check.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "publisher_freshness_check.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("publisher_freshness_check", SCRIPT_PATH)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["publisher_freshness_check"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


@pytest.fixture()
def stub_repo(tmp_path: Path) -> Path:
    (tmp_path / ".aragora" / "automation-github-status").mkdir(parents=True)
    (tmp_path / ".aragora" / "automation-outbox").mkdir(parents=True)
    return tmp_path


def _write_cache(tmp_path: Path, outbox_count: int) -> Path:
    cache_path = tmp_path / ".aragora" / "automation-github-status" / "latest.json"
    cache_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-29T17:00:00Z",
                "local_queue": {"outbox_count": outbox_count},
            }
        ),
        encoding="utf-8",
    )
    return cache_path


def _write_outbox_files(tmp_path: Path, count: int) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    for i in range(count):
        (outbox / f"open-pr-codex-test-{i}.json").write_text("{}", encoding="utf-8")


def test_ready_when_loaded_fresh_no_drift(monkeypatch: pytest.MonkeyPatch, stub_repo: Path) -> None:
    cache = _write_cache(stub_repo, outbox_count=3)
    _write_outbox_files(stub_repo, 3)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    now = cache.stat().st_mtime + 60
    report = mod.evaluate(stub_repo, now=now)
    assert report.verdict == "ready"
    assert report.launchd_loaded is True
    assert report.cache_present is True
    assert report.cache_stale is False
    assert report.outbox_drift is False
    assert report.outbox_real_count == 3
    assert report.outbox_cache_count == 3
    assert report.blockers == []


def test_degraded_when_launchd_not_loaded(monkeypatch: pytest.MonkeyPatch, stub_repo: Path) -> None:
    cache = _write_cache(stub_repo, outbox_count=2)
    _write_outbox_files(stub_repo, 2)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (False, "could not find service"))
    now = cache.stat().st_mtime + 60
    report = mod.evaluate(stub_repo, now=now)
    assert report.verdict == "degraded"
    assert report.launchd_loaded is False
    assert "launchd: could not find service" in report.blockers


def test_degraded_when_cache_stale(monkeypatch: pytest.MonkeyPatch, stub_repo: Path) -> None:
    cache = _write_cache(stub_repo, outbox_count=4)
    _write_outbox_files(stub_repo, 4)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (False, "not loaded"))
    now = cache.stat().st_mtime + 7200  # 2 hours stale
    report = mod.evaluate(stub_repo, now=now, stale_threshold_seconds=1800)
    assert report.verdict == "degraded"
    assert report.cache_stale is True
    assert any(b.startswith("cache:") for b in report.blockers)
    assert report.cache_age_human == "2.0h"


def test_warming_when_loaded_but_drift(monkeypatch: pytest.MonkeyPatch, stub_repo: Path) -> None:
    """When launchd is loaded and cache is fresh but drift exists, verdict is warming.

    This represents the transient state right after a publisher cycle that ran
    against a stale outbox snapshot — the cache should refresh on the next cycle.
    """
    cache = _write_cache(stub_repo, outbox_count=2)
    _write_outbox_files(stub_repo, 5)  # real count > cache count
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    now = cache.stat().st_mtime + 60
    report = mod.evaluate(stub_repo, now=now)
    # drift triggers degraded since drift means writes have happened that the
    # cache hasn't reflected yet
    assert report.outbox_drift is True
    assert "drift: outbox=5 cache=2" in report.blockers
    assert report.verdict == "degraded"


def test_warming_label_when_loaded_no_drift_but_cache_just_stale(
    monkeypatch: pytest.MonkeyPatch, stub_repo: Path
) -> None:
    cache = _write_cache(stub_repo, outbox_count=2)
    _write_outbox_files(stub_repo, 2)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    now = cache.stat().st_mtime + 7200
    report = mod.evaluate(stub_repo, now=now, stale_threshold_seconds=1800)
    # loaded + no drift but cache stale -> warming (not degraded)
    assert report.verdict == "warming"
    assert report.outbox_drift is False
    assert report.cache_stale is True


def test_missing_cache(monkeypatch: pytest.MonkeyPatch, stub_repo: Path) -> None:
    _write_outbox_files(stub_repo, 1)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (False, "not loaded"))
    report = mod.evaluate(stub_repo)
    assert report.cache_present is False
    assert report.cache_age_seconds is None
    assert report.cache_age_human == "n/a"
    assert "cache: missing" in report.blockers
    assert report.verdict == "degraded"


def test_outbox_count_excludes_non_json(monkeypatch: pytest.MonkeyPatch, stub_repo: Path) -> None:
    cache = _write_cache(stub_repo, outbox_count=2)
    outbox = stub_repo / ".aragora" / "automation-outbox"
    (outbox / "real-1.json").write_text("{}")
    (outbox / "real-2.json").write_text("{}")
    (outbox / "ignore.txt").write_text("ignore me")
    (outbox / ".DS_Store").write_text("apple metadata")
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    now = cache.stat().st_mtime + 60
    report = mod.evaluate(stub_repo, now=now)
    assert report.outbox_real_count == 2
    assert report.outbox_drift is False


def test_main_text_output_emits_summary(
    monkeypatch: pytest.MonkeyPatch, stub_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_cache(stub_repo, outbox_count=0)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    rc = mod.main(["--repo", str(stub_repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("publisher:")
    assert "launchd: loaded" in out


def test_main_json_output_includes_full_report(
    monkeypatch: pytest.MonkeyPatch, stub_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_cache(stub_repo, outbox_count=0)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    rc = mod.main(["--repo", str(stub_repo), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert parsed["verdict"] == "ready"
    assert "generated_at" in parsed
    assert parsed["launchd_loaded"] is True


def test_main_exit_nonzero_on_degraded(
    monkeypatch: pytest.MonkeyPatch, stub_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (False, "not loaded"))
    rc = mod.main(["--repo", str(stub_repo), "--exit-nonzero-on-degraded"])
    assert rc == 1


def test_main_exit_zero_on_degraded_without_flag(
    monkeypatch: pytest.MonkeyPatch, stub_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (False, "not loaded"))
    rc = mod.main(["--repo", str(stub_repo)])
    assert rc == 0


# ---------------------------------------------------------------------------
# Drift-suppression on stale cache (Round 2026-04-30b Phase E)
# ---------------------------------------------------------------------------


def test_stale_cache_with_count_disagreement_does_not_double_flag_drift(
    monkeypatch: pytest.MonkeyPatch, stub_repo: Path
) -> None:
    """A stale cache will *necessarily* disagree with the live outbox.

    Previously this surfaced both "cache: stale" and "drift: outbox=N cache=M"
    as separate blockers, double-flagging the same root cause. After the
    Phase E fix the drift signal is suppressed when the cache is already
    flagged stale; only "cache: stale" remains as the blocker.

    The raw ``outbox_drift`` field on the report is still True so consumers
    that want the unfiltered observability signal can still see it.
    """
    cache = _write_cache(stub_repo, outbox_count=20)
    _write_outbox_files(stub_repo, 17)  # real count != cache count
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    # Cache is 4 hours stale at default 1800s threshold.
    now = cache.stat().st_mtime + 4 * 3600
    report = mod.evaluate(stub_repo, now=now, stale_threshold_seconds=1800)

    # Cache is stale.
    assert report.cache_stale is True
    cache_blockers = [b for b in report.blockers if b.startswith("cache:")]
    assert cache_blockers and "stale" in cache_blockers[0]

    # Drift is NOT a blocker (suppressed because cache is stale).
    drift_blockers = [b for b in report.blockers if b.startswith("drift:")]
    assert drift_blockers == [], (
        f"drift should be suppressed when cache is stale; got {drift_blockers}"
    )

    # But raw observability fields are preserved.
    assert report.outbox_drift is True
    assert report.outbox_real_count == 17
    assert report.outbox_cache_count == 20
    assert report.drift_detail == "outbox=17 cache=20"

    # Verdict is "warming" not "degraded" — operator action is just
    # waiting for the next publisher cycle, which is exactly what
    # `warming` semantics communicate.
    assert report.verdict == "warming"


def test_fresh_cache_with_count_disagreement_still_flags_drift(
    monkeypatch: pytest.MonkeyPatch, stub_repo: Path
) -> None:
    """Drift signal is meaningful only when the cache is fresh.

    Confirms the inverse of the suppression rule: when the cache is fresh
    AND counts disagree, drift IS a real blocker (the publisher wrote
    inconsistent data).
    """
    cache = _write_cache(stub_repo, outbox_count=2)
    _write_outbox_files(stub_repo, 5)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    # Cache is 60s old → fresh at default 1800s threshold.
    now = cache.stat().st_mtime + 60
    report = mod.evaluate(stub_repo, now=now, stale_threshold_seconds=1800)

    assert report.cache_stale is False
    assert report.outbox_drift is True
    drift_blockers = [b for b in report.blockers if b.startswith("drift:")]
    assert drift_blockers == ["drift: outbox=5 cache=2"]
    assert report.verdict == "degraded"


def test_stale_cache_with_matching_counts_warming_only(
    monkeypatch: pytest.MonkeyPatch, stub_repo: Path
) -> None:
    """Sanity: stale cache without drift is still 'warming', not 'degraded'.

    Companion to the existing ``test_warming_label_when_loaded_no_drift_but_cache_just_stale``;
    this version also exercises the new suppression code path at the same time.
    """
    cache = _write_cache(stub_repo, outbox_count=3)
    _write_outbox_files(stub_repo, 3)
    monkeypatch.setattr(mod, "_launchd_loaded", lambda label: (True, "loaded"))
    now = cache.stat().st_mtime + 4 * 3600
    report = mod.evaluate(stub_repo, now=now, stale_threshold_seconds=1800)
    assert report.cache_stale is True
    assert report.outbox_drift is False
    assert report.verdict == "warming"
