"""Tests for ``scripts/probe_proof_surface_freshness.py``.

These tests exercise the lane P67 proof-surface freshness probe via
its module-level functions and through the CLI entrypoint:

1. Both surfaces fresh -> exit 0, payload reports ``fresh: true``.
2. One surface stale -> exit non-zero, offender listed on stderr.
3. Malformed ``Last updated:`` header -> exit non-zero with a clear
   error message on stderr.

The tests build a minimal mock repo root under ``tmp_path`` so they
never depend on the *real* B0 / TW-03 docs and never mutate them.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "probe_proof_surface_freshness.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("probe_proof_surface_freshness", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_proof_surface_freshness"] = module
    spec.loader.exec_module(module)
    return module


probe_mod = _load_module()


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _write_surface(repo_root: Path, rel: str, last_updated_line: str, *, body: str = "") -> Path:
    """Materialise a tiny markdown status doc inside ``repo_root``."""
    target = repo_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# Surface Heading\n"
        "\n"
        f"{last_updated_line}\n"
        "\n"
        "This is a synthetic fixture for the freshness probe tests.\n"
        f"{body}\n"
    )
    target.write_text(text, encoding="utf-8")
    return target


@pytest.fixture()
def mock_repo(tmp_path: Path) -> Path:
    """Build a mock repo root that mirrors the real surface layout."""
    (tmp_path / "docs" / "status").mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Module-level function tests.
# ---------------------------------------------------------------------------


def test_parse_last_updated_accepts_iso8601_z() -> None:
    text = "# Heading\n\nLast updated: 2026-05-17T14:36:51Z\n\nbody\n"
    parsed = probe_mod.parse_last_updated(text)
    assert parsed == dt.datetime(2026, 5, 17, 14, 36, 51, tzinfo=dt.timezone.utc)


def test_parse_last_updated_accepts_bare_date() -> None:
    text = "# Heading\n\nLast updated: 2026-04-17\n\nbody\n"
    parsed = probe_mod.parse_last_updated(text)
    assert parsed == dt.datetime(2026, 4, 17, tzinfo=dt.timezone.utc)


def test_parse_last_updated_rejects_garbage() -> None:
    text = "# Heading\n\nLast updated: not-a-date\n"
    with pytest.raises(probe_mod.FreshnessProbeError):
        probe_mod.parse_last_updated(text)


def test_parse_last_updated_rejects_missing_header() -> None:
    text = "# Heading\n\nNo header at all\n"
    with pytest.raises(probe_mod.FreshnessProbeError):
        probe_mod.parse_last_updated(text)


def test_probe_surface_marks_fresh_within_window(mock_repo: Path) -> None:
    _write_surface(
        mock_repo,
        "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
        "Last updated: 2026-05-17T00:00:00Z",
    )
    now = dt.datetime(2026, 5, 18, 12, 0, 0, tzinfo=dt.timezone.utc)
    result = probe_mod.probe_surface("b0", repo_root=mock_repo, max_age_days=7, now=now)
    assert result.surface == "b0"
    assert result.fresh is True
    assert result.last_updated == "2026-05-17T00:00:00Z"
    assert 1.4 < result.age_days < 1.6


def test_probe_surface_marks_stale_beyond_window(mock_repo: Path) -> None:
    _write_surface(
        mock_repo,
        "docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md",
        "Last updated: 2026-04-01T00:00:00Z",
    )
    now = dt.datetime(2026, 5, 18, 0, 0, 0, tzinfo=dt.timezone.utc)
    result = probe_mod.probe_surface("tw03", repo_root=mock_repo, max_age_days=7, now=now)
    assert result.fresh is False
    assert result.age_days > 7


def test_probe_surface_tolerates_small_future_clock_skew(mock_repo: Path) -> None:
    _write_surface(
        mock_repo,
        "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
        "Last updated: 2026-05-18T00:04:00Z",
    )
    now = dt.datetime(2026, 5, 18, 0, 0, 0, tzinfo=dt.timezone.utc)
    result = probe_mod.probe_surface("b0", repo_root=mock_repo, max_age_days=7, now=now)
    assert result.fresh is True
    assert result.age_days == 0.0


def test_probe_surface_rejects_material_future_timestamp(mock_repo: Path) -> None:
    _write_surface(
        mock_repo,
        "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
        "Last updated: 2026-05-20T00:00:00Z",
    )
    now = dt.datetime(2026, 5, 18, 0, 0, 0, tzinfo=dt.timezone.utc)
    with pytest.raises(probe_mod.FreshnessProbeError, match="future"):
        probe_mod.probe_surface("b0", repo_root=mock_repo, max_age_days=7, now=now)


# ---------------------------------------------------------------------------
# CLI / acceptance tests required by the lane brief.
# ---------------------------------------------------------------------------


def _run_cli(repo_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(repo_root),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _today_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_cli_both_fresh_exits_zero(mock_repo: Path) -> None:
    """Acceptance #1 — both surfaces fresh -> exit 0."""
    fresh = _today_iso()
    _write_surface(
        mock_repo,
        "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
        f"Last updated: {fresh}",
    )
    _write_surface(
        mock_repo,
        "docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md",
        f"Last updated: {fresh}",
    )

    proc = _run_cli(mock_repo, "--max-age-days", "7")

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["fresh"] is True
    surfaces = {entry["surface"]: entry for entry in payload["surfaces"]}
    assert set(surfaces) == {"b0", "tw03"}
    for entry in surfaces.values():
        assert entry["fresh"] is True
        assert entry["age_days"] < 1.0
    # Healthy case should not have a noisy stderr message.
    assert "stale proof surface" not in proc.stderr


def test_cli_one_stale_exits_non_zero_with_offender(mock_repo: Path) -> None:
    """Acceptance #2 — one stale surface -> non-zero exit + offender listed."""
    fresh = _today_iso()
    _write_surface(
        mock_repo,
        "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
        f"Last updated: {fresh}",
    )
    # Make TW-03 obviously stale (>30 days old, well past any sensible
    # max_age_days default).
    stale_ts = (dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _write_surface(
        mock_repo,
        "docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md",
        f"Last updated: {stale_ts}",
    )

    proc = _run_cli(mock_repo, "--max-age-days", "7")

    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    payload = json.loads(proc.stdout)
    assert payload["fresh"] is False
    surfaces = {entry["surface"]: entry for entry in payload["surfaces"]}
    assert surfaces["b0"]["fresh"] is True
    assert surfaces["tw03"]["fresh"] is False
    # Stderr must clearly identify the offending surface so an operator
    # can act on it without parsing JSON.
    assert "tw03" in proc.stderr
    assert "stale proof surface" in proc.stderr
    # The fresh surface should *not* be listed as an offender.
    assert "b0(" not in proc.stderr


def test_cli_malformed_last_updated_raises_clear_error(
    mock_repo: Path,
) -> None:
    """Acceptance #3 — malformed header -> clear error and non-zero exit."""
    fresh = _today_iso()
    _write_surface(
        mock_repo,
        "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
        "Last updated: garbage-not-a-date",
    )
    _write_surface(
        mock_repo,
        "docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md",
        f"Last updated: {fresh}",
    )

    proc = _run_cli(mock_repo, "--surfaces", "b0,tw03")

    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "malformed" in proc.stderr.lower()
    assert "garbage-not-a-date" in proc.stderr
    assert "b0" in proc.stderr


def test_cli_future_last_updated_raises_clear_error(mock_repo: Path) -> None:
    future = (dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(days=2)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _write_surface(
        mock_repo,
        "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
        f"Last updated: {future}",
    )
    _write_surface(
        mock_repo,
        "docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md",
        f"Last updated: {_today_iso()}",
    )

    proc = _run_cli(mock_repo, "--surfaces", "b0,tw03")

    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "future" in proc.stderr.lower()
    assert "b0" in proc.stderr
    assert future in proc.stderr


def test_cli_surfaces_flag_scopes_probe(mock_repo: Path) -> None:
    """``--surfaces b0`` should only probe B0 and ignore TW-03."""
    fresh = _today_iso()
    _write_surface(
        mock_repo,
        "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
        f"Last updated: {fresh}",
    )
    # Intentionally leave the TW-03 surface absent. The probe must not
    # crash because we scoped it out.

    proc = _run_cli(mock_repo, "--surfaces", "b0", "--max-age-days", "7")

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert [entry["surface"] for entry in payload["surfaces"]] == ["b0"]


def test_cli_rejects_unknown_surface(mock_repo: Path) -> None:
    proc = _run_cli(mock_repo, "--surfaces", "definitely-not-a-surface")
    # argparse exits with code 2 for usage errors.
    assert proc.returncode == 2
    assert "unknown surface" in proc.stderr.lower()


def test_gti_surface_is_registered() -> None:
    module = _load_module()
    assert "gti" in module.SURFACE_PATHS
    assert module.SURFACE_PATHS["gti"] == Path("docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md")
    # gti is opt-in: it must NOT change the default surface set.
    assert "gti" not in module.DEFAULT_SURFACES
