"""External invariants for scripts/regenerate_metrics.py.

The drift check in the script itself compares the live ground truth
against the committed docs/METRICS.md. That is a useful
staleness check, but it is self-referential: if the committed doc was
wrong to begin with, the check would happily keep reproducing the
same wrong numbers as long as the ground truth also didn't move.

This test suite holds external lower-bound invariants on the metrics
the script produces. They encode facts about the codebase that are
obviously true (aragora has more than 100 Python files, more than
100,000 test definitions, etc.) and would catch:

  * A counting function silently returning 0 (e.g. ripgrep not
    available, a path typo returning an empty directory).
  * A counting function producing a number orders of magnitude off
    from reality (e.g. xargs/wc batching bug returning only the
    last chunk's total).

The bounds are intentionally loose: they exist to catch catastrophic
regressions, not to enforce specific values (those are the script's
job). Keep the bounds well below current values so this suite almost
never needs updating.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "regenerate_metrics.py"


def _load_module():
    # Register in sys.modules before exec so @dataclass decorators
    # can resolve the module (otherwise cls.__module__ -> None).
    spec = importlib.util.spec_from_file_location("regenerate_metrics", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["regenerate_metrics"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def snapshot():
    mod = _load_module()
    return {m.key: m for m in mod.gather_metrics().metrics}


# Lower-bound invariants. Pick bounds ~20-50% below the current live
# value so normal repo growth never breaks the test but a catastrophic
# undercount (e.g. 0, or an off-by-1000 from xargs batching) does.

MIN_BOUNDS = {
    "python_files": 1000,  # actual ~4000
    "python_loc": 500_000,  # actual ~1.9M
    "top_level_modules": 50,  # actual ~136
    "test_files": 1000,  # actual ~5000
    "test_functions": 100_000,  # actual ~215K
    "parametrize_decorators": 100,
    "cli_command_modules": 20,  # actual ~60
    "openapi_paths": 1000,  # actual ~2800
    "openapi_operations": 1000,  # actual ~3200
    "rbac_permission_calls": 500,
    "rbac_unique_permissions": 100,
    "python_sdk_modules": 50,
    "typescript_sdk_modules": 50,
    "allowed_agent_types": 10,
    "knowledge_mound_adapter_specs": 20,
    "knowledge_mound_adapter_files": 20,
    "doc_files": 100,
    "ci_workflows": 20,
}


@pytest.mark.parametrize("metric_key,min_value", list(MIN_BOUNDS.items()))
def test_metric_above_lower_bound(snapshot, metric_key, min_value):
    """Every counted metric must exceed a sanity lower bound.

    If this fails for a real reason (aragora shrank a lot), lower the
    bound in MIN_BOUNDS above. If it fails because the counter returned
    0 or a much-lower number than expected, there is a bug in
    scripts/regenerate_metrics.py.
    """
    assert metric_key in snapshot, (
        f"metric {metric_key!r} missing from snapshot; did it get renamed?"
    )
    metric = snapshot[metric_key]
    assert isinstance(metric.value, int), (
        f"metric {metric_key!r} has non-int value {metric.value!r}"
    )
    assert metric.value > min_value, (
        f"metric {metric_key!r} = {metric.value} is below sanity "
        f"lower bound {min_value}. Either the codebase genuinely "
        f"shrank (lower the bound) or the counter is buggy."
    )


def test_markdown_has_no_timestamp_or_sha(snapshot):
    """Canonical doc must not embed generation timestamp or git SHA.

    Embedding either into a tracked file guarantees merge conflicts
    whenever two branches regenerate the doc. The authoritative
    timestamp and SHA live in --json output instead.
    """
    mod = _load_module()
    rendered = mod.render_markdown(
        mod.MetricsSnapshot(
            generated_at="1970-01-01T00:00:00+00:00",
            git_sha="deadbeef",
            metrics=list(snapshot.values()),
        )
    )
    assert "1970-01-01" not in rendered, "generation timestamp leaked into rendered markdown"
    assert "deadbeef" not in rendered, "git sha leaked into rendered markdown"


def test_loc_count_matches_python_sum():
    """LOC metric must match an independent git-tracked file sum.

    Guards against xargs/wc batching bugs: if the script ever
    reintroduces shell-pipe counting, this test catches it by
    computing the sum a second way. It intentionally uses git-tracked
    files, not Path.rglob, so ignored local artifacts cannot pollute
    canonical metrics.
    """
    mod = _load_module()
    snap = {m.key: m for m in mod.gather_metrics().metrics}
    naive_total = 0
    tracked_files = subprocess.check_output(
        ["git", "ls-files", "--", "aragora"],
        cwd=REPO_ROOT,
        text=True,
    ).splitlines()
    for rel_path in tracked_files:
        if not rel_path.endswith(".py"):
            continue
        p = REPO_ROOT / rel_path
        try:
            with p.open(encoding="utf-8", errors="replace") as f:
                naive_total += sum(1 for _ in f)
        except OSError:
            pass
    assert snap["python_loc"].value == naive_total, (
        f"python_loc metric {snap['python_loc'].value} disagrees with "
        f"independent Python sum {naive_total}"
    )


def test_counts_are_tracked_content_not_local_filesystem_noise(snapshot):
    """Canonical counts must ignore ignored/untracked local artifacts.

    Developer machines often have ignored scratch files such as CLAUDE.md,
    benchmark output, or generated reports under counted directories. The
    public metrics must describe repository content, not machine state.
    """
    tracked_aragora = subprocess.check_output(
        ["git", "ls-files", "--", "aragora"],
        cwd=REPO_ROOT,
        text=True,
    ).splitlines()
    tracked_docs = subprocess.check_output(
        ["git", "ls-files", "--", "docs"],
        cwd=REPO_ROOT,
        text=True,
    ).splitlines()

    expected_top_modules = len(
        {Path(p).parts[1] for p in tracked_aragora if len(Path(p).parts) > 2}
    )
    expected_doc_files = sum(1 for p in tracked_docs if p.endswith(".md"))

    assert snapshot["top_level_modules"].value == expected_top_modules
    assert snapshot["doc_files"].value == expected_doc_files


def test_check_mode_is_idempotent(tmp_path, monkeypatch):
    """Running regenerate twice in a row must report no drift.

    This is the core invariant the drift CI depends on.
    """
    mod = _load_module()
    snapshot_a = mod.gather_metrics()
    monkeypatch.setattr(mod, "METRICS_DOC", tmp_path / "METRICS.md")
    md = mod.render_markdown(snapshot_a)
    (tmp_path / "METRICS.md").write_text(md)

    snapshot_b = mod.gather_metrics()
    drifted, drifts = mod.check_drift(snapshot_b)
    assert not drifted, f"drift detected between two back-to-back regenerations: {drifts}"
