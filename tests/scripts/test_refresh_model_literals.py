"""Tests for ``scripts/refresh_model_literals.py``.

Controller ruling (frontier-model-refresh, Task 8, 2026-09-04): the sweep
must SKIP files that legitimately contain retired ids — the catalog and
upgrade-map source itself, legacy pricing/routing tables old receipts
still resolve through, tests that assert retired ids on purpose, and the
script's own source. That skip list lives in ``SKIP_PATHS`` and is matched
by path suffix so it works regardless of the cwd the sweep runs from.

Fix round 1 (2026-09-05): two Important findings from review — (1)
``--check`` output was non-deterministic (``rglob`` discovery order), now
fixed by sorting scanned files and offenders; (2) the historical-allowlist
membership check compared a raw (possibly cwd- or absolute-path-flavored)
string against repo-relative allowlist entries, now fixed by normalizing
both sides to repo-root-relative POSIX paths via ``REPO_ROOT``.

The allowlist-normalization test loads the script as a module (rather than
via subprocess against the real repo) and monkeypatches its ``REPO_ROOT``
to a throwaway tmp_path tree. This repo's own dev checkout lives under a
directory literally named ``.worktrees`` (see ``SKIP_DIRS`` in the script),
so a subprocess run with a genuinely absolute --paths into the real repo
used to get zero files back regardless of the allowlist fix — an
unrelated, pre-existing SKIP_DIRS hazard flagged in fix-round-1 and fixed
in fix-round-2 (below).

Fix round 2 (2026-09-05): the flagged SKIP_DIRS hazard was itself the
Important finding this round — SKIP_DIRS membership was tested against
each file's raw (as-given) path parts, so an ancestor directory *above*
the --paths scan root (e.g. this checkout's own ``.worktrees`` parent, or
a ``.venv`` somewhere upstream) could false-positive and silently zero
out an absolute-path scan. Fixed by checking SKIP_DIRS only against parts
*relative to* the scan root; see ``test_skip_dirs_apply_only_below_the_scan_root``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path("scripts/refresh_model_literals.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def _load_module() -> Any:
    """Load scripts/refresh_model_literals.py as a fresh, isolated module."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "refresh_model_literals.py"
    spec = importlib.util.spec_from_file_location("refresh_model_literals_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rewrites_bare_and_openrouter_spellings(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text('A = "gpt-4o"\nB = "anthropic/claude-fable-5"\nC = "claude-fable-5-1"\n')
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == (
        'A = "gpt-6-astra"\nB = "anthropic/claude-fable-5.1"\nC = "claude-fable-5-1"\n'
    )


def test_check_fails_on_retired_literal_and_respects_allowlist(tmp_path: Path) -> None:
    f = tmp_path / "old.md"
    f.write_text("we shipped gpt-4 in 2024\n")
    allow = tmp_path / "allow.txt"
    allow.write_text("")
    assert _run("--paths", str(tmp_path), "--check", "--allowlist", str(allow)).returncode == 1
    allow.write_text(f"{f}\n")
    assert _run("--paths", str(tmp_path), "--check", "--allowlist", str(allow)).returncode == 0


def test_does_not_touch_lockfiles_or_git(tmp_path: Path) -> None:
    (tmp_path / "package-lock.json").write_text('{"x":"gpt-4"}')
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0 and (tmp_path / "package-lock.json").read_text() == '{"x":"gpt-4"}'


def test_skip_paths_are_never_rewritten_or_reported(tmp_path: Path) -> None:
    """Files at known SKIP_PATHS suffixes must be left alone entirely.

    These are the catalog/upgrade-map source, legacy pricing/routing
    tables, tests/models/, and the sweep script itself — see the
    SKIP_PATHS comment in scripts/refresh_model_literals.py for why each
    one legitimately contains retired ids on purpose.
    """
    skip_files = {
        tmp_path / "aragora" / "models" / "catalog.py": 'RETIRED = "gpt-4o"\n',
        tmp_path / "aragora" / "billing" / "usage.py": 'LEGACY = "gpt-4"\n',
        tmp_path / "tests" / "models" / "test_retired_on_purpose.py": 'OLD = "grok-3"\n',
        tmp_path / "scripts" / "refresh_model_literals.py": 'SELF = "claude-3-opus"\n',
    }
    for path, content in skip_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    for path, content in skip_files.items():
        assert path.read_text() == content, f"{path} was rewritten but should be skipped"

    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, f"skip-path files were reported as offenders: {r.stdout}"


def test_check_output_is_deterministic_and_sorted_by_path(tmp_path: Path) -> None:
    """Two --check runs over the same tree must print byte-identical,
    path-sorted output — not whatever order the filesystem/rglob happens
    to discover files in.
    """
    for name in ("zeta.py", "alpha.py", "mu.py", "beta.py"):
        (tmp_path / name).write_text('X = "gpt-4"\n')

    r1 = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    r2 = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r1.returncode == 1 and r2.returncode == 1
    assert r1.stdout == r2.stdout, "identical --check runs produced different output"

    offender_lines = [ln for ln in r1.stdout.splitlines() if ": retired model id " in ln]
    assert len(offender_lines) == 4
    reported_paths = [ln.split(":", 1)[0] for ln in offender_lines]
    assert reported_paths == sorted(reported_paths), (
        f"offenders not sorted by path: {reported_paths}"
    )


def test_allowlist_matches_regardless_of_cwd_or_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The historical allowlist stores repo-relative paths (generated via
    ``git ls-files`` from the repo root). Membership must still match when
    the sweep is invoked from an unrelated cwd with an absolute --paths —
    not just when run from the repo root with relative --paths.

    Exercises this against a throwaway fake repo root (monkeypatched onto
    the loaded module) rather than the real one, so the test is hermetic
    and not confounded by this checkout's own SKIP_DIRS(".worktrees")
    layout — see the module docstring above.
    """
    module = _load_module()
    fake_repo_root = (tmp_path / "fake_repo").resolve()
    fixture_file = fake_repo_root / "tests" / "scripts" / "offender.py"
    fixture_file.parent.mkdir(parents=True)
    fixture_file.write_text('X = "gpt-4"\n')
    repo_relative = fixture_file.relative_to(fake_repo_root).as_posix()
    assert repo_relative == "tests/scripts/offender.py"

    monkeypatch.setattr(module, "REPO_ROOT", fake_repo_root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    empty_allow = tmp_path / "empty_allow.txt"
    empty_allow.write_text("")
    sanity = module.main(["--paths", str(fixture_file), "--check", "--allowlist", str(empty_allow)])
    assert sanity == 1, "fixture should be a genuine offender without an allowlist entry"

    allow = tmp_path / "allow.txt"
    allow.write_text(f"{repo_relative}\n")
    result = module.main(["--paths", str(fixture_file), "--check", "--allowlist", str(allow)])
    assert result == 0, (
        "repo-relative allowlist entry did not match a file given as an "
        "absolute path while running from an unrelated cwd"
    )


def test_skip_dirs_apply_only_below_the_scan_root(tmp_path: Path) -> None:
    """SKIP_DIRS (e.g. ".worktrees", ".venv") must only ever exclude
    directories *below* the scan root passed via --paths — never an
    ancestor directory *above* it. A checkout nested under a directory
    literally named ".worktrees" (as this repo's own dev checkouts are)
    must still be scanned when --paths points at or below that directory;
    ".worktrees" should only cause a skip when it appears *inside* the
    scanned tree, i.e. below the given root.
    """
    offender = tmp_path / ".worktrees" / "wt" / "pkg" / "x.py"
    offender.parent.mkdir(parents=True)
    offender.write_text('X = "gpt-4"\n')

    # ".worktrees" is part of the scan root itself here (an ancestor of
    # the file, but AT the root, not below it) — the offender must still
    # be scanned and reported.
    r = _run(
        "--paths",
        str(tmp_path / ".worktrees" / "wt"),
        "--check",
        "--allowlist",
        str(tmp_path / "none.txt"),
    )
    assert r.returncode == 1, f"offender under an absolute scan root was not reported: {r.stdout}"
    assert "x.py" in r.stdout

    # Scanning from tmp_path instead, ".worktrees" is now BELOW the scan
    # root, so SKIP_DIRS correctly excludes it as before.
    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, (
        f"'.worktrees' below the scan root should still be skipped: {r.stdout}"
    )


# ---------------------------------------------------------------------------
# Unresolvable literals (2026-09-05 merge-gate fix wave, finding C-P3 on
# #9989): a BARE literal whose current row is reachable only through
# OpenRouter has no real native id. ``ModelSpec.direct_id`` is a documented
# placeholder on those rows, so rewriting the bare literal to it would swap a
# working native model code for a slug that 400s on the native endpoint.
# ---------------------------------------------------------------------------


def test_replacement_returns_none_for_openrouter_only_bare_literals() -> None:
    mod = _load_module()
    # Both resolve to rows whose provider is "openrouter".
    assert mod.replacement("deepseek-v4-pro") is None
    assert mod.replacement("qwen3-coder") is None
    # A bare literal whose row has a REAL native provider still rewrites.
    assert mod.replacement("moonshot-v1-8k") == "kimi-k3"
    # And the OpenRouter-slug SHAPE always rewrites, openrouter-only or not.
    assert mod.replacement("deepseek/deepseek-v4-pro") == "deepseek/deepseek-v4-pro-0813"


def test_write_leaves_openrouter_only_bare_literals_untouched(tmp_path: Path) -> None:
    f = tmp_path / "cli.py"
    original = (
        'DEEPSEEK = "deepseek-v4-pro"\n'
        'QWEN = "qwen3-coder"\n'
        'KIMI = "moonshot-v1-8k"\n'
        'SLUG = "deepseek/deepseek-v4-pro"\n'
    )
    f.write_text(original)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == (
        'DEEPSEEK = "deepseek-v4-pro"\n'
        'QWEN = "qwen3-coder"\n'
        'KIMI = "kimi-k3"\n'
        'SLUG = "deepseek/deepseek-v4-pro-0813"\n'
    )


def test_check_reports_unresolvable_separately_and_does_not_fail(tmp_path: Path) -> None:
    f = tmp_path / "cli.py"
    f.write_text('DEEPSEEK = "deepseek-v4-pro"\nQWEN = "qwen3-coder"\n')
    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, f"unresolvable literals must not gate the sweep:\n{r.stdout}"
    assert "unresolvable: native spelling of an OpenRouter-only row" in r.stdout
    assert "unresolvable model id deepseek-v4-pro" in r.stdout
    assert "unresolvable model id qwen3-coder" in r.stdout
    assert "0 retired literal(s) outside allowlist" in r.stdout
    assert "2 unresolvable literal(s) (not counted as offenders)" in r.stdout


def test_check_still_fails_when_a_real_offender_shares_the_file(tmp_path: Path) -> None:
    """The unresolvable bucket must not swallow a genuine offender."""
    f = tmp_path / "cli.py"
    f.write_text('OK = "deepseek-v4-pro"\nBAD = "gpt-4o"\n')
    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 1
    assert "retired model id gpt-4o" in r.stdout
    assert "1 retired literal(s) outside allowlist" in r.stdout
    assert "1 unresolvable literal(s) (not counted as offenders)" in r.stdout


def test_check_sees_every_match_on_a_line(tmp_path: Path) -> None:
    """An unresolvable literal earlier on the line must not hide an offender
    later on the same line."""
    f = tmp_path / "cli.py"
    f.write_text('MODELS = ["deepseek-v4-pro", "gpt-4o"]\n')
    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 1
    assert "retired model id gpt-4o" in r.stdout
    assert "1 retired literal(s) outside allowlist" in r.stdout


# ---------------------------------------------------------------------------
# Class 2 — duplicate-key collapse (2026-09-04 controller ruling, from PR 3's
# trial-sweep report). Two DISTINCT retired spellings that rewrite to the SAME
# id silently collapse a hand-written dict/set/list literal onto one entry.
# ---------------------------------------------------------------------------


def test_write_leaves_both_sides_of_a_collision_untouched(tmp_path: Path) -> None:
    """A dict literal with two retired keys that share a replacement keeps
    BOTH keys — the whole point of the table is one row per old spelling."""
    f = tmp_path / "tiers.py"
    original = (
        "MODEL_TIERS = {\n"
        '    "claude-opus-4": {"tier": 1},\n'
        '    "claude-sonnet-4": {"tier": 2},\n'
        "}\n"
    )
    f.write_text(original)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == original


def test_collision_freezes_the_whole_file_not_just_one_line(tmp_path: Path) -> None:
    """File-level is the accepted over-approximation: a colliding spelling is
    frozen everywhere in the file, but a NON-colliding spelling in the same
    file still rewrites."""
    f = tmp_path / "mixed.py"
    f.write_text(
        'A = "claude-opus-4"\n'
        'B = "claude-sonnet-4"\n'
        'FAR_BELOW = "claude-opus-4"\n'
        'UNRELATED = "gemini-2.5-pro"\n'
    )
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == (
        'A = "claude-opus-4"\n'
        'B = "claude-sonnet-4"\n'
        'FAR_BELOW = "claude-opus-4"\n'
        'UNRELATED = "gemini-3.1-pro-preview"\n'
    )


def test_check_reports_collisions_separately_and_does_not_fail(tmp_path: Path) -> None:
    f = tmp_path / "tiers.py"
    f.write_text('A = "claude-opus-4"\nB = "claude-sonnet-4"\n')
    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, f"collisions must not gate the sweep:\n{r.stdout}"
    assert "collision: distinct retired spellings that collapse onto one id" in r.stdout
    assert "collision: claude-opus-4,claude-sonnet-4 -> claude-fable-5-1" in r.stdout
    assert "0 retired literal(s) outside allowlist" in r.stdout
    assert "1 collision(s) (not counted as offenders)" in r.stdout
    # A colliding literal must NOT also be counted as an offender.
    assert "retired model id claude-opus-4" not in r.stdout


def test_collision_does_not_swallow_a_genuine_offender_in_the_same_file(
    tmp_path: Path,
) -> None:
    f = tmp_path / "mixed.py"
    f.write_text('A = "claude-opus-4"\nB = "claude-sonnet-4"\nC = "gemini-2.5-pro"\n')
    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 1
    assert "retired model id gemini-2.5-pro" in r.stdout
    assert "1 retired literal(s) outside allowlist" in r.stdout
    assert "1 collision(s) (not counted as offenders)" in r.stdout


def test_two_spellings_with_different_targets_are_not_a_collision(tmp_path: Path) -> None:
    """Only a SHARED replacement is a collision. Two retired spellings that
    upgrade to different ids are both rewritten as normal."""
    f = tmp_path / "ok.py"
    f.write_text('A = "claude-opus-4"\nB = "gemini-2.5-pro"\n')
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == 'A = "claude-fable-5-1"\nB = "gemini-3.1-pro-preview"\n'


# ---------------------------------------------------------------------------
# Class 3a/3b — bare short tokens and regex sources. ``o1``/``o3`` are the
# only UPGRADES keys that are both hyphen-free and shorter than six
# characters; ``gpt-4`` is hyphenated and deliberately unaffected.
# ---------------------------------------------------------------------------


def test_short_bare_keys_are_exactly_o1_and_o3() -> None:
    mod = _load_module()
    assert mod.SHORT_BARE_KEYS == frozenset({"o1", "o3"})


def test_bare_identifier_named_o1_is_never_rewritten(tmp_path: Path) -> None:
    """PR 3's trial sweep turned ``o1 = _make_org(...)`` into an invalid
    assignment target — a hard SyntaxError no test caught."""
    f = tmp_path / "test_store.py"
    original = "o1 = _make_org(name='acme')\nassert o1.id\n"
    f.write_text(original)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == original


def test_o1_in_prose_is_never_rewritten(tmp_path: Path) -> None:
    f = tmp_path / "feasibility.md"
    original = "The evaluation covered GPT-4o, o1, o3 and the Claude line.\n"
    f.write_text(original)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == original


def test_guarded_short_tokens_are_not_reported_as_offenders(tmp_path: Path) -> None:
    """A guarded match was never a model id, so --check must not name it —
    otherwise the sweep could never reach a clean exit."""
    f = tmp_path / "prose.md"
    f.write_text("we shipped o1 and o3 in 2024\n")
    r = _run("--paths", str(tmp_path), "--check", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stdout
    assert "0 retired literal(s) outside allowlist" in r.stdout
    assert "0 collision(s) (not counted as offenders)" in r.stdout


def test_short_key_as_a_complete_string_literal_is_rewritten(tmp_path: Path) -> None:
    """The quoting rule ADMITS the model-id shapes: a complete string
    literal, or one followed by an id separator."""
    f = tmp_path / "pins.py"
    f.write_text('MODEL = "o1"\n')
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == 'MODEL = "gpt-6-astra"\n'


def test_short_key_followed_by_a_colon_inside_a_string_is_rewritten(tmp_path: Path) -> None:
    f = tmp_path / "route.py"
    f.write_text("PREFIX = 'o3:reasoning'\n")
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == "PREFIX = 'gpt-6-astra:reasoning'\n"


def test_raw_string_regex_source_is_never_rewritten(tmp_path: Path) -> None:
    """``aragora/debate/provider_diversity.py``'s ``r"gpt|o1|o3|chatgpt"``
    matcher: sweeping it made ``detect_provider("o1-preview")`` return
    "unknown"."""
    f = tmp_path / "provider_diversity.py"
    original = 'PATTERNS = {"openai": [r"gpt|o1|o3|chatgpt"]}\n'
    f.write_text(original)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == original


def test_raw_string_guard_covers_hyphenated_keys_too(tmp_path: Path) -> None:
    """The raw-string guard is not restricted to short tokens: a raw string
    in this repo is a regex source, never a model id."""
    f = tmp_path / "matcher.py"
    original = 'FAMILY = r"^(claude-opus-4|gpt-4o)$"\n'
    f.write_text(original)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == original


def test_re_compile_line_is_never_rewritten(tmp_path: Path) -> None:
    f = tmp_path / "safety.py"
    original = 'RX = re.compile("gpt|o1|o3")\n'
    f.write_text(original)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == original


def test_pipe_separated_alternation_is_never_rewritten(tmp_path: Path) -> None:
    """A non-raw, non-``re.compile`` alternation string is still a matcher."""
    f = tmp_path / "markers.py"
    original = 'OPENAI = "gpt-4o|o1|o3"\n'
    f.write_text(original)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == original


def test_a_pipe_elsewhere_in_the_string_does_not_freeze_a_real_id(tmp_path: Path) -> None:
    """The alternation guard requires the match to be BOUNDED by ``|`` or by
    the string edge — a markdown table cell in a quoted string still
    rewrites."""
    f = tmp_path / "row.py"
    f.write_text('ROW = "model gpt-4o costs | see table"\n')
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    assert f.read_text() == 'ROW = "model gpt-6-astra costs | see table"\n'


# ---------------------------------------------------------------------------
# Class 6 — a frozen pricing source's test must be frozen with it.
# ---------------------------------------------------------------------------


def test_frozen_pricing_source_tests_are_in_skip_paths() -> None:
    """A SKIP_PATHS pricing table is keyed on its historical spellings, so
    the test that looks those spellings up EXACTLY must be skipped too."""
    mod = _load_module()
    paired = {
        "aragora/billing/usage.py": (
            "tests/billing/test_usage.py",
            "tests/billing/test_billing_usage.py",
        ),
        "aragora/billing/debate_costs.py": ("tests/billing/test_debate_costs.py",),
        "aragora/services/metering_models.py": (
            "tests/services/test_usage_metering.py",
            "tests/services/test_usage_metering_service.py",
        ),
        "aragora/pdb/real_invoker.py": ("tests/pdb/test_real_invoker.py",),
        "aragora/server/handlers/debates/cost_estimation.py": (
            "tests/handlers/debates/test_cost_estimation.py",
        ),
    }
    for source, tests in paired.items():
        assert source in mod.SKIP_PATHS
        for t in tests:
            assert t in mod.SKIP_PATHS, f"{t} pairs with frozen {source} but is not skipped"
    assert "tests/e2e/test_billing_accuracy_e2e.py" in mod.SKIP_PATHS


def test_frozen_pricing_test_files_are_not_rewritten(tmp_path: Path) -> None:
    fixtures = {
        tmp_path / "tests" / "billing" / "test_usage.py": 'K = "gpt-4o"\n',
        tmp_path / "tests" / "pdb" / "test_real_invoker.py": 'K = "claude-opus-4"\n',
        tmp_path
        / "tests"
        / "handlers"
        / "debates"
        / "test_cost_estimation.py": 'K = "gemini-2.5-pro"\n',
        tmp_path / "tests" / "e2e" / "test_billing_accuracy_e2e.py": 'K = "grok-4"\n',
    }
    for path, content in fixtures.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    r = _run("--paths", str(tmp_path), "--write", "--allowlist", str(tmp_path / "none.txt"))
    assert r.returncode == 0, r.stderr
    for path, content in fixtures.items():
        assert path.read_text() == content, f"{path} was rewritten but pairs with a frozen source"


def test_module_docstring_documents_the_period_vs_hyphen_split() -> None:
    """The same literal maps to the hyphen form bare and the dotted form as a
    slug; that is by design and must be written down where the sweep's users
    look."""
    mod = _load_module()
    doc = mod.__doc__ or ""
    assert "direct_id" in doc and "openrouter_id" in doc
    assert "claude-fable-5-1" in doc and "anthropic/claude-fable-5.1" in doc
