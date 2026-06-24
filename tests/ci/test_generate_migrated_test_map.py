"""Drift guard for scripts/ci/generate_migrated_test_map.py.

Enforces that the committed ``_MIGRATED_TEST_MAP`` in
``scripts/nomic_ci_test_selector.py`` matches the map freshly derived from the
three P1 tests-migration commits (PRs #8387, #8404, #8415) -- i.e. the
"auto-generated" label is real, not aspirational.

The derivation needs git history; on a shallow clone (no migration commits)
the history-dependent tests skip rather than fail spuriously.  The
git-independent invariant tests (no dead entries; every destination exists)
always run regardless of clone depth.

CI guard rationale: no CI workflow runs this drift-guard test or the
generator's ``--check`` (they are a developer/local full-history guard), and
the in-CI selector ``scripts/nomic_ci_test_selector.py`` reads only the static
``_MIGRATED_TEST_MAP`` (no git history), so the soft-skip cannot weaken CI.  The
always-on git-independent invariants are the standing guard; the history-derived
completeness check is enforced wherever full history exists (local dev,
``fetch-depth: 0``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = REPO_ROOT / "scripts" / "ci" / "generate_migrated_test_map.py"

_spec = importlib.util.spec_from_file_location("generate_migrated_test_map", _GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _derive_or_skip() -> dict[str, str]:
    try:
        return gen.derive_map()
    except gen.HistoryUnavailable as exc:  # shallow clone -- nothing to enforce
        pytest.skip(str(exc))


class TestDerivedMap:
    """The derivation produces exactly the reachable (top-level) entries."""

    def test_only_top_level_reachable_relocation_is_exceptions(self):
        """aragora/exceptions.py is the sole relocated top-level module."""
        derived = _derive_or_skip()
        assert derived == {"tests/test_exceptions.py": "tests/agents/test_exceptions.py"}

    def test_committed_map_matches_derived(self):
        """The committed _MIGRATED_TEST_MAP equals the freshly-derived map."""
        derived = _derive_or_skip()
        assert gen.committed_map() == derived

    def test_check_passes_on_clean_tree(self):
        """`--check` exits 0 when committed and derived agree."""
        _derive_or_skip()  # ensures history is present, else skip
        assert gen.main(["--check"]) == 0


class TestCommittedMapInvariants:
    """Git-independent invariants -- always enforced, even on a shallow clone."""

    def test_every_committed_entry_is_top_level_reachable(self):
        """No dead entries: every key maps from an existing top-level module."""
        committed = gen.committed_map()
        assert committed, "the migration map should not be empty"
        for old in committed:
            assert old.startswith("tests/test_")
            assert "/" not in old[len("tests/") :], f"{old} is not a root-form key"
            assert gen._top_level_source_exists(old), (
                f"dead entry {old}: no top-level aragora source"
            )

    def test_every_committed_destination_exists(self):
        """Every migrated target still exists on disk (catches re-relocation)."""
        committed = gen.committed_map()
        missing = [new for new in committed.values() if not (REPO_ROOT / new).exists()]
        assert missing == []


class TestCheckDetectsDrift:
    """`--check` fails (non-zero, offender named) on a tampered committed map."""

    def test_dead_entry_fails_check(self, monkeypatch, capsys):
        _derive_or_skip()
        polluted = dict(gen.committed_map())
        polluted["tests/test_agent_grok.py"] = "tests/agents/test_agent_grok.py"
        monkeypatch.setattr(gen, "committed_map", lambda: polluted)

        rc = gen.main(["--check"])

        out = capsys.readouterr().out
        assert rc == 1
        assert "tests/test_agent_grok.py" in out
        assert "dead entry" in out

    def test_missing_entry_fails_check(self, monkeypatch, capsys):
        _derive_or_skip()
        monkeypatch.setattr(gen, "committed_map", lambda: {})

        rc = gen.main(["--check"])

        out = capsys.readouterr().out
        assert rc == 1
        assert "missing reachable entry" in out
        assert "tests/test_exceptions.py" in out

    def test_wrong_destination_fails_check(self, monkeypatch, capsys):
        _derive_or_skip()
        monkeypatch.setattr(
            gen,
            "committed_map",
            lambda: {"tests/test_exceptions.py": "tests/wrong/test_exceptions.py"},
        )

        rc = gen.main(["--check"])

        out = capsys.readouterr().out
        assert rc == 1
        assert "wrong destination" in out


def test_format_map_renders_literal_block():
    """format_map renders the canonical _MIGRATED_TEST_MAP literal block."""
    rendered = gen.format_map({"tests/test_exceptions.py": "tests/agents/test_exceptions.py"})
    assert rendered == (
        "_MIGRATED_TEST_MAP = {  # {old_root_path: new_subdir_path}\n"
        '    "tests/test_exceptions.py": "tests/agents/test_exceptions.py",\n'
        "}"
    )


def test_print_flag_removed():
    """The parsed-but-unused ``--print`` flag is gone (argparse rejects it)."""
    with pytest.raises(SystemExit):
        gen.main(["--print"])


class TestFindCommitTieBreak:
    """``_find_commit`` is PR-scoped and prefers the original (oldest) commit.

    All cases stub ``gen._git`` so they are git-independent (run even on a
    shallow clone).
    """

    def test_prefers_oldest_when_subject_duplicated(self, monkeypatch):
        """A later duplicate of the same subject must not win over the original."""
        # ``git log`` is newest-first; both lines carry "relocate batch" + "(#8387)".
        log_out = (
            "newsha111 refactor(tests): re-land relocate batch 1 (#8387) (#9001)\n"
            "oldsha000 refactor(tests): relocate batch 1 (#8387)\n"
        )
        monkeypatch.setattr(gen, "_git", lambda *args: log_out)
        assert gen._find_commit("8387") == "oldsha000"

    def test_falls_back_when_no_subject_match(self, monkeypatch):
        """With no matching subject, the immutable fallback SHA is used."""
        monkeypatch.setattr(gen, "_git", lambda *args: "")
        monkeypatch.setattr(gen, "_commit_exists", lambda sha: True)
        assert gen._find_commit("8387") == gen._FALLBACK_COMMITS["8387"]


class TestRootTestRenamesDetection:
    """``_root_test_renames`` detects R renames AND add+delete relocations."""

    def test_detects_git_rename(self, monkeypatch):
        diff = "R100\ttests/test_foo.py\ttests/foo/test_foo.py\n"
        monkeypatch.setattr(gen, "_git", lambda *args: diff)
        assert gen._root_test_renames("c") == [("tests/test_foo.py", "tests/foo/test_foo.py")]

    def test_detects_add_delete_relocation(self, monkeypatch):
        """A delete+add pair for the same basename is treated as a relocation."""
        diff = "M\tsome/other/file.py\nD\ttests/test_foo.py\nA\ttests/foo/test_foo.py\n"
        monkeypatch.setattr(gen, "_git", lambda *args: diff)
        assert gen._root_test_renames("c") == [("tests/test_foo.py", "tests/foo/test_foo.py")]

    def test_unpaired_root_delete_is_ignored(self, monkeypatch):
        """A deleted root test with no matching add yields no relocation."""
        diff = "D\ttests/test_foo.py\nA\ttests/foo/test_bar.py\n"
        monkeypatch.setattr(gen, "_git", lambda *args: diff)
        assert gen._root_test_renames("c") == []

    def test_ambiguous_add_delete_is_skipped(self, monkeypatch):
        """Two adds with the same basename cannot be disambiguated -> skipped."""
        diff = "D\ttests/test_foo.py\nA\ttests/foo/test_foo.py\nA\ttests/bar/test_foo.py\n"
        monkeypatch.setattr(gen, "_git", lambda *args: diff)
        assert gen._root_test_renames("c") == []

    def test_add_delete_destination_outside_tests_is_skipped(self, monkeypatch):
        """A same-basename add OUTSIDE tests/ is not a test relocation -> skipped."""
        diff = "D\ttests/test_foo.py\nA\tdocs/test_foo.py\n"
        monkeypatch.setattr(gen, "_git", lambda *args: diff)
        assert gen._root_test_renames("c") == []
