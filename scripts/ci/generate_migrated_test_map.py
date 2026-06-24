#!/usr/bin/env python3
"""Derive (and verify) the nomic CI test-selector migration map.

``scripts/nomic_ci_test_selector.py`` keeps a ``_MIGRATED_TEST_MAP`` that lets
the selector follow a relocated root test (``tests/test_<x>.py`` ->
``tests/<module>/test_<x>.py``) when a CHANGED top-level ``aragora/<x>.py``
module's legacy root test was moved during the P1 tests-migration.

That map is only ever consulted in ``infer_test_paths`` for a genuinely
top-level ``aragora/<x>.py`` (the ``len(parts) == 1`` branch); a source under a
subdirectory takes the ``len(parts) == 2`` branch and never reads the map.  So
the ONLY entries that can affect resolution are relocations whose top-level
source ``aragora/<x>.py`` still exists.  Every other relocated root test (the
overwhelming majority) is dead weight in the map.

This script regenerates that map directly from the three P1 tests-migration
commits (PRs #8387, #8404, #8415) so the "auto-generated" label is enforced
rather than aspirational:

* default / ``--print``  -- print the freshly-derived map as a Python literal.
* ``--check``           -- exit non-zero (offenders named) when the committed
                           ``_MIGRATED_TEST_MAP`` diverges from the derived map.

When the migration commits are unavailable (e.g. a shallow CI clone), the
derivation cannot run; the script emits a warning and exits 0 (soft skip) so a
shallow checkout never spuriously fails.  Full-history environments (local dev,
``fetch-depth: 0`` CI, the pytest drift guard) still enforce.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_REL = "scripts/nomic_ci_test_selector.py"
GENERATOR_REL = "scripts/ci/generate_migrated_test_map.py"

# The three P1 tests-migration PRs whose merge commits relocated loose root
# tests into mirrored subdirectories.
MIGRATION_PRS = ("8387", "8404", "8415")

# Immutable fallback SHAs for the three relocation commits, used only if the
# local clone's commit subjects no longer match the grep (grep-by-PR wins).
_FALLBACK_COMMITS = {
    "8387": "4d5188c81cde57e8361453ab9e29653684d6c0e9",
    "8404": "d6ef78d62b6a016f741e461cdf022dadfb646eff",
    "8415": "5e2218542821f1741a982958fc783db0b387d705",
}

# A pre-migration root test: ``tests/test_<x>.py`` with no further subdirectory.
_ROOT_TEST_RE = re.compile(r"^tests/test_[^/]+\.py$")


class HistoryUnavailable(RuntimeError):
    """Raised when a migration commit is not present in this clone."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _commit_exists(sha: str) -> bool:
    try:
        _git("cat-file", "-e", f"{sha}^{{commit}}")
        return True
    except subprocess.CalledProcessError:
        return False


def _find_commit(pr: str) -> str | None:
    """Locate the batch-relocation commit for a migration PR, or None."""
    try:
        out = _git("log", "--all", "--format=%H %s", "--grep=relocate batch")
    except subprocess.CalledProcessError:
        out = ""
    for line in out.splitlines():
        sha, _, subject = line.partition(" ")
        if "relocate batch" in subject and f"(#{pr})" in subject:
            return sha
    fallback = _FALLBACK_COMMITS.get(pr)
    if fallback and _commit_exists(fallback):
        return fallback
    return None


def _root_test_renames(commit: str) -> list[tuple[str, str]]:
    """Return ``(old_root_test, new_path)`` renames introduced by ``commit``."""
    out = _git("show", "--name-status", "-M", "-C", commit)
    pairs: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line.startswith("R"):
            continue
        cols = line.split("\t")
        if len(cols) == 3 and _ROOT_TEST_RE.match(cols[1]):
            pairs.append((cols[1], cols[2]))
    return pairs


def _top_level_source_exists(old_root_test: str) -> bool:
    """Whether the top-level module implied by ``tests/test_<x>.py`` exists."""
    module = old_root_test[len("tests/test_") :]  # "<x>.py"
    return (REPO_ROOT / "aragora" / module).exists()


def derive_map() -> dict[str, str]:
    """Derive the reachable migration map from the three migration commits.

    Only relocations whose top-level source ``aragora/<x>.py`` still exists are
    kept -- those are exactly the entries ``infer_test_paths`` can consult.

    Raises ``HistoryUnavailable`` if any migration commit is missing locally.
    """
    derived: dict[str, str] = {}
    for pr in MIGRATION_PRS:
        commit = _find_commit(pr)
        if commit is None:
            raise HistoryUnavailable(
                f"migration commit for PR #{pr} not found (shallow clone? re-run with full history)"
            )
        for old, new in _root_test_renames(commit):
            if _top_level_source_exists(old):
                derived[old] = new
    return dict(sorted(derived.items()))


def committed_map() -> dict[str, str]:
    """Load ``_MIGRATED_TEST_MAP`` from the committed selector module."""
    selector_path = REPO_ROOT / SELECTOR_REL
    spec = importlib.util.spec_from_file_location("nomic_ci_test_selector", selector_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {SELECTOR_REL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module._MIGRATED_TEST_MAP)


def format_map(mapping: dict[str, str]) -> str:
    """Render a mapping as the ``_MIGRATED_TEST_MAP`` Python literal block."""
    lines = ["_MIGRATED_TEST_MAP = {  # {old_root_path: new_subdir_path}"]
    for old, new in sorted(mapping.items()):
        lines.append(f'    "{old}": "{new}",')
    lines.append("}")
    return "\n".join(lines)


def diff_maps(
    committed: dict[str, str], derived: dict[str, str]
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(extra, missing, changed)`` keys between committed and derived."""
    extra = sorted(set(committed) - set(derived))
    missing = sorted(set(derived) - set(committed))
    changed = sorted(k for k in set(committed) & set(derived) if committed[k] != derived[k])
    return extra, missing, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive/verify the migrated-test map")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed map diverges from the derived map.",
    )
    group.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print the freshly-derived map (default action).",
    )
    args = parser.parse_args(argv)

    try:
        derived = derive_map()
    except HistoryUnavailable as exc:
        print(f"::warning::cannot derive migration map: {exc}")
        return 0

    if not args.check:
        print(format_map(derived))
        return 0

    committed = committed_map()
    if committed == derived:
        print(f"OK: committed _MIGRATED_TEST_MAP matches the derived map ({len(derived)} entries).")
        return 0

    extra, missing, changed = diff_maps(committed, derived)
    print("ERROR: committed _MIGRATED_TEST_MAP diverges from the P1 migration commits:")
    for key in extra:
        print(
            f"  ::error::dead entry (no top-level aragora source / not a migration rename): {key}"
        )
    for key in missing:
        print(f"  ::error::missing reachable entry: {key} -> {derived[key]}")
    for key in changed:
        print(
            f"  ::error::wrong destination for {key}: committed={committed[key]} derived={derived[key]}"
        )
    print(f"\nUpdate _MIGRATED_TEST_MAP in {SELECTOR_REL} to match (regenerate with")
    print(f"`python3 {GENERATOR_REL}`).\n\nExpected map:")
    print(format_map(derived))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
