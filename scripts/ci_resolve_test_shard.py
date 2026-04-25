#!/usr/bin/env python3
"""Resolve concrete pytest path arguments for a CI test shard.

Why this exists:
Pytest's `--ignore-glob` matcher uses ``fnmatch`` against absolute paths,
which makes mutually-exclusive alphabetic shards extremely tricky to
express purely in YAML. Stacking many ``--ignore-glob`` patterns also
leaves the resulting argv hard to audit at a glance.

This helper translates a logical shard name (``handlers-amk-no-features``,
``debate-am`` and friends) into a concrete list of test files / dirs that
pytest can be invoked with directly, leaving collection itself simple,
deterministic, and easy to verify locally.

Usage::

    python3 scripts/ci_resolve_test_shard.py debate-am
    python3 scripts/ci_resolve_test_shard.py handlers-lz --check

Each shard prints a single line with the test paths to stdout. The
``--check`` flag additionally prints partition diagnostics on stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _entries(parent: Path) -> tuple[list[str], list[str]]:
    """Return (subdirs, top_level_test_files) under ``parent``.

    Both sorted alphabetically. ``__pycache__`` and ``__init__.py`` are
    filtered out so they never end up in a shard partition.
    """
    if not parent.exists():
        return [], []
    subdirs: list[str] = []
    files: list[str] = []
    for entry in sorted(parent.iterdir()):
        if entry.name.startswith("__pycache__"):
            continue
        if entry.is_dir():
            subdirs.append(entry.name)
        elif entry.is_file() and entry.name.startswith("test_") and entry.name.endswith(".py"):
            files.append(entry.name)
    return subdirs, files


def _starts_in(name: str, predicate: Callable[[str], bool]) -> bool:
    return bool(name) and predicate(name[0].lower())


def _alpha_range(low: str, high: str) -> Callable[[str], bool]:
    return lambda c: low <= c <= high


# Top-level test file alphabet uses the leading char of the *suffix* after
# the ``test_`` prefix (so ``test_admin.py`` is "a", not "t").
def _file_letter(name: str) -> str:
    suffix = name[len("test_") :] if name.startswith("test_") else name
    return suffix[0].lower() if suffix else ""


# ---------------------------------------------------------------------------
# Per-shard resolvers
# ---------------------------------------------------------------------------


def _under(parent: Path, names: Iterable[str]) -> list[str]:
    return [str((parent / name).relative_to(REPO_ROOT)) for name in names]


def shard_debate_am() -> list[str]:
    """Top-level debate test files starting a-m, no subdir tests."""
    parent = TESTS_ROOT / "debate"
    _, files = _entries(parent)
    in_am = _alpha_range("a", "m")
    return _under(parent, [f for f in files if in_am(_file_letter(f))])


def shard_debate_nz() -> list[str]:
    """Top-level debate test files starting n-z, no subdir tests."""
    parent = TESTS_ROOT / "debate"
    _, files = _entries(parent)
    in_nz = _alpha_range("n", "z")
    return _under(parent, [f for f in files if in_nz(_file_letter(f))])


def shard_debate_phases() -> list[str]:
    """All subdirectories under tests/debate/."""
    parent = TESTS_ROOT / "debate"
    subdirs, _ = _entries(parent)
    return _under(parent, subdirs)


def _server_handlers_root() -> Path:
    return TESTS_ROOT / "server" / "handlers"


def shard_server_handlers_am() -> list[str]:
    parent = _server_handlers_root()
    subdirs, files = _entries(parent)
    in_am = _alpha_range("a", "m")
    chosen = [s for s in subdirs if _starts_in(s.lstrip("_"), in_am)]
    chosen += [f for f in files if in_am(_file_letter(f))]
    return _under(parent, chosen)


def shard_server_handlers_nz() -> list[str]:
    parent = _server_handlers_root()
    subdirs, files = _entries(parent)
    in_nz = _alpha_range("n", "z")
    chosen = [s for s in subdirs if _starts_in(s.lstrip("_"), in_nz)]
    chosen += [f for f in files if in_nz(_file_letter(f))]
    return _under(parent, chosen)


def shard_server_rest() -> list[str]:
    """Everything under tests/server/ except tests/server/handlers/."""
    return ["tests/server", "--ignore=tests/server/handlers"]


def _handlers_root() -> Path:
    return TESTS_ROOT / "handlers"


def shard_handlers_features() -> list[str]:
    """Single largest subdir of tests/handlers/."""
    return ["tests/handlers/features"]


def shard_handlers_amk_no_features() -> list[str]:
    """A-K subdirs (excluding ``features``) + A-K top-level files.

    Underscore-prefixed subdirs (e.g. ``_oauth``) are grouped here so that
    every test in tests/handlers/ runs in exactly one shard.
    """
    parent = _handlers_root()
    subdirs, files = _entries(parent)
    in_ak = _alpha_range("a", "k")
    chosen_subdirs = [
        s for s in subdirs if s != "features" and (s.startswith("_") or _starts_in(s, in_ak))
    ]
    chosen_files = [f for f in files if in_ak(_file_letter(f))]
    return _under(parent, chosen_subdirs + chosen_files)


def shard_handlers_lz() -> list[str]:
    """L-Z subdirs + L-Z top-level files (no underscore-prefixed dirs)."""
    parent = _handlers_root()
    subdirs, files = _entries(parent)
    in_lz = _alpha_range("l", "z")
    chosen_subdirs = [s for s in subdirs if not s.startswith("_") and _starts_in(s, in_lz)]
    chosen_files = [f for f in files if in_lz(_file_letter(f))]
    return _under(parent, chosen_subdirs + chosen_files)


SHARDS: dict[str, Callable[[], list[str]]] = {
    "debate-am": shard_debate_am,
    "debate-nz": shard_debate_nz,
    "debate-phases": shard_debate_phases,
    "server-handlers-am": shard_server_handlers_am,
    "server-handlers-nz": shard_server_handlers_nz,
    "server-rest": shard_server_rest,
    "handlers-features": shard_handlers_features,
    "handlers-amk-no-features": shard_handlers_amk_no_features,
    "handlers-lz": shard_handlers_lz,
}


# ---------------------------------------------------------------------------
# Self-check (used by tests/CI sanity)
# ---------------------------------------------------------------------------


def _print_diagnostics(name: str, paths: list[str]) -> None:
    print(f"shard: {name}", file=sys.stderr)
    print(f"  resolved {len(paths)} entries", file=sys.stderr)
    for p in paths[:5]:
        print(f"  - {p}", file=sys.stderr)
    if len(paths) > 5:
        print(f"  ... and {len(paths) - 5} more", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", choices=sorted(SHARDS), help="logical shard name")
    parser.add_argument(
        "--check",
        action="store_true",
        help="print partition diagnostics on stderr",
    )
    args = parser.parse_args(argv)

    paths = SHARDS[args.shard]()
    if args.check:
        _print_diagnostics(args.shard, paths)
    print(" ".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
