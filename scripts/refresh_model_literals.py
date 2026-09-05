#!/usr/bin/env python3
"""Rewrite retired model-ID literals to their current ids, or check that none remain.

Consumes ``UPGRADES`` and ``RETIRED_PATTERN`` from
``aragora.models.upgrade_map`` and ``CATALOG`` from
``aragora.models.catalog`` to map a retired literal to its current
spelling: a literal that was an OpenRouter slug (contains ``/``) is
rewritten to the new OpenRouter slug, a bare id to the new direct id.

``RETIRED_PATTERN`` is already built with token-boundary lookarounds (see
``aragora/models/upgrade_map.py``) so a retired key that is a literal
prefix of a longer active spelling — e.g. ``"claude-fable-5"`` inside
active ``"claude-fable-5-1"`` — never falsely matches. This script reuses
that pattern directly rather than re-wrapping it in another boundary
layer.

One class of retired literal is deliberately NOT rewritten: a bare
(native-shaped) spelling whose current row is reachable only through
OpenRouter has no real native id to be rewritten to. ``--write`` leaves it
exactly as written, and ``--check`` reports it under a separate
"unresolvable" list that does not affect the exit code. See
``replacement()``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.models.catalog import CATALOG  # noqa: E402
from aragora.models.upgrade_map import RETIRED_PATTERN, UPGRADES  # noqa: E402

SKIP_DIRS = {".git", "node_modules", ".worktrees", "__pycache__", ".venv", "dist", "build"}
SKIP_SUFFIXES = {".lock", ".png", ".jpg", ".pdf", ".ico", ".woff", ".woff2", ".pyc"}
SKIP_NAMES = {
    "package-lock.json",
    "uv.lock",
    "yarn.lock",
    "pnpm-lock.yaml",
    "catalog_snapshot.json",
    "upgrade_map.py",
}

# Repo-relative paths that legitimately contain retired model-id literals
# on purpose, and must therefore never be rewritten by --write or reported
# as offenders by --check. Matched by path SUFFIX (a trailing "/" entry
# matches anything under that directory), so this works regardless of the
# cwd the sweep is invoked from.
#
#   - aragora/models/catalog.py, upgrade_map.py, catalog_snapshot.json,
#     pricing_mirror.py: these ARE the retired-id source of truth — the
#     catalog rows with retired=True, the UPGRADES map itself (keyed by
#     the very literals this script hunts for), its generated JSON
#     mirror, and the pricing table that old receipts still resolve
#     through by their original spelling.
#   - aragora/billing/usage.py, aragora/billing/debate_costs.py,
#     aragora/services/metering_models.py, aragora/pdb/real_invoker.py,
#     aragora/routing/provider_config.py,
#     aragora/server/handlers/debates/cost_estimation.py: legacy pricing
#     keys and static routing hand-rows that old receipts and in-flight
#     cost estimates still resolve through by their original spelling.
#   - tests/models/: unit tests that assert retired ids on purpose (e.g.
#     RETIRED_PATTERN / UPGRADES coverage tests in
#     tests/models/test_upgrade_map.py).
#   - scripts/refresh_model_literals.py: this script's own source, which
#     necessarily names retired ids in comments, constants, and its test.
#   - tests/scripts/test_refresh_model_literals.py: not in the original
#     skip list, added deliberately — this test's fixtures are retired-id
#     string literals embedded in the test *source* (not just written at
#     runtime), so a --write pass over "tests" would rewrite them in
#     place and silently gut what the test verifies, the same hazard
#     tests/models/ and this script's own source are protected against.
SKIP_PATHS: tuple[str, ...] = (
    "aragora/models/catalog.py",
    "aragora/models/upgrade_map.py",
    "aragora/models/catalog_snapshot.json",
    "aragora/models/pricing_mirror.py",
    "aragora/billing/usage.py",
    "aragora/billing/debate_costs.py",
    "aragora/services/metering_models.py",
    "aragora/pdb/real_invoker.py",
    "aragora/routing/provider_config.py",
    "aragora/server/handlers/debates/cost_estimation.py",
    "tests/models/",
    "scripts/refresh_model_literals.py",
    "tests/scripts/test_refresh_model_literals.py",
)

DEFAULT_ALLOWLIST = REPO_ROOT / "scripts" / "baselines" / "retired_model_literals_allowlist.txt"


def replacement(old: str) -> str | None:
    """Current literal for a retired spelling, or ``None`` when the retired
    literal has no honest replacement in the SHAPE it was written in.

    An OpenRouter slug (contains ``/``) is rewritten to the new slug; a bare
    id to the new direct id. The exception is a BARE literal whose target
    row has ``provider == "openrouter"`` — a family Aragora reaches only
    through OpenRouter. ``ModelSpec.direct_id`` is a documented placeholder
    on those rows, "NOT a code any native endpoint would accept" (see the
    field's docstring in aragora/models/catalog.py), so rewriting e.g. the
    deliberately-kept ``deepseek-cli`` default ``deepseek-v4-pro`` to
    ``deepseek-v4-pro-0813`` would swap a working native model code for a
    slug that 400s on the native API (2026-09-05 merge-gate finding C-P3 on
    #9989). Such a literal is left exactly as written by ``--write`` and
    reported by ``--check`` as UNRESOLVABLE rather than as an offender: it
    is a real gap (the catalog owes that family a native row), but it is not
    something this sweep can fix, so it must not gate the sweep.
    """
    spec = CATALOG[UPGRADES[old]]
    if "/" in old:
        return spec.openrouter_id
    if spec.provider == "openrouter":
        return None
    return spec.direct_id


def _sub_one(match: "re.Match[str]") -> str:
    """``RETIRED_PATTERN.sub`` callback: rewrite, or keep the literal when
    it has no honest native replacement (see ``replacement``)."""
    literal = match.group(0)
    new = replacement(literal)
    return literal if new is None else new


def _is_skip_path(f: Path) -> bool:
    """True if ``f`` matches a SKIP_PATHS entry by suffix.

    ``f.resolve().as_posix()`` is always an absolute path, so a directory
    entry (trailing "/") is matched as "/<entry>" appearing anywhere in
    that path, and a file entry is matched as an exact path suffix.
    """
    posix = f.resolve().as_posix()
    for skip in SKIP_PATHS:
        if skip.endswith("/"):
            if f"/{skip}" in posix:
                return True
        elif posix == skip or posix.endswith(f"/{skip}"):
            return True
    return False


def _allowlist_key(f: Path) -> str:
    """Normalize ``f`` to the form allowlist entries are written in.

    The allowlist (``scripts/baselines/retired_model_literals_allowlist.txt``)
    is generated via ``git ls-files`` from the repo root, so its entries are
    repo-root-relative POSIX paths regardless of the cwd or --paths spelling
    (relative or absolute) the sweep happens to be invoked with. Normalize
    the scanned file the same way — relative to REPO_ROOT when it is under
    the repo, else its absolute POSIX path — so membership testing doesn't
    silently no-op just because the sweep ran from a different directory.
    """
    resolved = f.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def iter_files(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        root = Path(p)
        if root.is_file():
            # A file root has no "below the scan root" hierarchy at all, so
            # the SKIP_DIRS check below (which only ever excludes
            # directories *below* a scanned root) does not apply here.
            f = root
            if f.name in SKIP_NAMES or f.suffix in SKIP_SUFFIXES:
                continue
            if _is_skip_path(f):
                continue
            found.append(f)
            continue
        for f in root.rglob("*"):
            if not f.is_file() or f.name in SKIP_NAMES or f.suffix in SKIP_SUFFIXES:
                continue
            # SKIP_DIRS must only ever exclude directories *below* the scan
            # root that was passed in --paths (e.g. a nested node_modules/
            # or .venv/ found while walking that root) — never an ancestor
            # *above* it. Checking f.parts directly (as an earlier version
            # of this script did) inspects the WHOLE path, including
            # whatever lies above the scan root; if that ancestry happens
            # to include a directory named e.g. ".worktrees" (as this
            # repo's own dev checkouts do) or ".venv", an absolute --paths
            # would silently scan nothing. Relative-to-root parts contain
            # only what's actually below the given root.
            rel_parts = f.relative_to(root).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue
            if _is_skip_path(f):
                continue
            found.append(f)
    # Deterministic order: --check output and --write iteration order must
    # not depend on filesystem/rglob discovery order (which varies by OS,
    # directory entry layout, and run-to-run).
    found.sort(key=lambda f: f.as_posix())
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paths",
        nargs="+",
        default=["aragora", "scripts", "sdk", "docs", "docs-site", "tests", "README.md"],
    )
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    a = ap.parse_args(argv)
    if a.write == a.check:
        print("choose exactly one of --write / --check", file=sys.stderr)
        return 2

    allow: set[str] = set()
    allow_path = Path(a.allowlist)
    if allow_path.exists():
        allow = {
            ln.strip()
            for ln in allow_path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        }

    offenders: list[tuple[str, int, str]] = []
    # Retired literals this sweep deliberately cannot rewrite: a bare
    # (native-shaped) spelling of a row Aragora reaches only through
    # OpenRouter. Reported separately and NEVER counted as an offender —
    # see replacement().
    unresolvable: list[tuple[str, int, str]] = []
    changed = 0
    for f in iter_files(a.paths):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _allowlist_key(f) in allow:
            continue
        if not RETIRED_PATTERN.search(text):
            continue
        if a.check:
            for i, line in enumerate(text.splitlines(), 1):
                for m in RETIRED_PATTERN.finditer(line):
                    literal = m.group(0)
                    bucket = offenders if replacement(literal) is not None else unresolvable
                    bucket.append((str(f), i, literal))
        else:
            new = RETIRED_PATTERN.sub(_sub_one, text)
            if new != text:
                f.write_text(new, encoding="utf-8")
                changed += 1

    if a.check:
        offenders.sort(key=lambda o: (o[0], o[1], o[2]))
        unresolvable.sort(key=lambda o: (o[0], o[1], o[2]))
        for path, ln, lit in offenders:
            print(f"{path}:{ln}: retired model id {lit}")
        if unresolvable:
            print("unresolvable: native spelling of an OpenRouter-only row")
            for path, ln, lit in unresolvable:
                print(f"{path}:{ln}: unresolvable model id {lit}")
        print(f"{len(offenders)} retired literal(s) outside allowlist")
        print(f"{len(unresolvable)} unresolvable literal(s) (not counted as offenders)")
        # Exit code is deliberately driven by OFFENDERS only.
        return 1 if offenders else 0

    print(f"rewrote {changed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
