#!/usr/bin/env python3
"""Classify every top-level module under aragora/ into a maturity tier.

Produces aragora/module_tiers.yaml as the canonical truth surface for
what is shipped-and-live vs what exists-but-is-not-yet-real. Addresses
issue #6505 follow-on (T2.1 of cold-auditor truth surface lane):
public docs currently list all ~135 top-level modules as if each were
a product surface, which overclaims significantly.

Tier definitions (applied in order; first match wins):

  core          - Imported by >= CORE_IMPORT_THRESHOLD other modules
                  AND has >= CORE_TEST_THRESHOLD test files referencing
                  it. Directly exercised by the mainline debate flow.

  integrated    - Imported by >= INTEGRATED_IMPORT_THRESHOLD other
                  modules OR has >= INTEGRATED_TEST_THRESHOLD test
                  files referencing it. Wired in, covered, not
                  necessarily on the main debate path.

  experimental  - Exists with code but very few importers and very
                  few tests. May be a scaffold, a research branch, or
                  an opt-in extension.

  deprecated    - No importers AND no tests AND not listed in the
                  MANUAL_PROMOTIONS map. Candidate for removal.

Manual overrides: MANUAL_TIER_OVERRIDES lets the founder pin a module
to a specific tier regardless of evidence (e.g. a newly-landed
module hasn't accrued test coverage yet but IS core; or a well-tested
module is actually deprecated).

Usage:
    python scripts/regenerate_module_tiers.py              # rewrite yaml
    python scripts/regenerate_module_tiers.py --check      # fail if drift
    python scripts/regenerate_module_tiers.py --json       # emit JSON

This script is the mechanical truth surface: if the yaml disagrees
with another doc about whether module X is core vs experimental, the
yaml wins and the other doc is stale.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIERS_YAML = REPO_ROOT / "aragora" / "module_tiers.yaml"

# Classification thresholds. These are deliberately loose — catching
# "obviously core" vs "obviously experimental" is more valuable than
# drawing a perfect line.
CORE_IMPORT_THRESHOLD = 50
CORE_TEST_THRESHOLD = 50
INTEGRATED_IMPORT_THRESHOLD = 10
INTEGRATED_TEST_THRESHOLD = 5

# Manual overrides. Use sparingly; prefer letting evidence speak.
# Format: module_name -> (tier, reason)
MANUAL_TIER_OVERRIDES: dict[str, tuple[str, str]] = {
    # Core by design even if evidence lags
    "core": ("core", "root type hierarchy + Agent base class"),
    "debate": ("core", "Arena + DebateProtocol + consensus — mainline flow"),
    "agents": ("core", "Agent factory, 43 registered types"),
    "memory": ("core", "ContinuumMemory + CritiqueStore — persistent state"),
    "ranking": ("core", "ELO skill tracking — feeds team selection"),
    # Infrastructure that IS shipped but reads as 'utility' by pure import count
    "config": ("core", "Pydantic settings + allowlist — load bearing"),
    "cli": ("core", "aragora CLI — primary developer surface"),
    "server": ("core", "FastAPI server + 3K+ API operations"),
    "storage": ("core", "persistence layer"),
    "db": ("core", "database abstraction"),
    # Shipped non-Python surfaces that pure Python import/test counts underweight.
    "live": ("integrated", "Next.js frontend app — tracked outside Python import graph"),
}


@dataclass
class ModuleClassification:
    name: str
    path: Path
    tier: str
    importer_count: int
    test_file_count: int
    py_file_count: int
    override_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "importer_count": self.importer_count,
            "test_file_count": self.test_file_count,
            "py_file_count": self.py_file_count,
            "override_reason": self.override_reason,
        }


@dataclass
class TierReport:
    modules: list[ModuleClassification] = field(default_factory=list)
    thresholds: dict[str, int] = field(default_factory=dict)

    def by_tier(self) -> dict[str, list[ModuleClassification]]:
        out: dict[str, list[ModuleClassification]] = defaultdict(list)
        for m in self.modules:
            out[m.tier].append(m)
        for tier in out:
            out[tier].sort(key=lambda x: x.name)
        return out


def _git_ls_files(*pathspecs: str) -> list[Path]:
    """Return git-tracked files under pathspecs (repo-relative)."""
    cmd = ["git", "ls-files", "--", *pathspecs]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def _top_level_modules() -> list[str]:
    """Top-level packages under aragora/ that contain .py files."""
    tracked = _git_ls_files("aragora")
    top: set[str] = set()
    for p in tracked:
        # aragora/<module>/... or aragora/<module>/file.py
        if len(p.parts) >= 3 and p.parts[0] == "aragora":
            mod = p.parts[1]
            if mod.startswith("_") or mod.startswith("."):
                continue
            top.add(mod)
    return sorted(top)


_IMPORT_REFERENCE_RE = re.compile(r"(?:from\s+aragora\.(\w+)|import\s+aragora\.(\w+))")


def _build_import_index(
    files: list[Path],
) -> dict[Path, set[str]]:
    """For each file, extract the set of aragora.X modules it references.

    O(N) pass over files, O(1) lookup per (module, file) query. Avoids
    the O(M * N) cost of a naive per-module scan.
    """
    index: dict[Path, set[str]] = {}
    for p in files:
        try:
            text = (REPO_ROOT / p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found: set[str] = set()
        for m in _IMPORT_REFERENCE_RE.finditer(text):
            mod = m.group(1) or m.group(2)
            if mod:
                found.add(mod)
        if found:
            index[p] = found
    return index


def _count_importers(
    module: str,
    py_files: list[Path],
    import_index: dict[Path, set[str]],
) -> int:
    """Count aragora/ .py files that import from aragora.<module>.

    Excludes files living inside aragora/<module>/ itself (self-imports
    don't count toward "external importers").
    """
    count = 0
    for p in py_files:
        if p.parts[:2] == ("aragora", module):
            continue
        if module in import_index.get(p, ()):
            count += 1
    return count


def _count_test_files(
    module: str,
    test_files: list[Path],
    import_index: dict[Path, set[str]],
) -> int:
    """Count test files referencing aragora.<module>."""
    return sum(1 for p in test_files if module in import_index.get(p, ()))


def _count_py_files(module: str, py_files: list[Path]) -> int:
    """Count .py files living under aragora/<module>/."""
    return sum(1 for p in py_files if p.parts[:2] == ("aragora", module))


def _classify(
    module: str,
    importer_count: int,
    test_file_count: int,
) -> str:
    if module in MANUAL_TIER_OVERRIDES:
        return MANUAL_TIER_OVERRIDES[module][0]

    if importer_count == 0 and test_file_count == 0:
        return "deprecated"

    if importer_count >= CORE_IMPORT_THRESHOLD and test_file_count >= CORE_TEST_THRESHOLD:
        return "core"

    if (
        importer_count >= INTEGRATED_IMPORT_THRESHOLD
        or test_file_count >= INTEGRATED_TEST_THRESHOLD
    ):
        return "integrated"

    return "experimental"


def gather_tiers() -> TierReport:
    modules = _top_level_modules()
    all_py = _git_ls_files("aragora")
    py_files = [p for p in all_py if p.suffix == ".py"]
    test_files = [
        p for p in _git_ls_files("tests") if p.suffix == ".py" and p.name.startswith("test_")
    ]

    # Build import index once — O(N) file reads, then O(1) queries per
    # (module, file). Without this, classification is O(M * N) where
    # M=135 modules and N=4000+ files, which times out in CI.
    import_index = _build_import_index(py_files + test_files)

    classifications: list[ModuleClassification] = []
    for mod in modules:
        importer_count = _count_importers(mod, py_files, import_index)
        test_file_count = _count_test_files(mod, test_files, import_index)
        py_file_count = _count_py_files(mod, py_files)
        tier = _classify(mod, importer_count, test_file_count)
        override_reason = (
            MANUAL_TIER_OVERRIDES.get(mod, ("", ""))[1] if mod in MANUAL_TIER_OVERRIDES else ""
        )
        classifications.append(
            ModuleClassification(
                name=mod,
                path=Path("aragora") / mod,
                tier=tier,
                importer_count=importer_count,
                test_file_count=test_file_count,
                py_file_count=py_file_count,
                override_reason=override_reason,
            )
        )

    return TierReport(
        modules=classifications,
        thresholds={
            "core_import_threshold": CORE_IMPORT_THRESHOLD,
            "core_test_threshold": CORE_TEST_THRESHOLD,
            "integrated_import_threshold": INTEGRATED_IMPORT_THRESHOLD,
            "integrated_test_threshold": INTEGRATED_TEST_THRESHOLD,
        },
    )


def render_yaml(report: TierReport) -> str:
    lines: list[str] = []
    lines.append("# Aragora Module Tier Registry")
    lines.append("# ==============================")
    lines.append("#")
    lines.append("# AUTO-GENERATED by scripts/regenerate_module_tiers.py. Do not edit by hand.")
    lines.append("# Every top-level package under aragora/ is classified by maturity.")
    lines.append("#")
    lines.append("# Tiers:")
    lines.append("#   core         — mainline debate flow surfaces, load-bearing")
    lines.append("#   integrated   — shipped, tested, wired in; not mainline debate path")
    lines.append("#   experimental — scaffolded; limited importers or tests")
    lines.append("#   deprecated   — no importers, no tests; removal candidate")
    lines.append("#")
    lines.append("# Public-facing docs (README, EXTENDED_README, START_HERE, capability")
    lines.append("# generators) should default to showing only core + integrated, with")
    lines.append("# explicit opt-in for experimental/deprecated.")
    lines.append("")
    lines.append("version: 1")
    lines.append("generator: scripts/regenerate_module_tiers.py")
    lines.append("")
    lines.append("thresholds:")
    for k, v in report.thresholds.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    summary: dict[str, int] = defaultdict(int)
    for m in report.modules:
        summary[m.tier] += 1

    lines.append("summary:")
    for tier in ("core", "integrated", "experimental", "deprecated"):
        lines.append(f"  {tier}: {summary.get(tier, 0)}")
    lines.append(f"  total: {len(report.modules)}")
    lines.append("")

    lines.append("modules:")
    by_tier = report.by_tier()
    for tier in ("core", "integrated", "experimental", "deprecated"):
        items = by_tier.get(tier, [])
        if not items:
            continue
        lines.append(f"  # --- {tier} ({len(items)}) ---")
        for m in items:
            lines.append(f"  - name: {m.name}")
            lines.append(f"    tier: {tier}")
            lines.append(f"    py_files: {m.py_file_count}")
            lines.append(f"    importer_count: {m.importer_count}")
            lines.append(f"    test_file_count: {m.test_file_count}")
            if m.override_reason:
                ro = m.override_reason.replace('"', '\\"')
                lines.append(f'    override_reason: "{ro}"')
        lines.append("")
    return "\n".join(lines)


def parse_current(yaml_path: Path) -> dict[str, dict]:
    """Parse the previous yaml for a module->snapshot dict."""
    if not yaml_path.exists():
        return {}
    text = yaml_path.read_text()
    modules: dict[str, dict] = {}
    current: dict | None = None
    for line in text.splitlines():
        m = re.match(r"\s+- name:\s+(\S+)", line)
        if m:
            current = {"name": m.group(1)}
            modules[m.group(1)] = current
            continue
        if current is None:
            continue
        m = re.match(r"\s+(\w+):\s+(.+)", line)
        if m:
            key = m.group(1)
            val = m.group(2).strip().strip('"')
            try:
                current[key] = int(val)
            except ValueError:
                current[key] = val
    return modules


def check_drift(report: TierReport) -> tuple[bool, list[str]]:
    current = parse_current(TIERS_YAML)
    drifts: list[str] = []
    seen: set[str] = set()
    for m in report.modules:
        seen.add(m.name)
        prev = current.get(m.name)
        if prev is None:
            drifts.append(f"NEW module: {m.name} -> {m.tier}")
            continue
        prev_tier = prev.get("tier")
        if prev_tier != m.tier:
            drifts.append(
                f"TIER CHANGE: {m.name} {prev_tier!r} -> {m.tier!r} "
                f"(imports {prev.get('importer_count')} -> {m.importer_count}, "
                f"tests {prev.get('test_file_count')} -> {m.test_file_count})"
            )
    for name in current.keys() - seen:
        drifts.append(f"REMOVED module: {name}")
    return (len(drifts) > 0), drifts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any module's tier drifted.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of writing yaml.",
    )
    args = parser.parse_args()

    report = gather_tiers()

    if args.json:
        print(
            json.dumps(
                {
                    "thresholds": report.thresholds,
                    "modules": [m.as_dict() for m in report.modules],
                },
                indent=2,
            )
        )
        return 0

    if args.check:
        drifted, drifts = check_drift(report)
        if drifted:
            print("Module tier drift detected:")
            for d in drifts:
                print(f"  {d}")
            print()
            print(
                "Run `python scripts/regenerate_module_tiers.py` to refresh "
                "aragora/module_tiers.yaml and commit the result."
            )
            return 1
        print("No tier drift.")
        return 0

    TIERS_YAML.parent.mkdir(parents=True, exist_ok=True)
    TIERS_YAML.write_text(render_yaml(report))
    print(f"Wrote {TIERS_YAML.relative_to(REPO_ROOT)} with {len(report.modules)} modules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
