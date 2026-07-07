#!/usr/bin/env python3
"""Advisory checker for the Intended Architecture charter.

The checker consumes ``docs/architecture/charters.yaml`` and reports changed
paths, symbol touches, and new import edges that conflict with chartered
REMOVED/PENDING/PARKED/EXCLUSION entries. It is intentionally advisory-only
until wired into a reviewed gate.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


BINDING_STATES = {"REMOVED", "EXCLUSION"}
FREEZE_STATES = {"PENDING", "EXPIRING"}
PARK_STATES = {"PARKED"}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class ChangedFile:
    path: str
    added_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class CharterEntry:
    id: str
    state: str
    paths: tuple[str, ...]
    symbols: tuple[str, ...] = ()
    kept_symbols: tuple[str, ...] = ()
    binding_in_draft: bool = False
    owner: str | None = None
    deadline: str | None = None
    evidence: str = ""


@dataclass(frozen=True)
class CharterFinding:
    entry_id: str
    state: str
    severity: str
    binding: bool
    changed_path: str
    reason: str
    matched_path: str | None = None
    matched_symbol: str | None = None
    evidence: str = ""


@dataclass(frozen=True)
class CharterCheckResult:
    charter_path: str
    charter_status: str
    base_ref: str | None
    head_ref: str | None
    changed_paths: tuple[str, ...]
    findings: tuple[CharterFinding, ...]
    summary: dict[str, int] = field(default_factory=dict)

    @property
    def has_blocking_findings(self) -> bool:
        return any(f.severity == "blocking" for f in self.findings)


def _repo_root(start: Path | None = None) -> Path:
    cwd = start or Path.cwd()
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return Path(completed.stdout.strip())


def _run_git(args: Sequence[str], *, repo: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("./")


def _module_to_paths(module: str) -> set[str]:
    rel = module.replace(".", "/")
    return {f"{rel}.py", f"{rel}/__init__.py"}


def _path_matches(pattern: str, changed_path: str) -> bool:
    pattern = _normalize_path(pattern)
    changed_path = _normalize_path(changed_path)
    if not pattern:
        return False
    if any(ch in pattern for ch in "*?[]"):
        return fnmatch.fnmatch(changed_path, pattern)
    if pattern.endswith("/"):
        return changed_path.startswith(pattern)
    return changed_path == pattern


def _load_entries(charter_path: Path) -> tuple[str, list[CharterEntry]]:
    data = yaml.safe_load(charter_path.read_text()) or {}
    status = str((data.get("meta") or {}).get("status") or "UNKNOWN").upper()
    entries: list[CharterEntry] = []
    for raw in data.get("registry") or []:
        entries.append(
            CharterEntry(
                id=str(raw.get("id") or ""),
                state=str(raw.get("state") or "").upper(),
                paths=tuple(str(p) for p in (raw.get("paths") or [])),
                symbols=tuple(str(s) for s in (raw.get("symbols") or [])),
                kept_symbols=tuple(str(s) for s in (raw.get("kept_symbols") or [])),
                binding_in_draft=bool(raw.get("binding_in_draft")),
                owner=raw.get("owner"),
                deadline=raw.get("deadline"),
                evidence=str(raw.get("evidence") or ""),
            )
        )
    return status, entries


def _is_binding(entry: CharterEntry, charter_status: str) -> bool:
    if charter_status.startswith("RATIFIED"):
        return True
    if entry.binding_in_draft:
        return True
    if entry.state in BINDING_STATES and entry.id.startswith("CHR-E-"):
        return True
    return False


def _symbol_parts(symbol: str) -> tuple[str, str] | None:
    if ":" not in symbol:
        return None
    module, name = symbol.split(":", 1)
    module = module.strip()
    name = name.strip()
    if not module or not name:
        return None
    return module, name


def _entry_symbols_for_path(entry: CharterEntry, changed_path: str) -> tuple[str, ...]:
    matches: list[str] = []
    normalized = _normalize_path(changed_path)
    for symbol in entry.symbols:
        parts = _symbol_parts(symbol)
        if not parts:
            continue
        module, _name = parts
        if normalized in _module_to_paths(module):
            matches.append(symbol)
    return tuple(matches)


def _extract_defined_or_imported_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(TOKEN_RE.findall(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _symbol_touched(symbol: str, changed_file: ChangedFile, repo: Path) -> bool:
    parts = _symbol_parts(symbol)
    if not parts:
        return False
    _module, name = parts
    haystack = "\n".join(changed_file.added_lines)
    if name in TOKEN_RE.findall(haystack):
        return True
    path = repo / changed_file.path
    if path.exists() and path.suffix == ".py":
        return name in _extract_defined_or_imported_names(path.read_text())
    return False


def _new_imported_modules(lines: Iterable[str]) -> set[str]:
    modules: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if line.startswith("from "):
            match = re.match(r"from\s+([A-Za-z_][\w.]*)\s+import\s+", line)
            if match:
                modules.add(match.group(1))
        elif line.startswith("import "):
            imported = line.removeprefix("import ").split("#", 1)[0]
            for part in imported.split(","):
                module = part.strip().split(" as ", 1)[0].strip()
                if module:
                    modules.add(module)
    return modules


def _path_to_module_prefix(path: str) -> str | None:
    path = _normalize_path(path)
    if not path.startswith("aragora/"):
        return None
    if path.endswith("/"):
        return path.rstrip("/").replace("/", ".")
    if path.endswith("/__init__.py"):
        return path.removesuffix("/__init__.py").replace("/", ".")
    if path.endswith(".py"):
        return path.removesuffix(".py").replace("/", ".")
    return None


def _import_matches_entry(entry: CharterEntry, module: str) -> str | None:
    for path in entry.paths:
        prefix = _path_to_module_prefix(path)
        if prefix and (module == prefix or module.startswith(f"{prefix}.")):
            return path
    return None


def _finding_severity(entry: CharterEntry, binding: bool) -> str:
    if entry.state in PARK_STATES:
        return "operator"
    if binding:
        return "blocking"
    return "advisory"


def _classify_path_entry(
    *,
    entry: CharterEntry,
    changed_file: ChangedFile,
    repo: Path,
    binding: bool,
) -> list[CharterFinding]:
    findings: list[CharterFinding] = []
    matched_paths = [p for p in entry.paths if _path_matches(p, changed_file.path)]
    if not matched_paths:
        return findings

    severity = _finding_severity(entry, binding)
    path_scoped_symbols = _entry_symbols_for_path(entry, changed_file.path)
    if path_scoped_symbols:
        touched_symbols = [
            symbol
            for symbol in path_scoped_symbols
            if symbol not in entry.kept_symbols and _symbol_touched(symbol, changed_file, repo)
        ]
        if not touched_symbols:
            return [
                CharterFinding(
                    entry_id=entry.id,
                    state=entry.state,
                    severity="advisory",
                    binding=binding,
                    changed_path=changed_file.path,
                    matched_path=matched_paths[0],
                    reason="symbol-scoped entry path changed but chartered symbol was not touched",
                    evidence=entry.evidence,
                )
            ]
        for symbol in touched_symbols:
            findings.append(
                CharterFinding(
                    entry_id=entry.id,
                    state=entry.state,
                    severity=severity,
                    binding=binding,
                    changed_path=changed_file.path,
                    matched_path=matched_paths[0],
                    matched_symbol=symbol,
                    reason=f"{entry.state.lower()} symbol touched",
                    evidence=entry.evidence,
                )
            )
        return findings

    findings.append(
        CharterFinding(
            entry_id=entry.id,
            state=entry.state,
            severity=severity,
            binding=binding,
            changed_path=changed_file.path,
            matched_path=matched_paths[0],
            reason=f"{entry.state.lower()} path changed",
            evidence=entry.evidence,
        )
    )
    return findings


def classify_changes(
    *,
    charter_path: Path,
    changed_files: Sequence[ChangedFile],
    repo: Path | None = None,
    base_ref: str | None = None,
    head_ref: str | None = None,
) -> CharterCheckResult:
    repo = repo or _repo_root()
    charter_status, entries = _load_entries(charter_path)
    findings: list[CharterFinding] = []

    for changed_file in changed_files:
        imported_modules = _new_imported_modules(changed_file.added_lines)
        for entry in entries:
            if not entry.id:
                continue
            binding = _is_binding(entry, charter_status)
            if entry.state in (BINDING_STATES | FREEZE_STATES | PARK_STATES):
                findings.extend(
                    _classify_path_entry(
                        entry=entry,
                        changed_file=changed_file,
                        repo=repo,
                        binding=binding,
                    )
                )
            if entry.state in FREEZE_STATES:
                for module in imported_modules:
                    matched_path = _import_matches_entry(entry, module)
                    if not matched_path:
                        continue
                    findings.append(
                        CharterFinding(
                            entry_id=entry.id,
                            state=entry.state,
                            severity=_finding_severity(entry, binding),
                            binding=binding,
                            changed_path=changed_file.path,
                            matched_path=matched_path,
                            reason=f"new importer/caller for {entry.state.lower()} surface {module}",
                            evidence=entry.evidence,
                        )
                    )

    summary: dict[str, int] = {}
    for finding in findings:
        summary[finding.severity] = summary.get(finding.severity, 0) + 1
    return CharterCheckResult(
        charter_path=str(charter_path),
        charter_status=charter_status,
        base_ref=base_ref,
        head_ref=head_ref,
        changed_paths=tuple(changed_file.path for changed_file in changed_files),
        findings=tuple(findings),
        summary=summary,
    )


def _changed_paths(repo: Path, base_ref: str, head_ref: str) -> list[str]:
    out = _run_git(["diff", "--name-only", f"{base_ref}...{head_ref}"], repo=repo)
    return [_normalize_path(line) for line in out.splitlines() if line.strip()]


def _changed_file(repo: Path, path: str, base_ref: str, head_ref: str) -> ChangedFile:
    try:
        diff = _run_git(
            ["diff", "--unified=0", "--no-ext-diff", f"{base_ref}...{head_ref}", "--", path],
            repo=repo,
        )
    except subprocess.CalledProcessError:
        return ChangedFile(path=path)
    added_lines = []
    for line in diff.splitlines():
        if line.startswith("+++") or not line.startswith("+"):
            continue
        added_lines.append(line[1:])
    return ChangedFile(path=path, added_lines=tuple(added_lines))


def build_changed_files(
    *,
    repo: Path,
    base_ref: str,
    head_ref: str,
    paths: Sequence[str],
) -> list[ChangedFile]:
    selected = list(paths) if paths else _changed_paths(repo, base_ref, head_ref)
    return [
        _changed_file(repo, _normalize_path(path), base_ref, head_ref)
        for path in selected
        if _normalize_path(path)
    ]


def _as_json(result: CharterCheckResult) -> dict[str, Any]:
    return asdict(result) | {
        "blocking": result.has_blocking_findings,
        "finding_count": len(result.findings),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--charters", type=Path, default=None)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--changed-path", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve() if args.repo else _repo_root()
    charter_path = args.charters or repo / "docs/architecture/charters.yaml"
    changed_files = build_changed_files(
        repo=repo,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        paths=args.changed_path,
    )
    result = classify_changes(
        charter_path=charter_path,
        changed_files=changed_files,
        repo=repo,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
    )
    payload = _as_json(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"charter={result.charter_status} changed={len(result.changed_paths)} "
            f"findings={len(result.findings)} blocking={result.has_blocking_findings}"
        )
        for finding in result.findings:
            symbol = f" symbol={finding.matched_symbol}" if finding.matched_symbol else ""
            print(
                f"{finding.severity.upper()} {finding.entry_id} {finding.changed_path}: "
                f"{finding.reason}{symbol}"
            )
    return 1 if result.has_blocking_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
