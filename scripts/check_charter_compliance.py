#!/usr/bin/env python3
"""Advisory checker for chartered architecture removals and exclusions."""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import re
import subprocess
import sys
import token
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


DRAFT_BINDING_IDS = {
    "CHR-P4A-001",
    "CHR-P4A-002",
    "CHR-P4A-003",
    "CHR-P4A-004",
    "CHR-X-007",
}
ENFORCED_STATES = {"REMOVED", "EXCLUSION", "PENDING", "EXPIRING", "PARKED"}
FROM_IMPORT_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+(.+)$")
PLAIN_IMPORT_RE = re.compile(r"^\s*import\s+(.+)$")
HUNK_RE = re.compile(r"@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class CharterEntry:
    entry_id: str
    state: str
    paths: tuple[str, ...]
    symbols: tuple[str, ...]
    kept_symbols: tuple[str, ...]
    binding: str


@dataclass(frozen=True)
class AddedLine:
    path: str
    line_no: int | None
    line: str
    in_string_literal: bool = False
    code_line: str | None = None


@dataclass(frozen=True)
class Violation:
    binding: str
    entry_id: str
    state: str
    path: str
    line_no: int | None
    line: str
    reason: str
    authority_ids: list[str]


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    binding_violations: list[Violation]
    proposed_violations: list[Violation]
    violations: list[Violation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "binding_violations": [asdict(violation) for violation in self.binding_violations],
            "proposed_violations": [asdict(violation) for violation in self.proposed_violations],
            "violations": [asdict(violation) for violation in self.violations],
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_diff_path(raw: str) -> str | None:
    raw = raw.strip()
    if raw == "/dev/null":
        return None
    if raw.startswith("a/") or raw.startswith("b/"):
        raw = raw[2:]
    if raw.startswith("./"):
        raw = raw[2:]
    return raw or None


def _path_matches(pattern: str, path: str) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return False
    if pattern.startswith("./"):
        pattern = pattern[2:]
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return path == pattern or fnmatch.fnmatch(path, pattern)


def _path_to_module(path: str) -> str:
    if path.endswith("/__init__.py"):
        path = path[: -len("/__init__.py")]
    elif path.endswith(".py"):
        path = path[:-3]
    return path.replace("/", ".")


def _path_pattern_to_modules(pattern: str) -> tuple[str, ...]:
    pattern = pattern.strip().rstrip("/")
    if not pattern:
        return ()
    if pattern.endswith(".py") or pattern.endswith("/__init__.py"):
        return (_path_to_module(pattern),)
    return (pattern.replace("/", "."),)


def _symbol_module(symbol: str) -> str | None:
    if ":" not in symbol:
        return None
    return symbol.split(":", 1)[0]


def _symbol_export(symbol: str) -> str:
    export = symbol.split(":", 1)[-1]
    return export.rsplit(".", 1)[-1]


def _symbol_root_export(symbol: str) -> str:
    export = symbol.split(":", 1)[-1]
    return export.split(".", 1)[0]


def _symbol_export_roots(symbols: Iterable[str]) -> set[str]:
    roots: set[str] = set()
    for symbol in symbols:
        root = _symbol_root_export(symbol)
        if root:
            roots.add(root)
    return roots


def _is_top_level_symbol(symbol: str) -> bool:
    return "." not in symbol.split(":", 1)[-1]


def _parse_imported_names(imports: str) -> list[str]:
    cleaned = imports.split("#", 1)[0].strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    names: list[str] = []
    for part in cleaned.split(","):
        token = part.strip()
        if not token:
            continue
        names.append(token.split(" as ", 1)[0].strip())
    return names


def _from_import(line: str) -> tuple[str, list[str]] | None:
    match = FROM_IMPORT_RE.match(line)
    if not match:
        return None
    return match.group(1), _parse_imported_names(match.group(2))


def _plain_import_modules(line: str) -> list[str]:
    match = PLAIN_IMPORT_RE.match(line)
    if not match:
        return []
    modules: list[str] = []
    for part in match.group(1).split(","):
        token = part.strip()
        if not token:
            continue
        modules.append(token.split(" as ", 1)[0].strip())
    return modules


def _plain_import_aliases(line: str) -> dict[str, str]:
    match = PLAIN_IMPORT_RE.match(line)
    if not match:
        return {}
    aliases: dict[str, str] = {}
    for part in match.group(1).split(","):
        token = part.strip()
        if not token or " as " not in token:
            continue
        module, alias = [piece.strip() for piece in token.split(" as ", 1)]
        if module and alias:
            aliases[alias] = module
    return aliases


def _line_mentions_symbol(line: str, symbol: str) -> bool:
    export = _symbol_export(symbol)
    return bool(re.search(rf"\b{re.escape(export)}\b", line))


def _is_python_path(path: str) -> bool:
    return path.endswith((".py", ".pyi"))


_FSTRING_TOKEN_TYPES = {
    token_type
    for token_type in (
        getattr(token, "FSTRING_START", None),
        getattr(token, "FSTRING_MIDDLE", None),
        getattr(token, "FSTRING_END", None),
    )
    if token_type is not None
}
_NON_CODE_TOKEN_TYPES = {token.COMMENT, *_FSTRING_TOKEN_TYPES}


def _blank_token_span(
    masked_lines: list[list[str]], start: tuple[int, int], end: tuple[int, int]
) -> None:
    start_row, start_col = start
    end_row, end_col = end
    for row in range(start_row, end_row + 1):
        index = row - 1
        if index < 0 or index >= len(masked_lines):
            continue
        line = masked_lines[index]
        left = start_col if row == start_row else 0
        right = end_col if row == end_row else len(line)
        for column in range(max(0, left), min(len(line), right)):
            line[column] = " "


def _token_offset_positions(text: str, start: tuple[int, int]) -> list[tuple[int, int]]:
    row, column = start
    positions: list[tuple[int, int]] = []
    for char in text:
        positions.append((row, column))
        if char == "\n":
            row += 1
            column = 0
        else:
            column += 1
    return positions


def _restore_token_span(
    masked_lines: list[list[str]],
    token_text: str,
    token_start: tuple[int, int],
    start_offset: int,
    end_offset: int,
) -> None:
    positions = _token_offset_positions(token_text, token_start)
    for offset in range(max(0, start_offset), min(len(token_text), end_offset)):
        row, column = positions[offset]
        line_index = row - 1
        if line_index < 0 or line_index >= len(masked_lines):
            continue
        line = masked_lines[line_index]
        if 0 <= column < len(line):
            line[column] = token_text[offset]


def _string_prefix(text: str) -> str:
    index = 0
    while index < len(text) and text[index] in "rRuUbBfF":
        index += 1
    return text[:index]


def _is_f_string_token(text: str) -> bool:
    return "f" in _string_prefix(text).lower()


def _f_string_expression_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "{":
            index += 1
            continue
        if index + 1 < len(text) and text[index + 1] == "{":
            index += 2
            continue
        depth = 1
        expr_start = index + 1
        index += 1
        while index < len(text) and depth:
            char = text[index]
            if char == "{":
                if index + 1 < len(text) and text[index + 1] == "{":
                    index += 2
                    continue
                depth += 1
            elif char == "}":
                if index + 1 < len(text) and text[index + 1] == "}":
                    index += 2
                    continue
                depth -= 1
                if depth == 0:
                    spans.append((expr_start, index))
                    break
            index += 1
        index += 1
    return spans


def _blank_string_token_span(masked_lines: list[list[str]], tok: tokenize.TokenInfo) -> None:
    _blank_token_span(masked_lines, tok.start, tok.end)
    if not _is_f_string_token(tok.string):
        return
    for start_offset, end_offset in _f_string_expression_spans(tok.string):
        _restore_token_span(masked_lines, tok.string, tok.start, start_offset, end_offset)


def _python_code_lines_from_blob(source: str) -> dict[int, str] | None:
    """Return code-only text by physical line using Python's tokenizer.

    ``None`` means the blob could not be tokenized; callers must fail closed by
    treating added lines as live code.
    """
    physical_lines = source.splitlines()
    masked_lines = [list(line) for line in physical_lines]
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type == token.STRING:
                _blank_string_token_span(masked_lines, tok)
            elif tok.type in _NON_CODE_TOKEN_TYPES:
                _blank_token_span(masked_lines, tok.start, tok.end)
    except (IndentationError, tokenize.TokenError, UnicodeDecodeError):
        return None
    return {line_no: "".join(chars).rstrip() for line_no, chars in enumerate(masked_lines, 1)}


def _line_for_matching(added_line: AddedLine) -> str:
    if _is_python_path(added_line.path) and added_line.code_line is not None:
        return added_line.code_line
    return added_line.line


def _line_reexports_or_defines_symbol(line: str, symbol: str) -> bool:
    return _line_reexports_or_defines_export(line, _symbol_export(symbol))


def _line_reexports_or_defines_export(
    line: str,
    export: str,
    *,
    allow_wildcard: bool = True,
) -> bool:
    if line[:1].isspace():
        return False
    from_import = _from_import(line)
    if from_import is not None:
        _module, names = from_import
        return export in names or (allow_wildcard and "*" in names)
    return bool(
        re.match(rf"^(?:async\s+def|def|class)\s+{re.escape(export)}\b", line)
        or re.match(rf"^{re.escape(export)}\s*=", line)
    )


def _line_reexports_or_defines_kept_symbol(line: str, entry: CharterEntry) -> bool:
    return any(
        _line_reexports_or_defines_export(
            line,
            _symbol_root_export(symbol),
            allow_wildcard=False,
        )
        for symbol in entry.kept_symbols
        if _is_top_level_symbol(symbol)
    )


def _module_is_under(module: str, parent: str) -> bool:
    return module == parent or module.startswith(f"{parent}.")


def _line_imports_path(line: str, path_pattern: str) -> bool:
    modules = _path_pattern_to_modules(path_pattern)
    if not modules:
        return False
    from_import = _from_import(line)
    if from_import is not None:
        module, _names = from_import
        return any(_module_is_under(module, parent) for parent in modules)
    return any(
        _module_is_under(imported, parent)
        for imported in _plain_import_modules(line)
        for parent in modules
    )


def _line_uses_symbol_reference(
    line: str,
    symbol: str,
    aliases_by_module: dict[str, set[str]],
) -> bool:
    module_name = _symbol_module(symbol)
    if module_name is None:
        return False
    export = _symbol_export(symbol)
    refs = [module_name, *sorted(aliases_by_module.get(module_name, set()))]
    return any(
        re.search(rf"(?<![\w.]){re.escape(ref)}\.{re.escape(export)}\b", line) for ref in refs
    )


def _line_imports_symbol(
    line: str,
    symbol: str,
    aliases_by_module: dict[str, set[str]] | None = None,
) -> bool:
    aliases_by_module = aliases_by_module or {}
    if _line_uses_symbol_reference(line, symbol, aliases_by_module):
        return True
    module_name = _symbol_module(symbol)
    export = _symbol_export(symbol)
    if module_name is None:
        return _line_mentions_symbol(line, symbol)
    from_import = _from_import(line)
    if from_import is not None:
        imported_module, names = from_import
        return imported_module == module_name and (export in names or "*" in names)
    return any(
        imported == module_name for imported in _plain_import_modules(line)
    ) and _line_mentions_symbol(
        line,
        symbol,
    )


def _line_is_kept_only(added_line: AddedLine, entry: CharterEntry) -> bool:
    if not entry.kept_symbols:
        return False
    line = _line_for_matching(added_line)
    kept_exports = _symbol_export_roots(entry.kept_symbols)
    from_import = _from_import(line)
    if from_import is not None:
        imported_module, names = from_import
        matching_kept = [
            symbol for symbol in entry.kept_symbols if _symbol_module(symbol) == imported_module
        ]
        if matching_kept:
            return bool(names) and all(name != "*" and name in kept_exports for name in names)
        return False
    if any(_path_matches(path, added_line.path) for path in entry.paths):
        return _line_reexports_or_defines_kept_symbol(line, entry)
    return False


def _paths_with_added_lines(diff_text: str) -> set[str]:
    paths: set[str] = set()
    current_path: str | None = None
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            current_path = _normalize_diff_path(raw_line[4:].split("\t", 1)[0])
            continue
        if current_path is not None and raw_line.startswith("+") and not raw_line.startswith("+++"):
            paths.add(current_path)
    return paths


def _post_images_from_diff(diff_text: str) -> dict[str, str]:
    line_maps: dict[str, dict[int, str]] = {}
    current_path: str | None = None
    current_line: int | None = None
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            current_path = _normalize_diff_path(raw_line[4:].split("\t", 1)[0])
            current_line = None
            continue
        if raw_line.startswith("@@"):
            match = HUNK_RE.search(raw_line)
            current_line = int(match.group(1)) if match else None
            continue
        if current_path is None or current_line is None:
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            line_maps.setdefault(current_path, {})[current_line] = raw_line[1:]
            current_line += 1
        elif raw_line.startswith(" "):
            line_maps.setdefault(current_path, {})[current_line] = raw_line[1:]
            current_line += 1
        elif raw_line.startswith("-"):
            continue
    images: dict[str, str] = {}
    for path, line_map in line_maps.items():
        if not line_map:
            continue
        max_line = max(line_map)
        images[path] = "\n".join(line_map.get(line_no, "") for line_no in range(1, max_line + 1))
    return images


def _git_show_blob(repo_root: Path, head_ref: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{head_ref}:{path}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _line_classifications_for_diff(
    diff_text: str,
    *,
    head_ref: str | None,
    repo_root: Path | None,
    post_images: dict[str, str] | None,
) -> dict[str, dict[int, str] | None]:
    images = dict(post_images or {})
    fallback_images = _post_images_from_diff(diff_text)
    classifications: dict[str, dict[int, str] | None] = {}
    for path in _paths_with_added_lines(diff_text):
        if not _is_python_path(path):
            continue
        image = images.get(path)
        if image is None and head_ref is not None and repo_root is not None:
            image = _git_show_blob(repo_root, head_ref, path)
        if image is None:
            image = fallback_images.get(path)
        classifications[path] = None if image is None else _python_code_lines_from_blob(image)
    return classifications


def parse_diff(
    diff_text: str,
    *,
    head_ref: str | None = None,
    repo_root: Path | None = None,
    post_images: dict[str, str] | None = None,
) -> list[AddedLine]:
    added: list[AddedLine] = []
    current_path: str | None = None
    current_line: int | None = None
    classifications = _line_classifications_for_diff(
        diff_text,
        head_ref=head_ref,
        repo_root=repo_root,
        post_images=post_images,
    )
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            current_path = _normalize_diff_path(raw_line[4:].split("\t", 1)[0])
            current_line = None
            continue
        if raw_line.startswith("@@"):
            match = HUNK_RE.search(raw_line)
            current_line = int(match.group(1)) if match else None
            continue
        if current_path is None:
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            line = raw_line[1:]
            in_string_literal = False
            code_line = None
            if _is_python_path(current_path):
                code_lines = classifications.get(current_path)
                if code_lines is not None and current_line is not None:
                    code_line = code_lines.get(current_line, "")
                    in_string_literal = not code_line.strip()
                else:
                    code_line = line
                    in_string_literal = False
            added.append(AddedLine(current_path, current_line, line, in_string_literal, code_line))
            if current_line is not None:
                current_line += 1
        elif raw_line.startswith("-"):
            continue
        elif current_line is not None and raw_line.startswith(" "):
            current_line += 1
    return added


def load_charter_entries(
    charter_path: Path,
) -> tuple[list[CharterEntry], dict[str, list[str]], str]:
    data = yaml.safe_load(charter_path.read_text(encoding="utf-8")) or {}
    meta = data.get("meta") or {}
    status = str(meta.get("status") or "DRAFT").upper()
    authority_by_ref: dict[str, list[str]] = {}
    for authority in data.get("authorities") or []:
        authority_id = str(authority.get("id") or "")
        if not authority_id.startswith("ARCH-"):
            continue
        for ref in authority.get("registry_refs") or []:
            authority_by_ref.setdefault(str(ref), []).append(authority_id)

    entries: list[CharterEntry] = []
    for raw_entry in data.get("registry") or []:
        entry_id = str(raw_entry.get("id") or "")
        state = str(raw_entry.get("state") or "").upper()
        if not entry_id.startswith("CHR-") or state not in ENFORCED_STATES:
            continue
        is_binding = (
            status == "RATIFIED"
            or bool(raw_entry.get("binding_in_draft"))
            or entry_id in DRAFT_BINDING_IDS
        )
        entries.append(
            CharterEntry(
                entry_id=entry_id,
                state=state,
                paths=tuple(str(path) for path in raw_entry.get("paths") or []),
                symbols=tuple(str(symbol) for symbol in raw_entry.get("symbols") or []),
                kept_symbols=tuple(str(symbol) for symbol in raw_entry.get("kept_symbols") or []),
                binding="BINDING" if is_binding else "PROPOSED",
            )
        )
    return entries, authority_by_ref, status


def _entry_matches_line(
    entry: CharterEntry,
    added_line: AddedLine,
    aliases_by_module: dict[str, set[str]],
) -> str | None:
    line = _line_for_matching(added_line)
    if _line_is_kept_only(added_line, entry):
        return None
    path_matches = any(_path_matches(path, added_line.path) for path in entry.paths)
    if not entry.symbols:
        if path_matches:
            return f"adds code under chartered {entry.state.lower()} path"
        if _is_python_path(added_line.path) and not line.strip():
            return None
        for path in entry.paths:
            if _line_imports_path(line, path):
                return f"imports chartered {entry.state.lower()} path {path}"
        return None
    if added_line.in_string_literal and _is_python_path(added_line.path):
        return None
    if _is_python_path(added_line.path) and not line.strip():
        return None
    if entry.symbols:
        if not _is_python_path(added_line.path):
            return None
        for symbol in entry.symbols:
            if _line_imports_symbol(line, symbol, aliases_by_module) or (
                path_matches and _line_reexports_or_defines_symbol(line, symbol)
            ):
                return f"re-adds chartered symbol {symbol}"
        return None
    return None


def check_diff(
    diff_text: str,
    *,
    charter_path: Path | str,
    head_ref: str | None = None,
    repo_root: Path | None = None,
    post_images: dict[str, str] | None = None,
) -> CheckResult:
    entries, authority_by_ref, _status = load_charter_entries(Path(charter_path))
    added_lines = parse_diff(
        diff_text,
        head_ref=head_ref,
        repo_root=repo_root,
        post_images=post_images,
    )
    aliases_by_path: dict[str, dict[str, set[str]]] = {}
    for added_line in added_lines:
        for alias, module in _plain_import_aliases(_line_for_matching(added_line)).items():
            aliases_by_path.setdefault(added_line.path, {}).setdefault(module, set()).add(alias)
    violations: list[Violation] = []
    seen: set[tuple[str, str, int | None, str]] = set()
    for added_line in added_lines:
        for entry in entries:
            reason = _entry_matches_line(
                entry,
                added_line,
                aliases_by_path.get(added_line.path, {}),
            )
            if reason is None:
                continue
            if entry.symbols:
                key = (
                    entry.entry_id,
                    added_line.path,
                    added_line.line_no,
                    added_line.line.strip(),
                )
            else:
                key = (entry.entry_id, added_line.path, None, "")
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                Violation(
                    binding=entry.binding,
                    entry_id=entry.entry_id,
                    state=entry.state,
                    path=added_line.path,
                    line_no=added_line.line_no,
                    line=added_line.line,
                    reason=reason,
                    authority_ids=authority_by_ref.get(entry.entry_id, []),
                )
            )
    binding = [violation for violation in violations if violation.binding == "BINDING"]
    proposed = [violation for violation in violations if violation.binding == "PROPOSED"]
    return CheckResult(
        ok=not violations,
        binding_violations=binding,
        proposed_violations=proposed,
        violations=violations,
    )


def _git_diff(ref_range: str) -> str:
    proc = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--unified=0", ref_range, "--"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(f"git diff failed for {ref_range}: {proc.stderr.strip()}")
    return proc.stdout


def _read_diff(args: argparse.Namespace) -> str:
    if args.diff_file:
        if args.diff_file == "-":
            return sys.stdin.read()
        return Path(args.diff_file).read_text(encoding="utf-8")
    ref_range = args.ref_range or f"{args.base}...{args.head}"
    return _git_diff(ref_range)


def _head_ref(args: argparse.Namespace) -> str:
    if args.ref_range:
        if "..." in args.ref_range:
            return args.ref_range.rsplit("...", 1)[1]
        if ".." in args.ref_range:
            return args.ref_range.rsplit("..", 1)[1]
    return args.head


def _render_text(result: CheckResult) -> str:
    if result.ok:
        return "Charter compliance: PASS\nNo chartered re-adds or enforceable placement violations found."
    lines = ["Charter compliance: FAIL"]
    for title, violations in (
        ("Binding violations", result.binding_violations),
        ("Proposed violations", result.proposed_violations),
    ):
        if not violations:
            continue
        lines.append("")
        lines.append(f"{title}:")
        for violation in violations:
            location = violation.path
            if violation.line_no is not None:
                location = f"{location}:{violation.line_no}"
            authorities = ""
            if violation.authority_ids:
                authorities = f" authorities={','.join(violation.authority_ids)}"
            lines.append(
                f"- {violation.binding} {violation.entry_id} [{violation.state}]"
                f"{authorities} {location}: {violation.reason}"
            )
            lines.append(f"  added: {violation.line.strip()}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--charters",
        type=Path,
        default=_repo_root() / "docs" / "architecture" / "charters.yaml",
        help="Path to docs/architecture/charters.yaml.",
    )
    parser.add_argument("--diff-file", help="Unified diff file to inspect; use '-' for stdin.")
    parser.add_argument(
        "--range", dest="ref_range", help="Git diff range, for example base...head."
    )
    parser.add_argument("--base", default="origin/main", help="Base ref when --range is omitted.")
    parser.add_argument("--head", default="HEAD", help="Head ref when --range is omitted.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    diff_text = _read_diff(args)
    head_ref = None if args.diff_file else _head_ref(args)
    repo_root = None if args.diff_file else Path.cwd()
    result = check_diff(
        diff_text,
        charter_path=args.charters,
        head_ref=head_ref,
        repo_root=repo_root,
    )
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(_render_text(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
