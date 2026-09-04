#!/usr/bin/env python3
"""Per-tool output parsers for ``scripts/ci/check_tool_baseline.py``.

Each parser is a pure function ``parse(stdout: str) -> list[Finding]`` that
turns one tool's captured stdout into findings. Parsers never touch the file
system and never run anything, so they are trivially unit-testable against a
captured real-output fixture under ``tests/ci/fixtures/tool_baseline/``.

Adding a parser is a one-function change: write the function and decorate it
with ``@register("<tool>", ...)``. The runner discovers it through ``PARSERS``
and the ``--tool`` choices, usage text, and ``docs/RATCHETS.md`` table follow.

Finding keys
------------
A finding is keyed as ``<path>::<symbol>::<rule>``:

* ``path`` -- POSIX path relative to the runner's ``--cwd`` (parsers hand back
  whatever the tool printed; the runner normalises it);
* ``symbol`` -- either a symbol name the tool reports (vulture, deptry,
  jscpd, ...) or a 12-hex-digit SHA-256 prefix of the offending source line's
  stripped content. Line numbers are deliberately NOT part of the key, so a
  pure line shift never surfaces as a new finding. A parser that leaves
  ``symbol`` empty and sets ``line`` asks the runner to compute the content
  hash from the file (``ToolSpec.symbol_from_line``);
* ``rule`` -- the tool's rule/error code (``F401``, ``arg-type``, ``TODO``).

Stdlib only: this module runs in CI before project dependencies are installed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

KEY_SEPARATOR = "::"
LINE_HASH_LENGTH = 12


@dataclass(frozen=True)
class Finding:
    """One tool finding.

    ``symbol`` may be empty when ``line`` is set: the runner then fills it with
    the content hash of that source line (see ``ToolSpec.symbol_from_line``).
    ``line`` and ``message`` are informational only and never enter the key.
    """

    path: str
    rule: str
    symbol: str = ""
    line: int | None = None
    message: str = ""

    def key(self) -> str:
        return KEY_SEPARATOR.join((self.path, self.symbol, self.rule))


ParseFn = Callable[[str], list[Finding]]


@dataclass(frozen=True)
class ToolSpec:
    """Registry entry describing how the runner should treat one tool."""

    name: str
    parse: ParseFn
    description: str
    example_command: str
    # Exit codes that mean "ran fine, zero findings" when stdout parses to
    # nothing. Any other exit code with zero parsed findings is a tool crash.
    clean_exit_codes: frozenset[int] = field(default_factory=lambda: frozenset({0}))
    # True when the parser leaves ``symbol`` empty and the runner must hash the
    # offending source line read from ``--cwd``.
    symbol_from_line: bool = False


PARSERS: dict[str, ToolSpec] = {}


def register(
    name: str,
    *,
    description: str,
    example_command: str,
    clean_exit_codes: frozenset[int] | set[int] = frozenset({0}),
    symbol_from_line: bool = False,
) -> Callable[[ParseFn], ParseFn]:
    """Decorator registering ``parse`` under ``--tool <name>``."""

    def decorator(fn: ParseFn) -> ParseFn:
        if name in PARSERS:
            raise ValueError(f"parser already registered: {name}")
        PARSERS[name] = ToolSpec(
            name=name,
            parse=fn,
            description=description,
            example_command=example_command,
            clean_exit_codes=frozenset(clean_exit_codes),
            symbol_from_line=symbol_from_line,
        )
        return fn

    return decorator


def line_hash(content: str) -> str:
    """Stable short hash of a source line, whitespace-insensitive at the ends."""
    return hashlib.sha256(content.strip().encode("utf-8", "replace")).hexdigest()[:LINE_HASH_LENGTH]


def supported_tools() -> list[str]:
    return sorted(PARSERS)


# --- ruff -------------------------------------------------------------------

# ``ruff check --output-format concise``:
#   pkg/mod.py:1:8: F401 [*] `os` imported but unused
#   pkg/mod.py:3:1: SyntaxError: unexpected indentation
_RUFF_RE = re.compile(
    r"^(?P<path>[^:\n]+?):(?P<line>\d+):(?P<col>\d+): "
    r"(?P<rule>[A-Za-z][A-Za-z0-9]*):? (?:\[\*\] )?(?P<msg>.*)$"
)


@register(
    "ruff",
    description="ruff check, concise output; key = line-content hash",
    example_command="ruff check <paths> --select N --output-format concise",
    clean_exit_codes={0},
    symbol_from_line=True,
)
def parse_ruff(stdout: str) -> list[Finding]:
    findings: list[Finding] = []
    for raw in stdout.splitlines():
        match = _RUFF_RE.match(raw)
        if match is None:
            continue
        findings.append(
            Finding(
                path=match["path"],
                rule=match["rule"],
                line=int(match["line"]),
                message=match["msg"].strip(),
            )
        )
    return findings


# --- mypy -------------------------------------------------------------------

# ``mypy`` default text output (with or without --show-column-numbers):
#   pkg/mod.py:11: error: Incompatible return value type ...  [return-value]
#   pkg/mod.py:11:5: error: ...  [arg-type]
#   pkg/mod.py:11: note: ...            (notes are not findings)
_MYPY_RE = re.compile(
    r"^(?P<path>[^:\n]+?):(?P<line>\d+)(?::(?P<col>\d+))?: "
    r"(?P<severity>error|warning): (?P<msg>.*?)(?:  \[(?P<code>[\w-]+)\])?$"
)


@register(
    "mypy",
    description="mypy text output (errors and warnings; notes ignored); key = line-content hash",
    example_command="mypy --ignore-missing-imports <paths>",
    clean_exit_codes={0},
    symbol_from_line=True,
)
def parse_mypy(stdout: str) -> list[Finding]:
    findings: list[Finding] = []
    for raw in stdout.splitlines():
        match = _MYPY_RE.match(raw)
        if match is None:
            continue
        findings.append(
            Finding(
                path=match["path"],
                rule=match["code"] or match["severity"],
                line=int(match["line"]),
                message=match["msg"].strip(),
            )
        )
    return findings


# --- todo (grep) ------------------------------------------------------------

# ``grep -rn --include='*.py' -E 'TODO|FIXME' .``:
#   ./pkg/mod.py:10:    # TODO: make this real
# The content is already in the output, so the parser hashes it itself; the
# marker word becomes the rule so a TODO turned FIXME is a new finding.
_GREP_RE = re.compile(r"^(?P<path>[^:\n]+?):(?P<line>\d+):(?P<content>.*)$")
_TODO_MARKER_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")


@register(
    "todo",
    description="grep -rn output for TODO/FIXME markers; key = matched-line hash, rule = marker",
    example_command="grep -rn --include='*.py' -E 'TODO|FIXME' .",
    # grep exits 1 when nothing matches: that is zero findings, not a crash.
    clean_exit_codes={0, 1},
)
def parse_todo(stdout: str) -> list[Finding]:
    findings: list[Finding] = []
    for raw in stdout.splitlines():
        match = _GREP_RE.match(raw)
        if match is None:
            continue
        content = match["content"]
        marker = _TODO_MARKER_RE.search(content)
        findings.append(
            Finding(
                path=match["path"],
                rule=marker.group(1) if marker else "TODO",
                symbol=line_hash(content),
                line=int(match["line"]),
                message=content.strip(),
            )
        )
    return findings
