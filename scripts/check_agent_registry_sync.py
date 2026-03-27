#!/usr/bin/env python3
"""
Validate AGENTS.md against runtime registry and allowlist.

Checks:
1) Declared registered-agent count in AGENTS.md matches runtime list_available_agents().
2) Declared allowlist count in AGENTS.md matches ALLOWED_AGENT_TYPES.
3) Agent table in AGENTS.md matches runtime registry set exactly.
4) All allowlisted agent types are documented.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


def _extract_agent_section(text: str) -> str:
    start = text.find("## Agent Types")
    end = text.find("## Agent Creation")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate 'Agent Types' section boundaries in AGENTS.md")
    return text[start:end]


def _extract_doc_agent_types(agent_section: str) -> set[str]:
    # Agent rows in the top section follow: | `agent-type` | ...
    return {
        m.group(1).strip()
        for m in re.finditer(r"^\|\s*`([^`]+)`\s*\|", agent_section, re.MULTILINE)
    }


def _extract_declared_registered_count(agent_section: str) -> int:
    m = re.search(r"registers\s+(\d+)\s+agent types", agent_section)
    if not m:
        raise ValueError("Could not parse declared registered-agent count in AGENTS.md")
    return int(m.group(1))


def _extract_declared_allowlist_count(agent_section: str) -> int:
    m = re.search(r"ALLOWED_AGENT_TYPES[^\n]*?(\d+)\s+types", agent_section)
    if not m:
        raise ValueError("Could not parse declared allowlist count in AGENTS.md")
    return int(m.group(1))


def _load_runtime_registry() -> set[str]:
    from aragora.agents.base import list_available_agents

    return set(list_available_agents().keys())


def _load_allowlist() -> set[str]:
    from aragora.config.settings import ALLOWED_AGENT_TYPES

    return set(ALLOWED_AGENT_TYPES)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extract_string_literals(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Call):
        return _extract_string_literals(node.args[0]) if node.args else set()
    if isinstance(node, ast.Set | ast.List | ast.Tuple):
        values: set[str] = set()
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.add(elt.value)
        return values
    return set()


def _load_allowlist_static() -> set[str]:
    settings_path = _repo_root() / "aragora" / "config" / "settings.py"
    tree = ast.parse(settings_path.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "ALLOWED_AGENT_TYPES" and node.value is not None:
                values = _extract_string_literals(node.value)
                if values:
                    return values
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ALLOWED_AGENT_TYPES":
                    values = _extract_string_literals(node.value)
                    if values:
                        return values

    raise ValueError("Could not parse ALLOWED_AGENT_TYPES from settings.py")


def _load_runtime_registry_static() -> set[str]:
    agents_root = _repo_root() / "aragora" / "agents"
    registered: set[str] = set()

    for path in agents_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            decorators = getattr(node, "decorator_list", [])
            for decorator in decorators:
                if not isinstance(decorator, ast.Call) or not isinstance(
                    decorator.func, ast.Attribute
                ):
                    continue
                if decorator.func.attr != "register":
                    continue
                owner = decorator.func.value
                if not isinstance(owner, ast.Name) or owner.id != "AgentRegistry":
                    continue
                if not decorator.args:
                    continue
                first_arg = decorator.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    registered.add(first_arg.value)

    if not registered:
        raise ValueError("Could not parse registered agent types from source")

    return registered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check AGENTS.md registry/allowlist synchronization"
    )
    parser.add_argument("--agents-doc", default="AGENTS.md", help="Path to AGENTS.md")
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Skip runtime imports and only validate AGENTS.md internal consistency",
    )
    args = parser.parse_args()

    doc_path = Path(args.agents_doc)
    if not doc_path.exists():
        print(f"AGENTS doc not found: {doc_path}", file=sys.stderr)
        return 2

    text = doc_path.read_text(encoding="utf-8")

    try:
        agent_section = _extract_agent_section(text)
        doc_agent_types = _extract_doc_agent_types(agent_section)
        declared_registered = _extract_declared_registered_count(agent_section)
        declared_allowlisted = _extract_declared_allowlist_count(agent_section)
    except ValueError as exc:
        print(f"AGENTS.md parse error: {exc}", file=sys.stderr)
        return 2

    errors: list[str] = []

    if declared_registered != len(doc_agent_types):
        errors.append(
            "Declared registered-agent count does not match AGENTS.md table: "
            f"declared={declared_registered}, table={len(doc_agent_types)}"
        )

    if args.docs_only:
        if errors:
            print("Agent registry sync check failed:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(
            "Agent registry docs-only check passed "
            f"(declared={declared_registered}, table={len(doc_agent_types)}, allowlist-declared={declared_allowlisted})"
        )
        return 0

    try:
        runtime_registry = _load_runtime_registry()
        allowlist = _load_allowlist()
    except Exception as exc:  # pragma: no cover - CI/runtime dependency failures
        print(
            f"Runtime imports unavailable ({exc}); falling back to source parsing.",
            file=sys.stderr,
        )
        try:
            runtime_registry = _load_runtime_registry_static()
            allowlist = _load_allowlist_static()
        except Exception as fallback_exc:
            print(f"Failed to load runtime registry/allowlist: {exc}", file=sys.stderr)
            print(f"Static fallback failed: {fallback_exc}", file=sys.stderr)
            return 2

    if declared_registered != len(runtime_registry):
        errors.append(
            "Declared registered-agent count does not match runtime registry: "
            f"declared={declared_registered}, runtime={len(runtime_registry)}"
        )

    if declared_allowlisted != len(allowlist):
        errors.append(
            "Declared allowlist count does not match ALLOWED_AGENT_TYPES: "
            f"declared={declared_allowlisted}, runtime={len(allowlist)}"
        )

    if doc_agent_types != runtime_registry:
        only_in_doc = sorted(doc_agent_types - runtime_registry)
        only_in_runtime = sorted(runtime_registry - doc_agent_types)
        if only_in_doc:
            errors.append(f"Agent types documented but not registered at runtime: {only_in_doc}")
        if only_in_runtime:
            errors.append(
                f"Agent types registered at runtime but missing from AGENTS.md: {only_in_runtime}"
            )

    missing_allowlisted = sorted(allowlist - doc_agent_types)
    if missing_allowlisted:
        errors.append(f"Allowlisted agent types missing from AGENTS.md: {missing_allowlisted}")

    if errors:
        print("Agent registry sync check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        "Agent registry sync check passed "
        f"(registered={len(runtime_registry)}, allowlisted={len(allowlist)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
