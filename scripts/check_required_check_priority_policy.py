#!/usr/bin/env python3
"""Guard required-check-priority keep-lists against drift."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Violation:
    path: str
    message: str


@dataclass(frozen=True)
class _YamlMappingLine:
    indent: int
    key: str
    value: str


@dataclass(frozen=True)
class _MainPushTrigger:
    targets_main: bool
    path_filtered: bool


WORKFLOW_PATH = Path(".github/workflows/required-check-priority.yml")

REQUIRED_KEEP_WORKFLOW_PATHS = {
    ".github/workflows/aragora-merge-quorum.yml",
    ".github/workflows/aragora-review-gate.yml",
    ".github/workflows/autopilot-worktree-e2e.yml",
    ".github/workflows/core-suites.yml",
    ".github/workflows/lint.yml",
    ".github/workflows/sdk-parity.yml",
    ".github/workflows/sdk-generate.yml",
    ".github/workflows/sdk-test.yml",
    ".github/workflows/test.yml",
    ".github/workflows/openapi.yml",
    ".github/workflows/pr-admission-controller.yml",
    ".github/workflows/quality-smoke.yml",
    ".github/workflows/required-check-priority.yml",
    ".github/workflows/release-readiness.yml",
    ".github/workflows/security-gate.yml",
    ".github/workflows/self-hosted-shadow.yml",
    ".github/workflows/smoke.yml",
    ".github/workflows/smoke-offline.yml",
}

REQUIRED_KEEP_WORKFLOW_NAMES = {
    "Aragora Merge Quorum",
    "Aragora Code Review",
    "Autopilot Worktree E2E",
    "Core Suites (Decision Integrity)",
    "Generate SDK Types",
    "Offline Golden Path",
    "Required Check Priority",
    "Lint",
    "PR Admission Controller",
    "Quality Pipeline Smoke",
    "Release Readiness Gate",
    "Security Gate",
    "Self-Hosted Shadow CI",
    "SDK Parity Check",
    "SDK Tests",
    "Smoke Tests",
    "Tests",
    "OpenAPI Spec",
}

REQUIRED_CONTEXT_TO_WORKFLOW_PATH = {
    "lint": ".github/workflows/lint.yml",
    "typecheck": ".github/workflows/lint.yml",
    "sdk-parity": ".github/workflows/sdk-parity.yml",
    "Generate & Validate": ".github/workflows/openapi.yml",
    "TypeScript SDK Type Check": ".github/workflows/sdk-test.yml",
}


def _extract_js_set_items(workflow_text: str, set_name: str) -> list[str] | None:
    pattern = r"const\s+" + re.escape(set_name) + r"\s*=\s*new Set\(\[(?P<body>.*?)\]\);"
    match = re.search(pattern, workflow_text, flags=re.DOTALL)
    if not match:
        return None
    body = match.group("body")
    return re.findall(r"""["']([^"']+)["']""", body)


def _parse_yaml_mapping_line(line: str) -> _YamlMappingLine | None:
    if not line.strip() or line.lstrip().startswith("#"):
        return None
    match = re.match(
        r"^(?P<indent>\s*)(?P<key>['\"]?[A-Za-z0-9_-]+['\"]?)\s*:\s*(?P<value>.*)$",
        line,
    )
    if not match:
        return None
    return _YamlMappingLine(
        indent=len(match.group("indent")),
        key=match.group("key").strip("\"'"),
        value=match.group("value").strip(),
    )


def _nested_yaml_block(lines: list[str], index: int, parent_indent: int) -> list[str]:
    block: list[str] = []
    for line in lines[index + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            block.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= parent_indent:
            break
        block.append(line)
    return block


def _inline_yaml_value_mentions(value: str, expected: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_./-]){re.escape(expected)}(?![A-Za-z0-9_./-])",
            value,
        )
    )


def _flow_mapping_value(value: str, key: str) -> str | None:
    stripped = value.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None

    match = re.search(
        rf"(?<![A-Za-z0-9_-]){re.escape(key)}\s*:\s*(?P<value>\[[^\]]*\]|[^,}}]+)",
        stripped[1:-1],
    )
    if match is None:
        return None
    return match.group("value").strip()


def _flow_push_targets_main(value: str) -> bool:
    branches = _flow_mapping_value(value, "branches")
    if branches is not None:
        return _inline_yaml_value_mentions(branches, "main")

    branches_ignore = _flow_mapping_value(value, "branches-ignore")
    if branches_ignore is not None and _inline_yaml_value_mentions(
        branches_ignore,
        "main",
    ):
        return False

    return True


def _flow_push_has_path_filter(value: str) -> bool:
    return (
        _flow_mapping_value(value, "paths") is not None
        or _flow_mapping_value(
            value,
            "paths-ignore",
        )
        is not None
    )


def _block_value_for_key(lines: list[str], key: str) -> list[str] | None:
    for index, line in enumerate(lines):
        parsed = _parse_yaml_mapping_line(line)
        if parsed is None or parsed.key != key:
            continue

        values = [parsed.value]
        for child in lines[index + 1 :]:
            child_parsed = _parse_yaml_mapping_line(child)
            if child_parsed is not None and child_parsed.indent <= parsed.indent:
                break
            values.append(child.strip())
        return values
    return None


def _push_block_targets_main(push_block: list[str]) -> bool:
    branch_values = _block_value_for_key(push_block, "branches")
    if branch_values is not None:
        return _inline_yaml_value_mentions("\n".join(branch_values), "main")

    ignored_branch_values = _block_value_for_key(push_block, "branches-ignore")
    if ignored_branch_values is not None and _inline_yaml_value_mentions(
        "\n".join(ignored_branch_values),
        "main",
    ):
        return False

    return True


def _push_block_has_path_filter(push_block: list[str]) -> bool:
    for line in push_block:
        parsed = _parse_yaml_mapping_line(line)
        if parsed is not None and parsed.key in {"paths", "paths-ignore"}:
            return True
    return False


def _main_push_trigger(workflow_text: str) -> _MainPushTrigger:
    lines = workflow_text.splitlines()
    for index, line in enumerate(lines):
        parsed = _parse_yaml_mapping_line(line)
        if parsed is None or parsed.key != "on" or parsed.indent != 0:
            continue

        if parsed.value and _inline_yaml_value_mentions(parsed.value, "push"):
            return _MainPushTrigger(targets_main=True, path_filtered=False)

        on_block = _nested_yaml_block(lines, index, parsed.indent)
        for on_index, on_line in enumerate(on_block):
            on_parsed = _parse_yaml_mapping_line(on_line)
            if on_parsed is None or on_parsed.key != "push":
                continue

            if on_parsed.value:
                if not _flow_push_targets_main(on_parsed.value):
                    return _MainPushTrigger(targets_main=False, path_filtered=False)
                return _MainPushTrigger(
                    targets_main=True,
                    path_filtered=_flow_push_has_path_filter(on_parsed.value),
                )

            push_block = _nested_yaml_block(on_block, on_index, on_parsed.indent)
            if not _push_block_targets_main(push_block):
                return _MainPushTrigger(targets_main=False, path_filtered=False)
            return _MainPushTrigger(
                targets_main=True,
                path_filtered=_push_block_has_path_filter(push_block),
            )

    return _MainPushTrigger(targets_main=False, path_filtered=False)


def find_required_check_priority_violations(
    workflow_text: str,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    violations: list[str] = []

    path_items = _extract_js_set_items(workflow_text, "alwaysKeepWorkflowPaths")
    if path_items is None:
        return ["missing `alwaysKeepWorkflowPaths` set definition"]

    name_items = _extract_js_set_items(workflow_text, "alwaysKeepWorkflowNames")
    if name_items is None:
        return ["missing `alwaysKeepWorkflowNames` set definition"]

    if len(path_items) != len(set(path_items)):
        violations.append("duplicate entries found in alwaysKeepWorkflowPaths")
    if len(name_items) != len(set(name_items)):
        violations.append("duplicate entries found in alwaysKeepWorkflowNames")

    path_set = set(path_items)
    missing_required_paths = sorted(REQUIRED_KEEP_WORKFLOW_PATHS - path_set)
    for path in missing_required_paths:
        violations.append(f"missing required keep workflow path: {path}")

    for context, mapped_path in sorted(REQUIRED_CONTEXT_TO_WORKFLOW_PATH.items()):
        if mapped_path not in path_set:
            violations.append(
                f"required context `{context}` maps to workflow path not in keep-list: {mapped_path}"
            )

    name_set = set(name_items)
    missing_required_names = sorted(REQUIRED_KEEP_WORKFLOW_NAMES - name_set)
    for name in missing_required_names:
        violations.append(f"missing required keep workflow name: {name}")

    if repo_root is not None:
        mapped_workflows: dict[str, str] = {}
        for rel in sorted(path_set):
            wf_path = (repo_root / rel).resolve()
            if not wf_path.exists():
                violations.append(f"keep workflow path does not exist: {rel}")
        for context, rel in sorted(REQUIRED_CONTEXT_TO_WORKFLOW_PATH.items()):
            wf_path = (repo_root / rel).resolve()
            if not wf_path.exists():
                continue
            text = wf_path.read_text(encoding="utf-8")
            mapped_workflows[rel] = text
            if context not in text:
                violations.append(
                    f"required context marker `{context}` not found in mapped workflow: {rel}"
                )
        for context, rel in sorted(REQUIRED_CONTEXT_TO_WORKFLOW_PATH.items()):
            text = mapped_workflows.get(rel)
            if text is None:
                continue
            main_push = _main_push_trigger(text)
            if main_push.targets_main and main_push.path_filtered:
                violations.append(
                    f"required context `{context}` maps to path-filtered main push workflow: {rel}"
                )

    return violations


def check_repo(repo_root: Path) -> list[Violation]:
    workflow_file = repo_root / WORKFLOW_PATH
    if not workflow_file.exists():
        return [Violation(path=str(WORKFLOW_PATH), message="missing workflow file")]

    text = workflow_file.read_text(encoding="utf-8")
    return [
        Violation(path=str(WORKFLOW_PATH), message=message)
        for message in find_required_check_priority_violations(text, repo_root=repo_root)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce required-check-priority workflow keep-list policy."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to check",
    )
    args = parser.parse_args()

    violations = check_repo(Path(args.repo_root).resolve())
    if not violations:
        print("Required check priority policy check passed")
        return 0

    print("Required check priority policy violations detected:")
    for v in violations:
        print(f"- {v.path}: {v.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
