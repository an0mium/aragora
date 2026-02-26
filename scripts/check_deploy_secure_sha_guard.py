#!/usr/bin/env python3
"""Guard deploy-secure SHA verification hardening against workflow regressions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Violation:
    path: str
    message: str


WORKFLOW_PATH = Path(".github/workflows/deploy-secure.yml")
SHA_STEP_NAME = "- name: Post-deploy SHA verification"

REQUIRED_MARKERS: dict[str, str] = {
    "ec2_user_command": "sudo -u ec2-user git -C /home/ec2-user/aragora rev-parse HEAD",
    "safe_directory_fallback": "git -C /home/ec2-user/aragora -c safe.directory=/home/ec2-user/aragora rev-parse HEAD",
    "ssm_timeout": "--timeout-seconds 60",
    "stdout_diagnostics": "::warning::SHA stdout for $INST_ID:",
    "stderr_diagnostics": "::warning::SHA stderr for $INST_ID:",
}


def _extract_sha_step_block(workflow_text: str) -> str | None:
    lines = workflow_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == SHA_STEP_NAME:
            start_idx = i
            break

    if start_idx is None:
        return None

    block_lines: list[str] = []
    for line in lines[start_idx + 1 :]:
        if re.match(r"^  [A-Za-z0-9_-]+:\s*$", line):
            break
        block_lines.append(line)
    return "\n".join(block_lines)


def find_sha_verification_violations(workflow_text: str) -> list[str]:
    violations: list[str] = []
    block = _extract_sha_step_block(workflow_text)
    if block is None:
        return ["missing `Post-deploy SHA verification` step"]

    for name, marker in REQUIRED_MARKERS.items():
        if marker not in block:
            violations.append(f"missing required marker `{name}`: {marker}")
    return violations


def check_repo(repo_root: Path) -> list[Violation]:
    workflow_file = repo_root / WORKFLOW_PATH
    if not workflow_file.exists():
        return [Violation(path=str(WORKFLOW_PATH), message="missing workflow file")]

    text = workflow_file.read_text(encoding="utf-8")
    return [
        Violation(path=str(WORKFLOW_PATH), message=message)
        for message in find_sha_verification_violations(text)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce post-deploy SHA verification hardening in deploy-secure workflow."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to check",
    )
    args = parser.parse_args()

    violations = check_repo(Path(args.repo_root).resolve())
    if not violations:
        print("Deploy secure SHA guard check passed")
        return 0

    print("Deploy secure SHA guard violations detected:")
    for v in violations:
        print(f"- {v.path}: {v.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
