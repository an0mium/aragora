"""Verification helpers extracted from ``dev_coordination``."""

from __future__ import annotations

import re
import shlex
from typing import Any

_TEST_FILE_PATTERN = re.compile(r"(tests/[\w./-]+\.py)")


def _normalize_claim(value: str) -> str:
    return value.strip().strip("/")


def _extract_tests_value(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _canonical_verification_command(command: Any) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    for prefix in ("bash -lc ", "/bin/bash -lc "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
                text = text[1:-1]
            break
    from aragora.swarm.worker_launcher import WorkerLauncher

    text = re.sub(r"^(?P<prefix>\s*)python3(?=\s|$)", r"\g<prefix>python", text)
    return WorkerLauncher._normalize_verification_command(text).strip()


def _pytest_command_targets(command: Any) -> list[str]:
    text = _canonical_verification_command(command)
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return []
    if not tokens:
        return []
    start = 0
    if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
        start = 3
    elif tokens[0].endswith("pytest"):
        start = 1
    else:
        return []

    targets: list[str] = []
    skip_next = False
    options_with_values = {"-k", "-m", "--maxfail", "--timeout", "--tb", "-c", "--rootdir"}
    for token in tokens[start:]:
        if skip_next:
            skip_next = False
            continue
        if token in options_with_values:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        normalized = _normalize_claim(token.rstrip("/"))
        if token.endswith("/") or "/" in normalized or normalized.endswith(".py"):
            targets.append(normalized)
    return targets


def _is_overbroad_pytest_command(command: Any) -> bool:
    targets = _pytest_command_targets(command)
    if not targets:
        return False
    return any(not target.endswith(".py") for target in targets)


def _verification_command_covers_expected(recorded_command: Any, expected_command: Any) -> bool:
    recorded = _canonical_verification_command(recorded_command)
    expected = _canonical_verification_command(expected_command)
    if not recorded or not expected:
        return False
    if recorded == expected:
        return True
    recorded_targets = _pytest_command_targets(recorded)
    expected_targets = _pytest_command_targets(expected)
    if not recorded_targets or not expected_targets:
        return False
    for expected_target in expected_targets:
        if not any(
            expected_target == recorded_target
            or expected_target.startswith(recorded_target.rstrip("/") + "/")
            for recorded_target in recorded_targets
        ):
            return False
    return True


def _verification_timeout_for_command(command: Any, default_timeout: float) -> float:
    canonical = _canonical_verification_command(command)
    if not canonical:
        return default_timeout
    targets = _pytest_command_targets(canonical)
    if len(targets) == 1 and targets[0].endswith(".py"):
        return max(default_timeout, 300.0)
    return default_timeout


def _inferred_expected_tests_for_work_order(work_order: dict[str, Any]) -> list[str]:
    inferred: list[str] = []
    seen: set[str] = set()
    deferred_overbroad: list[str] = []
    deferred_seen: set[str] = set()

    def _append(command: str) -> None:
        normalized = str(command).strip()
        canonical = _canonical_verification_command(normalized)
        dedupe_key = canonical or normalized
        if not normalized or dedupe_key in seen:
            return
        if _is_overbroad_pytest_command(normalized):
            if dedupe_key in deferred_seen:
                return
            deferred_seen.add(dedupe_key)
            deferred_overbroad.append(normalized)
            return
        seen.add(dedupe_key)
        inferred.append(normalized)

    for entry in work_order.get("expected_tests", []):
        _append(str(entry))

    success_criteria = work_order.get("success_criteria")
    if isinstance(success_criteria, dict):
        for entry in _extract_tests_value(success_criteria.get("tests")):
            _append(entry)

    metadata = work_order.get("metadata")
    if isinstance(metadata, dict):
        for entry in metadata.get("acceptance_criteria", []):
            text = str(entry).strip()
            if text.startswith("python -m pytest") or text.startswith("pytest"):
                _append(text)
            for match in _TEST_FILE_PATTERN.findall(text):
                _append(f"python -m pytest {match} -q")

    for path in work_order.get("file_scope", []):
        normalized = str(path).strip()
        if normalized.startswith("tests/") and normalized.endswith(".py"):
            _append(f"python -m pytest {normalized} -q")

    for path in work_order.get("changed_paths", []):
        normalized = str(path).strip()
        if normalized.startswith("tests/") and normalized.endswith(".py"):
            _append(f"python -m pytest {normalized} -q")

    if not inferred:
        inferred.extend(deferred_overbroad)

    return inferred
