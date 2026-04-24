from __future__ import annotations

import re
from typing import Collection
from typing import cast

import yaml

from .exceptions import FooterValidationError
from .types import BridgeFooter
from .types import ParsedTurn

FOOTER_MARKER = "---BRIDGE-FOOTER---"
FOOTER_END_MARKER = "---BRIDGE-FOOTER-END---"
REQUIRED_KEYS = (
    "summary",
    "next_actor",
    "needs_human",
    "done",
    "artifacts",
    "tests_run",
)


def build_footer_instruction(*, roles: list[str]) -> str:
    role_list = ", ".join(sorted(roles))
    return (
        "End your response with this exact footer block:\n"
        f"{FOOTER_MARKER}\n"
        "summary: <single line>\n"
        "next_actor: <role_name or null>\n"
        "needs_human: <true|false>\n"
        "done: <true|false>\n"
        "artifacts: []\n"
        "tests_run: []\n"
        f"{FOOTER_END_MARKER}\n"
        f"Allowed next_actor values: {role_list}, null\n"
        "Unknown keys are not allowed."
    )


def build_repair_prompt(
    *,
    parse_errors: list[str],
    original_message: str,
    allowed_roles: Collection[str],
) -> str:
    excerpt = original_message.strip()
    if len(excerpt) > 1200:
        excerpt = excerpt[:1197].rstrip() + "..."
    errors = "\n".join(f"- {item}" for item in parse_errors) or "- footer_missing"
    role_list = ", ".join(sorted(allowed_roles))
    return (
        "Your previous response did not satisfy the bridge footer contract.\n"
        f"Return ONLY a corrected footer block that starts with {FOOTER_MARKER}.\n"
        f"Allowed next_actor values: {role_list}, null\n"
        "Required fields: summary, next_actor, needs_human, done, artifacts, tests_run\n"
        "Validation errors:\n"
        f"{errors}\n\n"
        "Previous response excerpt:\n"
        f"{excerpt}"
    )


def extract_footer_block(text: str) -> tuple[str | None, str]:
    stripped = text.rstrip()
    start = stripped.rfind(FOOTER_MARKER)
    end = stripped.rfind(FOOTER_END_MARKER)
    if start == -1 or end == -1 or end < start:
        return None, stripped
    end_pos = end + len(FOOTER_END_MARKER)
    footer_block = stripped[start:end_pos]
    body = stripped[:start].rstrip()
    suffix = stripped[end_pos:].strip()
    if suffix:
        return footer_block + suffix, body
    return footer_block, body


def parse_footer_block(block: str, *, allowed_roles: Collection[str]) -> BridgeFooter:
    if not block.startswith(FOOTER_MARKER) or not block.endswith(FOOTER_END_MARKER):
        raise FooterValidationError("footer_block_markers_invalid")
    inner = block[len(FOOTER_MARKER) : -len(FOOTER_END_MARKER)].strip()
    try:
        payload = yaml.safe_load(inner)
    except yaml.YAMLError as exc:
        raise FooterValidationError("footer_invalid_yaml") from exc
    if not isinstance(payload, dict):
        raise FooterValidationError("footer_not_mapping")

    raw_payload = cast(dict[object, object], payload)
    keys = {str(key) for key in raw_payload}
    missing = [key for key in REQUIRED_KEYS if key not in keys]
    if missing:
        joined = ",".join(missing)
        raise FooterValidationError(f"footer_missing_keys:{joined}")
    unknown = sorted(key for key in keys if key not in REQUIRED_KEYS)
    if unknown:
        raise FooterValidationError(f"footer_unknown_keys:{','.join(unknown)}")

    summary = raw_payload["summary"]
    if not isinstance(summary, str) or not summary.strip() or "\n" in summary:
        raise FooterValidationError("footer_summary_invalid")

    next_actor = raw_payload["next_actor"]
    if next_actor is not None:
        if not isinstance(next_actor, str) or next_actor not in set(allowed_roles):
            raise FooterValidationError("footer_next_actor_invalid")

    needs_human = raw_payload["needs_human"]
    if not isinstance(needs_human, bool):
        raise FooterValidationError("footer_needs_human_not_bool")

    done = raw_payload["done"]
    if not isinstance(done, bool):
        raise FooterValidationError("footer_done_not_bool")

    artifacts = raw_payload["artifacts"]
    if not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
        raise FooterValidationError("footer_artifacts_invalid")

    tests_run = raw_payload["tests_run"]
    if not isinstance(tests_run, list) or not all(isinstance(item, str) for item in tests_run):
        raise FooterValidationError("footer_tests_run_invalid")

    return BridgeFooter(
        summary=summary.strip(),
        next_actor=next_actor,
        needs_human=needs_human,
        done=done,
        artifacts=list(artifacts),
        tests_run=list(tests_run),
    )


def extract_footer(text: str, *, allowed_roles: Collection[str]) -> ParsedTurn:
    block, body = extract_footer_block(text)
    if block is None:
        return ParsedTurn(
            footer=None,
            body_without_footer=body,
            parse_status="missing",
            footer_raw=None,
            parse_errors=["footer_missing"],
        )

    if not re.search(
        rf"{re.escape(FOOTER_MARKER)}.*{re.escape(FOOTER_END_MARKER)}\s*$",
        text,
        flags=re.DOTALL,
    ):
        return ParsedTurn(
            footer=None,
            body_without_footer=body,
            parse_status="malformed",
            footer_raw=block,
            parse_errors=["footer_not_final_block"],
        )

    try:
        footer = parse_footer_block(block, allowed_roles=allowed_roles)
    except FooterValidationError as exc:
        return ParsedTurn(
            footer=None,
            body_without_footer=body,
            parse_status="malformed",
            footer_raw=block,
            parse_errors=[str(exc)],
        )

    return ParsedTurn(
        footer=footer,
        body_without_footer=body,
        parse_status="ok",
        footer_raw=block,
        parse_errors=[],
    )
