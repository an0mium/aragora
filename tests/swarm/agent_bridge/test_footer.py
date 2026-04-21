from __future__ import annotations

import pytest

from aragora.swarm.agent_bridge.footer import FOOTER_END_MARKER
from aragora.swarm.agent_bridge.footer import FOOTER_MARKER
from aragora.swarm.agent_bridge.footer import build_repair_prompt
from aragora.swarm.agent_bridge.footer import extract_footer

ALLOWED_ROLES = {"reviewer", "implementer"}


def _message(body: str, footer_lines: list[str]) -> str:
    footer = "\n".join([FOOTER_MARKER, *footer_lines, FOOTER_END_MARKER])
    return f"{body}\n\n{footer}"


def test_extract_footer_ok_and_missing() -> None:
    ok_message = _message(
        "Body",
        [
            "summary: Looks good",
            "next_actor: implementer",
            "needs_human: false",
            "done: false",
            "artifacts: []",
            "tests_run: []",
        ],
    )
    parsed_ok = extract_footer(ok_message, allowed_roles=ALLOWED_ROLES)
    parsed_missing = extract_footer("No footer here", allowed_roles=ALLOWED_ROLES)

    assert parsed_ok.parse_status == "ok"
    assert parsed_ok.footer is not None
    assert parsed_ok.footer.next_actor == "implementer"
    assert parsed_ok.body_without_footer == "Body"
    assert parsed_missing.parse_status == "missing"
    assert parsed_missing.parse_errors == ["footer_missing"]


@pytest.mark.parametrize(
    ("footer_lines", "reason"),
    [
        (
            [
                "summary: [",
                "next_actor: reviewer",
                "needs_human: false",
                "done: false",
                "artifacts: []",
                "tests_run: []",
            ],
            "footer_invalid_yaml",
        ),
        (
            [
                "summary: ok",
                "next_actor: reviewer",
                "needs_human: false",
                "done: false",
                "artifacts: []",
                "tests_run: []",
                "extra: nope",
            ],
            "footer_unknown_keys:extra",
        ),
        (
            [
                "summary: ''",
                "next_actor: reviewer",
                "needs_human: false",
                "done: false",
                "artifacts: []",
                "tests_run: []",
            ],
            "footer_summary_invalid",
        ),
        (
            [
                "summary: ok",
                "next_actor: qa",
                "needs_human: false",
                "done: false",
                "artifacts: []",
                "tests_run: []",
            ],
            "footer_next_actor_invalid",
        ),
        (
            [
                "summary: ok",
                "next_actor: reviewer",
                'needs_human: "false"',
                "done: false",
                "artifacts: []",
                "tests_run: []",
            ],
            "footer_needs_human_not_bool",
        ),
        (
            [
                "summary: ok",
                "next_actor: reviewer",
                "needs_human: false",
                'done: "false"',
                "artifacts: []",
                "tests_run: []",
            ],
            "footer_done_not_bool",
        ),
        (
            [
                "summary: ok",
                "next_actor: reviewer",
                "needs_human: false",
                "done: false",
                "artifacts: not-a-list",
                "tests_run: []",
            ],
            "footer_artifacts_invalid",
        ),
        (
            [
                "summary: ok",
                "next_actor: reviewer",
                "needs_human: false",
                "done: false",
                "artifacts: []",
                "tests_run: not-a-list",
            ],
            "footer_tests_run_invalid",
        ),
    ],
)
def test_extract_footer_strict_rejection_cases(
    footer_lines: list[str],
    reason: str,
) -> None:
    parsed = extract_footer(_message("Body", footer_lines), allowed_roles=ALLOWED_ROLES)

    assert parsed.parse_status == "malformed"
    assert parsed.parse_errors == [reason]


def test_extract_footer_rejects_missing_required_keys() -> None:
    parsed = extract_footer(
        _message(
            "Body",
            [
                "summary: ok",
                "next_actor: reviewer",
                "needs_human: false",
                "done: false",
                "artifacts: []",
            ],
        ),
        allowed_roles=ALLOWED_ROLES,
    )

    assert parsed.parse_status == "malformed"
    assert parsed.parse_errors == ["footer_missing_keys:tests_run"]


def test_extract_footer_rejects_non_final_footer_block() -> None:
    parsed = extract_footer(
        _message(
            "Body",
            [
                "summary: ok",
                "next_actor: reviewer",
                "needs_human: false",
                "done: false",
                "artifacts: []",
                "tests_run: []",
            ],
        )
        + "\nTrailing text",
        allowed_roles=ALLOWED_ROLES,
    )

    assert parsed.parse_status == "malformed"
    assert parsed.parse_errors == ["footer_not_final_block"]


def test_build_repair_prompt_surfaces_allowed_roles_and_errors() -> None:
    prompt = build_repair_prompt(
        parse_errors=["footer_next_actor_invalid"],
        original_message="Hello",
        allowed_roles=ALLOWED_ROLES,
    )

    assert "Return ONLY a corrected footer block" in prompt
    assert "footer_next_actor_invalid" in prompt
    assert "implementer, reviewer" in prompt
