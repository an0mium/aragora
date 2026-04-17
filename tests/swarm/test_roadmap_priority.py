from __future__ import annotations

from pathlib import Path

from aragora.swarm.roadmap_priority import (
    RoadmapPriority,
    RoadmapPriorityPolicy,
    expand_roadmap_token,
    extract_roadmap_codes,
    load_roadmap_priority_policy,
)


def test_expand_roadmap_token_handles_ranges() -> None:
    assert expand_roadmap_token("BC-07..09") == ("BC-07", "BC-08", "BC-09")
    assert expand_roadmap_token("CS-01..03") == ("CS-01", "CS-02", "CS-03")
    assert expand_roadmap_token("RS-07") == ("RS-07",)
    assert expand_roadmap_token("BC-09..07") == ("BC-09..07",)
    assert expand_roadmap_token("") == ()


def test_extract_roadmap_codes_deduplicates_and_expands() -> None:
    text = "Refs: (`RS-07`), delayed `BC-07..08`, plus repeated `RS-07`."
    assert extract_roadmap_codes(text) == ("RS-07", "BC-07", "BC-08")


def test_extract_roadmap_codes_preserves_first_seen_order_across_text_noise() -> None:
    text = """
    Follow `TW-01`, lowercase tw-02 is prose only, then `CS-01..03`.
    Repeating `CS-02` must not duplicate, and malformed `BC-09..07` stays atomic.
    """

    assert extract_roadmap_codes(text) == (
        "TW-01",
        "CS-01",
        "CS-02",
        "CS-03",
        "BC-09..07",
    )


def test_priority_policy_prefers_blocking_sections_and_reports_blocked_codes() -> None:
    policy = RoadmapPriorityPolicy(
        do_now=frozenset({"RS-07", "TW-01"}),
        delay=frozenset({"BC-07", "CS-02"}),
        avoid=frozenset({"UDW-08", "CS-03"}),
    )

    avoid = policy.priority_for_text("Do `RS-07`, delay `BC-07`, but avoid `UDW-08`.")
    assert avoid.priority == RoadmapPriority.AVOID
    assert avoid.codes == ("RS-07", "BC-07", "UDW-08")
    assert avoid.blocked_codes == ("UDW-08",)
    assert not policy.allows_boss_ready("Do `RS-07`, but avoid `UDW-08`.")

    delay = policy.priority_for_text("Do `TW-01`, but delay `BC-07`.")
    assert delay.priority == RoadmapPriority.DELAY
    assert delay.blocked_codes == ("BC-07",)
    assert not policy.allows_boss_ready("Delay `CS-02`.")

    do_now = policy.priority_for_text("Continue `RS-07`.")
    assert do_now.priority == RoadmapPriority.DO_NOW
    assert do_now.blocked_codes == ()
    assert policy.allows_boss_ready("Continue `RS-07`.")


def test_priority_policy_unknown_codes_do_not_block_boss_ready() -> None:
    policy = RoadmapPriorityPolicy(
        do_now=frozenset({"RS-07"}),
        delay=frozenset({"BC-07"}),
        avoid=frozenset({"UDW-08"}),
    )

    match = policy.priority_for_text("Unclassified `ZZ-99`.")

    assert match.priority == RoadmapPriority.UNKNOWN
    assert match.codes == ("ZZ-99",)
    assert match.blocked_codes == ()
    assert policy.allows_boss_ready("Unclassified `ZZ-99`.")
    assert not RoadmapPriority.DO_NOW.blocks_boss_ready
    assert not RoadmapPriority.UNKNOWN.blocks_boss_ready
    assert RoadmapPriority.DELAY.blocks_boss_ready
    assert RoadmapPriority.AVOID.blocks_boss_ready


def test_load_priority_policy_returns_none_when_canonical_file_is_missing(
    tmp_path: Path,
) -> None:
    assert load_roadmap_priority_policy(tmp_path) is None


def test_load_priority_policy_parses_canonical_sections(tmp_path: Path) -> None:
    docs = tmp_path / "docs" / "status"
    docs.mkdir(parents=True)
    (docs / "NEXT_STEPS_CANONICAL.md").write_text(
        "\n".join(
            [
                "## Do Now / Delay / Avoid",
                "",
                "### Do now",
                "- `RS-07`",
                "- `TW-01`",
                "",
                "### Delay",
                "- `BC-07..09` until truth exists",
                "",
                "### Avoid in this tranche",
                "- `UDW-07..08`",
            ]
        ),
        encoding="utf-8",
    )

    policy = load_roadmap_priority_policy(tmp_path)

    assert policy is not None
    assert policy.priority_for_text("Refs: (`RS-07`)").priority == RoadmapPriority.DO_NOW
    blocked = policy.priority_for_text("Refs: (`BC-07`) and (`TW-01`)")
    assert blocked.priority == RoadmapPriority.DELAY
    assert blocked.blocked_codes == ("BC-07",)
    avoid = policy.priority_for_text("Refs: (`UDW-08`)")
    assert avoid.priority == RoadmapPriority.AVOID


def test_load_priority_policy_resets_section_on_higher_level_heading(tmp_path: Path) -> None:
    docs = tmp_path / "docs" / "status"
    docs.mkdir(parents=True)
    (docs / "NEXT_STEPS_CANONICAL.md").write_text(
        "\n".join(
            [
                "### Avoid in this tranche",
                "- `CS-04..12`",
                "",
                "## Live Boss-Ready Queue",
                "- `TW-01`, `TW-02`, and `TW-03` publish through recurring status surfaces.",
            ]
        ),
        encoding="utf-8",
    )

    policy = load_roadmap_priority_policy(tmp_path)

    assert policy is not None
    assert "TW-02" not in policy.avoid
