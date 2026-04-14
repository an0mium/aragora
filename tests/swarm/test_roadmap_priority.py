from __future__ import annotations

from pathlib import Path

from aragora.swarm.roadmap_priority import (
    RoadmapPriority,
    expand_roadmap_token,
    extract_roadmap_codes,
    load_roadmap_priority_policy,
)


def test_expand_roadmap_token_handles_ranges() -> None:
    assert expand_roadmap_token("BC-07..09") == ("BC-07", "BC-08", "BC-09")
    assert expand_roadmap_token("CS-01..03") == ("CS-01", "CS-02", "CS-03")
    assert expand_roadmap_token("RS-07") == ("RS-07",)


def test_extract_roadmap_codes_deduplicates_and_expands() -> None:
    text = "Refs: (`RS-07`), delayed `BC-07..08`, plus repeated `RS-07`."
    assert extract_roadmap_codes(text) == ("RS-07", "BC-07", "BC-08")


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
