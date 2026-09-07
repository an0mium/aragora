"""The Intake Register carries a single Strategy Mission Queue with a valid,
single-active invariant. This is the durable cross-session state for the
bounded-mission cadence (see docs/superpowers/specs/2026-06-26-strategy-as-
bounded-mission-cadence-design.md)."""

from __future__ import annotations

from pathlib import Path

REGISTER = Path("docs/status/ROADMAP_INTAKE_REGISTER.md")
REQUIRED_COLUMNS = ["id", "title", "tier", "status", "external-proof gate", "tracking"]
VALID_STATUS = {"queued", "active", "blocked-on-proof", "blocked-on-human", "done"}


def _section(text: str, header: str) -> str:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _rows(section: str) -> list[list[str]]:
    rows = []
    for ln in section.splitlines():
        if ln.startswith("|") and "---" not in ln:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(cells)
    return rows


def test_mission_queue_section_exists_and_is_valid():
    text = REGISTER.read_text(encoding="utf-8")
    section = _section(text, "## Strategy Mission Queue")
    rows = _rows(section)
    assert rows, "Strategy Mission Queue table is empty"
    header = [c.lower() for c in rows[0]]
    assert header == REQUIRED_COLUMNS, f"unexpected columns: {header}"
    data = rows[1:]
    assert data, "no mission rows"
    statuses = [r[3] for r in data]
    for s in statuses:
        assert s in VALID_STATUS, f"invalid status {s!r}"
    assert statuses.count("active") <= 1, "at most one mission may be active"
    ids = [r[0] for r in data]
    assert {"M1", "M2", "M3"}.issubset(set(ids)), f"missing mission rows: {ids}"
