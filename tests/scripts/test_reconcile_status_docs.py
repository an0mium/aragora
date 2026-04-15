from __future__ import annotations

from pathlib import Path

from scripts import reconcile_status_docs as mod


def _write_execution_docs(tmp_path: Path) -> tuple[Path, Path]:
    docs_status = tmp_path / "docs" / "status"
    docs_status.mkdir(parents=True)

    active = docs_status / "ACTIVE_EXECUTION_ISSUES.md"
    active.write_text(
        "\n".join(
            [
                "# Active Execution Issues",
                "- Current execution epics: "
                "[#804](https://github.com/synaptent/aragora/issues/804), "
                "[#805](https://github.com/synaptent/aragora/issues/805), "
                "[#806](https://github.com/synaptent/aragora/issues/806)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    next_steps = docs_status / "NEXT_STEPS_CANONICAL.md"
    next_steps.write_text(
        "\n".join(
            [
                "# Next Steps",
                "[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md)",
                "Current gate uses "
                "[#804](https://github.com/synaptent/aragora/issues/804), "
                "[#805](https://github.com/synaptent/aragora/issues/805), "
                "[#806](https://github.com/synaptent/aragora/issues/806).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return active, next_steps


def test_check_execution_issue_tracking_flags_closed_current_epics(
    monkeypatch, tmp_path: Path
) -> None:
    active, next_steps = _write_execution_docs(tmp_path)

    monkeypatch.setattr(mod, "ACTIVE_EXECUTION_ISSUES", active)
    monkeypatch.setattr(mod, "NEXT_STEPS_CANONICAL", next_steps)
    monkeypatch.setattr(
        mod,
        "EXECUTION_TRACKING_DOCS",
        [(next_steps, "status/NEXT_STEPS_CANONICAL.md")],
    )
    monkeypatch.setattr(mod, "REQUIRED_EXECUTION_ISSUES", [804, 805, 806])
    monkeypatch.setattr(
        mod,
        "CURRENT_EXECUTION_EPIC_DOCS",
        [
            (next_steps, "status/NEXT_STEPS_CANONICAL.md"),
            (active, "status/ACTIVE_EXECUTION_ISSUES.md"),
        ],
    )
    monkeypatch.setattr(
        mod,
        "_load_github_issue_metadata",
        lambda issue_number, repo=mod.DEFAULT_GITHUB_REPO: {
            "number": issue_number,
            "state": "CLOSED",
            "state_reason": "COMPLETED",
            "title": f"Issue {issue_number}",
            "url": f"https://github.com/synaptent/aragora/issues/{issue_number}",
        },
    )

    findings = mod._check_execution_issue_tracking()

    critical_sources = {item["source"] for item in findings if item["severity"] == "critical"}
    assert critical_sources == {
        "status/NEXT_STEPS_CANONICAL.md",
        "status/ACTIVE_EXECUTION_ISSUES.md",
    }
    assert any(
        "#804 (completed)" in item["message"] for item in findings if item["severity"] == "critical"
    )


def test_check_execution_issue_tracking_ignores_open_current_epics(
    monkeypatch, tmp_path: Path
) -> None:
    active, next_steps = _write_execution_docs(tmp_path)

    monkeypatch.setattr(mod, "ACTIVE_EXECUTION_ISSUES", active)
    monkeypatch.setattr(mod, "NEXT_STEPS_CANONICAL", next_steps)
    monkeypatch.setattr(
        mod,
        "EXECUTION_TRACKING_DOCS",
        [(next_steps, "status/NEXT_STEPS_CANONICAL.md")],
    )
    monkeypatch.setattr(mod, "REQUIRED_EXECUTION_ISSUES", [804, 805, 806])
    monkeypatch.setattr(
        mod,
        "CURRENT_EXECUTION_EPIC_DOCS",
        [
            (next_steps, "status/NEXT_STEPS_CANONICAL.md"),
            (active, "status/ACTIVE_EXECUTION_ISSUES.md"),
        ],
    )
    monkeypatch.setattr(
        mod,
        "_load_github_issue_metadata",
        lambda issue_number, repo=mod.DEFAULT_GITHUB_REPO: {
            "number": issue_number,
            "state": "OPEN",
            "state_reason": "",
            "title": f"Issue {issue_number}",
            "url": f"https://github.com/synaptent/aragora/issues/{issue_number}",
        },
    )

    findings = mod._check_execution_issue_tracking()

    assert all(item["severity"] != "critical" for item in findings)
