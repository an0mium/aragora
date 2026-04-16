from __future__ import annotations

from pathlib import Path

from scripts import reconcile_proof_first_queue as mod


def test_reconcile_proof_first_queue_dry_run_removes_noncanonical_and_plans_docs_issue(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "list_open_queue_issues",
        lambda **kwargs: [
            {
                "number": 1,
                "title": "Replace silent exception swallowing in postgres_store.py",
                "body": "Generic cleanup only.",
                "labels": [{"name": "boss-ready"}],
                "url": "https://github.com/org/repo/issues/1",
            },
            {
                "number": 2,
                "title": "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-1",
                "body": "Refresh benchmark corpus freshness after stale closed issues were detected.",
                "labels": [{"name": "boss-ready"}],
                "url": "https://github.com/org/repo/issues/2",
            },
        ],
    )
    monkeypatch.setattr(
        mod,
        "load_status_reconciliation_report",
        lambda **kwargs: {
            "summary": {"critical": 0, "warning": 1, "info": 0},
            "findings": [
                {
                    "severity": "warning",
                    "source": "status/NEXT_STEPS_CANONICAL.md",
                    "message": "Docs claims outrun measured proof.",
                }
            ],
        },
    )
    monkeypatch.setattr(mod, "find_existing_open_issue_by_title", lambda **kwargs: None)

    report = mod.reconcile_proof_first_queue(
        repo="org/repo",
        repo_root=Path("/tmp/repo"),
        apply=False,
    )

    assert [item["number"] for item in report["kept"]] == [2]
    assert [item["number"] for item in report["removed"]] == [1]
    assert report["removed"][0]["action"] == "would_remove_label"
    assert report["docs_issue"] == {
        "action": "would_create_issue",
        "title": mod.DEFAULT_DOCS_ISSUE_TITLE,
    }


def test_reconcile_proof_first_queue_apply_removes_label_and_reuses_existing_docs_issue(
    monkeypatch,
) -> None:
    removed: list[int] = []
    monkeypatch.setattr(
        mod,
        "list_open_queue_issues",
        lambda **kwargs: [
            {
                "number": 5,
                "title": "Replace silent exception swallowing in postgres_store.py",
                "body": "Generic cleanup only.",
                "labels": [{"name": "boss-ready"}],
                "url": "https://github.com/org/repo/issues/5",
            }
        ],
    )
    monkeypatch.setattr(
        mod,
        "remove_queue_label",
        lambda **kwargs: removed.append(int(kwargs["issue_number"])),
    )
    monkeypatch.setattr(
        mod,
        "load_status_reconciliation_report",
        lambda **kwargs: {
            "summary": {"critical": 1, "warning": 0, "info": 0},
            "findings": [
                {
                    "severity": "critical",
                    "source": "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
                    "message": "Drift",
                }
            ],
        },
    )
    monkeypatch.setattr(
        mod,
        "find_existing_open_issue_by_title",
        lambda **kwargs: {
            "number": 42,
            "title": mod.DEFAULT_DOCS_ISSUE_TITLE,
            "url": "https://github.com/org/repo/issues/42",
        },
    )

    report = mod.reconcile_proof_first_queue(
        repo="org/repo",
        repo_root=Path("/tmp/repo"),
        apply=True,
    )

    assert removed == [5]
    assert report["docs_issue"] == {
        "action": "existing_issue",
        "title": mod.DEFAULT_DOCS_ISSUE_TITLE,
        "number": 42,
        "url": "https://github.com/org/repo/issues/42",
    }


def test_reconcile_proof_first_queue_reports_github_unavailable_without_crashing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "list_open_queue_issues",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("error connecting to api.github.com\ncheck your internet connection")
        ),
    )
    monkeypatch.setattr(
        mod,
        "load_status_reconciliation_report",
        lambda **kwargs: {
            "summary": {"critical": 0, "warning": 1, "info": 0},
            "findings": [
                {
                    "severity": "warning",
                    "source": "status/NEXT_STEPS_CANONICAL.md",
                    "message": "Docs claims outrun measured proof.",
                }
            ],
        },
    )

    report = mod.reconcile_proof_first_queue(
        repo="org/repo",
        repo_root=Path("/tmp/repo"),
        apply=True,
    )

    assert report["kept"] == []
    assert report["removed"] == []
    assert report["docs_issue"] == {
        "action": "deferred_github_unavailable",
        "title": mod.DEFAULT_DOCS_ISSUE_TITLE,
        "error": "error connecting to api.github.com\ncheck your internet connection",
    }
    assert report["github_status"] == {
        "available": False,
        "operation": "list_open_queue_issues",
        "error": "error connecting to api.github.com\ncheck your internet connection",
    }
