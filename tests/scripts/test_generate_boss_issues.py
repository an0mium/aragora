from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.swarm.issue_scanner import BossIssueCandidate
from aragora.swarm.roadmap_priority import RoadmapPriorityPolicy
from scripts import generate_boss_issues as mod


def _candidate(name: str, *, file_scope: list[str]) -> BossIssueCandidate:
    return BossIssueCandidate(
        category="test_coverage",
        title=f"Add tests for {name}",
        description=f"Create tests covering {name}.",
        file_scope=file_scope,
        new_files=[f"tests/test_{name}.py"],
        validation_command=f"pytest tests/test_{name}.py -v",
        acceptance_criteria=["All tests pass"],
    )


def _category_candidate(category: str, *, rel: str) -> BossIssueCandidate:
    validation_command = (
        f"pytest tests/test_{Path(rel).stem}.py -v"
        if category == "test_coverage"
        else f"ruff check {rel}"
    )
    new_files = [f"tests/test_{Path(rel).stem}.py"] if category == "test_coverage" else []
    return BossIssueCandidate(
        category=category,
        title=f"Candidate for {category}",
        description=f"Improve `{rel}` in a bounded way.",
        file_scope=[rel],
        new_files=new_files,
        validation_command=validation_command,
        acceptance_criteria=[f"`{validation_command}` passes"],
    )


def test_main_dry_run_fetches_and_filters_like_real_mode(
    monkeypatch,
    capsys,
) -> None:
    duplicate = _candidate("duplicate_module", file_scope=["aragora/duplicate_module.py"])
    pr_conflict = _candidate("pr_conflict_module", file_scope=["aragora/pr_conflict_module.py"])
    eligible = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])

    scan_calls: list[tuple[object, object, object]] = []
    fetch_existing_calls: list[str] = []
    fetch_pr_calls: list[str] = []
    create_calls: list[tuple[str, str, str, str]] = []

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: (
            scan_calls.append((repo_root, categories, min_success_rate))
            or [duplicate, pr_conflict, eligible]
        ),
    )
    monkeypatch.setattr(
        mod,
        "fetch_existing_boss_issues",
        lambda repo: (
            fetch_existing_calls.append(repo)
            or [{"title": "Other issue", "body": f"<!-- fingerprint:{duplicate.fingerprint} -->"}]
        ),
    )
    monkeypatch.setattr(
        mod,
        "fetch_open_pr_files",
        lambda repo: (fetch_pr_calls.append(repo) or {"aragora/pr_conflict_module.py"}),
    )
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        mod,
        "create_github_issue",
        lambda repo, title, body, label, **kwargs: (
            create_calls.append((repo, title, body, label, kwargs.get("extra_labels", []))) or True
        ),
    )
    monkeypatch.setattr(mod, "load_roadmap_priority_policy", lambda repo_root: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--dry-run",
            "--max-issues",
            "5",
            "--label",
            "lane:test",
        ],
    )

    mod.main()

    out = capsys.readouterr().out
    assert scan_calls
    assert scan_calls[0][2] == 0.3
    assert fetch_existing_calls == ["org/repo"]
    assert fetch_pr_calls == ["org/repo"]
    assert "DRY RUN — would create 1 issues" in out
    assert eligible.title in out
    assert (
        "Skipped: 1 duplicates, 1 PR conflicts, 0 canonical priority blocks, "
        "0 validation failures" in out
    )
    assert not create_calls


def test_main_create_mode_trims_to_max_and_writes_fingerprint(
    monkeypatch,
    capsys,
) -> None:
    first = _candidate("first_module", file_scope=["aragora/first_module.py"])
    second = _candidate("second_module", file_scope=["aragora/second_module.py"])

    created: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: [first, second],
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(mod, "load_roadmap_priority_policy", lambda repo_root: None)
    monkeypatch.setattr(
        mod,
        "create_github_issue",
        lambda repo, title, body, label, **kwargs: (
            created.append((repo, title, body, label, kwargs.get("extra_labels", []))) or True
        ),
    )
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--max-issues",
            "1",
            "--label",
            "lane:test",
        ],
    )

    mod.main()

    out = capsys.readouterr().out
    assert "Done: 1 created, 0 failed" in out
    assert len(created) == 1
    repo, title, body, label, extra_labels = created[0]
    assert repo == "org/repo"
    assert title == first.title
    assert label == "lane:test"
    assert extra_labels == ["autonomous"]
    assert f"<!-- fingerprint:{first.fingerprint} -->" in body


def test_main_passes_explicit_min_success_rate(monkeypatch, capsys) -> None:
    eligible = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])
    scan_calls: list[tuple[object, object, object]] = []

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: (
            scan_calls.append((repo_root, categories, min_success_rate)) or [eligible]
        ),
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(mod, "load_roadmap_priority_policy", lambda repo_root: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--dry-run",
            "--min-success-rate",
            "0.5",
        ],
    )

    mod.main()

    _ = capsys.readouterr()
    assert scan_calls
    assert scan_calls[0][2] == 0.5


def test_main_boss_ready_requires_explicit_do_now_priority(monkeypatch, capsys) -> None:
    unknown = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: [unknown],
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        mod,
        "load_roadmap_priority_policy",
        lambda repo_root: RoadmapPriorityPolicy(
            do_now=frozenset({"TW-01"}),
            delay=frozenset({"BC-07"}),
            avoid=frozenset({"CS-04"}),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--dry-run",
            "--label",
            "boss-ready",
        ],
    )

    mod.main()

    out = capsys.readouterr().out
    assert "DRY RUN — would create 0 issues" in out
    assert "1 canonical priority blocks" in out


def test_main_non_boss_ready_label_allows_unknown_priority(monkeypatch, capsys) -> None:
    unknown = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])
    created: list[tuple[str, str, str, str]] = []

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: [unknown],
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        mod,
        "load_roadmap_priority_policy",
        lambda repo_root: RoadmapPriorityPolicy(
            do_now=frozenset({"TW-01"}),
            delay=frozenset({"BC-07"}),
            avoid=frozenset({"CS-04"}),
        ),
    )
    monkeypatch.setattr(
        mod,
        "create_github_issue",
        lambda repo, title, body, label, **kwargs: (
            created.append((repo, title, body, label, kwargs.get("extra_labels", []))) or True
        ),
    )
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--max-issues",
            "1",
            "--label",
            "lane:test",
        ],
    )

    mod.main()

    out = capsys.readouterr().out
    assert "Done: 1 created, 0 failed" in out
    assert created[0][3] == "lane:test"


def test_main_boss_ready_allows_tw02_benchmark_follow_up_without_do_now_code(
    monkeypatch,
    capsys,
) -> None:
    candidate = BossIssueCandidate(
        category="test_coverage",
        title="[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-1",
        description=(
            "Refresh benchmark corpus freshness by updating docs/benchmarks/corpus.json "
            "after stale closed issues were detected."
        ),
        file_scope=["docs/benchmarks/corpus.json"],
        new_files=[],
        validation_command="python3 scripts/measure_b0_scorecard.py --json",
        acceptance_criteria=[
            "Recurring benchmark truth publication reports fresh corpus membership."
        ],
    )
    created: list[tuple[str, str, str, str]] = []

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: [candidate],
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        mod,
        "load_roadmap_priority_policy",
        lambda repo_root: RoadmapPriorityPolicy(
            do_now=frozenset({"CS-01", "CS-02", "CS-03"}),
            delay=frozenset({"BC-07"}),
            avoid=frozenset({"CS-04"}),
        ),
    )
    monkeypatch.setattr(
        mod,
        "create_github_issue",
        lambda repo, title, body, label, **kwargs: (
            created.append((repo, title, body, label, kwargs.get("extra_labels", []))) or True
        ),
    )
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--max-issues",
            "1",
            "--label",
            "boss-ready",
        ],
    )

    mod.main()

    out = capsys.readouterr().out
    assert "Done: 1 created, 0 failed" in out
    assert created[0][1] == candidate.title


def test_fetch_existing_boss_issues_includes_fingerprinted_open_issues_without_label_filter(
    monkeypatch,
) -> None:
    seen_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
        seen_cmds.append(cmd)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "number": 5529,
                        "title": "Duplicate fingerprint issue",
                        "body": "Task body\n\n<!-- fingerprint:abc123 -->",
                    },
                    {
                        "number": 5574,
                        "title": "Open issue without fingerprint",
                        "body": "Task body without dedupe marker",
                    },
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    issues = mod.fetch_existing_boss_issues("org/repo")

    assert issues == [
        {
            "number": 5529,
            "title": "Duplicate fingerprint issue",
            "body": "Task body\n\n<!-- fingerprint:abc123 -->",
        }
    ]
    assert seen_cmds == [
        [
            "gh",
            "issue",
            "list",
            "--repo",
            "org/repo",
            "--state",
            "open",
            "--limit",
            str(mod._OPEN_ISSUE_LIMIT),
            "--json",
            "number,title,body",
        ]
    ]


def test_fetch_existing_boss_issues_returns_empty_on_invalid_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda cmd, **_: SimpleNamespace(returncode=0, stdout=json.dumps({"items": []}), stderr=""),
    )

    assert mod.fetch_existing_boss_issues("org/repo") == []


@pytest.mark.parametrize(
    "category",
    ["test_coverage", "broad_exception", "silent_exception", "type_annotation"],
)
def test_format_boss_ready_body_uses_upgrader_for_supported_categories(
    monkeypatch,
    category: str,
) -> None:
    candidate = _category_candidate(category, rel="aragora/swarm/example_module.py")
    calls: list[dict[str, object]] = []

    def _upgrade(title, body, **kwargs):  # noqa: ANN001
        calls.append({"title": title, "body": body, **kwargs})
        return SimpleNamespace(upgraded_body="## Task\n\nUpgraded body")

    monkeypatch.setattr(mod, "upgrade_issue_heuristic", _upgrade)

    body = mod.format_boss_ready_body(candidate)

    assert body.startswith("## Task\n\nUpgraded body")
    assert f"<!-- fingerprint:{candidate.fingerprint} -->" in body
    assert calls[0]["category"] == category
    assert calls[0]["validation_command"] == candidate.validation_command
    assert calls[0]["acceptance_criteria"] == candidate.acceptance_criteria
    assert calls[0]["new_files"] == candidate.new_files


def test_format_boss_ready_body_falls_back_when_non_test_upgrade_is_unavailable(
    monkeypatch,
) -> None:
    candidate = _category_candidate("type_annotation", rel="aragora/swarm/example_module.py")
    monkeypatch.setattr(mod, "upgrade_issue_heuristic", lambda *args, **kwargs: None)

    body = mod.format_boss_ready_body(candidate)

    assert candidate.description in body
    assert "### File Scope" in body
    assert "`aragora/swarm/example_module.py`" in body
    assert candidate.validation_command in body
    assert f"<!-- fingerprint:{candidate.fingerprint} -->" in body


def test_maybe_decompose_candidates_noop_when_disabled(monkeypatch) -> None:
    parent = _candidate("parent_module", file_scope=["aragora/parent_module.py"])

    class UnexpectedBridge:
        def __init__(self, repo_root):  # noqa: D401, ANN001
            raise AssertionError("bridge should not be constructed when disabled")

    monkeypatch.setattr(mod, "DecompositionBridge", UnexpectedBridge)

    result = mod.maybe_decompose_candidates(
        [parent],
        enabled=False,
        max_children_per_parent=5,
        repo_root=mod.REPO_ROOT,
    )

    assert result == [parent]


def test_maybe_decompose_candidates_replaces_parent_when_children_emitted(monkeypatch) -> None:
    parent = _candidate("parent_module", file_scope=["aragora/parent_module.py"])
    child_a = BossIssueCandidate(
        category="test_coverage",
        title="Child A",
        description="Write focused tests for module A with bounded scope and validation.",
        file_scope=["aragora/module_a.py"],
        new_files=["tests/test_module_a.py"],
        validation_command="python3 -m pytest -q tests/test_module_a.py",
    )
    child_b = BossIssueCandidate(
        category="test_coverage",
        title="Child B",
        description="Write focused tests for module B with bounded scope and validation.",
        file_scope=["aragora/module_b.py"],
        new_files=["tests/test_module_b.py"],
        validation_command="python3 -m pytest -q tests/test_module_b.py",
    )
    monkeypatch.setattr(
        mod,
        "format_boss_ready_body",
        lambda candidate: (
            "## Task\n\nAdd comprehensive unit tests.\n\n"
            "### Requirements\n"
            "1. Read the module and identify all public functions.\n"
            "2. Create a test file with broad coverage.\n"
        ),
    )

    class FakeBridge:
        def __init__(self, repo_root):  # noqa: D401, ANN001
            self.repo_root = repo_root

        def decompose_issue_sync_with_stats(self, title, body, *, max_children):  # noqa: ANN001
            assert title == parent.title
            assert "## Task" in body
            assert max_children == 3
            return SimpleNamespace(
                children=[child_a, child_b],
                stats=SimpleNamespace(rejected_candidates=1, sanitizer_rejections=1),
            )

    monkeypatch.setattr(mod, "DecompositionBridge", FakeBridge)

    result = mod.maybe_decompose_candidates(
        [parent],
        enabled=True,
        max_children_per_parent=3,
        repo_root=mod.REPO_ROOT,
    )

    assert result == [child_a, child_b]


def test_maybe_decompose_candidates_keeps_parent_when_child_set_is_not_meaningful(
    monkeypatch,
) -> None:
    parent = _candidate("parent_module", file_scope=["aragora/parent_module.py"])
    single_child = BossIssueCandidate(
        category="test_coverage",
        title="Only child",
        description="Only child with bounded scope and validation command.",
        file_scope=["aragora/module_a.py"],
        validation_command="python3 -m ruff check aragora/module_a.py",
    )
    monkeypatch.setattr(
        mod,
        "format_boss_ready_body",
        lambda candidate: (
            "## Task\n\nAdd comprehensive unit tests.\n\n"
            "### Requirements\n"
            "1. Read the module and identify all public functions.\n"
            "2. Create a test file with broad coverage.\n"
        ),
    )

    class FakeBridge:
        def __init__(self, repo_root):  # noqa: D401, ANN001
            self.repo_root = repo_root

        def decompose_issue_sync_with_stats(self, title, body, *, max_children):  # noqa: ANN001
            return SimpleNamespace(
                children=[single_child],
                stats=SimpleNamespace(rejected_candidates=2, sanitizer_rejections=1),
            )

    monkeypatch.setattr(mod, "DecompositionBridge", FakeBridge)

    result = mod.maybe_decompose_candidates(
        [parent],
        enabled=True,
        max_children_per_parent=5,
        repo_root=mod.REPO_ROOT,
    )

    assert result == [parent]


def test_is_low_quality_parent_skips_bounded_module_aware_issue() -> None:
    candidate = _candidate(
        "analytics_core", file_scope=["aragora/server/handlers/analytics/core.py"]
    )
    body = (
        "## Task\n\n"
        "Write focused unit tests for `aragora/server/handlers/analytics/core.py` (53 lines, 0 public functions).\n\n"
        "**Module purpose:** Analytics Core Module.\n\n"
        "### What to test\n"
        "- happy path behavior\n\n"
        "### Validation\n```bash\npytest tests/test_analytics_core.py -v\n```"
    )

    assert mod.is_low_quality_parent(candidate, body) is False


def test_is_low_quality_parent_detects_generic_template() -> None:
    candidate = _candidate("module", file_scope=["aragora/pkg/module.py"])
    body = (
        "## Task\n\n"
        "Add comprehensive unit tests for `aragora/pkg/module.py`.\n\n"
        "### Requirements\n"
        "1. Read the module and identify all public functions.\n"
        "2. Create a test file with broad coverage.\n"
    )

    assert mod.is_low_quality_parent(candidate, body) is True


def test_maybe_decompose_candidates_with_telemetry_tracks_counts(monkeypatch) -> None:
    low_quality = _candidate("generic_module", file_scope=["aragora/generic_module.py"])
    bounded = _candidate("bounded_module", file_scope=["aragora/bounded_module.py"])
    child_a = _candidate("child_a", file_scope=["aragora/child_a.py"])
    child_b = _candidate("child_b", file_scope=["aragora/child_b.py"])

    original_formatter = mod.format_boss_ready_body

    def fake_format(candidate: BossIssueCandidate) -> str:
        if candidate.title == low_quality.title:
            return (
                "## Task\n\n"
                "Add comprehensive unit tests for `aragora/generic_module.py`.\n\n"
                "### Requirements\n"
                "1. Read the module and identify all public functions.\n"
                "2. Create a test file with broad coverage.\n"
            )
        return original_formatter(candidate)

    class FakeBridge:
        def __init__(self, repo_root):  # noqa: D401, ANN001
            self.repo_root = repo_root

        def decompose_issue_sync_with_stats(self, title, body, *, max_children):  # noqa: ANN001
            assert title == low_quality.title
            return SimpleNamespace(
                children=[child_a, child_b],
                stats=SimpleNamespace(rejected_candidates=3, sanitizer_rejections=2),
            )

    monkeypatch.setattr(mod, "format_boss_ready_body", fake_format)
    monkeypatch.setattr(mod, "DecompositionBridge", FakeBridge)

    result, telemetry = mod.maybe_decompose_candidates_with_telemetry(
        [low_quality, bounded],
        enabled=True,
        max_children_per_parent=4,
        repo_root=mod.REPO_ROOT,
    )

    assert result == [child_a, child_b, bounded]
    assert telemetry.parents_seen == 2
    assert telemetry.parents_eligible == 1
    assert telemetry.parents_replaced == 1
    assert telemetry.parents_preserved == 1
    assert telemetry.children_emitted == 2
    assert telemetry.children_rejected == 3
    assert telemetry.sanitizer_rejections == 2


def test_main_passes_decomposition_flags(monkeypatch, capsys) -> None:
    eligible = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])
    scan_calls: list[tuple[object, object, object]] = []
    decompose_calls: list[tuple[list[BossIssueCandidate], bool, int, object]] = []

    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: (
            scan_calls.append((repo_root, categories, min_success_rate)) or [eligible]
        ),
    )
    monkeypatch.setattr(
        mod,
        "maybe_decompose_candidates_with_telemetry",
        lambda candidates, *, enabled, max_children_per_parent, repo_root: (
            decompose_calls.append((list(candidates), enabled, max_children_per_parent, repo_root))
            or (
                list(candidates),
                mod.DecompositionTelemetry(
                    parents_seen=len(candidates),
                    parents_preserved=len(candidates),
                ),
            )
        ),
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--dry-run",
            "--decompose-low-quality",
            "--max-children-per-parent",
            "4",
        ],
    )

    mod.main()

    _ = capsys.readouterr()
    assert scan_calls
    assert decompose_calls
    assert decompose_calls[0][1] is True
    assert decompose_calls[0][2] == 4


def test_fetch_open_pr_files_paginates_open_prs_and_pr_files(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(mod, "_OPEN_PR_PAGE_SIZE", 2)
    monkeypatch.setattr(mod, "_OPEN_PR_FILES_PAGE_SIZE", 2)

    def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
        endpoint = cmd[-1]
        calls.append(endpoint)
        if endpoint.endswith("/pulls?state=open&per_page=2&page=1"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"number": 101}, {"number": 102}]),
                stderr="",
            )
        if endpoint.endswith("/pulls/101/files?per_page=2&page=1"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [{"filename": "aragora/first.py"}, {"filename": "aragora/second.py"}]
                ),
                stderr="",
            )
        if endpoint.endswith("/pulls/101/files?per_page=2&page=2"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"filename": "aragora/third.py"}]),
                stderr="",
            )
        if endpoint.endswith("/pulls/102/files?per_page=2&page=1"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"filename": "aragora/fourth.py"}]),
                stderr="",
            )
        if endpoint.endswith("/pulls?state=open&per_page=2&page=2"):
            return SimpleNamespace(returncode=0, stdout=json.dumps([{"number": 103}]), stderr="")
        if endpoint.endswith("/pulls/103/files?per_page=2&page=1"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"filename": "aragora/fifth.py"}]),
                stderr="",
            )
        if endpoint.endswith("/pulls?state=open&per_page=2&page=3"):
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        raise AssertionError(f"unexpected gh api call: {endpoint}")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    files = mod.fetch_open_pr_files("org/repo")

    assert files == {
        "aragora/first.py",
        "aragora/second.py",
        "aragora/third.py",
        "aragora/fourth.py",
        "aragora/fifth.py",
    }
    assert "repos/org/repo/pulls?state=open&per_page=2&page=2" in calls
    assert "repos/org/repo/pulls/101/files?per_page=2&page=2" in calls


def test_fetch_open_pr_files_raises_when_open_pr_pagination_cap_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_OPEN_PR_PAGE_SIZE", 1)
    monkeypatch.setattr(mod, "_OPEN_PR_MAX_PAGES", 2)

    def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
        endpoint = cmd[-1]
        if endpoint.endswith("/pulls?state=open&per_page=1&page=1"):
            return SimpleNamespace(returncode=0, stdout=json.dumps([{"number": 101}]), stderr="")
        if endpoint.endswith("/pulls/101/files?per_page=100&page=1"):
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        if endpoint.endswith("/pulls?state=open&per_page=1&page=2"):
            return SimpleNamespace(returncode=0, stdout=json.dumps([{"number": 102}]), stderr="")
        if endpoint.endswith("/pulls/102/files?per_page=100&page=1"):
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        raise AssertionError(f"unexpected gh api call: {endpoint}")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="open PR pagination exceeded configured cap"):
        mod.fetch_open_pr_files("org/repo")


def test_fetch_open_pr_files_raises_when_pr_files_pagination_cap_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_OPEN_PR_PAGE_SIZE", 1)
    monkeypatch.setattr(mod, "_OPEN_PR_FILES_PAGE_SIZE", 1)
    monkeypatch.setattr(mod, "_OPEN_PR_FILES_MAX_PAGES", 2)

    def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
        endpoint = cmd[-1]
        if endpoint.endswith("/pulls?state=open&per_page=1&page=1"):
            return SimpleNamespace(returncode=0, stdout=json.dumps([{"number": 101}]), stderr="")
        if endpoint.endswith("/pulls/101/files?per_page=1&page=1"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"filename": "aragora/first.py"}]),
                stderr="",
            )
        if endpoint.endswith("/pulls/101/files?per_page=1&page=2"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"filename": "aragora/second.py"}]),
                stderr="",
            )
        raise AssertionError(f"unexpected gh api call: {endpoint}")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="open PR file pagination exceeded configured cap"):
        mod.fetch_open_pr_files("org/repo")


def test_main_returns_error_when_open_pr_pagination_exhausted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: [],
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])

    def raise_pagination_error(repo: str) -> set[str]:
        raise RuntimeError("open PR pagination exceeded configured cap (10 pages) for org/repo")

    monkeypatch.setattr(mod, "fetch_open_pr_files", raise_pagination_error)
    monkeypatch.setattr(sys, "argv", ["generate_boss_issues.py", "--repo", "org/repo"])

    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "Error: open PR pagination exceeded configured cap (10 pages) for org/repo" in out


def test_create_github_issue_passes_extra_labels_as_repeated_flags(monkeypatch) -> None:
    """create_github_issue must add each extra label as its own --label flag.

    Without this, issues created by generate_boss_issues.py only carry the
    primary label and are skipped by the boss-loop dispatcher, which requires
    both `boss-ready` and `autonomous` (#5997 followup).
    """
    captured: list[list[str]] = []

    class _Result:
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    ok = mod.create_github_issue(
        "org/repo",
        "title",
        "body",
        "boss-ready",
        extra_labels=["autonomous", "boss-ready", "  "],
    )
    assert ok is True
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:3] == ["gh", "issue", "create"]
    label_flags = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--label"]
    assert label_flags == ["boss-ready", "autonomous"]


def test_main_omits_extra_labels_when_disabled(monkeypatch, capsys) -> None:
    """Passing --extra-labels='' yields a single primary label only."""

    eligible = _candidate("eligible_module", file_scope=["aragora/eligible_module.py"])
    created: list[tuple] = []
    monkeypatch.setattr(
        mod,
        "scan_all",
        lambda repo_root, categories=None, min_success_rate=0.3: [eligible],
    )
    monkeypatch.setattr(mod, "fetch_existing_boss_issues", lambda repo: [])
    monkeypatch.setattr(mod, "fetch_open_pr_files", lambda repo: set())
    monkeypatch.setattr(mod, "validate_body", lambda body: (True, ""))
    monkeypatch.setattr(mod, "load_roadmap_priority_policy", lambda repo_root: None)
    monkeypatch.setattr(
        mod,
        "create_github_issue",
        lambda repo, title, body, label, **kwargs: (
            created.append((repo, title, body, label, kwargs.get("extra_labels", []))) or True
        ),
    )
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_boss_issues.py",
            "--repo",
            "org/repo",
            "--max-issues",
            "1",
            "--label",
            "lane:test",
            "--extra-labels",
            "",
        ],
    )

    mod.main()

    assert len(created) == 1
    assert created[0][4] == []
