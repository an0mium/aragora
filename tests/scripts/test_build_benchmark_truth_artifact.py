from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import build_benchmark_truth_artifact as mod  # noqa: E402


class FakeGitHubTruthClient:
    def __init__(self, *, issues: dict[int, dict], prs: dict[int, dict]) -> None:
        self.issues = issues
        self.prs = prs

    def get_issue(self, repo: str, number: int) -> dict:
        return self.issues[number]

    def get_pr(self, repo: str, number: int) -> dict:
        return self.prs[number]

    def get_cross_referenced_pr_numbers(self, repo: str, number: int) -> list[int]:
        return []


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_build_benchmark_truth_artifact_links_corpus_revision_and_truth_metrics(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "issue_number": 1064,
                        "issue_title": "Dependency bump",
                        "terminal_class": "deliverable_pr_created",
                        "publish_action": "pr_created",
                        "worker_outcome": "pr_adopted",
                    }
                ),
                json.dumps(
                    {
                        "issue_number": 2712,
                        "issue_title": "Boolean parsing fix",
                        "terminal_class": "rescue_worker_crash",
                        "worker_outcome": "crash",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 3,
            "recorded_on": "2026-04-13",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [
                {"issue_id": 1064, "title": "Dependency bump"},
                {"issue_id": 2712, "title": "Boolean parsing fix"},
            ],
        },
    )
    client = FakeGitHubTruthClient(
        issues={
            1064: {
                "title": "Dependency bump",
                "comments": [{"body": "PR: https://github.com/synaptent/aragora/pull/6001"}],
            },
            2712: {"title": "Boolean parsing fix", "comments": []},
        },
        prs={
            6001: {
                "number": 6001,
                "title": "merged fix",
                "url": "https://github.com/synaptent/aragora/pull/6001",
                "state": "MERGED",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "mergedAt": "2026-04-13T12:00:00Z",
                "isDraft": False,
            }
        },
    )

    artifact = mod.build_benchmark_truth_artifact(
        repo="synaptent/aragora",
        metrics_file=metrics_path,
        corpus_path=corpus_path,
        client=client,
        generated_at="2026-04-13T20:00:00Z",
    )

    assert artifact["corpus"]["corpus_id"] == "tw-01-bounded-execution-v1"
    assert artifact["corpus"]["revision"] == 3
    assert (
        artifact["corpus"]["manifest_sha256"]
        == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    )
    assert artifact["corpus"]["membership_issue_numbers"] == [1064, 2712]
    assert artifact["corpus"]["membership_sha256"] == hashlib.sha256(b"[1064,2712]").hexdigest()
    assert artifact["run_status"] == "complete"
    assert artifact["coverage"]["attempted_issue_count"] == 2
    assert artifact["coverage"]["missing_issue_numbers"] == []
    assert artifact["coverage"]["is_complete"] is True
    assert artifact["primary_metrics"]["truth_success_rate"] == 0.5
    assert artifact["primary_metrics"]["no_rescue_truth_success_rate"] == 0.5
    assert artifact["primary_metrics"]["merged_only_rate"] == 0.5
    assert artifact["failure_class_distribution"] == {"rescue_worker_crash": 1}
    assert artifact["rescue_counts_by_type"] == {"rescue_worker_crash": 1}
    assert artifact["corpus_freshness"]["status"] == "fresh"
    assert artifact["corpus_freshness"]["stale_closed_issue_count"] == 0
    assert artifact["proxy_metrics"]["attempted_issue_count"] == 2
    assert [issue["truth_state"] for issue in artifact["issues"]] == ["merged_pr", "no_linked_pr"]
    assert artifact["issues"][0]["stale_corpus_issue"] is False


def test_build_benchmark_truth_artifact_marks_partial_corpus_runs_incomplete(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "issue_number": 1064,
                "issue_title": "Dependency bump",
                "terminal_class": "deliverable_pr_created",
                "publish_action": "pr_created",
                "worker_outcome": "pr_adopted",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 4,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [
                {"issue_id": 1064, "title": "Dependency bump"},
                {"issue_id": 873, "title": "ESLint bump"},
            ],
        },
    )
    client = FakeGitHubTruthClient(
        issues={
            1064: {
                "title": "Dependency bump",
                "comments": [{"body": "PR: https://github.com/synaptent/aragora/pull/6001"}],
            },
            873: {"title": "ESLint bump", "comments": []},
        },
        prs={
            6001: {
                "number": 6001,
                "title": "merged fix",
                "url": "https://github.com/synaptent/aragora/pull/6001",
                "state": "MERGED",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "mergedAt": "2026-04-13T12:00:00Z",
                "isDraft": False,
            }
        },
    )

    artifact = mod.build_benchmark_truth_artifact(
        repo="synaptent/aragora",
        metrics_file=metrics_path,
        corpus_path=corpus_path,
        client=client,
        generated_at="2026-04-14T01:00:00Z",
    )

    assert artifact["run_status"] == "incomplete"
    assert artifact["corpus"]["membership_issue_numbers"] == [873, 1064]
    assert artifact["coverage"]["attempted_issue_count"] == 1
    assert artifact["coverage"]["missing_issue_count"] == 1
    assert artifact["coverage"]["missing_issue_numbers"] == [873]
    assert artifact["coverage"]["is_complete"] is False
    assert artifact["primary_metrics"]["truth_success_rate"] == 0.5
    assert [issue["truth_state"] for issue in artifact["issues"]] == ["not_attempted", "merged_pr"]


def test_build_benchmark_truth_artifact_does_not_count_historical_truth_for_unattempted_issues(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "issue_number": 1064,
                "issue_title": "Dependency bump",
                "terminal_class": "rescue_worker_crash",
                "worker_outcome": "crash",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 4,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [
                {"issue_id": 1064, "title": "Dependency bump"},
                {"issue_id": 873, "title": "ESLint bump"},
            ],
        },
    )
    client = FakeGitHubTruthClient(
        issues={
            1064: {"title": "Dependency bump", "comments": []},
            873: {
                "title": "ESLint bump",
                "comments": [{"body": "PR: https://github.com/synaptent/aragora/pull/6001"}],
            },
        },
        prs={
            6001: {
                "number": 6001,
                "title": "merged fix",
                "url": "https://github.com/synaptent/aragora/pull/6001",
                "state": "MERGED",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "mergedAt": "2026-04-13T12:00:00Z",
                "isDraft": False,
            }
        },
    )

    artifact = mod.build_benchmark_truth_artifact(
        repo="synaptent/aragora",
        metrics_file=metrics_path,
        corpus_path=corpus_path,
        client=client,
        generated_at="2026-04-14T01:00:00Z",
    )

    assert artifact["run_status"] == "incomplete"
    assert artifact["coverage"]["missing_issue_numbers"] == [873]
    assert artifact["primary_metrics"]["truth_success_rate"] == 0.0
    assert artifact["primary_metrics"]["no_rescue_truth_success_rate"] == 0.0
    assert [issue["truth_state"] for issue in artifact["issues"]] == [
        "not_attempted",
        "no_linked_pr",
    ]


def test_build_benchmark_truth_artifact_reports_stale_closed_corpus_issues(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "issue_number": 1733,
                "issue_title": "Detached worker cleanup",
                "terminal_class": "issue_already_resolved",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 4,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [
                {"issue_id": 1733, "title": "Detached worker cleanup"},
            ],
        },
    )
    client = FakeGitHubTruthClient(
        issues={
            1733: {
                "title": "Detached worker cleanup",
                "url": "https://github.com/synaptent/aragora/issues/1733",
                "state": "CLOSED",
                "stateReason": "COMPLETED",
                "closedAt": "2026-03-31T23:45:29Z",
                "closedByPullRequestsReferences": [],
                "comments": [],
            }
        },
        prs={},
    )

    artifact = mod.build_benchmark_truth_artifact(
        repo="synaptent/aragora",
        metrics_file=metrics_path,
        corpus_path=corpus_path,
        client=client,
        generated_at="2026-04-14T01:00:00Z",
    )

    assert artifact["primary_metrics"]["truth_success_rate"] == 0.0
    assert artifact["corpus_freshness"]["status"] == "stale_closed_issues_detected"
    assert artifact["corpus_freshness"]["stale_closed_issue_numbers"] == [1733]
    assert artifact["issues"][0]["stale_corpus_issue"] is True
    assert artifact["issues"][0]["issue_state"] == "CLOSED"


def test_build_benchmark_truth_artifact_surfaces_freshness_issue_draft(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "issue_number": 1733,
                "issue_title": "Detached worker cleanup",
                "terminal_class": "issue_already_resolved",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 4,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [
                {"issue_id": 1733, "title": "Detached worker cleanup"},
            ],
        },
    )
    freshness_map_path = _write_json(
        tmp_path / "benchmark_corpus_freshness.json",
        {
            "schema_version": 1,
            "entries": [],
        },
    )
    client = FakeGitHubTruthClient(
        issues={
            1733: {
                "title": "Detached worker cleanup",
                "url": "https://github.com/synaptent/aragora/issues/1733",
                "state": "CLOSED",
                "stateReason": "COMPLETED",
                "closedAt": "2026-03-31T23:45:29Z",
                "closedByPullRequestsReferences": [],
                "comments": [],
            }
        },
        prs={},
    )

    artifact = mod.build_benchmark_truth_artifact(
        repo="synaptent/aragora",
        metrics_file=metrics_path,
        corpus_path=corpus_path,
        client=client,
        generated_at="2026-04-14T01:00:00Z",
        freshness_map_path=freshness_map_path,
    )

    assert artifact["corpus_freshness"]["issue_map_path"] == str(freshness_map_path)
    assert artifact["corpus_freshness"]["linked_issue_count"] == 0
    assert artifact["corpus_freshness"]["unlinked_issue_count"] == 1
    assert artifact["corpus_freshness"]["issue_drafts"][0]["title"] == (
        "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-4"
    )


def test_ensure_corpus_freshness_issue_linkage_updates_map(
    tmp_path: Path,
    monkeypatch,
) -> None:
    freshness_map_path = _write_json(
        tmp_path / "benchmark_corpus_freshness.json",
        {
            "schema_version": 1,
            "entries": [],
        },
    )
    monkeypatch.setattr(
        mod,
        "find_existing_issue_by_title",
        lambda **_: {
            "number": 6001,
            "title": "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-4",
            "url": "https://github.com/synaptent/aragora/issues/6001",
            "state": "open",
        },
    )

    results = mod.ensure_corpus_freshness_issue_linkage(
        issue_drafts=[
            {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 4,
                "stale_issue_numbers": [1733],
                "title": "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-4",
                "body": "body",
            }
        ],
        freshness_map_path=freshness_map_path,
        repo="synaptent/aragora",
    )

    assert results == [
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 4,
            "action": "linked_existing_issue",
            "target_kind": "issue",
            "target": "#6001",
            "url": "https://github.com/synaptent/aragora/issues/6001",
        }
    ]
    written_map = json.loads(freshness_map_path.read_text(encoding="utf-8"))
    assert written_map["entries"] == [
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "notes": "Auto-linked by recurring TW-02 publication.",
            "revision": 4,
            "stale_issue_numbers": [1733],
            "target": "#6001",
            "target_kind": "issue",
            "title": "[TW-02] Restock stale issues in tw-01-bounded-execution-v1 rev-4",
            "url": "https://github.com/synaptent/aragora/issues/6001",
        }
    ]


def test_main_fail_incomplete_returns_nonzero_and_emits_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "issue_number": 1064,
                "issue_title": "Dependency bump",
                "terminal_class": "deliverable_pr_created",
                "publish_action": "pr_created",
                "worker_outcome": "pr_adopted",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 5,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [
                {"issue_id": 1064, "title": "Dependency bump"},
                {"issue_id": 873, "title": "ESLint bump"},
            ],
        },
    )
    fake_client = FakeGitHubTruthClient(
        issues={
            1064: {
                "title": "Dependency bump",
                "comments": [{"body": "PR: https://github.com/synaptent/aragora/pull/6001"}],
            },
            873: {"title": "ESLint bump", "comments": []},
        },
        prs={
            6001: {
                "number": 6001,
                "title": "merged fix",
                "url": "https://github.com/synaptent/aragora/pull/6001",
                "state": "MERGED",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "mergedAt": "2026-04-13T12:00:00Z",
                "isDraft": False,
            }
        },
    )

    monkeypatch.setattr(mod, "GitHubTruthClient", lambda: fake_client)

    exit_code = mod.main(
        [
            "--repo",
            "synaptent/aragora",
            "--metrics-file",
            str(metrics_path),
            "--corpus",
            str(corpus_path),
            "--json",
            "--fail-incomplete",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["run_status"] == "incomplete"
    assert payload["coverage"]["missing_issue_numbers"] == [873]
    assert "incomplete corpus coverage" in captured.err


def test_main_fail_incomplete_does_not_publish_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 1,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issues": [
                    {"issue_id": 1064, "title": "Issue A"},
                    {"issue_id": 873, "title": "Issue B"},
                ],
            }
        ),
        encoding="utf-8",
    )
    fake_client = FakeGitHubTruthClient(
        issues={
            1064: {"title": "Issue A", "comments": []},
            873: {"title": "Issue B", "comments": []},
        },
        prs={
            1064: {
                "number": 6002,
                "title": "mergeable fix",
                "url": "https://github.com/synaptent/aragora/pull/6002",
                "state": "OPEN",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "mergedAt": None,
                "isDraft": False,
            }
        },
    )

    monkeypatch.setattr(mod, "GitHubTruthClient", lambda: fake_client)

    publish_dir = tmp_path / "published"
    exit_code = mod.main(
        [
            "--repo",
            "synaptent/aragora",
            "--metrics-file",
            str(metrics_path),
            "--corpus",
            str(corpus_path),
            "--publish-dir",
            str(publish_dir),
            "--fail-incomplete",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "incomplete corpus coverage" in captured.err
    assert not publish_dir.exists()


def test_write_artifact_emits_diffable_json(tmp_path: Path) -> None:
    payload = {
        "generated_at": "2026-04-13T20:00:00Z",
        "corpus": {"corpus_id": "tw-01", "revision": 1, "issue_count": 1},
        "primary_metrics": {"truth_success_rate": 1.0},
    }
    output = tmp_path / "artifact.json"

    written = mod.write_artifact(output, payload)

    assert written == output
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["corpus"]["revision"] == 1
    assert parsed["primary_metrics"]["truth_success_rate"] == 1.0


def test_build_benchmark_truth_artifact_normalizes_generated_at_and_repo_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    repo_root = tmp_path / "repo"
    corpus_path = repo_root / "docs" / "benchmarks" / "corpus.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        json.dumps(
            {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 7,
                "recorded_on": "2026-04-14",
                "success_contract": "mergeable_pr_or_merged_pr",
                "issues": [{"issue_id": 1064, "title": "Dependency bump"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)

    artifact = mod.build_benchmark_truth_artifact(
        repo="synaptent/aragora",
        metrics_file=metrics_path,
        corpus_path=corpus_path,
        client=FakeGitHubTruthClient(
            issues={1064: {"title": "Dependency bump", "comments": []}},
            prs={},
        ),
        generated_at="2026-04-14T02:03:04+00:00",
    )

    assert artifact["generated_at"] == "2026-04-14T02:03:04Z"
    assert artifact["corpus"]["path"] == "docs/benchmarks/corpus.json"
    assert (
        artifact["corpus"]["manifest_sha256"]
        == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    )
    assert artifact["corpus"]["membership_issue_numbers"] == [1064]
    assert artifact["metrics_file"] == str(metrics_path)


def test_resolve_published_artifact_path_uses_corpus_revision_and_timestamp() -> None:
    path = mod.resolve_published_artifact_path(
        publish_dir=Path("/tmp/published"),
        artifact={
            "generated_at": "2026-04-14T02:03:04Z",
            "corpus": {
                "corpus_id": "TW-01 Bounded Execution v1",
                "revision": 7,
            },
        },
    )

    assert path == Path(
        "/tmp/published/tw-01-bounded-execution-v1/rev-7/truth-20260414T020304Z.json"
    )


def test_resolve_latest_artifact_paths_use_corpus_and_revision_roots() -> None:
    paths = mod.resolve_latest_artifact_paths(
        publish_dir=Path("/tmp/published"),
        artifact={
            "generated_at": "2026-04-14T02:03:04Z",
            "corpus": {
                "corpus_id": "TW-01 Bounded Execution v1",
                "revision": 7,
            },
        },
    )

    assert paths == {
        "corpus_latest": Path("/tmp/published/tw-01-bounded-execution-v1/latest.json"),
        "revision_latest": Path("/tmp/published/tw-01-bounded-execution-v1/rev-7/latest.json"),
    }


def test_main_publish_dir_writes_timestamped_artifact_and_prints_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 8,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [{"issue_id": 1064, "title": "Dependency bump"}],
        },
    )
    monkeypatch.setattr(
        mod,
        "build_benchmark_truth_artifact",
        lambda **_: {
            "generated_at": "2026-04-14T05:06:07Z",
            "corpus": {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 8,
                "issue_count": 1,
            },
            "primary_metrics": {"truth_success_rate": 1.0},
        },
    )

    exit_code = mod.main(
        [
            "--repo",
            "synaptent/aragora",
            "--metrics-file",
            str(metrics_path),
            "--corpus",
            str(corpus_path),
            "--publish-dir",
            str(tmp_path / "published"),
        ]
    )

    captured = capsys.readouterr()
    written_path = Path(captured.out.strip())
    assert exit_code == 0
    assert written_path == Path(
        tmp_path
        / "published"
        / "tw-01-bounded-execution-v1"
        / "rev-8"
        / "truth-20260414T050607Z.json"
    )
    parsed = json.loads(written_path.read_text(encoding="utf-8"))
    assert parsed["generated_at"] == "2026-04-14T05:06:07Z"
    assert (
        json.loads(
            (tmp_path / "published" / "tw-01-bounded-execution-v1" / "latest.json").read_text(
                encoding="utf-8"
            )
        )["generated_at"]
        == "2026-04-14T05:06:07Z"
    )
    assert (
        json.loads(
            (
                tmp_path / "published" / "tw-01-bounded-execution-v1" / "rev-8" / "latest.json"
            ).read_text(encoding="utf-8")
        )["generated_at"]
        == "2026-04-14T05:06:07Z"
    )
    assert captured.err == ""


def test_main_publish_dir_with_json_keeps_stdout_json_and_reports_path_on_stderr(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 9,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [{"issue_id": 1064, "title": "Dependency bump"}],
        },
    )
    monkeypatch.setattr(
        mod,
        "build_benchmark_truth_artifact",
        lambda **_: {
            "generated_at": "2026-04-14T08:09:10Z",
            "corpus": {
                "corpus_id": "tw-01-bounded-execution-v1",
                "revision": 9,
                "issue_count": 1,
            },
            "primary_metrics": {"truth_success_rate": 1.0},
        },
    )

    exit_code = mod.main(
        [
            "--repo",
            "synaptent/aragora",
            "--metrics-file",
            str(metrics_path),
            "--corpus",
            str(corpus_path),
            "--publish-dir",
            str(tmp_path / "published"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["generated_at"] == "2026-04-14T08:09:10Z"
    assert (
        json.loads(
            (tmp_path / "published" / "tw-01-bounded-execution-v1" / "latest.json").read_text(
                encoding="utf-8"
            )
        )["generated_at"]
        == "2026-04-14T08:09:10Z"
    )
    assert captured.err.strip() == str(
        tmp_path
        / "published"
        / "tw-01-bounded-execution-v1"
        / "rev-9"
        / "truth-20260414T080910Z.json"
    )
