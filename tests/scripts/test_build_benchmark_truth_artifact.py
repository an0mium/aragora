from __future__ import annotations

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
    assert artifact["run_status"] == "complete"
    assert artifact["coverage"]["attempted_issue_count"] == 2
    assert artifact["coverage"]["missing_issue_numbers"] == []
    assert artifact["coverage"]["is_complete"] is True
    assert artifact["primary_metrics"]["truth_success_rate"] == 0.5
    assert artifact["primary_metrics"]["no_rescue_truth_success_rate"] == 0.5
    assert artifact["primary_metrics"]["merged_only_rate"] == 0.5
    assert artifact["failure_class_distribution"] == {"rescue_worker_crash": 1}
    assert artifact["rescue_counts_by_type"] == {"rescue_worker_crash": 1}
    assert artifact["proxy_metrics"]["attempted_issue_count"] == 2
    assert [issue["truth_state"] for issue in artifact["issues"]] == ["merged_pr", "no_linked_pr"]


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
    assert artifact["coverage"]["attempted_issue_count"] == 1
    assert artifact["coverage"]["missing_issue_count"] == 1
    assert artifact["coverage"]["missing_issue_numbers"] == [873]
    assert artifact["coverage"]["is_complete"] is False
    assert artifact["primary_metrics"]["truth_success_rate"] == 0.5


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
