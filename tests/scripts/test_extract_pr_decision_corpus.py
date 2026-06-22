from __future__ import annotations

import json
from pathlib import Path

import scripts.extract_pr_decision_corpus as mod


def _sample_prs() -> list[dict]:
    return [
        {
            "number": 1,
            "title": "docs: safe update",
            "state": "MERGED",
            "isDraft": False,
            "mergeStateStatus": "CLEAN",
            "mergedAt": "2026-05-22T00:00:00Z",
            "additions": 3,
            "deletions": 1,
            "changedFiles": 1,
            "files": [{"path": "docs/example.md"}],
            "author": {"login": "alice"},
            "labels": [{"name": "docs"}],
            "url": "https://github.example/pr/1",
        },
        {
            "number": 2,
            "title": "fix: do not leak token=sk-ant-secretsecretsecret",
            "state": "OPEN",
            "isDraft": True,
            "mergeStateStatus": "CLEAN",
            "additions": 5,
            "deletions": 2,
            "changedFiles": 1,
            "files": [{"path": "scripts/example.py"}],
            "author": {"login": "bob"},
            "labels": [],
            "url": "https://github.example/pr/2",
        },
    ]


def test_pr_to_example_redacts_and_labels() -> None:
    example = mod.pr_to_example(
        _sample_prs()[1],
        seed="test",
        holdout_ratio=0.2,
        packet={"pr_number": 2, "tier": 2, "requires_human_risk_settlement": False},
    )

    blob = json.dumps(example)
    assert "sk-ant" not in blob
    assert "[REDACTED_SECRET]" in blob
    assert example["label"] == "ask_user"
    assert example["context_features"]["tier"] == 2
    assert example["schema_version"] == mod.SCHEMA_VERSION


def test_extractor_writes_deterministic_jsonl(tmp_path: Path, capsys) -> None:
    source = tmp_path / "prs.json"
    output = tmp_path / "corpus.jsonl"
    source.write_text(json.dumps(_sample_prs()), encoding="utf-8")

    rc = mod.main(
        [
            "--source-json",
            str(source),
            "--output",
            str(output),
            "--seed",
            "fixed",
            "--holdout-ratio",
            "0.5",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["examples"] == 2
    assert [row["pr_number"] for row in rows] == [1, 2]
    assert {row["split"] for row in rows} <= {"train", "holdout"}


def test_assign_splits_uses_preregistered_50_example_holdout() -> None:
    examples = [{"pr_number": number, "split": "train"} for number in range(1, 201)]

    result = mod.assign_deterministic_splits(
        examples,
        seed="fixed",
        holdout_ratio=0.2,
        min_holdout_count=50,
    )

    assert sum(1 for row in result if row["split"] == "holdout") == 50
    assert sum(1 for row in result if row["split"] == "train") == 150


def test_assign_splits_keeps_early_corpus_ratio() -> None:
    examples = [{"pr_number": number, "split": "train"} for number in range(1, 41)]

    result = mod.assign_deterministic_splits(
        examples,
        seed="fixed",
        holdout_ratio=0.2,
        min_holdout_count=50,
    )

    assert sum(1 for row in result if row["split"] == "holdout") == 8
    assert sum(1 for row in result if row["split"] == "train") == 32


def test_extractor_enriches_existing_jsonl_with_changed_files(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "corpus.jsonl"
    output = tmp_path / "enriched.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema_version": mod.SCHEMA_VERSION,
                "task_type": "pr_triage",
                "artifact_id": "pr-42",
                "pr_number": 42,
                "artifact_summary": "PR #42: workflow update | changed_files=2",
                "proposed_action": "merge",
                "context_features": {
                    "pr_number": 42,
                    "changed_files_count": 2,
                    "changed_files": [],
                },
                "label": "challenge",
                "split": "holdout",
                "source": {"github_pr": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod,
        "fetch_pr_files",
        lambda repo, pr_number: [".github/workflows/test.yml", "docs/example.md"],
    )

    rc = mod.main(
        [
            "--source-jsonl",
            str(source),
            "--output",
            str(output),
            "--fetch-missing-files",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert summary["examples"] == 1
    assert row["context_features"]["changed_files"] == [
        ".github/workflows/test.yml",
        "docs/example.md",
    ]
    assert "files=.github/workflows/test.yml, docs/example.md" in row["artifact_summary"]
