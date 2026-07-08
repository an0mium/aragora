"""Tests for ``scripts/harness_weakness_miner.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "harness_weakness_miner.py"
    spec = importlib.util.spec_from_file_location(
        "harness_weakness_miner_under_test",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


miner = _load_module()


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _taxonomy(tmp_path: Path) -> Path:
    path = tmp_path / "taxonomy.md"
    path.write_text(
        "\n".join(
            [
                "# Taxonomy",
                "### 1. Diff-blind grounding",
                "A reviewer reasons over only the changed files.",
                "### 2. Stale-external-world grounding",
                "A reviewer relies on stale external state.",
            ],
        ),
        encoding="utf-8",
    )
    return path


def test_two_pass_clustering_uses_seeded_and_emergent_identities(tmp_path: Path) -> None:
    examples_path = _write_json(
        tmp_path / "examples.json",
        [
            {
                "id": "pr1-openai",
                "source": "gate",
                "target": "PR #1",
                "created_at": "2026-07-01T00:00:00Z",
                "severity": "P2",
                "text": "OpenAI said the docs artifact was missing because it was outside the diff.",
            },
            {
                "id": "pr2-claude",
                "source": "gate",
                "target": "PR #2",
                "created_at": "2026-07-02T00:00:00Z",
                "severity": "P1",
                "text": "Claude called a README link dead because the artifact was not changed.",
            },
            {
                "id": "pr3-openai",
                "source": "ledger",
                "target": "PR #3",
                "created_at": "2026-07-03T00:00:00Z",
                "severity": "P2",
                "text": "OpenAI saw PyPI 0.1.0 although 0.1.1 was already published.",
            },
            {
                "id": "pr4-claude",
                "source": "ledger",
                "target": "PR #4",
                "created_at": "2026-07-04T00:00:00Z",
                "severity": "P2",
                "text": "Claude read the release date using a stale local clock.",
            },
        ],
    )
    classifications_path = _write_json(
        tmp_path / "classifications.json",
        {
            "pr1-openai": {
                "taxonomy_id": "1",
                "finding_class": "Diff-blind grounding",
                "causal_mechanism": "Reviewer prompt lacked head-tree context",
                "harness_surface": "reviewer grounding prompt",
                "emergent_cluster": "review_context_not_tree_complete",
                "evidence_summary": "Reviewer reasoned from changed files instead of head tree.",
            },
            "pr2-claude": {
                "taxonomy_id": "1",
                "finding_class": "Diff-blind grounding",
                "causal_mechanism": "Reviewer prompt lacked head-tree context",
                "harness_surface": "reviewer grounding prompt",
                "emergent_cluster": "review_context_not_tree_complete",
                "evidence_summary": "Reviewer treated unchanged artifact as absent.",
            },
            "pr3-openai": {
                "taxonomy_id": "2",
                "finding_class": "Stale-external-world grounding",
                "causal_mechanism": "External fetch freshness is not timestamped",
                "harness_surface": "external evidence fetch template",
                "emergent_cluster": "freshness_proof_missing",
                "evidence_summary": "Reviewer trusted stale PyPI state.",
            },
            "pr4-claude": {
                "taxonomy_id": "2",
                "finding_class": "Stale-external-world grounding",
                "causal_mechanism": "External fetch freshness is not timestamped",
                "harness_surface": "external evidence fetch template",
                "emergent_cluster": "freshness_proof_missing",
                "evidence_summary": "Reviewer lacked timestamped freshness proof.",
            },
        },
    )

    result = miner.run_miner(
        input_json=examples_path,
        taxonomy_path=_taxonomy(tmp_path),
        classification_json=classifications_path,
        min_cluster_size=2,
    )

    assert result.ok is True
    assert [cluster.pass_name for cluster in result.clusters] == [
        "taxonomy_seeded",
        "taxonomy_seeded",
        "emergent_bottom_up",
        "emergent_bottom_up",
    ]
    assert {cluster.cluster_key for cluster in result.clusters} == {
        "taxonomy:1:reviewer-prompt-lacked-head-tree-context",
        "taxonomy:2:external-fetch-freshness-is-not-timestamped",
        "emergent:freshness_proof_missing",
        "emergent:review_context_not_tree_complete",
    }
    first = result.clusters[0]
    assert first.example_count == 2
    assert first.rank_score > 0
    assert first.harness_surfaces == ["reviewer grounding prompt"]
    assert first.examples[0].id in {"pr1-openai", "pr2-claude"}


def test_reads_ledger_and_comment_fixtures_with_redaction(tmp_path: Path) -> None:
    ledger_path = tmp_path / "long_run_ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-07-01T00:00:00Z",
                "target": {"pr": 10},
                "blockers": ["OpenAI P2: API key leaked sk-test-1234567890abcdef"],
                "progress_kind": "parked_pr",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    comments_path = _write_json(
        tmp_path / "comments.json",
        [
            {
                "pr": 11,
                "author": "openai",
                "created_at": "2026-07-02T00:00:00Z",
                "body": "CHANGES-REQUESTED [P1] The reviewer missed the base tree.",
                "url": "https://example.invalid/comment",
            }
        ],
    )

    examples = miner.collect_examples(
        ledger_paths=[ledger_path],
        comment_json_paths=[comments_path],
        since_days=30,
        now=miner.parse_timestamp("2026-07-08T00:00:00Z"),
    )

    assert [example.source for example in examples] == ["ledger", "github_comment"]
    assert examples[0].target == "PR #10"
    assert "sk-test-" not in examples[0].text
    assert "[REDACTED_SECRET]" in examples[0].text
    assert examples[1].url == "https://example.invalid/comment"


def test_cli_writes_markdown_and_json_reports(tmp_path: Path, capsys: Any) -> None:
    examples_path = _write_json(
        tmp_path / "examples.json",
        [
            {
                "id": "a",
                "source": "gate",
                "target": "PR #1",
                "created_at": "2026-07-01T00:00:00Z",
                "severity": "P2",
                "text": "A",
            },
            {
                "id": "b",
                "source": "gate",
                "target": "PR #2",
                "created_at": "2026-07-01T00:00:00Z",
                "severity": "P2",
                "text": "B",
            },
        ],
    )
    classifications_path = _write_json(
        tmp_path / "classifications.json",
        {
            "a": {
                "taxonomy_id": "1",
                "finding_class": "Diff-blind grounding",
                "causal_mechanism": "Prompt lacks tree context",
                "harness_surface": "reviewer grounding prompt",
                "emergent_cluster": "tree_context",
            },
            "b": {
                "taxonomy_id": "1",
                "finding_class": "Diff-blind grounding",
                "causal_mechanism": "Prompt lacks tree context",
                "harness_surface": "reviewer grounding prompt",
                "emergent_cluster": "tree_context",
            },
        },
    )
    output_path = tmp_path / "report.md"

    status = miner.main(
        [
            "--input-json",
            str(examples_path),
            "--taxonomy",
            str(_taxonomy(tmp_path)),
            "--classification-json",
            str(classifications_path),
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert status == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["ok"] is True
    assert stdout["cluster_count"] == 2
    report = output_path.read_text(encoding="utf-8")
    assert "# Harness Weakness Report" in report
    assert "taxonomy_seeded" in report
    assert "emergent_bottom_up" in report
