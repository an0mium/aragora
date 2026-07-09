"""Tests for ``scripts/harness_weakness_miner.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
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
                "emergent_cluster": "review context not tree complete",
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
        "emergent:freshness-proof-missing",
        "emergent:review-context-not-tree-complete",
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
                "body": (
                    "CHANGES-REQUESTED [P1] The reviewer missed the base tree. "
                    'Payload included "api_key": "json-secret-value".'
                ),
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
    assert examples[1].target == "PR #11"
    assert "json-secret-value" not in examples[1].text
    assert "[REDACTED_SECRET]" in examples[1].text
    assert examples[1].url == "https://example.invalid/comment"


def test_since_days_filters_every_source_and_rejects_untrusted_timestamps(
    tmp_path: Path,
) -> None:
    input_path = _write_json(
        tmp_path / "examples.json",
        [
            {"id": "input-recent", "created_at": "2026-07-01T00:00:00Z", "text": "recent"},
            {"id": "input-stale", "created_at": "2026-05-01T00:00:00Z", "text": "stale"},
            {"id": "input-future", "created_at": "2026-07-09T00:00:00Z", "text": "future"},
            {"id": "input-invalid", "created_at": "not-a-date", "text": "invalid"},
            {"id": "input-missing", "text": "missing"},
        ],
    )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-07-02T00:00:00Z",
                        "blocker_class": "recent-ledger",
                    }
                ),
                json.dumps({"timestamp": "invalid", "blocker_class": "invalid-ledger"}),
                json.dumps({"blocker_class": "missing-ledger"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    comments_path = _write_json(
        tmp_path / "comments.json",
        [
            {"id": "comment-recent", "created_at": "2026-07-03T00:00:00Z", "body": "[P2] recent"},
            {"id": "comment-missing", "body": "[P2] missing"},
        ],
    )

    examples = miner.collect_examples(
        input_json=input_path,
        ledger_paths=[ledger_path],
        comment_json_paths=[comments_path],
        since_days=30,
        now=miner.parse_timestamp("2026-07-08T00:00:00Z"),
    )

    assert [example.id for example in examples] == [
        "input-recent",
        "ledger:ledger.jsonl:1",
        "comment-recent",
    ]


def test_classifier_uses_prompt_file_instead_of_argv(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    consult = scripts_dir / "consult_claude.py"
    consult.write_text("# test stub\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert "--prompt-file" in command
        prompt_path = Path(command[command.index("--prompt-file") + 1])
        captured["command"] = command
        captured["prompt"] = json.loads(prompt_path.read_text(encoding="utf-8"))
        response = {
            "example-1": {
                "taxonomy_id": "1",
                "finding_class": "Diff-blind grounding",
                "causal_mechanism": "Prompt lacked tree context",
                "harness_surface": "reviewer grounding prompt",
                "emergent_cluster": "tree context",
            }
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"response": json.dumps(response)}),
            stderr="",
        )

    monkeypatch.setattr(miner, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(miner.subprocess, "run", fake_run)
    classifications = miner._consult_classifier(
        [
            miner.WeaknessExample(
                id="example-1",
                source="gate",
                target="PR #1",
                created_at="2026-07-01T00:00:00Z",
                severity="P2",
                text="A" * 10_000,
            )
        ],
        {"1": "Diff-blind grounding"},
        timeout=10,
    )

    assert classifications["example-1"]["taxonomy_id"] == "1"
    assert len(captured["command"]) == 5
    assert captured["prompt"]["examples"][0]["text"] == "A" * 1200


def test_render_markdown_escapes_untrusted_table_content() -> None:
    example = miner.WeaknessExample(
        id="example-1",
        source="gate|spoof",
        target="PR #1 | forged",
        created_at="2026-07-01T00:00:00Z",
        severity="P2",
        text="finding",
        url="https://example.invalid/a)b",
    )
    classified = miner.ClassifiedExample(
        example=example,
        taxonomy_id="1",
        finding_class="Diff-blind grounding",
        causal_mechanism="Prompt lacked tree context",
        harness_surface="reviewer grounding prompt",
        emergent_cluster="tree-context",
        evidence_summary="Evidence | with [misleading](https://evil.invalid)",
    )
    cluster = miner.WeaknessCluster(
        pass_name="emergent_bottom_up",
        cluster_key="emergent:tree-context",
        title="Diff-blind grounding",
        finding_class="Diff-blind grounding",
        causal_mechanism="Prompt lacked tree context",
        harness_surfaces=["reviewer grounding prompt"],
        rank_score=5,
        examples=[classified],
    )
    report = miner.render_markdown(
        miner.MiningResult(
            ok=True,
            generated_at="2026-07-01T00:00:00Z",
            input_count=1,
            classified_count=1,
            clusters=[cluster],
        )
    )

    assert r"PR #1 \| forged" in report
    assert "a%29b" in report
    assert r"gate\|spoof" in report
    assert r"Evidence \| with \[misleading\]" in report


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
