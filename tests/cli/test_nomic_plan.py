"""CLI integration tests for local generic Nomic planning."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.cli.commands.nomic import add_nomic_parser
from aragora.core_types import DebateResult
from aragora.nomic.meta_planner import MetaPlanner


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def cli_repository(tmp_path: Path) -> Path:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "cli@example.test")
    git(tmp_path, "config", "user.name", "CLI Test")
    git(tmp_path, "remote", "add", "origin", "https://github.com/example/cli-plan.git")
    (tmp_path / ".gitignore").write_text(".nomic/\n", encoding="utf-8")
    (tmp_path / ".aragora.yaml").write_text(
        """nomic:
  repository:
    name: CLI Plan
    id: example/cli-plan
  roadmap_paths:
    - ROADMAP.md
  context_entry_files:
    - README.md
  evaluation_criteria:
    - id: value
      description: Delivers measurable repository value
""",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# CLI Plan\n", encoding="utf-8")
    (tmp_path / "ROADMAP.md").write_text("# Roadmap\nShip the planner.\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def ready():\n    return False\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_nomic_parser(subparsers)
    return parser.parse_args(["nomic", *argv])


async def fake_multimodel_debate(self, prompt: str, context_pack) -> DebateResult:
    assert "CLI Plan" in prompt
    return DebateResult(
        task=prompt,
        final_answer=json.dumps(
            {
                "goals": [
                    {
                        "description": "Implement the roadmap planning milestone",
                        "rationale": "The tracked roadmap names it as the next deliverable.",
                        "estimated_impact": "high",
                        "criterion_scores": {"value": 0.95},
                        "evidence_paths": ["ROADMAP.md"],
                    }
                ]
            }
        ),
        confidence=0.9,
        consensus_reached=True,
        rounds_used=2,
        participants=["fake-alpha", "fake-beta"],
        proposals={
            "fake-alpha": "Implement the tracked roadmap milestone with tests.",
            "fake-beta": "Prioritize the same milestone because it has direct evidence.",
        },
        metadata={"nomic_planning_models": ["frontier-alpha", "frontier-beta"]},
    )


def test_plan_json_emits_pack_and_receipt(cli_repository: Path, capsys) -> None:
    args = parse(
        "plan",
        "Choose the next roadmap improvement",
        "--repo",
        str(cli_repository),
        "--config",
        str(cli_repository / ".aragora.yaml"),
        "--json",
    )
    with (
        patch.object(MetaPlanner, "_run_repository_planning_debate", fake_multimodel_debate),
        patch.object(MetaPlanner, "_ingest_receipt_to_km", lambda self, receipt: None),
    ):
        args.func(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned"
    assert payload["commit_sha"] == git(cli_repository, "rev-parse", "HEAD")
    assert payload["evidence_coverage"] == 1.0
    assert payload["verdict"] == "PASS"
    assert payload["receipt"]["schema_version"] == "1.3"
    assert Path(payload["receipt_json_path"]).is_file()
    assert Path(payload["receipt_markdown_path"]).is_file()
    pack_path = (
        cli_repository / ".nomic" / "context" / "packs" / payload["commit_sha"] / payload["pack_id"]
    )
    assert (pack_path / "context-pack.json").is_file()
    assert (pack_path / "manifest.tsv").is_file()


def test_plan_human_output_lists_artifacts_and_goal(cli_repository: Path, capsys) -> None:
    args = parse("plan", "Choose work", "--repo", str(cli_repository))
    with (
        patch.object(MetaPlanner, "_run_repository_planning_debate", fake_multimodel_debate),
        patch.object(MetaPlanner, "_ingest_receipt_to_km", lambda self, receipt: None),
    ):
        args.func(args)

    output = capsys.readouterr().out
    assert "Repository plan: CLI Plan" in output
    assert "Context pack:" in output
    assert "Receipt JSON:" in output
    assert "Implement the roadmap planning milestone" in output


def test_dirty_plan_fails_before_artifacts(cli_repository: Path, capsys) -> None:
    (cli_repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    args = parse("plan", "Choose work", "--repo", str(cli_repository), "--json")

    with pytest.raises(SystemExit) as exc_info:
        args.func(args)

    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "clean" in payload["error"]
    assert not (cli_repository / ".nomic").exists()


def test_no_multimodel_debate_returns_distinct_status(cli_repository: Path, capsys) -> None:
    async def single_model(self, prompt: str, context_pack) -> DebateResult:
        result = await fake_multimodel_debate(self, prompt, context_pack)
        result.metadata["nomic_planning_models"] = ["frontier-alpha"]
        result.proposals = {"fake-alpha": result.proposals["fake-alpha"]}
        return result

    args = parse("plan", "Choose work", "--repo", str(cli_repository), "--json")
    with (
        patch.object(MetaPlanner, "_run_repository_planning_debate", single_model),
        patch.object(MetaPlanner, "_ingest_receipt_to_km", lambda self, receipt: None),
        pytest.raises(SystemExit) as exc_info,
    ):
        args.func(args)

    assert exc_info.value.code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no_evidence"
    assert payload["goals"] == []
    assert payload["verdict"] == "NO_EVIDENCE"


def test_plan_rejects_api_mode(cli_repository: Path) -> None:
    args = parse("--api", "plan", "Choose work", "--repo", str(cli_repository))

    with pytest.raises(SystemExit) as exc_info:
        args.func(args)

    assert exc_info.value.code == 1
    assert not (cli_repository / ".nomic").exists()
