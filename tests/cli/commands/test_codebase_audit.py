from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aragora.cli.commands import codebase_audit
from aragora.cli.parser import build_parser


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_codebase_audit_parser_accepts_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "codebase-audit",
            "/tmp/repo",
            "--dry-run",
            "--disable-research",
            "--json",
            "--top-files",
            "5",
            "--artifact-dir",
            "/tmp/artifacts",
        ]
    )

    assert args.command == "codebase-audit"
    assert args.repo == "/tmp/repo"
    assert args.dry_run is True
    assert args.disable_research is True
    assert args.json is True
    assert args.top_files == 5
    assert args.artifact_dir == "/tmp/artifacts"


def test_collect_repo_triage_ignores_noise_and_runtime_artifacts(tmp_path: Path) -> None:
    _write(
        tmp_path / "aragora" / "swarm" / "boss_loop.py",
        "def dispatch():\n    return 'ok'\n",
    )
    _write(tmp_path / "node_modules" / "pkg" / "index.js", "console.log('ignore me')\n")
    _write(tmp_path / ".aragora" / "codebase-audit" / "old" / "run.json", '{"ignore": true}\n')
    _write(tmp_path / "uv.lock", "version = 1\n")

    triage = codebase_audit.collect_repo_triage(tmp_path)

    assert triage["scanned_files"] == 1
    assert triage["bespoke_loc"] == 2
    assert triage["largest_files"] == [
        {
            "path": "aragora/swarm/boss_loop.py",
            "loc": 2,
            "boundary_scores": {},
            "dominant_boundary": "general",
        }
    ]


def test_identify_threat_surface_marks_expected_boundaries(tmp_path: Path) -> None:
    _write(
        tmp_path / "aragora" / "swarm" / "boss_loop.py",
        "create_agent('codex')\nworker = 'dispatch worker'\nsubprocess.run(['echo', 'ok'])\n",
    )
    _write(
        tmp_path / "aragora" / "server" / "handlers" / "slack.py",
        "class BaseHandler:\n    def handle(self, request):\n        return 'oauth callback'\n",
    )
    _write(
        tmp_path / "contracts" / "erc8004" / "Token.sol",
        "contract Token {\n function transfer(address wallet) public {}\n}\n",
    )
    _write(
        tmp_path / "tests" / "server" / "handlers" / "test_slack.py",
        "class BaseHandler:\n    def handle(self, request, headers, oauth):\n        return request\n",
    )
    _write(
        tmp_path / "sdk" / "typescript" / "src" / "namespaces" / "openapi.ts",
        "export const request = { Authorization: 'x', oauth: true, callback: true };\n",
    )

    surface = codebase_audit.identify_threat_surface(tmp_path, top_files=6)
    by_path = {item["path"]: item for item in surface["top_risk_files"]}

    assert by_path["aragora/swarm/boss_loop.py"]["dominant_boundary"] == "llm_system"
    assert by_path["aragora/server/handlers/slack.py"]["dominant_boundary"] == "user_perimeter"
    assert by_path["contracts/erc8004/Token.sol"]["dominant_boundary"] == "crypto_fiat"
    assert "tests/server/handlers/test_slack.py" not in by_path
    assert "sdk/typescript/src/namespaces/openapi.ts" not in by_path


def test_cmd_codebase_audit_dry_run_writes_staged_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    _write(
        tmp_path / "aragora" / "swarm" / "boss_loop.py",
        "create_agent('codex')\nworker = 'dispatch worker'\n",
    )
    _write(
        tmp_path / "contracts" / "erc8004" / "Token.sol",
        "contract Token {\n function transfer(address wallet) public {}\n}\n",
    )
    artifact_dir = tmp_path / "artifacts"
    args = argparse.Namespace(
        repo=str(tmp_path),
        agents="claude,codex,openai",
        top_files=4,
        max_dirs=25,
        max_preview_chars=500,
        max_file_chars=1000,
        artifact_dir=str(artifact_dir),
        dry_run=True,
        json=True,
    )

    exit_code = codebase_audit.cmd_codebase_audit(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_dir"] == str(artifact_dir)
    assert (artifact_dir / "triage.json").exists()
    assert (artifact_dir / "surface.json").exists()
    assert (artifact_dir / "interrogate.json").exists()
    assert (artifact_dir / "blast_radius.json").exists()
    assert (artifact_dir / "run.json").exists()
    assert (artifact_dir / "summary.md").exists()

    interrogate = json.loads((artifact_dir / "interrogate.json").read_text())
    assert interrogate["status"] == "skipped_dry_run"
    assert "Highest-risk files:" in interrogate["task"]


def test_cmd_codebase_audit_persists_partial_artifacts_when_interrogate_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(
        tmp_path / "aragora" / "swarm" / "boss_loop.py",
        "create_agent('codex')\nworker = 'dispatch worker'\n",
    )
    artifact_dir = tmp_path / "artifacts"

    async def _boom(**_: object) -> dict[str, object]:
        raise RuntimeError("interrogate exploded")

    monkeypatch.setattr(codebase_audit, "_run_interrogate_stage", _boom)

    args = argparse.Namespace(
        repo=str(tmp_path),
        agents="claude,codex,openai",
        top_files=4,
        max_dirs=25,
        max_preview_chars=500,
        max_file_chars=1000,
        artifact_dir=str(artifact_dir),
        dry_run=False,
        json=False,
    )

    exit_code = codebase_audit.cmd_codebase_audit(args)

    assert exit_code == 1
    for name in [
        "triage.json",
        "surface.json",
        "blast_radius.json",
        "interrogate.json",
        "run.json",
        "summary.md",
    ]:
        assert (artifact_dir / name).exists()

    interrogate = json.loads((artifact_dir / "interrogate.json").read_text())
    assert interrogate["status"] == "failed"
    assert interrogate["errors"] == ["interrogate exploded"]


@pytest.mark.asyncio
async def test_run_interrogate_stage_can_disable_research(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(
        tmp_path / "aragora" / "swarm" / "boss_loop.py",
        "create_agent('codex')\nworker = 'dispatch worker'\n",
    )
    triage = codebase_audit.collect_repo_triage(tmp_path)
    surface = codebase_audit.identify_threat_surface(tmp_path, top_files=4)

    monkeypatch.setattr(codebase_audit, "_build_agents", lambda _: ([object()], []))
    observed: dict[str, object] = {}

    async def _fake_run_deep_audit(*, task, agents, context, config):
        observed["task"] = task
        observed["agents"] = agents
        observed["context"] = context
        observed["enable_research"] = config.enable_research
        return SimpleNamespace(
            recommendation="focus supervisor and webhook paths",
            confidence=0.72,
            findings=[],
            unanimous_issues=[],
            split_opinions=[],
            risk_areas=[],
            citations=[],
            cross_examination_notes="",
        )

    monkeypatch.setattr("aragora.modes.deep_audit.run_deep_audit", _fake_run_deep_audit)

    result = await codebase_audit._run_interrogate_stage(
        repo_root=tmp_path,
        triage=triage,
        surface=surface,
        agents_str="openai-api,gemini",
        top_files=4,
        max_file_chars=1000,
        dry_run=False,
        disable_research=True,
    )

    assert result["status"] == "completed"
    assert observed["enable_research"] is False
