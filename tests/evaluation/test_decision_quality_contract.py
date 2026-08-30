from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from aragora.evaluation.decision_quality_contract import load_benchmark_bundle

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/benchmarks/decision_quality"
MANIFEST = BENCHMARK_DIR / "benchmark-manifest.json"


def _copy_contract(tmp_path: Path) -> Path:
    destination = tmp_path / "decision_quality"
    shutil.copytree(BENCHMARK_DIR, destination)
    return destination / "benchmark-manifest.json"


def _issue_codes(report: object) -> set[str]:
    return {issue.code for issue in report.issues}  # type: ignore[attr-defined]


def test_live_contract_binds_complete_frozen_corpus() -> None:
    bundle, report = load_benchmark_bundle(MANIFEST)

    assert report.ok
    assert bundle is not None
    assert report.case_count == 24
    assert (
        report.corpus_sha256 == "3a46198fe33e4cc984cf777c6db7f046e4adb7db10840e775f83e9a46e87172b"
    )
    assert (
        report.outcomes_sha256 == "98702fe5d220d171e77fd0276d5803ec851e46c6900445dd73ccb4cbfdbf194d"
    )
    assert set(bundle.roster["conditions"]) == {
        "single_claude",
        "single_openai",
        "single_gemini",
        "aragora_team",
    }


def test_contract_rejects_prompt_digest_drift(tmp_path: Path) -> None:
    manifest_path = _copy_contract(tmp_path)
    prompt_path = manifest_path.parent / "prompt.md"
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8"
    )

    bundle, report = load_benchmark_bundle(manifest_path)

    assert bundle is None
    assert "artifact_hash_mismatch" in _issue_codes(report)


def test_contract_rejects_artifact_path_escape(tmp_path: Path) -> None:
    manifest_path = _copy_contract(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt"]["path"] = "../prompt.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bundle, report = load_benchmark_bundle(manifest_path)

    assert bundle is None
    assert "unsafe_artifact_path" in _issue_codes(report)


def test_contract_rejects_duplicate_roster_family(tmp_path: Path) -> None:
    manifest_path = _copy_contract(tmp_path)
    roster_path = manifest_path.parent / "roster.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    members = roster["conditions"]["aragora_team"]["members"]
    members.append(copy.deepcopy(members[0]))
    roster_path.write_text(json.dumps(roster), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from aragora.evaluation.decision_quality_corpus import canonical_sha256

    manifest["roster"]["sha256"] = canonical_sha256(roster)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bundle, report = load_benchmark_bundle(manifest_path)

    assert bundle is None
    assert "duplicate_family" in _issue_codes(report)
