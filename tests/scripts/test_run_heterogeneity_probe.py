from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_heterogeneity_probe.py"


def _run_with_classifications(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    limit: int | None = None,
) -> subprocess.CompletedProcess[str]:
    classifications_file = tmp_path / "classifications.json"
    classifications_file.write_text(json.dumps(payload), encoding="utf-8")
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--classifications-file",
        str(classifications_file),
        "--output-root",
        str(tmp_path / "out"),
        "--run-id",
        "validation-repro",
        "--json",
    ]
    if limit is not None:
        cmd[2:2] = ["--limit", str(limit)]
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_probe_script_dry_run_selects_pilot_subset() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["prompt_count"] == 23
    assert payload["class_counts"]["null_negative"] == 2


def test_probe_script_writes_synthetic_fixture_receipt(tmp_path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--synthetic-fixture",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "fixture",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    receipt_path = Path(payload["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "heterogeneity_probe_receipt.v1"
    assert receipt["judge_model"] == "synthetic-fixture"
    assert "synthetic fixture only" in receipt["scope_caveats"][0]


def test_probe_script_writes_receipt_from_classifications_file(tmp_path) -> None:
    fixture = ROOT / "tests" / "heterogeneity" / "fixtures" / "classifications_minimal.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--limit",
            "1",
            "--classifications-file",
            str(fixture),
            "--output-root",
            str(tmp_path),
            "--run-id",
            "external",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    receipt = json.loads(Path(payload["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["judge_model"] == "external-judge-fixture"
    assert receipt["verdict"] == "insufficient_pilot"
    assert "external judged classifications" in receipt["scope_caveats"][0]


def test_probe_script_accepts_partial_multi_seeded_classification(tmp_path) -> None:
    prompt_root = tmp_path / "prompts"
    prompt_dir = prompt_root / "multi_seeded_error"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "01_fixture.md").write_text(
        """---
prompt_id: mse_fixture
class: multi_seeded_error
seeded_errors:
  - description: "first seeded error"
  - description: "second seeded error"
expected_flags: 1
---

Fixture prompt with two seeded errors.
""",
        encoding="utf-8",
    )
    classifications_file = tmp_path / "classifications.json"
    classifications_file.write_text(
        json.dumps(
            {
                "judge_model": "fixture",
                "panel_models": ["solo"],
                "results": [
                    {
                        "prompt_id": "mse_fixture",
                        "classifications": [
                            {
                                "agent": "solo",
                                "verdict": "partial_multi_seeded",
                                "rationale": "caught one seeded error",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prompt-root",
            str(prompt_root),
            "--all-prompts",
            "--classifications-file",
            str(classifications_file),
            "--output-root",
            str(tmp_path / "out"),
            "--run-id",
            "partial",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    receipt = json.loads(Path(payload["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["metrics"]["independent_flag_successes"] == 0
    assert receipt["metrics"]["partial_multi_seeded_successes"] == 1
    assert receipt["metrics"]["partial_multi_seeded_trials"] == 1


def test_probe_script_rejects_duplicate_classification_agents(tmp_path) -> None:
    proc = _run_with_classifications(
        tmp_path,
        {
            "judge_model": "duplicate-repro",
            "panel_models": ["solo"],
            "results": [
                {
                    "prompt_id": "sse_01_revert_window_off_by_one",
                    "classifications": [
                        {
                            "agent": "solo",
                            "verdict": "flagged_correctly",
                            "rationale": "first",
                        },
                        {
                            "agent": "solo",
                            "verdict": "flagged_correctly",
                            "rationale": "duplicate",
                        },
                    ],
                }
            ],
        },
    )

    assert proc.returncode == 1
    assert "duplicate classification agent: solo" in proc.stderr


def test_probe_script_rejects_unknown_classification_verdict(tmp_path) -> None:
    proc = _run_with_classifications(
        tmp_path,
        {
            "judge_model": "unknown-verdict-repro",
            "panel_models": ["solo"],
            "results": [
                {
                    "prompt_id": "cn_01_invalidation_signals",
                    "classifications": [
                        {"agent": "solo", "verdict": "probably", "rationale": "bad verdict"}
                    ],
                }
            ],
        },
        limit=1,
    )

    assert proc.returncode == 1
    assert "unknown classification verdict: probably" in proc.stderr


def test_probe_script_rejects_unknown_classification_agent(tmp_path) -> None:
    proc = _run_with_classifications(
        tmp_path,
        {
            "judge_model": "unknown-agent-repro",
            "panel_models": ["solo"],
            "results": [
                {
                    "prompt_id": "cn_01_invalidation_signals",
                    "classifications": [
                        {"agent": "intruder", "verdict": "missed", "rationale": "not in panel"}
                    ],
                }
            ],
        },
        limit=1,
    )

    assert proc.returncode == 1
    assert "classification agent not in panel_models: intruder" in proc.stderr


def test_probe_script_rejects_missing_classification_agents(tmp_path) -> None:
    proc = _run_with_classifications(
        tmp_path,
        {
            "judge_model": "missing-agent-repro",
            "panel_models": ["solo", "missing"],
            "results": [
                {
                    "prompt_id": "cn_01_invalidation_signals",
                    "classifications": [
                        {"agent": "solo", "verdict": "missed", "rationale": "one panelist only"}
                    ],
                }
            ],
        },
        limit=1,
    )

    assert proc.returncode == 1
    assert "missing classifications for panel_models: missing" in proc.stderr


def test_probe_script_rejects_duplicate_prompt_ids(tmp_path) -> None:
    proc = _run_with_classifications(
        tmp_path,
        {
            "judge_model": "duplicate-prompt-repro",
            "panel_models": ["solo"],
            "results": [
                {
                    "prompt_id": "cn_01_invalidation_signals",
                    "classifications": [
                        {"agent": "solo", "verdict": "missed", "rationale": "first"}
                    ],
                },
                {
                    "prompt_id": "cn_01_invalidation_signals",
                    "classifications": [
                        {"agent": "solo", "verdict": "missed", "rationale": "duplicate"}
                    ],
                },
            ],
        },
        limit=1,
    )

    assert proc.returncode == 1
    assert "duplicate prompt_id in classifications file: cn_01_invalidation_signals" in proc.stderr


def test_probe_script_rejects_missing_prompt_results(tmp_path) -> None:
    proc = _run_with_classifications(
        tmp_path,
        {
            "judge_model": "missing-prompt-repro",
            "panel_models": ["solo"],
            "results": [
                {
                    "prompt_id": "cn_01_invalidation_signals",
                    "classifications": [
                        {"agent": "solo", "verdict": "missed", "rationale": "first only"}
                    ],
                }
            ],
        },
        limit=2,
    )

    assert proc.returncode == 1
    assert (
        "classifications file missing results for prompt_ids: cn_02_baseline_floor" in proc.stderr
    )
