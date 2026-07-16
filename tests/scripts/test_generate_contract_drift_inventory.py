"""Tests for scripts/generate_contract_drift_inventory.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.generate_contract_drift_inventory as gen

VERIFY = {
    "python_sdk_drift": ["GET /a", "GET /b"],
    "typescript_sdk_drift": ["ts1"],
    "missing_stable": [],
}
ROUTES = {"missing_in_spec": ["m1"], "orphaned_in_spec": ["o1"]}
PARITY = {"missing_from_both_sdks": []}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_baselines(repo: Path, verify: dict, routes: dict, parity: dict) -> None:
    _write_json(repo / "scripts/baselines/verify_sdk_contracts.json", verify)
    _write_json(repo / "scripts/baselines/validate_openapi_routes.json", routes)
    _write_json(repo / "scripts/baselines/check_sdk_parity.json", parity)


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_baselines(repo, VERIFY, ROUTES, PARITY)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "cohort"],
        cwd=repo,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, sha


def _run(monkeypatch, repo: Path, sha: str, *extra: str) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_contract_drift_inventory.py",
            "--repo-root",
            str(repo),
            "--cohort-commit",
            sha,
            *extra,
        ],
    )
    return gen.main()


def _inventory(repo: Path) -> dict:
    return json.loads((repo / gen.DEFAULT_INVENTORY).read_text())


def test_cohort_items_classified_start_cohort(monkeypatch, tmp_path: Path):
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha) == 0

    items = _inventory(repo)["items"]
    assert len(items) == 5
    for item in items:
        assert item["class"] == "start_cohort"
        assert item["discovered_on"] == gen.COHORT_DATE
        assert item["provenance"] == gen.COHORT_PROVENANCE
        assert item["status"] == "open"
    assert items == sorted(items, key=lambda i: i["id"])


def test_output_deterministic(monkeypatch, tmp_path: Path):
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha) == 0
    first = (repo / gen.DEFAULT_INVENTORY).read_bytes()
    assert _run(monkeypatch, repo, sha) == 0
    assert (repo / gen.DEFAULT_INVENTORY).read_bytes() == first


def test_resolved_items_retained_never_deleted(monkeypatch, tmp_path: Path):
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha) == 0

    verify = dict(VERIFY, python_sdk_drift=["GET /a"])  # GET /b fixed
    _write_baselines(repo, verify, ROUTES, PARITY)
    assert _run(monkeypatch, repo, sha, "--as-of", "2026-07-01") == 0

    items = {i["id"]: i for i in _inventory(repo)["items"]}
    assert len(items) == 5  # nothing deleted
    resolved = items["python_sdk_drift:GET /b"]
    assert resolved["status"] == "resolved"
    assert resolved["resolved_on"] == "2026-07-01"
    assert resolved["class"] == "start_cohort"
    assert sum(1 for i in items.values() if i["status"] == "open") == 4


def test_reopened_item_keeps_original_record(monkeypatch, tmp_path: Path):
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha) == 0
    _write_baselines(repo, dict(VERIFY, python_sdk_drift=["GET /a"]), ROUTES, PARITY)
    assert _run(monkeypatch, repo, sha, "--as-of", "2026-07-01") == 0

    _write_baselines(repo, VERIFY, ROUTES, PARITY)  # GET /b regresses back
    assert _run(monkeypatch, repo, sha, "--as-of", "2026-07-02") == 0
    item = {i["id"]: i for i in _inventory(repo)["items"]}["python_sdk_drift:GET /b"]
    assert item["status"] == "open"
    assert "resolved_on" not in item
    assert item["class"] == "start_cohort"
    assert item["discovered_on"] == gen.COHORT_DATE


def test_new_item_without_provenance_fails_closed(monkeypatch, tmp_path: Path, capsys):
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha) == 0
    before = (repo / gen.DEFAULT_INVENTORY).read_bytes()

    verify = dict(VERIFY, python_sdk_drift=[*VERIFY["python_sdk_drift"], "GET /new"])
    _write_baselines(repo, verify, ROUTES, PARITY)
    assert _run(monkeypatch, repo, sha) == 1
    out = capsys.readouterr().out
    assert "python_sdk_drift:GET /new" in out
    assert (repo / gen.DEFAULT_INVENTORY).read_bytes() == before  # not absorbed


def test_new_item_with_provenance_classified_discovered(monkeypatch, tmp_path: Path):
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha) == 0
    verify = dict(VERIFY, python_sdk_drift=[*VERIFY["python_sdk_drift"], "GET /new"])
    _write_baselines(repo, verify, ROUTES, PARITY)

    assert (
        _run(
            monkeypatch,
            repo,
            sha,
            "--provenance",
            "introduced by handler rework, tracked in #9999",
            "--as-of",
            "2026-07-02",
        )
        == 0
    )
    item = {i["id"]: i for i in _inventory(repo)["items"]}["python_sdk_drift:GET /new"]
    assert item["class"] == "discovered"
    assert item["discovered_on"] == "2026-07-02"
    assert "#9999" in item["provenance"]


def test_provenance_requires_pr_or_issue_reference(monkeypatch, tmp_path: Path):
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha, "--provenance", "because reasons") == 1


def test_check_mode(monkeypatch, tmp_path: Path):
    repo, sha = _init_repo(tmp_path)
    # Missing inventory fails.
    assert _run(monkeypatch, repo, sha, "--check") == 1

    assert _run(monkeypatch, repo, sha) == 0
    assert _run(monkeypatch, repo, sha, "--check") == 0

    # Baseline edited without regenerating -> out of sync.
    verify = dict(VERIFY, python_sdk_drift=["GET /a"])
    _write_baselines(repo, verify, ROUTES, PARITY)
    assert _run(monkeypatch, repo, sha, "--check") == 1


def test_check_fails_on_tampered_cohort_classification(monkeypatch, tmp_path: Path):
    """--check must reject a committed inventory whose cohort items were
    reclassified (derivable-metadata invariant)."""
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha) == 0

    inventory = _inventory(repo)
    inventory["items"][0]["class"] = "discovered"
    inventory["items"][0]["discovered_on"] = "2026-07-01"
    (repo / gen.DEFAULT_INVENTORY).write_text(json.dumps(inventory))

    assert _run(monkeypatch, repo, sha, "--check") == 1
