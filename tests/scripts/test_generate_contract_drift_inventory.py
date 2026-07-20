"""Tests for scripts/generate_contract_drift_inventory.py."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.generate_contract_drift_inventory as gen
from tests.scripts import cdg_synthetic_artifacts as cdg

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


def test_generator_never_mints_resolved(monkeypatch, tmp_path: Path):
    """An entry that vanished before the FIRST generation is simply absent —
    the generator never fabricates resolved history for items it never saw."""
    repo, sha = _init_repo(tmp_path)
    _write_baselines(repo, dict(VERIFY, python_sdk_drift=["GET /a"]), ROUTES, PARITY)

    assert _run(monkeypatch, repo, sha) == 0
    items = _inventory(repo)["items"]
    assert all(item["status"] == "open" for item in items)
    assert "python_sdk_drift:GET /b" not in {item["id"] for item in items}


def test_duplicate_inventory_ids_fail_sync(tmp_path):
    """Round-4 review P2: duplicated rows (especially resolved duplicates of a

    discovered item) would inflate batch_size past dict-collapsed append-only
    checks; every inventory id must be unique.
    """
    item = {
        "id": "python_sdk_drift:GET /api/x",
        "class": "discovered",
        "discovered_on": "2026-06-01",
        "provenance": "found via #9999",
        "status": "open",
    }
    dup_resolved = dict(item, status="resolved", resolved_on="2026-07-01")
    inventory = {"items": [item, dup_resolved]}
    issues = gen.find_sync_issues(inventory, {"python_sdk_drift:GET /api/x": "python_sdk_drift"})
    assert any("Duplicate inventory id" in i for i in issues)


def test_duplicate_baseline_entries_fail_closed(monkeypatch, tmp_path: Path, capsys):
    """A duplicate baseline entry inflates count-based ratchet totals while the
    id-deduped inventory sees one item — a latent count-decrease freebie. The
    generator refuses to run (check or write) over duplicated baselines."""
    repo, sha = _init_repo(tmp_path)
    assert _run(monkeypatch, repo, sha) == 0
    assert _run(monkeypatch, repo, sha, "--check") == 0

    verify = dict(VERIFY, python_sdk_drift=[*VERIFY["python_sdk_drift"], "GET /a"])
    _write_baselines(repo, verify, ROUTES, PARITY)
    assert _run(monkeypatch, repo, sha, "--check") == 1
    assert "Duplicate baseline entry: python_sdk_drift:GET /a" in capsys.readouterr().out

    before = (repo / gen.DEFAULT_INVENTORY).read_bytes()
    assert _run(monkeypatch, repo, sha) == 1  # write mode refuses too
    assert (repo / gen.DEFAULT_INVENTORY).read_bytes() == before


def test_find_duplicate_entry_issues_unit():
    docs = {
        "verify": {
            "python_sdk_drift": ["GET /a", "GET /a", "GET /b"],
            "typescript_sdk_drift": ["ts1"],
        },
        "routes": {"missing_in_spec": ["m1"], "orphaned_in_spec": []},
        "parity": {"missing_from_both_sdks": []},
    }
    issues = gen.find_duplicate_entry_issues(docs)
    assert issues == ["Duplicate baseline entry: python_sdk_drift:GET /a"]
    docs["verify"]["python_sdk_drift"] = ["GET /a", "GET /b"]
    assert gen.find_duplicate_entry_issues(docs) == []


def test_discovered_provenance_requires_reference_in_committed_inventory():
    """Round-5 review P2: hand-edited committed inventories must meet the

    generator's bar — discovered items need a PR/issue reference in
    provenance, not free text.
    """
    item = {
        "id": "python_sdk_drift:GET /api/y",
        "class": "discovered",
        "discovered_on": "2026-06-01",
        "provenance": "we decided this is fine",
        "status": "open",
    }
    issues = gen.find_sync_issues(
        {"items": [item]}, {"python_sdk_drift:GET /api/y": "python_sdk_drift"}
    )
    assert any("lacks a PR/issue reference" in i for i in issues)


# ---------------------------------------------------------------------------
# Standalone read-only authority/inventory surface tests (VAL-CDG-001/002)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(gen.__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_contract_drift_inventory.py"

# Located BEFORE any fixture patches env/constants: the real multi-MB mission
# artifacts, or None. Import/collection runs before fixtures, and the check
# is size-pinned, so a synthetic library can never masquerade as real.
REAL_ARTIFACTS = cdg.find_real_artifacts()


@pytest.fixture(scope="module", autouse=True)
def _cdg_artifact_env(tmp_path_factory):
    """Guarantee canonical mission artifacts for every test in this module.

    Real multi-MB artifacts are used when available; otherwise small
    synthetic artifacts are generated and the pinned digest constants are
    patched in-process so the standalone authority/inventory surface runs
    hermetically in CI instead of skipping. Tests that spawn the CLI in a
    subprocess cannot see the in-process patches and must use
    `_require_real_artifacts` (skip-if-absent) instead.
    """
    with cdg.artifact_environment(tmp_path_factory) as env:
        yield env


def _canonical_bytes(doc: dict) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _artifact_paths() -> tuple[Path, Path] | None:
    roots: list[Path] = []
    mission_dir = os.environ.get("FACTORY_MISSION_DIR", "").strip()
    if mission_dir:
        roots.append(Path(mission_dir) / "library")
    settings_path = os.environ.get("FACTORY_RUNTIME_SETTINGS_PATH", "").strip()
    if settings_path:
        roots.append(Path(settings_path).parent / "library")
    roots.append(REPO_ROOT / "library")
    for root in roots:
        cohort = root / gen.COHORT_ARTIFACT_FILENAME
        provenance = root / gen.PROVENANCE_ARTIFACT_FILENAME
        if cohort.is_file() and provenance.is_file():
            return cohort, provenance
    return None


def _require_artifacts() -> tuple[Path, Path]:
    """Canonical artifacts for IN-PROCESS tests (never skips: the module
    fixture provides synthetic artifacts when the real ones are absent)."""
    found = _artifact_paths()
    assert found is not None, "the _cdg_artifact_env fixture must provide the canonical artifacts"
    return found


def _require_real_artifacts() -> tuple[Path, Path]:
    """The REAL multi-MB mission artifacts, for tests that spawn the CLI in
    a subprocess where the synthetic fixture's in-process constant patches
    cannot apply."""
    if REAL_ARTIFACTS is None:
        pytest.skip(
            "real canonical mission artifacts unavailable "
            "(set FACTORY_MISSION_DIR or provide <repo>/library); "
            "hermetic coverage runs in-process via the synthetic-artifact fixture"
        )
    return REAL_ARTIFACTS


def _artifact_args(artifacts: tuple[Path, Path]) -> list[str]:
    return [
        "--cohort-artifact",
        str(artifacts[0]),
        "--sdk-provenance-artifact",
        str(artifacts[1]),
    ]


def _head_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_cli(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT_PATH), *argv],
        capture_output=True,
        cwd=str(cwd or REPO_ROOT),
    )


def _run_standalone(monkeypatch, capsys, *argv: str) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", ["generate_contract_drift_inventory.py", *argv])
    rc = gen.main()
    return rc, json.loads(capsys.readouterr().out)


def _independent_closure(sha: str) -> set[str]:
    rq = gen._review_queue_module()
    roots = list(rq.CONTRACT_DRIFT_AUTHORITY_PREFIXES)
    names = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-tree", "--name-only", sha, ".github/workflows/"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    closure = set(roots) | {"aragora/cli/commands/review_queue.py"}
    for name in names:
        # GitHub runs both .yml and .yaml workflows (top level only).
        if not name.endswith((".yml", ".yaml")):
            continue
        blob = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{sha}:{name}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if any(root in blob for root in roots):
            closure.add(name)
    return closure


def test_transitive_authority_dependencies_are_tier4():
    artifacts = _require_real_artifacts()  # subprocess CLI: needs the real artifacts
    sha = _head_sha()
    proc = _run_cli(
        "--authority-manifest",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
    )
    assert proc.returncode == 0, proc.stdout
    manifest = json.loads(proc.stdout)
    rq = gen._review_queue_module()

    repo_files = manifest["repo_files"]
    assert repo_files == sorted(set(repo_files))
    assert set(repo_files) == _independent_closure(sha)
    assert set(rq.CONTRACT_DRIFT_AUTHORITY_PREFIXES) <= set(repo_files)
    assert "aragora/cli/commands/review_queue.py" in repo_files
    assert "scripts/tier4_merge_train.py" not in repo_files

    for path in repo_files:
        tier, _name, _reason = rq._classify_model_review_tier([path])
        assert tier == 4, path
    bindings = {binding["path"]: binding for binding in manifest["file_bindings"]}
    assert sorted(bindings) == repo_files
    for path, binding in bindings.items():
        assert binding["effective_tier"] == 4
        assert binding["matched_authority_rule"] is not None
        assert len(binding["sha256"]) == 64
        assert binding["byte_length"] > 0

    mirror = manifest["merge_train_mirror"]
    assert mirror["path"] == "scripts/tier4_merge_train.py"
    assert mirror["in_repo_files"] is False
    assert len(mirror["sha256"]) == 64

    body = {k: v for k, v in manifest.items() if k != "authority_manifest_sha256"}
    assert (
        hashlib.sha256(_canonical_bytes(body)).hexdigest() == manifest["authority_manifest_sha256"]
    )


def test_classifier_and_merge_train_closure_match():
    rq = gen._review_queue_module()
    import scripts.tier4_merge_train as train

    exact_roots = [*rq.CONTRACT_DRIFT_AUTHORITY_PREFIXES, "aragora/cli/commands/review_queue.py"]
    matrix: list[str] = []
    for root in exact_roots:
        matrix += [root, f"{root}.bak", f"{root}.old", f"{root}.pyx", f"{root}/child"]
    matrix += [
        ".github/workflows/contract-drift-governance.yml",
        ".github/workflows/nested/deep.yml",
        "docs/CONTRACT_DRIFT.md",
        "scripts/unrelated_sibling_tool.py",
    ]
    for path in matrix:
        canonical_rule = next(
            (p for p in rq.TIER_4_PREFIXES if rq._matches_prefix(path, (p,))), None
        )
        train_rule = train.matches_serialized_path(path)
        assert canonical_rule == train_rule, path
        tier = rq._classify_model_review_tier([path])[0]
        assert (tier == 4) == (canonical_rule is not None), path
        serialized = train.serialized_paths_for([path])
        assert serialized == ({train_rule} if train_rule else set()), path
    for root in exact_roots:
        for suffix in (".bak", ".old", ".pyx", "/child"):
            hostile = f"{root}{suffix}"
            assert rq._classify_model_review_tier([hostile])[0] < 4, hostile
            assert train.matches_serialized_path(hostile) is None, hostile


def test_standalone_classifier_imports_and_calls_canonical_review_queue_policy(monkeypatch, capsys):
    artifacts = _require_artifacts()
    sha = _head_sha()
    rq = gen._review_queue_module()

    sentinel_path = "scripts/nonexistent_probe_file.py"
    real_classify = rq._classify_model_review_tier

    def spy(files, *, pr=None):
        if list(files) == [sentinel_path]:
            return (3, "tier_3_sentinel_from_canonical_policy", "sentinel classification")
        return real_classify(files, pr=pr)

    monkeypatch.setattr(rq, "_classify_model_review_tier", spy)
    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--classify-tier",
        "--changed-file",
        sentinel_path,
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
    )
    assert rc == 0
    assert doc["effective_tier"] == 3
    assert doc["tier_name"] == "tier_3_sentinel_from_canonical_policy"
    assert doc["changed_files"] == [sentinel_path]

    dropped_root = "scripts/sdk_path_normalize.py"
    trimmed = tuple(
        prefix for prefix in rq.CONTRACT_DRIFT_AUTHORITY_PREFIXES if prefix != dropped_root
    )
    assert len(trimmed) == len(rq.CONTRACT_DRIFT_AUTHORITY_PREFIXES) - 1
    monkeypatch.setattr(rq, "CONTRACT_DRIFT_AUTHORITY_PREFIXES", trimmed)
    rc, manifest = _run_standalone(
        monkeypatch,
        capsys,
        "--authority-manifest",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
    )
    assert rc == 0
    assert dropped_root not in manifest["repo_files"]
    assert set(trimmed) <= set(manifest["repo_files"])
    assert manifest["policy"]["authority_roots"] == sorted(trimmed)


def test_canonical_tier_cli_is_read_only_and_digest_bound(tmp_path):
    artifacts = _require_real_artifacts()  # subprocess CLI: needs the real artifacts
    sha = _head_sha()

    def _status() -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    before_status = _status()
    before_artifact_bytes = (artifacts[0].read_bytes(), artifacts[1].read_bytes())
    scratch_cwd = tmp_path / "empty-cwd"
    scratch_cwd.mkdir()

    classify = _run_cli(
        "--classify-tier",
        "--changed-file",
        "scripts/validate_openapi_routes.py",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
        cwd=scratch_cwd,
    )
    manifest_proc = _run_cli(
        "--authority-manifest",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
        cwd=scratch_cwd,
    )
    assert classify.returncode == 0, classify.stdout
    assert manifest_proc.returncode == 0, manifest_proc.stdout
    assert _status() == before_status
    assert (artifacts[0].read_bytes(), artifacts[1].read_bytes()) == before_artifact_bytes
    assert list(scratch_cwd.iterdir()) == []

    doc = json.loads(classify.stdout)
    manifest = json.loads(manifest_proc.stdout)
    assert doc["effective_tier"] == 4
    assert doc["matched_authority_path"] == "scripts/validate_openapi_routes.py"
    assert doc["authority_manifest_sha256"] == manifest["authority_manifest_sha256"]
    body = {k: v for k, v in manifest.items() if k != "authority_manifest_sha256"}
    assert (
        hashlib.sha256(_canonical_bytes(body)).hexdigest() == manifest["authority_manifest_sha256"]
    )
    facts = doc["inventory_facts"]
    assert facts == manifest["inventory_facts"]
    assert facts["original_record_id_set_sha256"] == gen.RATIFIED_ORIGINAL_ID_SET_SHA256
    assert len(facts["inventory_sha256"]) == 64


def test_read_only_cli_is_deterministic_across_double_run():
    artifacts = _require_real_artifacts()  # subprocess CLI: needs the real artifacts
    sha = _head_sha()
    art = _artifact_args(artifacts)
    invocations = (
        ["--authority-manifest", "--ref", sha, "--json", "--repo-root", str(REPO_ROOT), *art],
        [
            "--classify-tier",
            "--changed-file",
            "scripts/check_sdk_parity.py",
            "--ref",
            sha,
            "--json",
            "--repo-root",
            str(REPO_ROOT),
            *art,
        ],
        ["--ref", sha, "--json", "--repo-root", str(REPO_ROOT), *art],
    )
    for argv in invocations:
        first = _run_cli(*argv)
        second = _run_cli(*argv)
        assert first.returncode == 0, first.stdout
        assert second.returncode == 0, second.stdout
        assert first.stdout == second.stdout
        assert first.stdout.endswith(b"\n")


def test_original_cohort_descriptor_is_immutable_across_authority_versions(monkeypatch, capsys):
    artifacts = _require_artifacts()
    sha = _head_sha()
    argv = [
        "--authority-manifest",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
    ]
    rc, baseline = _run_standalone(monkeypatch, capsys, *argv)
    assert rc == 0

    rq = gen._review_queue_module()
    monkeypatch.setattr(
        rq,
        "CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION",
        rq.CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION + 1,
    )
    rc, bumped = _run_standalone(monkeypatch, capsys, *argv)
    assert rc == 0

    assert bumped["authority_manifest_sha256"] != baseline["authority_manifest_sha256"]
    assert bumped["artifacts"] == baseline["artifacts"]
    assert bumped["inventory_facts"] == baseline["inventory_facts"]

    descriptor = baseline["artifacts"]["original_cohort"]
    assert descriptor["logical_path"] == gen.COHORT_LOGICAL_PATH
    assert descriptor["byte_length"] == gen.COHORT_BYTE_LENGTH
    assert descriptor["sha256"] == gen.COHORT_SHA256
    provenance_descriptor = baseline["artifacts"]["sdk_provenance"]
    assert provenance_descriptor["logical_path"] == gen.PROVENANCE_LOGICAL_PATH
    assert provenance_descriptor["byte_length"] == gen.PROVENANCE_BYTE_LENGTH
    assert provenance_descriptor["sha256"] == gen.PROVENANCE_SHA256


def test_canonical_cohort_and_provenance_artifacts_are_in_authority_closure():
    artifacts = _require_real_artifacts()  # subprocess CLI: needs the real artifacts
    sha = _head_sha()
    proc = _run_cli(
        "--authority-manifest",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
    )
    assert proc.returncode == 0, proc.stdout
    manifest = json.loads(proc.stdout)

    assert manifest["artifacts"]["original_cohort"] == {
        "logical_path": gen.COHORT_LOGICAL_PATH,
        "byte_length": gen.COHORT_BYTE_LENGTH,
        "sha256": gen.COHORT_SHA256,
        "schema": gen.COHORT_SCHEMA,
        "canonical_bytes": True,
        "utf8_no_bom": True,
        "single_terminal_lf": True,
    }
    assert manifest["artifacts"]["sdk_provenance"] == {
        "logical_path": gen.PROVENANCE_LOGICAL_PATH,
        "byte_length": gen.PROVENANCE_BYTE_LENGTH,
        "sha256": gen.PROVENANCE_SHA256,
        "schema": gen.PROVENANCE_SCHEMA,
        "canonical_bytes": True,
        "utf8_no_bom": True,
        "single_terminal_lf": True,
    }

    facts = manifest["inventory_facts"]
    assert facts["membership_anchor"]["commit_sha"] == "6c4784330dca2d1709edf50b38f4d1201e92c83a"
    assert [source["path"] for source in facts["membership_sources"]] == [
        "scripts/baselines/verify_sdk_contracts.json",
        "scripts/baselines/validate_openapi_routes.json",
        "scripts/baselines/check_sdk_parity.json",
    ]
    assert facts["original_record_total"] == 655
    assert facts["original_record_id_set_sha256"] == gen.RATIFIED_ORIGINAL_ID_SET_SHA256
    assert facts["method_bearing_sdk_records"] == 598
    assert facts["method_null_route_parity_records"] == 57
    provenance_facts = facts["sdk_provenance"]
    assert (
        provenance_facts["baseline_birth"]["commit_sha"]
        == "af5edb22235d4c40a97fe3faa54168b406ab5696"
    )
    assert (
        provenance_facts["dependencies"]["verifier"]["git_blob_oid"]
        == "2d2a0e866f722dab0fad29f4283d1158aad0c408"
    )
    assert (
        provenance_facts["dependencies"]["normalizer"]["git_blob_oid"]
        == "5710fcd04234f997586a187ff8e2ace42f30dddc"
    )
    assert provenance_facts["record_count"] == 598
    assert provenance_facts["source_occurrence_count"] == 690
    assert provenance_facts["multi_atom_record_count"] == 12
    assert provenance_facts["missing_record_count"] == 0
    assert provenance_facts["record_digest_set_sha256"] == gen.RATIFIED_PROVENANCE_DIGEST_SET_SHA256
    partition = provenance_facts["partition"]
    assert partition["core_count"] == 75
    assert partition["extended_count"] == 523
    assert partition["sdk_original_record_id_set_sha256"] == gen.RATIFIED_SDK_ID_SET_SHA256
    assert partition["core_original_record_id_set_sha256"] == gen.RATIFIED_CORE_ID_SET_SHA256
    assert (
        partition["extended_original_record_id_set_sha256"] == gen.RATIFIED_EXTENDED_ID_SET_SHA256
    )
    projection = facts["operation_projection"]
    assert projection["membership_count"] == 655
    assert projection["operation_edge_count"] == 666
    assert projection["multi_edge_membership_count"] == 9
    assert projection["max_operation_edges"] == 4
    assert projection["record_digest_set_sha256"] == gen.RATIFIED_PROJECTION_DIGEST_SET_SHA256

    stdout_text = proc.stdout.decode("utf-8")
    assert str(artifacts[0]) not in stdout_text
    assert str(artifacts[0].parent) not in stdout_text


def test_standalone_ref_rejects_abbreviated_unresolved_or_malformed_shas(monkeypatch, capsys):
    head = _head_sha()
    cases = (
        (head[:12], "ref_invalid"),
        (head.upper(), "ref_invalid"),
        ("HEAD", "ref_invalid"),
        ("main", "ref_invalid"),
        ("f" * 40, "ref_unresolved"),
    )
    for ref, expected_code in cases:
        rc, doc = _run_standalone(
            monkeypatch,
            capsys,
            "--classify-tier",
            "--changed-file",
            "scripts/check_sdk_parity.py",
            "--ref",
            ref,
            "--json",
            "--repo-root",
            str(REPO_ROOT),
        )
        assert rc == 1, ref
        assert doc["status"] == "fail"
        assert doc["error"]["code"] == expected_code, ref


def test_standalone_artifact_tamper_fails_closed(monkeypatch, capsys, tmp_path):
    artifacts = _require_artifacts()
    sha = _head_sha()
    raw = artifacts[0].read_bytes()

    tampered = bytearray(raw)
    tampered[100] = (tampered[100] + 1) % 256
    flipped = tmp_path / "cohort-flipped.json"
    flipped.write_bytes(bytes(tampered))
    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        "--cohort-artifact",
        str(flipped),
        "--sdk-provenance-artifact",
        str(artifacts[1]),
    )
    assert rc == 1
    assert doc["error"]["code"] == "artifact_sha256_mismatch"

    truncated = tmp_path / "cohort-truncated.json"
    truncated.write_bytes(raw[:-1])
    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        "--cohort-artifact",
        str(truncated),
        "--sdk-provenance-artifact",
        str(artifacts[1]),
    )
    assert rc == 1
    assert doc["error"]["code"] == "artifact_byte_length_mismatch"

    symlink = tmp_path / "cohort-symlink.json"
    symlink.symlink_to(artifacts[0])
    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        "--cohort-artifact",
        str(symlink),
        "--sdk-provenance-artifact",
        str(artifacts[1]),
    )
    assert rc == 1
    assert doc["error"]["code"] == "artifact_symlink_forbidden"

    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        "--cohort-artifact",
        str(tmp_path / "absent.json"),
        "--sdk-provenance-artifact",
        str(artifacts[1]),
    )
    assert rc == 1
    assert doc["error"]["code"] == "artifact_missing"
    assert gen.COHORT_LOGICAL_PATH in doc["error"]["message"]
    assert str(tmp_path) not in doc["error"]["message"]


def test_classify_tier_rejects_traversal_absolute_and_non_normalized_paths(monkeypatch, capsys):
    sha = _head_sha()
    hostile_paths = (
        "/etc/passwd",
        "../secrets",
        "a/../b",
        "a//b",
        "./a",
        "a/./b",
        "a\\b",
        "scripts/",
        "",
    )
    for hostile in hostile_paths:
        rc, doc = _run_standalone(
            monkeypatch,
            capsys,
            "--classify-tier",
            "--changed-file",
            hostile,
            "--ref",
            sha,
            "--json",
            "--repo-root",
            str(REPO_ROOT),
        )
        assert rc == 1, hostile
        assert doc["error"]["code"] == "changed_file_invalid", hostile


def test_classify_tier_hostile_file_variants_stay_below_tier4(monkeypatch, capsys):
    artifacts = _require_artifacts()
    sha = _head_sha()
    root = "scripts/check_sdk_parity.py"

    def _classify(path: str) -> dict:
        rc, doc = _run_standalone(
            monkeypatch,
            capsys,
            "--classify-tier",
            "--changed-file",
            path,
            "--ref",
            sha,
            "--json",
            "--repo-root",
            str(REPO_ROOT),
            *_artifact_args(artifacts),
        )
        assert rc == 0, path
        return doc

    for hostile in (f"{root}.bak", f"{root}.old", f"{root}.pyx", f"{root}/child"):
        doc = _classify(hostile)
        assert doc["effective_tier"] < 4, hostile
        assert doc["matched_authority_path"] is None, hostile

    doc = _classify(root)
    assert doc["effective_tier"] == 4
    assert doc["matched_authority_path"] == root

    doc = _classify(".github/workflows/contract-drift-governance.yml")
    assert doc["effective_tier"] == 4
    assert doc["matched_authority_path"] == ".github/workflows/"


def test_authority_manifest_auto_discovers_mission_artifacts_and_fails_closed(
    monkeypatch, capsys, tmp_path
):
    artifacts = _require_artifacts()
    sha = _head_sha()

    monkeypatch.setenv("FACTORY_MISSION_DIR", str(artifacts[0].parent.parent))
    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--authority-manifest",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
    )
    assert rc == 0
    assert doc["status"] == "ok"
    assert doc["artifacts"]["original_cohort"]["logical_path"] == gen.COHORT_LOGICAL_PATH

    monkeypatch.delenv("FACTORY_MISSION_DIR", raising=False)
    monkeypatch.delenv("FACTORY_RUNTIME_SETTINGS_PATH", raising=False)
    bare_repo, bare_sha = _init_repo(tmp_path)
    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--authority-manifest",
        "--ref",
        bare_sha,
        "--json",
        "--repo-root",
        str(bare_repo),
    )
    assert rc == 1
    assert doc["error"]["code"] == "artifact_missing"


def test_authority_manifest_fails_closed_on_policy_module_ref_mismatch(monkeypatch, capsys):
    # Blobs are bound at --ref while classification uses the imported live
    # module; any byte difference between the two must refuse certification.
    artifacts = _require_artifacts()
    sha = _head_sha()
    monkeypatch.setenv("FACTORY_MISSION_DIR", str(artifacts[0].parent.parent))
    real_blob_at_ref = gen._git_blob_at_ref

    def _tampered(repo_root: Path, ref: str, path: str):
        blob = real_blob_at_ref(repo_root, ref, path)
        if path == gen.CANONICAL_CLASSIFIER_PATH and blob is not None:
            return blob + b"\n# drifted policy\n"
        return blob

    monkeypatch.setattr(gen, "_git_blob_at_ref", _tampered)
    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--authority-manifest",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
    )
    assert rc == 1
    assert doc["error"]["code"] == "policy_module_ref_mismatch"


def test_workflow_closure_includes_yaml_workflows(tmp_path: Path):
    """Quorum P2: GitHub runs both .yml and .yaml workflow files, so a .yaml
    workflow referencing an authority root must join the Tier-4 closure
    exactly like .yml. Workflows in nested subdirectories are never executed
    by GitHub (only the top level of .github/workflows runs) and stay
    excluded by design."""
    artifacts = _require_artifacts()
    root_ref = "scripts/check_sdk_parity.py"  # a canonical authority root
    workflows = {
        "gate.yml": f"jobs:\n  gate:\n    run: python {root_ref}\n",
        "gate.yaml": f"jobs:\n  gate:\n    run: python {root_ref}\n",
        "unrelated.yaml": "jobs:\n  noop:\n    run: echo ok\n",
        "nested/deep.yaml": f"jobs:\n  gate:\n    run: python {root_ref}\n",
    }
    repo, sha = cdg.init_authority_repo(tmp_path, workflows=workflows)
    manifest = gen._build_authority_manifest(repo, sha, artifacts[0], artifacts[1])
    repo_files = manifest["repo_files"]
    assert ".github/workflows/gate.yml" in repo_files
    assert ".github/workflows/gate.yaml" in repo_files
    assert ".github/workflows/unrelated.yaml" not in repo_files
    assert ".github/workflows/nested/deep.yaml" not in repo_files


def test_standalone_inventory_mode_happy_path(monkeypatch, capsys):
    """Hermetic happy path for the standalone inventory mode, including the
    inventory_sha256 self-digest construction that the boundary validator
    recomputes (canonical JSON of the document minus the digest field)."""
    artifacts = _require_artifacts()
    sha = _head_sha()
    rc, doc = _run_standalone(
        monkeypatch,
        capsys,
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
    )
    assert rc == 0
    assert doc["schema"] == gen.INVENTORY_OUTPUT_SCHEMA
    assert doc["status"] == "ok"
    assert doc["ref"] == sha
    assert doc["original_record_total"] == 655
    assert doc["method_bearing_sdk_records"] == 598
    assert doc["method_null_route_parity_records"] == 57
    assert doc["original_record_id_set_sha256"] == gen.RATIFIED_ORIGINAL_ID_SET_SHA256
    body = {key: value for key, value in doc.items() if key != "inventory_sha256"}
    assert doc["inventory_sha256"] == hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _git_text(*argv: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *argv],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout


def _top_level_ast_by_name(ref: str, path: str) -> dict[str, str]:
    import ast

    tree = ast.parse(_git_text("show", f"{ref}:{path}"))
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = ast.dump(node, include_attributes=False)
    return result


def _snapshot_read_only_state() -> dict[str, str]:
    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def digest_tree(root: Path, *, excluded: frozenset[str] = frozenset()) -> str:
        hashed = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] in excluded:
                continue
            hashed.update(relative.as_posix().encode())
            if path.is_symlink():
                hashed.update(os.readlink(path).encode())
            elif path.is_file():
                hashed.update(path.read_bytes())
        return hashed.hexdigest()

    def git_path(name: str) -> Path:
        raw = _git_text("rev-parse", "--path-format=absolute", "--git-path", name).strip()
        return Path(raw)

    index = git_path("index")
    worktree_gitdir = Path(_git_text("rev-parse", "--absolute-git-dir").strip())
    common_gitdir = Path(
        _git_text("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    )
    return {
        "tracked_untracked": digest(
            _git_text("status", "--porcelain=v1", "--untracked-files=all").encode()
        ),
        "index": digest(index.read_bytes()),
        "worktree_gitdir": digest_tree(worktree_gitdir, excluded=frozenset({"index"})),
        "common_gitdir": digest_tree(
            common_gitdir,
            excluded=frozenset({"objects", "refs", "logs", "worktrees"}),
        ),
        "objects": digest(
            (
                _git_text("rev-list", "--objects", "--all") + _git_text("count-objects", "-v")
            ).encode()
        ),
        "refs": digest(_git_text("show-ref", "--head").encode()),
        "reflogs": digest(_git_text("reflog", "show", "--all", "--format=%H %gD").encode()),
    }


def test_classifier_pr_changes_are_constants_only():
    commit = "ee686e9d116c704ede146a6ec69dfe013b6c32be"
    changed = set(_git_text("show", "--format=", "--name-only", commit).splitlines())
    assert changed == {
        "aragora/cli/commands/review_queue.py",
        "scripts/tier4_merge_train.py",
        "tests/governance/test_contract_drift_measurement_authority_tier.py",
    }
    allowed = {
        "CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION",
        "CONTRACT_DRIFT_AUTHORITY_TIER",
        "CONTRACT_DRIFT_AUTHORITY_PREFIXES",
        "CONTRACT_DRIFT_AUTHORITY_CANONICAL_SOURCE",
        "TIER_4_PREFIXES",
        "SERIALIZED_TIER4_PREFIXES",
    }
    for path in ("aragora/cli/commands/review_queue.py", "scripts/tier4_merge_train.py"):
        before = _top_level_ast_by_name(f"{commit}^", path)
        after = _top_level_ast_by_name(commit, path)
        changed_symbols = {
            name for name in set(before) | set(after) if before.get(name) != after.get(name)
        }
        assert changed_symbols <= allowed


def test_matcher_repair_preserves_eight_constants_and_single_line_authority_tuple():
    from aragora.cli.commands import review_queue

    commit = "e8a0d165242737d3226b6d3360aa9e8ec014fd75"
    before = _top_level_ast_by_name(f"{commit}^", "aragora/cli/commands/review_queue.py")
    after = _top_level_ast_by_name(commit, "aragora/cli/commands/review_queue.py")
    for name in (
        "CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION",
        "CONTRACT_DRIFT_AUTHORITY_TIER",
        "CONTRACT_DRIFT_AUTHORITY_PREFIXES",
        "TIER_4_PREFIXES",
    ):
        assert before[name] == after[name]
    assert len(review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES) == 8
    source = Path(review_queue.__file__).read_text(encoding="utf-8")
    assert "    *CONTRACT_DRIFT_AUTHORITY_PREFIXES," in source.splitlines()


def test_file_authority_roots_match_exact_equality_only():
    from aragora.cli.commands import review_queue

    exact_roots = [root for root in review_queue.TIER_4_PREFIXES if not root.endswith("/")]
    for root in exact_roots:
        assert review_queue._matches_prefix(root, (root,))
        for suffix in (".bak", ".old", ".pyx", "/child"):
            assert not review_queue._matches_prefix(f"{root}{suffix}", (root,))


def test_directory_authority_roots_match_descendants():
    from aragora.cli.commands import review_queue

    directory_roots = [root for root in review_queue.TIER_4_PREFIXES if root.endswith("/")]
    assert directory_roots
    for root in directory_roots:
        assert review_queue._matches_prefix(f"{root}one.yml", (root,))
        assert review_queue._matches_prefix(f"{root}nested/two.yml", (root,))
        assert not review_queue._matches_prefix(root.rstrip("/"), (root,))


def test_review_queue_and_merge_train_report_identical_match_and_serialization():
    from aragora.cli.commands import review_queue
    from scripts import tier4_merge_train

    matrix: list[str] = []
    for root in review_queue.TIER_4_PREFIXES:
        if root.endswith("/"):
            matrix.extend((f"{root}one.yml", f"{root}nested/two.yml"))
        else:
            matrix.extend((root, f"{root}.bak", f"{root}.old", f"{root}.pyx", f"{root}/child"))
    for path in matrix:
        canonical = next(
            (
                root
                for root in review_queue.TIER_4_PREFIXES
                if review_queue._matches_prefix(path, (root,))
            ),
            None,
        )
        mirrored = tier4_merge_train.matches_serialized_path(path)
        assert canonical == mirrored
        assert (review_queue._classify_model_review_tier([path])[0] == 4) == (mirrored is not None)


def test_file_root_suffix_and_child_variants_are_below_tier4_and_unserialized():
    from aragora.cli.commands import review_queue
    from scripts import tier4_merge_train

    for root in review_queue.TIER_4_PREFIXES:
        if root.endswith("/"):
            continue
        for suffix in (".bak", ".old", ".pyx", "/child"):
            path = f"{root}{suffix}"
            assert review_queue._classify_model_review_tier([path])[0] < 4
            assert tier4_merge_train.serialized_paths_for([path]) == set()


def test_exact_authority_roots_serialize_and_contend():
    from aragora.cli.commands import review_queue
    from scripts import tier4_merge_train

    for index, root in enumerate(review_queue.TIER_4_PREFIXES, start=1):
        path = f"{root}probe.yml" if root.endswith("/") else root
        empty = tier4_merge_train.evaluate_merge_train([path], [])
        occupied = tier4_merge_train.evaluate_merge_train(
            [path],
            [
                {
                    "number": 9000 + index,
                    "createdAt": "2026-07-17T00:00:00Z",
                    "files": [{"path": path}],
                }
            ],
        )
        assert empty["decision"] == tier4_merge_train.ALLOW
        assert occupied["decision"] == tier4_merge_train.QUEUE
        assert occupied["blocking_prs"] == [9000 + index]
        assert occupied["candidate_surfaces"] == [root]


def test_directory_descendants_remain_tier4():
    from aragora.cli.commands import review_queue
    from scripts import tier4_merge_train

    for root in review_queue.TIER_4_PREFIXES:
        if not root.endswith("/"):
            continue
        for path in (f"{root}one.yml", f"{root}nested/two.yml"):
            assert review_queue._classify_model_review_tier([path])[0] == 4
            assert tier4_merge_train.serialized_paths_for([path]) == {root}


def test_parser_prohibition_is_behavioral_or_targeted_ast_registration():
    from aragora.cli.parser import build_parser

    parser = build_parser()
    for command in ("classify-tier", "authority-manifest", "boundary-validator"):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["review-queue", command])
        assert exc_info.value.code == 2


def test_matcher_repair_has_no_parser_dispatch_handler_or_settlement_scope():
    commit = "e8a0d165242737d3226b6d3360aa9e8ec014fd75"
    changed = set(_git_text("show", "--format=", "--name-only", commit).splitlines())
    assert changed == {
        "aragora/cli/commands/review_queue.py",
        "tests/governance/test_contract_drift_measurement_authority_tier.py",
        "tests/scripts/test_tier4_merge_train.py",
    }
    before = _top_level_ast_by_name(f"{commit}^", "aragora/cli/commands/review_queue.py")
    after = _top_level_ast_by_name(commit, "aragora/cli/commands/review_queue.py")
    changed_symbols = {
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    }
    assert changed_symbols == {"_matches_prefix"}


def test_read_only_cli_hashes_worktree_index_gitdirs_objects_refs_and_reflogs():
    artifacts = _require_real_artifacts()  # subprocess CLI: needs the real artifacts
    sha = _head_sha()
    argv = (
        "--authority-manifest",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
    )
    for _attempt in range(3):
        before = _snapshot_read_only_state()
        first = _run_cli(*argv)
        second = _run_cli(*argv)
        after = _snapshot_read_only_state()
        if before == after:
            break
    assert before == after
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert set(before) == {
        "tracked_untracked",
        "index",
        "worktree_gitdir",
        "common_gitdir",
        "objects",
        "refs",
        "reflogs",
    }


def test_read_only_cli_allows_only_scratch_and_output_writes(tmp_path: Path):
    artifacts = _require_real_artifacts()  # subprocess CLI: needs the real artifacts
    sha = _head_sha()
    scratch = tmp_path / "scratch"
    output = tmp_path / "output"
    scratch.mkdir()
    output.mkdir()
    proc = _run_cli(
        "--classify-tier",
        "--changed-file",
        "scripts/check_contract_drift_ratchet.py",
        "--ref",
        sha,
        "--json",
        "--repo-root",
        str(REPO_ROOT),
        *_artifact_args(artifacts),
        cwd=scratch,
    )
    assert proc.returncode == 0
    (output / "classification.json").write_bytes(proc.stdout)
    assert list(scratch.iterdir()) == []
    assert [path.name for path in output.iterdir()] == ["classification.json"]


def test_reporting_only_paths_remain_below_tier4():
    from aragora.cli.commands import review_queue
    from scripts import tier4_merge_train

    for path in (
        "scripts/contract_drift_report.py",
        "scripts/generate_contract_drift_backlog.py",
        "scripts/generate_contract_drift_issue_plan.py",
    ):
        assert review_queue._classify_model_review_tier([path])[0] < 4
        assert tier4_merge_train.matches_serialized_path(path) is None


def test_unrelated_sibling_path_is_not_overclassified():
    from aragora.cli.commands import review_queue
    from scripts import tier4_merge_train

    for path in (
        "scripts/check_cross_sdk_parity.py",
        "scripts/sdk_parity_audit.py",
        "scripts/baselines/validate_openapi_routes.json",
        "scripts/baselines/check_sdk_parity.json",
    ):
        assert review_queue._classify_model_review_tier([path])[0] < 4
        assert tier4_merge_train.matches_serialized_path(path) is None
