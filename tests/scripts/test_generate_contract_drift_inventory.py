"""Tests for scripts/generate_contract_drift_inventory.py."""

from __future__ import annotations

import copy
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.check_contract_drift_ratchet as ratchet
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


FIXTURE_ROOT = "scripts/generate_contract_drift_inventory.py"
FIXTURE_DEPENDENCIES = (
    ".github/actions/nested/action.yml",
    ".github/actions/root/action.yaml",
    ".github/scripts/tool.py",
    "aragora/__init__.py",
    "aragora/cli/__init__.py",
    "aragora/cli/commands/__init__.py",
    "aragora/cli/commands/review_queue.py",
    "aragora/helper.py",
    "aragora/runtime_helper.py",
    "scripts/helper.py",
    "scripts/helper.sh",
    "scripts/pkg/__main__.py",
    "scripts/tier4_merge_train.py",
    "scripts/transitive.py",
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _authority_fixture(
    tmp_path: Path,
    *,
    dynamic_reference: bool = False,
    dynamic_run_reference: str | None = None,
    helper_source: str = "import aragora.helper\n",
    runtime_helper_source: str = "# runtime policy helper\n",
    missing_policy: bool = False,
    namespace_package: bool = False,
    policy_prelude: str = "",
    policy_epilogue: str = "",
    symlink_policy: bool = False,
) -> tuple[Path, str]:
    repo = tmp_path / "authority-repo"
    repo.mkdir()
    _write_baselines(repo, VERIFY, ROUTES, PARITY)
    _write_text(repo / FIXTURE_ROOT, "# exact-ref fixture authority root\n")
    _write_text(repo / "scripts/tier4_merge_train.py", _fixture_merge_train_source())
    _write_text(repo / "scripts/helper.py", helper_source)
    _write_text(repo / "scripts/helper.sh", "python scripts/transitive.py\n")
    _write_text(repo / "scripts/transitive.py", "# transitive workflow helper\n")
    _write_text(repo / "scripts/pkg/__main__.py", "# package module runner\n")
    _write_text(repo / ".github/scripts/tool.py", "# github helper\n")
    _write_text(repo / "scripts/measured.py", "# not an authority dependency\n")
    _write_text(repo / "sdk/python/client.py", "# measured SDK subject\n")
    if not namespace_package:
        _write_text(repo / "aragora/__init__.py", "")
    _write_text(repo / "aragora/helper.py", "# repository helper\n")
    _write_text(repo / "aragora/runtime_helper.py", runtime_helper_source)
    _write_text(repo / "aragora/cli/__init__.py", "")
    _write_text(repo / "aragora/cli/commands/__init__.py", "")
    if not missing_policy:
        policy_source = policy_prelude + _fixture_review_queue_source() + policy_epilogue
        if symlink_policy:
            _write_text(repo / "aragora/cli/copied_review_queue.py", policy_source)
            (repo / "aragora/cli/commands/review_queue.py").symlink_to("../copied_review_queue.py")
        else:
            _write_text(repo / "aragora/cli/commands/review_queue.py", policy_source)
    dynamic_uses = (
        "      - uses: ./.github/actions/${{ inputs.action }}\n"
        if dynamic_reference
        else "      - uses: ./.github/actions/root\n"
    )
    dynamic_runs = {
        "leading": '      - run: python "$HELPER_SCRIPT"\n',
        "leading_block": '      - run: |\n          python "$HELPER_SCRIPT"\n',
        "mid_token": "      - run: python scripts/$HELPER.py\n",
        "glob": "      - run: python ./*.py\n",
        "module": "      - run: python -m scripts.helper\n",
        "module_block": "      - run: |\n          python -m scripts.helper\n",
        "module_substitution": '      - run: echo "$(python -m scripts.helper)"\n',
        "package_module": "      - run: python -m scripts.pkg\n",
        "github_literal": "      - run: python .github/scripts/tool.py\n",
        "github_dynamic": "      - run: python .github/scripts/$HELPER.py\n",
    }
    helper_run = dynamic_runs.get(
        dynamic_run_reference, "      - run: python sdk/python/client.py\n"
    )
    _write_text(
        repo / ".github/workflows/authority.yml",
        "on:\n"
        "  push:\n"
        "    paths:\n"
        f"      - '{FIXTURE_ROOT}'\n"
        "jobs:\n"
        "  authority:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        f"{dynamic_uses}"
        f"{helper_run}"
        "  nested:\n"
        "    uses: ./.github/workflows/nested.yaml\n",
    )
    _write_text(
        repo / ".github/workflows/ignore-only.yml",
        "on:\n"
        "  push:\n"
        "    paths-ignore:\n"
        f"      - '{FIXTURE_ROOT}'\n"
        "jobs:\n"
        "  ignored:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: external/action@v1\n"
        "        with:\n"
        f"          paths: ['{FIXTURE_ROOT}']\n"
        "      - uses: ./.github/actions/does-not-exist\n",
    )
    _write_text(
        repo / ".github/workflows/ignore-unrelated.yaml",
        "on:\n"
        "  pull_request:\n"
        "    paths-ignore:\n"
        "      - 'docs/**'\n"
        "jobs:\n"
        "  authority:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: python .github/scripts/tool.py\n",
    )
    _write_text(
        repo / ".github/workflows/negated.yml",
        "on:\n"
        "  pull_request:\n"
        "    paths:\n"
        "      - 'scripts/**'\n"
        f"      - '!{FIXTURE_ROOT}'\n"
        "jobs:\n"
        "  ignored:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/does-not-exist\n",
    )
    _write_text(
        repo / ".github/workflows/nested.yaml",
        "on:\n"
        "  workflow_call:\n"
        "jobs:\n"
        "  helper:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: python scripts/helper.py\n"
        "      - uses: ./.github/actions/root\n"
        "      - uses: ./.github/actions/root\n"
        "  cycle:\n"
        "    uses: ./.github/workflows/authority.yml\n",
    )
    _write_text(
        repo / ".github/actions/root/action.yaml",
        "name: root\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - run: bash scripts/helper.sh\n"
        "      shell: bash\n"
        "    - uses: ./.github/actions/nested\n",
    )
    _write_text(
        repo / ".github/actions/nested/action.yml",
        "name: nested\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - run: python scripts/transitive.py\n"
        "      shell: bash\n",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


def _fixture_review_queue_source() -> str:
    dependencies = repr(FIXTURE_DEPENDENCIES)
    return (
        f"CONTRACT_DRIFT_AUTHORITY_PREFIXES = ({FIXTURE_ROOT!r},)\n"
        "CONTRACT_DRIFT_AUTHORITY_TIER = 4\n"
        "CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION = 7\n"
        f"CONTRACT_DRIFT_AUTHORITY_DEPENDENCY_PREFIXES = {dependencies}\n"
        "TIER_4_PREFIXES = (\n"
        "    '.github/workflows/',\n"
        "    *CONTRACT_DRIFT_AUTHORITY_PREFIXES,\n"
        "    *CONTRACT_DRIFT_AUTHORITY_DEPENDENCY_PREFIXES,\n"
        ")\n"
        "def _matches_prefix(path, prefixes):\n"
        "    return any(path.startswith(rule) if rule.endswith('/') else path == rule "
        "for rule in prefixes)\n"
        "def _classify_model_review_tier(files, *, pr=None):\n"
        "    if any(_matches_prefix(path, TIER_4_PREFIXES) for path in files):\n"
        "        return (4, 'tier_4_preapproval_required', 'fixture authority')\n"
        "    return (2, 'tier_2_live_automation', 'fixture non-authority')\n"
    )


def _fixture_merge_train_source() -> str:
    dependencies = repr(FIXTURE_DEPENDENCIES)
    return (
        f"CONTRACT_DRIFT_AUTHORITY_PREFIXES = ({FIXTURE_ROOT!r},)\n"
        f"CONTRACT_DRIFT_AUTHORITY_DEPENDENCY_PREFIXES = {dependencies}\n"
        "SERIALIZED_TIER4_PREFIXES = (\n"
        "    '.github/workflows/',\n"
        "    *CONTRACT_DRIFT_AUTHORITY_PREFIXES,\n"
        "    *CONTRACT_DRIFT_AUTHORITY_DEPENDENCY_PREFIXES,\n"
        ")\n"
        "def matches_serialized_path(path):\n"
        "    for rule in SERIALIZED_TIER4_PREFIXES:\n"
        "        if path.startswith(rule) if rule.endswith('/') else path == rule:\n"
        "            return rule\n"
        "    return None\n"
    )


def _manifest(repo: Path, sha: str) -> dict:
    return gen.build_authority_manifest(repo, sha)


def _filesystem_snapshot(root: Path) -> list[tuple[str, str, str]]:
    snapshot: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot.append((relative, "symlink", os.readlink(path)))
        elif path.is_file():
            snapshot.append((relative, "file", gen._sha256(path.read_bytes())))
        elif path.is_dir():
            snapshot.append((relative, "directory", ""))
    return snapshot


def test_deterministic_bounded_authority_dependency_closure_has_incoming_edges_and_exact_ref_digests(
    tmp_path: Path,
):
    repo, sha = _authority_fixture(tmp_path)
    first = _manifest(repo, sha)
    second = _manifest(repo, sha)
    assert gen._canonical_json_bytes(first) == gen._canonical_json_bytes(second)
    assert first["ref"] == sha
    paths = [entry["path"] for entry in first["repo_files"]]
    assert paths == sorted(set(paths))
    assert first["authority_roots"] == [FIXTURE_ROOT]
    for entry in first["repo_files"]:
        raw = subprocess.run(
            ["git", "cat-file", "blob", entry["git_blob_oid"]],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        assert entry["byte_length"] == len(raw)
        assert entry["sha256"] == gen._sha256(raw)
        if not entry["authority_root"]:
            assert entry["incoming_edges"]


def test_measured_sdk_handler_openapi_subjects_are_not_authority_dependencies(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    paths = {entry["path"] for entry in _manifest(repo, sha)["repo_files"]}
    assert "sdk/python/client.py" not in paths
    assert "scripts/measured.py" not in paths


def test_merge_train_mirror_is_normal_repo_file_authority_member(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    mirror = files["scripts/tier4_merge_train.py"]
    assert mirror["authority_root"] is False
    assert mirror["tier"] == 4
    assert mirror["incoming_edges"] == [{"from": FIXTURE_ROOT, "kind": "merge_train_mirror"}]


def test_workflows_yml_and_yaml_recurse_through_structural_run_uses_and_path_filters(
    tmp_path: Path,
):
    repo, sha = _authority_fixture(tmp_path)
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert ".github/workflows/authority.yml" in files
    assert ".github/workflows/ignore-unrelated.yaml" in files
    assert ".github/workflows/nested.yaml" in files
    assert any(
        edge["kind"] == "workflow_path_filter"
        for edge in files[".github/workflows/authority.yml"]["incoming_edges"]
    )
    assert "scripts/helper.py" in files
    assert "scripts/helper.sh" in files
    assert "scripts/transitive.py" in files
    assert "aragora/helper.py" in files
    assert any(
        edge["kind"] == "workflow_path_ignore"
        for edge in files[".github/workflows/ignore-unrelated.yaml"]["incoming_edges"]
    )
    assert any(
        edge
        == {
            "from": ".github/workflows/ignore-unrelated.yaml",
            "kind": "workflow_run_executable",
        }
        for edge in files[".github/scripts/tool.py"]["incoming_edges"]
    )


def test_local_reusable_workflows_and_composite_actions_join_closure(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert ".github/actions/root/action.yaml" in files
    assert ".github/actions/nested/action.yml" in files
    assert any(
        edge["kind"] == "local_reusable_workflow"
        for edge in files[".github/workflows/nested.yaml"]["incoming_edges"]
    )
    assert len(files[".github/actions/root/action.yaml"]["incoming_edges"]) == 2


def test_unresolved_or_dynamic_local_workflow_reference_fails_closed(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path, dynamic_reference=True)
    with pytest.raises(gen.AuthorityClosureError, match="dynamic local reference"):
        _manifest(repo, sha)


@pytest.mark.parametrize(
    "variant", ["leading", "leading_block", "mid_token", "glob", "github_dynamic"]
)
def test_dynamic_local_run_reference_fails_closed(tmp_path: Path, variant: str):
    repo, sha = _authority_fixture(tmp_path, dynamic_run_reference=variant)
    with pytest.raises(gen.AuthorityClosureError, match="dynamic local run target"):
        _manifest(repo, sha)


@pytest.mark.parametrize("variant", ["module", "module_block", "module_substitution"])
def test_repository_python_module_run_target_joins_closure(tmp_path: Path, variant: str):
    repo, sha = _authority_fixture(tmp_path, dynamic_run_reference=variant)
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert any(
        edge
        == {
            "from": ".github/workflows/authority.yml",
            "kind": "workflow_run_executable",
        }
        for edge in files["scripts/helper.py"]["incoming_edges"]
    )


def test_repository_python_package_module_uses_dunder_main(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path, dynamic_run_reference="package_module")
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert any(
        edge["kind"] == "workflow_run_executable"
        for edge in files["scripts/pkg/__main__.py"]["incoming_edges"]
    )


def test_dot_github_script_reference_preserves_repository_prefix(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path, dynamic_run_reference="github_literal")
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert ".github/scripts/tool.py" in files


def test_literal_shell_subprocess_helper_joins_closure(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import aragora.helper\n"
            "import subprocess\n"
            "subprocess.run('python scripts/transitive.py', shell=True)\n"
        ),
    )
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert any(
        edge["kind"] == "literal_subprocess_helper"
        for edge in files["scripts/transitive.py"]["incoming_edges"]
    )


def test_imported_member_function_subprocess_helper_joins_closure(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source="import aragora.runtime_helper\n",
        runtime_helper_source=(
            "import subprocess\n"
            "def run_helper():\n"
            "    subprocess.run(['python', '-m', 'scripts.pkg'])\n"
        ),
    )
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert {
        "from": "aragora/runtime_helper.py",
        "kind": "literal_subprocess_helper",
    } in files["scripts/pkg/__main__.py"]["incoming_edges"]


def test_imported_member_dynamic_function_subprocess_fails_closed(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source="import aragora.runtime_helper\n",
        runtime_helper_source=(
            "import subprocess\ndef run_helper(command):\n    subprocess.run(command)\n"
        ),
    )
    with pytest.raises(gen.AuthorityClosureError, match="dynamic subprocess command"):
        _manifest(repo, sha)


def test_dynamic_shell_subprocess_fails_closed(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import subprocess\n"
            "name = 'transitive'\n"
            "subprocess.run(f'python scripts/{name}.py', shell=True)\n"
        ),
    )
    with pytest.raises(gen.AuthorityClosureError, match="dynamic shell subprocess"):
        _manifest(repo, sha)


def test_literal_bash_c_subprocess_helper_joins_closure(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import subprocess\nsubprocess.run(['bash', '-c', 'python scripts/transitive.py'])\n"
        ),
    )
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert {
        "from": "scripts/helper.py",
        "kind": "literal_subprocess_helper",
    } in files["scripts/transitive.py"]["incoming_edges"]


def test_dynamic_bash_c_subprocess_fails_closed(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import subprocess\n"
            "command = f'python scripts/{input()}.py'\n"
            "subprocess.run(['bash', '-c', command])\n"
        ),
    )
    with pytest.raises(gen.AuthorityClosureError, match="dynamic shell subprocess command"):
        _manifest(repo, sha)


@pytest.mark.parametrize(
    "shell_command",
    [
        "scripts/${HELPER}_gate.sh",
        'scripts/"${HELPER}".sh',
    ],
)
def test_dynamic_direct_bash_c_repository_target_fails_closed(tmp_path: Path, shell_command: str):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(f"import subprocess\nsubprocess.run(['bash', '-c', {shell_command!r}])\n"),
    )
    with pytest.raises(gen.AuthorityClosureError, match="dynamic local run target"):
        _manifest(repo, sha)


def test_dynamic_subprocess_keyword_splat_fails_closed(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import subprocess\n"
            "options = {'shell': True}\n"
            "subprocess.run('python scripts/transitive.py', **options)\n"
        ),
    )
    with pytest.raises(gen.AuthorityClosureError, match="dynamic subprocess keyword"):
        _manifest(repo, sha)


@pytest.mark.parametrize(
    "helper_source",
    [
        ("import subprocess as sp\nsp.run(['python', 'scripts/transitive.py'], check=True)\n"),
        (
            "from subprocess import run as execute\n"
            "execute(['python', 'scripts/transitive.py'], check=True)\n"
        ),
    ],
)
def test_aliased_subprocess_helpers_join_closure(tmp_path: Path, helper_source: str):
    repo, sha = _authority_fixture(tmp_path, helper_source=helper_source)
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert any(
        edge["kind"] == "literal_subprocess_helper"
        for edge in files["scripts/transitive.py"]["incoming_edges"]
    )


def test_non_subprocess_run_with_keyword_splat_is_ignored(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import asyncio\n"
            "async def main():\n"
            "    return None\n"
            "options = {}\n"
            "asyncio.run(main(), **options)\n"
        ),
    )
    assert _manifest(repo, sha)["ref"] == sha


def test_unknown_dynamic_subprocess_executable_fails_closed(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import subprocess\n"
            "from pathlib import Path\n"
            "some_tmp_dir = Path('/tmp')\n"
            "subprocess.run([some_tmp_dir / 'scripts/missing.py'], check=False)\n"
        ),
    )
    with pytest.raises(gen.AuthorityClosureError, match="dynamic subprocess executable"):
        _manifest(repo, sha)


@pytest.mark.parametrize(
    "loader",
    [
        "from importlib import import_module\nimport_module('aragora.helper')\n",
        "import runpy\nrunpy.run_module('aragora.helper')\n",
        "__import__('aragora.helper')\n",
    ],
)
def test_static_dynamic_import_apis_join_closure(tmp_path: Path, loader: str):
    repo, sha = _authority_fixture(tmp_path, helper_source=loader)
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert any(
        edge["kind"] == "python_repository_import"
        for edge in files["aragora/helper.py"]["incoming_edges"]
    )


def test_dynamic_import_with_bounded_repository_literals_joins_closure(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "from importlib import import_module\n"
            "module_name = 'aragora.helper'\n"
            "import_module(module_name)\n"
        ),
    )
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert {
        "from": "scripts/helper.py",
        "kind": "python_repository_import",
    } in files["aragora/helper.py"]["incoming_edges"]


def test_unbounded_dynamic_repository_import_fails_closed(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "from importlib import import_module\n"
            "module_name = 'aragora.' + input()\n"
            "import_module(module_name)\n"
        ),
    )
    with pytest.raises(gen.AuthorityClosureError, match="dynamic repository import"):
        _manifest(repo, sha)


@pytest.mark.parametrize(
    "source_path",
    [
        "scripts/check_sdk_parity.py",
        "scripts/check_test_dependencies.py",
        "scripts/generate_api_docs.py",
        "scripts/generate_openapi.py",
    ],
)
def test_dynamic_non_authority_import_is_excluded_from_closure(tmp_path: Path, source_path: str):
    source = tmp_path / source_path
    _write_text(
        source,
        "import importlib\n"
        "def discover(module_path):\n"
        "    return importlib.import_module(module_path)\n",
    )
    assert gen._python_import_edges(tmp_path, source_path) == []


@pytest.mark.parametrize(
    "path",
    [
        "aragora/server/handler_registry.py",
        "aragora/server/handler_registry/core.py",
        "aragora/server/handlers/example.py",
        "aragora/server/openapi/schemas/example.py",
        "docs/api/openapi_generated.json",
        "sdk/python/client.py",
    ],
)
def test_measurement_subject_implementations_are_excluded(path: str):
    assert gen._is_measurement_subject(path)


def test_repository_import_resolution_matches_package_precedence(tmp_path: Path):
    _write_text(tmp_path / "aragora/registry.py", "# compatibility module\n")
    _write_text(tmp_path / "aragora/registry/__init__.py", "# canonical package\n")
    assert gen._resolve_module_path(tmp_path, "aragora.registry") == "aragora/registry/__init__.py"


@pytest.mark.parametrize(
    "helper_source",
    [
        "for _ in range(1):\n    import aragora.helper\n",
        "while False:\n    import aragora.helper\n",
        "match 1:\n    case 1:\n        import aragora.helper\n",
    ],
)
def test_compound_load_time_imports_join_closure(tmp_path: Path, helper_source: str):
    repo, sha = _authority_fixture(tmp_path, helper_source=helper_source)
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert any(
        edge["kind"] == "python_repository_import"
        for edge in files["aragora/helper.py"]["incoming_edges"]
    )


@pytest.mark.parametrize(
    "helper_source",
    [
        "def discover():\n    import aragora.helper\ndiscover()\n",
        "class Discovery:\n    import aragora.helper\n",
        (
            "@__import__('aragora.helper').decorate\n"
            "def discover(\n"
            "    value: __import__('aragora.helper').Value = __import__('aragora.helper'),\n"
            ") -> __import__('aragora.helper').Value:\n"
            "    return value\n"
        ),
    ],
)
def test_function_and_class_imports_join_closure(tmp_path: Path, helper_source: str):
    _write_text(tmp_path / "scripts/check_contract_drift_ratchet.py", helper_source)
    _write_text(tmp_path / "aragora/__init__.py", "")
    _write_text(tmp_path / "aragora/helper.py", "# helper\n")
    assert (
        "aragora/helper.py",
        "python_repository_import",
    ) in gen._python_import_edges(
        tmp_path,
        "scripts/check_contract_drift_ratchet.py",
        include_function_bodies=True,
    )


def test_non_designated_class_body_import_joins_closure(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source="class Discovery:\n    import aragora.helper\n",
    )
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert {
        "from": "scripts/helper.py",
        "kind": "python_repository_import",
    } in files["aragora/helper.py"]["incoming_edges"]


def test_transitive_executable_function_import_joins_closure(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "def load_runtime():\n"
            "    import aragora.helper\n"
            "def main():\n"
            "    load_runtime()\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    )
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert {
        "from": "scripts/helper.py",
        "kind": "python_repository_import",
    } in files["aragora/helper.py"]["incoming_edges"]


def test_measurement_runner_function_imports_are_not_authority_dependencies(tmp_path: Path):
    _write_text(
        tmp_path / "scripts/smoke_test.py",
        (
            "def check_product():\n"
            "    import aragora.helper\n"
            "def main():\n"
            "    check_product()\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    )
    _write_text(tmp_path / "aragora/__init__.py", "")
    _write_text(tmp_path / "aragora/helper.py", "# measured product subject\n")
    assert (
        gen._python_import_edges(
            tmp_path,
            "scripts/smoke_test.py",
            include_function_bodies=True,
        )
        == []
    )


@pytest.mark.parametrize(
    "helper_source",
    [
        (
            "import subprocess\n"
            "name = 'helper'\n"
            "subprocess.run(['python', f'scripts/{name}.py'], check=False)\n"
        ),
        (
            "import subprocess\n"
            "module = 'scripts.' + input()\n"
            "subprocess.run(['python', '-m', module], check=False)\n"
        ),
    ],
)
def test_dynamic_non_shell_python_subprocess_target_fails_closed(
    tmp_path: Path, helper_source: str
):
    repo, sha = _authority_fixture(tmp_path, helper_source=helper_source)
    with pytest.raises(gen.AuthorityClosureError, match="dynamic Python subprocess"):
        _manifest(repo, sha)


@pytest.mark.parametrize(
    "helper_source",
    [
        (
            "import subprocess\n"
            "module = 'scripts.transitive'\n"
            "subprocess.run(['python', '-m', module], check=False)\n"
        ),
        (
            "import subprocess\n"
            "script_path = 'scripts/transitive.py'\n"
            "subprocess.run(['python', script_path], check=False)\n"
        ),
        (
            "import subprocess\n"
            "import sys\n"
            "PYTHON = sys.executable\n"
            "subprocess.run([PYTHON, 'scripts/transitive.py'], check=False)\n"
        ),
        (
            "import subprocess\n"
            "def run():\n"
            "    cmd = ['python', 'scripts/transitive.py']\n"
            "    subprocess.run(cmd, check=False)\n"
            "run()\n"
        ),
    ],
)
def test_statically_bound_python_subprocess_targets_join_closure(
    tmp_path: Path, helper_source: str
):
    repo, sha = _authority_fixture(tmp_path, helper_source=helper_source)
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert {
        "from": "scripts/helper.py",
        "kind": "literal_subprocess_helper",
    } in files["scripts/transitive.py"]["incoming_edges"]


@pytest.mark.parametrize(
    "helper_source",
    [
        (
            "import subprocess\n"
            "cmd = ['python', 'scripts/transitive.py']\n"
            "cmd = build_command()\n"
            "subprocess.run(cmd, check=False)\n"
        ),
        (
            "import subprocess\n"
            "def unrelated():\n"
            "    cmd = ['python', 'scripts/transitive.py']\n"
            "cmd = build_command()\n"
            "subprocess.run(cmd, check=False)\n"
        ),
        (
            "import subprocess\n"
            "cmd = ['python', 'scripts/transitive.py']\n"
            "cmd.append('--quiet')\n"
            "subprocess.run(cmd, check=False)\n"
        ),
        (
            "import subprocess\n"
            "cmd = ['python', 'scripts/transitive.py']\n"
            "for cmd in commands:\n"
            "    pass\n"
            "subprocess.run(cmd, check=False)\n"
        ),
        (
            "import subprocess\n"
            "cmd = ['python', 'scripts/transitive.py']\n"
            "cmd, other = build_commands()\n"
            "subprocess.run(cmd, check=False)\n"
        ),
        (
            "import subprocess\n"
            "cmd = ['python', 'scripts/transitive.py']\n"
            "cmd[0] = 'bash'\n"
            "subprocess.run(cmd, check=False)\n"
        ),
    ],
)
def test_mixed_or_mutated_subprocess_bindings_fail_closed(tmp_path: Path, helper_source: str):
    repo, sha = _authority_fixture(tmp_path, helper_source=helper_source)
    with pytest.raises(gen.AuthorityClosureError, match="dynamic subprocess command"):
        _manifest(repo, sha)


def test_missing_shell_subprocess_helper_has_closure_error(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import subprocess\nsubprocess.run('python scripts/missing.py', shell=True)\n"
        ),
    )
    with pytest.raises(gen.AuthorityClosureError, match="literal shell helper is unavailable"):
        _manifest(repo, sha)


def test_standalone_classifier_extracts_and_calls_exact_ref_canonical_review_queue_policy_under_I_S(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, sha = _authority_fixture(tmp_path)
    hostile = tmp_path / "hostile"
    _write_text(
        hostile / "aragora/cli/commands/review_queue.py",
        "raise RuntimeError('ambient policy loaded')\n",
    )
    monkeypatch.setenv("PYTHONPATH", str(hostile))
    monkeypatch.setenv("PYTHONHOME", str(hostile))
    result = gen.classify_exact_ref_path(repo, sha, FIXTURE_ROOT)
    assert result["tier"] == 4
    assert result["matched_rule"] == FIXTURE_ROOT
    assert result["merge_train_matched_rule"] == FIXTURE_ROOT


def test_standalone_classifier_rejects_unavailable_or_ambient_policy(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path, missing_policy=True)
    with pytest.raises(gen.AuthorityClosureError, match="policy is unavailable"):
        _manifest(repo, sha)


@pytest.mark.parametrize(
    ("fixture_options", "message"),
    [
        ({"namespace_package": True}, "namespace-blended or noncanonical package"),
        ({"symlink_policy": True}, "policy cannot be a symlink"),
    ],
)
def test_standalone_classifier_rejects_namespace_blended_or_copied_policy(
    tmp_path: Path, fixture_options: dict[str, bool], message: str
):
    repo, sha = _authority_fixture(tmp_path, **fixture_options)
    with pytest.raises(gen.AuthorityClosureError, match=message):
        _manifest(repo, sha)


def test_all_loaded_repository_modules_are_under_exact_ref_extraction_root(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    loaded = _manifest(repo, sha)["policy"]["loaded_repository_modules"]
    assert any(entry["path"] == gen.POLICY_PATH for entry in loaded)
    assert all(not Path(entry["path"]).is_absolute() for entry in loaded)
    assert all(".." not in Path(entry["path"]).parts for entry in loaded)


def test_top_level_module_shadow_loaded_from_exact_ref_fails_closed(tmp_path: Path):
    repo, _sha = _authority_fixture(tmp_path, policy_prelude="import subprocess\n")
    _write_text(repo / "subprocess.py", "SHADOWED = True\n")
    subprocess.run(["git", "add", "subprocess.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "shadow"],
        cwd=repo,
        check=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(
        gen.AuthorityClosureError,
        match=r"authority closure members below Tier 4: subprocess\.py",
    ):
        _manifest(repo, sha)


def test_classifier_runtime_import_joins_and_recursively_closes_authority(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        policy_epilogue=(
            "def _classify_model_review_tier(files, *, pr=None):\n"
            "    import aragora.runtime_helper\n"
            "    import scripts.transitive\n"
            "    if any(_matches_prefix(path, TIER_4_PREFIXES) for path in files):\n"
            "        return (4, 'tier_4_preapproval_required', 'fixture authority')\n"
            "    return (2, 'tier_2_live_automation', 'fixture non-authority')\n"
        ),
    )
    manifest = _manifest(repo, sha)
    files = {entry["path"]: entry for entry in manifest["repo_files"]}
    assert {
        "from": "aragora/cli/commands/review_queue.py",
        "kind": "classifier_runtime_import",
    } in files["aragora/runtime_helper.py"]["incoming_edges"]
    assert any(
        entry["module"] == "scripts.transitive" and entry["path"] == "scripts/transitive.py"
        for entry in manifest["policy"]["loaded_repository_modules"]
    )


def test_exact_ref_classifier_timeout_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, sha = _authority_fixture(
        tmp_path,
        policy_prelude="import time\ntime.sleep(1)\n",
    )
    monkeypatch.setattr(gen, "EXACT_REF_POLICY_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(gen.AuthorityClosureError, match="classifier exceeded"):
        _manifest(repo, sha)


def test_classifier_and_merge_train_closure_match(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    manifest = _manifest(repo, sha)
    assert all(
        entry["matched_rule"] == entry["merge_train_matched_rule"]
        for entry in manifest["repo_files"]
    )
    hostile = gen.classify_exact_ref_path(repo, sha, f"{FIXTURE_ROOT}.bak")
    assert hostile["tier"] == 2
    assert hostile["matched_rule"] is None
    assert hostile["merge_train_matched_rule"] is None


def test_canonical_tier_cli_is_read_only_and_digest_bound(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    script = Path(gen.__file__).resolve()
    command = [
        sys.executable,
        "-B",
        str(script),
        "--repo-root",
        str(repo),
        "--classify-tier",
        "--changed-file",
        FIXTURE_ROOT,
        "--ref",
        sha,
        "--json",
    ]
    before_status = subprocess.run(
        ["git", "status", "--porcelain=v2", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    before_refs = subprocess.run(
        ["git", "show-ref"], cwd=repo, check=True, capture_output=True
    ).stdout
    before_filesystem = _filesystem_snapshot(repo)
    first = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True).stdout
    second = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True).stdout
    assert first == second
    payload = json.loads(first)
    assert payload["authority_manifest_sha256"]
    assert payload["ref"] == sha
    assert (
        subprocess.run(
            ["git", "status", "--porcelain=v2", "--untracked-files=all"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        == before_status
    )
    assert (
        subprocess.run(["git", "show-ref"], cwd=repo, check=True, capture_output=True).stdout
        == before_refs
    )
    assert _filesystem_snapshot(repo) == before_filesystem


def test_exact_ref_cli_is_read_only_against_bare_remote_repository(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True)
    before = _filesystem_snapshot(bare)
    manifest = gen.build_authority_manifest(bare, sha)
    after = _filesystem_snapshot(bare)
    assert manifest["ref"] == sha
    assert manifest["authority_manifest_sha256"]
    assert before == after


def test_external_authority_manifest_and_evidence_index_bytes_are_canonical_before_semantic_digest(
    tmp_path: Path,
):
    repo, sha = _authority_fixture(tmp_path)
    canonical = tmp_path / "canonical.json"
    canonical.write_bytes(gen._canonical_json_bytes({"schema": "fixture", "value": 1}))
    manifest = gen.build_authority_manifest(repo, sha, external_artifacts=(canonical,))
    assert manifest["inventory"]["external_artifacts"][0]["canonical_bytes"] is True

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps({"schema": "fixture"}, indent=2) + "\n")
    with pytest.raises(gen.AuthorityClosureError, match="compact JSON"):
        gen.build_authority_manifest(repo, sha, external_artifacts=(noncanonical,))


def test_external_artifact_binding_is_machine_path_independent(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    left = tmp_path / "left/canonical.json"
    right = tmp_path / "right/canonical.json"
    left.parent.mkdir()
    right.parent.mkdir()
    canonical = gen._canonical_json_bytes({"schema": "fixture", "value": 1})
    left.write_bytes(canonical)
    right.write_bytes(canonical)
    first = gen.build_authority_manifest(repo, sha, external_artifacts=(left,))
    second = gen.build_authority_manifest(repo, sha, external_artifacts=(right,))
    assert first["authority_manifest_sha256"] == second["authority_manifest_sha256"]
    assert first["inventory"]["external_artifacts"][0]["path"] == "canonical.json"


def test_exact_ref_modes_ignore_ambient_python_environment(tmp_path: Path, monkeypatch):
    repo, sha = _authority_fixture(tmp_path)
    user_base = tmp_path / "userbase"
    site = user_base / "lib/python/site-packages"
    site.mkdir(parents=True)
    (site / "hostile.pth").write_text(str(tmp_path / "missing") + os.linesep)
    monkeypatch.setenv("PYTHONUSERBASE", str(user_base))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "missing"))
    assert _manifest(repo, sha)["policy"]["module_path"] == gen.POLICY_PATH


@pytest.mark.parametrize(
    ("prelude", "message"),
    [
        (
            "import subprocess\nsubprocess.run(['git', 'status'], check=False)\n",
            "forbidden subprocess action",
        ),
        (
            "from pathlib import Path\nPath('/tmp/authority-escape').write_text('x')\n",
            "forbidden write action",
        ),
        (
            "from pathlib import Path\nPath('/etc/passwd').read_text()\n",
            "forbidden read outside extraction and standard-library roots",
        ),
        (
            "import socket\nsocket.create_connection(('127.0.0.1', 9))\n",
            "forbidden network action",
        ),
        (
            "import os\nos.truncate('/tmp/authority-escape', 0)\n",
            "forbidden filesystem mutation",
        ),
        (
            "import os\nos.mkdir('/tmp/authority-escape')\n",
            "forbidden filesystem mutation",
        ),
        (
            "import os\nos.symlink('/tmp', '/tmp/authority-escape')\n",
            "forbidden filesystem mutation",
        ),
        (
            "import os\nos.chmod('/tmp', 0o700)\n",
            "forbidden filesystem mutation",
        ),
        (
            "import sqlite3\nsqlite3.connect('/tmp/authority-escape.sqlite')\n",
            "forbidden filesystem mutation",
        ),
    ],
)
def test_exact_ref_policy_rejects_mutating_subprocess_and_filesystem_actions(
    tmp_path: Path, prelude: str, message: str
):
    repo, sha = _authority_fixture(tmp_path, policy_prelude=prelude)
    with pytest.raises(gen.AuthorityClosureError, match=message):
        _manifest(repo, sha)


@pytest.mark.parametrize("ref", ["HEAD", "main", "42bc9458", "A" * 40])
def test_exact_ref_modes_reject_symbolic_abbreviated_or_noncanonical_refs(tmp_path: Path, ref: str):
    repo, _sha = _authority_fixture(tmp_path)
    with pytest.raises(gen.AuthorityClosureError, match="full lowercase 40-hex"):
        gen.build_authority_manifest(repo, ref)


def test_exact_ref_mode_rejects_inventory_check_flag(tmp_path: Path):
    repo, sha = _authority_fixture(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(gen.__file__).resolve()),
            "--repo-root",
            str(repo),
            "--ref",
            sha,
            "--check",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "cannot be combined with --ref" in proc.stdout


# --- VAL-CDG-002: canonical census binds the ratified 655-record cohort -----

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_CATEGORIES = ("python_sdk_drift", "typescript_sdk_drift")
ROUTE_CATEGORIES = ("routes_missing_in_spec", "routes_orphaned_in_spec", "sdk_missing_from_both")
RUNTIME_METHODS = frozenset(
    {"CONNECT", "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"}
)
CATEGORY_LANGUAGES = {
    "python_sdk_drift": ["python"],
    "typescript_sdk_drift": ["typescript"],
    "routes_missing_in_spec": [],
    "routes_orphaned_in_spec": [],
    "sdk_missing_from_both": ["python", "typescript"],
}


@functools.cache
def _accepted_authority() -> dict:
    inventory = json.loads((REPO_ROOT / gen.DEFAULT_INVENTORY).read_bytes())
    authority = inventory["accepted_authority"]
    assert authority["schema"] == "contract-drift-accepted-authority-v1"
    return authority


def _cohort() -> dict:
    return _accepted_authority()["canonical_artifacts"]["original_cohort"]


def _provenance() -> dict:
    return _accepted_authority()["canonical_artifacts"]["sdk_provenance"]


def _record_digest(record: dict) -> str:
    return ratchet._sha256_bytes(
        ratchet._canonical_json_bytes(
            {key: value for key, value in record.items() if key != "record_sha256"}
        )
    )


def _rehash_projection(cohort: dict) -> None:
    projection = cohort["operation_projection"]
    for record in projection["records"]:
        record["record_sha256"] = _record_digest(record)
    projection["record_digest_set_sha256"] = ratchet._digest_set(
        "cdg-operation-projection-record-digest-set-v1",
        [record["record_sha256"] for record in projection["records"]],
        "record_sha256_values",
    )


def _original_id(category: str, literal: str) -> tuple[bytes, str]:
    payload = {
        "category": category,
        "exact_historical_literal_record": literal,
        "schema": "cdg-original-record-id-v1",
    }
    raw = ratchet._canonical_json_bytes(payload)
    return raw, f"cdg1:{ratchet._sha256_bytes(raw)}"


def _linked_provenance_mutation(mutate, *, index: int = 0) -> tuple[dict, dict]:
    """Mutate one provenance record while keeping its digest and cohort link
    self-consistent, so validation reaches the semantic check under test
    instead of failing at the digest-binding layer."""
    cohort = copy.deepcopy(_cohort())
    provenance = copy.deepcopy(_provenance())
    record = provenance["records"][index]
    mutate(record)
    record["record_sha256"] = _record_digest(record)
    linked = next(
        r
        for r in cohort["original_records"]
        if r["original_record_id"] == record["original_record_id"]
    )
    linked["sdk_provenance_record_sha256"] = record["record_sha256"]
    return cohort, provenance


def test_canonical_cohort_artifact_exact_length_sha_and_canonical_bytes(tmp_path: Path):
    raw = ratchet._canonical_json_bytes(_cohort(), terminal_lf=True)
    assert len(raw) == ratchet.COHORT_ARTIFACT["byte_length"] == 1_692_125
    digest = ratchet._sha256_bytes(raw)
    assert (
        digest
        == ratchet.COHORT_ARTIFACT["sha256"]
        == "565cd84a9a5d266f61b66bd7965e0a036e4817ef5fed32edb8c41a2dea6cc208"
    )
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n") and b"\n" not in raw[:-1]
    path = tmp_path / ratchet.COHORT_ARTIFACT["filename"]
    path.write_bytes(raw)
    parsed, descriptor, _ = ratchet._read_canonical_json_bytes(
        path,
        label="canonical original-cohort artifact",
        expected_byte_length=len(raw),
        expected_sha256=digest,
        terminal_lf=True,
    )
    assert parsed["schema"] == "contract-drift-original-cohort-v1"
    assert descriptor["canonical_bytes_valid"] is True
    # A parseable but differently serialized file fails even with matching pins.
    hostile = json.dumps(json.loads(raw), separators=(", ", ": ")).encode() + b"\n"
    hostile_path = tmp_path / "reserialized.json"
    hostile_path.write_bytes(hostile)
    with pytest.raises(ValueError, match="not canonical compact sorted-key JSON"):
        ratchet._read_canonical_json_bytes(
            hostile_path,
            label="canonical original-cohort artifact",
            expected_byte_length=len(hostile),
            expected_sha256=ratchet._sha256_bytes(hostile),
            terminal_lf=True,
        )


def test_canonical_sdk_provenance_artifact_exact_length_sha_and_canonical_bytes(tmp_path: Path):
    raw = ratchet._canonical_json_bytes(_provenance(), terminal_lf=True)
    assert len(raw) == ratchet.PROVENANCE_ARTIFACT["byte_length"] == 898_099
    digest = ratchet._sha256_bytes(raw)
    assert (
        digest
        == ratchet.PROVENANCE_ARTIFACT["sha256"]
        == "21ae1c30200cda6df51dbca7053bbbbde6241ab78a73347b0fe5e4d2ed79f07f"
    )
    path = tmp_path / ratchet.PROVENANCE_ARTIFACT["filename"]
    path.write_bytes(raw)
    parsed, _, _ = ratchet._read_canonical_json_bytes(
        path,
        label="canonical SDK-provenance artifact",
        expected_byte_length=len(raw),
        expected_sha256=digest,
        terminal_lf=True,
    )
    assert parsed["schema"] == "contract-drift-sdk-provenance-v1"
    # Duplicate keys are rejected before any use of the parsed payload.
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        json.loads('{"records": 1, "records": 2}', object_pairs_hook=ratchet._duplicate_key_object)
    truncated = raw[:-2] + b"\n"
    truncated_path = tmp_path / "truncated.json"
    truncated_path.write_bytes(truncated)
    with pytest.raises(ValueError, match="byte-length mismatch"):
        ratchet._read_canonical_json_bytes(
            truncated_path,
            label="canonical SDK-provenance artifact",
            expected_byte_length=ratchet.PROVENANCE_ARTIFACT["byte_length"],
            expected_sha256=digest,
            terminal_lf=True,
        )


def test_all_655_original_ids_reproduce_from_anchored_source_blobs():
    cohort = _cohort()
    records = cohort["original_records"]
    assert len(records) == 655
    by_source: dict[str, list[str]] = {}
    for record in records:
        raw, original_id = _original_id(
            record["category"], record["exact_historical_literal_record"]
        )
        assert record["id_payload_byte_length"] == len(raw)
        assert record["id_payload_sha256"] == ratchet._sha256_bytes(raw)
        assert record["original_record_id"] == original_id
        by_source.setdefault(record["source_json_key"], []).append(
            record["exact_historical_literal_record"]
        )
    assert sorted(by_source) == sorted(
        [
            "python_sdk_drift",
            "typescript_sdk_drift",
            "missing_in_spec",
            "orphaned_in_spec",
            "missing_from_both_sdks",
        ]  # fmt: skip
    )
    # Anchored source binding: every membership source pins blob bytes; when the
    # anchor blobs are reachable in this clone, the cohort reproduces from them.
    for source in cohort["membership_sources"]:
        assert ratchet.SHA256_RE.fullmatch(source["sha256"])
        assert source["commit_sha"] == cohort["membership_anchor"]["commit_sha"]
        probe = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "blob", source["git_blob_oid"]],
            capture_output=True,
        )
        if probe.returncode != 0:
            continue
        blob = probe.stdout
        assert len(blob) == source["byte_length"]
        assert ratchet._sha256_bytes(blob) == source["sha256"]
        arrays = json.loads(blob)
        for key, literals in by_source.items():
            if key in arrays:
                assert sorted(arrays[key]) == sorted(literals), key
        if source["path"].endswith("verify_sdk_contracts.json"):
            # Unrelated arrays such as missing_stable never join the cohort.
            assert "missing_stable" in arrays
            assert "missing_stable" not in by_source


def test_ratified_original_id_set_digest_supersedes_provisional_digest():
    cohort = _cohort()
    original_ids = sorted(record["original_record_id"] for record in cohort["original_records"])
    ratified = ratchet._digest_set(
        "cdg-original-record-id-set-v1", original_ids, "original_record_ids"
    )
    assert (
        ratified
        == ratchet.ORIGINAL_ID_SET_SHA256
        == "c1235670c183b1887ba3fe4280fa0320f9fd6f4a85b8f346d4332ac2aebbe269"
    )
    assert cohort["original_record_id_set"]["sha256"] == ratified
    # Any other digest value — e.g. a stale provisional pin — is a hard failure
    # even when it is internally consistent with the declared ID list.
    hostile = copy.deepcopy(cohort)
    provisional = ratchet._digest_set(
        "cdg-original-record-id-set-v0", original_ids, "original_record_ids"
    )
    assert provisional != ratified
    hostile["original_record_id_set"]["sha256"] = provisional
    with pytest.raises(ValueError, match="ID-set digest mismatch"):
        ratchet._validate_original_cohort(hostile)


def test_sdk_598_records_remain_exact_method_bearing_literals():
    records = [r for r in _cohort()["original_records"] if r["category"] in SDK_CATEGORIES]
    assert len(records) == 598
    assert sum(1 for r in records if r["category"] == "python_sdk_drift") == 74
    assert sum(1 for r in records if r["category"] == "typescript_sdk_drift") == 524
    for record in records:
        method = record["method"]
        assert isinstance(method, str) and method in RUNTIME_METHODS
        literal = record["exact_historical_literal_record"]
        token, _, path = literal.partition(" ")
        assert token == method and path.startswith("/"), literal
    hostile = copy.deepcopy(_cohort())
    sdk_record = next(r for r in hostile["original_records"] if r["category"] == "python_sdk_drift")
    sdk_record["method"] = None
    with pytest.raises(ValueError, match="lacks a method"):
        ratchet._validate_original_cohort(hostile)


def test_route_parity_57_records_remain_exact_path_literals_with_null_method():
    records = [r for r in _cohort()["original_records"] if r["category"] in ROUTE_CATEGORIES]
    assert len(records) == 57
    for record in records:
        assert record["method"] is None
        assert record["exact_historical_literal_record"].startswith("/")
    counts = {c: sum(1 for r in records if r["category"] == c) for c in ROUTE_CATEGORIES}
    assert counts == {
        "routes_missing_in_spec": 11,
        "routes_orphaned_in_spec": 17,
        "sdk_missing_from_both": 29,
    }
    # No other original may carry a null method, and a path-level original may
    # not grow one.
    hostile = copy.deepcopy(_cohort())
    route_record = next(
        r for r in hostile["original_records"] if r["category"] == "routes_missing_in_spec"
    )
    route_record["method"] = "GET"
    with pytest.raises(ValueError, match="carries a method"):
        ratchet._validate_original_cohort(hostile)


def test_null_method_is_forbidden_on_runtime_and_projection_edges():
    projection = _cohort()["operation_projection"]
    for record in projection["records"]:
        for edge in record["operation_edges"]:
            assert edge["method"] in RUNTIME_METHODS
            assert edge["normalized_path"].startswith("/")
            assert edge["evidence"]
    hostile = copy.deepcopy(_cohort())
    edge = hostile["operation_projection"]["records"][0]["operation_edges"][0]
    for placeholder in (None, "", "get", "ANY", "*"):
        edge["method"] = placeholder
        _rehash_projection(hostile)
        with pytest.raises(ValueError, match="invalid method"):
            ratchet._validate_original_cohort(hostile)


def test_sdk_provenance_reconstructs_from_pinned_birth_extractor_normalizer_and_source_closure():
    provenance = _provenance()
    birth = provenance["baseline_birth"]
    for field in (
        "commit_sha",
        "first_parent_sha",
        "commit_tree_git_oid",
        "membership_anchor_baseline_blob_git_oid",
        "parent_baseline_blob_git_oid",
        "ratified_baseline_blob_git_oid",
    ):
        assert isinstance(birth[field], str) and birth[field], field
    # The ratified birth blob is the membership-anchor baseline blob.
    assert birth["ratified_baseline_blob_git_oid"] == birth["membership_anchor_baseline_blob_git_oid"]  # fmt: skip
    dependencies = provenance["dependencies"]
    assert sorted(dependencies) == sorted(
        [
            "baseline_source",
            "normalizer",
            "openapi_sources",
            "python_namespace_tree",
            "typescript_namespace_tree",
            "verifier",
        ]
    )
    for record in provenance["records"]:
        assert record["record_sha256"] == _record_digest(record)
        atoms = record["provenance_atoms"]
        assert sorted({o["provenance_atom"] for o in record["source_occurrences"]}) == sorted(
            set(atoms)
        )
        for occurrence in record["source_occurrences"]:
            match_text = occurrence["match_text"]
            assert occurrence["match_text_sha256"] == ratchet._sha256_bytes(match_text.encode())
            assert occurrence["match_end_byte"] - occurrence["match_start_byte"] == len(
                match_text.encode()
            )
            # The provenance atom is the historical namespace filename stem.
            assert occurrence["provenance_atom"] == Path(occurrence["source_path"]).stem
            assert occurrence["source_commit_sha"] == birth["commit_sha"]


def test_sdk_provenance_has_598_records_690_occurrences_12_multi_atom_zero_missing():
    provenance = _provenance()
    records = provenance["records"]
    assert len(records) == 598
    occurrences = sum(len(record["source_occurrences"]) for record in records)
    multi_atom = sum(1 for record in records if len(set(record["provenance_atoms"])) > 1)
    assert (occurrences, multi_atom) == (690, 12)
    assert all(record["source_occurrences"] for record in records)  # zero missing
    summary = ratchet._validate_sdk_provenance(
        provenance, ratchet._validate_original_cohort(_cohort())
    )
    assert summary["record_count"] == 598
    assert summary["source_occurrence_count"] == 690
    assert summary["multiple_atom_record_count"] == 12
    assert summary["missing_provenance_count"] == 0
    cohort, hostile = _linked_provenance_mutation(
        lambda record: record.update(source_occurrences=[])
    )
    with pytest.raises(ValueError, match="lacks source occurrences"):
        ratchet._validate_sdk_provenance(hostile, ratchet._validate_original_cohort(cohort))


def test_sdk_provenance_per_record_links_and_digests_match_cohort():
    cohort_summary = ratchet._validate_original_cohort(_cohort())
    cohort_sdk = cohort_summary["sdk_records"]
    provenance = _provenance()
    for record in provenance["records"]:
        cohort_record = cohort_sdk[record["original_record_id"]]
        for field in (
            "category",
            "exact_historical_literal_record",
            "id_payload_byte_length",
            "id_payload_sha256",
            "source_array_index",
        ):
            assert record[field] == cohort_record[field], field
        assert cohort_record["sdk_language"] == [record["sdk_language"]]
        assert cohort_record["sdk_provenance_record_sha256"] == record["record_sha256"]
    hostile = copy.deepcopy(provenance)
    hostile["records"][0]["source_array_index"] += 1
    hostile["records"][0]["record_sha256"] = _record_digest(hostile["records"][0])
    with pytest.raises(ValueError, match="source_array_index link mismatch"):
        ratchet._validate_sdk_provenance(hostile, ratchet._validate_original_cohort(_cohort()))
    stale = copy.deepcopy(provenance)
    stale["records"][1]["record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        ratchet._validate_sdk_provenance(stale, ratchet._validate_original_cohort(_cohort()))


def test_sdk_partition_has_exact_75_core_523_extended_and_pinned_digests():
    provenance = _provenance()
    partitions: dict[str, list[str]] = {"core": [], "extended": []}
    for record in provenance["records"]:
        partition, _ = ratchet._partition_from_atoms(record["provenance_atoms"])
        assert partition == record["partition"]
        partitions[partition].append(record["original_record_id"])
    assert (len(partitions["core"]), len(partitions["extended"])) == (75, 523)
    descriptor = provenance["partition"]
    assert descriptor["intersection_count"] == 0 and descriptor["union_count"] == 598
    expectations = {
        "core_original_record_id_set_sha256": (
            "cdg-core-original-record-id-set-v1",
            partitions["core"],
            ratchet.CORE_ID_SET_SHA256,
        ),
        "extended_original_record_id_set_sha256": (
            "cdg-extended-original-record-id-set-v1",
            partitions["extended"],
            ratchet.EXTENDED_ID_SET_SHA256,
        ),
        "sdk_original_record_id_set_sha256": (
            "cdg-sdk-original-record-id-set-v1",
            partitions["core"] + partitions["extended"],
            ratchet.SDK_ID_SET_SHA256,
        ),
    }
    for field, (schema, ids, pinned) in expectations.items():
        recomputed = ratchet._digest_set(schema, sorted(ids), "original_record_ids")
        assert descriptor[field] == recomputed == pinned, field


def test_original_descriptor_and_provenance_are_identical_across_authority_transitions():
    authority = _accepted_authority()
    result = ratchet.compare_accepted_authorities(
        authority, copy.deepcopy(authority), repo_root=REPO_ROOT
    )
    assert result["status"] == "pass"
    assert result["added_original_record_ids"] == []
    assert result["removed_original_record_ids"] == []
    # A transition that touches one ratified literal is rejected outright.
    hostile = copy.deepcopy(authority)
    record = hostile["canonical_artifacts"]["original_cohort"]["original_records"][0]
    record["exact_historical_literal_record"] += "/renamed"
    with pytest.raises(ValueError):
        ratchet.compare_accepted_authorities(authority, hostile, repo_root=REPO_ROOT)
    # Swapping the analyzer bundle is an authority change, not a transition.
    rebundled = copy.deepcopy(authority)
    rebundled["analyzer_bundle"]["files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        ratchet.compare_accepted_authorities(authority, rebundled, repo_root=REPO_ROOT)


def test_operation_projection_has_one_membership_per_655_originals():
    cohort = _cohort()
    projection_ids = [
        record["original_record_id"] for record in cohort["operation_projection"]["records"]
    ]
    original_ids = [record["original_record_id"] for record in cohort["original_records"]]
    assert len(projection_ids) == len(original_ids) == 655
    assert len(set(projection_ids)) == 655
    assert sorted(projection_ids) == sorted(original_ids)
    hostile = copy.deepcopy(cohort)
    duplicated = hostile["operation_projection"]["records"]
    duplicated[1] = copy.deepcopy(duplicated[0])
    _rehash_projection(hostile)
    with pytest.raises(ValueError, match="does not biject"):
        ratchet._validate_original_cohort(hostile)


def test_operation_projection_has_666_edges_and_nine_multi_edge_originals_max_four():
    cohort = _cohort()
    sizes = [len(record["operation_edges"]) for record in cohort["operation_projection"]["records"]]
    assert sum(sizes) == 666
    assert sum(1 for size in sizes if size > 1) == 9
    assert max(sizes) == 4
    distribution = {size: sizes.count(size) for size in sorted(set(sizes))}
    assert distribution == {1: 646, 2: 8, 4: 1}
    by_id = {record["original_record_id"]: record for record in cohort["original_records"]}
    sdk_sizes = [
        len(record["operation_edges"])
        for record in cohort["operation_projection"]["records"]
        if by_id[record["original_record_id"]]["category"] in SDK_CATEGORIES
    ]
    assert sdk_sizes.count(1) == len(sdk_sizes) == 598
    route_sizes = sorted(
        len(record["operation_edges"])
        for record in cohort["operation_projection"]["records"]
        if by_id[record["original_record_id"]]["category"] in ROUTE_CATEGORIES
    )
    assert route_sizes == [1] * 48 + [2] * 8 + [4]
    assert sum(route_sizes) == 68


def test_every_witnessed_method_specific_edge_is_preserved():
    cohort = _cohort()
    for record in cohort["operation_projection"]["records"]:
        edges = {(edge["method"], edge["normalized_path"]) for edge in record["operation_edges"]}
        assert len(edges) == len(record["operation_edges"])  # distinct witnessed set
        for edge in record["operation_edges"]:
            assert edge["evidence"], edge["normalized_path"]
    # Omitting one witnessed method-specific edge from the four-edge membership
    # is caught by the pinned record-digest set even after rehashing.
    hostile = copy.deepcopy(cohort)
    victim = next(
        record
        for record in hostile["operation_projection"]["records"]
        if len(record["operation_edges"]) == 4
    )
    victim["operation_edges"].pop()
    _rehash_projection(hostile)
    with pytest.raises(ValueError, match="record-digest-set mismatch"):
        ratchet._validate_original_cohort(hostile)
    # Cross-method collapse (rewriting an edge onto a sibling method) fails too.
    collapsed = copy.deepcopy(cohort)
    twin = next(
        record
        for record in collapsed["operation_projection"]["records"]
        if len(record["operation_edges"]) == 2
    )
    twin["operation_edges"][1]["method"] = twin["operation_edges"][0]["method"]
    twin["operation_edges"][1]["normalized_path"] = twin["operation_edges"][0]["normalized_path"]
    _rehash_projection(collapsed)
    with pytest.raises(ValueError, match="duplicate edges"):
        ratchet._validate_original_cohort(collapsed)


def test_edge_count_never_substitutes_for_original_count():
    cohort = _cohort()
    edge_count = sum(
        len(record["operation_edges"]) for record in cohort["operation_projection"]["records"]
    )
    assert edge_count == 666 != 655
    assert cohort["counts"]["records"] == 655
    assert len(cohort["original_records"]) == 655
    hostile = copy.deepcopy(cohort)
    hostile["counts"]["records"] = 666
    with pytest.raises(ValueError, match="declared counts mismatch"):
        ratchet._validate_original_cohort(hostile)
    # Replacing the cohort list with one record per edge (666 rows) fails the
    # census outright: membership count is the original count, never edges.
    fanout = copy.deepcopy(cohort)
    fanout["original_records"].extend(copy.deepcopy(fanout["original_records"][:11]))
    with pytest.raises(ValueError, match="exactly 655 original records"):
        ratchet._validate_original_cohort(fanout)


def test_exactly_one_edge_assumption_fails(monkeypatch):
    cohort = copy.deepcopy(_cohort())
    for record in cohort["operation_projection"]["records"]:
        del record["operation_edges"][1:]
    _rehash_projection(cohort)
    # The flattened projection cannot pass against the ratified digest pin...
    with pytest.raises(ValueError, match="record-digest-set mismatch"):
        ratchet._validate_original_cohort(cohort)
    # ...and even a re-pinned digest cannot satisfy the (666, 9, 4) invariant:
    # exactly-one-edge assumptions are structurally rejected.
    monkeypatch.setattr(
        ratchet,
        "PROJECTION_RECORD_SET_SHA256",
        cohort["operation_projection"]["record_digest_set_sha256"],
    )
    with pytest.raises(ValueError, match="cardinality mismatch"):
        ratchet._validate_original_cohort(cohort)


def test_projection_revision_cannot_change_original_id_sets():
    cohort = _cohort()
    hostile = copy.deepcopy(cohort)
    projection = hostile["operation_projection"]
    victim = next(r for r in projection["records"] if len(r["operation_edges"]) == 4)
    projection["records"].remove(victim)
    for index, edge in enumerate(victim["operation_edges"]):
        replacement = {
            "operation_edges": [edge],
            "original_record_id": f"{victim['original_record_id']}:{index}",
        }
        projection["records"].append(replacement)
    _rehash_projection(hostile)
    with pytest.raises(ValueError, match="must contain 655 membership records"):
        ratchet._validate_original_cohort(hostile)
    # Same membership count but a swapped original ID also fails the bijection.
    swapped = copy.deepcopy(cohort)
    _, foreign_id = _original_id("python_sdk_drift", "GET /api/never-witnessed")
    swapped["operation_projection"]["records"][0]["original_record_id"] = foreign_id
    _rehash_projection(swapped)
    with pytest.raises(ValueError, match="does not biject"):
        ratchet._validate_original_cohort(swapped)


def test_sdk_language_exact_mapping_is_provenance_not_identity():
    cohort = _cohort()
    for record in cohort["original_records"]:
        assert record["sdk_language"] == CATEGORY_LANGUAGES[record["category"]]
        # Identity hashes only category + exact literal + schema: language is
        # provenance metadata and never enters the ID payload.
        raw, original_id = _original_id(
            record["category"], record["exact_historical_literal_record"]
        )
        assert record["original_record_id"] == original_id
        assert b"sdk_language" not in raw
    for record in _provenance()["records"]:
        assert [record["sdk_language"]] == CATEGORY_LANGUAGES[record["category"]]
        for occurrence in record["source_occurrences"]:
            assert occurrence["sdk_language"] == record["sdk_language"]


def test_language_metadata_change_cannot_change_original_record_id():
    cohort = _cohort()
    record = next(r for r in cohort["original_records"] if r["category"] == "python_sdk_drift")
    mutated = dict(record, sdk_language=["typescript"])
    _, mutated_id = _original_id(mutated["category"], mutated["exact_historical_literal_record"])
    assert mutated_id == record["original_record_id"]  # identity is unchanged
    # But the census rejects the inconsistent language mapping rather than
    # minting a new identity for it.
    hostile = copy.deepcopy(_provenance())
    victim = hostile["records"][0]
    victim["sdk_language"] = "typescript" if victim["category"] == "python_sdk_drift" else "python"
    victim["record_sha256"] = _record_digest(victim)
    with pytest.raises(ValueError, match="language link mismatch"):
        ratchet._validate_sdk_provenance(hostile, ratchet._validate_original_cohort(_cohort()))


def test_sdk_partition_is_disjoint_exhaustive_and_category_preserving():
    provenance = _provenance()
    core = {r["original_record_id"] for r in provenance["records"] if r["partition"] == "core"}
    extended = {
        r["original_record_id"] for r in provenance["records"] if r["partition"] == "extended"
    }
    assert not core & extended
    sdk_ids = {
        record["original_record_id"]
        for record in _cohort()["original_records"]
        if record["category"] in SDK_CATEGORIES
    }
    assert core | extended == sdk_ids
    by_category = {"python_sdk_drift": 0, "typescript_sdk_drift": 0}
    for record in provenance["records"]:
        by_category[record["category"]] += 1
    assert by_category == {"python_sdk_drift": 74, "typescript_sdk_drift": 524}
    assert provenance["counts"]["python_sdk_drift"] == 74
    assert provenance["counts"]["typescript_sdk_drift"] == 524
    cohort, hostile = _linked_provenance_mutation(
        lambda record: record.update(
            partition="extended" if record["partition"] == "core" else "core"
        )
    )
    with pytest.raises(ValueError, match="partition mismatch"):
        ratchet._validate_sdk_provenance(hostile, ratchet._validate_original_cohort(cohort))


def test_sdk_partition_exact_hyphen_and_single_plural_whole_atom_rules():
    partition, matches = ratchet._partition_from_atoms(["debate"])
    assert partition == "core"
    assert matches == [
        {
            "atom": "debate",
            "domain": "debate",
            "match_rule": "exact",
            "normalized_atom": "debate",
        }
    ]
    # Hyphens normalize to underscores before comparison.
    assert ratchet._partition_from_atoms(["de-bate"]) == ("extended", [])
    # Exactly one trailing s maps a plural onto its core domain.
    plural_partition, plural_matches = ratchet._partition_from_atoms(["rankings"])
    assert plural_partition == "core"
    assert plural_matches[0]["match_rule"] == "remove_exactly_one_trailing_s"
    assert plural_matches[0]["domain"] == "ranking"
    # A double plural is not reduced twice.
    assert ratchet._partition_from_atoms(["rankingss"]) == ("extended", [])
    # 'agents' is itself a core domain: matched exactly, not via plural rule.
    agents_partition, agents_matches = ratchet._partition_from_atoms(["agents"])
    assert agents_partition == "core" and agents_matches[0]["match_rule"] == "exact"
    # Mixed arrays with at least one matching whole atom are core.
    mixed_partition, mixed_matches = ratchet._partition_from_atoms(["billing", "memory"])
    assert mixed_partition == "core"
    assert [match["atom"] for match in mixed_matches] == ["memory"]


def test_sdk_partition_rejects_substrings_free_text_paths_and_malformed_atoms():
    for hostile_atom in (
        "debate_utils",  # embedded domain word
        "predebate",  # substring on the left
        "the memory subsystem is great",  # free text
        "aragora/memory/handler.py",  # path fragment
        "memory.py",  # filename, not a whole atom
    ):
        partition, matches = ratchet._partition_from_atoms([hostile_atom])
        assert (partition, matches) == ("extended", []), hostile_atom
    for malformed in ([], [""], [123], [None], ["memory", ""]):
        with pytest.raises(ValueError, match="nonempty string array"):
            ratchet._partition_from_atoms(malformed)


def test_sdk_partition_missing_or_empty_provenance_fails_closed():
    def _drop_atoms(record: dict) -> None:
        del record["provenance_atoms"]

    cohort, hostile = _linked_provenance_mutation(_drop_atoms)
    with pytest.raises(ValueError, match="lacks provenance atoms"):
        ratchet._validate_sdk_provenance(hostile, ratchet._validate_original_cohort(cohort))
    cohort, empty = _linked_provenance_mutation(lambda record: record.update(provenance_atoms=[]))
    with pytest.raises(ValueError, match="nonempty string array"):
        ratchet._validate_sdk_provenance(empty, ratchet._validate_original_cohort(cohort))


def test_sdk_partition_is_scheduling_metadata_only():
    provenance = _provenance()
    record = provenance["records"][0]
    # Partition never enters the identity payload...
    raw, original_id = _original_id(record["category"], record["exact_historical_literal_record"])
    assert b"partition" not in raw
    assert record["original_record_id"] == original_id
    # ...and never appears on active-inventory rows, which carry only identity,
    # category, status, and disposition history.
    for row in _accepted_authority()["active_inventory"]:
        assert set(row) == {"category", "disposition_history", "original_record_id", "status"}
    # Moving a record between partitions does not change its original ID.
    moved = dict(record, partition="extended" if record["partition"] == "core" else "core")
    _, moved_id = _original_id(moved["category"], moved["exact_historical_literal_record"])
    assert moved_id == record["original_record_id"]


def test_annotation_or_later_move_cannot_change_partition(monkeypatch):
    provenance = _provenance()
    # The partition is a pure function of the provenance atoms: annotating a
    # record (or "moving" it later) without changing atoms cannot change it.
    for record in provenance["records"][:25]:
        recomputed, matches = ratchet._partition_from_atoms(record["provenance_atoms"])
        assert recomputed == record["partition"]
        assert matches == record["matched_domains"]
    core_index = next(
        index for index, record in enumerate(provenance["records"]) if record["partition"] == "core"
    )
    cohort, hostile = _linked_provenance_mutation(
        lambda record: record.update(matched_domains=[]), index=core_index
    )
    with pytest.raises(ValueError, match="matched-domain proof mismatch"):
        ratchet._validate_sdk_provenance(hostile, ratchet._validate_original_cohort(cohort))
    # A wholesale rule swap (annotation layer redefining the grammar) is not
    # honored either: the validator recomputes from the pinned rule.
    monkeypatch.setattr(ratchet, "CORE_DOMAINS", frozenset({"billing"}))
    with pytest.raises(ValueError, match="partition mismatch"):
        ratchet._validate_sdk_provenance(
            copy.deepcopy(provenance), ratchet._validate_original_cohort(_cohort())
        )
