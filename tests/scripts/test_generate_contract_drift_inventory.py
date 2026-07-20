"""Tests for scripts/generate_contract_drift_inventory.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
    _write_text(repo / "aragora/runtime_helper.py", "# runtime policy helper\n")
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
        "mid_token": "      - run: python scripts/$HELPER.py\n",
        "glob": "      - run: python ./*.py\n",
        "module": "      - run: python -m scripts.helper\n",
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
    assert ".github/workflows/nested.yaml" in files
    assert any(
        edge["kind"] == "workflow_path_filter"
        for edge in files[".github/workflows/authority.yml"]["incoming_edges"]
    )
    assert "scripts/helper.py" in files
    assert "scripts/helper.sh" in files
    assert "scripts/transitive.py" in files
    assert "aragora/helper.py" in files


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


@pytest.mark.parametrize("variant", ["leading", "mid_token", "glob", "github_dynamic"])
def test_dynamic_local_run_reference_fails_closed(tmp_path: Path, variant: str):
    repo, sha = _authority_fixture(tmp_path, dynamic_run_reference=variant)
    with pytest.raises(gen.AuthorityClosureError, match="dynamic local run target"):
        _manifest(repo, sha)


@pytest.mark.parametrize("variant", ["module", "module_substitution"])
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


def test_unknown_path_root_does_not_become_repository_root(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        helper_source=(
            "import subprocess\n"
            "from pathlib import Path\n"
            "some_tmp_dir = Path('/tmp')\n"
            "subprocess.run([some_tmp_dir / 'scripts/missing.py'], check=False)\n"
        ),
    )
    assert _manifest(repo, sha)["ref"] == sha


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
        "def discover():\n    import aragora.helper\n",
        "class Discovery:\n    import aragora.helper\n",
    ],
)
def test_function_and_class_imports_join_closure(tmp_path: Path, helper_source: str):
    _write_text(tmp_path / "scripts/check_contract_drift_ratchet.py", helper_source)
    _write_text(tmp_path / "aragora/__init__.py", "")
    _write_text(tmp_path / "aragora/helper.py", "# helper\n")
    assert (
        "aragora/helper.py",
        "python_repository_import",
    ) in gen._python_import_edges(tmp_path, "scripts/check_contract_drift_ratchet.py")


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


def test_classifier_runtime_import_joins_and_recursively_closes_authority(tmp_path: Path):
    repo, sha = _authority_fixture(
        tmp_path,
        policy_epilogue=(
            "def _classify_model_review_tier(files, *, pr=None):\n"
            "    import aragora.runtime_helper\n"
            "    if any(_matches_prefix(path, TIER_4_PREFIXES) for path in files):\n"
            "        return (4, 'tier_4_preapproval_required', 'fixture authority')\n"
            "    return (2, 'tier_2_live_automation', 'fixture non-authority')\n"
        ),
    )
    files = {entry["path"]: entry for entry in _manifest(repo, sha)["repo_files"]}
    assert {
        "from": "aragora/cli/commands/review_queue.py",
        "kind": "classifier_runtime_import",
    } in files["aragora/runtime_helper.py"]["incoming_edges"]


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
