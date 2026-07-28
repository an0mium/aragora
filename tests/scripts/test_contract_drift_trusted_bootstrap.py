from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / ".github/workflows/contract_drift_trusted_bootstrap.py"
WORKFLOW_PATH = ROOT / ".github/workflows/contract-drift-trusted-bootstrap.yml"
LAUNCHER_PATH = ROOT / ".github/workflows/contract_drift_trusted_launcher.py"
MANIFEST_PATH = ROOT / ".github/workflows/contract-drift-trusted-bootstrap-manifest.json"
SPEC = importlib.util.spec_from_file_location("contract_drift_trusted_bootstrap", MODULE_PATH)
assert SPEC and SPEC.loader
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _write(repo: Path, relative: str, data: bytes | str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


def _canonical(value: Any) -> bytes:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (rendered + "\n").encode()


def _build_fixture(tmp_path: Path, mutation: str = "") -> SimpleNamespace:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Bootstrap Test")
    _write(repo, bootstrap.BASELINE_PATH, _canonical({"records": [], "schema": "test"}))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "genesis")
    genesis = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-qb", "trusted-base")
    _write(repo, bootstrap.BOOTSTRAP_PATH, MODULE_PATH.read_bytes())
    _write(repo, bootstrap.WORKFLOW_PATH, WORKFLOW_PATH.read_bytes())
    _write(repo, bootstrap.LAUNCHER_PATH, LAUNCHER_PATH.read_bytes())
    _write(repo, bootstrap.MANIFEST_PATH, MANIFEST_PATH.read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add base-owned bootstrap")
    base = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-qb", "candidate", base)
    if mutation == "analyzer-rejected":
        checker = (
            "import json,os,sys\n"
            "print(json.dumps({'comparison':"
            "os.environ['CDG_FIRST_TRANSITION_COMPARISON_SHA256'],'status':'rejected'}))\n"
            "sys.exit(1)\n"
        )
    elif mutation == "analyzer-garbage":
        checker = "print('not-json')\n"
    else:
        checker = (
            "import json,os\n"
            "print(json.dumps({'authority_root':os.environ['CDG_AUTHORITY_ROOT'],"
            "'comparison':os.environ['CDG_FIRST_TRANSITION_COMPARISON_SHA256'],"
            "'launcher':os.environ['CDG_EXECUTED_LAUNCHER_SHA256'],'status':'pass'}))\n"
        )
    _write(repo, bootstrap.ANALYZER_FILES[0], checker)
    _write(repo, bootstrap.ANALYZER_FILES[1], "VALUE = 1\n")
    _write(repo, bootstrap.ANALYZER_FILES[2], _canonical({"schema": "test-program"}))
    original_digests = tuple(
        hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in bootstrap.ANALYZER_FILES
    )

    artifacts = {
        "original_cohort": {"records": [{"id": "cohort"}], "schema": "test-cohort"},
        "sdk_provenance": {"records": [{"id": "provenance"}], "schema": "test-provenance"},
    }
    bindings = tuple(
        bootstrap.ArtifactBinding(
            path=f"library/{artifact['schema']}.json",
            schema=artifact["schema"],
            sha256=hashlib.sha256(_canonical(artifact)).hexdigest(),
            byte_length=len(_canonical(artifact)),
        )
        for artifact in artifacts.values()
    )
    files = [
        {"path": path, "sha256": digest}
        for path, digest in zip(bootstrap.ANALYZER_FILES, original_digests, strict=True)
    ]
    authority: dict[str, Any] = {
        "active_inventory": [],
        "active_inventory_sha256": hashlib.sha256(b"[]").hexdigest(),
        "analyzer_bundle": {
            "dependencies": [],
            "files": files,
            "interpreter_flags": list(bootstrap.ANALYZER_FLAGS),
            "launcher_sha256": hashlib.sha256(bootstrap.HERMETIC_LAUNCHER).hexdigest(),
            "schema": "contract-drift-analyzer-bundle-v1",
        },
        "canonical_artifact_bindings": [
            {
                "byte_length": item.byte_length,
                "path": item.path,
                "sha256": item.sha256,
            }
            for item in bindings
        ],
        "canonical_artifacts": artifacts,
        "categories": list(bootstrap.ACCEPTED_CATEGORIES),
        "manifest_sha256": "1" * 64,
        "publication": {
            "authority": "future-immutable-github-release-capsule",
            "status": "pending-merge",
        },
        "schema": "contract-drift-accepted-authority-v1",
        "transition": {
            "accepted_transition_head": "bound-by-release-capsule",
            "base_sha": genesis,
            "historical_nonconforming": bootstrap.EXPECTED_HISTORICAL_NONCONFORMING,
            "kind": "authority_transition",
        },
    }
    authority["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in authority.items() if key != "manifest_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    if mutation == "workflow":
        _write(repo, bootstrap.WORKFLOW_PATH, "ref: refs/pull/1/merge\n")
    elif mutation == "trusted-launcher":
        _write(repo, bootstrap.BOOTSTRAP_PATH, "raise SystemExit('candidate')\n")
    elif mutation == "trusted-exec-launcher":
        _write(repo, bootstrap.LAUNCHER_PATH, "raise SystemExit('candidate launcher')\n")
    elif mutation == "delete-trusted-launcher":
        _git(repo, "rm", "-q", bootstrap.LAUNCHER_PATH)
    elif mutation == "delete-manifest":
        _git(repo, "rm", "-q", bootstrap.MANIFEST_PATH)
    elif mutation == "checker":
        _write(repo, bootstrap.ANALYZER_FILES[0], "raise SystemExit('substituted')\n")
        authority["analyzer_bundle"]["files"][0]["sha256"] = hashlib.sha256(
            (repo / bootstrap.ANALYZER_FILES[0]).read_bytes()
        ).hexdigest()
    elif mutation == "empty-bundle":
        authority["analyzer_bundle"]["files"] = []
    elif mutation == "partial-bundle":
        authority["analyzer_bundle"]["files"] = files[:2]
    elif mutation == "accepted-launcher":
        authority["analyzer_bundle"]["launcher_sha256"] = "2" * 64
    elif mutation == "wrong-digest":
        authority["analyzer_bundle"]["files"][0]["sha256"] = "3" * 64
    elif mutation == "wrong-transition-base":
        authority["transition"]["base_sha"] = base
    elif mutation == "authority-manifest":
        authority["manifest_sha256"] = "4" * 64
    elif mutation == "active-inventory-digest":
        authority["active_inventory_sha256"] = "5" * 64
    elif mutation == "historical-nonconforming":
        authority["transition"]["historical_nonconforming"] = []
    elif mutation == "self-signal":
        authority["comparison_signal"] = {
            "candidate_head_sha": "candidate-controlled",
            "status": "accepted",
        }
    if mutation not in {"authority-manifest", "self-signal"}:
        authority["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in authority.items() if key != "manifest_sha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    baseline = {"accepted_authority": authority, "records": [], "schema": "test"}
    _write(repo, bootstrap.BASELINE_PATH, _canonical(baseline))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", f"candidate {mutation or 'valid'}")
    head = _git(repo, "rev-parse", "HEAD")
    event_data = {
        "pull_request": {
            "base": {
                "ref": "develop" if mutation == "wrong-base-ref" else "main",
                "repo": {"full_name": "synaptent/aragora"},
                "sha": base,
            },
            "head": {
                "repo": {
                    "full_name": (
                        "attacker/aragora" if mutation == "fork-head" else "synaptent/aragora"
                    )
                },
                "sha": base if mutation == "wrong-ref" else head,
            },
        },
        "repository": {"full_name": "synaptent/aragora"},
    }
    event = tmp_path / "event.json"
    event.write_bytes(_canonical(event_data))
    policy = bootstrap.BootstrapPolicy(
        analyzer_digests=original_digests,
        analyzer_source_sha=head,
        artifact_bindings=bindings,
        expected_head_sha=head,
        transition_base_sha=genesis,
    )
    return SimpleNamespace(repo=repo, base=base, head=head, event=event, policy=policy)


def _run(fixture: SimpleNamespace, tmp_path: Path) -> dict[str, Any]:
    return bootstrap.run_bootstrap(
        repo=fixture.repo,
        event_path=fixture.event,
        base_sha=fixture.base,
        head_sha=fixture.head,
        output_dir=tmp_path / "output",
        policy=fixture.policy,
        executed_launcher=MODULE_PATH,
        run_provenance=bootstrap.RunProvenance(
            event_name="pull_request_target",
            repository="synaptent/aragora",
            run_attempt=1,
            run_id=12345,
            workflow_ref=(
                "synaptent/aragora/.github/workflows/"
                "contract-drift-trusted-bootstrap.yml@refs/heads/main"
            ),
        ),
    )


def test_workflow_uses_base_owned_python_without_merge_checkout():
    workflow = WORKFLOW_PATH.read_text()
    assert "pull_request_target:" in workflow
    assert 'python-version: "3.11"' in workflow
    assert "actions/checkout" not in workflow
    assert '"$BASE_SHA" "$HEAD_SHA" "$ANALYZER_SOURCE_SHA"' in workflow
    assert f"ANALYZER_SOURCE_SHA: {bootstrap.ANALYZER_SOURCE_SHA}" in workflow
    assert 'for path in "$BOOTSTRAP_PATH" "$LAUNCHER_PATH" "$MANIFEST_PATH"' in workflow
    assert 'show "$BASE_SHA:$path"' in workflow
    assert "python -I -S -B" in workflow
    assert "github.event.pull_request.merge_commit_sha" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "checks: write" not in workflow
    assert "statuses: write" not in workflow
    for path in bootstrap.TRUSTED_SURFACE_FILES + (bootstrap.MANIFEST_PATH,):
        assert f'      - "{path}"' in workflow


def test_workflow_emits_run_bound_comparison_and_admission_artifact():
    workflow = WORKFLOW_PATH.read_text()
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
    assert "contract-drift-trusted-bootstrap-${{ github.event.pull_request.head.sha }}" in workflow
    assert 'if-no-files-found: "error"' in workflow
    assert "GITHUB_RUN_ID" in workflow
    assert "GITHUB_RUN_ATTEMPT" in workflow
    assert "GITHUB_WORKFLOW_REF" in workflow


def test_base_manifest_closes_bootstrap_and_launcher_bytes():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["schema"] == "contract-drift-trusted-bootstrap-manifest-v1"
    bindings = {item["path"]: item for item in manifest["files"]}
    for path in (WORKFLOW_PATH, MODULE_PATH, LAUNCHER_PATH):
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        assert bindings[relative] == {
            "byte_length": len(raw),
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }


def test_intended_exact_pair_bootstrap_executes_after_comparison(tmp_path):
    fixture = _build_fixture(tmp_path)
    result = _run(fixture, tmp_path)
    assert result["status"] == "pass"
    assert result["candidate_head_sha"] == fixture.head
    assert result["analysis"]["status"] == "pass"
    assert result["analysis"]["comparison"] == result["comparison_signal_sha256"]
    assert result["analysis"]["launcher"] == hashlib.sha256(LAUNCHER_PATH.read_bytes()).hexdigest()
    assert Path(result["analysis"]["authority_root"]).name.startswith("cdg-bootstrap-")
    signal = json.loads((tmp_path / "output/first-transition-comparison.json").read_text())
    assert signal["payload"]["workflow_run"] == {
        "event_name": "pull_request_target",
        "repository": "synaptent/aragora",
        "run_attempt": 1,
        "run_id": 12345,
        "workflow_ref": (
            "synaptent/aragora/.github/workflows/"
            "contract-drift-trusted-bootstrap.yml@refs/heads/main"
        ),
    }
    analysis_path = tmp_path / "output/trusted-bootstrap-analysis.json"
    receipt = json.loads((tmp_path / "output/trusted-bootstrap-admission.json").read_text())
    assert receipt["payload"]["analysis_path"] == analysis_path.name
    assert (
        receipt["payload"]["analysis_sha256"]
        == hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "workflow",
        "trusted-launcher",
        "trusted-exec-launcher",
        "delete-trusted-launcher",
        "delete-manifest",
    ],
)
def test_candidate_cannot_replace_base_owned_bootstrap_surface(tmp_path, mutation):
    fixture = _build_fixture(tmp_path, mutation)
    with pytest.raises(bootstrap.BootstrapError, match="proposed tree|trusted surface"):
        _run(fixture, tmp_path)


def test_head_checker_substitution_fails_closed(tmp_path):
    fixture = _build_fixture(tmp_path, "checker")
    with pytest.raises(bootstrap.BootstrapError, match="analyzer digest"):
        _run(fixture, tmp_path)


@pytest.mark.parametrize("mutation", ["empty-bundle", "partial-bundle"])
def test_non_exact_analyzer_bundle_fails_closed(tmp_path, mutation):
    fixture = _build_fixture(tmp_path, mutation)
    with pytest.raises(bootstrap.BootstrapError, match="file set"):
        _run(fixture, tmp_path)


def test_accepted_launcher_substitution_fails_closed(tmp_path):
    fixture = _build_fixture(tmp_path, "accepted-launcher")
    with pytest.raises(bootstrap.BootstrapError, match="launcher"):
        _run(fixture, tmp_path)


def test_base_owned_analyzer_source_substitution_fails_closed(tmp_path):
    fixture = _build_fixture(tmp_path)
    _git(fixture.repo, "checkout", "-qb", "substituted-source", fixture.base)
    for path in bootstrap.ANALYZER_FILES:
        _write(fixture.repo, path, bootstrap._git_blob(fixture.repo, fixture.head, path))
    _write(fixture.repo, bootstrap.ANALYZER_FILES[0], "raise SystemExit('source substitute')\n")
    _git(fixture.repo, "add", ".")
    _git(fixture.repo, "commit", "-qm", "substituted analyzer source")
    source_sha = _git(fixture.repo, "rev-parse", "HEAD")
    fixture.policy = bootstrap.BootstrapPolicy(
        analyzer_digests=fixture.policy.analyzer_digests,
        analyzer_source_sha=source_sha,
        artifact_bindings=fixture.policy.artifact_bindings,
        expected_head_sha=fixture.head,
        transition_base_sha=fixture.policy.transition_base_sha,
    )
    with pytest.raises(bootstrap.BootstrapError, match="analyzer digest"):
        _run(fixture, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong-ref", "event base/head"),
        ("wrong-base-ref", "event base/head"),
        ("fork-head", "event base/head"),
        ("wrong-digest", "analyzer digest"),
        ("wrong-transition-base", "transition base"),
        ("authority-manifest", "manifest digest"),
        ("active-inventory-digest", "active inventory digest"),
        ("historical-nonconforming", "self-authorization"),
    ],
)
def test_wrong_ref_or_digest_fails_closed(tmp_path, mutation, message):
    fixture = _build_fixture(tmp_path, mutation)
    with pytest.raises(bootstrap.BootstrapError, match=message):
        _run(fixture, tmp_path)


def test_missing_comparison_evidence_blocks_analyzer(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path)
    missing = tmp_path / "missing-comparison.json"
    monkeypatch.setattr(bootstrap, "_write_comparison_signal", lambda *args, **kwargs: missing)
    with pytest.raises(bootstrap.BootstrapError, match="comparison signal"):
        _run(fixture, tmp_path)


def test_wrong_comparison_run_provenance_blocks_analyzer(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path)
    write_signal = bootstrap._write_comparison_signal

    def _write_wrong_run(*args, **kwargs):
        path = write_signal(*args, **kwargs)
        envelope = json.loads(path.read_text())
        envelope["payload"]["workflow_run"]["run_id"] = 54321
        envelope["signal_sha256"] = hashlib.sha256(_canonical(envelope["payload"])).hexdigest()
        path.write_bytes(_canonical(envelope))
        return path

    monkeypatch.setattr(bootstrap, "_write_comparison_signal", _write_wrong_run)
    with pytest.raises(bootstrap.BootstrapError, match="comparison signal"):
        _run(fixture, tmp_path)


@pytest.mark.parametrize(
    ("event_name", "workflow_ref"),
    [
        (
            "pull_request",
            "synaptent/aragora/.github/workflows/"
            "contract-drift-trusted-bootstrap.yml@refs/heads/main",
        ),
        (
            "pull_request_target",
            "synaptent/aragora/.github/workflows/"
            "contract-drift-trusted-bootstrap.yml@refs/pull/1/merge",
        ),
    ],
)
def test_non_target_run_provenance_fails_closed(tmp_path, event_name, workflow_ref):
    fixture = _build_fixture(tmp_path)
    with pytest.raises(bootstrap.BootstrapError, match="workflow run provenance"):
        bootstrap.run_bootstrap(
            repo=fixture.repo,
            event_path=fixture.event,
            base_sha=fixture.base,
            head_sha=fixture.head,
            output_dir=tmp_path / "output",
            policy=fixture.policy,
            executed_launcher=MODULE_PATH,
            run_provenance=bootstrap.RunProvenance(
                event_name=event_name,
                repository="synaptent/aragora",
                run_attempt=1,
                run_id=12345,
                workflow_ref=workflow_ref,
            ),
        )


def test_analyzer_rejection_writes_nonpassing_admission(tmp_path):
    fixture = _build_fixture(tmp_path, "analyzer-rejected")
    result = _run(fixture, tmp_path)
    assert result["status"] == "analyzer-rejected"
    receipt = json.loads((tmp_path / "output/trusted-bootstrap-admission.json").read_text())
    assert receipt["payload"]["status"] == "analyzer-rejected"


def test_analyzer_invalid_output_fails_closed(tmp_path):
    fixture = _build_fixture(tmp_path, "analyzer-garbage")
    with pytest.raises(bootstrap.BootstrapError, match="did not emit JSON"):
        _run(fixture, tmp_path)


def test_existing_base_authority_writes_not_first_receipt(tmp_path):
    fixture = _build_fixture(tmp_path)
    _git(fixture.repo, "checkout", "-qb", "after-authority", fixture.head)
    _write(fixture.repo, "unrelated.txt", "next candidate\n")
    _git(fixture.repo, "add", ".")
    _git(fixture.repo, "commit", "-qm", "candidate after accepted authority")
    next_head = _git(fixture.repo, "rev-parse", "HEAD")
    event_data = json.loads(fixture.event.read_text())
    event_data["pull_request"]["base"]["sha"] = fixture.head
    event_data["pull_request"]["head"]["sha"] = next_head
    fixture.event.write_bytes(_canonical(event_data))
    fixture.base = fixture.head
    fixture.head = next_head
    fixture.policy = bootstrap.BootstrapPolicy(
        analyzer_digests=fixture.policy.analyzer_digests,
        analyzer_source_sha=fixture.policy.analyzer_source_sha,
        artifact_bindings=fixture.policy.artifact_bindings,
        expected_head_sha=next_head,
        transition_base_sha=fixture.policy.transition_base_sha,
    )
    result = _run(fixture, tmp_path)
    assert result["status"] == "not-first-transition"
    receipt = json.loads((tmp_path / "output/trusted-bootstrap-admission.json").read_text())
    assert receipt["payload"]["status"] == "not-first-transition"


def test_no_authority_candidate_writes_neutral_receipt(tmp_path):
    fixture = _build_fixture(tmp_path)
    _git(fixture.repo, "checkout", "-qb", "neutral-candidate", fixture.base)
    _write(fixture.repo, "unrelated.txt", "routine maintenance\n")
    _git(fixture.repo, "add", ".")
    _git(fixture.repo, "commit", "-qm", "routine candidate")
    neutral_head = _git(fixture.repo, "rev-parse", "HEAD")
    event_data = json.loads(fixture.event.read_text())
    event_data["pull_request"]["head"]["sha"] = neutral_head
    fixture.event.write_bytes(_canonical(event_data))
    fixture.head = neutral_head
    fixture.policy = bootstrap.BootstrapPolicy(
        analyzer_digests=fixture.policy.analyzer_digests,
        analyzer_source_sha=fixture.policy.analyzer_source_sha,
        artifact_bindings=fixture.policy.artifact_bindings,
        expected_head_sha=neutral_head,
        transition_base_sha=fixture.policy.transition_base_sha,
    )
    result = _run(fixture, tmp_path)
    assert result["status"] == "no-authority-proposed"
    receipt = json.loads((tmp_path / "output/trusted-bootstrap-admission.json").read_text())
    assert receipt["payload"]["status"] == "no-authority-proposed"


def test_self_authorized_first_transition_fails_closed(tmp_path):
    fixture = _build_fixture(tmp_path, "self-signal")
    with pytest.raises(bootstrap.BootstrapError, match="authority fields"):
        _run(fixture, tmp_path)
