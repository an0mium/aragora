from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import build_contract_drift_historical_backfill as backfill

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "ee989c889e51f911f1cf5dd5fe667417613bbeb6"
PR_BASE_SHA = "14d1ef53e23c5466c0491ed93f72752944c78cd4"
HEAD_SHA = "aba6b14c94eca3a9c825b1a303ea67684d5f8daa"
MERGE_SHA = "0b28f68b9f4d204ae14814169093723ea84c1364"
FIRST_PARENT_SHA = "e448b840dad03ee28accd218c14a27fa8b87c7b4"
HEAD_TREE_SHA = "e5c6c3d07a918cf43fffed6d4a9f472bc10a674a"
MERGE_TREE_SHA = "79c1c374eed261c42468dc526d837e726e73425a"
PATCH_SHA256 = "a5c94ff5c9d32a60c055d5ae67b21935dd7f98aae6f868ab1d68e300bb604455"
OLD_RELEASE_ID = 363450207
SYNTHETIC_RELEASE_ID = 990000001
SYNTHETIC_RULE_SUITE_ID = 990000002
SYNTHETIC_RECEIPT_RUN_ID = 990000003
SYNTHETIC_RECEIPT_JOB_ID = 990000004
APP_ID = 15368
EXPECTED_PATHS = [
    "aragora/server/handlers/social/__init__.py",
    "aragora/server/handlers/social/sharing.py",
    "tests/handlers/social/test_sharing.py",
]


def _git_text(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _authority() -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{SOURCE_SHA}:scripts/baselines/contract_drift_inventory.json",
        ]
    )
    return json.loads(raw)["accepted_authority"]


def _authority_manifest() -> dict[str, Any]:
    return json.loads(
        subprocess.check_output(
            [
                "python3",
                "-B",
                "scripts/generate_contract_drift_inventory.py",
                "--authority-manifest",
                "--ref",
                SOURCE_SHA,
            ],
            cwd=ROOT,
        )
    )


def _contexts() -> list[dict[str, Any]]:
    return [
        {
            "app_id": APP_ID,
            "check_run_id": 87709243174,
            "conclusion": "success",
            "job_id": 87709243174,
            "name": "lint",
            "run_attempt": 1,
            "workflow_run_id": 29524359563,
        },
        {
            "app_id": APP_ID,
            "check_run_id": 87709180560,
            "conclusion": "success",
            "job_id": 87709180560,
            "name": "typecheck",
            "run_attempt": 1,
            "workflow_run_id": 29524359563,
        },
        {
            "app_id": APP_ID,
            "check_run_id": 87709276751,
            "conclusion": "success",
            "job_id": 87709276751,
            "name": "sdk-parity",
            "run_attempt": 1,
            "workflow_run_id": 29524359665,
        },
        {
            "app_id": APP_ID,
            "check_run_id": 87709726971,
            "conclusion": "success",
            "job_id": 87709726971,
            "name": "Generate & Validate",
            "run_attempt": 1,
            "workflow_run_id": 29524359572,
        },
        {
            "app_id": APP_ID,
            "check_run_id": 87709013895,
            "conclusion": "success",
            "job_id": 87709013895,
            "name": "TypeScript SDK Type Check",
            "run_attempt": 1,
            "workflow_run_id": 29524359727,
        },
        {
            "app_id": APP_ID,
            "check_run_id": 87728267780,
            "conclusion": "success",
            "job_id": 87728267780,
            "name": "aragora-merge-quorum",
            "run_attempt": 3,
            "workflow_run_id": 29524359568,
        },
    ]


def _input_document() -> dict[str, Any]:
    return {
        "attestation": {
            "predicate_type": backfill.ratchet.RELEASE_ATTESTATION_PREDICATE_TYPE,
            "repository": "synaptent/aragora",
            "schema": backfill.ATTESTATION_SCHEMA,
            "signer_san_regexp": backfill.ratchet.RELEASE_ATTESTATION_SIGNER_SAN_REGEXP,
            "source_digest": MERGE_SHA,
            "subject_asset_names": list(backfill.EXPECTED_ASSET_NAMES),
            "verified": True,
            "workflow": "actions/attest@v4",
            "workflow_path": backfill.ATTESTATION_WORKFLOW_PATH,
        },
        "authority_source_sha": SOURCE_SHA,
        "disposition": {
            "authoritative_for_future_admission": False,
            "precedential": False,
            "status": "historical_nonconforming",
        },
        "historical_pull_request": {
            "actor": "scarmani",
            "base_sha": PR_BASE_SHA,
            "changed_files": [
                {
                    "additions": 3,
                    "deletions": 1,
                    "path": "aragora/server/handlers/social/__init__.py",
                },
                {
                    "additions": 23,
                    "deletions": 0,
                    "path": "aragora/server/handlers/social/sharing.py",
                },
                {
                    "additions": 89,
                    "deletions": 2,
                    "path": "tests/handlers/social/test_sharing.py",
                },
            ],
            "first_parent_patch_byte_length": 5874,
            "first_parent_patch_sha256": PATCH_SHA256,
            "first_parent_sha": FIRST_PARENT_SHA,
            "head_sha": HEAD_SHA,
            "head_tree_sha": HEAD_TREE_SHA,
            "merge_sha": MERGE_SHA,
            "merge_tree_sha": MERGE_TREE_SHA,
            "merged_at": "2026-07-16T20:00:49Z",
            "pr": 9320,
        },
        "receipt": {
            "artifact_name": f"contract-drift-main-receipt-{MERGE_SHA}",
            "base_sha": PR_BASE_SHA,
            "check_run_id": SYNTHETIC_RECEIPT_JOB_ID,
            "conclusion": "success",
            "first_parent_sha": FIRST_PARENT_SHA,
            "head_sha": HEAD_SHA,
            "job_id": SYNTHETIC_RECEIPT_JOB_ID,
            "merge_sha": MERGE_SHA,
            "required_contexts": _contexts(),
            "run_attempt": 1,
            "schema": backfill.RECEIPT_SCHEMA,
            "source_sha": MERGE_SHA,
            "workflow_name": "Contract Drift Governance",
            "workflow_path": backfill.WORKFLOW_PATH,
            "workflow_run_id": SYNTHETIC_RECEIPT_RUN_ID,
        },
        "release": {
            "asset_names": list(backfill.EXPECTED_ASSET_NAMES),
            "exact_full_sha_tag": MERGE_SHA,
            "immutable": True,
            "release_api_id": SYNTHETIC_RELEASE_ID,
            "tag_name": f"backfill-v2-{MERGE_SHA}",
            "tag_target_sha": MERGE_SHA,
            "verified": True,
        },
        "repository": "synaptent/aragora",
        "rule_suite": {
            "after_sha": MERGE_SHA,
            "id": SYNTHETIC_RULE_SUITE_ID,
            "ref": "refs/heads/main",
            "repository_id": 1126097105,
            "repository_name": "synaptent/aragora",
            "result": "pass",
            "schema": backfill.RULE_SUITE_SCHEMA,
        },
        "supersedes": {
            "reason": (
                "Immutable release 363450207 is retained as superseded historical "
                "evidence because its receipt and attestation planes were incomplete."
            ),
            "release_api_id": OLD_RELEASE_ID,
            "status": "superseded_historical_evidence",
            "tag_name": f"backfill-{MERGE_SHA}",
        },
    }


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return backfill.build_payload(
        repo_root=ROOT,
        input_document=_input_document(),
        authority_manifest=_authority_manifest(),
    )


@pytest.fixture(scope="module")
def assets(payload: dict[str, Any]) -> dict[str, bytes]:
    return backfill.build_capsule_bytes(payload)


def test_fixture_builds_byte_identically_twice(
    payload: dict[str, Any],
    assets: dict[str, bytes],
) -> None:
    second = backfill.build_capsule_bytes(copy.deepcopy(payload))
    assert assets == second
    assert list(assets) == ["checksums.txt", "manifest.json", "payload.json"]
    assert all(raw for raw in assets.values())
    assert OLD_RELEASE_ID not in {payload["release"]["release_api_id"]}
    assert payload["supersedes"]["release_api_id"] == OLD_RELEASE_ID


def test_fixture_binds_all_semantic_planes(payload: dict[str, Any]) -> None:
    authority = _authority()
    projection = payload["projection"]
    assert projection["membership_count"] == 655
    assert projection["edge_count"] == 666
    assert projection["multi_edge_originals"] == 9
    assert projection["max_edges"] == 4
    assert (
        projection["records"]
        == (authority["canonical_artifacts"]["original_cohort"]["operation_projection"]["records"])
    )
    assert payload["authority"]["original_record_id_set_sha256"] == (
        "c1235670c183b1887ba3fe4280fa0320f9fd6f4a85b8f346d4332ac2aebbe269"
    )
    assert payload["authority"]["operation_projection_record_digest_set_sha256"] == (
        "2d6790a6f825c53047639d9433f40e3e10b5bfc9e357bcd161f6b341134775e5"
    )
    assert payload["authority"]["operation_projection_schema_sha256"] == (
        "26bb802de06dcda62be8dc0b8f67b7ee9243af0692e712340771835a76e5b5ce"
    )
    assert payload["authority"]["sdk_provenance_record_digest_set_sha256"] == (
        "0d30ce3b083344f19949da12ae2d92952757af0aea800b3f99d447458b6eeba0"
    )
    assert payload["authority"]["sdk_original_record_id_set_sha256"] == (
        "51a963079136a92a86485b56f6cef42aafc7749bfad146ce5fb37293524c5762"
    )
    assert payload["authority"]["core_original_record_id_set_sha256"] == (
        "b3a1755f027c998d507f13f3ba9093f769cea8720d44bfac12be6beccd626787"
    )
    assert payload["authority"]["extended_original_record_id_set_sha256"] == (
        "bb1fc41548778022dab3041bc05fc40a4da239a1bd4ad8b1ccbcd1007d90b252"
    )
    assert payload["receipt"]["base_sha"] == PR_BASE_SHA
    assert payload["receipt"]["head_sha"] == HEAD_SHA
    assert (
        payload["historical_pull_request"]["changed_files"]
        == _input_document()["historical_pull_request"]["changed_files"]
    )


@pytest.mark.parametrize(
    "path",
    [
        *((field,) for field in sorted(backfill.PAYLOAD_FIELDS)),
        *(("historical_pull_request", field) for field in sorted(backfill.HISTORICAL_PR_FIELDS)),
        *(
            ("historical_pull_request", "changed_files", "0", field)
            for field in sorted(backfill.CHANGED_FILE_FIELDS)
        ),
        *(("authority", field) for field in sorted(backfill.AUTHORITY_FIELDS)),
        *(("projection", field) for field in sorted(backfill.PROJECTION_FIELDS)),
        *(
            ("projection", "records", "0", field)
            for field in sorted(backfill.PROJECTION_RECORD_FIELDS)
        ),
        *(
            ("projection", "records", "0", "operation_edges", "0", field)
            for field in sorted(backfill.PROJECTION_EDGE_FIELDS)
        ),
        *(("receipt", field) for field in sorted(backfill.RECEIPT_FIELDS)),
        *(
            ("receipt", "required_contexts", "0", field)
            for field in sorted(backfill.CONTEXT_FIELDS)
        ),
        *(("release", field) for field in sorted(backfill.RELEASE_FIELDS)),
        *(("attestation", field) for field in sorted(backfill.ATTESTATION_FIELDS)),
        *(("rule_suite", field) for field in sorted(backfill.RULE_SUITE_FIELDS)),
        *(("supersedes", field) for field in sorted(backfill.SUPERSEDES_FIELDS)),
    ],
)
def test_every_required_payload_field_is_fail_closed(
    payload: dict[str, Any],
    path: tuple[str, ...],
) -> None:
    candidate = copy.deepcopy(payload)
    normalized = tuple(int(item) if item.isdigit() else item for item in path)
    current: Any = candidate
    for key in normalized[:-1]:
        current = current[key]
    del current[normalized[-1]]
    with pytest.raises(ValueError):
        backfill.validate_payload(candidate)


def test_failed_receipt_and_missing_attestation_fail_closed(payload: dict[str, Any]) -> None:
    failed = copy.deepcopy(payload)
    failed["receipt"]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="did not succeed"):
        backfill.validate_payload(failed)

    missing = copy.deepcopy(payload)
    del missing["attestation"]
    with pytest.raises(ValueError, match="fields are incomplete"):
        backfill.validate_payload(missing)

    wrong_base = copy.deepcopy(payload)
    wrong_base["receipt"]["base_sha"] = FIRST_PARENT_SHA
    with pytest.raises(ValueError, match="receipt base SHA mismatch"):
        backfill.validate_payload(wrong_base)

    wrong_head = copy.deepcopy(payload)
    wrong_head["receipt"]["head_sha"] = MERGE_SHA
    with pytest.raises(ValueError, match="receipt head SHA mismatch"):
        backfill.validate_payload(wrong_head)


def test_tampered_assets_fail_closed(
    payload: dict[str, Any],
    assets: dict[str, bytes],
) -> None:
    with pytest.raises(ValueError, match="terminal LF"):
        backfill.validate_capsule_bytes(
            manifest_bytes=assets["manifest.json"],
            payload_bytes=assets["payload.json"] + b" ",
            checksums_bytes=assets["checksums.txt"],
        )
    with pytest.raises(ValueError, match="manifest binding"):
        tampered_manifest = json.loads(assets["manifest.json"])
        tampered_manifest["pr"] = 1
        backfill.validate_capsule_bytes(
            manifest_bytes=backfill._canonical_json_bytes(tampered_manifest),
            payload_bytes=assets["payload.json"],
            checksums_bytes=assets["checksums.txt"],
        )
    with pytest.raises(ValueError, match="checksum asset"):
        backfill.validate_capsule_bytes(
            manifest_bytes=assets["manifest.json"],
            payload_bytes=assets["payload.json"],
            checksums_bytes=b"0" * len(assets["checksums.txt"]),
        )


def test_attestation_subject_signer_and_repository_are_fail_closed(
    payload: dict[str, Any],
    assets: dict[str, bytes],
) -> None:
    evidence = backfill.build_external_attestation_evidence(
        payload=payload,
        assets=assets,
    )
    wrong_subject = copy.deepcopy(evidence)
    wrong_subject["subject_sha256s"]["payload.json"] = "0" * 64
    with pytest.raises(ValueError, match="exact asset bytes"):
        backfill.validate_external_attestation(
            wrong_subject,
            payload=payload,
            assets=assets,
        )
    for field, value in (
        ("signer_san_regexp", "^https://example.invalid$"),
        ("repository", "other/repository"),
        ("source_digest", "f" * 40),
    ):
        mismatch = copy.deepcopy(evidence)
        mismatch[field] = value
        with pytest.raises(ValueError, match=f"{field} mismatch"):
            backfill.validate_external_attestation(
                mismatch,
                payload=payload,
                assets=assets,
            )


def test_rule_suite_and_projection_edges_are_fail_closed(payload: dict[str, Any]) -> None:
    failed_rule = copy.deepcopy(payload)
    failed_rule["rule_suite"]["result"] = "fail"
    with pytest.raises(ValueError, match="did not pass"):
        backfill.validate_payload(failed_rule)

    missing_edge = copy.deepcopy(payload)
    missing_edge["projection"]["records"][0]["operation_edges"].clear()
    with pytest.raises(ValueError, match="no method-specific edges"):
        backfill.validate_payload(missing_edge)


def test_movement_is_fail_closed() -> None:
    stable = {
        "main_sha_after": SOURCE_SHA,
        "main_sha_before": SOURCE_SHA,
        "newest_run_attempt_after": 1,
        "newest_run_attempt_before": 1,
        "newest_workflow_run_id_after": SYNTHETIC_RECEIPT_RUN_ID,
        "newest_workflow_run_id_before": SYNTHETIC_RECEIPT_RUN_ID,
    }
    assert backfill.validate_movement_snapshot(stable) == stable
    moved_main = {**stable, "main_sha_after": "f" * 40}
    with pytest.raises(ValueError, match="main moved"):
        backfill.validate_movement_snapshot(moved_main)
    moved_run = {**stable, "newest_run_attempt_after": 2}
    with pytest.raises(ValueError, match="execution moved"):
        backfill.validate_movement_snapshot(moved_run)


def test_historical_git_fixture_is_exact() -> None:
    assert _git_text("rev-parse", f"{MERGE_SHA}^1") == FIRST_PARENT_SHA
    assert (
        subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", PR_BASE_SHA, HEAD_SHA]
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", PR_BASE_SHA, FIRST_PARENT_SHA]
        ).returncode
        == 0
    )
    assert _git_text("rev-parse", f"{HEAD_SHA}^{{tree}}") == HEAD_TREE_SHA
    assert _git_text("rev-parse", f"{MERGE_SHA}^{{tree}}") == MERGE_TREE_SHA
    assert _git_text("diff", "--name-only", FIRST_PARENT_SHA, MERGE_SHA).splitlines() == (
        EXPECTED_PATHS
    )
    head_patch = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--no-ext-diff",
            "--no-renames",
            f"{PR_BASE_SHA}^{{tree}}",
            f"{HEAD_SHA}^{{tree}}",
        ]
    )
    merge_patch = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--no-ext-diff",
            "--no-renames",
            f"{FIRST_PARENT_SHA}^{{tree}}",
            f"{MERGE_SHA}^{{tree}}",
        ]
    )
    assert head_patch == merge_patch
    assert len(merge_patch) == 5874
    assert backfill._sha256(merge_patch) == PATCH_SHA256


def test_builder_cli_is_supported_from_outside_the_repository(tmp_path: Path) -> None:
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ["PATH"],
        "PYTHONNOUSERSITE": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(ROOT / "scripts/build_contract_drift_historical_backfill.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--verify-dir" in result.stdout
