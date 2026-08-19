"""VAL-CDG-017 historical backfill and route-truth sequence regressions.

The tests deliberately use checked-in records and immutable local Git objects,
not live GitHub API calls. The later ``route_truth`` sealer owns live capsule
re-authentication; this file pins the facts it must rediscover.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TypedDict


class AssetBinding(TypedDict):
    api_id: int
    byte_length: int
    name: str
    sha256: str


class PatchBinding(TypedDict):
    byte_length: int
    sha256: str


class CorrectiveCapsuleBinding(TypedDict):
    end_sha: str
    release_id: int
    tag: str


class SequenceEntry(TypedDict):
    fulfills: tuple[str, ...]
    merge_sha: str
    pr: int
    role: str


ROOT = Path(__file__).resolve().parents[2]
BACKFILL_RECORD = ROOT / "docs/audits/2026-08-01-pr9320-historical-first-parent-backfill.md"
AUTHORITY_INVENTORY = ROOT / "scripts/baselines/contract_drift_inventory.json"

PR_9320 = 9320
PR_9320_HEAD = "aba6b14c94eca3a9c825b1a303ea67684d5f8daa"
PR_9320_OLD_HEAD = "73cf6d4831c2cb032f31108dada6251f45571748"
PR_9320_MERGE = "0b28f68b9f4d204ae14814169093723ea84c1364"
PR_9320_FIRST_PARENT = "e448b840dad03ee28accd218c14a27fa8b87c7b4"
PR_9320_MERGED_AT = "2026-07-16T20:00:49Z"
PR_9320_ACTOR = "scarmani"
PR_9320_HEAD_TREE = "e5c6c3d07a918cf43fffed6d4a9f472bc10a674a"
PR_9320_MERGE_TREE = "79c1c374eed261c42468dc526d837e726e73425a"

BACKFILL_TAG = f"backfill-{PR_9320_MERGE}"
BACKFILL_RELEASE_ID = 363450207
BACKFILL_PAYLOAD: AssetBinding = {
    "api_id": 497649474,
    "byte_length": 10437,
    "name": "payload.json",
    "sha256": "9c238f0aa2a7c69547a78900c6aba95f4771bdd8df68105de6e9a61cc2a4523e",
}
BACKFILL_MANIFEST: AssetBinding = {
    "api_id": 497649469,
    "byte_length": 299,
    "name": "manifest.json",
    "sha256": "b2c81cc56fba3749756690a39241153c90d17bc6b2ecfdc3c38c0bc1cdafa7b5",
}
BACKFILL_CHECKSUMS: AssetBinding = {
    "api_id": 497649488,
    "byte_length": 159,
    "name": "checksums.txt",
    "sha256": "0d3c7dc932f6860335edac2df08104107f63a78459341ff0fd4822bc19845d56",
}
BACKFILL_RULE_SUITE = {
    "after_sha": "486b24fbb131d27b90853c1d64dd949834427e1f",
    "id": 3525237532,
    "ref": "refs/heads/main",
    "repository_id": 1126097105,
    "result": "pass",
}
BACKFILL_RECEIPT_EXECUTION = {
    "check_id": 91336914751,
    "job_id": 91336914751,
    "run_attempt": 1,
    "workflow_run_id": 30687788027,
}
FIRST_PARENT_PATCH: PatchBinding = {
    "byte_length": 5874,
    "sha256": "a5c94ff5c9d32a60c055d5ae67b21935dd7f98aae6f868ab1d68e300bb604455",
}
SEMANTIC_PATHS = [
    "aragora/server/handlers/social/__init__.py",
    "aragora/server/handlers/social/sharing.py",
    "tests/handlers/social/test_sharing.py",
]
HISTORICAL_FACT = {
    "actor": PR_9320_ACTOR,
    "head_sha": PR_9320_HEAD,
    "merge_sha": PR_9320_MERGE,
    "merged_at": PR_9320_MERGED_AT,
    "pr": PR_9320,
}
CORRECTIVE_CAPSULE: CorrectiveCapsuleBinding = {
    "end_sha": "3aa420631c4e906ee4c12bf4fde06282c1f21489",
    "release_id": 364632369,
    "tag": "cdg-corrective_bootstrap-3aa420631c4e906ee4c12bf4fde06282c1f21489",
}
CONSTANTS_MERGE = "ee686e9d116c704ede146a6ec69dfe013b6c32be"
MATCHER_MERGE = "e8a0d165242737d3226b6d3360aa9e8ec014fd75"
STAGE1_MERGE = "9482fc2dffdb6425d2405389c13f46d5954ac467"
STAGE2_MERGE = "d20d22335e3459050f5c0e433120db309f503d13"
CORRECTIVE_MERGE = "d3e45fafe6dd04508882935c813f6896abc859d7"
BACKFILL_RECORDING_MERGE = "486b24fbb131d27b90853c1d64dd949834427e1f"
BACKFILL_CAPSULE_BINDING_MERGE = "4b36750a2ea3433e42a06927c79056b3bd5a9e3d"
ROUTE_CORE_MERGE = "9148ba293404e8d46c4c588d4169d32fb11e29b0"
OPENAPI_REARM_MERGE = "56af53b778f6d8ec80f81edc4f8ed6651d4410b7"
SEQUENCE: list[SequenceEntry] = [
    {
        "fulfills": (),
        "merge_sha": CONSTANTS_MERGE,
        "pr": 9406,
        "role": "constants",
    },
    {
        "fulfills": (),
        "merge_sha": MATCHER_MERGE,
        "pr": 9413,
        "role": "matcher",
    },
    {
        "fulfills": (),
        "merge_sha": STAGE1_MERGE,
        "pr": 9429,
        "role": "stage1",
    },
    {
        "fulfills": ("VAL-CDG-001",),
        "merge_sha": STAGE2_MERGE,
        "pr": 9449,
        "role": "stage2",
    },
    {
        "fulfills": (),
        "merge_sha": CORRECTIVE_MERGE,
        "pr": 9645,
        "role": "corrective",
    },
    {
        "fulfills": (),
        "merge_sha": BACKFILL_CAPSULE_BINDING_MERGE,
        "pr": 9692,
        "role": "backfill",
    },
    {
        "fulfills": ("VAL-CDG-011",),
        "merge_sha": ROUTE_CORE_MERGE,
        "pr": 9717,
        "role": "route_core",
    },
    {
        "fulfills": ("VAL-CDG-012",),
        "merge_sha": OPENAPI_REARM_MERGE,
        "pr": 9719,
        "role": "openapi_rearm",
    },
]


@lru_cache
def _record_text() -> str:
    return BACKFILL_RECORD.read_text(encoding="utf-8")


@lru_cache
def _authority() -> dict:
    inventory = json.loads(AUTHORITY_INVENTORY.read_text(encoding="utf-8"))
    return inventory["accepted_authority"]


def _canonical_json_bytes(value: object, *, terminal_lf: bool) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return raw + (b"\n" if terminal_lf else b"")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-c", "core.pager=cat", *args],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def _git_text(*args: str) -> str:
    return _git_bytes(*args).decode().strip()


def _all_commits_present(*shas: str) -> bool:
    return all(
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
        for sha in shas
    )


def _assert_recorded(*needles: str) -> None:
    text = _record_text()
    for needle in needles:
        assert needle in text


def test_pr_9320_is_exact_historical_nonconforming_head_and_merge() -> None:
    historical = _authority()["transition"]["historical_nonconforming"]
    matches = [fact for fact in historical if fact["pr"] == PR_9320]
    assert matches == [HISTORICAL_FACT]

    # Full clones recompute from immutable Git objects. Shallow CI clones retain
    # the always-on checked-in authority and recording assertions below.
    if _all_commits_present(PR_9320_MERGE, PR_9320_HEAD):
        merge_fields = _git_text(
            "show",
            "-s",
            "--format=%H|%P|%T|%s",
            PR_9320_MERGE,
        ).split("|")
        assert merge_fields == [
            PR_9320_MERGE,
            PR_9320_FIRST_PARENT,
            PR_9320_MERGE_TREE,
            "fix(sharing): warn on legacy share model access (#9320)",
        ]
        assert _git_text("show", "-s", "--format=%T", PR_9320_HEAD) == PR_9320_HEAD_TREE
    _assert_recorded(
        "Status: `historical_backfill` + `historical_nonconforming`",
        PR_9320_HEAD,
        PR_9320_MERGE,
        PR_9320_MERGED_AT,
        PR_9320_ACTOR,
    )


def test_pr_9320_requires_first_parent_historical_backfill_not_future_admission() -> None:
    if _all_commits_present(PR_9320_MERGE, PR_9320_FIRST_PARENT):
        assert _git_text("show", "-s", "--format=%P", PR_9320_MERGE) == PR_9320_FIRST_PARENT
    _assert_recorded(
        "required historical first-parent receipt/backfill",
        "pre-merge admission evidence (`contract-drift-pr-delta` never ran for this PR)",
        "forward immutable-boundary, chronology, settlement, or capsule proof",
    )
    assert BACKFILL_TAG == f"backfill-{PR_9320_MERGE}"
    assert "historical_backfill" in _record_text()
    assert "future admission" in _record_text()


def test_pr_9320_historical_record_cannot_supply_authority_chronology_no_admin_or_forward_capsule_proof() -> (
    None
):
    assert set(HISTORICAL_FACT) == {"actor", "head_sha", "merge_sha", "merged_at", "pr"}
    forbidden_authority = {
        "accepted_authority",
        "admission",
        "admin",
        "authority",
        "boundary",
        "capsule",
        "chronology",
        "post_settlement_quorum",
        "settlement",
    }
    assert not (set(HISTORICAL_FACT) & forbidden_authority)
    _assert_recorded(
        "accepted authority or an authority transition",
        "forward proof of no-admin merge behavior",
        "forward immutable-boundary, chronology, settlement, or capsule proof",
    )


def test_pr_9320_old_delegations_are_consumed_or_void() -> None:
    _assert_recorded(
        PR_9320_OLD_HEAD,
        "every delegation bound to it remain",
        "consumed or void",
    )
    assert PR_9320_OLD_HEAD != PR_9320_HEAD
    if _all_commits_present(PR_9320_HEAD, PR_9320_OLD_HEAD):
        assert PR_9320_OLD_HEAD in _git_text("show", "-s", "--format=%P", PR_9320_HEAD).split()


def test_pr_9320_backfill_payload_binds_canonical_artifacts_cohort_provenance_and_projection_edges() -> (
    None
):
    authority = _authority()
    bindings = {entry["path"]: entry for entry in authority["canonical_artifact_bindings"]}
    artifacts = authority["canonical_artifacts"]

    for key, path in (
        ("original_cohort", "library/contract-drift-original-cohort-v1.json"),
        ("sdk_provenance", "library/contract-drift-sdk-provenance-v1.json"),
    ):
        raw = _canonical_json_bytes(artifacts[key], terminal_lf=True)
        assert len(raw) == bindings[path]["byte_length"]
        assert _sha256(raw) == bindings[path]["sha256"]

    cohort = artifacts["original_cohort"]
    provenance = artifacts["sdk_provenance"]
    projection = cohort["operation_projection"]
    assert cohort["counts"] == {
        "by_category": {
            "python_sdk_drift": 74,
            "routes_missing_in_spec": 11,
            "routes_orphaned_in_spec": 17,
            "sdk_missing_from_both": 29,
            "typescript_sdk_drift": 524,
        },
        "method_bearing_sdk_records": 598,
        "method_null_route_parity_records": 57,
        "records": 655,
        "route_projection_max_operation_edges": 4,
        "route_projection_records_with_multiple_operation_edges": 9,
        "route_projection_unresolved_method_records": 0,
        "sdk_provenance_links": 598,
    }
    assert (
        cohort["original_record_id_set"]["sha256"]
        == "c1235670c183b1887ba3fe4280fa0320f9fd6f4a85b8f346d4332ac2aebbe269"
    )
    assert len(projection["records"]) == 655
    assert sum(len(record["operation_edges"]) for record in projection["records"]) == 666
    route_ids = {
        record["original_record_id"]
        for record in cohort["original_records"]
        if record["method"] is None
    }
    route_edge_counts = sorted(
        len(record["operation_edges"])
        for record in projection["records"]
        if record["original_record_id"] in route_ids
    )
    assert route_edge_counts == ([1] * 48 + [2] * 8 + [4])
    assert projection["record_digest_set_sha256"] == (
        "2d6790a6f825c53047639d9433f40e3e10b5bfc9e357bcd161f6b341134775e5"
    )
    assert provenance["counts"] == {
        "core": 75,
        "extended": 523,
        "python_sdk_drift": 74,
        "records": 598,
        "records_with_multiple_distinct_atoms": 12,
        "source_occurrences": 690,
        "typescript_sdk_drift": 524,
    }
    assert provenance["partition"]["intersection_count"] == 0
    assert provenance["partition"]["union_count"] == 598


def test_pr_9320_backfill_payload_binds_semantic_run_attempt_job_and_check_digests() -> None:
    active_inventory = _authority()["active_inventory"]
    assert (
        _sha256(_canonical_json_bytes(active_inventory, terminal_lf=False))
        == (_authority()["active_inventory_sha256"])
    )
    assert BACKFILL_RECEIPT_EXECUTION == {
        "check_id": 91336914751,
        "job_id": 91336914751,
        "run_attempt": 1,
        "workflow_run_id": 30687788027,
    }
    assert FIRST_PARENT_PATCH["sha256"] in _record_text()
    _assert_recorded(
        BACKFILL_PAYLOAD["sha256"],
        "run 30687788027 attempt 1",
        "job/check 91336914751",
        "analyzer-bundle digest = `1b977ec70e9400ec5239c22474a4400753c7d99acee78c4a6426c2ce8bc47356`",
        "active_inventory_sha256",
    )


def test_pr_9320_backfill_release_capsule_binds_manifest_payload_checksums_attestation_and_rule_suite() -> (
    None
):
    manifest_object = {
        "first_parent_sha": PR_9320_FIRST_PARENT,
        "merge_sha": PR_9320_MERGE,
        "payload_byte_length": BACKFILL_PAYLOAD["byte_length"],
        "payload_sha256": BACKFILL_PAYLOAD["sha256"],
        "pr": PR_9320,
        "schema": "contract-drift-historical-backfill-manifest-v1",
    }
    manifest_bytes = _canonical_json_bytes(manifest_object, terminal_lf=True)
    assert len(manifest_bytes) == BACKFILL_MANIFEST["byte_length"]
    assert _sha256(manifest_bytes) == BACKFILL_MANIFEST["sha256"]

    checksums_bytes = (
        f"{BACKFILL_MANIFEST['sha256']}  manifest.json\n"
        f"{BACKFILL_PAYLOAD['sha256']}  payload.json\n"
    ).encode()
    assert len(checksums_bytes) == BACKFILL_CHECKSUMS["byte_length"]
    assert _sha256(checksums_bytes) == BACKFILL_CHECKSUMS["sha256"]

    assert BACKFILL_RULE_SUITE == {
        "after_sha": "486b24fbb131d27b90853c1d64dd949834427e1f",
        "id": 3525237532,
        "ref": "refs/heads/main",
        "repository_id": 1126097105,
        "result": "pass",
    }
    _assert_recorded(
        f"release API ID {BACKFILL_RELEASE_ID}",
        BACKFILL_TAG,
        "`immutable=true`",
        "`gh release verify",
        "release attestation loaded from the GitHub API",
        "rule-suite record (ID 3525237532",
        "at publication `gh release verify",
        "`gh attestation verify` reported no SLSA-provenance attestation",
    )


def test_pr_9320_backfill_is_durable_after_actions_artifact_expiration() -> None:
    assets = [BACKFILL_MANIFEST, BACKFILL_PAYLOAD, BACKFILL_CHECKSUMS]
    assert [asset["name"] for asset in assets] == [
        "manifest.json",
        "payload.json",
        "checksums.txt",
    ]
    assert all(asset["api_id"] > 0 and asset["byte_length"] > 0 for asset in assets)
    assert all(len(asset["sha256"]) == 64 for asset in assets)
    _assert_recorded(
        "The canonical durable artifact is the published immutable GitHub Release",
        "asset deletion is rejected with HTTP 422",
        "Cannot delete\nasset from an immutable release",
        "the capsule identity — tag,",
        "target SHA, and asset bytes — is immutable",
    )
    manifest_keys = {
        "first_parent_sha",
        "merge_sha",
        "payload_byte_length",
        "payload_sha256",
        "pr",
        "schema",
    }
    assert not any("artifact" in key for key in manifest_keys)


def test_pr_9320_backfill_recomputes_first_parent_to_squash_semantics() -> None:
    assert PR_9320_HEAD_TREE != PR_9320_MERGE_TREE
    assert FIRST_PARENT_PATCH == {
        "byte_length": 5874,
        "sha256": "a5c94ff5c9d32a60c055d5ae67b21935dd7f98aae6f868ab1d68e300bb604455",
    }
    assert SEMANTIC_PATHS == [
        "aragora/server/handlers/social/__init__.py",
        "aragora/server/handlers/social/sharing.py",
        "tests/handlers/social/test_sharing.py",
    ]
    if _all_commits_present(PR_9320_MERGE, PR_9320_FIRST_PARENT):
        assert _git_text("show", "-s", "--format=%P", PR_9320_MERGE) == PR_9320_FIRST_PARENT
        patch = _git_bytes("diff", PR_9320_FIRST_PARENT, PR_9320_MERGE)
        assert len(patch) == FIRST_PARENT_PATCH["byte_length"]
        assert _sha256(patch) == FIRST_PARENT_PATCH["sha256"]
        changed_paths = _git_text(
            "diff",
            "--name-only",
            PR_9320_FIRST_PARENT,
            PR_9320_MERGE,
        ).splitlines()
        assert changed_paths == SEMANTIC_PATHS
        assert not any(
            path.startswith(("scripts/baselines/", "scripts/check_contract_drift"))
            for path in changed_paths
        )
    _assert_recorded(
        "Exact first-parent->merge patch: 5,874 bytes",
        FIRST_PARENT_PATCH["sha256"],
        "Head-tree vs merge-tree | NOT equal",
        "`governed_surface_delta=none`",
        "No\n`original_record_id` was added, removed, or replaced",
    )


def test_pr_9320_cannot_be_reopened_resettled_or_remerged() -> None:
    matching_facts = [
        fact
        for fact in _authority()["transition"]["historical_nonconforming"]
        if fact["pr"] == PR_9320
    ]
    assert matching_facts == [HISTORICAL_FACT]
    _assert_recorded(
        "must never be reopened, resettled, remerged, edited",
        "cannot supply",
        "sole accepted disposition remains `historical_nonconforming`",
    )
    assert "state" not in HISTORICAL_FACT
    assert "settlement_id" not in HISTORICAL_FACT


def test_constants_matcher_stage1_stage2_corrective_backfill_route_openapi_order_is_exact() -> None:
    assert [entry["role"] for entry in SEQUENCE] == [
        "constants",
        "matcher",
        "stage1",
        "stage2",
        "corrective",
        "backfill",
        "route_core",
        "openapi_rearm",
    ]
    chronological_shas = [entry["merge_sha"] for entry in SEQUENCE]
    anchors = [*chronological_shas, BACKFILL_RECORDING_MERGE, CORRECTIVE_CAPSULE["end_sha"]]
    if _all_commits_present(*anchors):
        first_parent = _git_text("rev-list", "--first-parent", "--reverse", "HEAD").splitlines()
        positions = {sha: first_parent.index(sha) for sha in chronological_shas}
        ordered_positions = [positions[sha] for sha in chronological_shas]
        assert ordered_positions == sorted(ordered_positions)
        assert len(set(ordered_positions)) == len(SEQUENCE)

        backfill_record_position = first_parent.index(BACKFILL_RECORDING_MERGE)
        assert positions[CORRECTIVE_MERGE] < backfill_record_position
        assert backfill_record_position < positions[BACKFILL_CAPSULE_BINDING_MERGE]
        assert positions[BACKFILL_CAPSULE_BINDING_MERGE] < positions[ROUTE_CORE_MERGE]
        assert positions[ROUTE_CORE_MERGE] < positions[OPENAPI_REARM_MERGE]

        corrective_capsule_position = first_parent.index(CORRECTIVE_CAPSULE["end_sha"])
        assert positions[CORRECTIVE_MERGE] < corrective_capsule_position
        assert corrective_capsule_position < positions[ROUTE_CORE_MERGE]
    assert CORRECTIVE_CAPSULE["release_id"] == 364632369
    assert BACKFILL_RELEASE_ID == 363450207
    assert CORRECTIVE_CAPSULE["release_id"] != BACKFILL_RELEASE_ID
    assert CORRECTIVE_CAPSULE["tag"].endswith(CORRECTIVE_CAPSULE["end_sha"])


def test_stage1_has_no_fulfills_and_stage2_is_sole_val_cdg_001_owner() -> None:
    by_role = {entry["role"]: entry for entry in SEQUENCE}
    assert by_role["stage1"]["fulfills"] == ()
    assert by_role["stage2"]["fulfills"] == ("VAL-CDG-001",)
    assert [entry["role"] for entry in SEQUENCE if "VAL-CDG-001" in entry["fulfills"]] == ["stage2"]

    if _all_commits_present(STAGE1_MERGE, STAGE2_MERGE):
        stage1_paths = set(
            _git_text("diff-tree", "--no-commit-id", "--name-only", "-r", STAGE1_MERGE).splitlines()
        )
        stage2_paths = set(
            _git_text("diff-tree", "--no-commit-id", "--name-only", "-r", STAGE2_MERGE).splitlines()
        )
        assert "scripts/generate_contract_drift_inventory.py" in stage1_paths
        assert "scripts/check_contract_drift_ratchet.py" not in stage1_paths
        assert stage2_paths == {
            "scripts/check_contract_drift_ratchet.py",
            "tests/scripts/test_check_contract_drift_ratchet.py",
        }
        assert _git_text("show", "-s", "--format=%s", STAGE1_MERGE).endswith("(#9429)")
        assert _git_text("show", "-s", "--format=%s", STAGE2_MERGE).endswith("(#9449)")


def test_route_core_and_openapi_rearm_are_separate_prs() -> None:
    by_role = {entry["role"]: entry for entry in SEQUENCE}
    route = by_role["route_core"]
    rearm = by_role["openapi_rearm"]
    assert route["pr"] == 9717
    assert rearm["pr"] == 9719
    assert route["pr"] != rearm["pr"]
    assert route["merge_sha"] != rearm["merge_sha"]
    if _all_commits_present(route["merge_sha"], rearm["merge_sha"]):
        assert set(
            _git_text(
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                route["merge_sha"],
            ).splitlines()
        ) == {
            "scripts/validate_openapi_routes.py",
            "tests/scripts/test_validate_openapi_routes.py",
        }
        assert set(
            _git_text(
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                rearm["merge_sha"],
            ).splitlines()
        ) == {
            ".github/workflows/openapi.yml",
            "tests/scripts/test_openapi_workflow_contract.py",
        }


def test_route_core_merges_before_openapi_rearm() -> None:
    roles = [entry["role"] for entry in SEQUENCE]
    assert roles.index("route_core") < roles.index("openapi_rearm")
    if _all_commits_present(ROUTE_CORE_MERGE, OPENAPI_REARM_MERGE):
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ROUTE_CORE_MERGE, OPENAPI_REARM_MERGE],
            cwd=ROOT,
            check=False,
        )
        assert ancestry.returncode == 0
        first_parent = _git_text("rev-list", "--first-parent", "--reverse", "HEAD").splitlines()
        assert first_parent.index(ROUTE_CORE_MERGE) < first_parent.index(OPENAPI_REARM_MERGE)
        assert _git_text("show", "-s", "--format=%s", ROUTE_CORE_MERGE).endswith("(#9717)")
        assert _git_text("show", "-s", "--format=%s", OPENAPI_REARM_MERGE).endswith("(#9719)")


def test_openapi_rearm_requires_method_aware_route_core() -> None:
    workflow = (ROOT / ".github/workflows/openapi.yml").read_text(encoding="utf-8")
    assert "Method-aware operation plane (VAL-CDG-011/012)" in workflow
    assert 'EXEC_SHA="$(git rev-parse HEAD)"' in workflow
    assert "scripts/validate_openapi_routes.py \\" in workflow
    assert '--ref "${EXEC_SHA}"' in workflow
    assert 'if plane.get("ref") != sys.argv[1]:' in workflow
    assert "method-aware plane ref" in workflow

    route_core_source = (ROOT / "scripts/validate_openapi_routes.py").read_text(encoding="utf-8")
    assert '"--ref"' in route_core_source
    assert '"served_operations"' in route_core_source
    assert '"operation_projection"' in route_core_source
    if _all_commits_present(ROUTE_CORE_MERGE, OPENAPI_REARM_MERGE):
        assert (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ROUTE_CORE_MERGE, OPENAPI_REARM_MERGE],
                cwd=ROOT,
                check=False,
            ).returncode
            == 0
        )
