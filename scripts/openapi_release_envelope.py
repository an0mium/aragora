#!/usr/bin/env python3
"""Read-only OpenAPI release-envelope builder and verifier (FIX-RT-017).

Expiring, fixed-name Actions artifacts are not durable contract evidence. The
``envelope`` job in ``.github/workflows/openapi.yml`` binds the exact
run-level payload bytes of the SHA/run-bound ``openapi-spec-<run>-<attempt>-
<head_sha>`` artifact into an immutable GitHub release at the exact-SHA tag
``openapi-envelope-<head_sha>``, attests the exact assets with the pinned
``actions/attest`` signer, and verifies the whole chain with this helper.

This module never mutates remote state. ``build`` writes deterministic
envelope bytes (canonical compact sorted-key JSON manifest plus an
``sha256sum --check --strict``-compatible checksums file) to a local
directory; ``verify`` re-authenticates a published envelope release, its
assets, its attestations (``--signer-workflow`` + ``--source-digest``), and
the passing rule-suite record for the bound head, re-querying
workflow/run/artifact/main before and after with restart-on-movement
semantics; ``dry-run`` proves deterministic byte construction from checked-in
fixture bytes with no network access.

Exit codes (CDG status vocabulary): 0 pass, 1 fail (verification
contradiction), 2 blocked (publication/attestation/rule-suite not yet
visible), 3 movement (selection moved; restart verification).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

WORKFLOW_PATH = ".github/workflows/openapi.yml"
ENVELOPE_SCHEMA = "aragora/openapi-release-envelope@v1"
TAG_PREFIX = "openapi-envelope-"
MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "checksums.txt"
ARTIFACT_PREFIX = "openapi-spec-"
API_VERSION_HEADER = "X-GitHub-Api-Version: 2022-11-28"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2
EXIT_MOVEMENT = 3

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_ASSET_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Post-verification re-query rule (VAL-CDG-012): if any of these move between
# the opening and closing snapshots, the whole selection restarts.
SELECTION_KEYS = (
    "main_sha",
    "workflow_id",
    "run_id",
    "run_attempt",
    "conclusion",
    "artifact_id",
    "artifact_size",
)


class Blocked(RuntimeError):
    """Publication, attestation, or rule-suite record is not visible yet."""


class Movement(RuntimeError):
    """The bound selection moved during verification; restart required."""


def sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Compact, sorted-key, ASCII JSON with one trailing newline: the only
    manifest serialization the envelope accepts, so identical content can
    never produce distinct release bytes."""
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def make_identity(
    *,
    repository: str,
    head_sha: str,
    run_id: int,
    run_attempt: int,
    artifact_name: str,
) -> dict[str, Any]:
    """Validate and freeze the execution identity every envelope operation is
    bound to. The artifact name must be exactly run-, attempt-, and SHA-bound;
    a fixed or foreign name is rejected before any bytes are touched."""
    if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
        raise ValueError(f"repository {repository!r} is not owner/name")
    if not isinstance(head_sha, str) or not _FULL_SHA.fullmatch(head_sha):
        raise ValueError("head SHA must be a full lowercase 40-hex commit SHA")
    for label, value in (("run id", run_id), ("run attempt", run_attempt)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    expected_name = f"{ARTIFACT_PREFIX}{run_id}-{run_attempt}-{head_sha}"
    if artifact_name != expected_name:
        raise ValueError(
            f"artifact name {artifact_name!r} is not run/SHA-bound (expected {expected_name!r})"
        )
    return {
        "repository": repository,
        "head_sha": head_sha,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "artifact_name": artifact_name,
        "tag": TAG_PREFIX + head_sha,
    }


def _checksum_bytes(digests: dict[str, str]) -> bytes:
    lines = [f"{digest}  {name}\n" for name, digest in sorted(digests.items())]
    return "".join(lines).encode("ascii")


def build_envelope(payload_files: dict[str, bytes], identity: dict[str, Any]) -> dict[str, bytes]:
    """Deterministically derive the complete release asset set (payload bytes
    passed through untouched, plus manifest.json and checksums.txt) from the
    exact run-level payload bytes and the execution identity."""
    if not payload_files:
        raise ValueError("release envelope requires at least one payload file")
    entries: list[dict[str, Any]] = []
    payload_by_name: dict[str, bytes] = {}
    for rel_path in sorted(payload_files):
        pure = Path(rel_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"payload path {rel_path!r} escapes the payload root")
        name = pure.name
        if not _ASSET_NAME.fullmatch(name):
            raise ValueError(f"payload file name {name!r} is not release-asset safe")
        if name in (MANIFEST_NAME, CHECKSUMS_NAME):
            raise ValueError(f"payload file name {name!r} collides with an envelope asset")
        if name in payload_by_name:
            raise ValueError(f"duplicate payload basename {name!r}")
        data = payload_files[rel_path]
        if not data:
            raise ValueError(f"payload file {rel_path!r} is empty")
        payload_by_name[name] = data
        entries.append(
            {
                "byte_length": len(data),
                "name": name,
                "path": Path(rel_path).as_posix(),
                "sha256": sha256_hexdigest(data),
            }
        )
    manifest = {
        "artifact_name": identity["artifact_name"],
        "head_sha": identity["head_sha"],
        "payload_assets": entries,
        "repository": identity["repository"],
        "run_attempt": identity["run_attempt"],
        "schema": ENVELOPE_SCHEMA,
        "tag": identity["tag"],
        "workflow_path": WORKFLOW_PATH,
        "workflow_run_id": identity["run_id"],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    digests = {entry["name"]: entry["sha256"] for entry in entries}
    digests[MANIFEST_NAME] = sha256_hexdigest(manifest_bytes)
    assets = dict(payload_by_name)
    assets[MANIFEST_NAME] = manifest_bytes
    assets[CHECKSUMS_NAME] = _checksum_bytes(digests)
    return assets


def verify_envelope_assets(assets: dict[str, bytes], identity: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed verification of exact downloaded asset bytes against the
    execution identity: canonical manifest bytes, identity binding, exact
    asset set, per-asset digests, and the checksums file itself."""
    for required in (MANIFEST_NAME, CHECKSUMS_NAME):
        if required not in assets:
            raise ValueError(f"envelope asset {required!r} is missing")
    manifest_bytes = assets[MANIFEST_NAME]
    try:
        manifest = json.loads(manifest_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest bytes are not parseable canonical JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not a JSON object")
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("manifest bytes are not in canonical form")
    bindings = {
        "schema": ENVELOPE_SCHEMA,
        "repository": identity["repository"],
        "head_sha": identity["head_sha"],
        "workflow_run_id": identity["run_id"],
        "run_attempt": identity["run_attempt"],
        "artifact_name": identity["artifact_name"],
        "tag": identity["tag"],
        "workflow_path": WORKFLOW_PATH,
    }
    for field, expected in bindings.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"manifest field {field!r} is {manifest.get(field)!r}, not the bound {expected!r}"
            )
    entries = manifest.get("payload_assets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest lists no payload assets")
    digests: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest payload asset entry is not an object")
        name = entry.get("name")
        if not isinstance(name, str) or name in digests or name in (MANIFEST_NAME, CHECKSUMS_NAME):
            raise ValueError(f"manifest payload asset name {name!r} is invalid or duplicated")
        if name not in assets:
            raise ValueError(f"payload asset {name!r} is missing from the release")
        data = assets[name]
        size = entry.get("byte_length")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size != len(data):
            raise ValueError(f"payload asset {name!r} byte length contradicts its exact bytes")
        digest = entry.get("sha256")
        if digest != sha256_hexdigest(data):
            raise ValueError(f"payload asset {name!r} bytes contradict the manifest digest")
        digests[name] = str(digest)
    expected_assets = set(digests) | {MANIFEST_NAME, CHECKSUMS_NAME}
    if set(assets) != expected_assets:
        raise ValueError(
            f"release asset set {sorted(assets)} does not equal the manifest set {sorted(expected_assets)}"
        )
    digests[MANIFEST_NAME] = sha256_hexdigest(manifest_bytes)
    if assets[CHECKSUMS_NAME] != _checksum_bytes(digests):
        raise ValueError("checksums.txt contradicts the recomputed asset digests")
    return manifest


def release_digest_set(assets: dict[str, bytes]) -> set[str]:
    """SHA-256 digests of every published asset's exact bytes: the
    ``release_digests`` plane a run-level artifact payload must bind into."""
    return {sha256_hexdigest(data) for data in assets.values()}


def selection_is_stable(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for key in SELECTION_KEYS:
        if key not in before or key not in after:
            raise ValueError(f"selection snapshot is missing {key!r}")
        if before[key] != after[key]:
            return False
    return True


def attestation_verify_argv(
    asset_path: str, *, repository: str, head_sha: str, signer_workflow: str
) -> list[str]:
    return [
        "gh",
        "attestation",
        "verify",
        asset_path,
        "-R",
        repository,
        "--signer-workflow",
        signer_workflow,
        "--source-digest",
        head_sha,
        "--format",
        "json",
    ]


def release_verify_argv(tag: str, *, repository: str) -> list[str]:
    return ["gh", "release", "verify", tag, "-R", repository, "--format", "json"]


def release_verify_asset_argv(tag: str, asset_path: str, *, repository: str) -> list[str]:
    return ["gh", "release", "verify-asset", tag, asset_path, "-R", repository, "--format", "json"]


_READ_ONLY_GH_ACTIONS = {
    ("api",),
    ("attestation", "verify"),
    ("release", "download"),
    ("release", "verify"),
    ("release", "verify-asset"),
    ("release", "view"),
    ("run", "list"),
    ("run", "view"),
}
_GH_API_WRITE_FLAGS = {"-f", "-F", "--field", "--raw-field", "--input"}


def require_read_only_argv(argv: list[str]) -> None:
    """Reject any subprocess invocation that could mutate remote state. The
    helper's contract is read-only verification; publication belongs to the
    workflow's own reviewed steps."""
    if not argv or Path(argv[0]).name != "gh":
        raise ValueError(f"only read-only gh invocations are allowed, not {argv!r}")
    if len(argv) < 2:
        raise ValueError("gh invocation names no subcommand")
    group = argv[1]
    if group == "api":
        index = 2
        while index < len(argv):
            item = argv[index]
            if item in {"-X", "--method"}:
                if index + 1 >= len(argv) or argv[index + 1].upper() != "GET":
                    raise ValueError("gh api may only issue GET requests")
                index += 2
                continue
            if item.startswith("--method=") and item.split("=", 1)[1].upper() != "GET":
                raise ValueError("gh api may only issue GET requests")
            if (
                item in _GH_API_WRITE_FLAGS
                or item.startswith("--field=")
                or item.startswith("--input=")
            ):
                raise ValueError("gh api write-shaped flags are rejected")
            index += 1
        return
    action = (group, argv[2] if len(argv) > 2 else "")
    if action not in _READ_ONLY_GH_ACTIONS:
        raise ValueError(f"mutating or unsupported gh action rejected: {' '.join(action)}")


def _run_gh(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    require_read_only_argv(argv)
    return subprocess.run(argv, capture_output=True, check=False)


def _gh_api_json(endpoint: str, *, paginate: bool = False) -> Any:
    argv = ["gh", "api", "-H", API_VERSION_HEADER, endpoint]
    if paginate:
        argv += ["--paginate", "--slurp"]
    completed = _run_gh(argv)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        if "HTTP 404" in stderr:
            raise Blocked(f"{endpoint} is not visible: {stderr}")
        raise RuntimeError(f"gh api {endpoint} failed: {stderr}")
    return json.loads(completed.stdout)


def _checked_verification(argv: list[str], *, kind: str, blocked_marker: str = "") -> None:
    completed = _run_gh(argv)
    if completed.returncode == 0:
        return
    stderr = completed.stderr.decode("utf-8", "replace").strip()
    if blocked_marker and blocked_marker in stderr.lower():
        raise Blocked(f"{kind} is not attested yet: {stderr}")
    raise RuntimeError(f"{kind} verification failed: {stderr}")


def live_selection_snapshot(identity: dict[str, Any]) -> dict[str, Any]:
    """Requery main, workflow, run, attempt, and the run-level artifact; the
    caller compares two snapshots and restarts on any movement."""
    repository = identity["repository"]
    main_ref = _gh_api_json(f"repos/{repository}/git/ref/heads/main")
    workflow = _gh_api_json(f"repos/{repository}/actions/workflows/{Path(WORKFLOW_PATH).name}")
    if workflow.get("path") != WORKFLOW_PATH or workflow.get("state") != "active":
        raise RuntimeError(
            f"live workflow moved: {workflow.get('path')!r} {workflow.get('state')!r}"
        )
    run = _gh_api_json(f"repos/{repository}/actions/runs/{identity['run_id']}")
    if run.get("path") != WORKFLOW_PATH:
        raise RuntimeError("bound run does not belong to the OpenAPI workflow")
    if run.get("head_sha") != identity["head_sha"]:
        raise RuntimeError("bound run head SHA contradicts the envelope identity")
    latest_attempt = run.get("run_attempt")
    if latest_attempt != identity["run_attempt"]:
        raise Movement(
            f"run attempt {latest_attempt} supersedes bound attempt "
            f"{identity['run_attempt']}; restart verification at the newest execution"
        )
    pages = _gh_api_json(
        f"repos/{repository}/actions/runs/{identity['run_id']}/artifacts?per_page=100",
        paginate=True,
    )
    artifacts: list[dict[str, Any]] = []
    for page in pages if isinstance(pages, list) else [pages]:
        artifacts.extend(page.get("artifacts", []))
    matching = [item for item in artifacts if item.get("name") == identity["artifact_name"]]
    if len(matching) != 1:
        raise RuntimeError(
            f"run-level artifact {identity['artifact_name']!r} is not uniquely present "
            f"({len(matching)} matches)"
        )
    artifact = matching[0]
    size = artifact.get("size_in_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RuntimeError("run-level artifact lacks a nonempty payload")
    return {
        "main_sha": main_ref.get("object", {}).get("sha"),
        "workflow_id": workflow.get("id"),
        "run_id": run.get("id"),
        "run_attempt": latest_attempt,
        "conclusion": run.get("conclusion"),
        "artifact_id": artifact.get("id"),
        "artifact_size": size,
    }


def _resolve_tag_commit(repository: str, tag: str) -> str:
    ref = _gh_api_json(f"repos/{repository}/git/ref/tags/{tag}")
    obj = ref.get("object", {})
    if obj.get("type") == "tag":
        obj = _gh_api_json(f"repos/{repository}/git/tags/{obj.get('sha')}").get("object", {})
    return str(obj.get("sha"))


def _passing_rule_suite(repository: str, head_sha: str) -> dict[str, Any]:
    pages = _gh_api_json(
        f"repos/{repository}/rulesets/rule-suites"
        f"?ref=refs/heads/main&time_period=month&per_page=100",
        paginate=True,
    )
    suites: list[dict[str, Any]] = []
    for page in pages if isinstance(pages, list) else [pages]:
        suites.extend(page if isinstance(page, list) else [page])
    matching = [item for item in suites if item.get("after_sha") == head_sha]
    if not matching:
        raise Blocked(
            f"no rule-suite record for {head_sha} on refs/heads/main within the month window"
        )
    newest = max(matching, key=lambda item: item.get("id") or 0)
    if newest.get("result") != "pass":
        raise RuntimeError(
            f"rule suite {newest.get('id')} for {head_sha} concluded {newest.get('result')!r}"
        )
    return {"id": newest.get("id"), "result": "pass"}


def _emit(report: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")


def cmd_build(args: argparse.Namespace) -> int:
    identity = make_identity(
        repository=args.repository,
        head_sha=args.head_sha,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        artifact_name=args.artifact_name,
    )
    payload_root = Path(args.payload_dir)
    payload_files = {
        path.relative_to(payload_root).as_posix(): path.read_bytes()
        for path in sorted(payload_root.rglob("*"))
        if path.is_file()
    }
    assets = build_envelope(payload_files, identity)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError(f"output dir {output_dir} must be empty for a deterministic build")
    for name, data in sorted(assets.items()):
        (output_dir / name).write_bytes(data)
    _emit(
        {
            "status": "pass",
            "tag": identity["tag"],
            "asset_names": sorted(assets),
            "manifest_sha256": sha256_hexdigest(assets[MANIFEST_NAME]),
        }
    )
    return EXIT_PASS


def cmd_verify(args: argparse.Namespace) -> int:
    identity = make_identity(
        repository=args.repository,
        head_sha=args.head_sha,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        artifact_name=args.artifact_name,
    )
    repository = identity["repository"]
    tag = identity["tag"]
    signer_workflow = args.signer_workflow or f"{repository}/{WORKFLOW_PATH}"
    report: dict[str, Any] = {"tag": tag, "head_sha": identity["head_sha"]}
    try:
        immutability = _gh_api_json(f"repos/{repository}/immutable-releases")
        if immutability.get("enabled") is not True:
            raise RuntimeError("release immutability is not enabled; the envelope is not durable")
        before = live_selection_snapshot(identity)
        try:
            release = _gh_api_json(f"repos/{repository}/releases/tags/{tag}")
        except Blocked as exc:
            raise Blocked(f"release {tag} is not published yet") from exc
        if release.get("draft"):
            raise Blocked(f"release {tag} is still an unpublished draft")
        report["release_api_id"] = release.get("id")
        tag_commit = _resolve_tag_commit(repository, tag)
        if tag_commit != identity["head_sha"]:
            raise RuntimeError(
                f"tag {tag} resolves to {tag_commit}, not the bound head {identity['head_sha']}"
            )
        hosted = {
            str(asset.get("name")): asset
            for asset in release.get("assets", [])
            if isinstance(asset, dict) and isinstance(asset.get("name"), str)
        }
        with tempfile.TemporaryDirectory() as scratch:
            download = _run_gh(
                ["gh", "release", "download", tag, "--repo", repository, "--dir", scratch]
            )
            if download.returncode != 0:
                raise RuntimeError(
                    "release asset download failed: "
                    + download.stderr.decode("utf-8", "replace").strip()
                )
            files = {
                path.name: path.read_bytes()
                for path in sorted(Path(scratch).iterdir())
                if path.is_file()
            }
            manifest = verify_envelope_assets(files, identity)
            if set(hosted) != set(files):
                raise RuntimeError(
                    f"hosted asset listing {sorted(hosted)} contradicts downloaded assets {sorted(files)}"
                )
            for name, data in files.items():
                if hosted[name].get("size") != len(data):
                    raise RuntimeError(f"hosted size for {name!r} contradicts its exact bytes")
            _checked_verification(
                release_verify_argv(tag, repository=repository), kind=f"release {tag}"
            )
            for name in sorted(files):
                asset_path = str(Path(scratch) / name)
                _checked_verification(
                    release_verify_asset_argv(tag, asset_path, repository=repository),
                    kind=f"release asset {name}",
                )
                _checked_verification(
                    attestation_verify_argv(
                        asset_path,
                        repository=repository,
                        head_sha=identity["head_sha"],
                        signer_workflow=signer_workflow,
                    ),
                    kind=f"attestation for {name}",
                    blocked_marker="no attestations",
                )
        report["payload_assets"] = [entry["name"] for entry in manifest["payload_assets"]]
        report["rule_suite"] = _passing_rule_suite(repository, identity["head_sha"])
        after = live_selection_snapshot(identity)
        if not selection_is_stable(before, after):
            raise Movement("workflow/run/artifact/main selection moved during verification")
        report["superseded_by_newer_main"] = before["main_sha"] != identity["head_sha"]
        report["status"] = "pass"
        _emit(report)
        return EXIT_PASS
    except Blocked as exc:
        _emit({**report, "status": "blocked", "reason": str(exc)})
        return EXIT_BLOCKED
    except Movement as exc:
        _emit({**report, "status": "movement", "reason": str(exc)})
        return EXIT_MOVEMENT
    except (RuntimeError, ValueError) as exc:
        _emit({**report, "status": "fail", "reason": str(exc)})
        return EXIT_FAIL


DRY_RUN_HEAD_SHA = "0123456789abcdef0123456789abcdef01234567"
DRY_RUN_PAYLOADS = {
    "docs/api/openapi_generated.json": (
        b'{"info":{"title":"aragora dry-run fixture","version":"0.0.0"},'
        b'"openapi":"3.1.0","paths":{}}\n'
    ),
    "docs/api/openapi_generated.yaml": (
        b"info:\n  title: aragora dry-run fixture\n  version: 0.0.0\nopenapi: 3.1.0\npaths: {}\n"
    ),
    "sdk/typescript/src/openapi-types.ts": b"export type DryRunFixture = never;\n",
}


def dry_run_report() -> dict[str, Any]:
    identity = make_identity(
        repository="synaptent/aragora",
        head_sha=DRY_RUN_HEAD_SHA,
        run_id=1,
        run_attempt=1,
        artifact_name=f"{ARTIFACT_PREFIX}1-1-{DRY_RUN_HEAD_SHA}",
    )
    first = build_envelope(DRY_RUN_PAYLOADS, identity)
    second = build_envelope(dict(DRY_RUN_PAYLOADS), identity)
    if first != second:
        raise RuntimeError("envelope bytes are not deterministic")
    verify_envelope_assets(first, identity)
    tampered = dict(first)
    tampered["openapi_generated.json"] = tampered["openapi_generated.json"] + b" "
    try:
        verify_envelope_assets(tampered, identity)
    except ValueError:
        pass
    else:
        raise RuntimeError("tampered payload bytes were not rejected")
    snapshot = {key: index for index, key in enumerate(SELECTION_KEYS)}
    if not selection_is_stable(snapshot, dict(snapshot)):
        raise RuntimeError("stable selection was misreported as movement")
    if selection_is_stable(snapshot, {**snapshot, "main_sha": "moved"}):
        raise RuntimeError("main movement was not detected")
    return {
        "status": "pass",
        "deterministic": True,
        "tag": identity["tag"],
        "asset_names": sorted(first),
        "manifest_sha256": sha256_hexdigest(first[MANIFEST_NAME]),
        "checksums_sha256": sha256_hexdigest(first[CHECKSUMS_NAME]),
        "release_digests": sorted(release_digest_set(first)),
    }


def cmd_dry_run(_args: argparse.Namespace) -> int:
    _emit(dry_run_report())
    return EXIT_PASS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_identity_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--repository", required=True)
        sub.add_argument("--head-sha", required=True)
        sub.add_argument("--run-id", type=int, required=True)
        sub.add_argument("--run-attempt", type=int, required=True)
        sub.add_argument("--artifact-name", required=True)

    build = subparsers.add_parser("build", help="write deterministic envelope bytes locally")
    add_identity_args(build)
    build.add_argument("--payload-dir", required=True)
    build.add_argument("--output-dir", required=True)
    build.set_defaults(func=cmd_build)

    verify = subparsers.add_parser("verify", help="verify a published envelope read-only")
    add_identity_args(verify)
    verify.add_argument("--signer-workflow", default="")
    verify.set_defaults(func=cmd_verify)

    dry_run = subparsers.add_parser("dry-run", help="offline deterministic self-check")
    dry_run.set_defaults(func=cmd_dry_run)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (RuntimeError, ValueError) as exc:
        _emit({"status": "fail", "reason": str(exc)})
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
