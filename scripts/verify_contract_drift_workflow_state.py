#!/usr/bin/env python3
"""Read-only verifier for the live Contract Drift Governance workflow state."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import yaml

DEFAULT_REPOSITORY = "synaptent/aragora"
CANONICAL_WORKFLOW_ID = 233974988
CANONICAL_WORKFLOW_PATH = ".github/workflows/contract-drift-governance.yml"
CANONICAL_WORKFLOW_NAME = "Contract Drift Governance"
EXPECTED_PR_DELTA_APP_ID = 15368
LIVE_CHECK_NAMES = (
    "contract-drift-pr-delta",
    "contract-drift-main-receipt",
    "contract-drift-program-trajectory",
)
PRE_CUTOVER_REQUIRED_CHECKS = (
    ("lint", 15368),
    ("typecheck", 15368),
    ("sdk-parity", 15368),
    ("Generate & Validate", 15368),
    ("TypeScript SDK Type Check", 15368),
    ("aragora-merge-quorum", 15368),
)
MAX_PAGES = 10_000
DATE_SHARD_CAP = 1000


class VerificationError(ValueError):
    """The authenticated state did not satisfy the fail-closed contract."""


class ApiReader(Protocol):
    def get_json(self, endpoint: str) -> dict[str, Any]: ...

    def get_bytes(self, endpoint: str) -> bytes: ...


class GhApiReader:
    """Authenticated read-only GitHub transport backed by ``gh api``."""

    def __init__(self, *, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds

    def _run(self, endpoint: str, *, accept: str) -> bytes:
        try:
            proc = subprocess.run(
                ["gh", "api", "--method", "GET", "-H", f"Accept: {accept}", endpoint],
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise VerificationError(f"authenticated GitHub GET timed out: {endpoint}") from exc
        if proc.returncode != 0:
            error = proc.stderr.decode("utf-8", errors="replace").strip()
            raise VerificationError(f"authenticated GitHub GET failed for {endpoint}: {error}")
        return proc.stdout

    def get_json(self, endpoint: str) -> dict[str, Any]:
        raw = self._run(endpoint, accept="application/vnd.github+json")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerificationError(f"GitHub response is malformed for {endpoint}: {exc}") from exc
        if not isinstance(payload, dict):
            raise VerificationError(f"GitHub response is not an object: {endpoint}")
        return payload

    def get_bytes(self, endpoint: str) -> bytes:
        accept = (
            "application/vnd.github.raw+json"
            if "/contents/" in urlparse(endpoint).path
            else "application/vnd.github+json"
        )
        return self._run(endpoint, accept=accept)


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VerificationError(f"{label} is malformed")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} is malformed")
    return value


def _with_page(endpoint: str, page: int) -> str:
    parsed = urlparse(endpoint)
    query = [
        (key, value) for key, value in parse_qsl(parsed.query) if key not in {"page", "per_page"}
    ]
    query.extend((("per_page", "100"), ("page", str(page))))
    return urlunparse(parsed._replace(query=urlencode(query)))


def _paginate_collection(
    reader: ApiReader,
    endpoint: str,
    *,
    collection_key: str,
    identity: Callable[[dict[str, Any]], Any],
    label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    endpoints: list[str] = []
    reported_total: int | None = None
    page = 1
    while True:
        page_endpoint = _with_page(endpoint, page)
        endpoints.append(page_endpoint)
        payload = reader.get_json(page_endpoint)
        page_records = payload.get(collection_key)
        if not isinstance(page_records, list) or not all(
            isinstance(record, dict) for record in page_records
        ):
            raise VerificationError(f"{label} page is malformed: {page_endpoint}")
        if "total_count" not in payload:
            raise VerificationError(f"{label} total_count is missing")
        page_total = _require_int(payload["total_count"], f"{label} total_count")
        if reported_total is None:
            reported_total = page_total
        elif page_total != reported_total:
            raise VerificationError(f"{label} total_count moved between pages")
        existing = {identity(record) for record in records}
        page_identities = [identity(record) for record in page_records]
        if any(value is None for value in page_identities):
            raise VerificationError(f"{label} record identity is missing")
        if len(page_identities) != len(set(page_identities)) or existing.intersection(
            page_identities
        ):
            raise VerificationError(f"{label} pagination returned duplicate identities")
        records.extend(page_records)
        if len(page_records) < 100:
            break
        page += 1
        if page > MAX_PAGES:
            raise VerificationError(f"{label} pagination did not terminate")

    identities = [identity(record) for record in records]
    if any(value is None for value in identities):
        raise VerificationError(f"{label} record identity is missing")
    if len(identities) != len(set(identities)):
        raise VerificationError(f"{label} pagination returned duplicate identities")
    if reported_total != len(records):
        raise VerificationError(
            f"{label} records do not reconcile to total_count ({len(records)} != {reported_total})"
        )
    return records, endpoints


def paginate_runs(
    fetch_page: Callable[[str], list[dict[str, Any]]],
) -> tuple[list[dict], list[str]]:
    """Compatibility helper used by contract tests for unfiltered run pagination."""
    records: list[dict[str, Any]] = []
    endpoints: list[str] = []
    page = 1
    while True:
        endpoint = f"repos/OWNER/REPO/actions/runs?per_page=100&page={page}"
        endpoints.append(endpoint)
        payload = fetch_page(endpoint)
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError("paginated workflow-run payload is malformed")
        records.extend(payload)
        if len(payload) < 100:
            break
        page += 1
        if page > MAX_PAGES:
            raise ValueError("workflow-run pagination did not terminate")
    ids = [run.get("id") for run in records]
    if len(ids) != len(set(ids)):
        raise ValueError("paginated workflow runs returned duplicate record IDs")
    return records, endpoints


def plan_date_shards(
    daily_counts: Mapping[str, int], *, cap: int = DATE_SHARD_CAP
) -> list[tuple[str, str, int]]:
    """Plan disjoint inclusive date shards, each strictly below the API cap."""
    shards: list[tuple[str, str, int]] = []
    current: list[str] = []
    total = 0
    for day in sorted(daily_counts):
        count = _require_int(daily_counts[day], f"daily count for {day}")
        if count >= cap:
            raise ValueError(f"single-day run volume {count} defeats the {cap}-result window")
        if current and total + count >= cap:
            shards.append((current[0], current[-1], total))
            current, total = [], 0
        current.append(day)
        total += count
    if current:
        shards.append((current[0], current[-1], total))
    for (_, end_a, _), (start_b, _, _) in zip(shards, shards[1:]):
        if not end_a < start_b:
            raise ValueError("date shards overlap")
    if any(count >= cap for _, _, count in shards):
        raise ValueError(f"date shard reaches the {cap}-result cap")
    return shards


def reconcile_run_ids(shards: list[dict], *, reported_total: int) -> list[int]:
    ids = [run["id"] for shard in shards for run in shard["workflow_runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("sharded run capture duplicated run IDs")
    for shard in shards:
        if shard["total_count"] != len(shard["workflow_runs"]):
            raise ValueError("shard total_count does not reconcile with captured runs")
    if reported_total != len(ids):
        raise ValueError("run IDs do not reconcile to the reported total_count")
    return sorted(ids)


def reconcile_date_shards(
    shards: list[dict[str, Any]], *, start_date: str, end_date: str
) -> list[int]:
    """Validate externally queried filtered-history shards and reconcile their IDs."""
    previous_end: str | None = None
    reported_total = 0
    for shard in shards:
        start = _require_string(shard.get("start"), "date shard start")
        end = _require_string(shard.get("end"), "date shard end")
        total = _require_int(shard.get("total_count"), "date shard total_count")
        if total >= DATE_SHARD_CAP:
            raise VerificationError("date shard reaches the 1000-result cap")
        if start > end or (previous_end is not None and not previous_end < start):
            raise VerificationError("date shards overlap or are out of order")
        if previous_end is None and start != start_date:
            raise VerificationError("date shards leave a leading gap")
        if previous_end is not None:
            try:
                expected_start = (date.fromisoformat(previous_end) + timedelta(days=1)).isoformat()
            except ValueError as exc:
                raise VerificationError("date shard boundary is malformed") from exc
            if start != expected_start:
                raise VerificationError("date shards leave an internal gap")
        records = shard.get("workflow_runs")
        if not isinstance(records, list) or not all(isinstance(run, dict) for run in records):
            raise VerificationError("date shard workflow_runs are malformed")
        if len(records) != total:
            raise VerificationError("date shard total_count does not reconcile")
        reported_total += total
        previous_end = end
    if not shards or previous_end != end_date:
        raise VerificationError("date shards leave a trailing gap")
    return reconcile_run_ids(shards, reported_total=reported_total)


def attempt_numbers(run: Mapping[str, Any]) -> list[int]:
    attempt = run.get("run_attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("run_attempt is malformed")
    return list(range(1, attempt + 1))


def attempt_jobs_endpoint(run_id: int, attempt: int) -> str:
    if attempt < 1:
        raise ValueError("attempt numbers start at 1")
    return f"repos/OWNER/REPO/actions/runs/{run_id}/attempts/{attempt}/jobs"


def validate_attempt_jobs(endpoint: str, jobs: list[dict], *, attempt: int) -> None:
    if f"/attempts/{attempt}/jobs" not in endpoint:
        raise ValueError("jobs endpoint is not attempt-specific")
    for job in jobs:
        if job.get("run_attempt") != attempt:
            raise ValueError("job record is not attempt-specific")
        check_url = str(job.get("check_url") or job.get("check_run_url") or "")
        if not check_url:
            raise ValueError("job check URL is missing")
        if "check_url" in job and f"/attempts/{attempt}" not in check_url:
            raise ValueError("check URL is not pinned to the requested attempt")


def validate_run_artifact(
    artifact: Mapping[str, Any], *, head_sha: str, release_digests: set[str]
) -> str:
    name = artifact.get("name")
    if not isinstance(name, str) or not name.endswith(f"-{head_sha}"):
        raise ValueError("run artifact name is not SHA-bound")
    size = artifact.get("size_in_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("run artifact lacks a nonempty payload")
    if artifact.get("payload_sha256") not in release_digests:
        raise ValueError("run artifact payload is not bound to the immutable release")
    return name


def verify_protection_cutover(
    before: dict,
    after: dict,
    *,
    added_context: str = "contract-drift-pr-delta",
    expected_app_id: int = EXPECTED_PR_DELTA_APP_ID,
) -> list[tuple[str, int]]:
    """Validate a historical before/after branch-protection cutover capture."""
    if before["strict"] != after["strict"]:
        raise ValueError("cutover changed required_status_checks.strict")
    before_tuples = [tuple(item) for item in before["checks"]]
    after_tuples = [tuple(item) for item in after["checks"]]
    if len(before_tuples) != len(set(before_tuples)) or len(after_tuples) != len(set(after_tuples)):
        raise ValueError("required check tuples are duplicated")
    removed = set(before_tuples) - set(after_tuples)
    if removed:
        raise ValueError("cutover removed or mutated a required (context, app_id) tuple")
    added = set(after_tuples) - set(before_tuples)
    if added != {(added_context, expected_app_id)}:
        if any(context == added_context and app_id != expected_app_id for context, app_id in added):
            raise ValueError("pr-delta was added with the wrong expected app identity")
        raise ValueError(
            "cutover must add exactly the pr-delta context with the expected app identity"
        )
    return sorted(set(after_tuples))


def paginated_protection_checks(
    pages: list[list[tuple[str, int]]],
) -> list[tuple[str, int]]:
    checks: list[tuple[str, int]] = []
    for page in pages:
        if len(page) > 100:
            raise ValueError("paginated protection page exceeds per_page=100")
        checks.extend((str(context), int(app_id)) for context, app_id in page)
    if len(checks) != len(set(checks)):
        raise ValueError("paginated protection capture returned duplicate tuples")
    return checks


def required_check_satisfied(
    check_runs: list[dict], *, context: str, app_id: int = EXPECTED_PR_DELTA_APP_ID
) -> bool:
    return any(
        run.get("name") == context
        and run.get("app_id") == app_id
        and run.get("conclusion") == "success"
        for run in check_runs
    )


def _workflow_job_names(source: bytes) -> set[str]:
    try:
        doc = yaml.load(source.decode("utf-8"), Loader=yaml.BaseLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise VerificationError(f"workflow source is malformed: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
        raise VerificationError("workflow source lacks jobs")
    return {
        str(job.get("name", job_id)) for job_id, job in doc["jobs"].items() if isinstance(job, dict)
    }


def _canonical_workflow(
    reader: ApiReader,
    *,
    repo: str,
    canonical_workflow_id: int,
) -> tuple[dict[str, Any], list[str]]:
    workflows, endpoints = _paginate_collection(
        reader,
        f"repos/{repo}/actions/workflows",
        collection_key="workflows",
        identity=lambda workflow: workflow.get("id"),
        label="workflow discovery",
    )
    candidates: list[dict[str, Any]] = []
    for workflow in workflows:
        path = workflow.get("path")
        if not isinstance(path, str) or not path.startswith(".github/workflows/"):
            continue
        name = str(workflow.get("name") or "")
        if path == CANONICAL_WORKFLOW_PATH or name == CANONICAL_WORKFLOW_NAME:
            candidates.append(workflow)
    if len(candidates) != 1:
        raise VerificationError(
            "exactly one registered workflow may emit the live check names "
            f"(found {len(candidates)})"
        )
    workflow = candidates[0]
    workflow_id = _require_int(workflow.get("id"), "canonical workflow ID", minimum=1)
    if workflow_id != canonical_workflow_id:
        raise VerificationError(
            f"canonical workflow ID moved ({workflow_id} != {canonical_workflow_id})"
        )
    if workflow.get("path") != CANONICAL_WORKFLOW_PATH:
        raise VerificationError("canonical workflow path moved")
    if workflow.get("name") != CANONICAL_WORKFLOW_NAME:
        raise VerificationError("canonical workflow name moved")
    if workflow.get("state") != "active":
        raise VerificationError("canonical workflow must be active")
    source = reader.get_bytes(f"repos/{repo}/contents/{CANONICAL_WORKFLOW_PATH}?ref=main")
    if _workflow_job_names(source) != set(LIVE_CHECK_NAMES):
        raise VerificationError("canonical workflow does not expose the exact three live checks")
    return workflow, endpoints


def _main_sha(reader: ApiReader, repo: str) -> str:
    payload = reader.get_json(f"repos/{repo}/git/ref/heads/main")
    obj = payload.get("object")
    if not isinstance(obj, dict):
        raise VerificationError("main ref object is malformed")
    sha = _require_string(obj.get("sha"), "main SHA")
    if len(sha) != 40:
        raise VerificationError("main SHA is not full length")
    return sha


def _workflow_runs(
    reader: ApiReader, *, repo: str, workflow_id: int
) -> tuple[list[dict[str, Any]], list[str]]:
    return _paginate_collection(
        reader,
        f"repos/{repo}/actions/workflows/{workflow_id}/runs",
        collection_key="workflow_runs",
        identity=lambda run: run.get("id"),
        label="workflow-run discovery",
    )


def _run_key(run: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        _require_string(run.get("run_started_at"), "run_started_at"),
        _require_int(run.get("id"), "run ID", minimum=1),
        attempt_numbers(run)[-1],
    )


def _selected_main_run(
    runs: Iterable[dict[str, Any]], *, workflow_id: int, main_sha: str
) -> dict[str, Any]:
    eligible = [
        run
        for run in runs
        if run.get("workflow_id") == workflow_id
        and run.get("path") == CANONICAL_WORKFLOW_PATH
        and run.get("head_branch") == "main"
        and run.get("head_sha") == main_sha
        and run.get("event") in {"push", "schedule", "workflow_dispatch"}
    ]
    if not eligible:
        raise VerificationError("no canonical main workflow execution matches current main")
    return max(eligible, key=_run_key)


def _check_summary(
    reader: ApiReader,
    *,
    endpoint: str,
    job: Mapping[str, Any],
    run: Mapping[str, Any],
    attempt: int,
) -> dict[str, Any]:
    if job.get("run_attempt") != attempt:
        raise VerificationError("job record is not attempt-specific")
    check_url = _require_string(job.get("check_run_url"), "attempt-specific check URL")
    check = reader.get_json(check_url)
    if check.get("id") is None or check.get("name") != job.get("name"):
        raise VerificationError("attempt job/check identity mismatch")
    if check.get("head_sha") != run.get("head_sha"):
        raise VerificationError("attempt check is bound to another SHA")
    app = check.get("app")
    app_id = app.get("id") if isinstance(app, dict) else None
    if app_id != EXPECTED_PR_DELTA_APP_ID:
        raise VerificationError("attempt check is not authenticated by the GitHub Actions app")
    details_url = _require_string(check.get("details_url"), "attempt check details URL")
    if f"/runs/{run['id']}/" not in details_url:
        raise VerificationError("attempt check details URL is bound to another run")
    return {
        "job_id": _require_int(job.get("id"), "job ID", minimum=1),
        "check_run_id": _require_int(check.get("id"), "check-run ID", minimum=1),
        "app_id": app_id,
        "status": check.get("status"),
        "conclusion": check.get("conclusion"),
        "check_url": check_url,
        "details_url": details_url,
        "jobs_endpoint": endpoint,
    }


def _attempts(reader: ApiReader, *, repo: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    run_id = _require_int(run.get("id"), "run ID", minimum=1)
    for attempt in attempt_numbers(run):
        endpoint = f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"
        jobs, _ = _paginate_collection(
            reader,
            endpoint,
            collection_key="jobs",
            identity=lambda job: job.get("id"),
            label=f"attempt {attempt} jobs",
        )
        try:
            validate_attempt_jobs(endpoint, jobs, attempt=attempt)
        except ValueError as exc:
            raise VerificationError(str(exc)) from exc
        names = [job.get("name") for job in jobs]
        if len(names) != len(set(names)):
            raise VerificationError("attempt jobs contain duplicate names")
        if set(names) != set(LIVE_CHECK_NAMES):
            raise VerificationError("attempt jobs do not expose the exact three live checks")
        checks = {
            str(job["name"]): _check_summary(
                reader,
                endpoint=endpoint,
                job=job,
                run=run,
                attempt=attempt,
            )
            for job in jobs
        }
        result.append({"attempt": attempt, "jobs_endpoint": endpoint, "checks": checks})
    return result


def _artifact_payload(raw_zip: bytes, *, expected_name: str) -> tuple[bytes, dict[str, Any]]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            if names != [expected_name]:
                raise VerificationError(
                    "run artifact zip does not contain exactly the expected file"
                )
            raw_payload = archive.read(names[0])
    except (zipfile.BadZipFile, KeyError) as exc:
        raise VerificationError(f"run artifact zip is malformed: {exc}") from exc
    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"run artifact payload is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationError("run artifact payload is not an object")
    return raw_payload, payload


def _payload_source_sha(payload: Mapping[str, Any]) -> str | None:
    direct = payload.get("source_sha")
    if isinstance(direct, str):
        return direct
    program = payload.get("program")
    if isinstance(program, dict) and isinstance(program.get("source_sha"), str):
        return str(program["source_sha"])
    return None


def _artifacts(reader: ApiReader, *, repo: str, run: Mapping[str, Any]) -> list[dict[str, Any]]:
    run_id = _require_int(run.get("id"), "run ID", minimum=1)
    artifacts, _ = _paginate_collection(
        reader,
        f"repos/{repo}/actions/runs/{run_id}/artifacts",
        collection_key="artifacts",
        identity=lambda artifact: artifact.get("id"),
        label="run artifacts",
    )
    head_sha = _require_string(run.get("head_sha"), "run head SHA")
    expected = {
        f"contract-drift-main-receipt-{head_sha}": "contract-drift-main-receipt.json",
        f"contract-drift-program-trajectory-{head_sha}": ("contract-drift-program-trajectory.json"),
    }
    selected = [artifact for artifact in artifacts if artifact.get("name") in expected]
    if {artifact.get("name") for artifact in selected} != set(expected):
        raise VerificationError("run-level receipt/trajectory artifacts are incomplete")
    summaries: list[dict[str, Any]] = []
    for artifact in selected:
        if artifact.get("expired") is not False:
            raise VerificationError("run artifact is expired")
        workflow_run = artifact.get("workflow_run")
        if (
            not isinstance(workflow_run, dict)
            or workflow_run.get("id") != run_id
            or workflow_run.get("head_sha") != head_sha
        ):
            raise VerificationError("run artifact is bound to another run or SHA")
        artifact_id = _require_int(artifact.get("id"), "artifact ID", minimum=1)
        name = str(artifact["name"])
        raw_payload, payload = _artifact_payload(
            reader.get_bytes(f"repos/{repo}/actions/artifacts/{artifact_id}/zip"),
            expected_name=expected[name],
        )
        if _payload_source_sha(payload) != head_sha:
            raise VerificationError("artifact payload is bound to another source SHA")
        summaries.append(
            {
                "id": artifact_id,
                "name": name,
                "size_in_bytes": _require_int(
                    artifact.get("size_in_bytes"), "artifact size", minimum=1
                ),
                "payload_sha256": hashlib.sha256(raw_payload).hexdigest(),
                "payload_status": payload.get("status"),
            }
        )
    return sorted(summaries, key=lambda item: item["id"])


def _branch_protection(reader: ApiReader, repo: str) -> dict[str, Any]:
    payload = reader.get_json(f"repos/{repo}/branches/main/protection/required_status_checks")
    if not isinstance(payload.get("strict"), bool):
        raise VerificationError("branch-protection strict is malformed")
    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, list) or not all(isinstance(item, dict) for item in raw_checks):
        raise VerificationError("branch-protection checks are malformed")
    checks = [
        (
            _require_string(item.get("context"), "required-check context"),
            _require_int(item.get("app_id"), "required-check app_id", minimum=1),
        )
        for item in raw_checks
    ]
    if len(checks) != len(set(checks)):
        raise VerificationError("branch-protection check tuples are duplicated")
    expected_before = set(PRE_CUTOVER_REQUIRED_CHECKS)
    actual = set(checks)
    expected_after = expected_before | {("contract-drift-pr-delta", EXPECTED_PR_DELTA_APP_ID)}
    if actual != expected_before and actual != expected_after:
        raise VerificationError("branch protection does not match a valid cutover state")
    phase = "after" if actual == expected_after else "before"
    return {"strict": payload["strict"], "checks": sorted(checks), "cutover_phase": phase}


@dataclass(frozen=True)
class _Snapshot:
    main_sha: str
    run: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "run_started_at": self.run["run_started_at"],
            "run_id": self.run["id"],
            "run_attempt": self.run["run_attempt"],
        }


def _snapshot(reader: ApiReader, *, repo: str, workflow_id: int) -> tuple[_Snapshot, list[str]]:
    main_sha = _main_sha(reader, repo)
    runs, endpoints = _workflow_runs(reader, repo=repo, workflow_id=workflow_id)
    run = _selected_main_run(runs, workflow_id=workflow_id, main_sha=main_sha)
    source = reader.get_bytes(
        f"repos/{repo}/contents/{CANONICAL_WORKFLOW_PATH}?ref={run['head_sha']}"
    )
    if _workflow_job_names(source) != set(LIVE_CHECK_NAMES):
        raise VerificationError("selected run workflow source does not expose exact live checks")
    return _Snapshot(main_sha=main_sha, run=run), endpoints


def _same_snapshot(left: _Snapshot, right: _Snapshot) -> bool:
    fields = ("id", "run_attempt", "run_started_at", "status", "conclusion", "head_sha")
    return left.main_sha == right.main_sha and all(
        left.run.get(field) == right.run.get(field) for field in fields
    )


def verify_workflow_state(
    reader: ApiReader,
    *,
    repo: str = DEFAULT_REPOSITORY,
    canonical_workflow_id: int = CANONICAL_WORKFLOW_ID,
) -> dict[str, Any]:
    """Verify complete live topology/history/protection state without mutations."""
    workflow, workflow_pages = _canonical_workflow(
        reader,
        repo=repo,
        canonical_workflow_id=canonical_workflow_id,
    )
    before, run_pages = _snapshot(reader, repo=repo, workflow_id=canonical_workflow_id)
    attempts = _attempts(reader, repo=repo, run=before.run)
    artifacts = _artifacts(reader, repo=repo, run=before.run)
    protection = _branch_protection(reader, repo)
    after, _ = _snapshot(reader, repo=repo, workflow_id=canonical_workflow_id)
    if not _same_snapshot(before, after):
        raise VerificationError(
            "current main or newest workflow execution moved during verification"
        )

    selected_attempt = attempts[-1]
    return {
        "repository": repo,
        "workflow": {key: workflow[key] for key in ("id", "name", "path", "state")},
        "workflow_pages": workflow_pages,
        "main_sha": before.main_sha,
        "selection": {
            "identity": before.identity,
            "status": before.run.get("status"),
            "conclusion": before.run.get("conclusion"),
            "attempts": attempts,
            "selected_attempt": selected_attempt,
            "artifacts": artifacts,
            "run_pages": run_pages,
        },
        "branch_protection": protection,
        "stable_requery": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify live Contract Drift Governance workflow state read-only."
    )
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    parser.add_argument("--workflow-id", type=int, default=CANONICAL_WORKFLOW_ID)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_workflow_state(
            GhApiReader(timeout_seconds=args.timeout_seconds),
            repo=args.repo,
            canonical_workflow_id=args.workflow_id,
        )
    except VerificationError as exc:
        print(f"contract drift workflow verification failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        identity = result["selection"]["identity"]
        print(
            "verified "
            f"{result['workflow']['path']} id={result['workflow']['id']} "
            f"main={result['main_sha']} run={identity['run_id']} "
            f"attempt={identity['run_attempt']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
