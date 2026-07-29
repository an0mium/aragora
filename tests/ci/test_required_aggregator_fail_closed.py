"""Guard the required-check aggregators against fail-OPEN conclusions.

Every required context in this repo is produced by the same three-job shape::

    changes / scope   -> classifies whether the work is in scope (a boolean output)
    <name>-run        -> the real worker, gated on `if: needs.<classifier>.outputs.<flag> == 'true'`
    <name>            -> the *required* aggregator, `if: always()`, reporting the verdict

The aggregator is the job whose name lands on the branch-protection list, so it
alone decides whether the required context is green. Historically each one
rejected a single worker result::

    if [[ "${{ needs.typecheck-run.result }}" == "failure" ]]; then exit 1; fi

That is fail-OPEN. GitHub's `needs.<job>.result` is one of `success`, `failure`,
`cancelled`, or `skipped`, and a job that is cancelled, timed out, or never
scheduled reports something *other* than `failure` — so it was converted into a
green required context.

This is not hypothetical. Issue #9084 records the exact instance: PR #8939 at head
`f8335380d1c7b25f2a6cb433e786e3e2b343a130`, Lint run 29059352439, where worker job
86257629023 was cancelled mid-typecheck and required aggregator job 86257810837
still succeeded. The typecheck the PR existed to validate never ran, and the gate
said it had.

The `skipped` result is the subtle half. A skip is legitimate *only* when the
classifier explicitly said the worker was out of scope. A skip while the
classifier said `true` means the worker was required and never ran — which must
be red, not green. So the aggregator has to consult the classifier's output, not
just the worker's result; the classifier is already in `needs` but its verdict
was never read.

## Why this test executes the script instead of matching it

Two guards in this repo shipped green while blind to the very thing they guarded:
the merge-quorum vocabulary guard (#9640) matched assignment syntax and missed the
dict-literal form, and the ssh-stdin guard matched `-n` inside a *remote* command
(`tail -n 1`) and so passed against the un-fixed workflow. Both were string
matching a proxy for the behavior.

This guard runs the aggregator's actual shell body under simulated `needs`
results and asserts the exit status, so it tests the semantics that GitHub will
execute. It is deliberately agnostic about *how* the fix is written: it resolves
`${{ }}` expressions whether they appear inline in `run:` or in the step's `env:`
block, so a rewrite that moves interpolation into `env:` (which is also the safer
quoting posture) stays covered.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


@dataclass(frozen=True)
class Aggregator:
    """One required-check aggregator and the classifier contract it must honor."""

    workflow: str
    job: str
    worker: str
    classifier: str
    scope_flag: str
    required_context: bool
    # quality-smoke additionally gates its worker on `!github.event.pull_request.draft`,
    # so on a *draft* PR an in-scope skip is legitimate there and nowhere else.
    defers_on_draft: bool = False


AGGREGATORS: tuple[Aggregator, ...] = (
    Aggregator("lint.yml", "lint", "lint-run", "changes", "python", True),
    Aggregator("lint.yml", "typecheck", "typecheck-run", "changes", "python", True),
    Aggregator("openapi.yml", "generate", "generate-run", "scope", "run_openapi", True),
    Aggregator("sdk-parity.yml", "sdk-parity", "sdk-parity-run", "changes", "relevant", True),
    Aggregator(
        "sdk-test.yml", "typescript-sdk", "typescript-sdk-run", "changes", "run_typescript", True
    ),
    Aggregator(
        "quality-smoke.yml",
        "quality-smoke",
        "quality-smoke-run",
        "changes",
        "quality",
        required_context=False,
        defers_on_draft=True,
    ),
)

# GitHub only ever reports these four in `needs.<job>.result`, but a required gate
# should also refuse anything it does not recognize rather than defaulting to green.
NON_SUCCESS_WORKER_RESULTS = ("failure", "cancelled", "timed_out", "action_required", "stale")

_EXPRESSION = re.compile(r"\$\{\{\s*(?P<body>[^}]+?)\s*\}\}")
_RESULT_REF = re.compile(r"^needs\.(?P<job>[A-Za-z0-9_-]+)\.result$")
_OUTPUT_REF = re.compile(r"^needs\.(?P<job>[A-Za-z0-9_-]+)\.outputs\.(?P<name>[A-Za-z0-9_-]+)$")
# The one non-`needs` signal an aggregator may legitimately read: whether the PR is a
# draft. Matching the whole `event_name == 'pull_request' && ....draft || 'false'`
# idiom rather than the bare field keeps any *other* github-context read unsupported,
# so a future aggregator cannot smuggle in an unsimulated input.
_DRAFT_REF = re.compile(
    r"^github\.event_name == 'pull_request' &&\s*"
    r"github\.event\.pull_request\.draft \|\| 'false'$"
)


def _load_job(spec: Aggregator) -> dict:
    workflow = yaml.safe_load((WORKFLOWS_DIR / spec.workflow).read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert spec.job in jobs, f"{spec.workflow}: aggregator job '{spec.job}' is gone"
    return jobs[spec.job]


def _resolve(
    expression: str,
    spec: Aggregator,
    worker_result: str,
    classifier_result: str,
    scope_flag: str,
    is_draft: str,
) -> str:
    """Resolve a single `${{ ... }}` body against the simulated run state."""
    if _DRAFT_REF.match(" ".join(expression.split())):
        assert spec.defers_on_draft, (
            f"{spec.workflow}::{spec.job} reads the draft flag but is not declared "
            "defers_on_draft — a gate that silently defers on drafts is a fail-open path."
        )
        return is_draft

    result_match = _RESULT_REF.match(expression)
    if result_match:
        job = result_match.group("job")
        if job == spec.worker:
            return worker_result
        if job == spec.classifier:
            return classifier_result
        raise AssertionError(
            f"{spec.workflow}::{spec.job} reads needs.{job}.result, which this guard "
            "does not simulate — extend the Aggregator spec so the new dependency is covered."
        )

    output_match = _OUTPUT_REF.match(expression)
    if output_match:
        job, name = output_match.group("job"), output_match.group("name")
        if job == spec.classifier and name == spec.scope_flag:
            return scope_flag
        raise AssertionError(
            f"{spec.workflow}::{spec.job} reads needs.{job}.outputs.{name}, which this guard "
            "does not simulate — extend the Aggregator spec so the new signal is covered."
        )

    raise AssertionError(
        f"{spec.workflow}::{spec.job} uses unsupported expression '{expression}'. "
        "A required aggregator must decide only from its dependencies' results and outputs."
    )


def _run_aggregator(
    spec: Aggregator,
    *,
    worker_result: str,
    classifier_result: str,
    scope_flag: str,
    is_draft: str = "false",
) -> subprocess.CompletedProcess[str]:
    """Execute the aggregator's real shell body under a simulated run state."""
    job = _load_job(spec)
    steps = [step for step in job.get("steps", []) if "run" in step]
    assert steps, f"{spec.workflow}::{spec.job} has no run steps to evaluate"

    def resolve_all(text: str) -> str:
        return _EXPRESSION.sub(
            lambda m: _resolve(
                m.group("body"), spec, worker_result, classifier_result, scope_flag, is_draft
            ),
            text,
        )

    # Each `run:` is a separate shell; GitHub stops the job at the first non-zero
    # step, so `&&`-chaining reproduces that short-circuit faithfully.
    script_parts: list[str] = []
    env: dict[str, str] = {"GITHUB_STEP_SUMMARY": "/dev/null", "PATH": "/usr/bin:/bin"}
    for step in steps:
        for key, value in (step.get("env") or {}).items():
            env[key] = resolve_all(str(value))
        script_parts.append(resolve_all(step["run"]))

    script = "\n".join(f"(\n{part}\n) || exit $?" for part in script_parts)
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60
    )


def _ids(specs: tuple[Aggregator, ...]) -> list[str]:
    return [f"{s.workflow}::{s.job}" for s in specs]


@pytest.mark.parametrize("spec", AGGREGATORS, ids=_ids(AGGREGATORS))
def test_in_scope_success_is_green(spec: Aggregator) -> None:
    """The honest pass: classifier says in scope, worker succeeded."""
    proc = _run_aggregator(
        spec, worker_result="success", classifier_result="success", scope_flag="true"
    )
    assert proc.returncode == 0, (
        f"{spec.workflow}::{spec.job} rejected a legitimate success "
        f"(exit {proc.returncode}).\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("spec", AGGREGATORS, ids=_ids(AGGREGATORS))
def test_out_of_scope_skip_is_green(spec: Aggregator) -> None:
    """The honest skip: classifier explicitly said the worker was not required."""
    proc = _run_aggregator(
        spec, worker_result="skipped", classifier_result="success", scope_flag="false"
    )
    assert proc.returncode == 0, (
        f"{spec.workflow}::{spec.job} rejected a legitimate scope skip "
        f"(exit {proc.returncode}).\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize("spec", AGGREGATORS, ids=_ids(AGGREGATORS))
@pytest.mark.parametrize("worker_result", NON_SUCCESS_WORKER_RESULTS)
def test_incomplete_worker_fails_closed(spec: Aggregator, worker_result: str) -> None:
    """A worker that did not succeed must never yield a green required context.

    `cancelled` is the #9084 instance verbatim; `timed_out` and the rest are the
    same hole with a different label.
    """
    proc = _run_aggregator(
        spec, worker_result=worker_result, classifier_result="success", scope_flag="true"
    )
    assert proc.returncode != 0, (
        f"{spec.workflow}::{spec.job} reported SUCCESS while its worker "
        f"'{spec.worker}' was '{worker_result}'. A required gate that passes without "
        f"checking is worse than no gate (#9084).\n{proc.stdout}"
    )


@pytest.mark.parametrize("spec", AGGREGATORS, ids=_ids(AGGREGATORS))
def test_skip_while_in_scope_fails_closed(spec: Aggregator) -> None:
    """A skip is only honest when the classifier said out-of-scope.

    Worker skipped while the scope flag is 'true' on a non-draft PR means the check was
    required and never ran — the failure mode #9084's acceptance criteria calls out
    explicitly. Draft deferral is covered separately below.
    """
    proc = _run_aggregator(
        spec,
        worker_result="skipped",
        classifier_result="success",
        scope_flag="true",
        is_draft="false",
    )
    assert proc.returncode != 0, (
        f"{spec.workflow}::{spec.job} reported SUCCESS while '{spec.worker}' was skipped "
        f"even though the classifier said it was in scope.\n{proc.stdout}"
    )


@pytest.mark.parametrize(
    "spec",
    [s for s in AGGREGATORS if s.defers_on_draft],
    ids=_ids(tuple(s for s in AGGREGATORS if s.defers_on_draft)),
)
def test_draft_deferral_is_green_but_only_for_drafts(spec: Aggregator) -> None:
    """The one legitimate in-scope skip: the worker itself declines to run on drafts."""
    proc = _run_aggregator(
        spec,
        worker_result="skipped",
        classifier_result="success",
        scope_flag="true",
        is_draft="true",
    )
    assert proc.returncode == 0, (
        f"{spec.workflow}::{spec.job} rejected a legitimate draft deferral "
        f"(exit {proc.returncode}).\n{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.parametrize(
    "spec",
    [s for s in AGGREGATORS if s.defers_on_draft],
    ids=_ids(tuple(s for s in AGGREGATORS if s.defers_on_draft)),
)
def test_draft_does_not_excuse_an_incomplete_worker(spec: Aggregator) -> None:
    """Draft deferral must not become a blanket amnesty.

    If the worker actually ran and was cancelled, `draft` is not the reason it is
    missing, and the gate must still fail closed.
    """
    proc = _run_aggregator(
        spec,
        worker_result="cancelled",
        classifier_result="success",
        scope_flag="true",
        is_draft="true",
    )
    assert proc.returncode != 0, (
        f"{spec.workflow}::{spec.job} reported SUCCESS for a cancelled worker just "
        f"because the PR is a draft.\n{proc.stdout}"
    )


@pytest.mark.parametrize("spec", AGGREGATORS, ids=_ids(AGGREGATORS))
@pytest.mark.parametrize("classifier_result", ["failure", "cancelled", "skipped", "timed_out"])
def test_broken_classifier_fails_closed(spec: Aggregator, classifier_result: str) -> None:
    """If classification did not complete, its scope verdict cannot be trusted.

    An empty scope flag is what GitHub actually supplies when the classifier never
    produced outputs, so a gate that treats "not 'true'" as "out of scope" would
    silently go green on classifier failure.
    """
    proc = _run_aggregator(
        spec, worker_result="skipped", classifier_result=classifier_result, scope_flag=""
    )
    assert proc.returncode != 0, (
        f"{spec.workflow}::{spec.job} reported SUCCESS while its classifier "
        f"'{spec.classifier}' was '{classifier_result}' and produced no scope verdict.\n"
        f"{proc.stdout}"
    )


# `aragora-review` is fail-open *on purpose*: it downgrades a non-success worker to a
# `::warning::` and exits 0, and it is not a required context. Excluding it keeps this
# guard focused on gates that claim to block. The exclusion is not free — the companion
# assertion below re-fires if the job ever stops declaring itself advisory, so it cannot
# quietly become a required gate while sitting on the exemption list.
ADVISORY_AGGREGATORS: dict[tuple[str, str], str] = {
    ("aragora-review-gate.yml", "aragora-review"): "not blocking merge",
}


@pytest.mark.parametrize(
    ("location", "advisory_marker"), sorted(ADVISORY_AGGREGATORS.items()), ids=lambda v: str(v)
)
def test_advisory_exemption_still_declares_itself_advisory(
    location: tuple[str, str], advisory_marker: str
) -> None:
    """An exempted aggregator must keep saying out loud that it does not block."""
    workflow_name, job_name = location
    workflow = yaml.safe_load((WORKFLOWS_DIR / workflow_name).read_text(encoding="utf-8"))
    job = workflow["jobs"][job_name]
    body = " ".join(str(step.get("run", "")) for step in job.get("steps", []))
    assert advisory_marker in body, (
        f"{workflow_name}::{job_name} is exempt from the fail-closed guard because it is "
        f"advisory, but it no longer says '{advisory_marker}'. If it now blocks merges, "
        "move it into AGGREGATORS instead of leaving it exempt."
    )


def test_every_required_aggregator_is_covered() -> None:
    """Catch a *new* fail-open aggregator being added without a guard entry.

    Without this, someone copies the three-job shape into a sixth required check and
    the guard above stays green purely because the new job is not in AGGREGATORS.
    """
    covered = {(s.workflow, s.job) for s in AGGREGATORS} | set(ADVISORY_AGGREGATORS)
    suspects: list[tuple[str, str]] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict):
            continue
        for name, job in (workflow.get("jobs") or {}).items():
            if not isinstance(job, dict) or str(job.get("if", "")).strip() != "always()":
                continue
            needs = job.get("needs")
            if not isinstance(needs, list) or len(needs) != 2:
                continue
            body = " ".join(str(step.get("run", "")) for step in job.get("steps", []))
            if ".result" not in body and "_RESULT" not in body:
                continue
            if (path.name, name) not in covered:
                suspects.append((path.name, name))

    assert not suspects, (
        "Un-guarded always() aggregator(s) that judge a dependency's result: "
        f"{suspects}. Add them to AGGREGATORS so their fail-closed behavior is proven."
    )
