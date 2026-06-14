"""Collect genuine model-review evidence for the merge-quorum gate.

This module powers ``review-queue collect-evidence`` (and the thin
``scripts/collect_quorum_evidence.py`` wrapper). It runs >=2 genuine,
heterogeneous model reviewers against a PR's *exact current head*, composes each
reviewer's output into an evidence comment whose heading the canonical quorum
parsers recognize, and validates every comment with the same
``review-queue evidence-lint`` parser the gate uses — *before* anything is
posted.

Two safety invariants are enforced here, not by the caller:

* **Never fabricate.** A comment is only ever composed from a reviewer that
  actually returned non-empty output; failed/empty reviewers are recorded as
  failures and produce no comment.
* **Tier-gated posting.** Only Tier 0-2 PRs may be auto-posted (and only with
  ``apply=True``). Tier 3-4 (and unknown tier) always *prepare* the evidence for
  an operator and never post — the same human-settlement boundary the rest of
  the boss loop respects.

The decision logic (:func:`decide_action`) and comment composition
(:func:`compose_evidence_comment`) are pure so they can be unit-tested offline;
all network/process I/O is injected so the orchestrator
(:func:`collect_evidence`) is fully testable with fakes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import multiprocessing
import os
import queue
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.swarm import merge_quorum_io

logger = logging.getLogger(__name__)

# Direct model families whose name appears in the evidence heading and is
# recognized by the quorum identity resolver as a countable model reviewer.
# Router surfaces (factory/codex/tesla/harvey) are intentionally excluded: they
# require a separate disclosed model family, which this collector does not emit.
FAMILY_PROVIDERS: dict[str, str] = {
    "claude": "anthropic",
    "grok": "xai",
    "gemini": "google",
    "openai": "openai",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "kimi": "moonshot",
    "yi": "yi",
    "glm": "zhipu",
    "minimax": "minimax",
    "hermes": "nous",
}

FAMILY_DISPLAY: dict[str, str] = {
    "claude": "Claude",
    "grok": "Grok",
    "gemini": "Gemini",
    "openai": "OpenAI",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "kimi": "Kimi",
    "yi": "Yi",
    "glm": "GLM",
    "minimax": "MiniMax",
    "hermes": "Hermes",
}

DEFAULT_FAMILIES: tuple[str, ...] = ("claude", "grok")

# Tiers at or above this require exact-head operator settlement; never auto-post.
SETTLEMENT_TIER_FLOOR = 3

QUORUM_RERUN_COOLDOWN_SECONDS = 10 * 60
QUORUM_RERUN_MAX_PER_HEAD = 3
QUORUM_STATE_LOCK_TIMEOUT_SECONDS = 60.0
QUORUM_STATE_LOCK_POLL_SECONDS = 0.2
QUORUM_STATE_LOCK_STALE_SECONDS = 15 * 60
_REVIEWER_RESULT_QUEUE_TIMEOUT = 1.0
# Cap the diff fed to reviewers so a huge PR cannot blow the model context.
_MAX_DIFF_CHARS = 60_000
# Cap reviewer output so a runaway model cannot exceed GitHub's per-comment limit.
_MAX_REVIEWER_CHARS = 32_000
_CLAUDE_TIMEOUT = 300
_CODEX_TIMEOUT = 300
_REVIEWER_TIMEOUT = 300
_CLAUDE_TIMEOUT_ENV = "ARAGORA_COLLECT_EVIDENCE_CLAUDE_TIMEOUT_SECONDS"
_CODEX_TIMEOUT_ENV = "ARAGORA_COLLECT_EVIDENCE_CODEX_TIMEOUT_SECONDS"
_CODEX_MODEL_ENV = "ARAGORA_COLLECT_EVIDENCE_CODEX_MODEL"
_CODEX_MODELS_ENV = "ARAGORA_COLLECT_EVIDENCE_CODEX_MODELS"
_CODEX_DEFAULT_MODELS = ("gpt-5.5", "gpt-5")
_CODEX_DEFAULT_MODEL = _CODEX_DEFAULT_MODELS[0]
_REVIEWER_TIMEOUT_ENV = "ARAGORA_COLLECT_EVIDENCE_REVIEWER_TIMEOUT_SECONDS"
_CODEX_OPENAI_HARNESS = "Codex CLI OpenAI harness"
_CODEX_APPROVAL_POLICY_CONFIG = 'approval_policy="never"'
_REVIEWER_CLEANUP_TIMEOUT = 10


def _cap_text(text: str) -> str:
    text = text.strip()
    if len(text) > _MAX_REVIEWER_CHARS:
        return text[:_MAX_REVIEWER_CHARS].rstrip() + "\n\n[reviewer output truncated]"
    return text


def _timeout_seconds(env_name: str, default: int) -> float:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if not math.isfinite(value) or value <= 0:
        return float(default)
    return value


def _format_seconds(seconds: float) -> str:
    return f"{seconds:g}"


@dataclass
class ReviewerResult:
    """Raw output of one genuine reviewer run."""

    family: str
    text: str
    ok: bool
    error: str = ""
    harness: str = ""


@dataclass
class EvidenceItem:
    """A composed evidence comment plus its evidence-lint verdict."""

    family: str
    body: str
    would_count: bool
    counted_reviewer_ids: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    verdict: str = "unknown"

    @property
    def supportive(self) -> bool:
        return self.would_count and self.verdict == "pass"

    @property
    def dissenting(self) -> bool:
        return self.verdict == "changes_requested"


@dataclass
class CollectOutcome:
    repo: str
    pr: int
    head_sha: str
    head_committed_at: str
    tier: int | None
    action: str
    action_reason: str
    items: list[EvidenceItem] = field(default_factory=list)
    failures: list[ReviewerResult] = field(default_factory=list)
    posted: list[str] = field(default_factory=list)
    post_errors: list[str] = field(default_factory=list)
    quorum_rerun: dict[str, Any] | None = None

    @property
    def counting_families(self) -> list[str]:
        return [item.family for item in self.items if item.would_count]

    @property
    def supportive_families(self) -> list[str]:
        return [item.family for item in self.items if item.supportive]

    @property
    def dissenting_families(self) -> list[str]:
        return [item.family for item in self.items if item.dissenting]

    @property
    def has_supportive_quorum(self) -> bool:
        return len(self.supportive_families) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "collect_evidence",
            "repo": self.repo,
            "pr_number": self.pr,
            "head_sha": self.head_sha,
            "head_committed_at": self.head_committed_at,
            "tier": self.tier,
            "action": self.action,
            "action_reason": self.action_reason,
            "counting_families": self.counting_families,
            "supportive_families": self.supportive_families,
            "dissenting_families": self.dissenting_families,
            "has_supportive_quorum": self.has_supportive_quorum,
            "posted_families": list(self.posted),
            "post_errors": list(self.post_errors),
            "quorum_rerun": self.quorum_rerun,
            "items": [
                {
                    "family": item.family,
                    "would_count": item.would_count,
                    "verdict": item.verdict,
                    "counted_reviewer_ids": item.counted_reviewer_ids,
                    "problems": item.problems,
                    "body": item.body,
                }
                for item in self.items
            ],
            "failures": [{"family": f.family, "error": f.error} for f in self.failures],
        }


def decide_action(tier: int | None, apply: bool) -> tuple[str, str]:
    """Return ``(action, reason)`` where action is ``"post"`` or ``"prepare"``.

    Tier 3+ (and unknown tier) always ``prepare`` — high-tier merge authority is
    only ever settled by an operator on the exact head, so this collector refuses
    to post there regardless of ``apply``. Tier 0-2 posts only when ``apply`` is
    set; otherwise it is a dry run.
    """
    if tier is None or tier < 0:
        return ("prepare", "tier unknown; preparing evidence only (fail-safe)")
    if tier >= SETTLEMENT_TIER_FLOOR:
        return (
            "prepare",
            f"tier {tier} requires exact-head operator settlement; preparing evidence only",
        )
    if not apply:
        return ("prepare", "dry-run; re-run with --apply to post")
    return ("post", f"tier {tier} is auto-postable")


def _neutralize_reviewer_text(text: str) -> str:
    """Quote reviewer lines that could hijack the quorum identity parser.

    The composed comment owns its identity via the first heading and a single
    ``Model family:`` disclosure line. A reviewer that happens to emit its own
    ``## ... model review`` heading or a ``Model family: <other>`` line must not
    be able to change the attributed family. Such lines are prefixed with ``> ``
    so the parser (which keys on a leading ``#`` or a ``model family:`` label)
    ignores them, while the text stays human-readable. Everything else passes
    through verbatim — the reviewer's findings are never altered.
    """
    out: list[str] = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        # Canonicalize the way the parser does (strip leading quote/list markers
        # and surrounding emphasis) so the neutralizer is a strict superset of
        # what the identity parser will accept as a heading or disclosure line.
        probe = stripped.lstrip(">").strip()
        probe = re.sub(r"^([-*+]\s+|\d+[.)]\s+)+", "", probe)
        probe = probe.strip("*_ ").strip()
        is_heading = probe.startswith("#")
        is_setext = bool(re.fullmatch(r"[=\-]{2,}", stripped))
        # Over-quoting is harmless; a missed disclosure is not, so match the
        # ``model family:`` label anywhere it could be parsed. The gate parser
        # strips surrounding emphasis from the label, so tolerate whitespace and
        # ``*``/``_`` between "family" and the colon (e.g. ``**Model family**:``).
        has_family = bool(re.search(r"model\s+family[\s*_]*:", lower))
        if is_heading or is_setext or has_family:
            out.append(f"> {line}")
        else:
            out.append(line)
    return "\n".join(out)


def _reviewer_verdict(text: str) -> str:
    """Parse the first reviewer verdict line without inventing support."""
    for line in text.splitlines():
        stripped = line.strip().lower()
        if not stripped:
            continue
        if stripped.startswith("verdict:"):
            verdict = stripped.split(":", 1)[1].strip()
            if verdict.startswith("pass"):
                return "pass"
            if verdict.startswith("changes-requested") or verdict.startswith("changes requested"):
                return "changes_requested"
            return "unknown"
    return "unknown"


def compose_evidence_comment(
    *,
    family: str,
    head_sha: str,
    head_committed_at: str,
    pr: int | str,
    reviewer_text: str,
    harness: str = "",
) -> str:
    """Compose an evidence comment the quorum parsers recognize and count.

    The heading carries the family name (so the identity resolver infers a
    countable direct model reviewer) and an ``independent model review`` review
    trigger; a ``Model family:`` disclosure line plus a 7-char head citation are
    placed immediately under the heading so the comment is grounded on the exact
    head. ``reviewer_text`` is the genuine reviewer output; only lines that could
    hijack the identity parser are quoted (see :func:`_neutralize_reviewer_text`).
    """
    fam = family.strip().lower()
    display = FAMILY_DISPLAY.get(fam, fam.title())
    provider = FAMILY_PROVIDERS.get(fam, fam)
    short = head_sha[:7]
    harness_label = harness or f"the Aragora {display} reviewer"
    # Sanitize the timestamp to a safe charset so the disclosure block can never
    # be hijacked even if the field ever carries caller-influenced text.
    safe_committed = re.sub(r"[^A-Za-z0-9:.+\- TZ]", "", head_committed_at)[:40]
    committed = f", committed {safe_committed}" if safe_committed else ""
    return (
        f"## {display} independent model review\n\n"
        f"Reviewer: {fam} ({provider}) — independent adversarial model review via "
        f"{harness_label}, grounded on the exact PR head.\n"
        f"Head: {short} ({head_sha}){committed}.\n"
        f"PR: #{pr}.\n"
        f"Model family: {fam}\n\n"
        f"{_neutralize_reviewer_text(reviewer_text)}\n\n"
        f"dogfood: yes\n"
    )


def build_review_prompt(*, repo: str, pr: int | str, head_sha: str, diff_text: str) -> str:
    """Adversarial review prompt grounded on the exact head; diff is bounded."""
    diff = diff_text.strip()
    truncated = ""
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS]
        truncated = "\n\n[diff truncated for length]"
    short = head_sha[:7]
    return (
        "You are an adversarial senior reviewer giving an independent model review. "
        f"Review ONLY the diff below for PR #{pr} in {repo} at head {short}. "
        "Look hard for correctness, security, and regression risks. "
        "Begin your reply with 'Verdict: PASS' or 'Verdict: CHANGES-REQUESTED', then a terse "
        "bullet list of concrete findings each tagged [P1]/[P2]/[P3] with a location, or state "
        "explicitly that there are no blocking issues. Be concise.\n\n"
        f"=== DIFF (head {short}) ===\n{diff}{truncated}\n"
    )


# --- Default (real) I/O callables ------------------------------------------


def default_reviewer_runner(family: str, prompt: str) -> ReviewerResult:
    """Run a genuine reviewer: ``claude`` via its CLI, others via the API agent."""
    fam = family.strip().lower()
    if fam == "claude":
        return _run_claude_cli(prompt)
    if fam == "openai":
        return _run_openai_reviewer(prompt)
    return _run_api_agent(fam, prompt)


def _run_claude_cli(prompt: str) -> ReviewerResult:
    timeout = _timeout_seconds(_CLAUDE_TIMEOUT_ENV, _CLAUDE_TIMEOUT)
    try:
        proc = subprocess.run(
            ["claude", "-p"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return ReviewerResult("claude", "", False, "claude CLI not found on PATH")
    except subprocess.TimeoutExpired:
        return ReviewerResult(
            "claude", "", False, f"claude CLI timed out after {_format_seconds(timeout)}s"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Convert any other subprocess error (e.g. broken pipe writing stdin)
        # into a recorded failure so one bad reviewer never aborts the run.
        return ReviewerResult("claude", "", False, f"{type(exc).__name__}: {str(exc)[:200]}")
    text = (proc.stdout or "").strip()
    if proc.returncode != 0 or not text:
        return ReviewerResult(
            "claude",
            "",
            False,
            f"claude CLI exit {proc.returncode}: {(proc.stderr or '').strip()[:200]}",
        )
    return ReviewerResult("claude", _cap_text(text), True)


def _run_openai_reviewer(prompt: str) -> ReviewerResult:
    """Run OpenAI evidence via direct API when available, else Codex CLI.

    Operator machines often have Codex subscription auth but no direct
    ``OPENAI_API_KEY``. In that case Codex CLI is the local OpenAI-family
    reviewer; the normal exact-head comment composition and lint-before-post
    paths still decide whether the resulting evidence can count.
    """
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return _run_api_agent("openai", prompt)
    return _run_codex_openai_cli(prompt)


def _run_codex_openai_cli(prompt: str) -> ReviewerResult:
    timeout = _timeout_seconds(_CODEX_TIMEOUT_ENV, _CODEX_TIMEOUT)
    model_candidates = _codex_model_candidates()
    model_errors: list[str] = []
    for index, model in enumerate(model_candidates):
        output_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".md", prefix="aragora-codex-openai-review-", delete=False
            ) as fh:
                output_path = fh.name
            cmd = _codex_openai_command(output_path, model=model)
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            text = ""
            if output_path and os.path.exists(output_path):
                with open(output_path, encoding="utf-8") as fh:
                    text = fh.read().strip()
            if not text:
                text = (proc.stdout or "").strip()
            if proc.returncode != 0 or not text:
                detail = (proc.stderr or proc.stdout or "").strip()[:200]
                if index < len(model_candidates) - 1 and _codex_model_selection_failed(detail):
                    model_errors.append(f"{model}: {detail}")
                    continue
                if model_errors:
                    detail = (
                        f"{detail}; previous model selection failures: {'; '.join(model_errors)}"
                    )
                return ReviewerResult(
                    "openai", "", False, f"codex CLI exit {proc.returncode}: {detail}"
                )
            return ReviewerResult("openai", _cap_text(text), True, harness=_CODEX_OPENAI_HARNESS)
        except FileNotFoundError:
            return ReviewerResult("openai", "", False, "codex CLI not found on PATH")
        except subprocess.TimeoutExpired:
            return ReviewerResult(
                "openai", "", False, f"codex CLI timed out after {_format_seconds(timeout)}s"
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ReviewerResult("openai", "", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        finally:
            if output_path:
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
    return ReviewerResult("openai", "", False, "codex CLI has no configured model candidates")


def _codex_model_candidates() -> list[str]:
    pinned_model = os.environ.get(_CODEX_MODEL_ENV, "").strip()
    if pinned_model:
        return [pinned_model]
    raw_models = os.environ.get(_CODEX_MODELS_ENV, "").strip()
    candidates = re.split(r"[\s,]+", raw_models) if raw_models else list(_CODEX_DEFAULT_MODELS)
    return list(dict.fromkeys(model.strip() for model in candidates if model.strip()))


def _codex_openai_command(output_path: str, *, model: str) -> list[str]:
    cmd = [
        "codex",
        "exec",
        "--ignore-user-config",
        "-c",
        _CODEX_APPROVAL_POLICY_CONFIG,
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--output-last-message",
        output_path,
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append("-")
    return cmd


def _codex_model_selection_failed(detail: str) -> bool:
    lower = detail.lower()
    if "model" not in lower:
        return False
    return any(
        marker in lower
        for marker in (
            "not supported",
            "not available",
            "unsupported",
            "unknown",
            "invalid",
            "unrecognized",
        )
    )


def _run_api_agent(family: str, prompt: str) -> ReviewerResult:
    timeout = _timeout_seconds(_REVIEWER_TIMEOUT_ENV, _REVIEWER_TIMEOUT)
    ctx = _api_agent_process_context()
    result_queue: multiprocessing.Queue = ctx.Queue(maxsize=1)
    process = _start_api_agent_worker_process(ctx, family, prompt, result_queue)
    process.start()
    process.join(timeout + _REVIEWER_CLEANUP_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():  # pragma: no cover - defensive hard kill.
            process.kill()
            process.join(5)
        return ReviewerResult(
            family, "", False, f"{family} reviewer timed out after {_format_seconds(timeout)}s"
        )
    try:
        payload = result_queue.get(timeout=_REVIEWER_RESULT_QUEUE_TIMEOUT)
    except queue.Empty:
        return ReviewerResult(
            family,
            "",
            False,
            f"{family} reviewer exited without returning a result",
        )
    if isinstance(payload, ReviewerResult):
        return payload
    if isinstance(payload, dict):
        return ReviewerResult(
            str(payload.get("family") or family),
            str(payload.get("text") or ""),
            bool(payload.get("ok")),
            str(payload.get("error") or ""),
        )
    return ReviewerResult(family, "", False, f"{family} reviewer returned invalid result")


def _api_agent_process_context() -> Any:
    """Use spawn so API reviewer children do not inherit parent connector state."""
    return multiprocessing.get_context("spawn")


def _start_api_agent_worker_process(
    ctx: Any,
    family: str,
    prompt: str,
    result_queue: multiprocessing.Queue,
) -> multiprocessing.Process:
    return ctx.Process(
        target=_api_agent_worker,
        args=(family, prompt, result_queue),
        daemon=True,
    )


def _api_agent_worker(
    family: str,
    prompt: str,
    result_queue: multiprocessing.Queue,
) -> None:
    result_queue.put(_run_api_agent_in_current_process(family, prompt))


def _run_api_agent_in_current_process(family: str, prompt: str) -> ReviewerResult:
    try:
        from aragora.agents import create_agent
    except Exception as exc:  # pragma: no cover - import guard
        return ReviewerResult(family, "", False, f"create_agent import failed: {exc}")
    try:
        agent = create_agent(family, name=f"{family}_reviewer", role="critic")
        text = asyncio.run(_generate_with_api_agent_cleanup(agent, prompt))
    except Exception as exc:
        return ReviewerResult(family, "", False, f"{type(exc).__name__}: {str(exc)[:200]}")
    text = (text or "").strip()
    if not text:
        return ReviewerResult(family, "", False, "empty reviewer output")
    return ReviewerResult(family, _cap_text(text), True)


async def _generate_with_api_agent_cleanup(agent: Any, prompt: str) -> str:
    """Generate with an API-backed agent and close one-shot network resources."""
    timeout = _timeout_seconds(_REVIEWER_TIMEOUT_ENV, _REVIEWER_TIMEOUT)
    try:
        return await asyncio.wait_for(agent.generate(prompt), timeout=timeout)
    finally:
        await _close_api_agent_resources(agent)


async def _close_api_agent_resources(agent: Any) -> None:
    """Best-effort cleanup for collect-evidence one-shot API reviewer runs."""
    close = getattr(agent, "close", None)
    if callable(close):
        try:
            result = close()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=_REVIEWER_CLEANUP_TIMEOUT)
        except TimeoutError:
            logger.debug("collect-evidence API agent close timed out")
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask reviewer results.
            logger.debug("collect-evidence API agent close failed: %s", exc)

    try:
        from aragora.agents.api_agents.common import close_shared_connector
    except ImportError as exc:
        logger.debug("collect-evidence shared connector cleanup unavailable: %s", exc)
        return

    try:
        # This collector calls API reviewers through a one-shot asyncio.run()
        # loop, so the shared aiohttp connector must be released before that
        # loop is torn down. The collector dispatches reviewers serially; if it
        # ever fans reviewers out, cleanup must move outside the per-reviewer path.
        await asyncio.wait_for(close_shared_connector(), timeout=_REVIEWER_CLEANUP_TIMEOUT)
    except TimeoutError:
        logger.debug("collect-evidence shared connector close timed out")
    except Exception as exc:  # noqa: BLE001 - cleanup must not mask reviewer results.
        logger.debug("collect-evidence shared connector close failed: %s", exc)


def default_prompt_builder(repo: str, pr: int, ctx: dict[str, Any]) -> str:
    head_sha = str(ctx.get("head_sha") or "")
    proc = merge_quorum_io.run(
        ["gh", "pr", "diff", str(pr), "--repo", repo],
        env=merge_quorum_io.aragora_env(),
        timeout=120,
    )
    # Refuse to review nothing: a failed or empty diff fetch would otherwise let
    # a reviewer emit a "PASS" against an empty prompt while the composed comment
    # still claims it is grounded on the head. Fail loudly instead.
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not fetch diff for PR #{pr}: {(proc.stderr or '').strip()[:200]}"
        )
    diff_text = proc.stdout or ""
    if not diff_text.strip():
        raise RuntimeError(f"PR #{pr} has an empty diff; nothing to review")
    # Pin the diff to the resolved head: `gh pr diff` returns whatever the head
    # is at call time, so if it moved between context resolution and now the
    # reviewer would see a different diff than the comment claims to ground on.
    live = merge_quorum_io.run(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "headRefOid",
            "--jq",
            ".headRefOid",
        ],
        env=merge_quorum_io.aragora_env(),
        timeout=30,
    )
    # Fail closed: if the head cannot be re-resolved, treat it as a pin failure
    # rather than silently skipping the check and grounding on a stale head.
    live_head = (live.stdout or "").strip()
    if live.returncode != 0 or not live_head:
        raise RuntimeError(f"could not re-resolve head for PR #{pr} to pin the diff")
    if head_sha and live_head and live_head != head_sha:
        raise RuntimeError(
            f"head moved during diff fetch for PR #{pr} ({head_sha[:7]} -> {live_head[:7]}); retry"
        )
    return build_review_prompt(repo=repo, pr=pr, head_sha=head_sha, diff_text=diff_text)


def default_linter(
    pr: int,
    head_sha: str,
    head_committed_at: str,
    author: str,
    body: str,
    env: dict[str, str],
) -> dict[str, Any]:
    return merge_quorum_io.lint_comment(
        pr, head_sha, head_committed_at, author, body, env or merge_quorum_io.aragora_env()
    )


def default_poster(repo: str, pr: int, body: str) -> None:
    import os
    import tempfile

    path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
            path = fh.name
            fh.write(body)
        proc = merge_quorum_io.run(
            ["gh", "pr", "comment", str(pr), "--repo", repo, "--body-file", path],
            env=merge_quorum_io.aragora_env(),
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gh pr comment failed: {(proc.stderr or '').strip()[:200]}")
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def resolve_author(default: str = "local") -> str:
    """Best-effort GitHub login used for offline evidence-lint simulation."""
    try:
        proc = merge_quorum_io.run(
            ["gh", "api", "user", "--jq", ".login"],
            env=merge_quorum_io.aragora_env(),
            timeout=30,
        )
    except Exception:
        return default
    login = (proc.stdout or "").strip() if proc.returncode == 0 else ""
    return login or default


@contextmanager
def _locked_quorum_reconcile_state(path: Path) -> Iterator[None]:
    """Serialize load/evaluate/rerun/save for the shared merge-quorum state file."""
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + QUORUM_STATE_LOCK_TIMEOUT_SECONDS
    fd: int | None = None
    while fd is None:
        try:
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(lock_path, flags)
            os.write(
                fd,
                f"pid={os.getpid()} acquired_at={datetime.now(timezone.utc).isoformat()}\n".encode(),
            )
        except FileExistsError:
            if _quorum_state_lock_is_stale(lock_path):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for merge-quorum state lock: {lock_path}")
            time.sleep(QUORUM_STATE_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except OSError:
            pass


def _quorum_state_lock_is_stale(lock_path: Path) -> bool:
    try:
        stat = lock_path.lstat()
    except OSError:
        return False
    if lock_path.is_symlink():
        raise RuntimeError(f"refusing symlink merge-quorum state lock: {lock_path}")
    age_seconds = max(0.0, time.time() - stat.st_mtime)
    if age_seconds < QUORUM_STATE_LOCK_STALE_SECONDS:
        return False
    try:
        text = lock_path.read_text(encoding="utf-8")
    except OSError:
        return False
    match = re.search(r"\bpid=(\d+)\b", text)
    if not match:
        return True
    pid = int(match.group(1))
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def default_quorum_reconciler(repo: str, pr: int) -> dict[str, Any]:
    """Run the A1 stale-quorum reconciler for one PR after evidence posting."""
    from scripts import reconcile_merge_quorum

    state_file = reconcile_merge_quorum.DEFAULT_STATE_FILE
    with _locked_quorum_reconcile_state(state_file):
        state = reconcile_merge_quorum._load_state(state_file)
        decision, quorum_run = reconcile_merge_quorum.evaluate_pr(
            repo,
            pr,
            now=datetime.now(timezone.utc),
            state=state,
            cooldown_seconds=QUORUM_RERUN_COOLDOWN_SECONDS,
            max_reruns=QUORUM_RERUN_MAX_PER_HEAD,
        )
        record: dict[str, Any] = {
            "should_rerun": decision.should_rerun,
            "reason": decision.reason,
            "run_id": decision.run_id,
            "applied": False,
        }
        if decision.next_prompt:
            record["next_prompt"] = decision.next_prompt
        if decision.should_rerun and quorum_run is not None:
            head_state = state.setdefault(
                quorum_run.head_sha,
                {"count": 0, "last_rerun_at": None},
            )
            if int(head_state.get("count", 0)) >= QUORUM_RERUN_MAX_PER_HEAD:
                record["should_rerun"] = False
                record["reason"] = "max_reruns_reached_in_locked_state"
                return record
            record["applied"] = reconcile_merge_quorum.execute_rerun(repo, quorum_run.run_id)
            if record["applied"]:
                head_state["count"] = int(head_state.get("count", 0)) + 1
                head_state["last_rerun_at"] = datetime.now(timezone.utc).isoformat()
                reconcile_merge_quorum._save_state(state_file, state)
        return record


# --- Orchestrator ----------------------------------------------------------


def collect_evidence(
    *,
    repo: str,
    pr: int,
    families: Sequence[str],
    author: str,
    apply: bool,
    context_fetcher: Callable[[str, int], dict[str, Any]] = merge_quorum_io.fetch_pr_context,
    tier_fetcher: Callable[[str, int], int | None] = merge_quorum_io.fetch_pr_tier,
    prompt_builder: Callable[[str, int, dict[str, Any]], str] = default_prompt_builder,
    reviewer_runner: Callable[[str, str], ReviewerResult] = default_reviewer_runner,
    linter: Callable[..., dict[str, Any]] = default_linter,
    poster: Callable[[str, int, str], None] = default_poster,
    quorum_reconciler: Callable[[str, int], dict[str, Any] | None] | None = None,
    env: dict[str, str] | None = None,
) -> CollectOutcome:
    """Run reviewers, validate evidence, and post only when tier-gating allows."""
    ctx = context_fetcher(repo, pr)
    head_sha = str(ctx.get("head_sha") or "").strip()
    head_committed_at = str(ctx.get("head_committed_at") or "")
    if not head_sha:
        raise ValueError(f"could not resolve head SHA for PR #{pr} in {repo}")

    tier = tier_fetcher(repo, pr)
    action, action_reason = decide_action(tier, apply)

    outcome = CollectOutcome(
        repo=repo,
        pr=pr,
        head_sha=head_sha,
        head_committed_at=head_committed_at,
        tier=tier,
        action=action,
        action_reason=action_reason,
    )

    prompt = prompt_builder(repo, pr, ctx)

    seen: set[str] = set()
    for raw_family in families:
        family = raw_family.strip().lower()
        if not family or family in seen:
            continue
        seen.add(family)
        if family not in FAMILY_PROVIDERS:
            # Only direct families the quorum parser can count are supported;
            # reject anything else early instead of producing an uncountable
            # (or malformed) comment.
            outcome.failures.append(
                ReviewerResult(family, "", False, f"unsupported reviewer family: {family}")
            )
            continue
        result = reviewer_runner(family, prompt)
        if not result.ok or not result.text.strip():
            outcome.failures.append(result)
            continue
        body = compose_evidence_comment(
            family=family,
            head_sha=head_sha,
            head_committed_at=head_committed_at,
            pr=pr,
            reviewer_text=result.text,
            harness=result.harness,
        )
        lint = linter(pr, head_sha, head_committed_at, author, body, env or {})
        outcome.items.append(
            EvidenceItem(
                family=family,
                body=body,
                would_count=bool(lint.get("would_count")),
                verdict=_reviewer_verdict(result.text),
                counted_reviewer_ids=list(lint.get("counted_reviewer_ids") or []),
                problems=list(lint.get("problems") or []),
            )
        )

    if action == "post":
        if outcome.dissenting_families:
            outcome.action = "prepare"
            outcome.action_reason = (
                "reviewer dissent present "
                f"({', '.join(outcome.dissenting_families)}); prepared evidence only"
            )
            return outcome
        if not outcome.has_supportive_quorum:
            outcome.action = "prepare"
            outcome.action_reason = (
                "supportive quorum incomplete "
                f"({len(outcome.supportive_families)}/2); prepared evidence only"
            )
            return outcome
        # Reviewers can take minutes; re-verify the head and tier immediately
        # before posting so a head that moved or a PR promoted to a settlement
        # tier in the meantime is never posted against.
        try:
            recheck_head = str((context_fetcher(repo, pr) or {}).get("head_sha") or "").strip()
            recheck_tier = tier_fetcher(repo, pr)
        except Exception as exc:
            outcome.action = "prepare"
            outcome.action_reason = (
                f"could not re-verify head/tier before posting ({str(exc)[:120]}); prepared only"
            )
            return outcome
        recheck_action, recheck_reason = decide_action(recheck_tier, apply)
        if recheck_head != head_sha or recheck_action != "post":
            outcome.action = "prepare"
            outcome.action_reason = (
                f"head/tier changed before posting "
                f"(head {head_sha[:7]}->{recheck_head[:7] or 'none'}, "
                f"tier {tier}->{recheck_tier}); prepared only: {recheck_reason}"
            )
        else:
            for item in outcome.items:
                if not item.supportive:
                    continue
                try:
                    poster(repo, pr, item.body)
                except Exception as exc:
                    # One failed post must not lose the record of the others.
                    outcome.post_errors.append(f"{item.family}: {str(exc)[:200]}")
                    continue
                outcome.posted.append(item.family)
        if outcome.posted and outcome.has_supportive_quorum and quorum_reconciler is not None:
            try:
                outcome.quorum_rerun = quorum_reconciler(repo, pr)
            except Exception as exc:  # noqa: BLE001 - evidence posts should remain reported.
                outcome.quorum_rerun = {"applied": False, "error": str(exc)[:200]}

    return outcome


def _render_outcome(outcome: CollectOutcome) -> str:
    lines = [
        f"collect-evidence: PR #{outcome.pr} ({outcome.repo})",
        f"  head: {outcome.head_sha[:10]}  tier: {outcome.tier}",
        f"  action: {outcome.action} ({outcome.action_reason})",
        f"  counting families: {', '.join(outcome.counting_families) or 'none'}",
        f"  supportive families: {', '.join(outcome.supportive_families) or 'none'}",
        f"  dissenting families: {', '.join(outcome.dissenting_families) or 'none'}",
    ]
    if outcome.posted:
        lines.append(f"  posted: {', '.join(outcome.posted)}")
    if outcome.post_errors:
        lines.append(f"  post errors: {'; '.join(outcome.post_errors)}")
    if outcome.quorum_rerun:
        rerun = outcome.quorum_rerun
        action = "applied" if rerun.get("applied") else "not applied"
        reason = rerun.get("reason") or rerun.get("error") or "unknown"
        lines.append(f"  quorum rerun: {action} ({reason})")
    for item in outcome.items:
        flag = "counts" if item.would_count else f"DOES NOT count ({', '.join(item.problems)})"
        lines.append(f"  - {item.family}: {flag}; verdict={item.verdict}")
    for failure in outcome.failures:
        lines.append(f"  - {failure.family}: reviewer failed ({failure.error})")
    if outcome.action == "prepare":
        lines.append("")
        lines.append("Prepared evidence comments (not posted):")
        for item in outcome.items:
            if not item.would_count:
                continue
            lines.append(f"\n----- {item.family} -----\n{item.body}")
    return "\n".join(lines)


def run_collect_cli(
    *,
    repo: str,
    pr: int,
    families: Sequence[str] | None,
    author: str | None,
    apply: bool,
    json_output: bool,
    printer: Callable[[str], None] = print,
) -> int:
    """Shared entry point for the script and ``review-queue collect-evidence``.

    Returns 0 when >=2 reviewers produced counting evidence, else 1. Note that a
    non-zero exit does not imply nothing was posted: with ``--apply`` on a
    low-tier PR a single genuine reviewer can post one counting comment and still
    return 1 (quorum is enforced as N-of-M elsewhere). Inspect ``posted_families``
    in the JSON output rather than treating exit-code 1 as "nothing posted".
    """
    fams = tuple(families) if families else DEFAULT_FAMILIES
    resolved_author = author or resolve_author()
    try:
        outcome = collect_evidence(
            repo=repo,
            pr=pr,
            families=fams,
            author=resolved_author,
            apply=apply,
            env=merge_quorum_io.aragora_env(),
            quorum_reconciler=default_quorum_reconciler if apply else None,
        )
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        if json_output:
            import json

            printer(json.dumps({"mode": "collect_evidence", "error": str(exc)}, indent=2))
        else:
            printer(f"error: {exc}")
        return 1

    if json_output:
        import json

        printer(json.dumps(outcome.to_dict(), indent=2))
    else:
        printer(_render_outcome(outcome))
    return 0 if outcome.has_supportive_quorum else 1
