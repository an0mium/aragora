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
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
# Cap the diff fed to reviewers so a huge PR cannot blow the model context.
_MAX_DIFF_CHARS = 60_000
# Cap reviewer output so a runaway model cannot exceed GitHub's per-comment limit.
_MAX_REVIEWER_CHARS = 32_000
_CLAUDE_TIMEOUT = 300
_REVIEWER_TIMEOUT = 300


def _cap_text(text: str) -> str:
    text = text.strip()
    if len(text) > _MAX_REVIEWER_CHARS:
        return text[:_MAX_REVIEWER_CHARS].rstrip() + "\n\n[reviewer output truncated]"
    return text


@dataclass
class ReviewerResult:
    """Raw output of one genuine reviewer run."""

    family: str
    text: str
    ok: bool
    error: str = ""


@dataclass
class EvidenceItem:
    """A composed evidence comment plus its evidence-lint verdict."""

    family: str
    body: str
    would_count: bool
    counted_reviewer_ids: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


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

    @property
    def counting_families(self) -> list[str]:
        return [item.family for item in self.items if item.would_count]

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
            "posted_families": list(self.posted),
            "post_errors": list(self.post_errors),
            "items": [
                {
                    "family": item.family,
                    "would_count": item.would_count,
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
    return _run_api_agent(fam, prompt)


def _run_claude_cli(prompt: str) -> ReviewerResult:
    try:
        proc = subprocess.run(
            ["claude", "-p"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=_CLAUDE_TIMEOUT,
            check=False,
        )
    except FileNotFoundError:
        return ReviewerResult("claude", "", False, "claude CLI not found on PATH")
    except subprocess.TimeoutExpired:
        return ReviewerResult("claude", "", False, f"claude CLI timed out after {_CLAUDE_TIMEOUT}s")
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


def _run_api_agent(family: str, prompt: str) -> ReviewerResult:
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
    try:
        return await asyncio.wait_for(agent.generate(prompt), timeout=_REVIEWER_TIMEOUT)
    finally:
        await _close_api_agent_resources(agent)


async def _close_api_agent_resources(agent: Any) -> None:
    """Best-effort cleanup for collect-evidence one-shot API reviewer runs."""
    close = getattr(agent, "close", None)
    if callable(close):
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
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
        await close_shared_connector()
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
        )
        lint = linter(pr, head_sha, head_committed_at, author, body, env or {})
        outcome.items.append(
            EvidenceItem(
                family=family,
                body=body,
                would_count=bool(lint.get("would_count")),
                counted_reviewer_ids=list(lint.get("counted_reviewer_ids") or []),
                problems=list(lint.get("problems") or []),
            )
        )

    if action == "post":
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
                if not item.would_count:
                    continue
                try:
                    poster(repo, pr, item.body)
                except Exception as exc:
                    # One failed post must not lose the record of the others.
                    outcome.post_errors.append(f"{item.family}: {str(exc)[:200]}")
                    continue
                outcome.posted.append(item.family)

    return outcome


def _render_outcome(outcome: CollectOutcome) -> str:
    lines = [
        f"collect-evidence: PR #{outcome.pr} ({outcome.repo})",
        f"  head: {outcome.head_sha[:10]}  tier: {outcome.tier}",
        f"  action: {outcome.action} ({outcome.action_reason})",
        f"  counting families: {', '.join(outcome.counting_families) or 'none'}",
    ]
    if outcome.posted:
        lines.append(f"  posted: {', '.join(outcome.posted)}")
    if outcome.post_errors:
        lines.append(f"  post errors: {'; '.join(outcome.post_errors)}")
    for item in outcome.items:
        flag = "counts" if item.would_count else f"DOES NOT count ({', '.join(item.problems)})"
        lines.append(f"  - {item.family}: {flag}")
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
    return 0 if len(outcome.counting_families) >= 2 else 1
