"""GitHub-event resolution adapter for synthetic markets.

The resolver looks up the current GitHub state for each market's target
and returns a deterministic :class:`ResolutionEvent`. It uses
:func:`aragora.swarm.github_app_auth.gh_subprocess_run` so the lookups
go through the App-token + rate-limit-aware path landed for AGT-04
substrate hardening.

Resolution rules:

- ``pr_merge``: YES if PR is merged at expiry; NO if closed unmerged at
  expiry; INCONCLUSIVE if still open at expiry.
- ``issue_close``: YES if closed at expiry (closed-via-PR counts);
  INCONCLUSIVE if still open at expiry (NO is reserved for future
  ``rejected``-state semantics).
- ``ci_pass``: YES if the most recent completed check suite for the
  target ref has conclusion ``success`` or ``neutral``;
  NO if any required check failed; INCONCLUSIVE if no completed run
  exists at expiry.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from aragora.markets.types import Market, ResolutionEvent

logger = logging.getLogger(__name__)


class ResolutionError(RuntimeError):
    """Raised when resolution cannot proceed (e.g. transient GitHub error)."""


GhRunner = Callable[..., subprocess.CompletedProcess[str]]


def _default_runner() -> GhRunner:
    # Imported lazily so tests can stub gh_subprocess_run before resolver init
    from aragora.swarm.github_app_auth import gh_subprocess_run

    return gh_subprocess_run


@dataclass
class GitHubMarketResolver:
    """Resolves synthetic markets against current GitHub state.

    Parameters:
        gh_runner: Callable matching ``gh_subprocess_run`` signature; lets
            tests inject a deterministic stub. Defaults to the real
            App-token-aware runner.
        require_expiry: If True (the default), only resolve markets that
            have passed ``expires_at``; otherwise, resolve immediately
            against current state (useful for tests and ad-hoc resolution).
    """

    gh_runner: GhRunner | None = None
    require_expiry: bool = True

    def __post_init__(self) -> None:
        self._runner: GhRunner = self.gh_runner or _default_runner()

    def resolve(self, market: Market, *, now: datetime | None = None) -> ResolutionEvent:
        """Resolve a single market or raise :class:`ResolutionError` on transient failure."""
        reference = now or datetime.now(tz=UTC)
        if self.require_expiry and not market.is_expired(now=reference):
            raise ResolutionError(
                f"market {market.market_id} is not yet expired ({market.expires_at})"
            )
        if market.question_kind == "pr_merge":
            return self._resolve_pr_merge(market, reference=reference)
        if market.question_kind == "issue_close":
            return self._resolve_issue_close(market, reference=reference)
        if market.question_kind == "ci_pass":
            return self._resolve_ci_pass(market, reference=reference)
        raise ResolutionError(f"unsupported question_kind: {market.question_kind}")

    def resolve_batch(
        self,
        markets: list[Market],
        *,
        now: datetime | None = None,
    ) -> list[ResolutionEvent]:
        """Resolve as many markets as possible; skip transient failures.

        Returns the successfully resolved events. Failed resolutions are
        logged at warning level and omitted from the result; callers can
        retry on the next pass.
        """
        out: list[ResolutionEvent] = []
        for market in markets:
            try:
                out.append(self.resolve(market, now=now))
            except ResolutionError as exc:
                logger.warning("resolve skipped for %s: %s", market.market_id, exc)
        return out

    # ------------------------------------------------------------------
    # Per-kind resolvers
    # ------------------------------------------------------------------

    def _resolve_pr_merge(
        self,
        market: Market,
        *,
        reference: datetime,
    ) -> ResolutionEvent:
        repo = str(market.target.get("repo") or "")
        number = int(market.target.get("number") or 0)
        result = self._runner(
            ["pr", "view", str(number), "--repo", repo, "--json", "state,merged,mergedAt,closedAt"],
            timeout=30,
        )
        if result.returncode != 0:
            raise ResolutionError(f"gh pr view failed: {result.stderr.strip() or '(no stderr)'}")
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise ResolutionError(f"gh pr view returned non-JSON: {exc}") from exc
        state = str(payload.get("state") or "").upper()
        merged = bool(payload.get("merged"))
        evidence = {
            "repo": repo,
            "number": number,
            "state": state,
            "merged": merged,
            "merged_at": payload.get("mergedAt"),
            "closed_at": payload.get("closedAt"),
            "checked_at": reference.isoformat().replace("+00:00", "Z"),
        }
        if merged:
            return ResolutionEvent.yes(
                market_id=market.market_id,
                resolution_source="github_pr_state",
                evidence=evidence,
                resolved_at=reference,
            )
        if state == "CLOSED":
            return ResolutionEvent.no(
                market_id=market.market_id,
                resolution_source="github_pr_state",
                evidence=evidence,
                resolved_at=reference,
            )
        return ResolutionEvent.inconclusive(
            market_id=market.market_id,
            resolution_source="github_pr_state",
            evidence=evidence,
            resolved_at=reference,
        )

    def _resolve_issue_close(
        self,
        market: Market,
        *,
        reference: datetime,
    ) -> ResolutionEvent:
        repo = str(market.target.get("repo") or "")
        number = int(market.target.get("number") or 0)
        result = self._runner(
            ["issue", "view", str(number), "--repo", repo, "--json", "state,closedAt"],
            timeout=30,
        )
        if result.returncode != 0:
            raise ResolutionError(f"gh issue view failed: {result.stderr.strip() or '(no stderr)'}")
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise ResolutionError(f"gh issue view returned non-JSON: {exc}") from exc
        state = str(payload.get("state") or "").upper()
        evidence = {
            "repo": repo,
            "number": number,
            "state": state,
            "closed_at": payload.get("closedAt"),
            "checked_at": reference.isoformat().replace("+00:00", "Z"),
        }
        if state == "CLOSED":
            return ResolutionEvent.yes(
                market_id=market.market_id,
                resolution_source="github_issue_state",
                evidence=evidence,
                resolved_at=reference,
            )
        return ResolutionEvent.inconclusive(
            market_id=market.market_id,
            resolution_source="github_issue_state",
            evidence=evidence,
            resolved_at=reference,
        )

    def _resolve_ci_pass(
        self,
        market: Market,
        *,
        reference: datetime,
    ) -> ResolutionEvent:
        repo = str(market.target.get("repo") or "")
        ref = str(market.target.get("ref") or "")
        result = self._runner(
            [
                "api",
                f"repos/{repo}/commits/{ref}/check-suites",
                "--jq",
                ".check_suites | map({status, conclusion, app_slug: .app.slug})",
            ],
            timeout=30,
        )
        if result.returncode != 0:
            raise ResolutionError(
                f"gh api check-suites failed: {result.stderr.strip() or '(no stderr)'}"
            )
        try:
            suites: list[dict[str, Any]] = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise ResolutionError(f"check-suites response was non-JSON: {exc}") from exc
        completed = [
            suite for suite in suites if str(suite.get("status") or "").lower() == "completed"
        ]
        evidence = {
            "repo": repo,
            "ref": ref,
            "suite_count": len(suites),
            "completed_count": len(completed),
            "checked_at": reference.isoformat().replace("+00:00", "Z"),
            "conclusions": [str(suite.get("conclusion") or "").lower() for suite in completed],
        }
        if not completed:
            return ResolutionEvent.inconclusive(
                market_id=market.market_id,
                resolution_source="github_ci_check",
                evidence=evidence,
                resolved_at=reference,
            )
        good = {"success", "neutral", "skipped"}
        bad = {"failure", "timed_out", "cancelled", "action_required", "stale"}
        conclusions = [str(suite.get("conclusion") or "").lower() for suite in completed]
        if any(conclusion in bad for conclusion in conclusions):
            return ResolutionEvent.no(
                market_id=market.market_id,
                resolution_source="github_ci_check",
                evidence=evidence,
                resolved_at=reference,
            )
        if all(conclusion in good for conclusion in conclusions):
            return ResolutionEvent.yes(
                market_id=market.market_id,
                resolution_source="github_ci_check",
                evidence=evidence,
                resolved_at=reference,
            )
        return ResolutionEvent.inconclusive(
            market_id=market.market_id,
            resolution_source="github_ci_check",
            evidence=evidence,
            resolved_at=reference,
        )


def resolve_market(
    market: Market,
    *,
    gh_runner: GhRunner | None = None,
    require_expiry: bool = True,
    now: datetime | None = None,
) -> ResolutionEvent:
    """Convenience wrapper around :class:`GitHubMarketResolver`."""
    resolver = GitHubMarketResolver(gh_runner=gh_runner, require_expiry=require_expiry)
    return resolver.resolve(market, now=now)


__all__ = ["GitHubMarketResolver", "ResolutionError", "resolve_market"]
