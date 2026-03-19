from __future__ import annotations

from typing import Any

from aragora.swarm.tranche import GhReferenceClient, parse_github_reference_url


def classify_source_ref(url: str) -> dict[str, Any]:
    value = str(url).strip()
    try:
        target = parse_github_reference_url(value)
    except ValueError:
        return {
            "url": value,
            "kind": "context",
            "gated": False,
        }
    return {
        "url": value,
        "kind": "github",
        "gated": True,
        "github_kind": target.kind,
        "owner": target.owner,
        "repo": target.repo,
        "repo_full_name": f"{target.owner}/{target.repo}",
        "number": target.number,
    }


def enrich_github_refs(
    refs: list[dict[str, Any]],
    client: GhReferenceClient,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        item = dict(ref)
        if item.get("kind") != "github":
            enriched.append(item)
            continue
        repo = str(item.get("repo_full_name", "")).strip()
        number = int(item.get("number", 0) or 0)
        github_kind = str(item.get("github_kind", "")).strip()
        payload = (
            client.get_pr(repo, number)
            if github_kind == "pull_request"
            else client.get_issue(repo, number)
        )
        observed_state = _observed_reference_state(github_kind, payload)
        item.update(
            {
                "observed_state": observed_state,
                "status": _reference_status(observed_state),
                "stale": observed_state in {"closed", "merged"},
                "title": str(payload.get("title", "")).strip(),
                "labels": [
                    str(label.get("name", "")).strip()
                    for label in payload.get("labels", [])
                    if isinstance(label, dict) and str(label.get("name", "")).strip()
                ],
                "closed_at": payload.get("closedAt"),
                "merged_at": payload.get("mergedAt"),
            }
        )
        enriched.append(item)
    return enriched


def _observed_reference_state(kind: str, payload: dict[str, Any]) -> str:
    state = str(payload.get("state", "")).strip().lower()
    if kind == "pull_request" and str(payload.get("mergedAt", "")).strip():
        return "merged"
    return state or "unknown"


def _reference_status(observed_state: str) -> str:
    return "actionable" if observed_state == "open" else "stale"
