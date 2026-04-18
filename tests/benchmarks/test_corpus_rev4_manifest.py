"""Schema + SHA-256 signature invariants for the rev-4 staging corpus.

The rev-4 manifest at ``tests/benchmarks/corpus_rev4.json`` is the staged
benchmark corpus for H1-01 (issue #6227). It is not yet the canonical corpus
— promotion to ``docs/benchmarks/corpus.json`` requires recorded dispatch
evidence per the rev-3 honesty invariant (see
``docs/benchmarks/corpus_rev4_staging.md`` and
``tests/benchmarks/test_corpus_freshness.py``).

These tests enforce the shape of the staging manifest so that promotion is a
mechanical, reviewable step rather than an ad-hoc edit:

1. ``corpus_rev4.json`` exists and parses as a JSON object.
2. The manifest declares ``revision == 4`` and ``status == "staging"``.
3. Membership is at least 30 bounded tasks (H1-01 acceptance floor).
4. Every entry carries issue_id, title, execution_class, scope_hint, and
   source_reference.
5. Execution-class coverage meets the per-class targets declared in the
   manifest.
6. ``sha256_signature`` matches the canonical hash recomputed from the body
   with the signature field removed — i.e. the manifest is self-signed and
   will detect silent tampering.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_REV4_PATH = REPO_ROOT / "tests" / "benchmarks" / "corpus_rev4.json"

REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "corpus_id",
        "revision",
        "status",
        "recorded_on",
        "success_contract",
        "run_cadence",
        "change_control",
        "promotion_target",
        "promotion_requirements",
        "membership_criteria",
        "execution_class_targets",
        "sha256_signature_scope",
        "sha256_signature",
        "issues",
    }
)

REQUIRED_ISSUE_KEYS = frozenset(
    {"issue_id", "title", "execution_class", "scope_hint", "source_reference"}
)

MIN_ISSUE_COUNT = 30


def _load() -> dict[str, Any]:
    assert CORPUS_REV4_PATH.exists(), (
        f"rev-4 staging corpus missing at {CORPUS_REV4_PATH} — H1-01 acceptance"
    )
    payload = json.loads(CORPUS_REV4_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "rev-4 staging corpus must be a JSON object"
    return payload


def _canonical_body_for_signature(payload: dict[str, Any]) -> bytes:
    """Return deterministic bytes used to compute the SHA-256 signature.

    The signature covers every field except ``sha256_signature`` itself.
    Issues are sorted by ``issue_id`` so order-shuffling edits do not change
    the digest, and JSON is emitted with sorted keys and tight separators so
    whitespace edits do not change it either.
    """

    body = {k: v for k, v in payload.items() if k != "sha256_signature"}
    issues = sorted(
        (dict(issue) for issue in body.get("issues", [])),
        key=lambda issue: int(issue.get("issue_id", 0)),
    )
    body["issues"] = issues
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_signature(payload: dict[str, Any]) -> str:
    """Public helper so promotion scripts can recompute on edit."""

    return hashlib.sha256(_canonical_body_for_signature(payload)).hexdigest()


def test_rev4_manifest_shape() -> None:
    payload = _load()

    missing = REQUIRED_TOP_LEVEL_KEYS - payload.keys()
    assert missing == set(), f"rev-4 manifest missing top-level keys: {sorted(missing)}"

    assert payload["revision"] == 4, "rev-4 manifest revision must be 4"
    assert payload["status"] == "staging", (
        "rev-4 manifest must be status=staging until dispatch evidence accumulates"
    )
    assert payload["success_contract"] == "mergeable_pr_or_merged_pr"
    assert payload["promotion_target"] == "docs/benchmarks/corpus.json"


def test_rev4_has_at_least_30_bounded_tasks() -> None:
    payload = _load()
    issues = payload["issues"]
    assert isinstance(issues, list), "issues must be a list"
    assert len(issues) >= MIN_ISSUE_COUNT, (
        f"rev-4 must have ≥{MIN_ISSUE_COUNT} bounded tasks (H1-01 acceptance); found {len(issues)}"
    )


def test_rev4_entry_fields_are_present() -> None:
    payload = _load()
    for issue in payload["issues"]:
        assert isinstance(issue, dict), "issue entries must be objects"
        missing = REQUIRED_ISSUE_KEYS - issue.keys()
        assert missing == set(), (
            f"issue #{issue.get('issue_id', '?')} missing fields: {sorted(missing)}"
        )
        assert isinstance(issue["issue_id"], int) and issue["issue_id"] > 0
        assert isinstance(issue["title"], str) and issue["title"].strip()
        assert isinstance(issue["execution_class"], str) and issue["execution_class"].strip()
        assert isinstance(issue["scope_hint"], list) and issue["scope_hint"]
        assert isinstance(issue["source_reference"], str) and issue["source_reference"].startswith(
            "github.com/"
        )


def test_rev4_issue_ids_are_unique() -> None:
    payload = _load()
    ids = [int(issue["issue_id"]) for issue in payload["issues"]]
    assert len(ids) == len(set(ids)), "rev-4 corpus must not contain duplicate issue_ids"


def test_rev4_execution_class_targets_are_met() -> None:
    payload = _load()
    targets = payload["execution_class_targets"]
    counts = Counter(issue["execution_class"] for issue in payload["issues"])
    deficits: list[str] = []
    for execution_class, target in targets.items():
        observed = counts.get(execution_class, 0)
        if observed < target:
            deficits.append(
                f"execution_class={execution_class}: target={target} observed={observed}"
            )
    assert deficits == [], "rev-4 execution-class coverage short of targets:\n  " + "\n  ".join(
        deficits
    )


def test_rev4_signature_matches_body() -> None:
    payload = _load()
    declared = payload.get("sha256_signature")
    assert isinstance(declared, str) and len(declared) == 64, (
        "rev-4 manifest must declare a 64-char hex sha256_signature"
    )
    recomputed = compute_signature(payload)
    assert declared == recomputed, (
        f"rev-4 sha256_signature mismatch — manifest body has been edited without "
        f"updating the signature.\n  declared:   {declared}\n  recomputed: {recomputed}\n"
        f"  fix: run `python3 -c \"import json, hashlib, pathlib; p=pathlib.Path('"
        f"tests/benchmarks/corpus_rev4.json'); d=json.loads(p.read_text()); "
        f"body={{k:v for k,v in d.items() if k!='sha256_signature'}}; "
        f"body['issues']=sorted((dict(i) for i in body['issues']), key=lambda x:int(x['issue_id'])); "
        f"d['sha256_signature']=hashlib.sha256(json.dumps(body, sort_keys=True, "
        f"separators=(',',':')).encode()).hexdigest(); p.write_text(json.dumps(d, indent=2)+chr(10))\"`"
    )
