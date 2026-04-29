"""
Invariant tests for ``aragora.swarm.handoff_contract``.

The eight clauses (C1..C8) the contract encodes are tested here. This is
the single test fixture proposed by the spec doc
``docs/plans/2026-04-28-handoff-contract-derivation.md`` to assert the
invariants the 17 outbox/handoff fix PRs converged on.

The module is a pure-function skeleton with no I/O; these tests are all
synchronous, no fixtures, no mocks needed.
"""

from __future__ import annotations

import json

import pytest

from aragora.swarm.handoff_contract import (
    HandoffIdentity,
    InvalidHandoff,
    PR_OPEN_REQUEST_ACTIONS,
    PR_OPEN_REQUEST_CANONICAL_ACTION,
    REQUIRED_OUTBOX_KEYS,
    ReconcilePlan,
    SatisfactionContext,
    SatisfactionKind,
    SatisfactionSignal,
    TERMINAL_RECEIPT_STATUSES,
    compute_fingerprint,
    evaluate_satisfaction,
    is_dry_run_safe,
    parse_outbox_entry,
    plan_archive_satisfied,
)


# ---------------------------------------------------------------------------
# C1: Idempotency-key as primary identity
# ---------------------------------------------------------------------------


class TestC1_IdempotencyKey:
    def test_canonical_action_is_in_set(self) -> None:
        assert PR_OPEN_REQUEST_CANONICAL_ACTION in PR_OPEN_REQUEST_ACTIONS

    def test_required_keys_includes_idempotency(self) -> None:
        assert "idempotency_key" in REQUIRED_OUTBOX_KEYS

    def test_missing_idempotency_key_invalidates(self) -> None:
        payload = _valid_payload()
        del payload["idempotency_key"]
        result = parse_outbox_entry(payload)
        assert isinstance(result, InvalidHandoff)
        assert "idempotency_key" in result.missing_keys

    def test_empty_idempotency_key_invalidates(self) -> None:
        payload = _valid_payload(idempotency_key="")
        result = parse_outbox_entry(payload)
        assert isinstance(result, InvalidHandoff)


# ---------------------------------------------------------------------------
# C2: Terminal-state precedence
# ---------------------------------------------------------------------------


class TestC2_TerminalPrecedence:
    def test_terminal_statuses_set(self) -> None:
        assert "published" in TERMINAL_RECEIPT_STATUSES
        assert "already_satisfied" in TERMINAL_RECEIPT_STATUSES

    def test_terminal_receipt_wins_over_other_signals(self) -> None:
        identity = _identity()
        ctx = SatisfactionContext(
            terminal_receipt_keys=frozenset({identity.idempotency_key}),
            merged_pr_branches=frozenset({identity.branch_name or ""}),
            patch_equivalent_branches=frozenset({identity.branch_name or ""}),
            open_pr_heads={identity.branch_name or "": 999},
            receipt_only_branches=frozenset({identity.branch_name or ""}),
        )
        signal = evaluate_satisfaction(identity, ctx)
        assert signal is not None
        assert signal.kind == SatisfactionKind.TERMINAL_RECEIPT

    def test_no_signal_returns_none(self) -> None:
        identity = _identity()
        ctx = SatisfactionContext()
        assert evaluate_satisfaction(identity, ctx) is None


# ---------------------------------------------------------------------------
# C3: Out of scope here (state-root resolution lives in CLI scripts).
# C3 is asserted by the morning briefing's outbox-empty check.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C4: Branch field fingerprinting canonical
# ---------------------------------------------------------------------------


class TestC4_Fingerprinting:
    def test_fingerprint_stable_across_calls(self) -> None:
        a = compute_fingerprint("k1", "feat/x", "0" * 16, "open_pr")
        b = compute_fingerprint("k1", "feat/x", "0" * 16, "open_pr")
        assert a == b

    def test_fingerprint_changes_with_action(self) -> None:
        a = compute_fingerprint("k1", "feat/x", "0" * 16, "open_pr")
        b = compute_fingerprint("k1", "feat/x", "0" * 16, "open_or_update_pr")
        assert a != b

    def test_fingerprint_changes_with_head_sha(self) -> None:
        a = compute_fingerprint("k1", "feat/x", "0" * 16, "open_pr")
        b = compute_fingerprint("k1", "feat/x", "1" * 16, "open_pr")
        assert a != b

    def test_branch_resolution_local_evidence_mapping(self) -> None:
        payload = _valid_payload()
        payload["local_evidence"] = {
            "branch": "feat/from-mapping",
            "head_sha": "0" * 16,
        }
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)
        assert ident.branch_name == "feat/from-mapping"

    def test_branch_resolution_local_evidence_list(self) -> None:
        # Schema-drift case introduced by PR #6755.
        payload = _valid_payload()
        payload["local_evidence"] = [
            {"branch": "feat/from-list", "head_sha": "0" * 16},
            {"branch": "ignored"},
        ]
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)
        assert ident.branch_name == "feat/from-list"

    def test_branch_resolution_top_level_fallback(self) -> None:
        # Schema-drift case introduced by PRs #6595/#6596.
        # Per spec C4, when local_evidence is empty/missing AND requested_action
        # carries no branch, the fingerprint extractor falls back to the
        # top-level `branch` field. The handoff MUST parse — falling back
        # is the contract, not an InvalidHandoff path.
        payload = _valid_payload()
        payload["local_evidence"] = {}  # no branch here
        payload["requested_action"] = "open_pr"  # plain string; no nested branch
        payload["branch"] = "feat/top-level"
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)
        assert ident.branch_name == "feat/top-level"

    def test_branch_resolution_falsy_evidence_does_not_invalidate(self) -> None:
        # B1 regression: empty/falsy local_evidence must not pre-empt C4.
        # The handoff parses; branch resolution falls through to the next
        # available source (requested_action.branch in this fixture).
        payload = _valid_payload()
        payload["local_evidence"] = []  # empty list, was falsy-rejected pre-fix
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)
        # _valid_payload's requested_action carries branch=feat/x.
        assert ident.branch_name == "feat/x"

    def test_branch_resolution_precedence_local_evidence_beats_requested_action(self) -> None:
        # When both local_evidence.branch and requested_action.branch are set,
        # local_evidence wins (it's the canonical worker-stamped record;
        # requested_action is the higher-level intent).
        payload = _valid_payload()
        payload["local_evidence"] = {"branch": "feat/le-wins", "head_sha": "0" * 16}
        payload["requested_action"] = {"type": "open_pr", "branch": "feat/loses"}
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)
        assert ident.branch_name == "feat/le-wins"


# ---------------------------------------------------------------------------
# C5: Patch-equivalence is a satisfaction signal
# ---------------------------------------------------------------------------


class TestC5_PatchEquivalence:
    def test_patch_equivalent_branch_satisfies(self) -> None:
        identity = _identity(branch_name="feat/landed-elsewhere")
        ctx = SatisfactionContext(
            patch_equivalent_branches=frozenset({"feat/landed-elsewhere"}),
        )
        signal = evaluate_satisfaction(identity, ctx)
        assert signal is not None
        assert signal.kind == SatisfactionKind.PATCH_EQUIVALENT

    def test_merged_pr_beats_patch_equivalent(self) -> None:
        identity = _identity(branch_name="feat/x")
        ctx = SatisfactionContext(
            merged_pr_branches=frozenset({"feat/x"}),
            patch_equivalent_branches=frozenset({"feat/x"}),
        )
        signal = evaluate_satisfaction(identity, ctx)
        assert signal is not None
        assert signal.kind == SatisfactionKind.MERGED_PR


# ---------------------------------------------------------------------------
# C6: Open-PR identity beats outbox identity
# ---------------------------------------------------------------------------


class TestC6_OpenPRIdentity:
    def test_open_pr_match_satisfies(self) -> None:
        identity = _identity(branch_name="feat/x")
        ctx = SatisfactionContext(open_pr_heads={"feat/x": 1234})
        signal = evaluate_satisfaction(identity, ctx)
        assert signal is not None
        assert signal.kind == SatisfactionKind.OPEN_PR_MATCH
        assert signal.evidence["pr"] == 1234

    def test_patch_equivalent_beats_open_pr_match(self) -> None:
        identity = _identity(branch_name="feat/x")
        ctx = SatisfactionContext(
            patch_equivalent_branches=frozenset({"feat/x"}),
            open_pr_heads={"feat/x": 999},
        )
        signal = evaluate_satisfaction(identity, ctx)
        assert signal is not None
        assert signal.kind == SatisfactionKind.PATCH_EQUIVALENT


# ---------------------------------------------------------------------------
# C7: Dry-run is read-only or it's a bug
# ---------------------------------------------------------------------------


class TestC7_DryRunSafety:
    def test_pure_module_is_dry_run_safe(self) -> None:
        plan = ReconcilePlan(entries=(), base_ref="origin/main")
        assert is_dry_run_safe(plan) is True

    def test_plan_archive_satisfied_returns_plan(self) -> None:
        identity = _identity()
        ctx = SatisfactionContext(terminal_receipt_keys=frozenset({identity.idempotency_key}))
        plan = plan_archive_satisfied([identity], ctx, "origin/main")
        assert plan.archive_count() == 1
        assert plan.skip_count() == 0
        assert plan.publish_count() == 0
        assert is_dry_run_safe(plan) is True

    def test_plan_skips_unsatisfied(self) -> None:
        identity = _identity()
        plan = plan_archive_satisfied([identity], SatisfactionContext(), "origin/main")
        assert plan.skip_count() == 1
        assert plan.archive_count() == 0


# ---------------------------------------------------------------------------
# C8: Evidence schema loosely-typed but defensively-validated
# ---------------------------------------------------------------------------


class TestC8_EvidenceSchema:
    def test_required_keys_complete(self) -> None:
        payload = _valid_payload()
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)

    def test_each_required_key_individually_invalidates(self) -> None:
        for key in REQUIRED_OUTBOX_KEYS:
            payload = _valid_payload()
            del payload[key]
            result = parse_outbox_entry(payload)
            assert isinstance(result, InvalidHandoff), f"{key} should invalidate"

    def test_action_can_be_mapping(self) -> None:
        payload = _valid_payload()
        payload["requested_action"] = {"type": "open_pr", "branch": "feat/x"}
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)
        assert ident.action_kind == "open_pr"

    def test_action_can_be_json_string(self) -> None:
        payload = _valid_payload()
        payload["requested_action"] = json.dumps(
            {"action": "open_or_update_pr", "branch": "feat/x"}
        )
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)
        assert ident.action_kind == "open_or_update_pr"

    def test_action_can_be_plain_string(self) -> None:
        payload = _valid_payload()
        payload["requested_action"] = "open_pr"
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)
        assert ident.action_kind == "open_pr"

    def test_unknown_action_kind_invalidates(self) -> None:
        # B2 regression: action_kind outside PR_OPEN_REQUEST_ACTIONS must
        # produce InvalidHandoff. The whitelist is the contract's binding
        # constraint on what publisher actions are valid.
        payload = _valid_payload()
        payload["requested_action"] = "frobnicate"
        result = parse_outbox_entry(payload)
        assert isinstance(result, InvalidHandoff)
        assert "frobnicate" in result.reason

    def test_unknown_action_kind_in_mapping_invalidates(self) -> None:
        # B2 regression: same enforcement when action arrives via Mapping.
        payload = _valid_payload()
        payload["requested_action"] = {"type": "frobnicate"}
        result = parse_outbox_entry(payload)
        assert isinstance(result, InvalidHandoff)

    def test_unknown_action_kind_in_json_string_invalidates(self) -> None:
        # B2 regression: same enforcement when action arrives via JSON-string.
        payload = _valid_payload()
        payload["requested_action"] = json.dumps({"action": "frobnicate"})
        result = parse_outbox_entry(payload)
        assert isinstance(result, InvalidHandoff)

    def test_requires_github_false_is_valid(self) -> None:
        # B1 regression: requires_github=False is a valid value (boolean
        # False, not "missing"). Falsy-empty check used to reject this.
        payload = _valid_payload()
        payload["requires_github"] = False
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)

    def test_validation_empty_list_is_valid(self) -> None:
        # B1 regression: validation=[] is a valid value (no validation ran).
        # Falsy-empty check used to reject this.
        payload = _valid_payload()
        payload["validation"] = []
        ident = parse_outbox_entry(payload)
        assert isinstance(ident, HandoffIdentity)

    def test_non_mapping_input_produces_invalid_handoff(self) -> None:
        # Non-Mapping payload is programmer error; module must NOT raise.
        result = parse_outbox_entry("not a dict")  # type: ignore[arg-type]
        assert isinstance(result, InvalidHandoff)

    def test_invalid_handoff_to_dict_serializable(self) -> None:
        payload = _valid_payload()
        del payload["task"]
        result = parse_outbox_entry(payload)
        assert isinstance(result, InvalidHandoff)
        # Must be JSON-friendly so quarantine receipts can record it.
        json.dumps(result.to_dict())

    def test_handoff_identity_to_dict_serializable(self) -> None:
        ident = _identity()
        json.dumps(ident.to_dict())

    def test_satisfaction_signal_to_dict_serializable(self) -> None:
        sig = SatisfactionSignal(kind=SatisfactionKind.MERGED_PR, evidence={"branch": "feat/x"})
        json.dumps(sig.to_dict())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Synthetic identifiers for fixture data. Not credentials; not secrets.
# pragma: allowlist secret
_TEST_IDENTITY_KEY = "FAKE-NOT-A-SECRET-test-identity"
_TEST_SHA_PADDING = "0" * 16


def _valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task": "Open PR for feat/x",
        "requires_github": True,
        "requested_action": {"type": "open_pr", "branch": "feat/x"},
        "repo": "synaptent/aragora",
        "local_evidence": {"branch": "feat/x", "head_sha": _TEST_SHA_PADDING},
        "validation": [{"command": "pytest", "result": "ok"}],
        "idempotency_key": _TEST_IDENTITY_KEY,
        "created_at": "2026-04-28T01:00:00Z",
    }
    payload.update(overrides)
    return payload


def _identity(
    *,
    idempotency_key: str = _TEST_IDENTITY_KEY,
    branch_name: str = "feat/x",
    head_sha: str = _TEST_SHA_PADDING,
    action_kind: str = "open_pr",
) -> HandoffIdentity:
    fp = compute_fingerprint(idempotency_key, branch_name, head_sha, action_kind)
    return HandoffIdentity(
        idempotency_key=idempotency_key,
        branch_name=branch_name,
        head_sha=head_sha,
        action_kind=action_kind,
        fingerprint=fp,
    )
