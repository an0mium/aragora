"""Tests for settlement resolver backends."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

from aragora.scheduler.settlement_resolvers import SettlementResolverRegistry


class TestSettlementResolverRegistry:
    def test_human_resolver_waits_without_adjudication(self) -> None:
        now = datetime.now(timezone.utc)
        registry = SettlementResolverRegistry()
        decision = registry.resolve(
            "human",
            receipt_data={"mode": "epistemic_hygiene"},
            settlement={},
            now=now,
        )
        assert decision.resolved is False
        assert decision.reason == "waiting_for_human_adjudication"

    def test_human_resolver_accepts_explicit_adjudication(self) -> None:
        now = datetime.now(timezone.utc)
        registry = SettlementResolverRegistry()
        decision = registry.resolve(
            "human",
            receipt_data={"mode": "epistemic_hygiene"},
            settlement={"human_outcome": True},
            now=now,
        )
        assert decision.resolved is True
        assert decision.outcome is True

    def test_deterministic_resolver_evaluates_rule(self) -> None:
        now = datetime.now(timezone.utc)
        registry = SettlementResolverRegistry()
        decision = registry.resolve(
            "deterministic",
            receipt_data={"mode": "epistemic_hygiene"},
            settlement={
                "deterministic_rule": {
                    "observed": 94,
                    "operator": ">=",
                    "target": 95,
                }
            },
            now=now,
        )
        assert decision.resolved is True
        assert decision.outcome is False
        assert decision.reason == "deterministic_rule_evaluated"

    def test_oracle_resolver_verifies_signature(self, monkeypatch) -> None:
        now = datetime.now(timezone.utc)
        monkeypatch.setenv("ARAGORA_ORACLE_HMAC_SECRET", "scope-secret")
        signed_at = now.isoformat()
        canonical = json.dumps(
            {"outcome": True, "source": "oracle-feed", "signed_at": signed_at},
            sort_keys=True,
            separators=(",", ":"),
        )
        signature = hmac.new(
            b"scope-secret",
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        registry = SettlementResolverRegistry()
        decision = registry.resolve(
            "oracle",
            receipt_data={"mode": "epistemic_hygiene"},
            settlement={
                "oracle_attestation": {
                    "outcome": True,
                    "source": "oracle-feed",
                    "signed_at": signed_at,
                    "signature": signature,
                }
            },
            now=now,
        )
        assert decision.resolved is True
        assert decision.outcome is True
        assert decision.reason == "oracle_attestation_verified"

    def test_unknown_resolver_type(self) -> None:
        now = datetime.now(timezone.utc)
        registry = SettlementResolverRegistry()
        decision = registry.resolve(
            "unknown",
            receipt_data={},
            settlement={},
            now=now,
        )
        assert decision.resolved is False
        assert decision.reason == "unknown_resolver_type"
