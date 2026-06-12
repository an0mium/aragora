"""Tests for the Art. 14 / NIST AI 600-1 oversight evidence pack (#8230)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aragora.compliance.oversight_pack import (
    ART14_NIST_CROSSWALK,
    build_oversight_pack,
    collect_github_settlements,
    collect_local_settlements,
    collect_trail_anchors,
)

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
WINDOW_START = datetime(2026, 6, 1, tzinfo=timezone.utc)

HEAD_8169 = "bdb4ee5b9c9f4d7c130ef75cb01cd0530a62f241"


def _local_receipt(pr_number: int, reviewed_at: str) -> dict:
    return {
        "session_id": f"recorded-{pr_number}",
        "reviewed_at": reviewed_at,
        "actor": "an0mium",
        "action": "admin_squash_merge",
        "reason": "external exact-head admin squash observed",
        "pr_number": pr_number,
        "pr_url": f"https://github.com/synaptent/aragora/pull/{pr_number}",
        "head_sha": "5af592dd619c02c023088a0625ff70f2a3c1d5c6",
        "packet_sha": "sha256:" + "f" * 64,
        "github_event": "ADMIN_SQUASH_MERGE",
    }


@pytest.fixture()
def receipts_dir(tmp_path):
    directory = tmp_path / "receipts"
    directory.mkdir()
    (directory / "pr-8154-recorded.json").write_text(
        json.dumps(_local_receipt(8154, "2026-06-11T04:34:42+00:00"))
    )
    # Outside the window: must be excluded.
    (directory / "pr-7000-recorded.json").write_text(
        json.dumps(_local_receipt(7000, "2026-04-01T00:00:00+00:00"))
    )
    # No reviewed_at: recency unknown, excluded from a windowed pack.
    undated = _local_receipt(7500, "")
    del undated["reviewed_at"]
    (directory / "pr-7500-recorded.json").write_text(json.dumps(undated))
    # Corrupt file: skipped, never invented.
    (directory / "pr-9999-bad.json").write_text("{not json")
    return directory


class TestCollectLocalSettlements:
    def test_window_filtering_and_honesty(self, receipts_dir):
        decisions = collect_local_settlements(receipts_dir, window_start=WINDOW_START)
        labels = [d["label"] for d in decisions]
        assert labels == ["PR #8154"]
        att = decisions[0]["attestation"]
        assert att["disposition"] == "human_attested"
        assert att["attestor_id"] == "an0mium"
        assert decisions[0]["odr_attestation"]["attestor"]["id"] == "an0mium"

    def test_missing_directory_is_empty_not_error(self, tmp_path):
        assert collect_local_settlements(tmp_path / "nope", window_start=WINDOW_START) == []


class TestCollectGithubSettlements:
    def _gh(self, calls):
        """Stubbed gh fetcher: one human-settled PR, one autonomous PR."""

        def gh_json(args):
            calls.append(args)
            if args[0] == "pr":
                return [
                    {
                        "number": 8169,
                        "title": "B1 evidence re-eval",
                        "mergedAt": "2026-06-11T19:39:26Z",
                        "headRefOid": HEAD_8169,
                        "url": "https://github.com/synaptent/aragora/pull/8169",
                    },
                    {
                        "number": 8239,
                        "title": "ODR spine",
                        "mergedAt": "2026-06-11T10:00:00Z",
                        "headRefOid": "c" * 40,
                        "url": "https://github.com/synaptent/aragora/pull/8239",
                    },
                ]
            if args[0] == "api" and f"commits/{HEAD_8169}/statuses" in args[1]:
                return [
                    {
                        "context": "aragora/human-settlement",
                        "state": "success",
                        "creator": {"login": "an0mium"},
                        "created_at": "2026-06-11T18:14:09Z",
                        "target_url": "https://github.com/synaptent/aragora/pull/8169#issuecomment-1",
                    }
                ]
            if args[0] == "api" and "/statuses" in args[1]:
                return []
            if args[0] == "api" and "issues/8169/comments" in args[1]:
                return []
            raise AssertionError(f"unexpected gh call: {args}")

        return gh_json

    def test_classification(self):
        calls: list = []
        decisions, notes = collect_github_settlements(
            repo="synaptent/aragora",
            window_start=WINDOW_START,
            gh_json=self._gh(calls),
        )
        assert notes == []
        by_pr = {d["attestation"]["subject"]["pr_number"]: d for d in decisions}
        assert by_pr[8169]["attestation"]["disposition"] == "human_attested"
        assert by_pr[8169]["attestation"]["attestor_id"] == "an0mium"
        assert by_pr[8239]["attestation"]["disposition"] == "autonomous"
        assert "model-quorum" in by_pr[8239]["attestation"]["observed"]["non_intervention_reason"]
        # Autonomous PRs must not trigger comment fetches (rate-light).
        assert not any("issues/8239" in str(c) for c in calls)

    def test_listing_failure_degrades_to_note(self):
        def gh_json(args):
            raise RuntimeError("rate limited")

        decisions, notes = collect_github_settlements(
            repo="synaptent/aragora", window_start=WINDOW_START, gh_json=gh_json
        )
        assert decisions == []
        assert any("failed" in n for n in notes)


class TestCollectTrailAnchors:
    def test_missing_chain_is_graceful(self, tmp_path):
        result = collect_trail_anchors(tmp_path / "intent-chain.jsonl", window_start=WINDOW_START)
        assert result["records"] == []
        assert "absent" in result["note"]

    def test_real_chain_records(self, tmp_path):
        from aragora.trail import intent_chain

        chain = tmp_path / "intent-chain.jsonl"
        intent_chain.append_intent(
            chain,
            actor_class="human",
            intent_type="merge_pr",
            target={"repo": "synaptent/aragora", "pr": 8169},
        )
        result = collect_trail_anchors(chain, window_start=WINDOW_START)
        assert result["verified"] is True
        assert result["head_hash"]
        assert len(result["records"]) == 1
        assert result["records"][0]["record_hash"]


class TestBuildOversightPack:
    def test_local_only_pack(self, receipts_dir, tmp_path):
        pack = build_oversight_pack(
            window_days=11,
            receipts_dir=receipts_dir,
            trail_chain_path=tmp_path / "intent-chain.jsonl",
            now=NOW,
        )
        assert pack.human_attested_count == 1
        assert pack.autonomous_count == 0
        assert any("local settlement-receipt store only" in n for n in pack.notes)
        data = pack.to_dict()
        assert data["integrity_hash"]
        assert data["regulatory_crosswalk"]["rows"] == ART14_NIST_CROSSWALK

    def test_markdown_contains_crosswalk_and_attestations(self, receipts_dir, tmp_path):
        pack = build_oversight_pack(
            window_days=11,
            receipts_dir=receipts_dir,
            trail_chain_path=tmp_path / "missing.jsonl",
            now=NOW,
        )
        md = pack.to_markdown()
        assert "EU AI Act Article 14 / NIST AI 600-1 crosswalk" in md
        assert "14(4)(e)" in md
        assert "GV-3.2" in md
        assert "PR #8154" in md
        assert "an0mium" in md
        assert "docs/specs/OPEN_DECISION_RECEIPT.md" in md
        # Crosswalk maps evidence availability, not legal conformity.
        assert "not legal conformity" in md

    def test_github_layer_merges_with_local_corroboration(self, receipts_dir, tmp_path):
        def gh_json(args):
            if args[0] == "pr":
                return [
                    {
                        "number": 8154,
                        "title": "settled while offline",
                        "mergedAt": "2026-06-11T04:34:42Z",
                        "headRefOid": "5af592dd619c02c023088a0625ff70f2a3c1d5c6",
                        "url": "https://github.com/synaptent/aragora/pull/8154",
                    }
                ]
            if args[0] == "api" and "/statuses" in args[1]:
                return []
            raise AssertionError(f"unexpected: {args}")

        pack = build_oversight_pack(
            window_days=11,
            repo="synaptent/aragora",
            receipts_dir=receipts_dir,
            trail_chain_path=tmp_path / "missing.jsonl",
            gh_json=gh_json,
            now=NOW,
        )
        # One decision for PR 8154 (GitHub classification), with the local
        # receipt attached as a corroborating source rather than duplicated.
        labels = [d["label"] for d in pack.decisions]
        assert labels.count("PR #8154") == 1
        decision = pack.decisions[0]
        assert decision["source"] == "github_merged_pr"
        corroborating = decision.get("corroborating_sources") or []
        assert corroborating and corroborating[0]["type"] == "local_settlement_receipt"

    def test_integrity_hash_is_stable(self, receipts_dir, tmp_path):
        kwargs = dict(
            window_days=11,
            receipts_dir=receipts_dir,
            trail_chain_path=tmp_path / "missing.jsonl",
            now=NOW,
        )
        assert (
            build_oversight_pack(**kwargs).to_dict()["integrity_hash"]
            == build_oversight_pack(**kwargs).to_dict()["integrity_hash"]
        )


class TestCrosswalkIsCanonical:
    def test_rows_match_odr_spec_fields(self):
        fields = [row["odr_field"] for row in ART14_NIST_CROSSWALK]
        # The ten rows of OPEN_DECISION_RECEIPT.md section 7, in order.
        assert fields == [
            "subject (binding + digest)",
            "claim.verdict",
            "reasoning.summary",
            "quorum.participants + independence",
            "quorum.dissent",
            "confidence + calibration",
            "cruxes",
            "attestation",
            "signatures / JCS digest (section 5-6)",
            "source",
        ]

    def test_no_invented_clauses(self):
        # Every Art. 14 reference must be one the ODR spec cites: 14(1) or 14(4)(a-e).
        import re

        for row in ART14_NIST_CROSSWALK:
            refs = re.findall(r"14\(\d\)(?:\([a-e]\))?", row["eu_ai_act_art14"])
            assert refs, f"row has no Art.14 citation: {row['odr_field']}"
            for ref in refs:
                assert ref in {
                    "14(1)",
                    "14(4)(a)",
                    "14(4)(b)",
                    "14(4)(c)",
                    "14(4)(d)",
                    "14(4)(e)",
                }


class TestCli:
    def test_cmd_oversight_pack_writes_bundle(self, receipts_dir, tmp_path, capsys):
        import argparse

        from aragora.cli.commands.compliance import _cmd_oversight_pack

        out_dir = tmp_path / "pack"
        args = argparse.Namespace(
            window=30,
            repo="",
            receipts_dir=str(receipts_dir),
            trail_chain=str(tmp_path / "missing.jsonl"),
            output=str(out_dir),
            output_format="all",
            github_limit=100,
            json=False,
        )
        _cmd_oversight_pack(args)
        captured = capsys.readouterr()
        assert "Human-Oversight Evidence Pack" in captured.out
        assert (out_dir / "oversight_pack.json").is_file()
        assert (out_dir / "oversight_pack.md").is_file()
        payload = json.loads((out_dir / "oversight_pack.json").read_text())
        assert payload["kind"] == "aragora.oversight_evidence_pack"
        assert payload["summary"]["decisions"] >= 1
