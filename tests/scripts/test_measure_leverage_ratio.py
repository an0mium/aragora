"""Tests for scripts/measure_leverage_ratio.py (steering program Phase 0.2).

All network and subprocess boundaries are injected; no gh / git / aragora
calls happen in these tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.measure_leverage_ratio import (
    METHODOLOGY_VERSION,
    SI_COMPONENTS_PENDING,
    compute_leverage,
    find_receipt_refs,
    main,
    render_lr_block,
    resolve_receipt_path,
    update_leverage_md,
)

NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
WINDOW_START = datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)


def _pr(number: int, body: str = "") -> dict:
    return {
        "number": number,
        "title": f"pr {number}",
        "body": body,
        "merged_at": "2026-06-10T08:00:00Z",
    }


# ---------------------------------------------------------------------------
# Receipt reference extraction / resolution
# ---------------------------------------------------------------------------


class TestFindReceiptRefs:
    def test_extracts_receipt_paths(self) -> None:
        text = (
            "Receipt artifact: .aragora/run-20260610/receipts/c03_instruments_grok.json\n"
            "and also `receipts/other_one.json` inline."
        )
        refs = find_receipt_refs(text)
        assert ".aragora/run-20260610/receipts/c03_instruments_grok.json" in refs
        assert any(r.endswith("receipts/other_one.json") for r in refs)

    def test_ignores_non_receipt_json(self) -> None:
        assert find_receipt_refs("see config/settings.json and data.json") == []

    def test_empty_text(self) -> None:
        assert find_receipt_refs("") == []

    def test_deduplicates(self) -> None:
        text = "receipts/a.json receipts/a.json"
        assert len(find_receipt_refs(text)) == 1


class TestResolveReceiptPath:
    def test_resolves_by_basename_in_receipts_dir(self, tmp_path: Path) -> None:
        rdir = tmp_path / "receipts"
        rdir.mkdir()
        (rdir / "x.json").write_text("{}")
        got = resolve_receipt_path(".aragora/run-x/receipts/x.json", [rdir])
        assert got == rdir / "x.json"

    def test_resolves_direct_path(self, tmp_path: Path) -> None:
        rdir = tmp_path / "receipts"
        rdir.mkdir()
        p = rdir / "y.json"
        p.write_text("{}")
        assert resolve_receipt_path(str(p), []) == p

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert resolve_receipt_path("receipts/nope.json", [tmp_path]) is None


# ---------------------------------------------------------------------------
# Core LR computation
# ---------------------------------------------------------------------------


class TestComputeLeverage:
    def _receipts_dir(self, tmp_path: Path) -> Path:
        rdir = tmp_path / "receipts"
        rdir.mkdir()
        (rdir / "good.json").write_text("{}")
        (rdir / "bad.json").write_text("{}")
        return rdir

    def test_counts_and_ratio(self, tmp_path: Path) -> None:
        rdir = self._receipts_dir(tmp_path)
        prs = [
            _pr(1, body="Receipt artifact: receipts/good.json"),
            _pr(2, body="Receipt artifact: receipts/bad.json"),
            _pr(3, body="no receipts here"),
        ]
        result = compute_leverage(
            merged_prs=prs,
            operator_minutes=25.0,
            receipts_dirs=[rdir],
            comments_fetcher=lambda n: [],
            verifier=lambda p: p.name == "good.json",
            window_start=WINDOW_START,
            window_end=NOW,
            window_days=7,
            repo="synaptent/aragora",
        )
        assert result["merged_total"] == 3
        assert result["merged_receipt_backed"] == 1
        assert result["receipts_failed_verify"] == 1
        assert result["failed_verify_paths"] == [str(rdir / "bad.json")]
        assert result["leverage_ratio"] == pytest.approx(1 / 25.0)
        assert result["methodology_version"] == METHODOLOGY_VERSION

    def test_steering_integrity_is_null_never_a_number(self, tmp_path: Path) -> None:
        result = compute_leverage(
            merged_prs=[],
            operator_minutes=10.0,
            receipts_dirs=[tmp_path],
            comments_fetcher=lambda n: [],
            verifier=lambda p: True,
            window_start=WINDOW_START,
            window_end=NOW,
            window_days=7,
            repo="synaptent/aragora",
        )
        assert result["steering_integrity"] is None
        assert result["si_components_pending"] == SI_COMPONENTS_PENDING
        assert result["si_components_pending"] == [
            "crux_shown",
            "within_attention_budget",
            "not_reversed_on_audit",
        ]

    def test_receipt_ref_in_comments_counts(self, tmp_path: Path) -> None:
        rdir = self._receipts_dir(tmp_path)
        result = compute_leverage(
            merged_prs=[_pr(7)],
            operator_minutes=5.0,
            receipts_dirs=[rdir],
            comments_fetcher=lambda n: ["Receipt artifact: receipts/good.json"],
            verifier=lambda p: True,
            window_start=WINDOW_START,
            window_end=NOW,
            window_days=7,
            repo="synaptent/aragora",
        )
        assert result["merged_receipt_backed"] == 1

    def test_ref_to_missing_local_receipt_not_backed_not_silent(self, tmp_path: Path) -> None:
        result = compute_leverage(
            merged_prs=[_pr(8, body="receipts/ghost.json")],
            operator_minutes=5.0,
            receipts_dirs=[tmp_path],
            comments_fetcher=lambda n: [],
            verifier=lambda p: True,
            window_start=WINDOW_START,
            window_end=NOW,
            window_days=7,
            repo="synaptent/aragora",
        )
        assert result["merged_receipt_backed"] == 0
        assert result["receipt_refs_unresolved"] == 1

    def test_failed_verify_pr_with_another_good_receipt_still_backed(self, tmp_path: Path) -> None:
        rdir = self._receipts_dir(tmp_path)
        result = compute_leverage(
            merged_prs=[_pr(9, body="receipts/bad.json and receipts/good.json")],
            operator_minutes=5.0,
            receipts_dirs=[rdir],
            comments_fetcher=lambda n: [],
            verifier=lambda p: p.name == "good.json",
            window_start=WINDOW_START,
            window_end=NOW,
            window_days=7,
            repo="synaptent/aragora",
        )
        assert result["merged_receipt_backed"] == 1
        assert result["receipts_failed_verify"] == 1


# ---------------------------------------------------------------------------
# CLI refusal contract: operator-minutes must never be invented
# ---------------------------------------------------------------------------


class TestOperatorMinutesRefusal:
    def test_refuses_without_operator_minutes(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--json"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "operator-minutes" in err
        assert "operator-estimated" in err

    def test_refuses_zero_operator_minutes(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--operator-minutes", "0"])
        assert rc == 2

    def test_refuses_negative_operator_minutes(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--operator-minutes", "-3"])
        assert rc == 2


class TestMainJson:
    def test_json_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        rdir = tmp_path / "receipts"
        rdir.mkdir()
        (rdir / "good.json").write_text("{}")

        import scripts.measure_leverage_ratio as mlr

        monkeypatch.setattr(
            mlr,
            "fetch_merged_prs",
            lambda repo, since: [_pr(1, body="receipts/good.json"), _pr(2)],
        )
        monkeypatch.setattr(mlr, "fetch_issue_comments", lambda repo, n: [])
        monkeypatch.setattr(mlr, "verify_receipt", lambda p: True)

        rc = main(
            [
                "--operator-minutes",
                "25",
                "--receipts-dir",
                str(rdir),
                "--json",
            ]
        )
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["merged_total"] == 2
        assert out["merged_receipt_backed"] == 1
        assert out["leverage_ratio"] == pytest.approx(1 / 25.0)
        assert out["steering_integrity"] is None
        assert out["operator_minutes"] == 25.0
        assert "self-reported" in out["operator_minutes_source"]


# ---------------------------------------------------------------------------
# LEVERAGE.md managed-region rendering
# ---------------------------------------------------------------------------


def _lr_result(**overrides) -> dict:
    base = {
        "methodology_version": METHODOLOGY_VERSION,
        "repo": "synaptent/aragora",
        "window_days": 7,
        "window_start": "2026-06-10T00:00:00Z",
        "window_end": "2026-06-10T12:00:00Z",
        "operator_minutes": 25.0,
        "operator_minutes_source": "self-reported",
        "operator_minutes_note": "baseline day estimate",
        "merged_total": 9,
        "merged_receipt_backed": 4,
        "unique_receipts_backed": 4,
        "split_factor": 1.0,
        "receipts_failed_verify": 0,
        "receipt_refs_unresolved": 1,
        "failed_verify_paths": [],
        "leverage_ratio": 0.16,
        "steering_integrity": None,
        "si_components_pending": SI_COMPONENTS_PENDING,
    }
    base.update(overrides)
    return base


class TestUpdateLeverageMd:
    def test_creates_skeleton_with_all_sections(self, tmp_path: Path) -> None:
        doc = tmp_path / "LEVERAGE.md"
        update_leverage_md(doc, lr_block=render_lr_block(_lr_result()))
        text = doc.read_text()
        assert "<!-- leverage-managed:begin -->" in text
        assert "<!-- leverage-managed:end -->" in text
        assert "Last updated:" in text
        assert "## Leverage Ratio" in text
        assert "## Waste Ratio" in text
        assert "## Blind-Period Log" in text
        assert "## Caveats" in text
        assert "self-reported" in text
        assert "not yet instrumented" in text
        # waste not yet measured -> placeholder
        assert "not yet measured" in text

    def test_manual_text_outside_region_preserved(self, tmp_path: Path) -> None:
        doc = tmp_path / "LEVERAGE.md"
        update_leverage_md(doc, lr_block=render_lr_block(_lr_result()))
        manual = "- 2026-05-27 -> 2026-06-05: loop was blind (manual entry).\n"
        doc.write_text(doc.read_text() + manual)
        update_leverage_md(doc, lr_block=render_lr_block(_lr_result(merged_total=11)))
        text = doc.read_text()
        assert manual in text
        assert "| Merged PRs in window (total) | 11 |" in text
        assert text.count("<!-- leverage-managed:begin -->") == 1

    def test_other_metric_block_preserved_on_partial_update(self, tmp_path: Path) -> None:
        doc = tmp_path / "LEVERAGE.md"
        update_leverage_md(doc, waste_block="WASTE-SENTINEL-CONTENT")
        update_leverage_md(doc, lr_block="LR-SENTINEL-CONTENT")
        text = doc.read_text()
        assert "WASTE-SENTINEL-CONTENT" in text
        assert "LR-SENTINEL-CONTENT" in text

    def test_appends_managed_region_to_unmanaged_file(self, tmp_path: Path) -> None:
        doc = tmp_path / "LEVERAGE.md"
        doc.write_text("# Hand-written header\n\nkeep me\n")
        update_leverage_md(doc, lr_block="LR-X")
        text = doc.read_text()
        assert text.startswith("# Hand-written header")
        assert "keep me" in text
        assert "LR-X" in text

    def test_render_lr_block_si_row_is_null(self) -> None:
        block = render_lr_block(_lr_result())
        assert "null" in block
        assert "crux_shown" in block
        assert "0.16" in block


class TestUpdateLeverageMdIdempotency:
    def test_rerender_does_not_glue_following_text_to_end_marker(self, tmp_path: Path) -> None:
        doc = tmp_path / "LEVERAGE.md"
        update_leverage_md(doc, lr_block="LR-1")
        update_leverage_md(doc, lr_block="LR-2")
        update_leverage_md(doc, waste_block="W-1")
        text = doc.read_text()
        assert "<!-- leverage-managed:end -->\n" in text
        assert "<!-- leverage-managed:end -->#" not in text
        # blind-period heading still starts at a line boundary
        assert "\n## Blind-Period Log" in text


class TestAntiSplitting:
    """LR gameability guard: PR-splitting inflates merged_receipt_backed but
    not unique_receipts_backed; split_factor surfaces the divergence."""

    def test_unique_receipts_and_split_factor(self, tmp_path: Path) -> None:
        rdir = tmp_path / "receipts"
        rdir.mkdir()
        (rdir / "shared.json").write_text("{}")
        prs = [
            _pr(1, body="receipts/shared.json"),
            _pr(2, body="receipts/shared.json"),
            _pr(3, body="receipts/shared.json"),
        ]
        result = compute_leverage(
            merged_prs=prs,
            operator_minutes=10.0,
            receipts_dirs=[rdir],
            comments_fetcher=lambda n: [],
            verifier=lambda p: True,
            window_start=WINDOW_START,
            window_end=NOW,
            window_days=7,
            repo="synaptent/aragora",
        )
        assert result["merged_receipt_backed"] == 3
        assert result["unique_receipts_backed"] == 1
        assert result["split_factor"] == pytest.approx(3.0)

    def test_split_factor_one_when_distinct(self, tmp_path: Path) -> None:
        rdir = tmp_path / "receipts"
        rdir.mkdir()
        (rdir / "a.json").write_text("{}")
        (rdir / "b.json").write_text("{}")
        prs = [_pr(1, body="receipts/a.json"), _pr(2, body="receipts/b.json")]
        result = compute_leverage(
            merged_prs=prs,
            operator_minutes=10.0,
            receipts_dirs=[rdir],
            comments_fetcher=lambda n: [],
            verifier=lambda p: True,
            window_start=WINDOW_START,
            window_end=NOW,
            window_days=7,
            repo="synaptent/aragora",
        )
        assert result["unique_receipts_backed"] == 2
        assert result["split_factor"] == pytest.approx(1.0)

    def test_no_backed_prs_zero_unique(self, tmp_path: Path) -> None:
        result = compute_leverage(
            merged_prs=[_pr(1)],
            operator_minutes=10.0,
            receipts_dirs=[tmp_path],
            comments_fetcher=lambda n: [],
            verifier=lambda p: True,
            window_start=WINDOW_START,
            window_end=NOW,
            window_days=7,
            repo="synaptent/aragora",
        )
        assert result["unique_receipts_backed"] == 0
        assert result["split_factor"] == 0.0
