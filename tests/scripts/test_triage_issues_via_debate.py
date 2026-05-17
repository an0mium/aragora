"""Tests for the calibration-only multi-model issue triage tool.

Coverage targets:
- Evidence gathering does not penalize automation authorship by default.
- Stratified sampler covers heterogeneous buckets.
- Prompt rubric is locked (regression guard).
- Per-model JSON parser is robust to fences, prose, malformed payloads.
- Aggregator handles unanimous / majority / split / all-error cases.
- Receipts persist all receipt-equivalent fields.
- CLI ``--estimate`` produces a cost projection without invoking models.
- Budget cap halts evaluation mid-run.

No tests hit external services. All GitHub + agent calls are stubbed.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.triage import (  # noqa: E402
    IssueDebateReceipt,
    IssueEvidence,
    IssueRecord,
    PanelMember,
    PerModelVerdict,
    aggregate_verdicts,
    build_panel,
    build_panel_prompt,
    estimate_cost_usd,
    evaluate_issue,
    gather_evidence,
    is_automation_generated,
    parse_model_response,
    write_jsonl_receipt,
    write_markdown_report,
)
from aragora.triage.evidence import extract_file_references, extract_issue_references
from aragora.triage.issue_evaluator import (
    AUTOMATION_VALUE_VALUES,
    CONFIDENCE_CLASSES,
    DEFAULT_PANEL,
    PANEL_PROMPT_RUBRIC,
    VERDICT_CATEGORIES,
    FounderRecommendation,
    _aggregate_confidence_class,
    _build_recommendation,
)
from collections import Counter


def _make_issue(
    number: int = 1,
    *,
    title: str = "Open PR for typo fix",
    body: str = "Auto-generated. See `aragora/foo.py`.",
    author: str = "an0mium",
    labels: tuple[str, ...] = ("boss-stuck",),
    state: str = "open",
) -> IssueRecord:
    return IssueRecord(
        number=number,
        title=title,
        body=body,
        author=author,
        labels=labels,
        state=state,
        url=f"https://github.com/synaptent/aragora/issues/{number}",
        created_at="2026-05-10T00:00:00Z",
        updated_at="2026-05-10T01:00:00Z",
        comments_count=0,
    )


def _stub_panel(*, members: Sequence[PanelMember] | None = None) -> list[PanelMember]:
    return list(members or DEFAULT_PANEL)


def _verdict(
    *,
    member: PanelMember,
    verdict: str = "keep",
    confidence: float = 0.8,
    automation_value: str = "valuable",
    confidence_class: str = "needs-spot-check",
    what_to_inspect: str = "inspect body",
    safety_note: str = "low risk",
    refined_title: str = "",
    refined_body_outline: str = "",
    consolidate_with: int | None = None,
    error: str | None = None,
) -> PerModelVerdict:
    return PerModelVerdict(
        panel_member=member,
        verdict=verdict,
        confidence=confidence,
        confidence_class=confidence_class,
        automation_value=automation_value,
        rationale="rationale",
        suggested_action="do thing",
        evidence_used=["body"],
        what_to_inspect=what_to_inspect,
        safety_note=safety_note,
        refined_title=refined_title,
        refined_body_outline=refined_body_outline,
        consolidate_with=consolidate_with,
        raw_response="{}",
        prompt_chars=100,
        response_chars=50,
        cost_usd=0.01,
        latency_seconds=0.5,
        error=error,
    )


def test_verdict_categories_are_stable():
    assert "keep" in VERDICT_CATEGORIES
    assert "flag-for-human" in VERDICT_CATEGORIES
    assert "close-duplicate" in VERDICT_CATEGORIES
    assert len(VERDICT_CATEGORIES) == 7


def test_automation_value_includes_positive_outcome():
    assert "valuable" in AUTOMATION_VALUE_VALUES
    assert "noise" in AUTOMATION_VALUE_VALUES
    assert "n/a" in AUTOMATION_VALUE_VALUES


def test_panel_prompt_rubric_explicit_about_automation_not_being_bad():
    assert "SUBSTANTIVE VALUE" in PANEL_PROMPT_RUBRIC
    assert "Automation-generated" in PANEL_PROMPT_RUBRIC
    assert "valuable" in PANEL_PROMPT_RUBRIC


def test_is_automation_generated_handles_bots_and_labels():
    assert is_automation_generated(author="github-actions[bot]")
    assert is_automation_generated(author="an0mium")
    assert is_automation_generated(author="alice", labels=("stage-gate-drift",))
    assert is_automation_generated(
        author="alice", labels=(), body="This issue was opened by the swarm boss loop."
    )
    assert not is_automation_generated(author="founder", labels=("bug",), body="repro: ...")


def test_extract_file_references_finds_python_paths():
    refs = extract_file_references(
        "See aragora/foo.py and scripts/bar.sh and docs/x.md. Also `aragora/baz.py`."
    )
    assert "aragora/foo.py" in refs
    assert "scripts/bar.sh" in refs
    assert "docs/x.md" in refs
    assert "aragora/baz.py" in refs


def test_extract_issue_references_excludes_self():
    refs = extract_issue_references("Related #123 fixes #456 closes #7172.", exclude=123)
    assert 123 not in refs
    assert 456 in refs
    assert 7172 in refs


def test_gather_evidence_marks_broken_file_refs(tmp_path: Path):
    (tmp_path / "aragora").mkdir()
    (tmp_path / "aragora" / "existing.py").write_text("# real")
    issue = _make_issue(
        body="Refers to aragora/existing.py and aragora/ghost.py for fixes.",
    )
    evidence = gather_evidence(
        issue,
        repo="synaptent/aragora",
        repo_root=tmp_path,
        open_issue_index=[],
        gh_runner=lambda args: None,
        now_iso="2026-05-14T12:00:00Z",
    )
    paths = {ref["path"]: ref["exists_in_head"] for ref in evidence.referenced_files}
    assert paths["aragora/existing.py"] is True
    assert paths["aragora/ghost.py"] is False
    assert evidence.is_automation_generated is True


def test_gather_evidence_marks_backslash_repo_file_refs_existing(tmp_path: Path):
    (tmp_path / "aragora" / "triage").mkdir(parents=True)
    (tmp_path / "aragora" / "triage" / "evidence.py").write_text("# real")
    issue = _make_issue(
        body=r"Windows logs mention aragora\triage\evidence.py during triage.",
    )

    evidence = gather_evidence(
        issue,
        repo="synaptent/aragora",
        repo_root=tmp_path,
        open_issue_index=[],
        gh_runner=lambda args: None,
        now_iso="2026-05-14T12:00:00Z",
    )

    paths = {ref["path"]: ref["exists_in_head"] for ref in evidence.referenced_files}
    assert paths[r"aragora\triage\evidence.py"] is True


def test_gather_evidence_suggests_duplicates(tmp_path: Path):
    target = _make_issue(number=1, title="Narrow broad except Exception in foo")
    others = [
        _make_issue(number=2, title="Narrow broad except Exception in bar"),
        _make_issue(number=3, title="Add unit tests for foo"),
    ]
    evidence = gather_evidence(
        target,
        repo="x/y",
        repo_root=tmp_path,
        open_issue_index=others,
        gh_runner=lambda args: None,
        now_iso="2026-05-14T12:00:00Z",
    )
    dup_numbers = [c["number"] for c in evidence.duplicate_candidates]
    assert 2 in dup_numbers
    assert 3 not in dup_numbers


def test_build_panel_default_and_filter():
    full = build_panel()
    assert len(full) == 3
    filtered = build_panel(["anthropic-api", "openai-api"])
    assert {m.agent_type for m in filtered} == {"anthropic-api", "openai-api"}


def test_build_panel_rejects_solo_panel():
    with pytest.raises(ValueError):
        build_panel(["anthropic-api"])


def test_build_panel_rejects_unknown_agent():
    with pytest.raises(ValueError):
        build_panel(["anthropic-api", "fictional-agent"])


def test_build_panel_prompt_contains_evidence_block(tmp_path: Path):
    issue = _make_issue(body="See aragora/x.py")
    evidence = gather_evidence(
        issue,
        repo="x/y",
        repo_root=tmp_path,
        open_issue_index=[],
        gh_runner=lambda args: None,
        now_iso="2026-05-14T12:00:00Z",
    )
    prompt = build_panel_prompt(evidence)
    assert "ISSUE" in prompt
    assert "BODY" in prompt
    assert "REFERENCED FILES" in prompt
    assert "DUPLICATE CANDIDATES" in prompt
    assert "EVIDENCE BLOCK" in prompt
    assert "aragora/x.py" in prompt


def test_parse_model_response_strict_object():
    raw = (
        '{"verdict":"keep","confidence":0.9,"automation_value":"valuable",'
        '"rationale":"r","suggested_action":"do","evidence_used":["body"]}'
    )
    parsed = parse_model_response(raw)
    assert parsed["verdict"] == "keep"
    assert parsed["confidence"] == pytest.approx(0.9)
    assert parsed["automation_value"] == "valuable"


def test_parse_model_response_handles_fence_and_prose():
    raw = (
        "Here is my analysis.\n"
        "```json\n"
        '{"verdict":"refine","confidence":0.6,"automation_value":"neutral",'
        '"rationale":"needs scope","suggested_action":"tighten","evidence_used":[]}\n'
        "```\nThanks."
    )
    parsed = parse_model_response(raw)
    assert parsed["verdict"] == "refine"


def test_parse_model_response_rejects_invalid_verdict():
    raw = '{"verdict":"explode","confidence":0.5,"automation_value":"n/a","rationale":"r"}'
    with pytest.raises(ValueError):
        parse_model_response(raw)


def test_parse_model_response_clamps_confidence():
    raw = '{"verdict":"keep","confidence":1.5,"automation_value":"valuable","rationale":"r","suggested_action":"x","evidence_used":[]}'
    parsed = parse_model_response(raw)
    assert parsed["confidence"] == pytest.approx(1.0)


def test_parse_model_response_empty_string():
    with pytest.raises(ValueError):
        parse_model_response("")


def test_aggregate_verdicts_unanimous():
    panel = _stub_panel()
    per_model = [_verdict(member=m, verdict="keep", confidence=0.9) for m in panel]
    agg = aggregate_verdicts(per_model)
    assert agg.verdict == "keep"
    assert agg.consensus == "unanimous"
    assert agg.confidence == pytest.approx(0.9)


def test_aggregate_verdicts_majority_wins():
    panel = _stub_panel()
    per_model = [
        _verdict(member=panel[0], verdict="keep", confidence=0.8),
        _verdict(member=panel[1], verdict="keep", confidence=0.75),
        _verdict(member=panel[2], verdict="refine", confidence=0.6),
    ]
    agg = aggregate_verdicts(per_model)
    assert agg.verdict == "keep"
    assert agg.consensus == "majority"


def test_aggregate_verdicts_split_low_confidence_flags_human():
    panel = _stub_panel()
    per_model = [
        _verdict(member=panel[0], verdict="keep", confidence=0.4),
        _verdict(member=panel[1], verdict="refine", confidence=0.5),
        _verdict(member=panel[2], verdict="close-duplicate", confidence=0.4),
    ]
    agg = aggregate_verdicts(per_model)
    assert agg.verdict == "flag-for-human"
    assert agg.consensus == "split"


def test_aggregate_verdicts_split_high_confidence_picks_highest():
    panel = _stub_panel()
    per_model = [
        _verdict(member=panel[0], verdict="keep", confidence=0.9),
        _verdict(member=panel[1], verdict="refine", confidence=0.7),
        _verdict(member=panel[2], verdict="close-duplicate", confidence=0.6),
    ]
    agg = aggregate_verdicts(per_model)
    assert agg.verdict == "keep"
    assert agg.consensus == "split"


def test_aggregate_verdicts_all_errors_flags_human():
    panel = _stub_panel()
    per_model = [_verdict(member=m, verdict="keep", confidence=0.0, error="timeout") for m in panel]
    agg = aggregate_verdicts(per_model)
    assert agg.verdict == "flag-for-human"
    assert agg.consensus == "unclear"


def test_estimate_cost_usd_scales_with_issue_count():
    panel = _stub_panel()
    one = estimate_cost_usd(panel=panel, issue_count=1)
    thirty = estimate_cost_usd(panel=panel, issue_count=30)
    assert thirty["total_usd"] > one["total_usd"]
    assert thirty["issues"] == 30
    assert set(one["per_model"].keys()) == {m.model_id for m in panel}


def test_evaluate_issue_uses_injected_generator(tmp_path: Path):
    issue = _make_issue()
    evidence = gather_evidence(
        issue,
        repo="x/y",
        repo_root=tmp_path,
        open_issue_index=[],
        gh_runner=lambda args: None,
        now_iso="2026-05-14T12:00:00Z",
    )

    responses = {
        "claude-opus-4-7": '{"verdict":"keep","confidence":0.9,"automation_value":"valuable","rationale":"r","suggested_action":"a","evidence_used":[]}',
        "gpt-4.1": '{"verdict":"keep","confidence":0.8,"automation_value":"valuable","rationale":"r","suggested_action":"a","evidence_used":[]}',
        "gemini-3.1-pro-preview": '{"verdict":"refine","confidence":0.6,"automation_value":"neutral","rationale":"r","suggested_action":"a","evidence_used":[]}',
    }

    async def generator(member: PanelMember, prompt: str) -> str:
        return responses[member.model_id]

    receipt = asyncio.run(evaluate_issue(evidence, generator=generator))
    assert receipt.aggregate_verdict == "keep"
    assert receipt.aggregate_consensus == "majority"
    assert len(receipt.per_model) == 3
    assert all(entry.get("raw_response") for entry in receipt.per_model)
    assert receipt.automation_value in AUTOMATION_VALUE_VALUES


def test_evaluate_issue_handles_model_error(tmp_path: Path):
    issue = _make_issue()
    evidence = gather_evidence(
        issue,
        repo="x/y",
        repo_root=tmp_path,
        open_issue_index=[],
        gh_runner=lambda args: None,
        now_iso="2026-05-14T12:00:00Z",
    )

    async def generator(member: PanelMember, prompt: str) -> str:
        if member.model_id == "gpt-4.1":
            raise RuntimeError("api down")
        return '{"verdict":"keep","confidence":0.9,"automation_value":"valuable","rationale":"r","suggested_action":"a","evidence_used":[]}'

    receipt = asyncio.run(evaluate_issue(evidence, generator=generator))
    errored = [pm for pm in receipt.per_model if pm.get("error")]
    assert len(errored) == 1
    assert receipt.aggregate_verdict == "keep"


def test_evaluate_issue_records_untyped_provider_exception(tmp_path: Path):
    class ProviderSDKError(Exception):
        pass

    issue = _make_issue()
    evidence = gather_evidence(
        issue,
        repo="x/y",
        repo_root=tmp_path,
        open_issue_index=[],
        gh_runner=lambda args: None,
        now_iso="2026-05-14T12:00:00Z",
    )

    async def generator(member: PanelMember, prompt: str) -> str:
        if member.model_id == "claude-opus-4-7":
            raise ProviderSDKError("provider overloaded")
        return '{"verdict":"keep","confidence":0.9,"automation_value":"valuable","rationale":"r","suggested_action":"a","evidence_used":[]}'

    receipt = asyncio.run(evaluate_issue(evidence, generator=generator))
    errored = [pm for pm in receipt.per_model if pm.get("error")]
    assert len(errored) == 1
    assert errored[0]["error"] == "ProviderSDKError: provider overloaded"
    assert receipt.aggregate_verdict == "keep"


def test_evaluate_issue_handles_parse_error(tmp_path: Path):
    issue = _make_issue()
    evidence = gather_evidence(
        issue,
        repo="x/y",
        repo_root=tmp_path,
        open_issue_index=[],
        gh_runner=lambda args: None,
        now_iso="2026-05-14T12:00:00Z",
    )

    async def generator(member: PanelMember, prompt: str) -> str:
        if member.model_id == "gemini-3.1-pro-preview":
            return "this is not json"
        return '{"verdict":"keep","confidence":0.85,"automation_value":"valuable","rationale":"r","suggested_action":"a","evidence_used":[]}'

    receipt = asyncio.run(evaluate_issue(evidence, generator=generator))
    parse_errors = [
        pm for pm in receipt.per_model if (pm.get("error") or "").startswith("parse_error")
    ]
    assert len(parse_errors) == 1
    assert receipt.aggregate_verdict == "keep"


def test_write_jsonl_and_markdown_roundtrip(tmp_path: Path):
    panel = _stub_panel()
    per_model = [_verdict(member=panel[0])]
    receipt = IssueDebateReceipt(
        issue_number=42,
        issue_title="Test",
        issue_url="https://example/issues/42",
        issue_author="alice",
        is_automation_generated=False,
        panel=[panel[0].model_id],
        prompt="prompt",
        per_model=[pm.to_dict() for pm in per_model],
        aggregate_verdict="keep",
        aggregate_confidence=0.9,
        aggregate_consensus="unanimous",
        aggregation_rationale="r",
        automation_value="n/a",
        suggested_action="action",
        evidence={"referenced_files": []},
        started_at="2026-05-14T00:00:00Z",
        finished_at="2026-05-14T00:01:00Z",
        cost_usd=0.01,
        latency_seconds=1.0,
    )
    jsonl_path = tmp_path / "receipts.jsonl"
    write_jsonl_receipt(jsonl_path, receipt)
    assert jsonl_path.exists()
    line = jsonl_path.read_text().strip().splitlines()[0]
    loaded = json.loads(line)
    assert loaded["issue_number"] == 42
    md_path = tmp_path / "report.md"
    write_markdown_report(md_path, [receipt])
    text = md_path.read_text()
    assert "Verdict distribution" in text
    assert "Automation-value cross-tab" in text
    assert "automation_value=valuable" in text


def test_stratified_sample_covers_buckets():
    from scripts.triage_issues_via_debate import stratified_sample

    issues: list[IssueRecord] = []
    for i in range(50):
        issues.append(_make_issue(number=i, author="an0mium", labels=("boss-stuck",)))
    for i in range(50, 60):
        issues.append(_make_issue(number=i, author="founder", labels=("bug",)))
    for i in range(60, 70):
        issues.append(
            _make_issue(number=i, author="github-actions[bot]", labels=("stage-gate-drift",))
        )

    sample = stratified_sample(issues, sample_size=15, seed=42)
    assert len(sample) == 15
    authors = {iss.author for iss in sample}
    assert authors >= {"an0mium", "founder", "github-actions[bot]"}


def test_stratified_sample_zero_returns_empty():
    from scripts.triage_issues_via_debate import stratified_sample

    assert stratified_sample([_make_issue()], sample_size=0) == []


def test_cli_estimate_prints_projection_without_calling_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from scripts import triage_issues_via_debate as cli

    issues = [_make_issue(number=i) for i in range(40)]
    monkeypatch.setattr(cli, "fetch_open_issues", lambda repo, **_: issues)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("estimate must not invoke agents")

    monkeypatch.setattr(cli, "_agent_generator_factory", boom)

    rc = cli.main(
        [
            "--repo",
            "x/y",
            "--sample",
            "10",
            "--estimate",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Cost projection" in out
    assert "total_usd" in out


def test_cli_budget_cap_aborts_when_projected_exceeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from scripts import triage_issues_via_debate as cli

    issues = [_make_issue(number=i) for i in range(100)]
    monkeypatch.setattr(cli, "fetch_open_issues", lambda repo, **_: issues)

    rc = cli.main(
        [
            "--repo",
            "x/y",
            "--sample",
            "30",
            "--budget-usd",
            "0.01",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "exceeds budget" in captured.out + captured.err


def test_cli_issues_mode_uses_full_open_index_for_duplicate_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from scripts import triage_issues_via_debate as cli

    target = _make_issue(number=10, title="Narrow broad except Exception in foo")
    duplicate = _make_issue(number=11, title="Narrow broad except Exception in bar")
    monkeypatch.setattr(cli, "fetch_specific_issues", lambda repo, numbers, **_: [target])
    monkeypatch.setattr(cli, "fetch_open_issues", lambda repo, **_: [target, duplicate])

    rc = cli.main(
        [
            "--repo",
            "x/y",
            "--issues",
            "10",
            "--dry-run-prompt",
            "--budget-usd",
            "10",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "DUPLICATE CANDIDATES" in out
    assert "#11" in out


def test_jsonl_resume_skips_completed_issues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scripts import triage_issues_via_debate as cli

    issues = [_make_issue(number=i) for i in range(3)]
    monkeypatch.setattr(cli, "fetch_specific_issues", lambda repo, numbers, **_: issues)
    monkeypatch.setattr(cli, "fetch_open_issues", lambda repo, **_: issues)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "receipts.jsonl").write_text(
        json.dumps(
            {
                "issue_number": 0,
                "issue_title": "x",
                "aggregate_verdict": "keep",
                "aggregate_confidence": 0.9,
                "aggregate_consensus": "unanimous",
            }
        )
        + "\n"
    )

    async def stub_generator(*_args: Any, **_kwargs: Any) -> Any:
        async def gen(_member: PanelMember, _prompt: str) -> str:
            return '{"verdict":"keep","confidence":0.9,"automation_value":"valuable","rationale":"r","suggested_action":"a","evidence_used":[]}'

        return gen

    monkeypatch.setattr(cli, "_agent_generator_factory", stub_generator)
    monkeypatch.setattr(
        cli,
        "gather_evidence",
        lambda issue, **kwargs: IssueEvidence(
            issue=issue,
            is_automation_generated=True,
            gathered_at="2026-05-14T00:00:00Z",
        ),
    )

    rc = cli.main(
        [
            "--repo",
            "x/y",
            "--issues",
            "0,1,2",
            "--budget-usd",
            "100",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    written = (output_dir / "receipts.jsonl").read_text().splitlines()
    issue_numbers = {json.loads(line)["issue_number"] for line in written}
    assert {0, 1, 2}.issubset(issue_numbers)


def test_confidence_classes_are_stable():
    assert "easy-call" in CONFIDENCE_CLASSES
    assert "needs-spot-check" in CONFIDENCE_CLASSES
    assert "do-not-act-without-human" in CONFIDENCE_CLASSES


def test_rubric_teaches_founder_to_inspect_evidence():
    assert "TEACH" in PANEL_PROMPT_RUBRIC
    assert "non-expert" in PANEL_PROMPT_RUBRIC
    assert "what_to_inspect" in PANEL_PROMPT_RUBRIC
    assert "safety_note" in PANEL_PROMPT_RUBRIC
    assert "refined_title" in PANEL_PROMPT_RUBRIC
    assert "consolidate_with" in PANEL_PROMPT_RUBRIC
    assert "confidence_class" in PANEL_PROMPT_RUBRIC


def test_parse_model_response_accepts_new_fields():
    raw = json.dumps(
        {
            "verdict": "refine",
            "confidence": 0.78,
            "confidence_class": "needs-spot-check",
            "automation_value": "valuable",
            "rationale": "real bug",
            "suggested_action": "refine",
            "evidence_used": ["body para 1"],
            "what_to_inspect": "inspect line 142",
            "safety_note": "reversible",
            "refined_title": "narrow except in cost_tracker.py",
            "refined_body_outline": "- repro\n- proposed catch list\n- test plan",
            "consolidate_with": 6371,
        }
    )
    parsed = parse_model_response(raw)
    assert parsed["confidence_class"] == "needs-spot-check"
    assert parsed["what_to_inspect"] == "inspect line 142"
    assert parsed["safety_note"] == "reversible"
    assert parsed["refined_title"].startswith("narrow except")
    assert "- repro" in parsed["refined_body_outline"]
    assert parsed["consolidate_with"] == 6371


def test_parse_model_response_falls_back_to_inferred_confidence_class():
    raw = json.dumps(
        {
            "verdict": "keep",
            "confidence": 0.91,
            "automation_value": "valuable",
            "rationale": "ok",
            "suggested_action": "keep",
            "evidence_used": [],
        }
    )
    parsed = parse_model_response(raw)
    assert parsed["confidence_class"] == "easy-call"


def test_parse_model_response_handles_invalid_consolidate_with():
    raw = json.dumps(
        {
            "verdict": "consolidate",
            "confidence": 0.6,
            "automation_value": "neutral",
            "rationale": "merge",
            "suggested_action": "consolidate",
            "evidence_used": [],
            "consolidate_with": "not-a-number",
        }
    )
    parsed = parse_model_response(raw)
    assert parsed["consolidate_with"] is None


def test_aggregate_confidence_class_unanimous_high_is_easy_call():
    panel = _stub_panel()
    pm = [
        _verdict(member=panel[0], confidence=0.92, confidence_class="easy-call"),
        _verdict(member=panel[1], confidence=0.88, confidence_class="easy-call"),
        _verdict(member=panel[2], confidence=0.85, confidence_class="easy-call"),
    ]
    cls = _aggregate_confidence_class("unanimous", 0.88, pm)
    assert cls == "easy-call"


def test_aggregate_confidence_class_any_do_not_act_dominates():
    panel = _stub_panel()
    pm = [
        _verdict(member=panel[0], confidence=0.95, confidence_class="easy-call"),
        _verdict(member=panel[1], confidence=0.92, confidence_class="easy-call"),
        _verdict(member=panel[2], confidence=0.6, confidence_class="do-not-act-without-human"),
    ]
    cls = _aggregate_confidence_class("majority", 0.82, pm)
    assert cls == "do-not-act-without-human"


def test_aggregate_confidence_class_split_is_do_not_act():
    panel = _stub_panel()
    pm = [
        _verdict(member=panel[0], confidence=0.9, confidence_class="easy-call"),
        _verdict(member=panel[1], confidence=0.85, confidence_class="easy-call"),
        _verdict(member=panel[2], confidence=0.8, confidence_class="easy-call"),
    ]
    cls = _aggregate_confidence_class("split", 0.85, pm)
    assert cls == "do-not-act-without-human"


def test_build_recommendation_carries_refined_title_and_consolidate_with():
    panel = _stub_panel()
    pm = [
        _verdict(
            member=panel[0],
            verdict="refine",
            refined_title="tighter scope",
            refined_body_outline="- repro\n- proposed fix",
            what_to_inspect="inspect line 142",
            safety_note="reversible",
        ),
        _verdict(
            member=panel[1],
            verdict="refine",
            what_to_inspect="check test coverage",
        ),
    ]
    rec = _build_recommendation(
        verdict="refine",
        confidence_class="needs-spot-check",
        consensus="majority",
        valid=pm,
        verdict_counts=Counter({"refine": 2}),
        suggested="refine the issue",
    )
    assert rec.refined_title == "tighter scope"
    assert "repro" in rec.refined_body_outline
    assert "inspect line 142" in rec.inspect
    assert "reversible" in rec.safety


def test_build_recommendation_do_not_act_warns_loudly():
    panel = _stub_panel()
    pm = [
        _verdict(
            member=panel[0],
            verdict="close-duplicate",
            consolidate_with=6371,
            what_to_inspect="check #6371",
        ),
    ]
    rec = _build_recommendation(
        verdict="close-duplicate",
        confidence_class="do-not-act-without-human",
        consensus="split",
        valid=pm,
        verdict_counts=Counter({"close-duplicate": 1, "refine": 1, "keep": 1}),
        suggested="close as dup",
    )
    assert rec.consolidate_with == 6371
    assert "DO NOT ACT" in rec.action


def test_aggregate_verdicts_attaches_recommendation_and_confidence_class():
    panel = _stub_panel()
    pm = [
        _verdict(member=panel[0], verdict="keep", confidence=0.92, confidence_class="easy-call"),
        _verdict(member=panel[1], verdict="keep", confidence=0.9, confidence_class="easy-call"),
        _verdict(member=panel[2], verdict="keep", confidence=0.88, confidence_class="easy-call"),
    ]
    agg = aggregate_verdicts(pm)
    assert agg.confidence_class == "easy-call"
    assert agg.recommendation is not None
    assert "Leave the issue open" in agg.recommendation.action


def test_aggregate_verdicts_all_errors_flag_do_not_act():
    panel = _stub_panel()
    pm = [_verdict(member=m, confidence=0.0, error="timeout") for m in panel]
    agg = aggregate_verdicts(pm)
    assert agg.confidence_class == "do-not-act-without-human"
    assert agg.recommendation is not None
    assert "UNSAFE" in (agg.recommendation.safety or "")


def test_markdown_report_includes_how_to_review_guide_and_cards(tmp_path: Path):
    panel = _stub_panel()
    pm = [
        _verdict(
            member=panel[0],
            verdict="refine",
            refined_title="tighter scope",
            what_to_inspect="inspect line 142",
            safety_note="reversible",
        ),
        _verdict(member=panel[1], verdict="refine", confidence=0.7),
        _verdict(member=panel[2], verdict="keep", confidence=0.6),
    ]
    agg = aggregate_verdicts(pm)
    receipt = IssueDebateReceipt(
        issue_number=99,
        issue_title="example",
        issue_url="https://example/99",
        issue_author="an0mium",
        is_automation_generated=True,
        panel=[m.model_id for m in panel],
        prompt="prompt",
        per_model=[v.to_dict() for v in pm],
        aggregate_verdict=agg.verdict,
        aggregate_confidence=agg.confidence,
        aggregate_consensus=agg.consensus,
        aggregation_rationale=agg.rationale,
        confidence_class=agg.confidence_class,
        recommendation=agg.recommendation.to_dict() if agg.recommendation else None,
        automation_value=agg.automation_value,
        suggested_action=agg.suggested_action,
        evidence={
            "issue": {
                "body": "Real bug in aragora/billing/cost_tracker.py line 142",
                "labels": ["boss-stuck", "automation"],
            },
            "referenced_files": [
                {"path": "aragora/billing/cost_tracker.py", "exists_in_head": True},
            ],
            "referenced_issues": [
                {"number": 6371, "title": "Related broad except", "state": "OPEN"},
            ],
            "duplicate_candidates": [
                {"number": 6371, "similarity": 0.62, "title": "Related broad except"},
            ],
        },
        started_at="2026-05-15T00:00:00Z",
        finished_at="2026-05-15T00:00:05Z",
        cost_usd=0.05,
        latency_seconds=4.5,
    )
    out = tmp_path / "report.md"
    write_markdown_report(out, [receipt])
    text = out.read_text(encoding="utf-8")
    assert "How to review this report" in text
    assert "Evidence summary" in text
    assert "Per-model verdicts" in text
    assert "Founder-facing recommendation" in text
    assert "Confidence-class distribution" in text
    assert "Suggested refined title" in text
    assert "What to inspect" in text


def test_sample_card_cli_renders_without_models(capsys: pytest.MonkeyPatch):
    from scripts import triage_issues_via_debate as cli

    rc = cli.main(["--sample-card"])
    out = capsys.readouterr().out if hasattr(capsys, "readouterr") else ""
    assert rc == 0
    assert "Sample" in out or "sample" in out or "Issue Triage" in out
    assert "How to review this report" in out
    assert "Founder-facing recommendation" in out
    assert "9999" in out


def test_receipt_schema_version_is_one_dot_one():
    from aragora.triage import RECEIPT_SCHEMA_VERSION

    assert RECEIPT_SCHEMA_VERSION == "triage-receipt/1.1"


def test_issue_triage_guide_documents_current_receipt_schema():
    from aragora.triage import RECEIPT_SCHEMA_VERSION

    guide = (REPO_ROOT / "docs/guides/ISSUE_TRIAGE.md").read_text(encoding="utf-8")
    assert f"Schema version: `{RECEIPT_SCHEMA_VERSION}`" in guide
