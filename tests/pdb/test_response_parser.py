"""Tests for :mod:`aragora.pdb.response_parser`.

Exercises the full set of malformed-response cases enumerated in the
mission brief: clean JSON, JSON in a code fence, JSON with comments,
truncated responses, leading prose, and pure-text-no-JSON. The parsers
must never raise.
"""

from __future__ import annotations

from aragora.pdb.response_parser import (
    PARSE_FAILURE_REASON,
    extract_json_object,
    parse_critique_response,
    parse_findings_response,
    parse_synthesis_response,
    position_from_string,
)
from aragora.review.protocol import DissentPosition


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_clean_json(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        text = 'Here is the output:\n```json\n{"recommendation": "approve"}\n```\n'
        assert extract_json_object(text) == {"recommendation": "approve"}

    def test_fenced_without_language(self) -> None:
        text = '```\n{"k": 2}\n```'
        assert extract_json_object(text) == {"k": 2}

    def test_with_line_comments(self) -> None:
        text = '// this is a comment\n{"a": 1, "b": 2}\n# trailing comment'
        assert extract_json_object(text) == {"a": 1, "b": 2}

    def test_leading_prose(self) -> None:
        text = 'My analysis:\n\n{"recommendation": "request_changes", "confidence": 0.7}\n\nThanks.'
        result = extract_json_object(text)
        assert result == {"recommendation": "request_changes", "confidence": 0.7}

    def test_truncated_returns_none(self) -> None:
        # Missing closing brace; last-brace fallback cannot recover.
        assert extract_json_object('{"a": 1, "b":') is None

    def test_plain_text_returns_none(self) -> None:
        assert extract_json_object("This is just prose, no JSON at all.") is None

    def test_empty_or_whitespace_returns_none(self) -> None:
        assert extract_json_object("") is None
        assert extract_json_object("   \n\t  ") is None

    def test_non_object_top_level_returns_none(self) -> None:
        # Arrays are not dicts — we only accept objects at the top level.
        assert extract_json_object("[1, 2, 3]") is None

    def test_prefers_fenced_body_when_outer_also_parses(self) -> None:
        # Outer + fenced both parse; fenced is the canonical model output.
        text = '```json\n{"source": "fenced"}\n```\n{"source": "outer"}'
        assert extract_json_object(text) == {"source": "fenced"}


# ---------------------------------------------------------------------------
# position_from_string
# ---------------------------------------------------------------------------


class TestPositionFromString:
    def test_approve(self) -> None:
        assert position_from_string("approve") is DissentPosition.APPROVE
        assert position_from_string("APPROVE") is DissentPosition.APPROVE
        assert position_from_string("approve_candidate") is DissentPosition.APPROVE

    def test_request_changes(self) -> None:
        assert position_from_string("request_changes") is DissentPosition.REQUEST_CHANGES
        assert position_from_string("request-changes") is DissentPosition.REQUEST_CHANGES
        assert position_from_string("block") is DissentPosition.REQUEST_CHANGES

    def test_defer(self) -> None:
        assert position_from_string("defer") is DissentPosition.DEFER
        assert position_from_string("escalate") is DissentPosition.DEFER

    def test_unknown_fallback(self) -> None:
        assert position_from_string("unknown") is DissentPosition.DEFER
        assert position_from_string(None) is DissentPosition.DEFER  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_findings_response
# ---------------------------------------------------------------------------


def _findings_json() -> str:
    return (
        '{"recommendation": "approve", '
        '"confidence": 0.85, '
        '"top_findings": [{"finding_id": "s1-F1", "category": "logic", '
        '"severity": "medium", "summary": "Race condition in counter", '
        '"evidence": ["rate_limit.py:42"]}], '
        '"contested_finding_ids": ["s1-F1"], '
        '"reason": "Code is correct modulo the race."}'
    )


class TestParseFindingsResponse:
    def test_happy_path(self) -> None:
        result = parse_findings_response(_findings_json(), slot_id="s1")
        assert result["parsed"] is True
        assert result["position"] is DissentPosition.APPROVE
        assert result["confidence"] == 0.85
        assert len(result["top_findings"]) == 1
        f = result["top_findings"][0]
        assert f["finding_id"] == "s1-F1"
        assert f["severity"] == "medium"
        assert f["evidence"] == ["rate_limit.py:42"]
        assert result["contested_finding_ids"] == ("s1-F1",)
        assert "correct" in result["reason"]

    def test_fenced_json_parses(self) -> None:
        text = f"Here's the analysis:\n```json\n{_findings_json()}\n```"
        result = parse_findings_response(text, slot_id="s1")
        assert result["parsed"] is True
        assert result["position"] is DissentPosition.APPROVE

    def test_unparseable_returns_safe_defaults(self) -> None:
        result = parse_findings_response("not json", slot_id="s1")
        assert result["parsed"] is False
        assert result["position"] is DissentPosition.DEFER
        assert result["confidence"] == 0.0
        assert result["top_findings"] == []
        assert result["contested_finding_ids"] == ()
        assert result["reason"] == PARSE_FAILURE_REASON

    def test_confidence_clamped_into_range(self) -> None:
        text = '{"recommendation": "approve", "confidence": 2.5}'
        result = parse_findings_response(text, slot_id="s1")
        assert result["confidence"] == 1.0

    def test_negative_confidence_clamped(self) -> None:
        text = '{"recommendation": "approve", "confidence": -0.3}'
        result = parse_findings_response(text, slot_id="s1")
        assert result["confidence"] == 0.0

    def test_non_numeric_confidence_defaults_to_zero(self) -> None:
        text = '{"recommendation": "approve", "confidence": "high"}'
        result = parse_findings_response(text, slot_id="s1")
        assert result["confidence"] == 0.0

    def test_severity_normalization(self) -> None:
        text = (
            '{"recommendation": "request_changes", "top_findings": '
            '[{"finding_id": "F1", "severity": "CRITICAL", "summary": "x"}]}'
        )
        result = parse_findings_response(text, slot_id="s1")
        assert result["top_findings"][0]["severity"] == "medium"  # fallback

    def test_caps_top_findings_at_five(self) -> None:
        items = ",".join(
            f'{{"finding_id": "F{i}", "category": "c", "severity": "low", "summary": "s{i}"}}'
            for i in range(10)
        )
        text = f'{{"recommendation": "approve", "top_findings": [{items}]}}'
        result = parse_findings_response(text, slot_id="s1")
        assert len(result["top_findings"]) == 5

    def test_missing_finding_id_is_synthesized(self) -> None:
        text = (
            '{"recommendation": "approve", "top_findings": '
            '[{"category": "c", "severity": "low", "summary": "s"}]}'
        )
        result = parse_findings_response(text, slot_id="alpha")
        assert result["top_findings"][0]["finding_id"] == "alpha-F1"

    def test_ignores_non_dict_findings(self) -> None:
        text = '{"recommendation": "approve", "top_findings": [123, "string", null]}'
        result = parse_findings_response(text, slot_id="s1")
        assert result["top_findings"] == []

    def test_raw_text_preview_included_on_failure(self) -> None:
        long_text = "random prose " * 100
        result = parse_findings_response(long_text, slot_id="s1")
        assert result["parsed"] is False
        assert "raw_text_preview" in result
        assert len(result["raw_text_preview"]) <= 500


# ---------------------------------------------------------------------------
# parse_critique_response
# ---------------------------------------------------------------------------


class TestParseCritiqueResponse:
    def test_happy_path(self) -> None:
        text = (
            '{"recommendation": "request_changes", '
            '"confidence": 0.6, '
            '"agrees_with": ["claude_core"], '
            '"disagrees_with": ["grok_heterodox"], '
            '"contested_finding_ids": ["F1"], '
            '"reason": "I stand by my critique."}'
        )
        result = parse_critique_response(text, slot_id="s1")
        assert result["parsed"] is True
        assert result["position"] is DissentPosition.REQUEST_CHANGES
        assert result["agrees_with"] == ("claude_core",)
        assert result["disagrees_with"] == ("grok_heterodox",)
        assert "stand by" in result["reason"]

    def test_unparseable_returns_defaults(self) -> None:
        result = parse_critique_response("garbage", slot_id="s1")
        assert result["parsed"] is False
        assert result["position"] is DissentPosition.DEFER
        assert result["confidence"] == 0.0
        assert result["agrees_with"] == ()

    def test_handles_missing_fields_gracefully(self) -> None:
        text = '{"recommendation": "approve"}'
        result = parse_critique_response(text, slot_id="s1")
        assert result["parsed"] is True
        assert result["confidence"] == 0.0
        assert result["reason"] == ""
        assert result["agrees_with"] == ()


# ---------------------------------------------------------------------------
# parse_synthesis_response
# ---------------------------------------------------------------------------


class TestParseSynthesisResponse:
    def test_happy_path(self) -> None:
        text = (
            '{"top_line": "Panel leans approve with one dissent.", '
            '"validation_summary": "CI green; one logic concern.", '
            '"preserved_dissent": [{"slot_id": "grok_h", "lens": "heterodox", '
            '"position": "request_changes", "reason": "Hidden coupling."}]}'
        )
        result = parse_synthesis_response(text)
        assert result["parsed"] is True
        assert "approve" in result["top_line"]
        assert "green" in result["validation_summary"]
        assert len(result["preserved_dissent"]) == 1
        d = result["preserved_dissent"][0]
        assert d["slot_id"] == "grok_h"
        assert d["position"] == "request_changes"

    def test_unparseable_returns_failure_top_line(self) -> None:
        result = parse_synthesis_response("not json at all")
        assert result["parsed"] is False
        assert PARSE_FAILURE_REASON in result["top_line"]
        assert result["preserved_dissent"] == ()

    def test_ignores_non_dict_dissent(self) -> None:
        text = '{"top_line": "t", "validation_summary": "v", "preserved_dissent": [123, "str"]}'
        result = parse_synthesis_response(text)
        assert result["preserved_dissent"] == ()
