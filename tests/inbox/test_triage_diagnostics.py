"""Tests for triage-scoped diagnostics capture."""

from __future__ import annotations

import io
import json
import logging
import warnings

from aragora.inbox.triage_diagnostics import TriageRunDiagnostics


def _root_logger_with_stream():
    root = logging.getLogger()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.WARNING)
    return root, handler, stream, previous_level


def test_capture_logging_suppresses_targeted_warning_and_records_event(tmp_path):
    diagnostics = TriageRunDiagnostics(
        profile="baseline",
        batch_size=1,
        auto_approve=False,
        dry_run=True,
        verbose=False,
        diagnostics_dir=tmp_path,
    )
    root, handler, stream, previous_level = _root_logger_with_stream()
    try:
        with (
            diagnostics.activate(),
            diagnostics.capture_logging(),
            diagnostics.message_scope("msg-1"),
        ):
            logging.getLogger("aragora.debate.phases.vote_collector").error(
                "vote_error agent=triage-proposer error=vote returned None",
                extra={
                    "triage_diag_code": "vote_none",
                    "triage_diag_severity": "degraded",
                },
            )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)

    assert stream.getvalue() == ""
    events = [json.loads(line) for line in diagnostics.events_path.read_text().splitlines() if line]
    assert len(events) == 1
    assert events[0]["code"] == "vote_none"
    assert events[0]["severity"] == "degraded"
    assert events[0]["message_id"] == "msg-1"


def test_capture_logging_mirrors_targeted_warning_in_verbose_mode(tmp_path):
    diagnostics = TriageRunDiagnostics(
        profile="baseline",
        batch_size=1,
        auto_approve=False,
        dry_run=True,
        verbose=True,
        diagnostics_dir=tmp_path,
    )
    root, handler, stream, previous_level = _root_logger_with_stream()
    try:
        with diagnostics.activate(), diagnostics.capture_logging():
            logging.getLogger("aragora.server.research_phase").warning(
                "[research] Anthropic generation failed (BadRequestError), falling back to OpenRouter",
                extra={
                    "triage_diag_code": "provider_fallback",
                    "triage_diag_severity": "degraded",
                },
            )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)

    assert "falling back to OpenRouter" in stream.getvalue()
    events = [json.loads(line) for line in diagnostics.events_path.read_text().splitlines() if line]
    assert events[0]["code"] == "provider_fallback"


def test_capture_logging_records_resource_warning_artifact(tmp_path):
    diagnostics = TriageRunDiagnostics(
        profile="baseline",
        batch_size=1,
        auto_approve=False,
        dry_run=True,
        verbose=False,
        diagnostics_dir=tmp_path,
    )
    root, handler, stream, previous_level = _root_logger_with_stream()
    try:
        with (
            diagnostics.activate(),
            diagnostics.capture_logging(),
            diagnostics.message_scope("msg-2"),
        ):
            warnings.warn("unclosed transport while testing", ResourceWarning)
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)

    assert stream.getvalue() == ""
    events = [json.loads(line) for line in diagnostics.events_path.read_text().splitlines() if line]
    assert events[0]["code"] == "resource_warning"
    assert events[0]["message_id"] == "msg-2"


def test_finalize_writes_meta_and_event_artifacts(tmp_path):
    diagnostics = TriageRunDiagnostics(
        profile="staged_v1",
        batch_size=2,
        auto_approve=False,
        dry_run=True,
        verbose=False,
        diagnostics_dir=tmp_path,
    )
    with diagnostics.activate(), diagnostics.message_scope("msg-3"):
        diagnostics.record_event(
            code="slow_round",
            severity="diagnostic",
            logger_name="aragora.debate.performance_monitor",
            summary="slow_round_detected debate_id=debate-3 round=1",
            details="threshold=30.0",
            tier="fast",
        )

    meta = diagnostics.finalize([])
    assert diagnostics.meta_path.exists()
    assert diagnostics.events_path.exists()
    written_meta = json.loads(diagnostics.meta_path.read_text())
    assert written_meta["profile"] == "staged_v1"
    assert written_meta["artifact_dir"] == str(diagnostics.artifact_dir)
    assert meta["severity_counts"]["diagnostic"] == 1
