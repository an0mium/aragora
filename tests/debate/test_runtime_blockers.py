"""Tests for stderr runtime-blocker classification."""

from __future__ import annotations

from aragora.debate.runtime_blockers import classify_stderr_signals


def test_classify_stderr_warning_only_resourcewarning():
    stderr = (
        "ResourceWarning: unclosed <socket.socket ...>\n"
        "ResourceWarning: Enable tracemalloc to get the object allocation traceback\n"
        "Unclosed connector\n"
    )
    result = classify_stderr_signals(stderr)
    assert result["runtime_blockers"] == []
    assert "resource_warning" in result["warning_signals"]
    assert result["warning_only"] is True


def test_classify_stderr_detects_traceback_blocker():
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "AttributeError: boom\n"
    )
    result = classify_stderr_signals(stderr)
    assert "traceback" in result["runtime_blockers"]
    assert "attribute_error" in result["runtime_blockers"]
    assert result["warning_only"] is False


def test_classify_stderr_detects_timeout_and_warning():
    stderr = (
        "Debate timed out after 120s\n"
        "ResourceWarning: unclosed transport <_SelectorSocketTransport fd=63>\n"
    )
    result = classify_stderr_signals(stderr)
    assert "debate_timeout" in result["runtime_blockers"]
    assert "unclosed_transport" in result["warning_signals"]
    assert result["warning_only"] is False
