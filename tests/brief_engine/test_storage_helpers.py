"""Tests for the public helpers on :mod:`aragora.brief_engine.storage`.

Covers the two convenience functions introduced alongside the
2026-04-22 Mode 3 dogfood P0 fixes (PR #6441):

- :func:`brief_to_dict` — canonical brief → ``dict`` serializer.
- :func:`persist_ready_from_executor` — one-call state-machine driver
  that turns a completed :class:`BriefExecutionResult` into an on-disk
  ready brief.
- :func:`ready_path` — public equivalent of the module-private
  ``_ready_path`` helper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aragora.brief_engine import storage as engine_storage
from aragora.brief_engine.lifecycle import BriefLifecycleState
from aragora.pdb import storage


PR = 6421
SHA = "deadbeefcafebabe" + "0" * 24
SHA_SHORT = SHA[:12]
FILENAME = f"pr-{PR}-{SHA_SHORT}.json"


@pytest.fixture
def briefs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the briefs root under a tmp dir and return the briefs dir."""
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", str(tmp_path))
    briefs = tmp_path / "briefs"
    briefs.mkdir(parents=True, exist_ok=True)
    return briefs


# ---------------------------------------------------------------------------
# ready_path
# ---------------------------------------------------------------------------


class TestReadyPath:
    def test_matches_private_helper_for_default_namespace(self, briefs_dir: Path) -> None:
        # ``_ready_path`` is module-private and lives on the
        # :mod:`aragora.brief_engine.storage` module; the PDB shim
        # re-exports the public ``ready_path`` only. Compare the two
        # directly to lock in the "public helper matches the private
        # legacy" contract.
        private = engine_storage._ready_path(PR, SHA)
        public = storage.ready_path(PR, SHA)
        assert public == private
        assert public.name == FILENAME

    def test_pdb_shim_reexports_public_helper(self) -> None:
        assert storage.ready_path is engine_storage.ready_path


# ---------------------------------------------------------------------------
# brief_to_dict
# ---------------------------------------------------------------------------


class _ReviewBriefLike:
    """Stand-in for :class:`aragora.review.protocol.ReviewBrief`."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class TestBriefToDict:
    def test_dataclass_with_to_dict(self) -> None:
        brief = _ReviewBriefLike({"recommendation": "approve_candidate", "confidence": 0.82})
        out = storage.brief_to_dict(brief)
        assert out == {"recommendation": "approve_candidate", "confidence": 0.82}
        # Must return a plain dict (JSON-serializable).
        assert json.dumps(out)

    def test_mapping_returned_as_dict(self) -> None:
        raw = {"top_line": "hello", "nested": {"k": 1}}
        out = storage.brief_to_dict(raw)
        assert out == raw
        assert type(out) is dict

    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            storage.brief_to_dict(None)

    def test_unknown_shape_raises_type_error(self) -> None:
        class Random:
            pass

        with pytest.raises(TypeError):
            storage.brief_to_dict(Random())

    def test_to_dict_returning_non_mapping_raises(self) -> None:
        class Bad:
            def to_dict(self) -> list[str]:
                return ["nope"]

        with pytest.raises(TypeError):
            storage.brief_to_dict(Bad())


# ---------------------------------------------------------------------------
# persist_ready_from_executor
# ---------------------------------------------------------------------------


@dataclass
class _FakeExecutorResult:
    """Minimal stand-in for :class:`BriefExecutionResult`."""

    brief: Any
    actual_cost_usd: float = 0.0
    active_roster: tuple[str, ...] = ()


class TestPersistReadyFromExecutor:
    def test_happy_path_absent_to_ready(self, briefs_dir: Path) -> None:
        brief = _ReviewBriefLike(
            {
                "pr_number": PR,
                "head_sha": SHA,
                "recommendation": "approve_candidate",
                "overall_confidence": 0.82,
                "top_line": "LGTM",
            }
        )
        result = _FakeExecutorResult(brief=brief, actual_cost_usd=0.1234)

        ready_path = storage.persist_ready_from_executor(
            result,
            pr_number=PR,
            head_sha=SHA,
            source="scripts/generate_one_brief.py",
            signature="cli-local-run",
            wall_clock_ms=8421,
        )

        # Lifecycle landed at READY and the file exists at the public path.
        assert storage.get_state(PR, SHA) == BriefLifecycleState.READY
        assert ready_path.exists()
        assert ready_path == storage.ready_path(PR, SHA)

        # The on-disk payload round-trips via the canonical loader.
        loaded = storage.load_ready_brief(PR, SHA)
        assert loaded is not None
        assert loaded["recommendation"] == "approve_candidate"
        assert loaded["overall_confidence"] == 0.82
        assert loaded["signature"] == "cli-local-run"

    def test_index_event_includes_source_and_cost(self, briefs_dir: Path) -> None:
        brief = _ReviewBriefLike({"recommendation": "repair_first"})
        result = _FakeExecutorResult(brief=brief, actual_cost_usd=0.456)

        storage.persist_ready_from_executor(
            result,
            pr_number=PR,
            head_sha=SHA,
            source="tests",
            wall_clock_ms=1000,
        )

        index_lines = (briefs_dir / storage.INDEX_FILENAME).read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in index_lines if line.strip()]
        # Expect: queued, running, ready, pdb_brief_generated
        event_types = [e["event"] for e in events]
        assert event_types[-1] == "pdb_brief_generated"
        final = events[-1]
        assert final["source"] == "tests"
        assert final["cost_usd"] == pytest.approx(0.456)
        assert final["wall_clock_ms"] == 1000

    def test_infers_active_roster_for_queue_panel_models(self, briefs_dir: Path) -> None:
        brief = _ReviewBriefLike({"recommendation": "approve_candidate"})
        result = _FakeExecutorResult(
            brief=brief,
            actual_cost_usd=0.0,
            active_roster=("claude_core", "gpt_core", "grok_heterodox"),
        )

        storage.persist_ready_from_executor(
            result,
            pr_number=PR,
            head_sha=SHA,
        )

        # Queued record is consumed by mark_running; inspect the
        # ``queued`` event instead.
        events = [
            json.loads(line)
            for line in (briefs_dir / storage.INDEX_FILENAME)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        queued_event = next(e for e in events if e["event"] == "queued")
        assert queued_event["panel_models"] == [
            "claude_core",
            "gpt_core",
            "grok_heterodox",
        ]

    def test_raises_when_brief_is_none(self, briefs_dir: Path) -> None:
        result = _FakeExecutorResult(brief=None)
        with pytest.raises(ValueError, match="result.brief must not be None"):
            storage.persist_ready_from_executor(result, pr_number=PR, head_sha=SHA)

    def test_idempotent_when_already_ready(self, briefs_dir: Path) -> None:
        brief = _ReviewBriefLike({"recommendation": "approve_candidate"})
        result = _FakeExecutorResult(brief=brief)

        first = storage.persist_ready_from_executor(result, pr_number=PR, head_sha=SHA)
        # Second call should be a no-op and return the same path
        # without exploding on ``queue_generation → already queued``.
        second = storage.persist_ready_from_executor(result, pr_number=PR, head_sha=SHA)
        assert first == second
        assert storage.get_state(PR, SHA) == BriefLifecycleState.READY

    def test_signature_defaults_override_work(self, briefs_dir: Path) -> None:
        brief = _ReviewBriefLike({"recommendation": "needs_human_attention"})
        result = _FakeExecutorResult(brief=brief)

        storage.persist_ready_from_executor(
            result,
            pr_number=PR,
            head_sha=SHA,
            signature="ed25519:sig-abc",
        )
        loaded = storage.load_ready_brief(PR, SHA)
        assert loaded is not None
        assert loaded["signature"] == "ed25519:sig-abc"

    def test_file_path_matches_ready_path_helper(self, briefs_dir: Path) -> None:
        brief = _ReviewBriefLike({"recommendation": "approve_candidate"})
        result = _FakeExecutorResult(brief=brief)
        returned = storage.persist_ready_from_executor(result, pr_number=PR, head_sha=SHA)
        assert returned == storage.ready_path(PR, SHA)
        assert returned == briefs_dir / FILENAME
