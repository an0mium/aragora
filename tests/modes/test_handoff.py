"""
Tests for mode handoff system.

Tests cover:
- HandoffContext dataclass
- ModeHandoff class
"""

from datetime import datetime

import pytest

from aragora.modes.handoff import HandoffContext, ModeHandoff


class TestHandoffContext:
    """Tests for HandoffContext dataclass."""

    def test_create_context(self):
        """Basic context creation."""
        ctx = HandoffContext(
            from_mode="architect",
            to_mode="coder",
            task_summary="Design a caching system",
        )

        assert ctx.from_mode == "architect"
        assert ctx.to_mode == "coder"
        assert ctx.task_summary == "Design a caching system"

    def test_default_values(self):
        """Default values are empty collections."""
        ctx = HandoffContext(
            from_mode="a",
            to_mode="b",
            task_summary="test",
        )

        assert ctx.key_findings == []
        assert ctx.files_touched == []
        assert ctx.open_questions == []
        assert ctx.artifacts == {}
        assert ctx.timestamp is not None

    def test_with_all_fields(self):
        """Context with all fields populated."""
        ctx = HandoffContext(
            from_mode="reviewer",
            to_mode="debugger",
            task_summary="Found bugs in auth module",
            key_findings=["SQL injection in login", "Missing rate limit"],
            files_touched=["auth.py", "login.py"],
            open_questions=["What is the expected rate limit?"],
            artifacts={"bug_count": 3},
        )

        assert len(ctx.key_findings) == 2
        assert len(ctx.files_touched) == 2
        assert len(ctx.open_questions) == 1
        assert ctx.artifacts["bug_count"] == 3

    def test_to_prompt_basic(self):
        """to_prompt generates proper markdown."""
        ctx = HandoffContext(
            from_mode="architect",
            to_mode="coder",
            task_summary="Design complete for API layer",
        )

        prompt = ctx.to_prompt()

        assert "## Handoff from architect to coder" in prompt
        assert "### Task Summary" in prompt
        assert "Design complete for API layer" in prompt

    def test_to_prompt_with_findings(self):
        """to_prompt includes key findings."""
        ctx = HandoffContext(
            from_mode="reviewer",
            to_mode="coder",
            task_summary="Review complete",
            key_findings=["Bug in auth", "Memory leak"],
        )

        prompt = ctx.to_prompt()

        assert "### Key Findings" in prompt
        assert "- Bug in auth" in prompt
        assert "- Memory leak" in prompt

    def test_to_prompt_with_files(self):
        """to_prompt includes files touched."""
        ctx = HandoffContext(
            from_mode="coder",
            to_mode="reviewer",
            task_summary="Implementation done",
            files_touched=["src/api.py", "tests/test_api.py"],
        )

        prompt = ctx.to_prompt()

        assert "### Files Touched" in prompt
        assert "`src/api.py`" in prompt
        assert "`tests/test_api.py`" in prompt

    def test_to_prompt_with_questions(self):
        """to_prompt includes open questions."""
        ctx = HandoffContext(
            from_mode="architect",
            to_mode="coder",
            task_summary="Design needs clarification",
            open_questions=["Redis or Memcached?", "How many replicas?"],
        )

        prompt = ctx.to_prompt()

        assert "### Open Questions" in prompt
        assert "- Redis or Memcached?" in prompt
        assert "- How many replicas?" in prompt

    def test_to_prompt_empty_sections_omitted(self):
        """Empty sections should not appear in prompt."""
        ctx = HandoffContext(
            from_mode="a",
            to_mode="b",
            task_summary="Simple task",
        )

        prompt = ctx.to_prompt()

        assert "### Key Findings" not in prompt
        assert "### Files Touched" not in prompt
        assert "### Open Questions" not in prompt


class TestModeHandoff:
    """Tests for ModeHandoff class."""

    def test_init(self):
        """Initializes with empty history."""
        handoff = ModeHandoff()
        assert handoff.history == []

    def test_create_context(self):
        """Creates and stores context."""
        handoff = ModeHandoff()

        ctx = handoff.create_context(
            from_mode="architect",
            to_mode="coder",
            task_summary="Design complete",
        )

        assert ctx.from_mode == "architect"
        assert ctx.to_mode == "coder"
        assert len(handoff.history) == 1

    def test_create_context_with_optional_fields(self):
        """Creates context with all optional fields."""
        handoff = ModeHandoff()

        ctx = handoff.create_context(
            from_mode="reviewer",
            to_mode="debugger",
            task_summary="Found issues",
            key_findings=["Bug 1", "Bug 2"],
            files_touched=["file.py"],
            open_questions=["Why?"],
            artifacts={"data": 123},
        )

        assert ctx.key_findings == ["Bug 1", "Bug 2"]
        assert ctx.files_touched == ["file.py"]
        assert ctx.open_questions == ["Why?"]
        assert ctx.artifacts == {"data": 123}

    def test_multiple_contexts(self):
        """Can create multiple contexts."""
        handoff = ModeHandoff()

        handoff.create_context("a", "b", "First")
        handoff.create_context("b", "c", "Second")
        handoff.create_context("c", "d", "Third")

        assert len(handoff.history) == 3

    def test_generate_transition_prompt(self):
        """Generates transition prompt combining mode prompt and context."""
        handoff = ModeHandoff()

        ctx = handoff.create_context(
            from_mode="architect",
            to_mode="coder",
            task_summary="API design complete",
            key_findings=["Use REST", "JSON responses"],
        )

        target_prompt = "You are a coder. Write clean, tested code."

        transition = handoff.generate_transition_prompt(ctx, target_prompt)

        assert "You are a coder" in transition
        assert "Handoff from architect to coder" in transition
        assert "API design complete" in transition
        assert "Use REST" in transition
        assert "Continue from where the previous mode left off" in transition

    def test_get_history_returns_copy(self):
        """get_history returns a copy, not the original."""
        handoff = ModeHandoff()
        handoff.create_context("a", "b", "Test")

        history = handoff.get_history()
        history.append(None)  # Modify the copy

        assert len(handoff.history) == 1  # Original unchanged

    def test_summarize_session_empty(self):
        """Summarize empty session."""
        handoff = ModeHandoff()

        summary = handoff.summarize_session()

        assert "No mode transitions recorded" in summary

    def test_summarize_session_with_history(self):
        """Summarize session with transitions."""
        handoff = ModeHandoff()
        handoff.create_context("architect", "coder", "Design the API layer")
        handoff.create_context("coder", "reviewer", "Implementation complete")

        summary = handoff.summarize_session()

        assert "## Session Mode History" in summary
        assert "architect -> coder" in summary
        assert "coder -> reviewer" in summary
        assert "Design the API" in summary

    def test_summarize_session_includes_file_count(self):
        """Summarize includes file count when files touched."""
        handoff = ModeHandoff()
        handoff.create_context(
            "coder",
            "reviewer",
            "Wrote code",
            files_touched=["a.py", "b.py", "c.py"],
        )

        summary = handoff.summarize_session()

        assert "Files: 3" in summary


class TestHandoffIntegration:
    """Integration tests for handoff workflows."""

    def test_full_workflow(self):
        """Test a full mode transition workflow."""
        handoff = ModeHandoff()

        # Architect designs
        ctx1 = handoff.create_context(
            from_mode="architect",
            to_mode="coder",
            task_summary="Designed a microservices architecture",
            key_findings=["Use event sourcing", "Deploy to K8s"],
            open_questions=["Which message broker?"],
        )

        # Generate transition for coder
        coder_prompt = "You are a coder. Write clean code."
        transition1 = handoff.generate_transition_prompt(ctx1, coder_prompt)

        assert "microservices" in transition1
        assert "event sourcing" in transition1

        # Coder implements
        ctx2 = handoff.create_context(
            from_mode="coder",
            to_mode="reviewer",
            task_summary="Implemented user service with Kafka",
            key_findings=["Used Kafka for messaging", "Added retry logic"],
            files_touched=["user_service.py", "kafka_client.py"],
            artifacts={"lines_added": 500},
        )

        # Generate transition for reviewer
        reviewer_prompt = "You are a reviewer. Check code quality."
        transition2 = handoff.generate_transition_prompt(ctx2, reviewer_prompt)

        assert "user service" in transition2.lower()
        assert "Kafka" in transition2

        # Check session summary
        summary = handoff.summarize_session()
        assert "architect -> coder" in summary
        assert "coder -> reviewer" in summary
        assert len(handoff.history) == 2
