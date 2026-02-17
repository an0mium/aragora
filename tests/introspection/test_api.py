"""
Comprehensive tests for aragora.introspection module.

Tests cover:
- IntrospectionSnapshot dataclass (types.py)
- get_agent_introspection and format_introspection_section (api.py)
- IntrospectionCache (cache.py)
- Module imports (__init__.py)
"""

import pytest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all public symbols are importable."""

    def test_import_snapshot(self):
        from aragora.introspection import IntrospectionSnapshot

        assert IntrospectionSnapshot is not None

    def test_import_cache(self):
        from aragora.introspection import IntrospectionCache

        assert IntrospectionCache is not None

    def test_import_get_agent_introspection(self):
        from aragora.introspection import get_agent_introspection

        assert callable(get_agent_introspection)

    def test_import_format_introspection_section(self):
        from aragora.introspection import format_introspection_section

        assert callable(format_introspection_section)

    def test_all_exports(self):
        import aragora.introspection as mod

        expected = {
            "IntrospectionSnapshot",
            "IntrospectionCache",
            "get_agent_introspection",
            "format_introspection_section",
            # Active introspection
            "ActiveIntrospectionTracker",
            "MetaReasoningEngine",
            "IntrospectionGoals",
            "RoundMetrics",
        }
        assert expected == set(mod.__all__)

    def test_import_types_module(self):
        from aragora.introspection.types import IntrospectionSnapshot

        assert IntrospectionSnapshot is not None

    def test_import_api_module(self):
        from aragora.introspection.api import get_agent_introspection, format_introspection_section

        assert callable(get_agent_introspection)
        assert callable(format_introspection_section)

    def test_import_cache_module(self):
        from aragora.introspection.cache import IntrospectionCache

        assert IntrospectionCache is not None


# ---------------------------------------------------------------------------
# IntrospectionSnapshot tests
# ---------------------------------------------------------------------------


class TestIntrospectionSnapshot:
    """Tests for the IntrospectionSnapshot dataclass."""

    def _make(self, **overrides):
        defaults = dict(agent_name="claude")
        defaults.update(overrides)
        from aragora.introspection.types import IntrospectionSnapshot

        return IntrospectionSnapshot(**defaults)

    # -- Construction ---------------------------------------------------------

    def test_default_values(self):
        snap = self._make()
        assert snap.agent_name == "claude"
        assert snap.reputation_score == 0.0
        assert snap.vote_weight == 1.0
        assert snap.proposals_made == 0
        assert snap.proposals_accepted == 0
        assert snap.critiques_given == 0
        assert snap.critiques_valuable == 0
        assert snap.calibration_score == 0.5
        assert snap.debate_count == 0
        assert snap.top_expertise == []
        assert snap.traits == []

    def test_custom_values(self):
        snap = self._make(
            reputation_score=0.85,
            vote_weight=1.4,
            proposals_made=10,
            proposals_accepted=8,
            critiques_given=5,
            critiques_valuable=4,
            calibration_score=0.72,
            debate_count=20,
            top_expertise=["python", "architecture"],
            traits=["analytical", "concise"],
        )
        assert snap.reputation_score == 0.85
        assert snap.vote_weight == 1.4
        assert snap.proposals_made == 10
        assert snap.proposals_accepted == 8
        assert snap.debate_count == 20
        assert snap.top_expertise == ["python", "architecture"]
        assert snap.traits == ["analytical", "concise"]

    def test_is_dataclass(self):
        snap = self._make()
        field_names = {f.name for f in fields(snap)}
        assert "agent_name" in field_names
        assert "reputation_score" in field_names

    # -- Properties -----------------------------------------------------------

    def test_proposal_acceptance_rate_zero_proposals(self):
        snap = self._make(proposals_made=0)
        assert snap.proposal_acceptance_rate == 0.0

    def test_proposal_acceptance_rate(self):
        snap = self._make(proposals_made=10, proposals_accepted=7)
        assert snap.proposal_acceptance_rate == pytest.approx(0.7)

    def test_proposal_acceptance_rate_all_accepted(self):
        snap = self._make(proposals_made=5, proposals_accepted=5)
        assert snap.proposal_acceptance_rate == pytest.approx(1.0)

    def test_critique_effectiveness_zero(self):
        snap = self._make(critiques_given=0)
        assert snap.critique_effectiveness == 0.0

    def test_critique_effectiveness(self):
        snap = self._make(critiques_given=10, critiques_valuable=6)
        assert snap.critique_effectiveness == pytest.approx(0.6)

    def test_calibration_label_excellent(self):
        snap = self._make(calibration_score=0.8)
        assert snap.calibration_label == "excellent"

    def test_calibration_label_excellent_boundary(self):
        snap = self._make(calibration_score=0.7)
        assert snap.calibration_label == "excellent"

    def test_calibration_label_good(self):
        snap = self._make(calibration_score=0.6)
        assert snap.calibration_label == "good"

    def test_calibration_label_good_boundary(self):
        snap = self._make(calibration_score=0.5)
        assert snap.calibration_label == "good"

    def test_calibration_label_fair(self):
        snap = self._make(calibration_score=0.4)
        assert snap.calibration_label == "fair"

    def test_calibration_label_fair_boundary(self):
        snap = self._make(calibration_score=0.3)
        assert snap.calibration_label == "fair"

    def test_calibration_label_developing(self):
        snap = self._make(calibration_score=0.2)
        assert snap.calibration_label == "developing"

    def test_calibration_label_zero(self):
        snap = self._make(calibration_score=0.0)
        assert snap.calibration_label == "developing"

    # -- to_prompt_section ----------------------------------------------------

    def test_prompt_section_header(self):
        snap = self._make()
        section = snap.to_prompt_section()
        assert section.startswith("## YOUR TRACK RECORD")

    def test_prompt_section_includes_reputation(self):
        snap = self._make(reputation_score=0.75, vote_weight=1.3)
        section = snap.to_prompt_section()
        assert "Reputation: 75%" in section
        assert "Vote weight: 1.3x" in section

    def test_prompt_section_includes_proposals(self):
        snap = self._make(proposals_made=10, proposals_accepted=7)
        section = snap.to_prompt_section()
        assert "Proposals: 7/10 accepted (70%)" in section

    def test_prompt_section_no_proposals_when_zero(self):
        snap = self._make(proposals_made=0)
        section = snap.to_prompt_section()
        assert "Proposals:" not in section

    def test_prompt_section_includes_critiques(self):
        snap = self._make(critiques_given=8, critiques_valuable=6, calibration_score=0.8)
        section = snap.to_prompt_section()
        assert "Critiques: 75% valuable" in section
        assert "Calibration: excellent" in section

    def test_prompt_section_no_critiques_when_zero(self):
        snap = self._make(critiques_given=0)
        section = snap.to_prompt_section()
        assert "Critiques:" not in section

    def test_prompt_section_includes_expertise(self):
        snap = self._make(top_expertise=["python", "ml", "architecture"])
        section = snap.to_prompt_section()
        assert "Expertise: python, ml, architecture" in section

    def test_prompt_section_expertise_limited_to_three(self):
        snap = self._make(top_expertise=["a", "b", "c", "d", "e"])
        section = snap.to_prompt_section()
        assert "Expertise: a, b, c" in section
        assert "d" not in section

    def test_prompt_section_includes_traits(self):
        snap = self._make(traits=["analytical", "concise"])
        section = snap.to_prompt_section()
        assert "Style: analytical, concise" in section

    def test_prompt_section_traits_limited_to_three(self):
        snap = self._make(traits=["a", "b", "c", "d"])
        section = snap.to_prompt_section()
        assert "Style: a, b, c" in section
        assert "d" not in section

    def test_prompt_section_truncation(self):
        """Section should respect max_chars limit."""
        snap = self._make(
            reputation_score=0.9,
            vote_weight=1.5,
            proposals_made=100,
            proposals_accepted=80,
            critiques_given=50,
            critiques_valuable=40,
            calibration_score=0.8,
            top_expertise=["python", "architecture", "databases"],
            traits=["analytical", "concise", "thorough"],
        )
        section = snap.to_prompt_section(max_chars=100)
        assert len(section) <= 100

    def test_prompt_section_always_keeps_header_and_reputation(self):
        """Even with aggressive truncation, header + reputation remain."""
        snap = self._make(
            reputation_score=0.5,
            vote_weight=1.0,
            proposals_made=10,
            proposals_accepted=5,
            critiques_given=5,
            critiques_valuable=3,
            top_expertise=["x"],
            traits=["y"],
        )
        section = snap.to_prompt_section(max_chars=120)
        assert "## YOUR TRACK RECORD" in section
        assert "Reputation:" in section

    def test_prompt_section_default_max_chars(self):
        snap = self._make()
        section = snap.to_prompt_section()
        assert len(section) <= 600

    # -- to_dict --------------------------------------------------------------

    def test_to_dict_keys(self):
        snap = self._make()
        d = snap.to_dict()
        expected_keys = {
            "agent_name",
            "reputation_score",
            "vote_weight",
            "proposals_made",
            "proposals_accepted",
            "proposal_acceptance_rate",
            "critiques_given",
            "critiques_valuable",
            "critique_effectiveness",
            "calibration_score",
            "calibration_label",
            "debate_count",
            "top_expertise",
            "traits",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        snap = self._make(
            agent_name="gpt",
            reputation_score=0.9,
            proposals_made=10,
            proposals_accepted=8,
        )
        d = snap.to_dict()
        assert d["agent_name"] == "gpt"
        assert d["reputation_score"] == 0.9
        assert d["proposal_acceptance_rate"] == pytest.approx(0.8)

    def test_to_dict_includes_computed_properties(self):
        snap = self._make(
            proposals_made=4,
            proposals_accepted=2,
            critiques_given=10,
            critiques_valuable=7,
            calibration_score=0.75,
        )
        d = snap.to_dict()
        assert d["proposal_acceptance_rate"] == pytest.approx(0.5)
        assert d["critique_effectiveness"] == pytest.approx(0.7)
        assert d["calibration_label"] == "excellent"


# ---------------------------------------------------------------------------
# get_agent_introspection tests
# ---------------------------------------------------------------------------


class TestGetAgentIntrospection:
    """Tests for the get_agent_introspection function."""

    def test_minimal_call(self):
        from aragora.introspection.api import get_agent_introspection

        snap = get_agent_introspection("claude")
        assert snap.agent_name == "claude"
        assert snap.reputation_score == 0.0

    def test_with_memory_reputation(self):
        from aragora.introspection.api import get_agent_introspection

        rep = SimpleNamespace(
            score=0.85,
            vote_weight=1.4,
            proposals_made=20,
            proposals_accepted=15,
            critiques_given=10,
            critiques_valuable=8,
            calibration_score=0.72,
        )
        memory = MagicMock()
        memory.get_reputation.return_value = rep

        snap = get_agent_introspection("claude", memory=memory)
        memory.get_reputation.assert_called_once_with("claude")
        assert snap.reputation_score == 0.85
        assert snap.vote_weight == 1.4
        assert snap.proposals_made == 20
        assert snap.proposals_accepted == 15
        assert snap.critiques_given == 10
        assert snap.critiques_valuable == 8
        assert snap.calibration_score == 0.72

    def test_memory_returns_none(self):
        from aragora.introspection.api import get_agent_introspection

        memory = MagicMock()
        memory.get_reputation.return_value = None

        snap = get_agent_introspection("claude", memory=memory)
        # Should have defaults when rep is None
        assert snap.reputation_score == 0.0
        assert snap.vote_weight == 1.0

    def test_memory_raises_exception(self):
        from aragora.introspection.api import get_agent_introspection

        memory = MagicMock()
        memory.get_reputation.side_effect = RuntimeError("db error")

        snap = get_agent_introspection("claude", memory=memory)
        # Graceful degradation - returns defaults
        assert snap.reputation_score == 0.0

    def test_with_persona_top_expertise(self):
        from aragora.introspection.api import get_agent_introspection

        persona = SimpleNamespace(
            top_expertise=[("python", 0.9), ("ml", 0.8), ("arch", 0.7), ("extra", 0.5)],
            traits=["analytical", "concise", "verbose", "extra"],
        )
        persona_manager = MagicMock()
        persona_manager.get_persona.return_value = persona

        snap = get_agent_introspection("claude", persona_manager=persona_manager)
        persona_manager.get_persona.assert_called_once_with("claude")
        assert snap.top_expertise == ["python", "ml", "arch"]
        assert snap.traits == ["analytical", "concise", "verbose"]

    def test_with_persona_expertise_dict_fallback(self):
        from aragora.introspection.api import get_agent_introspection

        persona = SimpleNamespace(
            expertise={"python": 0.9, "ml": 0.8, "arch": 0.7, "db": 0.6},
            traits=["careful"],
        )
        persona_manager = MagicMock()
        persona_manager.get_persona.return_value = persona

        snap = get_agent_introspection("claude", persona_manager=persona_manager)
        # Should sort by score descending, take top 3
        assert len(snap.top_expertise) == 3
        assert snap.top_expertise[0] == "python"
        assert snap.top_expertise[1] == "ml"
        assert snap.top_expertise[2] == "arch"
        assert snap.traits == ["careful"]

    def test_persona_returns_none(self):
        from aragora.introspection.api import get_agent_introspection

        persona_manager = MagicMock()
        persona_manager.get_persona.return_value = None

        snap = get_agent_introspection("claude", persona_manager=persona_manager)
        assert snap.top_expertise == []
        assert snap.traits == []

    def test_persona_raises_exception(self):
        from aragora.introspection.api import get_agent_introspection

        persona_manager = MagicMock()
        persona_manager.get_persona.side_effect = ValueError("oops")

        snap = get_agent_introspection("claude", persona_manager=persona_manager)
        assert snap.top_expertise == []
        assert snap.traits == []

    def test_persona_no_traits(self):
        from aragora.introspection.api import get_agent_introspection

        persona = SimpleNamespace(
            top_expertise=[("python", 0.9)],
            traits=[],
        )
        persona_manager = MagicMock()
        persona_manager.get_persona.return_value = persona

        snap = get_agent_introspection("claude", persona_manager=persona_manager)
        assert snap.top_expertise == ["python"]
        assert snap.traits == []

    def test_persona_no_expertise_attrs(self):
        """Persona has neither top_expertise nor expertise attribute."""
        from aragora.introspection.api import get_agent_introspection

        persona = SimpleNamespace()  # no expertise/traits attrs
        persona_manager = MagicMock()
        persona_manager.get_persona.return_value = persona

        snap = get_agent_introspection("claude", persona_manager=persona_manager)
        assert snap.top_expertise == []

    def test_both_sources(self):
        from aragora.introspection.api import get_agent_introspection

        rep = SimpleNamespace(
            score=0.9,
            vote_weight=1.5,
            proposals_made=10,
            proposals_accepted=9,
            critiques_given=5,
            critiques_valuable=4,
            calibration_score=0.8,
        )
        memory = MagicMock()
        memory.get_reputation.return_value = rep

        persona = SimpleNamespace(
            top_expertise=[("security", 0.95)],
            traits=["thorough"],
        )
        persona_manager = MagicMock()
        persona_manager.get_persona.return_value = persona

        snap = get_agent_introspection("claude", memory=memory, persona_manager=persona_manager)
        assert snap.reputation_score == 0.9
        assert snap.top_expertise == ["security"]
        assert snap.traits == ["thorough"]


# ---------------------------------------------------------------------------
# format_introspection_section tests
# ---------------------------------------------------------------------------


class TestFormatIntrospectionSection:
    """Tests for the format_introspection_section convenience function."""

    def test_delegates_to_snapshot(self):
        from aragora.introspection.api import format_introspection_section
        from aragora.introspection.types import IntrospectionSnapshot

        snap = IntrospectionSnapshot(agent_name="gpt", reputation_score=0.6, vote_weight=1.2)
        result = format_introspection_section(snap)
        assert "## YOUR TRACK RECORD" in result
        assert "Reputation: 60%" in result

    def test_custom_max_chars(self):
        from aragora.introspection.api import format_introspection_section
        from aragora.introspection.types import IntrospectionSnapshot

        snap = IntrospectionSnapshot(
            agent_name="gpt",
            reputation_score=0.5,
            vote_weight=1.0,
            proposals_made=10,
            proposals_accepted=5,
            critiques_given=10,
            critiques_valuable=5,
            top_expertise=["a", "b", "c"],
            traits=["x", "y", "z"],
        )
        result = format_introspection_section(snap, max_chars=100)
        assert len(result) <= 100

    def test_matches_direct_call(self):
        from aragora.introspection.api import format_introspection_section
        from aragora.introspection.types import IntrospectionSnapshot

        snap = IntrospectionSnapshot(agent_name="test", reputation_score=0.5)
        assert format_introspection_section(snap) == snap.to_prompt_section()
        assert format_introspection_section(snap, max_chars=200) == snap.to_prompt_section(
            max_chars=200
        )


# ---------------------------------------------------------------------------
# IntrospectionCache tests
# ---------------------------------------------------------------------------


class TestIntrospectionCache:
    """Tests for the IntrospectionCache class."""

    def _make_agent(self, name):
        return SimpleNamespace(name=name)

    def test_empty_cache(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        assert not cache.is_warm
        assert cache.agent_count == 0
        assert cache.get("any") is None
        assert cache.get_all() == {}

    def test_warm_populates_cache(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()

        agents = [self._make_agent("claude"), self._make_agent("gpt")]
        cache.warm(agents=agents)

        assert cache.is_warm
        assert cache.agent_count == 2
        assert cache.get("claude") is not None
        assert cache.get("gpt") is not None
        assert cache.get("claude").agent_name == "claude"
        assert cache.get("gpt").agent_name == "gpt"

    def test_warm_with_memory(self):
        from aragora.introspection.cache import IntrospectionCache

        rep = SimpleNamespace(
            score=0.8,
            vote_weight=1.3,
            proposals_made=5,
            proposals_accepted=4,
            critiques_given=3,
            critiques_valuable=2,
            calibration_score=0.7,
        )
        memory = MagicMock()
        memory.get_reputation.return_value = rep

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")], memory=memory)

        snap = cache.get("claude")
        assert snap.reputation_score == 0.8

    def test_warm_with_persona_manager(self):
        from aragora.introspection.cache import IntrospectionCache

        persona = SimpleNamespace(
            top_expertise=[("rust", 0.9)],
            traits=["precise"],
        )
        pm = MagicMock()
        pm.get_persona.return_value = persona

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")], persona_manager=pm)

        snap = cache.get("claude")
        assert snap.top_expertise == ["rust"]
        assert snap.traits == ["precise"]

    def test_warm_clears_previous(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()

        cache.warm(agents=[self._make_agent("a"), self._make_agent("b")])
        assert cache.agent_count == 2

        cache.warm(agents=[self._make_agent("c")])
        assert cache.agent_count == 1
        assert cache.get("a") is None
        assert cache.get("c") is not None

    def test_get_nonexistent(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")])
        assert cache.get("nonexistent") is None

    def test_invalidate(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")])
        assert cache.is_warm

        cache.invalidate()
        assert not cache.is_warm
        assert cache.agent_count == 0
        assert cache.get("claude") is None

    def test_is_warm_false_after_invalidation(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("x")])
        cache.invalidate()
        assert not cache.is_warm

    def test_loaded_at_set_on_warm(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        assert cache._loaded_at is None
        cache.warm(agents=[self._make_agent("a")])
        assert cache._loaded_at is not None

    def test_loaded_at_cleared_on_invalidate(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("a")])
        cache.invalidate()
        assert cache._loaded_at is None

    def test_get_all_returns_copy(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")])

        all_data = cache.get_all()
        assert "claude" in all_data
        # Mutating the copy should not affect the cache
        all_data.pop("claude")
        assert cache.get("claude") is not None

    def test_agent_without_name_attr(self):
        """Agents without .name should fall back to str()."""
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()

        agent = "raw_string_agent"
        cache.warm(agents=[agent])
        assert cache.agent_count == 1
        assert cache.get("raw_string_agent") is not None

    def test_warm_multiple_agents_all_cached(self):
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        names = ["a", "b", "c", "d", "e"]
        agents = [self._make_agent(n) for n in names]
        cache.warm(agents=agents)
        assert cache.agent_count == 5
        for n in names:
            snap = cache.get(n)
            assert snap is not None
            assert snap.agent_name == n
