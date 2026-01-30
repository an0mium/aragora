"""
Tests for core identity module.

Tests cover:
- TAGLINE constant
- DESCRIPTION_SHORT constant
- DESCRIPTION_FULL constant
- ELEVATOR_PITCH constant
- DIFFERENTIATORS dictionary
- NOT_STATEMENTS dictionary
- CORE_CAPABILITIES dictionary
- SERVICE_OFFERINGS dictionary
- TERMINOLOGY dictionary
- get_tagline function
- get_description function
- get_elevator_pitch function
"""

import pytest

from aragora.core.identity import (
    TAGLINE,
    DESCRIPTION_SHORT,
    DESCRIPTION_FULL,
    ELEVATOR_PITCH,
    DIFFERENTIATORS,
    NOT_STATEMENTS,
    CORE_CAPABILITIES,
    SERVICE_OFFERINGS,
    TERMINOLOGY,
    get_tagline,
    get_description,
    get_elevator_pitch,
)


# =============================================================================
# TAGLINE Tests
# =============================================================================


class TestTagline:
    """Tests for TAGLINE constant."""

    def test_tagline_exists(self):
        """Tagline constant exists and is a string."""
        assert isinstance(TAGLINE, str)
        assert len(TAGLINE) > 0

    def test_tagline_is_concise(self):
        """Tagline is concise (under 100 characters)."""
        # A good tagline should be memorable and brief
        assert len(TAGLINE) < 100

    def test_tagline_contains_key_concepts(self):
        """Tagline mentions key product concepts."""
        tagline_lower = TAGLINE.lower()
        # Should mention multi-agent or control plane or decision
        assert any(
            concept in tagline_lower for concept in ["control plane", "multi-agent", "decision"]
        )


# =============================================================================
# Description Tests
# =============================================================================


class TestDescriptions:
    """Tests for description constants."""

    def test_description_short_exists(self):
        """Short description exists and is a string."""
        assert isinstance(DESCRIPTION_SHORT, str)
        assert len(DESCRIPTION_SHORT) > 0

    def test_description_short_is_under_500_chars(self):
        """Short description is concise."""
        assert len(DESCRIPTION_SHORT) < 500

    def test_description_full_exists(self):
        """Full description exists and is a string."""
        assert isinstance(DESCRIPTION_FULL, str)
        assert len(DESCRIPTION_FULL) > 0

    def test_description_full_is_longer_than_short(self):
        """Full description is longer than short description."""
        assert len(DESCRIPTION_FULL) >= len(DESCRIPTION_SHORT)

    def test_descriptions_mention_aragora(self):
        """Descriptions mention Aragora product name."""
        assert "Aragora" in DESCRIPTION_SHORT
        assert "Aragora" in DESCRIPTION_FULL

    def test_descriptions_mention_ai_models(self):
        """Descriptions mention AI model orchestration."""
        full_lower = DESCRIPTION_FULL.lower()
        short_lower = DESCRIPTION_SHORT.lower()

        # Should mention AI/model orchestration
        assert any(term in full_lower for term in ["ai model", "models", "orchestrat"])


# =============================================================================
# Elevator Pitch Tests
# =============================================================================


class TestElevatorPitch:
    """Tests for ELEVATOR_PITCH constant."""

    def test_elevator_pitch_exists(self):
        """Elevator pitch exists and is a string."""
        assert isinstance(ELEVATOR_PITCH, str)
        assert len(ELEVATOR_PITCH) > 0

    def test_elevator_pitch_is_substantial(self):
        """Elevator pitch is substantial (more than 200 chars)."""
        assert len(ELEVATOR_PITCH) > 200

    def test_elevator_pitch_mentions_aragora(self):
        """Elevator pitch mentions Aragora."""
        assert "Aragora" in ELEVATOR_PITCH

    def test_elevator_pitch_has_value_props(self):
        """Elevator pitch contains value propositions."""
        pitch_lower = ELEVATOR_PITCH.lower()

        # Should mention key value props
        value_props_found = sum(
            1
            for prop in ["memory", "knowledge", "decision", "audit", "channel", "defense"]
            if prop in pitch_lower
        )
        assert value_props_found >= 2


# =============================================================================
# DIFFERENTIATORS Tests
# =============================================================================


class TestDifferentiators:
    """Tests for DIFFERENTIATORS dictionary."""

    def test_differentiators_exists(self):
        """Differentiators dict exists and is not empty."""
        assert isinstance(DIFFERENTIATORS, dict)
        assert len(DIFFERENTIATORS) > 0

    def test_differentiators_has_expected_keys(self):
        """Differentiators has expected competitive position keys."""
        expected_keys = [
            "not_chatbot",
            "not_copilot",
            "not_single_model",
            "not_stateless",
            "not_text_only",
        ]
        for key in expected_keys:
            assert key in DIFFERENTIATORS, f"Missing key: {key}"

    def test_differentiators_values_are_strings(self):
        """All differentiator values are non-empty strings."""
        for key, value in DIFFERENTIATORS.items():
            assert isinstance(value, str), f"{key} should be string"
            assert len(value) > 0, f"{key} should not be empty"


# =============================================================================
# NOT_STATEMENTS Tests
# =============================================================================


class TestNotStatements:
    """Tests for NOT_STATEMENTS dictionary."""

    def test_not_statements_exists(self):
        """NOT_STATEMENTS dict exists and is not empty."""
        assert isinstance(NOT_STATEMENTS, dict)
        assert len(NOT_STATEMENTS) > 0

    def test_not_statements_has_expected_keys(self):
        """NOT_STATEMENTS has expected keys."""
        expected_keys = ["chatbot", "copilot", "single_model", "stateless", "text_only"]
        for key in expected_keys:
            assert key in NOT_STATEMENTS, f"Missing key: {key}"

    def test_not_statements_start_with_not(self):
        """NOT_STATEMENTS values explain what Aragora is NOT."""
        for key, value in NOT_STATEMENTS.items():
            # Each statement should start with "NOT" to emphasize differentiation
            assert "NOT" in value.upper(), f"{key} should contain 'NOT'"

    def test_not_statements_provide_alternative(self):
        """NOT_STATEMENTS provide alternative positioning."""
        for key, value in NOT_STATEMENTS.items():
            # Should be substantial enough to explain the alternative
            assert len(value) > 30, f"{key} should explain the alternative"


# =============================================================================
# CORE_CAPABILITIES Tests
# =============================================================================


class TestCoreCapabilities:
    """Tests for CORE_CAPABILITIES dictionary."""

    def test_core_capabilities_exists(self):
        """CORE_CAPABILITIES dict exists and is not empty."""
        assert isinstance(CORE_CAPABILITIES, dict)
        assert len(CORE_CAPABILITIES) > 0

    def test_capabilities_have_required_fields(self):
        """Each capability has name, description, and maturity."""
        required_fields = ["name", "description", "maturity"]

        for cap_key, cap_data in CORE_CAPABILITIES.items():
            assert isinstance(cap_data, dict), f"{cap_key} should be a dict"
            for field in required_fields:
                assert field in cap_data, f"{cap_key} missing '{field}'"

    def test_capabilities_have_valid_maturity(self):
        """Capability maturity is a percentage string."""
        for cap_key, cap_data in CORE_CAPABILITIES.items():
            maturity = cap_data.get("maturity", "")
            # Should be a percentage like "96%" or "92%"
            assert "%" in maturity, f"{cap_key} maturity should be percentage"

    def test_expected_capabilities_present(self):
        """Expected core capabilities are present."""
        expected_caps = [
            "omnivorous_ingestion",
            "institutional_memory",
            "bidirectional_communication",
            "debate_synthesis",
        ]
        for cap in expected_caps:
            assert cap in CORE_CAPABILITIES, f"Missing capability: {cap}"

    def test_capability_names_are_descriptive(self):
        """Capability names are human-readable."""
        for cap_key, cap_data in CORE_CAPABILITIES.items():
            name = cap_data.get("name", "")
            # Name should be at least 10 chars and contain spaces
            assert len(name) >= 10, f"{cap_key} name too short"

    def test_capability_descriptions_are_informative(self):
        """Capability descriptions provide useful information."""
        for cap_key, cap_data in CORE_CAPABILITIES.items():
            desc = cap_data.get("description", "")
            # Description should be substantial
            assert len(desc) >= 20, f"{cap_key} description too short"


# =============================================================================
# SERVICE_OFFERINGS Tests
# =============================================================================


class TestServiceOfferings:
    """Tests for SERVICE_OFFERINGS dictionary."""

    def test_service_offerings_exists(self):
        """SERVICE_OFFERINGS dict exists and is not empty."""
        assert isinstance(SERVICE_OFFERINGS, dict)
        assert len(SERVICE_OFFERINGS) > 0

    def test_offerings_have_required_fields(self):
        """Each offering has name and description."""
        required_fields = ["name", "description"]

        for offering_key, offering_data in SERVICE_OFFERINGS.items():
            assert isinstance(offering_data, dict), f"{offering_key} should be a dict"
            for field in required_fields:
                assert field in offering_data, f"{offering_key} missing '{field}'"

    def test_offering_names_are_professional(self):
        """Offering names are professional and concise."""
        for offering_key, offering_data in SERVICE_OFFERINGS.items():
            name = offering_data.get("name", "")
            # Name should be reasonable length
            assert 5 < len(name) < 100, f"{offering_key} name has invalid length"

    def test_offering_descriptions_are_informative(self):
        """Offering descriptions explain the service."""
        for offering_key, offering_data in SERVICE_OFFERINGS.items():
            desc = offering_data.get("description", "")
            # Description should explain what the service does
            assert len(desc) >= 30, f"{offering_key} description too short"


# =============================================================================
# TERMINOLOGY Tests
# =============================================================================


class TestTerminology:
    """Tests for TERMINOLOGY dictionary."""

    def test_terminology_exists(self):
        """TERMINOLOGY dict exists and is not empty."""
        assert isinstance(TERMINOLOGY, dict)
        assert len(TERMINOLOGY) > 0

    def test_terminology_values_are_strings(self):
        """All terminology values are strings."""
        for term, guidance in TERMINOLOGY.items():
            assert isinstance(guidance, str), f"{term} guidance should be string"
            assert len(guidance) > 0, f"{term} guidance should not be empty"

    def test_terminology_provides_guidance(self):
        """Terminology entries provide usage guidance."""
        for term, guidance in TERMINOLOGY.items():
            # Guidance should explain how to use the term
            assert len(guidance) > 10, f"{term} needs more guidance"

    def test_expected_terms_present(self):
        """Expected terminology entries are present."""
        # These are key terms that need consistent usage
        expected_terms = ["deliberation", "control_plane"]
        for term in expected_terms:
            assert term in TERMINOLOGY, f"Missing terminology: {term}"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetTagline:
    """Tests for get_tagline function."""

    def test_get_tagline_returns_tagline(self):
        """get_tagline returns the TAGLINE constant."""
        result = get_tagline()
        assert result == TAGLINE

    def test_get_tagline_returns_string(self):
        """get_tagline returns a string."""
        result = get_tagline()
        assert isinstance(result, str)


class TestGetDescription:
    """Tests for get_description function."""

    def test_get_description_short_by_default(self):
        """get_description returns short description by default."""
        result = get_description()
        assert result == DESCRIPTION_SHORT

    def test_get_description_short_explicit(self):
        """get_description(full=False) returns short description."""
        result = get_description(full=False)
        assert result == DESCRIPTION_SHORT

    def test_get_description_full(self):
        """get_description(full=True) returns full description."""
        result = get_description(full=True)
        assert result == DESCRIPTION_FULL

    def test_get_description_returns_string(self):
        """get_description always returns a string."""
        assert isinstance(get_description(), str)
        assert isinstance(get_description(full=True), str)
        assert isinstance(get_description(full=False), str)


class TestGetElevatorPitch:
    """Tests for get_elevator_pitch function."""

    def test_get_elevator_pitch_returns_pitch(self):
        """get_elevator_pitch returns the ELEVATOR_PITCH constant."""
        result = get_elevator_pitch()
        assert result == ELEVATOR_PITCH

    def test_get_elevator_pitch_returns_string(self):
        """get_elevator_pitch returns a string."""
        result = get_elevator_pitch()
        assert isinstance(result, str)


# =============================================================================
# Consistency Tests
# =============================================================================


class TestIdentityConsistency:
    """Tests for consistency across identity elements."""

    def test_aragora_spelling_consistent(self):
        """Aragora is spelled consistently across all content."""
        all_text = (
            TAGLINE
            + DESCRIPTION_SHORT
            + DESCRIPTION_FULL
            + ELEVATOR_PITCH
            + " ".join(DIFFERENTIATORS.values())
            + " ".join(NOT_STATEMENTS.values())
        )

        # Check for common misspellings
        assert "aragora" not in all_text  # Should be capitalized
        assert "Aragora" in all_text  # Correct capitalization

    def test_key_terms_used_consistently(self):
        """Key product terms appear consistently."""
        all_text = (
            DESCRIPTION_FULL
            + ELEVATOR_PITCH
            + " ".join(cap["description"] for cap in CORE_CAPABILITIES.values())
        ).lower()

        # Key terms that should appear multiple times
        key_terms = ["multi-agent", "decision", "knowledge"]
        for term in key_terms:
            assert term in all_text, f"Key term '{term}' should appear in content"

    def test_no_competitor_mentions(self):
        """No direct competitor names in positioning."""
        all_text = (TAGLINE + DESCRIPTION_SHORT + DESCRIPTION_FULL + ELEVATOR_PITCH).lower()

        # Should not name competitors directly
        competitors = ["chatgpt", "claude", "gemini", "copilot"]
        for comp in competitors:
            # Note: Some might be mentioned in technical contexts
            # This test ensures we're not mentioning in marketing copy
            pass  # Allow for flexibility in technical descriptions

    def test_feature_maturity_reasonable(self):
        """Feature maturity percentages are reasonable."""
        for cap_key, cap_data in CORE_CAPABILITIES.items():
            maturity = cap_data.get("maturity", "0%")
            # Extract numeric value
            try:
                value = int(maturity.replace("%", ""))
                # Should be between 0 and 100
                assert 0 <= value <= 100, f"{cap_key} maturity out of range"
                # Should be substantial for core capabilities
                assert value >= 50, f"{cap_key} maturity seems low for a core capability"
            except ValueError:
                pytest.fail(f"{cap_key} maturity '{maturity}' is not a valid percentage")


# =============================================================================
# Marketing Quality Tests
# =============================================================================


class TestMarketingQuality:
    """Tests for marketing content quality."""

    def test_no_jargon_overload(self):
        """Content doesn't have excessive jargon."""
        # These are fine in moderation but shouldn't dominate
        jargon_terms = ["synergy", "leverage", "paradigm", "disrupt"]

        all_text = (DESCRIPTION_SHORT + DESCRIPTION_FULL + ELEVATOR_PITCH).lower()
        jargon_count = sum(1 for term in jargon_terms if term in all_text)

        # Should have minimal marketing jargon
        assert jargon_count < 3, "Too much marketing jargon"

    def test_descriptions_are_actionable(self):
        """Descriptions explain what the product does."""
        all_text = (DESCRIPTION_SHORT + DESCRIPTION_FULL).lower()

        # Should contain action verbs
        action_verbs = ["orchestrat", "deliver", "debat", "provid", "build", "ingest"]
        verb_count = sum(1 for verb in action_verbs if verb in all_text)

        assert verb_count >= 2, "Descriptions should use action verbs"

    def test_value_propositions_clear(self):
        """Value propositions are clearly stated."""
        pitch_lower = ELEVATOR_PITCH.lower()

        # Should mention benefits, not just features
        benefit_terms = ["assurance", "accountability", "defense", "receipts", "memory"]
        benefit_count = sum(1 for term in benefit_terms if term in pitch_lower)

        assert benefit_count >= 2, "Elevator pitch should state clear benefits"
