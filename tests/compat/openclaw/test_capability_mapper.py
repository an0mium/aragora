"""Tests for OpenClaw capability mapper."""

from __future__ import annotations

import logging

import pytest

from aragora.compat.openclaw.capability_mapper import CapabilityMapper
from aragora.skills.base import SkillCapability


# ---------------------------------------------------------------------------
# Tests: to_aragora
# ---------------------------------------------------------------------------


class TestToAragora:
    """Test converting OpenClaw capability strings to Aragora SkillCapability."""

    @pytest.mark.parametrize(
        "openclaw, expected",
        [
            ("browser", SkillCapability.WEB_FETCH),
            ("file_read", SkillCapability.READ_LOCAL),
            ("file_write", SkillCapability.WRITE_LOCAL),
            ("code_execution", SkillCapability.CODE_EXECUTION),
            ("shell", SkillCapability.SHELL_EXECUTION),
            ("web_search", SkillCapability.WEB_SEARCH),
            ("api", SkillCapability.EXTERNAL_API),
            ("database", SkillCapability.READ_DATABASE),
        ],
    )
    def test_known_single_capability(self, openclaw: str, expected: SkillCapability) -> None:
        """Each known OpenClaw string maps to the correct Aragora capability."""
        result = CapabilityMapper.to_aragora([openclaw])
        assert result == [expected]

    def test_multiple_capabilities(self) -> None:
        """Convert a list of several OpenClaw capabilities at once."""
        result = CapabilityMapper.to_aragora(["browser", "file_read", "shell"])
        assert result == [
            SkillCapability.WEB_FETCH,
            SkillCapability.READ_LOCAL,
            SkillCapability.SHELL_EXECUTION,
        ]

    def test_unknown_capability_skipped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown OpenClaw capability is logged and excluded from results."""
        with caplog.at_level(logging.WARNING):
            result = CapabilityMapper.to_aragora(["browser", "teleportation"])

        assert result == [SkillCapability.WEB_FETCH]
        assert any("Unknown OpenClaw capability" in msg for msg in caplog.messages)
        assert any("teleportation" in msg for msg in caplog.messages)

    def test_deduplicates(self) -> None:
        """Duplicate mappings are collapsed (browser and screenshot both map to WEB_FETCH)."""
        result = CapabilityMapper.to_aragora(["browser", "screenshot"])
        assert result == [SkillCapability.WEB_FETCH]

    def test_empty_list(self) -> None:
        """Empty input produces empty output."""
        assert CapabilityMapper.to_aragora([]) == []


# ---------------------------------------------------------------------------
# Tests: to_openclaw
# ---------------------------------------------------------------------------


class TestToOpenClaw:
    """Test converting Aragora SkillCapability values to OpenClaw strings."""

    @pytest.mark.parametrize(
        "aragora, expected",
        [
            (SkillCapability.WEB_FETCH, "browser"),
            (SkillCapability.READ_LOCAL, "file_read"),
            (SkillCapability.WRITE_LOCAL, "file_write"),
            (SkillCapability.CODE_EXECUTION, "code_execution"),
            (SkillCapability.SHELL_EXECUTION, "shell"),
            (SkillCapability.WEB_SEARCH, "web_search"),
            (SkillCapability.EXTERNAL_API, "api"),
            (SkillCapability.READ_DATABASE, "database"),
        ],
    )
    def test_known_single_capability(self, aragora: SkillCapability, expected: str) -> None:
        """Each mapped Aragora capability produces the correct OpenClaw string."""
        result = CapabilityMapper.to_openclaw([aragora])
        assert result == [expected]

    def test_unmapped_capability_skipped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Aragora capabilities without an OpenClaw mapping are logged and skipped."""
        with caplog.at_level(logging.WARNING):
            result = CapabilityMapper.to_openclaw(
                [SkillCapability.READ_LOCAL, SkillCapability.DEBATE_CONTEXT]
            )

        assert result == ["file_read"]
        assert any("No OpenClaw mapping" in msg for msg in caplog.messages)

    def test_deduplicates(self) -> None:
        """Duplicate OpenClaw strings are collapsed (READ_DATABASE and WRITE_DATABASE both map to 'database')."""
        result = CapabilityMapper.to_openclaw(
            [SkillCapability.READ_DATABASE, SkillCapability.WRITE_DATABASE]
        )
        assert result == ["database"]

    def test_empty_list(self) -> None:
        """Empty input produces empty output."""
        assert CapabilityMapper.to_openclaw([]) == []


# ---------------------------------------------------------------------------
# Tests: round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test that round-trip conversion preserves semantics."""

    def test_openclaw_to_aragora_to_openclaw(self) -> None:
        """Round-trip: OpenClaw -> Aragora -> OpenClaw gives back the original strings."""
        original = ["file_read", "shell", "web_search"]
        aragora = CapabilityMapper.to_aragora(original)
        back = CapabilityMapper.to_openclaw(aragora)
        assert back == original

    def test_aragora_to_openclaw_to_aragora(self) -> None:
        """Round-trip: Aragora -> OpenClaw -> Aragora gives back the original capabilities."""
        original = [
            SkillCapability.READ_LOCAL,
            SkillCapability.CODE_EXECUTION,
            SkillCapability.WEB_SEARCH,
        ]
        openclaw = CapabilityMapper.to_openclaw(original)
        back = CapabilityMapper.to_aragora(openclaw)
        assert back == original


# ---------------------------------------------------------------------------
# Tests: is_supported
# ---------------------------------------------------------------------------


class TestIsSupported:
    """Test CapabilityMapper.is_supported."""

    @pytest.mark.parametrize(
        "cap",
        [
            "browser",
            "file_read",
            "file_write",
            "code_execution",
            "shell",
            "web_search",
            "api",
            "database",
        ],
    )
    def test_known_capabilities_supported(self, cap: str) -> None:
        assert CapabilityMapper.is_supported(cap) is True

    @pytest.mark.parametrize("cap", ["teleportation", "time_travel", ""])
    def test_unknown_capabilities_not_supported(self, cap: str) -> None:
        assert CapabilityMapper.is_supported(cap) is False


# ---------------------------------------------------------------------------
# Tests: listing helpers
# ---------------------------------------------------------------------------


class TestListingHelpers:
    """Test all_openclaw_capabilities and all_aragora_capabilities."""

    def test_all_openclaw_capabilities_sorted(self) -> None:
        """Returned list should be sorted alphabetically."""
        caps = CapabilityMapper.all_openclaw_capabilities()
        assert caps == sorted(caps)
        # Spot-check a few known entries
        assert "browser" in caps
        assert "shell" in caps
        assert "database" in caps

    def test_all_aragora_capabilities_unique(self) -> None:
        """Returned list should contain no duplicates."""
        caps = CapabilityMapper.all_aragora_capabilities()
        assert len(caps) == len(set(caps))

    def test_all_aragora_capabilities_sorted_by_value(self) -> None:
        """Capabilities should be sorted by their string value."""
        caps = CapabilityMapper.all_aragora_capabilities()
        values = [c.value for c in caps]
        assert values == sorted(values)
