"""
Tests for the centralized feature flag registry.

Tests cover:
- Flag registration
- Flag retrieval
- Default value handling
- Environment variable override
- Type validation
- Flag listing and filtering
- Thread safety
- Hierarchical resolution
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
from unittest import mock

import pytest

from aragora.config.feature_flags import (
    FeatureFlagRegistry,
    FlagCategory,
    FlagDefinition,
    FlagStatus,
    FlagUsage,
    RegistryStats,
    get_flag,
    get_flag_registry,
    is_enabled,
    reset_flag_registry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fresh_registry():
    """Create a fresh registry for each test."""
    # Reset global registry to ensure isolation
    reset_flag_registry()
    yield FeatureFlagRegistry(warn_on_unknown=False)
    reset_flag_registry()


@pytest.fixture
def registry_with_flags(fresh_registry):
    """Create a registry with some test flags registered."""
    fresh_registry.register(
        name="test_bool_flag",
        flag_type=bool,
        default=True,
        description="A test boolean flag",
        category=FlagCategory.CORE,
    )
    fresh_registry.register(
        name="test_int_flag",
        flag_type=int,
        default=42,
        description="A test integer flag",
        category=FlagCategory.PERFORMANCE,
    )
    fresh_registry.register(
        name="test_float_flag",
        flag_type=float,
        default=3.14,
        description="A test float flag",
        category=FlagCategory.EXPERIMENTAL,
    )
    fresh_registry.register(
        name="test_str_flag",
        flag_type=str,
        default="hello",
        description="A test string flag",
        category=FlagCategory.DEBUG,
    )
    return fresh_registry


# =============================================================================
# Flag Registration Tests
# =============================================================================


class TestFlagRegistration:
    """Tests for flag registration."""

    def test_register_basic_flag(self, fresh_registry):
        """Test registering a basic boolean flag."""
        flag = fresh_registry.register(
            name="my_feature",
            flag_type=bool,
            default=False,
            description="My feature flag",
        )

        assert flag.name == "my_feature"
        assert flag.flag_type is bool
        assert flag.default is False
        assert flag.description == "My feature flag"
        assert flag.category == FlagCategory.CORE  # default
        assert flag.status == FlagStatus.ACTIVE  # default

    def test_register_with_all_options(self, fresh_registry):
        """Test registering a flag with all options specified."""
        flag = fresh_registry.register(
            name="deprecated_feature",
            flag_type=int,
            default=100,
            description="An old feature",
            category=FlagCategory.DEPRECATED,
            status=FlagStatus.DEPRECATED,
            env_var="CUSTOM_ENV_VAR",
            deprecated_since="v1.0.0",
            removed_in="v2.0.0",
            replacement="new_feature",
        )

        assert flag.name == "deprecated_feature"
        assert flag.flag_type is int
        assert flag.default == 100
        assert flag.category == FlagCategory.DEPRECATED
        assert flag.status == FlagStatus.DEPRECATED
        assert flag.env_var == "CUSTOM_ENV_VAR"
        assert flag.deprecated_since == "v1.0.0"
        assert flag.removed_in == "v2.0.0"
        assert flag.replacement == "new_feature"

    def test_register_auto_generates_env_var(self, fresh_registry):
        """Test that env_var is auto-generated if not provided."""
        flag = fresh_registry.register(
            name="auto_env_feature",
            flag_type=bool,
            default=True,
        )

        assert flag.env_var == "ARAGORA_AUTO_ENV_FEATURE"

    def test_register_multiple_flags(self, fresh_registry):
        """Test registering multiple flags."""
        fresh_registry.register(name="flag_a", default=True)
        fresh_registry.register(name="flag_b", default=False)
        fresh_registry.register(name="flag_c", flag_type=int, default=10)

        assert fresh_registry.is_registered("flag_a")
        assert fresh_registry.is_registered("flag_b")
        assert fresh_registry.is_registered("flag_c")

    def test_register_overwrites_existing(self, fresh_registry):
        """Test that re-registering a flag overwrites it."""
        fresh_registry.register(name="overwrite_me", default=True)
        assert fresh_registry.get_value("overwrite_me") is True

        fresh_registry.register(name="overwrite_me", default=False)
        assert fresh_registry.get_value("overwrite_me") is False


# =============================================================================
# Flag Retrieval Tests
# =============================================================================


class TestFlagRetrieval:
    """Tests for flag value retrieval."""

    def test_get_registered_flag_value(self, registry_with_flags):
        """Test getting value of a registered flag."""
        value = registry_with_flags.get_value("test_bool_flag")
        assert value is True

        value = registry_with_flags.get_value("test_int_flag")
        assert value == 42

    def test_get_definition(self, registry_with_flags):
        """Test getting flag definition."""
        defn = registry_with_flags.get_definition("test_bool_flag")

        assert defn is not None
        assert defn.name == "test_bool_flag"
        assert defn.flag_type is bool
        assert defn.default is True

    def test_get_definition_nonexistent(self, registry_with_flags):
        """Test getting definition for non-existent flag returns None."""
        defn = registry_with_flags.get_definition("nonexistent")
        assert defn is None

    def test_is_registered(self, registry_with_flags):
        """Test is_registered check."""
        assert registry_with_flags.is_registered("test_bool_flag") is True
        assert registry_with_flags.is_registered("nonexistent") is False


# =============================================================================
# Default Value Handling Tests
# =============================================================================


class TestDefaultValueHandling:
    """Tests for default value handling."""

    def test_unregistered_flag_returns_provided_default(self, fresh_registry):
        """Test that unregistered flags return the provided default."""
        value = fresh_registry.get_value("unknown_flag", default="fallback")
        assert value == "fallback"

    def test_unregistered_flag_returns_none_without_default(self, fresh_registry):
        """Test that unregistered flags return None if no default provided."""
        value = fresh_registry.get_value("unknown_flag")
        assert value is None

    def test_registered_flag_ignores_caller_default(self, registry_with_flags):
        """Test that registered flags use their registered default, not caller's."""
        # The registered default is True
        value = registry_with_flags.get_value("test_bool_flag", default=False)
        assert value is True  # Should use registered default, not caller's


# =============================================================================
# Environment Variable Override Tests
# =============================================================================


class TestEnvironmentVariableOverride:
    """Tests for environment variable override functionality."""

    def test_env_var_overrides_default(self, fresh_registry):
        """Test that environment variable overrides the default."""
        fresh_registry.register(
            name="env_override_test",
            flag_type=bool,
            default=False,
            env_var="TEST_ENV_OVERRIDE",
        )

        with mock.patch.dict(os.environ, {"TEST_ENV_OVERRIDE": "true"}):
            value = fresh_registry.get_value("env_override_test")
            assert value is True

    def test_env_var_int_parsing(self, fresh_registry):
        """Test parsing integer from environment variable."""
        fresh_registry.register(
            name="env_int_test",
            flag_type=int,
            default=10,
            env_var="TEST_ENV_INT",
        )

        with mock.patch.dict(os.environ, {"TEST_ENV_INT": "999"}):
            value = fresh_registry.get_value("env_int_test")
            assert value == 999

    def test_env_var_float_parsing(self, fresh_registry):
        """Test parsing float from environment variable."""
        fresh_registry.register(
            name="env_float_test",
            flag_type=float,
            default=1.0,
            env_var="TEST_ENV_FLOAT",
        )

        with mock.patch.dict(os.environ, {"TEST_ENV_FLOAT": "2.718"}):
            value = fresh_registry.get_value("env_float_test")
            assert value == pytest.approx(2.718)

    def test_env_var_string_parsing(self, fresh_registry):
        """Test parsing string from environment variable."""
        fresh_registry.register(
            name="env_str_test",
            flag_type=str,
            default="default",
            env_var="TEST_ENV_STR",
        )

        with mock.patch.dict(os.environ, {"TEST_ENV_STR": "custom_value"}):
            value = fresh_registry.get_value("env_str_test")
            assert value == "custom_value"

    def test_env_var_bool_truthy_values(self, fresh_registry):
        """Test various truthy string values for boolean env vars."""
        fresh_registry.register(
            name="bool_test",
            flag_type=bool,
            default=False,
            env_var="TEST_BOOL",
        )

        truthy_values = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]
        for truthy in truthy_values:
            with mock.patch.dict(os.environ, {"TEST_BOOL": truthy}):
                value = fresh_registry.get_value("bool_test")
                assert value is True, f"Expected True for '{truthy}'"

    def test_env_var_bool_falsy_values(self, fresh_registry):
        """Test various falsy string values for boolean env vars."""
        fresh_registry.register(
            name="bool_test",
            flag_type=bool,
            default=True,
            env_var="TEST_BOOL",
        )

        falsy_values = ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]
        for falsy in falsy_values:
            with mock.patch.dict(os.environ, {"TEST_BOOL": falsy}):
                value = fresh_registry.get_value("bool_test")
                assert value is False, f"Expected False for '{falsy}'"

    def test_missing_env_var_uses_default(self, fresh_registry):
        """Test that missing env var falls back to default."""
        fresh_registry.register(
            name="missing_env_test",
            flag_type=bool,
            default=True,
            env_var="NONEXISTENT_ENV_VAR_12345",
        )

        value = fresh_registry.get_value("missing_env_test")
        assert value is True


# =============================================================================
# Type Validation Tests
# =============================================================================


class TestTypeValidation:
    """Tests for type validation."""

    def test_validate_flags_valid_types(self, registry_with_flags):
        """Test validation passes for correct types."""
        errors = registry_with_flags.validate_flags(
            {
                "test_bool_flag": True,
                "test_int_flag": 100,
                "test_float_flag": 1.5,
                "test_str_flag": "world",
            }
        )
        assert errors == []

    def test_validate_flags_wrong_types(self, registry_with_flags):
        """Test validation fails for wrong types."""
        errors = registry_with_flags.validate_flags(
            {
                "test_bool_flag": "not_a_bool",
                "test_int_flag": "not_an_int",
            }
        )
        assert len(errors) == 2
        assert any("test_bool_flag" in e for e in errors)
        assert any("test_int_flag" in e for e in errors)

    def test_validate_flags_unknown_flag(self, registry_with_flags):
        """Test validation reports unknown flags."""
        errors = registry_with_flags.validate_flags(
            {
                "unknown_flag": True,
            }
        )
        assert len(errors) == 1
        assert "Unknown flag" in errors[0]

    def test_is_enabled_returns_bool(self, registry_with_flags):
        """Test is_enabled always returns a boolean."""
        result = registry_with_flags.is_enabled("test_bool_flag")
        assert isinstance(result, bool)
        assert result is True

    def test_is_enabled_converts_string_to_bool(self, fresh_registry):
        """Test is_enabled handles string values correctly."""
        fresh_registry.register(
            name="string_bool",
            flag_type=str,
            default="true",
        )
        result = fresh_registry.is_enabled("string_bool")
        assert result is True

    def test_is_enabled_unregistered_returns_false(self, fresh_registry):
        """Test is_enabled returns False for unregistered flags."""
        result = fresh_registry.is_enabled("nonexistent_flag")
        assert result is False


# =============================================================================
# Flag Listing and Category Filtering Tests
# =============================================================================


class TestFlagListing:
    """Tests for flag listing and filtering."""

    def test_get_all_flags(self, registry_with_flags):
        """Test getting all registered flags."""
        flags = registry_with_flags.get_all_flags()
        # Should have built-in flags plus our 4 test flags
        assert len(flags) >= 4

    def test_get_all_flags_by_category(self, registry_with_flags):
        """Test filtering flags by category."""
        core_flags = registry_with_flags.get_all_flags(category=FlagCategory.CORE)
        for flag in core_flags:
            assert flag.category == FlagCategory.CORE

        experimental_flags = registry_with_flags.get_all_flags(category=FlagCategory.EXPERIMENTAL)
        for flag in experimental_flags:
            assert flag.category == FlagCategory.EXPERIMENTAL

    def test_get_all_flags_by_status(self, fresh_registry):
        """Test filtering flags by status."""
        fresh_registry.register(
            name="active_flag",
            status=FlagStatus.ACTIVE,
        )
        fresh_registry.register(
            name="beta_flag",
            status=FlagStatus.BETA,
        )
        fresh_registry.register(
            name="deprecated_flag",
            status=FlagStatus.DEPRECATED,
        )

        active_flags = fresh_registry.get_all_flags(status=FlagStatus.ACTIVE)
        assert any(f.name == "active_flag" for f in active_flags)

        beta_flags = fresh_registry.get_all_flags(status=FlagStatus.BETA)
        assert any(f.name == "beta_flag" for f in beta_flags)

    def test_get_all_flags_combined_filter(self, fresh_registry):
        """Test filtering by both category and status."""
        fresh_registry.register(
            name="core_active",
            category=FlagCategory.CORE,
            status=FlagStatus.ACTIVE,
        )
        fresh_registry.register(
            name="core_deprecated",
            category=FlagCategory.CORE,
            status=FlagStatus.DEPRECATED,
        )
        fresh_registry.register(
            name="experimental_active",
            category=FlagCategory.EXPERIMENTAL,
            status=FlagStatus.ACTIVE,
        )

        flags = fresh_registry.get_all_flags(category=FlagCategory.CORE, status=FlagStatus.ACTIVE)
        flag_names = [f.name for f in flags]
        assert "core_active" in flag_names
        assert "core_deprecated" not in flag_names
        assert "experimental_active" not in flag_names

    def test_flags_are_sorted(self, registry_with_flags):
        """Test that flags are sorted by category then name."""
        flags = registry_with_flags.get_all_flags()
        # Verify sorted by category then name
        for i in range(len(flags) - 1):
            curr = flags[i]
            next_flag = flags[i + 1]
            assert (curr.category.value, curr.name) <= (
                next_flag.category.value,
                next_flag.name,
            )


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_registration(self, fresh_registry):
        """Test concurrent flag registration is thread-safe."""
        num_threads = 10
        flags_per_thread = 100

        def register_flags(thread_id):
            for i in range(flags_per_thread):
                fresh_registry.register(
                    name=f"thread_{thread_id}_flag_{i}",
                    default=True,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_flags, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        # Verify all flags were registered
        for t in range(num_threads):
            for i in range(flags_per_thread):
                assert fresh_registry.is_registered(f"thread_{t}_flag_{i}")

    def test_concurrent_read_write(self, fresh_registry):
        """Test concurrent reads and writes are thread-safe."""
        fresh_registry.register(name="concurrent_flag", flag_type=int, default=0)
        errors = []
        results = []

        def reader():
            try:
                for _ in range(100):
                    value = fresh_registry.get_value("concurrent_flag")
                    results.append(value)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    fresh_registry.register(name=f"writer_flag_{i}", flag_type=int, default=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)] + [
            threading.Thread(target=writer) for _ in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

    def test_concurrent_get_value(self, fresh_registry):
        """Test concurrent get_value calls are thread-safe."""
        fresh_registry.register(name="read_test", flag_type=int, default=42)

        results = []
        lock = threading.Lock()

        def read_flag():
            for _ in range(1000):
                value = fresh_registry.get_value("read_test")
                with lock:
                    results.append(value)

        threads = [threading.Thread(target=read_flag) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should return the same value
        assert all(v == 42 for v in results)
        assert len(results) == 10000


# =============================================================================
# Usage Tracking Tests
# =============================================================================


class TestUsageTracking:
    """Tests for flag usage tracking."""

    def test_usage_tracking_on_access(self, fresh_registry):
        """Test that accessing a flag updates usage stats."""
        fresh_registry.register(name="tracked_flag", default=True)

        # Access the flag multiple times
        for _ in range(5):
            fresh_registry.get_value("tracked_flag")

        usage = fresh_registry.get_usage("tracked_flag")
        assert usage is not None
        assert usage.access_count == 5
        assert usage.last_accessed is not None

    def test_unknown_flag_tracking(self, fresh_registry):
        """Test that unknown flags are tracked."""
        fresh_registry.get_value("unknown_flag_1", default=None)
        fresh_registry.get_value("unknown_flag_2", default=None)

        unknown = fresh_registry.get_unknown_flags()
        assert "unknown_flag_1" in unknown
        assert "unknown_flag_2" in unknown


# =============================================================================
# Registry Statistics Tests
# =============================================================================


class TestRegistryStats:
    """Tests for registry statistics."""

    def test_get_stats(self, registry_with_flags):
        """Test getting registry statistics."""
        # Add a deprecated flag
        registry_with_flags.register(
            name="deprecated_stat_flag",
            status=FlagStatus.DEPRECATED,
            category=FlagCategory.DEPRECATED,
        )

        stats = registry_with_flags.get_stats()

        assert isinstance(stats, RegistryStats)
        assert stats.total_flags >= 5  # 4 test flags + 1 deprecated + built-ins
        assert stats.deprecated_flags >= 1
        assert FlagCategory.CORE.value in stats.flags_by_category

    def test_stats_to_dict(self, fresh_registry):
        """Test stats serialization to dict."""
        fresh_registry.register(name="stat_flag", default=True)
        stats = fresh_registry.get_stats()

        stats_dict = stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert "total_flags" in stats_dict
        assert "active_flags" in stats_dict
        assert "deprecated_flags" in stats_dict
        assert "flags_by_category" in stats_dict


# =============================================================================
# Deprecation Warning Tests
# =============================================================================


class TestDeprecationWarnings:
    """Tests for deprecation warning functionality."""

    def test_deprecated_flag_logs_warning(self, fresh_registry, caplog):
        """Test that accessing deprecated flag logs a warning."""
        fresh_registry.register(
            name="old_feature",
            status=FlagStatus.DEPRECATED,
            replacement="new_feature",
            removed_in="v2.0.0",
        )

        import logging

        with caplog.at_level(logging.WARNING):
            fresh_registry.get_value("old_feature")

        assert "deprecated" in caplog.text.lower()
        assert "new_feature" in caplog.text
        assert "v2.0.0" in caplog.text


# =============================================================================
# Global Registry / Convenience Function Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry and convenience functions."""

    def test_get_flag_registry_returns_singleton(self):
        """Test that get_flag_registry returns the same instance."""
        reset_flag_registry()
        reg1 = get_flag_registry()
        reg2 = get_flag_registry()
        assert reg1 is reg2
        reset_flag_registry()

    def test_reset_flag_registry(self):
        """Test that reset_flag_registry clears the global instance."""
        reset_flag_registry()
        reg1 = get_flag_registry()
        reg1.register(name="test_reset", default=True)

        reset_flag_registry()
        reg2 = get_flag_registry()

        # The new registry should still have built-in flags
        # but test_reset should be gone (it's not a built-in)
        assert reg1 is not reg2

    def test_is_enabled_convenience_function(self):
        """Test the is_enabled convenience function."""
        reset_flag_registry()
        registry = get_flag_registry()
        registry.register(name="convenience_test", default=True)

        result = is_enabled("convenience_test")
        assert result is True
        reset_flag_registry()

    def test_get_flag_convenience_function(self):
        """Test the get_flag convenience function."""
        reset_flag_registry()
        registry = get_flag_registry()
        registry.register(name="get_flag_test", flag_type=int, default=100)

        result = get_flag("get_flag_test")
        assert result == 100
        reset_flag_registry()


# =============================================================================
# Documentation Export Tests
# =============================================================================


class TestDocumentationExport:
    """Tests for documentation export functionality."""

    def test_export_documentation(self, registry_with_flags):
        """Test exporting flag documentation as markdown."""
        markdown = registry_with_flags.export_documentation()

        assert "# Feature Flags" in markdown
        assert "test_bool_flag" in markdown
        assert "test_int_flag" in markdown
        assert "bool" in markdown.lower()
        assert "int" in markdown.lower()

    def test_export_includes_deprecated_badge(self, fresh_registry):
        """Test that deprecated flags get marked in documentation."""
        fresh_registry.register(
            name="doc_deprecated",
            status=FlagStatus.DEPRECATED,
            description="Old feature",
        )

        markdown = fresh_registry.export_documentation()
        assert "[DEPRECATED]" in markdown

    def test_export_includes_beta_badge(self, fresh_registry):
        """Test that beta flags get marked in documentation."""
        fresh_registry.register(
            name="doc_beta",
            status=FlagStatus.BETA,
            description="Beta feature",
        )

        markdown = fresh_registry.export_documentation()
        assert "[BETA]" in markdown


# =============================================================================
# Built-in Flags Tests
# =============================================================================


class TestBuiltinFlags:
    """Tests for built-in flags that are registered automatically."""

    def test_builtin_flags_are_registered(self, fresh_registry):
        """Test that built-in flags are automatically registered."""
        # These are some known built-in flags from the implementation
        expected_builtins = [
            "enable_knowledge_retrieval",
            "enable_knowledge_ingestion",
            "enable_performance_monitor",
            "enable_coordinated_writes",
            "enable_hook_handlers",
        ]

        for flag_name in expected_builtins:
            assert fresh_registry.is_registered(flag_name), (
                f"Expected built-in flag '{flag_name}' to be registered"
            )

    def test_builtin_knowledge_flags(self, fresh_registry):
        """Test knowledge-related built-in flags."""
        knowledge_flags = fresh_registry.get_all_flags(category=FlagCategory.KNOWLEDGE)
        assert len(knowledge_flags) > 0

        flag_names = [f.name for f in knowledge_flags]
        assert "enable_knowledge_retrieval" in flag_names
        assert "enable_knowledge_ingestion" in flag_names


# =============================================================================
# FlagDefinition Tests
# =============================================================================


class TestFlagDefinition:
    """Tests for the FlagDefinition dataclass."""

    def test_flag_definition_post_init_auto_env_var(self):
        """Test that FlagDefinition auto-generates env_var."""
        flag = FlagDefinition(
            name="my_test_flag",
            flag_type=bool,
            default=True,
            description="Test",
        )
        assert flag.env_var == "ARAGORA_MY_TEST_FLAG"

    def test_flag_definition_custom_env_var(self):
        """Test that custom env_var is preserved."""
        flag = FlagDefinition(
            name="my_test_flag",
            flag_type=bool,
            default=True,
            description="Test",
            env_var="CUSTOM_VAR",
        )
        assert flag.env_var == "CUSTOM_VAR"


# =============================================================================
# FlagUsage Tests
# =============================================================================


class TestFlagUsage:
    """Tests for the FlagUsage dataclass."""

    def test_flag_usage_record_access(self):
        """Test recording access on FlagUsage."""
        usage = FlagUsage(name="test")
        assert usage.access_count == 0
        assert usage.last_accessed is None

        usage.record_access()
        assert usage.access_count == 1
        assert usage.last_accessed is not None

        usage.record_access(location="test_location")
        assert usage.access_count == 2
        assert usage.access_locations["test_location"] == 1

    def test_flag_usage_location_tracking(self):
        """Test tracking access locations."""
        usage = FlagUsage(name="test")
        usage.record_access(location="module_a")
        usage.record_access(location="module_a")
        usage.record_access(location="module_b")

        assert usage.access_locations["module_a"] == 2
        assert usage.access_locations["module_b"] == 1
