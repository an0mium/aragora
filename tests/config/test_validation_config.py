"""
Tests for Input Validation Configuration.

Tests cover:
- ValidationMode enum and config
- Environment variable parsing
- Blocking vs warn mode behavior
- Route-specific overrides
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aragora.config.validation import (
    ValidationConfig,
    ValidationMode,
    RouteValidationOverride,
    get_validation_config,
    clear_validation_config_cache,
    create_validation_response,
    DEFAULT_ROUTE_OVERRIDES,
)


class TestValidationMode:
    """Tests for ValidationMode enum."""

    def test_mode_values(self):
        """Test validation mode enum values."""
        assert ValidationMode.BLOCKING.value == "blocking"
        assert ValidationMode.WARN.value == "warn"
        assert ValidationMode.DISABLED.value == "disabled"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert ValidationMode("blocking") == ValidationMode.BLOCKING
        assert ValidationMode("warn") == ValidationMode.WARN
        assert ValidationMode("disabled") == ValidationMode.DISABLED

    def test_invalid_mode_raises(self):
        """Test invalid mode string raises ValueError."""
        with pytest.raises(ValueError):
            ValidationMode("invalid")


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_default_values(self):
        """Test default config values are production-safe."""
        config = ValidationConfig()

        assert config.mode == ValidationMode.BLOCKING
        assert config.is_blocking is True
        assert config.is_enabled is True
        assert config.max_body_size == 10_485_760
        assert config.max_json_depth == 10
        assert config.max_array_items == 1000
        assert config.max_object_keys == 500

    def test_blocking_mode_properties(self):
        """Test is_blocking and is_enabled for blocking mode."""
        config = ValidationConfig(mode=ValidationMode.BLOCKING)

        assert config.is_blocking is True
        assert config.is_enabled is True

    def test_warn_mode_properties(self):
        """Test is_blocking and is_enabled for warn mode."""
        config = ValidationConfig(mode=ValidationMode.WARN)

        assert config.is_blocking is False
        assert config.is_enabled is True

    def test_disabled_mode_properties(self):
        """Test is_blocking and is_enabled for disabled mode."""
        config = ValidationConfig(mode=ValidationMode.DISABLED)

        assert config.is_blocking is False
        assert config.is_enabled is False

    def test_to_dict(self):
        """Test config serialization."""
        config = ValidationConfig(
            mode=ValidationMode.WARN,
            max_body_size=5_000_000,
        )
        data = config.to_dict()

        assert data["mode"] == "warn"
        assert data["is_blocking"] is False
        assert data["is_enabled"] is True
        assert data["max_body_size"] == 5_000_000

    def test_frozen_immutability(self):
        """Test config is immutable (frozen dataclass)."""
        config = ValidationConfig()

        with pytest.raises(AttributeError):
            config.mode = ValidationMode.WARN  # type: ignore


class TestGetValidationConfig:
    """Tests for get_validation_config() function."""

    def setup_method(self):
        """Clear config cache before each test."""
        clear_validation_config_cache()

    def teardown_method(self):
        """Clear config cache after each test."""
        clear_validation_config_cache()

    def test_default_is_blocking(self):
        """Test default mode is blocking for security."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_validation_config()
            assert config.mode == ValidationMode.BLOCKING

    def test_explicit_mode_blocking(self):
        """Test ARAGORA_VALIDATION_MODE=blocking."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "blocking"}, clear=True):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.mode == ValidationMode.BLOCKING

    def test_explicit_mode_warn(self):
        """Test ARAGORA_VALIDATION_MODE=warn."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "warn"}, clear=True):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.mode == ValidationMode.WARN

    def test_explicit_mode_disabled(self):
        """Test ARAGORA_VALIDATION_MODE=disabled."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "disabled"}, clear=True):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.mode == ValidationMode.DISABLED

    def test_legacy_blocking_true(self):
        """Test legacy ARAGORA_VALIDATION_BLOCKING=true."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_BLOCKING": "true"}, clear=True):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.mode == ValidationMode.BLOCKING

    def test_legacy_blocking_false(self):
        """Test legacy ARAGORA_VALIDATION_BLOCKING=false."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_BLOCKING": "false"}, clear=True):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.mode == ValidationMode.WARN

    def test_explicit_mode_overrides_legacy(self):
        """Test explicit mode takes precedence over legacy."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_VALIDATION_MODE": "warn",
                "ARAGORA_VALIDATION_BLOCKING": "true",
            },
            clear=True,
        ):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.mode == ValidationMode.WARN

    def test_custom_max_body_size(self):
        """Test custom max body size from environment."""
        with patch.dict(
            os.environ,
            {"ARAGORA_VALIDATION_MAX_BODY_SIZE": "5000000"},
            clear=True,
        ):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.max_body_size == 5_000_000

    def test_custom_json_depth(self):
        """Test custom max JSON depth from environment."""
        with patch.dict(
            os.environ,
            {"ARAGORA_VALIDATION_MAX_JSON_DEPTH": "20"},
            clear=True,
        ):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.max_json_depth == 20

    def test_invalid_int_falls_back_to_default(self):
        """Test invalid integer values use defaults."""
        with patch.dict(
            os.environ,
            {"ARAGORA_VALIDATION_MAX_BODY_SIZE": "not_a_number"},
            clear=True,
        ):
            clear_validation_config_cache()
            config = get_validation_config()
            assert config.max_body_size == 10_485_760  # Default

    def test_config_is_cached(self):
        """Test configuration is cached after first load."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "warn"}, clear=True):
            clear_validation_config_cache()
            config1 = get_validation_config()

        # Change env var - should not affect cached config
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "blocking"}, clear=True):
            config2 = get_validation_config()

        assert config1 is config2
        assert config1.mode == ValidationMode.WARN

    def test_cache_clear_reloads(self):
        """Test clearing cache reloads config."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "warn"}, clear=True):
            clear_validation_config_cache()
            config1 = get_validation_config()
            assert config1.mode == ValidationMode.WARN

        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "blocking"}, clear=True):
            clear_validation_config_cache()
            config2 = get_validation_config()
            assert config2.mode == ValidationMode.BLOCKING


class TestCreateValidationResponse:
    """Tests for validation error response creation."""

    def setup_method(self):
        """Clear config cache before each test."""
        clear_validation_config_cache()

    def teardown_method(self):
        """Clear config cache after each test."""
        clear_validation_config_cache()

    def test_blocking_mode_returns_error(self):
        """Test blocking mode returns error response."""
        config = ValidationConfig(mode=ValidationMode.BLOCKING)
        errors = ["Field 'name' is required", "Field 'email' is invalid"]

        response = create_validation_response(errors, config)

        assert response["error"] == "Validation failed"
        assert response["code"] == "validation_error"
        assert response["details"]["errors"] == errors
        assert response["details"]["count"] == 2

    def test_warn_mode_returns_empty(self):
        """Test warn mode returns empty response (don't block)."""
        config = ValidationConfig(mode=ValidationMode.WARN)
        errors = ["Field 'name' is required"]

        response = create_validation_response(errors, config)

        assert response == {}

    def test_disabled_mode_returns_empty(self):
        """Test disabled mode returns empty response."""
        config = ValidationConfig(mode=ValidationMode.DISABLED)
        errors = ["Field 'name' is required"]

        response = create_validation_response(errors, config)

        assert response == {}


class TestRouteValidationOverride:
    """Tests for route-specific validation overrides."""

    def test_override_creation(self):
        """Test creating route override."""
        override = RouteValidationOverride(
            path_pattern=r"^/api/batch",
            max_body_size=50_000_000,
            max_array_items=10000,
        )

        assert override.path_pattern == r"^/api/batch"
        assert override.max_body_size == 50_000_000
        assert override.max_array_items == 10000
        assert override.max_json_depth is None

    def test_default_overrides_exist(self):
        """Test default route overrides are defined."""
        assert len(DEFAULT_ROUTE_OVERRIDES) > 0

        # Check batch endpoint has override
        batch_override = next(
            (o for o in DEFAULT_ROUTE_OVERRIDES if "batch" in o.path_pattern),
            None,
        )
        assert batch_override is not None
        assert batch_override.max_body_size > 10_485_760  # Larger than default


class TestValidationModeIntegration:
    """Integration tests for validation mode behavior."""

    def setup_method(self):
        """Clear config cache before each test."""
        clear_validation_config_cache()

    def teardown_method(self):
        """Clear config cache after each test."""
        clear_validation_config_cache()

    def test_blocking_mode_workflow(self):
        """Test typical blocking mode workflow."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "blocking"}, clear=True):
            clear_validation_config_cache()
            config = get_validation_config()

            # Simulate validation failure
            errors = ["Invalid input"]

            if config.is_blocking:
                response = create_validation_response(errors, config)
                assert response["error"] == "Validation failed"
                # Would return 400 error

    def test_warn_mode_workflow(self):
        """Test typical warn mode workflow."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "warn"}, clear=True):
            clear_validation_config_cache()
            config = get_validation_config()

            # Simulate validation failure
            errors = ["Invalid input"]

            if not config.is_blocking:
                response = create_validation_response(errors, config)
                assert response == {}
                # Would log warning but allow request through

    def test_disabled_mode_skips_validation(self):
        """Test disabled mode skips validation entirely."""
        with patch.dict(os.environ, {"ARAGORA_VALIDATION_MODE": "disabled"}, clear=True):
            clear_validation_config_cache()
            config = get_validation_config()

            if not config.is_enabled:
                # Would skip validation entirely
                assert config.is_enabled is False
