"""Tests for aragora.debate.feature_validator — Feature dependency validation."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from aragora.debate.feature_validator import (
    FEATURE_DEPENDENCIES,
    FeatureDependency,
    FeatureValidationResult,
    ValidationResult,
    _check_module,
    validate_and_warn,
    validate_feature_dependencies,
)


# ---------------------------------------------------------------------------
# FeatureDependency
# ---------------------------------------------------------------------------


class TestFeatureDependency:
    def test_fields(self):
        fd = FeatureDependency(name="test", description="A test feature")
        assert fd.name == "test"
        assert fd.description == "A test feature"
        assert fd.required_modules == []
        assert fd.required_config == []
        assert fd.check_fn is None

    def test_with_modules(self):
        fd = FeatureDependency(name="test", description="test", required_modules=["z3"])
        assert fd.required_modules == ["z3"]

    def test_with_check_fn(self):
        fd = FeatureDependency(name="test", description="test", check_fn=lambda: True)
        assert fd.check_fn() is True


# ---------------------------------------------------------------------------
# FeatureValidationResult
# ---------------------------------------------------------------------------


class TestFeatureValidationResult:
    def test_defaults(self):
        r = FeatureValidationResult(valid=True)
        assert r.valid is True
        assert r.warnings == []
        assert r.errors == []

    def test_with_warnings(self):
        r = FeatureValidationResult(valid=True, warnings=["warn1"])
        assert len(r.warnings) == 1

    def test_backward_compat_alias(self):
        assert ValidationResult is FeatureValidationResult


# ---------------------------------------------------------------------------
# _check_module
# ---------------------------------------------------------------------------


class TestCheckModule:
    def test_available_module(self):
        assert _check_module("os") is True

    def test_unavailable_module(self):
        assert _check_module("nonexistent_module_xyz") is False


# ---------------------------------------------------------------------------
# FEATURE_DEPENDENCIES registry
# ---------------------------------------------------------------------------


class TestFeatureDependenciesRegistry:
    def test_has_expected_keys(self):
        expected = {
            "formal_verification",
            "belief_guidance",
            "knowledge_mound",
            "rlm_compression",
            "checkpointing",
            "calibration",
            "performance_monitoring",
            "population_evolution",
        }
        assert set(FEATURE_DEPENDENCIES.keys()) == expected

    def test_formal_verification_has_check_fn(self):
        dep = FEATURE_DEPENDENCIES["formal_verification"]
        assert dep.check_fn is not None
        assert dep.required_modules == ["z3"]


# ---------------------------------------------------------------------------
# validate_feature_dependencies
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create a mock ArenaConfig with common defaults."""
    config = MagicMock()
    config.enable_belief_guidance = False
    config.enable_knowledge_retrieval = False
    config.enable_knowledge_ingestion = False
    config.use_rlm_limiter = False
    config.enable_checkpointing = False
    config.enable_performance_monitor = False
    config.auto_evolve = False
    config.protocol = None
    config.dissent_retriever = None
    config.consensus_memory = None
    config.knowledge_mound = None
    config.checkpoint_manager = None
    config.calibration_tracker = None
    config.performance_monitor = None
    config.population_manager = None
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


class TestValidateFeatureDependencies:
    def test_all_disabled(self):
        config = _make_config()
        result = validate_feature_dependencies(config)
        assert result.valid is True
        assert result.warnings == []
        assert result.errors == []

    def test_belief_guidance_no_deps(self):
        config = _make_config(enable_belief_guidance=True)
        result = validate_feature_dependencies(config)
        assert any("belief_guidance" in w for w in result.warnings)

    def test_belief_guidance_with_deps(self):
        config = _make_config(
            enable_belief_guidance=True,
            dissent_retriever=MagicMock(),
        )
        result = validate_feature_dependencies(config)
        assert not any("belief_guidance" in w for w in result.warnings)

    def test_knowledge_mound_no_deps(self):
        config = _make_config(enable_knowledge_retrieval=True)
        result = validate_feature_dependencies(config)
        assert any("knowledge_mound" in w for w in result.warnings)

    def test_knowledge_mound_with_deps(self):
        config = _make_config(
            enable_knowledge_retrieval=True,
            knowledge_mound=MagicMock(),
        )
        result = validate_feature_dependencies(config)
        assert not any("knowledge_mound" in w for w in result.warnings)

    def test_knowledge_ingestion_no_deps(self):
        config = _make_config(enable_knowledge_ingestion=True)
        result = validate_feature_dependencies(config)
        assert any("knowledge_mound" in w for w in result.warnings)

    def test_checkpointing_no_deps(self):
        config = _make_config(enable_checkpointing=True)
        result = validate_feature_dependencies(config)
        assert any("checkpointing" in w for w in result.warnings)

    def test_checkpointing_with_deps(self):
        config = _make_config(
            enable_checkpointing=True,
            checkpoint_manager=MagicMock(),
        )
        result = validate_feature_dependencies(config)
        assert not any("checkpointing" in w for w in result.warnings)

    def test_formal_verification_no_z3(self):
        protocol = MagicMock()
        protocol.enable_formal_verification = True
        config = _make_config(protocol=protocol)
        with patch("aragora.debate.feature_validator._check_module", return_value=False):
            result = validate_feature_dependencies(config)
        assert result.valid is False
        assert any("formal_verification" in e for e in result.errors)

    def test_formal_verification_with_z3(self):
        protocol = MagicMock()
        protocol.enable_formal_verification = True
        config = _make_config(protocol=protocol)
        with patch("aragora.debate.feature_validator._check_module", return_value=True):
            result = validate_feature_dependencies(config)
        assert not any("formal_verification" in e for e in result.errors)

    def test_calibration_no_deps(self):
        protocol = MagicMock()
        protocol.enable_calibration = True
        config = _make_config(protocol=protocol)
        result = validate_feature_dependencies(config)
        assert any("calibration" in w for w in result.warnings)

    def test_calibration_with_deps(self):
        protocol = MagicMock()
        protocol.enable_calibration = True
        config = _make_config(
            protocol=protocol,
            calibration_tracker=MagicMock(),
        )
        result = validate_feature_dependencies(config)
        assert not any("calibration" in w for w in result.warnings)

    def test_performance_monitoring_no_deps(self):
        config = _make_config(enable_performance_monitor=True)
        result = validate_feature_dependencies(config)
        assert any("performance_monitoring" in w for w in result.warnings)

    def test_population_evolution_no_deps(self):
        config = _make_config(auto_evolve=True)
        result = validate_feature_dependencies(config)
        assert result.valid is False
        assert any("population_evolution" in e for e in result.errors)

    def test_population_evolution_with_deps(self):
        config = _make_config(
            auto_evolve=True,
            population_manager=MagicMock(),
        )
        result = validate_feature_dependencies(config)
        assert not any("population_evolution" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_and_warn
# ---------------------------------------------------------------------------


class TestValidateAndWarn:
    def test_no_errors(self):
        config = _make_config()
        # Should not raise
        validate_and_warn(config)

    def test_with_errors(self):
        config = _make_config(auto_evolve=True)
        # Should not raise even with errors
        validate_and_warn(config)

    def test_rlm_compression_no_module(self):
        config = _make_config(use_rlm_limiter=True)
        with patch("aragora.debate.feature_validator._check_module", return_value=False):
            result = validate_feature_dependencies(config)
        assert any("rlm_compression" in w for w in result.warnings)
