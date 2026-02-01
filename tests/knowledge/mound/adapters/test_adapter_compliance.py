"""
Compliance tests for Knowledge Mound adapters.

Ensures all adapters follow the unified KnowledgeMoundAdapter pattern
with proper inheritance, adapter_name, and observability methods.

This is Phase 0 of the KM Adapter Unification effort.
Adapters that don't yet comply are marked xfail and will pass
once migrated in subsequent phases.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest


# Adapter specifications: module path, class name, and expected compliance status
# xfail_reason: if set, adapter is expected to fail compliance (not yet migrated)
ADAPTER_SPECS: list[dict[str, Any]] = [
    # Pattern A: Already inherit KnowledgeMoundAdapter (should pass)
    {
        "module": "aragora.knowledge.mound.adapters.openclaw_adapter",
        "class_name": "OpenClawAdapter",
        "xfail_reason": None,
    },
    {
        "module": "aragora.knowledge.mound.adapters.fabric_adapter",
        "class_name": "FabricAdapter",
        "xfail_reason": None,
    },
    {
        "module": "aragora.knowledge.mound.adapters.gateway_adapter",
        "class_name": "GatewayAdapter",
        "xfail_reason": None,
    },
    {
        "module": "aragora.knowledge.mound.adapters.calibration_fusion_adapter",
        "class_name": "CalibrationFusionAdapter",
        "xfail_reason": None,
    },
    {
        "module": "aragora.knowledge.mound.adapters.erc8004_adapter",
        "class_name": "ERC8004Adapter",
        "xfail_reason": None,
    },
    {
        "module": "aragora.knowledge.mound.adapters.computer_use_adapter",
        "class_name": "ComputerUseAdapter",
        "xfail_reason": None,
    },
    {
        "module": "aragora.knowledge.mound.adapters.extraction_adapter",
        "class_name": "ExtractionAdapter",
        "xfail_reason": None,
    },
    {
        "module": "aragora.knowledge.mound.adapters.workspace_adapter",
        "class_name": "WorkspaceAdapter",
        "xfail_reason": None,
    },
    # Pattern B: Use mixins + ResilientAdapterMixin (need migration)
    {
        "module": "aragora.knowledge.mound.adapters.consensus_adapter",
        "class_name": "ConsensusAdapter",
        "xfail_reason": "Uses mixin composition, not KnowledgeMoundAdapter",
    },
    {
        "module": "aragora.knowledge.mound.adapters.continuum_adapter",
        "class_name": "ContinuumAdapter",
        "xfail_reason": "Uses mixin composition, not KnowledgeMoundAdapter",
    },
    {
        "module": "aragora.knowledge.mound.adapters.belief_adapter",
        "class_name": "BeliefAdapter",
        "xfail_reason": "Uses mixin composition, not KnowledgeMoundAdapter",
    },
    {
        "module": "aragora.knowledge.mound.adapters.evidence_adapter",
        "class_name": "EvidenceAdapter",
        "xfail_reason": "Uses mixin composition, not KnowledgeMoundAdapter",
    },
    {
        "module": "aragora.knowledge.mound.adapters.insights_adapter",
        "class_name": "InsightsAdapter",
        "xfail_reason": "Uses mixin composition, not KnowledgeMoundAdapter",
    },
    {
        "module": "aragora.knowledge.mound.adapters.cost_adapter",
        "class_name": "CostAdapter",
        "xfail_reason": "Uses mixin composition, not KnowledgeMoundAdapter",
    },
    {
        "module": "aragora.knowledge.mound.adapters.pulse_adapter",
        "class_name": "PulseAdapter",
        "xfail_reason": "Uses mixin composition, not KnowledgeMoundAdapter",
    },
    {
        "module": "aragora.knowledge.mound.adapters.performance.adapter",
        "class_name": "PerformanceAdapter",
        "xfail_reason": "Uses mixin composition, not KnowledgeMoundAdapter",
    },
    # Pattern C: Standalone adapters (need migration)
    {
        "module": "aragora.knowledge.mound.adapters.critique_adapter",
        "class_name": "CritiqueAdapter",
        "xfail_reason": "Standalone adapter, no KnowledgeMoundAdapter inheritance",
    },
    {
        "module": "aragora.knowledge.mound.adapters.control_plane_adapter",
        "class_name": "ControlPlaneAdapter",
        "xfail_reason": "Standalone adapter, no KnowledgeMoundAdapter inheritance",
    },
    {
        "module": "aragora.knowledge.mound.adapters.receipt_adapter",
        "class_name": "ReceiptAdapter",
        "xfail_reason": "Standalone adapter, no KnowledgeMoundAdapter inheritance",
    },
    {
        "module": "aragora.knowledge.mound.adapters.culture_adapter",
        "class_name": "CultureAdapter",
        "xfail_reason": "Standalone adapter, no KnowledgeMoundAdapter inheritance",
    },
    {
        "module": "aragora.knowledge.mound.adapters.rlm_adapter",
        "class_name": "RlmAdapter",
        "xfail_reason": "Standalone adapter, no KnowledgeMoundAdapter inheritance",
    },
    {
        "module": "aragora.knowledge.mound.adapters.provenance_adapter",
        "class_name": "ProvenanceAdapter",
        "xfail_reason": "Standalone adapter, no KnowledgeMoundAdapter inheritance",
    },
]


def _load_adapter_class(module_path: str, class_name: str) -> type | None:
    """Dynamically load an adapter class."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Could not import {module_path}.{class_name}: {e}")
        return None


def _load_base_class() -> type | None:
    """Load the KnowledgeMoundAdapter base class."""
    try:
        from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter

        return KnowledgeMoundAdapter
    except ImportError as e:
        pytest.skip(f"Could not import KnowledgeMoundAdapter: {e}")
        return None


def _get_adapter_id(spec: dict[str, Any]) -> str:
    """Generate a test ID from adapter spec."""
    return spec["class_name"]


class TestAdapterCompliance:
    """Test that all adapters comply with the unified adapter pattern."""

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param(
                spec,
                marks=[pytest.mark.xfail(reason=spec["xfail_reason"])]
                if spec["xfail_reason"]
                else [],
                id=_get_adapter_id(spec),
            )
            for spec in ADAPTER_SPECS
        ],
    )
    def test_inherits_from_knowledge_mound_adapter(self, spec: dict[str, Any]) -> None:
        """Adapter should inherit from KnowledgeMoundAdapter."""
        base_class = _load_base_class()
        adapter_class = _load_adapter_class(spec["module"], spec["class_name"])

        if base_class is None or adapter_class is None:
            pytest.skip("Could not load required classes")

        assert issubclass(adapter_class, base_class), (
            f"{spec['class_name']} should inherit from KnowledgeMoundAdapter"
        )

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param(
                spec,
                marks=[pytest.mark.xfail(reason=spec["xfail_reason"])]
                if spec["xfail_reason"]
                else [],
                id=_get_adapter_id(spec),
            )
            for spec in ADAPTER_SPECS
        ],
    )
    def test_has_unique_adapter_name(self, spec: dict[str, Any]) -> None:
        """Adapter should have a unique adapter_name that is not 'base'."""
        adapter_class = _load_adapter_class(spec["module"], spec["class_name"])

        if adapter_class is None:
            pytest.skip("Could not load adapter class")

        # Check adapter_name is defined
        assert hasattr(adapter_class, "adapter_name"), (
            f"{spec['class_name']} should have adapter_name attribute"
        )

        adapter_name = getattr(adapter_class, "adapter_name", None)
        assert adapter_name is not None, f"{spec['class_name']}.adapter_name should not be None"
        assert adapter_name != "base", f"{spec['class_name']}.adapter_name should not be 'base'"
        assert isinstance(adapter_name, str), (
            f"{spec['class_name']}.adapter_name should be a string"
        )
        assert len(adapter_name) > 0, f"{spec['class_name']}.adapter_name should not be empty"

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param(
                spec,
                marks=[pytest.mark.xfail(reason=spec["xfail_reason"])]
                if spec["xfail_reason"]
                else [],
                id=_get_adapter_id(spec),
            )
            for spec in ADAPTER_SPECS
        ],
    )
    def test_has_emit_event_method(self, spec: dict[str, Any]) -> None:
        """Adapter should have _emit_event method (inherited or defined)."""
        adapter_class = _load_adapter_class(spec["module"], spec["class_name"])

        if adapter_class is None:
            pytest.skip("Could not load adapter class")

        assert hasattr(adapter_class, "_emit_event"), (
            f"{spec['class_name']} should have _emit_event method"
        )

        # Verify it's callable
        emit_event = getattr(adapter_class, "_emit_event")
        assert callable(emit_event), f"{spec['class_name']}._emit_event should be callable"

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param(
                spec,
                marks=[pytest.mark.xfail(reason=spec["xfail_reason"])]
                if spec["xfail_reason"]
                else [],
                id=_get_adapter_id(spec),
            )
            for spec in ADAPTER_SPECS
        ],
    )
    def test_has_record_metric_method(self, spec: dict[str, Any]) -> None:
        """Adapter should have _record_metric method (inherited or defined)."""
        adapter_class = _load_adapter_class(spec["module"], spec["class_name"])

        if adapter_class is None:
            pytest.skip("Could not load adapter class")

        assert hasattr(adapter_class, "_record_metric"), (
            f"{spec['class_name']} should have _record_metric method"
        )

        # Verify it's callable
        record_metric = getattr(adapter_class, "_record_metric")
        assert callable(record_metric), f"{spec['class_name']}._record_metric should be callable"

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param(
                spec,
                marks=[pytest.mark.xfail(reason=spec["xfail_reason"])]
                if spec["xfail_reason"]
                else [],
                id=_get_adapter_id(spec),
            )
            for spec in ADAPTER_SPECS
        ],
    )
    def test_has_health_check_method(self, spec: dict[str, Any]) -> None:
        """Adapter should have health_check method (inherited or defined)."""
        adapter_class = _load_adapter_class(spec["module"], spec["class_name"])

        if adapter_class is None:
            pytest.skip("Could not load adapter class")

        assert hasattr(adapter_class, "health_check"), (
            f"{spec['class_name']} should have health_check method"
        )

        # Verify it's callable
        health_check = getattr(adapter_class, "health_check")
        assert callable(health_check), f"{spec['class_name']}.health_check should be callable"


class TestAdapterNameUniqueness:
    """Test that adapter_name values are unique across all adapters."""

    def test_no_duplicate_adapter_names(self) -> None:
        """All adapters should have unique adapter_name values."""
        adapter_names: dict[str, str] = {}  # adapter_name -> class_name

        for spec in ADAPTER_SPECS:
            adapter_class = _load_adapter_class(spec["module"], spec["class_name"])
            if adapter_class is None:
                continue

            adapter_name = getattr(adapter_class, "adapter_name", None)
            if adapter_name is None or adapter_name == "base":
                continue  # Skip adapters without proper adapter_name (xfail tests cover this)

            if adapter_name in adapter_names:
                # Only fail if both adapters are expected to pass compliance
                # (skip this check for xfail adapters)
                if spec["xfail_reason"] is None:
                    pytest.fail(
                        f"Duplicate adapter_name '{adapter_name}' found in "
                        f"{spec['class_name']} and {adapter_names[adapter_name]}"
                    )
            else:
                adapter_names[adapter_name] = spec["class_name"]

        # Report how many unique adapter names we found
        # (useful for tracking migration progress)
        assert len(adapter_names) > 0, "Should find at least one valid adapter_name"


class TestComplianceSummary:
    """Summary tests for tracking migration progress."""

    def test_count_compliant_adapters(self) -> None:
        """Count how many adapters are fully compliant."""
        base_class = _load_base_class()
        if base_class is None:
            pytest.skip("Could not load KnowledgeMoundAdapter")

        compliant_count = 0
        total_count = len(ADAPTER_SPECS)

        for spec in ADAPTER_SPECS:
            adapter_class = _load_adapter_class(spec["module"], spec["class_name"])
            if adapter_class is None:
                continue

            # Check all compliance criteria
            is_subclass = issubclass(adapter_class, base_class)
            has_valid_name = (
                hasattr(adapter_class, "adapter_name") and adapter_class.adapter_name != "base"
            )
            has_emit_event = hasattr(adapter_class, "_emit_event")
            has_record_metric = hasattr(adapter_class, "_record_metric")
            has_health_check = hasattr(adapter_class, "health_check")

            if all(
                [is_subclass, has_valid_name, has_emit_event, has_record_metric, has_health_check]
            ):
                compliant_count += 1

        # Report progress
        compliance_pct = (compliant_count / total_count) * 100 if total_count > 0 else 0

        # This test passes as long as we have at least some compliant adapters
        # (Pattern A adapters should pass)
        assert compliant_count >= 8, (
            f"Expected at least 8 compliant adapters (Pattern A), found {compliant_count}"
        )

        # Log progress for visibility
        print(f"\nAdapter Compliance: {compliant_count}/{total_count} ({compliance_pct:.1f}%)")
