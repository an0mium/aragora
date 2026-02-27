"""Tests for Obsidian adapter factory registration."""

from __future__ import annotations

from aragora.knowledge.mound.adapters.factory import _ADAPTER_DEFS, ADAPTER_SPECS


def _adapter_names() -> list[str]:
    """Return the list of adapter names from _ADAPTER_DEFS."""
    return [spec_kwargs["name"] for _, _, spec_kwargs in _ADAPTER_DEFS]


def test_obsidian_in_adapter_definitions() -> None:
    """Verify 'obsidian' is present in the _ADAPTER_DEFS name list."""
    names = _adapter_names()
    assert "obsidian" in names, f"'obsidian' not found in adapter definitions: {names}"

    # Also check it was registered in ADAPTER_SPECS (populated at import time)
    assert "obsidian" in ADAPTER_SPECS, "obsidian not in ADAPTER_SPECS after init"


def test_obsidian_adapter_has_bidirectional_methods() -> None:
    """Verify the obsidian adapter spec has both forward and reverse methods set."""
    # Find the obsidian entry in _ADAPTER_DEFS
    obsidian_defs = [
        (module, cls, kwargs)
        for module, cls, kwargs in _ADAPTER_DEFS
        if kwargs.get("name") == "obsidian"
    ]
    assert len(obsidian_defs) == 1, f"Expected 1 obsidian def, found {len(obsidian_defs)}"

    _, _, kwargs = obsidian_defs[0]
    assert kwargs["forward_method"] == "sync_to_km", (
        f"Expected forward_method='sync_to_km', got '{kwargs['forward_method']}'"
    )
    assert kwargs["reverse_method"] == "sync_from_km", (
        f"Expected reverse_method='sync_from_km', got '{kwargs['reverse_method']}'"
    )

    # Also verify through the initialized ADAPTER_SPECS
    spec = ADAPTER_SPECS["obsidian"]
    assert spec.forward_method == "sync_to_km"
    assert spec.reverse_method == "sync_from_km"
