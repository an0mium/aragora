"""
Tests for PlaybookRegistry — discovery, registration, and YAML loading.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from aragora.playbooks.models import Playbook
from aragora.playbooks.registry import PlaybookRegistry, get_playbook_registry


@pytest.fixture
def registry():
    """Create a fresh registry (no builtins)."""
    r = PlaybookRegistry()
    r._loaded_builtins = True  # Skip auto-loading
    return r


class TestRegistration:
    """Tests for playbook registration."""

    def test_register_and_get(self, registry):
        pb = Playbook(id="test", name="Test", description="", category="general")
        registry.register(pb)
        assert registry.get("test") is pb

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_register_overwrite(self, registry):
        pb1 = Playbook(id="x", name="First", description="", category="general")
        pb2 = Playbook(id="x", name="Second", description="", category="general")
        registry.register(pb1)
        registry.register(pb2)
        assert registry.get("x").name == "Second"

    def test_count(self, registry):
        assert registry.count == 0
        registry.register(Playbook(id="a", name="A", description="", category="general"))
        registry.register(Playbook(id="b", name="B", description="", category="finance"))
        assert registry.count == 2


class TestListing:
    """Tests for listing playbooks."""

    def test_list_all(self, registry):
        registry.register(Playbook(id="a", name="A", description="", category="general"))
        registry.register(Playbook(id="b", name="B", description="", category="finance"))
        assert len(registry.list()) == 2

    def test_list_by_category(self, registry):
        registry.register(Playbook(id="a", name="A", description="", category="general"))
        registry.register(Playbook(id="b", name="B", description="", category="finance"))
        registry.register(Playbook(id="c", name="C", description="", category="finance"))
        results = registry.list(category="finance")
        assert len(results) == 2

    def test_list_by_tags(self, registry):
        registry.register(Playbook(id="a", name="A", description="", category="general", tags=["hipaa"]))
        registry.register(Playbook(id="b", name="B", description="", category="general", tags=["sox"]))
        results = registry.list(tags=["hipaa"])
        assert len(results) == 1
        assert results[0].id == "a"

    def test_list_sorted_by_name(self, registry):
        registry.register(Playbook(id="z", name="Zebra", description="", category="general"))
        registry.register(Playbook(id="a", name="Alpha", description="", category="general"))
        results = registry.list()
        assert results[0].name == "Alpha"
        assert results[1].name == "Zebra"


class TestYAMLLoading:
    """Tests for YAML file loading."""

    def test_load_from_yaml(self, registry):
        yaml_content = """
id: test_yaml
name: YAML Test
description: Loaded from YAML
category: general
min_agents: 2
steps:
  - name: step1
    action: debate
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                pb = registry.from_yaml(f.name)
                assert pb.id == "test_yaml"
                assert pb.name == "YAML Test"
                assert len(pb.steps) == 1
                assert registry.get("test_yaml") is not None
            finally:
                os.unlink(f.name)

    def test_load_from_yaml_derives_id(self, registry):
        yaml_content = """
name: No ID Playbook
description: ID from filename
category: general
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="my_playbook_", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                pb = registry.from_yaml(f.name)
                assert pb.id == Path(f.name).stem
            finally:
                os.unlink(f.name)

    def test_load_from_yaml_not_found(self, registry):
        with pytest.raises(FileNotFoundError):
            registry.from_yaml("/nonexistent/path.yaml")

    def test_load_from_yaml_invalid(self, registry):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- just\n- a\n- list\n")
            f.flush()
            try:
                with pytest.raises(ValueError, match="expected mapping"):
                    registry.from_yaml(f.name)
            finally:
                os.unlink(f.name)


class TestBuiltinLoading:
    """Tests for built-in playbook auto-discovery."""

    def test_builtins_load_on_first_access(self):
        r = PlaybookRegistry()
        # Access triggers loading
        playbooks = r.list()
        assert len(playbooks) >= 7  # We have 7 builtin YAML files

    def test_builtin_hipaa(self):
        r = PlaybookRegistry()
        pb = r.get("hipaa_vendor_assessment")
        assert pb is not None
        assert pb.category == "healthcare"
        assert "hipaa" in pb.tags

    def test_builtin_sox(self):
        r = PlaybookRegistry()
        pb = r.get("sox_financial_decision")
        assert pb is not None
        assert pb.category == "finance"

    def test_builtin_eu_ai_act(self):
        r = PlaybookRegistry()
        pb = r.get("eu_ai_act_conformity")
        assert pb is not None
        assert pb.category == "compliance"

    def test_builtin_hiring(self):
        r = PlaybookRegistry()
        pb = r.get("hiring_committee")
        assert pb is not None
        assert pb.category == "general"

    def test_builtin_architecture(self):
        r = PlaybookRegistry()
        pb = r.get("architecture_review")
        assert pb is not None
        assert pb.category == "engineering"

    def test_builtin_pricing(self):
        r = PlaybookRegistry()
        pb = r.get("pricing_change")
        assert pb is not None

    def test_builtin_postmortem(self):
        r = PlaybookRegistry()
        pb = r.get("incident_postmortem")
        assert pb is not None
        assert "blameless" in pb.tags


class TestSingleton:
    """Tests for singleton management."""

    def test_get_playbook_registry_returns_instance(self):
        import aragora.playbooks.registry as mod

        mod._registry_singleton = None
        r = get_playbook_registry()
        assert isinstance(r, PlaybookRegistry)
        mod._registry_singleton = None

    def test_singleton_returns_same(self):
        import aragora.playbooks.registry as mod

        mod._registry_singleton = None
        r1 = get_playbook_registry()
        r2 = get_playbook_registry()
        assert r1 is r2
        mod._registry_singleton = None
