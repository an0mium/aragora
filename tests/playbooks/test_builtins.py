"""
Tests for built-in playbook YAML files.

Validates that all built-in playbooks:
- Parse successfully from YAML
- Have required fields
- Have valid structure (steps, gates, etc.)
- Cover expected categories
"""

import pytest
from pathlib import Path

from aragora.playbooks.models import Playbook
from aragora.playbooks.registry import PlaybookRegistry, BUILTIN_DIR


@pytest.fixture
def all_builtins():
    """Load all builtin playbooks."""
    registry = PlaybookRegistry()
    registry._ensure_builtins()
    return registry.list()


class TestBuiltinStructure:
    """Tests validating built-in playbook structure."""

    def test_all_have_id(self, all_builtins):
        for pb in all_builtins:
            assert pb.id, f"Playbook missing id: {pb.name}"

    def test_all_have_name(self, all_builtins):
        for pb in all_builtins:
            assert pb.name, f"Playbook missing name: {pb.id}"

    def test_all_have_description(self, all_builtins):
        for pb in all_builtins:
            assert pb.description, f"Playbook missing description: {pb.id}"

    def test_all_have_category(self, all_builtins):
        for pb in all_builtins:
            assert pb.category, f"Playbook missing category: {pb.id}"

    def test_all_have_steps(self, all_builtins):
        for pb in all_builtins:
            assert len(pb.steps) > 0, f"Playbook has no steps: {pb.id}"

    def test_all_steps_have_action(self, all_builtins):
        for pb in all_builtins:
            for step in pb.steps:
                assert step.action, f"Step missing action in playbook {pb.id}: {step.name}"

    def test_valid_agent_counts(self, all_builtins):
        for pb in all_builtins:
            assert pb.min_agents >= 1, f"min_agents < 1 in {pb.id}"
            assert pb.max_agents >= pb.min_agents, f"max_agents < min_agents in {pb.id}"

    def test_valid_consensus_threshold(self, all_builtins):
        for pb in all_builtins:
            assert 0.0 < pb.consensus_threshold <= 1.0, (
                f"Invalid consensus_threshold in {pb.id}: {pb.consensus_threshold}"
            )

    def test_all_have_tags(self, all_builtins):
        for pb in all_builtins:
            assert len(pb.tags) > 0, f"Playbook has no tags: {pb.id}"

    def test_unique_ids(self, all_builtins):
        ids = [pb.id for pb in all_builtins]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"


class TestBuiltinCoverage:
    """Tests ensuring built-in playbooks cover key categories."""

    def test_healthcare_playbook_exists(self, all_builtins):
        categories = {pb.category for pb in all_builtins}
        assert "healthcare" in categories

    def test_finance_playbook_exists(self, all_builtins):
        categories = {pb.category for pb in all_builtins}
        assert "finance" in categories

    def test_engineering_playbook_exists(self, all_builtins):
        categories = {pb.category for pb in all_builtins}
        assert "engineering" in categories

    def test_compliance_playbook_exists(self, all_builtins):
        categories = {pb.category for pb in all_builtins}
        assert "compliance" in categories

    def test_minimum_seven_builtins(self, all_builtins):
        assert len(all_builtins) >= 7


class TestBuiltinYAMLFiles:
    """Tests for YAML file integrity."""

    def test_builtin_dir_exists(self):
        assert BUILTIN_DIR.exists()

    def test_yaml_files_exist(self):
        yaml_files = list(BUILTIN_DIR.glob("*.yaml"))
        assert len(yaml_files) >= 7

    def test_all_yaml_files_parseable(self):
        import yaml

        for yaml_file in BUILTIN_DIR.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"Invalid YAML structure in {yaml_file.name}"
            assert "name" in data, f"Missing 'name' in {yaml_file.name}"
