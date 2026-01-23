"""
Tests for deliberation templates module.
"""

import pytest
from pathlib import Path
import tempfile

from aragora.deliberation.templates import (
    DeliberationTemplate,
    TemplateRegistry,
    OutputFormat,
    TeamStrategy,
    TemplateCategory,
    get_template,
    list_templates,
    register_template,
    BUILTIN_TEMPLATES,
)


class TestDeliberationTemplate:
    """Tests for DeliberationTemplate dataclass."""

    def test_create_minimal(self):
        """Test creating template with minimal fields."""
        template = DeliberationTemplate(
            name="test",
            description="Test template",
        )
        assert template.name == "test"
        assert template.description == "Test template"
        assert template.category == TemplateCategory.GENERAL
        assert template.default_agents == []
        assert template.consensus_threshold == 0.7

    def test_create_full(self):
        """Test creating template with all fields."""
        template = DeliberationTemplate(
            name="full_test",
            description="Full test template",
            category=TemplateCategory.CODE,
            default_agents=["agent1", "agent2"],
            team_strategy=TeamStrategy.DIVERSE,
            default_knowledge_sources=["github:pr"],
            output_format=OutputFormat.GITHUB_REVIEW,
            consensus_threshold=0.8,
            max_rounds=3,
            personas=["security", "performance"],
            tags=["test", "code"],
        )
        assert template.name == "full_test"
        assert template.category == TemplateCategory.CODE
        assert template.team_strategy == TeamStrategy.DIVERSE
        assert template.output_format == OutputFormat.GITHUB_REVIEW
        assert len(template.default_agents) == 2
        assert len(template.personas) == 2

    def test_to_dict(self):
        """Test converting template to dictionary."""
        template = DeliberationTemplate(
            name="test",
            description="Test",
            category=TemplateCategory.LEGAL,
            default_agents=["claude"],
            output_format=OutputFormat.DECISION_RECEIPT,
        )
        data = template.to_dict()
        assert data["name"] == "test"
        assert data["category"] == "legal"
        assert data["output_format"] == "decision_receipt"
        assert data["default_agents"] == ["claude"]

    def test_from_dict(self):
        """Test creating template from dictionary."""
        data = {
            "name": "from_dict_test",
            "description": "Created from dict",
            "category": "finance",
            "team_strategy": "fast",
            "output_format": "summary",
            "consensus_threshold": 0.5,
            "max_rounds": 2,
            "tags": ["finance", "audit"],
        }
        template = DeliberationTemplate.from_dict(data)
        assert template.name == "from_dict_test"
        assert template.category == TemplateCategory.FINANCE
        assert template.team_strategy == TeamStrategy.FAST
        assert template.output_format == OutputFormat.SUMMARY
        assert template.consensus_threshold == 0.5
        assert "finance" in template.tags

    def test_from_dict_invalid_enums(self):
        """Test that invalid enum values fall back to defaults."""
        data = {
            "name": "test",
            "description": "Test",
            "category": "invalid_category",
            "team_strategy": "invalid_strategy",
            "output_format": "invalid_format",
        }
        template = DeliberationTemplate.from_dict(data)
        assert template.category == TemplateCategory.GENERAL
        assert template.team_strategy == TeamStrategy.BEST_FOR_DOMAIN
        assert template.output_format == OutputFormat.STANDARD

    def test_merge_with_request(self):
        """Test merging template with request data."""
        template = DeliberationTemplate(
            name="test",
            description="Test",
            default_agents=["default1", "default2"],
            consensus_threshold=0.7,
            max_rounds=5,
        )

        # Request overrides some values
        request = {
            "agents": ["custom1"],
            "max_rounds": 3,
            "custom_field": "custom_value",
        }

        merged = template.merge_with_request(request)
        assert merged["agents"] == ["custom1"]  # Request override
        assert merged["consensus_threshold"] == 0.7  # Template default
        assert merged["max_rounds"] == 3  # Request override
        assert merged["custom_field"] == "custom_value"  # Passed through


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving a template."""
        registry = TemplateRegistry()
        template = DeliberationTemplate(
            name="registry_test",
            description="Registry test",
        )
        registry.register(template)

        retrieved = registry.get("registry_test")
        assert retrieved is not None
        assert retrieved.name == "registry_test"

    def test_get_nonexistent(self):
        """Test getting a template that doesn't exist."""
        registry = TemplateRegistry()
        assert registry.get("nonexistent") is None

    def test_unregister(self):
        """Test unregistering a template."""
        registry = TemplateRegistry()
        template = DeliberationTemplate(name="to_remove", description="Test")
        registry.register(template)

        assert registry.unregister("to_remove") is True
        assert registry.get("to_remove") is None
        assert registry.unregister("to_remove") is False

    def test_list_all(self):
        """Test listing all templates."""
        registry = TemplateRegistry()
        registry.register(DeliberationTemplate(name="a", description="A"))
        registry.register(DeliberationTemplate(name="b", description="B"))

        templates = registry.list()
        names = [t.name for t in templates]
        assert "a" in names
        assert "b" in names

    def test_list_by_category(self):
        """Test filtering templates by category."""
        registry = TemplateRegistry()
        registry.register(
            DeliberationTemplate(name="code1", description="Code", category=TemplateCategory.CODE)
        )
        registry.register(
            DeliberationTemplate(
                name="legal1", description="Legal", category=TemplateCategory.LEGAL
            )
        )

        code_templates = registry.list(category=TemplateCategory.CODE)
        assert all(t.category == TemplateCategory.CODE for t in code_templates)

    def test_list_by_tags(self):
        """Test filtering templates by tags."""
        registry = TemplateRegistry()
        registry.register(
            DeliberationTemplate(name="tagged1", description="Tagged", tags=["security", "code"])
        )
        registry.register(
            DeliberationTemplate(name="tagged2", description="Tagged", tags=["legal"])
        )

        security_templates = registry.list(tags=["security"])
        names = [t.name for t in security_templates]
        assert "tagged1" in names
        assert "tagged2" not in names

    def test_list_with_search(self):
        """Test searching templates."""
        registry = TemplateRegistry()
        registry.register(
            DeliberationTemplate(name="code_review", description="Review code for quality")
        )
        registry.register(
            DeliberationTemplate(name="contract_review", description="Review contracts")
        )

        results = registry.list(search="code")
        names = [t.name for t in results]
        assert "code_review" in names

    def test_list_pagination(self):
        """Test pagination in listing."""
        registry = TemplateRegistry()
        for i in range(10):
            registry.register(
                DeliberationTemplate(name=f"template_{i:02d}", description=f"Template {i}")
            )

        page1 = registry.list(limit=3, offset=0)
        page2 = registry.list(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].name != page2[0].name

    def test_categories(self):
        """Test getting category counts."""
        registry = TemplateRegistry()

        # Force loading of any built-in templates first
        _ = list(registry.list())

        # Get baseline counts (includes any built-in templates)
        initial_counts = registry.categories()
        initial_code = initial_counts.get("code", 0)
        initial_legal = initial_counts.get("legal", 0)

        # Register new templates
        registry.register(
            DeliberationTemplate(
                name="code1_test", description="Code", category=TemplateCategory.CODE
            )
        )
        registry.register(
            DeliberationTemplate(
                name="code2_test", description="Code", category=TemplateCategory.CODE
            )
        )
        registry.register(
            DeliberationTemplate(
                name="legal1_test", description="Legal", category=TemplateCategory.LEGAL
            )
        )

        counts = registry.categories()
        # Verify new templates were added
        assert counts.get("code", 0) == initial_code + 2
        assert counts.get("legal", 0) == initial_legal + 1


class TestBuiltinTemplates:
    """Tests for built-in templates."""

    def test_builtin_templates_exist(self):
        """Test that expected built-in templates exist."""
        expected = [
            "code_review",
            "security_audit",
            "architecture_decision",
            "contract_review",
            "compliance_check",
            "quick_decision",
            "hipaa_compliance",
            "financial_audit",
        ]
        for name in expected:
            assert name in BUILTIN_TEMPLATES, f"Missing template: {name}"

    def test_code_review_template(self):
        """Test code review template configuration."""
        template = BUILTIN_TEMPLATES["code_review"]
        assert template.category == TemplateCategory.CODE
        assert template.output_format == OutputFormat.GITHUB_REVIEW
        assert "anthropic-api" in template.default_agents
        assert "security" in template.personas

    def test_contract_review_template(self):
        """Test contract review template configuration."""
        template = BUILTIN_TEMPLATES["contract_review"]
        assert template.category == TemplateCategory.LEGAL
        assert template.output_format == OutputFormat.DECISION_RECEIPT
        assert template.consensus_threshold >= 0.8  # High threshold for legal

    def test_compliance_templates(self):
        """Test compliance templates have high thresholds."""
        compliance_templates = [
            "compliance_check",
            "hipaa_compliance",
            "soc2_audit",
            "gdpr_assessment",
        ]
        for name in compliance_templates:
            if name in BUILTIN_TEMPLATES:
                template = BUILTIN_TEMPLATES[name]
                assert template.consensus_threshold >= 0.8, f"{name} should have high threshold"

    def test_quick_decision_template(self):
        """Test quick decision template is fast."""
        template = BUILTIN_TEMPLATES["quick_decision"]
        assert template.max_rounds <= 3
        assert template.consensus_threshold <= 0.6
        assert len(template.default_agents) <= 3


class TestGlobalFunctions:
    """Tests for global registry functions."""

    def test_get_template(self):
        """Test getting template via global function."""
        template = get_template("code_review")
        assert template is not None
        assert template.name == "code_review"

    def test_list_templates(self):
        """Test listing templates via global function."""
        templates = list_templates()
        assert len(templates) > 0
        assert any(t.name == "code_review" for t in templates)

    def test_list_templates_with_category(self):
        """Test listing templates by category."""
        templates = list_templates(category="code")
        assert all(t.category == TemplateCategory.CODE for t in templates)

    def test_register_custom_template(self):
        """Test registering a custom template."""
        custom = DeliberationTemplate(
            name="custom_test_template",
            description="Custom test",
            tags=["test"],
        )
        register_template(custom)

        retrieved = get_template("custom_test_template")
        assert retrieved is not None
        assert retrieved.name == "custom_test_template"


class TestYAMLLoading:
    """Tests for YAML template loading."""

    def test_load_from_yaml(self):
        """Test loading templates from YAML file."""
        yaml_content = """
templates:
  - name: yaml_test
    description: Template from YAML
    category: general
    default_agents:
      - anthropic-api
    consensus_threshold: 0.6
    tags:
      - yaml
      - test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            registry = TemplateRegistry()
            count = registry.load_from_yaml(yaml_path)
            assert count == 1

            template = registry.get("yaml_test")
            assert template is not None
            assert template.description == "Template from YAML"
            assert "yaml" in template.tags
        finally:
            yaml_path.unlink()

    def test_load_from_invalid_yaml(self):
        """Test handling invalid YAML gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            yaml_path = Path(f.name)

        try:
            registry = TemplateRegistry()
            count = registry.load_from_yaml(yaml_path)
            assert count == 0  # Should handle error gracefully
        finally:
            yaml_path.unlink()
