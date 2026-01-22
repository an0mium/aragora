"""Tests for marketplace template registry."""

import json
import pytest
import tempfile
from pathlib import Path

from aragora.marketplace.models import (
    AgentTemplate,
    DebateTemplate,
    TemplateMetadata,
    TemplateRating,
    TemplateCategory,
)
from aragora.marketplace.registry import TemplateRegistry


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def registry(self, temp_db):
        """Create a registry with temporary database."""
        return TemplateRegistry(db_path=temp_db)

    @pytest.fixture
    def sample_agent_template(self):
        """Create a sample agent template."""
        return AgentTemplate(
            metadata=TemplateMetadata(
                id="custom-agent-1",
                name="Custom Agent",
                description="A custom agent template",
                version="1.0.0",
                author="test_user",
                category=TemplateCategory.CODING,
                tags=["custom", "testing"],
            ),
            agent_type="claude",
            system_prompt="You are a custom agent for testing.",
            capabilities=["testing"],
        )

    @pytest.fixture
    def sample_debate_template(self):
        """Create a sample debate template."""
        return DebateTemplate(
            metadata=TemplateMetadata(
                id="custom-debate-1",
                name="Custom Debate",
                description="A custom debate format",
                version="1.0.0",
                author="test_user",
                category=TemplateCategory.DEBATE,
            ),
            task_template="Debate: {topic}",
            agent_roles=[{"role": "speaker", "team": "neutral"}],
            protocol={"rounds": 2},
        )

    def test_registry_initialization(self, registry):
        """Test registry initializes with built-in templates."""
        # Built-in templates should be loaded
        templates = registry.search()
        assert len(templates) > 0

    def test_register_template(self, registry, sample_agent_template):
        """Test registering a new template."""
        template_id = registry.register(sample_agent_template)
        assert template_id == "custom-agent-1"

        # Verify it can be retrieved
        retrieved = registry.get(template_id)
        assert retrieved is not None
        assert retrieved.metadata.name == "Custom Agent"

    def test_get_nonexistent_template(self, registry):
        """Test getting a template that doesn't exist."""
        result = registry.get("nonexistent-id")
        assert result is None

    def test_search_by_query(self, registry, sample_agent_template):
        """Test searching templates by text query."""
        registry.register(sample_agent_template)

        # Search by name
        results = registry.search(query="Custom")
        assert any(t.metadata.id == "custom-agent-1" for t in results)

        # Search by description
        results = registry.search(query="custom agent template")
        assert any(t.metadata.id == "custom-agent-1" for t in results)

    def test_search_by_category(self, registry, sample_agent_template):
        """Test searching templates by category."""
        registry.register(sample_agent_template)

        results = registry.search(category=TemplateCategory.CODING)
        assert any(t.metadata.id == "custom-agent-1" for t in results)

        results = registry.search(category=TemplateCategory.CREATIVE)
        assert not any(t.metadata.id == "custom-agent-1" for t in results)

    def test_search_by_type(self, registry, sample_agent_template, sample_debate_template):
        """Test searching by template type."""
        registry.register(sample_agent_template)
        registry.register(sample_debate_template)

        agent_results = registry.search(template_type="AgentTemplate")
        debate_results = registry.search(template_type="DebateTemplate")

        assert any(t.metadata.id == "custom-agent-1" for t in agent_results)
        assert any(t.metadata.id == "custom-debate-1" for t in debate_results)

    def test_search_by_tags(self, registry, sample_agent_template):
        """Test searching templates by tags."""
        registry.register(sample_agent_template)

        results = registry.search(tags=["testing"])
        assert any(t.metadata.id == "custom-agent-1" for t in results)

        results = registry.search(tags=["nonexistent"])
        assert not any(t.metadata.id == "custom-agent-1" for t in results)

    def test_search_pagination(self, registry):
        """Test search pagination."""
        # Built-in templates provide enough for pagination test
        all_results = registry.search(limit=100)
        first_page = registry.search(limit=2, offset=0)
        second_page = registry.search(limit=2, offset=2)

        assert len(first_page) <= 2
        assert len(second_page) <= 2
        if len(all_results) > 4:
            assert first_page[0].metadata.id != second_page[0].metadata.id

    def test_list_categories(self, registry):
        """Test listing categories with counts."""
        categories = registry.list_categories()
        assert len(categories) > 0
        assert all("category" in c and "count" in c for c in categories)

    def test_rate_template(self, registry, sample_agent_template):
        """Test rating a template."""
        registry.register(sample_agent_template)

        rating = TemplateRating(
            user_id="user-1",
            template_id="custom-agent-1",
            score=5,
            review="Excellent template!",
        )
        registry.rate(rating)

        ratings = registry.get_ratings("custom-agent-1")
        assert len(ratings) == 1
        assert ratings[0].score == 5
        assert ratings[0].review == "Excellent template!"

    def test_update_rating(self, registry, sample_agent_template):
        """Test updating an existing rating."""
        registry.register(sample_agent_template)

        # First rating
        rating1 = TemplateRating(
            user_id="user-1",
            template_id="custom-agent-1",
            score=3,
        )
        registry.rate(rating1)

        # Update rating
        rating2 = TemplateRating(
            user_id="user-1",
            template_id="custom-agent-1",
            score=5,
            review="Changed my mind, it's great!",
        )
        registry.rate(rating2)

        ratings = registry.get_ratings("custom-agent-1")
        assert len(ratings) == 1  # Should replace, not add
        assert ratings[0].score == 5

    def test_average_rating(self, registry, sample_agent_template):
        """Test calculating average rating."""
        registry.register(sample_agent_template)

        registry.rate(TemplateRating(user_id="user-1", template_id="custom-agent-1", score=5))
        registry.rate(TemplateRating(user_id="user-2", template_id="custom-agent-1", score=3))

        avg = registry.get_average_rating("custom-agent-1")
        assert avg == 4.0

    def test_increment_downloads(self, registry, sample_agent_template):
        """Test incrementing download count."""
        registry.register(sample_agent_template)

        registry.increment_downloads("custom-agent-1")
        registry.increment_downloads("custom-agent-1")

        template = registry.get("custom-agent-1")
        assert template.metadata.downloads == 2

    def test_star_template(self, registry, sample_agent_template):
        """Test starring a template."""
        registry.register(sample_agent_template)

        registry.star("custom-agent-1")
        registry.star("custom-agent-1")

        template = registry.get("custom-agent-1")
        assert template.metadata.stars == 2

    def test_delete_template(self, registry, sample_agent_template):
        """Test deleting a custom template."""
        registry.register(sample_agent_template)

        result = registry.delete("custom-agent-1")
        assert result is True
        assert registry.get("custom-agent-1") is None

    def test_cannot_delete_builtin(self, registry):
        """Test that built-in templates cannot be deleted."""
        result = registry.delete("devil-advocate")
        assert result is False
        assert registry.get("devil-advocate") is not None

    def test_export_template(self, registry, sample_agent_template):
        """Test exporting template as JSON."""
        registry.register(sample_agent_template)

        json_str = registry.export_template("custom-agent-1")
        assert json_str is not None

        data = json.loads(json_str)
        assert data["metadata"]["id"] == "custom-agent-1"
        assert data["agent_type"] == "claude"

    def test_import_agent_template(self, registry, sample_agent_template):
        """Test importing an agent template from JSON."""
        json_str = json.dumps(sample_agent_template.to_dict())
        sample_agent_template.metadata.id = "imported-agent"

        json_str = json.dumps(
            {
                **sample_agent_template.to_dict(),
                "metadata": {
                    **sample_agent_template.metadata.to_dict(),
                    "id": "imported-agent",
                },
            }
        )

        template_id = registry.import_template(json_str)
        assert template_id == "imported-agent"

        imported = registry.get("imported-agent")
        assert imported is not None
        assert imported.agent_type == "claude"

    def test_import_debate_template(self, registry, sample_debate_template):
        """Test importing a debate template from JSON."""
        json_str = json.dumps(
            {
                **sample_debate_template.to_dict(),
                "metadata": {
                    **sample_debate_template.metadata.to_dict(),
                    "id": "imported-debate",
                },
            }
        )

        template_id = registry.import_template(json_str)
        assert template_id == "imported-debate"

        imported = registry.get("imported-debate")
        assert imported is not None
        assert isinstance(imported, DebateTemplate)
