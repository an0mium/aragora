"""End-to-end validation tests for marketplace templates.

Validates that all built-in templates are well-formed, can be serialized
and deserialized, and produce valid configurations for their target systems.
"""

import json
import tempfile
from pathlib import Path

import pytest

from aragora.marketplace.models import (
    AgentTemplate,
    DebateTemplate,
    WorkflowTemplate,
    TemplateCategory,
    TemplateMetadata,
    TemplateRating,
    BUILTIN_AGENT_TEMPLATES,
    BUILTIN_DEBATE_TEMPLATES,
)
from aragora.marketplace.registry import TemplateRegistry


class TestBuiltinTemplateIntegrity:
    """Validate all built-in templates are well-formed."""

    @pytest.fixture
    def registry(self, tmp_path):
        return TemplateRegistry(db_path=tmp_path / "test.db")

    @pytest.mark.parametrize(
        "template",
        BUILTIN_AGENT_TEMPLATES,
        ids=[t.metadata.id for t in BUILTIN_AGENT_TEMPLATES],
    )
    def test_agent_template_schema(self, template):
        """Each agent template has all required fields."""
        assert template.metadata.id, "Template must have an ID"
        assert template.metadata.name, "Template must have a name"
        assert template.metadata.description, "Template must have a description"
        assert template.metadata.version, "Template must have a version"
        assert template.metadata.author, "Template must have an author"
        assert isinstance(template.metadata.category, TemplateCategory)
        assert template.agent_type, "Agent template must have agent_type"
        assert template.system_prompt, "Agent template must have system_prompt"
        assert len(template.system_prompt) >= 10, "System prompt must be substantive"

    @pytest.mark.parametrize(
        "template",
        BUILTIN_DEBATE_TEMPLATES,
        ids=[t.metadata.id for t in BUILTIN_DEBATE_TEMPLATES],
    )
    def test_debate_template_schema(self, template):
        """Each debate template has all required fields."""
        assert template.metadata.id, "Template must have an ID"
        assert template.metadata.name, "Template must have a name"
        assert template.metadata.description, "Template must have a description"
        assert template.task_template, "Debate template must have task_template"
        assert template.agent_roles, "Debate template must have agent_roles"
        assert len(template.agent_roles) >= 1, "Must have at least one role"
        assert template.protocol, "Debate template must have protocol"
        assert "rounds" in template.protocol, "Protocol must specify rounds"

    @pytest.mark.parametrize(
        "template",
        BUILTIN_AGENT_TEMPLATES + BUILTIN_DEBATE_TEMPLATES,
        ids=[t.metadata.id for t in BUILTIN_AGENT_TEMPLATES + BUILTIN_DEBATE_TEMPLATES],
    )
    def test_content_hash_deterministic(self, template):
        """Content hash is consistent across calls."""
        hash1 = template.content_hash()
        hash2 = template.content_hash()
        assert hash1 == hash2, "Content hash must be deterministic"
        assert len(hash1) == 16, "Content hash must be 16 chars (SHA-256 prefix)"

    @pytest.mark.parametrize(
        "template",
        BUILTIN_AGENT_TEMPLATES + BUILTIN_DEBATE_TEMPLATES,
        ids=[t.metadata.id for t in BUILTIN_AGENT_TEMPLATES + BUILTIN_DEBATE_TEMPLATES],
    )
    def test_serialization_roundtrip(self, template):
        """Templates survive JSON serialization and deserialization."""
        data = template.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["metadata"]["id"] == template.metadata.id
        assert parsed["metadata"]["name"] == template.metadata.name
        assert parsed["content_hash"] == template.content_hash()

    def test_no_duplicate_ids(self):
        """All built-in template IDs are unique."""
        all_templates = BUILTIN_AGENT_TEMPLATES + BUILTIN_DEBATE_TEMPLATES
        ids = [t.metadata.id for t in all_templates]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_featured_templates_exist(self, registry):
        """At least one featured template exists."""
        featured = registry.featured()
        assert len(featured) > 0, "Must have at least one featured template"

    def test_all_categories_covered(self):
        """Built-in templates cover multiple categories."""
        all_templates = BUILTIN_AGENT_TEMPLATES + BUILTIN_DEBATE_TEMPLATES
        categories = {t.metadata.category for t in all_templates}
        assert len(categories) >= 3, f"Only {len(categories)} categories covered"


class TestTemplateInstallFlow:
    """Validate the full install/use lifecycle."""

    @pytest.fixture
    def registry(self, tmp_path):
        return TemplateRegistry(db_path=tmp_path / "test.db")

    def test_register_retrieve_use_agent(self, registry):
        """Full lifecycle: register → retrieve → validate config."""
        template = AgentTemplate(
            metadata=TemplateMetadata(
                id="test-lifecycle-agent",
                name="Lifecycle Test Agent",
                description="Tests the full lifecycle",
                version="1.0.0",
                author="test",
                category=TemplateCategory.CUSTOM,
            ),
            agent_type="claude",
            system_prompt="You are a test agent.",
            model_config={"temperature": 0.7, "max_tokens": 1000},
            capabilities=["testing"],
        )

        # Register
        tid = registry.register(template)
        assert tid == "test-lifecycle-agent"

        # Retrieve
        retrieved = registry.get(tid)
        assert retrieved is not None
        assert isinstance(retrieved, AgentTemplate)
        assert retrieved.agent_type == "claude"
        assert retrieved.model_config["temperature"] == 0.7

        # Export → Import roundtrip
        exported = registry.export_template(tid)
        assert exported is not None

        # Modify the ID for re-import
        data = json.loads(exported)
        data["metadata"]["id"] = "test-lifecycle-agent-copy"
        reimported_id = registry.import_template(json.dumps(data))
        assert reimported_id == "test-lifecycle-agent-copy"

        copy = registry.get(reimported_id)
        assert copy is not None
        assert copy.system_prompt == template.system_prompt

    def test_register_retrieve_use_debate(self, registry):
        """Full lifecycle for debate templates."""
        template = DebateTemplate(
            metadata=TemplateMetadata(
                id="test-lifecycle-debate",
                name="Lifecycle Test Debate",
                description="Tests debate lifecycle",
                version="1.0.0",
                author="test",
                category=TemplateCategory.DEBATE,
            ),
            task_template="Debate: {topic}",
            agent_roles=[
                {"role": "proposer", "team": "for"},
                {"role": "opponent", "team": "against"},
                {"role": "judge", "team": "neutral"},
            ],
            protocol={"rounds": 3, "consensus_mode": "vote"},
            evaluation_criteria=["logic", "evidence"],
        )

        tid = registry.register(template)
        retrieved = registry.get(tid)
        assert isinstance(retrieved, DebateTemplate)
        assert len(retrieved.agent_roles) == 3
        assert retrieved.protocol["rounds"] == 3

    def test_register_retrieve_use_workflow(self, registry):
        """Full lifecycle for workflow templates."""
        template = WorkflowTemplate(
            metadata=TemplateMetadata(
                id="test-lifecycle-workflow",
                name="Lifecycle Test Workflow",
                description="Tests workflow lifecycle",
                version="1.0.0",
                author="test",
                category=TemplateCategory.PLANNING,
            ),
            nodes=[
                {"id": "start", "type": "trigger", "config": {}},
                {"id": "process", "type": "action", "config": {"action": "analyze"}},
                {"id": "end", "type": "output", "config": {}},
            ],
            edges=[
                {"source": "start", "target": "process"},
                {"source": "process", "target": "end"},
            ],
            inputs={"topic": {"type": "string", "required": True}},
            outputs={"result": {"type": "object"}},
        )

        tid = registry.register(template)
        retrieved = registry.get(tid)
        assert isinstance(retrieved, WorkflowTemplate)
        assert len(retrieved.nodes) == 3
        assert len(retrieved.edges) == 2

    def test_rating_lifecycle(self, registry):
        """Rate, update, aggregate ratings."""
        template = AgentTemplate(
            metadata=TemplateMetadata(
                id="rated-agent",
                name="Rated Agent",
                description="For rating tests",
                version="1.0.0",
                author="test",
                category=TemplateCategory.CUSTOM,
            ),
            agent_type="claude",
            system_prompt="Test.",
        )
        registry.register(template)

        # Multiple users rate
        for i, score in enumerate([5, 4, 3, 5, 4]):
            registry.rate(
                TemplateRating(
                    user_id=f"user-{i}",
                    template_id="rated-agent",
                    score=score,
                )
            )

        avg = registry.get_average_rating("rated-agent")
        assert avg is not None
        assert 4.0 <= avg <= 4.5

        ratings = registry.get_ratings("rated-agent")
        assert len(ratings) == 5

    def test_search_filters_combined(self, registry):
        """Search with multiple filters applied."""
        # Register templates in different categories
        for i, cat in enumerate(
            [TemplateCategory.CODING, TemplateCategory.RESEARCH, TemplateCategory.CODING]
        ):
            registry.register(
                AgentTemplate(
                    metadata=TemplateMetadata(
                        id=f"filter-test-{i}",
                        name=f"Filter Test {i}",
                        description="For filter testing",
                        version="1.0.0",
                        author="test-author",
                        category=cat,
                        tags=["filter-test"],
                    ),
                    agent_type="claude",
                    system_prompt="Test.",
                )
            )

        # Filter by category + author
        results = registry.search(
            category=TemplateCategory.CODING,
            author="test-author",
        )
        assert len(results) == 2

        # Filter by tags
        results = registry.search(tags=["filter-test"])
        assert len(results) == 3


class TestTemplateValidation:
    """Test template validation edge cases."""

    def test_rating_score_validation(self):
        """Ratings must be 1-5."""
        with pytest.raises(ValueError, match="between 1 and 5"):
            TemplateRating(user_id="u", template_id="t", score=0)
        with pytest.raises(ValueError, match="between 1 and 5"):
            TemplateRating(user_id="u", template_id="t", score=6)

    def test_empty_system_prompt_allowed(self):
        """Agent templates can have empty system prompt (not recommended)."""
        template = AgentTemplate(
            metadata=TemplateMetadata(
                id="empty-prompt",
                name="Empty",
                description="Test",
                version="1.0.0",
                author="test",
                category=TemplateCategory.CUSTOM,
            ),
            agent_type="claude",
            system_prompt="",
        )
        assert template.content_hash()

    def test_workflow_empty_dag(self):
        """Workflow with no nodes/edges is technically valid."""
        template = WorkflowTemplate(
            metadata=TemplateMetadata(
                id="empty-dag",
                name="Empty DAG",
                description="Test",
                version="1.0.0",
                author="test",
                category=TemplateCategory.CUSTOM,
            ),
            nodes=[],
            edges=[],
        )
        data = template.to_dict()
        assert data["nodes"] == []
        assert data["edges"] == []


class TestRegistryHealth:
    """Tests that would power a /marketplace/health endpoint."""

    @pytest.fixture
    def registry(self, tmp_path):
        return TemplateRegistry(db_path=tmp_path / "test.db")

    def test_all_builtins_loadable(self, registry):
        """Every built-in template can be loaded from the database."""
        all_ids = [t.metadata.id for t in BUILTIN_AGENT_TEMPLATES + BUILTIN_DEBATE_TEMPLATES]
        for tid in all_ids:
            template = registry.get(tid)
            assert template is not None, f"Built-in template {tid} not loadable"

    def test_registry_categories_nonempty(self, registry):
        """Category listing returns at least one result."""
        categories = registry.list_categories()
        assert len(categories) > 0
        total = sum(c["count"] for c in categories)
        assert total == len(BUILTIN_AGENT_TEMPLATES) + len(BUILTIN_DEBATE_TEMPLATES)

    def test_popular_returns_results(self, registry):
        """Popular endpoint returns templates."""
        popular = registry.popular(limit=5)
        assert len(popular) > 0

    def test_featured_all_tagged(self, registry):
        """All featured templates actually have the 'featured' tag."""
        featured = registry.featured()
        for t in featured:
            assert "featured" in t.metadata.tags, f"{t.metadata.id} missing 'featured' tag"
