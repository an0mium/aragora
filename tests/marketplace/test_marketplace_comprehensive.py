"""
Comprehensive tests for the Marketplace module.

Covers:
- TemplateCategory enum
- TemplateMetadata dataclass and serialization
- TemplateRating validation
- AgentTemplate (creation, content_hash, to_dict)
- DebateTemplate (creation, content_hash, to_dict)
- WorkflowTemplate (creation, content_hash, to_dict)
- Built-in templates integrity
- TemplateRegistry (CRUD, search, ratings, stars, downloads, export/import, featured, popular, delete)
- MarketplaceClient (mocked HTTP operations)
- MarketplaceError
- Edge cases: empty fields, unicode, large data, unknown formats
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.marketplace.models import (
    BUILTIN_AGENT_TEMPLATES,
    BUILTIN_DEBATE_TEMPLATES,
    AgentTemplate,
    DebateTemplate,
    TemplateCategory,
    TemplateMetadata,
    TemplateRating,
    WorkflowTemplate,
)
from aragora.marketplace.registry import TemplateRegistry
from aragora.marketplace.client import MarketplaceClient, MarketplaceConfig, MarketplaceError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metadata():
    """Create minimal valid metadata."""
    return TemplateMetadata(
        id="test-1",
        name="Test Template",
        description="Description",
        version="1.0.0",
        author="tester",
        category=TemplateCategory.CODING,
    )


@pytest.fixture
def agent_template(metadata):
    return AgentTemplate(
        metadata=metadata,
        agent_type="claude",
        system_prompt="You are helpful.",
        capabilities=["cap1"],
        constraints=["con1"],
    )


@pytest.fixture
def debate_template():
    return DebateTemplate(
        metadata=TemplateMetadata(
            id="debate-1",
            name="Debate Template",
            description="Desc",
            version="1.0.0",
            author="tester",
            category=TemplateCategory.DEBATE,
        ),
        task_template="Debate: {topic}",
        agent_roles=[{"role": "proposer"}, {"role": "opposer"}],
        protocol={"rounds": 3, "consensus_mode": "majority"},
        evaluation_criteria=["logic", "evidence"],
        success_metrics={"clarity": 0.8},
    )


@pytest.fixture
def workflow_template():
    return WorkflowTemplate(
        metadata=TemplateMetadata(
            id="workflow-1",
            name="Workflow Template",
            description="Desc",
            version="1.0.0",
            author="tester",
            category=TemplateCategory.PLANNING,
        ),
        nodes=[{"id": "start"}, {"id": "end"}],
        edges=[{"source": "start", "target": "end"}],
        inputs={"topic": {"type": "string"}},
        outputs={"result": {"type": "object"}},
        variables={"timeout": 30},
    )


@pytest.fixture
def registry(tmp_path):
    """Create a registry with a temp database."""
    return TemplateRegistry(db_path=tmp_path / "test.db")


# ===========================================================================
# TemplateCategory
# ===========================================================================


class TestTemplateCategory:
    def test_all_values(self):
        expected = {
            "analysis",
            "coding",
            "creative",
            "debate",
            "research",
            "decision",
            "brainstorm",
            "review",
            "planning",
            "custom",
        }
        actual = {c.value for c in TemplateCategory}
        assert actual == expected

    def test_member_count(self):
        assert len(TemplateCategory) == 10

    def test_from_value(self):
        assert TemplateCategory("coding") == TemplateCategory.CODING


# ===========================================================================
# TemplateMetadata
# ===========================================================================


class TestTemplateMetadata:
    def test_required_fields(self, metadata):
        assert metadata.id == "test-1"
        assert metadata.name == "Test Template"
        assert metadata.version == "1.0.0"

    def test_defaults(self, metadata):
        assert metadata.downloads == 0
        assert metadata.stars == 0
        assert metadata.license == "MIT"
        assert metadata.tags == []
        assert metadata.repository_url is None
        assert metadata.documentation_url is None

    def test_to_dict_keys(self, metadata):
        d = metadata.to_dict()
        required_keys = {
            "id",
            "name",
            "description",
            "version",
            "author",
            "category",
            "tags",
            "created_at",
            "updated_at",
            "downloads",
            "stars",
            "license",
            "repository_url",
            "documentation_url",
        }
        assert set(d.keys()) == required_keys

    def test_to_dict_category_value(self, metadata):
        d = metadata.to_dict()
        assert d["category"] == "coding"

    def test_timestamps_are_set(self, metadata):
        assert metadata.created_at is not None
        assert metadata.updated_at is not None


# ===========================================================================
# TemplateRating
# ===========================================================================


class TestTemplateRating:
    def test_valid_scores(self):
        for score in [1, 2, 3, 4, 5]:
            r = TemplateRating(user_id="u", template_id="t", score=score)
            assert r.score == score

    def test_score_zero_raises(self):
        with pytest.raises(ValueError, match="between 1 and 5"):
            TemplateRating(user_id="u", template_id="t", score=0)

    def test_score_six_raises(self):
        with pytest.raises(ValueError, match="between 1 and 5"):
            TemplateRating(user_id="u", template_id="t", score=6)

    def test_score_negative_raises(self):
        with pytest.raises(ValueError, match="between 1 and 5"):
            TemplateRating(user_id="u", template_id="t", score=-1)

    def test_review_optional(self):
        r = TemplateRating(user_id="u", template_id="t", score=3)
        assert r.review is None

    def test_created_at_default(self):
        r = TemplateRating(user_id="u", template_id="t", score=3)
        assert r.created_at is not None


# ===========================================================================
# AgentTemplate
# ===========================================================================


class TestAgentTemplate:
    def test_basic_creation(self, agent_template):
        assert agent_template.agent_type == "claude"
        assert len(agent_template.capabilities) == 1
        assert len(agent_template.constraints) == 1

    def test_default_fields(self, metadata):
        t = AgentTemplate(
            metadata=metadata,
            agent_type="gpt4",
            system_prompt="Test.",
        )
        assert t.model_config == {}
        assert t.capabilities == []
        assert t.constraints == []
        assert t.examples == []

    def test_content_hash_deterministic(self, agent_template):
        h1 = agent_template.content_hash()
        h2 = agent_template.content_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_different_for_different_content(self, metadata):
        t1 = AgentTemplate(metadata=metadata, agent_type="claude", system_prompt="A")
        t2 = AgentTemplate(metadata=metadata, agent_type="claude", system_prompt="B")
        assert t1.content_hash() != t2.content_hash()

    def test_content_hash_ignores_metadata(self, metadata):
        t1 = AgentTemplate(metadata=metadata, agent_type="claude", system_prompt="X")
        metadata2 = TemplateMetadata(
            id="different",
            name="Different",
            description="Different",
            version="2.0.0",
            author="other",
            category=TemplateCategory.RESEARCH,
        )
        t2 = AgentTemplate(metadata=metadata2, agent_type="claude", system_prompt="X")
        assert t1.content_hash() == t2.content_hash()

    def test_to_dict(self, agent_template):
        d = agent_template.to_dict()
        assert d["agent_type"] == "claude"
        assert d["system_prompt"] == "You are helpful."
        assert "content_hash" in d
        assert "metadata" in d
        assert d["metadata"]["id"] == "test-1"

    def test_to_dict_with_examples(self, metadata):
        t = AgentTemplate(
            metadata=metadata,
            agent_type="claude",
            system_prompt=".",
            examples=[{"input": "hi", "output": "hello"}],
        )
        d = t.to_dict()
        assert len(d["examples"]) == 1


# ===========================================================================
# DebateTemplate
# ===========================================================================


class TestDebateTemplate:
    def test_basic_creation(self, debate_template):
        assert debate_template.task_template == "Debate: {topic}"
        assert len(debate_template.agent_roles) == 2
        assert debate_template.protocol["rounds"] == 3

    def test_content_hash_deterministic(self, debate_template):
        h1 = debate_template.content_hash()
        h2 = debate_template.content_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_to_dict(self, debate_template):
        d = debate_template.to_dict()
        assert "task_template" in d
        assert "agent_roles" in d
        assert "protocol" in d
        assert "evaluation_criteria" in d
        assert "success_metrics" in d
        assert "content_hash" in d

    def test_success_metrics(self, debate_template):
        assert debate_template.success_metrics == {"clarity": 0.8}


# ===========================================================================
# WorkflowTemplate
# ===========================================================================


class TestWorkflowTemplate:
    def test_basic_creation(self, workflow_template):
        assert len(workflow_template.nodes) == 2
        assert len(workflow_template.edges) == 1
        assert "topic" in workflow_template.inputs
        assert "result" in workflow_template.outputs
        assert workflow_template.variables == {"timeout": 30}

    def test_content_hash_deterministic(self, workflow_template):
        h1 = workflow_template.content_hash()
        h2 = workflow_template.content_hash()
        assert h1 == h2

    def test_to_dict(self, workflow_template):
        d = workflow_template.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "inputs" in d
        assert "outputs" in d
        assert "variables" in d
        assert "content_hash" in d

    def test_empty_dag(self):
        t = WorkflowTemplate(
            metadata=TemplateMetadata(
                id="empty",
                name="Empty",
                description="",
                version="1.0.0",
                author="t",
                category=TemplateCategory.CUSTOM,
            ),
            nodes=[],
            edges=[],
        )
        d = t.to_dict()
        assert d["nodes"] == []
        assert d["edges"] == []
        assert t.content_hash()  # should not raise


# ===========================================================================
# Built-in Templates
# ===========================================================================


class TestBuiltinTemplates:
    def test_agent_templates_count(self):
        assert len(BUILTIN_AGENT_TEMPLATES) >= 3

    def test_debate_templates_count(self):
        assert len(BUILTIN_DEBATE_TEMPLATES) >= 3

    def test_all_agent_templates_have_featured_tag(self):
        for t in BUILTIN_AGENT_TEMPLATES:
            assert "featured" in t.metadata.tags

    def test_all_debate_templates_have_featured_tag(self):
        for t in BUILTIN_DEBATE_TEMPLATES:
            assert "featured" in t.metadata.tags

    def test_no_duplicate_ids(self):
        all_templates = BUILTIN_AGENT_TEMPLATES + BUILTIN_DEBATE_TEMPLATES
        ids = [t.metadata.id for t in all_templates]
        assert len(ids) == len(set(ids))

    def test_all_agent_templates_serializable(self):
        for t in BUILTIN_AGENT_TEMPLATES:
            data = json.dumps(t.to_dict())
            parsed = json.loads(data)
            assert parsed["metadata"]["id"] == t.metadata.id

    def test_all_debate_templates_serializable(self):
        for t in BUILTIN_DEBATE_TEMPLATES:
            data = json.dumps(t.to_dict())
            parsed = json.loads(data)
            assert parsed["metadata"]["id"] == t.metadata.id


# ===========================================================================
# TemplateRegistry
# ===========================================================================


class TestTemplateRegistry:
    def test_initialization_loads_builtins(self, registry):
        templates = registry.search()
        assert len(templates) >= len(BUILTIN_AGENT_TEMPLATES) + len(BUILTIN_DEBATE_TEMPLATES)

    def test_register_and_get(self, registry, agent_template):
        tid = registry.register(agent_template)
        assert tid == "test-1"
        retrieved = registry.get(tid)
        assert retrieved is not None
        assert isinstance(retrieved, AgentTemplate)
        assert retrieved.metadata.name == "Test Template"

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_register_debate_template(self, registry, debate_template):
        tid = registry.register(debate_template)
        retrieved = registry.get(tid)
        assert isinstance(retrieved, DebateTemplate)
        assert retrieved.protocol["rounds"] == 3

    def test_register_workflow_template(self, registry, workflow_template):
        tid = registry.register(workflow_template)
        retrieved = registry.get(tid)
        assert isinstance(retrieved, WorkflowTemplate)
        assert len(retrieved.nodes) == 2

    def test_search_by_query(self, registry, agent_template):
        registry.register(agent_template)
        results = registry.search(query="Test Template")
        assert any(t.metadata.id == "test-1" for t in results)

    def test_search_by_category(self, registry, agent_template):
        registry.register(agent_template)
        results = registry.search(category=TemplateCategory.CODING)
        assert any(t.metadata.id == "test-1" for t in results)

    def test_search_by_category_no_match(self, registry, agent_template):
        registry.register(agent_template)
        results = registry.search(category=TemplateCategory.CREATIVE)
        assert not any(t.metadata.id == "test-1" for t in results)

    def test_search_by_type(self, registry, agent_template, debate_template):
        registry.register(agent_template)
        registry.register(debate_template)
        agent_results = registry.search(template_type="AgentTemplate")
        debate_results = registry.search(template_type="DebateTemplate")
        assert any(t.metadata.id == "test-1" for t in agent_results)
        assert any(t.metadata.id == "debate-1" for t in debate_results)

    def test_search_by_author(self, registry, agent_template):
        registry.register(agent_template)
        results = registry.search(author="tester")
        assert any(t.metadata.id == "test-1" for t in results)

    def test_search_by_tags(self, registry):
        t = AgentTemplate(
            metadata=TemplateMetadata(
                id="tagged",
                name="Tagged",
                description="D",
                version="1.0.0",
                author="t",
                category=TemplateCategory.CUSTOM,
                tags=["python", "ai"],
            ),
            agent_type="claude",
            system_prompt=".",
        )
        registry.register(t)
        results = registry.search(tags=["python"])
        assert any(r.metadata.id == "tagged" for r in results)
        results = registry.search(tags=["nonexistent"])
        assert not any(r.metadata.id == "tagged" for r in results)

    def test_search_pagination(self, registry):
        all_results = registry.search(limit=100)
        page1 = registry.search(limit=2, offset=0)
        page2 = registry.search(limit=2, offset=2)
        assert len(page1) <= 2
        assert len(page2) <= 2
        if len(all_results) > 4:
            assert page1[0].metadata.id != page2[0].metadata.id

    def test_search_combined_filters(self, registry):
        for i in range(3):
            registry.register(
                AgentTemplate(
                    metadata=TemplateMetadata(
                        id=f"combo-{i}",
                        name=f"Combo {i}",
                        description="Combo test",
                        version="1.0.0",
                        author="combo-author",
                        category=TemplateCategory.CODING if i < 2 else TemplateCategory.RESEARCH,
                        tags=["combo"],
                    ),
                    agent_type="claude",
                    system_prompt=".",
                )
            )
        results = registry.search(
            category=TemplateCategory.CODING,
            author="combo-author",
        )
        assert len(results) == 2

    def test_list_categories(self, registry):
        categories = registry.list_categories()
        assert len(categories) > 0
        for c in categories:
            assert "category" in c
            assert "count" in c

    def test_rate_template(self, registry, agent_template):
        registry.register(agent_template)
        registry.rate(TemplateRating(user_id="u1", template_id="test-1", score=5))
        ratings = registry.get_ratings("test-1")
        assert len(ratings) == 1
        assert ratings[0].score == 5

    def test_rate_update_replaces(self, registry, agent_template):
        registry.register(agent_template)
        registry.rate(TemplateRating(user_id="u1", template_id="test-1", score=2))
        registry.rate(TemplateRating(user_id="u1", template_id="test-1", score=5))
        ratings = registry.get_ratings("test-1")
        assert len(ratings) == 1
        assert ratings[0].score == 5

    def test_rate_multiple_users(self, registry, agent_template):
        registry.register(agent_template)
        for i in range(5):
            registry.rate(TemplateRating(user_id=f"u{i}", template_id="test-1", score=3 + (i % 3)))
        ratings = registry.get_ratings("test-1")
        assert len(ratings) == 5

    def test_get_average_rating(self, registry, agent_template):
        registry.register(agent_template)
        registry.rate(TemplateRating(user_id="u1", template_id="test-1", score=5))
        registry.rate(TemplateRating(user_id="u2", template_id="test-1", score=3))
        avg = registry.get_average_rating("test-1")
        assert avg == 4.0

    def test_get_average_rating_no_ratings(self, registry, agent_template):
        registry.register(agent_template)
        avg = registry.get_average_rating("test-1")
        assert avg is None

    def test_get_average_rating_nonexistent_template(self, registry):
        avg = registry.get_average_rating("nonexistent")
        assert avg is None

    def test_increment_downloads(self, registry, agent_template):
        registry.register(agent_template)
        registry.increment_downloads("test-1")
        registry.increment_downloads("test-1")
        registry.increment_downloads("test-1")
        t = registry.get("test-1")
        assert t.metadata.downloads == 3

    def test_star_template(self, registry, agent_template):
        registry.register(agent_template)
        registry.star("test-1")
        registry.star("test-1")
        t = registry.get("test-1")
        assert t.metadata.stars == 2

    def test_delete_custom_template(self, registry, agent_template):
        registry.register(agent_template)
        result = registry.delete("test-1")
        assert result is True
        assert registry.get("test-1") is None

    def test_cannot_delete_builtin(self, registry):
        result = registry.delete("devil-advocate")
        assert result is False
        assert registry.get("devil-advocate") is not None

    def test_delete_nonexistent(self, registry):
        result = registry.delete("nonexistent-xyz")
        assert result is False

    def test_featured(self, registry):
        featured = registry.featured()
        assert len(featured) > 0
        for t in featured:
            assert "featured" in t.metadata.tags

    def test_featured_limit(self, registry):
        featured = registry.featured(limit=2)
        assert len(featured) <= 2

    def test_popular(self, registry, agent_template):
        registry.register(agent_template)
        for _ in range(10):
            registry.increment_downloads("test-1")
        popular = registry.popular(limit=10)
        # The template with most downloads should be first or near top
        assert len(popular) > 0

    def test_export_template(self, registry, agent_template):
        registry.register(agent_template)
        exported = registry.export_template("test-1")
        assert exported is not None
        data = json.loads(exported)
        assert data["metadata"]["id"] == "test-1"
        assert data["agent_type"] == "claude"

    def test_export_nonexistent(self, registry):
        assert registry.export_template("nonexistent") is None

    def test_import_agent_template(self, registry, agent_template):
        data = agent_template.to_dict()
        data["metadata"]["id"] = "imported-agent"
        tid = registry.import_template(json.dumps(data))
        assert tid == "imported-agent"
        imported = registry.get("imported-agent")
        assert isinstance(imported, AgentTemplate)

    def test_import_debate_template(self, registry, debate_template):
        data = debate_template.to_dict()
        data["metadata"]["id"] = "imported-debate"
        tid = registry.import_template(json.dumps(data))
        assert tid == "imported-debate"
        imported = registry.get("imported-debate")
        assert isinstance(imported, DebateTemplate)

    def test_import_workflow_template(self, registry, workflow_template):
        data = workflow_template.to_dict()
        data["metadata"]["id"] = "imported-workflow"
        tid = registry.import_template(json.dumps(data))
        assert tid == "imported-workflow"
        imported = registry.get("imported-workflow")
        assert isinstance(imported, WorkflowTemplate)

    def test_import_unknown_format_raises(self, registry):
        bad_data = json.dumps(
            {
                "metadata": {
                    "id": "x",
                    "name": "x",
                    "description": "x",
                    "version": "1.0.0",
                    "author": "t",
                    "category": "custom",
                }
            }
        )
        with pytest.raises(ValueError, match="Unknown template format"):
            registry.import_template(bad_data)

    def test_export_import_roundtrip(self, registry, agent_template):
        registry.register(agent_template)
        exported = registry.export_template("test-1")
        assert exported is not None

        data = json.loads(exported)
        data["metadata"]["id"] = "roundtrip-copy"
        tid = registry.import_template(json.dumps(data))
        copy = registry.get(tid)
        assert copy is not None
        assert copy.system_prompt == agent_template.system_prompt

    def test_upsert_overwrites(self, registry, agent_template):
        registry.register(agent_template)
        # Modify and re-register
        agent_template.system_prompt = "Updated prompt"
        registry.register(agent_template)
        retrieved = registry.get("test-1")
        assert retrieved.system_prompt == "Updated prompt"

    def test_row_to_template_unknown_type_raises(self, registry):
        """If an unknown type is stored, _row_to_template raises."""
        with registry._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO templates
                (id, type, name, description, version, author, category, tags,
                 content, content_hash, created_at, updated_at, downloads, stars, is_builtin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "bad-type",
                    "UnknownType",
                    "Bad",
                    "Bad type",
                    "1.0.0",
                    "t",
                    "custom",
                    "[]",
                    "{}",
                    "abc",
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:00",
                    0,
                    0,
                    0,
                ),
            )
            conn.commit()
        with pytest.raises(ValueError, match="Unknown template type"):
            registry.get("bad-type")

    def test_get_ratings_empty(self, registry, agent_template):
        registry.register(agent_template)
        ratings = registry.get_ratings("test-1")
        assert ratings == []

    def test_rating_with_review(self, registry, agent_template):
        registry.register(agent_template)
        registry.rate(
            TemplateRating(
                user_id="u1",
                template_id="test-1",
                score=4,
                review="Great template!",
            )
        )
        ratings = registry.get_ratings("test-1")
        assert ratings[0].review == "Great template!"


# ===========================================================================
# MarketplaceClient
# ===========================================================================


class TestMarketplaceConfig:
    def test_defaults(self):
        c = MarketplaceConfig()
        assert "marketplace.aragora.ai" in c.base_url
        assert c.api_key is None
        assert c.timeout == 30.0

    def test_custom_values(self):
        c = MarketplaceConfig(
            base_url="https://custom.api.com",
            api_key="key123",
            timeout=60.0,
        )
        assert c.base_url == "https://custom.api.com"
        assert c.api_key == "key123"


class TestMarketplaceClient:
    @pytest.fixture
    def client(self):
        config = MarketplaceConfig(base_url="https://test.api.com", api_key="testkey")
        return MarketplaceClient(config=config)

    @pytest.mark.asyncio
    async def test_close_when_no_session(self, client):
        """Closing without a session should not raise."""
        await client.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        config = MarketplaceConfig()
        async with MarketplaceClient(config=config) as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_search_templates(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"templates": [{"id": "t1"}, {"id": "t2"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        results = await client.search_templates(query="test", limit=10)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_template(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"metadata": {"id": "t1"}})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = await client.get_template("t1")
        assert result["metadata"]["id"] == "t1"

    @pytest.mark.asyncio
    async def test_request_error_raises(self, client):
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not found")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        with pytest.raises(MarketplaceError, match="API error 404"):
            await client.get_template("nonexistent")

    @pytest.mark.asyncio
    async def test_publish_template(self, client, agent_template):
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"id": "test-1", "status": "published"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = await client.publish_template(agent_template)
        assert result["id"] == "test-1"

    @pytest.mark.asyncio
    async def test_delete_template(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"deleted": True})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = await client.delete_template("t1")
        assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_rate_template(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = await client.rate_template("t1", score=5, review="Excellent")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_star_template(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"starred": True})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = await client.star_template("t1")
        assert result["starred"] is True

    @pytest.mark.asyncio
    async def test_data_to_template_agent(self, client):
        data = {
            "metadata": {
                "id": "a1",
                "name": "Agent",
                "description": "D",
                "version": "1.0.0",
                "author": "t",
                "category": "coding",
            },
            "agent_type": "claude",
            "system_prompt": "You are helpful.",
            "model_config": {},
            "capabilities": [],
            "constraints": [],
            "examples": [],
        }
        template = client._data_to_template(data)
        assert isinstance(template, AgentTemplate)
        assert template.metadata.id == "a1"

    @pytest.mark.asyncio
    async def test_data_to_template_debate(self, client):
        data = {
            "metadata": {
                "id": "d1",
                "name": "Debate",
                "description": "D",
                "version": "1.0.0",
                "author": "t",
                "category": "debate",
            },
            "task_template": "T: {x}",
            "agent_roles": [{"role": "r1"}],
            "protocol": {"rounds": 2},
        }
        template = client._data_to_template(data)
        assert isinstance(template, DebateTemplate)

    @pytest.mark.asyncio
    async def test_data_to_template_workflow(self, client):
        data = {
            "metadata": {
                "id": "w1",
                "name": "Wf",
                "description": "D",
                "version": "1.0.0",
                "author": "t",
                "category": "planning",
            },
            "nodes": [{"id": "n1"}],
            "edges": [{"source": "n1", "target": "n2"}],
        }
        template = client._data_to_template(data)
        assert isinstance(template, WorkflowTemplate)

    @pytest.mark.asyncio
    async def test_data_to_template_unknown_raises(self, client):
        data = {
            "metadata": {
                "id": "u1",
                "name": "Unknown",
                "description": "D",
                "version": "1.0.0",
                "author": "t",
                "category": "custom",
            },
        }
        with pytest.raises(ValueError, match="Unknown template format"):
            client._data_to_template(data)

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"templates": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        await client.search_templates(
            category=TemplateCategory.CODING,
            template_type="AgentTemplate",
            tags=["python", "ai"],
        )
        # Verify the request was made with correct params
        call_args = mock_session.request.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params", {})
        assert params.get("category") == "coding"
        assert params.get("type") == "AgentTemplate"
        assert params.get("tags") == "python,ai"


class TestMarketplaceError:
    def test_basic(self):
        err = MarketplaceError("connection failed")
        assert str(err) == "connection failed"

    def test_is_exception(self):
        with pytest.raises(MarketplaceError):
            raise MarketplaceError("oops")


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_unicode_template_name(self, registry):
        t = AgentTemplate(
            metadata=TemplateMetadata(
                id="unicode-test",
                name="Modele d'analyse",
                description="Description avec des caracteres speciaux",
                version="1.0.0",
                author="utilisateur",
                category=TemplateCategory.ANALYSIS,
            ),
            agent_type="claude",
            system_prompt="Vous etes un assistant.",
        )
        tid = registry.register(t)
        retrieved = registry.get(tid)
        assert retrieved.metadata.name == "Modele d'analyse"

    def test_very_long_system_prompt(self, registry):
        t = AgentTemplate(
            metadata=TemplateMetadata(
                id="long-prompt",
                name="Long",
                description="D",
                version="1.0.0",
                author="t",
                category=TemplateCategory.CUSTOM,
            ),
            agent_type="claude",
            system_prompt="x" * 100_000,
        )
        tid = registry.register(t)
        retrieved = registry.get(tid)
        assert len(retrieved.system_prompt) == 100_000

    def test_empty_description(self, registry):
        t = AgentTemplate(
            metadata=TemplateMetadata(
                id="no-desc",
                name="No Desc",
                description="",
                version="1.0.0",
                author="t",
                category=TemplateCategory.CUSTOM,
            ),
            agent_type="claude",
            system_prompt=".",
        )
        tid = registry.register(t)
        retrieved = registry.get(tid)
        assert retrieved.metadata.description == ""

    def test_many_tags(self, registry):
        tags = [f"tag{i}" for i in range(50)]
        t = AgentTemplate(
            metadata=TemplateMetadata(
                id="many-tags",
                name="Tags",
                description="D",
                version="1.0.0",
                author="t",
                category=TemplateCategory.CUSTOM,
                tags=tags,
            ),
            agent_type="claude",
            system_prompt=".",
        )
        tid = registry.register(t)
        retrieved = registry.get(tid)
        assert len(retrieved.metadata.tags) == 50

    def test_special_characters_in_query(self, registry):
        """Search with special SQL characters should not crash."""
        results = registry.search(query="'; DROP TABLE templates; --")
        # Should not raise, just return results (likely empty match)
        assert isinstance(results, list)

    def test_large_model_config(self, registry):
        t = AgentTemplate(
            metadata=TemplateMetadata(
                id="large-config",
                name="Large Config",
                description="D",
                version="1.0.0",
                author="t",
                category=TemplateCategory.CUSTOM,
            ),
            agent_type="claude",
            system_prompt=".",
            model_config={f"param_{i}": i for i in range(100)},
        )
        tid = registry.register(t)
        retrieved = registry.get(tid)
        assert len(retrieved.model_config) == 100

    def test_workflow_with_many_nodes(self, registry):
        nodes = [{"id": f"node_{i}", "type": "action"} for i in range(100)]
        edges = [{"source": f"node_{i}", "target": f"node_{i + 1}"} for i in range(99)]
        t = WorkflowTemplate(
            metadata=TemplateMetadata(
                id="big-dag",
                name="Big DAG",
                description="D",
                version="1.0.0",
                author="t",
                category=TemplateCategory.PLANNING,
            ),
            nodes=nodes,
            edges=edges,
        )
        tid = registry.register(t)
        retrieved = registry.get(tid)
        assert len(retrieved.nodes) == 100
        assert len(retrieved.edges) == 99
