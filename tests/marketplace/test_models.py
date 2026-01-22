"""Tests for marketplace data models."""

import pytest
from datetime import datetime

from aragora.marketplace.models import (
    AgentTemplate,
    DebateTemplate,
    WorkflowTemplate,
    TemplateMetadata,
    TemplateRating,
    TemplateCategory,
    BUILTIN_AGENT_TEMPLATES,
    BUILTIN_DEBATE_TEMPLATES,
)


class TestTemplateCategory:
    """Tests for TemplateCategory enum."""

    def test_category_values(self):
        """Test all category values exist."""
        assert TemplateCategory.ANALYSIS.value == "analysis"
        assert TemplateCategory.CODING.value == "coding"
        assert TemplateCategory.CREATIVE.value == "creative"
        assert TemplateCategory.DEBATE.value == "debate"
        assert TemplateCategory.RESEARCH.value == "research"
        assert TemplateCategory.DECISION.value == "decision"
        assert TemplateCategory.BRAINSTORM.value == "brainstorm"
        assert TemplateCategory.REVIEW.value == "review"
        assert TemplateCategory.PLANNING.value == "planning"
        assert TemplateCategory.CUSTOM.value == "custom"


class TestTemplateMetadata:
    """Tests for TemplateMetadata dataclass."""

    def test_required_fields(self):
        """Test metadata with required fields only."""
        metadata = TemplateMetadata(
            id="test-1",
            name="Test Template",
            description="A test template",
            version="1.0.0",
            author="test_user",
            category=TemplateCategory.CODING,
        )
        assert metadata.id == "test-1"
        assert metadata.name == "Test Template"
        assert metadata.version == "1.0.0"
        assert metadata.downloads == 0
        assert metadata.stars == 0
        assert metadata.license == "MIT"

    def test_metadata_to_dict(self):
        """Test serialization to dictionary."""
        metadata = TemplateMetadata(
            id="test-1",
            name="Test",
            description="Desc",
            version="1.0.0",
            author="user",
            category=TemplateCategory.CODING,
            tags=["python", "testing"],
        )
        d = metadata.to_dict()
        assert d["id"] == "test-1"
        assert d["category"] == "coding"
        assert d["tags"] == ["python", "testing"]
        assert "created_at" in d

    def test_optional_urls(self):
        """Test optional repository and documentation URLs."""
        metadata = TemplateMetadata(
            id="test-1",
            name="Test",
            description="Desc",
            version="1.0.0",
            author="user",
            category=TemplateCategory.CODING,
            repository_url="https://github.com/test/repo",
            documentation_url="https://docs.example.com",
        )
        assert metadata.repository_url == "https://github.com/test/repo"
        assert metadata.documentation_url == "https://docs.example.com"


class TestTemplateRating:
    """Tests for TemplateRating dataclass."""

    def test_valid_rating(self):
        """Test creating a valid rating."""
        rating = TemplateRating(
            user_id="user-1",
            template_id="template-1",
            score=5,
            review="Great template!",
        )
        assert rating.score == 5
        assert rating.review == "Great template!"

    def test_invalid_score_too_low(self):
        """Test that score below 1 raises error."""
        with pytest.raises(ValueError):
            TemplateRating(
                user_id="user-1",
                template_id="template-1",
                score=0,
            )

    def test_invalid_score_too_high(self):
        """Test that score above 5 raises error."""
        with pytest.raises(ValueError):
            TemplateRating(
                user_id="user-1",
                template_id="template-1",
                score=6,
            )


class TestAgentTemplate:
    """Tests for AgentTemplate dataclass."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return TemplateMetadata(
            id="agent-1",
            name="Test Agent",
            description="A test agent",
            version="1.0.0",
            author="test",
            category=TemplateCategory.CODING,
        )

    def test_create_agent_template(self, sample_metadata):
        """Test creating an agent template."""
        template = AgentTemplate(
            metadata=sample_metadata,
            agent_type="claude",
            system_prompt="You are a helpful assistant.",
            capabilities=["code_review", "debugging"],
            constraints=["no_personal_data"],
        )
        assert template.agent_type == "claude"
        assert len(template.capabilities) == 2
        assert len(template.constraints) == 1

    def test_content_hash(self, sample_metadata):
        """Test content hash generation."""
        template = AgentTemplate(
            metadata=sample_metadata,
            agent_type="claude",
            system_prompt="You are helpful.",
        )
        hash1 = template.content_hash()
        assert len(hash1) == 16  # First 16 chars of SHA256

        # Same content should produce same hash
        template2 = AgentTemplate(
            metadata=sample_metadata,
            agent_type="claude",
            system_prompt="You are helpful.",
        )
        assert template2.content_hash() == hash1

        # Different content should produce different hash
        template3 = AgentTemplate(
            metadata=sample_metadata,
            agent_type="gpt4",
            system_prompt="You are helpful.",
        )
        assert template3.content_hash() != hash1

    def test_to_dict(self, sample_metadata):
        """Test serialization to dictionary."""
        template = AgentTemplate(
            metadata=sample_metadata,
            agent_type="claude",
            system_prompt="You are helpful.",
            model_config={"temperature": 0.7},
            examples=[{"input": "Hi", "output": "Hello!"}],
        )
        d = template.to_dict()
        assert d["agent_type"] == "claude"
        assert d["system_prompt"] == "You are helpful."
        assert d["model_config"] == {"temperature": 0.7}
        assert len(d["examples"]) == 1
        assert "content_hash" in d


class TestDebateTemplate:
    """Tests for DebateTemplate dataclass."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return TemplateMetadata(
            id="debate-1",
            name="Test Debate",
            description="A test debate format",
            version="1.0.0",
            author="test",
            category=TemplateCategory.DEBATE,
        )

    def test_create_debate_template(self, sample_metadata):
        """Test creating a debate template."""
        template = DebateTemplate(
            metadata=sample_metadata,
            task_template="Discuss: {topic}",
            agent_roles=[
                {"role": "proposer", "team": "pro"},
                {"role": "opposer", "team": "con"},
            ],
            protocol={"rounds": 3, "consensus_mode": "majority"},
            evaluation_criteria=["logic", "evidence"],
        )
        assert template.task_template == "Discuss: {topic}"
        assert len(template.agent_roles) == 2
        assert template.protocol["rounds"] == 3

    def test_content_hash(self, sample_metadata):
        """Test content hash generation."""
        template = DebateTemplate(
            metadata=sample_metadata,
            task_template="Topic: {topic}",
            agent_roles=[],
            protocol={"rounds": 2},
        )
        hash1 = template.content_hash()
        assert len(hash1) == 16


class TestWorkflowTemplate:
    """Tests for WorkflowTemplate dataclass."""

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return TemplateMetadata(
            id="workflow-1",
            name="Test Workflow",
            description="A test workflow",
            version="1.0.0",
            author="test",
            category=TemplateCategory.PLANNING,
        )

    def test_create_workflow_template(self, sample_metadata):
        """Test creating a workflow template."""
        template = WorkflowTemplate(
            metadata=sample_metadata,
            nodes=[
                {"id": "start", "type": "input"},
                {"id": "process", "type": "debate"},
                {"id": "end", "type": "output"},
            ],
            edges=[
                {"from": "start", "to": "process"},
                {"from": "process", "to": "end"},
            ],
            inputs={"topic": {"type": "string"}},
            outputs={"result": {"type": "string"}},
        )
        assert len(template.nodes) == 3
        assert len(template.edges) == 2
        assert "topic" in template.inputs


class TestBuiltinTemplates:
    """Tests for built-in templates."""

    def test_builtin_agent_templates_exist(self):
        """Test that built-in agent templates are defined."""
        assert len(BUILTIN_AGENT_TEMPLATES) >= 3

    def test_devils_advocate_template(self):
        """Test Devil's Advocate template."""
        template = next(
            (t for t in BUILTIN_AGENT_TEMPLATES if t.metadata.id == "devil-advocate"),
            None,
        )
        assert template is not None
        assert template.agent_type == "claude"
        assert "challenge" in template.system_prompt.lower()
        assert "counterargument" in template.capabilities

    def test_code_reviewer_template(self):
        """Test Code Reviewer template."""
        template = next(
            (t for t in BUILTIN_AGENT_TEMPLATES if t.metadata.id == "code-reviewer"),
            None,
        )
        assert template is not None
        assert "security" in template.system_prompt.lower()
        assert "code_analysis" in template.capabilities

    def test_builtin_debate_templates_exist(self):
        """Test that built-in debate templates are defined."""
        assert len(BUILTIN_DEBATE_TEMPLATES) >= 3

    def test_oxford_style_template(self):
        """Test Oxford-Style Debate template."""
        template = next(
            (t for t in BUILTIN_DEBATE_TEMPLATES if t.metadata.id == "oxford-style"),
            None,
        )
        assert template is not None
        assert len(template.agent_roles) == 4
        assert template.protocol["rounds"] == 4

    def test_brainstorm_template(self):
        """Test Brainstorm Session template."""
        template = next(
            (t for t in BUILTIN_DEBATE_TEMPLATES if t.metadata.id == "brainstorm-session"),
            None,
        )
        assert template is not None
        assert template.protocol["no_criticism_phase"] is True
