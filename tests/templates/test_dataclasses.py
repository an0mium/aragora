"""
Tests for the template dataclasses.

Tests cover:
- DebateRole creation and validation
- DebatePhase creation and validation
- DebateTemplate creation and validation
- Default values and field behavior
"""

from dataclasses import asdict, fields

import pytest

from aragora.templates import (
    DebatePhase,
    DebateRole,
    DebateTemplate,
    TemplateType,
)


class TestDebateRole:
    """Tests for the DebateRole dataclass."""

    def test_create_debate_role(self):
        """Test creating a DebateRole with required fields."""
        role = DebateRole(
            name="test_role",
            description="A test role for debate",
            objectives=["objective1", "objective2"],
            evaluation_criteria=["criteria1", "criteria2"],
        )

        assert role.name == "test_role"
        assert role.description == "A test role for debate"
        assert len(role.objectives) == 2
        assert len(role.evaluation_criteria) == 2
        assert role.example_prompts == []  # Default value

    def test_create_debate_role_with_example_prompts(self):
        """Test creating a DebateRole with example prompts."""
        role = DebateRole(
            name="reviewer",
            description="Reviews code changes",
            objectives=["find bugs"],
            evaluation_criteria=["accuracy"],
            example_prompts=["This code has a bug because..."],
        )

        assert len(role.example_prompts) == 1
        assert role.example_prompts[0] == "This code has a bug because..."

    def test_debate_role_to_dict(self):
        """Test converting DebateRole to dictionary."""
        role = DebateRole(
            name="analyst",
            description="Analyzes data",
            objectives=["analyze"],
            evaluation_criteria=["thoroughness"],
        )

        role_dict = asdict(role)

        assert role_dict["name"] == "analyst"
        assert role_dict["description"] == "Analyzes data"
        assert role_dict["objectives"] == ["analyze"]
        assert role_dict["evaluation_criteria"] == ["thoroughness"]
        assert role_dict["example_prompts"] == []

    def test_debate_role_equality(self):
        """Test DebateRole equality comparison."""
        role1 = DebateRole(
            name="test",
            description="desc",
            objectives=["obj"],
            evaluation_criteria=["crit"],
        )
        role2 = DebateRole(
            name="test",
            description="desc",
            objectives=["obj"],
            evaluation_criteria=["crit"],
        )

        assert role1 == role2

    def test_debate_role_fields(self):
        """Test that DebateRole has expected fields."""
        field_names = [f.name for f in fields(DebateRole)]

        assert "name" in field_names
        assert "description" in field_names
        assert "objectives" in field_names
        assert "evaluation_criteria" in field_names
        assert "example_prompts" in field_names


class TestDebatePhase:
    """Tests for the DebatePhase dataclass."""

    def test_create_debate_phase(self):
        """Test creating a DebatePhase with all fields."""
        phase = DebatePhase(
            name="initial_review",
            description="First review phase",
            duration_rounds=2,
            roles_active=["critic", "author"],
            objectives=["identify issues"],
            outputs=["issue list"],
        )

        assert phase.name == "initial_review"
        assert phase.description == "First review phase"
        assert phase.duration_rounds == 2
        assert len(phase.roles_active) == 2
        assert len(phase.objectives) == 1
        assert len(phase.outputs) == 1

    def test_debate_phase_single_round(self):
        """Test creating a single-round phase."""
        phase = DebatePhase(
            name="synthesis",
            description="Final synthesis",
            duration_rounds=1,
            roles_active=["synthesizer"],
            objectives=["summarize findings"],
            outputs=["summary document"],
        )

        assert phase.duration_rounds == 1

    def test_debate_phase_multiple_rounds(self):
        """Test creating a multi-round phase."""
        phase = DebatePhase(
            name="debate",
            description="Main debate phase",
            duration_rounds=5,
            roles_active=["analyst", "skeptic", "advocate"],
            objectives=["reach consensus"],
            outputs=["decision"],
        )

        assert phase.duration_rounds == 5
        assert len(phase.roles_active) == 3

    def test_debate_phase_to_dict(self):
        """Test converting DebatePhase to dictionary."""
        phase = DebatePhase(
            name="test",
            description="test phase",
            duration_rounds=3,
            roles_active=["role1"],
            objectives=["obj1"],
            outputs=["out1"],
        )

        phase_dict = asdict(phase)

        assert phase_dict["name"] == "test"
        assert phase_dict["duration_rounds"] == 3

    def test_debate_phase_fields(self):
        """Test that DebatePhase has expected fields."""
        field_names = [f.name for f in fields(DebatePhase)]

        assert "name" in field_names
        assert "description" in field_names
        assert "duration_rounds" in field_names
        assert "roles_active" in field_names
        assert "objectives" in field_names
        assert "outputs" in field_names


class TestDebateTemplate:
    """Tests for the DebateTemplate dataclass."""

    @pytest.fixture
    def sample_roles(self):
        """Create sample roles for testing."""
        return [
            DebateRole(
                name="author",
                description="Presents the work",
                objectives=["explain decisions"],
                evaluation_criteria=["clarity"],
            ),
            DebateRole(
                name="critic",
                description="Critiques the work",
                objectives=["find issues"],
                evaluation_criteria=["accuracy"],
            ),
        ]

    @pytest.fixture
    def sample_phases(self):
        """Create sample phases for testing."""
        return [
            DebatePhase(
                name="presentation",
                description="Author presents",
                duration_rounds=1,
                roles_active=["author"],
                objectives=["present work"],
                outputs=["overview"],
            ),
            DebatePhase(
                name="critique",
                description="Critic reviews",
                duration_rounds=2,
                roles_active=["critic"],
                objectives=["find issues"],
                outputs=["issue list"],
            ),
        ]

    def test_create_debate_template_minimal(self, sample_roles, sample_phases):
        """Test creating a DebateTemplate with minimal configuration."""
        template = DebateTemplate(
            template_id="test-v1",
            template_type=TemplateType.CODE_REVIEW,
            name="Test Template",
            description="A test template",
            roles=sample_roles,
            phases=sample_phases,
            recommended_agents=2,
            max_rounds=3,
            consensus_threshold=0.7,
            rubric={"accuracy": 0.5, "clarity": 0.5},
            output_format="# Result\n{result}",
            domain="testing",
        )

        assert template.template_id == "test-v1"
        assert template.template_type == TemplateType.CODE_REVIEW
        assert template.name == "Test Template"
        assert len(template.roles) == 2
        assert len(template.phases) == 2
        assert template.recommended_agents == 2
        assert template.max_rounds == 3
        assert template.consensus_threshold == 0.7
        assert template.difficulty == 0.5  # Default
        assert template.tags == []  # Default

    def test_create_debate_template_full(self, sample_roles, sample_phases):
        """Test creating a DebateTemplate with all fields."""
        template = DebateTemplate(
            template_id="full-test-v1",
            template_type=TemplateType.DESIGN_DOC,
            name="Full Test Template",
            description="A complete test template",
            roles=sample_roles,
            phases=sample_phases,
            recommended_agents=4,
            max_rounds=5,
            consensus_threshold=0.8,
            rubric={"a": 0.3, "b": 0.7},
            output_format="# Full\n{content}",
            domain="software",
            difficulty=0.9,
            tags=["test", "example"],
        )

        assert template.difficulty == 0.9
        assert template.tags == ["test", "example"]
        assert len(template.tags) == 2

    def test_debate_template_rubric_sum(self, sample_roles, sample_phases):
        """Test that rubric weights are properly stored."""
        rubric = {
            "category1": 0.25,
            "category2": 0.25,
            "category3": 0.25,
            "category4": 0.25,
        }
        template = DebateTemplate(
            template_id="rubric-test",
            template_type=TemplateType.POLICY_REVIEW,
            name="Rubric Test",
            description="Test rubric handling",
            roles=sample_roles,
            phases=sample_phases,
            recommended_agents=3,
            max_rounds=4,
            consensus_threshold=0.6,
            rubric=rubric,
            output_format="",
            domain="policy",
        )

        assert sum(template.rubric.values()) == 1.0
        assert len(template.rubric) == 4

    def test_debate_template_output_format_placeholders(self, sample_roles, sample_phases):
        """Test that output format contains placeholders."""
        output_format = """
# Summary
## Score: {score}
## Findings
{findings}
## Recommendations
{recommendations}
"""
        template = DebateTemplate(
            template_id="format-test",
            template_type=TemplateType.SECURITY_AUDIT,
            name="Format Test",
            description="Test output format",
            roles=sample_roles,
            phases=sample_phases,
            recommended_agents=4,
            max_rounds=5,
            consensus_threshold=0.75,
            rubric={"security": 1.0},
            output_format=output_format,
            domain="security",
        )

        assert "{score}" in template.output_format
        assert "{findings}" in template.output_format
        assert "{recommendations}" in template.output_format

    def test_debate_template_to_dict(self, sample_roles, sample_phases):
        """Test converting DebateTemplate to dictionary."""
        template = DebateTemplate(
            template_id="dict-test",
            template_type=TemplateType.INCIDENT_RESPONSE,
            name="Dict Test",
            description="Test serialization",
            roles=sample_roles,
            phases=sample_phases,
            recommended_agents=4,
            max_rounds=5,
            consensus_threshold=0.7,
            rubric={"a": 1.0},
            output_format="",
            domain="ops",
        )

        template_dict = asdict(template)

        assert template_dict["template_id"] == "dict-test"
        assert template_dict["template_type"] == TemplateType.INCIDENT_RESPONSE
        assert len(template_dict["roles"]) == 2
        assert len(template_dict["phases"]) == 2

    def test_debate_template_phase_total_rounds(self, sample_roles, sample_phases):
        """Test calculating total rounds from phases."""
        template = DebateTemplate(
            template_id="rounds-test",
            template_type=TemplateType.RESEARCH_SYNTHESIS,
            name="Rounds Test",
            description="Test round calculation",
            roles=sample_roles,
            phases=sample_phases,  # 1 + 2 = 3 total rounds
            recommended_agents=3,
            max_rounds=5,
            consensus_threshold=0.6,
            rubric={"a": 1.0},
            output_format="",
            domain="research",
        )

        total_phase_rounds = sum(p.duration_rounds for p in template.phases)
        assert total_phase_rounds == 3

    def test_debate_template_fields(self):
        """Test that DebateTemplate has expected fields."""
        field_names = [f.name for f in fields(DebateTemplate)]

        expected_fields = [
            "template_id",
            "template_type",
            "name",
            "description",
            "roles",
            "phases",
            "recommended_agents",
            "max_rounds",
            "consensus_threshold",
            "rubric",
            "output_format",
            "domain",
            "difficulty",
            "tags",
        ]

        for field in expected_fields:
            assert field in field_names


class TestDataclassImmutability:
    """Tests for dataclass behavior and mutability."""

    def test_debate_role_objectives_mutable(self):
        """Test that objectives list is mutable (not frozen dataclass)."""
        role = DebateRole(
            name="test",
            description="test",
            objectives=["obj1"],
            evaluation_criteria=["crit1"],
        )

        # Lists are mutable in regular dataclasses
        role.objectives.append("obj2")
        assert len(role.objectives) == 2

    def test_debate_phase_roles_active_mutable(self):
        """Test that roles_active list is mutable."""
        phase = DebatePhase(
            name="test",
            description="test",
            duration_rounds=1,
            roles_active=["role1"],
            objectives=["obj1"],
            outputs=["out1"],
        )

        phase.roles_active.append("role2")
        assert len(phase.roles_active) == 2

    def test_debate_template_tags_mutable(self):
        """Test that tags list is mutable."""
        role = DebateRole(name="r", description="d", objectives=[], evaluation_criteria=[])
        phase = DebatePhase(
            name="p",
            description="d",
            duration_rounds=1,
            roles_active=[],
            objectives=[],
            outputs=[],
        )

        template = DebateTemplate(
            template_id="t",
            template_type=TemplateType.CODE_REVIEW,
            name="n",
            description="d",
            roles=[role],
            phases=[phase],
            recommended_agents=1,
            max_rounds=1,
            consensus_threshold=0.5,
            rubric={},
            output_format="",
            domain="test",
            tags=["tag1"],
        )

        template.tags.append("tag2")
        assert len(template.tags) == 2


class TestDataclassDefaults:
    """Tests for dataclass default values."""

    def test_debate_role_default_example_prompts(self):
        """Test that example_prompts defaults to empty list."""
        role = DebateRole(
            name="test",
            description="test",
            objectives=["obj"],
            evaluation_criteria=["crit"],
        )

        assert role.example_prompts == []
        assert isinstance(role.example_prompts, list)

    def test_debate_template_default_difficulty(self):
        """Test that difficulty defaults to 0.5."""
        role = DebateRole(name="r", description="d", objectives=[], evaluation_criteria=[])
        phase = DebatePhase(
            name="p",
            description="d",
            duration_rounds=1,
            roles_active=[],
            objectives=[],
            outputs=[],
        )

        template = DebateTemplate(
            template_id="t",
            template_type=TemplateType.CODE_REVIEW,
            name="n",
            description="d",
            roles=[role],
            phases=[phase],
            recommended_agents=1,
            max_rounds=1,
            consensus_threshold=0.5,
            rubric={},
            output_format="",
            domain="test",
        )

        assert template.difficulty == 0.5

    def test_debate_template_default_tags(self):
        """Test that tags defaults to empty list."""
        role = DebateRole(name="r", description="d", objectives=[], evaluation_criteria=[])
        phase = DebatePhase(
            name="p",
            description="d",
            duration_rounds=1,
            roles_active=[],
            objectives=[],
            outputs=[],
        )

        template = DebateTemplate(
            template_id="t",
            template_type=TemplateType.CODE_REVIEW,
            name="n",
            description="d",
            roles=[role],
            phases=[phase],
            recommended_agents=1,
            max_rounds=1,
            consensus_threshold=0.5,
            rubric={},
            output_format="",
            domain="test",
        )

        assert template.tags == []
        assert isinstance(template.tags, list)

    def test_default_list_independence(self):
        """Test that default lists are independent between instances."""
        role1 = DebateRole(name="r1", description="d", objectives=[], evaluation_criteria=[])
        role2 = DebateRole(name="r2", description="d", objectives=[], evaluation_criteria=[])

        role1.example_prompts.append("prompt1")

        # Role2 should not be affected
        assert role2.example_prompts == []
