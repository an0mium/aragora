"""
Tests for Workflow Pattern base classes.

Tests cover:
- PatternType enum: all values exist, string values correct
- ResourceLimits dataclass: defaults, custom values, to_dict serialization
- PatternConfig dataclass: defaults, custom values, field types
- WorkflowPattern ABC: abstract enforcement, concrete subclass, init defaults/custom
- WorkflowPattern.create classmethod: factory method behavior
- WorkflowPattern._generate_id: prefix, format, uniqueness
- WorkflowPattern._create_agent_step: StepDefinition with agent config and visuals
- WorkflowPattern._create_task_step: StepDefinition with task config and visuals
- WorkflowPattern._create_debate_step: StepDefinition with debate config and visuals
- WorkflowPattern._create_transition: TransitionRule with condition and visuals
- WorkflowPattern._get_agent_color: known agents, unknown fallback, case insensitivity
- WorkflowPattern._get_category_color: known categories, unknown fallback
"""

import pytest

from aragora.workflow.patterns.base import (
    PatternType,
    ResourceLimits,
    PatternConfig,
    WorkflowPattern,
)
from aragora.workflow.types import (
    WorkflowDefinition,
    StepDefinition,
    TransitionRule,
    Position,
    NodeCategory,
    EdgeType,
    WorkflowCategory,
)


# ============================================================================
# Concrete subclass for testing the abstract WorkflowPattern
# ============================================================================


class ConcretePattern(WorkflowPattern):
    """Minimal concrete implementation for testing."""

    pattern_type = PatternType.CUSTOM

    def create_workflow(self) -> WorkflowDefinition:
        return WorkflowDefinition(
            id=self._generate_id(),
            name=self.name,
            steps=[],
            transitions=[],
        )


# ============================================================================
# PatternType Enum Tests
# ============================================================================


class TestPatternType:
    """Tests for PatternType enumeration."""

    def test_hive_mind_value(self):
        """HIVE_MIND has value 'hive_mind'."""
        assert PatternType.HIVE_MIND.value == "hive_mind"

    def test_sequential_value(self):
        """SEQUENTIAL has value 'sequential'."""
        assert PatternType.SEQUENTIAL.value == "sequential"

    def test_map_reduce_value(self):
        """MAP_REDUCE has value 'map_reduce'."""
        assert PatternType.MAP_REDUCE.value == "map_reduce"

    def test_hierarchical_value(self):
        """HIERARCHICAL has value 'hierarchical'."""
        assert PatternType.HIERARCHICAL.value == "hierarchical"

    def test_review_cycle_value(self):
        """REVIEW_CYCLE has value 'review_cycle'."""
        assert PatternType.REVIEW_CYCLE.value == "review_cycle"

    def test_dialectic_value(self):
        """DIALECTIC has value 'dialectic'."""
        assert PatternType.DIALECTIC.value == "dialectic"

    def test_debate_value(self):
        """DEBATE has value 'debate'."""
        assert PatternType.DEBATE.value == "debate"

    def test_custom_value(self):
        """CUSTOM has value 'custom'."""
        assert PatternType.CUSTOM.value == "custom"

    def test_total_member_count(self):
        """PatternType has exactly 9 members."""
        assert len(PatternType) == 9

    def test_all_values_are_strings(self):
        """All PatternType values are strings."""
        for member in PatternType:
            assert isinstance(member.value, str)

    def test_lookup_by_value(self):
        """PatternType members can be looked up by string value."""
        assert PatternType("hive_mind") is PatternType.HIVE_MIND
        assert PatternType("debate") is PatternType.DEBATE


# ============================================================================
# ResourceLimits Tests
# ============================================================================


class TestResourceLimits:
    """Tests for ResourceLimits dataclass."""

    def test_default_max_tokens(self):
        """Default max_tokens is 100000."""
        limits = ResourceLimits()
        assert limits.max_tokens == 100000

    def test_default_max_cost_usd(self):
        """Default max_cost_usd is 10.0."""
        limits = ResourceLimits()
        assert limits.max_cost_usd == 10.0

    def test_default_timeout_seconds(self):
        """Default timeout_seconds is 600.0."""
        limits = ResourceLimits()
        assert limits.timeout_seconds == 600.0

    def test_default_max_parallel_agents(self):
        """Default max_parallel_agents is 5."""
        limits = ResourceLimits()
        assert limits.max_parallel_agents == 5

    def test_default_max_retries(self):
        """Default max_retries is 3."""
        limits = ResourceLimits()
        assert limits.max_retries == 3

    def test_custom_values(self):
        """Custom values override all defaults."""
        limits = ResourceLimits(
            max_tokens=50000,
            max_cost_usd=5.0,
            timeout_seconds=300.0,
            max_parallel_agents=10,
            max_retries=1,
        )
        assert limits.max_tokens == 50000
        assert limits.max_cost_usd == 5.0
        assert limits.timeout_seconds == 300.0
        assert limits.max_parallel_agents == 10
        assert limits.max_retries == 1

    def test_to_dict_keys(self):
        """to_dict includes all five fields."""
        limits = ResourceLimits()
        d = limits.to_dict()
        expected_keys = {
            "max_tokens",
            "max_cost_usd",
            "timeout_seconds",
            "max_parallel_agents",
            "max_retries",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_defaults(self):
        """to_dict values match the default field values."""
        limits = ResourceLimits()
        d = limits.to_dict()
        assert d["max_tokens"] == 100000
        assert d["max_cost_usd"] == 10.0
        assert d["timeout_seconds"] == 600.0
        assert d["max_parallel_agents"] == 5
        assert d["max_retries"] == 3

    def test_to_dict_with_custom_values(self):
        """to_dict reflects custom values."""
        limits = ResourceLimits(max_tokens=200, max_cost_usd=0.5)
        d = limits.to_dict()
        assert d["max_tokens"] == 200
        assert d["max_cost_usd"] == 0.5


# ============================================================================
# PatternConfig Tests
# ============================================================================


class TestPatternConfig:
    """Tests for PatternConfig dataclass."""

    def test_required_name(self):
        """name is the only required field."""
        config = PatternConfig(name="test")
        assert config.name == "test"

    def test_default_description(self):
        """Default description is empty string."""
        config = PatternConfig(name="test")
        assert config.description == ""

    def test_default_agents(self):
        """Default agents are ['claude', 'gpt4']."""
        config = PatternConfig(name="test")
        assert config.agents == ["claude", "gpt4"]

    def test_default_task(self):
        """Default task is empty string."""
        config = PatternConfig(name="test")
        assert config.task == ""

    def test_default_category(self):
        """Default category is WorkflowCategory.GENERAL."""
        config = PatternConfig(name="test")
        assert config.category == WorkflowCategory.GENERAL

    def test_default_limits_is_resource_limits(self):
        """Default limits is a ResourceLimits instance with defaults."""
        config = PatternConfig(name="test")
        assert isinstance(config.limits, ResourceLimits)
        assert config.limits.max_tokens == 100000

    def test_default_tags(self):
        """Default tags is an empty list."""
        config = PatternConfig(name="test")
        assert config.tags == []

    def test_default_metadata(self):
        """Default metadata is an empty dict."""
        config = PatternConfig(name="test")
        assert config.metadata == {}

    def test_default_agent_configs(self):
        """Default agent_configs is an empty dict."""
        config = PatternConfig(name="test")
        assert config.agent_configs == {}

    def test_default_output_format(self):
        """Default output_format is 'json'."""
        config = PatternConfig(name="test")
        assert config.output_format == "json"

    def test_custom_values(self):
        """All fields can be customized."""
        custom_limits = ResourceLimits(max_tokens=999)
        config = PatternConfig(
            name="Custom Config",
            description="A custom pattern",
            agents=["gemini", "mistral"],
            task="Analyze data",
            category=WorkflowCategory.FINANCE,
            limits=custom_limits,
            tags=["finance", "analysis"],
            metadata={"priority": "high"},
            agent_configs={"gemini": {"temperature": 0.5}},
            output_format="text",
        )
        assert config.name == "Custom Config"
        assert config.description == "A custom pattern"
        assert config.agents == ["gemini", "mistral"]
        assert config.task == "Analyze data"
        assert config.category == WorkflowCategory.FINANCE
        assert config.limits.max_tokens == 999
        assert config.tags == ["finance", "analysis"]
        assert config.metadata == {"priority": "high"}
        assert config.agent_configs == {"gemini": {"temperature": 0.5}}
        assert config.output_format == "text"

    def test_agents_list_independent_across_instances(self):
        """Each instance gets its own agents list (no shared mutable default)."""
        config1 = PatternConfig(name="a")
        config2 = PatternConfig(name="b")
        config1.agents.append("grok")
        assert "grok" not in config2.agents


# ============================================================================
# WorkflowPattern Abstract Enforcement Tests
# ============================================================================


class TestWorkflowPatternAbstract:
    """Tests for WorkflowPattern abstract class behavior."""

    def test_cannot_instantiate_directly(self):
        """WorkflowPattern cannot be instantiated because it is abstract."""
        with pytest.raises(TypeError):
            WorkflowPattern(name="fail")

    def test_subclass_without_create_workflow_raises(self):
        """A subclass that does not implement create_workflow cannot be instantiated."""

        class IncompletePattern(WorkflowPattern):
            pass

        with pytest.raises(TypeError):
            IncompletePattern(name="fail")

    def test_concrete_subclass_instantiates(self):
        """A properly implemented subclass can be instantiated."""
        pattern = ConcretePattern(name="Valid")
        assert pattern.name == "Valid"


# ============================================================================
# WorkflowPattern __init__ Tests
# ============================================================================


class TestWorkflowPatternInit:
    """Tests for WorkflowPattern.__init__ through ConcretePattern."""

    def test_default_agents(self):
        """Default agents are ['claude', 'gpt4'] when None is passed."""
        pattern = ConcretePattern(name="Test")
        assert pattern.agents == ["claude", "gpt4"]

    def test_explicit_none_agents_uses_default(self):
        """Passing agents=None explicitly uses the default list."""
        pattern = ConcretePattern(name="Test", agents=None)
        assert pattern.agents == ["claude", "gpt4"]

    def test_custom_agents(self):
        """Custom agents list is stored correctly."""
        pattern = ConcretePattern(name="Test", agents=["gemini", "grok"])
        assert pattern.agents == ["gemini", "grok"]

    def test_default_task(self):
        """Default task is empty string."""
        pattern = ConcretePattern(name="Test")
        assert pattern.task == ""

    def test_custom_task(self):
        """Custom task string is stored."""
        pattern = ConcretePattern(name="Test", task="Summarize document")
        assert pattern.task == "Summarize document"

    def test_kwargs_stored_in_config(self):
        """Extra keyword arguments are stored in self.config dict."""
        pattern = ConcretePattern(name="Test", rounds=5, mode="strict")
        assert pattern.config == {"rounds": 5, "mode": "strict"}

    def test_empty_kwargs_gives_empty_config(self):
        """No extra kwargs results in an empty config dict."""
        pattern = ConcretePattern(name="Test")
        assert pattern.config == {}

    def test_pattern_type_class_attribute(self):
        """pattern_type is accessible from the instance."""
        pattern = ConcretePattern(name="Test")
        assert pattern.pattern_type == PatternType.CUSTOM


# ============================================================================
# WorkflowPattern.create Classmethod Tests
# ============================================================================


class TestWorkflowPatternCreate:
    """Tests for WorkflowPattern.create factory classmethod."""

    def test_create_returns_workflow_definition(self):
        """create() returns a WorkflowDefinition."""
        result = ConcretePattern.create(name="Factory Test")
        assert isinstance(result, WorkflowDefinition)

    def test_create_passes_name(self):
        """create() passes the name through to the workflow."""
        result = ConcretePattern.create(name="My Workflow")
        assert result.name == "My Workflow"

    def test_create_passes_agents(self):
        """create() passes agents to the pattern instance."""

        # ConcretePattern.create_workflow doesn't use agents in the output,
        # but the pattern should receive them. We test via a custom subclass.
        class AgentCapture(WorkflowPattern):
            captured_agents = None

            def create_workflow(self):
                AgentCapture.captured_agents = self.agents
                return WorkflowDefinition(
                    id=self._generate_id(), name=self.name, steps=[], transitions=[]
                )

        AgentCapture.create(name="Test", agents=["deepseek"])
        assert AgentCapture.captured_agents == ["deepseek"]

    def test_create_passes_kwargs(self):
        """create() forwards extra kwargs to the pattern constructor."""

        class KwargsCapture(WorkflowPattern):
            captured_config = None

            def create_workflow(self):
                KwargsCapture.captured_config = self.config
                return WorkflowDefinition(
                    id=self._generate_id(), name=self.name, steps=[], transitions=[]
                )

        KwargsCapture.create(name="Test", rounds=7, mode="fast")
        assert KwargsCapture.captured_config == {"rounds": 7, "mode": "fast"}


# ============================================================================
# WorkflowPattern._generate_id Tests
# ============================================================================


class TestGenerateId:
    """Tests for WorkflowPattern._generate_id."""

    def test_default_prefix(self):
        """Default prefix is 'wf'."""
        pattern = ConcretePattern(name="Test")
        generated = pattern._generate_id()
        assert generated.startswith("wf_")

    def test_custom_prefix(self):
        """Custom prefix is applied."""
        pattern = ConcretePattern(name="Test")
        generated = pattern._generate_id(prefix="step")
        assert generated.startswith("step_")

    def test_format_prefix_underscore_hex(self):
        """ID has format: {prefix}_{12 hex chars}."""
        pattern = ConcretePattern(name="Test")
        generated = pattern._generate_id(prefix="node")
        parts = generated.split("_", 1)
        assert parts[0] == "node"
        hex_part = parts[1]
        assert len(hex_part) == 12
        # Verify it is valid hexadecimal
        int(hex_part, 16)

    def test_uniqueness(self):
        """Repeated calls produce unique IDs."""
        pattern = ConcretePattern(name="Test")
        ids = {pattern._generate_id() for _ in range(100)}
        assert len(ids) == 100


# ============================================================================
# WorkflowPattern._create_agent_step Tests
# ============================================================================


class TestCreateAgentStep:
    """Tests for WorkflowPattern._create_agent_step."""

    def _make_step(self, **kwargs):
        """Helper to create an agent step with defaults."""
        pattern = ConcretePattern(name="Test")
        defaults = dict(
            step_id="step_1",
            name="Agent Step",
            agent_type="claude",
            prompt="Do something",
            position=Position(x=100, y=200),
        )
        defaults.update(kwargs)
        return pattern._create_agent_step(**defaults)

    def test_returns_step_definition(self):
        """Returns a StepDefinition instance."""
        step = self._make_step()
        assert isinstance(step, StepDefinition)

    def test_step_type_is_agent(self):
        """step_type is 'agent'."""
        step = self._make_step()
        assert step.step_type == "agent"

    def test_step_id_set(self):
        """Step id matches the provided step_id."""
        step = self._make_step(step_id="my_step")
        assert step.id == "my_step"

    def test_step_name_set(self):
        """Step name matches the provided name."""
        step = self._make_step(name="Claude Analysis")
        assert step.name == "Claude Analysis"

    def test_config_has_agent_type(self):
        """Config dict contains agent_type."""
        step = self._make_step(agent_type="gemini")
        assert step.config["agent_type"] == "gemini"

    def test_config_has_prompt_template(self):
        """Config dict contains prompt_template."""
        step = self._make_step(prompt="Analyze this code")
        assert step.config["prompt_template"] == "Analyze this code"

    def test_default_timeout(self):
        """Default timeout is 120.0 seconds."""
        step = self._make_step()
        assert step.timeout_seconds == 120.0

    def test_custom_timeout(self):
        """Custom timeout is applied."""
        step = self._make_step(timeout=60.0)
        assert step.timeout_seconds == 60.0

    def test_default_retries(self):
        """Default retries is 1."""
        step = self._make_step()
        assert step.retries == 1

    def test_custom_retries(self):
        """Custom retries value is applied."""
        step = self._make_step(retries=3)
        assert step.retries == 3

    def test_visual_position(self):
        """Visual data has the correct position."""
        step = self._make_step(position=Position(x=50, y=75))
        assert step.visual.position.x == 50
        assert step.visual.position.y == 75

    def test_visual_category_is_agent(self):
        """Visual category is NodeCategory.AGENT."""
        step = self._make_step()
        assert step.visual.category == NodeCategory.AGENT

    def test_visual_color_matches_agent(self):
        """Visual color corresponds to the agent type."""
        step = self._make_step(agent_type="claude")
        assert step.visual.color == "#7c3aed"


# ============================================================================
# WorkflowPattern._create_task_step Tests
# ============================================================================


class TestCreateTaskStep:
    """Tests for WorkflowPattern._create_task_step."""

    def _make_step(self, **kwargs):
        """Helper to create a task step with defaults."""
        pattern = ConcretePattern(name="Test")
        defaults = dict(
            step_id="task_1",
            name="Task Step",
            task_type="merge",
            config={"key": "value"},
            position=Position(x=0, y=0),
        )
        defaults.update(kwargs)
        return pattern._create_task_step(**defaults)

    def test_returns_step_definition(self):
        """Returns a StepDefinition instance."""
        step = self._make_step()
        assert isinstance(step, StepDefinition)

    def test_step_type_is_task(self):
        """step_type is 'task'."""
        step = self._make_step()
        assert step.step_type == "task"

    def test_config_has_task_type(self):
        """Config dict contains task_type."""
        step = self._make_step(task_type="split")
        assert step.config["task_type"] == "split"

    def test_config_merges_extra_config(self):
        """Extra config dict entries are merged into config."""
        step = self._make_step(config={"strategy": "round_robin"})
        assert step.config["strategy"] == "round_robin"

    def test_default_category_is_task(self):
        """Default visual category is NodeCategory.TASK."""
        step = self._make_step()
        assert step.visual.category == NodeCategory.TASK

    def test_custom_category(self):
        """Custom category is applied."""
        step = self._make_step(category=NodeCategory.CONTROL)
        assert step.visual.category == NodeCategory.CONTROL

    def test_visual_position(self):
        """Visual data has the correct position."""
        step = self._make_step(position=Position(x=300, y=400))
        assert step.visual.position.x == 300
        assert step.visual.position.y == 400

    def test_visual_color_matches_category(self):
        """Visual color corresponds to the node category."""
        step = self._make_step(category=NodeCategory.TASK)
        assert step.visual.color == "#48bb78"


# ============================================================================
# WorkflowPattern._create_debate_step Tests
# ============================================================================


class TestCreateDebateStep:
    """Tests for WorkflowPattern._create_debate_step."""

    def _make_step(self, **kwargs):
        """Helper to create a debate step with defaults."""
        pattern = ConcretePattern(name="Test")
        defaults = dict(
            step_id="debate_1",
            name="Debate Step",
            topic="Best architecture",
            agents=["claude", "gpt4"],
            position=Position(x=200, y=100),
        )
        defaults.update(kwargs)
        return pattern._create_debate_step(**defaults)

    def test_returns_step_definition(self):
        """Returns a StepDefinition instance."""
        step = self._make_step()
        assert isinstance(step, StepDefinition)

    def test_step_type_is_debate(self):
        """step_type is 'debate'."""
        step = self._make_step()
        assert step.step_type == "debate"

    def test_config_has_topic(self):
        """Config dict contains the topic."""
        step = self._make_step(topic="Rate limiter design")
        assert step.config["topic"] == "Rate limiter design"

    def test_config_has_agents(self):
        """Config dict contains the agents list."""
        step = self._make_step(agents=["gemini", "grok"])
        assert step.config["agents"] == ["gemini", "grok"]

    def test_default_rounds(self):
        """Default rounds is 3."""
        step = self._make_step()
        assert step.config["rounds"] == 3

    def test_custom_rounds(self):
        """Custom rounds value is applied."""
        step = self._make_step(rounds=5)
        assert step.config["rounds"] == 5

    def test_default_consensus_mechanism(self):
        """Default consensus_mechanism is 'majority'."""
        step = self._make_step()
        assert step.config["consensus_mechanism"] == "majority"

    def test_custom_consensus_mechanism(self):
        """Custom consensus_mechanism is applied."""
        step = self._make_step(consensus_mechanism="unanimous")
        assert step.config["consensus_mechanism"] == "unanimous"

    def test_visual_category_is_debate(self):
        """Visual category is NodeCategory.DEBATE."""
        step = self._make_step()
        assert step.visual.category == NodeCategory.DEBATE

    def test_visual_color_is_teal(self):
        """Debate step visual color is #38b2ac (teal)."""
        step = self._make_step()
        assert step.visual.color == "#38b2ac"

    def test_visual_position(self):
        """Visual data has the correct position."""
        step = self._make_step(position=Position(x=500, y=600))
        assert step.visual.position.x == 500
        assert step.visual.position.y == 600


# ============================================================================
# WorkflowPattern._create_transition Tests
# ============================================================================


class TestCreateTransition:
    """Tests for WorkflowPattern._create_transition."""

    def _make_transition(self, **kwargs):
        """Helper to create a transition with defaults."""
        pattern = ConcretePattern(name="Test")
        defaults = dict(
            from_step="step_a",
            to_step="step_b",
        )
        defaults.update(kwargs)
        return pattern._create_transition(**defaults)

    def test_returns_transition_rule(self):
        """Returns a TransitionRule instance."""
        tr = self._make_transition()
        assert isinstance(tr, TransitionRule)

    def test_from_step(self):
        """from_step is set correctly."""
        tr = self._make_transition(from_step="start")
        assert tr.from_step == "start"

    def test_to_step(self):
        """to_step is set correctly."""
        tr = self._make_transition(to_step="end")
        assert tr.to_step == "end"

    def test_default_condition(self):
        """Default condition is 'True'."""
        tr = self._make_transition()
        assert tr.condition == "True"

    def test_custom_condition(self):
        """Custom condition string is applied."""
        tr = self._make_transition(condition="result.score > 0.8")
        assert tr.condition == "result.score > 0.8"

    def test_default_label(self):
        """Default label is empty string."""
        tr = self._make_transition()
        assert tr.label == ""

    def test_custom_label(self):
        """Custom label is applied."""
        tr = self._make_transition(label="On success")
        assert tr.label == "On success"

    def test_default_edge_type(self):
        """Default visual edge_type is EdgeType.DATA_FLOW."""
        tr = self._make_transition()
        assert tr.visual.edge_type == EdgeType.DATA_FLOW

    def test_custom_edge_type(self):
        """Custom edge_type is applied."""
        tr = self._make_transition(edge_type=EdgeType.CONDITIONAL)
        assert tr.visual.edge_type == EdgeType.CONDITIONAL

    def test_id_starts_with_tr(self):
        """Transition ID starts with 'tr_'."""
        tr = self._make_transition()
        assert tr.id.startswith("tr_")

    def test_visual_label_matches(self):
        """Visual edge label matches the transition label."""
        tr = self._make_transition(label="next")
        assert tr.visual.label == "next"


# ============================================================================
# WorkflowPattern._get_agent_color Tests
# ============================================================================


class TestGetAgentColor:
    """Tests for WorkflowPattern._get_agent_color."""

    def _get_color(self, agent_type):
        pattern = ConcretePattern(name="Test")
        return pattern._get_agent_color(agent_type)

    def test_claude_color(self):
        """claude returns purple (#7c3aed)."""
        assert self._get_color("claude") == "#7c3aed"

    def test_gpt4_color(self):
        """gpt4 returns green (#10b981)."""
        assert self._get_color("gpt4") == "#10b981"

    def test_gpt_4_color(self):
        """gpt-4 (with hyphen) also returns green (#10b981)."""
        assert self._get_color("gpt-4") == "#10b981"

    def test_gemini_color(self):
        """gemini returns blue (#3b82f6)."""
        assert self._get_color("gemini") == "#3b82f6"

    def test_mistral_color(self):
        """mistral returns amber (#f59e0b)."""
        assert self._get_color("mistral") == "#f59e0b"

    def test_grok_color(self):
        """grok returns red (#ef4444)."""
        assert self._get_color("grok") == "#ef4444"

    def test_deepseek_color(self):
        """deepseek returns cyan (#06b6d4)."""
        assert self._get_color("deepseek") == "#06b6d4"

    def test_llama_color(self):
        """llama returns violet (#8b5cf6)."""
        assert self._get_color("llama") == "#8b5cf6"

    def test_unknown_agent_returns_gray(self):
        """An unknown agent returns gray (#6b7280)."""
        assert self._get_color("unknown_model") == "#6b7280"

    def test_case_insensitive(self):
        """Agent type lookup is case-insensitive."""
        assert self._get_color("Claude") == "#7c3aed"
        assert self._get_color("GPT4") == "#10b981"
        assert self._get_color("GEMINI") == "#3b82f6"


# ============================================================================
# WorkflowPattern._get_category_color Tests
# ============================================================================


class TestGetCategoryColor:
    """Tests for WorkflowPattern._get_category_color."""

    def _get_color(self, category):
        pattern = ConcretePattern(name="Test")
        return pattern._get_category_color(category)

    def test_agent_category_color(self):
        """AGENT category returns #4299e1."""
        assert self._get_color(NodeCategory.AGENT) == "#4299e1"

    def test_task_category_color(self):
        """TASK category returns #48bb78."""
        assert self._get_color(NodeCategory.TASK) == "#48bb78"

    def test_control_category_color(self):
        """CONTROL category returns #ed8936."""
        assert self._get_color(NodeCategory.CONTROL) == "#ed8936"

    def test_memory_category_color(self):
        """MEMORY category returns #9f7aea."""
        assert self._get_color(NodeCategory.MEMORY) == "#9f7aea"

    def test_human_category_color(self):
        """HUMAN category returns #f56565."""
        assert self._get_color(NodeCategory.HUMAN) == "#f56565"

    def test_debate_category_color(self):
        """DEBATE category returns #38b2ac."""
        assert self._get_color(NodeCategory.DEBATE) == "#38b2ac"

    def test_integration_category_color(self):
        """INTEGRATION category returns #667eea."""
        assert self._get_color(NodeCategory.INTEGRATION) == "#667eea"

    def test_unknown_category_returns_gray(self):
        """A category not in the map returns gray (#6b7280)."""
        # Use a sentinel value that won't be in the dict
        assert self._get_color("nonexistent") == "#6b7280"
