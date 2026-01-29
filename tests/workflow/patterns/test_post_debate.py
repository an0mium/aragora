"""
Tests for Post-Debate Workflow Pattern.

Tests cover:
- PostDebateConfig defaults and customization
- PostDebatePattern initialization
- WorkflowDefinition generation with all features enabled
- WorkflowDefinition generation with features selectively disabled
- Step types, IDs, and visual metadata
- Transition rules and flow
- Entry step selection logic
- Inputs/outputs schema
- Tags and category
- Factory classmethod (PostDebatePattern.create)
- get_default_post_debate_workflow helper
- Workspace ID propagation
- Webhook notification step inclusion
"""

import pytest


# ============================================================================
# PostDebateConfig Tests
# ============================================================================


class TestPostDebateConfig:
    """Tests for PostDebateConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from aragora.workflow.patterns.post_debate import PostDebateConfig

        config = PostDebateConfig()
        assert config.store_consensus is True
        assert config.extract_facts is True
        assert config.notify_webhook is None
        assert config.generate_summary is True
        assert config.workspace_id is None

    def test_custom_config(self):
        """Test custom configuration values."""
        from aragora.workflow.patterns.post_debate import PostDebateConfig

        config = PostDebateConfig(
            store_consensus=False,
            extract_facts=False,
            notify_webhook="https://example.com/hook",
            generate_summary=True,
            workspace_id="ws_abc",
        )
        assert config.store_consensus is False
        assert config.extract_facts is False
        assert config.notify_webhook == "https://example.com/hook"
        assert config.workspace_id == "ws_abc"


# ============================================================================
# PostDebatePattern Initialization Tests
# ============================================================================


class TestPostDebatePatternInit:
    """Tests for PostDebatePattern initialization."""

    def test_default_init(self):
        """Test default initialization."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        pattern = PostDebatePattern()
        assert pattern.name == "Post-Debate Workflow"
        assert pattern.agents == ["claude"]
        assert pattern.task == "Process debate outcome"

    def test_custom_init(self):
        """Test custom initialization."""
        from aragora.workflow.patterns.post_debate import (
            PostDebatePattern,
            PostDebateConfig,
        )

        config = PostDebateConfig(store_consensus=False)
        pattern = PostDebatePattern(
            name="Custom Post",
            config=config,
            agents=["gpt4", "claude"],
            task="Custom task",
        )
        assert pattern.name == "Custom Post"
        assert pattern.agents == ["gpt4", "claude"]
        assert pattern.post_config.store_consensus is False

    def test_default_config_when_none(self):
        """Test that None config creates default PostDebateConfig."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        pattern = PostDebatePattern(config=None)
        assert pattern.post_config.store_consensus is True
        assert pattern.post_config.extract_facts is True


# ============================================================================
# Workflow Generation Tests - All Features Enabled
# ============================================================================


class TestPostDebateWorkflowAllEnabled:
    """Tests for workflow generation with all features enabled."""

    def _create_workflow(self):
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        pattern = PostDebatePattern()
        return pattern.create_workflow()

    def test_returns_workflow_definition(self):
        """Test that create_workflow returns a WorkflowDefinition."""
        from aragora.workflow.types import WorkflowDefinition

        wf = self._create_workflow()
        assert isinstance(wf, WorkflowDefinition)

    def test_workflow_id_prefix(self):
        """Test that workflow ID starts with post_debate_."""
        wf = self._create_workflow()
        assert wf.id.startswith("post_debate_")

    def test_workflow_has_all_steps(self):
        """Test that all enabled steps are present."""
        wf = self._create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "extract_knowledge" in step_ids
        assert "store_consensus" in step_ids
        assert "generate_summary" in step_ids
        assert "complete" in step_ids

    def test_workflow_step_count(self):
        """Test step count with all features enabled (no webhook)."""
        wf = self._create_workflow()
        # extract, store, summary, complete = 4 steps
        assert len(wf.steps) == 4

    def test_entry_step_is_extract_knowledge(self):
        """Test that entry step is extract_knowledge when facts enabled."""
        wf = self._create_workflow()
        assert wf.entry_step == "extract_knowledge"

    def test_transitions_count(self):
        """Test the number of transitions."""
        wf = self._create_workflow()
        # extract -> store, store -> summary, summary -> complete = 3
        assert len(wf.transitions) == 3

    def test_transition_flow(self):
        """Test the transition flow order."""
        wf = self._create_workflow()
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["extract_knowledge"] == "store_consensus"
        assert flow["store_consensus"] == "generate_summary"
        assert flow["generate_summary"] == "complete"

    def test_workflow_description(self):
        """Test workflow description is set."""
        wf = self._create_workflow()
        assert "post-debate" in wf.description.lower()

    def test_workflow_inputs(self):
        """Test workflow input definitions."""
        wf = self._create_workflow()
        assert "debate_id" in wf.inputs
        assert "consensus" in wf.inputs
        assert "confidence" in wf.inputs
        assert "agents" in wf.inputs

    def test_workflow_outputs(self):
        """Test workflow output definitions."""
        wf = self._create_workflow()
        assert "status" in wf.outputs
        assert "knowledge_stored" in wf.outputs
        assert "summary" in wf.outputs

    def test_workflow_tags(self):
        """Test workflow tags."""
        wf = self._create_workflow()
        assert "post-debate" in wf.tags
        assert "automation" in wf.tags
        assert "knowledge-extraction" in wf.tags

    def test_workflow_category(self):
        """Test workflow category is GENERAL."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow()
        assert wf.category == WorkflowCategory.GENERAL


# ============================================================================
# Workflow Generation Tests - Selective Features
# ============================================================================


class TestPostDebateWorkflowSelective:
    """Tests for workflow generation with selective features."""

    def test_no_extract_facts(self):
        """Test workflow without fact extraction."""
        from aragora.workflow.patterns.post_debate import (
            PostDebatePattern,
            PostDebateConfig,
        )

        config = PostDebateConfig(extract_facts=False)
        wf = PostDebatePattern(config=config).create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "extract_knowledge" not in step_ids
        assert wf.entry_step == "store_consensus"

    def test_no_store_consensus(self):
        """Test workflow without consensus storage."""
        from aragora.workflow.patterns.post_debate import (
            PostDebatePattern,
            PostDebateConfig,
        )

        config = PostDebateConfig(store_consensus=False)
        wf = PostDebatePattern(config=config).create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "store_consensus" not in step_ids
        # extract_knowledge should transition to generate_summary
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["extract_knowledge"] == "generate_summary"

    def test_no_generate_summary(self):
        """Test workflow without summary generation."""
        from aragora.workflow.patterns.post_debate import (
            PostDebatePattern,
            PostDebateConfig,
        )

        config = PostDebateConfig(generate_summary=False)
        wf = PostDebatePattern(config=config).create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "generate_summary" not in step_ids
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["store_consensus"] == "complete"

    def test_all_disabled_only_complete(self):
        """Test workflow with all features disabled has only complete step."""
        from aragora.workflow.patterns.post_debate import (
            PostDebatePattern,
            PostDebateConfig,
        )

        config = PostDebateConfig(
            extract_facts=False,
            store_consensus=False,
            generate_summary=False,
        )
        wf = PostDebatePattern(config=config).create_workflow()
        assert len(wf.steps) == 1
        assert wf.steps[0].id == "complete"
        assert wf.entry_step == "complete"
        assert len(wf.transitions) == 0

    def test_with_webhook(self):
        """Test workflow with webhook notification step."""
        from aragora.workflow.patterns.post_debate import (
            PostDebatePattern,
            PostDebateConfig,
        )

        config = PostDebateConfig(notify_webhook="https://hooks.example.com/notify")
        wf = PostDebatePattern(config=config).create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "notify" in step_ids
        # 5 steps: extract, store, summary, notify, complete
        assert len(wf.steps) == 5
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["generate_summary"] == "notify"
        assert flow["notify"] == "complete"

    def test_workspace_id_propagated(self):
        """Test that workspace_id is set in store step config."""
        from aragora.workflow.patterns.post_debate import (
            PostDebatePattern,
            PostDebateConfig,
        )

        config = PostDebateConfig(workspace_id="ws_custom")
        wf = PostDebatePattern(config=config).create_workflow()
        store_step = next(s for s in wf.steps if s.id == "store_consensus")
        assert store_step.config["args"]["workspace_id"] == "ws_custom"

    def test_default_workspace_id(self):
        """Test that workspace_id defaults to 'default'."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        wf = PostDebatePattern().create_workflow()
        store_step = next(s for s in wf.steps if s.id == "store_consensus")
        assert store_step.config["args"]["workspace_id"] == "default"


# ============================================================================
# Visual Metadata Tests
# ============================================================================


class TestPostDebateVisualMetadata:
    """Tests for visual metadata in generated workflow."""

    def test_all_steps_have_visual(self):
        """Test that all steps have visual metadata."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        wf = PostDebatePattern().create_workflow()
        for step in wf.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_steps_have_categories(self):
        """Test that steps have appropriate node categories."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern
        from aragora.workflow.types import NodeCategory

        wf = PostDebatePattern().create_workflow()
        step_map = {s.id: s for s in wf.steps}

        assert step_map["extract_knowledge"].visual.category == NodeCategory.TASK
        assert step_map["store_consensus"].visual.category == NodeCategory.MEMORY
        assert step_map["generate_summary"].visual.category == NodeCategory.AGENT

    def test_transitions_have_visual(self):
        """Test that transitions have visual edge data."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern
        from aragora.workflow.types import EdgeType

        wf = PostDebatePattern().create_workflow()
        for transition in wf.transitions:
            assert transition.visual is not None
            assert transition.visual.edge_type == EdgeType.DATA_FLOW


# ============================================================================
# Step Configuration Tests
# ============================================================================


class TestPostDebateStepConfig:
    """Tests for step configuration details."""

    def test_extract_step_config(self):
        """Test extract_knowledge step configuration."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        wf = PostDebatePattern().create_workflow()
        step = next(s for s in wf.steps if s.id == "extract_knowledge")
        assert step.step_type == "function"
        assert step.config["handler"] == "extract_debate_facts"
        assert step.config["args"]["include_dissent"] is True
        assert step.config["args"]["min_confidence"] == 0.5

    def test_store_step_config(self):
        """Test store_consensus step configuration."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        wf = PostDebatePattern().create_workflow()
        step = next(s for s in wf.steps if s.id == "store_consensus")
        assert step.step_type == "function"
        assert step.config["handler"] == "store_debate_consensus"
        assert step.config["args"]["include_provenance"] is True

    def test_summary_step_config(self):
        """Test generate_summary step configuration."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        wf = PostDebatePattern().create_workflow()
        step = next(s for s in wf.steps if s.id == "generate_summary")
        assert step.step_type == "agent"
        assert "agent_type" in step.config
        assert step.config["agent_type"] == "claude"
        assert "prompt" in step.config
        assert step.config["max_tokens"] == 500

    def test_summary_step_uses_first_agent(self):
        """Test that summary step uses the first agent from the list."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        wf = PostDebatePattern(agents=["gpt4", "gemini"]).create_workflow()
        step = next(s for s in wf.steps if s.id == "generate_summary")
        assert step.config["agent_type"] == "gpt4"

    def test_complete_step_config(self):
        """Test complete step configuration."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        wf = PostDebatePattern().create_workflow()
        step = next(s for s in wf.steps if s.id == "complete")
        assert step.step_type == "transform"
        assert "transform" in step.config

    def test_notify_step_config(self):
        """Test notify step configuration with webhook."""
        from aragora.workflow.patterns.post_debate import (
            PostDebatePattern,
            PostDebateConfig,
        )

        config = PostDebateConfig(notify_webhook="https://hooks.example.com/notify")
        wf = PostDebatePattern(config=config).create_workflow()
        step = next(s for s in wf.steps if s.id == "notify")
        assert step.step_type == "function"
        assert step.config["handler"] == "webhook_notify"
        assert step.config["args"]["url"] == "https://hooks.example.com/notify"
        assert step.config["args"]["include_summary"] is True


# ============================================================================
# Factory Method and Helper Tests
# ============================================================================


class TestPostDebateFactory:
    """Tests for factory methods and helpers."""

    def test_create_classmethod(self):
        """Test PostDebatePattern.create factory method."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern
        from aragora.workflow.types import WorkflowDefinition

        wf = PostDebatePattern.create(
            name="Factory Test",
            store_consensus=True,
            extract_facts=False,
        )
        assert isinstance(wf, WorkflowDefinition)
        step_ids = [s.id for s in wf.steps]
        assert "extract_knowledge" not in step_ids
        assert "store_consensus" in step_ids

    def test_create_with_all_options(self):
        """Test create with all configuration options."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern

        wf = PostDebatePattern.create(
            name="Full Options",
            store_consensus=True,
            extract_facts=True,
            generate_summary=True,
            notify_webhook="https://example.com/hook",
            workspace_id="ws_test",
        )
        step_ids = [s.id for s in wf.steps]
        assert "notify" in step_ids
        assert len(wf.steps) == 5

    def test_get_default_post_debate_workflow(self):
        """Test get_default_post_debate_workflow helper."""
        from aragora.workflow.patterns.post_debate import get_default_post_debate_workflow
        from aragora.workflow.types import WorkflowDefinition

        wf = get_default_post_debate_workflow()
        assert isinstance(wf, WorkflowDefinition)
        assert wf.name == "Default Post-Debate Workflow"
        step_ids = [s.id for s in wf.steps]
        assert "extract_knowledge" in step_ids
        assert "store_consensus" in step_ids
        assert "generate_summary" in step_ids

    def test_get_default_with_workspace_id(self):
        """Test get_default_post_debate_workflow with workspace_id."""
        from aragora.workflow.patterns.post_debate import get_default_post_debate_workflow

        wf = get_default_post_debate_workflow(workspace_id="ws_custom")
        store_step = next(s for s in wf.steps if s.id == "store_consensus")
        assert store_step.config["args"]["workspace_id"] == "ws_custom"

    def test_pattern_type_is_custom(self):
        """Test that pattern_type is CUSTOM."""
        from aragora.workflow.patterns.post_debate import PostDebatePattern
        from aragora.workflow.patterns.base import PatternType

        assert PostDebatePattern.pattern_type == PatternType.CUSTOM
