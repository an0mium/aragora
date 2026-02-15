"""
Tests for Workflow Memory Nodes (MemoryReadStep and MemoryWriteStep).

Tests cover:
- MemoryReadStep initialization and config handling
- MemoryReadStep query execution with mocked KnowledgeMound
- MemoryReadStep empty query handling
- MemoryReadStep ImportError and Exception handling
- MemoryReadStep _interpolate_query with inputs, step outputs, and state
- MemoryReadStep config defaults (hybrid, limit 10)
- MemoryReadStep tenant_id resolution from config and context.metadata
- MemoryWriteStep initialization and config handling
- MemoryWriteStep content execution with mocked KnowledgeMound
- MemoryWriteStep empty content handling
- MemoryWriteStep source_type parsing (FACT, CONSENSUS, invalid->FACT)
- MemoryWriteStep relationship parsing (supports, contradicts, derived_from)
- MemoryWriteStep ImportError and Exception handling
- MemoryWriteStep _interpolate_content with dict step outputs
- MemoryWriteStep workflow metadata injection
- Config merge with current_step_config for both steps
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Helpers
# ============================================================================


def _make_context(
    inputs=None,
    state=None,
    step_outputs=None,
    metadata=None,
    current_step_config=None,
    workflow_id="wf_test",
    current_step_id="step_test",
):
    from aragora.workflow.step import WorkflowContext

    return WorkflowContext(
        workflow_id=workflow_id,
        definition_id="def_test",
        inputs=inputs or {},
        state=state or {},
        step_outputs=step_outputs or {},
        metadata=metadata or {},
        current_step_config=current_step_config or {},
        current_step_id=current_step_id,
    )


def _mock_query_result(items=None, total_count=None, execution_time_ms=50):
    """Create a mock query result with items that have to_dict()."""
    mock_items = []
    for item_data in items or []:
        mock_item = MagicMock()
        mock_item.to_dict.return_value = item_data
        mock_items.append(mock_item)

    result = MagicMock()
    result.items = mock_items
    result.total_count = total_count if total_count is not None else len(mock_items)
    result.execution_time_ms = execution_time_ms
    return result


def _mock_ingestion_result(
    success=True,
    node_id="km_001",
    deduplicated=False,
    existing_node_id=None,
    relationships_created=0,
):
    """Create a mock ingestion result."""
    result = MagicMock()
    result.success = success
    result.node_id = node_id
    result.deduplicated = deduplicated
    result.existing_node_id = existing_node_id
    result.relationships_created = relationships_created
    return result


# ============================================================================
# MemoryReadStep Initialization Tests
# ============================================================================


class TestMemoryReadStepInit:
    """Tests for MemoryReadStep initialization."""

    def test_basic_init(self):
        """Test basic MemoryReadStep initialization."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Read Knowledge")
        assert step.name == "Read Knowledge"
        assert step.config == {}

    def test_init_with_config(self):
        """Test MemoryReadStep initialization with config."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        config = {
            "query": "What are the key findings?",
            "query_type": "semantic",
            "limit": 5,
        }
        step = MemoryReadStep(name="Semantic Search", config=config)
        assert step.config["query"] == "What are the key findings?"
        assert step.config["query_type"] == "semantic"
        assert step.config["limit"] == 5

    def test_init_none_config(self):
        """Test MemoryReadStep with None config defaults to empty dict."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Empty Config", config=None)
        assert step.config == {}

    def test_init_full_config(self):
        """Test MemoryReadStep with all config options."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        config = {
            "query": "test query",
            "query_type": "keyword",
            "sources": ["fact", "consensus"],
            "domain_filter": "legal/contracts",
            "min_confidence": 0.7,
            "limit": 20,
            "include_graph": True,
            "graph_depth": 3,
            "tenant_id": "tenant_abc",
        }
        step = MemoryReadStep(name="Full Config", config=config)
        assert step.config == config


# ============================================================================
# MemoryReadStep Query Interpolation Tests
# ============================================================================


class TestMemoryReadStepInterpolation:
    """Tests for MemoryReadStep._interpolate_query."""

    def test_interpolate_with_inputs(self):
        """Test query interpolation with input values."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context(inputs={"topic": "machine learning", "domain": "AI"})
        result = step._interpolate_query("Find resources about {topic} in {domain}", ctx)
        assert result == "Find resources about machine learning in AI"

    def test_interpolate_with_step_outputs_string(self):
        """Test query interpolation with string step outputs."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context(step_outputs={"analysis": "contract termination clauses"})
        result = step._interpolate_query("Expand on {step.analysis}", ctx)
        assert result == "Expand on contract termination clauses"

    def test_interpolate_with_step_outputs_dict_response(self):
        """Test query interpolation with dict step outputs containing 'response'."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context(step_outputs={"prev": {"response": "key legal findings"}})
        result = step._interpolate_query("Summarize {step.prev}", ctx)
        assert result == "Summarize key legal findings"

    def test_interpolate_with_step_outputs_dict_no_response(self):
        """Test that dict step outputs without 'response' are not interpolated."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context(step_outputs={"prev": {"data": "something"}})
        result = step._interpolate_query("Check {step.prev}", ctx)
        assert result == "Check {step.prev}"

    def test_interpolate_with_state(self):
        """Test query interpolation with state values."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context(state={"current_topic": "contracts", "iteration": "3"})
        result = step._interpolate_query("Iteration {state.iteration}: {state.current_topic}", ctx)
        assert result == "Iteration 3: contracts"

    def test_interpolate_mixed_sources(self):
        """Test query interpolation mixing inputs, step outputs, and state."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context(
            inputs={"user_query": "contracts"},
            step_outputs={"refine": "termination clauses"},
            state={"domain": "legal"},
        )
        result = step._interpolate_query("{user_query} about {step.refine} in {state.domain}", ctx)
        assert result == "contracts about termination clauses in legal"

    def test_interpolate_no_placeholders(self):
        """Test query interpolation with no placeholders."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context()
        result = step._interpolate_query("plain query string", ctx)
        assert result == "plain query string"

    def test_interpolate_empty_template(self):
        """Test query interpolation with empty template."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context()
        result = step._interpolate_query("", ctx)
        assert result == ""

    def test_interpolate_missing_key_preserved(self):
        """Test that unresolved placeholders are preserved."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Test")
        ctx = _make_context(inputs={"existing": "value"})
        result = step._interpolate_query("{existing} and {missing}", ctx)
        assert result == "value and {missing}"


# ============================================================================
# MemoryReadStep Execution Tests
# ============================================================================


class TestMemoryReadStepExecution:
    """Tests for MemoryReadStep.execute."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_result(self):
        """Test that an empty query returns an empty result set."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="Empty Query", config={"query": ""})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["items"] == []
        assert result["total_count"] == 0
        assert result["query"] == ""

    @pytest.mark.asyncio
    async def test_empty_query_no_config(self):
        """Test that missing query config returns empty result."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(name="No Query Config", config={})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["items"] == []
        assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_successful_query(self):
        """Test successful query with mocked KnowledgeMound."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Search",
            config={"query": "Find contracts", "limit": 5},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_query_result(
            items=[{"id": "km_1", "content": "Contract A"}],
            total_count=1,
            execution_time_ms=42,
        )
        mock_mound.query.return_value = mock_result

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_request_class = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=mock_request_class,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert result["items"] == [{"id": "km_1", "content": "Contract A"}]
        assert result["total_count"] == 1
        assert result["query"] == "Find contracts"
        assert result["execution_time_ms"] == 42

    @pytest.mark.asyncio
    async def test_query_with_interpolation(self):
        """Test that query template is interpolated before querying."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Interp Search",
            config={"query": "Find {topic} documents"},
        )
        ctx = _make_context(inputs={"topic": "legal"})

        mock_mound = AsyncMock()
        mock_result = _mock_query_result(items=[], total_count=0)
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_request_class = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=mock_request_class,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert result["query"] == "Find legal documents"

    @pytest.mark.asyncio
    async def test_import_error_returns_error_dict(self):
        """Test that ImportError returns an error dict."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Import Fail",
            config={"query": "test query"},
        )
        ctx = _make_context()

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.knowledge.mound":
                raise ImportError("No module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await step.execute(ctx)

        assert result["items"] == []
        assert result["total_count"] == 0
        assert result["query"] == "test query"
        assert "Knowledge Mound not available" in result["error"]

    @pytest.mark.asyncio
    async def test_general_exception_returns_error_dict(self):
        """Test that general exceptions return an error dict."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Exception Step",
            config={"query": "test query"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_mound.query.side_effect = RuntimeError("Database connection lost")

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_request_class = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=mock_request_class,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert result["items"] == []
        assert result["total_count"] == 0
        assert result["error"] == "Memory read failed"

    @pytest.mark.asyncio
    async def test_default_query_type_is_hybrid(self):
        """Test that the default query_type is 'hybrid'."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Default Type",
            config={"query": "test"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            return MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=capture_request,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("search_mode") == "hybrid"

    @pytest.mark.asyncio
    async def test_default_limit_is_10(self):
        """Test that the default limit is 10."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Default Limit",
            config={"query": "test"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            return MagicMock(tenant_id="default", limit=10, query="test")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=capture_request,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("limit") == 10

    @pytest.mark.asyncio
    async def test_tenant_id_from_config(self):
        """Test that tenant_id is read from config."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Tenant Config",
            config={"query": "test", "tenant_id": "tenant_from_config"},
        )
        ctx = _make_context(metadata={"tenant_id": "tenant_from_metadata"})

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            return MagicMock(tenant_id=kwargs.get("tenant_id"), limit=10, query="test")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=capture_request,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("tenant_id") == "tenant_from_config"

    @pytest.mark.asyncio
    async def test_tenant_id_from_metadata(self):
        """Test that tenant_id falls back to context.metadata."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Tenant Metadata",
            config={"query": "test"},
        )
        ctx = _make_context(metadata={"tenant_id": "tenant_from_metadata"})

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            return MagicMock(tenant_id=kwargs.get("tenant_id"), limit=10, query="test")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=capture_request,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("tenant_id") == "tenant_from_metadata"

    @pytest.mark.asyncio
    async def test_tenant_id_default(self):
        """Test that tenant_id defaults to 'default' when not set anywhere."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Tenant Default",
            config={"query": "test"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            return MagicMock(tenant_id=kwargs.get("tenant_id"), limit=10, query="test")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=capture_request,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("tenant_id") == "default"

    @pytest.mark.asyncio
    async def test_config_merged_with_current_step_config(self):
        """Test that step config is merged with current_step_config."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Merge Config",
            config={"query": "original query", "limit": 5},
        )
        ctx = _make_context(current_step_config={"query": "overridden query"})

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            return MagicMock(tenant_id="default", limit=5, query="overridden query")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=capture_request,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert result["query"] == "overridden query"

    @pytest.mark.asyncio
    async def test_sources_passed_to_query(self):
        """Test that sources config is forwarded to mound.query."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="With Sources",
            config={"query": "test", "sources": ["fact", "consensus"]},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_request_class = MagicMock(
            return_value=MagicMock(tenant_id="default", limit=10, query="test")
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=mock_request_class,
                ),
            },
        ):
            await step.execute(ctx)

        mock_mound.query.assert_called_once()
        call_kwargs = mock_mound.query.call_args
        assert call_kwargs.kwargs.get("sources") == ["fact", "consensus"]

    @pytest.mark.asyncio
    async def test_mound_initialize_called(self):
        """Test that KnowledgeMound.initialize() is called."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Init Check",
            config={"query": "test"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_request_class = MagicMock(
            return_value=MagicMock(tenant_id="default", limit=10, query="test")
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=mock_request_class,
                ),
            },
        ):
            await step.execute(ctx)

        mock_mound.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multiple_items_returned(self):
        """Test that multiple items are returned correctly."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Multi Results",
            config={"query": "find all"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        items_data = [
            {"id": "km_1", "content": "Item 1"},
            {"id": "km_2", "content": "Item 2"},
            {"id": "km_3", "content": "Item 3"},
        ]
        mock_result = _mock_query_result(items=items_data, total_count=3)
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_request_class = MagicMock(
            return_value=MagicMock(tenant_id="default", limit=10, query="find all")
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=mock_request_class,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert len(result["items"]) == 3
        assert result["total_count"] == 3
        assert result["items"][0] == {"id": "km_1", "content": "Item 1"}
        assert result["items"][2] == {"id": "km_3", "content": "Item 3"}

    @pytest.mark.asyncio
    async def test_include_graph_config(self):
        """Test that include_graph and graph_depth are passed to request."""
        from aragora.workflow.nodes.memory import MemoryReadStep

        step = MemoryReadStep(
            name="Graph Search",
            config={
                "query": "test",
                "include_graph": True,
                "graph_depth": 3,
            },
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_query_result()
        mock_mound.query.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            return MagicMock(tenant_id="default", limit=10, query="test")

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                ),
                "aragora.knowledge.mound.types": MagicMock(
                    UnifiedQueryRequest=capture_request,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("include_graph") is True
        assert captured_request.get("graph_depth") == 3


# ============================================================================
# MemoryWriteStep Initialization Tests
# ============================================================================


class TestMemoryWriteStepInit:
    """Tests for MemoryWriteStep initialization."""

    def test_basic_init(self):
        """Test basic MemoryWriteStep initialization."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Write Knowledge")
        assert step.name == "Write Knowledge"
        assert step.config == {}

    def test_init_with_config(self):
        """Test MemoryWriteStep initialization with config."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        config = {
            "content": "Analysis result: {analysis}",
            "source_type": "consensus",
            "domain": "legal/contracts",
            "confidence": 0.85,
        }
        step = MemoryWriteStep(name="Store Analysis", config=config)
        assert step.config["content"] == "Analysis result: {analysis}"
        assert step.config["source_type"] == "consensus"
        assert step.config["confidence"] == 0.85

    def test_init_none_config(self):
        """Test MemoryWriteStep with None config defaults to empty dict."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Empty Config", config=None)
        assert step.config == {}

    def test_init_full_config(self):
        """Test MemoryWriteStep with all config options."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        config = {
            "content": "test content",
            "source_type": "fact",
            "domain": "engineering/backend",
            "confidence": 0.9,
            "importance": 0.8,
            "relationships": [{"type": "supports", "target": "km_123"}],
            "tenant_id": "tenant_xyz",
            "deduplicate": True,
            "metadata": {"author": "system"},
        }
        step = MemoryWriteStep(name="Full Config", config=config)
        assert step.config == config


# ============================================================================
# MemoryWriteStep Content Interpolation Tests
# ============================================================================


class TestMemoryWriteStepInterpolation:
    """Tests for MemoryWriteStep._interpolate_content."""

    def test_interpolate_with_inputs(self):
        """Test content interpolation with input values."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context(inputs={"analysis": "Contract is valid"})
        result = step._interpolate_content("Result: {analysis}", ctx)
        assert result == "Result: Contract is valid"

    def test_interpolate_with_step_outputs_string(self):
        """Test content interpolation with string step outputs."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context(step_outputs={"summarize": "brief summary"})
        result = step._interpolate_content("Summary: {step.summarize}", ctx)
        assert result == "Summary: brief summary"

    def test_interpolate_with_step_outputs_dict_response(self):
        """Test content interpolation with dict step output containing 'response'."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context(step_outputs={"agent_step": {"response": "Agent generated this"}})
        result = step._interpolate_content("{step.agent_step}", ctx)
        assert result == "Agent generated this"

    def test_interpolate_with_step_outputs_dict_content(self):
        """Test content interpolation with dict step output containing 'content'."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context(step_outputs={"fetch_step": {"content": "Fetched content"}})
        result = step._interpolate_content("{step.fetch_step}", ctx)
        assert result == "Fetched content"

    def test_interpolate_with_step_outputs_dict_result(self):
        """Test content interpolation with dict step output containing 'result'."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context(step_outputs={"compute_step": {"result": "Computed value"}})
        result = step._interpolate_content("{step.compute_step}", ctx)
        assert result == "Computed value"

    def test_interpolate_dict_response_priority(self):
        """Test that 'response' key takes priority over 'content' and 'result'."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context(
            step_outputs={
                "multi_key": {
                    "response": "from response",
                    "content": "from content",
                    "result": "from result",
                }
            }
        )
        result = step._interpolate_content("{step.multi_key}", ctx)
        assert result == "from response"

    def test_interpolate_dict_no_known_keys(self):
        """Test that dict step outputs without known keys are not interpolated."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context(step_outputs={"unknown": {"data": "value", "info": "extra"}})
        result = step._interpolate_content("Output: {step.unknown}", ctx)
        assert result == "Output: {step.unknown}"

    def test_interpolate_with_state(self):
        """Test content interpolation with state values."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context(state={"session_id": "sess_123"})
        result = step._interpolate_content("Session: {state.session_id}", ctx)
        assert result == "Session: sess_123"

    def test_interpolate_empty_template(self):
        """Test content interpolation with empty template."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Test")
        ctx = _make_context()
        result = step._interpolate_content("", ctx)
        assert result == ""


# ============================================================================
# MemoryWriteStep Execution Tests
# ============================================================================


class TestMemoryWriteStepExecution:
    """Tests for MemoryWriteStep.execute."""

    @pytest.mark.asyncio
    async def test_empty_content_returns_error(self):
        """Test that empty content returns an error dict."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="Empty Write", config={"content": ""})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["success"] is False
        assert "Empty content" in result["error"]

    @pytest.mark.asyncio
    async def test_no_content_config_returns_error(self):
        """Test that missing content config returns an error."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(name="No Content", config={})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["success"] is False
        assert "Empty content" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_write(self):
        """Test successful write with mocked KnowledgeMound."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Store Data",
            config={
                "content": "Important finding",
                "source_type": "fact",
                "domain": "research",
                "confidence": 0.9,
            },
        )
        ctx = _make_context(workflow_id="wf_42", current_step_id="step_7")

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result(
            success=True,
            node_id="km_999",
            deduplicated=False,
            relationships_created=0,
        )
        mock_mound.store.return_value = mock_result

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")
        mock_request_class = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["node_id"] == "km_999"
        assert result["deduplicated"] is False
        assert result["existing_node_id"] is None
        assert result["relationships_created"] == 0

    @pytest.mark.asyncio
    async def test_write_with_interpolated_content(self):
        """Test that content template is interpolated before writing."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Interp Write",
            config={"content": "Finding: {result_text}"},
        )
        ctx = _make_context(inputs={"result_text": "Contract is valid"})

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("content") == "Finding: Contract is valid"

    @pytest.mark.asyncio
    async def test_source_type_fact(self):
        """Test parsing source_type as FACT."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Fact Write",
            config={"content": "A fact", "source_type": "fact"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        mock_source_enum = MagicMock()
        fact_value = MagicMock(name="FACT")
        mock_source_enum.__getitem__ = MagicMock(return_value=fact_value)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        mock_source_enum.__getitem__.assert_called_with("FACT")
        assert captured_request.get("source_type") == fact_value

    @pytest.mark.asyncio
    async def test_source_type_consensus(self):
        """Test parsing source_type as CONSENSUS."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Consensus Write",
            config={"content": "Consensus result", "source_type": "consensus"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        mock_source_enum = MagicMock()
        consensus_value = MagicMock(name="CONSENSUS")
        mock_source_enum.__getitem__ = MagicMock(return_value=consensus_value)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        mock_source_enum.__getitem__.assert_called_with("CONSENSUS")

    @pytest.mark.asyncio
    async def test_source_type_invalid_falls_back_to_fact(self):
        """Test that an invalid source_type falls back to FACT."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Invalid Source",
            config={"content": "Something", "source_type": "nonexistent_type"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)

        mock_source_enum = MagicMock()
        fact_value = MagicMock(name="FACT")
        mock_source_enum.FACT = fact_value

        def enum_getitem(key):
            if key == "NONEXISTENT_TYPE":
                raise KeyError(key)
            return fact_value

        mock_source_enum.__getitem__ = MagicMock(side_effect=enum_getitem)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("source_type") == fact_value

    @pytest.mark.asyncio
    async def test_relationship_supports(self):
        """Test relationship parsing for 'supports' type."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Supports Rel",
            config={
                "content": "Supporting evidence",
                "relationships": [{"type": "supports", "target": "km_100"}],
            },
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert "km_100" in mock_request.supports

    @pytest.mark.asyncio
    async def test_relationship_contradicts(self):
        """Test relationship parsing for 'contradicts' type."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Contradicts Rel",
            config={
                "content": "Contradicting evidence",
                "relationships": [{"type": "contradicts", "target": "km_200"}],
            },
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert "km_200" in mock_request.contradicts

    @pytest.mark.asyncio
    async def test_relationship_derived_from(self):
        """Test relationship parsing for 'derived_from' type."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Derived Rel",
            config={
                "content": "Derived conclusion",
                "relationships": [{"type": "derived_from", "target": "km_300"}],
            },
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert "km_300" in mock_request.derived_from

    @pytest.mark.asyncio
    async def test_multiple_relationships(self):
        """Test parsing multiple relationships of different types."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Multi Rels",
            config={
                "content": "Complex knowledge",
                "relationships": [
                    {"type": "supports", "target": "km_1"},
                    {"type": "contradicts", "target": "km_2"},
                    {"type": "derived_from", "target": "km_3"},
                    {"type": "supports", "target": "km_4"},
                ],
            },
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert mock_request.supports == ["km_1", "km_4"]
        assert mock_request.contradicts == ["km_2"]
        assert mock_request.derived_from == ["km_3"]

    @pytest.mark.asyncio
    async def test_relationship_target_interpolated(self):
        """Test that relationship targets are interpolated."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Interp Rel",
            config={
                "content": "Related finding",
                "relationships": [{"type": "derived_from", "target": "{source_doc_id}"}],
            },
        )
        ctx = _make_context(inputs={"source_doc_id": "km_orig_42"})

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert "km_orig_42" in mock_request.derived_from

    @pytest.mark.asyncio
    async def test_relationship_empty_target_skipped(self):
        """Test that relationships with empty target are skipped."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Empty Target Rel",
            config={
                "content": "Some content",
                "relationships": [{"type": "supports", "target": ""}],
            },
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert mock_request.supports == []

    @pytest.mark.asyncio
    async def test_import_error_returns_error_dict(self):
        """Test that ImportError returns an error dict for write."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Import Fail Write",
            config={"content": "test content"},
        )
        ctx = _make_context()

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.knowledge.mound":
                raise ImportError("No module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await step.execute(ctx)

        assert result["success"] is False
        assert "Knowledge Mound not available" in result["error"]

    @pytest.mark.asyncio
    async def test_general_exception_returns_error_dict(self):
        """Test that general exceptions return an error dict for write."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Exception Write",
            config={"content": "test content"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_mound.store.side_effect = RuntimeError("Storage engine crashed")

        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert result["success"] is False
        assert result["error"] == "Memory write failed"

    @pytest.mark.asyncio
    async def test_workflow_metadata_injected(self):
        """Test that workflow_id and step_id are injected into metadata."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Metadata Write",
            config={"content": "test content"},
        )
        ctx = _make_context(workflow_id="wf_meta_test", current_step_id="step_meta_7")

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        metadata = captured_request.get("metadata", {})
        assert metadata.get("workflow_id") == "wf_meta_test"
        assert metadata.get("step_id") == "step_meta_7"

    @pytest.mark.asyncio
    async def test_custom_metadata_merged(self):
        """Test that custom metadata is merged with workflow metadata."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Custom Meta Write",
            config={
                "content": "test content",
                "metadata": {"author": "system", "priority": "high"},
            },
        )
        ctx = _make_context(workflow_id="wf_123", current_step_id="step_456")

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        metadata = captured_request.get("metadata", {})
        assert metadata.get("workflow_id") == "wf_123"
        assert metadata.get("step_id") == "step_456"
        assert metadata.get("author") == "system"
        assert metadata.get("priority") == "high"

    @pytest.mark.asyncio
    async def test_tenant_id_from_config(self):
        """Test tenant_id comes from config when specified."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Tenant Write",
            config={"content": "data", "tenant_id": "tenant_cfg"},
        )
        ctx = _make_context(metadata={"tenant_id": "tenant_meta"})

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("workspace_id") == "tenant_cfg"

    @pytest.mark.asyncio
    async def test_tenant_id_from_metadata(self):
        """Test tenant_id falls back to context.metadata."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Tenant Meta Write",
            config={"content": "data"},
        )
        ctx = _make_context(metadata={"tenant_id": "tenant_meta"})

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("workspace_id") == "tenant_meta"

    @pytest.mark.asyncio
    async def test_tenant_id_default(self):
        """Test tenant_id defaults to 'default'."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Tenant Default Write",
            config={"content": "data"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("workspace_id") == "default"

    @pytest.mark.asyncio
    async def test_domain_defaults_to_general(self):
        """Test that domain defaults to 'general' when not specified."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Default Domain",
            config={"content": "data"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("topics") == ["general"]

    @pytest.mark.asyncio
    async def test_confidence_default(self):
        """Test that confidence defaults to 0.5."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Default Confidence",
            config={"content": "data"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("confidence") == 0.5

    @pytest.mark.asyncio
    async def test_mound_initialize_called(self):
        """Test that KnowledgeMound.initialize() is called for write."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Init Check Write",
            config={"content": "data"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        mock_mound.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deduplicated_result(self):
        """Test write result when deduplication is detected."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Dedup Write",
            config={"content": "duplicate content", "deduplicate": True},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result(
            success=True,
            node_id="km_new",
            deduplicated=True,
            existing_node_id="km_existing",
            relationships_created=0,
        )
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["deduplicated"] is True
        assert result["existing_node_id"] == "km_existing"

    @pytest.mark.asyncio
    async def test_config_merged_with_current_step_config(self):
        """Test that step config is merged with current_step_config for write."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Merge Config Write",
            config={"content": "original content", "confidence": 0.5},
        )
        ctx = _make_context(
            current_step_config={"content": "overridden content", "domain": "override_domain"}
        )

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        assert captured_request.get("content") == "overridden content"
        assert captured_request.get("topics") == ["override_domain"]
        assert captured_request.get("confidence") == 0.5

    @pytest.mark.asyncio
    async def test_default_source_type_is_fact(self):
        """Test that default source_type is 'fact' when not specified."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Default Source",
            config={"content": "data"},
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result()
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        fact_value = MagicMock(name="FACT")
        mock_source_enum.__getitem__ = MagicMock(return_value=fact_value)

        captured_request = {}

        def capture_request(**kwargs):
            captured_request.update(kwargs)
            req = MagicMock()
            req.supports = []
            req.contradicts = []
            req.derived_from = []
            return req

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=capture_request,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            await step.execute(ctx)

        mock_source_enum.__getitem__.assert_called_with("FACT")

    @pytest.mark.asyncio
    async def test_relationships_created_count(self):
        """Test that relationships_created count is returned."""
        from aragora.workflow.nodes.memory import MemoryWriteStep

        step = MemoryWriteStep(
            name="Rel Count",
            config={
                "content": "data",
                "relationships": [
                    {"type": "supports", "target": "km_1"},
                    {"type": "contradicts", "target": "km_2"},
                ],
            },
        )
        ctx = _make_context()

        mock_mound = AsyncMock()
        mock_result = _mock_ingestion_result(
            success=True, node_id="km_new", relationships_created=2
        )
        mock_mound.store.return_value = mock_result
        mock_km_class = MagicMock(return_value=mock_mound)
        mock_source_enum = MagicMock()
        mock_source_enum.__getitem__ = MagicMock(return_value="FACT")

        mock_request = MagicMock()
        mock_request.supports = []
        mock_request.contradicts = []
        mock_request.derived_from = []
        mock_request_class = MagicMock(return_value=mock_request)

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": MagicMock(
                    KnowledgeMound=mock_km_class,
                    IngestionRequest=mock_request_class,
                    KnowledgeSource=mock_source_enum,
                ),
            },
        ):
            result = await step.execute(ctx)

        assert result["relationships_created"] == 2
