"""
Tests for Workflow Connector Node.

Tests cover:
- ConnectorOperation enum values
- ConnectorMetadata dataclass
- Connector registry (register, get, list)
- create_connector dynamic creation
- ConnectorStep initialization and config
- ConnectorStep.execute with various operations (search, fetch, list, create, update, delete, sync, custom)
- Credential resolution (direct, context state, workflow inputs, env fallback)
- Parameter interpolation (inputs, step outputs, state, nested dicts, lists)
- Error handling and retries (timeout, retryable errors, non-retryable errors)
- Result formatting (to_dict, list of Evidence, plain results)
- Connector caching
- Connector alias
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# ConnectorOperation Tests
# ============================================================================


class TestConnectorOperation:
    """Tests for ConnectorOperation enum."""

    def test_operation_values(self):
        """Test all expected operation values exist."""
        from aragora.workflow.nodes.connector import ConnectorOperation

        assert ConnectorOperation.SEARCH.value == "search"
        assert ConnectorOperation.FETCH.value == "fetch"
        assert ConnectorOperation.LIST.value == "list"
        assert ConnectorOperation.CREATE.value == "create"
        assert ConnectorOperation.UPDATE.value == "update"
        assert ConnectorOperation.DELETE.value == "delete"
        assert ConnectorOperation.SYNC.value == "sync"
        assert ConnectorOperation.CUSTOM.value == "custom"

    def test_operation_is_string_enum(self):
        """Test that ConnectorOperation is a string enum."""
        from aragora.workflow.nodes.connector import ConnectorOperation

        assert isinstance(ConnectorOperation.SEARCH, str)
        assert ConnectorOperation.SEARCH == "search"


# ============================================================================
# ConnectorMetadata Tests
# ============================================================================


class TestConnectorMetadata:
    """Tests for ConnectorMetadata dataclass."""

    def test_metadata_defaults(self):
        """Test default values for ConnectorMetadata."""
        from aragora.workflow.nodes.connector import ConnectorMetadata

        meta = ConnectorMetadata(
            name="Test",
            description="Test connector",
            module_path="test.module",
            class_name="TestConnector",
            operations=["search", "fetch"],
        )
        assert meta.auth_required is True
        assert meta.auth_type == "api_key"

    def test_metadata_no_auth(self):
        """Test ConnectorMetadata with auth disabled."""
        from aragora.workflow.nodes.connector import ConnectorMetadata

        meta = ConnectorMetadata(
            name="Public",
            description="Public connector",
            module_path="test.public",
            class_name="PublicConnector",
            operations=["search"],
            auth_required=False,
        )
        assert meta.auth_required is False

    def test_metadata_oauth2(self):
        """Test ConnectorMetadata with OAuth2 auth type."""
        from aragora.workflow.nodes.connector import ConnectorMetadata

        meta = ConnectorMetadata(
            name="OAuth",
            description="OAuth connector",
            module_path="test.oauth",
            class_name="OAuthConnector",
            operations=["fetch"],
            auth_type="oauth2",
        )
        assert meta.auth_type == "oauth2"


# ============================================================================
# Registry Tests
# ============================================================================


class TestConnectorRegistry:
    """Tests for connector registry functions."""

    def test_get_known_connector(self):
        """Test getting metadata for a known connector."""
        from aragora.workflow.nodes.connector import get_connector_metadata

        meta = get_connector_metadata("github")
        assert meta is not None
        assert meta.name == "GitHub"
        assert "search" in meta.operations
        assert "fetch" in meta.operations

    def test_get_unknown_connector(self):
        """Test getting metadata for an unknown connector returns None."""
        from aragora.workflow.nodes.connector import get_connector_metadata

        meta = get_connector_metadata("nonexistent_connector_xyz")
        assert meta is None

    def test_list_connectors(self):
        """Test listing all registered connectors."""
        from aragora.workflow.nodes.connector import list_connectors

        connectors = list_connectors()
        assert len(connectors) > 0
        names = [c.name for c in connectors]
        assert "GitHub" in names
        assert "Slack" in names

    def test_register_connector(self):
        """Test registering a new connector type."""
        from aragora.workflow.nodes.connector import (
            ConnectorMetadata,
            get_connector_metadata,
            register_connector,
            _CONNECTOR_REGISTRY,
        )

        test_key = "_test_custom_connector"
        try:
            meta = ConnectorMetadata(
                name="Custom",
                description="Custom connector",
                module_path="test.custom",
                class_name="CustomConnector",
                operations=["search", "create"],
            )
            register_connector(test_key, meta)
            result = get_connector_metadata(test_key)
            assert result is not None
            assert result.name == "Custom"
        finally:
            _CONNECTOR_REGISTRY.pop(test_key, None)

    def test_registry_has_web_connector(self):
        """Test that web connector is registered and has no auth required."""
        from aragora.workflow.nodes.connector import get_connector_metadata

        meta = get_connector_metadata("web")
        assert meta is not None
        assert meta.auth_required is False

    def test_registry_has_enterprise_connectors(self):
        """Test that enterprise connectors (docusign, quickbooks) are registered."""
        from aragora.workflow.nodes.connector import get_connector_metadata

        docusign = get_connector_metadata("docusign")
        assert docusign is not None
        assert docusign.auth_type == "oauth2"

        qb = get_connector_metadata("quickbooks")
        assert qb is not None
        assert "sync" in qb.operations


# ============================================================================
# create_connector Tests
# ============================================================================


class TestCreateConnector:
    """Tests for dynamic connector creation."""

    @pytest.mark.asyncio
    async def test_unknown_type_raises_value_error(self):
        """Test that unknown connector type raises ValueError."""
        from aragora.workflow.nodes.connector import create_connector

        with pytest.raises(ValueError, match="Unknown connector type"):
            await create_connector("nonexistent_type", {})

    @pytest.mark.asyncio
    async def test_import_error_wraps_exception(self):
        """Test that import failure wraps into ImportError."""
        from aragora.workflow.nodes.connector import (
            ConnectorMetadata,
            register_connector,
            create_connector,
            _CONNECTOR_REGISTRY,
        )

        test_key = "_test_bad_import"
        try:
            meta = ConnectorMetadata(
                name="Bad",
                description="Bad import",
                module_path="nonexistent.module.path",
                class_name="BadConnector",
                operations=["fetch"],
            )
            register_connector(test_key, meta)

            with pytest.raises(ImportError, match="not available"):
                await create_connector(test_key, {})
        finally:
            _CONNECTOR_REGISTRY.pop(test_key, None)

    @pytest.mark.asyncio
    async def test_successful_creation(self):
        """Test successful dynamic connector creation with mock."""
        from aragora.workflow.nodes.connector import create_connector

        mock_class = MagicMock()
        mock_module = MagicMock()
        mock_module.GitHubConnector = mock_class

        with patch("importlib.import_module", return_value=mock_module):
            result = await create_connector("github", {"token": "abc"})
            mock_class.assert_called_once_with(token="abc")


# ============================================================================
# ConnectorStep Initialization Tests
# ============================================================================


class TestConnectorStepInit:
    """Tests for ConnectorStep initialization."""

    def test_basic_init(self):
        """Test basic ConnectorStep initialization."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="Test Step",
            config={"connector_type": "github", "operation": "fetch"},
        )
        assert step.name == "Test Step"
        assert step.config["connector_type"] == "github"

    def test_default_config(self):
        """Test ConnectorStep with no config."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Empty Step")
        assert step.config == {}

    def test_connector_alias(self):
        """Test that Connector is an alias for ConnectorStep."""
        from aragora.workflow.nodes.connector import ConnectorStep, Connector

        assert Connector is ConnectorStep


# ============================================================================
# Credential Resolution Tests
# ============================================================================


class TestCredentialResolution:
    """Tests for credential resolution in ConnectorStep."""

    def _make_context(self, inputs=None, state=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
        )

    def test_direct_credentials(self):
        """Test credentials provided directly in config."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="Direct",
            config={
                "connector_type": "github",
                "credentials": {"token": "abc123"},
            },
        )
        ctx = self._make_context()
        creds = step._get_credentials(ctx)
        assert creds == {"token": "abc123"}

    def test_credentials_key_from_state_dict(self):
        """Test credentials from context state as dict."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="State",
            config={
                "connector_type": "github",
                "credentials_key": "my_creds",
            },
        )
        ctx = self._make_context(state={"my_creds": {"token": "from_state"}})
        creds = step._get_credentials(ctx)
        assert creds == {"token": "from_state"}

    def test_credentials_key_from_state_string(self):
        """Test credentials from context state as string wraps to api_key."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="State String",
            config={
                "connector_type": "github",
                "credentials_key": "my_key",
            },
        )
        ctx = self._make_context(state={"my_key": "plain_token"})
        creds = step._get_credentials(ctx)
        assert creds == {"api_key": "plain_token"}

    def test_credentials_from_inputs(self):
        """Test credentials from workflow inputs using connector_type convention."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="Inputs",
            config={"connector_type": "github"},
        )
        ctx = self._make_context(inputs={"github_credentials": {"token": "from_input"}})
        creds = step._get_credentials(ctx)
        assert creds == {"token": "from_input"}

    def test_empty_credentials_fallback(self):
        """Test that empty dict is returned when no credentials found."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="No Creds",
            config={"connector_type": "github"},
        )
        ctx = self._make_context()
        creds = step._get_credentials(ctx)
        assert creds == {}


# ============================================================================
# Parameter Interpolation Tests
# ============================================================================


class TestParameterInterpolation:
    """Tests for parameter interpolation in ConnectorStep."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
        )

    def test_interpolate_inputs(self):
        """Test interpolation of {inputs.key} placeholders."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Interp", config={})
        ctx = self._make_context(inputs={"issue_id": "42"})
        result = step._interpolate_params(
            {"issue_number": "{inputs.issue_id}", "static": "value"},
            ctx,
        )
        assert result["issue_number"] == "42"
        assert result["static"] == "value"

    def test_interpolate_step_outputs(self):
        """Test interpolation of {step.step_id.key} placeholders."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Interp", config={})
        ctx = self._make_context(step_outputs={"prev_step": {"result": "data"}})
        result = step._interpolate_params(
            {"data": "{step.prev_step.result}"},
            ctx,
        )
        assert result["data"] == "data"

    def test_interpolate_state(self):
        """Test interpolation of {state.key} placeholders."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Interp", config={})
        ctx = self._make_context(state={"workspace": "ws_123"})
        result = step._interpolate_params(
            {"workspace_id": "{state.workspace}"},
            ctx,
        )
        assert result["workspace_id"] == "ws_123"

    def test_interpolate_nested_dict(self):
        """Test interpolation of nested dictionary params."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Interp", config={})
        ctx = self._make_context(inputs={"repo": "myrepo"})
        result = step._interpolate_params(
            {"config": {"repo_name": "{inputs.repo}"}},
            ctx,
        )
        assert result["config"]["repo_name"] == "myrepo"

    def test_interpolate_list_params(self):
        """Test interpolation of list params."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Interp", config={})
        ctx = self._make_context(inputs={"tag": "urgent"})
        result = step._interpolate_params(
            {"tags": ["{inputs.tag}", "static"]},
            ctx,
        )
        assert result["tags"] == ["urgent", "static"]

    def test_interpolate_non_string_preserved(self):
        """Test that non-string values are preserved unchanged."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Interp", config={})
        ctx = self._make_context()
        result = step._interpolate_params(
            {"count": 5, "active": True, "ratio": 0.7},
            ctx,
        )
        assert result["count"] == 5
        assert result["active"] is True
        assert result["ratio"] == 0.7

    def test_interpolate_full_match_returns_actual_value(self):
        """Test that a full placeholder match returns the actual value, not stringified."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Interp", config={})
        ctx = self._make_context(inputs={"data": {"key": "value"}})
        result = step._interpolate_params(
            {"payload": "{inputs.data}"},
            ctx,
        )
        # Should return the dict itself, not its string repr
        assert result["payload"] == {"key": "value"}

    def test_interpolate_missing_placeholder_kept(self):
        """Test that missing placeholders are kept as-is."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Interp", config={})
        ctx = self._make_context()
        result = step._interpolate_params(
            {"query": "{inputs.missing}"},
            ctx,
        )
        assert result["query"] == "{inputs.missing}"


# ============================================================================
# Execute Operation Tests
# ============================================================================


class TestExecuteOperation:
    """Tests for ConnectorStep._execute_operation."""

    @pytest.mark.asyncio
    async def test_search_operation(self):
        """Test search operation calls connector.search."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Search", config={})
        connector = AsyncMock()
        connector.search.return_value = [{"title": "Result"}]

        result = await step._execute_operation(connector, "search", {"query": "test", "limit": 10})
        connector.search.assert_called_once_with("test", limit=10)
        assert result == [{"title": "Result"}]

    @pytest.mark.asyncio
    async def test_fetch_with_source_id(self):
        """Test fetch operation with source_id."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Fetch", config={})
        connector = AsyncMock()
        connector.fetch.return_value = {"data": "found"}

        result = await step._execute_operation(connector, "fetch", {"source_id": "123"})
        connector.fetch.assert_called_once_with("123")

    @pytest.mark.asyncio
    async def test_fetch_with_kwargs(self):
        """Test fetch operation with keyword arguments when no source_id."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Fetch", config={})
        connector = AsyncMock()
        connector.fetch.return_value = {"data": "found"}

        result = await step._execute_operation(
            connector, "fetch", {"owner": "org", "repo": "myrepo"}
        )
        connector.fetch.assert_called_once_with(owner="org", repo="myrepo")

    @pytest.mark.asyncio
    async def test_list_operation(self):
        """Test list operation."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="List", config={})
        connector = AsyncMock()
        connector.list = AsyncMock(return_value=[1, 2, 3])

        result = await step._execute_operation(connector, "list", {"page": 1})
        connector.list.assert_called_once_with(page=1)

    @pytest.mark.asyncio
    async def test_list_all_fallback(self):
        """Test list operation falls back to list_all."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="List", config={})
        connector = MagicMock(spec=[])
        connector.list_all = AsyncMock(return_value=[1, 2])
        # Ensure 'list' is missing but 'list_all' exists
        assert not hasattr(connector, "list")
        connector.list_all = AsyncMock(return_value=[1, 2])

        result = await step._execute_operation(connector, "list", {})
        connector.list_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_operation(self):
        """Test create operation."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Create", config={})
        connector = AsyncMock()
        connector.create.return_value = {"id": "new_123"}

        result = await step._execute_operation(connector, "create", {"name": "Test"})
        connector.create.assert_called_once_with(name="Test")

    @pytest.mark.asyncio
    async def test_update_operation(self):
        """Test update operation."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Update", config={})
        connector = AsyncMock()
        connector.update.return_value = {"updated": True}

        result = await step._execute_operation(
            connector, "update", {"id": "123", "status": "closed"}
        )
        connector.update.assert_called_once_with(id="123", status="closed")

    @pytest.mark.asyncio
    async def test_delete_operation(self):
        """Test delete operation."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Delete", config={})
        connector = AsyncMock()
        connector.delete.return_value = None

        await step._execute_operation(connector, "delete", {"id": "123"})
        connector.delete.assert_called_once_with(id="123")

    @pytest.mark.asyncio
    async def test_sync_operation(self):
        """Test sync operation."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Sync", config={})
        connector = AsyncMock()
        connector.sync.return_value = {"synced": 5}

        result = await step._execute_operation(connector, "sync", {})
        connector.sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_operation_async(self):
        """Test custom operation with async method."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="Custom",
            config={"custom_method": "get_envelope_status"},
        )
        connector = MagicMock()
        connector.get_envelope_status = AsyncMock(return_value={"status": "sent"})

        result = await step._execute_operation(connector, "custom", {"envelope_id": "env_1"})
        connector.get_envelope_status.assert_called_once_with(envelope_id="env_1")

    @pytest.mark.asyncio
    async def test_custom_operation_sync(self):
        """Test custom operation with sync method."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="Custom Sync",
            config={"custom_method": "compute_total"},
        )
        connector = MagicMock()
        connector.compute_total = MagicMock(return_value=100)

        result = await step._execute_operation(connector, "custom", {"items": [1, 2]})
        connector.compute_total.assert_called_once_with(items=[1, 2])
        assert result == 100

    @pytest.mark.asyncio
    async def test_custom_operation_missing_method_name(self):
        """Test custom operation without custom_method raises ValueError."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Custom", config={})
        connector = MagicMock()

        with pytest.raises(ValueError, match="custom_method required"):
            await step._execute_operation(connector, "custom", {})

    @pytest.mark.asyncio
    async def test_custom_operation_missing_method(self):
        """Test custom operation with nonexistent method raises ValueError."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="Custom",
            config={"custom_method": "nonexistent_method"},
        )
        connector = MagicMock(spec=[])

        with pytest.raises(ValueError, match="no method"):
            await step._execute_operation(connector, "custom", {})

    @pytest.mark.asyncio
    async def test_unknown_operation_raises(self):
        """Test that unknown operation raises ValueError."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Bad Op", config={})
        connector = MagicMock()

        with pytest.raises(ValueError, match="Unknown operation"):
            await step._execute_operation(connector, "nonexistent_op", {})

    @pytest.mark.asyncio
    async def test_list_no_method_raises(self):
        """Test list on connector without list or list_all raises."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="No List", config={})
        connector = MagicMock(spec=[])

        with pytest.raises(ValueError, match="does not support list"):
            await step._execute_operation(connector, "list", {})


# ============================================================================
# Result Formatting Tests
# ============================================================================


class TestResultFormatting:
    """Tests for ConnectorStep._format_result."""

    def test_format_dict_result(self):
        """Test formatting a plain dict result."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Format", config={})
        result = step._format_result({"key": "value"})
        assert result == {"key": "value"}

    def test_format_evidence_object(self):
        """Test formatting an object with to_dict."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Format", config={})
        obj = MagicMock()
        obj.to_dict.return_value = {"evidence": "data"}
        result = step._format_result(obj)
        assert result == {"evidence": "data"}

    def test_format_list_of_evidence(self):
        """Test formatting a list of objects with to_dict."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Format", config={})
        obj1 = MagicMock()
        obj1.to_dict.return_value = {"id": 1}
        obj2 = MagicMock()
        obj2.to_dict.return_value = {"id": 2}
        result = step._format_result([obj1, obj2])
        assert result == [{"id": 1}, {"id": 2}]

    def test_format_plain_value(self):
        """Test formatting a plain value."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="Format", config={})
        assert step._format_result("hello") == "hello"
        assert step._format_result(42) == 42
        assert step._format_result(None) is None


# ============================================================================
# Full Execute Tests
# ============================================================================


class TestConnectorStepExecute:
    """Tests for ConnectorStep.execute (full flow)."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
        )

    @pytest.mark.asyncio
    async def test_execute_missing_connector_type(self):
        """Test execute raises when connector_type is missing."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(name="No Type", config={})
        ctx = self._make_context()
        with pytest.raises(ValueError, match="connector_type is required"):
            await step.execute(ctx)

    @pytest.mark.asyncio
    async def test_execute_unknown_connector_type(self):
        """Test execute raises for unknown connector type."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="Unknown",
            config={"connector_type": "totally_fake_connector"},
        )
        ctx = self._make_context()
        with pytest.raises(ValueError, match="Unknown connector type"):
            await step.execute(ctx)

    @pytest.mark.asyncio
    async def test_execute_unsupported_operation(self):
        """Test execute raises for unsupported operation on connector."""
        from aragora.workflow.nodes.connector import ConnectorStep

        step = ConnectorStep(
            name="Bad Op",
            config={
                "connector_type": "web",
                "operation": "delete",
            },
        )
        ctx = self._make_context()
        with pytest.raises(ValueError, match="not supported"):
            await step.execute(ctx)

    @pytest.mark.asyncio
    async def test_execute_successful_flow(self):
        """Test full successful execution flow."""
        from aragora.workflow.nodes.connector import ConnectorStep

        mock_connector = AsyncMock()
        mock_connector.search.return_value = [{"title": "Found"}]

        step = ConnectorStep(
            name="Full Flow",
            config={
                "connector_type": "github",
                "operation": "search",
                "params": {"query": "{inputs.query}"},
                "credentials": {"token": "test_token"},
            },
        )

        ctx = self._make_context(inputs={"query": "workflow engine"})

        with patch(
            "aragora.workflow.nodes.connector.create_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await step.execute(ctx)

        assert result == [{"title": "Found"}]
        mock_connector.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_caches_connector(self):
        """Test that connectors are cached across calls."""
        from aragora.workflow.nodes.connector import ConnectorStep

        mock_connector = AsyncMock()
        mock_connector.fetch.return_value = {"data": "cached"}

        step = ConnectorStep(
            name="Cache Test",
            config={
                "connector_type": "github",
                "operation": "fetch",
                "params": {"source_id": "1"},
                "credentials": {"token": "t"},
            },
        )
        ctx = self._make_context()

        create_mock = AsyncMock(return_value=mock_connector)
        with patch(
            "aragora.workflow.nodes.connector.create_connector",
            create_mock,
        ):
            await step.execute(ctx)
            await step.execute(ctx)

        # Should only create the connector once due to caching
        assert create_mock.call_count == 1
