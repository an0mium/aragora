"""
Tests for MarketplaceHandler graduation from EXPERIMENTAL to STABLE.

This test file provides comprehensive coverage for code paths not covered
by the existing test_features_marketplace.py, targeting 80%+ overall coverage.

Areas covered:
- Template loading with circuit breaker interactions
- YAML template parsing (all duration estimation branches)
- Full template content loading
- Search with tags and has_checkpoint filters
- Deploy with default name, config validation, exception handling
- Rate with circuit breaker failure, review sanitization
- Request routing for all HTTP methods and path patterns
- JSON body parsing edge cases
- Tenant deployment creation for new tenants
- Related templates fallback filling when < 5 matches
- Dataclass to_dict with optional fields populated
- Module-level helper functions
- _get_json_body without json attribute
- _validate_search_query with wrong type
- _validate_category with wrong type
- Deployment to_dict with last_run
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
from uuid import uuid4

import pytest

from aragora.server.handlers.features.marketplace import (
    MarketplaceHandler,
    MarketplaceCircuitBreaker,
    TemplateCategory,
    DeploymentStatus,
    TemplateMetadata,
    TemplateDeployment,
    TemplateRating,
    CATEGORY_INFO,
    get_marketplace_circuit_breaker_status,
    get_marketplace_handler,
    handle_marketplace,
    _clear_marketplace_components,
    _validate_template_id,
    _validate_deployment_id,
    _validate_deployment_name,
    _validate_review,
    _validate_rating,
    _validate_search_query,
    _validate_category,
    _validate_config,
    _clamp_pagination,
    _load_templates,
    _parse_template_file,
    _get_full_template,
    _get_tenant_deployments,
    _templates_cache,
    _deployments,
    _ratings,
    _download_counts,
    MAX_TEMPLATE_NAME_LENGTH,
    MAX_DEPLOYMENT_NAME_LENGTH,
    MAX_REVIEW_LENGTH,
    MAX_SEARCH_QUERY_LENGTH,
    MAX_CONFIG_SIZE,
    MAX_LIMIT,
    DEFAULT_LIMIT,
    MIN_LIMIT,
    MIN_RATING,
    MAX_RATING,
    SAFE_TEMPLATE_ID_PATTERN,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_marketplace_state():
    """Clear marketplace state before and after each test."""
    _clear_marketplace_components()
    yield
    _clear_marketplace_components()


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.tenant_id = "test-tenant"
    request.user_id = "test-user"
    request.query = {}
    return request


@pytest.fixture
def handler():
    """Create a MarketplaceHandler instance without loading templates."""
    with patch.object(
        MarketplaceHandler, "__init__", lambda self, ctx=None: setattr(self, "ctx", ctx or {})
    ):
        h = MarketplaceHandler.__new__(MarketplaceHandler)
        h.ctx = {}
        return h


@pytest.fixture
def sample_template():
    """Create a sample TemplateMetadata."""
    return TemplateMetadata(
        id="test-template-1",
        name="Test Template",
        description="A test template for testing",
        version="1.0.0",
        category=TemplateCategory.SOFTWARE,
        tags=["test", "software"],
        downloads=100,
        rating=4.5,
        rating_count=10,
    )


@pytest.fixture
def many_templates():
    """Create many templates across different categories for related-templates testing."""
    templates = {}
    categories = list(TemplateCategory)
    for i in range(12):
        cat = categories[i % len(categories)]
        tid = f"tpl-{i:03d}"
        templates[tid] = TemplateMetadata(
            id=tid,
            name=f"Template {i}",
            description=f"Description for template {i}",
            version="1.0.0",
            category=cat,
            tags=["common", f"tag-{i}", f"group-{i % 3}"],
            downloads=100 - i * 5,
            rating=4.0 + (i % 5) * 0.2,
        )
    return templates


# =============================================================================
# Template Loading Tests
# =============================================================================


class TestLoadTemplates:
    """Tests for _load_templates function."""

    def test_load_templates_cache_hit(self):
        """Test that cached templates are returned immediately."""
        mock_template = TemplateMetadata(
            id="cached-1",
            name="Cached",
            description="desc",
            version="1.0.0",
            category=TemplateCategory.GENERAL,
        )
        _templates_cache["cached-1"] = mock_template

        result = _load_templates()
        assert "cached-1" in result
        assert result["cached-1"].name == "Cached"

    def test_load_templates_circuit_breaker_open_returns_cache(self):
        """Test that open circuit breaker returns cached templates."""
        cb = MarketplaceCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN

        _templates_cache["old-1"] = TemplateMetadata(
            id="old-1",
            name="Old",
            description="desc",
            version="1.0.0",
            category=TemplateCategory.GENERAL,
        )

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=cb,
        ):
            result = _load_templates()

        assert "old-1" in result

    def test_load_templates_directory_not_found(self):
        """Test loading templates when directory doesn't exist."""
        with patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=Path("/nonexistent/path"),
        ):
            result = _load_templates()
            assert isinstance(result, dict)

    def test_load_templates_exception_records_failure(self):
        """Test that exceptions during loading record circuit breaker failure."""
        cb = MarketplaceCircuitBreaker()

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=cb,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace._get_templates_dir",
                side_effect=RuntimeError("disk error"),
            ):
                _load_templates()

        assert cb.get_status()["failure_count"] == 1


# =============================================================================
# Template Parsing Tests
# =============================================================================


class TestParseTemplateFile:
    """Tests for _parse_template_file function."""

    def test_parse_template_with_debate_step(self, tmp_path):
        """Test parsing template with debate step."""
        yaml_content = """
is_template: true
id: debate-tpl
name: Debate Template
description: Has debate
version: 1.0.0
category: software
steps:
  - step_type: debate
    name: Main debate
"""
        tpl_file = tmp_path / "debate.yaml"
        tpl_file.write_text(yaml_content)

        result = _parse_template_file(tpl_file)
        assert result is not None
        assert result.has_debate is True
        assert result.estimated_duration == "minutes to hours"

    def test_parse_template_with_human_checkpoint(self, tmp_path):
        """Test parsing template with human checkpoint step."""
        yaml_content = """
is_template: true
id: checkpoint-tpl
name: Checkpoint Template
description: Has human checkpoint
version: 1.0.0
category: legal
steps:
  - step_type: human_checkpoint
    name: Human review
"""
        tpl_file = tmp_path / "checkpoint.yaml"
        tpl_file.write_text(yaml_content)

        result = _parse_template_file(tpl_file)
        assert result is not None
        assert result.has_human_checkpoint is True
        assert result.estimated_duration == "hours to days"

    def test_parse_template_many_steps(self, tmp_path):
        """Test parsing template with many steps (> 5)."""
        yaml_content = """
is_template: true
id: many-steps
name: Many Steps
description: More than 5 steps
version: 1.0.0
steps:
  - step_type: process
  - step_type: process
  - step_type: process
  - step_type: process
  - step_type: process
  - step_type: process
"""
        tpl_file = tmp_path / "many.yaml"
        tpl_file.write_text(yaml_content)

        result = _parse_template_file(tpl_file)
        assert result is not None
        assert result.steps_count == 6
        assert result.estimated_duration == "1-5 minutes"

    def test_parse_template_few_steps(self, tmp_path):
        """Test parsing template with few steps (<= 5)."""
        yaml_content = """
is_template: true
id: few-steps
name: Few Steps
description: 3 steps
version: 1.0.0
steps:
  - step_type: process
  - step_type: process
  - step_type: process
"""
        tpl_file = tmp_path / "few.yaml"
        tpl_file.write_text(yaml_content)

        result = _parse_template_file(tpl_file)
        assert result is not None
        assert result.steps_count == 3
        assert result.estimated_duration == "< 1 minute"

    def test_parse_template_invalid_category_defaults_to_general(self, tmp_path):
        """Test that invalid category defaults to GENERAL."""
        yaml_content = """
is_template: true
id: bad-cat
name: Bad Category
description: Invalid category
version: 1.0.0
category: nonexistent_category
"""
        tpl_file = tmp_path / "badcat.yaml"
        tpl_file.write_text(yaml_content)

        result = _parse_template_file(tpl_file)
        assert result is not None
        assert result.category == TemplateCategory.GENERAL

    def test_parse_template_not_a_template(self, tmp_path):
        """Test parsing a YAML file that is not a template."""
        yaml_content = """
name: Not a template
is_template: false
"""
        tpl_file = tmp_path / "not_template.yaml"
        tpl_file.write_text(yaml_content)

        result = _parse_template_file(tpl_file)
        assert result is None

    def test_parse_template_empty_yaml(self, tmp_path):
        """Test parsing empty YAML file."""
        tpl_file = tmp_path / "empty.yaml"
        tpl_file.write_text("")

        result = _parse_template_file(tpl_file)
        assert result is None

    def test_parse_template_corrupted_yaml(self, tmp_path):
        """Test parsing corrupted YAML file."""
        tpl_file = tmp_path / "bad.yaml"
        tpl_file.write_text("{{{not: valid: yaml:")

        result = _parse_template_file(tpl_file)
        assert result is None

    def test_parse_template_uses_stem_for_defaults(self, tmp_path):
        """Test that file stem is used for missing id and name."""
        yaml_content = """
is_template: true
description: Minimal template
version: 1.0.0
"""
        tpl_file = tmp_path / "my_awesome_template.yaml"
        tpl_file.write_text(yaml_content)

        result = _parse_template_file(tpl_file)
        assert result is not None
        assert result.id == "my_awesome_template"
        assert result.name == "My Awesome Template"

    def test_parse_template_with_inputs_outputs(self, tmp_path):
        """Test parsing template with inputs and outputs."""
        yaml_content = """
is_template: true
id: io-tpl
name: IO Template
description: Has inputs/outputs
version: 1.0.0
inputs:
  query: "The search query"
outputs:
  result: "The analysis result"
icon: search
"""
        tpl_file = tmp_path / "io.yaml"
        tpl_file.write_text(yaml_content)

        result = _parse_template_file(tpl_file)
        assert result is not None
        assert result.inputs == {"query": "The search query"}
        assert result.outputs == {"result": "The analysis result"}
        assert result.icon == "search"


# =============================================================================
# Full Template Loading Tests
# =============================================================================


class TestGetFullTemplate:
    """Tests for _get_full_template function."""

    def test_get_full_template_success(self, tmp_path):
        """Test loading full template content."""
        yaml_content = """
is_template: true
id: full-tpl
name: Full Template
steps:
  - step_type: process
"""
        tpl_file = tmp_path / "full.yaml"
        tpl_file.write_text(yaml_content)

        meta = TemplateMetadata(
            id="full-tpl",
            name="Full Template",
            description="desc",
            version="1.0.0",
            category=TemplateCategory.GENERAL,
            file_path=str(tpl_file),
        )
        _templates_cache["full-tpl"] = meta

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=_templates_cache,
        ):
            result = _get_full_template("full-tpl")

        assert result is not None
        assert result["name"] == "Full Template"

    def test_get_full_template_not_found(self):
        """Test loading non-existent template."""
        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value={},
        ):
            result = _get_full_template("nonexistent")

        assert result is None

    def test_get_full_template_no_file_path(self):
        """Test loading template with no file_path."""
        meta = TemplateMetadata(
            id="no-path",
            name="No Path",
            description="desc",
            version="1.0.0",
            category=TemplateCategory.GENERAL,
            file_path=None,
        )
        _templates_cache["no-path"] = meta

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=_templates_cache,
        ):
            result = _get_full_template("no-path")

        assert result is None

    def test_get_full_template_file_read_error(self, tmp_path):
        """Test loading template when file read fails."""
        meta = TemplateMetadata(
            id="broken",
            name="Broken",
            description="desc",
            version="1.0.0",
            category=TemplateCategory.GENERAL,
            file_path="/nonexistent/path/file.yaml",
        )
        _templates_cache["broken"] = meta

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=_templates_cache,
        ):
            result = _get_full_template("broken")

        assert result is None


# =============================================================================
# Tenant Deployments Tests
# =============================================================================


class TestGetTenantDeployments:
    """Tests for _get_tenant_deployments function."""

    def test_creates_empty_dict_for_new_tenant(self):
        """Test that new tenants get an empty deployments dict."""
        result = _get_tenant_deployments("new-tenant")
        assert result == {}
        assert "new-tenant" in _deployments

    def test_returns_existing_deployments(self):
        """Test that existing deployments are returned."""
        deployment = TemplateDeployment(
            id="deploy_abc",
            template_id="tpl-1",
            tenant_id="existing",
            name="Existing",
            status=DeploymentStatus.ACTIVE,
        )
        _deployments["existing"] = {"deploy_abc": deployment}

        result = _get_tenant_deployments("existing")
        assert "deploy_abc" in result


# =============================================================================
# Search with Advanced Filters Tests
# =============================================================================


class TestSearchAdvancedFilters:
    """Tests for search with tags and feature filters."""

    @pytest.mark.asyncio
    async def test_search_with_tags_filter(self, handler, mock_request):
        """Test search with tags filter."""
        templates = {
            "t1": TemplateMetadata(
                id="t1",
                name="Template One",
                description="desc",
                version="1.0.0",
                category=TemplateCategory.SOFTWARE,
                tags=["python", "automation"],
            ),
            "t2": TemplateMetadata(
                id="t2",
                name="Template Two",
                description="desc",
                version="1.0.0",
                category=TemplateCategory.LEGAL,
                tags=["compliance", "audit"],
            ),
        }
        mock_request.query = {"tags": "python,automation"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_search_with_has_checkpoint_filter(self, handler, mock_request):
        """Test search with has_checkpoint filter."""
        templates = {
            "t1": TemplateMetadata(
                id="t1",
                name="Template One",
                description="desc",
                version="1.0.0",
                category=TemplateCategory.SOFTWARE,
                has_human_checkpoint=True,
            ),
            "t2": TemplateMetadata(
                id="t2",
                name="Template Two",
                description="desc",
                version="1.0.0",
                category=TemplateCategory.LEGAL,
                has_human_checkpoint=False,
            ),
        }
        mock_request.query = {"has_checkpoint": "true"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_search_with_invalid_query(self, handler, mock_request):
        """Test search with invalid search query (too long)."""
        mock_request.query = {"q": "a" * 600}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value={},
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_search_with_invalid_category(self, handler, mock_request):
        """Test search with invalid category."""
        mock_request.query = {"category": "nonexistent"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value={},
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_search_with_limit(self, handler, mock_request):
        """Test search with custom limit."""
        templates = {
            f"t{i}": TemplateMetadata(
                id=f"t{i}",
                name=f"Template {i}",
                description="common desc",
                version="1.0.0",
                category=TemplateCategory.GENERAL,
            )
            for i in range(10)
        }
        mock_request.query = {"limit": "3"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["results"]) <= 3


# =============================================================================
# Deploy Edge Cases Tests
# =============================================================================


class TestDeployEdgeCases:
    """Tests for deployment edge cases."""

    @pytest.mark.asyncio
    async def test_deploy_with_default_name(self, handler, mock_request, sample_template):
        """Test deploying without specifying a name uses template name."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({}, None),
            ):
                result = await handler._handle_deploy(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deployment"]["name"] == "Test Template"

    @pytest.mark.asyncio
    async def test_deploy_with_invalid_config(self, handler, mock_request, sample_template):
        """Test deploying with invalid config type."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"name": "Deploy", "config": "not-a-dict"}, None),
            ):
                result = await handler._handle_deploy(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_deploy_with_too_many_config_keys(self, handler, mock_request, sample_template):
        """Test deploying with too many config keys."""
        templates = {"test-template-1": sample_template}
        big_config = {f"key_{i}": i for i in range(MAX_CONFIG_SIZE + 1)}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"name": "Deploy", "config": big_config}, None),
            ):
                result = await handler._handle_deploy(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_deploy_exception_records_failure(self, handler, mock_request, sample_template):
        """Test that exceptions during deploy record circuit breaker failure."""
        templates = {"test-template-1": sample_template}
        cb = MarketplaceCircuitBreaker()

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=cb,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace._load_templates",
                return_value=templates,
            ):
                with patch(
                    "aragora.server.handlers.features.marketplace.parse_json_body",
                    side_effect=RuntimeError("unexpected"),
                ):
                    result = await handler._handle_deploy(
                        mock_request, "tenant-1", "test-template-1"
                    )

        assert result.status_code == 500
        assert cb.get_status()["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_deploy_with_valid_config(self, handler, mock_request, sample_template):
        """Test deploying with valid config."""
        templates = {"test-template-1": sample_template}
        config = {"auto_approve": True, "threshold": 100}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"name": "Config Deploy", "config": config}, None),
            ):
                result = await handler._handle_deploy(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deployment"]["config"] == config


# =============================================================================
# Rate Edge Cases Tests
# =============================================================================


class TestRateEdgeCases:
    """Tests for rating edge cases."""

    @pytest.mark.asyncio
    async def test_rate_circuit_breaker_open(self, handler, mock_request, sample_template):
        """Test rating when circuit breaker is open."""
        cb = MarketplaceCircuitBreaker(failure_threshold=1)
        cb.record_failure()

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=cb,
        ):
            result = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_rate_exception_records_failure(self, handler, mock_request, sample_template):
        """Test that exceptions during rating record circuit breaker failure."""
        templates = {"test-template-1": sample_template}
        cb = MarketplaceCircuitBreaker()

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=cb,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace._load_templates",
                return_value=templates,
            ):
                with patch(
                    "aragora.server.handlers.features.marketplace.parse_json_body",
                    side_effect=RuntimeError("unexpected"),
                ):
                    result = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 500
        assert cb.get_status()["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_rate_with_review_sanitization(self, handler, mock_request, sample_template):
        """Test that review text is sanitized."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"rating": 4, "review": "Great template!"}, None),
            ):
                with patch(
                    "aragora.server.handlers.features.marketplace.sanitize_string",
                    return_value="Great template!",
                ) as mock_sanitize:
                    result = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 200
        mock_sanitize.assert_called_once_with("Great template!")

    @pytest.mark.asyncio
    async def test_rate_invalid_review_type(self, handler, mock_request, sample_template):
        """Test rating with invalid review type."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"rating": 4, "review": 12345}, None),
            ):
                result = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_rate_review_too_long(self, handler, mock_request, sample_template):
        """Test rating with review exceeding max length."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"rating": 4, "review": "x" * (MAX_REVIEW_LENGTH + 1)}, None),
            ):
                result = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_rate_multiple_ratings_updates_average(
        self, handler, mock_request, sample_template
    ):
        """Test that multiple ratings correctly update the average."""
        templates = {"test-template-1": sample_template}

        # First rating
        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"rating": 5}, None),
            ):
                result1 = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        # Second rating
        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=templates,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"rating": 3}, None),
            ):
                result2 = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result2.status_code == 200
        body = json.loads(result2.body)
        assert body["template_rating"]["count"] == 2
        assert body["template_rating"]["average"] == 4.0


# =============================================================================
# Routing Tests
# =============================================================================


class TestRoutingComprehensive:
    """Comprehensive tests for request routing."""

    @pytest.mark.asyncio
    async def test_route_to_deploy(self, handler, mock_request, sample_template):
        """Test routing POST to deploy."""
        templates = {"test-id": sample_template}

        with patch.object(handler, "_handle_deploy", new_callable=AsyncMock) as mock_deploy:
            mock_deploy.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(
                mock_request, "/api/v1/marketplace/templates/test-id/deploy", "POST"
            )
            mock_deploy.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_rate(self, handler, mock_request):
        """Test routing POST to rate."""
        with patch.object(handler, "_handle_rate", new_callable=AsyncMock) as mock_rate:
            mock_rate.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/templates/test-id/rate", "POST")
            mock_rate.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_categories(self, handler, mock_request):
        """Test routing GET to categories."""
        with patch.object(handler, "_handle_list_categories", new_callable=AsyncMock) as mock_cats:
            mock_cats.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/categories", "GET")
            mock_cats.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_search(self, handler, mock_request):
        """Test routing GET to search."""
        with patch.object(handler, "_handle_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/search", "GET")
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_popular(self, handler, mock_request):
        """Test routing GET to popular."""
        with patch.object(handler, "_handle_popular", new_callable=AsyncMock) as mock_popular:
            mock_popular.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/popular", "GET")
            mock_popular.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_deployments(self, handler, mock_request):
        """Test routing GET to deployments."""
        with patch.object(handler, "_handle_list_deployments", new_callable=AsyncMock) as mock_deps:
            mock_deps.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/deployments", "GET")
            mock_deps.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_demo(self, handler, mock_request):
        """Test routing GET to demo."""
        with patch.object(handler, "_handle_demo", new_callable=AsyncMock) as mock_demo:
            mock_demo.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/demo", "GET")
            mock_demo.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_status(self, handler, mock_request):
        """Test routing GET to status."""
        with patch.object(handler, "_handle_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/status", "GET")
            mock_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_get_deployment(self, handler, mock_request):
        """Test routing GET to get deployment."""
        with patch.object(handler, "_handle_get_deployment", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(
                mock_request, "/api/v1/marketplace/deployments/deploy_abc123def456", "GET"
            )
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_delete_deployment(self, handler, mock_request):
        """Test routing DELETE to delete deployment."""
        with patch.object(
            handler, "_handle_delete_deployment", new_callable=AsyncMock
        ) as mock_delete:
            mock_delete.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(
                mock_request, "/api/v1/marketplace/deployments/deploy_abc123def456", "DELETE"
            )
            mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_invalid_deployment_id(self, handler, mock_request):
        """Test routing with invalid deployment ID."""
        result = await handler.handle(
            mock_request,
            "/api/v1/marketplace/deployments/../../etc/passwd",
            "GET",
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_route_unknown_action_on_template(self, handler, mock_request):
        """Test routing with unknown action on template."""
        result = await handler.handle(
            mock_request,
            "/api/v1/marketplace/templates/test-id/unknown-action",
            "POST",
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_route_template_get_only(self, handler, mock_request):
        """Test that template detail only works with GET."""
        # POST to a template detail (not deploy/rate) should return 404
        result = await handler.handle(
            mock_request,
            "/api/v1/marketplace/templates/test-id",
            "POST",
        )
        assert result.status_code == 404


# =============================================================================
# JSON Body Parsing Tests
# =============================================================================


class TestGetJsonBody:
    """Tests for _get_json_body method."""

    @pytest.mark.asyncio
    async def test_get_json_body_without_json_attr(self, handler):
        """Test _get_json_body when request has no json attribute."""
        request = MagicMock(spec=[])  # No attributes at all
        result = await handler._get_json_body(request)
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_json_body_with_json_attr(self, handler):
        """Test _get_json_body when request has json attribute."""
        request = MagicMock()
        request.json = MagicMock()

        with patch(
            "aragora.server.handlers.features.marketplace.parse_json_body",
            return_value=({"key": "value"}, None),
        ):
            result = await handler._get_json_body(request)

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_json_body_returns_empty_on_none(self, handler):
        """Test _get_json_body returns empty dict when parse returns None."""
        request = MagicMock()
        request.json = MagicMock()

        with patch(
            "aragora.server.handlers.features.marketplace.parse_json_body",
            return_value=(None, "parse error"),
        ):
            result = await handler._get_json_body(request)

        assert result == {}


# =============================================================================
# Related Templates Tests
# =============================================================================


class TestRelatedTemplatesFallback:
    """Tests for related templates with fallback filling."""

    def test_related_templates_fills_with_popular_fallback(self, handler, many_templates):
        """Test that related templates fills with popular when < 5 same-category."""
        # Pick a template in a category with few other templates
        target = many_templates["tpl-000"]

        result = handler._get_related_templates(target, many_templates)

        # Should fill up to 5 with fallback
        assert len(result) == 5

    def test_related_templates_shared_tags_increase_score(self, handler):
        """Test that shared tags increase relatedness score."""
        templates = {
            "base": TemplateMetadata(
                id="base",
                name="Base",
                description="desc",
                version="1.0.0",
                category=TemplateCategory.SOFTWARE,
                tags=["python", "automation", "testing"],
            ),
            "high-match": TemplateMetadata(
                id="high-match",
                name="High Match",
                description="desc",
                version="1.0.0",
                category=TemplateCategory.SOFTWARE,
                tags=["python", "automation", "testing"],
            ),
            "low-match": TemplateMetadata(
                id="low-match",
                name="Low Match",
                description="desc",
                version="1.0.0",
                category=TemplateCategory.LEGAL,
                tags=["compliance"],
            ),
        }

        result = handler._get_related_templates(templates["base"], templates)
        assert len(result) >= 2
        # high-match should appear before low-match
        ids = [r["id"] for r in result]
        if "high-match" in ids and "low-match" in ids:
            assert ids.index("high-match") < ids.index("low-match")


# =============================================================================
# Input Validation Edge Cases
# =============================================================================


class TestInputValidationEdgeCases:
    """Additional edge case tests for input validation functions."""

    def test_validate_search_query_wrong_type(self):
        """Test search query validation with wrong type."""
        is_valid, sanitized, err = _validate_search_query(123)
        assert is_valid is False
        assert "string" in err.lower()

    def test_validate_category_wrong_type(self):
        """Test category validation with wrong type."""
        is_valid, cat, err = _validate_category(123)
        assert is_valid is False
        assert "string" in err.lower()

    def test_validate_category_empty_string(self):
        """Test category validation with empty string."""
        is_valid, cat, err = _validate_category("")
        assert is_valid is True
        assert cat is None

    def test_validate_template_id_special_chars(self):
        """Test template ID with various special characters."""
        # Spaces
        is_valid, err = _validate_template_id("has space")
        assert is_valid is False

        # Dots
        is_valid, err = _validate_template_id("has.dot")
        assert is_valid is False

        # Forward slash
        is_valid, err = _validate_template_id("has/slash")
        assert is_valid is False

    def test_validate_deployment_id_too_long(self):
        """Test deployment ID exceeding max length."""
        long_id = "a" * 200
        is_valid, err = _validate_deployment_id(long_id)
        assert is_valid is False
        assert "128" in err

    def test_validate_rating_boundary_values(self):
        """Test rating validation at boundaries."""
        # Exactly MIN_RATING
        is_valid, value, err = _validate_rating(MIN_RATING)
        assert is_valid is True
        assert value == MIN_RATING

        # Exactly MAX_RATING
        is_valid, value, err = _validate_rating(MAX_RATING)
        assert is_valid is True
        assert value == MAX_RATING

    def test_clamp_pagination_exact_max(self):
        """Test pagination clamping at exact maximum."""
        limit, offset = _clamp_pagination(MAX_LIMIT, 0)
        assert limit == MAX_LIMIT

    def test_clamp_pagination_exact_min(self):
        """Test pagination clamping at exact minimum."""
        limit, offset = _clamp_pagination(MIN_LIMIT, 0)
        assert limit == MIN_LIMIT


# =============================================================================
# Dataclass Edge Cases
# =============================================================================


class TestDataclassEdgeCases:
    """Tests for dataclass edge cases."""

    def test_template_deployment_to_dict_with_last_run(self):
        """Test TemplateDeployment to_dict with last_run populated."""
        now = datetime.now(timezone.utc)
        deployment = TemplateDeployment(
            id="deploy_test",
            template_id="tpl-1",
            tenant_id="tenant-1",
            name="Test",
            status=DeploymentStatus.ACTIVE,
            last_run=now,
            run_count=5,
        )
        d = deployment.to_dict()

        assert d["last_run"] == now.isoformat()
        assert d["run_count"] == 5

    def test_template_deployment_to_dict_without_last_run(self):
        """Test TemplateDeployment to_dict without last_run."""
        deployment = TemplateDeployment(
            id="deploy_test",
            template_id="tpl-1",
            tenant_id="tenant-1",
            name="Test",
            status=DeploymentStatus.ACTIVE,
        )
        d = deployment.to_dict()

        assert d["last_run"] is None

    def test_template_metadata_all_fields(self):
        """Test TemplateMetadata to_dict with all fields."""
        meta = TemplateMetadata(
            id="full",
            name="Full Template",
            description="Full desc",
            version="2.0.0",
            category=TemplateCategory.HEALTHCARE,
            tags=["health", "compliance"],
            icon="heart",
            author="Doctor",
            downloads=500,
            rating=4.8,
            rating_count=50,
            inputs={"patient_id": "string"},
            outputs={"report": "string"},
            steps_count=10,
            has_debate=True,
            has_human_checkpoint=True,
            estimated_duration="hours to days",
        )
        d = meta.to_dict()

        assert d["category"] == "healthcare"
        assert d["icon"] == "heart"
        assert d["author"] == "Doctor"
        assert d["has_debate"] is True
        assert d["has_human_checkpoint"] is True
        assert d["estimated_duration"] == "hours to days"

    def test_template_rating_without_review(self):
        """Test TemplateRating to_dict without review."""
        rating = TemplateRating(
            id="rating_test",
            template_id="tpl-1",
            tenant_id="tenant-1",
            user_id="user-1",
            rating=3,
            review=None,
        )
        d = rating.to_dict()

        assert d["review"] is None
        assert d["rating"] == 3

    def test_all_deployment_statuses(self):
        """Test all DeploymentStatus values."""
        for status in DeploymentStatus:
            deployment = TemplateDeployment(
                id=f"deploy_{status.value}",
                template_id="tpl-1",
                tenant_id="tenant-1",
                name="Test",
                status=status,
            )
            d = deployment.to_dict()
            assert d["status"] == status.value

    def test_all_template_categories(self):
        """Test all TemplateCategory values."""
        for cat in TemplateCategory:
            meta = TemplateMetadata(
                id=f"tpl-{cat.value}",
                name=f"Template {cat.value}",
                description="desc",
                version="1.0.0",
                category=cat,
            )
            d = meta.to_dict()
            assert d["category"] == cat.value


# =============================================================================
# Module-level Helper Tests
# =============================================================================


class TestModuleLevelHelpers:
    """Tests for module-level helper functions."""

    def test_get_marketplace_handler(self):
        """Test get_marketplace_handler creates instance."""
        with patch("aragora.server.handlers.features.marketplace._load_templates"):
            handler = get_marketplace_handler()
        assert isinstance(handler, MarketplaceHandler)

    @pytest.mark.asyncio
    async def test_handle_marketplace(self, mock_request):
        """Test handle_marketplace delegates to handler."""
        with patch("aragora.server.handlers.features.marketplace._load_templates"):
            with patch.object(MarketplaceHandler, "handle", new_callable=AsyncMock) as mock_handle:
                mock_handle.return_value = MagicMock(status_code=200, body=b"{}")
                await handle_marketplace(mock_request, "/api/v1/marketplace/templates", "GET")
                mock_handle.assert_called_once()


# =============================================================================
# Category Info Tests
# =============================================================================


class TestCategoryInfo:
    """Tests for CATEGORY_INFO constant."""

    def test_all_categories_have_info(self):
        """Test that all TemplateCategory values have CATEGORY_INFO entries."""
        for cat in TemplateCategory:
            assert cat in CATEGORY_INFO, f"Missing CATEGORY_INFO for {cat.value}"

    def test_category_info_has_required_fields(self):
        """Test that each category info has required fields."""
        for cat, info in CATEGORY_INFO.items():
            assert "name" in info, f"Missing 'name' for {cat.value}"
            assert "description" in info, f"Missing 'description' for {cat.value}"
            assert "icon" in info, f"Missing 'icon' for {cat.value}"
            assert "color" in info, f"Missing 'color' for {cat.value}"


# =============================================================================
# Can Handle Tests - Additional
# =============================================================================


class TestCanHandleAdditional:
    """Additional tests for can_handle method."""

    def test_can_handle_demo(self, handler):
        """Test can_handle for demo endpoint."""
        assert handler.can_handle("/api/v1/marketplace/demo") is True

    def test_can_handle_template_rate(self, handler):
        """Test can_handle for template rate endpoint."""
        assert handler.can_handle("/api/v1/marketplace/templates/tpl-id/rate") is True

    def test_cannot_handle_partial_path(self, handler):
        """Test can_handle for partial marketplace path."""
        assert handler.can_handle("/api/v1/marketplace") is False

    def test_cannot_handle_similar_path(self, handler):
        """Test can_handle for similar but different path."""
        assert handler.can_handle("/api/v2/marketplace/templates") is False


# =============================================================================
# Tenant ID Extraction Tests - Additional
# =============================================================================


class TestTenantIdExtraction:
    """Additional tests for tenant ID extraction."""

    def test_valid_tenant_id(self, handler, mock_request):
        """Test extraction of valid tenant ID."""
        mock_request.tenant_id = "valid-tenant-123"
        result = handler._get_tenant_id(mock_request)
        assert result == "valid-tenant-123"

    def test_missing_tenant_id_attribute(self, handler):
        """Test extraction when request has no tenant_id attribute."""
        request = MagicMock(spec=[])
        result = handler._get_tenant_id(request)
        assert result == "default"


# =============================================================================
# Constants Verification
# =============================================================================


class TestConstants:
    """Tests to verify constant values are sane."""

    def test_pagination_limits(self):
        """Test pagination limit constants."""
        assert MIN_LIMIT >= 1
        assert DEFAULT_LIMIT > 0
        assert MAX_LIMIT >= DEFAULT_LIMIT

    def test_rating_bounds(self):
        """Test rating bound constants."""
        assert MIN_RATING >= 1
        assert MAX_RATING >= MIN_RATING
        assert MAX_RATING <= 10

    def test_string_length_limits(self):
        """Test string length limit constants."""
        assert MAX_TEMPLATE_NAME_LENGTH > 0
        assert MAX_DEPLOYMENT_NAME_LENGTH > 0
        assert MAX_REVIEW_LENGTH > 0
        assert MAX_SEARCH_QUERY_LENGTH > 0

    def test_safe_template_id_pattern(self):
        """Test the safe template ID regex pattern."""
        assert SAFE_TEMPLATE_ID_PATTERN.match("valid-id") is not None
        assert SAFE_TEMPLATE_ID_PATTERN.match("valid_id_123") is not None
        assert SAFE_TEMPLATE_ID_PATTERN.match("UPPER") is not None
        assert SAFE_TEMPLATE_ID_PATTERN.match("") is None
        assert SAFE_TEMPLATE_ID_PATTERN.match("../evil") is None
        assert SAFE_TEMPLATE_ID_PATTERN.match("has space") is None


# =============================================================================
# Circuit Breaker - Additional Edge Cases
# =============================================================================


class TestCircuitBreakerAdditional:
    """Additional circuit breaker edge cases."""

    def test_success_in_closed_state_no_prior_failures(self):
        """Test that success in closed state with no failures is a no-op."""
        cb = MarketplaceCircuitBreaker()
        cb.record_success()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED
        assert cb.get_status()["failure_count"] == 0

    def test_multiple_resets(self):
        """Test multiple resets."""
        cb = MarketplaceCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        cb.reset()
        cb.reset()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED

    def test_custom_parameters(self):
        """Test circuit breaker with custom parameters."""
        cb = MarketplaceCircuitBreaker(
            failure_threshold=10,
            cooldown_seconds=60.0,
            half_open_max_calls=5,
        )
        status = cb.get_status()
        assert status["failure_threshold"] == 10
        assert status["cooldown_seconds"] == 60.0

    def test_last_failure_time_is_set(self):
        """Test that last_failure_time is recorded."""
        cb = MarketplaceCircuitBreaker()
        assert cb.get_status()["last_failure_time"] is None

        before = time.time()
        cb.record_failure()
        after = time.time()

        last_fail = cb.get_status()["last_failure_time"]
        assert last_fail is not None
        assert before <= last_fail <= after
