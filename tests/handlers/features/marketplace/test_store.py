"""Comprehensive tests for marketplace in-memory storage and template discovery.

Covers every public function in:
    aragora/server/handlers/features/marketplace/store.py (199 lines)

Functions tested:
- _clear_marketplace_state()       - Clears all in-memory stores and circuit breaker
- _clear_marketplace_components()  - Compatibility wrapper for _clear_marketplace_state
- _get_templates_dir()             - Returns workflow templates directory path
- _load_templates()                - Loads templates from YAML files with circuit breaker
- _parse_template_file()           - Parses a single YAML template file into metadata
- _get_full_template()             - Loads full template content by ID
- _get_tenant_deployments()        - Returns deployment dict for a tenant
- get_ratings()                    - Returns the global ratings dict
- get_download_counts()            - Returns the global download counts dict
- get_deployments()                - Returns the global deployments dict
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from aragora.server.handlers.features.marketplace.models import (
    DeploymentStatus,
    TemplateCategory,
    TemplateDeployment,
    TemplateMetadata,
    TemplateRating,
)
from aragora.server.handlers.features.marketplace.store import (
    _clear_marketplace_components,
    _clear_marketplace_state,
    _deployments,
    _download_counts,
    _get_full_template,
    _get_templates_dir,
    _get_tenant_deployments,
    _load_templates,
    _parse_template_file,
    _ratings,
    _templates_cache,
    get_deployments,
    get_download_counts,
    get_ratings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_template(
    template_id: str = "tpl-1",
    name: str = "Test Template",
    category: TemplateCategory = TemplateCategory.GENERAL,
    file_path: str | None = "/tmp/test.yaml",
    **kwargs: Any,
) -> TemplateMetadata:
    """Create a template metadata object for testing."""
    defaults = dict(
        id=template_id,
        name=name,
        description="A test template",
        version="1.0.0",
        category=category,
        tags=["test"],
        steps_count=3,
        has_debate=False,
        has_human_checkpoint=False,
        estimated_duration="< 1 minute",
        file_path=file_path,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return TemplateMetadata(**defaults)


class MockCircuitBreaker:
    """Minimal circuit breaker stub for testing."""

    def __init__(self, *, allowed: bool = True):
        self._allowed = allowed
        self._successes = 0
        self._failures = 0

    def is_allowed(self) -> bool:
        return self._allowed

    def record_success(self) -> None:
        self._successes += 1

    def record_failure(self) -> None:
        self._failures += 1

    def get_status(self) -> dict[str, Any]:
        return {
            "state": "closed" if self._allowed else "open",
            "failure_count": self._failures,
            "success_count": self._successes,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_store():
    """Clear all marketplace state before and after each test."""
    _clear_marketplace_state()
    yield
    _clear_marketplace_state()


@pytest.fixture
def mock_cb():
    """Allowed circuit breaker."""
    return MockCircuitBreaker(allowed=True)


@pytest.fixture
def mock_cb_open():
    """Blocked (open) circuit breaker."""
    return MockCircuitBreaker(allowed=False)


# =============================================================================
# _clear_marketplace_state
# =============================================================================


class TestClearMarketplaceState:
    """Tests for _clear_marketplace_state()."""

    def test_clears_templates_cache(self):
        _templates_cache["tpl-1"] = _make_template()
        assert len(_templates_cache) == 1
        _clear_marketplace_state()
        assert len(_templates_cache) == 0

    def test_clears_deployments(self):
        _deployments["tenant-1"] = {
            "dep-1": TemplateDeployment(
                id="dep-1",
                template_id="tpl-1",
                tenant_id="tenant-1",
                name="My Deployment",
                status=DeploymentStatus.ACTIVE,
            )
        }
        _clear_marketplace_state()
        assert len(_deployments) == 0

    def test_clears_ratings(self):
        _ratings["tpl-1"] = [
            TemplateRating(
                id="r-1",
                template_id="tpl-1",
                tenant_id="t-1",
                user_id="u-1",
                rating=5,
            )
        ]
        _clear_marketplace_state()
        assert len(_ratings) == 0

    def test_clears_download_counts(self):
        _download_counts["tpl-1"] = 42
        _clear_marketplace_state()
        assert len(_download_counts) == 0

    def test_resets_circuit_breaker(self):
        with patch(
            "aragora.server.handlers.features.marketplace.store._reset_circuit_breaker"
        ) as mock_reset:
            _clear_marketplace_state()
            mock_reset.assert_called_once()

    def test_idempotent_on_empty_state(self):
        """Calling clear on already-empty state does not raise."""
        _clear_marketplace_state()
        _clear_marketplace_state()
        assert len(_templates_cache) == 0

    def test_clears_multiple_tenants(self):
        _deployments["t1"] = {}
        _deployments["t2"] = {}
        _deployments["t3"] = {}
        _clear_marketplace_state()
        assert len(_deployments) == 0


# =============================================================================
# _clear_marketplace_components
# =============================================================================


class TestClearMarketplaceComponents:
    """Tests for _clear_marketplace_components() compatibility wrapper."""

    def test_delegates_to_clear_state(self):
        _templates_cache["tpl-1"] = _make_template()
        _download_counts["tpl-1"] = 10
        _clear_marketplace_components()
        assert len(_templates_cache) == 0
        assert len(_download_counts) == 0

    def test_also_clears_ratings(self):
        _ratings["tpl-1"] = []
        _clear_marketplace_components()
        assert len(_ratings) == 0


# =============================================================================
# _get_templates_dir
# =============================================================================


class TestGetTemplatesDir:
    """Tests for _get_templates_dir()."""

    def test_returns_path_object(self):
        result = _get_templates_dir()
        assert isinstance(result, Path)

    def test_path_ends_with_workflow_templates(self):
        result = _get_templates_dir()
        assert result.parts[-2:] == ("workflow", "templates")

    def test_path_is_under_aragora(self):
        result = _get_templates_dir()
        # The path should be relative to the aragora source tree
        parts = result.parts
        assert "aragora" in parts


# =============================================================================
# _parse_template_file
# =============================================================================


class TestParseTemplateFile:
    """Tests for _parse_template_file()."""

    def _write_yaml(self, tmp_path: Path, data: dict, filename: str = "tpl.yaml") -> Path:
        """Write a YAML file and return its path."""
        fp = tmp_path / filename
        fp.write_text(yaml.dump(data))
        return fp

    def test_parses_minimal_template(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-min",
            "name": "Minimal",
            "description": "A minimal template",
            "version": "1.0.0",
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.id == "tpl-min"
        assert result.name == "Minimal"
        assert result.category == TemplateCategory.GENERAL

    def test_returns_none_when_not_template(self, tmp_path):
        fp = self._write_yaml(tmp_path, {"is_template": False, "name": "Not a template"})
        result = _parse_template_file(fp)
        assert result is None

    def test_returns_none_when_is_template_missing(self, tmp_path):
        fp = self._write_yaml(tmp_path, {"name": "Missing flag"})
        result = _parse_template_file(fp)
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        fp = tmp_path / "empty.yaml"
        fp.write_text("")
        result = _parse_template_file(fp)
        assert result is None

    def test_returns_none_for_null_yaml(self, tmp_path):
        fp = tmp_path / "null.yaml"
        fp.write_text("null")
        result = _parse_template_file(fp)
        assert result is None

    def test_parses_known_category(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-sw",
            "category": "software",
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.category == TemplateCategory.SOFTWARE

    def test_unknown_category_defaults_to_general(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-unk",
            "category": "blockchain_stuff",
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.category == TemplateCategory.GENERAL

    def test_category_case_insensitive(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-uc",
            "category": "LEGAL",
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.category == TemplateCategory.LEGAL

    def test_detects_debate_steps(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-debate",
            "steps": [
                {"step_type": "transform"},
                {"step_type": "debate"},
                {"step_type": "output"},
            ],
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.has_debate is True
        assert result.steps_count == 3

    def test_detects_human_checkpoint_steps(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-human",
            "steps": [
                {"step_type": "human_checkpoint"},
                {"step_type": "output"},
            ],
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.has_human_checkpoint is True

    def test_estimated_duration_with_human_checkpoint(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-dur-hc",
            "steps": [{"step_type": "human_checkpoint"}],
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.estimated_duration == "hours to days"

    def test_estimated_duration_with_debate_only(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-dur-debate",
            "steps": [{"step_type": "debate"}],
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.estimated_duration == "minutes to hours"

    def test_estimated_duration_many_steps(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-dur-many",
            "steps": [{"step_type": "x"} for _ in range(6)],
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.estimated_duration == "1-5 minutes"

    def test_estimated_duration_few_steps(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-dur-few",
            "steps": [{"step_type": "x"}, {"step_type": "y"}],
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.estimated_duration == "< 1 minute"

    def test_estimated_duration_no_steps(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-dur-none",
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.estimated_duration == "< 1 minute"

    def test_id_defaults_to_stem(self, tmp_path):
        fp = self._write_yaml(tmp_path, {"is_template": True}, filename="my_cool_template.yaml")
        result = _parse_template_file(fp)
        assert result is not None
        assert result.id == "my_cool_template"

    def test_name_defaults_to_titlecased_stem(self, tmp_path):
        fp = self._write_yaml(tmp_path, {"is_template": True}, filename="my_cool_template.yaml")
        result = _parse_template_file(fp)
        assert result is not None
        assert result.name == "My Cool Template"

    def test_file_path_recorded(self, tmp_path):
        fp = self._write_yaml(tmp_path, {"is_template": True, "id": "tpl-fp"})
        result = _parse_template_file(fp)
        assert result is not None
        assert result.file_path == str(fp)

    def test_tags_parsed(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-tags",
            "tags": ["alpha", "beta", "gamma"],
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.tags == ["alpha", "beta", "gamma"]

    def test_inputs_and_outputs_parsed(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-io",
            "inputs": {"query": "string"},
            "outputs": {"result": "json"},
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.inputs == {"query": "string"}
        assert result.outputs == {"result": "json"}

    def test_icon_parsed(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-icon",
            "icon": "rocket",
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.icon == "rocket"

    def test_icon_defaults_to_document(self, tmp_path):
        fp = self._write_yaml(tmp_path, {"is_template": True, "id": "tpl-noicon"})
        result = _parse_template_file(fp)
        assert result is not None
        assert result.icon == "document"

    def test_returns_none_on_yaml_error(self, tmp_path):
        fp = tmp_path / "bad.yaml"
        fp.write_text(":\n  :\n    - [bad yaml {{{")
        result = _parse_template_file(fp)
        assert result is None

    def test_returns_none_on_missing_file(self):
        result = _parse_template_file(Path("/nonexistent/path/template.yaml"))
        assert result is None

    def test_human_checkpoint_takes_priority_in_duration(self, tmp_path):
        """When both debate and human_checkpoint are present, 'hours to days' wins."""
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-both",
            "steps": [
                {"step_type": "debate"},
                {"step_type": "human_checkpoint"},
            ],
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.has_debate is True
        assert result.has_human_checkpoint is True
        assert result.estimated_duration == "hours to days"

    def test_version_parsed(self, tmp_path):
        fp = self._write_yaml(tmp_path, {
            "is_template": True,
            "id": "tpl-ver",
            "version": "3.2.1",
        })
        result = _parse_template_file(fp)
        assert result is not None
        assert result.version == "3.2.1"

    def test_version_defaults(self, tmp_path):
        fp = self._write_yaml(tmp_path, {"is_template": True, "id": "tpl-nover"})
        result = _parse_template_file(fp)
        assert result is not None
        assert result.version == "1.0.0"


# =============================================================================
# _load_templates
# =============================================================================


class TestLoadTemplates:
    """Tests for _load_templates()."""

    def test_returns_empty_when_dir_missing(self, mock_cb):
        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = False

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=mock_dir,
        ):
            result = _load_templates()
        assert result == {}
        assert mock_cb._successes == 1

    def test_returns_cached_when_populated(self, mock_cb):
        tpl = _make_template()
        _templates_cache["tpl-1"] = tpl

        # _load_templates should return cache without touching CB
        result = _load_templates()
        assert result == {"tpl-1": tpl}
        assert mock_cb._successes == 0  # Never called

    def test_returns_empty_when_circuit_open(self, mock_cb_open):
        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb_open,
        ):
            result = _load_templates()
        assert result == {}

    def test_loads_yaml_files_from_directory(self, tmp_path, mock_cb):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "t1.yaml").write_text(yaml.dump({
            "is_template": True,
            "id": "loaded-1",
            "name": "Loaded One",
        }))
        (tpl_dir / "t2.yaml").write_text(yaml.dump({
            "is_template": True,
            "id": "loaded-2",
            "name": "Loaded Two",
        }))

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=tpl_dir,
        ):
            result = _load_templates()

        assert "loaded-1" in result
        assert "loaded-2" in result
        assert result["loaded-1"].name == "Loaded One"
        assert mock_cb._successes == 1

    def test_skips_non_template_files(self, tmp_path, mock_cb):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "config.yaml").write_text(yaml.dump({
            "is_template": False,
            "setting": "value",
        }))

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=tpl_dir,
        ):
            result = _load_templates()

        assert len(result) == 0

    def test_skips_invalid_yaml_files(self, tmp_path, mock_cb):
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "bad.yaml").write_text(":\n  [bad yaml {{{")
        (tpl_dir / "good.yaml").write_text(yaml.dump({
            "is_template": True,
            "id": "good-1",
            "name": "Good One",
        }))

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=tpl_dir,
        ):
            result = _load_templates()

        assert "good-1" in result
        assert mock_cb._successes == 1

    def test_records_failure_on_runtime_error(self, mock_cb):
        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = True
        mock_dir.rglob.side_effect = RuntimeError("disk error")

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=mock_dir,
        ):
            result = _load_templates()

        assert result == {}
        assert mock_cb._failures == 1

    def test_loads_from_subdirectories(self, tmp_path, mock_cb):
        tpl_dir = tmp_path / "templates"
        sub = tpl_dir / "accounting"
        sub.mkdir(parents=True)
        (sub / "invoice.yaml").write_text(yaml.dump({
            "is_template": True,
            "id": "accounting-invoice",
            "category": "accounting",
        }))

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=tpl_dir,
        ):
            result = _load_templates()

        assert "accounting-invoice" in result
        assert result["accounting-invoice"].category == TemplateCategory.ACCOUNTING

    def test_records_failure_on_os_error(self, mock_cb):
        mock_dir = MagicMock(spec=Path)
        mock_dir.exists.return_value = True
        mock_dir.rglob.side_effect = OSError("permission denied")

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=mock_dir,
        ):
            result = _load_templates()

        assert mock_cb._failures == 1


# =============================================================================
# _get_full_template
# =============================================================================


class TestGetFullTemplate:
    """Tests for _get_full_template()."""

    def test_returns_none_for_unknown_template(self, mock_cb):
        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace._get_templates_dir",
            return_value=MagicMock(exists=MagicMock(return_value=False)),
        ):
            result = _get_full_template("nonexistent-id")
        assert result is None

    def test_returns_none_when_no_file_path(self, mock_cb):
        tpl = _make_template(template_id="tpl-nofp", file_path=None)
        _templates_cache["tpl-nofp"] = tpl

        result = _get_full_template("tpl-nofp")
        assert result is None

    def test_returns_full_yaml_content(self, tmp_path, mock_cb):
        yaml_data = {
            "is_template": True,
            "id": "tpl-full",
            "name": "Full Template",
            "steps": [
                {"step_type": "debate", "config": {"rounds": 3}},
            ],
        }
        fp = tmp_path / "full.yaml"
        fp.write_text(yaml.dump(yaml_data))

        tpl = _make_template(template_id="tpl-full", file_path=str(fp))
        _templates_cache["tpl-full"] = tpl

        result = _get_full_template("tpl-full")
        assert result is not None
        assert result["id"] == "tpl-full"
        assert len(result["steps"]) == 1

    def test_returns_none_on_yaml_read_error(self, tmp_path, mock_cb):
        fp = tmp_path / "corrupt.yaml"
        fp.write_text(":\n  [bad yaml {{{")

        tpl = _make_template(template_id="tpl-corrupt", file_path=str(fp))
        _templates_cache["tpl-corrupt"] = tpl

        result = _get_full_template("tpl-corrupt")
        assert result is None

    def test_returns_none_on_file_not_found(self, mock_cb):
        tpl = _make_template(
            template_id="tpl-gone",
            file_path="/nonexistent/path/template.yaml",
        )
        _templates_cache["tpl-gone"] = tpl

        result = _get_full_template("tpl-gone")
        assert result is None


# =============================================================================
# _get_tenant_deployments
# =============================================================================


class TestGetTenantDeployments:
    """Tests for _get_tenant_deployments()."""

    def test_returns_empty_dict_for_new_tenant(self):
        result = _get_tenant_deployments("new-tenant")
        assert result == {}

    def test_creates_entry_for_new_tenant(self):
        _get_tenant_deployments("tenant-x")
        assert "tenant-x" in _deployments
        assert _deployments["tenant-x"] == {}

    def test_returns_existing_deployments(self):
        dep = TemplateDeployment(
            id="dep-1",
            template_id="tpl-1",
            tenant_id="tenant-1",
            name="Existing",
            status=DeploymentStatus.ACTIVE,
        )
        _deployments["tenant-1"] = {"dep-1": dep}

        result = _get_tenant_deployments("tenant-1")
        assert "dep-1" in result
        assert result["dep-1"].name == "Existing"

    def test_returns_mutable_reference(self):
        result = _get_tenant_deployments("tenant-mut")
        dep = TemplateDeployment(
            id="dep-new",
            template_id="tpl-1",
            tenant_id="tenant-mut",
            name="New",
            status=DeploymentStatus.PENDING,
        )
        result["dep-new"] = dep
        assert "dep-new" in _deployments["tenant-mut"]

    def test_multiple_tenants_isolated(self):
        _get_tenant_deployments("t1")["d1"] = TemplateDeployment(
            id="d1", template_id="tpl-1", tenant_id="t1",
            name="T1 Dep", status=DeploymentStatus.ACTIVE,
        )
        _get_tenant_deployments("t2")["d2"] = TemplateDeployment(
            id="d2", template_id="tpl-2", tenant_id="t2",
            name="T2 Dep", status=DeploymentStatus.ACTIVE,
        )
        assert "d1" not in _get_tenant_deployments("t2")
        assert "d2" not in _get_tenant_deployments("t1")


# =============================================================================
# get_ratings
# =============================================================================


class TestGetRatings:
    """Tests for get_ratings()."""

    def test_returns_ratings_dict(self):
        result = get_ratings()
        assert isinstance(result, dict)
        assert result is _ratings

    def test_mutations_reflected_in_global(self):
        ratings = get_ratings()
        ratings["tpl-1"] = [
            TemplateRating(
                id="r-1",
                template_id="tpl-1",
                tenant_id="t-1",
                user_id="u-1",
                rating=5,
            )
        ]
        assert "tpl-1" in _ratings
        assert len(_ratings["tpl-1"]) == 1


# =============================================================================
# get_download_counts
# =============================================================================


class TestGetDownloadCounts:
    """Tests for get_download_counts()."""

    def test_returns_download_counts_dict(self):
        result = get_download_counts()
        assert isinstance(result, dict)
        assert result is _download_counts

    def test_mutations_reflected_in_global(self):
        counts = get_download_counts()
        counts["tpl-1"] = 100
        assert _download_counts["tpl-1"] == 100


# =============================================================================
# get_deployments
# =============================================================================


class TestGetDeployments:
    """Tests for get_deployments()."""

    def test_returns_deployments_dict(self):
        result = get_deployments()
        assert isinstance(result, dict)
        assert result is _deployments

    def test_mutations_reflected_in_global(self):
        deps = get_deployments()
        deps["tenant-1"] = {}
        assert "tenant-1" in _deployments
