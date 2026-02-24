"""
Tests for SDK code generation pipeline (scripts/sdk_codegen.py).

Validates endpoint extraction via AST parsing and code generation for
both TypeScript and Python targets.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.sdk_codegen import (
    CoverageReport,
    EndpointDef,
    SDKCodeGenerator,
    _extract_path_params,
    _group_by_namespace,
    _normalize_path,
    _py_format_path,
    _to_camel_case,
    _to_pascal_case,
    _ts_template_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def generator() -> SDKCodeGenerator:
    return SDKCodeGenerator()


@pytest.fixture()
def sample_handler_dir(tmp_path: Path) -> Path:
    """Create a temporary handler directory with sample handler files."""
    handler_dir = tmp_path / "handlers"
    handler_dir.mkdir()

    # Handler with docstring endpoints (dash style)
    (handler_dir / "orders.py").write_text(
        textwrap.dedent(
            '''\
        """
        Order Handler.

        Endpoints:
        - GET /api/v1/orders - List all orders
        - POST /api/v1/orders - Create a new order
        - GET /api/v1/orders/:order_id - Get order by ID
        - PUT /api/v1/orders/:order_id - Update an order
        - DELETE /api/v1/orders/:order_id - Delete an order
        """

        class OrderHandler:
            ROUTES = [
                "/api/v1/orders",
                "/api/v1/orders/*",
            ]
    '''
        )
    )

    # Handler with indented docstring endpoints (backup_handler style)
    (handler_dir / "items.py").write_text(
        textwrap.dedent(
            '''\
        """
        Item Handler.

        Endpoints:
            GET  /api/v2/items              - List items
            POST /api/v2/items              - Create item
            GET  /api/v2/items/:item_id     - Get item details
        """

        class ItemHandler:
            pass
    '''
        )
    )

    # Handler with ROUTES only (no docstring endpoints)
    (handler_dir / "tags.py").write_text(
        textwrap.dedent(
            '''\
        """Tag Handler."""

        class TagHandler:
            ROUTES = [
                "/api/v1/tags",
                "/api/v1/tags/popular",
            ]
    '''
        )
    )

    # File that should be skipped (private)
    (handler_dir / "_internal.py").write_text('"""Internal."""\n')

    # Subdirectory handler
    sub = handler_dir / "billing"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "invoices.py").write_text(
        textwrap.dedent(
            '''\
        """
        Invoice Handler.

        Endpoints:
        - GET /api/v1/billing/invoices - List invoices
        - POST /api/v1/billing/invoices/:invoice_id/pay - Pay an invoice
        """
    '''
        )
    )

    return handler_dir


@pytest.fixture()
def sample_endpoints() -> list[EndpointDef]:
    """A small set of endpoints for generation tests."""
    return [
        EndpointDef(
            method="GET",
            path="/api/v1/widgets",
            params=[],
            description="List all widgets",
            handler_file="widgets.py",
        ),
        EndpointDef(
            method="POST",
            path="/api/v1/widgets",
            params=[],
            description="Create a widget",
            handler_file="widgets.py",
        ),
        EndpointDef(
            method="GET",
            path="/api/v1/widgets/:widget_id",
            params=["widget_id"],
            description="Get widget by ID",
            handler_file="widgets.py",
        ),
        EndpointDef(
            method="DELETE",
            path="/api/v1/widgets/:widget_id",
            params=["widget_id"],
            description="Delete a widget",
            handler_file="widgets.py",
        ),
    ]


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_path_params_colon(self):
        assert _extract_path_params("/api/v1/orders/:order_id") == ["order_id"]

    def test_extract_path_params_braces(self):
        assert _extract_path_params("/api/v1/orders/{order_id}") == ["order_id"]

    def test_extract_path_params_multiple(self):
        params = _extract_path_params("/api/v1/:org_id/teams/:team_id")
        assert params == ["org_id", "team_id"]

    def test_extract_path_params_none(self):
        assert _extract_path_params("/api/v1/orders") == []

    def test_normalize_path(self):
        assert _normalize_path("/api/{id}/sub") == "/api/:id/sub"

    def test_to_pascal_case(self):
        assert _to_pascal_case("my-namespace") == "MyNamespace"
        assert _to_pascal_case("some_name") == "SomeName"

    def test_to_camel_case(self):
        assert _to_camel_case("order_id") == "orderId"
        assert _to_camel_case("name") == "name"

    def test_ts_template_path_no_params(self):
        assert _ts_template_path("/api/v1/orders") == "'/api/v1/orders'"

    def test_ts_template_path_with_params(self):
        result = _ts_template_path("/api/v1/orders/:order_id")
        assert result == "`/api/v1/orders/${orderId}`"

    def test_py_format_path_no_params(self):
        assert _py_format_path("/api/v1/orders") == '"/api/v1/orders"'

    def test_py_format_path_with_params(self):
        result = _py_format_path("/api/v1/orders/:order_id")
        assert result == 'f"/api/v1/orders/{order_id}"'

    def test_group_by_namespace(self):
        eps = [
            EndpointDef(method="GET", path="/api/v1/orders"),
            EndpointDef(method="GET", path="/api/v2/orders/:id"),
            EndpointDef(method="GET", path="/api/billing/invoices"),
        ]
        groups = _group_by_namespace(eps)
        assert "orders" in groups
        assert "billing" in groups
        assert len(groups["orders"]) == 2
        assert len(groups["billing"]) == 1


# ---------------------------------------------------------------------------
# Endpoint extraction tests
# ---------------------------------------------------------------------------


class TestExtractEndpoints:
    def test_scan_handlers_finds_all(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        endpoints = generator.scan_handlers(sample_handler_dir)
        # orders: 5 docstring + ROUTES deduped, items: 3, tags: 2, invoices: 2
        assert len(endpoints) >= 10

    def test_scan_skips_private_files(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        endpoints = generator.scan_handlers(sample_handler_dir)
        handler_files = {ep.handler_file for ep in endpoints}
        assert not any("_internal" in f for f in handler_files)

    def test_extract_dash_style_docstring(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        eps = generator.extract_endpoints(sample_handler_dir / "orders.py")
        methods = {(e.method, e.path) for e in eps}
        assert ("GET", "/api/v1/orders") in methods
        assert ("POST", "/api/v1/orders") in methods
        assert ("GET", "/api/v1/orders/:order_id") in methods
        assert ("PUT", "/api/v1/orders/:order_id") in methods
        assert ("DELETE", "/api/v1/orders/:order_id") in methods

    def test_extract_indent_style_docstring(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        eps = generator.extract_endpoints(sample_handler_dir / "items.py")
        methods = {(e.method, e.path) for e in eps}
        assert ("GET", "/api/v2/items") in methods
        assert ("POST", "/api/v2/items") in methods
        assert ("GET", "/api/v2/items/:item_id") in methods

    def test_extract_routes_only(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        eps = generator.extract_endpoints(sample_handler_dir / "tags.py")
        paths = {e.path for e in eps}
        assert "/api/v1/tags" in paths
        assert "/api/v1/tags/popular" in paths

    def test_extract_subdirectory(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        eps = generator.extract_endpoints(
            sample_handler_dir / "billing" / "invoices.py"
        )
        methods = {(e.method, e.path) for e in eps}
        assert ("GET", "/api/v1/billing/invoices") in methods
        assert ("POST", "/api/v1/billing/invoices/:invoice_id/pay") in methods

    def test_extract_path_params_populated(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        eps = generator.extract_endpoints(sample_handler_dir / "orders.py")
        parameterized = [e for e in eps if e.params]
        assert len(parameterized) >= 2
        assert all("order_id" in e.params for e in parameterized)

    def test_extract_description(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        eps = generator.extract_endpoints(sample_handler_dir / "orders.py")
        get_list = [
            e for e in eps if e.method == "GET" and e.path == "/api/v1/orders"
        ]
        assert len(get_list) == 1
        assert "List all orders" in get_list[0].description

    def test_deduplication(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        """Endpoints appearing in both docstring and ROUTES are deduplicated."""
        eps = generator.extract_endpoints(sample_handler_dir / "orders.py")
        get_orders = [
            (e.method, e.path)
            for e in eps
            if e.path == "/api/v1/orders" and e.method == "GET"
        ]
        assert len(get_orders) == 1

    def test_nonexistent_dir_raises(
        self, generator: SDKCodeGenerator, tmp_path: Path
    ):
        with pytest.raises(FileNotFoundError):
            generator.scan_handlers(tmp_path / "nonexistent")

    def test_syntax_error_skipped(
        self, generator: SDKCodeGenerator, tmp_path: Path
    ):
        """Files with syntax errors are silently skipped."""
        handler_dir = tmp_path / "handlers"
        handler_dir.mkdir()
        (handler_dir / "bad.py").write_text("def broken(:\n")
        endpoints = generator.scan_handlers(handler_dir)
        assert endpoints == []


# ---------------------------------------------------------------------------
# Code generation tests
# ---------------------------------------------------------------------------


class TestTypeScriptGeneration:
    def test_generates_class(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_typescript_namespace(
            sample_endpoints, "widgets"
        )
        assert "export class WidgetsAPI" in code
        assert "interface WidgetsAPIClient" in code

    def test_generates_methods(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_typescript_namespace(
            sample_endpoints, "widgets"
        )
        assert "async getWidgets(" in code
        assert "async createWidgets(" in code
        assert "async deleteWidgets" in code

    def test_path_params_in_signature(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_typescript_namespace(
            sample_endpoints, "widgets"
        )
        assert "widgetId: string" in code

    def test_template_literals(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_typescript_namespace(
            sample_endpoints, "widgets"
        )
        assert "${widgetId}" in code

    def test_post_has_body(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_typescript_namespace(
            sample_endpoints, "widgets"
        )
        assert "data: Record<string, unknown>" in code
        assert "{ body: data }" in code

    def test_auto_generated_header(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_typescript_namespace(
            sample_endpoints, "widgets"
        )
        assert "Auto-generated by sdk_codegen.py" in code

    def test_empty_endpoints(self, generator: SDKCodeGenerator):
        code = generator.generate_typescript_namespace([], "empty")
        assert "export class EmptyAPI" in code


class TestPythonGeneration:
    def test_generates_class(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_python_client(sample_endpoints, "widgets")
        assert "class WidgetsAPI:" in code

    def test_generates_methods(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_python_client(sample_endpoints, "widgets")
        assert "def get_widgets(self)" in code
        assert "def create_widgets(self" in code
        assert "def delete_widgets" in code

    def test_path_params_in_signature(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_python_client(sample_endpoints, "widgets")
        assert "widget_id: str" in code

    def test_fstring_paths(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_python_client(sample_endpoints, "widgets")
        assert "{widget_id}" in code
        assert 'f"' in code

    def test_post_has_json_param(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_python_client(sample_endpoints, "widgets")
        assert "json=data" in code

    def test_auto_generated_header(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_python_client(sample_endpoints, "widgets")
        assert "auto-generated by sdk_codegen.py" in code

    def test_type_annotations(
        self,
        generator: SDKCodeGenerator,
        sample_endpoints: list[EndpointDef],
    ):
        code = generator.generate_python_client(sample_endpoints, "widgets")
        assert "-> dict[str, Any]" in code
        assert "from typing import TYPE_CHECKING, Any" in code


# ---------------------------------------------------------------------------
# Coverage validation tests
# ---------------------------------------------------------------------------


class TestCoverageValidation:
    def test_full_coverage(
        self, generator: SDKCodeGenerator, tmp_path: Path
    ):
        sdk_dir = tmp_path / "sdk"
        ts_dir = sdk_dir / "ts"
        ts_dir.mkdir(parents=True)
        (ts_dir / "widgets.ts").write_text(
            "this.client.request('GET', '/api/v1/widgets');\n"
            "this.client.request('POST', '/api/v1/widgets');\n"
        )
        endpoints = [
            EndpointDef(method="GET", path="/api/v1/widgets"),
            EndpointDef(method="POST", path="/api/v1/widgets"),
        ]
        report = generator.validate_sdk_coverage(sdk_dir, endpoints)
        assert report.total_endpoints == 2
        assert report.covered == 2
        assert report.coverage_percent == 100.0
        assert report.missing == []

    def test_partial_coverage(
        self, generator: SDKCodeGenerator, tmp_path: Path
    ):
        sdk_dir = tmp_path / "sdk"
        ts_dir = sdk_dir / "ts"
        ts_dir.mkdir(parents=True)
        (ts_dir / "widgets.ts").write_text(
            "this.client.request('GET', '/api/v1/widgets');\n"
        )
        endpoints = [
            EndpointDef(method="GET", path="/api/v1/widgets"),
            EndpointDef(method="POST", path="/api/v1/widgets/special"),
        ]
        report = generator.validate_sdk_coverage(sdk_dir, endpoints)
        assert report.total_endpoints == 2
        assert report.covered == 1
        assert len(report.missing) == 1
        assert report.coverage_percent == 50.0

    def test_no_coverage(
        self, generator: SDKCodeGenerator, tmp_path: Path
    ):
        sdk_dir = tmp_path / "sdk"
        sdk_dir.mkdir()
        endpoints = [EndpointDef(method="GET", path="/api/v1/missing")]
        report = generator.validate_sdk_coverage(sdk_dir, endpoints)
        assert report.coverage_percent == 0.0

    def test_empty_endpoints(
        self, generator: SDKCodeGenerator, tmp_path: Path
    ):
        sdk_dir = tmp_path / "sdk"
        sdk_dir.mkdir()
        report = generator.validate_sdk_coverage(sdk_dir, [])
        assert report.coverage_percent == 100.0

    def test_template_literal_matching(
        self, generator: SDKCodeGenerator, tmp_path: Path
    ):
        """SDK files using template literals should match :id paths."""
        sdk_dir = tmp_path / "sdk"
        ts_dir = sdk_dir / "ts"
        ts_dir.mkdir(parents=True)
        (ts_dir / "orders.ts").write_text(
            "this.client.request('GET', `/api/v1/orders/${orderId}`);\n"
        )
        endpoints = [
            EndpointDef(method="GET", path="/api/v1/orders/:orderId"),
        ]
        report = generator.validate_sdk_coverage(sdk_dir, endpoints)
        assert report.covered == 1


# ---------------------------------------------------------------------------
# Integration: run method
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_generate_typescript(
        self,
        generator: SDKCodeGenerator,
        sample_handler_dir: Path,
        tmp_path: Path,
    ):
        output = tmp_path / "output"
        report = generator.run(
            sample_handler_dir, output_dir=output, language="typescript"
        )
        assert report.total_endpoints >= 10
        ts_dir = output / "typescript"
        assert ts_dir.is_dir()
        ts_files = list(ts_dir.glob("*.ts"))
        assert len(ts_files) >= 1

    def test_run_generate_python(
        self,
        generator: SDKCodeGenerator,
        sample_handler_dir: Path,
        tmp_path: Path,
    ):
        output = tmp_path / "output"
        report = generator.run(
            sample_handler_dir, output_dir=output, language="python"
        )
        py_dir = output / "python"
        assert py_dir.is_dir()
        py_files = list(py_dir.glob("*.py"))
        assert len(py_files) >= 1

    def test_run_generate_both(
        self,
        generator: SDKCodeGenerator,
        sample_handler_dir: Path,
        tmp_path: Path,
    ):
        output = tmp_path / "output"
        generator.run(
            sample_handler_dir, output_dir=output, language="both"
        )
        assert (output / "typescript").is_dir()
        assert (output / "python").is_dir()

    def test_run_scan_only(
        self, generator: SDKCodeGenerator, sample_handler_dir: Path
    ):
        report = generator.run(sample_handler_dir)
        assert report.total_endpoints >= 10

    def test_generated_ts_is_valid_syntax(
        self,
        generator: SDKCodeGenerator,
        sample_handler_dir: Path,
        tmp_path: Path,
    ):
        """Generated TypeScript should not have obvious syntax issues."""
        output = tmp_path / "output"
        generator.run(
            sample_handler_dir, output_dir=output, language="typescript"
        )
        for ts_file in (output / "typescript").glob("*.ts"):
            content = ts_file.read_text()
            assert content.count("{") == content.count(
                "}"
            ), f"Unbalanced braces in {ts_file.name}"

    def test_generated_py_is_valid_syntax(
        self,
        generator: SDKCodeGenerator,
        sample_handler_dir: Path,
        tmp_path: Path,
    ):
        """Generated Python should parse without syntax errors."""
        import ast as _ast

        output = tmp_path / "output"
        generator.run(
            sample_handler_dir, output_dir=output, language="python"
        )
        for py_file in (output / "python").glob("*.py"):
            content = py_file.read_text()
            _ast.parse(content, filename=str(py_file))


# ---------------------------------------------------------------------------
# EndpointDef and CoverageReport dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_endpoint_def_defaults(self):
        ep = EndpointDef(method="GET", path="/api/v1/test")
        assert ep.params == []
        assert ep.return_type == "unknown"
        assert ep.description == ""
        assert ep.handler_file == ""

    def test_coverage_report_percent(self):
        report = CoverageReport(total_endpoints=200, covered=150)
        assert report.coverage_percent == 75.0

    def test_coverage_report_zero_total(self):
        report = CoverageReport(total_endpoints=0, covered=0)
        assert report.coverage_percent == 100.0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_scan(
        self, sample_handler_dir: Path, capsys: pytest.CaptureFixture
    ):
        from scripts.sdk_codegen import main

        ret = main(["--scan", "--handler-dir", str(sample_handler_dir)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Endpoint Coverage Report" in captured.out
        assert "Total endpoints scanned:" in captured.out

    def test_main_no_args(self, capsys: pytest.CaptureFixture):
        from scripts.sdk_codegen import main

        ret = main([])
        assert ret == 1
