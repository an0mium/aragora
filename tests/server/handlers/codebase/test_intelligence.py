"""
Tests for the Code Intelligence API handlers.

Tests HTTP endpoints for:
- Codebase analysis
- Symbol extraction
- Call graph generation
- Dead code detection
- Impact analysis
- Codebase understanding queries
- Comprehensive audits
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def parse_result(result):
    """Parse a HandlerResult into status and data/error."""
    body = json.loads(result.body.decode("utf-8"))
    return result.status_code, body


class TestHandleAnalyzeCodebase:
    """Tests for handle_analyze_codebase endpoint."""

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a sample codebase for testing."""
        # Create a simple Python file
        py_file = tmp_path / "main.py"
        py_file.write_text('''
"""Main module."""

class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

def main():
    """Entry point."""
    calc = Calculator()
    print(calc.add(1, 2))
''')

        # Create a utils file
        utils_file = tmp_path / "utils.py"
        utils_file.write_text('''
"""Utility functions."""

from typing import List

def format_list(items: List[str]) -> str:
    """Format a list as a string."""
    return ", ".join(items)
''')

        return tmp_path

    @pytest.mark.asyncio
    async def test_analyze_codebase_success(self, sample_codebase: Path):
        """Test successful codebase analysis."""
        from aragora.server.handlers.codebase.intelligence import handle_analyze_codebase

        result = await handle_analyze_codebase("test-repo", {"path": str(sample_codebase)})

        status, body = parse_result(result)
        assert status == 200
        assert body["success"] is True
        data = body["data"]
        assert data["status"] == "completed"
        assert data["analysis_id"] is not None
        assert data["summary"]["total_files"] >= 2
        assert data["summary"]["classes"] >= 1
        assert data["summary"]["functions"] >= 1
        assert "python" in data["summary"]["languages"]

    @pytest.mark.asyncio
    async def test_analyze_codebase_missing_path(self):
        """Test analysis with missing path parameter."""
        from aragora.server.handlers.codebase.intelligence import handle_analyze_codebase

        result = await handle_analyze_codebase("test-repo", {})

        status, body = parse_result(result)
        assert status == 400
        assert "Missing required field: path" in body["error"]

    @pytest.mark.asyncio
    async def test_analyze_codebase_nonexistent_path(self):
        """Test analysis with non-existent path."""
        from aragora.server.handlers.codebase.intelligence import handle_analyze_codebase

        result = await handle_analyze_codebase("test-repo", {"path": "/nonexistent/path/to/code"})

        status, body = parse_result(result)
        assert status == 404
        assert "does not exist" in body["error"]

    @pytest.mark.asyncio
    async def test_analyze_codebase_with_exclude_patterns(self, sample_codebase: Path):
        """Test analysis with custom exclude patterns."""
        from aragora.server.handlers.codebase.intelligence import handle_analyze_codebase

        # Create a file that should be excluded
        excluded = sample_codebase / "test_stuff.py"
        excluded.write_text("# Test file")

        result = await handle_analyze_codebase(
            "test-repo",
            {
                "path": str(sample_codebase),
                "exclude_patterns": ["test_*"],
            },
        )

        status, body = parse_result(result)
        assert status == 200
        # Check that test_stuff.py is not in the files
        files = [f["path"] for f in body["data"]["files"]]
        assert not any("test_stuff" in f for f in files)


class TestHandleGetSymbols:
    """Tests for handle_get_symbols endpoint."""

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a sample codebase for testing."""
        py_file = tmp_path / "models.py"
        py_file.write_text('''
"""Models module."""

class User:
    """User model."""

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

class Admin(User):
    """Admin user model."""

    def has_permission(self, permission: str) -> bool:
        return True

def create_user(name: str) -> User:
    """Factory function for User."""
    return User(name)
''')
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_symbols_all(self, sample_codebase: Path):
        """Test getting all symbols."""
        from aragora.server.handlers.codebase.intelligence import handle_get_symbols

        result = await handle_get_symbols("test-repo", {"path": str(sample_codebase)})

        status, body = parse_result(result)
        assert status == 200
        symbols = body["data"]["symbols"]
        assert len(symbols) > 0

        # Check we have classes and functions
        kinds = [s["kind"] for s in symbols]
        assert "class" in kinds
        assert "function" in kinds

    @pytest.mark.asyncio
    async def test_get_symbols_filter_by_type(self, sample_codebase: Path):
        """Test filtering symbols by type."""
        from aragora.server.handlers.codebase.intelligence import handle_get_symbols

        result = await handle_get_symbols(
            "test-repo", {"path": str(sample_codebase), "type": "class"}
        )

        status, body = parse_result(result)
        assert status == 200
        symbols = body["data"]["symbols"]
        assert all(s["kind"] == "class" for s in symbols)

    @pytest.mark.asyncio
    async def test_get_symbols_filter_by_name(self, sample_codebase: Path):
        """Test filtering symbols by name."""
        from aragora.server.handlers.codebase.intelligence import handle_get_symbols

        result = await handle_get_symbols(
            "test-repo", {"path": str(sample_codebase), "name": "user"}
        )

        status, body = parse_result(result)
        assert status == 200
        symbols = body["data"]["symbols"]
        assert all("user" in s["name"].lower() for s in symbols)


class TestHandleGetCallgraph:
    """Tests for handle_get_callgraph endpoint."""

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a sample codebase with function calls."""
        py_file = tmp_path / "app.py"
        py_file.write_text('''
"""Application module."""

def helper():
    """Helper function."""
    return 42

def process():
    """Process data."""
    value = helper()
    return value * 2

def main():
    """Entry point."""
    result = process()
    print(result)
''')
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_callgraph_success(self, sample_codebase: Path):
        """Test successful call graph generation."""
        from aragora.server.handlers.codebase.intelligence import handle_get_callgraph

        result = await handle_get_callgraph("test-repo", {"path": str(sample_codebase)})

        status, body = parse_result(result)
        assert status == 200
        data = body["data"]
        assert "metrics" in data
        assert "nodes" in data
        assert "edges" in data
        assert "hotspots" in data

    @pytest.mark.asyncio
    async def test_get_callgraph_missing_path(self):
        """Test call graph with missing path."""
        from aragora.server.handlers.codebase.intelligence import handle_get_callgraph

        result = await handle_get_callgraph("test-repo", {})

        status, body = parse_result(result)
        assert status == 400
        assert "Missing required parameter: path" in body["error"]


class TestHandleFindDeadcode:
    """Tests for handle_find_deadcode endpoint."""

    @pytest.fixture
    def codebase_with_deadcode(self, tmp_path: Path) -> Path:
        """Create a codebase with dead code."""
        py_file = tmp_path / "code.py"
        py_file.write_text('''
"""Module with dead code."""

def used_function():
    """This is used."""
    return 1

def unused_function():
    """This is never called."""
    return 2

def main():
    """Entry point."""
    return used_function()
''')
        return tmp_path

    @pytest.mark.asyncio
    async def test_find_deadcode_success(self, codebase_with_deadcode: Path):
        """Test finding dead code."""
        from aragora.server.handlers.codebase.intelligence import handle_find_deadcode

        result = await handle_find_deadcode(
            "test-repo", {"path": str(codebase_with_deadcode), "entry_points": "main"}
        )

        status, body = parse_result(result)
        assert status == 200
        data = body["data"]
        assert "unreachable_functions" in data
        assert "summary" in data


class TestHandleAnalyzeImpact:
    """Tests for handle_analyze_impact endpoint."""

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a sample codebase for impact analysis."""
        py_file = tmp_path / "services.py"
        py_file.write_text('''
"""Services module."""

def get_data():
    """Get some data."""
    return [1, 2, 3]

def process_data(data):
    """Process the data."""
    return sum(data)

def render_result(value):
    """Render the result."""
    return f"Result: {value}"

def main():
    """Main flow."""
    data = get_data()
    processed = process_data(data)
    return render_result(processed)
''')
        return tmp_path

    @pytest.mark.asyncio
    async def test_analyze_impact_success(self, sample_codebase: Path):
        """Test successful impact analysis."""
        from aragora.server.handlers.codebase.intelligence import handle_analyze_impact

        result = await handle_analyze_impact(
            "test-repo",
            {
                "path": str(sample_codebase),
                "symbol": "get_data",
            },
        )

        status, body = parse_result(result)
        assert status == 200
        data = body["data"]
        assert "changed_node" in data
        assert "directly_affected" in data
        assert "transitively_affected" in data
        assert "risk_level" in data

    @pytest.mark.asyncio
    async def test_analyze_impact_missing_symbol(self, sample_codebase: Path):
        """Test impact analysis with missing symbol."""
        from aragora.server.handlers.codebase.intelligence import handle_analyze_impact

        result = await handle_analyze_impact("test-repo", {"path": str(sample_codebase)})

        status, body = parse_result(result)
        assert status == 400
        assert "Missing required field: symbol" in body["error"]


class TestHandleUnderstand:
    """Tests for handle_understand endpoint."""

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a sample codebase for understanding."""
        py_file = tmp_path / "auth.py"
        py_file.write_text('''
"""Authentication module."""

class AuthService:
    """Service for handling authentication."""

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate a user with username and password."""
        # Check credentials
        return self._verify_credentials(username, password)

    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials against the database."""
        return True
''')
        return tmp_path

    @pytest.mark.asyncio
    async def test_understand_missing_question(self, sample_codebase: Path):
        """Test understanding with missing question."""
        from aragora.server.handlers.codebase.intelligence import handle_understand

        result = await handle_understand("test-repo", {"path": str(sample_codebase)})

        status, body = parse_result(result)
        assert status == 400
        assert "Missing required field: question" in body["error"]

    @pytest.mark.asyncio
    async def test_understand_missing_path(self):
        """Test understanding with missing path."""
        from aragora.server.handlers.codebase.intelligence import handle_understand

        result = await handle_understand("test-repo", {"question": "How does authentication work?"})

        status, body = parse_result(result)
        assert status == 400
        assert "Missing required field: path" in body["error"]


class TestHandleAudit:
    """Tests for handle_audit endpoint."""

    @pytest.fixture
    def vulnerable_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase with vulnerabilities."""
        py_file = tmp_path / "insecure.py"
        py_file.write_text('''
"""Insecure code for testing."""

import os

# Hardcoded credentials
API_KEY = "sk-1234567890abcdef1234567890abcdef"

def execute_command(user_input):
    """Execute a shell command."""
    # Command injection vulnerability
    os.system(f"echo {user_input}")

def query_database(user_id):
    """Query the database."""
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query
''')
        return tmp_path

    @pytest.mark.asyncio
    async def test_audit_sync_success(self, vulnerable_codebase: Path):
        """Test synchronous audit execution."""
        from aragora.server.handlers.codebase.intelligence import handle_audit

        result = await handle_audit(
            "test-repo",
            {
                "path": str(vulnerable_codebase),
                "include_security": True,
                "include_bugs": True,
                "async": False,
            },
        )

        status, body = parse_result(result)
        assert status == 200
        data = body["data"]
        assert data["status"] == "completed"
        assert "security_findings" in data
        assert "bug_findings" in data
        assert "risk_score" in data
        # Should find security issues
        assert len(data["security_findings"]) > 0

    @pytest.mark.asyncio
    async def test_audit_async_success(self, vulnerable_codebase: Path):
        """Test asynchronous audit execution."""
        from aragora.server.handlers.codebase.intelligence import handle_audit

        result = await handle_audit(
            "test-repo",
            {
                "path": str(vulnerable_codebase),
                "async": True,
            },
        )

        status, body = parse_result(result)
        assert status == 200
        data = body["data"]
        assert data["status"] == "running"
        assert "audit_id" in data

    @pytest.mark.asyncio
    async def test_audit_missing_path(self):
        """Test audit with missing path."""
        from aragora.server.handlers.codebase.intelligence import handle_audit

        result = await handle_audit("test-repo", {})

        status, body = parse_result(result)
        assert status == 400
        assert "Missing required field: path" in body["error"]


class TestHandleGetAuditStatus:
    """Tests for handle_get_audit_status endpoint."""

    @pytest.mark.asyncio
    async def test_get_audit_status_not_found(self):
        """Test getting status for non-existent audit."""
        from aragora.server.handlers.codebase.intelligence import handle_get_audit_status

        result = await handle_get_audit_status("test-repo", "nonexistent-audit-id", {})

        status, body = parse_result(result)
        assert status == 404
        assert "not found" in body["error"].lower()


class TestIntelligenceHandler:
    """Tests for the IntelligenceHandler class."""

    def test_can_handle_routes(self):
        """Test route matching."""
        from aragora.server.handlers.codebase.intelligence import IntelligenceHandler

        # Should handle exact routes
        assert IntelligenceHandler.can_handle("/api/codebase/analyze")
        assert IntelligenceHandler.can_handle("/api/v1/codebase/symbols")
        assert IntelligenceHandler.can_handle("/api/codebase/callgraph")

        # Should handle routes with repo_id
        assert IntelligenceHandler.can_handle("/api/codebase/my-repo/analyze")
        assert IntelligenceHandler.can_handle("/api/v1/codebase/my-repo/audit")

        # Should not handle unrelated routes
        assert not IntelligenceHandler.can_handle("/api/debates/")
        assert not IntelligenceHandler.can_handle("/api/agents/")

    def test_routes_attribute(self):
        """Test ROUTES attribute is defined."""
        from aragora.server.handlers.codebase.intelligence import IntelligenceHandler

        assert hasattr(IntelligenceHandler, "ROUTES")
        assert len(IntelligenceHandler.ROUTES) > 0
        assert "/api/codebase/analyze" in IntelligenceHandler.ROUTES


class TestServiceRegistry:
    """Tests for service registry integration."""

    def test_code_intelligence_lazy_loading(self):
        """Test CodeIntelligence is lazily loaded."""
        from aragora.server.handlers.codebase.intelligence import _get_code_intelligence

        intel = _get_code_intelligence()
        assert intel is not None

    def test_call_graph_builder_lazy_loading(self):
        """Test CallGraphBuilder is lazily loaded."""
        from aragora.server.handlers.codebase.intelligence import _get_call_graph_builder

        builder = _get_call_graph_builder()
        assert builder is not None

    def test_security_scanner_lazy_loading(self):
        """Test SecurityScanner is lazily loaded."""
        from aragora.server.handlers.codebase.intelligence import _get_security_scanner

        scanner = _get_security_scanner()
        assert scanner is not None

    def test_bug_detector_lazy_loading(self):
        """Test BugDetector is lazily loaded."""
        from aragora.server.handlers.codebase.intelligence import _get_bug_detector

        detector = _get_bug_detector()
        assert detector is not None


class TestQuickHelpers:
    """Tests for quick helper functions."""

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a minimal codebase."""
        py_file = tmp_path / "simple.py"
        py_file.write_text("def foo(): pass")
        return tmp_path

    @pytest.mark.asyncio
    async def test_quick_analyze(self, sample_codebase: Path):
        """Test quick_analyze helper."""
        from aragora.server.handlers.codebase.intelligence import quick_analyze

        result = await quick_analyze(str(sample_codebase))
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_quick_audit(self, sample_codebase: Path):
        """Test quick_audit helper."""
        from aragora.server.handlers.codebase.intelligence import quick_audit

        result = await quick_audit(str(sample_codebase))
        assert "status" in result
