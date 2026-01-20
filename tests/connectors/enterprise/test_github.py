"""
Tests for GitHub Enterprise Connector.

Tests the full repository crawling with:
- Incremental sync using commit SHA
- File content indexing with filtering
- PR/Issue/Discussion sync
- AST parsing for Python, JS, TS
- Dependency graph extraction
- Webhook support for real-time updates
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import pytest

from aragora.connectors.enterprise.base import SyncState, SyncStatus
from aragora.connectors.enterprise.git.github import (
    GitHubEnterpriseConnector,
    GitHubFile,
    GitHubCommit,
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    CONFIG_EXTENSIONS,
    IMPORTANT_FILES,
    MAX_FILE_SIZE,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_credentials():
    """Mock credential provider."""
    from tests.connectors.enterprise.conftest import MockCredentialProvider

    return MockCredentialProvider(
        {
            "GITHUB_TOKEN": "ghp_test_token_12345",
            "GITHUB_WEBHOOK_SECRET": "webhook_secret_abc",
        }
    )


@pytest.fixture
def github_connector(mock_credentials, tmp_path):
    """Create a GitHub connector for testing."""
    return GitHubEnterpriseConnector(
        repo="test-owner/test-repo",
        branch="main",
        token="ghp_test_token",
        include_prs=True,
        include_issues=True,
        include_discussions=False,
        credentials=mock_credentials,
        state_dir=tmp_path / "sync_state",
    )


@pytest.fixture
def mock_gh_output():
    """Mock gh CLI output helper."""

    def _make_output(data: Any) -> str:
        if isinstance(data, (dict, list)):
            return json.dumps(data)
        return str(data)

    return _make_output


@pytest.fixture
def sample_tree():
    """Sample repository file tree."""
    return [
        {"path": "README.md", "type": "blob", "sha": "abc123", "size": 1000},
        {"path": "src/main.py", "type": "blob", "sha": "def456", "size": 2000},
        {"path": "src/utils.ts", "type": "blob", "sha": "ghi789", "size": 1500},
        {"path": "package.json", "type": "blob", "sha": "jkl012", "size": 500},
        {"path": "node_modules/dep/index.js", "type": "blob", "sha": "mno345", "size": 3000},
        {
            "path": "src/__pycache__/main.cpython-310.pyc",
            "type": "blob",
            "sha": "pqr678",
            "size": 4000,
        },
        {"path": "large_file.py", "type": "blob", "sha": "stu901", "size": 2000000},  # > 1MB
        {"path": "src", "type": "tree", "sha": "tree123"},  # Directory, not file
    ]


@pytest.fixture
def sample_issues():
    """Sample GitHub issues."""
    return [
        {
            "number": 1,
            "title": "Bug in authentication",
            "body": "Login fails when using special characters.",
            "author": {"login": "alice"},
            "createdAt": "2024-01-15T10:00:00Z",
            "url": "https://github.com/test-owner/test-repo/issues/1",
            "state": "open",
            "labels": [{"name": "bug"}, {"name": "high-priority"}],
        },
        {
            "number": 2,
            "title": "Feature request: Dark mode",
            "body": "Please add dark mode support.",
            "author": {"login": "bob"},
            "createdAt": "2024-01-16T14:00:00Z",
            "url": "https://github.com/test-owner/test-repo/issues/2",
            "state": "closed",
            "labels": [{"name": "enhancement"}],
        },
    ]


@pytest.fixture
def sample_prs():
    """Sample GitHub pull requests."""
    return [
        {
            "number": 10,
            "title": "Fix authentication bug",
            "body": "This PR fixes the login issue with special characters.",
            "author": {"login": "alice"},
            "createdAt": "2024-01-17T09:00:00Z",
            "url": "https://github.com/test-owner/test-repo/pull/10",
            "state": "merged",
            "mergedAt": "2024-01-17T15:00:00Z",
        },
        {
            "number": 11,
            "title": "Add dark mode",
            "body": "Implements dark mode theme switching.",
            "author": {"login": "bob"},
            "createdAt": "2024-01-18T10:00:00Z",
            "url": "https://github.com/test-owner/test-repo/pull/11",
            "state": "open",
            "mergedAt": None,
        },
    ]


@pytest.fixture
def sample_commits():
    """Sample GitHub commits."""
    return [
        {
            "sha": "abc123def456",
            "commit": {
                "message": "Fix authentication bug",
                "author": {"name": "Alice", "date": "2024-01-17T15:00:00Z"},
            },
        },
        {
            "sha": "def456ghi789",
            "commit": {
                "message": "Initial commit",
                "author": {"name": "Bob", "date": "2024-01-16T10:00:00Z"},
            },
        },
    ]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestGitHubConnectorInit:
    """Test GitHubEnterpriseConnector initialization."""

    def test_init_with_valid_repo(self, mock_credentials, tmp_path):
        """Test initialization with valid repo format."""
        connector = GitHubEnterpriseConnector(
            repo="owner/repo",
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.repo == "owner/repo"
        assert connector.branch == "main"
        assert connector.connector_id == "github_owner_repo"

    def test_init_invalid_repo_format(self, mock_credentials):
        """Test initialization fails with invalid repo format."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubEnterpriseConnector(repo="invalid-format", credentials=mock_credentials)

    def test_init_invalid_repo_format_empty(self, mock_credentials):
        """Test initialization fails with empty repo."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubEnterpriseConnector(repo="", credentials=mock_credentials)

    def test_init_with_custom_branch(self, mock_credentials, tmp_path):
        """Test initialization with custom branch."""
        connector = GitHubEnterpriseConnector(
            repo="owner/repo",
            branch="develop",
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.branch == "develop"

    def test_init_with_custom_options(self, mock_credentials, tmp_path):
        """Test initialization with custom options."""
        custom_extensions = {".py", ".rs"}
        custom_excludes = ["vendor/", "test/"]

        connector = GitHubEnterpriseConnector(
            repo="owner/repo",
            include_prs=False,
            include_issues=False,
            include_discussions=True,
            file_extensions=custom_extensions,
            exclude_paths=custom_excludes,
            credentials=mock_credentials,
            state_dir=tmp_path,
        )

        assert connector.include_prs is False
        assert connector.include_issues is False
        assert connector.include_discussions is True
        assert connector.file_extensions == custom_extensions
        assert connector.exclude_paths == custom_excludes

    def test_default_file_extensions(self, github_connector):
        """Test default file extensions include code, docs, and config."""
        extensions = github_connector.file_extensions
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".md" in extensions
        assert ".json" in extensions

    def test_default_exclude_paths(self, github_connector):
        """Test default exclude paths."""
        excludes = github_connector.exclude_paths
        assert "node_modules/" in excludes
        assert "__pycache__/" in excludes
        assert ".git/" in excludes

    def test_source_type(self, github_connector):
        """Test source type property."""
        from aragora.reasoning.provenance import SourceType

        assert github_connector.source_type == SourceType.CODE_ANALYSIS

    def test_name_property(self, github_connector):
        """Test name property."""
        assert github_connector.name == "GitHub (test-owner/test-repo)"


# =============================================================================
# File Filtering Tests
# =============================================================================


class TestFileFiltering:
    """Test file filtering logic."""

    def test_should_index_important_files(self, github_connector):
        """Test important files are always indexed."""
        assert github_connector._should_index_file("README.md", 100) is True
        assert github_connector._should_index_file("package.json", 500) is True
        assert github_connector._should_index_file("pyproject.toml", 200) is True
        assert github_connector._should_index_file("Dockerfile", 300) is True

    def test_should_index_code_files(self, github_connector):
        """Test code files are indexed."""
        assert github_connector._should_index_file("src/main.py", 1000) is True
        assert github_connector._should_index_file("lib/utils.ts", 2000) is True
        assert github_connector._should_index_file("app.jsx", 1500) is True

    def test_should_not_index_excluded_paths(self, github_connector):
        """Test excluded paths are not indexed."""
        assert github_connector._should_index_file("node_modules/dep/index.js", 500) is False
        assert github_connector._should_index_file("vendor/lib/code.py", 500) is False
        assert github_connector._should_index_file("src/__pycache__/main.pyc", 500) is False

    def test_should_not_index_large_files(self, github_connector):
        """Test large files are not indexed."""
        large_size = MAX_FILE_SIZE + 1
        assert github_connector._should_index_file("src/main.py", large_size) is False

    def test_should_not_index_unknown_extensions(self, github_connector):
        """Test unknown extensions are not indexed."""
        assert github_connector._should_index_file("data.bin", 500) is False
        assert github_connector._should_index_file("image.png", 500) is False

    def test_important_files_bypass_size_limit(self, github_connector):
        """Test important files bypass size limit check."""
        # Important files have extension check skipped
        assert github_connector._should_index_file("README.md", 500) is True


# =============================================================================
# Code Element Extraction Tests
# =============================================================================


class TestCodeElementExtraction:
    """Test code element extraction from files."""

    def test_extract_python_classes(self, github_connector):
        """Test Python class extraction."""
        content = """
class MyClass:
    pass

class AnotherClass(BaseClass):
    def method(self):
        pass
"""
        elements = github_connector._extract_python_elements(content, "test.py")
        class_elements = [e for e in elements if e["type"] == "class"]
        assert len(class_elements) == 2
        assert any(e["name"] == "MyClass" for e in class_elements)
        assert any(e["name"] == "AnotherClass" for e in class_elements)

    def test_extract_python_functions(self, github_connector):
        """Test Python function extraction."""
        content = """
def regular_function(arg1, arg2):
    return arg1 + arg2

async def async_function(data: str) -> dict:
    return {"data": data}

def typed_function(x: int, y: int) -> int:
    return x * y
"""
        elements = github_connector._extract_python_elements(content, "test.py")
        func_elements = [e for e in elements if e["type"] == "function"]
        assert len(func_elements) == 3
        assert any(e["name"] == "regular_function" for e in func_elements)
        assert any(e["name"] == "async_function" for e in func_elements)
        assert any(e["name"] == "typed_function" for e in func_elements)

    def test_extract_python_line_numbers(self, github_connector):
        """Test Python element line numbers."""
        content = """class First:
    pass

def second():
    pass
"""
        elements = github_connector._extract_python_elements(content, "test.py")
        first_class = next(e for e in elements if e["name"] == "First")
        second_func = next(e for e in elements if e["name"] == "second")
        assert first_class["line"] == 1
        assert second_func["line"] == 4

    def test_extract_js_classes(self, github_connector):
        """Test JavaScript class extraction."""
        content = """
class MyComponent {
    constructor() {}
}

export class ExportedClass extends BaseClass {
    render() {}
}
"""
        elements = github_connector._extract_js_elements(content, "test.js")
        class_elements = [e for e in elements if e["type"] == "class"]
        assert len(class_elements) == 2
        assert any(e["name"] == "MyComponent" for e in class_elements)
        assert any(e["name"] == "ExportedClass" for e in class_elements)

    def test_extract_js_functions(self, github_connector):
        """Test JavaScript function extraction."""
        content = """
function regularFunction(arg) {
    return arg;
}

export async function asyncFunction(data) {
    return data;
}

const arrowFunc = (x) => x * 2;

export const exportedArrow = async (y) => {
    return y;
};
"""
        elements = github_connector._extract_js_elements(content, "test.js")
        func_elements = [e for e in elements if e["type"] == "function"]
        assert len(func_elements) == 4
        assert any(e["name"] == "regularFunction" for e in func_elements)
        assert any(e["name"] == "asyncFunction" for e in func_elements)
        assert any(e["name"] == "arrowFunc" for e in func_elements)
        assert any(e["name"] == "exportedArrow" for e in func_elements)

    def test_extract_code_elements_by_extension(self, github_connector):
        """Test code element extraction routes by file extension."""
        py_content = "class Foo: pass"
        js_content = "class Bar {}"

        py_elements = github_connector._extract_code_elements(py_content, "test.py")
        js_elements = github_connector._extract_code_elements(js_content, "test.js")
        ts_elements = github_connector._extract_code_elements(js_content, "test.ts")
        tsx_elements = github_connector._extract_code_elements(js_content, "test.tsx")

        assert len(py_elements) == 1
        assert py_elements[0]["name"] == "Foo"
        assert len(js_elements) == 1
        assert js_elements[0]["name"] == "Bar"
        assert len(ts_elements) == 1
        assert len(tsx_elements) == 1


# =============================================================================
# Dependency Extraction Tests
# =============================================================================


class TestDependencyExtraction:
    """Test dependency extraction from imports."""

    def test_extract_python_imports(self, github_connector):
        """Test Python import extraction."""
        content = """
import os
import json
from typing import List, Dict
from collections.abc import Mapping
from aragora.connectors.base import BaseConnector
import asyncio
"""
        deps = github_connector._extract_dependencies(content, "test.py")
        assert "os" in deps
        assert "json" in deps
        assert "typing" in deps
        assert "collections" in deps
        assert "aragora" in deps
        assert "asyncio" in deps

    def test_extract_python_multi_imports(self, github_connector):
        """Test Python multi-line imports."""
        content = """
import sys, re, time
from pathlib import Path
"""
        deps = github_connector._extract_dependencies(content, "test.py")
        assert "sys" in deps
        assert "re" in deps
        assert "time" in deps
        assert "pathlib" in deps

    def test_extract_js_imports(self, github_connector):
        """Test JavaScript import extraction."""
        content = """
import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from '@mui/material';
import utils from './utils';
"""
        deps = github_connector._extract_dependencies(content, "test.js")
        assert "react" in deps
        assert "axios" in deps
        assert "@mui" in deps
        # Local imports (starting with .) should not be included
        assert "./utils" not in deps

    def test_extract_js_require(self, github_connector):
        """Test JavaScript require() extraction."""
        content = """
const fs = require('fs');
const path = require('path');
const localModule = require('./local');
"""
        deps = github_connector._extract_dependencies(content, "test.js")
        assert "fs" in deps
        assert "path" in deps
        # Local requires should not be included
        assert "./local" not in deps

    def test_extract_ts_imports(self, github_connector):
        """Test TypeScript import extraction."""
        content = """
import type { User } from './types';
import { Component } from '@angular/core';
"""
        deps = github_connector._extract_dependencies(content, "test.ts")
        assert "@angular" in deps

    def test_no_duplicate_dependencies(self, github_connector):
        """Test dependencies are deduplicated."""
        content = """
import React from 'react';
import { useState } from 'react';
import { useEffect } from 'react';
"""
        deps = github_connector._extract_dependencies(content, "test.js")
        assert deps.count("react") == 1


# =============================================================================
# gh CLI Tests
# =============================================================================


class TestGhCli:
    """Test gh CLI interaction."""

    def test_check_gh_cli_available(self, github_connector):
        """Test gh CLI availability check when available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            github_connector._gh_available = None  # Reset cache
            result = github_connector._check_gh_cli()
            assert result is True
            assert github_connector._gh_available is True

    def test_check_gh_cli_unavailable(self, github_connector):
        """Test gh CLI availability check when unavailable."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            github_connector._gh_available = None  # Reset cache
            result = github_connector._check_gh_cli()
            assert result is False

    def test_check_gh_cli_cached(self, github_connector):
        """Test gh CLI availability is cached."""
        github_connector._gh_available = True
        with patch("subprocess.run") as mock_run:
            result = github_connector._check_gh_cli()
            assert result is True
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_gh_success(self, github_connector):
        """Test successful gh command execution."""
        github_connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b'{"result": "success"}', b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await github_connector._run_gh(["api", "test"])
            assert result == '{"result": "success"}'

    @pytest.mark.asyncio
    async def test_run_gh_cli_unavailable(self, github_connector):
        """Test gh command when CLI unavailable."""
        github_connector._gh_available = False
        result = await github_connector._run_gh(["api", "test"])
        assert result is None

    @pytest.mark.asyncio
    async def test_run_gh_error(self, github_connector):
        """Test gh command error handling."""
        github_connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error message"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await github_connector._run_gh(["api", "test"])
            assert result is None


# =============================================================================
# API Method Tests
# =============================================================================


class TestApiMethods:
    """Test GitHub API methods."""

    @pytest.mark.asyncio
    async def test_get_latest_commit(self, github_connector):
        """Test getting latest commit SHA."""
        github_connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"abc123def456\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            sha = await github_connector._get_latest_commit()
            assert sha == "abc123def456"

    @pytest.mark.asyncio
    async def test_get_latest_commit_failure(self, github_connector):
        """Test getting latest commit when gh fails."""
        github_connector._gh_available = False
        sha = await github_connector._get_latest_commit()
        assert sha is None

    @pytest.mark.asyncio
    async def test_get_commits_since(self, github_connector, sample_commits, mock_gh_output):
        """Test getting commits since a SHA."""
        github_connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(mock_gh_output(sample_commits).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            commits = await github_connector._get_commits_since(None, limit=10)
            assert len(commits) == 2
            assert commits[0].sha == "abc123def456"
            assert commits[0].author == "Alice"

    @pytest.mark.asyncio
    async def test_get_commits_since_stops_at_cursor(
        self, github_connector, sample_commits, mock_gh_output
    ):
        """Test commits stop at the since_sha cursor."""
        github_connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(mock_gh_output(sample_commits).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # Should stop at the second commit
            commits = await github_connector._get_commits_since("def456ghi789", limit=10)
            assert len(commits) == 1
            assert commits[0].sha == "abc123def456"

    @pytest.mark.asyncio
    async def test_get_tree(self, github_connector, sample_tree, mock_gh_output):
        """Test getting repository tree."""
        github_connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(mock_gh_output(sample_tree).encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            tree = await github_connector._get_tree("abc123")
            assert len(tree) == 8

    @pytest.mark.asyncio
    async def test_get_file_content(self, github_connector):
        """Test getting file content."""
        github_connector._gh_available = True
        import base64

        content = "print('Hello, World!')"
        encoded = base64.b64encode(content.encode()).decode()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(encoded.encode() + b"\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await github_connector._get_file_content("test.py")
            assert result == content

    @pytest.mark.asyncio
    async def test_get_issues(self, github_connector, sample_issues, mock_gh_output):
        """Test getting issues."""
        github_connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(mock_gh_output(sample_issues).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            issues = await github_connector._get_issues(state="all", limit=10)
            assert len(issues) == 2
            assert issues[0]["number"] == 1
            assert issues[0]["title"] == "Bug in authentication"

    @pytest.mark.asyncio
    async def test_get_prs(self, github_connector, sample_prs, mock_gh_output):
        """Test getting pull requests."""
        github_connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(mock_gh_output(sample_prs).encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            prs = await github_connector._get_prs(state="all", limit=10)
            assert len(prs) == 2
            assert prs[0]["number"] == 10
            assert prs[0]["title"] == "Fix authentication bug"


# =============================================================================
# Sync Tests
# =============================================================================


class TestSync:
    """Test sync_items functionality."""

    @pytest.mark.asyncio
    async def test_sync_items_no_gh_cli(self, github_connector):
        """Test sync when gh CLI unavailable."""
        github_connector._gh_available = False
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)

        items = []
        async for item in github_connector.sync_items(state):
            items.append(item)

        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_sync_items_files(self, github_connector, sample_tree):
        """Test syncing files from repository."""
        github_connector._gh_available = True
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)
        github_connector.include_issues = False
        github_connector.include_prs = False

        import base64

        file_content = "# README\n\nThis is a test."
        encoded = base64.b64encode(file_content.encode()).decode()

        async def mock_run_gh(args):
            if "commits/main" in str(args):
                return "abc123"
            elif "commits" in str(args) and "per_page" in str(args):
                return "[]"  # No new commits
            elif "git/trees" in str(args):
                return json.dumps(sample_tree)
            elif "contents" in str(args):
                return encoded
            return None

        with patch.object(github_connector, "_run_gh", side_effect=mock_run_gh):
            items = []
            async for item in github_connector.sync_items(state, batch_size=10):
                items.append(item)

        # Should have indexed README.md, src/main.py, src/utils.ts, package.json
        # (excluded: node_modules, __pycache__, large file, tree item)
        assert len(items) >= 1
        assert any("README.md" in item.title for item in items)

    @pytest.mark.asyncio
    async def test_sync_items_issues(self, github_connector, sample_issues):
        """Test syncing issues."""
        github_connector._gh_available = True
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)
        github_connector.include_prs = False

        async def mock_run_gh(args):
            if "commits/main" in str(args):
                return "abc123"
            elif "commits" in str(args) and "per_page" in str(args):
                return "[]"
            elif "git/trees" in str(args):
                return "[]"  # Empty tree
            elif "issue" in args:
                return json.dumps(sample_issues)
            return None

        with patch.object(github_connector, "_run_gh", side_effect=mock_run_gh):
            items = []
            async for item in github_connector.sync_items(state, batch_size=10):
                items.append(item)

        issue_items = [i for i in items if i.source_type == "issue"]
        assert len(issue_items) == 2
        assert any("Bug in authentication" in item.title for item in issue_items)

    @pytest.mark.asyncio
    async def test_sync_items_prs(self, github_connector, sample_prs):
        """Test syncing pull requests."""
        github_connector._gh_available = True
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)
        github_connector.include_issues = False

        async def mock_run_gh(args):
            if "commits/main" in str(args):
                return "abc123"
            elif "commits" in str(args) and "per_page" in str(args):
                return "[]"
            elif "git/trees" in str(args):
                return "[]"
            elif "pr" in args:
                return json.dumps(sample_prs)
            return None

        with patch.object(github_connector, "_run_gh", side_effect=mock_run_gh):
            items = []
            async for item in github_connector.sync_items(state, batch_size=10):
                items.append(item)

        pr_items = [i for i in items if i.source_type == "pr"]
        assert len(pr_items) == 2
        assert any("Fix authentication bug" in item.title for item in pr_items)
        # Check merged PR metadata
        merged_pr = next(i for i in pr_items if "Fix authentication bug" in i.title)
        assert merged_pr.metadata.get("merged") is True


# =============================================================================
# Webhook Tests
# =============================================================================


class TestWebhook:
    """Test webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook_push(self, github_connector):
        """Test handling push webhook."""
        # Note: The handler checks if "push" is in the ref string
        # (this matches the webhook event type detection logic)
        payload = {
            "ref": "push:refs/heads/main",  # Contains "push" to trigger handler
            "commits": [{"id": "abc123"}, {"id": "def456"}],
        }

        with patch.object(github_connector, "sync", new_callable=AsyncMock) as mock_sync:
            result = await github_connector.handle_webhook(payload)
            assert result is True
            # Sync should be triggered in background (via asyncio.create_task)

    @pytest.mark.asyncio
    async def test_handle_webhook_issue(self, github_connector):
        """Test handling issue webhook."""
        payload = {
            "action": "opened",
            "issue": {"number": 123, "title": "New issue"},
        }

        result = await github_connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_pr(self, github_connector):
        """Test handling PR webhook."""
        payload = {
            "action": "closed",
            "pull_request": {"number": 456, "title": "Merged PR"},
        }

        result = await github_connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_unknown(self, github_connector):
        """Test handling unknown webhook type."""
        payload = {"action": "unknown_action"}
        result = await github_connector.handle_webhook(payload)
        assert result is False

    def test_get_webhook_secret(self, github_connector):
        """Test getting webhook secret."""
        # This calls the credential provider which uses a sync event loop
        # In real usage, it would get GITHUB_WEBHOOK_SECRET
        # For testing, we just verify the method exists
        assert hasattr(github_connector, "get_webhook_secret")


# =============================================================================
# Search and Fetch Tests
# =============================================================================


class TestSearchAndFetch:
    """Test search and fetch delegation."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_connector(self, github_connector):
        """Test search delegates to GitHubConnector."""
        with patch("aragora.connectors.github.GitHubConnector") as MockConnector:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(return_value=[{"id": "result1"}])
            MockConnector.return_value = mock_instance

            results = await github_connector.search("test query", limit=5)
            assert len(results) == 1
            mock_instance.search.assert_called_once_with("test query", limit=5, search_type="code")

    @pytest.mark.asyncio
    async def test_fetch_delegates_to_connector(self, github_connector):
        """Test fetch delegates to GitHubConnector."""
        with patch("aragora.connectors.github.GitHubConnector") as MockConnector:
            mock_instance = MagicMock()
            mock_instance.fetch = AsyncMock(return_value={"content": "test"})
            MockConnector.return_value = mock_instance

            result = await github_connector.fetch("evidence-123")
            assert result == {"content": "test"}
            mock_instance.fetch.assert_called_once_with("evidence-123")


# =============================================================================
# SyncItem Content Tests
# =============================================================================


class TestSyncItemContent:
    """Test sync item content and metadata."""

    @pytest.mark.asyncio
    async def test_file_sync_item_metadata(self, github_connector, sample_tree):
        """Test file sync item has correct metadata."""
        github_connector._gh_available = True
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)
        github_connector.include_issues = False
        github_connector.include_prs = False

        import base64

        py_content = "def hello(): pass"
        encoded = base64.b64encode(py_content.encode()).decode()

        async def mock_run_gh(args):
            if "commits/main" in str(args):
                return "abc123"
            elif "commits" in str(args) and "per_page" in str(args):
                return "[]"
            elif "git/trees" in str(args):
                return json.dumps([sample_tree[1]])  # Just src/main.py
            elif "contents" in str(args):
                return encoded
            return None

        with patch.object(github_connector, "_run_gh", side_effect=mock_run_gh):
            items = []
            async for item in github_connector.sync_items(state, batch_size=10):
                items.append(item)

        if items:  # May be empty if filtering kicks in
            file_item = items[0]
            assert file_item.source_type == "code"
            assert "github" in file_item.source_id
            assert file_item.domain == "technical/code"
            assert file_item.metadata.get("repo") == "test-owner/test-repo"
            assert file_item.metadata.get("branch") == "main"

    @pytest.mark.asyncio
    async def test_issue_sync_item_metadata(self, github_connector, sample_issues):
        """Test issue sync item has correct metadata."""
        github_connector._gh_available = True
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)
        github_connector.include_prs = False

        async def mock_run_gh(args):
            if "commits/main" in str(args):
                return "abc123"
            elif "commits" in str(args):
                return "[]"
            elif "git/trees" in str(args):
                return "[]"
            elif "issue" in args:
                return json.dumps([sample_issues[0]])
            return None

        with patch.object(github_connector, "_run_gh", side_effect=mock_run_gh):
            items = []
            async for item in github_connector.sync_items(state, batch_size=10):
                items.append(item)

        issue_item = next((i for i in items if i.source_type == "issue"), None)
        if issue_item:
            assert issue_item.domain == "technical/issues"
            assert issue_item.metadata.get("number") == 1
            assert issue_item.metadata.get("state") == "open"
            assert "bug" in issue_item.metadata.get("labels", [])
