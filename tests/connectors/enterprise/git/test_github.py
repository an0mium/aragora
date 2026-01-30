"""
Tests for GitHub Enterprise Connector.

Comprehensive test coverage for repository operations including:
- Client initialization and configuration
- Repository operations (trees, commits, files)
- Pull request operations
- Issue operations
- Webhook handling
- File/content operations
- Error handling and edge cases
"""

import base64
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.base import SyncState, SyncStatus
from aragora.connectors.enterprise.git.github import (
    CODE_EXTENSIONS,
    CONFIG_EXTENSIONS,
    DOC_EXTENSIONS,
    IMPORTANT_FILES,
    MAX_FILE_SIZE,
    GitHubCommit,
    GitHubEnterpriseConnector,
    GitHubFile,
)
from aragora.reasoning.provenance import SourceType


# =============================================================================
# Fixtures
# =============================================================================


class MockCredentialProvider:
    """Mock credential provider for testing."""

    def __init__(self, credentials: dict[str, str] | None = None):
        self._credentials = credentials or {}

    async def get_credential(self, key: str) -> str | None:
        return self._credentials.get(key)

    async def set_credential(self, key: str, value: str) -> None:
        self._credentials[key] = value


@pytest.fixture
def mock_credentials():
    """Mock credential provider with GitHub tokens."""
    return MockCredentialProvider(
        {
            "GITHUB_TOKEN": "ghp_test_token_12345",
            "GITHUB_WEBHOOK_SECRET": "webhook_secret_abc123",
        }
    )


@pytest.fixture
def connector(mock_credentials, tmp_path):
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
        {"path": "large_file.py", "type": "blob", "sha": "stu901", "size": 2000000},
        {"path": "src", "type": "tree", "sha": "tree123"},
        {"path": "config.yaml", "type": "blob", "sha": "cfg111", "size": 300},
        {"path": "Dockerfile", "type": "blob", "sha": "dkr222", "size": 400},
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
        {
            "number": 3,
            "title": "Documentation update needed",
            "body": None,
            "author": {"login": "charlie"},
            "createdAt": "2024-01-17T08:00:00Z",
            "url": "https://github.com/test-owner/test-repo/issues/3",
            "state": "open",
            "labels": [],
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
            "sha": "abc123def456789012345678901234567890abcd",
            "commit": {
                "message": "Fix authentication bug\n\nAdded input sanitization",
                "author": {"name": "Alice", "date": "2024-01-17T15:00:00Z"},
            },
        },
        {
            "sha": "def456ghi789012345678901234567890bcdef",
            "commit": {
                "message": "Initial commit",
                "author": {"name": "Bob", "date": "2024-01-16T10:00:00Z"},
            },
        },
    ]


# =============================================================================
# Test Class: Client Initialization
# =============================================================================


class TestClientInitialization:
    """Tests for GitHubEnterpriseConnector initialization."""

    def test_init_valid_repo_format(self, mock_credentials, tmp_path):
        """Test connector initializes with valid owner/repo format."""
        connector = GitHubEnterpriseConnector(
            repo="owner/repo",
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.repo == "owner/repo"
        assert connector.branch == "main"
        assert connector.connector_id == "github_owner_repo"

    def test_init_invalid_repo_format_no_slash(self, mock_credentials):
        """Test initialization fails without owner/repo format."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubEnterpriseConnector(repo="invalid-format", credentials=mock_credentials)

    def test_init_invalid_repo_format_empty(self, mock_credentials):
        """Test initialization fails with empty repo string."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubEnterpriseConnector(repo="", credentials=mock_credentials)

    def test_init_invalid_repo_format_multiple_slashes(self, mock_credentials):
        """Test initialization fails with too many slashes."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubEnterpriseConnector(repo="owner/repo/extra", credentials=mock_credentials)

    def test_init_custom_branch(self, mock_credentials, tmp_path):
        """Test initialization with custom branch."""
        connector = GitHubEnterpriseConnector(
            repo="owner/repo",
            branch="develop",
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.branch == "develop"

    def test_init_with_token(self, mock_credentials, tmp_path):
        """Test initialization with explicit token."""
        connector = GitHubEnterpriseConnector(
            repo="owner/repo",
            token="ghp_custom_token",
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.token == "ghp_custom_token"

    def test_init_custom_options(self, mock_credentials, tmp_path):
        """Test initialization with custom include/exclude options."""
        custom_extensions = {".py", ".rs", ".go"}
        custom_excludes = ["vendor/", "test_data/"]

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

    def test_default_file_extensions(self, connector):
        """Test default file extensions include code, docs, and config."""
        extensions = connector.file_extensions
        # Check code extensions
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions
        # Check doc extensions
        assert ".md" in extensions
        assert ".rst" in extensions
        # Check config extensions
        assert ".json" in extensions
        assert ".yaml" in extensions

    def test_default_exclude_paths(self, connector):
        """Test default exclude paths."""
        excludes = connector.exclude_paths
        assert "node_modules/" in excludes
        assert "__pycache__/" in excludes
        assert ".git/" in excludes
        assert "vendor/" in excludes

    def test_source_type_property(self, connector):
        """Test source_type property returns CODE_ANALYSIS."""
        assert connector.source_type == SourceType.CODE_ANALYSIS

    def test_name_property(self, connector):
        """Test name property returns formatted name."""
        assert connector.name == "GitHub (test-owner/test-repo)"


# =============================================================================
# Test Class: Repository Operations
# =============================================================================


class TestRepositoryOperations:
    """Tests for repository operations (trees, commits, files)."""

    @pytest.mark.asyncio
    async def test_get_latest_commit_success(self, connector):
        """Test getting latest commit SHA."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"abc123def456\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            sha = await connector._get_latest_commit()
            assert sha == "abc123def456"

    @pytest.mark.asyncio
    async def test_get_latest_commit_failure(self, connector):
        """Test getting latest commit when gh CLI unavailable."""
        connector._gh_available = False
        sha = await connector._get_latest_commit()
        assert sha is None

    @pytest.mark.asyncio
    async def test_get_tree_success(self, connector, sample_tree):
        """Test getting repository file tree."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json.dumps(sample_tree).encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            tree = await connector._get_tree("abc123")
            assert len(tree) == len(sample_tree)

    @pytest.mark.asyncio
    async def test_get_tree_empty_response(self, connector):
        """Test getting tree with empty response."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            tree = await connector._get_tree("abc123")
            assert tree == []

    @pytest.mark.asyncio
    async def test_get_tree_invalid_json(self, connector):
        """Test getting tree with invalid JSON response."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"not valid json", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            tree = await connector._get_tree("abc123")
            assert tree == []

    @pytest.mark.asyncio
    async def test_get_commits_since_success(self, connector, sample_commits):
        """Test getting commits since a specific SHA."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json.dumps(sample_commits).encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            commits = await connector._get_commits_since(None, limit=10)
            assert len(commits) == 2
            assert commits[0].sha == "abc123def456789012345678901234567890abcd"
            assert commits[0].author == "Alice"

    @pytest.mark.asyncio
    async def test_get_commits_since_stops_at_cursor(self, connector, sample_commits):
        """Test commits collection stops at the cursor SHA."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json.dumps(sample_commits).encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            commits = await connector._get_commits_since(
                "def456ghi789012345678901234567890bcdef", limit=10
            )
            assert len(commits) == 1


# =============================================================================
# Test Class: Pull Request Operations
# =============================================================================


class TestPullRequestOperations:
    """Tests for pull request operations."""

    @pytest.mark.asyncio
    async def test_get_prs_success(self, connector, sample_prs):
        """Test getting pull requests."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json.dumps(sample_prs).encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            prs = await connector._get_prs(state="all", limit=10)
            assert len(prs) == 2
            assert prs[0]["number"] == 10
            assert prs[0]["title"] == "Fix authentication bug"

    @pytest.mark.asyncio
    async def test_get_prs_empty(self, connector):
        """Test getting PRs when none exist."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"[]", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            prs = await connector._get_prs(state="open", limit=10)
            assert prs == []

    @pytest.mark.asyncio
    async def test_sync_prs_creates_sync_items(self, connector, sample_prs):
        """Test syncing PRs creates proper SyncItems."""
        connector._gh_available = True
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)
        connector.include_issues = False

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

        with patch.object(connector, "_run_gh", side_effect=mock_run_gh):
            items = []
            async for item in connector.sync_items(state, batch_size=10):
                items.append(item)

        pr_items = [i for i in items if i.source_type == "pr"]
        assert len(pr_items) == 2

        merged_pr = next(i for i in pr_items if "Fix authentication bug" in i.title)
        assert merged_pr.metadata.get("merged") is True
        assert merged_pr.domain == "technical/pull-requests"


# =============================================================================
# Test Class: Issue Operations
# =============================================================================


class TestIssueOperations:
    """Tests for issue operations."""

    @pytest.mark.asyncio
    async def test_get_issues_success(self, connector, sample_issues):
        """Test getting issues."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json.dumps(sample_issues).encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            issues = await connector._get_issues(state="all", limit=10)
            assert len(issues) == 3
            assert issues[0]["number"] == 1
            assert issues[0]["title"] == "Bug in authentication"

    @pytest.mark.asyncio
    async def test_get_issues_with_state_filter(self, connector, sample_issues):
        """Test getting issues with state filter."""
        connector._gh_available = True
        open_issues = [i for i in sample_issues if i["state"] == "open"]

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json.dumps(open_issues).encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            issues = await connector._get_issues(state="open", limit=10)
            assert len(issues) == 2

    @pytest.mark.asyncio
    async def test_sync_issues_creates_sync_items(self, connector, sample_issues):
        """Test syncing issues creates proper SyncItems."""
        connector._gh_available = True
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)
        connector.include_prs = False

        async def mock_run_gh(args):
            if "commits/main" in str(args):
                return "abc123"
            elif "commits" in str(args) and "per_page" in str(args):
                return "[]"
            elif "git/trees" in str(args):
                return "[]"
            elif "issue" in args:
                return json.dumps(sample_issues)
            return None

        with patch.object(connector, "_run_gh", side_effect=mock_run_gh):
            items = []
            async for item in connector.sync_items(state, batch_size=10):
                items.append(item)

        issue_items = [i for i in items if i.source_type == "issue"]
        assert len(issue_items) == 3

        bug_issue = next(i for i in issue_items if "Bug in authentication" in i.title)
        assert bug_issue.domain == "technical/issues"
        assert "bug" in bug_issue.metadata.get("labels", [])


# =============================================================================
# Test Class: Webhook Handling
# =============================================================================


class TestWebhookHandling:
    """Tests for webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook_push_event(self, connector):
        """Test handling push webhook triggers sync."""
        payload = {
            "ref": "push:refs/heads/main",
            "commits": [{"id": "abc123"}, {"id": "def456"}],
        }

        with patch.object(connector, "sync", new_callable=AsyncMock) as mock_sync:
            result = await connector.handle_webhook(payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_issue_opened(self, connector):
        """Test handling issue opened webhook."""
        payload = {
            "action": "opened",
            "issue": {"number": 123, "title": "New issue"},
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_issue_closed(self, connector):
        """Test handling issue closed webhook."""
        payload = {
            "action": "closed",
            "issue": {"number": 123, "title": "Closed issue"},
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_issue_reopened(self, connector):
        """Test handling issue reopened webhook."""
        payload = {
            "action": "reopened",
            "issue": {"number": 123, "title": "Reopened issue"},
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_pr_opened(self, connector):
        """Test handling PR opened webhook."""
        payload = {
            "action": "opened",
            "pull_request": {"number": 456, "title": "New PR"},
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_pr_merged(self, connector):
        """Test handling PR merged webhook."""
        payload = {
            "action": "closed",
            "pull_request": {"number": 456, "title": "Merged PR", "merged": True},
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_unknown_event(self, connector):
        """Test handling unknown webhook event returns False."""
        payload = {"action": "unknown_action_type"}
        result = await connector.handle_webhook(payload)
        assert result is False

    def test_get_webhook_secret_exists(self, connector):
        """Test get_webhook_secret method exists."""
        assert hasattr(connector, "get_webhook_secret")


# =============================================================================
# Test Class: File/Content Operations
# =============================================================================


class TestFileContentOperations:
    """Tests for file and content operations."""

    @pytest.mark.asyncio
    async def test_get_file_content_success(self, connector):
        """Test getting file content with base64 decoding."""
        connector._gh_available = True
        content = "print('Hello, World!')"
        encoded = base64.b64encode(content.encode()).decode()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(encoded.encode() + b"\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await connector._get_file_content("test.py")
            assert result == content

    @pytest.mark.asyncio
    async def test_get_file_content_invalid_base64(self, connector):
        """Test getting file content with invalid base64."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"not-valid-base64!!!", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await connector._get_file_content("test.py")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_file_content_failure(self, connector):
        """Test getting file content when gh fails."""
        connector._gh_available = False
        result = await connector._get_file_content("test.py")
        assert result is None

    def test_should_index_important_files(self, connector):
        """Test important files are always indexed."""
        assert connector._should_index_file("README.md", 100) is True
        assert connector._should_index_file("package.json", 500) is True
        assert connector._should_index_file("pyproject.toml", 200) is True
        assert connector._should_index_file("Dockerfile", 300) is True
        assert connector._should_index_file("Makefile", 400) is True

    def test_should_index_code_files(self, connector):
        """Test code files with valid extensions are indexed."""
        assert connector._should_index_file("src/main.py", 1000) is True
        assert connector._should_index_file("lib/utils.ts", 2000) is True
        assert connector._should_index_file("app.jsx", 1500) is True
        assert connector._should_index_file("server.go", 3000) is True

    def test_should_not_index_excluded_paths(self, connector):
        """Test excluded paths are not indexed."""
        assert connector._should_index_file("node_modules/dep/index.js", 500) is False
        assert connector._should_index_file("vendor/lib/code.py", 500) is False
        assert connector._should_index_file("src/__pycache__/main.pyc", 500) is False
        assert connector._should_index_file(".git/config", 500) is False

    def test_should_not_index_large_files(self, connector):
        """Test large files exceeding MAX_FILE_SIZE are not indexed."""
        large_size = MAX_FILE_SIZE + 1
        assert connector._should_index_file("src/main.py", large_size) is False

    def test_should_not_index_unknown_extensions(self, connector):
        """Test files with unknown extensions are not indexed."""
        assert connector._should_index_file("data.bin", 500) is False
        assert connector._should_index_file("image.png", 500) is False
        assert connector._should_index_file("archive.zip", 500) is False


# =============================================================================
# Test Class: Code Element Extraction
# =============================================================================


class TestCodeElementExtraction:
    """Tests for code element extraction from files."""

    def test_extract_python_classes(self, connector):
        """Test extracting Python class definitions."""
        content = """
class SimpleClass:
    pass

class InheritedClass(BaseClass):
    def method(self):
        pass

class MultiInherit(Base1, Base2):
    pass
"""
        elements = connector._extract_python_elements(content, "test.py")
        class_elements = [e for e in elements if e["type"] == "class"]
        assert len(class_elements) == 3
        assert any(e["name"] == "SimpleClass" for e in class_elements)
        assert any(e["name"] == "InheritedClass" for e in class_elements)
        assert any(e["name"] == "MultiInherit" for e in class_elements)

    def test_extract_python_functions(self, connector):
        """Test extracting Python function definitions."""
        content = """
def regular_function(arg1, arg2):
    return arg1 + arg2

async def async_function(data: str) -> dict:
    return {"data": data}

def typed_function(x: int, y: int) -> int:
    return x * y
"""
        elements = connector._extract_python_elements(content, "test.py")
        func_elements = [e for e in elements if e["type"] == "function"]
        assert len(func_elements) == 3
        assert any(e["name"] == "regular_function" for e in func_elements)
        assert any(e["name"] == "async_function" for e in func_elements)
        assert any(e["name"] == "typed_function" for e in func_elements)

    def test_extract_js_classes(self, connector):
        """Test extracting JavaScript class definitions."""
        content = """
class MyComponent {
    constructor() {}
}

export class ExportedClass extends BaseClass {
    render() {}
}
"""
        elements = connector._extract_js_elements(content, "test.js")
        class_elements = [e for e in elements if e["type"] == "class"]
        assert len(class_elements) == 2
        assert any(e["name"] == "MyComponent" for e in class_elements)
        assert any(e["name"] == "ExportedClass" for e in class_elements)

    def test_extract_js_functions(self, connector):
        """Test extracting JavaScript function definitions."""
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
        elements = connector._extract_js_elements(content, "test.js")
        func_elements = [e for e in elements if e["type"] == "function"]
        assert len(func_elements) == 4
        assert any(e["name"] == "regularFunction" for e in func_elements)
        assert any(e["name"] == "asyncFunction" for e in func_elements)
        assert any(e["name"] == "arrowFunc" for e in func_elements)
        assert any(e["name"] == "exportedArrow" for e in func_elements)


# =============================================================================
# Test Class: Dependency Extraction
# =============================================================================


class TestDependencyExtraction:
    """Tests for dependency extraction from imports."""

    def test_extract_python_imports(self, connector):
        """Test extracting Python import statements."""
        content = """
import os
import json
from typing import List, Dict
from collections.abc import Mapping
from aragora.connectors.base import BaseConnector
"""
        deps = connector._extract_dependencies(content, "test.py")
        assert "os" in deps
        assert "json" in deps
        assert "typing" in deps
        assert "collections" in deps
        assert "aragora" in deps

    def test_extract_js_imports(self, connector):
        """Test extracting JavaScript ES6 imports."""
        content = """
import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from '@mui/material';
import utils from './utils';
"""
        deps = connector._extract_dependencies(content, "test.js")
        assert "react" in deps
        assert "axios" in deps
        assert "@mui" in deps

    def test_extract_js_require(self, connector):
        """Test extracting JavaScript require() calls."""
        content = """
const fs = require('fs');
const path = require('path');
const localModule = require('./local');
"""
        deps = connector._extract_dependencies(content, "test.js")
        assert "fs" in deps
        assert "path" in deps

    def test_no_duplicate_dependencies(self, connector):
        """Test dependencies are deduplicated."""
        content = """
import React from 'react';
import { useState } from 'react';
import { useEffect } from 'react';
"""
        deps = connector._extract_dependencies(content, "test.js")
        assert deps.count("react") == 1


# =============================================================================
# Test Class: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_check_gh_cli_available(self, connector):
        """Test gh CLI availability check when available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            connector._gh_available = None
            result = connector._check_gh_cli()
            assert result is True
            assert connector._gh_available is True

    def test_check_gh_cli_unavailable(self, connector):
        """Test gh CLI availability check when unavailable."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            connector._gh_available = None
            result = connector._check_gh_cli()
            assert result is False
            assert connector._gh_available is False

    def test_check_gh_cli_cached(self, connector):
        """Test gh CLI availability is cached."""
        connector._gh_available = True
        with patch("subprocess.run") as mock_run:
            result = connector._check_gh_cli()
            assert result is True
            mock_run.assert_not_called()

    def test_check_gh_cli_oserror(self, connector):
        """Test gh CLI check handles OSError."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Command not found")
            connector._gh_available = None
            result = connector._check_gh_cli()
            assert result is False

    @pytest.mark.asyncio
    async def test_run_gh_timeout(self, connector):
        """Test gh command handles timeout."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=TimeoutError()):
                result = await connector._run_gh(["api", "test"])
                assert result is None

    @pytest.mark.asyncio
    async def test_run_gh_error_output(self, connector):
        """Test gh command handles error output."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"API rate limit exceeded"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await connector._run_gh(["api", "test"])
            assert result is None

    @pytest.mark.asyncio
    async def test_sync_items_no_gh_cli(self, connector):
        """Test sync when gh CLI unavailable returns empty."""
        connector._gh_available = False
        state = SyncState(connector_id="test", status=SyncStatus.IDLE)

        items = []
        async for item in connector.sync_items(state):
            items.append(item)

        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_perform_health_check_success(self, connector):
        """Test health check succeeds when API accessible."""
        connector._gh_available = True

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b'{"name": "test-repo"}', b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await connector._perform_health_check(timeout=10.0)
            assert result is True

    @pytest.mark.asyncio
    async def test_perform_health_check_failure(self, connector):
        """Test health check fails when API inaccessible."""
        connector._gh_available = False
        result = await connector._perform_health_check(timeout=10.0)
        assert result is False

    def test_is_available_property(self, connector):
        """Test is_available property."""
        with patch.object(connector, "_check_gh_cli", return_value=True):
            assert connector.is_available is True

    def test_is_configured_with_token(self, connector):
        """Test is_configured when token is set."""
        connector.token = "ghp_test_token"
        assert connector.is_configured is True

    def test_is_configured_without_token_but_cli(self, connector):
        """Test is_configured when no token but CLI available."""
        connector.token = None
        with patch.object(connector, "_check_gh_cli", return_value=True):
            assert connector.is_configured is True


# =============================================================================
# Test Class: Search and Fetch Delegation
# =============================================================================


class TestSearchAndFetch:
    """Tests for search and fetch delegation to GitHubConnector."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_connector(self, connector):
        """Test search method delegates to GitHubConnector."""
        with patch("aragora.connectors.github.GitHubConnector") as MockConnector:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(return_value=[{"id": "result1"}])
            MockConnector.return_value = mock_instance

            results = await connector.search("test query", limit=5)
            assert len(results) == 1
            mock_instance.search.assert_called_once_with("test query", limit=5, search_type="code")

    @pytest.mark.asyncio
    async def test_fetch_delegates_to_connector(self, connector):
        """Test fetch method delegates to GitHubConnector."""
        with patch("aragora.connectors.github.GitHubConnector") as MockConnector:
            mock_instance = MagicMock()
            mock_instance.fetch = AsyncMock(return_value={"content": "test content"})
            MockConnector.return_value = mock_instance

            result = await connector.fetch("evidence-123")
            assert result == {"content": "test content"}
            mock_instance.fetch.assert_called_once_with("evidence-123")


# =============================================================================
# Test Class: Dataclass Tests
# =============================================================================


class TestDataclasses:
    """Tests for GitHubFile and GitHubCommit dataclasses."""

    def test_github_file_dataclass(self):
        """Test GitHubFile dataclass creation."""
        file = GitHubFile(
            path="src/main.py",
            sha="abc123",
            content="print('hello')",
            size=100,
            url="https://github.com/owner/repo/blob/main/src/main.py",
            last_modified=datetime.now(timezone.utc),
        )
        assert file.path == "src/main.py"
        assert file.sha == "abc123"
        assert file.size == 100

    def test_github_commit_dataclass(self):
        """Test GitHubCommit dataclass creation."""
        commit = GitHubCommit(
            sha="abc123def456",
            message="Initial commit",
            author="Alice",
            date=datetime.now(timezone.utc),
            files_changed=["README.md", "src/main.py"],
        )
        assert commit.sha == "abc123def456"
        assert commit.message == "Initial commit"
        assert commit.author == "Alice"
        assert len(commit.files_changed) == 2

    def test_github_commit_default_files_changed(self):
        """Test GitHubCommit default files_changed is empty list."""
        commit = GitHubCommit(
            sha="abc123",
            message="Test",
            author="Test",
            date=datetime.now(timezone.utc),
        )
        assert commit.files_changed == []
