"""Tests for CodebaseAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.codebase import (
    AnalysisResult,
    CodebaseAPI,
    CodeFile,
    DependencyInfo,
    Repository,
    SecurityFinding,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> CodebaseAPI:
    return CodebaseAPI(mock_client)


SAMPLE_REPO = {
    "id": "repo-001",
    "name": "aragora",
    "url": "https://github.com/org/aragora",
    "provider": "github",
    "branch": "main",
    "status": "active",
    "last_synced_at": "2026-01-20T12:00:00Z",
    "file_count": 350,
    "language_stats": {"python": 80.5, "typescript": 15.0, "shell": 4.5},
}

SAMPLE_FILE = {
    "path": "aragora/debate/orchestrator.py",
    "language": "python",
    "size_bytes": 12400,
    "lines": 420,
    "last_modified": "2026-01-18T09:30:00Z",
    "complexity_score": 7.2,
}

SAMPLE_DEPENDENCY = {
    "name": "fastapi",
    "version": "0.104.1",
    "type": "direct",
    "ecosystem": "pip",
    "latest_version": "0.105.0",
    "has_vulnerabilities": False,
    "vulnerability_count": 0,
}

SAMPLE_FINDING = {
    "id": "finding-001",
    "type": "sql_injection",
    "severity": "high",
    "title": "Potential SQL injection in query builder",
    "description": "User input passed directly to SQL query without sanitization.",
    "file_path": "aragora/storage/queries.py",
    "line_number": 42,
    "cwe_id": "CWE-89",
    "recommendation": "Use parameterized queries.",
    "status": "open",
}

SAMPLE_ANALYSIS = {
    "repository_id": "repo-001",
    "analysis_type": "security_scan",
    "status": "completed",
    "started_at": "2026-01-20T12:00:00Z",
    "completed_at": "2026-01-20T12:05:00Z",
    "findings_count": 3,
    "summary": {"critical": 0, "high": 1, "medium": 2, "low": 0},
}


# =========================================================================
# Repository Management
# =========================================================================


class TestListRepositories:
    def test_list_default(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"repositories": [SAMPLE_REPO], "total": 1}
        repos, total = api.list_repositories()
        assert len(repos) == 1
        assert total == 1
        assert isinstance(repos[0], Repository)
        assert repos[0].id == "repo-001"
        assert repos[0].name == "aragora"

    def test_list_with_status_filter(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"repositories": [], "total": 0}
        api.list_repositories(status="syncing")
        params = mock_client._get.call_args[1]["params"]
        assert params["status"] == "syncing"

    def test_list_with_pagination(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"repositories": [], "total": 0}
        api.list_repositories(limit=10, offset=20)
        params = mock_client._get.call_args[1]["params"]
        assert params["limit"] == 10
        assert params["offset"] == 20

    def test_list_no_status_omits_param(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"repositories": [], "total": 0}
        api.list_repositories()
        params = mock_client._get.call_args[1]["params"]
        assert "status" not in params

    def test_list_total_fallback(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"repositories": [SAMPLE_REPO]}
        repos, total = api.list_repositories()
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"repositories": [SAMPLE_REPO], "total": 1})
        repos, total = await api.list_repositories_async()
        assert len(repos) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_async_with_status(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"repositories": [], "total": 0})
        await api.list_repositories_async(status="error")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["status"] == "error"


class TestConnectRepository:
    def test_connect_minimal(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"repository": SAMPLE_REPO}
        repo = api.connect_repository(
            url="https://github.com/org/aragora",
            provider="github",
        )
        assert isinstance(repo, Repository)
        assert repo.id == "repo-001"
        mock_client._post.assert_called_once()
        body = mock_client._post.call_args[0][1]
        assert body["url"] == "https://github.com/org/aragora"
        assert body["provider"] == "github"
        assert body["branch"] == "main"
        assert "access_token" not in body

    def test_connect_with_branch_and_token(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"repository": SAMPLE_REPO}
        api.connect_repository(
            url="https://github.com/org/repo",
            provider="gitlab",
            branch="develop",
            access_token="tok-secret",
        )
        body = mock_client._post.call_args[0][1]
        assert body["branch"] == "develop"
        assert body["access_token"] == "tok-secret"

    def test_connect_response_without_wrapper(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_REPO
        repo = api.connect_repository(url="https://example.com/repo", provider="bitbucket")
        assert repo.provider == "github"

    @pytest.mark.asyncio
    async def test_connect_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"repository": SAMPLE_REPO})
        repo = await api.connect_repository_async(
            url="https://github.com/org/aragora",
            provider="github",
        )
        assert repo.id == "repo-001"

    @pytest.mark.asyncio
    async def test_connect_async_with_token(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"repository": SAMPLE_REPO})
        await api.connect_repository_async(
            url="https://github.com/org/repo",
            provider="github",
            access_token="secret",
        )
        body = mock_client._post_async.call_args[0][1]
        assert body["access_token"] == "secret"


class TestGetRepository:
    def test_get(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"repository": SAMPLE_REPO}
        repo = api.get_repository("repo-001")
        assert repo.id == "repo-001"
        assert repo.url == "https://github.com/org/aragora"
        mock_client._get.assert_called_once_with("/api/v1/codebase/repositories/repo-001")

    def test_get_unwrapped_response(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_REPO
        repo = api.get_repository("repo-001")
        assert repo.name == "aragora"

    @pytest.mark.asyncio
    async def test_get_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"repository": SAMPLE_REPO})
        repo = await api.get_repository_async("repo-001")
        assert repo.provider == "github"


class TestSyncRepository:
    def test_sync(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        syncing_repo = {**SAMPLE_REPO, "status": "syncing"}
        mock_client._post.return_value = {"repository": syncing_repo}
        repo = api.sync_repository("repo-001")
        assert repo.status == "syncing"
        mock_client._post.assert_called_once_with("/api/v1/codebase/repositories/repo-001/sync", {})

    @pytest.mark.asyncio
    async def test_sync_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(
            return_value={"repository": {**SAMPLE_REPO, "status": "syncing"}}
        )
        repo = await api.sync_repository_async("repo-001")
        assert repo.status == "syncing"


class TestDisconnectRepository:
    def test_disconnect(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = {}
        result = api.disconnect_repository("repo-001")
        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/codebase/repositories/repo-001")

    @pytest.mark.asyncio
    async def test_disconnect_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._delete_async = AsyncMock(return_value={})
        result = await api.disconnect_repository_async("repo-001")
        assert result is True


# =========================================================================
# File Operations
# =========================================================================


class TestListFiles:
    def test_list_default(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"files": [SAMPLE_FILE], "total": 1}
        files, total = api.list_files("repo-001")
        assert len(files) == 1
        assert total == 1
        assert isinstance(files[0], CodeFile)
        assert files[0].path == "aragora/debate/orchestrator.py"

    def test_list_with_filters(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"files": [], "total": 0}
        api.list_files("repo-001", path="aragora/", language="python", limit=25, offset=5)
        params = mock_client._get.call_args[1]["params"]
        assert params["path"] == "aragora/"
        assert params["language"] == "python"
        assert params["limit"] == 25
        assert params["offset"] == 5

    def test_list_omits_empty_path_and_language(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"files": [], "total": 0}
        api.list_files("repo-001")
        params = mock_client._get.call_args[1]["params"]
        assert "path" not in params
        assert "language" not in params

    def test_list_total_fallback(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"files": [SAMPLE_FILE, SAMPLE_FILE]}
        files, total = api.list_files("repo-001")
        assert total == 2

    @pytest.mark.asyncio
    async def test_list_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"files": [SAMPLE_FILE], "total": 1})
        files, total = await api.list_files_async("repo-001")
        assert len(files) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_async_with_language(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"files": [], "total": 0})
        await api.list_files_async("repo-001", language="typescript")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["language"] == "typescript"


class TestGetFileContent:
    def test_get_content(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"content": "import os\nprint('hello')"}
        content = api.get_file_content("repo-001", "main.py")
        assert content == "import os\nprint('hello')"
        mock_client._get.assert_called_once_with(
            "/api/v1/codebase/repositories/repo-001/files/content",
            params={"path": "main.py"},
        )

    def test_get_content_empty(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        content = api.get_file_content("repo-001", "empty.py")
        assert content == ""

    @pytest.mark.asyncio
    async def test_get_content_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"content": "# async content"})
        content = await api.get_file_content_async("repo-001", "main.py")
        assert content == "# async content"


# =========================================================================
# Dependencies
# =========================================================================


class TestListDependencies:
    def test_list_default(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dependencies": [SAMPLE_DEPENDENCY]}
        deps = api.list_dependencies("repo-001")
        assert len(deps) == 1
        assert isinstance(deps[0], DependencyInfo)
        assert deps[0].name == "fastapi"
        assert deps[0].version == "0.104.1"

    def test_list_with_ecosystem(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dependencies": []}
        api.list_dependencies("repo-001", ecosystem="npm")
        params = mock_client._get.call_args[1]["params"]
        assert params["ecosystem"] == "npm"

    def test_list_vulnerable_only(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        vuln_dep = {**SAMPLE_DEPENDENCY, "has_vulnerabilities": True, "vulnerability_count": 2}
        mock_client._get.return_value = {"dependencies": [vuln_dep]}
        deps = api.list_dependencies("repo-001", vulnerable_only=True)
        params = mock_client._get.call_args[1]["params"]
        assert params["vulnerable_only"] is True
        assert deps[0].has_vulnerabilities is True
        assert deps[0].vulnerability_count == 2

    def test_list_omits_empty_filters(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dependencies": []}
        api.list_dependencies("repo-001")
        params = mock_client._get.call_args[1]["params"]
        assert "ecosystem" not in params
        assert "vulnerable_only" not in params

    @pytest.mark.asyncio
    async def test_list_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"dependencies": [SAMPLE_DEPENDENCY]})
        deps = await api.list_dependencies_async("repo-001")
        assert len(deps) == 1
        assert deps[0].ecosystem == "pip"

    @pytest.mark.asyncio
    async def test_list_async_with_filters(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"dependencies": []})
        await api.list_dependencies_async("repo-001", ecosystem="maven", vulnerable_only=True)
        params = mock_client._get_async.call_args[1]["params"]
        assert params["ecosystem"] == "maven"
        assert params["vulnerable_only"] is True


# =========================================================================
# Security Analysis
# =========================================================================


class TestRunSecurityScan:
    def test_run_scan(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_ANALYSIS
        result = api.run_security_scan("repo-001")
        assert isinstance(result, AnalysisResult)
        assert result.repository_id == "repo-001"
        assert result.analysis_type == "security_scan"
        assert result.status == "completed"
        assert result.findings_count == 3
        mock_client._post.assert_called_once_with(
            "/api/v1/codebase/repositories/repo-001/security/scan", {}
        )

    @pytest.mark.asyncio
    async def test_run_scan_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_ANALYSIS)
        result = await api.run_security_scan_async("repo-001")
        assert result.findings_count == 3


class TestListSecurityFindings:
    def test_list_default(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"findings": [SAMPLE_FINDING], "total": 1}
        findings, total = api.list_security_findings("repo-001")
        assert len(findings) == 1
        assert total == 1
        assert isinstance(findings[0], SecurityFinding)
        assert findings[0].id == "finding-001"
        assert findings[0].severity == "high"

    def test_list_with_filters(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"findings": [], "total": 0}
        api.list_security_findings(
            "repo-001", severity="critical", status="open", limit=25, offset=10
        )
        params = mock_client._get.call_args[1]["params"]
        assert params["severity"] == "critical"
        assert params["status"] == "open"
        assert params["limit"] == 25
        assert params["offset"] == 10

    def test_list_omits_empty_filters(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"findings": [], "total": 0}
        api.list_security_findings("repo-001")
        params = mock_client._get.call_args[1]["params"]
        assert "severity" not in params
        assert "status" not in params

    def test_list_total_fallback(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"findings": [SAMPLE_FINDING]}
        findings, total = api.list_security_findings("repo-001")
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"findings": [SAMPLE_FINDING], "total": 1})
        findings, total = await api.list_security_findings_async("repo-001")
        assert len(findings) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_async_with_severity(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"findings": [], "total": 0})
        await api.list_security_findings_async("repo-001", severity="low")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["severity"] == "low"


class TestUpdateFindingStatus:
    def test_update(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        updated = {**SAMPLE_FINDING, "status": "fixed"}
        mock_client._patch.return_value = {"finding": updated}
        finding = api.update_finding_status("repo-001", "finding-001", "fixed")
        assert isinstance(finding, SecurityFinding)
        assert finding.status == "fixed"
        mock_client._patch.assert_called_once_with(
            "/api/v1/codebase/repositories/repo-001/security/findings/finding-001",
            {"status": "fixed"},
        )

    def test_update_ignored(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        updated = {**SAMPLE_FINDING, "status": "ignored"}
        mock_client._patch.return_value = {"finding": updated}
        finding = api.update_finding_status("repo-001", "finding-001", "ignored")
        assert finding.status == "ignored"

    def test_update_unwrapped_response(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        updated = {**SAMPLE_FINDING, "status": "fixed"}
        mock_client._patch.return_value = updated
        finding = api.update_finding_status("repo-001", "finding-001", "fixed")
        assert finding.status == "fixed"

    @pytest.mark.asyncio
    async def test_update_async(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        updated = {**SAMPLE_FINDING, "status": "fixed"}
        mock_client._patch_async = AsyncMock(return_value={"finding": updated})
        finding = await api.update_finding_status_async("repo-001", "finding-001", "fixed")
        assert finding.status == "fixed"


# =========================================================================
# Parser / Helper Methods
# =========================================================================


class TestParseRepository:
    def test_parse_full(self, api: CodebaseAPI) -> None:
        repo = api._parse_repository(SAMPLE_REPO)
        assert repo.id == "repo-001"
        assert repo.name == "aragora"
        assert repo.url == "https://github.com/org/aragora"
        assert repo.provider == "github"
        assert repo.branch == "main"
        assert repo.status == "active"
        assert repo.last_synced_at is not None
        assert repo.last_synced_at.year == 2026
        assert repo.file_count == 350
        assert repo.language_stats == {"python": 80.5, "typescript": 15.0, "shell": 4.5}

    def test_parse_missing_fields(self, api: CodebaseAPI) -> None:
        repo = api._parse_repository({})
        assert repo.id == ""
        assert repo.name == ""
        assert repo.branch == "main"
        assert repo.status == "active"
        assert repo.last_synced_at is None
        assert repo.file_count == 0
        assert repo.language_stats == {}

    def test_parse_invalid_datetime(self, api: CodebaseAPI) -> None:
        data = {**SAMPLE_REPO, "last_synced_at": "not-a-date"}
        repo = api._parse_repository(data)
        assert repo.last_synced_at is None

    def test_parse_null_datetime(self, api: CodebaseAPI) -> None:
        data = {**SAMPLE_REPO, "last_synced_at": None}
        repo = api._parse_repository(data)
        assert repo.last_synced_at is None


class TestParseFile:
    def test_parse_full(self, api: CodebaseAPI) -> None:
        f = api._parse_file(SAMPLE_FILE)
        assert f.path == "aragora/debate/orchestrator.py"
        assert f.language == "python"
        assert f.size_bytes == 12400
        assert f.lines == 420
        assert f.last_modified is not None
        assert f.last_modified.year == 2026
        assert f.complexity_score == 7.2

    def test_parse_missing_fields(self, api: CodebaseAPI) -> None:
        f = api._parse_file({})
        assert f.path == ""
        assert f.language == ""
        assert f.size_bytes == 0
        assert f.lines == 0
        assert f.last_modified is None
        assert f.complexity_score == 0.0

    def test_parse_invalid_datetime(self, api: CodebaseAPI) -> None:
        data = {**SAMPLE_FILE, "last_modified": "bad-date"}
        f = api._parse_file(data)
        assert f.last_modified is None


class TestParseDependency:
    def test_parse_full(self, api: CodebaseAPI) -> None:
        dep = api._parse_dependency(SAMPLE_DEPENDENCY)
        assert dep.name == "fastapi"
        assert dep.version == "0.104.1"
        assert dep.type == "direct"
        assert dep.ecosystem == "pip"
        assert dep.latest_version == "0.105.0"
        assert dep.has_vulnerabilities is False
        assert dep.vulnerability_count == 0

    def test_parse_missing_fields(self, api: CodebaseAPI) -> None:
        dep = api._parse_dependency({})
        assert dep.name == ""
        assert dep.version == ""
        assert dep.type == "direct"
        assert dep.ecosystem == ""
        assert dep.latest_version is None
        assert dep.has_vulnerabilities is False
        assert dep.vulnerability_count == 0


class TestParseFinding:
    def test_parse_full(self, api: CodebaseAPI) -> None:
        finding = api._parse_finding(SAMPLE_FINDING)
        assert finding.id == "finding-001"
        assert finding.type == "sql_injection"
        assert finding.severity == "high"
        assert finding.title == "Potential SQL injection in query builder"
        assert finding.file_path == "aragora/storage/queries.py"
        assert finding.line_number == 42
        assert finding.cwe_id == "CWE-89"
        assert finding.recommendation == "Use parameterized queries."
        assert finding.status == "open"

    def test_parse_missing_fields(self, api: CodebaseAPI) -> None:
        finding = api._parse_finding({})
        assert finding.id == ""
        assert finding.type == ""
        assert finding.severity == "medium"
        assert finding.title == ""
        assert finding.description == ""
        assert finding.file_path is None
        assert finding.line_number is None
        assert finding.cwe_id is None
        assert finding.recommendation is None
        assert finding.status == "open"


class TestParseAnalysisResult:
    def test_parse_full(self, api: CodebaseAPI) -> None:
        result = api._parse_analysis_result(SAMPLE_ANALYSIS)
        assert result.repository_id == "repo-001"
        assert result.analysis_type == "security_scan"
        assert result.status == "completed"
        assert result.started_at is not None
        assert result.started_at.year == 2026
        assert result.completed_at is not None
        assert result.findings_count == 3
        assert result.summary == {"critical": 0, "high": 1, "medium": 2, "low": 0}

    def test_parse_missing_fields(self, api: CodebaseAPI) -> None:
        result = api._parse_analysis_result({})
        assert result.repository_id == ""
        assert result.analysis_type == ""
        assert result.status == "pending"
        assert result.started_at is None
        assert result.completed_at is None
        assert result.findings_count == 0
        assert result.summary == {}

    def test_parse_invalid_started_at(self, api: CodebaseAPI) -> None:
        data = {**SAMPLE_ANALYSIS, "started_at": "not-a-date"}
        result = api._parse_analysis_result(data)
        assert result.started_at is None

    def test_parse_invalid_completed_at(self, api: CodebaseAPI) -> None:
        data = {**SAMPLE_ANALYSIS, "completed_at": "not-a-date"}
        result = api._parse_analysis_result(data)
        assert result.completed_at is None


# =========================================================================
# Dataclass Construction
# =========================================================================


class TestDataclasses:
    def test_repository_defaults(self) -> None:
        repo = Repository(id="r1", name="test", url="https://example.com", provider="github")
        assert repo.branch == "main"
        assert repo.status == "active"
        assert repo.last_synced_at is None
        assert repo.file_count == 0
        assert repo.language_stats == {}

    def test_code_file_defaults(self) -> None:
        f = CodeFile(path="main.py", language="python", size_bytes=100, lines=10)
        assert f.last_modified is None
        assert f.complexity_score == 0.0

    def test_dependency_info_defaults(self) -> None:
        dep = DependencyInfo(name="requests", version="2.31.0", type="direct", ecosystem="pip")
        assert dep.latest_version is None
        assert dep.has_vulnerabilities is False
        assert dep.vulnerability_count == 0

    def test_security_finding_defaults(self) -> None:
        finding = SecurityFinding(
            id="f1", type="xss", severity="medium", title="XSS", description="Cross-site scripting"
        )
        assert finding.file_path is None
        assert finding.line_number is None
        assert finding.cwe_id is None
        assert finding.recommendation is None
        assert finding.status == "open"

    def test_analysis_result_defaults(self) -> None:
        result = AnalysisResult(repository_id="r1", analysis_type="scan", status="pending")
        assert result.started_at is None
        assert result.completed_at is None
        assert result.findings_count == 0
        assert result.summary == {}

    def test_repository_language_stats_independence(self) -> None:
        r1 = Repository(id="a", name="a", url="u", provider="p")
        r2 = Repository(id="b", name="b", url="u", provider="p")
        r1.language_stats["python"] = 100.0
        assert "python" not in r2.language_stats

    def test_analysis_result_summary_independence(self) -> None:
        a1 = AnalysisResult(repository_id="r1", analysis_type="scan", status="done")
        a2 = AnalysisResult(repository_id="r2", analysis_type="scan", status="done")
        a1.summary["key"] = "val"
        assert "key" not in a2.summary


# =========================================================================
# URL / Endpoint Verification
# =========================================================================


class TestEndpoints:
    def test_list_repositories_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"repositories": [], "total": 0}
        api.list_repositories()
        assert mock_client._get.call_args[0][0] == "/api/v1/codebase/repositories"

    def test_connect_repository_endpoint(
        self, api: CodebaseAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_REPO
        api.connect_repository(url="u", provider="github")
        assert mock_client._post.call_args[0][0] == "/api/v1/codebase/repositories"

    def test_get_repository_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_REPO
        api.get_repository("repo-xyz")
        mock_client._get.assert_called_once_with("/api/v1/codebase/repositories/repo-xyz")

    def test_sync_repository_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_REPO
        api.sync_repository("repo-xyz")
        mock_client._post.assert_called_once_with("/api/v1/codebase/repositories/repo-xyz/sync", {})

    def test_disconnect_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = {}
        api.disconnect_repository("repo-xyz")
        mock_client._delete.assert_called_once_with("/api/v1/codebase/repositories/repo-xyz")

    def test_list_files_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"files": [], "total": 0}
        api.list_files("repo-xyz")
        assert mock_client._get.call_args[0][0] == "/api/v1/codebase/repositories/repo-xyz/files"

    def test_get_file_content_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"content": ""}
        api.get_file_content("repo-xyz", "src/main.py")
        mock_client._get.assert_called_once_with(
            "/api/v1/codebase/repositories/repo-xyz/files/content",
            params={"path": "src/main.py"},
        )

    def test_list_dependencies_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"dependencies": []}
        api.list_dependencies("repo-xyz")
        assert (
            mock_client._get.call_args[0][0]
            == "/api/v1/codebase/repositories/repo-xyz/dependencies"
        )

    def test_security_scan_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_ANALYSIS
        api.run_security_scan("repo-xyz")
        mock_client._post.assert_called_once_with(
            "/api/v1/codebase/repositories/repo-xyz/security/scan", {}
        )

    def test_security_findings_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"findings": [], "total": 0}
        api.list_security_findings("repo-xyz")
        assert (
            mock_client._get.call_args[0][0]
            == "/api/v1/codebase/repositories/repo-xyz/security/findings"
        )

    def test_update_finding_endpoint(self, api: CodebaseAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = SAMPLE_FINDING
        api.update_finding_status("repo-xyz", "f-001", "fixed")
        mock_client._patch.assert_called_once_with(
            "/api/v1/codebase/repositories/repo-xyz/security/findings/f-001",
            {"status": "fixed"},
        )
