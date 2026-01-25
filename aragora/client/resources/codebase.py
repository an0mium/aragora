"""
Codebase API resource for the Aragora client.

Provides methods for codebase analysis integration:
- Repository connections
- Code analysis
- Dependency tracking
- Security scanning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class Repository:
    """A connected code repository."""

    id: str
    name: str
    url: str
    provider: str  # github, gitlab, bitbucket
    branch: str = "main"
    status: str = "active"  # active, syncing, error
    last_synced_at: Optional[datetime] = None
    file_count: int = 0
    language_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class CodeFile:
    """A file in the codebase."""

    path: str
    language: str
    size_bytes: int
    lines: int
    last_modified: Optional[datetime] = None
    complexity_score: float = 0.0


@dataclass
class DependencyInfo:
    """Information about a dependency."""

    name: str
    version: str
    type: str  # direct, transitive
    ecosystem: str  # npm, pip, maven, etc.
    latest_version: Optional[str] = None
    has_vulnerabilities: bool = False
    vulnerability_count: int = 0


@dataclass
class SecurityFinding:
    """A security finding in the codebase."""

    id: str
    type: str
    severity: str  # critical, high, medium, low
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    recommendation: Optional[str] = None
    status: str = "open"  # open, fixed, ignored


@dataclass
class AnalysisResult:
    """Result of a codebase analysis."""

    repository_id: str
    analysis_type: str
    status: str  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    findings_count: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)


class CodebaseAPI:
    """API interface for codebase analysis."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Repository Management
    # =========================================================================

    def list_repositories(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Repository], int]:
        """
        List connected repositories.

        Args:
            status: Filter by status.
            limit: Maximum number of repositories.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of Repository objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self._client._get("/api/v1/codebase/repositories", params=params)
        repos = [self._parse_repository(r) for r in response.get("repositories", [])]
        return repos, response.get("total", len(repos))

    async def list_repositories_async(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Repository], int]:
        """Async version of list_repositories()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = await self._client._get_async("/api/v1/codebase/repositories", params=params)
        repos = [self._parse_repository(r) for r in response.get("repositories", [])]
        return repos, response.get("total", len(repos))

    def connect_repository(
        self,
        url: str,
        provider: str,
        branch: str = "main",
        access_token: Optional[str] = None,
    ) -> Repository:
        """
        Connect a new repository.

        Args:
            url: Repository URL.
            provider: Provider type (github, gitlab, bitbucket).
            branch: Default branch to analyze.
            access_token: Access token for private repositories.

        Returns:
            Connected Repository object.
        """
        body: Dict[str, Any] = {
            "url": url,
            "provider": provider,
            "branch": branch,
        }
        if access_token:
            body["access_token"] = access_token

        response = self._client._post("/api/v1/codebase/repositories", body)
        return self._parse_repository(response.get("repository", response))

    async def connect_repository_async(
        self,
        url: str,
        provider: str,
        branch: str = "main",
        access_token: Optional[str] = None,
    ) -> Repository:
        """Async version of connect_repository()."""
        body: Dict[str, Any] = {
            "url": url,
            "provider": provider,
            "branch": branch,
        }
        if access_token:
            body["access_token"] = access_token

        response = await self._client._post_async("/api/v1/codebase/repositories", body)
        return self._parse_repository(response.get("repository", response))

    def get_repository(self, repository_id: str) -> Repository:
        """
        Get repository details.

        Args:
            repository_id: The repository ID.

        Returns:
            Repository object.
        """
        response = self._client._get(f"/api/v1/codebase/repositories/{repository_id}")
        return self._parse_repository(response.get("repository", response))

    async def get_repository_async(self, repository_id: str) -> Repository:
        """Async version of get_repository()."""
        response = await self._client._get_async(f"/api/v1/codebase/repositories/{repository_id}")
        return self._parse_repository(response.get("repository", response))

    def sync_repository(self, repository_id: str) -> Repository:
        """
        Trigger repository sync.

        Args:
            repository_id: The repository ID.

        Returns:
            Updated Repository object.
        """
        response = self._client._post(f"/api/v1/codebase/repositories/{repository_id}/sync", {})
        return self._parse_repository(response.get("repository", response))

    async def sync_repository_async(self, repository_id: str) -> Repository:
        """Async version of sync_repository()."""
        response = await self._client._post_async(
            f"/api/v1/codebase/repositories/{repository_id}/sync", {}
        )
        return self._parse_repository(response.get("repository", response))

    def disconnect_repository(self, repository_id: str) -> bool:
        """
        Disconnect a repository.

        Args:
            repository_id: The repository ID.

        Returns:
            True if successful.
        """
        self._client._delete(f"/api/v1/codebase/repositories/{repository_id}")
        return True

    async def disconnect_repository_async(self, repository_id: str) -> bool:
        """Async version of disconnect_repository()."""
        await self._client._delete_async(f"/api/v1/codebase/repositories/{repository_id}")
        return True

    # =========================================================================
    # File Operations
    # =========================================================================

    def list_files(
        self,
        repository_id: str,
        path: str = "",
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[CodeFile], int]:
        """
        List files in a repository.

        Args:
            repository_id: The repository ID.
            path: Filter by path prefix.
            language: Filter by language.
            limit: Maximum number of files.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of CodeFile objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if path:
            params["path"] = path
        if language:
            params["language"] = language

        response = self._client._get(
            f"/api/v1/codebase/repositories/{repository_id}/files", params=params
        )
        files = [self._parse_file(f) for f in response.get("files", [])]
        return files, response.get("total", len(files))

    async def list_files_async(
        self,
        repository_id: str,
        path: str = "",
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[CodeFile], int]:
        """Async version of list_files()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if path:
            params["path"] = path
        if language:
            params["language"] = language

        response = await self._client._get_async(
            f"/api/v1/codebase/repositories/{repository_id}/files", params=params
        )
        files = [self._parse_file(f) for f in response.get("files", [])]
        return files, response.get("total", len(files))

    def get_file_content(self, repository_id: str, file_path: str) -> str:
        """
        Get file content.

        Args:
            repository_id: The repository ID.
            file_path: Path to the file.

        Returns:
            File content as string.
        """
        response = self._client._get(
            f"/api/v1/codebase/repositories/{repository_id}/files/content",
            params={"path": file_path},
        )
        return response.get("content", "")

    async def get_file_content_async(self, repository_id: str, file_path: str) -> str:
        """Async version of get_file_content()."""
        response = await self._client._get_async(
            f"/api/v1/codebase/repositories/{repository_id}/files/content",
            params={"path": file_path},
        )
        return response.get("content", "")

    # =========================================================================
    # Dependencies
    # =========================================================================

    def list_dependencies(
        self,
        repository_id: str,
        ecosystem: Optional[str] = None,
        vulnerable_only: bool = False,
    ) -> List[DependencyInfo]:
        """
        List dependencies for a repository.

        Args:
            repository_id: The repository ID.
            ecosystem: Filter by ecosystem (npm, pip, maven).
            vulnerable_only: Only show vulnerable dependencies.

        Returns:
            List of DependencyInfo objects.
        """
        params: Dict[str, Any] = {}
        if ecosystem:
            params["ecosystem"] = ecosystem
        if vulnerable_only:
            params["vulnerable_only"] = vulnerable_only

        response = self._client._get(
            f"/api/v1/codebase/repositories/{repository_id}/dependencies", params=params
        )
        return [self._parse_dependency(d) for d in response.get("dependencies", [])]

    async def list_dependencies_async(
        self,
        repository_id: str,
        ecosystem: Optional[str] = None,
        vulnerable_only: bool = False,
    ) -> List[DependencyInfo]:
        """Async version of list_dependencies()."""
        params: Dict[str, Any] = {}
        if ecosystem:
            params["ecosystem"] = ecosystem
        if vulnerable_only:
            params["vulnerable_only"] = vulnerable_only

        response = await self._client._get_async(
            f"/api/v1/codebase/repositories/{repository_id}/dependencies", params=params
        )
        return [self._parse_dependency(d) for d in response.get("dependencies", [])]

    # =========================================================================
    # Security Analysis
    # =========================================================================

    def run_security_scan(self, repository_id: str) -> AnalysisResult:
        """
        Run security scan on a repository.

        Args:
            repository_id: The repository ID.

        Returns:
            AnalysisResult object.
        """
        response = self._client._post(
            f"/api/v1/codebase/repositories/{repository_id}/security/scan", {}
        )
        return self._parse_analysis_result(response)

    async def run_security_scan_async(self, repository_id: str) -> AnalysisResult:
        """Async version of run_security_scan()."""
        response = await self._client._post_async(
            f"/api/v1/codebase/repositories/{repository_id}/security/scan", {}
        )
        return self._parse_analysis_result(response)

    def list_security_findings(
        self,
        repository_id: str,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[SecurityFinding], int]:
        """
        List security findings for a repository.

        Args:
            repository_id: The repository ID.
            severity: Filter by severity.
            status: Filter by status.
            limit: Maximum number of findings.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of SecurityFinding objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if severity:
            params["severity"] = severity
        if status:
            params["status"] = status

        response = self._client._get(
            f"/api/v1/codebase/repositories/{repository_id}/security/findings",
            params=params,
        )
        findings = [self._parse_finding(f) for f in response.get("findings", [])]
        return findings, response.get("total", len(findings))

    async def list_security_findings_async(
        self,
        repository_id: str,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[SecurityFinding], int]:
        """Async version of list_security_findings()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if severity:
            params["severity"] = severity
        if status:
            params["status"] = status

        response = await self._client._get_async(
            f"/api/v1/codebase/repositories/{repository_id}/security/findings",
            params=params,
        )
        findings = [self._parse_finding(f) for f in response.get("findings", [])]
        return findings, response.get("total", len(findings))

    def update_finding_status(
        self,
        repository_id: str,
        finding_id: str,
        status: str,
    ) -> SecurityFinding:
        """
        Update a security finding status.

        Args:
            repository_id: The repository ID.
            finding_id: The finding ID.
            status: New status (open, fixed, ignored).

        Returns:
            Updated SecurityFinding object.
        """
        body = {"status": status}
        response = self._client._patch(
            f"/api/v1/codebase/repositories/{repository_id}/security/findings/{finding_id}",
            body,
        )
        return self._parse_finding(response.get("finding", response))

    async def update_finding_status_async(
        self,
        repository_id: str,
        finding_id: str,
        status: str,
    ) -> SecurityFinding:
        """Async version of update_finding_status()."""
        body = {"status": status}
        response = await self._client._patch_async(
            f"/api/v1/codebase/repositories/{repository_id}/security/findings/{finding_id}",
            body,
        )
        return self._parse_finding(response.get("finding", response))

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_repository(self, data: Dict[str, Any]) -> Repository:
        """Parse repository data into Repository object."""
        last_synced_at = None
        if data.get("last_synced_at"):
            try:
                last_synced_at = datetime.fromisoformat(
                    data["last_synced_at"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return Repository(
            id=data.get("id", ""),
            name=data.get("name", ""),
            url=data.get("url", ""),
            provider=data.get("provider", ""),
            branch=data.get("branch", "main"),
            status=data.get("status", "active"),
            last_synced_at=last_synced_at,
            file_count=data.get("file_count", 0),
            language_stats=data.get("language_stats", {}),
        )

    def _parse_file(self, data: Dict[str, Any]) -> CodeFile:
        """Parse file data into CodeFile object."""
        last_modified = None
        if data.get("last_modified"):
            try:
                last_modified = datetime.fromisoformat(data["last_modified"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return CodeFile(
            path=data.get("path", ""),
            language=data.get("language", ""),
            size_bytes=data.get("size_bytes", 0),
            lines=data.get("lines", 0),
            last_modified=last_modified,
            complexity_score=data.get("complexity_score", 0.0),
        )

    def _parse_dependency(self, data: Dict[str, Any]) -> DependencyInfo:
        """Parse dependency data into DependencyInfo object."""
        return DependencyInfo(
            name=data.get("name", ""),
            version=data.get("version", ""),
            type=data.get("type", "direct"),
            ecosystem=data.get("ecosystem", ""),
            latest_version=data.get("latest_version"),
            has_vulnerabilities=data.get("has_vulnerabilities", False),
            vulnerability_count=data.get("vulnerability_count", 0),
        )

    def _parse_finding(self, data: Dict[str, Any]) -> SecurityFinding:
        """Parse finding data into SecurityFinding object."""
        return SecurityFinding(
            id=data.get("id", ""),
            type=data.get("type", ""),
            severity=data.get("severity", "medium"),
            title=data.get("title", ""),
            description=data.get("description", ""),
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            cwe_id=data.get("cwe_id"),
            recommendation=data.get("recommendation"),
            status=data.get("status", "open"),
        )

    def _parse_analysis_result(self, data: Dict[str, Any]) -> AnalysisResult:
        """Parse analysis result data into AnalysisResult object."""
        started_at = None
        completed_at = None

        if data.get("started_at"):
            try:
                started_at = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("completed_at"):
            try:
                completed_at = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return AnalysisResult(
            repository_id=data.get("repository_id", ""),
            analysis_type=data.get("analysis_type", ""),
            status=data.get("status", "pending"),
            started_at=started_at,
            completed_at=completed_at,
            findings_count=data.get("findings_count", 0),
            summary=data.get("summary", {}),
        )


__all__ = [
    "CodebaseAPI",
    "Repository",
    "CodeFile",
    "DependencyInfo",
    "SecurityFinding",
    "AnalysisResult",
]
