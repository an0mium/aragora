"""
Repository Namespace API

Provides methods for codebase repository indexing operations:
- Full and incremental repository indexing
- Entity extraction (classes, functions, etc.)
- Relationship graph building
- Entity search and filtering
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class RepositoryAPI:
    """
    Synchronous Repository API.

    Provides methods for codebase indexing and entity extraction:
    - Full and incremental repository indexing
    - Batch indexing of multiple repositories
    - Entity extraction (classes, functions, etc.)
    - Relationship graph building
    - Entity search and filtering

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.repository.index(
        ...     repository_url="https://github.com/org/repo",
        ...     branch="main",
        ...     extract_entities=True,
        ...     build_graph=True,
        ... )
        >>> status = client.repository.get_status(result["index_id"])
        >>> entities = client.repository.list_entities(
        ...     result["index_id"],
        ...     type="class",
        ...     language="python",
        ... )
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Indexing Operations
    # ===========================================================================

    def index(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        branch: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size_kb: int | None = None,
        extract_entities: bool | None = None,
        build_graph: bool | None = None,
    ) -> dict[str, Any]:
        """
        Start a full repository index.

        Args:
            repository_url: URL of the repository to index (e.g., GitHub URL)
            local_path: Local filesystem path to the repository
            branch: Branch to index (default: main/master)
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            max_file_size_kb: Maximum file size in KB to index
            extract_entities: Whether to extract code entities (classes, functions, etc.)
            build_graph: Whether to build relationship graph

        Returns:
            Dict with index_id, status, started_at, and optional estimated_completion_at
        """
        data: dict[str, Any] = {}
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        if branch is not None:
            data["branch"] = branch
        if include_patterns is not None:
            data["include_patterns"] = include_patterns
        if exclude_patterns is not None:
            data["exclude_patterns"] = exclude_patterns
        if max_file_size_kb is not None:
            data["max_file_size_kb"] = max_file_size_kb
        if extract_entities is not None:
            data["extract_entities"] = extract_entities
        if build_graph is not None:
            data["build_graph"] = build_graph
        return self._client.request("POST", "/api/v1/repository/index", json=data)

    def incremental_index(
        self,
        index_id: str,
        changed_files: list[str] | None = None,
        since_commit: str | None = None,
    ) -> dict[str, Any]:
        """
        Run an incremental index update.

        Args:
            index_id: ID of the existing index to update
            changed_files: List of file paths that have changed
            since_commit: Git commit SHA to use as baseline for changes

        Returns:
            Dict with index_id, status, started_at, and optional estimated_completion_at
        """
        data: dict[str, Any] = {"index_id": index_id}
        if changed_files is not None:
            data["changed_files"] = changed_files
        if since_commit is not None:
            data["since_commit"] = since_commit
        return self._client.request("POST", "/api/v1/repository/incremental", json=data)

    def batch_index(
        self,
        repositories: list[dict[str, Any]],
        parallel: bool | None = None,
        max_concurrent: int | None = None,
    ) -> dict[str, Any]:
        """
        Index multiple repositories in batch.

        Args:
            repositories: List of repository index requests (each with repository_url,
                local_path, branch, include_patterns, exclude_patterns, etc.)
            parallel: Whether to run indexing in parallel (default: True)
            max_concurrent: Maximum number of concurrent indexing operations

        Returns:
            Dict with batch_id, index_ids array, and status
        """
        data: dict[str, Any] = {"repositories": repositories}
        if parallel is not None:
            data["parallel"] = parallel
        if max_concurrent is not None:
            data["max_concurrent"] = max_concurrent
        return self._client.request("POST", "/api/v1/repository/batch", json=data)

    # ===========================================================================
    # Status and Queries
    # ===========================================================================

    def get_status(self, index_id: str) -> dict[str, Any]:
        """
        Get index status.

        Args:
            index_id: ID of the index

        Returns:
            Dict with index_id, status, progress_percent, files_indexed, files_total,
            entities_extracted, relationships_found, started_at, completed_at, and error
        """
        return self._client.request("GET", f"/api/v1/repository/{index_id}/status")

    def list_entities(
        self,
        index_id: str,
        type: str | None = None,
        language: str | None = None,
        file_pattern: str | None = None,
        name_pattern: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        List entities from an indexed repository.

        Args:
            index_id: ID of the index
            type: Filter by entity type (class, function, method, variable, module, interface, type)
            language: Filter by programming language
            file_pattern: Glob pattern to filter by file path
            name_pattern: Pattern to filter by entity name
            visibility: Filter by visibility (public, private, protected, internal)
            limit: Maximum number of entities to return
            offset: Offset for pagination

        Returns:
            Dict with entities array and total count
        """
        params: dict[str, Any] = {}
        if type is not None:
            params["type"] = type
        if language is not None:
            params["language"] = language
        if file_pattern is not None:
            params["file_pattern"] = file_pattern
        if name_pattern is not None:
            params["name_pattern"] = name_pattern
        if visibility is not None:
            params["visibility"] = visibility
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request("GET", f"/api/v1/repository/{index_id}/entities", params=params)

    def get_repository(self, repo_id: str) -> dict[str, Any]:
        """
        Get repository information.

        Args:
            repo_id: Repository ID

        Returns:
            Dict with repository metadata, indexing status, and statistics
        """
        return self._client.request("GET", f"/api/v1/repository/{repo_id}")

    def delete_repository(self, repo_id: str) -> dict[str, Any]:
        """
        Remove an indexed repository.

        Args:
            repo_id: Repository ID to remove

        Returns:
            Deletion confirmation
        """
        return self._client.request("DELETE", f"/api/v1/repository/{repo_id}")

    def get_graph(self, index_id: str) -> dict[str, Any]:
        """
        Get the relationship graph for an indexed repository.

        Args:
            index_id: ID of the index

        Returns:
            Dict with nodes (entities), edges (relationships), and statistics
        """
        return self._client.request("GET", f"/api/v1/repository/{index_id}/graph")

    def wait_for_index(
        self,
        index_id: str,
        poll_interval_ms: int = 2000,
        timeout_ms: int = 300000,
    ) -> dict[str, Any]:
        """
        Wait for index to complete.

        Polls the index status until it reaches 'completed' or 'failed' state,
        or until the timeout is reached.

        Args:
            index_id: ID of the index to wait for
            poll_interval_ms: Polling interval in milliseconds (default: 2000)
            timeout_ms: Maximum wait time in milliseconds (default: 300000 / 5 minutes)

        Returns:
            Dict with final index status

        Raises:
            TimeoutError: If index does not complete within the timeout
        """
        poll_interval_s = poll_interval_ms / 1000.0
        timeout_s = timeout_ms / 1000.0
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout_s:
            status = self.get_status(index_id)
            if status.get("status") in ("completed", "failed"):
                return status
            time.sleep(poll_interval_s)

        raise TimeoutError(f"Index {index_id} did not complete within {timeout_ms}ms")

    def index_and_wait(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        branch: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size_kb: int | None = None,
        extract_entities: bool | None = None,
        build_graph: bool | None = None,
        poll_interval_ms: int = 2000,
        timeout_ms: int = 300000,
    ) -> dict[str, Any]:
        """
        Index a repository and wait for completion.

        Combines index() and wait_for_index() into a single convenience method.

        Args:
            repository_url: URL of the repository to index
            local_path: Local filesystem path to the repository
            branch: Branch to index
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            max_file_size_kb: Maximum file size in KB to index
            extract_entities: Whether to extract code entities
            build_graph: Whether to build relationship graph
            poll_interval_ms: Polling interval in milliseconds (default: 2000)
            timeout_ms: Maximum wait time in milliseconds (default: 300000 / 5 minutes)

        Returns:
            Dict with final index status

        Raises:
            TimeoutError: If index does not complete within the timeout
        """
        result = self.index(
            repository_url=repository_url,
            local_path=local_path,
            branch=branch,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            max_file_size_kb=max_file_size_kb,
            extract_entities=extract_entities,
            build_graph=build_graph,
        )
        return self.wait_for_index(
            result["index_id"],
            poll_interval_ms=poll_interval_ms,
            timeout_ms=timeout_ms,
        )

class AsyncRepositoryAPI:
    """
    Asynchronous Repository API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.repository.index(
        ...         repository_url="https://github.com/org/repo",
        ...         branch="main",
        ...         extract_entities=True,
        ...     )
        ...     status = await client.repository.wait_for_index(result["index_id"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Indexing Operations
    # ===========================================================================

    async def index(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        branch: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size_kb: int | None = None,
        extract_entities: bool | None = None,
        build_graph: bool | None = None,
    ) -> dict[str, Any]:
        """
        Start a full repository index.

        Args:
            repository_url: URL of the repository to index (e.g., GitHub URL)
            local_path: Local filesystem path to the repository
            branch: Branch to index (default: main/master)
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            max_file_size_kb: Maximum file size in KB to index
            extract_entities: Whether to extract code entities (classes, functions, etc.)
            build_graph: Whether to build relationship graph

        Returns:
            Dict with index_id, status, started_at, and optional estimated_completion_at
        """
        data: dict[str, Any] = {}
        if repository_url is not None:
            data["repository_url"] = repository_url
        if local_path is not None:
            data["local_path"] = local_path
        if branch is not None:
            data["branch"] = branch
        if include_patterns is not None:
            data["include_patterns"] = include_patterns
        if exclude_patterns is not None:
            data["exclude_patterns"] = exclude_patterns
        if max_file_size_kb is not None:
            data["max_file_size_kb"] = max_file_size_kb
        if extract_entities is not None:
            data["extract_entities"] = extract_entities
        if build_graph is not None:
            data["build_graph"] = build_graph
        return await self._client.request("POST", "/api/v1/repository/index", json=data)

    async def incremental_index(
        self,
        index_id: str,
        changed_files: list[str] | None = None,
        since_commit: str | None = None,
    ) -> dict[str, Any]:
        """
        Run an incremental index update.

        Args:
            index_id: ID of the existing index to update
            changed_files: List of file paths that have changed
            since_commit: Git commit SHA to use as baseline for changes

        Returns:
            Dict with index_id, status, started_at, and optional estimated_completion_at
        """
        data: dict[str, Any] = {"index_id": index_id}
        if changed_files is not None:
            data["changed_files"] = changed_files
        if since_commit is not None:
            data["since_commit"] = since_commit
        return await self._client.request("POST", "/api/v1/repository/incremental", json=data)

    async def batch_index(
        self,
        repositories: list[dict[str, Any]],
        parallel: bool | None = None,
        max_concurrent: int | None = None,
    ) -> dict[str, Any]:
        """
        Index multiple repositories in batch.

        Args:
            repositories: List of repository index requests
            parallel: Whether to run indexing in parallel (default: True)
            max_concurrent: Maximum number of concurrent indexing operations

        Returns:
            Dict with batch_id, index_ids array, and status
        """
        data: dict[str, Any] = {"repositories": repositories}
        if parallel is not None:
            data["parallel"] = parallel
        if max_concurrent is not None:
            data["max_concurrent"] = max_concurrent
        return await self._client.request("POST", "/api/v1/repository/batch", json=data)

    # ===========================================================================
    # Status and Queries
    # ===========================================================================

    async def get_status(self, index_id: str) -> dict[str, Any]:
        """
        Get index status.

        Args:
            index_id: ID of the index

        Returns:
            Dict with index_id, status, progress_percent, files_indexed, files_total,
            entities_extracted, relationships_found, started_at, completed_at, and error
        """
        return await self._client.request("GET", f"/api/v1/repository/{index_id}/status")

    async def list_entities(
        self,
        index_id: str,
        type: str | None = None,
        language: str | None = None,
        file_pattern: str | None = None,
        name_pattern: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        List entities from an indexed repository.

        Args:
            index_id: ID of the index
            type: Filter by entity type (class, function, method, variable, module, interface, type)
            language: Filter by programming language
            file_pattern: Glob pattern to filter by file path
            name_pattern: Pattern to filter by entity name
            visibility: Filter by visibility (public, private, protected, internal)
            limit: Maximum number of entities to return
            offset: Offset for pagination

        Returns:
            Dict with entities array and total count
        """
        params: dict[str, Any] = {}
        if type is not None:
            params["type"] = type
        if language is not None:
            params["language"] = language
        if file_pattern is not None:
            params["file_pattern"] = file_pattern
        if name_pattern is not None:
            params["name_pattern"] = name_pattern
        if visibility is not None:
            params["visibility"] = visibility
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request(
            "GET", f"/api/v1/repository/{index_id}/entities", params=params
        )

    async def get_repository(self, repo_id: str) -> dict[str, Any]:
        """Get repository information."""
        return await self._client.request("GET", f"/api/v1/repository/{repo_id}")

    async def delete_repository(self, repo_id: str) -> dict[str, Any]:
        """Remove an indexed repository."""
        return await self._client.request("DELETE", f"/api/v1/repository/{repo_id}")

    async def get_graph(self, index_id: str) -> dict[str, Any]:
        """
        Get the relationship graph for an indexed repository.

        Args:
            index_id: ID of the index

        Returns:
            Dict with nodes (entities), edges (relationships), and statistics
        """
        return await self._client.request("GET", f"/api/v1/repository/{index_id}/graph")

    async def wait_for_index(
        self,
        index_id: str,
        poll_interval_ms: int = 2000,
        timeout_ms: int = 300000,
    ) -> dict[str, Any]:
        """
        Wait for index to complete.

        Polls the index status until it reaches 'completed' or 'failed' state,
        or until the timeout is reached.

        Args:
            index_id: ID of the index to wait for
            poll_interval_ms: Polling interval in milliseconds (default: 2000)
            timeout_ms: Maximum wait time in milliseconds (default: 300000 / 5 minutes)

        Returns:
            Dict with final index status

        Raises:
            TimeoutError: If index does not complete within the timeout
        """
        import asyncio

        poll_interval_s = poll_interval_ms / 1000.0
        timeout_s = timeout_ms / 1000.0
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout_s:
            status = await self.get_status(index_id)
            if status.get("status") in ("completed", "failed"):
                return status
            await asyncio.sleep(poll_interval_s)

        raise TimeoutError(f"Index {index_id} did not complete within {timeout_ms}ms")

    async def index_and_wait(
        self,
        repository_url: str | None = None,
        local_path: str | None = None,
        branch: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size_kb: int | None = None,
        extract_entities: bool | None = None,
        build_graph: bool | None = None,
        poll_interval_ms: int = 2000,
        timeout_ms: int = 300000,
    ) -> dict[str, Any]:
        """
        Index a repository and wait for completion.

        Combines index() and wait_for_index() into a single convenience method.

        Args:
            repository_url: URL of the repository to index
            local_path: Local filesystem path to the repository
            branch: Branch to index
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            max_file_size_kb: Maximum file size in KB to index
            extract_entities: Whether to extract code entities
            build_graph: Whether to build relationship graph
            poll_interval_ms: Polling interval in milliseconds (default: 2000)
            timeout_ms: Maximum wait time in milliseconds (default: 300000 / 5 minutes)

        Returns:
            Dict with final index status

        Raises:
            TimeoutError: If index does not complete within the timeout
        """
        result = await self.index(
            repository_url=repository_url,
            local_path=local_path,
            branch=branch,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            max_file_size_kb=max_file_size_kb,
            extract_entities=extract_entities,
            build_graph=build_graph,
        )
        return await self.wait_for_index(
            result["index_id"],
            poll_interval_ms=poll_interval_ms,
            timeout_ms=timeout_ms,
        )
