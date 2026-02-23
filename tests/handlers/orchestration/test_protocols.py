"""Tests for orchestration type protocols.

Covers all Protocol classes and the RecommendAgentsFunc type alias in
``aragora.server.handlers.orchestration.protocols``:

- ConfluenceConnectorProtocol (structural conformance, method signature)
- GitHubConnectorProtocol (structural conformance, dual methods)
- JiraConnectorProtocol (structural conformance, dict return)
- EmailSenderProtocol (callable protocol, __call__ signature)
- KnowledgeMoundProtocol (structural conformance, default param)
- RecommendAgentsFunc (type alias verification)
- Concrete implementations satisfying each protocol
- AsyncMock as protocol stand-in
- Partial / non-conforming implementations
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, Protocol, get_type_hints
from unittest.mock import AsyncMock

import pytest

from aragora.server.handlers.orchestration.protocols import (
    ConfluenceConnectorProtocol,
    EmailSenderProtocol,
    GitHubConnectorProtocol,
    JiraConnectorProtocol,
    KnowledgeMoundProtocol,
    RecommendAgentsFunc,
)


# ============================================================================
# Helpers: concrete implementations that structurally match protocols
# ============================================================================


class ConcreteConfluence:
    """Concrete class structurally matching ConfluenceConnectorProtocol."""

    async def get_page_content(self, page_id: str) -> str | None:
        return f"Page content for {page_id}"


class ConcreteGitHub:
    """Concrete class structurally matching GitHubConnectorProtocol."""

    async def get_pr_content(self, owner: str, repo: str, number: int) -> str | None:
        return f"PR {owner}/{repo}#{number}"

    async def get_issue_content(self, owner: str, repo: str, number: int) -> str | None:
        return f"Issue {owner}/{repo}#{number}"


class ConcreteJira:
    """Concrete class structurally matching JiraConnectorProtocol."""

    async def get_issue(self, issue_key: str) -> dict[str, Any] | None:
        return {"key": issue_key, "summary": "Test issue"}


class ConcreteEmailSender:
    """Concrete callable class structurally matching EmailSenderProtocol."""

    async def __call__(self, to: str, subject: str, body: str) -> None:
        pass


class ConcreteKnowledgeMound:
    """Concrete class structurally matching KnowledgeMoundProtocol."""

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return [{"query": query, "limit": limit}]


# ============================================================================
# Helpers: partial / non-conforming implementations
# ============================================================================


class PartialGitHub:
    """Only implements get_pr_content, missing get_issue_content."""

    async def get_pr_content(self, owner: str, repo: str, number: int) -> str | None:
        return "partial"


class EmptyClass:
    """Has no methods at all."""

    pass


class WrongSignatureConfluence:
    """Has get_page_content but with wrong signature (missing page_id)."""

    async def get_page_content(self) -> str | None:
        return "wrong"


class SyncConfluence:
    """Has get_page_content but synchronous."""

    def get_page_content(self, page_id: str) -> str | None:
        return f"sync {page_id}"


# ============================================================================
# A. Module-Level Imports and Structure
# ============================================================================


class TestModuleStructure:
    """Verify the protocols module exports the expected names."""

    def test_confluence_protocol_is_importable(self):
        assert ConfluenceConnectorProtocol is not None

    def test_github_protocol_is_importable(self):
        assert GitHubConnectorProtocol is not None

    def test_jira_protocol_is_importable(self):
        assert JiraConnectorProtocol is not None

    def test_email_sender_protocol_is_importable(self):
        assert EmailSenderProtocol is not None

    def test_knowledge_mound_protocol_is_importable(self):
        assert KnowledgeMoundProtocol is not None

    def test_recommend_agents_func_is_importable(self):
        assert RecommendAgentsFunc is not None

    def test_all_protocols_are_protocol_subclasses(self):
        """All protocol classes should be subclasses of typing.Protocol."""
        for cls in [
            ConfluenceConnectorProtocol,
            GitHubConnectorProtocol,
            JiraConnectorProtocol,
            EmailSenderProtocol,
            KnowledgeMoundProtocol,
        ]:
            assert issubclass(cls, Protocol)

    def test_protocol_count(self):
        """The module defines exactly 5 Protocol classes."""
        import aragora.server.handlers.orchestration.protocols as mod

        protocol_classes = [
            v
            for v in vars(mod).values()
            if isinstance(v, type)
            and issubclass(v, Protocol)
            and v is not Protocol
            and v.__module__ == mod.__name__
        ]
        assert len(protocol_classes) == 5


# ============================================================================
# B. ConfluenceConnectorProtocol
# ============================================================================


class TestConfluenceConnectorProtocol:
    """Tests for ConfluenceConnectorProtocol."""

    def test_has_get_page_content_method(self):
        assert hasattr(ConfluenceConnectorProtocol, "get_page_content")

    def test_get_page_content_is_coroutine_function(self):
        method = ConfluenceConnectorProtocol.get_page_content
        assert inspect.iscoroutinefunction(method)

    def test_get_page_content_signature(self):
        sig = inspect.signature(ConfluenceConnectorProtocol.get_page_content)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "page_id" in params
        assert len(params) == 2

    def test_concrete_implementation_has_correct_method(self):
        impl = ConcreteConfluence()
        assert hasattr(impl, "get_page_content")
        assert inspect.iscoroutinefunction(impl.get_page_content)

    @pytest.mark.asyncio
    async def test_concrete_implementation_returns_content(self):
        impl = ConcreteConfluence()
        result = await impl.get_page_content("pg-123")
        assert result == "Page content for pg-123"

    @pytest.mark.asyncio
    async def test_async_mock_as_confluence(self):
        mock = AsyncMock()
        mock.get_page_content = AsyncMock(return_value="mocked content")
        result = await mock.get_page_content("pg-1")
        assert result == "mocked content"
        mock.get_page_content.assert_awaited_once_with("pg-1")

    @pytest.mark.asyncio
    async def test_concrete_returns_none(self):
        """Protocol allows returning None."""

        class NoneConfluence:
            async def get_page_content(self, page_id: str) -> str | None:
                return None

        impl = NoneConfluence()
        result = await impl.get_page_content("missing")
        assert result is None

    def test_docstring_present(self):
        assert ConfluenceConnectorProtocol.__doc__ is not None
        assert "Confluence" in ConfluenceConnectorProtocol.__doc__


# ============================================================================
# C. GitHubConnectorProtocol
# ============================================================================


class TestGitHubConnectorProtocol:
    """Tests for GitHubConnectorProtocol."""

    def test_has_get_pr_content_method(self):
        assert hasattr(GitHubConnectorProtocol, "get_pr_content")

    def test_has_get_issue_content_method(self):
        assert hasattr(GitHubConnectorProtocol, "get_issue_content")

    def test_both_methods_are_coroutines(self):
        assert inspect.iscoroutinefunction(GitHubConnectorProtocol.get_pr_content)
        assert inspect.iscoroutinefunction(GitHubConnectorProtocol.get_issue_content)

    def test_get_pr_content_signature(self):
        sig = inspect.signature(GitHubConnectorProtocol.get_pr_content)
        params = list(sig.parameters.keys())
        assert params == ["self", "owner", "repo", "number"]

    def test_get_issue_content_signature(self):
        sig = inspect.signature(GitHubConnectorProtocol.get_issue_content)
        params = list(sig.parameters.keys())
        assert params == ["self", "owner", "repo", "number"]

    @pytest.mark.asyncio
    async def test_concrete_get_pr_content(self):
        impl = ConcreteGitHub()
        result = await impl.get_pr_content("org", "repo", 42)
        assert result == "PR org/repo#42"

    @pytest.mark.asyncio
    async def test_concrete_get_issue_content(self):
        impl = ConcreteGitHub()
        result = await impl.get_issue_content("org", "repo", 7)
        assert result == "Issue org/repo#7"

    @pytest.mark.asyncio
    async def test_async_mock_as_github(self):
        mock = AsyncMock()
        mock.get_pr_content = AsyncMock(return_value="PR body")
        mock.get_issue_content = AsyncMock(return_value="Issue body")
        pr = await mock.get_pr_content("o", "r", 1)
        issue = await mock.get_issue_content("o", "r", 2)
        assert pr == "PR body"
        assert issue == "Issue body"

    def test_docstring_present(self):
        assert GitHubConnectorProtocol.__doc__ is not None
        assert "GitHub" in GitHubConnectorProtocol.__doc__

    def test_partial_implementation_missing_method(self):
        """PartialGitHub lacks get_issue_content."""
        impl = PartialGitHub()
        assert hasattr(impl, "get_pr_content")
        assert not hasattr(impl, "get_issue_content")


# ============================================================================
# D. JiraConnectorProtocol
# ============================================================================


class TestJiraConnectorProtocol:
    """Tests for JiraConnectorProtocol."""

    def test_has_get_issue_method(self):
        assert hasattr(JiraConnectorProtocol, "get_issue")

    def test_get_issue_is_coroutine(self):
        assert inspect.iscoroutinefunction(JiraConnectorProtocol.get_issue)

    def test_get_issue_signature(self):
        sig = inspect.signature(JiraConnectorProtocol.get_issue)
        params = list(sig.parameters.keys())
        assert params == ["self", "issue_key"]

    @pytest.mark.asyncio
    async def test_concrete_get_issue(self):
        impl = ConcreteJira()
        result = await impl.get_issue("PROJ-123")
        assert result == {"key": "PROJ-123", "summary": "Test issue"}

    @pytest.mark.asyncio
    async def test_concrete_returns_none(self):
        class NoneJira:
            async def get_issue(self, issue_key: str) -> dict[str, Any] | None:
                return None

        impl = NoneJira()
        result = await impl.get_issue("MISSING-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_async_mock_as_jira(self):
        mock = AsyncMock()
        mock.get_issue = AsyncMock(return_value={"key": "T-1", "status": "open"})
        result = await mock.get_issue("T-1")
        assert result["key"] == "T-1"
        assert result["status"] == "open"

    def test_docstring_present(self):
        assert JiraConnectorProtocol.__doc__ is not None
        assert "Jira" in JiraConnectorProtocol.__doc__


# ============================================================================
# E. EmailSenderProtocol
# ============================================================================


class TestEmailSenderProtocol:
    """Tests for EmailSenderProtocol."""

    def test_has_call_method(self):
        assert hasattr(EmailSenderProtocol, "__call__")

    def test_call_is_coroutine(self):
        assert inspect.iscoroutinefunction(EmailSenderProtocol.__call__)

    def test_call_signature(self):
        sig = inspect.signature(EmailSenderProtocol.__call__)
        params = list(sig.parameters.keys())
        assert params == ["self", "to", "subject", "body"]

    @pytest.mark.asyncio
    async def test_concrete_email_sender(self):
        sender = ConcreteEmailSender()
        result = await sender("user@example.com", "Hello", "Body text")
        assert result is None  # Returns None

    @pytest.mark.asyncio
    async def test_async_mock_as_email_sender(self):
        mock = AsyncMock()
        await mock("admin@co.com", "Subject", "Body")
        mock.assert_awaited_once_with("admin@co.com", "Subject", "Body")

    @pytest.mark.asyncio
    async def test_email_sender_async_function(self):
        """A plain async function can also serve as an EmailSender."""
        calls = []

        async def send_email(to: str, subject: str, body: str) -> None:
            calls.append((to, subject, body))

        await send_email("test@example.com", "Test", "Hello")
        assert len(calls) == 1
        assert calls[0] == ("test@example.com", "Test", "Hello")

    def test_docstring_present(self):
        assert EmailSenderProtocol.__doc__ is not None
        assert "email" in EmailSenderProtocol.__doc__.lower()


# ============================================================================
# F. KnowledgeMoundProtocol
# ============================================================================


class TestKnowledgeMoundProtocol:
    """Tests for KnowledgeMoundProtocol."""

    def test_has_search_method(self):
        assert hasattr(KnowledgeMoundProtocol, "search")

    def test_search_is_coroutine(self):
        assert inspect.iscoroutinefunction(KnowledgeMoundProtocol.search)

    def test_search_signature(self):
        sig = inspect.signature(KnowledgeMoundProtocol.search)
        params = list(sig.parameters.keys())
        assert params == ["self", "query", "limit"]

    def test_search_limit_has_default(self):
        sig = inspect.signature(KnowledgeMoundProtocol.search)
        limit_param = sig.parameters["limit"]
        assert limit_param.default == 10

    @pytest.mark.asyncio
    async def test_concrete_search(self):
        impl = ConcreteKnowledgeMound()
        results = await impl.search("test query")
        assert len(results) == 1
        assert results[0]["query"] == "test query"
        assert results[0]["limit"] == 10

    @pytest.mark.asyncio
    async def test_concrete_search_with_custom_limit(self):
        impl = ConcreteKnowledgeMound()
        results = await impl.search("q", limit=5)
        assert results[0]["limit"] == 5

    @pytest.mark.asyncio
    async def test_async_mock_as_knowledge_mound(self):
        mock = AsyncMock()
        mock.search = AsyncMock(return_value=[{"id": "doc-1", "text": "result"}])
        results = await mock.search("query", limit=20)
        assert len(results) == 1
        assert results[0]["id"] == "doc-1"
        mock.search.assert_awaited_once_with("query", limit=20)

    @pytest.mark.asyncio
    async def test_search_returns_empty_list(self):
        class EmptyKM:
            async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
                return []

        impl = EmptyKM()
        results = await impl.search("nothing")
        assert results == []

    def test_docstring_present(self):
        assert KnowledgeMoundProtocol.__doc__ is not None
        assert "Knowledge Mound" in KnowledgeMoundProtocol.__doc__


# ============================================================================
# G. RecommendAgentsFunc Type Alias
# ============================================================================


class TestRecommendAgentsFunc:
    """Tests for the RecommendAgentsFunc type alias."""

    def test_is_callable_alias(self):
        """RecommendAgentsFunc should be a Callable type."""
        origin = getattr(RecommendAgentsFunc, "__origin__", None)
        # Callable aliases have __origin__ of collections.abc.Callable
        assert origin is Callable or str(RecommendAgentsFunc).startswith("collections.abc.Callable")

    def test_accepts_string_argument(self):
        """RecommendAgentsFunc takes a single str argument."""
        args = getattr(RecommendAgentsFunc, "__args__", None)
        assert args is not None
        # For Callable[[str], Any] the __args__ tuple is (str, 'Any')
        # The first element(s) before the return type are the parameter types
        assert len(args) >= 2
        param_type = args[0]
        assert param_type is str

    def test_returns_any(self):
        """RecommendAgentsFunc returns Any."""
        args = getattr(RecommendAgentsFunc, "__args__", None)
        assert args is not None
        return_type = args[-1]
        # Due to `from __future__ import annotations`, the return type may be
        # the string "Any" (forward ref) rather than typing.Any itself.
        assert return_type is Any or str(return_type) == "Any"

    def test_sync_function_matches(self):
        """A regular sync function with matching signature can be used."""

        def recommend(query: str) -> list[str]:
            return ["claude", "gpt-4"]

        result = recommend("design a rate limiter")
        assert "claude" in result

    def test_async_function_matches_shape(self):
        """An async function can also be used if the caller awaits."""

        async def recommend(query: str) -> list[str]:
            return ["gemini"]

        coro = recommend("test")
        # Should produce a coroutine
        assert asyncio.iscoroutine(coro)
        coro.close()


# ============================================================================
# H. Cross-Cutting: Structural Typing Conformance
# ============================================================================


class TestStructuralConformance:
    """Verify structural typing concepts across protocols."""

    def test_empty_class_lacks_confluence_method(self):
        impl = EmptyClass()
        assert not hasattr(impl, "get_page_content")

    def test_wrong_signature_has_method_but_different_params(self):
        impl = WrongSignatureConfluence()
        assert hasattr(impl, "get_page_content")
        sig = inspect.signature(impl.get_page_content)
        # Only has 'self' (bound method shows no params)
        assert "page_id" not in sig.parameters

    def test_sync_implementation_is_not_coroutine(self):
        impl = SyncConfluence()
        assert not inspect.iscoroutinefunction(impl.get_page_content)

    def test_all_protocol_methods_are_async(self):
        """Every method defined in the protocols should be async."""
        protocols = [
            ConfluenceConnectorProtocol,
            GitHubConnectorProtocol,
            JiraConnectorProtocol,
            EmailSenderProtocol,
            KnowledgeMoundProtocol,
        ]
        for proto in protocols:
            for name, method in inspect.getmembers(proto, predicate=inspect.isfunction):
                if name.startswith("_") and name != "__call__":
                    continue
                assert inspect.iscoroutinefunction(method), (
                    f"{proto.__name__}.{name} should be a coroutine function"
                )

    def test_github_protocol_has_two_public_methods(self):
        """GitHubConnectorProtocol defines exactly 2 protocol methods."""
        methods = [
            name
            for name, _ in inspect.getmembers(
                GitHubConnectorProtocol, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        ]
        assert set(methods) == {"get_pr_content", "get_issue_content"}

    def test_confluence_protocol_has_one_public_method(self):
        methods = [
            name
            for name, _ in inspect.getmembers(
                ConfluenceConnectorProtocol, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        ]
        assert set(methods) == {"get_page_content"}

    def test_jira_protocol_has_one_public_method(self):
        methods = [
            name
            for name, _ in inspect.getmembers(
                JiraConnectorProtocol, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        ]
        assert set(methods) == {"get_issue"}

    def test_knowledge_mound_protocol_has_one_public_method(self):
        methods = [
            name
            for name, _ in inspect.getmembers(
                KnowledgeMoundProtocol, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        ]
        assert set(methods) == {"search"}

    def test_email_sender_protocol_callable_method(self):
        """EmailSenderProtocol defines __call__ as its protocol method."""
        assert inspect.iscoroutinefunction(EmailSenderProtocol.__call__)


# ============================================================================
# I. AsyncMock Integration
# ============================================================================


class TestAsyncMockIntegration:
    """Verify AsyncMock objects work seamlessly as protocol stand-ins."""

    @pytest.mark.asyncio
    async def test_mock_confluence_side_effect(self):
        mock = AsyncMock()
        mock.get_page_content = AsyncMock(side_effect=ConnectionError("timeout"))
        with pytest.raises(ConnectionError, match="timeout"):
            await mock.get_page_content("pg-1")

    @pytest.mark.asyncio
    async def test_mock_github_multiple_calls(self):
        mock = AsyncMock()
        mock.get_pr_content = AsyncMock(side_effect=["PR1", "PR2", "PR3"])
        assert await mock.get_pr_content("o", "r", 1) == "PR1"
        assert await mock.get_pr_content("o", "r", 2) == "PR2"
        assert await mock.get_pr_content("o", "r", 3) == "PR3"

    @pytest.mark.asyncio
    async def test_mock_jira_returns_complex_dict(self):
        issue_data = {
            "key": "PROJ-999",
            "summary": "Complex issue",
            "status": "in_progress",
            "assignee": {"name": "dev", "email": "dev@co.com"},
            "labels": ["bug", "priority"],
        }
        mock = AsyncMock()
        mock.get_issue = AsyncMock(return_value=issue_data)
        result = await mock.get_issue("PROJ-999")
        assert result["assignee"]["email"] == "dev@co.com"
        assert "bug" in result["labels"]

    @pytest.mark.asyncio
    async def test_mock_knowledge_mound_with_limit(self):
        mock = AsyncMock()
        mock.search = AsyncMock(return_value=[{"id": str(i)} for i in range(5)])
        results = await mock.search("test", limit=5)
        assert len(results) == 5
        assert results[4]["id"] == "4"

    @pytest.mark.asyncio
    async def test_mock_email_sender_tracks_calls(self):
        mock = AsyncMock()
        await mock("a@b.com", "Sub1", "Body1")
        await mock("c@d.com", "Sub2", "Body2")
        assert mock.await_count == 2
        mock.assert_any_await("a@b.com", "Sub1", "Body1")
        mock.assert_any_await("c@d.com", "Sub2", "Body2")
