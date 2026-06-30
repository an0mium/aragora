"""
Tests for orchestration connector type protocols.

Covers the public API surface of
``aragora.server.handlers.orchestration.protocols``:

- ConfluenceConnectorProtocol
- GitHubConnectorProtocol
- JiraConnectorProtocol
- EmailSenderProtocol
- KnowledgeMoundProtocol
- RecommendAgentsFunc type alias

Verifies runtime_checkable structural conformance (positive and negative),
async method behavior of conforming implementations, default parameter
values, protocol non-instantiability, and explicit-subclass usage.
All connectors are local fakes; no external services are touched.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, Protocol, get_args, get_origin

import pytest

from aragora.server.handlers.orchestration.protocols import (
    ConfluenceConnectorProtocol,
    EmailSenderProtocol,
    GitHubConnectorProtocol,
    JiraConnectorProtocol,
    KnowledgeMoundProtocol,
    RecommendAgentsFunc,
)

# ---------------------------------------------------------------------------
# Fakes used as structural implementations (no inheritance from protocols)
# ---------------------------------------------------------------------------


class FakeConfluence:
    """Structurally conforming Confluence connector."""

    def __init__(self, pages: dict[str, str] | None = None) -> None:
        self._pages = pages or {}

    async def get_page_content(self, page_id: str) -> str | None:
        return self._pages.get(page_id)


class FakeGitHub:
    """Structurally conforming GitHub connector."""

    async def get_pr_content(self, owner: str, repo: str, number: int) -> str | None:
        if number <= 0:
            return None
        return f"PR {owner}/{repo}#{number}"

    async def get_issue_content(self, owner: str, repo: str, number: int) -> str | None:
        if number <= 0:
            return None
        return f"Issue {owner}/{repo}#{number}"


class FakeJira:
    """Structurally conforming Jira connector."""

    async def get_issue(self, issue_key: str) -> dict[str, Any] | None:
        if not issue_key:
            return None
        return {"key": issue_key, "fields": {"summary": "stub"}}


class FakeEmailSender:
    """Callable object conforming to EmailSenderProtocol."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    async def __call__(self, to: str, subject: str, body: str) -> None:
        self.sent.append((to, subject, body))


class FakeKnowledgeMound:
    """Structurally conforming Knowledge Mound search interface."""

    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self._items = items or []

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not query:
            return []
        return self._items[:limit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def confluence() -> FakeConfluence:
    return FakeConfluence({"page-1": "Hello Confluence"})


@pytest.fixture
def github() -> FakeGitHub:
    return FakeGitHub()


@pytest.fixture
def jira() -> FakeJira:
    return FakeJira()


@pytest.fixture
def email_sender() -> FakeEmailSender:
    return FakeEmailSender()


@pytest.fixture
def knowledge_mound() -> FakeKnowledgeMound:
    return FakeKnowledgeMound([{"id": i, "text": f"doc-{i}"} for i in range(25)])


# ---------------------------------------------------------------------------
# Protocol class properties
# ---------------------------------------------------------------------------

ALL_PROTOCOLS = [
    ConfluenceConnectorProtocol,
    GitHubConnectorProtocol,
    JiraConnectorProtocol,
    EmailSenderProtocol,
    KnowledgeMoundProtocol,
]


class TestProtocolDefinitions:
    """Properties of the protocol classes themselves."""

    @pytest.mark.parametrize("proto", ALL_PROTOCOLS)
    def test_is_protocol_subclass(self, proto: type) -> None:
        assert issubclass(proto, Protocol)  # type: ignore[arg-type]

    @pytest.mark.parametrize("proto", ALL_PROTOCOLS)
    def test_is_runtime_checkable(self, proto: type) -> None:
        # runtime_checkable protocols set this flag; isinstance() requires it
        assert getattr(proto, "_is_runtime_protocol", False) is True

    @pytest.mark.parametrize("proto", ALL_PROTOCOLS)
    def test_protocol_cannot_be_instantiated(self, proto: type) -> None:
        with pytest.raises(TypeError):
            proto()  # type: ignore[call-arg]

    @pytest.mark.parametrize("proto", ALL_PROTOCOLS)
    def test_protocol_has_docstring(self, proto: type) -> None:
        assert proto.__doc__, f"{proto.__name__} should have a docstring"

    def test_protocol_method_names(self) -> None:
        assert callable(ConfluenceConnectorProtocol.get_page_content)
        assert callable(GitHubConnectorProtocol.get_pr_content)
        assert callable(GitHubConnectorProtocol.get_issue_content)
        assert callable(JiraConnectorProtocol.get_issue)
        assert callable(KnowledgeMoundProtocol.search)

    def test_protocol_methods_are_coroutine_functions(self) -> None:
        assert inspect.iscoroutinefunction(ConfluenceConnectorProtocol.get_page_content)
        assert inspect.iscoroutinefunction(GitHubConnectorProtocol.get_pr_content)
        assert inspect.iscoroutinefunction(GitHubConnectorProtocol.get_issue_content)
        assert inspect.iscoroutinefunction(JiraConnectorProtocol.get_issue)
        assert inspect.iscoroutinefunction(KnowledgeMoundProtocol.search)
        assert inspect.iscoroutinefunction(EmailSenderProtocol.__call__)

    def test_knowledge_mound_search_default_limit(self) -> None:
        sig = inspect.signature(KnowledgeMoundProtocol.search)
        assert sig.parameters["limit"].default == 10

    def test_email_sender_call_parameters(self) -> None:
        sig = inspect.signature(EmailSenderProtocol.__call__)
        names = [p for p in sig.parameters if p != "self"]
        assert names == ["to", "subject", "body"]

    def test_github_protocol_parameter_order(self) -> None:
        for method in (
            GitHubConnectorProtocol.get_pr_content,
            GitHubConnectorProtocol.get_issue_content,
        ):
            names = [p for p in inspect.signature(method).parameters if p != "self"]
            assert names == ["owner", "repo", "number"]


# ---------------------------------------------------------------------------
# Structural conformance (isinstance, positive)
# ---------------------------------------------------------------------------


class TestStructuralConformance:
    """Objects with matching method names satisfy the runtime checks."""

    def test_confluence_conformance(self, confluence: FakeConfluence) -> None:
        assert isinstance(confluence, ConfluenceConnectorProtocol)

    def test_github_conformance(self, github: FakeGitHub) -> None:
        assert isinstance(github, GitHubConnectorProtocol)

    def test_jira_conformance(self, jira: FakeJira) -> None:
        assert isinstance(jira, JiraConnectorProtocol)

    def test_email_sender_conformance(self, email_sender: FakeEmailSender) -> None:
        assert isinstance(email_sender, EmailSenderProtocol)

    def test_knowledge_mound_conformance(self, knowledge_mound: FakeKnowledgeMound) -> None:
        assert isinstance(knowledge_mound, KnowledgeMoundProtocol)

    def test_plain_async_function_satisfies_email_sender(self) -> None:
        async def send(to: str, subject: str, body: str) -> None:
            return None

        # Functions expose __call__, so the structural check passes
        assert isinstance(send, EmailSenderProtocol)

    def test_protocols_are_independent(
        self, confluence: FakeConfluence, github: FakeGitHub
    ) -> None:
        # A conforming Confluence connector is not a GitHub connector
        assert not isinstance(confluence, GitHubConnectorProtocol)
        assert not isinstance(github, ConfluenceConnectorProtocol)
        assert not isinstance(github, JiraConnectorProtocol)
        assert not isinstance(confluence, KnowledgeMoundProtocol)


# ---------------------------------------------------------------------------
# Structural non-conformance (isinstance, negative)
# ---------------------------------------------------------------------------


class TestNonConformance:
    """Objects missing required methods fail the runtime checks."""

    def test_empty_object_fails_all_method_protocols(self) -> None:
        class Empty:
            pass

        obj = Empty()
        assert not isinstance(obj, ConfluenceConnectorProtocol)
        assert not isinstance(obj, GitHubConnectorProtocol)
        assert not isinstance(obj, JiraConnectorProtocol)
        assert not isinstance(obj, KnowledgeMoundProtocol)

    def test_partial_github_implementation_fails(self) -> None:
        class OnlyPr:
            async def get_pr_content(self, owner: str, repo: str, number: int) -> str | None:
                return None

        assert not isinstance(OnlyPr(), GitHubConnectorProtocol)

    def test_non_callable_fails_email_sender(self) -> None:
        class NotCallable:
            pass

        assert not isinstance(NotCallable(), EmailSenderProtocol)

    def test_method_as_attribute_value_still_passes_runtime_check(self) -> None:
        # runtime_checkable only checks member presence, not callability
        # of the attribute or its signature; document that limitation.
        class AttrOnly:
            get_page_content = "not-a-method"

        assert isinstance(AttrOnly(), ConfluenceConnectorProtocol)

    def test_wrong_signature_still_passes_runtime_check(self) -> None:
        # isinstance() does not validate signatures, only names
        class WrongSig:
            async def get_issue(self) -> None:  # missing issue_key param
                return None

        assert isinstance(WrongSig(), JiraConnectorProtocol)


# ---------------------------------------------------------------------------
# Behavior of conforming implementations (async round trips)
# ---------------------------------------------------------------------------


class TestConformingBehavior:
    """Conforming fakes behave as the protocol contracts describe."""

    @pytest.mark.asyncio
    async def test_confluence_returns_content(self, confluence: FakeConfluence) -> None:
        assert await confluence.get_page_content("page-1") == "Hello Confluence"

    @pytest.mark.asyncio
    async def test_confluence_missing_page_returns_none(self, confluence: FakeConfluence) -> None:
        assert await confluence.get_page_content("missing") is None

    @pytest.mark.asyncio
    async def test_github_pr_and_issue_content(self, github: FakeGitHub) -> None:
        assert await github.get_pr_content("synaptent", "aragora", 42) == (
            "PR synaptent/aragora#42"
        )
        assert await github.get_issue_content("synaptent", "aragora", 7) == (
            "Issue synaptent/aragora#7"
        )

    @pytest.mark.asyncio
    async def test_github_invalid_number_returns_none(self, github: FakeGitHub) -> None:
        assert await github.get_pr_content("o", "r", 0) is None
        assert await github.get_issue_content("o", "r", -1) is None

    @pytest.mark.asyncio
    async def test_jira_returns_issue_dict(self, jira: FakeJira) -> None:
        issue = await jira.get_issue("ARA-123")
        assert issue is not None
        assert issue["key"] == "ARA-123"
        assert "fields" in issue

    @pytest.mark.asyncio
    async def test_jira_empty_key_returns_none(self, jira: FakeJira) -> None:
        assert await jira.get_issue("") is None

    @pytest.mark.asyncio
    async def test_email_sender_records_send(self, email_sender: FakeEmailSender) -> None:
        result = await email_sender("a@example.com", "subj", "body")
        assert result is None
        assert email_sender.sent == [("a@example.com", "subj", "body")]

    @pytest.mark.asyncio
    async def test_knowledge_mound_default_limit_applied(
        self, knowledge_mound: FakeKnowledgeMound
    ) -> None:
        results = await knowledge_mound.search("anything")
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_knowledge_mound_explicit_limit(
        self, knowledge_mound: FakeKnowledgeMound
    ) -> None:
        assert len(await knowledge_mound.search("q", limit=3)) == 3
        assert len(await knowledge_mound.search("q", limit=100)) == 25

    @pytest.mark.asyncio
    async def test_knowledge_mound_empty_query_returns_empty_list(
        self, knowledge_mound: FakeKnowledgeMound
    ) -> None:
        assert await knowledge_mound.search("") == []


# ---------------------------------------------------------------------------
# Explicit subclassing of protocols
# ---------------------------------------------------------------------------


class TestExplicitSubclassing:
    """Protocols can be used as explicit base classes."""

    def test_explicit_confluence_subclass(self) -> None:
        class Impl(ConfluenceConnectorProtocol):
            async def get_page_content(self, page_id: str) -> str | None:
                return page_id.upper()

        impl = Impl()
        assert isinstance(impl, ConfluenceConnectorProtocol)

    @pytest.mark.asyncio
    async def test_explicit_subclass_method_works(self) -> None:
        class Impl(KnowledgeMoundProtocol):
            async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
                return [{"query": query, "limit": limit}]

        results = await Impl().search("x", limit=2)
        assert results == [{"query": "x", "limit": 2}]


# ---------------------------------------------------------------------------
# RecommendAgentsFunc type alias
# ---------------------------------------------------------------------------


class TestRecommendAgentsFunc:
    """The RecommendAgentsFunc alias is Callable[[str], Any]."""

    def test_alias_origin_is_callable(self) -> None:
        origin = get_origin(RecommendAgentsFunc)
        assert origin in (
            Callable,
            getattr(__import__("collections.abc", fromlist=["Callable"]), "Callable"),
        )

    def test_alias_argument_types(self) -> None:
        args = get_args(RecommendAgentsFunc)
        assert args[0] == [str]
        assert args[1] is Any

    def test_matching_function_assignable(self) -> None:
        def recommend(task: str) -> list[str]:
            return [task]

        func: RecommendAgentsFunc = recommend
        assert func("debate") == ["debate"]

    def test_lambda_assignable(self) -> None:
        func: RecommendAgentsFunc = lambda task: {"task": task}
        assert func("triage") == {"task": "triage"}


# ---------------------------------------------------------------------------
# Package re-export surface (real consumption point of these protocols)
# ---------------------------------------------------------------------------


class TestPackageReExports:
    """The orchestration package re-exports the protocols unchanged."""

    def test_package_reexports_same_objects(self) -> None:
        import aragora.server.handlers.orchestration as pkg

        assert pkg.ConfluenceConnectorProtocol is ConfluenceConnectorProtocol
        assert pkg.GitHubConnectorProtocol is GitHubConnectorProtocol
        assert pkg.JiraConnectorProtocol is JiraConnectorProtocol
        assert pkg.EmailSenderProtocol is EmailSenderProtocol
        assert pkg.KnowledgeMoundProtocol is KnowledgeMoundProtocol
        assert pkg.RecommendAgentsFunc is RecommendAgentsFunc

    def test_fakes_satisfy_reexported_protocols(
        self,
        confluence: FakeConfluence,
        github: FakeGitHub,
        jira: FakeJira,
        email_sender: FakeEmailSender,
        knowledge_mound: FakeKnowledgeMound,
    ) -> None:
        import aragora.server.handlers.orchestration as pkg

        assert isinstance(confluence, pkg.ConfluenceConnectorProtocol)
        assert isinstance(github, pkg.GitHubConnectorProtocol)
        assert isinstance(jira, pkg.JiraConnectorProtocol)
        assert isinstance(email_sender, pkg.EmailSenderProtocol)
        assert isinstance(knowledge_mound, pkg.KnowledgeMoundProtocol)


# ---------------------------------------------------------------------------
# Error propagation through protocol-typed call sites
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    """Exceptions raised by implementations propagate through protocol calls."""

    @pytest.mark.asyncio
    async def test_confluence_connection_error_propagates(self) -> None:
        class FailingConfluence:
            async def get_page_content(self, page_id: str) -> str | None:
                raise ConnectionError("confluence unreachable")

        connector: ConfluenceConnectorProtocol = FailingConfluence()
        assert isinstance(connector, ConfluenceConnectorProtocol)
        with pytest.raises(ConnectionError, match="confluence unreachable"):
            await connector.get_page_content("page-1")

    @pytest.mark.asyncio
    async def test_email_sender_error_propagates(self) -> None:
        class FailingSender:
            async def __call__(self, to: str, subject: str, body: str) -> None:
                raise ConnectionError("smtp down")

        sender: EmailSenderProtocol = FailingSender()
        assert isinstance(sender, EmailSenderProtocol)
        with pytest.raises(ConnectionError, match="smtp down"):
            await sender("a@example.com", "s", "b")

    @pytest.mark.asyncio
    async def test_knowledge_mound_timeout_error_propagates(self) -> None:
        class FailingMound:
            async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
                raise TimeoutError("search timed out")

        mound: KnowledgeMoundProtocol = FailingMound()
        assert isinstance(mound, KnowledgeMoundProtocol)
        with pytest.raises(TimeoutError, match="search timed out"):
            await mound.search("query")


# ---------------------------------------------------------------------------
# Concurrency safety of conforming implementations
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Concurrent protocol calls do not corrupt shared state."""

    @pytest.mark.asyncio
    async def test_concurrent_knowledge_mound_searches(
        self, knowledge_mound: FakeKnowledgeMound
    ) -> None:
        results = await asyncio.gather(
            *(knowledge_mound.search(f"query-{i}", limit=5) for i in range(10))
        )
        assert len(results) == 10
        for batch in results:
            assert len(batch) == 5
        # Underlying store unchanged after concurrent reads
        assert len(knowledge_mound._items) == 25

    @pytest.mark.asyncio
    async def test_concurrent_email_sends_record_all(self, email_sender: FakeEmailSender) -> None:
        await asyncio.gather(
            *(email_sender(f"user{i}@example.com", f"subj-{i}", "body") for i in range(10))
        )
        assert len(email_sender.sent) == 10
        assert {to for to, _, _ in email_sender.sent} == {f"user{i}@example.com" for i in range(10)}
