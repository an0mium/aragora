"""
Gateway integration test fixtures.

Provides reusable fixtures for testing Secure Gateway components including
external framework servers, credential vaults, policy engines, and tenant isolation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest


class MockExternalFrameworkServer:
    """Mock server simulating an external framework (CrewAI, AutoGen, etc.).

    This class provides a mock HTTP server that can be configured to simulate
    various external agent framework behaviors for integration testing.
    """

    def __init__(self) -> None:
        """Initialize the mock external framework server."""
        self.requests: list[dict] = []
        self.responses: dict[str, Any] = {}
        self._healthy = True

    def set_healthy(self, healthy: bool) -> None:
        """Set the health status of the mock server.

        Args:
            healthy: Whether the server should report as healthy.
        """
        self._healthy = healthy

    def set_response(self, path: str, response: Any) -> None:
        """Configure a response for a specific path.

        Args:
            path: The URL path to configure.
            response: The response to return for requests to this path.
        """
        self.responses[path] = response

    def get_requests(self, path: str | None = None) -> list[dict]:
        """Get recorded requests, optionally filtered by path.

        Args:
            path: Optional path to filter requests by.

        Returns:
            List of request dictionaries matching the filter.
        """
        if path is None:
            return self.requests.copy()
        return [r for r in self.requests if r.get("path") == path]

    async def handle_request(self, path: str, method: str, body: dict | None = None) -> dict:
        """Handle an incoming request to the mock server.

        Args:
            path: The URL path of the request.
            method: The HTTP method (GET, POST, etc.).
            body: Optional request body.

        Returns:
            The configured response or a default response.

        Raises:
            Exception: If the server is unhealthy.
        """
        self.requests.append({"path": path, "method": method, "body": body})

        if not self._healthy:
            raise Exception("Server unhealthy")

        if path in self.responses:
            return self.responses[path]

        return {"status": "ok", "path": path}


class MockCredentialVault:
    """Mock credential vault for testing.

    Simulates a secure credential storage system with seal/unseal functionality.
    """

    def __init__(self) -> None:
        """Initialize the mock credential vault."""
        self._credentials: dict[str, dict] = {}
        self._sealed = False

    def store(self, credential_id: str, value: str, metadata: dict | None = None) -> None:
        """Store a credential in the vault.

        Args:
            credential_id: Unique identifier for the credential.
            value: The credential value to store.
            metadata: Optional metadata about the credential.

        Raises:
            RuntimeError: If the vault is sealed.
        """
        if self._sealed:
            raise RuntimeError("Vault is sealed")
        self._credentials[credential_id] = {
            "value": value,
            "metadata": metadata or {},
        }

    def retrieve(self, credential_id: str) -> str | None:
        """Retrieve a credential from the vault.

        Args:
            credential_id: The identifier of the credential to retrieve.

        Returns:
            The credential value, or None if not found.

        Raises:
            RuntimeError: If the vault is sealed.
        """
        if self._sealed:
            raise RuntimeError("Vault is sealed")
        cred = self._credentials.get(credential_id)
        return cred["value"] if cred else None

    def delete(self, credential_id: str) -> bool:
        """Delete a credential from the vault.

        Args:
            credential_id: The identifier of the credential to delete.

        Returns:
            True if the credential was deleted, False if not found.

        Raises:
            RuntimeError: If the vault is sealed.
        """
        if self._sealed:
            raise RuntimeError("Vault is sealed")
        if credential_id in self._credentials:
            del self._credentials[credential_id]
            return True
        return False

    def seal(self) -> None:
        """Seal the vault, preventing further operations."""
        self._sealed = True

    def unseal(self) -> None:
        """Unseal the vault, allowing operations."""
        self._sealed = False

    @property
    def is_sealed(self) -> bool:
        """Check if the vault is sealed.

        Returns:
            True if the vault is sealed.
        """
        return self._sealed


class MockPolicyEngine:
    """Mock policy engine for testing RBAC.

    Provides a simple permission-based access control system for testing.
    """

    def __init__(self) -> None:
        """Initialize the mock policy engine."""
        self._permissions: dict[str, set[str]] = {}  # user_id -> permissions

    def grant(self, user_id: str, *permissions: str) -> None:
        """Grant permissions to a user.

        Args:
            user_id: The user to grant permissions to.
            *permissions: Variable number of permission strings to grant.
        """
        if user_id not in self._permissions:
            self._permissions[user_id] = set()
        self._permissions[user_id].update(permissions)

    def revoke(self, user_id: str, permission: str) -> None:
        """Revoke a permission from a user.

        Args:
            user_id: The user to revoke the permission from.
            permission: The permission to revoke.
        """
        if user_id in self._permissions:
            self._permissions[user_id].discard(permission)

    def check(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission.

        Args:
            user_id: The user to check.
            permission: The permission to check for.

        Returns:
            True if the user has the permission.
        """
        return permission in self._permissions.get(user_id, set())


@dataclass
class TenantContext:
    """Context representing a tenant for testing tenant isolation.

    Attributes:
        tenant_id: Unique identifier for the tenant.
        user_id: Unique identifier for the user within the tenant.
        permissions: List of permissions granted to the user.
        quotas: Resource quotas for the tenant.
    """

    tenant_id: str
    user_id: str
    permissions: list[str]
    quotas: dict[str, int] = field(default_factory=dict)


class MockAgent:
    """Mock agent for testing.

    Provides a basic agent implementation that can be used in tests.
    """

    def __init__(self, name: str = "mock-agent", proposal: str = "Mock proposal") -> None:
        """Initialize the mock agent.

        Args:
            name: The name of the agent.
            proposal: The proposal text the agent will generate.
        """
        self.name = name
        self._proposal = proposal
        self._healthy = True

    async def is_available(self) -> bool:
        """Check if the agent is available.

        Returns:
            True if the agent is healthy and available.
        """
        return self._healthy

    async def generate(self, task: str, context: list | None = None) -> str:
        """Generate a proposal for a task.

        Args:
            task: The task description.
            context: Optional context for the task.

        Returns:
            The agent's proposal.
        """
        return self._proposal

    async def critique(self, proposal: str, task: str = "") -> str:
        """Critique a proposal.

        Args:
            proposal: The proposal to critique.
            task: Optional task description for context.

        Returns:
            A critique of the proposal.
        """
        return f"Critique of: {proposal[:50]}..."

    async def vote(self, proposals: list, task: str = "") -> dict:
        """Vote on proposals.

        Args:
            proposals: List of proposals to vote on.
            task: Optional task description for context.

        Returns:
            A vote dictionary with rankings.
        """
        return {"ranking": list(range(len(proposals))), "scores": [1.0] * len(proposals)}


class FailingAgent(MockAgent):
    """Agent that always fails.

    Used for testing error handling and resilience patterns.
    """

    async def generate(self, task: str, context: list | None = None) -> str:
        """Attempt to generate a proposal, but always fail.

        Args:
            task: The task description.
            context: Optional context for the task.

        Raises:
            Exception: Always raises an exception.
        """
        raise Exception("Agent failure")

    async def critique(self, proposal: str, task: str = "") -> str:
        """Attempt to critique a proposal, but always fail.

        Args:
            proposal: The proposal to critique.
            task: Optional task description.

        Raises:
            Exception: Always raises an exception.
        """
        raise Exception("Agent failure")

    async def vote(self, proposals: list, task: str = "") -> dict:
        """Attempt to vote on proposals, but always fail.

        Args:
            proposals: List of proposals to vote on.
            task: Optional task description.

        Raises:
            Exception: Always raises an exception.
        """
        raise Exception("Agent failure")


class SlowAgent(MockAgent):
    """Agent with configurable delay.

    Used for testing timeout handling and async coordination.
    """

    def __init__(
        self, name: str = "slow-agent", delay: float = 1.0, proposal: str = "Slow proposal"
    ) -> None:
        """Initialize the slow agent.

        Args:
            name: The name of the agent.
            delay: The delay in seconds before responding.
            proposal: The proposal text the agent will generate.
        """
        super().__init__(name, proposal)
        self._delay = delay

    async def generate(self, task: str, context: list | None = None) -> str:
        """Generate a proposal after a delay.

        Args:
            task: The task description.
            context: Optional context for the task.

        Returns:
            The agent's proposal after the configured delay.
        """
        await asyncio.sleep(self._delay)
        return self._proposal

    async def critique(self, proposal: str, task: str = "") -> str:
        """Critique a proposal after a delay.

        Args:
            proposal: The proposal to critique.
            task: Optional task description.

        Returns:
            A critique after the configured delay.
        """
        await asyncio.sleep(self._delay)
        return f"Slow critique of: {proposal[:50]}..."

    async def vote(self, proposals: list, task: str = "") -> dict:
        """Vote on proposals after a delay.

        Args:
            proposals: List of proposals to vote on.
            task: Optional task description.

        Returns:
            A vote dictionary after the configured delay.
        """
        await asyncio.sleep(self._delay)
        return {"ranking": list(range(len(proposals))), "scores": [1.0] * len(proposals)}


def register_external_agent(
    ctx: dict,
    name: str = "test-agent",
    framework: str = "crewai",
    base_url: str = "https://example.com/api",
) -> dict:
    """Helper to register an external agent in server context.

    Args:
        ctx: The server context dictionary.
        name: The name of the agent to register.
        framework: The framework type (crewai, autogen, etc.).
        base_url: The base URL of the external agent.

    Returns:
        The registered agent information dictionary.
    """
    agent_info = {
        "name": name,
        "framework_type": framework,
        "base_url": base_url,
        "timeout": 30,
        "status": "registered",
    }
    ctx.setdefault("external_agents", {})[name] = agent_info
    return agent_info


# --- Pytest Fixtures ---


@pytest.fixture
def mock_external_server() -> MockExternalFrameworkServer:
    """Provide a mock external framework server.

    Returns:
        A configured MockExternalFrameworkServer instance.
    """
    return MockExternalFrameworkServer()


@pytest.fixture
def mock_credential_vault() -> MockCredentialVault:
    """Provide a mock credential vault.

    Returns:
        A configured MockCredentialVault instance.
    """
    return MockCredentialVault()


@pytest.fixture
def mock_policy_engine() -> MockPolicyEngine:
    """Provide a mock policy engine.

    Returns:
        A configured MockPolicyEngine instance.
    """
    return MockPolicyEngine()


@pytest.fixture
def tenant_context() -> TenantContext:
    """Provide a default tenant context for testing.

    Returns:
        A TenantContext with standard test permissions.
    """
    return TenantContext(
        tenant_id="test-tenant",
        user_id="test-user",
        permissions=["gateway:agent.create", "gateway:agent.read"],
        quotas={"max_agents": 10, "max_credentials": 100},
    )


@pytest.fixture
def alt_tenant_context() -> TenantContext:
    """Provide an alternative tenant context for isolation tests.

    Returns:
        A TenantContext representing a different tenant with limited permissions.
    """
    return TenantContext(
        tenant_id="alt-tenant",
        user_id="alt-user",
        permissions=["gateway:agent.read"],
        quotas={"max_agents": 5},
    )


@pytest.fixture
def mock_agent() -> MockAgent:
    """Provide a mock agent for testing.

    Returns:
        A configured MockAgent instance.
    """
    return MockAgent()


@pytest.fixture
def failing_agent() -> FailingAgent:
    """Provide a failing agent for error handling tests.

    Returns:
        A FailingAgent instance that always raises exceptions.
    """
    return FailingAgent("failing-agent")


@pytest.fixture
def slow_agent() -> SlowAgent:
    """Provide a slow agent for timeout tests.

    Returns:
        A SlowAgent instance with a 0.5 second delay.
    """
    return SlowAgent("slow-agent", delay=0.5)


@pytest.fixture
def gateway_server_context(
    mock_credential_vault: MockCredentialVault,
    mock_policy_engine: MockPolicyEngine,
) -> dict:
    """Provide a server context with gateway components.

    Args:
        mock_credential_vault: The mock credential vault fixture.
        mock_policy_engine: The mock policy engine fixture.

    Returns:
        A dictionary containing the gateway server context.
    """
    return {
        "external_agents": {},
        "credential_vault": mock_credential_vault,
        "policy_engine": mock_policy_engine,
    }
