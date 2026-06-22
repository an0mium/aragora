"""Shared fixtures for agent tests."""

import pytest

from aragora.agents.registry import AgentRegistry, register_all_agents
from aragora.resilience import reset_all_v2_circuit_breakers

# Populate the lazily-built agent registry once, before any test can clear it,
# and snapshot the full set of registered specs. register_all_agents() only
# triggers the @register decorators via module imports, so after a clear() the
# already-cached modules are not re-imported and the registry cannot be rebuilt
# for the rest of the process. Capturing the populated state here lets the
# autouse fixture below restore a complete registry between tests.
register_all_agents()
_AGENT_REGISTRY_GOLDEN = dict(AgentRegistry._registry)


@pytest.fixture(autouse=True)
def _restore_agent_registry():
    """Restore the shared ``AgentRegistry`` after every agent test.

    Registry tests call ``AgentRegistry.clear()``/``register()`` on the shared
    class-level ``_registry``. Because population is lazy and import-based, a
    cleared registry cannot be rebuilt via ``register_all_agents()`` within the
    same process, so a clear() in one test file otherwise leaves
    ``create_agent()`` broken (e.g. "Unknown agent type: demo") for every later
    test file. Restoring the captured registry after each test keeps it intact
    across files regardless of execution order.
    """
    yield
    AgentRegistry._registry.clear()
    AgentRegistry._registry.update(_AGENT_REGISTRY_GOLDEN)


@pytest.fixture(autouse=True)
def _reset_agent_circuit_breakers():
    """Reset the agent circuit-breaker registry around every agent test.

    Agents acquire breakers from the shared ``get_v2_circuit_breaker`` registry
    keyed by agent name, so a breaker tripped in one test (e.g. ollama/lm-studio
    connection-failure cases) stays open for later tests and raises an
    order-dependent ``AgentCircuitOpenError``. The suite-wide autouse reset only
    clears the v1 registry, which is a distinct dict, so the v2 registry must be
    reset separately here.
    """
    reset_all_v2_circuit_breakers()
    yield
    reset_all_v2_circuit_breakers()
