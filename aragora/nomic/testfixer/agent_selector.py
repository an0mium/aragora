"""ELO-based agent selection for TestFixer batch processing.

Selects the best-performing agents for each failure category based on
historical ELO rankings, falling back to configured defaults when
ranking data is unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aragora.nomic.testfixer.analyzer import FailureCategory

if TYPE_CHECKING:
    from aragora.nomic.testfixer.proposer import CodeGenerator

logger = logging.getLogger(__name__)

# Map failure categories to ELO domain strings
CATEGORY_TO_DOMAIN: dict[FailureCategory, str] = {
    FailureCategory.TEST_ASSERTION: "test_assertion",
    FailureCategory.TEST_SETUP: "test_setup",
    FailureCategory.TEST_ASYNC: "test_async",
    FailureCategory.TEST_MOCK: "test_mock",
    FailureCategory.TEST_IMPORT: "test_import",
    FailureCategory.IMPL_BUG: "impl_bug",
    FailureCategory.IMPL_MISSING: "impl_missing",
    FailureCategory.IMPL_TYPE: "impl_type",
    FailureCategory.IMPL_API_CHANGE: "impl_api_change",
    FailureCategory.ENV_DEPENDENCY: "env_dependency",
    FailureCategory.ENV_CONFIG: "env_config",
    FailureCategory.ENV_RESOURCE: "env_resource",
    FailureCategory.RACE_CONDITION: "race_condition",
    FailureCategory.FLAKY: "flaky",
    FailureCategory.UNKNOWN: "general",
}

DEFAULT_FALLBACK_AGENTS = ["anthropic-api", "openai-api"]


class AgentSelector:
    """Select agents based on ELO rankings for a given failure category.

    Uses the existing EloSystem to find top-performing agents for the
    domain corresponding to a failure category. Falls back to a
    configurable list of default agents when ELO data is unavailable.
    """

    def __init__(
        self,
        elo_system: Any | None = None,
        fallback_agents: list[str] | None = None,
    ) -> None:
        self._elo = elo_system
        self._fallback_agents = fallback_agents or list(DEFAULT_FALLBACK_AGENTS)

    def select_agents_for_category(
        self,
        category: FailureCategory,
        limit: int = 3,
    ) -> list[CodeGenerator]:
        """Select the best agents for a failure category.

        Args:
            category: The failure category to select agents for.
            limit: Maximum number of agents to return.

        Returns:
            List of CodeGenerator instances for the selected agents.
        """
        domain = CATEGORY_TO_DOMAIN.get(category, "general")
        agent_types = self._get_agent_types(domain, limit)
        return self._create_generators(agent_types)

    def _get_agent_types(self, domain: str, limit: int) -> list[str]:
        """Get agent type names ranked by ELO for domain."""
        if self._elo is not None:
            try:
                ratings = self._elo.get_top_agents_for_domain(domain, limit=limit)
                if ratings:
                    names = [r.agent_name for r in ratings]
                    logger.info(
                        "agent_selector.elo domain=%s agents=%s",
                        domain,
                        names,
                    )
                    return names[:limit]
            except Exception as exc:
                logger.warning("agent_selector.elo_error domain=%s error=%s", domain, exc)

        # Fallback
        result = self._fallback_agents[:limit]
        logger.info("agent_selector.fallback domain=%s agents=%s", domain, result)
        return result

    @staticmethod
    def _create_generators(agent_types: list[str]) -> list[CodeGenerator]:
        """Create CodeGenerator instances from agent type names."""
        from aragora.nomic.testfixer.generators import (
            AgentCodeGenerator,
            AgentGeneratorConfig,
        )

        generators: list[CodeGenerator] = []
        for agent_type in agent_types:
            try:
                cfg = AgentGeneratorConfig(agent_type=agent_type)
                generators.append(AgentCodeGenerator(cfg))
            except Exception as exc:
                logger.warning(
                    "agent_selector.create_generator_error agent=%s error=%s",
                    agent_type,
                    exc,
                )
        return generators
