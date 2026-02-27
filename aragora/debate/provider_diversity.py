"""Provider Diversity Filter for debate team selection.

Ensures multi-provider representation in debate teams to prevent
single-provider groupthink. Composes with existing TeamSelector
as a post-selection filter.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field


# Model name → provider mapping patterns
PROVIDER_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "anthropic": [re.compile(r"claude", re.I)],
    "openai": [re.compile(r"gpt|o1|o3|chatgpt", re.I)],
    "google": [re.compile(r"gemini|palm|bard", re.I)],
    "mistral": [re.compile(r"mistral|mixtral|codestral", re.I)],
    "xai": [re.compile(r"grok", re.I)],
    "meta": [re.compile(r"llama", re.I)],
    "deepseek": [re.compile(r"deepseek", re.I)],
    "cohere": [re.compile(r"command", re.I)],
    "alibaba": [re.compile(r"qwen", re.I)],
}


@dataclass
class DiversityReport:
    """Report on provider diversity in a debate team."""

    providers: dict[str, list[str]]  # provider → [agent_names]
    provider_count: int
    meets_minimum: bool
    swaps_made: list[tuple[str, str]]  # (removed, added)
    min_providers: int = 2


@dataclass
class AgentInfo:
    """Minimal agent info for diversity filtering."""

    name: str
    model: str
    score: float = 0.0
    provider: str = ""

    def __post_init__(self) -> None:
        if not self.provider:
            self.provider = detect_provider(self.model)


def detect_provider(model_name: str) -> str:
    """Detect provider from model name."""
    for provider, patterns in PROVIDER_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(model_name):
                return provider
    return "unknown"


@dataclass
class ProviderDiversityFilter:
    """Enforces minimum provider diversity in debate teams.

    Operates as a post-selection filter: after TeamSelector picks
    the best agents, this filter ensures at least min_providers
    different model providers are represented.
    """

    min_providers: int = 2
    available_alternatives: list[AgentInfo] = field(default_factory=list)

    def check(self, agents: list[AgentInfo]) -> DiversityReport:
        """Check provider diversity without modifying the team."""
        providers: dict[str, list[str]] = defaultdict(list)
        for agent in agents:
            providers[agent.provider].append(agent.name)

        return DiversityReport(
            providers=dict(providers),
            provider_count=len(providers),
            meets_minimum=len(providers) >= self.min_providers,
            swaps_made=[],
            min_providers=self.min_providers,
        )

    def enforce(
        self,
        agents: list[AgentInfo],
        alternatives: list[AgentInfo] | None = None,
    ) -> tuple[list[AgentInfo], DiversityReport]:
        """Enforce provider diversity, swapping agents if needed.

        Strategy: replace lowest-scoring agents from over-represented
        providers with highest-scoring alternatives from missing providers.

        Returns:
            Tuple of (possibly modified agent list, diversity report).
        """
        alt_pool = alternatives or self.available_alternatives
        providers: dict[str, list[AgentInfo]] = defaultdict(list)
        for agent in agents:
            providers[agent.provider].append(agent)

        swaps: list[tuple[str, str]] = []

        if len(providers) >= self.min_providers:
            return agents, self._make_report(agents, swaps)

        # Find providers not in current team
        current_providers = set(providers.keys())
        alt_by_provider: dict[str, list[AgentInfo]] = defaultdict(list)
        for alt in alt_pool:
            if alt.provider not in current_providers and alt.name not in {a.name for a in agents}:
                alt_by_provider[alt.provider].append(alt)

        # Sort alternatives by score descending
        for p in alt_by_provider:
            alt_by_provider[p].sort(key=lambda a: a.score, reverse=True)

        # Find over-represented provider (most agents)
        result = list(agents)
        needed = self.min_providers - len(current_providers)

        for _ in range(needed):
            if not alt_by_provider:
                break

            # Pick best alternative from any missing provider
            best_alt: AgentInfo | None = None
            best_provider = ""
            for p, alts in alt_by_provider.items():
                if alts and (best_alt is None or alts[0].score > best_alt.score):
                    best_alt = alts[0]
                    best_provider = p

            if best_alt is None:
                break

            # Find lowest-scoring agent from most-represented provider
            rep_counts: dict[str, int] = defaultdict(int)
            for a in result:
                rep_counts[a.provider] += 1

            over_rep = max(rep_counts, key=lambda p: rep_counts[p])
            if rep_counts[over_rep] <= 1:
                break  # Can't remove without eliminating provider

            # Remove lowest scorer from over-represented provider
            candidates = [a for a in result if a.provider == over_rep]
            candidates.sort(key=lambda a: a.score)
            to_remove = candidates[0]

            result.remove(to_remove)
            result.append(best_alt)
            swaps.append((to_remove.name, best_alt.name))

            # Update tracking
            alt_by_provider[best_provider].pop(0)
            if not alt_by_provider[best_provider]:
                del alt_by_provider[best_provider]
            current_providers.add(best_provider)

        return result, self._make_report(result, swaps)

    def _make_report(
        self, agents: list[AgentInfo], swaps: list[tuple[str, str]]
    ) -> DiversityReport:
        """Build diversity report from final agent list."""
        providers: dict[str, list[str]] = defaultdict(list)
        for agent in agents:
            providers[agent.provider].append(agent.name)

        return DiversityReport(
            providers=dict(providers),
            provider_count=len(providers),
            meets_minimum=len(providers) >= self.min_providers,
            swaps_made=swaps,
            min_providers=self.min_providers,
        )
