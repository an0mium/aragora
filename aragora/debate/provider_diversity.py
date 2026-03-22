"""Provider Diversity Filter for debate team selection.

Ensures multi-provider representation in debate teams to prevent
single-provider groupthink. Composes with existing TeamSelector
as a post-selection filter.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

DEFAULT_RECEIPT_AGENT_SAMPLE_LIMIT = 3
DEFAULT_RECEIPT_SWAP_LIMIT = 5


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
class DiversityBenchmark:
    """Compact benchmark summary for provider-diversity enforcement."""

    path: str
    roster_size: int
    alternative_pool_size: int
    iterations: int
    average_runtime_ms: float
    max_runtime_ms: float
    provider_shortfall: int
    swap_budget: int
    swaps_made: int
    receipt_payload_bytes: int

    def to_dict(self) -> dict[str, int | float | str]:
        """Serialize benchmark data for receipts and diagnostics."""
        return {
            "path": self.path,
            "roster_size": self.roster_size,
            "alternative_pool_size": self.alternative_pool_size,
            "iterations": self.iterations,
            "average_runtime_ms": round(self.average_runtime_ms, 4),
            "max_runtime_ms": round(self.max_runtime_ms, 4),
            "provider_shortfall": self.provider_shortfall,
            "swap_budget": self.swap_budget,
            "swaps_made": self.swaps_made,
            "receipt_payload_bytes": self.receipt_payload_bytes,
        }


@dataclass
class DiversityReport:
    """Report on provider diversity in a debate team."""

    providers: dict[str, list[str]]  # provider → [agent_names]
    provider_count: int
    meets_minimum: bool
    swaps_made: list[tuple[str, str]]  # (removed, added)
    min_providers: int = 2
    roster_size: int = 0
    alternative_pool_size: int = 0
    provider_shortfall: int = 0
    swap_budget: int = 0
    runtime_ms: float = 0.0
    receipt_payload_bytes: int = 0
    benchmark: DiversityBenchmark | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the full diversity report."""
        result: dict[str, object] = {
            "providers": {
                provider: list(agent_names) for provider, agent_names in self.providers.items()
            },
            "provider_count": self.provider_count,
            "meets_minimum": self.meets_minimum,
            "swaps_made": [
                {"removed": removed, "added": added} for removed, added in self.swaps_made
            ],
            "min_providers": self.min_providers,
            "roster_size": self.roster_size,
            "alternative_pool_size": self.alternative_pool_size,
            "provider_shortfall": self.provider_shortfall,
            "swap_budget": self.swap_budget,
            "runtime_ms": round(self.runtime_ms, 4),
            "receipt_payload_bytes": self.receipt_payload_bytes,
        }
        if self.benchmark is not None:
            result["benchmark"] = self.benchmark.to_dict()
        return result

    def to_receipt_payload(
        self,
        *,
        max_agents_per_provider: int = DEFAULT_RECEIPT_AGENT_SAMPLE_LIMIT,
        max_swaps: int = DEFAULT_RECEIPT_SWAP_LIMIT,
    ) -> dict[str, object]:
        """Serialize a bounded receipt payload for large rosters."""
        provider_samples: dict[str, dict[str, object]] = {}
        sample_limit = max(max_agents_per_provider, 0)
        for provider in sorted(self.providers):
            agent_names = list(self.providers[provider])
            sampled_agents = agent_names[:sample_limit]
            provider_samples[provider] = {
                "count": len(agent_names),
                "sample_agents": sampled_agents,
                "truncated_agents": max(0, len(agent_names) - len(sampled_agents)),
            }

        swap_limit = max(max_swaps, 0)
        sampled_swaps = self.swaps_made[:swap_limit]
        payload: dict[str, object] = {
            "provider_count": self.provider_count,
            "meets_minimum": self.meets_minimum,
            "min_providers": self.min_providers,
            "roster_size": self.roster_size,
            "alternative_pool_size": self.alternative_pool_size,
            "provider_shortfall": self.provider_shortfall,
            "swap_budget": self.swap_budget,
            "swaps_made": [
                {"removed": removed, "added": added} for removed, added in sampled_swaps
            ],
            "swaps_truncated": max(0, len(self.swaps_made) - len(sampled_swaps)),
            "runtime_ms": round(self.runtime_ms, 4),
            "providers": provider_samples,
        }
        if self.benchmark is not None:
            payload["benchmark"] = self.benchmark.to_dict()
        return payload

    def to_runtime_summary(self) -> dict[str, object]:
        """Serialize the bounded runtime summary for observability receipts."""
        summary: dict[str, object] = {
            "path": "provider_diversity_filter",
            "roster_size": self.roster_size,
            "provider_count": self.provider_count,
            "min_providers": self.min_providers,
            "provider_shortfall": self.provider_shortfall,
            "swap_budget": self.swap_budget,
            "swaps_made": len(self.swaps_made),
            "runtime_ms": round(self.runtime_ms, 4),
            "receipt_payload_bytes": self.receipt_payload_bytes,
        }
        if self.benchmark is not None:
            summary["benchmark"] = self.benchmark.to_dict()
        return summary

    def estimate_receipt_payload_bytes(
        self,
        *,
        max_agents_per_provider: int = DEFAULT_RECEIPT_AGENT_SAMPLE_LIMIT,
        max_swaps: int = DEFAULT_RECEIPT_SWAP_LIMIT,
    ) -> int:
        """Estimate serialized receipt payload size in bytes."""
        payload = self.to_receipt_payload(
            max_agents_per_provider=max_agents_per_provider,
            max_swaps=max_swaps,
        )
        return len(json.dumps(payload, sort_keys=True).encode("utf-8"))


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
        started_at = time.perf_counter()
        providers: dict[str, list[str]] = defaultdict(list)
        for agent in agents:
            providers[agent.provider].append(agent.name)

        runtime_ms = (time.perf_counter() - started_at) * 1000
        report = DiversityReport(
            providers=dict(providers),
            provider_count=len(providers),
            meets_minimum=len(providers) >= self.min_providers,
            swaps_made=[],
            min_providers=self.min_providers,
            roster_size=len(agents),
            provider_shortfall=max(0, self.min_providers - len(providers)),
            swap_budget=max(0, self.min_providers - len(providers)),
            runtime_ms=runtime_ms,
        )
        report.receipt_payload_bytes = report.estimate_receipt_payload_bytes()
        return report

    def enforce(
        self,
        agents: list[AgentInfo],
        alternatives: list[AgentInfo] | None = None,
        benchmark_iterations: int = 0,
    ) -> tuple[list[AgentInfo], DiversityReport]:
        """Enforce provider diversity, swapping agents if needed.

        Strategy: replace lowest-scoring agents from over-represented
        providers with highest-scoring alternatives from missing providers.

        Returns:
            Tuple of (possibly modified agent list, diversity report).
        """
        started_at = time.perf_counter()
        alt_pool = list(self.available_alternatives if alternatives is None else alternatives)
        providers: dict[str, list[AgentInfo]] = defaultdict(list)
        for agent in agents:
            providers[agent.provider].append(agent)

        swaps: list[tuple[str, str]] = []
        provider_shortfall = max(0, self.min_providers - len(providers))

        if len(providers) >= self.min_providers:
            report = self._make_report(
                agents,
                swaps,
                alternative_pool_size=len(alt_pool),
                swap_budget=provider_shortfall,
                runtime_ms=(time.perf_counter() - started_at) * 1000,
            )
            if benchmark_iterations > 0:
                report.benchmark = self.benchmark(
                    agents,
                    alternatives=alt_pool,
                    iterations=benchmark_iterations,
                )
                report.receipt_payload_bytes = report.estimate_receipt_payload_bytes()
            return agents, report

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

        report = self._make_report(
            result,
            swaps,
            alternative_pool_size=len(alt_pool),
            swap_budget=provider_shortfall,
            runtime_ms=(time.perf_counter() - started_at) * 1000,
        )
        if benchmark_iterations > 0:
            report.benchmark = self.benchmark(
                agents,
                alternatives=alt_pool,
                iterations=benchmark_iterations,
            )
            report.receipt_payload_bytes = report.estimate_receipt_payload_bytes()
        return result, report

    def benchmark(
        self,
        agents: list[AgentInfo],
        alternatives: list[AgentInfo] | None = None,
        *,
        iterations: int = 5,
    ) -> DiversityBenchmark:
        """Benchmark provider-diversity enforcement for a roster."""
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        alt_pool = list(self.available_alternatives if alternatives is None else alternatives)
        runtimes: list[float] = []
        final_report: DiversityReport | None = None

        for _ in range(iterations):
            _, final_report = self.enforce(list(agents), alternatives=list(alt_pool))
            runtimes.append(final_report.runtime_ms)

        if final_report is None:
            raise RuntimeError("provider diversity benchmark did not produce a report")
        return DiversityBenchmark(
            path="provider_diversity_filter",
            roster_size=len(agents),
            alternative_pool_size=len(alt_pool),
            iterations=iterations,
            average_runtime_ms=sum(runtimes) / len(runtimes),
            max_runtime_ms=max(runtimes),
            provider_shortfall=final_report.provider_shortfall,
            swap_budget=final_report.swap_budget,
            swaps_made=len(final_report.swaps_made),
            receipt_payload_bytes=final_report.receipt_payload_bytes,
        )

    def _make_report(
        self,
        agents: list[AgentInfo],
        swaps: list[tuple[str, str]],
        *,
        alternative_pool_size: int = 0,
        swap_budget: int = 0,
        runtime_ms: float = 0.0,
    ) -> DiversityReport:
        """Build diversity report from final agent list."""
        providers: dict[str, list[str]] = defaultdict(list)
        for agent in agents:
            providers[agent.provider].append(agent.name)

        report = DiversityReport(
            providers=dict(providers),
            provider_count=len(providers),
            meets_minimum=len(providers) >= self.min_providers,
            swaps_made=swaps,
            min_providers=self.min_providers,
            roster_size=len(agents),
            alternative_pool_size=alternative_pool_size,
            provider_shortfall=max(0, self.min_providers - len(providers)),
            swap_budget=swap_budget,
            runtime_ms=runtime_ms,
        )
        report.receipt_payload_bytes = report.estimate_receipt_payload_bytes()
        return report
