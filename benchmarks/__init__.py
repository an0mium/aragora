"""
Aragora Performance Benchmarks.

This package contains benchmarks for measuring performance of core Aragora components.

Benchmarks:
- memory_tiers: Measure latency across fast/medium/slow/glacial memory tiers
- debate_throughput: Measure concurrent debate handling and agent response times
- api_endpoints: Measure API endpoint latency against SLO targets
- gauntlet_evaluation: Measure Gauntlet attack evaluation performance

Usage:
    # Run memory tier benchmarks
    python -m benchmarks.memory_tiers

    # Run debate throughput benchmarks
    python -m benchmarks.debate_throughput

    # Run API endpoint benchmarks
    python -m benchmarks.api_endpoints

    # Run all benchmarks
    python -m pytest benchmarks/ -v
"""

__all__ = ["memory_tiers", "debate_throughput", "api_endpoints", "gauntlet_evaluation"]
