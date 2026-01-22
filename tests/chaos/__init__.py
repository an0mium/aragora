"""
Chaos testing module for Aragora.

Provides infrastructure and tests for validating system resilience under:
- Agent failures and timeouts
- Memory/storage failures
- Network partitions and latency
- Circuit breaker behavior
- Concurrent load stress

Usage:
    pytest tests/chaos/ -v --timeout=120
    pytest tests/chaos/ -v -m "not slow"  # Skip slow tests
"""
