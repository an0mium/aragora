"""
Chaos Engineering Test Suite.

Tests for system resilience under failure conditions:
- Circuit breaker behavior under load
- Agent failure recovery
- Memory tier fallback mechanisms

Run with extended timeout:
    pytest tests/chaos/ -v --timeout=300
"""
