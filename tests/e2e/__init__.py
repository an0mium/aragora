"""
End-to-end integration tests for Aragora.

Tests full-stack scenarios including:
- Connector sync workflows
- Debate lifecycle (including 50+ rounds)
- Multi-tenant isolation
- Knowledge system integration

Usage:
    pytest tests/e2e/ -v --tb=short
    pytest tests/e2e/ -k "connector" -v
    pytest tests/e2e/ -k "debate" -v
    pytest tests/e2e/ -k "tenant" -v
"""
