"""
Shared test fixtures for the Aragora test suite.

This package provides reusable fixtures and utilities for testing,
reducing duplication across conftest files.

Modules:
    auth: RBAC and authentication mocking
    api_mocks: OpenAI, Anthropic, HTTPX client mocks
    stores: Mock store implementations (ELO, storage, knowledge, etc.)
"""

from .auth import (
    create_mock_auth_context,
    create_admin_context,
    create_viewer_context,
    create_editor_context,
    MockAuthorizationContext,
    patch_rbac_decorators,
    patch_get_auth_context,
)

from .api_mocks import (
    MockOpenAIClient,
    MockAsyncOpenAIClient,
    MockAnthropicClient,
    MockAsyncAnthropicClient,
    MockHTTPXClient,
    MockAsyncHTTPXClient,
    MockHTTPXResponse,
    apply_api_mocks,
)

from .stores import (
    create_mock_debate_storage,
    create_mock_user_store,
    create_mock_elo_system,
    create_mock_knowledge_store,
    create_mock_workflow_store,
    create_mock_server_context,
    MockMetaStore,
    MockConnection,
    MockSemanticStore,
)

__all__ = [
    # Auth
    "create_mock_auth_context",
    "create_admin_context",
    "create_viewer_context",
    "create_editor_context",
    "MockAuthorizationContext",
    "patch_rbac_decorators",
    "patch_get_auth_context",
    # API Mocks
    "MockOpenAIClient",
    "MockAsyncOpenAIClient",
    "MockAnthropicClient",
    "MockAsyncAnthropicClient",
    "MockHTTPXClient",
    "MockAsyncHTTPXClient",
    "MockHTTPXResponse",
    "apply_api_mocks",
    # Stores
    "create_mock_debate_storage",
    "create_mock_user_store",
    "create_mock_elo_system",
    "create_mock_knowledge_store",
    "create_mock_workflow_store",
    "create_mock_server_context",
    "MockMetaStore",
    "MockConnection",
    "MockSemanticStore",
]
