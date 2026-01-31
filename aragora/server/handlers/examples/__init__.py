"""
Example handlers demonstrating the typed base classes.

This module provides example implementations showing how to use the new
typed handler base classes introduced in base.py. These examples are
designed to:

1. Demonstrate best practices for handler implementation
2. Serve as templates for new handlers
3. Show migration patterns for existing handlers

Available Examples:
    ExampleTypedHandler: Basic TypedHandler usage with type annotations
    ExampleAuthenticatedHandler: Handler requiring authentication
    ExamplePermissionHandler: Handler with RBAC permission checking
    ExampleResourceHandler: RESTful resource CRUD handler
    ExampleAsyncHandler: Async handler for I/O operations

Usage:
    These examples can be used as templates. Copy and modify the relevant
    example for your use case:

        from aragora.server.handlers.examples import ExampleResourceHandler

        class MyResourceHandler(ExampleResourceHandler):
            RESOURCE_NAME = "my_resource"
            ROUTES = ["/api/v1/my_resources", "/api/v1/my_resources/*"]

            def _get_resource(self, resource_id, handler):
                # Your implementation
                ...

Testing:
    The typed base classes support dependency injection for testing:

        from aragora.server.handlers.base import TypedHandler
        from unittest.mock import Mock

        # Create handler with mocked dependencies
        mock_store = Mock()
        handler = MyHandler.with_dependencies(
            server_context,
            user_store=mock_store,
        )
"""

from .typed_example import ExampleTypedHandler
from .authenticated_example import ExampleAuthenticatedHandler
from .permission_example import ExamplePermissionHandler
from .resource_example import ExampleResourceHandler
from .async_example import ExampleAsyncHandler

__all__ = [
    "ExampleTypedHandler",
    "ExampleAuthenticatedHandler",
    "ExamplePermissionHandler",
    "ExampleResourceHandler",
    "ExampleAsyncHandler",
]
