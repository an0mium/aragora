"""
GraphQL HTTP Handler for Aragora API.

Provides the HTTP endpoint for GraphQL queries and mutations at /graphql.
Includes optional GraphiQL playground for development.

Usage:
    from aragora.server.graphql.handler import GraphQLHandler

    # Initialize handler with server context
    handler = GraphQLHandler(server_context)

    # Check if a path is handled
    if handler.can_handle("/graphql"):
        result = handler.handle(path, query_params, http_handler)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit

from .resolvers import (
    MUTATION_RESOLVERS,
    QUERY_RESOLVERS,
    ResolverContext,
    ResolverResult,
)
from .schema import (
    GraphQLParser,
    GraphQLValidator,
    OperationType,
    SCHEMA_SDL,
)

if TYPE_CHECKING:
    from aragora.server.handlers.base import ServerContext

logger = logging.getLogger(__name__)


# =============================================================================
# GraphiQL HTML Template
# =============================================================================

GRAPHIQL_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Aragora GraphQL Playground</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/graphiql@3/graphiql.min.css" />
</head>
<body style="margin: 0; height: 100vh;">
    <div id="graphiql" style="height: 100vh;"></div>
    <script crossorigin src="https://cdn.jsdelivr.net/npm/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://cdn.jsdelivr.net/npm/react-dom@18/umd/react-dom.production.min.js"></script>
    <script crossorigin src="https://cdn.jsdelivr.net/npm/graphiql@3/graphiql.min.js"></script>
    <script>
        const fetcher = GraphiQL.createFetcher({
            url: window.location.origin + '/graphql',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        const root = ReactDOM.createRoot(document.getElementById('graphiql'));
        root.render(
            React.createElement(GraphiQL, {
                fetcher,
                defaultEditorToolsVisibility: true,
                defaultQuery: `# Welcome to Aragora GraphQL Playground
#
# Try a query:
query GetDebates {
  debates(limit: 5) {
    debates {
      id
      topic
      status
      consensusReached
      participants {
        name
        elo
      }
    }
    total
    hasMore
  }
}

# Or get the leaderboard:
# query GetLeaderboard {
#   leaderboard(limit: 10) {
#     name
#     elo
#     stats {
#       wins
#       losses
#       winRate
#     }
#   }
# }
`,
            })
        );
    </script>
</body>
</html>
"""


# =============================================================================
# GraphQL Handler
# =============================================================================


class GraphQLHandler(BaseHandler):
    """HTTP handler for GraphQL requests.

    Handles both GraphQL queries/mutations via POST and optionally
    serves the GraphiQL playground via GET in development mode.

    Routes:
        - POST /graphql - Execute GraphQL query/mutation
        - POST /api/graphql - Execute GraphQL query/mutation (alternate)
        - POST /api/v1/graphql - Execute GraphQL query/mutation (versioned)
        - GET /graphql - GraphiQL playground (development only)

    Example request:
        POST /graphql
        Content-Type: application/json

        {
            "query": "query { debates(limit: 10) { debates { id topic } } }",
            "variables": {},
            "operationName": null
        }
    """

    ROUTES = [
        "/graphql",
        "/api/graphql",
        "/api/v1/graphql",
    ]

    def __init__(self, server_context: "ServerContext"):
        """Initialize GraphQL handler.

        Args:
            server_context: Server context with storage, ELO system, etc.
        """
        super().__init__(server_context)
        self._parser = GraphQLParser()
        self._validator = GraphQLValidator()
        self._enable_graphiql = os.environ.get("ARAGORA_ENV", "development") != "production"

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = path.rstrip("/")
        return normalized in ("/graphql", "/api/graphql", "/api/v1/graphql")

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Handle GET requests (GraphiQL playground).

        Args:
            path: Request path
            query_params: Query parameters
            handler: HTTP request handler

        Returns:
            HandlerResult with GraphiQL HTML or None
        """
        if not self.can_handle(path):
            return None

        # Serve GraphiQL playground in development mode
        if self._enable_graphiql:
            return HandlerResult(
                status_code=200,
                content_type="text/html; charset=utf-8",
                body=GRAPHIQL_HTML.encode("utf-8"),
            )

        # In production, return method not allowed for GET
        return error_response(
            "Use POST to execute GraphQL queries",
            405,
            headers={"Allow": "POST"},
        )

    @rate_limit(rpm=60, limiter_name="graphql")
    def handle_post(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Handle POST requests (GraphQL execution).

        Args:
            path: Request path
            query_params: Query parameters
            handler: HTTP request handler

        Returns:
            HandlerResult with GraphQL response
        """
        if not self.can_handle(path):
            return None

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Extract GraphQL request fields
        query = body.get("query")
        variables = body.get("variables", {})
        operation_name = body.get("operationName")

        if not query:
            return error_response("Missing 'query' field", 400)

        # Build resolver context
        user = self.get_current_user(handler)
        ctx = ResolverContext(
            server_context=self.ctx,
            user_id=user.user_id if user else None,
            org_id=user.org_id if user else None,
            trace_id=getattr(handler, "trace_id", None),
            variables=variables or {},
        )

        # Execute the query
        try:
            result = self._execute(query, variables, operation_name, ctx)
            return json_response(result, status=200 if "errors" not in result else 400)
        except Exception as e:
            logger.exception(f"GraphQL execution error: {e}")
            return json_response(
                {
                    "data": None,
                    "errors": [{"message": f"Internal server error: {e}"}],
                },
                status=500,
            )

    def _execute(
        self,
        query: str,
        variables: Optional[Dict[str, Any]],
        operation_name: Optional[str],
        ctx: ResolverContext,
    ) -> Dict[str, Any]:
        """Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables
            operation_name: Operation to execute (for multi-operation documents)
            ctx: Resolver context

        Returns:
            GraphQL response with data and/or errors
        """
        # Parse the query
        parsed = self._parser.parse(query)

        if parsed.errors:
            return {
                "data": None,
                "errors": [{"message": e} for e in parsed.errors],
            }

        # Validate the query
        validation = self._validator.validate(parsed)

        if not validation.valid:
            return {
                "data": None,
                "errors": [{"message": e} for e in validation.errors],
            }

        # Find the operation to execute
        operation = None
        if operation_name:
            for op in parsed.operations:
                if op.name == operation_name:
                    operation = op
                    break
            if not operation:
                return {
                    "data": None,
                    "errors": [{"message": f"Operation '{operation_name}' not found"}],
                }
        elif len(parsed.operations) == 1:
            operation = parsed.operations[0]
        elif len(parsed.operations) > 1:
            return {
                "data": None,
                "errors": [{"message": "Multiple operations found, specify operationName"}],
            }
        else:
            return {
                "data": None,
                "errors": [{"message": "No operation found"}],
            }

        # Resolve variables
        ctx.variables = variables or {}

        # Execute the operation
        try:
            data, errors = self._resolve_operation(operation, ctx, parsed.fragments)

            result: Dict[str, Any] = {"data": data}
            if errors:
                result["errors"] = [{"message": e} for e in errors]

            return result

        except Exception as e:
            logger.exception(f"Error executing operation: {e}")
            return {
                "data": None,
                "errors": [{"message": str(e)}],
            }

    def _resolve_operation(
        self,
        operation: Any,
        ctx: ResolverContext,
        fragments: Dict[str, Any],
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        """Resolve a GraphQL operation.

        Args:
            operation: Parsed operation
            ctx: Resolver context
            fragments: Fragment definitions

        Returns:
            Tuple of (data, errors)
        """
        errors: List[str] = []
        data: Dict[str, Any] = {}

        # Get the appropriate resolver map
        if operation.type == OperationType.QUERY:
            resolvers = QUERY_RESOLVERS
        elif operation.type == OperationType.MUTATION:
            resolvers = MUTATION_RESOLVERS
        else:
            return None, ["Subscriptions not supported via HTTP. Use WebSocket."]

        # Resolve each field
        for field in operation.selections:
            field_name = field.name
            resolver = resolvers.get(field_name)

            if not resolver:
                errors.append(f"No resolver for field '{field_name}'")
                continue

            # Resolve arguments
            args = self._resolve_arguments(field.arguments, ctx.variables)

            try:
                # Execute resolver (handle both sync and async)
                result = self._execute_resolver(resolver, ctx, args)

                if result.errors:
                    errors.extend(result.errors)
                    data[field.alias or field_name] = None
                else:
                    # Filter nested fields if selections are specified
                    resolved_data = result.data
                    if field.selections and resolved_data is not None:
                        resolved_data = self._filter_selections(
                            resolved_data, field.selections, fragments
                        )
                    data[field.alias or field_name] = resolved_data

            except Exception as e:
                logger.exception(f"Error resolving field {field_name}: {e}")
                errors.append(f"Error resolving '{field_name}': {e}")
                data[field.alias or field_name] = None

        return data if data else None, errors

    def _execute_resolver(
        self,
        resolver: Any,
        ctx: ResolverContext,
        args: Dict[str, Any],
    ) -> ResolverResult:
        """Execute a resolver function.

        Handles both sync and async resolvers.

        Args:
            resolver: Resolver function
            ctx: Resolver context
            args: Resolved arguments

        Returns:
            ResolverResult
        """
        # Determine if resolver is async
        if asyncio.iscoroutinefunction(resolver):
            # Run async resolver
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create new event loop for sync context
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, resolver(ctx, **args))
                        return future.result(timeout=30)
                else:
                    return asyncio.run(resolver(ctx, **args))
            except RuntimeError:
                # No event loop exists
                return asyncio.run(resolver(ctx, **args))
        else:
            # Sync resolver
            return resolver(ctx, **args)

    def _resolve_arguments(
        self,
        arguments: Dict[str, Any],
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve argument values, substituting variables.

        Args:
            arguments: Field arguments
            variables: Query variables

        Returns:
            Resolved arguments
        """
        resolved = {}

        for key, value in arguments.items():
            if isinstance(value, dict) and "$var" in value:
                # Variable reference
                var_name = value["$var"]
                resolved[key] = variables.get(var_name)
            elif isinstance(value, dict):
                # Nested object - recursively resolve
                resolved[key] = self._resolve_arguments(value, variables)
            elif isinstance(value, list):
                # Array - resolve each element
                resolved[key] = [
                    (
                        self._resolve_arguments({"_": v}, variables).get("_", v)
                        if isinstance(v, dict)
                        else v
                    )
                    for v in value
                ]
            else:
                resolved[key] = value

        return resolved

    def _filter_selections(
        self,
        data: Any,
        selections: List[Any],
        fragments: Dict[str, Any],
    ) -> Any:
        """Filter data to only include selected fields.

        Args:
            data: Resolved data
            selections: Field selections
            fragments: Fragment definitions

        Returns:
            Filtered data
        """
        if data is None:
            return None

        if isinstance(data, list):
            return [self._filter_selections(item, selections, fragments) for item in data]

        if not isinstance(data, dict):
            return data

        # Build set of selected field names
        selected_fields = set()
        nested_selections: Dict[str, List[Any]] = {}

        for selection in selections:
            field_name = selection.name
            selected_fields.add(field_name)

            if selection.selections:
                nested_selections[field_name] = selection.selections

        # Filter to selected fields
        result = {}
        for key, value in data.items():
            if key in selected_fields:
                if key in nested_selections:
                    result[key] = self._filter_selections(value, nested_selections[key], fragments)
                else:
                    result[key] = value

        return result


# =============================================================================
# Schema Introspection Handler
# =============================================================================


class GraphQLSchemaHandler(BaseHandler):
    """Handler for GraphQL schema introspection requests.

    Provides the raw SDL schema at /graphql/schema endpoint.
    """

    ROUTES = ["/graphql/schema", "/api/graphql/schema", "/api/v1/graphql/schema"]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.rstrip("/") in (
            "/graphql/schema",
            "/api/graphql/schema",
            "/api/v1/graphql/schema",
        )

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Handle GET requests for schema.

        Args:
            path: Request path
            query_params: Query parameters
            handler: HTTP request handler

        Returns:
            HandlerResult with schema SDL
        """
        if not self.can_handle(path):
            return None

        # Return schema in requested format
        format = query_params.get("format", "sdl")

        if format == "json":
            # Return as JSON introspection result
            return json_response(
                {
                    "data": {
                        "__schema": {
                            "description": "Aragora GraphQL Schema",
                            "queryType": {"name": "Query"},
                            "mutationType": {"name": "Mutation"},
                            "subscriptionType": {"name": "Subscription"},
                        }
                    }
                }
            )

        # Return raw SDL
        return HandlerResult(
            status_code=200,
            content_type="text/plain; charset=utf-8",
            body=SCHEMA_SDL.encode("utf-8"),
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "GraphQLHandler",
    "GraphQLSchemaHandler",
    "GRAPHIQL_HTML",
]
