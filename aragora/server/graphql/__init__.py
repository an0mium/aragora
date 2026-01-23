"""
GraphQL API Layer for Aragora.

Provides a GraphQL endpoint that wraps existing REST functionality
to enable efficient data fetching and reduce over-fetching.

Features:
    - Query debates, agents, tasks, and system health
    - Start debates and submit tasks via mutations
    - Real-time updates via subscriptions (WebSocket)
    - GraphiQL playground in development mode
    - Lightweight implementation without external dependencies

Usage:
    # In unified_server.py or handler registration
    from aragora.server.graphql import GraphQLHandler, GraphQLSchemaHandler

    # Register handlers
    graphql_handler = GraphQLHandler(server_context)
    schema_handler = GraphQLSchemaHandler(server_context)

Example queries:
    # Get recent debates with participants
    query {
        debates(limit: 10) {
            debates {
                id
                topic
                status
                participants {
                    name
                    elo
                }
            }
        }
    }

    # Start a new debate
    mutation {
        startDebate(input: {
            question: "What is the best programming language?"
            agents: "claude,gpt4,gemini"
            rounds: 3
        }) {
            id
            topic
            status
        }
    }

Endpoints:
    POST /graphql - Execute GraphQL queries and mutations
    GET /graphql - GraphiQL playground (development only)
    GET /graphql/schema - Get the GraphQL schema SDL
"""

from .handler import GraphQLHandler, GraphQLSchemaHandler, GRAPHIQL_HTML
from .resolvers import (
    QueryResolvers,
    MutationResolvers,
    SubscriptionResolvers,
    ResolverContext,
    ResolverResult,
    QUERY_RESOLVERS,
    MUTATION_RESOLVERS,
    SUBSCRIPTION_RESOLVERS,
)
from .schema import (
    SCHEMA,
    SCHEMA_SDL,
    GraphQLParser,
    GraphQLValidator,
    ParsedQuery,
    ValidationResult,
    OperationType,
    Field,
    Operation,
    parse_and_validate_query,
    QUERY_FIELDS,
    MUTATION_FIELDS,
    SUBSCRIPTION_FIELDS,
    OBJECT_TYPES,
    INPUT_TYPES,
    ENUM_VALUES,
)

__all__ = [
    # Handlers
    "GraphQLHandler",
    "GraphQLSchemaHandler",
    "GRAPHIQL_HTML",
    # Resolvers
    "QueryResolvers",
    "MutationResolvers",
    "SubscriptionResolvers",
    "ResolverContext",
    "ResolverResult",
    "QUERY_RESOLVERS",
    "MUTATION_RESOLVERS",
    "SUBSCRIPTION_RESOLVERS",
    # Schema
    "SCHEMA",
    "SCHEMA_SDL",
    "GraphQLParser",
    "GraphQLValidator",
    "ParsedQuery",
    "ValidationResult",
    "OperationType",
    "Field",
    "Operation",
    "parse_and_validate_query",
    # Schema metadata
    "QUERY_FIELDS",
    "MUTATION_FIELDS",
    "SUBSCRIPTION_FIELDS",
    "OBJECT_TYPES",
    "INPUT_TYPES",
    "ENUM_VALUES",
]
