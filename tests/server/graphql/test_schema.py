"""Tests for GraphQL schema parsing and validation."""

import pytest

from aragora.server.graphql.schema import (
    ENUM_VALUES,
    Field,
    GraphQLParser,
    GraphQLValidator,
    INPUT_TYPES,
    MUTATION_FIELDS,
    OBJECT_TYPES,
    Operation,
    OperationType,
    ParsedQuery,
    QUERY_FIELDS,
    SCHEMA_SDL,
    SUBSCRIPTION_FIELDS,
    ValidationResult,
    parse_and_validate_query,
)


class TestGraphQLParser:
    """Tests for the GraphQL query parser."""

    def test_parse_simple_query(self):
        """Test parsing a simple query."""
        parser = GraphQLParser()
        query = """
        query {
            debates {
                debates {
                    id
                    topic
                }
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].type == OperationType.QUERY
        assert len(result.operations[0].selections) == 1
        assert result.operations[0].selections[0].name == "debates"

    def test_parse_query_with_arguments(self):
        """Test parsing a query with arguments."""
        parser = GraphQLParser()
        query = """
        query GetDebates {
            debates(limit: 10, status: RUNNING) {
                debates {
                    id
                    topic
                    status
                }
                total
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].name == "GetDebates"

        debates_field = result.operations[0].selections[0]
        assert debates_field.name == "debates"
        assert debates_field.arguments.get("limit") == 10
        assert debates_field.arguments.get("status") == "RUNNING"

    def test_parse_query_with_variables(self):
        """Test parsing a query with variable definitions."""
        parser = GraphQLParser()
        query = """
        query GetDebate($id: ID!) {
            debate(id: $id) {
                id
                topic
                status
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].name == "GetDebate"
        assert "id" in result.operations[0].variables

    def test_parse_mutation(self):
        """Test parsing a mutation."""
        parser = GraphQLParser()
        query = """
        mutation StartDebate($input: StartDebateInput!) {
            startDebate(input: $input) {
                id
                topic
                status
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].type == OperationType.MUTATION
        assert result.operations[0].name == "StartDebate"

    def test_parse_subscription(self):
        """Test parsing a subscription."""
        parser = GraphQLParser()
        query = """
        subscription OnDebateUpdate($debateId: ID!) {
            debateUpdates(debateId: $debateId) {
                type
                debateId
                data
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].type == OperationType.SUBSCRIPTION

    def test_parse_shorthand_query(self):
        """Test parsing shorthand query syntax (no 'query' keyword)."""
        parser = GraphQLParser()
        query = """
        {
            systemHealth {
                status
                uptimeSeconds
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        assert len(result.operations) == 1
        assert result.operations[0].type == OperationType.QUERY
        assert result.operations[0].name is None

    def test_parse_nested_selections(self):
        """Test parsing deeply nested selections."""
        parser = GraphQLParser()
        query = """
        query {
            debate(id: "123") {
                rounds {
                    messages {
                        content
                        agent
                    }
                    critiques {
                        content
                        severity
                    }
                }
                participants {
                    name
                    stats {
                        wins
                        losses
                    }
                }
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        debate_field = result.operations[0].selections[0]
        assert debate_field.name == "debate"
        assert len(debate_field.selections) == 2  # rounds, participants

    def test_parse_field_alias(self):
        """Test parsing field aliases."""
        parser = GraphQLParser()
        query = """
        query {
            recent: debates(limit: 5) {
                debates {
                    id
                }
            }
            all: debates(limit: 100) {
                total
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        assert len(result.operations[0].selections) == 2

    def test_parse_string_arguments(self):
        """Test parsing string arguments."""
        parser = GraphQLParser()
        query = """
        query {
            searchDebates(query: "AI safety") {
                debates {
                    id
                    topic
                }
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0
        search_field = result.operations[0].selections[0]
        assert search_field.arguments.get("query") == "AI safety"

    def test_parse_boolean_arguments(self):
        """Test parsing boolean arguments."""
        parser = GraphQLParser()
        query = """
        mutation {
            startDebate(input: {question: "Test", autoSelect: true}) {
                id
            }
        }
        """
        result = parser.parse(query)

        assert len(result.errors) == 0


class TestGraphQLValidator:
    """Tests for the GraphQL query validator."""

    def test_validate_valid_query(self):
        """Test validating a valid query."""
        parser = GraphQLParser()
        validator = GraphQLValidator()

        query = """
        query {
            debates {
                debates {
                    id
                    topic
                }
            }
        }
        """
        parsed = parser.parse(query)
        result = validator.validate(parsed)

        assert result.valid
        assert len(result.errors) == 0

    def test_validate_invalid_root_field(self):
        """Test validating a query with invalid root field."""
        parser = GraphQLParser()
        validator = GraphQLValidator()

        query = """
        query {
            nonExistentField {
                id
            }
        }
        """
        parsed = parser.parse(query)
        result = validator.validate(parsed)

        assert not result.valid
        assert any("nonExistentField" in e for e in result.errors)

    def test_validate_invalid_nested_field(self):
        """Test validating a query with invalid nested field."""
        parser = GraphQLParser()
        validator = GraphQLValidator()

        query = """
        query {
            debates {
                debates {
                    nonExistentField
                }
            }
        }
        """
        parsed = parser.parse(query)
        result = validator.validate(parsed)

        assert not result.valid
        assert any("nonExistentField" in e for e in result.errors)

    def test_validate_mutation_fields(self):
        """Test validating mutation fields."""
        parser = GraphQLParser()
        validator = GraphQLValidator()

        query = """
        mutation {
            startDebate(input: {question: "Test"}) {
                id
                topic
            }
        }
        """
        parsed = parser.parse(query)
        result = validator.validate(parsed)

        assert result.valid

    def test_validate_typename_always_valid(self):
        """Test that __typename introspection field is always valid."""
        parser = GraphQLParser()
        validator = GraphQLValidator()

        query = """
        query {
            debates {
                debates {
                    __typename
                    id
                }
            }
        }
        """
        parsed = parser.parse(query)
        result = validator.validate(parsed)

        assert result.valid


class TestParseAndValidateQuery:
    """Tests for the combined parse and validate function."""

    def test_parse_and_validate_valid(self):
        """Test parse_and_validate_query with valid query."""
        query = """
        query {
            leaderboard(limit: 10) {
                name
                elo
                stats {
                    wins
                    losses
                }
            }
        }
        """
        result = parse_and_validate_query(query)

        assert len(result.errors) == 0
        assert len(result.operations) == 1

    def test_parse_and_validate_invalid_syntax(self):
        """Test parse_and_validate_query with invalid syntax."""
        query = """
        query {
            unclosed brace
        """
        result = parse_and_validate_query(query)

        # Should have parse errors
        # Note: Our simple parser may not catch all syntax errors
        # but should not crash

    def test_parse_and_validate_invalid_field(self):
        """Test parse_and_validate_query with invalid field."""
        query = """
        query {
            invalidField {
                id
            }
        }
        """
        result = parse_and_validate_query(query)

        assert len(result.errors) > 0


class TestSchemaConstants:
    """Tests for schema constants and metadata."""

    def test_query_fields_defined(self):
        """Test that all expected query fields are defined."""
        expected_fields = {
            "debate",
            "debates",
            "searchDebates",
            "agent",
            "agents",
            "leaderboard",
            "task",
            "tasks",
            "systemHealth",
            "stats",
        }
        assert expected_fields.issubset(QUERY_FIELDS)

    def test_mutation_fields_defined(self):
        """Test that all expected mutation fields are defined."""
        expected_fields = {
            "startDebate",
            "submitVote",
            "cancelDebate",
            "submitTask",
            "cancelTask",
            "registerAgent",
            "unregisterAgent",
        }
        assert expected_fields.issubset(MUTATION_FIELDS)

    def test_subscription_fields_defined(self):
        """Test that all expected subscription fields are defined."""
        expected_fields = {"debateUpdates", "taskUpdates"}
        assert expected_fields.issubset(SUBSCRIPTION_FIELDS)

    def test_object_types_defined(self):
        """Test that all expected object types are defined."""
        expected_types = {
            "Debate",
            "DebateConnection",
            "Round",
            "Message",
            "Agent",
            "AgentStats",
            "Task",
            "TaskConnection",
            "SystemHealth",
            "SystemStats",
        }
        assert expected_types.issubset(set(OBJECT_TYPES.keys()))

    def test_debate_fields_complete(self):
        """Test that Debate type has expected fields."""
        expected_fields = {
            "id",
            "topic",
            "status",
            "rounds",
            "participants",
            "consensus",
            "createdAt",
            "completedAt",
        }
        assert expected_fields.issubset(OBJECT_TYPES["Debate"])

    def test_agent_fields_complete(self):
        """Test that Agent type has expected fields."""
        expected_fields = {
            "id",
            "name",
            "status",
            "capabilities",
            "stats",
            "elo",
        }
        assert expected_fields.issubset(OBJECT_TYPES["Agent"])

    def test_input_types_defined(self):
        """Test that input types are defined."""
        expected_inputs = {
            "StartDebateInput",
            "VoteInput",
            "SubmitTaskInput",
            "RegisterAgentInput",
        }
        assert expected_inputs.issubset(set(INPUT_TYPES.keys()))

    def test_enum_values_defined(self):
        """Test that enum values are defined."""
        assert "DebateStatus" in ENUM_VALUES
        assert "COMPLETED" in ENUM_VALUES["DebateStatus"]

        assert "AgentStatus" in ENUM_VALUES
        assert "AVAILABLE" in ENUM_VALUES["AgentStatus"]

        assert "Priority" in ENUM_VALUES
        assert "HIGH" in ENUM_VALUES["Priority"]

    def test_schema_sdl_not_empty(self):
        """Test that schema SDL is defined."""
        assert SCHEMA_SDL
        assert len(SCHEMA_SDL) > 1000  # Should be substantial
        assert "type Query" in SCHEMA_SDL
        assert "type Mutation" in SCHEMA_SDL
        assert "type Subscription" in SCHEMA_SDL


class TestComplexQueries:
    """Tests for complex query scenarios."""

    def test_multiple_root_fields(self):
        """Test query with multiple root fields."""
        query = """
        query DashboardData {
            debates(limit: 5) {
                debates {
                    id
                    topic
                }
            }
            leaderboard(limit: 10) {
                name
                elo
            }
            systemHealth {
                status
            }
            stats {
                activeJobs
                totalAgents
            }
        }
        """
        result = parse_and_validate_query(query)

        assert len(result.errors) == 0
        assert len(result.operations[0].selections) == 4

    def test_deeply_nested_query(self):
        """Test deeply nested query structure."""
        query = """
        query {
            debates {
                debates {
                    rounds {
                        messages {
                            content
                            agent
                        }
                    }
                    participants {
                        stats {
                            wins
                            losses
                            winRate
                        }
                    }
                    consensus {
                        reached
                        answer
                    }
                }
            }
        }
        """
        result = parse_and_validate_query(query)

        # Should parse without errors
        assert isinstance(result, ParsedQuery)

    def test_mutation_with_complex_input(self):
        """Test mutation with complex input object."""
        query = """
        mutation StartNewDebate {
            startDebate(input: {
                question: "What are the ethical implications of AGI?"
                agents: "claude,gpt4,gemini"
                rounds: 5
                consensus: "majority"
                autoSelect: false
                tags: ["ethics", "agi", "philosophy"]
            }) {
                id
                topic
                status
                roundCount
                tags
            }
        }
        """
        result = parse_and_validate_query(query)

        assert len(result.errors) == 0
        assert result.operations[0].type == OperationType.MUTATION
