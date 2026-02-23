"""Playbook endpoint definitions.

Decision playbooks are repeatable, pre-configured debate workflows
tailored to specific domains (healthcare, finance, legal, engineering).
Each playbook encodes agent selection, consensus strategy, compliance
requirements, and output channels into a single runnable package.
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

# ---------------------------------------------------------------------------
# Shared schema fragments
# ---------------------------------------------------------------------------

_PLAYBOOK_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Step name"},
        "action": {"type": "string", "description": "Action to execute"},
        "config": {"type": "object", "description": "Step configuration"},
    },
}

_APPROVAL_GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
    },
}

_PLAYBOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Unique playbook identifier"},
        "name": {"type": "string", "description": "Human-readable playbook name"},
        "description": {"type": "string", "description": "What this playbook does"},
        "category": {
            "type": "string",
            "enum": ["healthcare", "finance", "legal", "engineering", "general"],
            "description": "Domain category",
        },
        "template_name": {
            "type": "string",
            "description": "Deliberation template to use",
        },
        "vertical_profile": {
            "type": "string",
            "nullable": True,
            "description": "Vertical weight profile for domain-specific scoring",
        },
        "compliance_artifacts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Required compliance artifacts to generate",
        },
        "min_agents": {
            "type": "integer",
            "minimum": 1,
            "description": "Minimum number of agents for the debate",
        },
        "max_agents": {
            "type": "integer",
            "maximum": 42,
            "description": "Maximum number of agents for the debate",
        },
        "required_agent_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agent types that must participate",
        },
        "agent_selection_strategy": {
            "type": "string",
            "description": "How agents are selected (best_for_domain, random, elo_ranked)",
        },
        "max_rounds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 12,
            "description": "Maximum debate rounds",
        },
        "consensus_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Consensus threshold (0.5=majority, 0.75=strong, 1.0=unanimous)",
        },
        "timeout_seconds": {
            "type": "number",
            "description": "Maximum time for the entire playbook execution",
        },
        "output_format": {
            "type": "string",
            "description": "Output format (decision_receipt, markdown, json)",
        },
        "output_channels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Channels to deliver results (slack, email, webhook)",
        },
        "approval_gates": {
            "type": "array",
            "items": _APPROVAL_GATE_SCHEMA,
            "description": "Human approval checkpoints",
        },
        "steps": {
            "type": "array",
            "items": _PLAYBOOK_STEP_SCHEMA,
            "description": "Ordered steps in the playbook",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tags for filtering and discovery",
        },
        "version": {"type": "string", "description": "Playbook version"},
        "metadata": {"type": "object", "description": "Additional metadata"},
    },
}

_PLAYBOOK_EXAMPLE = {
    "id": "hipaa-clinical-review",
    "name": "HIPAA Clinical Decision Review",
    "description": "Vet clinical decisions against HIPAA requirements with domain-expert agents",
    "category": "healthcare",
    "template_name": "clinical_review",
    "vertical_profile": "healthcare_hipaa",
    "compliance_artifacts": ["article_12", "article_13"],
    "min_agents": 3,
    "max_agents": 5,
    "required_agent_types": ["medical_specialist", "compliance_auditor"],
    "agent_selection_strategy": "best_for_domain",
    "max_rounds": 5,
    "consensus_threshold": 0.75,
    "timeout_seconds": 300.0,
    "output_format": "decision_receipt",
    "output_channels": ["email"],
    "approval_gates": [],
    "steps": [
        {"name": "gather_context", "action": "knowledge_query", "config": {"domain": "healthcare"}},
        {"name": "run_debate", "action": "debate", "config": {"template": "clinical_review"}},
        {"name": "generate_receipt", "action": "receipt", "config": {}},
    ],
    "tags": ["healthcare", "hipaa", "clinical"],
    "version": "1.0.0",
    "metadata": {},
}

# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------

PLAYBOOK_ENDPOINTS = {
    "/api/v1/playbooks": {
        "get": {
            "tags": ["Playbooks"],
            "summary": "List available playbooks",
            "operationId": "listPlaybooks",
            "description": """List all available decision playbooks.

**Purpose:** Browse pre-configured debate workflows for specific domains and use cases.
Playbooks encode best practices for agent selection, consensus strategy, compliance
requirements, and output channels into repeatable packages.

**Filtering:** Use `category` to filter by domain and `tags` to narrow by capability.

**Rate Limit:** 60 requests per minute.""",
            "parameters": [
                {
                    "name": "category",
                    "in": "query",
                    "description": "Filter by domain category",
                    "schema": {
                        "type": "string",
                        "enum": ["healthcare", "finance", "legal", "engineering", "general"],
                    },
                },
                {
                    "name": "tags",
                    "in": "query",
                    "description": "Comma-separated tags to filter by (e.g. hipaa,clinical)",
                    "schema": {"type": "string"},
                    "example": "hipaa,clinical",
                },
            ],
            "responses": {
                "200": _ok_response(
                    "List of playbooks",
                    {
                        "type": "object",
                        "properties": {
                            "playbooks": {
                                "type": "array",
                                "items": _PLAYBOOK_SCHEMA,
                            },
                            "count": {
                                "type": "integer",
                                "description": "Total number of playbooks returned",
                            },
                        },
                        "example": {
                            "playbooks": [_PLAYBOOK_EXAMPLE],
                            "count": 1,
                        },
                    },
                ),
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v1/playbooks/{id}": {
        "get": {
            "tags": ["Playbooks"],
            "summary": "Get playbook details",
            "operationId": "getPlaybook",
            "description": """Get full details for a specific playbook by ID.

**Response includes:** All configuration fields including agent selection criteria,
debate parameters, compliance artifacts, steps, approval gates, and output channels.""",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Playbook ID",
                    "example": "hipaa-clinical-review",
                }
            ],
            "responses": {
                "200": _ok_response(
                    "Playbook details",
                    {
                        **_PLAYBOOK_SCHEMA,
                        "example": _PLAYBOOK_EXAMPLE,
                    },
                ),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/playbooks/{id}/run": {
        "post": {
            "tags": ["Playbooks"],
            "summary": "Run a playbook",
            "operationId": "runPlaybook",
            "description": """Execute a playbook with the given input.

**Purpose:** Start a full decision workflow: gather context, run a multi-agent debate
with domain-appropriate agents, generate compliance artifacts, and deliver a decision
receipt through configured output channels.

**Authentication:** Required. The run will be owned by the authenticated user.

**Response:** Returns a run object with status `queued`. The execution proceeds
asynchronously -- poll the run status or connect via WebSocket for real-time updates.

**Rate Limit:** 10 runs per hour (free), 100/hour (pro), unlimited (enterprise).""",
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Playbook ID to run",
                    "example": "hipaa-clinical-review",
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "The question or topic for the playbook to evaluate",
                                },
                                "context": {
                                    "type": "object",
                                    "description": "Additional context variables for the playbook steps",
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["input"],
                        },
                        "examples": {
                            "clinical_review": {
                                "summary": "Clinical decision review",
                                "value": {
                                    "input": "Should we adopt AI-assisted radiology screening for our clinic?",
                                    "context": {
                                        "clinic_size": "50 beds",
                                        "patient_volume": "200/day",
                                        "budget": "$500k",
                                    },
                                },
                            },
                            "simple": {
                                "summary": "Simple question",
                                "value": {
                                    "input": "Should we migrate our data warehouse to Snowflake?",
                                },
                            },
                        },
                    }
                },
            },
            "responses": {
                "202": _ok_response(
                    "Playbook run queued",
                    {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Unique run identifier for tracking",
                            },
                            "playbook_id": {"type": "string"},
                            "playbook_name": {"type": "string"},
                            "input": {"type": "string"},
                            "context": {"type": "object"},
                            "status": {
                                "type": "string",
                                "enum": ["queued", "running", "completed", "failed"],
                            },
                            "created_at": {
                                "type": "string",
                                "format": "date-time",
                            },
                            "steps": {
                                "type": "array",
                                "items": _PLAYBOOK_STEP_SCHEMA,
                                "description": "Steps that will be executed",
                            },
                            "config": {
                                "type": "object",
                                "description": "Resolved playbook configuration",
                                "properties": {
                                    "template_name": {"type": "string"},
                                    "vertical_profile": {"type": "string", "nullable": True},
                                    "min_agents": {"type": "integer"},
                                    "max_agents": {"type": "integer"},
                                    "max_rounds": {"type": "integer"},
                                    "consensus_threshold": {"type": "number"},
                                },
                            },
                        },
                        "example": {
                            "run_id": "run_a1b2c3d4e5f6",
                            "playbook_id": "hipaa-clinical-review",
                            "playbook_name": "HIPAA Clinical Decision Review",
                            "input": "Should we adopt AI-assisted radiology screening?",
                            "context": {"clinic_size": "50 beds"},
                            "status": "queued",
                            "created_at": "2026-02-21T12:00:00Z",
                            "steps": [
                                {
                                    "name": "gather_context",
                                    "action": "knowledge_query",
                                    "config": {"domain": "healthcare"},
                                },
                                {
                                    "name": "run_debate",
                                    "action": "debate",
                                    "config": {"template": "clinical_review"},
                                },
                                {"name": "generate_receipt", "action": "receipt", "config": {}},
                            ],
                            "config": {
                                "template_name": "clinical_review",
                                "vertical_profile": "healthcare_hipaa",
                                "min_agents": 3,
                                "max_agents": 5,
                                "max_rounds": 5,
                                "consensus_threshold": 0.75,
                            },
                        },
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}

__all__ = ["PLAYBOOK_ENDPOINTS"]
