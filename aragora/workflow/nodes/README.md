# Workflow Nodes - Step Type Implementations

Reusable step type implementations for the Aragora Workflow Engine. Nodes are the atomic building blocks that execute discrete operations within workflows.

## Quick Start

```python
from aragora.workflow.nodes import (
    DebateStep,
    HumanCheckpointStep,
    MemoryReadStep,
    ConnectorStep,
)

# Execute a multi-agent debate as a workflow step
debate_step = DebateStep(
    name="Risk Analysis",
    config={
        "topic": "Evaluate risk factors in {document_name}",
        "agents": ["claude", "gpt4", "gemini"],
        "rounds": 3,
        "consensus_mechanism": "unanimous",
    }
)

# Add human approval gate
approval_step = HumanCheckpointStep(
    name="Legal Review",
    config={
        "title": "Contract Review Required",
        "checklist": [
            {"label": "Verified compliance terms", "required": True},
            {"label": "Checked liability clauses", "required": True},
        ],
        "timeout_seconds": 7200,
        "escalation_emails": ["legal@company.com"],
    }
)
```

## Available Nodes

| Node | Type | Purpose |
|------|------|---------|
| `HumanCheckpointStep` | Human-in-loop | Approval gates with checklists and escalation |
| `DebateStep` | AI orchestration | Execute Aragora multi-agent debates |
| `QuickDebateStep` | AI orchestration | Lightweight single-round consultations |
| `MemoryReadStep` | Knowledge | Query the Knowledge Mound |
| `MemoryWriteStep` | Knowledge | Store data in the Knowledge Mound |
| `DecisionStep` | Control flow | Conditional branching with expressions |
| `SwitchStep` | Control flow | Value-based branching (switch/case) |
| `TaskStep` | Generic | HTTP, transforms, validation, aggregation |
| `ConnectorStep` | Integration | 20+ external service integrations |
| `GauntletStep` | Validation | Adversarial security and compliance checks |
| `NomicLoopStep` | Self-improvement | Execute Nomic Loop phases |
| `KnowledgePipelineStep` | Knowledge | Document ingestion and processing |
| `KnowledgePruningStep` | Knowledge | Automatic knowledge maintenance |
| `KnowledgeDedupStep` | Knowledge | Duplicate detection and merging |
| `ConfidenceDecayStep` | Knowledge | Time-based confidence decay |
| `OpenClawActionStep` | Gateway | Enterprise Gateway shell/file/browser actions |
| `OpenClawSessionStep` | Gateway | Manage Gateway proxy sessions |

## Node Architecture

### Base Classes

All nodes extend from `BaseStep` which provides:

```python
from aragora.workflow.step import BaseStep, WorkflowContext

class CustomStep(BaseStep):
    """Custom workflow step implementation."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the step with given context."""
        # Access configuration
        config = {**self._config, **context.current_step_config}

        # Access workflow inputs
        input_value = context.get_input("key")

        # Access previous step outputs
        prev_output = context.get_step_output("previous_step")

        # Access/modify shared state
        context.set_state("my_key", "value")

        return {"result": "success"}

    async def checkpoint(self) -> dict[str, Any]:
        """Save state for pause/resume (optional)."""
        return {"internal_state": self._state}

    async def restore(self, state: dict[str, Any]) -> None:
        """Restore from checkpoint (optional)."""
        self._state = state.get("internal_state")

    def validate_config(self) -> bool:
        """Validate configuration (optional)."""
        return "required_key" in self._config
```

### WorkflowContext

The context object provides access to the workflow execution state:

| Property | Type | Description |
|----------|------|-------------|
| `workflow_id` | `str` | Unique workflow execution ID |
| `definition_id` | `str` | Workflow definition ID |
| `inputs` | `dict` | Workflow input parameters |
| `step_outputs` | `dict` | Outputs from previous steps |
| `state` | `dict` | Shared mutable state |
| `metadata` | `dict` | Workflow metadata (tenant_id, etc.) |
| `current_step_id` | `str` | Current step identifier |
| `current_step_config` | `dict` | Step-specific configuration |

### Execution Model

1. **Initialization**: Step receives name and config at construction
2. **Validation**: Engine calls `validate_config()` before execution
3. **Execution**: Engine calls `execute(context)` with workflow context
4. **Output**: Return value is stored in `context.step_outputs[step_id]`
5. **Checkpointing**: Engine may call `checkpoint()` to save state

## Human-in-the-Loop Nodes

### HumanCheckpointStep

Approval gates with checklist validation, timeout handling, and escalation.

```python
from aragora.workflow.nodes import HumanCheckpointStep

step = HumanCheckpointStep(
    name="Compliance Review",
    config={
        "title": "Compliance Approval Required",
        "description": "Review the generated report for compliance",
        "checklist": [
            {"label": "Data handling complies with GDPR", "required": True},
            {"label": "PII is properly anonymized", "required": True},
            {"label": "Audit trail is complete", "required": False},
        ],
        "timeout_seconds": 3600,           # 1 hour timeout
        "escalation_emails": ["manager@company.com"],
        "auto_approve_if": "inputs.risk_level < 0.3",  # Auto-approve low risk
        "require_all_checklist": True,
    }
)
```

**Config Options:**
- `title`: Approval request title
- `description`: Detailed description for approvers
- `checklist`: List of items `{label, required}`
- `timeout_seconds`: Timeout before escalation (default: 3600)
- `escalation_emails`: Emails for timeout notification
- `auto_approve_if`: Expression for automatic approval
- `require_all_checklist`: Require all checklist items (default: True)
- `assignees`: Specific users to notify

**Programmatic Approval:**

```python
from aragora.workflow.nodes.human_checkpoint import (
    resolve_approval,
    get_pending_approvals,
    ApprovalStatus,
)

# Get pending approvals
pending = get_pending_approvals(workflow_id="wf_123")

# Resolve an approval
resolve_approval(
    request_id="apr_abc123",
    status=ApprovalStatus.APPROVED,
    responder_id="user@company.com",
    notes="Looks good",
    checklist_updates={"item_0": True, "item_1": True},
)
```

## Memory Nodes

### MemoryReadStep

Query the Knowledge Mound for relevant knowledge.

```python
from aragora.workflow.nodes import MemoryReadStep

step = MemoryReadStep(
    name="Retrieve Context",
    config={
        "query": "What are the compliance requirements for {document_type}?",
        "query_type": "hybrid",         # semantic, keyword, or hybrid
        "domain_filter": "legal/compliance",
        "min_confidence": 0.7,
        "limit": 5,
        "include_graph": True,          # Include related nodes
        "graph_depth": 2,
        "tenant_id": "org_123",
    }
)
```

**Query Types:**
- `semantic`: Vector similarity search
- `keyword`: Traditional keyword matching
- `hybrid`: Combined semantic and keyword (default)

### MemoryWriteStep

Store knowledge with relationships in the Knowledge Mound.

```python
from aragora.workflow.nodes import MemoryWriteStep

step = MemoryWriteStep(
    name="Store Analysis",
    config={
        "content": "{step.debate.synthesis}",
        "source_type": "consensus",      # fact, consensus, insight
        "domain": "legal/analysis",
        "confidence": 0.85,
        "relationships": [
            {"type": "derived_from", "target": "{input.source_doc_id}"},
            {"type": "supports", "target": "km_existing_fact"},
        ],
        "deduplicate": True,
        "metadata": {
            "reviewed_by": "{input.reviewer}",
        },
    }
)
```

**Relationship Types:**
- `supports`: New knowledge supports existing item
- `contradicts`: New knowledge contradicts existing item
- `derived_from`: New knowledge derived from source

## External Integration Nodes

### ConnectorStep

First-class connector integration with 20+ services.

```python
from aragora.workflow.nodes import ConnectorStep

# GitHub integration
github_step = ConnectorStep(
    name="Fetch Issue",
    config={
        "connector_type": "github",
        "operation": "fetch",
        "params": {
            "owner": "myorg",
            "repo": "myrepo",
            "issue_number": "{inputs.issue_id}",
        },
        "credentials_key": "github_token",
    }
)

# QuickBooks integration
qbo_step = ConnectorStep(
    name="Create Invoice",
    config={
        "connector_type": "quickbooks",
        "operation": "create",
        "params": {
            "resource_type": "invoice",
            "data": "{step.invoice_data.result}",
        },
        "timeout_seconds": 30,
        "retry_on_error": True,
        "max_retries": 3,
    }
)

# Custom method
docusign_step = ConnectorStep(
    name="Check Status",
    config={
        "connector_type": "docusign",
        "operation": "custom",
        "custom_method": "get_envelope_status",
        "params": {"envelope_id": "{inputs.envelope_id}"},
    }
)
```

**Available Connectors:**

| Category | Connectors |
|----------|------------|
| Core | `github`, `web`, `local_docs` |
| News/Social | `hackernews`, `reddit`, `twitter`, `newsapi` |
| Academic | `arxiv`, `wikipedia` |
| Enterprise | `sql`, `sec` |
| Legal | `docusign` |
| Accounting | `quickbooks`, `xero`, `plaid` |
| DevOps | `pagerduty` |
| Chat | `slack`, `discord` |
| E-commerce | `shopify` |
| Support | `zendesk` |

**Operations:**
- `search`: Query for items
- `fetch`: Retrieve specific item
- `list`: List all items
- `create`: Create new item
- `update`: Update existing item
- `delete`: Remove item
- `sync`: Synchronize data
- `custom`: Call custom method

**Registering Custom Connectors:**

```python
from aragora.workflow.nodes.connector import register_connector, ConnectorMetadata

register_connector(
    "my_service",
    ConnectorMetadata(
        name="My Service",
        description="Custom integration",
        module_path="myapp.connectors.my_service",
        class_name="MyServiceConnector",
        operations=["fetch", "create"],
        auth_type="api_key",
    )
)
```

### GauntletStep

Adversarial validation for security and compliance.

```python
from aragora.workflow.nodes import GauntletStep

step = GauntletStep(
    name="Security Validation",
    config={
        "input_key": "content",
        "attack_categories": [
            "prompt_injection",
            "jailbreak",
            "data_extraction",
            "privacy",
        ],
        "probe_categories": ["reasoning", "consistency"],
        "compliance_frameworks": ["gdpr", "hipaa"],
        "require_passing": True,
        "severity_threshold": "medium",  # low, medium, high, critical
        "max_findings": 100,
        "timeout_seconds": 300,
        "parallel_attacks": 3,
    }
)
```

**Attack Categories:**
- `prompt_injection`: Injection attack detection
- `jailbreak`: Jailbreak attempt detection
- `data_extraction`: Data exfiltration attempts
- `hallucination`: Hallucination detection
- `bias`: Bias analysis
- `privacy`: Privacy violation detection
- `safety`: General safety checks

**Compliance Frameworks:**
- `gdpr`: EU General Data Protection Regulation
- `hipaa`: US Health Insurance Portability
- `soc2`: Service Organization Control 2
- `pci_dss`: Payment Card Industry DSS
- `nist_csf`: NIST Cybersecurity Framework
- `ai_act`: EU AI Act
- `sox`: Sarbanes-Oxley Act

## AI Orchestration Nodes

### DebateStep

Execute full Aragora multi-agent debates.

```python
from aragora.workflow.nodes import DebateStep

step = DebateStep(
    name="Contract Review Debate",
    config={
        "topic": "Review the terms of {contract_name}",
        "agents": ["legal_analyst", "risk_assessor", "compliance_officer"],
        "rounds": 3,
        "topology": "round_robin",
        "consensus_mechanism": "unanimous",
        "enable_critique": True,
        "enable_synthesis": True,
        "timeout_seconds": 120,
        "memory_enabled": True,
        "arena_config": {
            "enable_knowledge_retrieval": True,
            "enable_knowledge_ingestion": True,
            "org_id": "{inputs.org_id}",
        },
    }
)
```

**Topologies:**
- `round_robin`: Each agent speaks in turn
- `graph`: Free-form discussion
- `adversarial`: Two opposing teams
- `hive_mind`: Parallel responses
- `dialectic`: Thesis-antithesis-synthesis
- `socratic`: Question-driven exploration

### QuickDebateStep

Lightweight single-round consultations.

```python
from aragora.workflow.nodes.debate import QuickDebateStep

step = QuickDebateStep(
    name="Quick Consultation",
    config={
        "question": "What are the key risks in {document}?",
        "agents": ["claude", "gpt4"],
        "max_response_length": 500,
        "synthesize": True,
    }
)
```

## Control Flow Nodes

### DecisionStep

Conditional branching with expression evaluation.

```python
from aragora.workflow.nodes import DecisionStep

step = DecisionStep(
    name="Risk Routing",
    config={
        "conditions": [
            {
                "name": "high_risk",
                "expression": "step.risk_assessment.score > 0.8",
                "next_step": "manual_review",
            },
            {
                "name": "medium_risk",
                "expression": "step.risk_assessment.score > 0.5",
                "next_step": "enhanced_review",
            },
        ],
        "default_branch": "auto_approve",
        "evaluation_mode": "first_match",  # or "all"
        "ai_fallback": False,
    }
)
```

### SwitchStep

Value-based branching (switch/case pattern).

```python
from aragora.workflow.nodes.decision import SwitchStep

step = SwitchStep(
    name="Route by Category",
    config={
        "value": "inputs.document_type",
        "cases": {
            "contract": "contract_review",
            "invoice": "invoice_processing",
            "policy": "policy_review",
        },
        "default": "general_review",
    }
)
```

### TaskStep

Generic task execution for transforms, HTTP, validation.

```python
from aragora.workflow.nodes import TaskStep

# Data transformation
transform_step = TaskStep(
    name="Extract Key Points",
    config={
        "task_type": "transform",
        "transform": "[p['content'] for p in inputs.paragraphs if p.get('important')]",
        "output_format": "list",
    }
)

# HTTP webhook
webhook_step = TaskStep(
    name="Notify Webhook",
    config={
        "task_type": "http",
        "url": "https://api.example.com/webhook",
        "method": "POST",
        "headers": {"Authorization": "Bearer {api_token}"},
        "body": {"result": "{step.analysis.summary}"},
        "timeout_seconds": 30,
    }
)

# Validation
validate_step = TaskStep(
    name="Validate Input",
    config={
        "task_type": "validate",
        "data": "inputs",
        "validation": {
            "email": {"required": True, "pattern": r"^[\w.-]+@[\w.-]+\.\w+$"},
            "amount": {"required": True, "type": "float", "min": 0, "max": 10000},
        },
    }
)
```

**Task Types:**
- `function`: Execute registered Python handler
- `http`: Make HTTP requests
- `transform`: Data transformation expressions
- `validate`: Validate data against rules
- `aggregate`: Combine outputs from multiple steps

## Knowledge Management Nodes

### KnowledgePipelineStep

Document ingestion and processing.

```python
from aragora.workflow.nodes import KnowledgePipelineStep

step = KnowledgePipelineStep(
    name="Ingest Contracts",
    config={
        "sources": ["/path/to/contracts/", "https://docs.example.com"],
        "workspace_id": "legal",
        "chunk_strategy": "semantic",  # semantic, sliding, recursive, sentence
        "chunk_size": 512,
        "chunk_overlap": 64,
        "extract_facts": True,
        "connector_type": "local_docs",
        "timeout_seconds": 600,
    }
)
```

### KnowledgePruningStep

Automatic knowledge maintenance.

```python
from aragora.workflow.nodes import KnowledgePruningStep

step = KnowledgePruningStep(
    name="Nightly Pruning",
    config={
        "workspace_id": "production",
        "staleness_threshold": 0.85,
        "min_age_days": 30,
        "action": "archive",  # archive, delete, demote, flag
        "dry_run": False,
        "max_items": 100,
        "tier_exceptions": ["glacial"],
    }
)
```

### KnowledgeDedupStep

Duplicate detection and merging.

```python
from aragora.workflow.nodes import KnowledgeDedupStep

step = KnowledgeDedupStep(
    name="Weekly Dedup",
    config={
        "workspace_id": "production",
        "similarity_threshold": 0.95,
        "auto_merge": True,
        "dry_run": False,
        "max_clusters": 50,
    }
)
```

### ConfidenceDecayStep

Time-based confidence decay.

```python
from aragora.workflow.nodes import ConfidenceDecayStep

step = ConfidenceDecayStep(
    name="Daily Decay",
    config={
        "workspace_id": "production",
        "decay_rate": 0.005,      # 0.5% per day
        "min_confidence": 0.1,
    }
)
```

## Self-Improvement Nodes

### NomicLoopStep

Execute Nomic Loop self-improvement phases.

```python
from aragora.workflow.nodes import NomicLoopStep

step = NomicLoopStep(
    name="Self-Improvement Cycle",
    config={
        "cycles": 1,
        "phases": ["context", "debate", "design"],  # Available: context, debate, design, implement, verify, commit
        "workspace_id": "development",
        "enable_code_execution": False,  # Set True for implement/verify/commit
        "require_approval": True,
        "agents": ["claude", "gpt4"],
        "recovery_enabled": True,
        "max_retries": 3,
    }
)
```

**Phases:**
1. `context`: Gather codebase understanding
2. `debate`: Agents propose improvements
3. `design`: Architecture planning
4. `implement`: Code generation
5. `verify`: Tests and validation
6. `commit`: Commit changes

## Gateway Nodes

### OpenClawSessionStep

Manage Enterprise Gateway proxy sessions.

```python
from aragora.workflow.nodes import OpenClawSessionStep

create_session = OpenClawSessionStep(
    name="Create Session",
    config={
        "operation": "create",
        "workspace_id": "/workspace/project",
        "roles": ["developer"],
    }
)
```

### OpenClawActionStep

Execute actions through the Enterprise Gateway.

```python
from aragora.workflow.nodes import OpenClawActionStep

# Shell command
shell_step = OpenClawActionStep(
    name="List Files",
    config={
        "action_type": "shell",
        "session_id": "{step.create_session.session_id}",
        "command": "ls -la /workspace",
        "timeout_seconds": 30,
    }
)

# File operations
read_step = OpenClawActionStep(
    name="Read Config",
    config={
        "action_type": "file_read",
        "session_id": "{step.create_session.session_id}",
        "path": "/workspace/config.yaml",
    }
)

# Browser automation
browse_step = OpenClawActionStep(
    name="Fetch Page",
    config={
        "action_type": "browser",
        "session_id": "{step.create_session.session_id}",
        "url": "https://example.com",
    }
)
```

**Action Types:**
- `shell`: Execute shell commands
- `file_read`: Read file contents
- `file_write`: Write file contents
- `file_delete`: Delete files
- `browser`: Browser automation
- `screenshot`: Capture screenshots
- `api`: Custom API calls

## Creating Custom Nodes

### Step 1: Create the Node Class

```python
# myapp/workflow/nodes/custom.py
from aragora.workflow.step import BaseStep, WorkflowContext
from typing import Any, Optional

class CustomStep(BaseStep):
    """Custom workflow step for specific business logic."""

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None):
        super().__init__(name, config)
        self._internal_state = {}

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the custom step."""
        # Merge instance config with step-specific config
        config = {**self._config, **context.current_step_config}

        # Get inputs with template interpolation
        input_value = self._interpolate(config.get("input_key", ""), context)

        # Perform your logic
        result = await self._do_work(input_value)

        # Store intermediate state if needed
        context.set_state(f"{self.name}_result", result)

        return {
            "success": True,
            "result": result,
        }

    def _interpolate(self, template: str, context: WorkflowContext) -> str:
        """Replace {placeholders} with context values."""
        text = template
        for key, value in context.inputs.items():
            text = text.replace(f"{{{key}}}", str(value))
        for step_id, output in context.step_outputs.items():
            if isinstance(output, dict) and "result" in output:
                text = text.replace(f"{{step.{step_id}}}", str(output["result"]))
        return text

    async def checkpoint(self) -> dict[str, Any]:
        """Save state for workflow checkpointing."""
        return {"internal_state": self._internal_state}

    async def restore(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self._internal_state = state.get("internal_state", {})

    def validate_config(self) -> bool:
        """Validate step configuration."""
        required = ["input_key"]
        return all(k in self._config for k in required)

    async def _do_work(self, value: str) -> str:
        """Internal method for business logic."""
        return f"Processed: {value}"
```

### Step 2: Register the Node

```python
# myapp/workflow/__init__.py
from aragora.workflow.nodes import register_step_type
from myapp.workflow.nodes.custom import CustomStep

# Register for use in workflow definitions
register_step_type("custom", CustomStep)
```

### Step 3: Use in Workflows

```python
from aragora.workflow import WorkflowDefinition, StepDefinition

definition = WorkflowDefinition(
    id="my_workflow",
    name="My Custom Workflow",
    steps=[
        StepDefinition(
            id="custom_step",
            name="Run Custom Logic",
            step_type="custom",
            config={
                "input_key": "{document_text}",
            },
            next_steps=["next_step"],
        ),
    ],
    entry_step="custom_step",
)
```

## Examples

### Complete Workflow with Multiple Nodes

```python
from aragora.workflow import WorkflowEngine, WorkflowDefinition, StepDefinition
from aragora.workflow.nodes import (
    MemoryReadStep,
    DebateStep,
    HumanCheckpointStep,
    MemoryWriteStep,
    ConnectorStep,
)

definition = WorkflowDefinition(
    id="contract_review",
    name="Contract Review Workflow",
    steps=[
        # 1. Retrieve relevant knowledge
        StepDefinition(
            id="retrieve_context",
            name="Retrieve Contract Knowledge",
            step_type="memory_read",
            config={
                "query": "Contract terms for {contract_type}",
                "domain_filter": "legal",
                "limit": 10,
            },
            next_steps=["debate"],
        ),
        # 2. Multi-agent debate
        StepDefinition(
            id="debate",
            name="Contract Analysis Debate",
            step_type="debate",
            config={
                "topic": "Analyze the contract: {contract_text}",
                "agents": ["legal_analyst", "risk_assessor"],
                "rounds": 3,
            },
            next_steps=["approval"],
        ),
        # 3. Human approval
        StepDefinition(
            id="approval",
            name="Legal Review",
            step_type="human_checkpoint",
            config={
                "title": "Contract Analysis Review",
                "checklist": [
                    {"label": "Risk assessment verified", "required": True},
                    {"label": "Compliance confirmed", "required": True},
                ],
                "timeout_seconds": 7200,
            },
            next_steps=["store_result"],
        ),
        # 4. Store results
        StepDefinition(
            id="store_result",
            name="Store Analysis",
            step_type="memory_write",
            config={
                "content": "{step.debate.synthesis}",
                "source_type": "consensus",
                "domain": "legal/analysis",
            },
            next_steps=["notify"],
        ),
        # 5. Notify via Slack
        StepDefinition(
            id="notify",
            name="Notify Team",
            step_type="connector",
            config={
                "connector_type": "slack",
                "operation": "create",
                "params": {
                    "channel": "#legal-reviews",
                    "message": "Contract review completed for {contract_name}",
                },
            },
        ),
    ],
    entry_step="retrieve_context",
)

# Execute
engine = WorkflowEngine()
result = await engine.execute(
    definition,
    inputs={
        "contract_type": "SaaS",
        "contract_name": "Acme Corp Agreement",
        "contract_text": "...",
    }
)
```

## Related

- [Workflow Engine](../README.md) - Main workflow documentation
- [Workflow Patterns](../patterns/README.md) - Reusable workflow patterns
- [Workflow Templates](../templates/README.md) - Industry-specific templates
- [CLAUDE.md](../../../CLAUDE.md) - Project overview
