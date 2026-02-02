# Marketplace Module

Template registry and discovery system for Aragora, providing local storage and management of reusable agent, debate, and workflow templates with search, versioning, and rating capabilities.

## Overview

The marketplace module enables:

- **Template Registry**: SQLite-based local storage for templates with full CRUD operations
- **Template Types**: Support for Agent, Debate, and Workflow templates
- **Search and Discovery**: Query templates by category, type, author, tags, and text search
- **Versioning**: Track template versions with content hashing
- **Ratings and Reviews**: User ratings (1-5) with optional reviews
- **Import/Export**: JSON-based template sharing and portability
- **Built-in Templates**: Pre-configured templates for common use cases

## Architecture

```
aragora/marketplace/
├── __init__.py          # Module exports
├── models.py            # Template dataclasses and enums
├── registry.py          # SQLite-based template registry
├── client.py            # HTTP client for remote marketplaces
└── sync.py              # Template synchronization
```

## Key Classes

### Template Models

- **`AgentTemplate`**: Configuration for creating agents
  - System prompt, model config, capabilities, constraints
  - Examples for few-shot learning

- **`DebateTemplate`**: Configuration for debate sessions
  - Task template, agent roles, protocol settings
  - Evaluation criteria and success metrics

- **`WorkflowTemplate`**: DAG-based workflow configuration
  - Nodes, edges, inputs, outputs, variables

### Metadata and Categories

- **`TemplateMetadata`**: Common metadata (id, name, version, author, tags)
- **`TemplateCategory`**: Categories (ANALYSIS, CODING, CREATIVE, DEBATE, RESEARCH, DECISION, BRAINSTORM, REVIEW, PLANNING, CUSTOM)
- **`TemplateRating`**: User rating with score (1-5) and optional review

### Registry

- **`TemplateRegistry`**: Local SQLite storage with:
  - `register(template)`: Add or update a template
  - `get(template_id)`: Retrieve by ID
  - `search(query, category, type, author, tags)`: Search templates
  - `rate(rating)`: Add user rating
  - `star(template_id)`: Increment star count
  - `delete(template_id)`: Remove non-builtin templates
  - `export_template(template_id)`: Export as JSON
  - `import_template(json_str)`: Import from JSON

## Usage Example

### Using Built-in Templates

```python
from aragora.marketplace import (
    TemplateRegistry,
    TemplateCategory,
    AgentTemplate,
    DebateTemplate,
)
from aragora import Arena, Environment, DebateProtocol, Agent

# Create registry (loads built-in templates automatically)
registry = TemplateRegistry()

# Get a built-in agent template
devil_advocate = registry.get("devil-advocate")
print(f"Template: {devil_advocate.metadata.name}")
print(f"System prompt: {devil_advocate.system_prompt[:100]}...")

# Create an agent from template
agent = Agent(
    name="devils-advocate-1",
    provider="anthropic",
    system_prompt=devil_advocate.system_prompt,
    capabilities=devil_advocate.capabilities,
)

# Get a debate template
oxford_debate = registry.get("oxford-style")
print(f"Debate: {oxford_debate.metadata.name}")
print(f"Roles: {[r['role'] for r in oxford_debate.agent_roles]}")

# Use debate template to configure arena
protocol = DebateProtocol(
    rounds=oxford_debate.protocol["rounds"],
    consensus_mode=oxford_debate.protocol["consensus_mode"],
)
```

### Creating Custom Templates

```python
from aragora.marketplace import (
    TemplateRegistry,
    AgentTemplate,
    DebateTemplate,
    TemplateMetadata,
    TemplateCategory,
)
from datetime import datetime

registry = TemplateRegistry()

# Create a custom agent template
security_reviewer = AgentTemplate(
    metadata=TemplateMetadata(
        id="security-reviewer-v1",
        name="Security Code Reviewer",
        description="Reviews code for security vulnerabilities and best practices",
        version="1.0.0",
        author="your-org",
        category=TemplateCategory.CODING,
        tags=["security", "code-review", "vulnerabilities", "best-practices"],
    ),
    agent_type="claude",
    system_prompt="""You are a Security Code Reviewer. Your role is to:
1. Identify security vulnerabilities (OWASP Top 10)
2. Check for authentication and authorization issues
3. Review input validation and sanitization
4. Detect sensitive data exposure risks
5. Suggest security improvements with examples

Always cite specific line numbers and provide remediation steps.""",
    capabilities=["vulnerability_detection", "code_analysis", "security_audit"],
    constraints=["must_cite_lines", "provide_severity_ratings"],
    model_config={"temperature": 0.2, "max_tokens": 4000},
)

# Register the template
template_id = registry.register(security_reviewer)
print(f"Registered template: {template_id}")

# Create a custom debate template
architecture_review = DebateTemplate(
    metadata=TemplateMetadata(
        id="architecture-review-v1",
        name="Architecture Review Session",
        description="Multi-perspective architecture review with specialists",
        version="1.0.0",
        author="your-org",
        category=TemplateCategory.REVIEW,
        tags=["architecture", "review", "scalability", "security"],
    ),
    task_template="Review this architecture:\n{architecture_description}",
    agent_roles=[
        {"role": "scalability_expert", "focus": "scalability"},
        {"role": "security_expert", "focus": "security"},
        {"role": "cost_analyst", "focus": "cost"},
        {"role": "maintainability_expert", "focus": "maintainability"},
        {"role": "synthesizer", "aggregates": True},
    ],
    protocol={
        "rounds": 3,
        "consensus_mode": "synthesis",
        "require_specific_feedback": True,
        "allow_rebuttals": True,
    },
    evaluation_criteria=["thoroughness", "practicality", "risk_identification"],
    success_metrics={"coverage": 0.9, "agreement": 0.7},
)

registry.register(architecture_review)
```

### Searching Templates

```python
from aragora.marketplace import TemplateRegistry, TemplateCategory

registry = TemplateRegistry()

# Search by text query
results = registry.search(query="code review")
for template in results:
    print(f"{template.metadata.name} - {template.metadata.description}")

# Search by category
coding_templates = registry.search(category=TemplateCategory.CODING)

# Search by type
debate_templates = registry.search(template_type="DebateTemplate")

# Search by author
our_templates = registry.search(author="your-org")

# Search by tags
security_templates = registry.search(tags=["security", "vulnerabilities"])

# Combined search
results = registry.search(
    query="review",
    category=TemplateCategory.REVIEW,
    tags=["architecture"],
    limit=10,
)
```

### Ratings and Reviews

```python
from aragora.marketplace import TemplateRegistry, TemplateRating

registry = TemplateRegistry()

# Rate a template
rating = TemplateRating(
    user_id="user-123",
    template_id="devil-advocate",
    score=5,
    review="Excellent for challenging assumptions. Really improved our decision quality.",
)
registry.rate(rating)

# Get ratings for a template
ratings = registry.get_ratings("devil-advocate")
for r in ratings:
    print(f"Score: {r.score}/5 - {r.review}")

# Get average rating
avg = registry.get_average_rating("devil-advocate")
print(f"Average rating: {avg:.1f}/5")

# Star a template
registry.star("devil-advocate")
```

### Import/Export Templates

```python
from aragora.marketplace import TemplateRegistry

registry = TemplateRegistry()

# Export a template
json_str = registry.export_template("security-reviewer-v1")
print(json_str)

# Save to file
with open("security-reviewer.json", "w") as f:
    f.write(json_str)

# Import from file
with open("security-reviewer.json", "r") as f:
    json_str = f.read()
template_id = registry.import_template(json_str)
print(f"Imported: {template_id}")
```

## Built-in Templates

### Agent Templates

| ID | Name | Category | Description |
|----|------|----------|-------------|
| `devil-advocate` | Devil's Advocate | DEBATE | Challenges assumptions and presents counterarguments |
| `code-reviewer` | Code Reviewer | CODING | Reviews code for quality, security, and best practices |
| `research-analyst` | Research Analyst | RESEARCH | Conducts thorough research and synthesizes information |

### Debate Templates

| ID | Name | Category | Description |
|----|------|----------|-------------|
| `oxford-style` | Oxford-Style Debate | DEBATE | Formal debate with proposition and opposition teams |
| `brainstorm-session` | Brainstorm Session | BRAINSTORM | Collaborative ideation for creative solutions |
| `code-review-session` | Code Review Session | REVIEW | Multi-agent code review with different perspectives |

## Integration Points

### With Debate Engine
- Templates provide pre-configured debate protocols
- Agent templates define roles and system prompts
- Evaluation criteria from templates used for scoring

### With Agent System
- Agent templates include model configuration
- Capabilities and constraints for agent behavior
- Examples for few-shot learning

### With Workflow Engine
- Workflow templates define DAG structure
- Node and edge configurations
- Input/output schemas

### With Observability
- Marketplace metrics (downloads, ratings)
- Template usage tracking
- Search analytics

## Database Schema

```sql
CREATE TABLE templates (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    version TEXT NOT NULL,
    author TEXT NOT NULL,
    category TEXT NOT NULL,
    tags TEXT,  -- JSON array
    content TEXT NOT NULL,  -- JSON
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    downloads INTEGER DEFAULT 0,
    stars INTEGER DEFAULT 0,
    is_builtin INTEGER DEFAULT 0
);

CREATE TABLE ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    score INTEGER NOT NULL,  -- 1-5
    review TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(template_id, user_id)
);
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_MARKETPLACE_DB` | Path to marketplace SQLite database | `~/.aragora/marketplace.db` |
| `ARAGORA_MARKETPLACE_URL` | Remote marketplace API URL | - |
| `ARAGORA_MARKETPLACE_TOKEN` | API token for remote marketplace | - |

## See Also

- `aragora/workflow/templates/` - Workflow template definitions
- `aragora/agents/` - Agent implementations
- `aragora/debate/` - Debate engine
- `docs/MARKETPLACE.md` - Full marketplace guide
