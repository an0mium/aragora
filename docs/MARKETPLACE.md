# Agent Template Marketplace

The Aragora Marketplace enables sharing and discovering reusable templates for agents, debates, and workflows across the community.

## Overview

The marketplace provides:

- **Agent Templates**: Pre-configured agent personalities with system prompts, capabilities, and constraints
- **Debate Templates**: Structured debate formats with roles, protocols, and evaluation criteria
- **Workflow Templates**: DAG-based automation workflows with inputs, outputs, and node configurations

## Quick Start

### Using the Local Registry

```python
from aragora.marketplace import TemplateRegistry, AgentTemplate, TemplateMetadata, TemplateCategory

# Initialize registry (uses ~/.aragora/marketplace.db by default)
registry = TemplateRegistry()

# List built-in templates
templates = registry.search(category=TemplateCategory.DEBATE)
for t in templates:
    print(f"{t.metadata.name}: {t.metadata.description}")

# Get a specific template
template = registry.get("devil-advocate")
print(template.system_prompt)
```

### Creating Custom Templates

```python
from aragora.marketplace import AgentTemplate, TemplateMetadata, TemplateCategory

# Create a custom agent template
my_template = AgentTemplate(
    metadata=TemplateMetadata(
        id="my-analyst",
        name="Data Analyst",
        description="An agent specialized in data analysis and visualization",
        version="1.0.0",
        author="your-username",
        category=TemplateCategory.ANALYSIS,
        tags=["data", "analysis", "visualization"],
    ),
    agent_type="claude",
    system_prompt="""You are a Data Analyst. Your role is to:
1. Analyze datasets for patterns and insights
2. Create clear visualizations
3. Provide statistical summaries
4. Recommend data-driven decisions

Always cite your methodology and acknowledge limitations.""",
    capabilities=["data_analysis", "visualization", "statistics"],
    constraints=["must_cite_sources", "acknowledge_limitations"],
)

# Register the template
registry.register(my_template)
```

### Searching Templates

```python
# Search by query
results = registry.search(query="code review")

# Search by category
results = registry.search(category=TemplateCategory.CODING)

# Search by tags
results = registry.search(tags=["security", "review"])

# Combined search with pagination
results = registry.search(
    query="analysis",
    category=TemplateCategory.RESEARCH,
    limit=10,
    offset=0,
)
```

## Built-in Templates

### Agent Templates

| ID | Name | Description |
|----|------|-------------|
| `devil-advocate` | Devil's Advocate | Challenges assumptions and presents counterarguments |
| `code-reviewer` | Code Reviewer | Reviews code for quality, security, and best practices |
| `research-analyst` | Research Analyst | Conducts thorough research and synthesizes information |

### Debate Templates

| ID | Name | Description |
|----|------|-------------|
| `oxford-style` | Oxford-Style Debate | Formal debate with proposing and opposing teams |
| `brainstorm-session` | Brainstorm Session | Collaborative ideation with no-criticism phase |
| `code-review-session` | Code Review Session | Multi-agent code review with different perspectives |

## Template Types

### AgentTemplate

```python
@dataclass
class AgentTemplate:
    metadata: TemplateMetadata
    agent_type: str              # e.g., "claude", "gpt4", "custom"
    system_prompt: str           # The agent's system prompt
    model_config: dict           # Model-specific configuration
    capabilities: list[str]      # What the agent can do
    constraints: list[str]       # Behavioral constraints
    examples: list[dict]         # Example interactions
```

### DebateTemplate

```python
@dataclass
class DebateTemplate:
    metadata: TemplateMetadata
    task_template: str           # Template with {placeholders}
    agent_roles: list[dict]      # Role definitions
    protocol: dict               # Debate protocol settings
    evaluation_criteria: list[str]
    success_metrics: dict[str, float]
```

### WorkflowTemplate

```python
@dataclass
class WorkflowTemplate:
    metadata: TemplateMetadata
    nodes: list[dict]            # Workflow nodes
    edges: list[dict]            # Node connections
    inputs: dict                 # Input definitions
    outputs: dict                # Output definitions
    variables: dict              # Template variables
```

## Ratings and Downloads

```python
from aragora.marketplace import TemplateRating

# Rate a template (1-5 stars)
rating = TemplateRating(
    user_id="user-123",
    template_id="devil-advocate",
    score=5,
    review="Excellent for critical thinking exercises!",
)
registry.rate(rating)

# Get average rating
avg = registry.get_average_rating("devil-advocate")
print(f"Average rating: {avg}")

# Track downloads
registry.increment_downloads("devil-advocate")
```

## Import/Export

```python
# Export a template to JSON
json_str = registry.export_template("my-analyst")
with open("my-template.json", "w") as f:
    f.write(json_str)

# Import a template from JSON
with open("shared-template.json") as f:
    template_id = registry.import_template(f.read())
```

## Remote Marketplace Client

For sharing templates with the community:

```python
from aragora.marketplace import MarketplaceClient, MarketplaceConfig

# Configure client
config = MarketplaceConfig(
    base_url="https://marketplace.aragora.ai/api/v1",
    api_key="your-api-key",
)

async with MarketplaceClient(config) as client:
    # Search remote marketplace
    templates = await client.search_templates(
        query="code review",
        category=TemplateCategory.CODING,
    )

    # Download a template
    template = await client.download_template("popular-template-id")

    # Publish your template
    await client.publish_template(my_template)

    # Get featured templates
    featured = await client.get_featured(limit=10)

    # Star a template
    await client.star_template("template-id")
```

## Best Practices

### Template Design

1. **Clear Purpose**: Define a specific use case for your template
2. **Detailed Prompts**: Include comprehensive system prompts with examples
3. **Appropriate Constraints**: Add constraints that enforce desired behavior
4. **Good Documentation**: Write clear descriptions and add relevant tags

### Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Document breaking changes in new major versions
- Keep backward compatibility when possible

### Community Guidelines

- Test templates before publishing
- Provide example usage in descriptions
- Respond to user feedback and ratings
- Update templates based on community input

## API Reference

See the full API documentation at `/api/v1/templates` endpoints:

- `GET /api/v1/templates` - List/search templates
- `GET /api/v1/templates/{id}` - Get template by ID
- `POST /api/v1/templates` - Publish a template
- `PUT /api/v1/templates/{id}` - Update a template
- `DELETE /api/v1/templates/{id}` - Delete a template
- `POST /api/v1/templates/{id}/ratings` - Rate a template
- `POST /api/v1/templates/{id}/star` - Star a template
