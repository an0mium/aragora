# Skills Module

Modular, pluggable capabilities for extending agent functionality with standardized interfaces.

## Quick Start

```python
from aragora.skills import (
    Skill,
    SkillManifest,
    SkillResult,
    SkillContext,
    SkillCapability,
    get_skill_registry,
)

# Define a skill
class MySkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="my_skill",
            version="1.0.0",
            capabilities=[SkillCapability.EXTERNAL_API],
            input_schema={"query": {"type": "string", "required": True}},
        )

    async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
        result = await self._do_work(input_data["query"])
        return SkillResult.create_success(result)

# Register and invoke
registry = get_skill_registry()
registry.register(MySkill())

result = await registry.invoke("my_skill", {"query": "test"}, SkillContext(user_id="user123"))
```

## Key Components

| File | Purpose |
|------|---------|
| `base.py` | Core types: Skill, SkillManifest, SkillResult, SkillContext |
| `registry.py` | SkillRegistry for discovery and invocation |
| `loader.py` | Dynamic skill loading from files/packages |
| `marketplace.py` | Skill marketplace for publishing/installing |
| `installer.py` | Skill package installation |
| `publisher.py` | Skill publishing utilities |
| `builtin/` | Built-in skill implementations |

## Architecture

```
skills/
├── base.py              # Core types and interfaces
│   ├── Skill            # Abstract base class
│   ├── SyncSkill        # Synchronous skill variant
│   ├── SkillManifest    # Declarative metadata
│   ├── SkillResult      # Typed result container
│   ├── SkillContext     # Execution context
│   └── SkillCapability  # Capability enumeration
├── registry.py          # Central registry
│   ├── SkillRegistry    # Registration and invocation
│   └── get_skill_registry()
├── loader.py            # Dynamic loading
│   ├── SkillLoader      # File/package loader
│   └── DeclarativeSkill # YAML-defined skills
├── marketplace.py       # Publishing ecosystem
│   ├── SkillMarketplace # Browse/search skills
│   ├── SkillListing     # Marketplace entry
│   └── InstallResult    # Installation outcome
├── installer.py         # Package management
└── builtin/             # Built-in skills
    ├── web_search.py
    ├── knowledge_query.py
    ├── evidence_fetch.py
    ├── fact_check.py
    ├── code_execution.py
    ├── data_extraction.py
    ├── file_analysis.py
    ├── summarization.py
    ├── translation.py
    ├── calculation.py
    └── openclaw_skill.py
```

## Skill Capabilities

Skills declare their capabilities in the manifest:

| Capability | Description |
|------------|-------------|
| `READ_LOCAL` | Read local files |
| `WRITE_LOCAL` | Write local files |
| `READ_DATABASE` | Query databases |
| `WRITE_DATABASE` | Modify databases |
| `EXTERNAL_API` | Call external APIs |
| `WEB_SEARCH` | Search the web |
| `WEB_FETCH` | Fetch web pages |
| `CODE_EXECUTION` | Execute code |
| `SHELL_EXECUTION` | Run shell commands |
| `LLM_INFERENCE` | Call LLM APIs |
| `EMBEDDING` | Generate embeddings |
| `DEBATE_CONTEXT` | Access debate context |
| `EVIDENCE_COLLECTION` | Collect evidence |
| `KNOWLEDGE_QUERY` | Query knowledge mound |

## Creating Skills

### Async Skill

```python
from aragora.skills import Skill, SkillManifest, SkillResult, SkillContext, SkillCapability

class WebSearchSkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="web_search",
            version="1.0.0",
            description="Search the web for information",
            capabilities=[SkillCapability.WEB_SEARCH, SkillCapability.EXTERNAL_API],
            input_schema={
                "query": {"type": "string", "required": True},
                "max_results": {"type": "integer", "default": 10},
            },
            output_schema={
                "results": {"type": "array"},
                "count": {"type": "integer"},
            },
            rate_limit=100,  # requests per minute
            timeout_seconds=30,
        )

    async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
        query = input_data["query"]
        max_results = input_data.get("max_results", 10)

        try:
            results = await self._search(query, max_results)
            return SkillResult.create_success({
                "results": results,
                "count": len(results),
            })
        except Exception as e:
            return SkillResult.create_failure(str(e))
```

### Sync Skill

```python
from aragora.skills import SyncSkill, SkillManifest, SkillResult, SkillContext

class CalculationSkill(SyncSkill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="calculate",
            version="1.0.0",
            capabilities=[],  # No special capabilities needed
            input_schema={"expression": {"type": "string", "required": True}},
        )

    def execute_sync(self, input_data: dict, context: SkillContext) -> SkillResult:
        expr = input_data["expression"]
        result = eval(expr)  # Use safe evaluation in production!
        return SkillResult.create_success({"result": result})
```

### Declarative Skill (YAML)

```yaml
# skills/my_skill.yaml
name: my_skill
version: 1.0.0
description: A declarative skill
capabilities:
  - external_api
input_schema:
  query:
    type: string
    required: true
handler: my_module.handler_function
```

## Registry Usage

```python
from aragora.skills import get_skill_registry, SkillContext

registry = get_skill_registry()

# Register a skill
registry.register(MySkill())

# List registered skills
skills = registry.list_skills()

# Invoke by name
result = await registry.invoke(
    "web_search",
    {"query": "Aragora multi-agent debate"},
    SkillContext(user_id="user123"),
)

# Get function schemas for LLM tool use
schemas = registry.get_function_schemas()
```

## Marketplace

```python
from aragora.skills import get_marketplace

marketplace = get_marketplace()

# Search for skills
listings = await marketplace.search("web search")

# Install a skill
result = await marketplace.install("community/enhanced-search", version="2.0.0")

# Publish your skill
await marketplace.publish(MySkill(), description="My awesome skill")
```

## Built-in Skills

| Skill | Capabilities | Description |
|-------|--------------|-------------|
| `web_search` | WEB_SEARCH, EXTERNAL_API | Search the web |
| `knowledge_query` | KNOWLEDGE_QUERY, READ_DATABASE | Query knowledge mound |
| `evidence_fetch` | EVIDENCE_COLLECTION, WEB_FETCH | Fetch evidence from URLs |
| `fact_check` | LLM_INFERENCE, WEB_SEARCH | Verify factual claims |
| `code_execution` | CODE_EXECUTION | Execute Python code safely |
| `data_extraction` | LLM_INFERENCE | Extract structured data |
| `file_analysis` | READ_LOCAL | Analyze local files |
| `summarization` | LLM_INFERENCE | Summarize text |
| `translation` | LLM_INFERENCE | Translate text |
| `calculation` | - | Evaluate math expressions |
| `openclaw` | EXTERNAL_API | OpenClaw gateway integration |

## LLM Function Calling

Skills integrate with LLM function calling:

```python
# Get schemas for tool use
schemas = registry.get_function_schemas()

# Use in LLM prompt
response = await llm.complete(
    prompt="Find information about...",
    tools=schemas,
)

# Execute the chosen skill
if response.tool_calls:
    for call in response.tool_calls:
        result = await registry.invoke(
            call.function.name,
            call.function.arguments,
            context,
        )
```

## Integration Points

| Module | Integration |
|--------|-------------|
| `aragora.debate` | Skills invoked during debates |
| `aragora.agents` | Agents use skills for capabilities |
| `aragora.rbac` | Permission checking via SkillContext |
| `aragora.mcp` | Skills exposed as MCP tools |

## Related

- `aragora/mcp/` - Model Context Protocol server
- `aragora/agents/` - Agent implementations
- `aragora/knowledge/` - Knowledge mound integration
- `docs/SKILLS_GUIDE.md` - Detailed skill development guide
