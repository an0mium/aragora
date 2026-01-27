# Skills System

Aragora’s skills system provides a modular, pluggable way to extend capabilities
with clear manifests, permissions, and execution constraints.

Skills are used by agents, workflows, and handlers to access external services,
run tools, or query internal systems in a controlled and auditable way.

## Core Concepts

### SkillManifest

Declarative metadata for discovery, permissions, and validation.

```python
from aragora.skills import SkillManifest, SkillCapability

manifest = SkillManifest(
    name="web_search",
    version="1.0.0",
    capabilities=[SkillCapability.WEB_SEARCH, SkillCapability.EXTERNAL_API],
    input_schema={"query": {"type": "string", "required": True}},
    description="Search the web",
    required_permissions=["skills:web_search"],
    max_execution_time_seconds=30.0,
)
```

### SkillResult

Typed result container with timing and metadata.

```python
from aragora.skills import SkillResult

return SkillResult.success({"results": [...]})
```

### SkillContext

Execution context containing actor, workspace, and optional debate context.

```python
from aragora.skills import SkillContext

context = SkillContext(
    actor_id="user-123",
    workspace_id="ws-001",
    debate_id="debate-456",
    permissions={"skills:web_search"},
)
```

## Implementing a Skill

```python
from aragora.skills import Skill, SkillManifest, SkillCapability, SkillResult, SkillContext

class WebSearchSkill(Skill):
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="web_search",
            version="1.0.0",
            capabilities=[SkillCapability.WEB_SEARCH],
            input_schema={"query": {"type": "string", "required": True}},
        )

    async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
        query = input_data["query"]
        results = await self._search(query)
        return SkillResult.success({"results": results})
```

### SyncSkill

For synchronous implementations:

```python
from aragora.skills import SyncSkill

class StaticSkill(SyncSkill):
    def run(self, input_data: dict, context: SkillContext):
        return {"answer": "static"}
```

## Registry and Loader

```python
from aragora.skills import get_skill_registry, SkillLoader

registry = get_skill_registry()

# Load built‑in skills
loader = SkillLoader()
loader.load_builtin_skills(register=True)

# Register a custom skill instance
registry.register(WebSearchSkill())
```

## Capabilities

Common capability categories:
- Data access: `read_local`, `write_local`, `read_database`, `write_database`
- External: `external_api`, `web_search`, `web_fetch`
- Execution: `code_execution`, `shell_execution`
- AI: `llm_inference`, `embedding`
- Debate: `debate_context`, `evidence_collection`, `knowledge_query`
- System: `system_info`, `network`

## Security & Permissions

Skills declare `required_permissions` and `required_env_vars` in their manifest.
Handlers or agents should check these before execution.

Example permission keys:
- `skills:web_search`
- `skills:code_execution`
- `skills:knowledge_query`

## Best Practices

- Prefer explicit `input_schema` and `output_schema`.
- Enforce timeouts with `max_execution_time_seconds`.
- Avoid logging raw secrets; rely on structured logging redaction.
- Mark `debate_compatible=False` for skills that should not run mid‑debate.

