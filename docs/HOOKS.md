# Hooks

> **Last Updated:** 2026-01-27

Declarative event hooks let you attach automation to debate and audit lifecycle
events without writing glue code. Hooks are configured in YAML, evaluated with
conditions, and executed through the existing `HookManager`.

## When to Use Hooks

- Trigger notifications when debates complete
- Store high-confidence outcomes as facts
- Emit audit or compliance metrics
- Run lightweight automation on lifecycle events

## Core Components

- **HookManager**: Registers and triggers lifecycle hooks.
- **HookConfigLoader**: Loads YAML hook configs and applies them to a manager.
- **ConditionEvaluator**: Evaluates conditional logic using dot-path fields.
- **Built-in actions**: Common handlers in `aragora.hooks.builtin`.

## Hook Triggers

Hook triggers are string keys (from `HookType`) grouped by lifecycle:

- **Debate**: `pre_debate`, `post_debate`, `pre_round`, `post_round`,
  `pre_phase`, `post_phase`
- **Agent**: `pre_generate`, `post_generate`, `pre_critique`, `post_critique`,
  `pre_vote`, `post_vote`
- **Consensus**: `pre_consensus`, `post_consensus`, `on_convergence`
- **Audit**: `on_finding`, `on_contradiction`, `on_inconsistency`,
  `on_evidence`, `on_progress`
- **Errors**: `on_error`, `on_timeout`, `on_cancellation`, `on_agent_error`
- **Session**: `on_pause`, `on_resume`, `on_checkpoint`
- **Propulsion**: `on_ready`, `on_propel`, `on_escalate`, `on_molecule_complete`

## YAML Format

```yaml
hooks:
  - name: log_debate_complete
    trigger: post_debate
    priority: low
    enabled: true
    one_shot: false
    description: "Log debate completion"
    conditions:
      - field: consensus_reached
        operator: eq
        value: true
    action:
      handler: aragora.hooks.builtin.log_event
      args:
        message: "Debate {debate_id} complete: confidence={confidence:.2%}"
        level: info
```

### Condition Operators

| Operator | Meaning |
|----------|---------|
| `eq`, `ne` | Equal / not equal |
| `gt`, `gte`, `lt`, `lte` | Numeric comparisons |
| `contains`, `not_contains` | String or collection contains |
| `starts_with`, `ends_with` | String prefix/suffix |
| `matches` | Regex match |
| `is_null`, `is_not_null` | None checks |
| `is_empty`, `is_not_empty` | Empty string/list/dict |
| `in`, `not_in` | Membership |
| `has_key` | Dict has key |
| `is_true`, `is_false` | Boolean checks |

Field paths support dot notation and list indexing (e.g. `finding.severity`,
`agents.0.name`).

### Action Fields

- `handler`: Fully-qualified callable path
- `args`: Dict of arguments to merge with trigger context
- `async_execution`: Execute asynchronously (default: true)
- `timeout`: Optional execution timeout in seconds

## Built-in Actions

| Action | Purpose |
|--------|---------|
| `log_event` | Log messages with context interpolation |
| `log_metric` | Emit metrics to logs |
| `send_webhook` | Send HTTP webhook payload |
| `send_slack_notification` | Post to Slack via webhook |
| `save_checkpoint` | Persist JSON checkpoints to disk |
| `store_fact` | Store facts in Knowledge Mound |
| `set_context_var` | Mutate context for downstream hooks |
| `delay_execution` | Sleep before continuing |

Slack notifications require `SLACK_WEBHOOK_URL`. Webhooks use `httpx`.

## Loading Hooks

```python
from aragora.debate.hooks import HookManager
from aragora.hooks import get_hook_loader

hook_manager = HookManager()
loader = get_hook_loader()
loader.discover_and_load("hooks/")  # Default directory for hook YAML
loader.apply_to_manager(hook_manager)
```

To wire hooks into debate execution, pass the manager into the arena builder:

```python
from aragora.debate.arena_builder import ArenaBuilder

arena = (
    ArenaBuilder()
    .with_hook_manager(hook_manager)
    .build()
)
```

## Examples

Reference configurations live in `aragora/hooks/examples/`:

- `debate_hooks.yaml`
- `audit_hooks.yaml`

Use them as templates for your own `hooks/*.yaml` files.

## Notes

- `on_agent_error` receives: `agent`, `error_type`, `message`, `recoverable`, `phase`.
- Hook config loading does not expand environment variables automatically.
  If you want `${VAR}` expansion, preprocess your YAML with
  `os.path.expandvars()` before loading.
- Hooks are evaluated in priority order (critical → cleanup).
