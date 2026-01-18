# Workflow Expression Security Model

This document describes the security model for dynamic expression evaluation in Aragora workflows.

## Overview

Aragora workflows support dynamic expressions for:
- **Conditional branching** - Deciding which path to take
- **Data transformation** - Transforming outputs between steps
- **Validation rules** - Validating data against business rules
- **Loop conditions** - Controlling iteration

These expressions use Python's `eval()` function with strict sandboxing to prevent arbitrary code execution.

## Security Architecture

### Sandboxed Evaluation

All expression evaluation uses a restricted execution environment:

```python
eval(expression, {"__builtins__": {}}, namespace)
```

**Key Security Features:**

1. **No Builtins Access** - `{"__builtins__": {}}` removes access to:
   - `import` / `__import__` - No module imports
   - `open` / `exec` / `compile` - No file/code execution
   - `eval` (recursive) - No nested evaluation
   - `globals` / `locals` - No scope escape
   - `getattr` / `setattr` / `delattr` - No attribute manipulation

2. **Controlled Namespace** - Only safe objects are exposed:
   ```python
   namespace = {
       "result": <previous_step_result>,
       "data": <workflow_data>,
       "state": <workflow_state>,
       "str": str,
       "int": int,
       "float": float,
       "len": len,
       "bool": bool,
   }
   ```

3. **Error Containment** - All evaluations are wrapped in try/except to prevent crashes

### What Expressions CAN Do

- Access workflow data: `result["items"]`, `state["counter"]`
- Type conversions: `int(result["count"])`, `str(value)`
- Comparisons: `result["status"] == "approved"`, `len(items) > 0`
- Boolean logic: `condition_a and condition_b`
- Attribute access on allowed objects: `result.get("key", default)`

### What Expressions CANNOT Do

- Import modules: `import os` - No `import` statement available
- Access files: `open("/etc/passwd")` - No `open` function
- Execute code: `exec("...")` - No `exec` function
- Network calls: `urllib.request.urlopen(...)` - No network modules
- System commands: `os.system(...)` - No `os` module
- Escape sandbox: `__builtins__.__dict__[...]` - Builtins dict is empty

## Expression Locations

| File | Line | Purpose |
|------|------|---------|
| `aragora/workflow/nodes/task.py` | 210 | Transform expressions |
| `aragora/workflow/nodes/task.py` | 238 | Data validation |
| `aragora/workflow/nodes/task.py` | 292 | Validation rules |
| `aragora/workflow/nodes/decision.py` | 166 | Conditional branching |
| `aragora/workflow/nodes/decision.py` | 288 | Switch value extraction |
| `aragora/workflow/nodes/human_checkpoint.py` | 277 | Auto-approval conditions |
| `aragora/workflow/step.py` | 267 | Conditional step execution |
| `aragora/workflow/step.py` | 332 | Loop conditions |
| `aragora/workflow/engine.py` | 508 | Transition conditions |

## Example Expressions

### Safe Expressions

```python
# Conditional branching
"result['status'] == 'approved' and result['score'] > 0.8"

# Data transformation
"[item['name'] for item in result['items'] if item['active']]"

# Validation
"len(value) >= 3 and len(value) <= 100"

# Loop condition
"state['iteration'] < 10 and not state.get('done', False)"
```

### Blocked Expressions

```python
# These will fail due to missing builtins:
"import os; os.system('rm -rf /')"      # No import
"open('/etc/passwd').read()"            # No open
"__builtins__['eval']('...')"          # Empty builtins
"().__class__.__bases__[0].__subclasses__()"  # No class introspection
```

## Audit Logging

Expression evaluation is logged for security auditing:

```python
logger.debug(f"Evaluating expression: {expression[:100]}...")
```

Failed evaluations are logged with warnings:

```python
logger.warning(f"Expression evaluation failed: {expression} -> {error}")
```

## Best Practices for Workflow Authors

1. **Keep expressions simple** - Complex logic belongs in task handlers
2. **Validate input data** - Don't assume data structure
3. **Use safe defaults** - `state.get("key", default)` instead of `state["key"]`
4. **Test expressions** - Verify behavior with edge cases
5. **Document expressions** - Add comments explaining complex conditions

## Security Considerations

### Denial of Service

While arbitrary code execution is prevented, resource exhaustion is possible:

```python
# Could cause memory issues:
"'x' * 10**9"

# Could cause CPU issues:
"[x**2 for x in range(10**8)]"
```

**Mitigation:** Workflow execution has configurable timeouts (`ARAGORA_WORKFLOW_TIMEOUT`).

### Information Disclosure

Expressions can access workflow state and results:

```python
# Could expose sensitive data in logs:
"str(state)"  # Logs entire state
```

**Mitigation:** Sensitive data should be marked and filtered from logs.

### Complexity Attacks

Deeply nested expressions could cause stack overflow:

```python
# Many nested operations
"((((((((((result))))))))))"
```

**Mitigation:** Expression length limits and timeout controls.

## Future Considerations

1. **AST-based evaluation** - Consider `ast.literal_eval()` for simple cases
2. **Expression allowlisting** - Pre-approve expression patterns
3. **Resource limits** - Add memory/CPU limits to expression evaluation
4. **Static analysis** - Lint expressions before execution

## Related Documentation

- [WORKFLOWS.md](./WORKFLOWS.md) - Workflow system overview
- [SECURITY.md](./SECURITY.md) - Overall security architecture
- [API_RATE_LIMITS.md](./API_RATE_LIMITS.md) - Rate limiting for workflow APIs
