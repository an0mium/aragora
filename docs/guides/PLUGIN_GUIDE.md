# Aragora Plugin Development Guide

This guide explains how to create plugins for the Aragora control plane.

## Overview

Plugins extend Aragora's capabilities by providing:
- Code analysis (linting, security scanning)
- Test execution
- Evidence collection
- Custom verification logic
- External tool integration

## Quick Start

### 1. Create a Plugin Directory

```bash
mkdir -p plugins/my-plugin
cd plugins/my-plugin
```

### 2. Create the Manifest

Create `manifest.json`:

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "My custom plugin",
  "author": "your-name",
  "capabilities": ["custom"],
  "requirements": ["read_files"],
  "entry_point": "main:run",
  "timeout_seconds": 30
}
```

### 3. Implement the Entry Point

Create `main.py`:

```python
from aragora.plugins.runner import PluginContext

async def run(context: PluginContext) -> dict:
    """Plugin entry point."""
    # Read input
    files = context.input_data.get("files", [])

    # Do work
    context.log(f"Processing {len(files)} files")
    results = []

    for f in files:
        results.append({"file": f, "status": "ok"})

    # Return output
    return {"results": results, "count": len(results)}
```

### 4. Run Your Plugin

```bash
# Via API
curl -X POST http://localhost:8080/api/plugins/my-plugin/run \
  -H "Content-Type: application/json" \
  -d '{"input": {"files": ["test.py"]}}'
```

## Plugin Manifest

The manifest declares what your plugin does and needs.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique plugin identifier (alphanumeric, hyphens, underscores) |
| `entry_point` | string | Module and function in `module:function` format |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | string | "1.0.0" | Semantic version |
| `description` | string | "" | Human-readable description |
| `author` | string | "unknown" | Plugin author |
| `capabilities` | list | [] | What the plugin can do |
| `requirements` | list | [] | What the plugin needs |
| `timeout_seconds` | float | 60 | Max execution time |
| `max_memory_mb` | int | 512 | Memory limit (soft) |
| `python_packages` | list | [] | Required pip packages |
| `system_tools` | list | [] | Required system binaries |
| `config_schema` | object | {} | JSON Schema for config validation |
| `default_config` | object | {} | Default configuration |

### Capabilities

Declare what your plugin can do:

| Capability | Description |
|------------|-------------|
| `code_analysis` | Analyze code structure |
| `lint` | Check code style |
| `security_scan` | Find vulnerabilities |
| `type_check` | Static type checking |
| `test_runner` | Execute tests |
| `benchmark` | Performance benchmarks |
| `formatter` | Code formatting |
| `evidence_fetch` | Gather external evidence |
| `documentation` | Generate/check docs |
| `formal_verify` | Formal verification |
| `property_check` | Property-based testing |
| `custom` | Custom capability |

### Requirements

Declare what your plugin needs:

| Requirement | Description |
|-------------|-------------|
| `read_files` | Read local files |
| `write_files` | Write local files |
| `run_commands` | Execute shell commands |
| `network` | Make network requests |
| `high_memory` | > 1GB RAM |
| `long_running` | > 60s execution |
| `python_packages` | External Python packages |
| `system_tools` | External system tools |

## Plugin Context

Your entry point receives a `PluginContext` with:

```python
@dataclass
class PluginContext:
    # Input from API request
    input_data: dict
    config: dict

    # Environment
    working_dir: str
    debate_id: Optional[str]

    # Allowed operations (set by runner)
    allowed_operations: set

    # Output (write to these)
    output: dict
    logs: list[str]
    errors: list[str]

    # Methods
    def log(self, message: str): ...
    def error(self, message: str): ...
    def set_output(self, key: str, value: Any): ...
    def can(self, operation: str) -> bool: ...
```

### Input Data

Access via `context.input_data`:

```python
files = context.input_data.get("files", [])
query = context.input_data.get("query", "")
```

### Configuration

Access via `context.config`:

```python
max_depth = context.config.get("max_depth", 10)
verbose = context.config.get("verbose", False)
```

### Logging

Use context methods for logging:

```python
context.log("Starting analysis")
context.error("Failed to process file")
```

### Checking Permissions

Check if operations are allowed:

```python
if context.can("read_files"):
    with open(path) as f:
        content = f.read()
else:
    context.error("File reading not permitted")
```

## Return Values

Return a dict from your entry point:

```python
async def run(context: PluginContext) -> dict:
    return {
        "success": True,
        "items_processed": 42,
        "results": [...],
    }
```

The returned dict is merged with `context.output`.

## File Access

Plugins can only access files under the working directory:

```python
async def run(context: PluginContext) -> dict:
    # This works - relative path
    with open("src/main.py") as f:
        content = f.read()

    # This fails - path traversal
    with open("../../../etc/passwd") as f:  # PermissionError
        ...
```

## Running External Tools

For tools that need shell execution:

```python
import asyncio
import shutil

async def run(context: PluginContext) -> dict:
    if not shutil.which("mycommand"):
        context.error("mycommand not installed")
        return {"error": "Missing dependency"}

    process = await asyncio.create_subprocess_exec(
        "mycommand", "--json", "input.txt",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=context.working_dir,
    )

    stdout, stderr = await process.communicate()
    return {"output": stdout.decode()}
```

## Example: Custom Analyzer

Complete example of a code complexity analyzer:

```python
"""
Complexity Analyzer Plugin

Analyzes Python code complexity using radon.
"""

import asyncio
import json
import shutil

from aragora.plugins.runner import PluginContext


async def run(context: PluginContext) -> dict:
    """Analyze code complexity."""

    # Check dependencies
    if not shutil.which("radon"):
        context.error("radon not installed. Run: pip install radon")
        return {"error": "Missing radon"}

    # Get input
    paths = context.input_data.get("paths", ["."])
    threshold = context.config.get("threshold", "C")  # A, B, C, D, E, F

    context.log(f"Analyzing complexity of: {', '.join(paths)}")

    # Run radon
    cmd = ["radon", "cc", "-j", "-n", threshold] + paths

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=context.working_dir,
    )
    stdout, _ = await process.communicate()

    # Parse results
    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError:
        return {"error": "Failed to parse radon output"}

    # Transform results
    functions = []
    for file_path, items in data.items():
        for item in items:
            functions.append({
                "file": file_path,
                "name": item.get("name", ""),
                "type": item.get("type", ""),
                "complexity": item.get("complexity", 0),
                "rank": item.get("rank", ""),
                "line": item.get("lineno", 0),
            })

    # Sort by complexity
    functions.sort(key=lambda x: x["complexity"], reverse=True)

    return {
        "functions": functions[:50],  # Top 50 most complex
        "total_analyzed": len(functions),
        "average_complexity": (
            sum(f["complexity"] for f in functions) / len(functions)
            if functions else 0
        ),
    }
```

Manifest for this plugin:

```json
{
  "name": "complexity-analyzer",
  "version": "1.0.0",
  "description": "Analyze code complexity using radon",
  "author": "aragora",
  "capabilities": ["code_analysis"],
  "requirements": ["read_files", "run_commands"],
  "entry_point": "analyzer:run",
  "timeout_seconds": 60,
  "python_packages": ["radon"],
  "config_schema": {
    "type": "object",
    "properties": {
      "threshold": {
        "type": "string",
        "enum": ["A", "B", "C", "D", "E", "F"],
        "description": "Minimum complexity rank to report"
      }
    }
  },
  "default_config": {
    "threshold": "C"
  },
  "tags": ["analysis", "complexity", "metrics"]
}
```

## API Reference

### List Plugins

```
GET /api/plugins
```

Response:
```json
{
  "plugins": [
    {
      "name": "lint",
      "version": "1.0.0",
      "description": "Check code for style issues",
      "capabilities": ["lint", "code_analysis"],
      ...
    }
  ],
  "count": 3
}
```

### Get Plugin Details

```
GET /api/plugins/{name}
```

Response includes manifest plus runtime info:
```json
{
  "name": "lint",
  "version": "1.0.0",
  ...
  "requirements_satisfied": true,
  "missing_requirements": []
}
```

### Run Plugin

```
POST /api/plugins/{name}/run
```

Request:
```json
{
  "input": {
    "files": ["src/"]
  },
  "config": {
    "max_line_length": 120
  },
  "working_dir": "."
}
```

Response:
```json
{
  "success": true,
  "output": {
    "issues": [...],
    "summary": {...}
  },
  "logs": ["[2024-01-01T12:00:00] Starting..."],
  "errors": [],
  "duration_seconds": 1.23,
  "plugin_name": "lint",
  "plugin_version": "1.0.0"
}
```

## Built-in Plugins

Aragora ships with these plugins:

| Name | Description | Requirements |
|------|-------------|--------------|
| `lint` | Code style checking | ruff or flake8 |
| `security-scan` | Security vulnerability scanning | bandit |
| `test-runner` | pytest execution | pytest |

## Security Considerations

1. **Sandboxing**: Plugins run with restricted builtins (no `exec`, `eval`, `compile`)
2. **File Access**: Limited to working directory
3. **Timeouts**: Enforced execution limits
4. **Memory**: Soft memory limits on Unix
5. **Permissions**: Capability-based access control

## Troubleshooting

### Plugin Not Found

Ensure the plugin is in a directory scanned by the registry:
- Built-in: `aragora/plugins/builtin/`
- External: Directories passed to `PluginRegistry`

### Missing Requirements

Check that all dependencies are installed:
```bash
pip install -e ".[dev]"  # Python packages
which ruff               # System tools
```

### Timeout Errors

Increase `timeout_seconds` in manifest or optimize plugin code.

### Permission Denied

Ensure your plugin declares necessary requirements:
- `read_files` for file reading
- `write_files` for file writing
- `run_commands` for shell execution
- `network` for HTTP requests

## Best Practices

1. **Validate Input**: Check input data before processing
2. **Handle Errors**: Use `context.error()` for user-facing errors
3. **Log Progress**: Use `context.log()` for debugging
4. **Respect Timeouts**: Break work into chunks for long operations
5. **Return Structured Data**: Use consistent output formats
6. **Document**: Include clear docstrings and examples
