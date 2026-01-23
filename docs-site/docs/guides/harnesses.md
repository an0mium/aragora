---
title: Harnesses Integration Guide
description: Harnesses Integration Guide
---

# Harnesses Integration Guide

## Overview

Aragora's harness system provides a unified interface for integrating external code analysis tools. Harnesses wrap tools like Claude Code CLI and Codex to perform security audits, code quality reviews, and architectural analysis.

**Key Benefit**: Run multiple analysis tools through a consistent API, aggregate findings, and convert results to Aragora's audit format.

## Installation

```bash
# Core Aragora (includes harness framework)
pip install aragora

# Ensure Claude Code CLI is installed for ClaudeCodeHarness
npm install -g @anthropic/claude-code
```

## Quick Start

### Basic Analysis

```python
from pathlib import Path
from aragora.harnesses import ClaudeCodeHarness, AnalysisType

# Create harness
harness = ClaudeCodeHarness()

# Initialize (checks CLI availability)
await harness.initialize()

# Analyze repository
result = await harness.analyze_repository(
    repo_path=Path("/path/to/repo"),
    analysis_type=AnalysisType.SECURITY,
)

# Process findings
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}")
    print(f"  File: {finding.file_path}:{finding.line_start}")
    print(f"  {finding.description}")
```

### Converting to Audit Findings

```python
from aragora.harnesses import adapt_to_audit_findings

# Run analysis
result = await harness.analyze_repository(repo_path, AnalysisType.SECURITY)

# Convert to Aragora audit format
audit_findings = adapt_to_audit_findings(result)

# Use with document auditor
for finding in audit_findings:
    print(f"[{finding.severity.value}] {finding.title}")
    print(f"  Type: {finding.audit_type.value}")
    print(f"  Recommendation: {finding.recommendation}")
```

## Analysis Types

| Type | Description | Use Case |
|------|-------------|----------|
| `SECURITY` | Vulnerabilities, secrets, injections | Security audits |
| `QUALITY` | Code smells, duplication, complexity | Code reviews |
| `PERFORMANCE` | Bottlenecks, inefficiencies | Optimization |
| `ARCHITECTURE` | Structure, patterns, SOLID | Design reviews |
| `DEPENDENCIES` | Outdated, vulnerable packages | Supply chain |
| `DOCUMENTATION` | Missing/outdated docs | Doc audits |
| `TESTING` | Coverage, test quality | Test reviews |
| `GENERAL` | Comprehensive review | Initial scan |

## Harness Implementations

### ClaudeCodeHarness

Integrates with Claude Code CLI for AI-powered analysis.

```python
from aragora.harnesses import ClaudeCodeHarness, ClaudeCodeConfig

# Custom configuration
config = ClaudeCodeConfig(
    claude_code_path="claude",           # CLI path
    model="claude-sonnet-4-20250514",    # Model to use
    timeout_seconds=600,                  # 10 minute timeout
    max_thinking_tokens=10000,
    include_file_contents=True,
    parse_structured_output=True,
)

harness = ClaudeCodeHarness(config)
```

#### Custom Prompts

```python
# Use custom analysis prompt
result = await harness.analyze_repository(
    repo_path=Path("./my-project"),
    analysis_type=AnalysisType.SECURITY,
    prompt="""Analyze this Python codebase for:
    1. SQL injection vulnerabilities
    2. Hardcoded API keys
    3. Insecure deserialization

    Format findings as JSON array with severity, file, line, description.""",
)
```

#### Streaming Analysis

```python
# Stream output in real-time
async for chunk in harness.stream_analysis(
    repo_path=Path("./my-project"),
    analysis_type=AnalysisType.QUALITY,
):
    print(chunk, end="", flush=True)
```

### Interactive Sessions

For multi-turn conversations about a codebase:

```python
from aragora.harnesses import ClaudeCodeHarness, SessionContext

harness = ClaudeCodeHarness()
await harness.initialize()

# Start session
context = SessionContext(
    session_id="review-001",
    repo_path=Path("./my-project"),
    files_in_context=["src/auth.py", "src/api.py"],
)

# Initial analysis
session = await harness.start_interactive_session(context)
print(session.response)

# Follow-up questions
result = await harness.continue_session(
    context,
    "What authentication method is being used?"
)
print(result.response)

# End session
await harness.end_session(context)
```

## Configuration

### HarnessConfig (Base)

```python
from aragora.harnesses import HarnessConfig

config = HarnessConfig(
    # Execution
    timeout_seconds=300,           # Max execution time
    max_retries=2,                 # Retry on failure
    retry_delay_seconds=1.0,

    # Output
    verbose=False,
    stream_output=True,
    capture_stderr=True,

    # Resource limits
    max_file_size_mb=10,           # Skip large files
    max_files=1000,                # Limit file count
    max_output_size_mb=50,

    # File patterns
    include_patterns=["**/*"],
    exclude_patterns=[
        "**/.git/**",
        "**/node_modules/**",
        "**/__pycache__/**",
        "**/venv/**",
    ],
)
```

### ClaudeCodeConfig

```python
from aragora.harnesses import ClaudeCodeConfig, AnalysisType

config = ClaudeCodeConfig(
    # Inherit base config options...
    timeout_seconds=600,

    # Claude Code specific
    claude_code_path="claude",
    model="claude-sonnet-4-20250514",
    max_thinking_tokens=10000,
    include_file_contents=True,
    use_mcp_tools=True,
    parse_structured_output=True,
    extract_code_blocks=True,

    # Custom prompts per analysis type
    analysis_prompts={
        AnalysisType.SECURITY.value: "Custom security prompt...",
        AnalysisType.QUALITY.value: "Custom quality prompt...",
    },
)
```

## Result Adaptation

### Adapter Configuration

```python
from aragora.harnesses import HarnessResultAdapter, AdapterConfig

config = AdapterConfig(
    # Severity mapping
    severity_mapping={
        "critical": "critical",
        "high": "high",
        "medium": "medium",
        "low": "low",
        "warning": "medium",  # Map warning to medium
    },

    # Analysis type to audit type
    type_mapping={
        "security": "security",
        "quality": "quality",
        "architecture": "consistency",
    },

    # Confidence adjustments by harness
    confidence_adjustments={
        "claude-code": 0.0,    # No adjustment
        "codex": -0.05,        # Slight decrease
    },

    min_confidence=0.5,
    id_prefix="harness",
)

adapter = HarnessResultAdapter(config)
findings = adapter.adapt(result)
```

### Multi-Harness Analysis

```python
from aragora.harnesses import (
    ClaudeCodeHarness,
    adapt_multiple_results,
)

# Run multiple harnesses
harnesses = [ClaudeCodeHarness()]

results = []
for harness in harnesses:
    await harness.initialize()
    result = await harness.analyze_repository(
        repo_path,
        AnalysisType.SECURITY,
    )
    results.append(result)

# Combine and deduplicate findings
all_findings = adapt_multiple_results(
    results,
    merge_duplicates=True,  # Deduplicate across harnesses
)
```

## Data Classes

### AnalysisFinding

```python
@dataclass
class AnalysisFinding:
    id: str                           # Unique identifier
    title: str                        # Short title
    description: str                  # Detailed description
    severity: str                     # critical/high/medium/low/info
    confidence: float                 # 0.0-1.0
    category: str                     # Finding category
    file_path: str                    # Relative file path
    line_start: int | None = None     # Starting line
    line_end: int | None = None       # Ending line
    code_snippet: str = ""            # Relevant code
    recommendation: str = ""          # How to fix
    references: list[str] = []        # External references
    metadata: dict[str, Any] = {}     # Additional data
```

### HarnessResult

```python
@dataclass
class HarnessResult:
    harness: str                      # Harness name
    analysis_type: AnalysisType       # Type of analysis
    success: bool                     # Whether analysis succeeded
    findings: list[AnalysisFinding]   # List of findings

    # Execution metadata
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float

    # Stats
    files_analyzed: int
    lines_analyzed: int
    findings_by_severity: dict[str, int]

    # Output
    raw_output: str                   # Raw harness output
    error_output: str                 # Stderr output
    error_message: str | None         # Error if failed

    metadata: dict[str, Any]
```

## Creating Custom Harnesses

Extend `CodeAnalysisHarness` to integrate new tools:

```python
from aragora.harnesses import (
    CodeAnalysisHarness,
    HarnessConfig,
    HarnessResult,
    AnalysisType,
    AnalysisFinding,
)

class MyCustomHarness(CodeAnalysisHarness):
    """Custom harness for MyTool."""

    @property
    def name(self) -> str:
        return "my-tool"

    @property
    def supported_analysis_types(self) -> list[AnalysisType]:
        return [AnalysisType.SECURITY, AnalysisType.QUALITY]

    async def initialize(self) -> bool:
        # Check tool availability
        # Return True if ready
        self._initialized = True
        return True

    async def analyze_repository(
        self,
        repo_path: Path,
        analysis_type: AnalysisType = AnalysisType.GENERAL,
        prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> HarnessResult:
        self._validate_path(repo_path)

        # Run your tool
        findings = []
        raw_output = ""

        # ... tool execution logic ...

        return HarnessResult(
            harness=self.name,
            analysis_type=analysis_type,
            success=True,
            findings=findings,
            raw_output=raw_output,
        )

    async def analyze_files(
        self,
        files: list[Path],
        analysis_type: AnalysisType = AnalysisType.GENERAL,
        prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> HarnessResult:
        # Implement file-specific analysis
        ...
```

## Error Handling

```python
from aragora.harnesses import (
    HarnessError,
    HarnessTimeoutError,
    HarnessConfigError,
)

try:
    result = await harness.analyze_repository(repo_path)
except HarnessTimeoutError as e:
    print(f"Analysis timed out: \{e\}")
    # Handle timeout - maybe increase timeout or reduce scope
except HarnessConfigError as e:
    print(f"Configuration error: \{e\}")
    # Handle config issues - check tool installation
except HarnessError as e:
    print(f"Harness error: {e.harness} - \{e\}")
    # Handle general harness errors
```

## Integration with Debates

Use harness findings as input to agent debates:

```python
from aragora.debate import DebateService, DebateOptions
from aragora.harnesses import ClaudeCodeHarness, adapt_to_audit_findings

# Run security analysis
harness = ClaudeCodeHarness()
await harness.initialize()
result = await harness.analyze_repository(repo_path, AnalysisType.SECURITY)

# Convert to findings
findings = adapt_to_audit_findings(result)

# Create debate task from findings
findings_summary = "\n".join(
    f"- [{f.severity.value}] {f.title}: {f.description[:100]}"
    for f in findings[:10]
)

# Debate the findings
service = DebateService()
debate_result = await service.run(
    task=f"""Review these security findings and prioritize fixes:

\{findings_summary\}

Which findings are most critical? What's the recommended fix order?""",
    options=DebateOptions(rounds=3),
)
```

## Best Practices

1. **Initialize harnesses** - Always call `initialize()` before analysis
2. **Handle timeouts** - Set appropriate timeouts for large repositories
3. **Use patterns** - Configure include/exclude patterns to focus analysis
4. **Merge duplicates** - Use `merge_duplicates=True` when combining results
5. **Stream for feedback** - Use `stream_analysis()` for long-running analyses
6. **Session cleanup** - Always call `end_session()` to free resources

## API Reference

### Exports from `aragora.harnesses`

```python
from aragora.harnesses import (
    # Harness implementations
    ClaudeCodeHarness,
    ClaudeCodeConfig,
    CodexHarness,       # If available

    # Base classes
    CodeAnalysisHarness,
    HarnessConfig,
    HarnessResult,

    # Data types
    AnalysisType,
    AnalysisFinding,
    SessionContext,
    SessionResult,

    # Errors
    HarnessError,
    HarnessTimeoutError,
    HarnessConfigError,

    # Adapters
    HarnessResultAdapter,
    AdapterConfig,
    adapt_to_audit_findings,
    adapt_multiple_results,
)
```

## Further Reading

- [Gauntlet](./gauntlet)
- [Debate Internals](../core-concepts/debate-internals)
- [Memory Strategy](../core-concepts/memory-strategy)
