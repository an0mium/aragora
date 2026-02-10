# Code Analysis Harnesses

The harnesses module (`aragora/harnesses/`) provides an abstraction layer for integrating with external code analysis tools. It enables consistent interaction with CLI-based and API-based code analysis systems.

## Overview

| Component | Purpose |
|-----------|---------|
| `CodeAnalysisHarness` | Abstract base class defining the interface |
| `HarnessConfig` | Configuration settings |
| `HarnessResult` | Analysis results container |
| `AnalysisFinding` | Individual finding data |
| `ClaudeCodeHarness` | Claude Code CLI integration |

## Architecture

```
aragora/harnesses/
├── base.py          # Abstract interface, config, result types
├── adapter.py       # Generic adapter patterns
├── claude_code.py   # Claude Code CLI harness
└── codex.py         # Codex harness (planned)
```

## Core Types

### HarnessConfig

Base configuration for all harnesses:

```python
@dataclass
class HarnessConfig:
    timeout_seconds: int = 300     # Max analysis time
    max_retries: int = 2           # Retry count on failure
    verbose: bool = False          # Detailed output
    stream_output: bool = True     # Real-time streaming
    max_file_size_mb: int = 10     # Skip large files
    max_files: int = 1000          # File count limit
    include_patterns: list[str]    # Glob patterns to include
    exclude_patterns: list[str]    # Glob patterns to exclude
```

Default exclude patterns skip common non-source directories:
- `**/.git/**`
- `**/node_modules/**`
- `**/__pycache__/**`
- `**/venv/**`

### AnalysisType

Types of analysis supported:

| Type | Description |
|------|-------------|
| `SECURITY` | Security vulnerability scanning |
| `QUALITY` | Code quality assessment |
| `PERFORMANCE` | Performance analysis |
| `ARCHITECTURE` | Architecture review |
| `DEPENDENCIES` | Dependency analysis |
| `DOCUMENTATION` | Documentation gaps |
| `TESTING` | Test coverage/quality |
| `GENERAL` | Comprehensive review |

### AnalysisFinding

Individual finding from analysis:

```python
@dataclass
class AnalysisFinding:
    id: str                    # Unique identifier
    title: str                 # Short description
    description: str           # Detailed explanation
    severity: str              # critical/high/medium/low/info
    confidence: float          # 0.0 to 1.0
    category: str              # Finding category
    file_path: str             # Affected file
    line_start: int | None     # Start line (optional)
    line_end: int | None       # End line (optional)
    code_snippet: str          # Relevant code
    recommendation: str        # Suggested fix
    references: list[str]      # External references
    metadata: dict[str, Any]   # Additional data
```

### HarnessResult

Complete analysis result:

```python
@dataclass
class HarnessResult:
    harness: str                      # Harness name
    analysis_type: AnalysisType       # Type performed
    success: bool                     # Operation success
    findings: list[AnalysisFinding]   # All findings
    started_at: datetime              # Start time
    completed_at: datetime | None     # End time
    duration_seconds: float           # Auto-calculated
    files_analyzed: int               # File count
    lines_analyzed: int               # Line count
    findings_by_severity: dict        # Severity counts
    raw_output: str                   # Raw tool output
    error_message: str | None         # Error if failed
```

## CodeAnalysisHarness Interface

### Required Properties

```python
@property
@abstractmethod
def name(self) -> str:
    """Return the harness name."""
    ...

@property
@abstractmethod
def supported_analysis_types(self) -> list[AnalysisType]:
    """Return list of supported analysis types."""
    ...
```

### Optional Property

```python
@property
def supports_interactive(self) -> bool:
    """Return whether interactive sessions are supported.

    Default is False. Override if the harness implements
    start_interactive_session() and continue_session().
    """
    return False
```

### Required Methods

```python
async def analyze_repository(
    self,
    repo_path: Path,
    analysis_type: AnalysisType = AnalysisType.GENERAL,
    prompt: str | None = None,
    options: dict[str, Any] | None = None,
) -> HarnessResult:
    """Analyze a repository or directory."""
    ...

async def analyze_files(
    self,
    files: list[Path],
    analysis_type: AnalysisType = AnalysisType.GENERAL,
    prompt: str | None = None,
    options: dict[str, Any] | None = None,
) -> HarnessResult:
    """Analyze specific files."""
    ...
```

### Optional Methods

```python
async def initialize(self) -> bool:
    """Initialize the harness. Returns success status."""
    return True

async def stream_analysis(
    self,
    repo_path: Path,
    analysis_type: AnalysisType = AnalysisType.GENERAL,
    prompt: str | None = None,
) -> AsyncIterator[str]:
    """Stream analysis output in real-time."""
    ...

async def start_interactive_session(
    self,
    context: SessionContext,
) -> SessionResult:
    """Start an interactive analysis session."""
    raise NotImplementedError(...)

async def continue_session(
    self,
    context: SessionContext,
    user_input: str,
) -> SessionResult:
    """Continue an interactive session."""
    raise NotImplementedError(...)

async def end_session(self, context: SessionContext) -> None:
    """End an interactive session."""
    pass
```

## ClaudeCodeHarness

Integration with Claude Code CLI:

### Configuration

```python
@dataclass
class ClaudeCodeConfig(HarnessConfig):
    claude_code_path: str = "claude"      # CLI path
    model: str = "claude-sonnet-4-..."    # Model to use
    max_thinking_tokens: int = 10000      # Thinking budget
    include_file_contents: bool = True    # Send file contents
    parse_structured_output: bool = True  # Parse JSON findings
    analysis_prompts: dict[str, str]      # Per-type prompts
```

### Usage

```python
from pathlib import Path
from aragora.harnesses.claude_code import ClaudeCodeHarness, ClaudeCodeConfig
from aragora.harnesses.base import AnalysisType

# Create harness with custom config
config = ClaudeCodeConfig(
    timeout_seconds=120,
    model="claude-sonnet-4-20250514",
)
harness = ClaudeCodeHarness(config)

# Check if CLI is available
if await harness.initialize():
    # Run security analysis
    result = await harness.analyze_repository(
        Path("/path/to/repo"),
        analysis_type=AnalysisType.SECURITY,
    )

    if result.success:
        for finding in result.findings:
            print(f"[{finding.severity.upper()}] {finding.title}")
            print(f"  File: {finding.file_path}:{finding.line_start}")
            print(f"  {finding.description}")
    else:
        print(f"Analysis failed: {result.error_message}")
```

### Interactive Sessions

ClaudeCodeHarness supports interactive sessions:

```python
from aragora.harnesses.base import SessionContext

# Start session
context = SessionContext(
    session_id="my-session",
    repo_path=Path("/path/to/repo"),
    files_in_context=["src/main.py", "src/api.py"],
)

result = await harness.start_interactive_session(context)
print(result.response)

# Continue conversation
result = await harness.continue_session(
    context,
    "What are the security concerns in the authentication code?"
)
print(result.response)

# End session
await harness.end_session(context)
```

### Custom Prompts

Override default prompts for specific analysis types:

```python
config = ClaudeCodeConfig(
    analysis_prompts={
        AnalysisType.SECURITY.value: """
Analyze this codebase for security vulnerabilities with focus on:
- API key exposure
- SQL/NoSQL injection
- Authentication bypass
- Authorization issues

Format findings as JSON array with fields:
id, title, description, severity, confidence, category, file_path
""",
    }
)
```

## Implementing Custom Harnesses

### Basic Implementation

```python
from aragora.harnesses.base import (
    CodeAnalysisHarness,
    HarnessConfig,
    HarnessResult,
    AnalysisType,
    AnalysisFinding,
)
from pathlib import Path
from typing import Any

class MyToolHarness(CodeAnalysisHarness):
    @property
    def name(self) -> str:
        return "my-tool"

    @property
    def supported_analysis_types(self) -> list[AnalysisType]:
        return [AnalysisType.SECURITY, AnalysisType.QUALITY]

    async def analyze_repository(
        self,
        repo_path: Path,
        analysis_type: AnalysisType = AnalysisType.GENERAL,
        prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> HarnessResult:
        self._validate_path(repo_path)

        # Run your tool
        findings = await self._run_analysis(repo_path, analysis_type)

        return HarnessResult(
            harness=self.name,
            analysis_type=analysis_type,
            success=True,
            findings=findings,
        )

    async def analyze_files(
        self,
        files: list[Path],
        analysis_type: AnalysisType = AnalysisType.GENERAL,
        prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> HarnessResult:
        if not files:
            return HarnessResult(
                harness=self.name,
                analysis_type=analysis_type,
                success=False,
                findings=[],
                error_message="No files provided",
            )

        return await self.analyze_repository(
            files[0].parent,
            analysis_type,
            prompt,
            options,
        )
```

### With Interactive Support

```python
class InteractiveHarness(CodeAnalysisHarness):
    def __init__(self, config: HarnessConfig | None = None):
        super().__init__(config)
        self._sessions: dict[str, SessionContext] = {}

    @property
    def supports_interactive(self) -> bool:
        return True

    async def start_interactive_session(
        self,
        context: SessionContext,
    ) -> SessionResult:
        session_id = context.session_id
        self._sessions[session_id] = context

        # Initialize session with your tool
        response = await self._init_session(context)

        return SessionResult(
            session_id=session_id,
            response=response,
            continue_conversation=True,
        )

    async def continue_session(
        self,
        context: SessionContext,
        user_input: str,
    ) -> SessionResult:
        if context.session_id not in self._sessions:
            raise HarnessError("Session not found", self.name)

        # Process input
        response = await self._process_input(context, user_input)
        context.conversation_history.append({
            "role": "user", "content": user_input
        })
        context.conversation_history.append({
            "role": "assistant", "content": response
        })

        return SessionResult(
            session_id=context.session_id,
            response=response,
            continue_conversation=True,
        )
```

## Error Handling

### Exception Hierarchy

```
HarnessError (base)
├── HarnessTimeoutError    # Operation timed out
└── HarnessConfigError     # Invalid configuration
```

### Error Handling Pattern

```python
from aragora.harnesses.base import (
    HarnessError,
    HarnessTimeoutError,
    HarnessConfigError,
)

try:
    result = await harness.analyze_repository(repo_path)
except HarnessTimeoutError as e:
    print(f"Analysis timed out after {harness.config.timeout_seconds}s")
except HarnessConfigError as e:
    print(f"Configuration error: {e}")
except HarnessError as e:
    print(f"Harness error: {e}")
```

## Integration with Aragora

### In Debate Context

```python
from aragora.harnesses.claude_code import ClaudeCodeHarness
from aragora.debate.orchestrator import Arena

async def analyze_before_debate(repo_path: Path) -> list[str]:
    """Get analysis findings to inform debate."""
    harness = ClaudeCodeHarness()

    if not await harness.initialize():
        return []

    result = await harness.analyze_repository(
        repo_path,
        AnalysisType.ARCHITECTURE,
    )

    return [f.description for f in result.findings if f.severity in ("high", "critical")]
```

### In Gauntlet Testing

```python
from aragora.gauntlet import GauntletRunner
from aragora.harnesses.claude_code import ClaudeCodeHarness

async def code_review_gauntlet(repo_path: Path):
    harness = ClaudeCodeHarness()

    # Run security analysis
    security = await harness.analyze_repository(
        repo_path, AnalysisType.SECURITY
    )

    # Run quality analysis
    quality = await harness.analyze_repository(
        repo_path, AnalysisType.QUALITY
    )

    # Combine findings
    all_findings = security.findings + quality.findings
    critical_count = sum(1 for f in all_findings if f.severity == "critical")

    return {
        "passed": critical_count == 0,
        "findings": len(all_findings),
        "critical": critical_count,
    }
```

## See Also

- [MODES.md](./MODES.md) - Operational mode system
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [GAUNTLET.md](./GAUNTLET.md) - Gauntlet testing
