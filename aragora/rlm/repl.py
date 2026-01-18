"""
RLM REPL Environment for programmatic context interaction.

Based on the paper's approach of treating long prompts as an external
environment that the LLM can interact with through a Python REPL.
"""

import ast
import io
import logging
import re
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import AbstractionLevel, RLMConfig, RLMContext, RLMResult

logger = logging.getLogger(__name__)


@dataclass
class REPLState:
    """State of the REPL environment."""

    # Variables accessible to the LLM
    namespace: dict[str, Any] = field(default_factory=dict)

    # Execution history
    history: list[tuple[str, str]] = field(default_factory=list)  # (code, output)

    # Sub-LM tracking
    sub_calls: list[dict[str, Any]] = field(default_factory=list)
    sub_call_count: int = 0

    # Final result
    final_answer: Optional[str] = None
    final_var: Optional[str] = None


class RLMEnvironment:
    """
    REPL environment for RLM context interaction.

    Provides a sandboxed Python environment where:
    - Context is pre-loaded as a variable (CONTEXT)
    - LLM outputs code to interact with context
    - Results are truncated and returned to LLM's context
    - RLM_M() spawns sub-LM instances
    - FINAL()/FINAL_VAR() return results
    """

    # Maximum output length per execution
    MAX_OUTPUT_CHARS = 2000

    # Safe builtins for the sandbox
    SAFE_BUILTINS = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "abs": abs,
        "round": round,
        "print": print,
        "type": type,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
        "callable": callable,
    }

    def __init__(
        self,
        config: RLMConfig,
        context: RLMContext,
        agent_call: Optional[Callable[[str, str, str], str]] = None,
    ):
        """
        Initialize REPL environment.

        Args:
            config: RLM configuration
            context: Hierarchical context to make available
            agent_call: Callback to invoke agents for sub-LM calls
                       Signature: (model, query, context) -> response
        """
        self.config = config
        self.context = context
        self.agent_call = agent_call

        self.state = REPLState()
        self._initialize_namespace()

    def _initialize_namespace(self) -> None:
        """Set up the namespace with context and helper functions."""
        # Core context access
        self.state.namespace["CONTEXT"] = self.context.original_content
        self.state.namespace["CONTEXT_TOKENS"] = self.context.original_tokens

        # Hierarchical access
        self.state.namespace["get_level"] = self._get_level
        self.state.namespace["get_abstract"] = lambda: self._get_level(AbstractionLevel.ABSTRACT)
        self.state.namespace["get_summary"] = lambda: self._get_level(AbstractionLevel.SUMMARY)
        self.state.namespace["get_detailed"] = lambda: self._get_level(AbstractionLevel.DETAILED)
        self.state.namespace["get_full"] = lambda: self._get_level(AbstractionLevel.FULL)

        # Node navigation
        self.state.namespace["get_node"] = self._get_node
        self.state.namespace["drill_down"] = self._drill_down
        self.state.namespace["get_children"] = self._get_children

        # Search utilities
        self.state.namespace["search"] = self._search
        self.state.namespace["grep"] = self._grep
        self.state.namespace["find_sections"] = self._find_sections

        # RLM primitives
        self.state.namespace["RLM_M"] = self._rlm_call
        self.state.namespace["FINAL"] = self._final
        self.state.namespace["FINAL_VAR"] = self._final_var

        # String utilities
        self.state.namespace["truncate"] = self._truncate
        self.state.namespace["split_chunks"] = self._split_chunks
        self.state.namespace["count_tokens"] = self._count_tokens

        # Safe builtins
        for name, func in self.SAFE_BUILTINS.items():
            self.state.namespace[name] = func

        # Re module for regex
        self.state.namespace["re"] = re

    def _get_level(self, level: AbstractionLevel) -> str:
        """Get context at a specific abstraction level."""
        return self.context.get_at_level(level)

    def _get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """Get a specific node by ID."""
        node = self.context.get_node(node_id)
        if not node:
            return None
        return {
            "id": node.id,
            "level": node.level.name,
            "content": node.content,
            "token_count": node.token_count,
            "key_topics": node.key_topics,
            "parent_id": node.parent_id,
            "child_ids": node.child_ids,
        }

    def _drill_down(self, node_id: str) -> list[dict[str, Any]]:
        """Get more detailed nodes under a given node."""
        children = self.context.drill_down(node_id)
        return [
            {
                "id": c.id,
                "level": c.level.name,
                "content": c.content[:500] + "..." if len(c.content) > 500 else c.content,
                "token_count": c.token_count,
                "key_topics": c.key_topics,
            }
            for c in children
        ]

    def _get_children(self, node_id: str) -> list[str]:
        """Get child node IDs for navigation."""
        node = self.context.get_node(node_id)
        return node.child_ids if node else []

    def _search(self, query: str, level: Optional[str] = None) -> list[dict[str, Any]]:
        """Search for content matching query."""
        search_level = AbstractionLevel[level.upper()] if level else AbstractionLevel.SUMMARY

        results = []
        if search_level in self.context.levels:
            for node in self.context.levels[search_level]:
                if query.lower() in node.content.lower():
                    results.append({
                        "id": node.id,
                        "level": node.level.name,
                        "snippet": self._extract_snippet(node.content, query),
                        "topics": node.key_topics,
                    })
        return results[:10]  # Limit results

    def _grep(self, pattern: str, content: Optional[str] = None) -> list[str]:
        """Search using regex pattern."""
        text = content or self.context.original_content
        try:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            return matches[:20]  # Limit results
        except re.error:
            return []

    def _find_sections(self, heading_pattern: str = r"^#+\s+.+$") -> list[dict[str, Any]]:
        """Find sections in the content by heading pattern."""
        sections = []
        lines = self.context.original_content.split("\n")
        current_heading = None
        current_start = 0

        for i, line in enumerate(lines):
            if re.match(heading_pattern, line, re.MULTILINE):
                if current_heading:
                    sections.append({
                        "heading": current_heading,
                        "start_line": current_start,
                        "end_line": i - 1,
                    })
                current_heading = line.strip()
                current_start = i

        if current_heading:
            sections.append({
                "heading": current_heading,
                "start_line": current_start,
                "end_line": len(lines) - 1,
            })

        return sections

    def _rlm_call(self, query: str, context_snippet: str) -> str:
        """
        Spawn a sub-LM instance (RLM_M primitive from paper).

        Args:
            query: Query for the sub-LM
            context_snippet: Transformed/filtered context to provide

        Returns:
            Response from sub-LM
        """
        if self.state.sub_call_count >= self.config.max_sub_calls:
            return "[ERROR: Maximum sub-LM calls exceeded]"

        if not self.agent_call:
            return "[ERROR: No agent callback configured for sub-LM calls]"

        self.state.sub_call_count += 1

        # Track the call
        call_record = {
            "index": self.state.sub_call_count,
            "query": query[:200],
            "context_length": len(context_snippet),
        }
        self.state.sub_calls.append(call_record)

        try:
            # Use configured sub-model
            response = self.agent_call(
                self.config.sub_model,
                query,
                context_snippet[:self.config.target_tokens * 4],  # Rough char limit
            )
            call_record["response_length"] = len(response)
            return response
        except Exception as e:
            logger.error(f"Sub-LM call failed: {e}")
            call_record["error"] = str(e)
            return f"[ERROR: Sub-LM call failed: {e}]"

    def _final(self, answer: str) -> None:
        """Mark final answer (FINAL primitive from paper)."""
        self.state.final_answer = str(answer)

    def _final_var(self, var_name: str) -> None:
        """Mark variable as final answer (FINAL_VAR primitive from paper)."""
        self.state.final_var = var_name

    def _truncate(self, text: str, max_chars: int = 500) -> str:
        """Truncate text to max characters."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"... [truncated, {len(text) - max_chars} more chars]"

    def _split_chunks(self, text: str, chunk_size: int = 1000) -> list[str]:
        """Split text into chunks."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4  # ~4 chars per token approximation

    def _extract_snippet(self, content: str, query: str, context_chars: int = 100) -> str:
        """Extract snippet around query match."""
        query_lower = query.lower()
        content_lower = content.lower()
        idx = content_lower.find(query_lower)
        if idx == -1:
            return content[:200] + "..."

        start = max(0, idx - context_chars)
        end = min(len(content), idx + len(query) + context_chars)
        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def execute(self, code: str) -> tuple[str, bool]:
        """
        Execute code in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            (output, is_final) - output string and whether FINAL was called
        """
        # Validate code safety (basic AST check)
        try:
            tree = ast.parse(code)
            self._validate_ast(tree)
        except SyntaxError as e:
            return f"SyntaxError: {e}", False
        except SecurityError as e:
            return f"SecurityError: {e}", False

        # Capture stdout
        stdout_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stdout_capture):
                exec(code, {"__builtins__": {}}, self.state.namespace)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}", False

        output = stdout_capture.getvalue()

        # Truncate output if too long
        if len(output) > self.MAX_OUTPUT_CHARS:
            output = output[:self.MAX_OUTPUT_CHARS] + f"\n[Output truncated, {len(output) - self.MAX_OUTPUT_CHARS} more chars]"

        # Record in history
        self.state.history.append((code, output))

        # Check for final answer
        is_final = self.state.final_answer is not None or self.state.final_var is not None

        return output, is_final

    def _validate_ast(self, tree: ast.AST) -> None:
        """Validate AST for security concerns."""
        for node in ast.walk(tree):
            # Block imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Allow only re module
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name != "re":
                            raise SecurityError(f"Import not allowed: {alias.name}")
                elif node.module != "re":
                    raise SecurityError(f"Import not allowed: {node.module}")

            # Block dangerous builtins
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "open", "__import__"):
                        raise SecurityError(f"Function not allowed: {node.func.id}")

            # Block attribute access to dangerous methods
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    raise SecurityError(f"Private attribute access not allowed: {node.attr}")

    def get_result(self) -> RLMResult:
        """Get the final result after execution."""
        if self.state.final_var:
            answer = str(self.state.namespace.get(self.state.final_var, "[Variable not found]"))
        elif self.state.final_answer:
            answer = self.state.final_answer
        else:
            answer = "[No final answer provided]"

        # Collect nodes examined (from search operations, etc.)
        nodes_examined = []
        for code, _ in self.state.history:
            # Extract node IDs from code (basic heuristic)
            for match in re.findall(r"get_node\(['\"](\w+)['\"]\)", code):
                nodes_examined.append(match)
            for match in re.findall(r"drill_down\(['\"](\w+)['\"]\)", code):
                nodes_examined.append(match)

        return RLMResult(
            answer=answer,
            nodes_examined=list(set(nodes_examined)),
            levels_traversed=[],  # Would need more tracking
            citations=[],
            tokens_processed=sum(len(output) // 4 for _, output in self.state.history),
            sub_calls_made=self.state.sub_call_count,
            time_seconds=0.0,  # Caller should track
            confidence=0.8 if self.state.final_answer else 0.5,
            uncertainty_sources=[],
        )

    def get_prompt_for_agent(self) -> str:
        """Generate the system prompt for an agent to use this REPL."""
        return f"""You are interacting with a context through a Python REPL environment.

## Available Variables and Functions

### Context Access
- `CONTEXT` - Full original content ({self.context.original_tokens} tokens)
- `CONTEXT_TOKENS` - Token count of full context
- `get_abstract()` - Get highest-level summary
- `get_summary()` - Get summary level
- `get_detailed()` - Get detailed level
- `get_full()` - Get full content

### Navigation
- `get_node(node_id)` - Get specific node by ID
- `drill_down(node_id)` - Get child nodes (more detail)
- `get_children(node_id)` - Get child node IDs

### Search
- `search(query, level=None)` - Search for content
- `grep(pattern, content=None)` - Regex search
- `find_sections()` - Find document sections

### Sub-LM Calls
- `RLM_M(query, context_snippet)` - Spawn sub-LM to process context snippet

### Utilities
- `truncate(text, max_chars)` - Truncate text
- `split_chunks(text, chunk_size)` - Split into chunks
- `count_tokens(text)` - Estimate token count
- `re` - Python re module for regex

### Finishing
- `FINAL(answer)` - Return final answer
- `FINAL_VAR(var_name)` - Return variable as final answer

## Strategy

1. Start by examining the abstract/summary level
2. Search for relevant sections
3. Drill down into detailed content as needed
4. Use RLM_M() for complex sub-queries
5. Call FINAL() when you have the answer

## Example

```python
# Peek at structure
summary = get_summary()
print(truncate(summary, 500))

# Search for relevant content
results = search("authentication")
print(results)

# Drill into a result
if results:
    details = drill_down(results[0]["id"])
    print(details)

# Final answer
FINAL("The authentication system uses JWT tokens...")
```
"""


class SecurityError(Exception):
    """Raised when code attempts unsafe operations."""
    pass
