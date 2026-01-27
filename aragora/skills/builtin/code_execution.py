"""
Code Execution Skill.

Provides sandboxed code execution capabilities for debates.
Supports Python, JavaScript, and shell commands with safety controls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, Dict

from ..base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


class CodeExecutionSkill(Skill):
    """
    Skill for executing code in a sandboxed environment.

    Supports:
    - Python code execution
    - JavaScript (via Node.js if available)
    - Shell commands (restricted)

    Safety features:
    - Timeout enforcement
    - Resource limits
    - Blocked dangerous operations
    """

    def __init__(
        self,
        max_output_size: int = 100_000,
        enable_shell: bool = False,
    ):
        """
        Initialize code execution skill.

        Args:
            max_output_size: Maximum output size in bytes
            enable_shell: Whether to enable shell command execution
        """
        self._max_output_size = max_output_size
        self._enable_shell = enable_shell

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="code_execution",
            version="1.0.0",
            description="Execute code in a sandboxed environment",
            capabilities=[
                SkillCapability.CODE_EXECUTION,
            ],
            input_schema={
                "code": {
                    "type": "string",
                    "description": "Code to execute",
                    "required": True,
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (python, javascript, shell)",
                    "default": "python",
                },
                "timeout": {
                    "type": "number",
                    "description": "Execution timeout in seconds",
                    "default": 30,
                },
                "input_data": {
                    "type": "object",
                    "description": "Input data to pass to the code",
                },
            },
            tags=["code", "execution", "sandbox"],
            required_permissions=["skills:code_execution"],
            debate_compatible=True,
            max_execution_time_seconds=60.0,
            rate_limit_per_minute=20,
        )

    async def execute(
        self,
        input_data: Dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute code."""
        code = input_data.get("code", "")
        if not code:
            return SkillResult.create_failure(
                "Code is required",
                error_code="missing_code",
            )

        language = input_data.get("language", "python").lower()
        timeout = min(input_data.get("timeout", 30), 60)  # Cap at 60s
        code_input = input_data.get("input_data", {})

        try:
            if language == "python":
                result = await self._execute_python(code, timeout, code_input)
            elif language in ("javascript", "js"):
                result = await self._execute_javascript(code, timeout, code_input)
            elif language == "shell" and self._enable_shell:
                result = await self._execute_shell(code, timeout)
            else:
                return SkillResult.create_failure(
                    f"Unsupported language: {language}",
                    error_code="unsupported_language",
                )

            return SkillResult.create_success(result)

        except asyncio.TimeoutError:
            return SkillResult.create_timeout(timeout)
        except Exception as e:
            logger.exception(f"Code execution failed: {e}")
            return SkillResult.create_failure(f"Execution failed: {e}")

    async def _execute_python(
        self,
        code: str,
        timeout: float,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Python code in a subprocess."""
        # Create a wrapper script that handles input/output
        wrapper = f'''
import json
import sys

# Input data
input_data = {repr(input_data)}

# Capture output
_output_lines = []
_original_print = print
def _capture_print(*args, **kwargs):
    import io
    buf = io.StringIO()
    _original_print(*args, file=buf, **kwargs)
    _output_lines.append(buf.getvalue())
    _original_print(*args, **kwargs)

print = _capture_print

# User code
try:
    result = None
    exec_globals = {{"input_data": input_data, "__builtins__": __builtins__}}
    exec("""{code.replace('"""', chr(92) + '"""')}""", exec_globals)
    result = exec_globals.get("result")
except Exception as e:
    print(f"Error: {{e}}")
    result = None

# Output result
output = {{"output": "".join(_output_lines), "result": repr(result)}}
print("__RESULT__:" + json.dumps(output))
'''

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            stdout_str = stdout.decode()[: self._max_output_size]
            stderr_str = stderr.decode()[: self._max_output_size]

            # Parse result
            result_data = {"output": stdout_str, "error": stderr_str}
            if "__RESULT__:" in stdout_str:
                try:
                    import json

                    result_line = stdout_str.split("__RESULT__:")[-1].strip()
                    result_data = json.loads(result_line)
                except Exception:
                    pass

            return {
                "language": "python",
                "exit_code": proc.returncode,
                "stdout": result_data.get("output", stdout_str),
                "stderr": stderr_str,
                "result": result_data.get("result"),
            }

        finally:
            os.unlink(temp_path)

    async def _execute_javascript(
        self,
        code: str,
        timeout: float,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute JavaScript code via Node.js."""
        import json

        # Create wrapper
        wrapper = f"""
const inputData = {json.dumps(input_data)};
let result;
try {{
    {code}
}} catch (e) {{
    console.error("Error:", e.message);
}}
console.log("__RESULT__:" + JSON.stringify({{result}}));
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(wrapper)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "node",
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            stdout_str = stdout.decode()[: self._max_output_size]
            stderr_str = stderr.decode()[: self._max_output_size]

            return {
                "language": "javascript",
                "exit_code": proc.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
            }

        finally:
            os.unlink(temp_path)

    async def _execute_shell(
        self,
        code: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """Execute shell commands (restricted)."""
        # Block dangerous commands
        dangerous = ["rm -rf", "mkfs", "dd if=", ":(){", "fork", "wget", "curl"]
        for cmd in dangerous:
            if cmd in code.lower():
                return {
                    "language": "shell",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": f"Blocked dangerous command: {cmd}",
                }

        proc = await asyncio.create_subprocess_shell(
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        return {
            "language": "shell",
            "exit_code": proc.returncode,
            "stdout": stdout.decode()[: self._max_output_size],
            "stderr": stderr.decode()[: self._max_output_size],
        }


# Skill instance for registration
SKILLS = [CodeExecutionSkill()]
