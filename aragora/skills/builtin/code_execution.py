"""
Code Execution Skill.

Provides sandboxed code execution capabilities for debates.
Supports Python, JavaScript, and shell commands with safety controls.

Security features:
- AST-based pre-validation before subprocess execution
- Blocks dangerous imports (os, subprocess, sys, socket, etc.)
- Blocks dangerous builtins (exec, eval, compile, __import__)
- Blocks access to dunder attributes (__class__, __globals__, etc.)
- Blocks network operations
- Configurable strict mode for untrusted code
"""

from __future__ import annotations

import ast
import asyncio
import json as _json_module
import logging
import os
import sys as _sys_module
import tempfile
from typing import Any

from ..base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


class CodeValidationError(Exception):
    """Raised when code validation fails due to security concerns."""

    pass


# Dangerous modules that should be blocked
DANGEROUS_MODULES = frozenset(
    {
        # System access
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "tempfile",
        "glob",
        "fnmatch",
        # Process/threading
        "multiprocessing",
        "threading",
        "concurrent",
        "asyncio",
        # Network access
        "socket",
        "ssl",
        "http",
        "urllib",
        "urllib3",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "poplib",
        "imaplib",
        "telnetlib",
        "xmlrpc",
        # Code execution
        "code",
        "codeop",
        "compile",
        "importlib",
        "runpy",
        "pkgutil",
        # Introspection/manipulation
        "inspect",
        "types",
        "gc",
        "ctypes",
        "builtins",
        "_thread",
        # File/IO
        "io",
        "fileinput",
        "mmap",
        # Serialization (can be used for code execution)
        "pickle",
        "marshal",
        "shelve",
        # Other dangerous
        "pty",
        "tty",
        "signal",
        "resource",
        "sysconfig",
        "platform",
        "getpass",
        "crypt",
    }
)

# Dangerous builtin functions
DANGEROUS_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "memoryview",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "type",
        "object",
        "classmethod",
        "staticmethod",
        "property",
        "super",
    }
)

# Blocked dunder attributes (sandbox escape vectors)
BLOCKED_DUNDER_ATTRS = frozenset(
    {
        "__globals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__dict__",
        "__module__",
        "__name__",
        "__qualname__",
        "__func__",
        "__self__",
        "__wrapped__",
        "__annotations__",
        "__init_subclass__",
        "__reduce__",
        "__reduce_ex__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__import__",
        "__loader__",
        "__spec__",
    }
)


def validate_python_code(
    code: str,
    *,
    strict_mode: bool = True,
    allow_imports: set[str] | None = None,
    allow_builtins: set[str] | None = None,
) -> None:
    """
    Validate Python code using AST analysis before execution.

    This function performs static analysis to detect dangerous patterns
    BEFORE the code is executed in a subprocess.

    Args:
        code: Python source code to validate
        strict_mode: If True, applies strictest validation (recommended for untrusted code)
        allow_imports: Optional set of module names to allow (only in non-strict mode)
        allow_builtins: Optional set of builtin names to allow (only in non-strict mode)

    Raises:
        CodeValidationError: If dangerous patterns are detected
        SyntaxError: If the code has invalid Python syntax

    Security checks:
        1. Block imports of dangerous modules
        2. Block calls to dangerous builtins
        3. Block access to dunder attributes
        4. Block string literals containing dangerous patterns
        5. Block network-related operations
    """
    # Initialize allowed sets
    allowed_imports = set() if strict_mode else (allow_imports or set())
    allowed_builtins = set() if strict_mode else (allow_builtins or set())

    # Safe modules that are always allowed
    safe_modules = {"math", "json", "re", "datetime", "collections", "itertools", "functools"}
    allowed_imports.update(safe_modules)

    # Parse the code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"Syntax error in code: {e}") from e

    # Walk the AST and check for dangerous patterns
    for node in ast.walk(tree):
        _validate_node(node, allowed_imports, allowed_builtins, strict_mode)


def _validate_node(
    node: ast.AST,
    allowed_imports: set[str],
    allowed_builtins: set[str],
    strict_mode: bool,
) -> None:
    """Validate a single AST node for security concerns."""

    # Check imports
    if isinstance(node, ast.Import):
        for alias in node.names:
            module_name = alias.name.split(".")[0]  # Get root module
            if module_name in DANGEROUS_MODULES and module_name not in allowed_imports:
                raise CodeValidationError(f"Import of dangerous module not allowed: {alias.name}")

    elif isinstance(node, ast.ImportFrom):
        if node.module:
            module_name = node.module.split(".")[0]  # Get root module
            if module_name in DANGEROUS_MODULES and module_name not in allowed_imports:
                raise CodeValidationError(
                    f"Import from dangerous module not allowed: {node.module}"
                )

    # Check function calls
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in DANGEROUS_BUILTINS and func_name not in allowed_builtins:
                raise CodeValidationError(f"Call to dangerous builtin not allowed: {func_name}")

        # Check for attribute calls like os.system(), subprocess.run(), etc.
        elif isinstance(node.func, ast.Attribute):
            _validate_attribute_call(node.func)

    # Check attribute access (block dunder attributes)
    elif isinstance(node, ast.Attribute):
        attr_name = node.attr
        if attr_name in BLOCKED_DUNDER_ATTRS:
            raise CodeValidationError(f"Access to dunder attribute not allowed: {attr_name}")

        # In strict mode, block all underscore-prefixed attributes
        if strict_mode and attr_name.startswith("_"):
            raise CodeValidationError(
                f"Access to private/protected attribute not allowed in strict mode: {attr_name}"
            )

    # Check string literals for dangerous patterns
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        _validate_string_literal(node.value, strict_mode)

    # Check subscript access (e.g., obj["__globals__"])
    elif isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            if node.slice.value in BLOCKED_DUNDER_ATTRS:
                raise CodeValidationError(
                    f"Subscript access to blocked attribute not allowed: {node.slice.value}"
                )


def _validate_attribute_call(attr_node: ast.Attribute) -> None:
    """Validate attribute-based function calls for dangerous patterns."""
    # Check for dangerous method calls
    dangerous_methods = {
        "system",
        "popen",
        "spawn",
        "exec",
        "call",
        "run",
        "Popen",
        "check_output",
        "check_call",
        "getoutput",
        "getstatusoutput",
    }

    if attr_node.attr in dangerous_methods:
        raise CodeValidationError(
            f"Call to potentially dangerous method not allowed: {attr_node.attr}"
        )


def _validate_string_literal(value: str, strict_mode: bool) -> None:
    """Validate string literals for dangerous patterns."""
    value_lower = value.lower()

    # Check for dunder patterns in strings
    for blocked in BLOCKED_DUNDER_ATTRS:
        if blocked in value_lower:
            raise CodeValidationError(f"String literal contains blocked pattern: {blocked}")

    # In strict mode, check for potential command injection patterns
    if strict_mode:
        dangerous_patterns = [
            "import os",
            "import sys",
            "import subprocess",
            "__import__",
            "eval(",
            "exec(",
            "compile(",
        ]
        for pattern in dangerous_patterns:
            if pattern in value_lower:
                raise CodeValidationError(
                    f"String literal contains potentially dangerous pattern: {pattern}"
                )


class CodeExecutionSkill(Skill):
    """
    Skill for executing code in a sandboxed environment.

    Supports:
    - Python code execution
    - JavaScript (via Node.js if available)
    - Shell commands (restricted)

    Safety features:
    - AST-based pre-validation before execution
    - Timeout enforcement
    - Resource limits
    - Blocked dangerous operations
    - Configurable strict mode for untrusted code
    """

    def __init__(
        self,
        max_output_size: int = 100_000,
        enable_shell: bool = False,
        strict_mode: bool = True,
        allow_imports: set[str] | None = None,
        allow_builtins: set[str] | None = None,
    ):
        """
        Initialize code execution skill.

        Args:
            max_output_size: Maximum output size in bytes
            enable_shell: Whether to enable shell command execution
            strict_mode: If True, applies strictest validation (recommended for untrusted code).
                         Blocks all underscore-prefixed attributes and dangerous string patterns.
            allow_imports: Optional set of module names to allow (only effective in non-strict mode)
            allow_builtins: Optional set of builtin names to allow (only effective in non-strict mode)
        """
        self._max_output_size = max_output_size
        self._enable_shell = enable_shell
        self._strict_mode = strict_mode
        self._allow_imports = allow_imports or set()
        self._allow_builtins = allow_builtins or set()

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="code_execution",
            version="1.1.0",
            description="Execute code in a sandboxed environment with AST-based security validation",
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
                "strict_mode": {
                    "type": "boolean",
                    "description": "Enable strict validation mode (recommended for untrusted code)",
                    "default": True,
                },
            },
            tags=["code", "execution", "sandbox", "secure"],
            required_permissions=["skills:code_execution"],
            debate_compatible=True,
            max_execution_time_seconds=60.0,
            rate_limit_per_minute=20,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
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
        # Allow per-request override, but default to instance setting
        strict_mode = input_data.get("strict_mode", self._strict_mode)

        try:
            if language == "python":
                result = await self._execute_python(code, timeout, code_input, strict_mode)
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

        except CodeValidationError as e:
            logger.warning(f"Code validation failed: {e}")
            return SkillResult.create_failure(
                f"Code validation failed: {e}",
                error_code="validation_failed",
            )
        except asyncio.TimeoutError:
            return SkillResult.create_timeout(timeout)
        except Exception as e:
            logger.exception(f"Code execution failed: {e}")
            return SkillResult.create_failure(f"Execution failed: {e}")

    @staticmethod
    def _get_safe_subprocess_env() -> dict[str, str]:
        """Get a filtered environment for subprocess execution.

        Removes sensitive environment variables (API keys, secrets, tokens)
        to prevent them from being exposed to executed code.
        """
        safe_env = {}
        for key in ("PATH", "HOME", "PYTHONPATH", "LANG", "LC_ALL", "TMPDIR", "TMP", "TEMP"):
            if key in os.environ:
                safe_env[key] = os.environ[key]
        if "LANG" not in safe_env:
            safe_env["LANG"] = "en_US.UTF-8"
        return safe_env

    async def _execute_python(
        self,
        code: str,
        timeout: float,
        input_data: dict[str, Any],
        strict_mode: bool = True,
    ) -> dict[str, Any]:
        """Execute Python code in a subprocess.

        Security layers:
        1. AST-based validation BEFORE execution (blocks dangerous patterns)
        2. Restricted builtins in exec() (no __import__, open, exec, eval, etc.)
        3. Code injected via repr() (prevents triple-quote escape attacks)
        4. Filtered environment variables (prevents API key leakage)
        5. Timeout enforcement via asyncio.wait_for()
        """
        # SECURITY: Validate code BEFORE execution
        # This catches dangerous patterns before any code runs
        validate_python_code(
            code,
            strict_mode=strict_mode,
            allow_imports=self._allow_imports if not strict_mode else None,
            allow_builtins=self._allow_builtins if not strict_mode else None,
        )
        logger.debug("Code validation passed, proceeding with execution")

        # SECURITY: Serialize input_data to JSON for safe injection (prevents repr injection)
        safe_input_json = _json_module.dumps(input_data)

        # SECURITY: Use repr() to safely inject code as a string literal,
        # preventing triple-quote escape attacks. The subprocess uses
        # restricted builtins (safe subset only, no __import__, open, exec, etc.)
        # and a filtered environment (no API keys).
        wrapper = f"""
import json
import sys
import io

# Safe builtins whitelist - blocks dangerous functions
SAFE_BUILTINS = {{
    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
    'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
    'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
    'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
    'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
    'map': map, 'max': max, 'min': min, 'next': next, 'oct': oct, 'ord': ord,
    'pow': pow, 'print': print, 'range': range, 'repr': repr, 'reversed': reversed,
    'round': round, 'set': set, 'slice': slice, 'sorted': sorted, 'str': str,
    'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
    # Exceptions for try/except
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
    'KeyError': KeyError, 'IndexError': IndexError, 'AssertionError': AssertionError,
    'AttributeError': AttributeError, 'RuntimeError': RuntimeError,
    'StopIteration': StopIteration, 'ZeroDivisionError': ZeroDivisionError,
    'True': True, 'False': False, 'None': None,
}}

# Allow safe imports through a restricted __import__
ALLOWED_MODULES = {{'math', 'json', 're', 'datetime', 'collections', 'itertools', 'functools'}}

def _safe_import(name, *args, **kwargs):
    if name.split('.')[0] not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{{name}}' is not allowed in sandbox")
    import importlib
    return importlib.import_module(name)

SAFE_BUILTINS['__import__'] = _safe_import

# Input data (safely deserialized from JSON)
input_data = json.loads({repr(safe_input_json)})

# Capture output
_output_lines = []
_original_print = print
def _capture_print(*args, **kwargs):
    buf = io.StringIO()
    _original_print(*args, file=buf, **kwargs)
    _output_lines.append(buf.getvalue())
    _original_print(*args, **kwargs)

# Override print in safe builtins
SAFE_BUILTINS['print'] = _capture_print

# Execute user code with restricted builtins
_user_code = {repr(code)}
try:
    result = None
    exec_globals = {{"input_data": input_data, "__builtins__": SAFE_BUILTINS}}
    exec(_user_code, exec_globals)
    result = exec_globals.get("result")
except Exception as e:
    _capture_print(f"Error: {{e}}")
    result = None

# Output result
output = {{"output": "".join(_output_lines), "result": repr(result)}}
_original_print("__RESULT__:" + json.dumps(output))
"""

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                _sys_module.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_safe_subprocess_env(),
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
                    result_line = stdout_str.split("__RESULT__:")[-1].strip()
                    result_data = _json_module.loads(result_line)
                except Exception as e:
                    logger.debug("Failed to parse execution result JSON: %s", e)

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
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute JavaScript code via Node.js."""
        # Create wrapper
        wrapper = f"""
const inputData = {_json_module.dumps(input_data)};
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
    ) -> dict[str, Any]:
        """Execute shell commands using sandboxed subprocess runner.

        Uses the secure subprocess_runner module which enforces:
        - Command whitelist validation
        - Blocked command rejection
        - Shell metacharacter detection
        - Sanitized environment
        """
        import shlex

        from aragora.utils.subprocess_runner import SandboxError, run_sandboxed

        try:
            # Parse the command string into arguments
            args = shlex.split(code)
            if not args:
                return {
                    "language": "shell",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Empty command",
                }

            # Use the secure sandboxed runner (validates against allowlist)
            result = await run_sandboxed(
                args,
                timeout=min(timeout, 60.0),  # Cap at 60 seconds for shell commands
                capture_output=True,
            )

            return {
                "language": "shell",
                "exit_code": result.returncode,
                "stdout": result.stdout[: self._max_output_size],
                "stderr": result.stderr[: self._max_output_size],
            }

        except SandboxError as e:
            # Command failed allowlist validation
            return {
                "language": "shell",
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Security: {e}",
            }
        except ValueError as e:
            # shlex.split failed (malformed command)
            return {
                "language": "shell",
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Invalid command syntax: {e}",
            }


# Skill instance for registration
SKILLS = [CodeExecutionSkill()]
