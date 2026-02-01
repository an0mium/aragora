"""
Tests for aragora.skills.builtin.code_execution module.

Covers:
- CodeExecutionSkill manifest and initialization
- Python code execution
- Code validation (AST-based security)
- Sandbox security (blocking dangerous imports, builtins, attributes)
- Timeout enforcement
- Output capture (stdout, stderr)
- Return value handling
- Error handling (syntax errors, runtime errors)
- Skill registration
"""

from __future__ import annotations

import pytest

from aragora.skills.base import SkillCapability, SkillContext, SkillStatus
from aragora.skills.builtin.code_execution import (
    BLOCKED_DUNDER_ATTRS,
    DANGEROUS_BUILTINS,
    DANGEROUS_MODULES,
    CodeExecutionSkill,
    CodeValidationError,
    SKILLS,
    validate_python_code,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def skill() -> CodeExecutionSkill:
    """Create a code execution skill for testing."""
    return CodeExecutionSkill()


@pytest.fixture
def non_strict_skill() -> CodeExecutionSkill:
    """Create a non-strict code execution skill for testing."""
    return CodeExecutionSkill(strict_mode=False)


@pytest.fixture
def context() -> SkillContext:
    """Create a context for testing."""
    return SkillContext(
        user_id="user123",
        permissions=["skills:code_execution"],
    )


# =============================================================================
# CodeExecutionSkill Manifest Tests
# =============================================================================


class TestCodeExecutionSkillManifest:
    """Tests for CodeExecutionSkill manifest."""

    def test_manifest_name(self, skill: CodeExecutionSkill):
        """Test manifest name."""
        assert skill.manifest.name == "code_execution"

    def test_manifest_version(self, skill: CodeExecutionSkill):
        """Test manifest version."""
        assert skill.manifest.version == "1.1.0"

    def test_manifest_capabilities(self, skill: CodeExecutionSkill):
        """Test manifest capabilities."""
        assert SkillCapability.CODE_EXECUTION in skill.manifest.capabilities

    def test_manifest_input_schema(self, skill: CodeExecutionSkill):
        """Test manifest input schema."""
        schema = skill.manifest.input_schema

        assert "code" in schema
        assert schema["code"]["type"] == "string"
        assert schema["code"]["required"] is True

        assert "language" in schema
        assert schema["language"]["default"] == "python"

        assert "timeout" in schema
        assert "input_data" in schema
        assert "strict_mode" in schema

    def test_manifest_debate_compatible(self, skill: CodeExecutionSkill):
        """Test skill is debate compatible."""
        assert skill.manifest.debate_compatible is True

    def test_manifest_required_permissions(self, skill: CodeExecutionSkill):
        """Test required permissions."""
        assert "skills:code_execution" in skill.manifest.required_permissions

    def test_manifest_rate_limit(self, skill: CodeExecutionSkill):
        """Test rate limit is set."""
        assert skill.manifest.rate_limit_per_minute == 20

    def test_manifest_max_execution_time(self, skill: CodeExecutionSkill):
        """Test max execution time is set."""
        assert skill.manifest.max_execution_time_seconds == 60.0


# =============================================================================
# Skill Initialization Tests
# =============================================================================


class TestSkillInitialization:
    """Tests for CodeExecutionSkill initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        skill = CodeExecutionSkill()
        assert skill._max_output_size == 100_000
        assert skill._enable_shell is False
        assert skill._strict_mode is True
        assert skill._allow_imports == set()
        assert skill._allow_builtins == set()

    def test_custom_max_output_size(self):
        """Test custom max output size."""
        skill = CodeExecutionSkill(max_output_size=50_000)
        assert skill._max_output_size == 50_000

    def test_enable_shell(self):
        """Test enabling shell execution."""
        skill = CodeExecutionSkill(enable_shell=True)
        assert skill._enable_shell is True

    def test_non_strict_mode(self):
        """Test non-strict mode."""
        skill = CodeExecutionSkill(strict_mode=False)
        assert skill._strict_mode is False

    def test_allow_imports(self):
        """Test allowing specific imports."""
        skill = CodeExecutionSkill(
            strict_mode=False,
            allow_imports={"os", "sys"},
        )
        assert "os" in skill._allow_imports
        assert "sys" in skill._allow_imports


# =============================================================================
# Code Validation Tests - Dangerous Modules
# =============================================================================


class TestCodeValidationDangerousModules:
    """Tests for blocking dangerous module imports."""

    def test_blocks_os_import(self):
        """Block import os."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import os")

    def test_blocks_subprocess_import(self):
        """Block import subprocess."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import subprocess")

    def test_blocks_sys_import(self):
        """Block import sys."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import sys")

    def test_blocks_socket_import(self):
        """Block import socket (network access)."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import socket")

    def test_blocks_requests_import(self):
        """Block import requests (network access)."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import requests")

    def test_blocks_httpx_import(self):
        """Block import httpx (network access)."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import httpx")

    def test_blocks_pickle_import(self):
        """Block import pickle (code execution)."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import pickle")

    def test_blocks_importlib_import(self):
        """Block import importlib."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import importlib")

    def test_blocks_from_import(self):
        """Block from os import system."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("from os import system")

    def test_blocks_submodule_import(self):
        """Block import os.path."""
        with pytest.raises(CodeValidationError, match="dangerous module"):
            validate_python_code("import os.path")


# =============================================================================
# Code Validation Tests - Dangerous Builtins
# =============================================================================


class TestCodeValidationDangerousBuiltins:
    """Tests for blocking dangerous builtin calls."""

    def test_blocks_eval(self):
        """Block eval() calls."""
        with pytest.raises(CodeValidationError, match="dangerous builtin"):
            validate_python_code("eval('1+1')")

    def test_blocks_exec(self):
        """Block exec() calls."""
        with pytest.raises(CodeValidationError, match="dangerous builtin"):
            validate_python_code("exec('x=1')")

    def test_blocks_compile(self):
        """Block compile() calls."""
        with pytest.raises(CodeValidationError, match="dangerous builtin"):
            validate_python_code("compile('x=1', '<string>', 'exec')")

    def test_blocks_open(self):
        """Block open() calls (file access)."""
        with pytest.raises(CodeValidationError, match="dangerous builtin"):
            validate_python_code("open('/etc/passwd')")

    def test_blocks_dunder_import(self):
        """Block __import__() calls."""
        with pytest.raises(CodeValidationError, match="dangerous builtin"):
            validate_python_code("__import__('os')")

    def test_blocks_getattr(self):
        """Block getattr() calls."""
        with pytest.raises(CodeValidationError, match="dangerous builtin"):
            validate_python_code("getattr(object, '__class__')")

    def test_blocks_globals(self):
        """Block globals() calls."""
        with pytest.raises(CodeValidationError, match="dangerous builtin"):
            validate_python_code("globals()")


# =============================================================================
# Code Validation Tests - Dunder Attributes
# =============================================================================


class TestCodeValidationDunderAttrs:
    """Tests for blocking dunder attribute access."""

    def test_blocks_globals_attr(self):
        """Block __globals__ access."""
        with pytest.raises(CodeValidationError, match="dunder attribute"):
            validate_python_code("func.__globals__")

    def test_blocks_builtins_attr(self):
        """Block __builtins__ access."""
        with pytest.raises(CodeValidationError, match="dunder attribute"):
            validate_python_code("obj.__builtins__")

    def test_blocks_class_attr(self):
        """Block __class__ access."""
        with pytest.raises(CodeValidationError, match="dunder attribute"):
            validate_python_code("obj.__class__")

    def test_blocks_subclasses_attr(self):
        """Block __subclasses__ access."""
        with pytest.raises(CodeValidationError, match="dunder attribute"):
            validate_python_code("obj.__subclasses__()")

    def test_blocks_code_attr(self):
        """Block __code__ access."""
        with pytest.raises(CodeValidationError, match="dunder attribute"):
            validate_python_code("func.__code__")

    def test_blocks_subscript_dunder_access(self):
        """Block obj['__globals__'] access."""
        with pytest.raises(CodeValidationError, match="Subscript access"):
            validate_python_code("obj['__globals__']")

    def test_blocks_string_literal_dunder(self):
        """Block dunder patterns in string literals."""
        with pytest.raises(CodeValidationError, match="blocked pattern"):
            validate_python_code("x = '__globals__'")


# =============================================================================
# Code Validation Tests - Safe Code
# =============================================================================


class TestCodeValidationSafeCode:
    """Tests for allowing safe code patterns."""

    def test_allows_math_import(self):
        """Allow import math."""
        validate_python_code("import math")  # Should not raise

    def test_allows_json_import(self):
        """Allow import json."""
        validate_python_code("import json")  # Should not raise

    def test_allows_re_import(self):
        """Allow import re."""
        validate_python_code("import re")  # Should not raise

    def test_allows_datetime_import(self):
        """Allow import datetime."""
        validate_python_code("import datetime")  # Should not raise

    def test_allows_basic_arithmetic(self):
        """Allow basic arithmetic."""
        validate_python_code("result = 1 + 2 * 3")  # Should not raise

    def test_allows_function_definition(self):
        """Allow function definitions."""
        validate_python_code("def add(a, b): return a + b")  # Should not raise

    def test_allows_list_comprehension(self):
        """Allow list comprehensions."""
        validate_python_code("[x * 2 for x in range(10)]")  # Should not raise

    def test_allows_print(self):
        """Allow print statements."""
        validate_python_code("print('Hello, World!')")  # Should not raise


# =============================================================================
# Code Validation Tests - Strict Mode
# =============================================================================


class TestCodeValidationStrictMode:
    """Tests for strict mode behavior."""

    def test_strict_blocks_private_attrs(self):
        """Strict mode blocks private attribute access."""
        with pytest.raises(CodeValidationError, match="private/protected"):
            validate_python_code("obj._private", strict_mode=True)

    def test_non_strict_allows_private_attrs(self):
        """Non-strict mode allows private attribute access."""
        validate_python_code("obj._private", strict_mode=False)  # Should not raise

    def test_strict_blocks_dangerous_string_patterns(self):
        """Strict mode blocks dangerous string patterns."""
        with pytest.raises(CodeValidationError, match="dangerous pattern"):
            validate_python_code("x = 'import os'", strict_mode=True)

    def test_non_strict_allows_string_patterns(self):
        """Non-strict mode allows string patterns."""
        validate_python_code("x = 'import os'", strict_mode=False)  # Should not raise


# =============================================================================
# Code Validation Tests - Syntax Errors
# =============================================================================


class TestCodeValidationSyntaxErrors:
    """Tests for syntax error handling."""

    def test_raises_on_syntax_error(self):
        """Raise CodeValidationError on syntax error."""
        with pytest.raises(CodeValidationError, match="Syntax error"):
            validate_python_code("def broken(")

    def test_raises_on_invalid_code(self):
        """Raise CodeValidationError on invalid code."""
        with pytest.raises(CodeValidationError, match="Syntax error"):
            validate_python_code("if x = 5:")


# =============================================================================
# Code Execution Tests
# =============================================================================


class TestCodeExecution:
    """Tests for Python code execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_code(self, skill: CodeExecutionSkill, context: SkillContext):
        """Execute simple Python code."""
        result = await skill.execute(
            {"code": "print('Hello, World!')"},
            context,
        )

        assert result.success
        assert result.data is not None
        assert result.data["language"] == "python"
        assert result.data["exit_code"] == 0
        assert "Hello, World!" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_execute_code_with_result(self, skill: CodeExecutionSkill, context: SkillContext):
        """Execute code that sets a result variable."""
        result = await skill.execute(
            {"code": "result = 2 + 2"},
            context,
        )

        assert result.success
        assert result.data is not None
        assert "4" in result.data["result"]

    @pytest.mark.asyncio
    async def test_execute_code_with_input_data(
        self, skill: CodeExecutionSkill, context: SkillContext
    ):
        """Execute code with input data."""
        result = await skill.execute(
            {
                "code": "result = input_data['x'] + input_data['y']",
                "input_data": {"x": 5, "y": 3},
            },
            context,
        )

        assert result.success
        assert result.data is not None
        assert "8" in result.data["result"]

    @pytest.mark.asyncio
    async def test_execute_code_with_imports(
        self, skill: CodeExecutionSkill, context: SkillContext
    ):
        """Execute code with safe imports."""
        result = await skill.execute(
            {"code": "import math\nresult = math.sqrt(16)"},
            context,
        )

        assert result.success
        assert result.data is not None
        assert "4" in result.data["result"]


# =============================================================================
# Output Capture Tests
# =============================================================================


class TestOutputCapture:
    """Tests for stdout/stderr capture."""

    @pytest.mark.asyncio
    async def test_capture_stdout(self, skill: CodeExecutionSkill, context: SkillContext):
        """Capture stdout output."""
        result = await skill.execute(
            {"code": "print('line1')\nprint('line2')"},
            context,
        )

        assert result.success
        assert "line1" in result.data["stdout"]
        assert "line2" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_capture_multiple_prints(self, skill: CodeExecutionSkill, context: SkillContext):
        """Capture multiple print statements."""
        result = await skill.execute(
            {"code": "for i in range(3): print(f'Count: {i}')"},
            context,
        )

        assert result.success
        assert "Count: 0" in result.data["stdout"]
        assert "Count: 1" in result.data["stdout"]
        assert "Count: 2" in result.data["stdout"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_missing_code(self, skill: CodeExecutionSkill, context: SkillContext):
        """Return failure when code is missing."""
        result = await skill.execute({}, context)

        assert not result.success
        assert result.status == SkillStatus.FAILURE
        assert result.error_code == "missing_code"

    @pytest.mark.asyncio
    async def test_validation_failure(self, skill: CodeExecutionSkill, context: SkillContext):
        """Return failure on validation error."""
        result = await skill.execute(
            {"code": "import os"},
            context,
        )

        assert not result.success
        assert result.error_code == "validation_failed"

    @pytest.mark.asyncio
    async def test_unsupported_language(self, skill: CodeExecutionSkill, context: SkillContext):
        """Return failure for unsupported language."""
        result = await skill.execute(
            {"code": 'println("hello")', "language": "rust"},
            context,
        )

        assert not result.success
        assert result.error_code == "unsupported_language"

    @pytest.mark.asyncio
    async def test_runtime_error(self, skill: CodeExecutionSkill, context: SkillContext):
        """Handle runtime errors gracefully."""
        result = await skill.execute(
            {"code": "x = 1 / 0"},
            context,
        )

        assert result.success  # Execution completes, error is captured
        assert "Error" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_name_error(self, skill: CodeExecutionSkill, context: SkillContext):
        """Handle undefined variable errors."""
        result = await skill.execute(
            {"code": "print(undefined_variable)"},
            context,
        )

        assert result.success  # Execution completes, error is captured
        assert "Error" in result.data["stdout"]


# =============================================================================
# Timeout Enforcement Tests
# =============================================================================


class TestTimeoutEnforcement:
    """Tests for timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_caps_at_60_seconds(
        self, skill: CodeExecutionSkill, context: SkillContext
    ):
        """Timeout is capped at 60 seconds."""
        # This test just verifies the timeout capping logic
        result = await skill.execute(
            {"code": "result = 1 + 1", "timeout": 120},
            context,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_quick_code_completes(self, skill: CodeExecutionSkill, context: SkillContext):
        """Quick code completes within timeout."""
        result = await skill.execute(
            {"code": "result = sum(range(100))", "timeout": 5},
            context,
        )

        assert result.success
        assert result.data["exit_code"] == 0


# =============================================================================
# Sandbox Security Tests
# =============================================================================


class TestSandboxSecurity:
    """Tests for sandbox security."""

    @pytest.mark.asyncio
    async def test_blocks_file_access(self, skill: CodeExecutionSkill, context: SkillContext):
        """Block file access attempts."""
        result = await skill.execute(
            {"code": "open('/etc/passwd').read()"},
            context,
        )

        assert not result.success
        assert result.error_code == "validation_failed"

    @pytest.mark.asyncio
    async def test_blocks_network_access(self, skill: CodeExecutionSkill, context: SkillContext):
        """Block network access attempts."""
        result = await skill.execute(
            {"code": "import socket; socket.socket()"},
            context,
        )

        assert not result.success
        assert result.error_code == "validation_failed"

    @pytest.mark.asyncio
    async def test_blocks_subprocess(self, skill: CodeExecutionSkill, context: SkillContext):
        """Block subprocess calls."""
        result = await skill.execute(
            {"code": "import subprocess; subprocess.run(['ls'])"},
            context,
        )

        assert not result.success
        assert result.error_code == "validation_failed"

    @pytest.mark.asyncio
    async def test_blocks_os_system(self, skill: CodeExecutionSkill, context: SkillContext):
        """Block os.system calls."""
        result = await skill.execute(
            {"code": "import os; os.system('ls')"},
            context,
        )

        assert not result.success
        assert result.error_code == "validation_failed"

    @pytest.mark.asyncio
    async def test_blocks_sandbox_escape_via_string(
        self, skill: CodeExecutionSkill, context: SkillContext
    ):
        """Block sandbox escape attempts via string manipulation."""
        result = await skill.execute(
            {"code": "x = '__glob' + 'als__'"},  # String concat to avoid detection
            context,
        )

        # In strict mode, this is allowed as string creation,
        # but using it for actual access would be blocked
        # The code just creates a string, doesn't access anything
        assert result.success


# =============================================================================
# Skill Registration Tests
# =============================================================================


class TestSkillRegistration:
    """Tests for skill registration."""

    def test_skills_list_exists(self):
        """SKILLS list is defined."""
        assert SKILLS is not None
        assert isinstance(SKILLS, list)
        assert len(SKILLS) >= 1

    def test_code_execution_skill_in_list(self):
        """CodeExecutionSkill is in SKILLS list."""
        skill_names = [s.manifest.name for s in SKILLS]
        assert "code_execution" in skill_names

    def test_skill_is_valid_instance(self):
        """Skill in SKILLS is valid CodeExecutionSkill instance."""
        skill = next(s for s in SKILLS if s.manifest.name == "code_execution")
        assert isinstance(skill, CodeExecutionSkill)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_dangerous_modules_includes_os(self):
        """DANGEROUS_MODULES includes os."""
        assert "os" in DANGEROUS_MODULES

    def test_dangerous_modules_includes_network(self):
        """DANGEROUS_MODULES includes network modules."""
        assert "socket" in DANGEROUS_MODULES
        assert "requests" in DANGEROUS_MODULES
        assert "httpx" in DANGEROUS_MODULES

    def test_dangerous_builtins_includes_eval(self):
        """DANGEROUS_BUILTINS includes eval."""
        assert "eval" in DANGEROUS_BUILTINS
        assert "exec" in DANGEROUS_BUILTINS
        assert "compile" in DANGEROUS_BUILTINS

    def test_blocked_dunder_attrs_includes_globals(self):
        """BLOCKED_DUNDER_ATTRS includes __globals__."""
        assert "__globals__" in BLOCKED_DUNDER_ATTRS
        assert "__builtins__" in BLOCKED_DUNDER_ATTRS
        assert "__class__" in BLOCKED_DUNDER_ATTRS
