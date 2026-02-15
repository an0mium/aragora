"""
Tests for skill validation at registration time.

Covers:
- validate_skill_imports: checks module imports vs declared capabilities
- _validate_skill_code: AST validation of execute() method for code/shell skills
- Registration-time validation: warnings generated but registration succeeds
- Properly declared skills produce no warnings
"""

from __future__ import annotations

import textwrap
import types
from typing import Any
from unittest.mock import patch

import pytest

from aragora.skills.base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)
from aragora.skills.registry import (
    SkillRegistry,
    _validate_skill_code,
    validate_skill_imports,
)


# =============================================================================
# Helper: create a skill class from inline source at module level
# =============================================================================


def _make_skill_class_in_module(
    module_source: str,
    class_name: str = "TestSkill",
    module_name: str = "__test_module__",
) -> type:
    """
    Create a skill class that lives inside a synthetic module with the given source.

    This lets validate_skill_imports inspect the module source via inspect.getsource().
    The module is registered in sys.modules so inspect.getmodule() can find it.
    """
    import linecache
    import sys

    mod = types.ModuleType(module_name)
    mod.__file__ = f"<{module_name}>"
    # Store source so inspect.getsource can retrieve it
    linecache.cache[f"<{module_name}>"] = (
        len(module_source),
        None,
        module_source.splitlines(True),
        f"<{module_name}>",
    )
    code = compile(module_source, f"<{module_name}>", "exec")
    exec(code, mod.__dict__)  # noqa: S102 -- test helper only

    # Register in sys.modules so inspect.getmodule() can find it
    sys.modules[module_name] = mod

    cls = getattr(mod, class_name)
    # Set __module__ so inspect.getmodule uses our synthetic module
    cls.__module__ = module_name
    return cls


# =============================================================================
# Concrete skill stubs used across tests
# =============================================================================


class SafeSkill(Skill):
    """A skill with no dangerous imports and properly declared capabilities."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="safe_skill",
            version="1.0.0",
            capabilities=[SkillCapability.WEB_SEARCH],
            input_schema={"query": {"type": "string"}},
        )

    async def execute(self, input_data: dict[str, Any], context: SkillContext) -> SkillResult:
        return SkillResult.create_success({"answer": 42})


class CodeExecSkillClean(Skill):
    """A CODE_EXECUTION skill whose execute() uses only safe code."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="clean_code_exec",
            version="1.0.0",
            capabilities=[SkillCapability.CODE_EXECUTION],
            input_schema={"code": {"type": "string"}},
        )

    async def execute(self, input_data: dict[str, Any], context: SkillContext) -> SkillResult:
        result = input_data.get("code", "")
        return SkillResult.create_success({"output": result})


class ShellSkillWithDeclaredCap(Skill):
    """A skill that declares SHELL_EXECUTION -- should pass import validation."""

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="shell_declared",
            version="1.0.0",
            capabilities=[SkillCapability.SHELL_EXECUTION],
            input_schema={"cmd": {"type": "string"}},
        )

    async def execute(self, input_data: dict[str, Any], context: SkillContext) -> SkillResult:
        return SkillResult.create_success({})


# =============================================================================
# Tests: validate_skill_imports
# =============================================================================


class TestValidateSkillImports:
    """Tests for the validate_skill_imports function."""

    def test_safe_skill_no_warnings(self):
        """A skill with no dangerous imports produces no warnings."""
        skill = SafeSkill()
        warnings = validate_skill_imports(type(skill), skill.manifest)
        # SafeSkill lives in this test module which imports things like pytest,
        # but those are not in the shell/network blocklists, so should be clean
        # or at most reflect this test file's own imports. We check that the
        # skill-specific warnings are empty (no SHELL_EXECUTION / NETWORK mismatch).
        shell_warnings = [w for w in warnings if "SHELL_EXECUTION" in w]
        network_warnings = [w for w in warnings if "NETWORK" in w]
        assert shell_warnings == []
        assert network_warnings == []

    def test_undeclared_subprocess_import_generates_warning(self):
        """A skill module importing subprocess without SHELL_EXECUTION triggers a warning."""
        source = textwrap.dedent("""\
            import subprocess
            from aragora.skills.base import Skill, SkillManifest, SkillCapability, SkillContext, SkillResult
            from typing import Any

            class TestSkill(Skill):
                @property
                def manifest(self):
                    return SkillManifest(
                        name="sneaky_skill",
                        version="1.0.0",
                        capabilities=[SkillCapability.CODE_EXECUTION],
                        input_schema={},
                    )

                async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
                    return SkillResult.create_success({})
        """)
        cls = _make_skill_class_in_module(source, "TestSkill", "__test_subprocess__")
        manifest = SkillManifest(
            name="sneaky_skill",
            version="1.0.0",
            capabilities=[SkillCapability.CODE_EXECUTION],
            input_schema={},
        )
        warnings = validate_skill_imports(cls, manifest)
        assert len(warnings) >= 1
        assert any("subprocess" in w for w in warnings)
        assert any("SHELL_EXECUTION" in w for w in warnings)

    def test_undeclared_socket_import_generates_warning(self):
        """A skill module importing socket without NETWORK triggers a warning."""
        source = textwrap.dedent("""\
            import socket
            from aragora.skills.base import Skill, SkillManifest, SkillCapability, SkillContext, SkillResult
            from typing import Any

            class TestSkill(Skill):
                @property
                def manifest(self):
                    return SkillManifest(
                        name="network_skill",
                        version="1.0.0",
                        capabilities=[SkillCapability.WEB_SEARCH],
                        input_schema={},
                    )

                async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
                    return SkillResult.create_success({})
        """)
        cls = _make_skill_class_in_module(source, "TestSkill", "__test_socket__")
        manifest = SkillManifest(
            name="network_skill",
            version="1.0.0",
            capabilities=[SkillCapability.WEB_SEARCH],
            input_schema={},
        )
        warnings = validate_skill_imports(cls, manifest)
        assert len(warnings) >= 1
        assert any("socket" in w for w in warnings)
        assert any("NETWORK" in w for w in warnings)

    def test_network_import_with_external_api_no_warning(self):
        """Importing requests with EXTERNAL_API declared should NOT warn."""
        source = textwrap.dedent("""\
            import requests
            from aragora.skills.base import Skill, SkillManifest, SkillCapability, SkillContext, SkillResult
            from typing import Any

            class TestSkill(Skill):
                @property
                def manifest(self):
                    return SkillManifest(
                        name="api_skill",
                        version="1.0.0",
                        capabilities=[SkillCapability.EXTERNAL_API],
                        input_schema={},
                    )

                async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
                    return SkillResult.create_success({})
        """)
        cls = _make_skill_class_in_module(source, "TestSkill", "__test_requests__")
        manifest = SkillManifest(
            name="api_skill",
            version="1.0.0",
            capabilities=[SkillCapability.EXTERNAL_API],
            input_schema={},
        )
        warnings = validate_skill_imports(cls, manifest)
        network_warnings = [w for w in warnings if "NETWORK" in w]
        assert network_warnings == []

    def test_shell_import_with_shell_execution_no_warning(self):
        """Importing os with SHELL_EXECUTION declared should NOT warn."""
        source = textwrap.dedent("""\
            import os
            from aragora.skills.base import Skill, SkillManifest, SkillCapability, SkillContext, SkillResult
            from typing import Any

            class TestSkill(Skill):
                @property
                def manifest(self):
                    return SkillManifest(
                        name="shell_skill",
                        version="1.0.0",
                        capabilities=[SkillCapability.SHELL_EXECUTION],
                        input_schema={},
                    )

                async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
                    return SkillResult.create_success({})
        """)
        cls = _make_skill_class_in_module(source, "TestSkill", "__test_os_shell__")
        manifest = SkillManifest(
            name="shell_skill",
            version="1.0.0",
            capabilities=[SkillCapability.SHELL_EXECUTION],
            input_schema={},
        )
        warnings = validate_skill_imports(cls, manifest)
        shell_warnings = [w for w in warnings if "SHELL_EXECUTION" in w]
        assert shell_warnings == []

    def test_uninspectable_module_returns_empty(self):
        """A skill class whose module cannot be inspected returns no warnings."""
        # Built-in types have no gettable source
        manifest = SkillManifest(
            name="builtin_test",
            version="1.0.0",
            capabilities=[],
            input_schema={},
        )
        warnings = validate_skill_imports(int, manifest)
        assert warnings == []


# =============================================================================
# Tests: _validate_skill_code
# =============================================================================


class TestValidateSkillCode:
    """Tests for AST validation of execute() method source."""

    def test_clean_code_exec_skill_no_warnings(self):
        """A CODE_EXECUTION skill with safe execute() produces no warnings."""
        skill = CodeExecSkillClean()
        warnings = _validate_skill_code(skill)
        assert warnings == []

    def test_code_validation_produces_warning_not_exception(self):
        """
        Even if validate_python_code raises CodeValidationError,
        _validate_skill_code returns it as a warning string, not an exception.
        """
        skill = CodeExecSkillClean()
        with patch(
            "aragora.skills.registry.inspect.getsource",
            return_value="import os\nos.system('rm -rf /')\n",
        ):
            warnings = _validate_skill_code(skill)
        assert len(warnings) >= 1
        assert any("validation warning" in w.lower() or "dangerous" in w.lower() for w in warnings)

    def test_uninspectable_execute_returns_empty(self):
        """If inspect.getsource fails, return empty warnings."""
        skill = CodeExecSkillClean()
        with patch(
            "aragora.skills.registry.inspect.getsource",
            side_effect=OSError("no source"),
        ):
            warnings = _validate_skill_code(skill)
        assert warnings == []


# =============================================================================
# Tests: Registration-time validation integration
# =============================================================================


class TestRegistrationValidation:
    """Tests that validation happens during register() and warnings are stored."""

    def test_safe_skill_registers_without_warnings(self):
        """A safe skill registers with no validation warnings."""
        registry = SkillRegistry()
        registry.register(SafeSkill())
        assert registry.has_skill("safe_skill")
        assert "safe_skill" not in registry._validation_warnings

    def test_code_exec_skill_registers_successfully_even_with_warnings(self):
        """
        A CODE_EXECUTION skill that triggers validation warnings still gets
        registered -- warnings are advisory only.
        """
        registry = SkillRegistry()

        # Patch _validate_skill_code to return a warning
        with patch(
            "aragora.skills.registry._validate_skill_code",
            return_value=["Skill 'clean_code_exec' execute() code validation warning: test"],
        ):
            skill = CodeExecSkillClean()
            registry.register(skill)

        # Skill is registered
        assert registry.has_skill("clean_code_exec")
        # Warnings are stored
        assert "clean_code_exec" in registry._validation_warnings
        assert len(registry._validation_warnings["clean_code_exec"]) >= 1

    def test_import_validation_runs_during_registration(self):
        """validate_skill_imports is called during register()."""
        registry = SkillRegistry()

        with patch(
            "aragora.skills.registry.validate_skill_imports",
            return_value=[
                "Skill 'safe_skill' imports ['os'] but does not declare SHELL_EXECUTION capability"
            ],
        ) as mock_validate:
            skill = SafeSkill()
            registry.register(skill)

        mock_validate.assert_called_once_with(type(skill), skill.manifest)
        assert registry.has_skill("safe_skill")
        assert "safe_skill" in registry._validation_warnings

    def test_code_validation_only_for_code_shell_capabilities(self):
        """_validate_skill_code is NOT called for skills without CODE/SHELL capabilities."""
        registry = SkillRegistry()

        with patch(
            "aragora.skills.registry._validate_skill_code",
        ) as mock_code_val:
            # SafeSkill only has WEB_SEARCH capability
            registry.register(SafeSkill())

        mock_code_val.assert_not_called()

    def test_code_validation_called_for_code_execution_cap(self):
        """_validate_skill_code IS called for skills with CODE_EXECUTION."""
        registry = SkillRegistry()

        with patch(
            "aragora.skills.registry._validate_skill_code",
            return_value=[],
        ) as mock_code_val:
            registry.register(CodeExecSkillClean())

        mock_code_val.assert_called_once()

    def test_code_validation_called_for_shell_execution_cap(self):
        """_validate_skill_code IS called for skills with SHELL_EXECUTION."""
        registry = SkillRegistry()

        with patch(
            "aragora.skills.registry._validate_skill_code",
            return_value=[],
        ) as mock_code_val:
            registry.register(ShellSkillWithDeclaredCap())

        mock_code_val.assert_called_once()

    def test_validation_warnings_logged(self, caplog):
        """Validation warnings are logged via logger.warning."""
        registry = SkillRegistry()
        warning_msg = "Skill 'clean_code_exec' test warning"

        with patch(
            "aragora.skills.registry._validate_skill_code",
            return_value=[warning_msg],
        ):
            import logging

            with caplog.at_level(logging.WARNING, logger="aragora.skills.registry"):
                registry.register(CodeExecSkillClean())

        assert any(warning_msg in record.message for record in caplog.records)

    def test_replace_clears_old_warnings(self):
        """Re-registering with replace=True runs validation fresh."""
        registry = SkillRegistry()

        # First registration with a warning
        with patch(
            "aragora.skills.registry._validate_skill_code",
            return_value=["old warning"],
        ):
            registry.register(CodeExecSkillClean())
        assert "clean_code_exec" in registry._validation_warnings

        # Re-register with no warnings
        with patch(
            "aragora.skills.registry._validate_skill_code",
            return_value=[],
        ):
            registry.register(CodeExecSkillClean(), replace=True)

        # Old warnings should be gone (no new warnings to store)
        assert "clean_code_exec" not in registry._validation_warnings
