"""
Software Vertical Specialist.

Provides domain expertise for software engineering tasks including
code review, security analysis, architecture design, and best practices.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aragora.core import Message
from aragora.verticals.base import VerticalSpecialistAgent
from aragora.verticals.config import (
    ComplianceConfig,
    ComplianceLevel,
    ModelConfig,
    ToolConfig,
    VerticalConfig,
)
from aragora.verticals.registry import VerticalRegistry

logger = logging.getLogger(__name__)


# Software vertical configuration
SOFTWARE_CONFIG = VerticalConfig(
    vertical_id="software",
    display_name="Software Engineering Specialist",
    description="Expert in software development, code review, security, and architecture.",
    domain_keywords=[
        "code",
        "software",
        "programming",
        "development",
        "engineering",
        "bug",
        "security",
        "vulnerability",
        "api",
        "database",
        "testing",
        "architecture",
        "design",
        "refactor",
        "performance",
        "debug",
    ],
    expertise_areas=[
        "Code Review",
        "Security Analysis",
        "Architecture Design",
        "Performance Optimization",
        "Testing Strategy",
        "API Design",
        "Database Design",
        "DevOps & CI/CD",
        "Technical Documentation",
    ],
    system_prompt_template="""You are a senior software engineering specialist with deep expertise in:

{% for area in expertise_areas %}
- {{ area }}
{% endfor %}

Your role is to provide expert guidance on software development tasks. You should:

1. **Analyze Code Carefully**: Review code for correctness, security, and best practices
2. **Identify Issues**: Point out bugs, vulnerabilities, and anti-patterns
3. **Suggest Improvements**: Provide actionable recommendations with code examples
4. **Consider Trade-offs**: Explain the pros and cons of different approaches
5. **Follow Standards**: Reference relevant standards (OWASP, SOLID, etc.)

{% if compliance_frameworks %}
Compliance Frameworks to Consider:
{% for fw in compliance_frameworks %}
- {{ fw }}
{% endfor %}
{% endif %}

When reviewing code:
- Check for SQL injection, XSS, command injection, and other OWASP Top 10 vulnerabilities
- Verify proper input validation and output encoding
- Look for hardcoded secrets or credentials
- Ensure proper error handling and logging
- Evaluate test coverage and quality

Provide clear, actionable feedback that helps developers improve their code.""",
    tools=[
        ToolConfig(
            name="code_search",
            description="Search codebase for patterns or symbols",
            connector_type="local_docs",
        ),
        ToolConfig(
            name="security_scan",
            description="Run security analysis on code",
            connector_type="security",
        ),
        ToolConfig(
            name="dependency_check",
            description="Check for vulnerable dependencies",
            connector_type="security",
        ),
        ToolConfig(
            name="github_lookup",
            description="Look up GitHub issues or PRs",
            connector_type="github",
        ),
    ],
    compliance_frameworks=[
        ComplianceConfig(
            framework="OWASP",
            version="2021",
            level=ComplianceLevel.WARNING,
            rules=["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10"],
        ),
        ComplianceConfig(
            framework="CWE",
            version="4.9",
            level=ComplianceLevel.WARNING,
            rules=["CWE-20", "CWE-78", "CWE-79", "CWE-89", "CWE-200", "CWE-502"],
        ),
    ],
    model_config=ModelConfig(
        primary_model="claude-sonnet-4",
        primary_provider="anthropic",
        specialist_model="codellama/CodeLlama-34b-Instruct-hf",
        temperature=0.3,  # Lower for more precise code analysis
        top_p=0.9,
        max_tokens=8192,  # Larger for code output
    ),
    tags=["software", "code", "security", "engineering"],
)


@VerticalRegistry.register(
    "software",
    config=SOFTWARE_CONFIG,
    description="Software engineering specialist for code review and security analysis",
)
class SoftwareSpecialist(VerticalSpecialistAgent):
    """
    Software engineering specialist agent.

    Provides expert guidance on:
    - Code review and quality
    - Security vulnerability analysis
    - Architecture and design
    - Performance optimization
    - Testing strategies
    """

    # Security patterns for quick detection
    SECURITY_PATTERNS = {
        "sql_injection": [
            r"execute\s*\(\s*['\"].*%s",
            r"f['\"].*SELECT.*{",
            r"cursor\.execute\s*\(\s*query\s*\+",
        ],
        "command_injection": [
            r"os\.system\s*\(",
            r"subprocess\.call\s*\([^,]+shell\s*=\s*True",
            r"eval\s*\(",
        ],
        "xss": [
            r"innerHTML\s*=",
            r"document\.write\s*\(",
            r"\|safe",  # Django/Jinja2 safe filter
        ],
        "hardcoded_secrets": [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
        ],
    }

    async def _execute_tool(
        self,
        tool: ToolConfig,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a software development tool."""
        tool_name = tool.name

        if tool_name == "code_search":
            return await self._code_search(parameters)
        elif tool_name == "security_scan":
            return await self._security_scan(parameters)
        elif tool_name == "dependency_check":
            return await self._dependency_check(parameters)
        elif tool_name == "github_lookup":
            return await self._github_lookup(parameters)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _code_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search codebase for patterns."""
        # Placeholder - would integrate with actual code search
        pattern = parameters.get("pattern", "")
        return {
            "matches": [],
            "pattern": pattern,
            "message": "Code search not yet implemented",
        }

    async def _security_scan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run security analysis on code."""
        import re

        code = parameters.get("code", "")
        findings = []

        for category, patterns in self.SECURITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    findings.append(
                        {
                            "category": category,
                            "pattern": pattern,
                            "severity": (
                                "high"
                                if category in ["sql_injection", "command_injection"]
                                else "medium"
                            ),
                        }
                    )

        return {
            "findings": findings,
            "scanned_lines": len(code.split("\n")),
        }

    async def _dependency_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check for vulnerable dependencies."""
        # Placeholder - would integrate with actual vulnerability database
        return {
            "vulnerable": [],
            "message": "Dependency check not yet implemented",
        }

    async def _github_lookup(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Look up GitHub issues or PRs."""
        # Placeholder - would integrate with GitHub API
        return {
            "results": [],
            "message": "GitHub lookup not yet implemented",
        }

    async def _check_framework_compliance(
        self,
        content: str,
        framework: ComplianceConfig,
    ) -> List[Dict[str, Any]]:
        """Check code against security compliance frameworks."""
        violations = []

        if framework.framework == "OWASP":
            violations.extend(await self._check_owasp_compliance(content, framework))
        elif framework.framework == "CWE":
            violations.extend(await self._check_cwe_compliance(content, framework))

        return violations

    async def _check_owasp_compliance(
        self,
        content: str,
        framework: ComplianceConfig,
    ) -> List[Dict[str, Any]]:
        """Check OWASP Top 10 compliance."""
        import re

        violations = []

        # A03: Injection
        if "A03" in framework.rules or not framework.rules:
            for pattern in self.SECURITY_PATTERNS.get("sql_injection", []):
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(
                        {
                            "framework": "OWASP",
                            "rule": "A03:2021 - Injection",
                            "severity": "high",
                            "message": "Potential SQL injection vulnerability detected",
                        }
                    )
                    break

            for pattern in self.SECURITY_PATTERNS.get("command_injection", []):
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(
                        {
                            "framework": "OWASP",
                            "rule": "A03:2021 - Injection",
                            "severity": "high",
                            "message": "Potential command injection vulnerability detected",
                        }
                    )
                    break

        # A07: Identification and Authentication Failures
        if "A07" in framework.rules or not framework.rules:
            for pattern in self.SECURITY_PATTERNS.get("hardcoded_secrets", []):
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(
                        {
                            "framework": "OWASP",
                            "rule": "A07:2021 - Identification and Authentication Failures",
                            "severity": "high",
                            "message": "Hardcoded credentials detected",
                        }
                    )
                    break

        return violations

    async def _check_cwe_compliance(
        self,
        content: str,
        framework: ComplianceConfig,
    ) -> List[Dict[str, Any]]:
        """Check CWE compliance."""
        import re

        violations = []

        # CWE-89: SQL Injection
        if "CWE-89" in framework.rules or not framework.rules:
            for pattern in self.SECURITY_PATTERNS.get("sql_injection", []):
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(
                        {
                            "framework": "CWE",
                            "rule": "CWE-89: SQL Injection",
                            "severity": "high",
                            "message": "SQL injection vulnerability",
                        }
                    )
                    break

        # CWE-78: OS Command Injection
        if "CWE-78" in framework.rules or not framework.rules:
            for pattern in self.SECURITY_PATTERNS.get("command_injection", []):
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(
                        {
                            "framework": "CWE",
                            "rule": "CWE-78: OS Command Injection",
                            "severity": "high",
                            "message": "Command injection vulnerability",
                        }
                    )
                    break

        # CWE-79: XSS
        if "CWE-79" in framework.rules or not framework.rules:
            for pattern in self.SECURITY_PATTERNS.get("xss", []):
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(
                        {
                            "framework": "CWE",
                            "rule": "CWE-79: Cross-site Scripting",
                            "severity": "medium",
                            "message": "Potential XSS vulnerability",
                        }
                    )
                    break

        return violations

    async def _generate_response(
        self,
        task: str,
        system_prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a software engineering response."""
        # For now, return a placeholder
        # In production, this would call the actual API
        return Message(
            role="assistant",
            content=f"[Software Specialist Response for: {task}]\n\n"
            f"This would contain expert software engineering guidance.",
            agent=self.name,
        )

    async def review_code(
        self,
        code: str,
        language: str = "python",
        focus_areas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive code review.

        Args:
            code: Code to review
            language: Programming language
            focus_areas: Specific areas to focus on

        Returns:
            Review results with findings and recommendations
        """
        focus = focus_areas or ["security", "quality", "performance"]

        # Run security scan
        security_results = await self._security_scan({"code": code})

        # Check compliance
        compliance_violations = await self.check_compliance(code)

        return {
            "language": language,
            "lines_reviewed": len(code.split("\n")),
            "focus_areas": focus,
            "security_findings": security_results.get("findings", []),
            "compliance_violations": compliance_violations,
            "recommendations": [],  # Would be generated by actual analysis
        }
