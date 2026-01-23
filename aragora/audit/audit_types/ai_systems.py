"""
AI Systems Audit Type.

Specialized auditor for AI/ML systems and LLM applications targeting:
- Prompt injection vulnerabilities
- Missing guardrails and output validation
- Hallucination risk indicators
- Model configuration issues
- Data leakage risks
- API key and secrets exposure specific to AI providers
- Model card and documentation completeness

Designed for organizations deploying LLMs, AI agents, and ML pipelines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

from ..base_auditor import AuditorCapabilities, AuditContext, BaseAuditor, ChunkData
from ..document_auditor import AuditFinding, FindingSeverity

logger = logging.getLogger(__name__)


class AIRiskCategory(str, Enum):
    """Categories of AI-specific security findings."""

    PROMPT_INJECTION = "prompt_injection"
    MISSING_GUARDRAILS = "missing_guardrails"
    HALLUCINATION_RISK = "hallucination_risk"
    MODEL_CONFIG = "model_configuration"
    DATA_LEAKAGE = "data_leakage"
    API_SECRETS = "api_secrets"
    OUTPUT_VALIDATION = "output_validation"
    CONTEXT_OVERFLOW = "context_overflow"
    JAILBREAK_VECTOR = "jailbreak_vector"
    COMPLIANCE = "ai_compliance"
    DOCUMENTATION = "documentation"


class AIFramework(str, Enum):
    """Known AI/ML frameworks for pattern targeting."""

    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    TRANSFORMERS = "transformers"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    GUIDANCE = "guidance"
    SEMANTIC_KERNEL = "semantic_kernel"


@dataclass
class AIVulnerabilityPattern:
    """Pattern for detecting AI-specific vulnerabilities."""

    name: str
    pattern: str
    category: AIRiskCategory
    severity: FindingSeverity
    description: str
    recommendation: str
    cwe: Optional[str] = None
    frameworks: list[str] = field(default_factory=lambda: ["*"])
    flags: int = re.IGNORECASE | re.MULTILINE


@dataclass
class AISecretPattern:
    """Pattern for detecting AI provider secrets."""

    name: str
    pattern: str
    severity: FindingSeverity
    provider: str
    description: str = ""


class AISystemsAuditor(BaseAuditor):
    """
    Auditor for AI/ML systems and LLM applications.

    Detects:
    - Prompt injection vulnerabilities
    - Missing guardrails and safety checks
    - Hallucination risk indicators
    - Insecure model configurations
    - Data leakage risks
    - AI provider API key exposure
    - Missing model documentation
    """

    # AI-specific vulnerability patterns
    VULNERABILITY_PATTERNS: list[AIVulnerabilityPattern] = [
        # Prompt Injection - Direct User Input
        AIVulnerabilityPattern(
            name="prompt_injection_direct",
            pattern=r"(?:user_input|user_message|query|request\.body|input_text)\s*[+]?\s*(?:prompt|template|system)",
            category=AIRiskCategory.PROMPT_INJECTION,
            severity=FindingSeverity.CRITICAL,
            cwe="CWE-94",
            description="User input directly concatenated with prompt template - high prompt injection risk",
            recommendation="Use structured prompt templates with proper input sanitization. Consider using guardrail libraries like NeMo Guardrails or Guardrails AI",
        ),
        AIVulnerabilityPattern(
            name="prompt_injection_fstring",
            pattern=r"f['\"].*\{(?:user|input|query|message|request).*\}.*(?:system|assistant|prompt)",
            category=AIRiskCategory.PROMPT_INJECTION,
            severity=FindingSeverity.CRITICAL,
            cwe="CWE-94",
            description="F-string prompt construction with user input - vulnerable to injection",
            recommendation="Use dedicated prompt template libraries with automatic escaping",
        ),
        AIVulnerabilityPattern(
            name="prompt_injection_format",
            pattern=r"\.format\s*\([^)]*(?:user|input|query|message)[^)]*\).*(?:prompt|llm|chat)",
            category=AIRiskCategory.PROMPT_INJECTION,
            severity=FindingSeverity.HIGH,
            cwe="CWE-94",
            description="String format with user input in prompt context",
            recommendation="Use parameterized prompt templates instead of string formatting",
        ),
        # Missing Guardrails
        AIVulnerabilityPattern(
            name="no_output_validation",
            pattern=r"(?:response|completion|output|result)\.(?:content|text|message)\s*$",
            category=AIRiskCategory.MISSING_GUARDRAILS,
            severity=FindingSeverity.MEDIUM,
            description="LLM output used directly without validation",
            recommendation="Add output validation, content filtering, or structured output parsing",
        ),
        AIVulnerabilityPattern(
            name="no_content_filter",
            pattern=r"(?:ChatCompletion|chat|complete)\s*\([^)]*\)\s*(?!.*(?:filter|moderate|validate|guard))",
            category=AIRiskCategory.MISSING_GUARDRAILS,
            severity=FindingSeverity.MEDIUM,
            description="LLM call without apparent content filtering",
            recommendation="Implement content moderation for both input and output",
        ),
        AIVulnerabilityPattern(
            name="missing_length_limit",
            pattern=r"max_tokens\s*=\s*None|max_length\s*=\s*None",
            category=AIRiskCategory.CONTEXT_OVERFLOW,
            severity=FindingSeverity.LOW,
            description="No token/length limit set - potential for excessive resource usage",
            recommendation="Set appropriate max_tokens limits based on use case",
        ),
        # Hallucination Risk
        AIVulnerabilityPattern(
            name="no_grounding",
            pattern=r"(?:generate|complete|chat)\s*\([^)]*\)(?!.*(?:context|documents|retriev|rag|ground))",
            category=AIRiskCategory.HALLUCINATION_RISK,
            severity=FindingSeverity.MEDIUM,
            description="LLM generation without apparent grounding/RAG context",
            recommendation="Consider adding retrieval augmentation or knowledge base grounding",
        ),
        AIVulnerabilityPattern(
            name="high_temperature",
            pattern=r"temperature\s*=\s*(?:1\.(?:[5-9]\d*|[0-4]\d+)|[2-9](?:\.\d+)?)",
            category=AIRiskCategory.HALLUCINATION_RISK,
            severity=FindingSeverity.MEDIUM,
            description="High temperature setting (>1.5) increases hallucination risk",
            recommendation="Use lower temperature (0.0-1.0) for factual tasks, higher only for creative use cases",
        ),
        # Model Configuration Issues
        AIVulnerabilityPattern(
            name="hardcoded_model_version",
            pattern=r"model\s*=\s*['\"](?:gpt-3|gpt-4|claude-2|claude-3)['\"](?!.*(?:config|env|settings))",
            category=AIRiskCategory.MODEL_CONFIG,
            severity=FindingSeverity.LOW,
            description="Hardcoded model version - may miss updates and security patches",
            recommendation="Use configuration/environment variables for model selection",
        ),
        AIVulnerabilityPattern(
            name="deprecated_model",
            pattern=r"model\s*=\s*['\"](?:text-davinci|code-davinci|gpt-3\.5-turbo-0301|claude-instant)['\"]",
            category=AIRiskCategory.MODEL_CONFIG,
            severity=FindingSeverity.MEDIUM,
            description="Using deprecated or legacy model version",
            recommendation="Migrate to current model versions for better security and performance",
        ),
        AIVulnerabilityPattern(
            name="no_timeout",
            pattern=r"(?:openai|anthropic|client)\.(?:chat|complete|generate)\s*\([^)]*\)(?!.*timeout)",
            category=AIRiskCategory.MODEL_CONFIG,
            severity=FindingSeverity.LOW,
            description="LLM API call without timeout setting",
            recommendation="Set appropriate timeout values to prevent hanging requests",
        ),
        # Data Leakage
        AIVulnerabilityPattern(
            name="logging_prompt",
            pattern=r"(?:log|print|console|debug)\s*\([^)]*(?:prompt|message|content|user_input)[^)]*\)",
            category=AIRiskCategory.DATA_LEAKAGE,
            severity=FindingSeverity.HIGH,
            description="Logging user prompts or LLM content - potential PII exposure",
            recommendation="Sanitize or redact sensitive data before logging",
        ),
        AIVulnerabilityPattern(
            name="training_data_exposure",
            pattern=r"(?:training|finetune|dataset).*(?:log|save|export|upload|send)",
            category=AIRiskCategory.DATA_LEAKAGE,
            severity=FindingSeverity.HIGH,
            description="Training data potentially exposed to external systems",
            recommendation="Ensure training data handling complies with data retention policies",
        ),
        AIVulnerabilityPattern(
            name="embedding_storage_unencrypted",
            pattern=r"(?:embedding|vector).*(?:save|store|persist|write)(?!.*(?:encrypt|secure|vault))",
            category=AIRiskCategory.DATA_LEAKAGE,
            severity=FindingSeverity.MEDIUM,
            description="Vector embeddings stored without apparent encryption",
            recommendation="Encrypt embeddings at rest, especially if derived from sensitive data",
        ),
        # Jailbreak Vectors
        AIVulnerabilityPattern(
            name="role_override",
            pattern=r"(?:system|role)\s*[=:]\s*['\"].*(?:ignore|forget|pretend|roleplay|you are now)['\"]",
            category=AIRiskCategory.JAILBREAK_VECTOR,
            severity=FindingSeverity.HIGH,
            description="System prompt contains jailbreak-susceptible patterns",
            recommendation="Harden system prompts against role manipulation attempts",
        ),
        AIVulnerabilityPattern(
            name="no_system_prompt",
            pattern=r"messages\s*=\s*\[\s*\{[^}]*role[^}]*user[^}]*\}",
            category=AIRiskCategory.JAILBREAK_VECTOR,
            severity=FindingSeverity.MEDIUM,
            description="User message sent without system prompt - easier to manipulate",
            recommendation="Always include a system prompt with clear boundaries and instructions",
        ),
        # Output Validation
        AIVulnerabilityPattern(
            name="eval_llm_output",
            pattern=r"eval\s*\([^)]*(?:response|completion|output|result)",
            category=AIRiskCategory.OUTPUT_VALIDATION,
            severity=FindingSeverity.CRITICAL,
            description="eval() used on LLM output - severe code injection risk",
            recommendation="Never execute LLM output directly. Use structured parsing and validation",
        ),
        AIVulnerabilityPattern(
            name="exec_llm_output",
            pattern=r"exec\s*\([^)]*(?:response|completion|output|result|code)",
            category=AIRiskCategory.OUTPUT_VALIDATION,
            severity=FindingSeverity.CRITICAL,
            description="exec() used on LLM output - code execution vulnerability",
            recommendation="Never execute LLM-generated code without sandboxing and approval",
        ),
        AIVulnerabilityPattern(
            name="json_loads_no_validation",
            pattern=r"json\.loads\s*\([^)]*(?:response|completion|output)[^)]*\)(?!.*(?:schema|validate|pydantic))",
            category=AIRiskCategory.OUTPUT_VALIDATION,
            severity=FindingSeverity.MEDIUM,
            description="JSON parsing of LLM output without schema validation",
            recommendation="Use Pydantic or JSON schema validation for structured outputs",
        ),
        # Compliance
        AIVulnerabilityPattern(
            name="no_usage_tracking",
            pattern=r"(?:openai|anthropic|client)\.[^(]*\([^)]*\)(?!.*(?:track|log|metric|usage|token))",
            category=AIRiskCategory.COMPLIANCE,
            severity=FindingSeverity.LOW,
            description="LLM API call without usage tracking",
            recommendation="Track token usage for cost management and compliance reporting",
        ),
    ]

    # AI Provider Secret Patterns
    SECRET_PATTERNS: list[AISecretPattern] = [
        AISecretPattern(
            name="openai_api_key",
            pattern=r"sk-[a-zA-Z0-9]{48}",
            severity=FindingSeverity.CRITICAL,
            provider="OpenAI",
            description="OpenAI API key",
        ),
        AISecretPattern(
            name="anthropic_api_key",
            pattern=r"sk-ant-[a-zA-Z0-9]{40,}",
            severity=FindingSeverity.CRITICAL,
            provider="Anthropic",
            description="Anthropic API key",
        ),
        AISecretPattern(
            name="cohere_api_key",
            pattern=r"[a-zA-Z0-9]{40}",  # Cohere keys are 40 chars
            severity=FindingSeverity.HIGH,
            provider="Cohere",
            description="Possible Cohere API key",
        ),
        AISecretPattern(
            name="huggingface_token",
            pattern=r"hf_[a-zA-Z0-9]{34}",
            severity=FindingSeverity.HIGH,
            provider="Hugging Face",
            description="Hugging Face access token",
        ),
        AISecretPattern(
            name="replicate_token",
            pattern=r"r8_[a-zA-Z0-9]{36}",
            severity=FindingSeverity.HIGH,
            provider="Replicate",
            description="Replicate API token",
        ),
        AISecretPattern(
            name="pinecone_api_key",
            pattern=r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
            severity=FindingSeverity.HIGH,
            provider="Pinecone",
            description="Possible Pinecone API key (UUID format)",
        ),
        AISecretPattern(
            name="wandb_api_key",
            pattern=r"[a-f0-9]{40}",
            severity=FindingSeverity.MEDIUM,
            provider="Weights & Biases",
            description="Possible W&B API key",
        ),
    ]

    # Documentation check patterns
    DOCUMENTATION_PATTERNS = [
        {
            "name": "model_card_missing",
            "check": "model_card",
            "required_fields": ["model_name", "capabilities", "limitations", "training_data"],
        },
        {
            "name": "safety_guidelines_missing",
            "check": "safety",
            "required_sections": ["content_policy", "usage_restrictions", "harm_prevention"],
        },
    ]

    def __init__(self, **kwargs):
        """Initialize AI Systems Auditor."""
        super().__init__(**kwargs)
        self._compiled_patterns: list[tuple] = []
        self._compiled_secrets: list[tuple] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for p in self.VULNERABILITY_PATTERNS:
            try:
                self._compiled_patterns.append((re.compile(p.pattern, p.flags), p))
            except re.error as e:
                logger.warning(f"Invalid pattern {p.name}: {e}")

        for s in self.SECRET_PATTERNS:
            try:
                self._compiled_secrets.append((re.compile(s.pattern), s))
            except re.error as e:
                logger.warning(f"Invalid secret pattern {s.name}: {e}")

    @property
    def audit_type_id(self) -> str:
        return "ai_systems"

    @property
    def display_name(self) -> str:
        return "AI Systems Security Audit"

    @property
    def description(self) -> str:
        return (
            "Audits AI/ML systems for prompt injection, missing guardrails, "
            "hallucination risks, data leakage, and compliance issues"
        )

    @property
    def capabilities(self) -> AuditorCapabilities:
        return AuditorCapabilities(
            file_types=["py", "js", "ts", "yaml", "json", "md", "txt"],
            languages=["en"],
            specialized_domains=["ai", "ml", "llm", "nlp"],
            requires_llm=False,  # Pattern-based, no LLM needed
            max_chunk_size=50000,
        )

    def _is_example_or_test(self, content: str, line: str) -> bool:
        """Check if line is in example/test context."""
        indicators = [
            "example",
            "test",
            "demo",
            "sample",
            "mock",
            "fixture",
            "TODO",
            "FIXME",
            "placeholder",
            "fake",
            "dummy",
        ]
        line_lower = line.lower()
        return any(ind in line_lower for ind in indicators)

    async def analyze_chunk(
        self,
        chunk: ChunkData,
        context: AuditContext,
    ) -> Sequence[AuditFinding]:
        """
        Analyze a single code chunk for AI-specific vulnerabilities.

        Args:
            chunk: Code chunk to analyze
            context: Audit context with session info

        Returns:
            List of audit findings
        """
        findings: list[AuditFinding] = []
        content = chunk.content
        lines = content.split("\n")

        # Check vulnerability patterns
        for compiled, pattern in self._compiled_patterns:
            for match in compiled.finditer(content):
                # Get line number
                line_start = content[: match.start()].count("\n") + 1
                matched_line = lines[line_start - 1] if line_start <= len(lines) else ""

                # Skip examples/tests
                if self._is_example_or_test(content, matched_line):
                    continue

                # Create finding
                finding = context.create_finding(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    title=f"AI Security: {pattern.name}",
                    description=pattern.description,
                    severity=pattern.severity,
                    category=pattern.category.value,
                    confidence=0.8,
                    evidence_text=matched_line.strip()[:200],
                    evidence_location=f"Line {line_start}",
                    recommendation=pattern.recommendation,
                    found_by="ai_systems_auditor",
                    tags=["ai", "llm", pattern.category.value],
                )
                if pattern.cwe:
                    finding.tags.append(pattern.cwe)
                findings.append(finding)

        # Check secret patterns
        for compiled, secret in self._compiled_secrets:
            for match in compiled.finditer(content):
                line_start = content[: match.start()].count("\n") + 1
                matched_line = lines[line_start - 1] if line_start <= len(lines) else ""

                # Skip if in example/test context
                if self._is_example_or_test(content, matched_line):
                    continue

                # Mask the secret in evidence
                masked = match.group()[:8] + "..." + match.group()[-4:]

                finding = context.create_finding(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    title=f"AI API Key Exposed: {secret.provider}",
                    description=f"Potential {secret.description} found in code",
                    severity=secret.severity,
                    category=AIRiskCategory.API_SECRETS.value,
                    confidence=0.85,
                    evidence_text=f"Key: {masked}",
                    evidence_location=f"Line {line_start}",
                    recommendation=f"Remove {secret.provider} API key from code. Use environment variables or secret management.",
                    found_by="ai_systems_auditor",
                    tags=["ai", "secrets", secret.provider.lower().replace(" ", "_")],
                )
                findings.append(finding)

        return findings

    async def cross_document_analysis(
        self,
        chunks: Sequence[ChunkData],
        context: AuditContext,
    ) -> Sequence[AuditFinding]:
        """
        Analyze patterns across multiple documents.

        Checks for:
        - Consistent guardrail usage across files
        - Missing model documentation
        - Inconsistent model configurations
        """
        findings: list[AuditFinding] = []
        all_content = "\n".join(c.content for c in chunks)

        # Check for guardrail library usage
        guardrail_libs = [
            "nemo_guardrails",
            "guardrails",
            "rebuff",
            "llm-guard",
            "prompt-injection-detector",
        ]
        has_guardrails = any(lib in all_content.lower() for lib in guardrail_libs)

        # Check for LLM usage
        llm_usage_patterns = [
            r"openai\.",
            r"anthropic\.",
            r"ChatCompletion",
            r"llm\.(?:generate|chat|complete)",
            r"langchain",
            r"llama_index",
        ]
        has_llm_usage = any(re.search(p, all_content, re.IGNORECASE) for p in llm_usage_patterns)

        if has_llm_usage and not has_guardrails:
            finding = context.create_finding(
                document_id="cross_document",
                title="AI Systems: No Guardrail Library Detected",
                description="LLM usage detected but no guardrail library found in codebase",
                severity=FindingSeverity.MEDIUM,
                category=AIRiskCategory.MISSING_GUARDRAILS.value,
                confidence=0.7,
                evidence_text="LLM calls found without guardrail imports",
                evidence_location="Project-wide",
                recommendation=(
                    "Consider adding a guardrail library like NeMo Guardrails, "
                    "Guardrails AI, or Rebuff for input/output protection"
                ),
                found_by="ai_systems_auditor",
                tags=["ai", "guardrails", "cross_document"],
            )
            findings.append(finding)

        # Check for model documentation
        doc_patterns = [
            r"model.?card",
            r"MODEL\.md",
            r"model.?documentation",
            r"capabilities.*limitations",
        ]
        has_model_docs = any(re.search(p, all_content, re.IGNORECASE) for p in doc_patterns)

        if has_llm_usage and not has_model_docs:
            finding = context.create_finding(
                document_id="cross_document",
                title="AI Systems: Missing Model Documentation",
                description="No model card or AI documentation detected",
                severity=FindingSeverity.LOW,
                category=AIRiskCategory.DOCUMENTATION.value,
                confidence=0.6,
                evidence_text="No model card patterns found",
                evidence_location="Project-wide",
                recommendation=(
                    "Create a MODEL_CARD.md documenting model capabilities, "
                    "limitations, intended use, and safety considerations"
                ),
                found_by="ai_systems_auditor",
                tags=["ai", "documentation", "cross_document"],
            )
            findings.append(finding)

        return findings

    def get_risk_summary(self, findings: Sequence[AuditFinding]) -> dict[str, Any]:
        """
        Generate a summary of AI-specific risks from findings.

        Returns:
            Dictionary with risk categories and counts
        """
        summary = {
            "total_findings": len(findings),
            "by_severity": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            },
            "by_category": {},
            "top_risks": [],
            "recommendations": [],
        }

        for finding in findings:
            # Count by severity
            severity_key = finding.severity.value.lower()
            if severity_key in summary["by_severity"]:
                summary["by_severity"][severity_key] += 1

            # Count by category
            cat = finding.category
            if cat not in summary["by_category"]:
                summary["by_category"][cat] = 0
            summary["by_category"][cat] += 1

        # Top risks
        critical_high = [
            f for f in findings if f.severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH]
        ]
        summary["top_risks"] = [
            {"title": f.title, "severity": f.severity.value, "location": f.evidence_location}
            for f in critical_high[:5]
        ]

        # Unique recommendations
        seen_recs = set()
        for finding in findings:
            if finding.recommendation and finding.recommendation not in seen_recs:
                summary["recommendations"].append(finding.recommendation)
                seen_recs.add(finding.recommendation)

        return summary


# Convenience function for quick audit
async def audit_ai_code(
    code: str,
    file_path: str = "unknown",
) -> list[dict[str, Any]]:
    """
    Quick audit of AI/ML code for vulnerabilities.

    Args:
        code: Source code to audit
        file_path: Optional file path for context

    Returns:
        List of findings as dictionaries
    """
    from ..document_auditor import AuditSession

    session = AuditSession(workspace_id="quick_audit")
    context = AuditContext(session=session)

    chunk = ChunkData(
        document_id=file_path,
        chunk_id="0",
        content=code,
        metadata={"file_path": file_path},
    )

    auditor = AISystemsAuditor()
    findings = await auditor.analyze_chunk(chunk, context)

    return [
        {
            "title": f.title,
            "description": f.description,
            "severity": f.severity.value,
            "category": f.category,
            "evidence": f.evidence_text,
            "location": f.evidence_location,
            "recommendation": f.recommendation,
        }
        for f in findings
    ]
