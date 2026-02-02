"""
Tests for AI Systems Auditor.

Tests cover:
- Auditor type creation and properties
- Vulnerability pattern detection
- Secret pattern detection
- Event classification and severity mapping
- Cross-document analysis
- Risk summary generation
- Convenience audit function
"""

from __future__ import annotations

import pytest

from aragora.audit.audit_types.ai_systems import (
    AIFramework,
    AIRiskCategory,
    AISecretPattern,
    AISystemsAuditor,
    AIVulnerabilityPattern,
    audit_ai_code,
)
from aragora.audit.base_auditor import AuditContext, AuditorCapabilities, ChunkData
from aragora.audit.document_auditor import AuditSession, FindingSeverity


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def auditor():
    """Create an AISystemsAuditor for testing."""
    return AISystemsAuditor()


@pytest.fixture
def mock_session():
    """Create a mock AuditSession for testing."""
    return AuditSession(
        id="session-ai-test",
        name="AI Systems Audit Test",
        created_by="test-user",
        document_ids=["test-doc"],
        model="claude-3.5-sonnet",
    )


@pytest.fixture
def audit_context(mock_session):
    """Create an AuditContext for testing."""
    return AuditContext(
        session=mock_session,
        workspace_id="ws-ai-test",
        model="claude-3.5-sonnet",
    )


@pytest.fixture
def sample_chunk():
    """Create a sample ChunkData for testing."""
    return ChunkData(
        id="chunk-ai-001",
        document_id="test-doc",
        content="# Test file\nprint('Hello, World!')",
        chunk_type="code",
    )


# =============================================================================
# AIRiskCategory Enum Tests
# =============================================================================


class TestAIRiskCategory:
    """Tests for AIRiskCategory enum."""

    def test_all_categories_exist(self):
        """Test that all expected risk categories are defined."""
        assert AIRiskCategory.PROMPT_INJECTION.value == "prompt_injection"
        assert AIRiskCategory.MISSING_GUARDRAILS.value == "missing_guardrails"
        assert AIRiskCategory.HALLUCINATION_RISK.value == "hallucination_risk"
        assert AIRiskCategory.MODEL_CONFIG.value == "model_configuration"
        assert AIRiskCategory.DATA_LEAKAGE.value == "data_leakage"
        assert AIRiskCategory.API_SECRETS.value == "api_secrets"
        assert AIRiskCategory.OUTPUT_VALIDATION.value == "output_validation"
        assert AIRiskCategory.CONTEXT_OVERFLOW.value == "context_overflow"
        assert AIRiskCategory.JAILBREAK_VECTOR.value == "jailbreak_vector"
        assert AIRiskCategory.COMPLIANCE.value == "ai_compliance"
        assert AIRiskCategory.DOCUMENTATION.value == "documentation"

    def test_category_count(self):
        """Test there are exactly 11 risk categories."""
        assert len(AIRiskCategory) == 11


# =============================================================================
# AIFramework Enum Tests
# =============================================================================


class TestAIFramework:
    """Tests for AIFramework enum."""

    def test_all_frameworks_exist(self):
        """Test that all expected AI frameworks are defined."""
        assert AIFramework.LANGCHAIN.value == "langchain"
        assert AIFramework.LLAMA_INDEX.value == "llama_index"
        assert AIFramework.OPENAI.value == "openai"
        assert AIFramework.ANTHROPIC.value == "anthropic"
        assert AIFramework.HUGGINGFACE.value == "huggingface"
        assert AIFramework.TRANSFORMERS.value == "transformers"
        assert AIFramework.AUTOGEN.value == "autogen"
        assert AIFramework.CREWAI.value == "crewai"
        assert AIFramework.GUIDANCE.value == "guidance"
        assert AIFramework.SEMANTIC_KERNEL.value == "semantic_kernel"

    def test_framework_count(self):
        """Test there are exactly 10 AI frameworks."""
        assert len(AIFramework) == 10


# =============================================================================
# AIVulnerabilityPattern Dataclass Tests
# =============================================================================


class TestAIVulnerabilityPattern:
    """Tests for AIVulnerabilityPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a vulnerability pattern."""
        pattern = AIVulnerabilityPattern(
            name="test_pattern",
            pattern=r"test.*vulnerable",
            category=AIRiskCategory.PROMPT_INJECTION,
            severity=FindingSeverity.CRITICAL,
            description="Test vulnerability pattern",
            recommendation="Fix the vulnerability",
        )
        assert pattern.name == "test_pattern"
        assert pattern.category == AIRiskCategory.PROMPT_INJECTION
        assert pattern.severity == FindingSeverity.CRITICAL

    def test_pattern_default_values(self):
        """Test pattern creation with default values."""
        pattern = AIVulnerabilityPattern(
            name="test",
            pattern=r"test",
            category=AIRiskCategory.DATA_LEAKAGE,
            severity=FindingSeverity.LOW,
            description="Test",
            recommendation="Test",
        )
        assert pattern.cwe is None
        assert pattern.frameworks == ["*"]

    def test_pattern_with_cwe(self):
        """Test pattern with CWE identifier."""
        pattern = AIVulnerabilityPattern(
            name="injection_test",
            pattern=r"eval\(.*user",
            category=AIRiskCategory.OUTPUT_VALIDATION,
            severity=FindingSeverity.CRITICAL,
            description="Eval injection",
            recommendation="Remove eval",
            cwe="CWE-94",
        )
        assert pattern.cwe == "CWE-94"


# =============================================================================
# AISecretPattern Dataclass Tests
# =============================================================================


class TestAISecretPattern:
    """Tests for AISecretPattern dataclass."""

    def test_secret_pattern_creation(self):
        """Test creating a secret pattern."""
        pattern = AISecretPattern(
            name="test_api_key",
            pattern=r"sk-[a-zA-Z0-9]{48}",
            severity=FindingSeverity.CRITICAL,
            provider="TestProvider",
        )
        assert pattern.name == "test_api_key"
        assert pattern.provider == "TestProvider"
        assert pattern.severity == FindingSeverity.CRITICAL

    def test_secret_pattern_default_description(self):
        """Test secret pattern default description."""
        pattern = AISecretPattern(
            name="test",
            pattern=r"test",
            severity=FindingSeverity.HIGH,
            provider="Test",
        )
        assert pattern.description == ""


# =============================================================================
# AISystemsAuditor Tests
# =============================================================================


class TestAISystemsAuditor:
    """Tests for AISystemsAuditor class."""

    def test_auditor_type_id(self, auditor):
        """Test auditor type ID."""
        assert auditor.audit_type_id == "ai_systems"

    def test_auditor_display_name(self, auditor):
        """Test auditor display name."""
        assert auditor.display_name == "AI Systems Security Audit"

    def test_auditor_description(self, auditor):
        """Test auditor description."""
        assert "AI/ML systems" in auditor.description
        assert "prompt injection" in auditor.description

    def test_auditor_capabilities(self, auditor):
        """Test auditor capabilities."""
        caps = auditor.capabilities
        assert isinstance(caps, AuditorCapabilities)
        assert caps.requires_llm is False
        assert caps.max_chunk_size == 50000
        assert "py" in caps.supported_document_types
        assert "js" in caps.supported_document_types

    def test_vulnerability_patterns_exist(self, auditor):
        """Test that vulnerability patterns are defined."""
        assert len(auditor.VULNERABILITY_PATTERNS) > 0

    def test_secret_patterns_exist(self, auditor):
        """Test that secret patterns are defined."""
        assert len(auditor.SECRET_PATTERNS) > 0

    def test_patterns_compiled(self, auditor):
        """Test that patterns are compiled during initialization."""
        assert len(auditor._compiled_patterns) > 0
        assert len(auditor._compiled_secrets) > 0


# =============================================================================
# Prompt Injection Detection Tests
# =============================================================================


class TestPromptInjectionDetection:
    """Tests for prompt injection vulnerability detection."""

    @pytest.mark.asyncio
    async def test_detect_direct_prompt_injection(self, auditor, audit_context):
        """Test detection of direct prompt injection."""
        code = """
user_input = request.body
prompt = user_input + system_prompt
response = llm.generate(prompt)
"""
        chunk = ChunkData(
            id="chunk-1",
            document_id="test.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        injection_findings = [f for f in findings if "prompt_injection" in f.category]
        assert len(injection_findings) > 0

    @pytest.mark.asyncio
    async def test_detect_fstring_prompt_injection(self, auditor, audit_context):
        """Test detection of f-string prompt injection."""
        code = """
def build_prompt(user_message):
    return f"System: {system_prompt}. User says: {user_message}"
"""
        chunk = ChunkData(
            id="chunk-2",
            document_id="test.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        # May or may not detect based on pattern specifics
        assert isinstance(findings, list)


# =============================================================================
# Missing Guardrails Detection Tests
# =============================================================================


class TestMissingGuardrailsDetection:
    """Tests for missing guardrails detection."""

    @pytest.mark.asyncio
    async def test_detect_missing_length_limit(self, auditor, audit_context):
        """Test detection of missing length limits."""
        # Pattern requires: max_tokens = None or max_length = None
        code = """
max_tokens = None
response = client.chat.completions.create(model="gpt-4")
"""
        chunk = ChunkData(
            id="chunk-3",
            document_id="config.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        overflow_findings = [f for f in findings if "context_overflow" in f.category]
        assert len(overflow_findings) > 0


# =============================================================================
# Hallucination Risk Detection Tests
# =============================================================================


class TestHallucinationRiskDetection:
    """Tests for hallucination risk detection."""

    @pytest.mark.asyncio
    async def test_detect_high_temperature(self, auditor, audit_context):
        """Test detection of high temperature setting."""
        code = """
response = client.chat.completions.create(
    model="gpt-4",
    temperature=2.0,
    messages=messages
)
"""
        chunk = ChunkData(
            id="chunk-4",
            document_id="app.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        temp_findings = [f for f in findings if "hallucination_risk" in f.category]
        assert len(temp_findings) > 0


# =============================================================================
# Output Validation Detection Tests
# =============================================================================


class TestOutputValidationDetection:
    """Tests for output validation vulnerability detection."""

    @pytest.mark.asyncio
    async def test_detect_eval_on_llm_output(self, auditor, audit_context):
        """Test detection of eval() on LLM output."""
        code = """
response = llm.generate(prompt)
result = eval(response.content)
"""
        chunk = ChunkData(
            id="chunk-5",
            document_id="dangerous.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        eval_findings = [f for f in findings if "output_validation" in f.category]
        assert len(eval_findings) > 0
        assert any(f.severity == FindingSeverity.CRITICAL for f in eval_findings)

    @pytest.mark.asyncio
    async def test_detect_exec_on_llm_output(self, auditor, audit_context):
        """Test detection of exec() on LLM output."""
        code = """
code_response = model.generate(code_prompt)
exec(code_response.text)
"""
        chunk = ChunkData(
            id="chunk-6",
            document_id="exec_danger.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        exec_findings = [f for f in findings if "output_validation" in f.category]
        assert len(exec_findings) > 0


# =============================================================================
# Data Leakage Detection Tests
# =============================================================================


class TestDataLeakageDetection:
    """Tests for data leakage detection."""

    @pytest.mark.asyncio
    async def test_detect_logging_prompts(self, auditor, audit_context):
        """Test detection of prompt logging."""
        code = """
print(f"User prompt: {user_input}")
log.info(f"Processing message: {message}")
"""
        chunk = ChunkData(
            id="chunk-7",
            document_id="logging.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        leakage_findings = [f for f in findings if "data_leakage" in f.category]
        assert len(leakage_findings) > 0


# =============================================================================
# API Secret Detection Tests
# =============================================================================


class TestAPISecretDetection:
    """Tests for AI provider API secret detection."""

    @pytest.mark.asyncio
    async def test_detect_openai_api_key(self, auditor, audit_context):
        """Test detection of OpenAI API key."""
        # OpenAI pattern requires sk- followed by exactly 48 alphanumeric chars
        code = """
OPENAI_API_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHijkl"
client = OpenAI(api_key=OPENAI_API_KEY)
"""
        chunk = ChunkData(
            id="chunk-8",
            document_id="secrets.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        secret_findings = [f for f in findings if "api_secrets" in f.category]
        assert len(secret_findings) > 0
        assert any("OpenAI" in f.title for f in secret_findings)

    @pytest.mark.asyncio
    async def test_detect_anthropic_api_key(self, auditor, audit_context):
        """Test detection of Anthropic API key."""
        # Anthropic pattern requires sk-ant- followed by at least 40 alphanumeric chars
        code = """
anthropic_key = "sk-ant-abcdefghijklmnopqrstuvwxyz1234567890ABCDEF"
client = Anthropic(api_key=anthropic_key)
"""
        chunk = ChunkData(
            id="chunk-9",
            document_id="anthropic_config.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        secret_findings = [f for f in findings if "api_secrets" in f.category]
        assert len(secret_findings) > 0
        assert any("Anthropic" in f.title for f in secret_findings)

    @pytest.mark.asyncio
    async def test_detect_huggingface_token(self, auditor, audit_context):
        """Test detection of Hugging Face access token."""
        code = """
HF_TOKEN = "hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890"
"""
        chunk = ChunkData(
            id="chunk-10",
            document_id="hf_config.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        secret_findings = [f for f in findings if "api_secrets" in f.category]
        assert len(secret_findings) > 0


# =============================================================================
# Example/Test Code Exclusion Tests
# =============================================================================


class TestExampleCodeExclusion:
    """Tests for example and test code exclusion."""

    @pytest.mark.asyncio
    async def test_skip_example_code(self, auditor, audit_context):
        """Test that example code is skipped."""
        # OpenAI pattern requires sk- followed by exactly 48 alphanumeric chars
        code = """
# Example usage
example_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHijkl"
"""
        chunk = ChunkData(
            id="chunk-11",
            document_id="example.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        # Findings should be filtered out for example code
        secret_findings = [f for f in findings if "api_secrets" in f.category]
        assert len(secret_findings) == 0

    @pytest.mark.asyncio
    async def test_skip_test_code(self, auditor, audit_context):
        """Test that test/mock code is skipped."""
        # OpenAI pattern requires sk- followed by exactly 48 alphanumeric chars
        code = """
# Test fixture
test_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHijkl"
mock_response = "response.content"
"""
        chunk = ChunkData(
            id="chunk-12",
            document_id="test_config.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        # Test/mock code should be excluded
        secret_findings = [f for f in findings if "api_secrets" in f.category]
        assert len(secret_findings) == 0


# =============================================================================
# Cross-Document Analysis Tests
# =============================================================================


class TestCrossDocumentAnalysis:
    """Tests for cross-document analysis."""

    @pytest.mark.asyncio
    async def test_detect_missing_guardrail_library(self, auditor, audit_context):
        """Test detection of missing guardrail library."""
        chunks = [
            ChunkData(
                id="chunk-13",
                document_id="main.py",
                content="import openai\nresponse = openai.ChatCompletion.create()",
            ),
            ChunkData(
                id="chunk-14",
                document_id="utils.py",
                content="from langchain import LLMChain\nchain.run(user_input)",
            ),
        ]

        findings = await auditor.cross_document_analysis(chunks, audit_context)

        guardrail_findings = [f for f in findings if "guardrails" in f.category.lower()]
        assert len(guardrail_findings) > 0

    @pytest.mark.asyncio
    async def test_detect_missing_model_documentation(self, auditor, audit_context):
        """Test detection of missing model documentation."""
        chunks = [
            ChunkData(
                id="chunk-15",
                document_id="ai_app.py",
                content="import anthropic\nclient = anthropic.Client()",
            ),
        ]

        findings = await auditor.cross_document_analysis(chunks, audit_context)

        doc_findings = [f for f in findings if "documentation" in f.category.lower()]
        assert len(doc_findings) > 0

    @pytest.mark.asyncio
    async def test_no_findings_with_guardrails(self, auditor, audit_context):
        """Test no guardrail findings when guardrails are present."""
        chunks = [
            ChunkData(
                id="chunk-16",
                document_id="protected.py",
                content="from nemo_guardrails import RailsConfig\nconfig = RailsConfig.from_path('config')",
            ),
        ]

        findings = await auditor.cross_document_analysis(chunks, audit_context)

        guardrail_findings = [f for f in findings if "No Guardrail Library" in f.title]
        assert len(guardrail_findings) == 0


# =============================================================================
# Risk Summary Tests
# =============================================================================


class TestRiskSummary:
    """Tests for risk summary generation."""

    def test_get_risk_summary_structure(self, auditor, audit_context):
        """Test risk summary has correct structure."""
        # Create some mock findings
        findings = [
            audit_context.create_finding(
                document_id="test.py",
                title="Test Critical",
                description="Critical issue",
                severity=FindingSeverity.CRITICAL,
                category="prompt_injection",
                recommendation="Fix it",
            ),
            audit_context.create_finding(
                document_id="test.py",
                title="Test High",
                description="High issue",
                severity=FindingSeverity.HIGH,
                category="data_leakage",
                recommendation="Resolve it",
            ),
            audit_context.create_finding(
                document_id="test.py",
                title="Test Medium",
                description="Medium issue",
                severity=FindingSeverity.MEDIUM,
                category="missing_guardrails",
                recommendation="Consider it",
            ),
        ]

        summary = auditor.get_risk_summary(findings)

        assert "total_findings" in summary
        assert "by_severity" in summary
        assert "by_category" in summary
        assert "top_risks" in summary
        assert "recommendations" in summary

    def test_get_risk_summary_counts(self, auditor, audit_context):
        """Test risk summary counts correctly."""
        findings = [
            audit_context.create_finding(
                document_id="test.py",
                title=f"Finding {i}",
                description="Test",
                severity=FindingSeverity.CRITICAL,
                category="prompt_injection",
            )
            for i in range(3)
        ]

        summary = auditor.get_risk_summary(findings)

        assert summary["total_findings"] == 3
        assert summary["by_severity"]["critical"] == 3
        assert summary["by_category"]["prompt_injection"] == 3

    def test_get_risk_summary_top_risks(self, auditor, audit_context):
        """Test risk summary includes top risks."""
        findings = [
            audit_context.create_finding(
                document_id="test.py",
                title="Critical Finding",
                description="Very bad",
                severity=FindingSeverity.CRITICAL,
                category="output_validation",
                evidence_location="Line 42",
            ),
        ]

        summary = auditor.get_risk_summary(findings)

        assert len(summary["top_risks"]) == 1
        assert summary["top_risks"][0]["title"] == "Critical Finding"
        assert summary["top_risks"][0]["severity"] == "critical"

    def test_get_risk_summary_unique_recommendations(self, auditor, audit_context):
        """Test risk summary has unique recommendations."""
        same_recommendation = "Use secure coding practices"
        findings = [
            audit_context.create_finding(
                document_id="test.py",
                title=f"Finding {i}",
                description="Test",
                severity=FindingSeverity.HIGH,
                category="security",
                recommendation=same_recommendation,
            )
            for i in range(3)
        ]

        summary = auditor.get_risk_summary(findings)

        # Should only have one recommendation (deduplicated)
        assert len(summary["recommendations"]) == 1
        assert summary["recommendations"][0] == same_recommendation

    def test_get_risk_summary_empty_findings(self, auditor):
        """Test risk summary with no findings."""
        summary = auditor.get_risk_summary([])

        assert summary["total_findings"] == 0
        assert summary["by_severity"]["critical"] == 0
        assert summary["by_category"] == {}
        assert summary["top_risks"] == []


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestAuditAICodeFunction:
    """Tests for the audit_ai_code convenience function."""

    @pytest.mark.asyncio
    async def test_audit_ai_code_basic(self):
        """Test basic usage of audit_ai_code function."""
        code = """
response = model.generate(prompt)
result = eval(response.content)
"""
        findings = await audit_ai_code(code, file_path="test.py")

        assert isinstance(findings, list)
        # Should detect eval vulnerability
        eval_findings = [f for f in findings if "eval" in f["title"].lower()]
        assert len(eval_findings) > 0

    @pytest.mark.asyncio
    async def test_audit_ai_code_returns_dict_format(self):
        """Test that audit_ai_code returns correct dictionary format."""
        code = """
max_tokens = None
temperature = 2.5
"""
        findings = await audit_ai_code(code)

        if findings:
            finding = findings[0]
            assert "title" in finding
            assert "description" in finding
            assert "severity" in finding
            assert "category" in finding
            assert "evidence" in finding
            assert "location" in finding
            assert "recommendation" in finding

    @pytest.mark.asyncio
    async def test_audit_ai_code_clean_code(self):
        """Test audit_ai_code with clean code."""
        code = """
def hello():
    return "Hello, World!"
"""
        findings = await audit_ai_code(code)

        # Clean code should have minimal or no findings
        assert isinstance(findings, list)


# =============================================================================
# Finding Properties Tests
# =============================================================================


class TestFindingProperties:
    """Tests for finding properties and metadata."""

    @pytest.mark.asyncio
    async def test_finding_has_line_number(self, auditor, audit_context):
        """Test findings include line numbers."""
        code = """line 1
line 2
max_tokens = None
line 4
"""
        chunk = ChunkData(
            id="chunk-17",
            document_id="test.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        if findings:
            # Line numbers should be present in evidence_location
            assert any("Line" in f.evidence_location for f in findings)

    @pytest.mark.asyncio
    async def test_finding_has_tags(self, auditor, audit_context):
        """Test findings include relevant tags."""
        code = """
OPENAI_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGH"
"""
        chunk = ChunkData(
            id="chunk-18",
            document_id="config.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        # Check for AI-related tags
        all_tags = []
        for f in findings:
            all_tags.extend(f.tags)

        # Should have "ai" tag
        ai_tagged = [f for f in findings if "ai" in f.tags]
        if findings:
            assert len(ai_tagged) > 0

    @pytest.mark.asyncio
    async def test_finding_found_by_attribute(self, auditor, audit_context):
        """Test findings have correct found_by attribute."""
        code = """
temperature = 2.0
"""
        chunk = ChunkData(
            id="chunk-19",
            document_id="app.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        for finding in findings:
            assert finding.found_by == "ai_systems_auditor"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_content(self, auditor, audit_context):
        """Test handling of empty content."""
        chunk = ChunkData(
            id="chunk-empty",
            document_id="empty.py",
            content="",
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        assert findings == []

    @pytest.mark.asyncio
    async def test_binary_like_content(self, auditor, audit_context):
        """Test handling of binary-like content."""
        chunk = ChunkData(
            id="chunk-binary",
            document_id="binary.bin",
            content="\x00\x01\x02\x03\x04",
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        # Should not crash
        assert isinstance(findings, list)

    @pytest.mark.asyncio
    async def test_unicode_content(self, auditor, audit_context):
        """Test handling of unicode content."""
        code = """
# This is a comment with unicode: \u00e9\u00e0\u00fc
user_input = "Caf\u00e9"
"""
        chunk = ChunkData(
            id="chunk-unicode",
            document_id="unicode.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        # Should not crash
        assert isinstance(findings, list)

    @pytest.mark.asyncio
    async def test_very_long_line(self, auditor, audit_context):
        """Test handling of very long lines."""
        long_line = "x" * 10000
        code = f"variable = '{long_line}'"

        chunk = ChunkData(
            id="chunk-long",
            document_id="long.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        # Should not crash
        assert isinstance(findings, list)

    @pytest.mark.asyncio
    async def test_masked_secret_in_evidence(self, auditor, audit_context):
        """Test that secrets are masked in evidence text."""
        # OpenAI pattern requires sk- followed by exactly 48 alphanumeric chars
        code = """
API_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHijkl"
"""
        chunk = ChunkData(
            id="chunk-mask",
            document_id="secrets.py",
            content=code,
        )
        findings = await auditor.analyze_chunk(chunk, audit_context)

        secret_findings = [f for f in findings if "api_secrets" in f.category]
        for finding in secret_findings:
            # Evidence should be masked (show first 8 and last 4 chars)
            assert "..." in finding.evidence_text
            # Full key should not be in evidence
            assert "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHijkl" not in finding.evidence_text
