"""
Integration Tests for Enterprise Multi-Agent Control Plane.

Tests end-to-end functionality of:
- Enterprise connectors (FHIR, templates)
- Workflow templates
- API handlers
- Full sync pipeline
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.enterprise.healthcare.fhir import (
    FHIRConnector,
    PHIRedactor,
    FHIRAuditLogger,
    FHIRResourceType,
)
from aragora.workflow.templates import (
    WORKFLOW_TEMPLATES,
    get_template,
    list_templates,
    CONTRACT_REVIEW_TEMPLATE,
    HIPAA_ASSESSMENT_TEMPLATE,
    SECURITY_AUDIT_TEMPLATE,
    FINANCIAL_AUDIT_TEMPLATE,
)
from aragora.server.handlers.connectors import (
    get_scheduler,
    handle_list_connectors,
    handle_create_connector,
    handle_trigger_sync,
    handle_get_scheduler_stats,
    handle_list_workflow_templates,
    handle_get_workflow_template,
    handle_connector_health,
)
from aragora.agents.personas import DEFAULT_PERSONAS


# =============================================================================
# PHI Redactor Tests
# =============================================================================


class TestPHIRedactor:
    """Tests for HIPAA-compliant PHI redaction."""

    def test_redact_ssn(self):
        """Test SSN redaction."""
        redactor = PHIRedactor()
        text = "Patient SSN is 123-45-6789 and phone is 555-123-4567"

        result = redactor.redact_text(text)

        assert "123-45-6789" not in result.redacted_text
        assert "[REDACTED-SSN]" in result.redacted_text
        assert result.redactions_count >= 1

    def test_redact_email(self):
        """Test email redaction."""
        redactor = PHIRedactor()
        text = "Contact at patient@example.com for follow-up"

        result = redactor.redact_text(text)

        assert "patient@example.com" not in result.redacted_text
        assert "[REDACTED-EMAIL]" in result.redacted_text

    def test_redact_phone(self):
        """Test phone number redaction."""
        redactor = PHIRedactor()
        text = "Call (555) 123-4567 or 555.987.6543"

        result = redactor.redact_text(text)

        assert "555" not in result.redacted_text or "[REDACTED" in result.redacted_text

    def test_redact_fhir_patient_resource(self):
        """Test PHI redaction from FHIR Patient resource."""
        redactor = PHIRedactor()

        patient = {
            "resourceType": "Patient",
            "id": "12345",
            "name": [
                {
                    "use": "official",
                    "family": "Smith",
                    "given": ["John", "Robert"],
                }
            ],
            "telecom": [
                {"system": "phone", "value": "555-123-4567"},
                {"system": "email", "value": "john@example.com"},
            ],
            "address": [
                {
                    "use": "home",
                    "line": ["123 Main Street"],
                    "city": "Boston",
                    "state": "MA",
                    "postalCode": "02101",
                }
            ],
            "birthDate": "1985-06-15",
        }

        redacted = redactor.redact_fhir_resource(patient, "Patient")

        # Name should be redacted
        assert redacted["name"][0]["family"] == "[REDACTED]"
        assert redacted["name"][0]["given"] == ["[REDACTED]"]

        # Telecom should be redacted
        assert redacted["telecom"][0]["value"] == "[REDACTED]"

        # Address - city redacted, state preserved (Safe Harbor)
        assert redacted["address"][0]["city"] == "[REDACTED]"
        assert redacted["address"][0]["state"] == "MA"

        # Birth date - year preserved
        assert redacted["birthDate"] == "1985"

    def test_redact_preserves_clinical_data(self):
        """Test that clinical data is preserved."""
        redactor = PHIRedactor()

        condition = {
            "resourceType": "Condition",
            "id": "cond-123",
            "code": {
                "text": "Type 2 Diabetes Mellitus",
                "coding": [{"code": "E11", "display": "Type 2 diabetes"}],
            },
            "clinicalStatus": {
                "coding": [{"code": "active"}],
            },
        }

        redacted = redactor.redact_fhir_resource(condition, "Condition")

        # Clinical data should be preserved
        assert redacted["code"]["text"] == "Type 2 Diabetes Mellitus"
        assert redacted["clinicalStatus"]["coding"][0]["code"] == "active"


# =============================================================================
# FHIR Audit Logger Tests
# =============================================================================


class TestFHIRAuditLogger:
    """Tests for HIPAA audit logging."""

    def test_log_read(self):
        """Test logging read operations."""
        logger = FHIRAuditLogger(
            organization_id="org-123",
            user_id="user-456",
            user_role="nurse",
        )

        event = logger.log_read(
            resource_type="Patient",
            resource_id="patient-789",
            reason="clinical_care",
        )

        assert event.action == "R"
        assert event.resource_type == "Patient"
        assert event.resource_id == "patient-789"
        assert event.user_id == "user-456"
        assert event.outcome == "0"

    def test_log_search(self):
        """Test logging search operations."""
        logger = FHIRAuditLogger(organization_id="org-123")

        event = logger.log_search(
            resource_type="Observation",
            query_params={"patient": "123", "_count": "10"},
            results_count=5,
        )

        assert event.action == "R"
        assert event.resource_type == "Observation"
        assert "5" in event.resource_id

    def test_get_events_filtered(self):
        """Test retrieving filtered audit events."""
        logger = FHIRAuditLogger(organization_id="org-123")

        # Log some events
        logger.log_read("Patient", "p1")
        logger.log_read("Observation", "o1")
        logger.log_read("Patient", "p2")

        # Filter by resource type
        patient_events = logger.get_events(resource_type="Patient")

        assert len(patient_events) == 2


# =============================================================================
# Workflow Template Tests
# =============================================================================


class TestWorkflowTemplates:
    """Tests for industry workflow templates."""

    def test_all_templates_have_required_fields(self):
        """Test all templates have required fields."""
        required_fields = ["name", "description", "category", "steps", "transitions"]

        for template_id, template in WORKFLOW_TEMPLATES.items():
            for field in required_fields:
                assert field in template, f"Template {template_id} missing {field}"

    def test_legal_templates_use_legal_personas(self):
        """Test legal templates reference legal personas."""
        legal_templates = [t for tid, t in WORKFLOW_TEMPLATES.items() if tid.startswith("legal/")]

        for template in legal_templates:
            debate_steps = [s for s in template["steps"] if s.get("type") == "debate"]

            for step in debate_steps:
                agents = step.get("config", {}).get("agents", [])
                # Should include at least one legal persona
                legal_personas = {
                    "contract_analyst",
                    "compliance_officer",
                    "litigation_support",
                    "m_and_a_counsel",
                }
                has_legal = any(a in legal_personas for a in agents)
                assert has_legal, f"Legal template step {step['id']} should use legal personas"

    def test_healthcare_templates_use_healthcare_personas(self):
        """Test healthcare templates reference healthcare personas."""
        healthcare_templates = [
            t for tid, t in WORKFLOW_TEMPLATES.items() if tid.startswith("healthcare/")
        ]

        for template in healthcare_templates:
            debate_steps = [s for s in template["steps"] if s.get("type") == "debate"]

            for step in debate_steps:
                agents = step.get("config", {}).get("agents", [])
                healthcare_personas = {
                    "hipaa_auditor",
                    "clinical_reviewer",
                    "research_analyst_clinical",
                    "medical_coder",
                }
                has_healthcare = any(a in healthcare_personas for a in agents)
                # Allow compliance personas too
                compliance_personas = {"compliance_officer", "sox", "security_engineer"}
                has_compliance = any(a in compliance_personas for a in agents)
                assert has_healthcare or has_compliance, (
                    f"Healthcare template step {step['id']} should use healthcare/compliance personas"
                )

    def test_code_templates_use_code_personas(self):
        """Test code templates reference code review personas."""
        code_templates = [t for tid, t in WORKFLOW_TEMPLATES.items() if tid.startswith("code/")]

        for template in code_templates:
            debate_steps = [s for s in template["steps"] if s.get("type") == "debate"]

            for step in debate_steps:
                agents = step.get("config", {}).get("agents", [])
                code_personas = {
                    "code_security_specialist",
                    "architecture_reviewer",
                    "code_quality_reviewer",
                    "api_design_reviewer",
                    "security_engineer",
                    "performance_engineer",
                    "data_architect",
                    "devops_engineer",
                }
                has_code = any(a in code_personas for a in agents)
                assert has_code, f"Code template step {step['id']} should use code personas"

    def test_accounting_templates_use_financial_personas(self):
        """Test accounting templates reference financial personas."""
        accounting_templates = [
            t for tid, t in WORKFLOW_TEMPLATES.items() if tid.startswith("accounting/")
        ]

        for template in accounting_templates:
            debate_steps = [s for s in template["steps"] if s.get("type") == "debate"]

            for step in debate_steps:
                agents = step.get("config", {}).get("agents", [])
                financial_personas = {
                    "financial_auditor",
                    "tax_specialist",
                    "forensic_accountant",
                    "internal_auditor",
                    "sox",
                    "compliance_officer",
                }
                has_financial = any(a in financial_personas for a in agents)
                assert has_financial, (
                    f"Accounting template step {step['id']} should use financial personas"
                )

    def test_templates_have_human_checkpoints(self):
        """Test templates include human review steps."""
        for template_id, template in WORKFLOW_TEMPLATES.items():
            human_steps = [s for s in template["steps"] if s.get("type") == "human_checkpoint"]
            assert len(human_steps) >= 1, f"Template {template_id} should have human checkpoint"

    def test_most_templates_have_memory_operations(self):
        """Test most templates include knowledge mound integration."""
        templates_with_memory = 0
        for template_id, template in WORKFLOW_TEMPLATES.items():
            memory_steps = [
                s for s in template["steps"] if s.get("type") in ("memory_read", "memory_write")
            ]
            if len(memory_steps) >= 1:
                templates_with_memory += 1

        # At least 80% of templates should have memory operations
        assert templates_with_memory >= len(WORKFLOW_TEMPLATES) * 0.8

    def test_get_template(self):
        """Test get_template function."""
        template = get_template("legal/contract-review")
        assert template is not None
        assert template["name"] == "Contract Review"

        missing = get_template("nonexistent/template")
        assert missing is None

    def test_list_templates_by_category(self):
        """Test list_templates with category filter."""
        legal = list_templates(category="legal")
        assert len(legal) == 3
        assert all(t["category"] == "legal" for t in legal)

        healthcare = list_templates(category="healthcare")
        assert len(healthcare) == 3

        code = list_templates(category="code")
        assert len(code) == 3


# =============================================================================
# API Handler Tests
# =============================================================================


class TestConnectorAPIHandlers:
    """Tests for connector API handlers."""

    @pytest.mark.asyncio
    async def test_list_connectors_empty(self):
        """Test listing connectors when none registered."""
        result = await handle_list_connectors(tenant_id="test-tenant")

        assert "connectors" in result
        assert "total" in result

    @pytest.mark.asyncio
    async def test_create_connector_github(self):
        """Test creating a GitHub connector."""
        result = await handle_create_connector(
            connector_type="github",
            config={
                "owner": "test-org",
                "repo": "test-repo",
            },
            tenant_id="test-tenant",
        )

        assert result["type"] == "github"
        assert result["status"] == "registered"

    @pytest.mark.asyncio
    async def test_create_connector_s3(self):
        """Test creating an S3 connector."""
        result = await handle_create_connector(
            connector_type="s3",
            config={
                "bucket": "test-bucket",
                "prefix": "documents/",
            },
            tenant_id="test-tenant",
        )

        assert result["type"] == "s3"
        assert result["status"] == "registered"

    @pytest.mark.asyncio
    async def test_create_connector_fhir(self):
        """Test creating a FHIR connector."""
        result = await handle_create_connector(
            connector_type="fhir",
            config={
                "base_url": "https://fhir.example.com/r4",
                "organization_id": "org-123",
                "enable_phi_redaction": True,
            },
            tenant_id="test-tenant",
        )

        assert result["type"] == "fhir"
        assert result["status"] == "registered"

    @pytest.mark.asyncio
    async def test_scheduler_stats(self):
        """Test getting scheduler statistics."""
        stats = await handle_get_scheduler_stats()

        assert "total_jobs" in stats
        assert "running_syncs" in stats
        assert "success_rate" in stats

    @pytest.mark.asyncio
    async def test_connector_health(self):
        """Test connector health check."""
        health = await handle_connector_health()

        assert health["status"] == "healthy"
        assert "total_connectors" in health


class TestWorkflowTemplateAPIHandlers:
    """Tests for workflow template API handlers."""

    @pytest.mark.asyncio
    async def test_list_templates(self):
        """Test listing all templates."""
        result = await handle_list_workflow_templates()

        assert "templates" in result
        assert result["total"] == len(WORKFLOW_TEMPLATES)  # Dynamic count
        assert "categories" in result

    @pytest.mark.asyncio
    async def test_list_templates_by_category(self):
        """Test listing templates by category."""
        result = await handle_list_workflow_templates(category="healthcare")

        assert result["total"] == 3
        assert all(t["category"] == "healthcare" for t in result["templates"])

    @pytest.mark.asyncio
    async def test_get_template(self):
        """Test getting a specific template."""
        result = await handle_get_workflow_template("code/security-audit")

        assert result is not None
        assert result["id"] == "code/security-audit"
        assert "template" in result

    @pytest.mark.asyncio
    async def test_get_template_not_found(self):
        """Test getting a non-existent template."""
        result = await handle_get_workflow_template("nonexistent/template")

        assert result is None


# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_persona_coverage_for_templates(self):
        """Test that all personas used in templates exist in DEFAULT_PERSONAS or are known role-based agents."""
        # Known model names that are valid agents
        KNOWN_MODELS = {
            "claude",
            "codex",
            "gemini",
            "grok",
            "deepseek",
            "qwen",
            "yi",
            "llama",
            "gpt4",
            "gpt-4",
        }
        # Enterprise role-based agents used in workflow templates
        ENTERPRISE_ROLES = {
            "security_engineer",
            "compliance_officer",
            "legal_analyst",
            "auditor",
            "data_scientist",
            "ml_engineer",
            "data_engineer",
            "devops_engineer",
            "sre",
            "architect",
            "tech_lead",
            "product_manager",
            "project_manager",
            "business_analyst",
            "ux_researcher",
            "designer",
            "technical_writer",
            "qa_engineer",
            "data_analyst",
            "statistician",
            "ai_ethics_specialist",
            "data_governance_specialist",
            "product_marketing",
            "customer_success",
            "financial_auditor",
            "tax_specialist",
            "forensic_accountant",
            "internal_auditor",
            "sox",
            "hipaa_specialist",
            "clinical_analyst",
            "medical_informaticist",
            "healthcare_compliance",
            "phi_specialist",
            "prompt_engineer",
        }

        missing_personas = set()
        for template_id, template in WORKFLOW_TEMPLATES.items():
            for step in template["steps"]:
                if step.get("type") == "debate":
                    agents = step.get("config", {}).get("agents", [])
                    for agent in agents:
                        if (
                            agent not in DEFAULT_PERSONAS
                            and agent not in KNOWN_MODELS
                            and agent not in ENTERPRISE_ROLES
                        ):
                            missing_personas.add(agent)

        assert len(missing_personas) == 0, f"Missing personas: {missing_personas}"

    def test_template_transition_validity(self):
        """Test that all template transitions reference valid steps."""
        for template_id, template in WORKFLOW_TEMPLATES.items():
            step_ids = {s["id"] for s in template["steps"]}

            # Also include branch step IDs for parallel steps
            for step in template["steps"]:
                if step.get("type") == "parallel":
                    for branch in step.get("branches", []):
                        step_ids.add(branch["id"])
                        for substep in branch.get("steps", []):
                            step_ids.add(substep["id"])

            for transition in template["transitions"]:
                from_step = transition["from"]
                to_step = transition["to"]

                assert from_step in step_ids, (
                    f"Template {template_id}: transition from unknown step {from_step}"
                )
                assert to_step in step_ids, (
                    f"Template {template_id}: transition to unknown step {to_step}"
                )

    def test_fhir_connector_with_redaction(self):
        """Test FHIR connector with PHI redaction enabled."""
        connector = FHIRConnector(
            base_url="https://fhir.example.com",
            organization_id="org-123",
            enable_phi_redaction=True,
        )

        assert connector.enable_phi_redaction is True
        assert connector._redactor is not None
        assert connector._audit_logger is not None

    @pytest.mark.asyncio
    async def test_full_sync_pipeline_mock(self):
        """Test full sync pipeline with mocked FHIR server."""
        connector = FHIRConnector(
            base_url="https://fhir.example.com",
            organization_id="org-123",
            resource_types=[FHIRResourceType.PATIENT],
        )

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "test-patient-1",
                        "name": [{"family": "Test", "given": ["User"]}],
                        "meta": {"lastUpdated": "2024-01-01T00:00:00Z"},
                    }
                }
            ],
            "link": [],
        }

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            from aragora.connectors.enterprise.base import SyncState

            state = SyncState(connector_id=connector.connector_id)
            items = []

            async for item in connector.sync_items(state, batch_size=10):
                items.append(item)

            # Should have synced one patient (with PHI redacted)
            assert len(items) == 1
            assert items[0].metadata["phi_redacted"] is True

            # Verify audit logging occurred
            audit_events = connector.get_audit_events()
            assert len(audit_events) >= 1
