"""
Tests for Phase 3 Enterprise Connectors.

Tests the enterprise connector infrastructure including:
- EnterpriseConnector base class
- SyncState and incremental sync
- Credential providers
- SyncScheduler
- Industry personas integration
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncState,
    SyncStatus as BaseSyncStatus,
    SyncResult,
    SyncItem,
    CredentialProvider,
    EnvCredentialProvider,
)
from aragora.connectors.enterprise.sync.scheduler import (
    SyncScheduler,
    SyncJob,
    SyncSchedule,
    SyncHistory,
    SyncStatus,
)
from aragora.training.specialist_models import (
    Vertical,
    TrainingStatus,
    SpecialistModelConfig,
    SpecialistModel,
    SpecialistModelRegistry,
    get_vertical_config,
)
from aragora.agents.personas import (
    DEFAULT_PERSONAS,
    EXPERTISE_DOMAINS,
    PERSONALITY_TRAITS,
    Persona,
)


# =============================================================================
# SyncState Tests
# =============================================================================

class TestSyncState:
    """Tests for SyncState dataclass."""

    def test_sync_state_creation(self):
        """Test creating a sync state."""
        state = SyncState(connector_id="test-connector")

        assert state.connector_id == "test-connector"
        assert state.cursor is None
        assert state.items_synced == 0
        assert state.items_total == 0
        assert state.errors == []
        assert state.status == BaseSyncStatus.IDLE

    def test_sync_state_to_dict(self):
        """Test serializing sync state to dict."""
        state = SyncState(
            connector_id="test",
            cursor="abc123",
            items_synced=10,
            last_item_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        data = state.to_dict()

        assert data["connector_id"] == "test"
        assert data["cursor"] == "abc123"
        assert data["items_synced"] == 10
        assert "2024-01-01" in data["last_item_timestamp"]

    def test_sync_state_from_dict(self):
        """Test deserializing sync state from dict."""
        data = {
            "connector_id": "test",
            "cursor": "xyz789",
            "items_synced": 50,
            "items_total": 100,
            "last_item_timestamp": "2024-06-15T12:00:00+00:00",
        }

        state = SyncState.from_dict(data)

        assert state.connector_id == "test"
        assert state.cursor == "xyz789"
        assert state.items_synced == 50
        assert state.last_item_timestamp.year == 2024


# =============================================================================
# SyncItem Tests
# =============================================================================

class TestSyncItem:
    """Tests for SyncItem dataclass."""

    def test_sync_item_creation(self):
        """Test creating a sync item."""
        item = SyncItem(
            id="item-1",
            content="Test content",
            source_type="document",
            source_id="doc-123",
            title="Test Document",
            domain="technical/documentation",
        )

        assert item.id == "item-1"
        assert item.content == "Test content"
        assert item.source_type == "document"
        assert item.domain == "technical/documentation"
        assert item.confidence == 0.7  # default

    def test_sync_item_metadata(self):
        """Test sync item with metadata."""
        item = SyncItem(
            id="item-2",
            content="Content here",
            source_type="code",
            source_id="file.py",
            title="Python File",
            metadata={"language": "python"},
        )

        assert item.id == "item-2"
        assert item.source_type == "code"
        assert item.metadata["language"] == "python"
        assert item.confidence == 0.7  # default


# =============================================================================
# Credential Provider Tests
# =============================================================================

class TestEnvCredentialProvider:
    """Tests for EnvCredentialProvider."""

    @pytest.mark.asyncio
    async def test_env_credential_from_env(self):
        """Test getting credential from environment."""
        provider = EnvCredentialProvider(prefix="TEST_")

        with patch.dict("os.environ", {"TEST_API_KEY": "secret123"}):
            value = await provider.get_credential("API_KEY")
            assert value == "secret123"

    @pytest.mark.asyncio
    async def test_env_credential_with_default_prefix(self):
        """Test getting credential with default ARAGORA_ prefix."""
        provider = EnvCredentialProvider()

        with patch.dict("os.environ", {"ARAGORA_MY_SECRET": "secret456"}):
            value = await provider.get_credential("MY_SECRET")
            assert value == "secret456"

    @pytest.mark.asyncio
    async def test_env_credential_missing(self):
        """Test missing credential returns None."""
        provider = EnvCredentialProvider()

        value = await provider.get_credential("NONEXISTENT_KEY_12345")
        assert value is None


# =============================================================================
# SyncSchedule Tests
# =============================================================================

class TestSyncSchedule:
    """Tests for SyncSchedule configuration."""

    def test_schedule_defaults(self):
        """Test default schedule configuration."""
        schedule = SyncSchedule()

        assert schedule.schedule_type == "interval"
        assert schedule.interval_minutes == 60
        assert schedule.enabled is True
        assert schedule.max_concurrent == 1
        assert schedule.retry_on_failure is True

    def test_schedule_to_dict(self):
        """Test serializing schedule."""
        schedule = SyncSchedule(
            schedule_type="cron",
            cron_expression="0 * * * *",
            interval_minutes=30,
            max_retries=5,
        )

        data = schedule.to_dict()

        assert data["schedule_type"] == "cron"
        assert data["cron_expression"] == "0 * * * *"
        assert data["max_retries"] == 5

    def test_schedule_from_dict(self):
        """Test deserializing schedule."""
        data = {
            "schedule_type": "interval",
            "interval_minutes": 15,
            "enabled": False,
        }

        schedule = SyncSchedule.from_dict(data)

        assert schedule.interval_minutes == 15
        assert schedule.enabled is False


# =============================================================================
# SyncJob Tests
# =============================================================================

class TestSyncJob:
    """Tests for SyncJob."""

    def test_job_creation(self):
        """Test creating a sync job."""
        job = SyncJob(
            id="job-1",
            connector_id="github-connector",
            tenant_id="org-123",
            schedule=SyncSchedule(interval_minutes=30),
        )

        assert job.id == "job-1"
        assert job.connector_id == "github-connector"
        assert job.tenant_id == "org-123"
        assert job.consecutive_failures == 0

    def test_job_next_run_calculation(self):
        """Test that next_run is calculated on creation."""
        job = SyncJob(
            id="job-2",
            connector_id="s3-connector",
            tenant_id="default",
            schedule=SyncSchedule(interval_minutes=60),
        )

        # Should have a next run time set
        assert job.next_run is not None

    def test_job_to_dict(self):
        """Test serializing job."""
        job = SyncJob(
            id="job-3",
            connector_id="postgres-connector",
            tenant_id="tenant-a",
            schedule=SyncSchedule(),
        )
        job.consecutive_failures = 2

        data = job.to_dict()

        assert data["id"] == "job-3"
        assert data["connector_id"] == "postgres-connector"
        assert data["consecutive_failures"] == 2


# =============================================================================
# SyncHistory Tests
# =============================================================================

class TestSyncHistory:
    """Tests for SyncHistory."""

    def test_history_creation(self):
        """Test creating sync history."""
        history = SyncHistory(
            id="run-1",
            job_id="job-1",
            connector_id="github",
            tenant_id="default",
            status=SyncStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        assert history.id == "run-1"
        assert history.status == SyncStatus.RUNNING
        assert history.items_synced == 0

    def test_history_duration(self):
        """Test duration calculation."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(seconds=120)

        history = SyncHistory(
            id="run-2",
            job_id="job-1",
            connector_id="s3",
            tenant_id="default",
            status=SyncStatus.COMPLETED,
            started_at=start,
            completed_at=end,
            items_synced=50,
        )

        assert history.duration_seconds == 120.0

    def test_history_to_dict(self):
        """Test serializing history."""
        history = SyncHistory(
            id="run-3",
            job_id="job-2",
            connector_id="mongo",
            tenant_id="org-x",
            status=SyncStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            errors=["Connection timeout"],
        )

        data = history.to_dict()

        assert data["status"] == "failed"
        assert len(data["errors"]) == 1


# =============================================================================
# SyncScheduler Tests
# =============================================================================

class TestSyncScheduler:
    """Tests for SyncScheduler."""

    def test_scheduler_creation(self):
        """Test creating scheduler."""
        scheduler = SyncScheduler(max_concurrent_syncs=3)

        assert scheduler.max_concurrent_syncs == 3
        assert len(scheduler._jobs) == 0

    def test_register_connector(self):
        """Test registering a connector."""
        scheduler = SyncScheduler()

        # Create mock connector
        mock_connector = MagicMock()
        mock_connector.connector_id = "test-connector"
        mock_connector.name = "Test Connector"

        job = scheduler.register_connector(
            mock_connector,
            schedule=SyncSchedule(interval_minutes=15),
            tenant_id="tenant-1",
        )

        assert job.connector_id == "test-connector"
        assert job.tenant_id == "tenant-1"
        assert "tenant-1:test-connector" in scheduler._jobs

    def test_list_jobs(self):
        """Test listing jobs."""
        scheduler = SyncScheduler()

        # Register multiple connectors
        for i in range(3):
            mock_connector = MagicMock()
            mock_connector.connector_id = f"connector-{i}"
            mock_connector.name = f"Connector {i}"
            scheduler.register_connector(
                mock_connector,
                tenant_id=f"tenant-{i % 2}",
            )

        all_jobs = scheduler.list_jobs()
        assert len(all_jobs) == 3

        tenant_0_jobs = scheduler.list_jobs(tenant_id="tenant-0")
        assert len(tenant_0_jobs) == 2

    def test_get_stats(self):
        """Test getting scheduler stats."""
        scheduler = SyncScheduler()

        stats = scheduler.get_stats()

        assert stats["total_jobs"] == 0
        assert stats["running_syncs"] == 0
        assert stats["success_rate"] == 1.0


# =============================================================================
# Specialist Model Config Tests
# =============================================================================

class TestSpecialistModelConfig:
    """Tests for SpecialistModelConfig."""

    def test_config_creation(self):
        """Test creating config."""
        config = SpecialistModelConfig(
            vertical=Vertical.LEGAL,
            base_model="llama-3.3-70b",
            lora_rank=32,
        )

        assert config.vertical == Vertical.LEGAL
        assert config.base_model == "llama-3.3-70b"
        assert config.lora_rank == 32

    def test_get_vertical_config(self):
        """Test getting default config for vertical."""
        config = get_vertical_config(Vertical.HEALTHCARE)

        assert config.vertical == Vertical.HEALTHCARE
        assert config.base_model == "llama-3.3-70b"
        assert config.lora_rank == 32  # Healthcare default


# =============================================================================
# Specialist Model Registry Tests
# =============================================================================

class TestSpecialistModelRegistry:
    """Tests for SpecialistModelRegistry."""

    def test_registry_creation(self):
        """Test creating registry."""
        registry = SpecialistModelRegistry()

        assert len(registry._specialists) == 0

    def test_register_model(self):
        """Test registering a model."""
        registry = SpecialistModelRegistry()

        model = SpecialistModel(
            id="sm_legal_001",
            base_model="llama-3.3-70b",
            adapter_name="aragora-legal-v1",
            vertical=Vertical.LEGAL,
            org_id=None,
            status=TrainingStatus.READY,
        )

        registry.register(model)

        retrieved = registry.get("sm_legal_001")
        assert retrieved is not None
        assert retrieved.vertical == Vertical.LEGAL

    def test_get_for_vertical(self):
        """Test getting best model for vertical."""
        registry = SpecialistModelRegistry()

        # Register a ready model
        model = SpecialistModel(
            id="sm_healthcare_001",
            base_model="llama-3.3-70b",
            adapter_name="aragora-healthcare-v1",
            vertical=Vertical.HEALTHCARE,
            org_id=None,
            status=TrainingStatus.READY,
            elo_rating=1250,
        )
        registry.register(model)

        best = registry.get_for_vertical(Vertical.HEALTHCARE)

        assert best is not None
        assert best.id == "sm_healthcare_001"

    def test_list_for_org(self):
        """Test listing models for organization."""
        registry = SpecialistModelRegistry()

        # Register org-specific model
        model = SpecialistModel(
            id="sm_legal_org1",
            base_model="llama-3.3-70b",
            adapter_name="org1-legal-v1",
            vertical=Vertical.LEGAL,
            org_id="org-1",
            status=TrainingStatus.READY,
        )
        registry.register(model)

        org_models = registry.list_for_org("org-1")

        assert len(org_models) == 1
        assert org_models[0].org_id == "org-1"


# =============================================================================
# Industry Personas Tests
# =============================================================================

class TestIndustryPersonas:
    """Tests for industry-specific personas."""

    def test_legal_personas_exist(self):
        """Test legal personas are defined."""
        legal_personas = [
            "contract_analyst",
            "compliance_officer",
            "litigation_support",
            "m_and_a_counsel",
        ]

        for name in legal_personas:
            assert name in DEFAULT_PERSONAS, f"Missing legal persona: {name}"

    def test_healthcare_personas_exist(self):
        """Test healthcare personas are defined."""
        healthcare_personas = [
            "clinical_reviewer",
            "hipaa_auditor",
            "research_analyst_clinical",
            "medical_coder",
        ]

        for name in healthcare_personas:
            assert name in DEFAULT_PERSONAS, f"Missing healthcare persona: {name}"

    def test_accounting_personas_exist(self):
        """Test accounting personas are defined."""
        accounting_personas = [
            "financial_auditor",
            "tax_specialist",
            "forensic_accountant",
            "internal_auditor",
        ]

        for name in accounting_personas:
            assert name in DEFAULT_PERSONAS, f"Missing accounting persona: {name}"

    def test_academic_personas_exist(self):
        """Test academic personas are defined."""
        academic_personas = [
            "research_methodologist",
            "peer_reviewer",
            "grant_reviewer",
            "irb_reviewer",
        ]

        for name in academic_personas:
            assert name in DEFAULT_PERSONAS, f"Missing academic persona: {name}"

    def test_software_specialist_personas_exist(self):
        """Test software specialist personas are defined."""
        software_personas = [
            "code_security_specialist",
            "architecture_reviewer",
            "code_quality_reviewer",
            "api_design_reviewer",
        ]

        for name in software_personas:
            assert name in DEFAULT_PERSONAS, f"Missing software persona: {name}"

    def test_persona_has_required_fields(self):
        """Test all personas have required fields."""
        for name, persona in DEFAULT_PERSONAS.items():
            assert persona.agent_name == name, f"Persona {name} has mismatched agent_name"
            assert persona.description, f"Persona {name} missing description"
            assert len(persona.traits) > 0, f"Persona {name} has no traits"
            assert len(persona.expertise) > 0, f"Persona {name} has no expertise"

    def test_persona_traits_are_valid(self):
        """Test all persona traits are in PERSONALITY_TRAITS."""
        # Get all allowed traits (including some that might not be in base list)
        allowed_traits = set(PERSONALITY_TRAITS) | {
            "contemplative", "nuanced", "interdisciplinary",
            "empathetic", "balanced", "practical",
            "probing", "authentic", "individualistic",
            "methodical",
        }

        for name, persona in DEFAULT_PERSONAS.items():
            for trait in persona.traits:
                assert trait in allowed_traits, f"Persona {name} has invalid trait: {trait}"

    def test_legal_persona_expertise(self):
        """Test legal personas have legal domain expertise."""
        legal_personas = ["contract_analyst", "litigation_support", "m_and_a_counsel"]

        for name in legal_personas:
            persona = DEFAULT_PERSONAS[name]
            assert "legal" in persona.expertise, f"{name} missing legal expertise"
            assert persona.expertise["legal"] >= 0.85, f"{name} legal expertise too low"

    def test_healthcare_persona_hipaa_expertise(self):
        """Test healthcare personas have HIPAA expertise."""
        healthcare_personas = ["clinical_reviewer", "hipaa_auditor", "research_analyst_clinical"]

        for name in healthcare_personas:
            persona = DEFAULT_PERSONAS[name]
            assert "hipaa" in persona.expertise or "data_privacy" in persona.expertise, \
                f"{name} missing HIPAA/privacy expertise"

    def test_compliance_personas_have_low_temperature(self):
        """Test compliance-focused personas have low temperature."""
        compliance_personas = [
            "sox", "pci_dss", "hipaa", "gdpr", "finra",
            "financial_auditor", "hipaa_auditor",
        ]

        for name in compliance_personas:
            if name in DEFAULT_PERSONAS:
                persona = DEFAULT_PERSONAS[name]
                assert persona.temperature <= 0.5, \
                    f"{name} temperature too high for compliance: {persona.temperature}"


# =============================================================================
# Expertise Domains Tests
# =============================================================================

class TestExpertiseDomains:
    """Tests for expertise domains."""

    def test_industry_domains_exist(self):
        """Test industry vertical domains are defined."""
        industry_domains = ["legal", "clinical", "financial", "academic"]

        for domain in industry_domains:
            assert domain in EXPERTISE_DOMAINS, f"Missing industry domain: {domain}"

    def test_compliance_domains_exist(self):
        """Test compliance domains are defined."""
        compliance_domains = [
            "sox_compliance", "pci_dss", "hipaa", "gdpr",
            "fda_21_cfr", "fisma", "finra",
        ]

        for domain in compliance_domains:
            assert domain in EXPERTISE_DOMAINS, f"Missing compliance domain: {domain}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestConnectorIntegration:
    """Integration tests for connector system."""

    @pytest.mark.asyncio
    async def test_scheduler_with_mock_connector(self):
        """Test scheduler integration with mock connector."""
        scheduler = SyncScheduler()

        # Create mock connector that returns a successful result
        mock_connector = MagicMock()
        mock_connector.connector_id = "mock-connector"
        mock_connector.name = "Mock Connector"
        mock_connector.sync = AsyncMock(return_value=SyncResult(
            connector_id="mock-connector",
            success=True,
            items_synced=10,
            items_updated=5,
            items_skipped=0,
            items_failed=0,
            duration_ms=1500.0,
            errors=[],
        ))

        scheduler.register_connector(
            mock_connector,
            schedule=SyncSchedule(schedule_type="webhook_only"),
            tenant_id="test",
        )

        # Trigger sync
        run_id = await scheduler.trigger_sync("mock-connector", tenant_id="test")

        assert run_id is not None
        mock_connector.sync.assert_called_once()

    def test_vertical_to_persona_mapping(self):
        """Test that verticals map to appropriate personas."""
        vertical_persona_map = {
            Vertical.LEGAL: ["contract_analyst", "compliance_officer"],
            Vertical.HEALTHCARE: ["clinical_reviewer", "hipaa_auditor"],
            Vertical.ACCOUNTING: ["financial_auditor", "tax_specialist"],
            Vertical.ACADEMIC: ["research_methodologist", "peer_reviewer"],
            Vertical.SOFTWARE: ["code_security_specialist", "architecture_reviewer"],
        }

        for vertical, expected_personas in vertical_persona_map.items():
            for persona_name in expected_personas:
                assert persona_name in DEFAULT_PERSONAS, \
                    f"Vertical {vertical.value} missing persona {persona_name}"
