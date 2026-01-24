"""Tests for Knowledge Mound lifecycle management."""

from datetime import datetime, timedelta, timezone

import pytest

from aragora.knowledge.mound.lifecycle import (
    LifecycleManager,
    LifecycleReport,
    LifecycleStage,
    LifecycleTransition,
    RetentionAction,
    RetentionPolicy,
    create_lifecycle_manager,
)


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_creation(self):
        """Test policy creation with defaults."""
        policy = RetentionPolicy(
            name="default-policy",
            description="Default retention policy",
        )

        assert policy.name == "default-policy"
        assert policy.active_period == timedelta(days=30)
        assert policy.enabled is True
        assert policy.id is not None

    def test_matches_all(self):
        """Test that empty filters match everything."""
        policy = RetentionPolicy(name="match-all")

        assert policy.matches()
        assert policy.matches(workspace_id="any")
        assert policy.matches(knowledge_type="any")
        assert policy.matches(tier="any")

    def test_matches_workspace(self):
        """Test workspace matching."""
        policy = RetentionPolicy(
            name="workspace-policy",
            workspace_ids=["ws-1", "ws-2"],
        )

        assert policy.matches(workspace_id="ws-1")
        assert policy.matches(workspace_id="ws-2")
        assert not policy.matches(workspace_id="ws-3")

    def test_matches_knowledge_type(self):
        """Test knowledge type matching."""
        policy = RetentionPolicy(
            name="type-policy",
            knowledge_types=["fact", "consensus"],
        )

        assert policy.matches(knowledge_type="fact")
        assert policy.matches(knowledge_type="consensus")
        assert not policy.matches(knowledge_type="debate")

    def test_matches_tier(self):
        """Test tier matching."""
        policy = RetentionPolicy(
            name="tier-policy",
            tiers=["fast", "medium"],
        )

        assert policy.matches(tier="fast")
        assert policy.matches(tier="medium")
        assert not policy.matches(tier="slow")

    def test_disabled_policy_doesnt_match(self):
        """Test that disabled policies don't match."""
        policy = RetentionPolicy(
            name="disabled",
            enabled=False,
        )

        assert not policy.matches()

    def test_to_dict(self):
        """Test policy serialization."""
        policy = RetentionPolicy(
            name="test-policy",
            active_period=timedelta(days=60),
            max_versions_to_keep=5,
        )

        data = policy.to_dict()

        assert data["name"] == "test-policy"
        assert data["active_period_days"] == 60
        assert data["max_versions_to_keep"] == 5


class TestLifecycleTransition:
    """Tests for LifecycleTransition dataclass."""

    def test_creation(self):
        """Test transition creation."""
        transition = LifecycleTransition(
            knowledge_id="know-123",
            from_stage=LifecycleStage.ACTIVE,
            to_stage=LifecycleStage.COLD,
            reason="Inactivity",
        )

        assert transition.knowledge_id == "know-123"
        assert transition.from_stage == LifecycleStage.ACTIVE
        assert transition.to_stage == LifecycleStage.COLD
        assert transition.id is not None

    def test_to_dict(self):
        """Test transition serialization."""
        transition = LifecycleTransition(
            knowledge_id="know-123",
            from_stage=LifecycleStage.WARM,
            to_stage=LifecycleStage.COLD,
            reason="Age threshold",
            policy_id="policy-1",
        )

        data = transition.to_dict()

        assert data["knowledge_id"] == "know-123"
        assert data["from_stage"] == "warm"
        assert data["to_stage"] == "cold"
        assert data["reason"] == "Age threshold"
        assert data["policy_id"] == "policy-1"


class TestLifecycleReport:
    """Tests for LifecycleReport dataclass."""

    def test_to_dict(self):
        """Test report serialization."""
        report = LifecycleReport(
            total_knowledge_items=1000,
            active_count=800,
            warm_count=150,
            cold_count=50,
            stale_count=25,
        )

        data = report.to_dict()

        assert data["total_knowledge_items"] == 1000
        assert data["stage_distribution"]["active"] == 800
        assert data["stage_distribution"]["warm"] == 150
        assert data["stage_distribution"]["cold"] == 50
        assert data["quality_metrics"]["stale_count"] == 25


class TestLifecycleManager:
    """Tests for LifecycleManager class."""

    def test_init(self):
        """Test manager initialization."""
        manager = LifecycleManager()

        assert len(manager._policies) == 0
        assert len(manager._stages) == 0

    def test_add_policy(self):
        """Test adding a retention policy."""
        manager = LifecycleManager()

        policy = RetentionPolicy(name="test-policy")
        manager.add_policy(policy)

        assert policy.id in manager._policies
        assert manager.get_policy(policy.id) == policy

    def test_remove_policy(self):
        """Test removing a retention policy."""
        manager = LifecycleManager()

        policy = RetentionPolicy(name="test-policy")
        manager.add_policy(policy)
        result = manager.remove_policy(policy.id)

        assert result is True
        assert manager.get_policy(policy.id) is None

    def test_remove_nonexistent_policy(self):
        """Test removing a policy that doesn't exist."""
        manager = LifecycleManager()

        result = manager.remove_policy("nonexistent")

        assert result is False

    def test_list_policies(self):
        """Test listing policies."""
        manager = LifecycleManager()

        policy1 = RetentionPolicy(name="policy-1", priority=1)
        policy2 = RetentionPolicy(name="policy-2", priority=2)
        manager.add_policy(policy1)
        manager.add_policy(policy2)

        policies = manager.list_policies()

        assert len(policies) == 2
        # Higher priority first
        assert policies[0].name == "policy-2"

    def test_record_access(self):
        """Test recording knowledge access."""
        manager = LifecycleManager()

        manager.record_access("know-123")
        manager.record_access("know-123")
        manager.record_access("know-456")

        assert manager._access_log["know-123"]["access_count"] == 2
        assert manager._access_log["know-456"]["access_count"] == 1

    def test_get_stage_default(self):
        """Test default stage is ACTIVE."""
        manager = LifecycleManager()

        stage = manager.get_stage("unknown-id")

        assert stage == LifecycleStage.ACTIVE

    def test_set_stage(self):
        """Test setting lifecycle stage."""
        manager = LifecycleManager()

        transition = manager.set_stage(
            "know-123",
            LifecycleStage.COLD,
            reason="Manual archive",
        )

        assert manager.get_stage("know-123") == LifecycleStage.COLD
        assert transition.from_stage == LifecycleStage.ACTIVE
        assert transition.to_stage == LifecycleStage.COLD
        assert transition.reason == "Manual archive"

    def test_set_stage_transition_history(self):
        """Test that transitions are recorded."""
        manager = LifecycleManager()

        manager.set_stage("know-123", LifecycleStage.WARM, "Age")
        manager.set_stage("know-123", LifecycleStage.COLD, "Inactivity")

        history = manager.get_transition_history(knowledge_id="know-123")

        assert len(history) == 2
        assert history[0].to_stage == LifecycleStage.COLD  # Most recent first
        assert history[1].to_stage == LifecycleStage.WARM

    def test_evaluate_retention_no_policies(self):
        """Test retention evaluation with no policies returns KEEP."""
        manager = LifecycleManager()

        action = manager.evaluate_retention(
            knowledge_id="know-123",
        )

        assert action == RetentionAction.KEEP

    def test_evaluate_retention_max_age_exceeded(self):
        """Test that max age triggers DELETE."""
        manager = LifecycleManager()

        policy = RetentionPolicy(
            name="max-age-policy",
            max_age=timedelta(days=30),
        )
        manager.add_policy(policy)

        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=60)  # 60 days old

        action = manager.evaluate_retention(
            knowledge_id="know-123",
            created_at=old_date,
        )

        assert action == RetentionAction.DELETE

    def test_evaluate_retention_low_confidence(self):
        """Test that low confidence triggers REVALIDATE."""
        manager = LifecycleManager()

        policy = RetentionPolicy(
            name="quality-policy",
            min_confidence_for_keep=0.7,
        )
        manager.add_policy(policy)

        action = manager.evaluate_retention(
            knowledge_id="know-123",
            confidence=0.5,
        )

        assert action == RetentionAction.REVALIDATE

    def test_evaluate_retention_not_accessed(self):
        """Test that inactivity triggers ARCHIVE."""
        manager = LifecycleManager()

        policy = RetentionPolicy(
            name="access-policy",
            last_access_threshold=timedelta(days=7),
            min_access_count_for_keep=2,
        )
        manager.add_policy(policy)

        # Record one old access
        manager._access_log["know-123"] = {
            "first_access": datetime.now(timezone.utc) - timedelta(days=30),
            "last_access": datetime.now(timezone.utc) - timedelta(days=14),
            "access_count": 1,  # Below threshold
        }

        action = manager.evaluate_retention(
            knowledge_id="know-123",
        )

        assert action == RetentionAction.ARCHIVE

    def test_generate_report(self):
        """Test report generation."""
        manager = LifecycleManager()

        # Set up some stages
        manager._stages["know-1"] = LifecycleStage.ACTIVE
        manager._stages["know-2"] = LifecycleStage.ACTIVE
        manager._stages["know-3"] = LifecycleStage.WARM
        manager._stages["know-4"] = LifecycleStage.COLD

        report = manager.generate_report()

        assert report.total_knowledge_items == 4
        assert report.active_count == 2
        assert report.warm_count == 1
        assert report.cold_count == 1

    def test_get_transition_history_filtered(self):
        """Test filtered transition history."""
        manager = LifecycleManager()

        manager.set_stage("know-1", LifecycleStage.COLD, "Archive")
        manager.set_stage("know-2", LifecycleStage.DELETED, "Expired")
        manager.set_stage("know-1", LifecycleStage.DELETED, "Expired")

        # Filter by stage
        deleted_transitions = manager.get_transition_history(stage=LifecycleStage.DELETED)
        assert len(deleted_transitions) == 2

        # Filter by knowledge ID
        know1_transitions = manager.get_transition_history(knowledge_id="know-1")
        assert len(know1_transitions) == 2

    def test_archive_callback(self):
        """Test archive callback is triggered."""
        manager = LifecycleManager()

        archived = []
        manager.add_archive_callback(lambda kid, reason: archived.append((kid, reason)))

        manager.set_stage("know-123", LifecycleStage.COLD, "Test archive")

        assert len(archived) == 1
        assert archived[0][0] == "know-123"
        assert archived[0][1] == "Test archive"

    def test_delete_callback(self):
        """Test delete callback is triggered."""
        manager = LifecycleManager()

        deleted = []
        manager.add_delete_callback(lambda kid, reason: deleted.append((kid, reason)))

        manager.set_stage("know-123", LifecycleStage.DELETED, "Retention expired")

        assert len(deleted) == 1
        assert deleted[0][0] == "know-123"

    @pytest.mark.asyncio
    async def test_run_cleanup_dry_run(self):
        """Test cleanup in dry run mode."""
        manager = LifecycleManager()

        # Set up some stages
        manager._stages["know-1"] = LifecycleStage.ACTIVE
        manager._stages["know-2"] = LifecycleStage.WARM

        report = await manager.run_cleanup(dry_run=True)

        assert report["dry_run"] is True
        assert report["evaluated"] == 2
        # Stages should not change in dry run
        assert manager.get_stage("know-1") == LifecycleStage.ACTIVE

    def test_max_transitions_bounded(self):
        """Test that transition history is bounded."""
        manager = LifecycleManager()
        manager._max_transitions = 10

        for i in range(20):
            manager.set_stage(f"know-{i}", LifecycleStage.WARM, f"Reason {i}")

        assert len(manager._transitions) == 10


class TestCreateLifecycleManager:
    """Tests for factory function."""

    def test_create(self):
        """Test factory creates manager."""
        manager = create_lifecycle_manager()

        assert isinstance(manager, LifecycleManager)
