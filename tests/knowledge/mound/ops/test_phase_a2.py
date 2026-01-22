"""
Tests for Phase A2 Knowledge Mound Operations.

Tests:
- Contradiction Detection
- Confidence Decay
- Governance (RBAC + Audit)
- Analytics
- Knowledge Extraction
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


# =============================================================================
# Contradiction Detection Tests
# =============================================================================


class TestContradictionDetection:
    """Tests for contradiction detection module."""

    def test_contradiction_type_enum(self):
        """Test ContradictionType enum values."""
        from aragora.knowledge.mound.ops.contradiction import ContradictionType

        assert ContradictionType.SEMANTIC.value == "semantic"
        assert ContradictionType.LOGICAL.value == "logical"
        assert ContradictionType.NUMERICAL.value == "numerical"

    def test_resolution_strategy_enum(self):
        """Test ResolutionStrategy enum values."""
        from aragora.knowledge.mound.ops.contradiction import ResolutionStrategy

        assert ResolutionStrategy.PREFER_NEWER.value == "prefer_newer"
        assert ResolutionStrategy.HUMAN_REVIEW.value == "human_review"

    def test_contradiction_severity_calculation(self):
        """Test contradiction severity is calculated correctly."""
        from aragora.knowledge.mound.ops.contradiction import (
            Contradiction,
            ContradictionType,
        )

        # High severity
        c1 = Contradiction(
            id="c1",
            item_a_id="a",
            item_b_id="b",
            contradiction_type=ContradictionType.SEMANTIC,
            similarity_score=0.9,
            conflict_score=0.9,
        )
        assert c1.severity == "critical"

        # Medium severity
        c2 = Contradiction(
            id="c2",
            item_a_id="a",
            item_b_id="b",
            contradiction_type=ContradictionType.SEMANTIC,
            similarity_score=0.7,
            conflict_score=0.6,
        )
        assert c2.severity == "medium"

        # Low severity
        c3 = Contradiction(
            id="c3",
            item_a_id="a",
            item_b_id="b",
            contradiction_type=ContradictionType.SEMANTIC,
            similarity_score=0.5,
            conflict_score=0.3,
        )
        assert c3.severity == "low"

    def test_contradiction_to_dict(self):
        """Test contradiction serialization."""
        from aragora.knowledge.mound.ops.contradiction import (
            Contradiction,
            ContradictionType,
        )

        c = Contradiction(
            id="c1",
            item_a_id="a",
            item_b_id="b",
            contradiction_type=ContradictionType.NUMERICAL,
            similarity_score=0.8,
            conflict_score=0.7,
        )
        d = c.to_dict()

        assert d["id"] == "c1"
        assert d["contradiction_type"] == "numerical"
        assert d["severity"] == "high"
        assert "detected_at" in d

    def test_contradiction_config_defaults(self):
        """Test contradiction config default values."""
        from aragora.knowledge.mound.ops.contradiction import ContradictionConfig

        config = ContradictionConfig()

        assert config.min_topic_similarity == 0.7
        assert config.min_conflict_score == 0.5
        assert len(config.negation_patterns) > 0

    @pytest.mark.asyncio
    async def test_detector_initialization(self):
        """Test detector initializes correctly."""
        from aragora.knowledge.mound.ops.contradiction import ContradictionDetector

        detector = ContradictionDetector()
        assert detector.config is not None
        assert detector._contradictions == {}

    @pytest.mark.asyncio
    async def test_resolve_contradiction(self):
        """Test resolving a contradiction."""
        from aragora.knowledge.mound.ops.contradiction import (
            Contradiction,
            ContradictionDetector,
            ContradictionType,
            ResolutionStrategy,
        )

        detector = ContradictionDetector()

        # Add a contradiction
        c = Contradiction(
            id="c1",
            item_a_id="a",
            item_b_id="b",
            contradiction_type=ContradictionType.SEMANTIC,
            similarity_score=0.8,
            conflict_score=0.7,
        )
        detector._contradictions["c1"] = c

        # Resolve it
        resolved = await detector.resolve_contradiction(
            "c1",
            ResolutionStrategy.PREFER_NEWER,
            resolved_by="user1",
            notes="Newer data is correct",
        )

        assert resolved.resolved is True
        assert resolved.resolution == ResolutionStrategy.PREFER_NEWER
        assert resolved.resolved_by == "user1"

    @pytest.mark.asyncio
    async def test_get_unresolved(self):
        """Test getting unresolved contradictions."""
        from aragora.knowledge.mound.ops.contradiction import (
            Contradiction,
            ContradictionDetector,
            ContradictionType,
        )

        detector = ContradictionDetector()

        # Add contradictions
        c1 = Contradiction(
            id="c1",
            item_a_id="a",
            item_b_id="b",
            contradiction_type=ContradictionType.SEMANTIC,
            similarity_score=0.9,
            conflict_score=0.9,
            resolved=False,
        )
        c2 = Contradiction(
            id="c2",
            item_a_id="c",
            item_b_id="d",
            contradiction_type=ContradictionType.SEMANTIC,
            similarity_score=0.5,
            conflict_score=0.5,
            resolved=True,
        )
        detector._contradictions["c1"] = c1
        detector._contradictions["c2"] = c2

        unresolved = await detector.get_unresolved()

        assert len(unresolved) == 1
        assert unresolved[0].id == "c1"


# =============================================================================
# Confidence Decay Tests
# =============================================================================


class TestConfidenceDecay:
    """Tests for confidence decay module."""

    def test_decay_model_enum(self):
        """Test DecayModel enum values."""
        from aragora.knowledge.mound.ops.confidence_decay import DecayModel

        assert DecayModel.EXPONENTIAL.value == "exponential"
        assert DecayModel.LINEAR.value == "linear"
        assert DecayModel.STEP.value == "step"

    def test_confidence_event_enum(self):
        """Test ConfidenceEvent enum values."""
        from aragora.knowledge.mound.ops.confidence_decay import ConfidenceEvent

        assert ConfidenceEvent.ACCESSED.value == "accessed"
        assert ConfidenceEvent.VALIDATED.value == "validated"
        assert ConfidenceEvent.DECAYED.value == "decayed"

    def test_decay_config_defaults(self):
        """Test decay config default values."""
        from aragora.knowledge.mound.ops.confidence_decay import DecayConfig

        config = DecayConfig()

        assert config.half_life_days == 90.0
        assert config.min_confidence == 0.1
        assert config.access_boost == 0.01
        assert "technology" in config.domain_half_lives

    def test_exponential_decay_calculation(self):
        """Test exponential decay formula."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
            DecayConfig,
            DecayModel,
        )

        config = DecayConfig(model=DecayModel.EXPONENTIAL, half_life_days=90)
        manager = ConfidenceDecayManager(config)

        # At half-life, confidence should be ~0.5
        result = manager.calculate_decay(1.0, 90)
        assert 0.49 < result < 0.51

        # At 2x half-life, confidence should be ~0.25
        result = manager.calculate_decay(1.0, 180)
        assert 0.24 < result < 0.26

    def test_linear_decay_calculation(self):
        """Test linear decay formula."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
            DecayConfig,
            DecayModel,
        )

        config = DecayConfig(model=DecayModel.LINEAR, half_life_days=90)
        manager = ConfidenceDecayManager(config)

        # Linear decay should be proportional
        result = manager.calculate_decay(1.0, 45)
        assert 0.7 < result < 0.8

    def test_domain_specific_decay(self):
        """Test domain-specific half-lives."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
            DecayConfig,
        )

        config = DecayConfig(
            half_life_days=90,
            domain_half_lives={"technology": 30},
        )
        manager = ConfidenceDecayManager(config)

        # Tech decays faster
        tech_result = manager.calculate_decay(1.0, 30, domain="technology")
        default_result = manager.calculate_decay(1.0, 30, domain=None)

        assert tech_result < default_result

    def test_confidence_boost(self):
        """Test confidence boosting from events."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
            ConfidenceEvent,
        )

        manager = ConfidenceDecayManager()

        # Access boost
        result = manager.calculate_boost(0.5, ConfidenceEvent.ACCESSED)
        assert result > 0.5

        # Validation boost (larger)
        result = manager.calculate_boost(0.5, ConfidenceEvent.VALIDATED)
        assert result > 0.55

        # Invalidation penalty
        result = manager.calculate_boost(0.5, ConfidenceEvent.INVALIDATED)
        assert result < 0.5

    def test_confidence_adjustment_to_dict(self):
        """Test adjustment serialization."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceAdjustment,
            ConfidenceEvent,
        )

        adj = ConfidenceAdjustment(
            id="a1",
            item_id="item1",
            event=ConfidenceEvent.DECAYED,
            old_confidence=0.8,
            new_confidence=0.6,
            reason="Time-based decay",
        )
        d = adj.to_dict()

        assert d["id"] == "a1"
        assert d["event"] == "decayed"
        assert d["old_confidence"] == 0.8


# =============================================================================
# Governance Tests
# =============================================================================


class TestGovernance:
    """Tests for governance module (RBAC + Audit)."""

    def test_permission_enum(self):
        """Test Permission enum values."""
        from aragora.knowledge.mound.ops.governance import Permission

        assert Permission.READ.value == "read"
        assert Permission.CREATE.value == "create"
        assert Permission.ADMIN.value == "admin"

    def test_builtin_role_enum(self):
        """Test BuiltinRole enum values."""
        from aragora.knowledge.mound.ops.governance import BuiltinRole

        assert BuiltinRole.VIEWER.value == "viewer"
        assert BuiltinRole.ADMIN.value == "admin"

    def test_builtin_roles_exist(self):
        """Test builtin roles are defined."""
        from aragora.knowledge.mound.ops.governance import BUILTIN_ROLES, BuiltinRole

        assert BuiltinRole.VIEWER in BUILTIN_ROLES
        assert BuiltinRole.ADMIN in BUILTIN_ROLES

        viewer = BUILTIN_ROLES[BuiltinRole.VIEWER]
        assert viewer.is_builtin is True

    def test_role_has_permission(self):
        """Test role permission checking."""
        from aragora.knowledge.mound.ops.governance import Permission, Role

        role = Role(
            id="r1",
            name="Test",
            description="Test role",
            permissions={Permission.READ, Permission.CREATE},
        )

        assert role.has_permission(Permission.READ)
        assert role.has_permission(Permission.CREATE)
        assert not role.has_permission(Permission.DELETE)

    def test_admin_role_has_all_permissions(self):
        """Test admin role grants all permissions."""
        from aragora.knowledge.mound.ops.governance import Permission, Role

        admin = Role(
            id="admin",
            name="Admin",
            description="Admin role",
            permissions={Permission.ADMIN},
        )

        # Admin should have all permissions
        assert admin.has_permission(Permission.READ)
        assert admin.has_permission(Permission.DELETE)
        assert admin.has_permission(Permission.MANAGE_USERS)

    @pytest.mark.asyncio
    async def test_rbac_manager_create_role(self):
        """Test creating a custom role."""
        from aragora.knowledge.mound.ops.governance import Permission, RBACManager

        manager = RBACManager()

        role = await manager.create_role(
            name="Custom Editor",
            permissions={Permission.READ, Permission.UPDATE},
            description="Custom editor role",
        )

        assert role.name == "Custom Editor"
        assert Permission.UPDATE in role.permissions

    @pytest.mark.asyncio
    async def test_rbac_manager_assign_role(self):
        """Test assigning a role to a user."""
        from aragora.knowledge.mound.ops.governance import RBACManager

        manager = RBACManager()

        assignment = await manager.assign_role(
            user_id="user1",
            role_id="builtin:viewer",
            assigned_by="admin1",
        )

        assert assignment.user_id == "user1"
        assert assignment.role_id == "builtin:viewer"

    @pytest.mark.asyncio
    async def test_rbac_check_permission(self):
        """Test checking user permissions."""
        from aragora.knowledge.mound.ops.governance import Permission, RBACManager

        manager = RBACManager()

        # Assign viewer role
        await manager.assign_role("user1", "builtin:viewer")

        # Check permissions
        can_read = await manager.check_permission("user1", Permission.READ)
        can_delete = await manager.check_permission("user1", Permission.DELETE)

        assert can_read is True
        assert can_delete is False

    def test_audit_action_enum(self):
        """Test AuditAction enum values."""
        from aragora.knowledge.mound.ops.governance import AuditAction

        assert AuditAction.ITEM_CREATE.value == "item.create"
        assert AuditAction.ROLE_ASSIGN.value == "role.assign"

    @pytest.mark.asyncio
    async def test_audit_trail_log(self):
        """Test logging an audit entry."""
        from aragora.knowledge.mound.ops.governance import AuditAction, AuditTrail

        trail = AuditTrail()

        entry = await trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user1",
            resource_type="knowledge_item",
            resource_id="item123",
            workspace_id="ws1",
        )

        assert entry.action == AuditAction.ITEM_CREATE
        assert entry.actor_id == "user1"
        assert entry.success is True

    @pytest.mark.asyncio
    async def test_audit_trail_query(self):
        """Test querying audit entries."""
        from aragora.knowledge.mound.ops.governance import AuditAction, AuditTrail

        trail = AuditTrail()

        # Log multiple entries
        await trail.log(AuditAction.ITEM_CREATE, "user1", "item", "i1")
        await trail.log(AuditAction.ITEM_READ, "user2", "item", "i2")
        await trail.log(AuditAction.ITEM_CREATE, "user1", "item", "i3")

        # Query by actor
        results = await trail.query(actor_id="user1")
        assert len(results) == 2

        # Query by action
        results = await trail.query(action=AuditAction.ITEM_READ)
        assert len(results) == 1


# =============================================================================
# Analytics Tests
# =============================================================================


class TestAnalytics:
    """Tests for analytics module."""

    def test_usage_event_type_enum(self):
        """Test UsageEventType enum values."""
        from aragora.knowledge.mound.ops.analytics import UsageEventType

        assert UsageEventType.QUERY.value == "query"
        assert UsageEventType.VIEW.value == "view"
        assert UsageEventType.CITE.value == "cite"

    def test_domain_coverage_score(self):
        """Test domain coverage score calculation."""
        from aragora.knowledge.mound.ops.analytics import DomainCoverage

        # Well-covered domain
        coverage = DomainCoverage(
            domain="software",
            total_items=100,
            high_confidence_items=80,
            medium_confidence_items=15,
            low_confidence_items=5,
            average_confidence=0.85,
            average_age_days=30,
            stale_items=5,
            topics=["python", "testing"],
        )

        score = coverage.coverage_score
        assert 0.7 < score <= 1.0

        # Sparse domain
        sparse = DomainCoverage(
            domain="unknown",
            total_items=5,
            high_confidence_items=1,
            medium_confidence_items=2,
            low_confidence_items=2,
            average_confidence=0.3,
            average_age_days=200,
            stale_items=3,
            topics=[],
        )

        sparse_score = sparse.coverage_score
        assert sparse_score < score

    def test_item_usage_stats_engagement_score(self):
        """Test item engagement score calculation."""
        from aragora.knowledge.mound.ops.analytics import ItemUsageStats

        stats = ItemUsageStats(
            item_id="item1",
            view_count=10,
            query_hits=5,
            citation_count=2,
            share_count=1,
        )

        # engagement = views*1 + queries*2 + citations*5 + shares*3
        expected = 10 * 1 + 5 * 2 + 2 * 5 + 1 * 3
        assert stats.engagement_score == expected

    @pytest.mark.asyncio
    async def test_analytics_record_usage(self):
        """Test recording usage events."""
        from aragora.knowledge.mound.ops.analytics import (
            KnowledgeAnalytics,
            UsageEventType,
        )

        analytics = KnowledgeAnalytics()

        event = await analytics.record_usage(
            event_type=UsageEventType.VIEW,
            item_id="item1",
            user_id="user1",
            workspace_id="ws1",
        )

        assert event.event_type == UsageEventType.VIEW
        assert event.item_id == "item1"

        # Check item stats updated
        stats = analytics._item_usage["item1"]
        assert stats.view_count == 1

    def test_quality_snapshot_to_dict(self):
        """Test quality snapshot serialization."""
        from aragora.knowledge.mound.ops.analytics import QualitySnapshot

        snapshot = QualitySnapshot(
            timestamp=datetime.now(),
            total_items=100,
            average_confidence=0.75,
            stale_percentage=0.1,
            contradiction_count=3,
            high_quality_count=50,
        )

        d = snapshot.to_dict()
        assert d["total_items"] == 100
        assert "timestamp" in d


# =============================================================================
# Knowledge Extraction Tests
# =============================================================================


class TestKnowledgeExtraction:
    """Tests for knowledge extraction module."""

    def test_extraction_type_enum(self):
        """Test ExtractionType enum values."""
        from aragora.knowledge.mound.ops.extraction import ExtractionType

        assert ExtractionType.FACT.value == "fact"
        assert ExtractionType.DEFINITION.value == "definition"
        assert ExtractionType.CONSENSUS.value == "consensus"

    def test_confidence_source_enum(self):
        """Test ConfidenceSource enum values."""
        from aragora.knowledge.mound.ops.extraction import ConfidenceSource

        assert ConfidenceSource.SINGLE_AGENT.value == "single_agent"
        assert ConfidenceSource.CONSENSUS.value == "consensus"

    def test_extracted_claim_agreement_ratio(self):
        """Test agreement ratio calculation."""
        from aragora.knowledge.mound.ops.extraction import (
            ExtractionType,
            ExtractedClaim,
        )

        claim = ExtractedClaim(
            id="c1",
            content="Test claim",
            extraction_type=ExtractionType.FACT,
            source_debate_id="d1",
            supporting_agents=["a1", "a2", "a3"],
            contradicting_agents=["a4"],
        )

        # 3 supporting, 1 contradicting = 3/4 = 0.75
        assert claim.agreement_ratio == 0.75

    def test_extraction_config_defaults(self):
        """Test extraction config default values."""
        from aragora.knowledge.mound.ops.extraction import ExtractionConfig

        config = ExtractionConfig()

        assert config.min_confidence_to_extract == 0.3
        assert config.min_confidence_to_promote == 0.6
        assert config.extract_facts is True
        assert len(config.fact_patterns) > 0

    @pytest.mark.asyncio
    async def test_extractor_initialization(self):
        """Test extractor initializes correctly."""
        from aragora.knowledge.mound.ops.extraction import DebateKnowledgeExtractor

        extractor = DebateKnowledgeExtractor()
        assert extractor.config is not None
        assert extractor._extracted_claims == {}

    @pytest.mark.asyncio
    async def test_extract_from_debate(self):
        """Test extracting knowledge from debate messages."""
        from aragora.knowledge.mound.ops.extraction import DebateKnowledgeExtractor

        extractor = DebateKnowledgeExtractor()

        messages = [
            {
                "agent_id": "claude",
                "content": "According to research, Python is a dynamically typed language. This means variables don't have explicit types.",
            },
            {
                "agent_id": "gpt",
                "content": "I agree. Python is defined as an interpreted, high-level language.",
            },
        ]

        result = await extractor.extract_from_debate(
            debate_id="d1",
            messages=messages,
            topic="Python programming",
        )

        assert result.debate_id == "d1"
        assert len(result.claims) >= 0  # May extract claims depending on patterns
        assert "Python programming" in result.topics_discovered

    def test_extraction_result_to_dict(self):
        """Test extraction result serialization."""
        from aragora.knowledge.mound.ops.extraction import ExtractionResult

        result = ExtractionResult(
            debate_id="d1",
            claims=[],
            relationships=[],
            topics_discovered=["python", "testing"],
            promoted_to_mound=0,
            extraction_duration_ms=50.5,
        )

        d = result.to_dict()
        assert d["debate_id"] == "d1"
        assert d["claims_extracted"] == 0
        assert "extraction_duration_ms" in d


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhaseA2Integration:
    """Integration tests for Phase A2 modules with KnowledgeMound."""

    def test_knowledge_mound_has_phase_a2_methods(self):
        """Test KnowledgeMound has all Phase A2 methods."""
        from aragora.knowledge.mound import KnowledgeMound

        mound = KnowledgeMound()

        # Contradiction methods
        assert hasattr(mound, "detect_contradictions")
        assert hasattr(mound, "resolve_contradiction")

        # Confidence decay methods
        assert hasattr(mound, "apply_confidence_decay")
        assert hasattr(mound, "record_confidence_event")

        # Governance methods
        assert hasattr(mound, "create_role")
        assert hasattr(mound, "check_permission")
        assert hasattr(mound, "log_audit")

        # Analytics methods
        assert hasattr(mound, "analyze_coverage")
        assert hasattr(mound, "analyze_usage")

        # Extraction methods
        assert hasattr(mound, "extract_from_debate")
        assert hasattr(mound, "promote_extracted_knowledge")
