"""Tests for GauntletOrchestrator and related dataclasses from aragora.gauntlet."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aragora.gauntlet.orchestrator import (
    CODE_REVIEW_GAUNTLET,
    Finding,
    GauntletConfig,
    GauntletOrchestrator,
    GauntletProgress,
    GauntletResult,
    POLICY_GAUNTLET,
    QUICK_GAUNTLET,
    THOROUGH_GAUNTLET,
    VerifiedClaim,
)

from aragora.gauntlet.types import InputType, Verdict
from aragora.modes.redteam import AttackType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finding(
    severity: float,
    category: str = "test",
    finding_id: str = "f-001",
    title: str = "Test finding",
    description: str = "A test finding",
    **kwargs,
) -> Finding:
    """Helper to create a Finding with sensible defaults."""
    return Finding(
        finding_id=finding_id,
        category=category,
        severity=severity,
        title=title,
        description=description,
        **kwargs,
    )


def _make_result(
    verdict: Verdict = Verdict.APPROVED,
    critical: list[Finding] | None = None,
    high: list[Finding] | None = None,
    medium: list[Finding] | None = None,
    low: list[Finding] | None = None,
    verified_claims: list[VerifiedClaim] | None = None,
    **kwargs,
) -> GauntletResult:
    """Helper to create a GauntletResult with sensible defaults."""
    return GauntletResult(
        gauntlet_id="gauntlet-test-001",
        input_type=InputType.SPEC,
        input_summary="Test input summary",
        verdict=verdict,
        confidence=0.85,
        risk_score=0.3,
        robustness_score=0.8,
        coverage_score=0.7,
        critical_findings=critical or [],
        high_findings=high or [],
        medium_findings=medium or [],
        low_findings=low or [],
        verified_claims=verified_claims or [],
        **kwargs,
    )


# ===========================================================================
# GauntletProgress tests
# ===========================================================================


class TestGauntletProgress:
    """Tests for GauntletProgress dataclass."""

    def test_all_fields(self):
        """GauntletProgress stores all provided fields."""
        progress = GauntletProgress(
            phase="Risk Assessment",
            phase_number=1,
            total_phases=5,
            percent=20.0,
            message="Analyzing risks...",
            findings_so_far=3,
            current_task="risk_scan",
        )
        assert progress.phase == "Risk Assessment"
        assert progress.phase_number == 1
        assert progress.total_phases == 5
        assert progress.percent == 20.0
        assert progress.message == "Analyzing risks..."
        assert progress.findings_so_far == 3
        assert progress.current_task == "risk_scan"

    def test_defaults(self):
        """GauntletProgress default values for optional fields."""
        progress = GauntletProgress(
            phase="Init",
            phase_number=0,
            total_phases=3,
            percent=0.0,
            message="Starting",
        )
        assert progress.findings_so_far == 0
        assert progress.current_task is None

    def test_percent_boundary_zero(self):
        """Progress percent can be zero."""
        progress = GauntletProgress(
            phase="Init", phase_number=1, total_phases=3, percent=0.0, message="Start"
        )
        assert progress.percent == 0.0

    def test_percent_boundary_hundred(self):
        """Progress percent can be 100."""
        progress = GauntletProgress(
            phase="Done", phase_number=3, total_phases=3, percent=100.0, message="Complete"
        )
        assert progress.percent == 100.0


# ===========================================================================
# Finding tests
# ===========================================================================


class TestFinding:
    """Tests for Finding dataclass."""

    def test_all_fields(self):
        """Finding stores all provided fields."""
        finding = Finding(
            finding_id="f-001",
            category="attack",
            severity=0.85,
            title="SQL injection",
            description="Found SQL injection in login form",
            evidence="SELECT * FROM users WHERE ...",
            mitigation="Use parameterized queries",
            source="RedTeam/agent1",
            verified=True,
        )
        assert finding.finding_id == "f-001"
        assert finding.category == "attack"
        assert finding.severity == 0.85
        assert finding.title == "SQL injection"
        assert finding.description == "Found SQL injection in login form"
        assert finding.evidence == "SELECT * FROM users WHERE ..."
        assert finding.mitigation == "Use parameterized queries"
        assert finding.source == "RedTeam/agent1"
        assert finding.verified is True
        assert finding.timestamp  # auto-generated, non-empty

    def test_defaults(self):
        """Finding default values for optional fields."""
        finding = _make_finding(severity=0.5)
        assert finding.evidence == ""
        assert finding.mitigation is None
        assert finding.source == ""
        assert finding.verified is False
        assert isinstance(finding.timestamp, str)
        assert len(finding.timestamp) > 0

    def test_severity_level_critical_at_boundary(self):
        """Severity level is CRITICAL at exactly 0.9."""
        finding = _make_finding(severity=0.9)
        assert finding.severity_level == "CRITICAL"

    def test_severity_level_critical_above(self):
        """Severity level is CRITICAL above 0.9."""
        finding = _make_finding(severity=0.95)
        assert finding.severity_level == "CRITICAL"

    def test_severity_level_critical_at_one(self):
        """Severity level is CRITICAL at 1.0."""
        finding = _make_finding(severity=1.0)
        assert finding.severity_level == "CRITICAL"

    def test_severity_level_high_at_boundary(self):
        """Severity level is HIGH at exactly 0.7."""
        finding = _make_finding(severity=0.7)
        assert finding.severity_level == "HIGH"

    def test_severity_level_high_just_above(self):
        """Severity level is HIGH at 0.89 (just below CRITICAL)."""
        finding = _make_finding(severity=0.89)
        assert finding.severity_level == "HIGH"

    def test_severity_level_medium_at_boundary(self):
        """Severity level is MEDIUM at exactly 0.4."""
        finding = _make_finding(severity=0.4)
        assert finding.severity_level == "MEDIUM"

    def test_severity_level_medium_just_above(self):
        """Severity level is MEDIUM at 0.69 (just below HIGH)."""
        finding = _make_finding(severity=0.69)
        assert finding.severity_level == "MEDIUM"

    def test_severity_level_low_just_below_medium(self):
        """Severity level is LOW at 0.39 (just below MEDIUM)."""
        finding = _make_finding(severity=0.39)
        assert finding.severity_level == "LOW"

    def test_severity_level_low_at_zero(self):
        """Severity level is LOW at 0.0."""
        finding = _make_finding(severity=0.0)
        assert finding.severity_level == "LOW"

    def test_severity_level_low_small_value(self):
        """Severity level is LOW at a small positive value."""
        finding = _make_finding(severity=0.1)
        assert finding.severity_level == "LOW"


# ===========================================================================
# VerifiedClaim tests
# ===========================================================================


class TestVerifiedClaim:
    """Tests for VerifiedClaim dataclass."""

    def test_all_fields(self):
        """VerifiedClaim stores all provided fields."""
        claim = VerifiedClaim(
            claim="All inputs are validated",
            verified=True,
            verification_method="z3",
            proof_hash="abc123def456",
            verification_time_ms=150.5,
        )
        assert claim.claim == "All inputs are validated"
        assert claim.verified is True
        assert claim.verification_method == "z3"
        assert claim.proof_hash == "abc123def456"
        assert claim.verification_time_ms == 150.5

    def test_defaults(self):
        """VerifiedClaim default values for optional fields."""
        claim = VerifiedClaim(
            claim="System is consistent",
            verified=False,
            verification_method="lean",
        )
        assert claim.proof_hash is None
        assert claim.verification_time_ms == 0.0

    def test_unverified_claim(self):
        """VerifiedClaim can represent an unverified claim."""
        claim = VerifiedClaim(
            claim="Performance is adequate",
            verified=False,
            verification_method="manual",
        )
        assert claim.verified is False

    def test_verification_methods(self):
        """VerifiedClaim accepts various verification methods."""
        for method in ("z3", "lean", "manual"):
            claim = VerifiedClaim(claim="Test", verified=True, verification_method=method)
            assert claim.verification_method == method


# ===========================================================================
# GauntletConfig tests
# ===========================================================================


class TestGauntletConfig:
    """Tests for GauntletConfig dataclass."""

    def test_defaults(self):
        """GauntletConfig has sensible defaults."""
        config = GauntletConfig()
        assert config.input_type == InputType.SPEC
        assert config.input_content == ""
        assert config.input_path is None
        assert config.attack_types is None
        assert config.probe_types is None
        assert config.severity_threshold == 0.5
        assert config.risk_threshold == 0.7
        assert config.max_duration_seconds == 600
        assert config.verification_timeout_seconds == 60.0
        assert config.parallel_attacks == 5
        assert config.parallel_probes == 3
        assert config.enable_redteam is True
        assert config.enable_probing is True
        assert config.enable_deep_audit is True
        assert config.enable_verification is True
        assert config.enable_risk_assessment is True
        assert config.deep_audit_rounds == 4
        assert config.persona is None

    def test_custom_values(self):
        """GauntletConfig accepts custom values."""
        config = GauntletConfig(
            input_type=InputType.CODE,
            input_content="def hello(): pass",
            severity_threshold=0.8,
            risk_threshold=0.9,
            max_duration_seconds=300,
            parallel_attacks=10,
            parallel_probes=8,
            deep_audit_rounds=6,
        )
        assert config.input_type == InputType.CODE
        assert config.input_content == "def hello(): pass"
        assert config.severity_threshold == 0.8
        assert config.risk_threshold == 0.9
        assert config.max_duration_seconds == 300
        assert config.parallel_attacks == 10
        assert config.parallel_probes == 8
        assert config.deep_audit_rounds == 6

    def test_feature_toggles_all_disabled(self):
        """GauntletConfig feature toggles can all be disabled."""
        config = GauntletConfig(
            enable_redteam=False,
            enable_probing=False,
            enable_deep_audit=False,
            enable_verification=False,
            enable_risk_assessment=False,
        )
        assert config.enable_redteam is False
        assert config.enable_probing is False
        assert config.enable_deep_audit is False
        assert config.enable_verification is False
        assert config.enable_risk_assessment is False

    def test_post_init_reads_file_from_path(self, tmp_path):
        """__post_init__ reads file content when input_path is provided."""
        test_file = tmp_path / "test_input.txt"
        test_file.write_text("Content from file")
        config = GauntletConfig(input_path=test_file)
        assert config.input_content == "Content from file"

    def test_post_init_does_not_overwrite_existing_content(self, tmp_path):
        """__post_init__ does not read file if input_content is already set."""
        test_file = tmp_path / "test_input.txt"
        test_file.write_text("File content")
        config = GauntletConfig(input_path=test_file, input_content="Existing content")
        assert config.input_content == "Existing content"

    def test_persona_string_kept_when_personas_unavailable(self):
        """Persona string is kept as-is when personas module is not available."""
        with patch("aragora.gauntlet.orchestrator.PERSONAS_AVAILABLE", False):
            config = GauntletConfig(persona="gdpr")
            # When personas aren't available, string stays as string
            assert config.persona == "gdpr"

    def test_attack_types_list(self):
        """GauntletConfig accepts explicit attack types."""
        attacks = [AttackType.SECURITY, AttackType.EDGE_CASE]
        config = GauntletConfig(attack_types=attacks)
        assert config.attack_types == attacks
        assert len(config.attack_types) == 2

    def test_input_type_values(self):
        """GauntletConfig accepts all InputType values."""
        for input_type in InputType:
            config = GauntletConfig(input_type=input_type)
            assert config.input_type == input_type


# ===========================================================================
# GauntletResult tests
# ===========================================================================


class TestGauntletResult:
    """Tests for GauntletResult dataclass."""

    def test_minimal_result(self):
        """GauntletResult can be created with required fields only."""
        result = _make_result()
        assert result.gauntlet_id == "gauntlet-test-001"
        assert result.input_type == InputType.SPEC
        assert result.verdict == Verdict.APPROVED
        assert result.confidence == 0.85

    def test_all_findings_combined(self):
        """all_findings returns critical + high + medium + low in order."""
        c = _make_finding(0.95, finding_id="c1")
        h = _make_finding(0.75, finding_id="h1")
        m = _make_finding(0.5, finding_id="m1")
        lo = _make_finding(0.2, finding_id="l1")
        result = _make_result(critical=[c], high=[h], medium=[m], low=[lo])
        all_f = result.all_findings
        assert len(all_f) == 4
        assert all_f[0].finding_id == "c1"
        assert all_f[1].finding_id == "h1"
        assert all_f[2].finding_id == "m1"
        assert all_f[3].finding_id == "l1"

    def test_total_findings(self):
        """total_findings returns the count of all findings."""
        result = _make_result(
            critical=[_make_finding(0.95, finding_id="c1")],
            high=[_make_finding(0.75, finding_id="h1"), _make_finding(0.8, finding_id="h2")],
            medium=[_make_finding(0.5, finding_id="m1")],
        )
        assert result.total_findings == 4

    def test_total_findings_zero(self):
        """total_findings is zero when no findings exist."""
        result = _make_result()
        assert result.total_findings == 0

    def test_severity_counts(self):
        """severity_counts returns per-level counts."""
        result = _make_result(
            critical=[_make_finding(0.95, finding_id="c1")],
            high=[_make_finding(0.75, finding_id="h1"), _make_finding(0.8, finding_id="h2")],
            medium=[_make_finding(0.5, finding_id="m1")],
            low=[],
        )
        counts = result.severity_counts
        assert counts["critical"] == 1
        assert counts["high"] == 2
        assert counts["medium"] == 1
        assert counts["low"] == 0

    def test_verdict_types(self):
        """GauntletResult works with all verdict types."""
        for verdict in (
            Verdict.APPROVED,
            Verdict.NEEDS_REVIEW,
            Verdict.REJECTED,
            Verdict.APPROVED_WITH_CONDITIONS,
            Verdict.PASS,
            Verdict.FAIL,
            Verdict.CONDITIONAL,
        ):
            result = _make_result(verdict=verdict)
            assert result.verdict == verdict

    def test_checksum_deterministic(self):
        """Checksum is deterministic for same inputs."""
        result1 = _make_result()
        result2 = _make_result()
        assert result1.checksum == result2.checksum

    def test_checksum_changes_with_verdict(self):
        """Checksum differs when verdict changes."""
        result1 = _make_result(verdict=Verdict.APPROVED)
        result2 = _make_result(verdict=Verdict.REJECTED)
        assert result1.checksum != result2.checksum

    def test_checksum_format(self):
        """Checksum is a 16-character hex string."""
        result = _make_result()
        assert len(result.checksum) == 16
        assert all(c in "0123456789abcdef" for c in result.checksum)

    def test_verification_coverage_default(self):
        """Default verification coverage is 0.0."""
        result = _make_result()
        assert result.verification_coverage == 0.0

    def test_verification_coverage_custom(self):
        """Custom verification coverage is stored correctly."""
        result = _make_result(verification_coverage=0.75)
        assert result.verification_coverage == 0.75

    def test_to_dict_structure(self):
        """to_dict returns expected keys."""
        result = _make_result()
        d = result.to_dict()
        expected_keys = {
            "gauntlet_id",
            "input_type",
            "input_summary",
            "input_hash",
            "verdict",
            "confidence",
            "risk_score",
            "robustness_score",
            "coverage_score",
            "verification_coverage",
            "severity_counts",
            "findings",
            "consensus_reached",
            "agents_involved",
            "duration_seconds",
            "created_at",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_values(self):
        """to_dict serializes values correctly."""
        result = _make_result(verdict=Verdict.NEEDS_REVIEW)
        d = result.to_dict()
        assert d["gauntlet_id"] == "gauntlet-test-001"
        assert d["input_type"] == "spec"
        assert d["verdict"] == "needs_review"
        assert d["confidence"] == 0.85
        assert d["risk_score"] == 0.3
        assert d["robustness_score"] == 0.8
        assert d["coverage_score"] == 0.7

    def test_to_dict_findings_serialized(self):
        """to_dict includes serialized findings."""
        finding = _make_finding(
            severity=0.95,
            finding_id="f-critical",
            category="attack",
            title="XSS Attack",
            description="Cross-site scripting",
            evidence="<script>alert(1)</script>",
            mitigation="Escape output",
            source="RedTeam/agent1",
        )
        result = _make_result(critical=[finding])
        d = result.to_dict()
        assert len(d["findings"]) == 1
        f = d["findings"][0]
        assert f["id"] == "f-critical"
        assert f["category"] == "attack"
        assert f["severity"] == 0.95
        assert f["severity_level"] == "CRITICAL"
        assert f["title"] == "XSS Attack"
        assert f["description"] == "Cross-site scripting"
        assert f["evidence"] == "<script>alert(1)</script>"
        assert f["mitigation"] == "Escape output"
        assert f["source"] == "RedTeam/agent1"
        assert f["verified"] is False

    def test_to_dict_empty_findings(self):
        """to_dict with no findings returns empty list."""
        result = _make_result()
        d = result.to_dict()
        assert d["findings"] == []

    def test_summary_contains_verdict(self):
        """summary() includes the verdict text."""
        result = _make_result(verdict=Verdict.REJECTED)
        s = result.summary()
        assert "REJECTED" in s

    def test_summary_contains_gauntlet_id(self):
        """summary() includes the gauntlet ID."""
        result = _make_result()
        s = result.summary()
        assert "gauntlet-test-001" in s

    def test_summary_contains_scores(self):
        """summary() includes the score values."""
        result = _make_result()
        s = result.summary()
        assert "Risk Score" in s
        assert "Robustness Score" in s
        assert "Coverage Score" in s

    def test_summary_contains_finding_counts(self):
        """summary() includes finding counts per severity."""
        result = _make_result(
            critical=[_make_finding(0.95, finding_id="c1")],
            high=[_make_finding(0.75, finding_id="h1")],
        )
        s = result.summary()
        assert "Critical: 1" in s
        assert "High: 1" in s

    def test_summary_lists_critical_issues(self):
        """summary() lists titles of critical findings."""
        c = _make_finding(0.95, finding_id="c1", title="Buffer Overflow")
        result = _make_result(critical=[c])
        s = result.summary()
        assert "Buffer Overflow" in s
        assert "CRITICAL ISSUES:" in s

    def test_summary_no_critical_section_when_empty(self):
        """summary() omits CRITICAL ISSUES section when there are none."""
        result = _make_result()
        s = result.summary()
        assert "CRITICAL ISSUES:" not in s

    def test_summary_includes_dissenting_views_count(self):
        """summary() includes dissenting views count if present."""
        from aragora.debate.consensus import DissentRecord

        dissent = DissentRecord(
            agent="agent1",
            claim_id="claim-1",
            dissent_type="partial",
            reasons=["Insufficient evidence"],
            severity=0.5,
        )
        result = _make_result(dissenting_views=[dissent])
        s = result.summary()
        assert "Dissenting Views: 1" in s

    def test_summary_includes_unresolved_tensions_count(self):
        """summary() includes unresolved tensions count if present."""
        from aragora.debate.consensus import UnresolvedTension

        tension = UnresolvedTension(
            tension_id="t-1",
            description="Performance vs Security tradeoff",
            agents_involved=["agent1", "agent2"],
            options=["option-a", "option-b"],
            impact="System design",
        )
        result = _make_result(unresolved_tensions=[tension])
        s = result.summary()
        assert "Unresolved Tensions: 1" in s

    def test_summary_includes_checksum(self):
        """summary() includes the integrity checksum."""
        result = _make_result()
        s = result.summary()
        assert result.checksum in s

    def test_input_hash_default(self):
        """Default input_hash is empty string."""
        result = _make_result()
        assert result.input_hash == ""

    def test_input_hash_custom(self):
        """Custom input_hash is stored correctly."""
        h = hashlib.sha256(b"test").hexdigest()
        result = _make_result(input_hash=h)
        assert result.input_hash == h

    def test_agents_involved_default(self):
        """Default agents_involved is empty list."""
        result = _make_result()
        assert result.agents_involved == []

    def test_consensus_reached_default(self):
        """Default consensus_reached is False."""
        result = _make_result()
        assert result.consensus_reached is False


# ===========================================================================
# Pre-configured profiles tests
# ===========================================================================


class TestPreConfiguredProfiles:
    """Tests for pre-configured GauntletConfig profiles."""

    def test_quick_gauntlet_exists(self):
        """QUICK_GAUNTLET is a GauntletConfig instance."""
        assert isinstance(QUICK_GAUNTLET, GauntletConfig)

    def test_quick_gauntlet_fast_settings(self):
        """QUICK_GAUNTLET has reduced rounds and duration."""
        assert QUICK_GAUNTLET.deep_audit_rounds == 2
        assert QUICK_GAUNTLET.parallel_attacks == 2
        assert QUICK_GAUNTLET.enable_verification is False
        assert QUICK_GAUNTLET.max_duration_seconds == 120

    def test_thorough_gauntlet_exists(self):
        """THOROUGH_GAUNTLET is a GauntletConfig instance."""
        assert isinstance(THOROUGH_GAUNTLET, GauntletConfig)

    def test_thorough_gauntlet_comprehensive_settings(self):
        """THOROUGH_GAUNTLET has more rounds and longer duration."""
        assert THOROUGH_GAUNTLET.deep_audit_rounds == 6
        assert THOROUGH_GAUNTLET.parallel_attacks == 5
        assert THOROUGH_GAUNTLET.parallel_probes == 5
        assert THOROUGH_GAUNTLET.enable_verification is True
        assert THOROUGH_GAUNTLET.max_duration_seconds == 900

    def test_code_review_gauntlet_exists(self):
        """CODE_REVIEW_GAUNTLET is a GauntletConfig instance."""
        assert isinstance(CODE_REVIEW_GAUNTLET, GauntletConfig)

    def test_code_review_gauntlet_input_type(self):
        """CODE_REVIEW_GAUNTLET targets code input."""
        assert CODE_REVIEW_GAUNTLET.input_type == InputType.CODE

    def test_code_review_gauntlet_attack_types(self):
        """CODE_REVIEW_GAUNTLET includes security-related attack types."""
        attacks = CODE_REVIEW_GAUNTLET.attack_types
        assert attacks is not None
        assert AttackType.SECURITY in attacks
        assert AttackType.EDGE_CASE in attacks
        assert AttackType.RACE_CONDITION in attacks
        assert AttackType.RESOURCE_EXHAUSTION in attacks

    def test_code_review_gauntlet_verification_enabled(self):
        """CODE_REVIEW_GAUNTLET has verification enabled."""
        assert CODE_REVIEW_GAUNTLET.enable_verification is True

    def test_policy_gauntlet_exists(self):
        """POLICY_GAUNTLET is a GauntletConfig instance."""
        assert isinstance(POLICY_GAUNTLET, GauntletConfig)

    def test_policy_gauntlet_input_type(self):
        """POLICY_GAUNTLET targets policy input."""
        assert POLICY_GAUNTLET.input_type == InputType.POLICY

    def test_policy_gauntlet_attack_types(self):
        """POLICY_GAUNTLET includes logic-related attack types."""
        attacks = POLICY_GAUNTLET.attack_types
        assert attacks is not None
        assert AttackType.LOGICAL_FALLACY in attacks
        assert AttackType.UNSTATED_ASSUMPTION in attacks
        assert AttackType.EDGE_CASE in attacks
        assert AttackType.COUNTEREXAMPLE in attacks

    def test_policy_gauntlet_lower_severity_threshold(self):
        """POLICY_GAUNTLET uses a lower severity threshold for sensitivity."""
        assert POLICY_GAUNTLET.severity_threshold == 0.3

    def test_policy_gauntlet_deep_audit_rounds(self):
        """POLICY_GAUNTLET has 5 deep audit rounds."""
        assert POLICY_GAUNTLET.deep_audit_rounds == 5


# ===========================================================================
# GauntletOrchestrator tests
# ===========================================================================


class TestGauntletOrchestrator:
    """Tests for GauntletOrchestrator initialization and helpers."""

    def _make_mock_agent(self, name: str = "agent1") -> MagicMock:
        """Create a mock Agent with a name attribute."""
        agent = MagicMock()
        agent.name = name
        return agent

    def test_init_with_agents(self):
        """GauntletOrchestrator stores agents."""
        agents = [self._make_mock_agent("a1"), self._make_mock_agent("a2")]
        orchestrator = GauntletOrchestrator(agents)
        assert orchestrator.agents == agents
        assert len(orchestrator.agents) == 2

    def test_init_with_empty_agents(self):
        """GauntletOrchestrator can be initialized with empty agent list."""
        orchestrator = GauntletOrchestrator([])
        assert orchestrator.agents == []

    def test_init_with_run_agent_fn(self):
        """GauntletOrchestrator accepts custom run_agent_fn."""
        custom_fn = MagicMock()
        orchestrator = GauntletOrchestrator([], run_agent_fn=custom_fn)
        assert orchestrator.run_agent_fn is custom_fn

    def test_init_default_run_agent_fn(self):
        """GauntletOrchestrator uses default run_agent_fn when none provided."""
        orchestrator = GauntletOrchestrator([])
        assert orchestrator.run_agent_fn is not None
        assert callable(orchestrator.run_agent_fn)

    def test_init_with_progress_callback(self):
        """GauntletOrchestrator accepts progress callback."""
        callback = MagicMock()
        orchestrator = GauntletOrchestrator([], on_progress=callback)
        assert orchestrator.on_progress is callback

    def test_init_default_progress_callback(self):
        """GauntletOrchestrator defaults to no progress callback."""
        orchestrator = GauntletOrchestrator([])
        assert orchestrator.on_progress is None

    def test_finding_counter_starts_at_zero(self):
        """Finding counter starts at zero."""
        orchestrator = GauntletOrchestrator([])
        assert orchestrator._finding_counter == 0

    def test_next_finding_id_increments(self):
        """_next_finding_id generates sequential IDs."""
        orchestrator = GauntletOrchestrator([])
        id1 = orchestrator._next_finding_id()
        id2 = orchestrator._next_finding_id()
        id3 = orchestrator._next_finding_id()
        assert id1 == "finding-0001"
        assert id2 == "finding-0002"
        assert id3 == "finding-0003"

    def test_risk_level_to_severity_mapping(self):
        """_risk_level_to_severity maps RiskLevel to float correctly."""
        from aragora.debate.risk_assessor import RiskLevel

        orchestrator = GauntletOrchestrator([])
        assert orchestrator._risk_level_to_severity(RiskLevel.LOW) == 0.25
        assert orchestrator._risk_level_to_severity(RiskLevel.MEDIUM) == 0.5
        assert orchestrator._risk_level_to_severity(RiskLevel.HIGH) == 0.75
        assert orchestrator._risk_level_to_severity(RiskLevel.CRITICAL) == 0.95

    def test_emit_progress_with_callback(self):
        """_emit_progress calls the callback with a GauntletProgress."""
        callback = MagicMock()
        orchestrator = GauntletOrchestrator([], on_progress=callback)
        orchestrator._findings_count = 5
        orchestrator._emit_progress(
            "Testing", 2, 3, 50.0, "Running tests", current_task="test_task"
        )
        callback.assert_called_once()
        progress = callback.call_args[0][0]
        assert isinstance(progress, GauntletProgress)
        assert progress.phase == "Testing"
        assert progress.phase_number == 2
        assert progress.total_phases == 3
        assert progress.percent == 50.0
        assert progress.message == "Running tests"
        assert progress.findings_so_far == 5
        assert progress.current_task == "test_task"

    def test_emit_progress_without_callback(self):
        """_emit_progress is a no-op when no callback is set."""
        orchestrator = GauntletOrchestrator([])
        # Should not raise
        orchestrator._emit_progress("Testing", 1, 3, 10.0, "msg")

    def test_sub_components_initialized(self):
        """GauntletOrchestrator initializes sub-components."""
        orchestrator = GauntletOrchestrator([])
        assert orchestrator.redteam_mode is not None
        assert orchestrator.prober is not None
        assert orchestrator.risk_assessor is not None
        assert orchestrator.verification_manager is not None

    def test_determine_verdict_approved_clean(self):
        """_determine_verdict returns APPROVED when no serious issues."""
        orchestrator = GauntletOrchestrator([])
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=[],
            medium=[],
            risk_score=0.1,
            robustness_score=0.9,
            dissents=[],
        )
        assert verdict == Verdict.APPROVED
        assert 0.0 < confidence <= 0.95

    def test_determine_verdict_rejected_multiple_critical(self):
        """_determine_verdict returns REJECTED with 2+ critical findings."""
        orchestrator = GauntletOrchestrator([])
        c1 = _make_finding(0.95, finding_id="c1")
        c2 = _make_finding(0.92, finding_id="c2")
        verdict, confidence = orchestrator._determine_verdict(
            critical=[c1, c2],
            high=[],
            medium=[],
            risk_score=0.5,
            robustness_score=0.3,
            dissents=[],
        )
        assert verdict == Verdict.REJECTED
        assert confidence == 0.9

    def test_determine_verdict_critical_plus_high_needs_review(self):
        """_determine_verdict returns NEEDS_REVIEW with 1 critical (regardless of high count)."""
        orchestrator = GauntletOrchestrator([])
        c1 = _make_finding(0.95, finding_id="c1")
        h1 = _make_finding(0.75, finding_id="h1")
        h2 = _make_finding(0.8, finding_id="h2")
        h3 = _make_finding(0.72, finding_id="h3")
        verdict, confidence = orchestrator._determine_verdict(
            critical=[c1],
            high=[h1, h2, h3],
            medium=[],
            risk_score=0.5,
            robustness_score=0.3,
            dissents=[],
        )
        # 1 critical triggers NEEDS_REVIEW (2+ critical needed for REJECTED)
        assert verdict == Verdict.NEEDS_REVIEW
        assert confidence == 0.7

    def test_determine_verdict_rejected_high_risk_score(self):
        """_determine_verdict returns REJECTED when risk_score > 0.8."""
        orchestrator = GauntletOrchestrator([])
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=[],
            medium=[],
            risk_score=0.85,
            robustness_score=0.5,
            dissents=[],
        )
        assert verdict == Verdict.REJECTED
        assert confidence == 0.8

    def test_determine_verdict_needs_review_one_critical(self):
        """_determine_verdict returns NEEDS_REVIEW with exactly 1 critical."""
        orchestrator = GauntletOrchestrator([])
        c1 = _make_finding(0.95, finding_id="c1")
        verdict, confidence = orchestrator._determine_verdict(
            critical=[c1],
            high=[],
            medium=[],
            risk_score=0.3,
            robustness_score=0.7,
            dissents=[],
        )
        assert verdict == Verdict.NEEDS_REVIEW
        assert confidence == 0.7

    def test_determine_verdict_needs_review_many_high(self):
        """_determine_verdict returns NEEDS_REVIEW with 3+ high and no critical."""
        orchestrator = GauntletOrchestrator([])
        highs = [_make_finding(0.75, finding_id=f"h{i}") for i in range(3)]
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=highs,
            medium=[],
            risk_score=0.3,
            robustness_score=0.7,
            dissents=[],
        )
        assert verdict == Verdict.NEEDS_REVIEW
        assert confidence == 0.65

    def test_determine_verdict_needs_review_many_dissents(self):
        """_determine_verdict returns NEEDS_REVIEW with 3+ dissents."""
        from aragora.debate.consensus import DissentRecord

        orchestrator = GauntletOrchestrator([])
        dissents = [
            DissentRecord(
                agent=f"a{i}", claim_id="", dissent_type="full", reasons=["reason"], severity=0.5
            )
            for i in range(3)
        ]
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=[],
            medium=[],
            risk_score=0.3,
            robustness_score=0.7,
            dissents=dissents,
        )
        assert verdict == Verdict.NEEDS_REVIEW
        assert confidence == 0.6

    def test_determine_verdict_needs_review_moderate_risk(self):
        """_determine_verdict returns NEEDS_REVIEW when risk_score > 0.6."""
        orchestrator = GauntletOrchestrator([])
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=[],
            medium=[],
            risk_score=0.65,
            robustness_score=0.7,
            dissents=[],
        )
        assert verdict == Verdict.NEEDS_REVIEW
        assert confidence == 0.6

    def test_determine_verdict_approved_with_conditions(self):
        """_determine_verdict returns APPROVED_WITH_CONDITIONS with some high findings."""
        orchestrator = GauntletOrchestrator([])
        h1 = _make_finding(0.75, finding_id="h1")
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=[h1],
            medium=[],
            risk_score=0.2,
            robustness_score=0.8,
            dissents=[],
        )
        assert verdict == Verdict.APPROVED_WITH_CONDITIONS

    def test_determine_verdict_approved_with_conditions_many_medium(self):
        """_determine_verdict returns APPROVED_WITH_CONDITIONS with 3+ medium findings."""
        orchestrator = GauntletOrchestrator([])
        mediums = [_make_finding(0.5, finding_id=f"m{i}") for i in range(3)]
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=[],
            medium=mediums,
            risk_score=0.2,
            robustness_score=0.8,
            dissents=[],
        )
        assert verdict == Verdict.APPROVED_WITH_CONDITIONS

    def test_calculate_risk_score_no_findings(self):
        """_calculate_risk_score returns 0.0 with no findings or assessments."""
        orchestrator = GauntletOrchestrator([])
        score = orchestrator._calculate_risk_score([], [])
        assert score == 0.0

    def test_calculate_risk_score_with_findings(self):
        """_calculate_risk_score returns nonzero with findings."""
        orchestrator = GauntletOrchestrator([])
        findings = [_make_finding(0.8)]
        score = orchestrator._calculate_risk_score(findings, [])
        assert 0.0 < score <= 1.0

    def test_calculate_coverage_score_no_components(self):
        """_calculate_coverage_score returns 0.0 with no components."""
        orchestrator = GauntletOrchestrator([])
        score = orchestrator._calculate_coverage_score(None, None, None)
        assert score == 0.0
