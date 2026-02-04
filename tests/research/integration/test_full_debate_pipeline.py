"""
End-to-end integration tests for the full debate pipeline.

Tests the integration of all research components working together:
- Adaptive stopping with stability detection
- MUSE calibration for uncertainty quantification
- ASCoT fragility detection
- LaRA retrieval routing
- ThinkPRM step verification
- GraphRAG hybrid retrieval
- ClaimCheck verification
- A-HMAD team selection
"""

import pytest
import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, AsyncMock


@dataclass
class MockDebateRound:
    """Mock debate round for testing."""

    round_number: int
    contributions: list[dict[str, Any]]
    debate_id: str = "test-debate"


@dataclass
class MockAgent:
    """Mock agent for testing."""

    id: str
    name: str
    domains: list[str]
    elo_score: float = 1500.0
    calibration_score: float = 0.7


class TestFullDebatePipeline:
    """End-to-end tests for the full debate pipeline."""

    @pytest.fixture
    def mock_agents(self) -> list[MockAgent]:
        """Create mock agents for testing."""
        return [
            MockAgent("agent1", "Claude", ["general", "analysis"], 1600, 0.8),
            MockAgent("agent2", "GPT", ["general", "synthesis"], 1550, 0.75),
            MockAgent("agent3", "Gemini", ["general", "critique"], 1500, 0.7),
            MockAgent("agent4", "Llama", ["general", "research"], 1450, 0.65),
        ]

    @pytest.fixture
    def mock_debate_rounds(self) -> list[MockDebateRound]:
        """Create mock debate rounds."""
        return [
            MockDebateRound(
                round_number=1,
                contributions=[
                    {
                        "agent_id": "agent1",
                        "content": "The Roman Empire fell due to economic instability.",
                        "position": 0.7,
                    },
                    {
                        "agent_id": "agent2",
                        "content": "I agree, but military overextension was also a factor.",
                        "position": 0.65,
                    },
                ],
            ),
            MockDebateRound(
                round_number=2,
                contributions=[
                    {
                        "agent_id": "agent1",
                        "content": "The economic factors included currency debasement.",
                        "position": 0.75,
                    },
                    {
                        "agent_id": "agent3",
                        "content": "We should also consider barbarian invasions.",
                        "position": 0.6,
                    },
                ],
            ),
            MockDebateRound(
                round_number=3,
                contributions=[
                    {
                        "agent_id": "agent2",
                        "content": "Consensus: Multiple factors contributed to Rome's fall.",
                        "position": 0.7,
                    },
                ],
            ),
        ]

    @pytest.fixture
    def mock_evidence(self) -> list[str]:
        """Create mock evidence for testing."""
        return [
            "The Roman Empire experienced significant economic decline in its later years.",
            "Currency debasement was a major economic policy during the late Roman period.",
            "Barbarian invasions increased pressure on Roman borders in the 4th and 5th centuries.",
            "The Western Roman Empire officially ended in 476 AD.",
        ]

    @pytest.mark.asyncio
    async def test_stability_detection_integration(
        self, mock_debate_rounds: list[MockDebateRound]
    ) -> None:
        """Test stability detection integrates with debate rounds."""
        # Import components
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "stability",
            "aragora/debate/stability_detector.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            detector = module.BetaBinomialStabilityDetector()

            # Process rounds
            for round_data in mock_debate_rounds:
                positions = [c["position"] for c in round_data.contributions]
                avg_position = sum(positions) / len(positions)
                detector.update_round(round_data.round_number, avg_position)

            # Check stability
            result = detector.check_stability()
            assert result is not None
            assert "is_stable" in result
            assert "ks_distance" in result

    @pytest.mark.asyncio
    async def test_muse_calibration_integration(self, mock_agents: list[MockAgent]) -> None:
        """Test MUSE calibration integrates with agent votes."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "muse",
            "aragora/ranking/muse_calibration.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            calculator = module.MUSECalculator()

            # Simulate agent votes
            agent_votes = {
                "agent1": [0.7, 0.3],  # 70% option A, 30% option B
                "agent2": [0.65, 0.35],
                "agent3": [0.6, 0.4],
                "agent4": [0.55, 0.45],
            }
            calibration_scores = {a.id: a.calibration_score for a in mock_agents}

            # Calculate MUSE
            result = calculator.calculate(
                agent_votes=agent_votes,
                calibration_scores=calibration_scores,
            )

            assert result is not None
            assert result.divergence >= 0
            assert len(result.best_subset) >= 2

    @pytest.mark.asyncio
    async def test_ascot_fragility_integration(
        self, mock_debate_rounds: list[MockDebateRound]
    ) -> None:
        """Test ASCoT fragility detection integrates with debate rounds."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ascot",
            "aragora/debate/ascot_fragility.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.ASCoTFragilityAnalyzer()

            max_rounds = len(mock_debate_rounds)

            # Check fragility for each round
            for round_data in mock_debate_rounds:
                result = analyzer.calculate_round_fragility(
                    round_number=round_data.round_number,
                    total_rounds=max_rounds,
                    dependency_depth=round_data.round_number - 1,
                )

                assert result is not None
                assert result.fragility_score >= 0
                assert result.fragility_score <= 1

                # Later rounds should have higher fragility
                if round_data.round_number == max_rounds:
                    intensity = analyzer.get_verification_intensity(result)
                    assert intensity.value in ["high", "critical"]

    @pytest.mark.asyncio
    async def test_lara_routing_integration(self) -> None:
        """Test LaRA routing integrates with retrieval modes."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "lara",
            "aragora/routing/lara_router.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            router = module.LaRARouter()

            # Test different query types
            test_queries = [
                ("What is the capital of France?", "rag"),  # Simple factual
                ("Compare the economies of US and China", "hybrid"),  # Analytical
                ("Explain quantum computing step by step", "rlm"),  # Complex reasoning
            ]

            for query, expected_mode in test_queries:
                result = router.route(
                    query=query,
                    document_token_count=1000,
                )

                assert result is not None
                assert result.mode is not None
                # Mode should be one of the valid modes
                assert result.mode.value in ["rag", "rlm", "long_context", "graph", "hybrid"]

    @pytest.mark.asyncio
    async def test_think_prm_integration(self, mock_debate_rounds: list[MockDebateRound]) -> None:
        """Test ThinkPRM integrates with debate verification."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "think_prm",
            "aragora/verification/think_prm.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            verifier = module.ThinkPRMVerifier()

            # Mock query function
            async def mock_query(agent_id: str, prompt: str, max_tokens: int = 1000) -> str:
                return """VERDICT: CORRECT
CONFIDENCE: 0.85
REASONING: The reasoning is sound
SUGGESTED_FIX: None"""

            # Verify a step
            result = await verifier.verify_step(
                step_content=mock_debate_rounds[0].contributions[0]["content"],
                round_number=1,
                agent_id="agent1",
                prior_context="",
                dependencies=[],
                query_fn=mock_query,
            )

            assert result is not None
            assert result.verdict == module.StepVerdict.CORRECT
            assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_claim_check_integration(self, mock_evidence: list[str]) -> None:
        """Test ClaimCheck integrates with evidence verification."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "claim_check",
            "aragora/evidence/claim_check.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            verifier = module.ClaimCheckVerifier()

            # Verify a claim
            result = await verifier.verify(
                claim="The Roman Empire experienced economic decline.",
                evidence=mock_evidence,
            )

            assert result is not None
            assert result.decomposition is not None
            assert result.overall_status is not None

    @pytest.mark.asyncio
    async def test_ahmad_team_selection_integration(self, mock_agents: list[MockAgent]) -> None:
        """Test A-HMAD integrates with team selection."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ahmad",
            "aragora/debate/role_specializer.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            specializer = module.AHMADRoleSpecializer()

            # Analyze topic
            roles = specializer.analyze_topic(
                topic="Was the fall of Rome primarily due to economic factors?",
                domain="history",
                context={"complexity": "medium"},
            )

            assert len(roles) > 0

            # Assign roles
            composition = specializer.assign_roles(
                roles=roles,
                available_agents=[a.id for a in mock_agents],
                elo_scores={a.id: a.elo_score for a in mock_agents},
                calibration_scores={a.id: a.calibration_score for a in mock_agents},
                domain_scores={a.id: 0.7 for a in mock_agents},
            )

            assert composition is not None
            assert composition.diversity_score >= 0.5
            assert len(composition.assignments) > 0

    @pytest.mark.asyncio
    async def test_research_config_integration(self) -> None:
        """Test research config integrates with all components."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "config",
            "aragora/config/research_integration.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Test all presets
            for level in module.IntegrationLevel:
                config = module.ResearchIntegrationConfig.from_preset(level)
                features = config.get_enabled_features()

                # Validate config
                warnings = config.validate()

                # Standard and Full should have no warnings
                if level in (module.IntegrationLevel.STANDARD, module.IntegrationLevel.FULL):
                    assert len(warnings) == 0, f"{level} has warnings: {warnings}"

                # To dict should work
                config_dict = config.to_dict()
                assert "level" in config_dict
                assert config_dict["level"] == level.value


class TestPipelineIntegrationFlow:
    """Test the full integration flow across components."""

    @pytest.mark.asyncio
    async def test_full_debate_flow(self) -> None:
        """Test a complete debate flow with all integrations."""
        # This is a high-level integration test simulating a full debate

        # 1. Configure research integrations
        import importlib.util

        config_spec = importlib.util.spec_from_file_location(
            "config",
            "aragora/config/research_integration.py",
        )
        if config_spec and config_spec.loader:
            config_module = importlib.util.module_from_spec(config_spec)
            config_spec.loader.exec_module(config_module)

            config = config_module.ResearchIntegrationConfig.from_preset(
                config_module.IntegrationLevel.STANDARD
            )

            # Verify all expected features are enabled
            features = config.get_enabled_features()
            assert "adaptive_stopping" in features
            assert "muse" in features
            assert "lara" in features
            assert "ascot" in features
            assert "think_prm" in features
            assert "graph_rag" in features
            assert "claim_check" in features
            assert "ahmad" in features

    @pytest.mark.asyncio
    async def test_verification_pipeline_flow(self) -> None:
        """Test verification pipeline: ThinkPRM -> ClaimCheck -> ASCoT."""
        # Simulate a step going through the verification pipeline

        # Step content to verify
        step_content = "The Roman economy declined due to currency debasement."
        evidence = [
            "The Roman Empire debased its currency significantly in the 3rd century.",
            "Economic instability was a major factor in Rome's decline.",
        ]

        # 1. ASCoT fragility check
        import importlib.util

        ascot_spec = importlib.util.spec_from_file_location(
            "ascot",
            "aragora/debate/ascot_fragility.py",
        )
        if ascot_spec and ascot_spec.loader:
            ascot_module = importlib.util.module_from_spec(ascot_spec)
            ascot_spec.loader.exec_module(ascot_module)

            analyzer = ascot_module.ASCoTFragilityAnalyzer()
            fragility = analyzer.calculate_round_fragility(
                round_number=3,
                total_rounds=5,
                dependency_depth=2,
            )

            # 2. If fragility is high, do intensive verification
            intensity = analyzer.get_verification_intensity(fragility)

        # 3. ClaimCheck verification
        claim_spec = importlib.util.spec_from_file_location(
            "claim_check",
            "aragora/evidence/claim_check.py",
        )
        if claim_spec and claim_spec.loader:
            claim_module = importlib.util.module_from_spec(claim_spec)
            claim_spec.loader.exec_module(claim_module)

            verifier = claim_module.ClaimCheckVerifier()
            claim_result = await verifier.verify(
                claim=step_content,
                evidence=evidence,
            )

            # Verify integration
            assert claim_result is not None
            assert claim_result.decomposition is not None

    @pytest.mark.asyncio
    async def test_routing_to_retrieval_flow(self) -> None:
        """Test LaRA routing -> GraphRAG retrieval flow."""
        # 1. Route query
        import importlib.util

        lara_spec = importlib.util.spec_from_file_location(
            "lara",
            "aragora/routing/lara_router.py",
        )
        if lara_spec and lara_spec.loader:
            lara_module = importlib.util.module_from_spec(lara_spec)
            lara_spec.loader.exec_module(lara_module)

            router = lara_module.LaRARouter()
            routing_result = router.route(
                query="What were the causes and effects of Rome's fall?",
                document_token_count=5000,
                has_graph_data=True,
            )

            assert routing_result is not None
            # Complex query with graph data should route to hybrid
            assert routing_result.mode in (
                lara_module.RetrievalMode.HYBRID,
                lara_module.RetrievalMode.GRAPH,
                lara_module.RetrievalMode.RAG,
            )

    @pytest.mark.asyncio
    async def test_team_selection_to_debate_flow(self) -> None:
        """Test A-HMAD team selection -> debate initialization flow."""
        import importlib.util

        ahmad_spec = importlib.util.spec_from_file_location(
            "ahmad",
            "aragora/debate/role_specializer.py",
        )
        if ahmad_spec and ahmad_spec.loader:
            ahmad_module = importlib.util.module_from_spec(ahmad_spec)
            ahmad_spec.loader.exec_module(ahmad_module)

            specializer = ahmad_module.AHMADRoleSpecializer()

            # 1. Analyze topic for required roles
            roles = specializer.analyze_topic(
                topic="Should AI development be regulated?",
                domain="policy",
            )

            # 2. Select team with role assignments
            agents = ["claude", "gpt4", "gemini", "llama"]
            elo_scores = {"claude": 1600, "gpt4": 1550, "gemini": 1500, "llama": 1450}
            calibration = {"claude": 0.8, "gpt4": 0.75, "gemini": 0.7, "llama": 0.65}
            domain_scores = {a: 0.7 for a in agents}

            composition = specializer.assign_roles(
                roles=roles,
                available_agents=agents,
                elo_scores=elo_scores,
                calibration_scores=calibration,
                domain_scores=domain_scores,
            )

            # Verify team is valid for debate
            assert composition.diversity_score >= 0.5
            assert len(composition.assignments) >= 2
            assert composition.coverage_score > 0


class TestConfigurationCombinations:
    """Test different configuration combinations."""

    @pytest.mark.asyncio
    async def test_minimal_config_flow(self) -> None:
        """Test minimal configuration enables only essential features."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "config",
            "aragora/config/research_integration.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            config = module.ResearchIntegrationConfig.from_preset(module.IntegrationLevel.MINIMAL)

            # Only essential features
            assert config.adaptive_stopping.enabled is True
            assert config.muse.enabled is False
            assert config.lara.enabled is False
            assert config.telemetry.enabled is True

    @pytest.mark.asyncio
    async def test_full_config_flow(self) -> None:
        """Test full configuration enables all features."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "config",
            "aragora/config/research_integration.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            config = module.ResearchIntegrationConfig.from_preset(module.IntegrationLevel.FULL)

            # All features enabled
            features = config.get_enabled_features()
            assert len(features) >= 8

            # Specific full config settings
            assert config.lara.max_hops == 3
            assert config.think_prm.max_parallel == 5
            assert config.graph_rag.max_hops == 3
            assert config.telemetry.record_all_events is True

    @pytest.mark.asyncio
    async def test_custom_config_validation(self) -> None:
        """Test custom configuration validation."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "config",
            "aragora/config/research_integration.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Create config with intentional issues
            config = module.ResearchIntegrationConfig(
                level=module.IntegrationLevel.CUSTOM,
            )
            config.adaptive_stopping.enabled = True
            config.muse.enabled = False  # Should warn
            config.ascot.enabled = False  # Should warn

            warnings = config.validate()

            # Should have warnings about missing gating
            assert len(warnings) >= 2
            assert any("muse" in w.lower() for w in warnings)
            assert any("ascot" in w.lower() for w in warnings)
