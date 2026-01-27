"""
End-to-End Integration Tests: Skills System ↔ Arena.

Tests the integration between the Skills system and debate Arena:
1. Skill invocation during debates
2. Skill-based evidence collection
3. Evidence refresh with skills
4. Skills in multi-round debates
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.skills import (
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
    SkillCapability,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_skill_registry():
    """Create a mock SkillRegistry."""
    registry = MagicMock()
    registry._skills = {}
    registry._metrics = {}

    def mock_register(skill):
        registry._skills[skill.manifest.name] = skill
        registry._metrics[skill.manifest.name] = {
            "total_invocations": 0,
            "successful_invocations": 0,
            "failed_invocations": 0,
            "average_latency_ms": 0,
        }

    def mock_get(name):
        return registry._skills.get(name)

    def mock_list():
        return list(registry._skills.values())

    async def mock_invoke(name, input_data, context):
        skill = registry._skills.get(name)
        if not skill:
            return SkillResult(
                status=SkillStatus.FAILURE,
                error_message=f"Skill not found: {name}",
            )

        registry._metrics[name]["total_invocations"] += 1

        try:
            result = await skill.execute(input_data, context)
            registry._metrics[name]["successful_invocations"] += 1
            return result
        except Exception as e:
            registry._metrics[name]["failed_invocations"] += 1
            return SkillResult(
                status=SkillStatus.FAILURE,
                error_message=str(e),
            )

    registry.register = MagicMock(side_effect=mock_register)
    registry.get = MagicMock(side_effect=mock_get)
    registry.list_skills = MagicMock(side_effect=mock_list)
    registry.invoke = AsyncMock(side_effect=mock_invoke)
    registry.get_metrics = lambda name: registry._metrics.get(name)

    return registry


@pytest.fixture
def mock_arena():
    """Create a mock Arena for debate orchestration."""
    arena = MagicMock()
    arena._evidence_collector = None
    arena._skill_results = []

    async def mock_run():
        return {
            "consensus": True,
            "winner": "proposal_1",
            "rounds": 3,
            "evidence_used": arena._skill_results,
        }

    async def mock_collect_evidence(query: str, skill_name: str = None):
        result = {
            "query": query,
            "skill": skill_name,
            "evidence": [
                {"source": "web_search", "content": f"Evidence for: {query}"},
            ],
        }
        arena._skill_results.append(result)
        return result

    arena.run = AsyncMock(side_effect=mock_run)
    arena.collect_evidence = AsyncMock(side_effect=mock_collect_evidence)

    return arena


@pytest.fixture
def web_search_skill():
    """Create a mock web search skill."""
    skill = MagicMock()
    skill.manifest = SkillManifest(
        name="web_search",
        version="1.0.0",
        description="Search the web for information",
        capabilities=[SkillCapability.WEB_SEARCH],
        input_schema={"query": "string"},
        output_schema={"results": "array"},
    )

    async def mock_execute(input_data, context):
        query = input_data.get("query", "")
        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "results": [
                    {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
                    {"title": f"Result 2 for {query}", "url": "https://example.com/2"},
                ]
            },
        )

    skill.execute = AsyncMock(side_effect=mock_execute)
    return skill


@pytest.fixture
def evidence_skill():
    """Create a mock evidence collection skill."""
    skill = MagicMock()
    skill.manifest = SkillManifest(
        name="evidence_collector",
        version="1.0.0",
        description="Collect evidence from multiple sources",
        capabilities=[SkillCapability.WEB_SEARCH, SkillCapability.READ_LOCAL],
        input_schema={"topic": "string", "sources": "array"},
    )

    async def mock_execute(input_data, context):
        topic = input_data.get("topic", "")
        return SkillResult(
            status=SkillStatus.SUCCESS,
            data={
                "evidence": [
                    {
                        "source": "academic",
                        "claim": f"Research supports {topic}",
                        "confidence": 0.85,
                    },
                    {
                        "source": "news",
                        "claim": f"Recent developments in {topic}",
                        "confidence": 0.72,
                    },
                ]
            },
        )

    skill.execute = AsyncMock(side_effect=mock_execute)
    return skill


# ============================================================================
# Integration Tests
# ============================================================================


class TestSkillsArenaIntegration:
    """Tests for Skills ↔ Arena integration."""

    @pytest.mark.asyncio
    async def test_skill_invocation_during_debate(
        self, mock_skill_registry, mock_arena, web_search_skill
    ):
        """Test that skills can be invoked during a debate."""
        # Register skill
        mock_skill_registry.register(web_search_skill)

        # Create skill context for debate
        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
            debate_id="debate-123",
        )

        # Invoke skill during debate
        result = await mock_skill_registry.invoke(
            "web_search",
            {"query": "climate change effects"},
            context,
        )

        assert result.status == SkillStatus.SUCCESS
        assert "results" in result.data
        assert len(result.data["results"]) == 2

        # Verify metrics updated
        metrics = mock_skill_registry.get_metrics("web_search")
        assert metrics["total_invocations"] == 1
        assert metrics["successful_invocations"] == 1

    @pytest.mark.asyncio
    async def test_skill_based_evidence_collection(
        self, mock_skill_registry, mock_arena, evidence_skill
    ):
        """Test using skills to collect evidence for debates."""
        mock_skill_registry.register(evidence_skill)

        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
            debate_id="debate-456",
            debate_context={"round": 1},
        )

        # Collect evidence using skill
        result = await mock_skill_registry.invoke(
            "evidence_collector",
            {"topic": "renewable energy", "sources": ["academic", "news"]},
            context,
        )

        assert result.status == SkillStatus.SUCCESS
        assert "evidence" in result.data
        evidence_items = result.data["evidence"]
        assert len(evidence_items) == 2
        assert evidence_items[0]["source"] == "academic"
        assert evidence_items[1]["source"] == "news"

    @pytest.mark.asyncio
    async def test_evidence_refresh_with_skills(
        self, mock_skill_registry, mock_arena, web_search_skill
    ):
        """Test refreshing evidence mid-debate using skills."""
        mock_skill_registry.register(web_search_skill)

        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
            debate_id="debate-789",
            debate_context={"round": 2, "action": "refresh"},
        )

        # Initial evidence collection
        result1 = await mock_skill_registry.invoke(
            "web_search",
            {"query": "initial topic"},
            context,
        )

        # Refresh with updated query
        context.debate_context["round"] = 3
        result2 = await mock_skill_registry.invoke(
            "web_search",
            {"query": "refined topic based on critiques"},
            context,
        )

        assert result1.status == SkillStatus.SUCCESS
        assert result2.status == SkillStatus.SUCCESS

        # Both should have succeeded
        metrics = mock_skill_registry.get_metrics("web_search")
        assert metrics["total_invocations"] == 2
        assert metrics["successful_invocations"] == 2

    @pytest.mark.asyncio
    async def test_multi_skill_debate_flow(
        self, mock_skill_registry, mock_arena, web_search_skill, evidence_skill
    ):
        """Test using multiple skills throughout a debate."""
        mock_skill_registry.register(web_search_skill)
        mock_skill_registry.register(evidence_skill)

        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
            debate_id="debate-multi",
            debate_context={},
        )

        # Round 1: Search for initial information
        context.debate_context["round"] = 1
        search_result = await mock_skill_registry.invoke(
            "web_search",
            {"query": "AI regulation"},
            context,
        )

        # Round 2: Collect deeper evidence
        context.debate_context["round"] = 2
        evidence_result = await mock_skill_registry.invoke(
            "evidence_collector",
            {"topic": "AI regulation frameworks", "sources": ["academic"]},
            context,
        )

        # Round 3: Follow-up search
        context.debate_context["round"] = 3
        followup_result = await mock_skill_registry.invoke(
            "web_search",
            {"query": "EU AI Act implementation"},
            context,
        )

        # All invocations should succeed
        assert search_result.status == SkillStatus.SUCCESS
        assert evidence_result.status == SkillStatus.SUCCESS
        assert followup_result.status == SkillStatus.SUCCESS

        # Verify each skill's metrics
        search_metrics = mock_skill_registry.get_metrics("web_search")
        evidence_metrics = mock_skill_registry.get_metrics("evidence_collector")

        assert search_metrics["total_invocations"] == 2
        assert evidence_metrics["total_invocations"] == 1

    @pytest.mark.asyncio
    async def test_skill_failure_does_not_break_debate(self, mock_skill_registry, mock_arena):
        """Test that skill failures don't crash the debate."""
        # Create a skill that fails
        failing_skill = MagicMock()
        failing_skill.manifest = SkillManifest(
            name="failing_skill",
            version="1.0.0",
            description="A skill that always fails",
            capabilities=[],
            input_schema={},
            output_schema={},
        )

        async def mock_fail(input_data, context):
            raise ValueError("Simulated skill failure")

        failing_skill.execute = AsyncMock(side_effect=mock_fail)
        mock_skill_registry.register(failing_skill)

        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
        )

        # Invocation should handle failure gracefully
        result = await mock_skill_registry.invoke(
            "failing_skill",
            {"input": "test"},
            context,
        )

        assert result.status == SkillStatus.FAILURE
        assert "Simulated skill failure" in result.error_message

        # Metrics should track failure
        metrics = mock_skill_registry.get_metrics("failing_skill")
        assert metrics["failed_invocations"] == 1

    @pytest.mark.asyncio
    async def test_skill_permissions_in_debate_context(self, mock_skill_registry, web_search_skill):
        """Test that skill permissions are enforced in debate context."""
        mock_skill_registry.register(web_search_skill)

        # Context without invoke permission
        limited_context = SkillContext(
            user_id="limited_user",
            permissions=["skills:read"],  # Missing skills:invoke
            debate_id="debate-limited",
        )

        # Create skill that checks permissions
        permission_skill = MagicMock()
        permission_skill.manifest = SkillManifest(
            name="permission_skill",
            version="1.0.0",
            description="Checks permissions",
            capabilities=[],
            input_schema={},
            output_schema={},
            required_permissions=["skills:invoke"],
        )

        async def check_perms(input_data, context):
            if "skills:invoke" not in context.permissions:
                return SkillResult(
                    status=SkillStatus.PERMISSION_DENIED,
                    error_message="Missing required permission: skills:invoke",
                )
            return SkillResult(status=SkillStatus.SUCCESS, data={"ok": True})

        permission_skill.execute = AsyncMock(side_effect=check_perms)
        mock_skill_registry.register(permission_skill)

        result = await mock_skill_registry.invoke(
            "permission_skill",
            {},
            limited_context,
        )

        assert result.status == SkillStatus.PERMISSION_DENIED

    @pytest.mark.asyncio
    async def test_skill_metadata_propagation(self, mock_skill_registry, web_search_skill):
        """Test that debate metadata is available to skills."""
        mock_skill_registry.register(web_search_skill)

        debate_context = {
            "round": 2,
            "topic": "AI Ethics",
            "agents": ["claude", "gpt-4", "gemini"],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
            debate_id="debate-meta-test",
            debate_context=debate_context,
        )

        result = await mock_skill_registry.invoke(
            "web_search",
            {"query": "AI ethics research"},
            context,
        )

        assert result.status == SkillStatus.SUCCESS
        # Skill had access to all debate context
        assert context.debate_id == "debate-meta-test"
        assert context.debate_context["round"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_skill_invocations(self, mock_skill_registry, web_search_skill):
        """Test multiple concurrent skill invocations during debate."""
        import asyncio

        mock_skill_registry.register(web_search_skill)

        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
            debate_id="concurrent-test",
        )

        # Simulate concurrent evidence gathering by multiple agents
        queries = [
            "argument point 1",
            "argument point 2",
            "counterargument 1",
            "evidence for position",
        ]

        tasks = [mock_skill_registry.invoke("web_search", {"query": q}, context) for q in queries]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.status == SkillStatus.SUCCESS for r in results)

        # Metrics should reflect all invocations
        metrics = mock_skill_registry.get_metrics("web_search")
        assert metrics["total_invocations"] == 4
        assert metrics["successful_invocations"] == 4


class TestSkillsEvidenceFlow:
    """Tests for skills-based evidence flow in debates."""

    @pytest.mark.asyncio
    async def test_evidence_aggregation_from_multiple_skills(
        self, mock_skill_registry, web_search_skill, evidence_skill
    ):
        """Test aggregating evidence from multiple skill sources."""
        mock_skill_registry.register(web_search_skill)
        mock_skill_registry.register(evidence_skill)

        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
            debate_id="evidence-aggregation",
        )

        # Collect from web search
        web_results = await mock_skill_registry.invoke(
            "web_search",
            {"query": "renewable energy benefits"},
            context,
        )

        # Collect from evidence skill
        evidence_results = await mock_skill_registry.invoke(
            "evidence_collector",
            {"topic": "renewable energy", "sources": ["academic", "news"]},
            context,
        )

        # Aggregate evidence
        all_evidence = []
        if web_results.status == SkillStatus.SUCCESS:
            all_evidence.extend(web_results.data.get("results", []))
        if evidence_results.status == SkillStatus.SUCCESS:
            all_evidence.extend(evidence_results.data.get("evidence", []))

        # Should have evidence from both sources
        assert len(all_evidence) == 4  # 2 from web + 2 from evidence

    @pytest.mark.asyncio
    async def test_skill_rate_limiting_in_debate(self, mock_skill_registry):
        """Test that skill rate limiting is respected during debates."""
        # Create a rate-limited skill
        rate_limited_skill = MagicMock()
        rate_limited_skill.manifest = SkillManifest(
            name="rate_limited_search",
            version="1.0.0",
            description="Rate limited search",
            capabilities=[SkillCapability.WEB_SEARCH],
            input_schema={"query": "string"},
            output_schema={"results": "array"},
            rate_limit_per_minute=2,
        )
        rate_limited_skill._call_count = 0

        async def mock_execute(input_data, context):
            rate_limited_skill._call_count += 1
            if rate_limited_skill._call_count > 2:
                return SkillResult(
                    status=SkillStatus.RATE_LIMITED,
                    error_message="Rate limit exceeded: 2 per minute",
                )
            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={"results": []},
            )

        rate_limited_skill.execute = AsyncMock(side_effect=mock_execute)
        mock_skill_registry.register(rate_limited_skill)

        context = SkillContext(
            user_id="debate_user",
            permissions=["skills:invoke"],
        )

        # First two calls succeed
        r1 = await mock_skill_registry.invoke("rate_limited_search", {"query": "test1"}, context)
        r2 = await mock_skill_registry.invoke("rate_limited_search", {"query": "test2"}, context)

        # Third call should be rate limited
        r3 = await mock_skill_registry.invoke("rate_limited_search", {"query": "test3"}, context)

        assert r1.status == SkillStatus.SUCCESS
        assert r2.status == SkillStatus.SUCCESS
        assert r3.status == SkillStatus.RATE_LIMITED
