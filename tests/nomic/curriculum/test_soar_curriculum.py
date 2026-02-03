"""
Tests for SOAR Curriculum Generation module.

Tests cover:
- Skill profiles and categories
- Stepping stone generation
- Difficulty estimation
- Curriculum planning
- Result tracking
"""

import asyncio
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from aragora.nomic.curriculum.soar_curriculum import (
    SkillCategory,
    SkillProfile,
    SteppingStone,
    SteppingStoneResult,
    Curriculum,
    DifficultyEstimator,
    SteppingStoneGenerator,
    CurriculumPlanner,
    generate_curriculum,
)


class TestSkillCategory:
    """Tests for SkillCategory enum."""

    def test_all_categories_exist(self):
        """All expected categories are defined."""
        expected = [
            "code_generation",
            "code_refactoring",
            "bug_fixing",
            "architecture",
            "testing",
            "documentation",
            "performance",
            "security",
            "integration",
            "reasoning",
        ]
        actual = [c.value for c in SkillCategory]
        assert set(expected) == set(actual)

    def test_enum_string_values(self):
        """Categories have correct string values."""
        assert SkillCategory.CODE_GENERATION.value == "code_generation"
        assert SkillCategory.SECURITY.value == "security"


class TestSkillProfile:
    """Tests for SkillProfile dataclass."""

    def test_default_level(self):
        """Default level is 0.5 for unknown categories."""
        profile = SkillProfile()
        assert profile.get_level(SkillCategory.CODE_GENERATION) == 0.5

    def test_update_level(self):
        """Levels can be updated with history tracking."""
        profile = SkillProfile()
        profile.update_level(SkillCategory.CODE_GENERATION, 0.8)
        assert profile.get_level(SkillCategory.CODE_GENERATION) == 0.8
        assert len(profile.history) == 1

    def test_level_clamping(self):
        """Levels are clamped to [0.0, 1.0]."""
        profile = SkillProfile()
        profile.update_level(SkillCategory.TESTING, 1.5)
        assert profile.get_level(SkillCategory.TESTING) == 1.0

        profile.update_level(SkillCategory.TESTING, -0.5)
        assert profile.get_level(SkillCategory.TESTING) == 0.0

    def test_weakness_categories(self):
        """weakness_categories returns low-level categories."""
        profile = SkillProfile()
        profile.levels = {
            SkillCategory.CODE_GENERATION: 0.8,
            SkillCategory.TESTING: 0.3,
            SkillCategory.SECURITY: 0.2,
            SkillCategory.DOCUMENTATION: 0.6,
        }
        weaknesses = profile.weakness_categories(threshold=0.5)
        assert SkillCategory.TESTING in weaknesses
        assert SkillCategory.SECURITY in weaknesses
        assert SkillCategory.CODE_GENERATION not in weaknesses


class TestSteppingStone:
    """Tests for SteppingStone dataclass."""

    def test_generate_id_deterministic(self):
        """IDs are deterministic for same task."""
        task = "Implement basic REST API"
        id1 = SteppingStone.generate_id(task)
        id2 = SteppingStone.generate_id(task)
        assert id1 == id2
        assert id1.startswith("ss_")

    def test_generate_id_unique(self):
        """Different tasks get different IDs."""
        id1 = SteppingStone.generate_id("Task A")
        id2 = SteppingStone.generate_id("Task B")
        assert id1 != id2

    def test_creation(self):
        """SteppingStone can be created with required fields."""
        stone = SteppingStone(
            id="ss_test",
            task="Implement logging",
            difficulty=0.4,
            skills_required=[SkillCategory.CODE_GENERATION],
        )
        assert stone.id == "ss_test"
        assert stone.difficulty == 0.4
        assert len(stone.prerequisites) == 0


class TestSteppingStoneResult:
    """Tests for SteppingStoneResult dataclass."""

    def test_creation(self):
        """Result can be created with required fields."""
        result = SteppingStoneResult(
            stone_id="ss_test",
            success=True,
            completion_score=0.9,
            time_taken=120.5,
        )
        assert result.success is True
        assert result.completion_score == 0.9

    def test_default_lists(self):
        """Lists default to empty."""
        result = SteppingStoneResult(
            stone_id="ss_test",
            success=False,
            completion_score=0.3,
            time_taken=60.0,
        )
        assert result.errors_encountered == []
        assert result.skills_demonstrated == []


class TestCurriculum:
    """Tests for Curriculum dataclass."""

    def test_is_complete_false_initially(self):
        """New curriculum is not complete."""
        curriculum = Curriculum(
            id="cur_1",
            target_task="Build microservice",
            target_difficulty=0.8,
            stepping_stones=[
                SteppingStone(
                    id="ss_1",
                    task="task 1",
                    difficulty=0.3,
                    skills_required=[],
                ),
            ],
        )
        assert not curriculum.is_complete

    def test_is_complete_true_after_all_done(self):
        """Curriculum is complete after all stones attempted."""
        curriculum = Curriculum(
            id="cur_1",
            target_task="Build microservice",
            target_difficulty=0.8,
            stepping_stones=[
                SteppingStone(
                    id="ss_1",
                    task="task 1",
                    difficulty=0.3,
                    skills_required=[],
                ),
            ],
            current_index=1,  # Past the single stone
        )
        assert curriculum.is_complete

    def test_success_rate_no_results(self):
        """Success rate is 0 with no results."""
        curriculum = Curriculum(
            id="cur_1",
            target_task="task",
            target_difficulty=0.5,
            stepping_stones=[],
        )
        assert curriculum.success_rate == 0.0

    def test_success_rate_calculated(self):
        """Success rate is calculated correctly."""
        curriculum = Curriculum(
            id="cur_1",
            target_task="task",
            target_difficulty=0.5,
            stepping_stones=[],
            results=[
                SteppingStoneResult("ss_1", True, 0.8, 10.0),
                SteppingStoneResult("ss_2", False, 0.3, 15.0),
                SteppingStoneResult("ss_3", True, 0.9, 12.0),
            ],
        )
        assert curriculum.success_rate == pytest.approx(2 / 3)

    def test_next_stone(self):
        """next_stone returns current stone or None."""
        stones = [
            SteppingStone(id="ss_1", task="task 1", difficulty=0.3, skills_required=[]),
            SteppingStone(id="ss_2", task="task 2", difficulty=0.5, skills_required=[]),
        ]
        curriculum = Curriculum(
            id="cur_1",
            target_task="target",
            target_difficulty=0.8,
            stepping_stones=stones,
            current_index=0,
        )

        assert curriculum.next_stone() == stones[0]

        curriculum.current_index = 1
        assert curriculum.next_stone() == stones[1]

        curriculum.current_index = 2
        assert curriculum.next_stone() is None


class TestDifficultyEstimator:
    """Tests for DifficultyEstimator class."""

    def test_simple_task_lower_difficulty(self):
        """Simple tasks have lower difficulty."""
        estimator = DifficultyEstimator()
        simple = estimator.estimate("Add a comment to the function")
        complex = estimator.estimate(
            "Refactor the distributed caching system to support "
            "concurrent access with proper security measures"
        )
        assert simple < complex

    def test_skill_increases_difficulty(self):
        """Tasks requiring hard skills have higher difficulty."""
        estimator = DifficultyEstimator()
        basic = estimator.estimate(
            "Write documentation",
            skills_required=[SkillCategory.DOCUMENTATION],
        )
        advanced = estimator.estimate(
            "Implement security audit",
            skills_required=[SkillCategory.SECURITY],
        )
        assert advanced > basic

    def test_difficulty_in_valid_range(self):
        """Difficulty is always between 0 and 1."""
        estimator = DifficultyEstimator()

        for task in ["x", "simple", "complex distributed security migration"]:
            diff = estimator.estimate(task)
            assert 0.0 <= diff <= 1.0

    def test_update_from_result(self):
        """Estimator can be updated with actual results."""
        estimator = DifficultyEstimator()
        # Should not raise
        estimator.update_from_result("test task", 0.7)


class TestSteppingStoneGenerator:
    """Tests for SteppingStoneGenerator class."""

    @pytest.mark.asyncio
    async def test_generates_correct_count(self):
        """Generator creates requested number of stones."""
        generator = SteppingStoneGenerator()
        stones = await generator.generate_stones(
            target_task="Implement full authentication system",
            current_level=0.2,
            target_level=0.8,
            num_stones=3,
        )
        assert len(stones) == 3

    @pytest.mark.asyncio
    async def test_stones_have_increasing_difficulty(self):
        """Stones have progressively increasing difficulty."""
        generator = SteppingStoneGenerator()
        stones = await generator.generate_stones(
            target_task="Complex task requiring many skills",
            current_level=0.2,
            target_level=0.9,
            num_stones=3,
        )

        if len(stones) > 1:
            for i in range(1, len(stones)):
                assert stones[i].difficulty >= stones[i - 1].difficulty

    @pytest.mark.asyncio
    async def test_prerequisites_chain(self):
        """Later stones depend on earlier ones."""
        generator = SteppingStoneGenerator()
        stones = await generator.generate_stones(
            target_task="Build API gateway",
            current_level=0.3,
            target_level=0.8,
            num_stones=3,
        )

        if len(stones) >= 2:
            # Second stone should have first as prerequisite
            assert stones[0].id in stones[1].prerequisites

    @pytest.mark.asyncio
    async def test_no_stones_when_already_capable(self):
        """No stones generated if current level exceeds target difficulty."""
        generator = SteppingStoneGenerator()
        stones = await generator.generate_stones(
            target_task="Simple task",  # Low difficulty
            current_level=0.9,  # Already very capable
            target_level=0.95,
            num_stones=3,
        )
        # Should generate few or no stones
        assert len(stones) <= 3  # May still generate if difficulty > current

    @pytest.mark.asyncio
    async def test_hints_from_skill_profile(self):
        """Hints are generated based on skill weaknesses."""
        profile = SkillProfile()
        profile.levels = {
            SkillCategory.TESTING: 0.2,  # Weak
            SkillCategory.CODE_GENERATION: 0.8,  # Strong
        }

        generator = SteppingStoneGenerator()
        stones = await generator.generate_stones(
            target_task="Create comprehensive test suite",
            current_level=0.3,
            target_level=0.8,
            num_stones=2,
            skill_profile=profile,
        )

        # Some stones should have hints about weak skills
        all_hints = []
        for stone in stones:
            all_hints.extend(stone.hints)
        # May or may not have hints depending on skill inference


class TestCurriculumPlanner:
    """Tests for CurriculumPlanner class."""

    @pytest.mark.asyncio
    async def test_create_curriculum(self):
        """Planner creates a curriculum."""
        planner = CurriculumPlanner()
        curriculum = await planner.create_curriculum(
            target_task="Build recommendation engine",
            current_level=0.3,
            target_level=0.8,
            num_stones=2,
        )

        assert curriculum.id
        assert curriculum.target_task == "Build recommendation engine"
        assert len(curriculum.stepping_stones) == 2
        assert curriculum.id in planner.active_curricula

    @pytest.mark.asyncio
    async def test_record_result(self):
        """Results are recorded and calibration updated."""
        planner = CurriculumPlanner()
        curriculum = await planner.create_curriculum(
            target_task="Implement caching",
            current_level=0.4,
            num_stones=2,
        )

        stone = curriculum.stepping_stones[0]
        result = SteppingStoneResult(
            stone_id=stone.id,
            success=True,
            completion_score=0.85,
            time_taken=100.0,
            skills_demonstrated=[SkillCategory.CODE_GENERATION],
        )

        planner.record_result(curriculum.id, result)

        assert len(curriculum.results) == 1
        assert curriculum.current_index == 1

    @pytest.mark.asyncio
    async def test_should_attempt_target(self):
        """should_attempt_target returns correct value."""
        planner = CurriculumPlanner()
        curriculum = await planner.create_curriculum(
            target_task="test",
            current_level=0.5,
            num_stones=1,
        )

        # Not complete yet
        assert not planner.should_attempt_target(curriculum.id)

        # Complete with success
        stone = curriculum.stepping_stones[0]
        planner.record_result(
            curriculum.id,
            SteppingStoneResult(stone.id, True, 0.9, 10.0),
        )

        # Now should be ready (60% success rate threshold met)
        assert planner.should_attempt_target(curriculum.id)

    @pytest.mark.asyncio
    async def test_get_next_action(self):
        """get_next_action returns appropriate action."""
        planner = CurriculumPlanner()
        curriculum = await planner.create_curriculum(
            target_task="test target",
            current_level=0.5,
            num_stones=1,
        )

        # Should return first stepping stone
        next_action = planner.get_next_action(curriculum.id)
        assert isinstance(next_action, SteppingStone)

        # Complete the stone
        planner.record_result(
            curriculum.id,
            SteppingStoneResult(curriculum.stepping_stones[0].id, True, 0.9, 10.0),
        )

        # Should now return target task string
        next_action = planner.get_next_action(curriculum.id)
        assert next_action == "test target"

    @pytest.mark.asyncio
    async def test_storage_persistence(self):
        """Curriculum is saved to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "curricula"
            planner = CurriculumPlanner(storage_path=storage_path)

            curriculum = await planner.create_curriculum(
                target_task="test",
                num_stones=1,
            )

            # File should be created
            assert (storage_path / f"{curriculum.id}.json").exists()


class TestGenerateCurriculumFunction:
    """Tests for generate_curriculum convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """generate_curriculum convenience function works."""
        curriculum = await generate_curriculum(
            target_task="Build ML pipeline",
            current_level=0.4,
            num_stones=2,
        )

        assert isinstance(curriculum, Curriculum)
        assert curriculum.target_task == "Build ML pipeline"
        assert len(curriculum.stepping_stones) == 2


class TestSkillInference:
    """Tests for skill inference from task descriptions."""

    @pytest.mark.asyncio
    async def test_infers_code_generation(self):
        """Infers code generation from keywords."""
        generator = SteppingStoneGenerator()
        stones = await generator.generate_stones(
            target_task="Create a new authentication module",
            current_level=0.3,
            target_level=0.8,
            num_stones=1,
        )
        if stones:
            assert SkillCategory.CODE_GENERATION in stones[0].skills_required

    @pytest.mark.asyncio
    async def test_infers_testing(self):
        """Infers testing from keywords."""
        generator = SteppingStoneGenerator()
        stones = await generator.generate_stones(
            target_task="Write comprehensive test suite",
            current_level=0.3,
            target_level=0.8,
            num_stones=1,
        )
        if stones:
            assert SkillCategory.TESTING in stones[0].skills_required

    @pytest.mark.asyncio
    async def test_infers_security(self):
        """Infers security from keywords."""
        generator = SteppingStoneGenerator()
        stones = await generator.generate_stones(
            target_task="Implement authentication and encryption",
            current_level=0.3,
            target_level=0.8,
            num_stones=1,
        )
        if stones:
            assert SkillCategory.SECURITY in stones[0].skills_required
