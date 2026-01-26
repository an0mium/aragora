"""
Tests for lazy subsystem initialization.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestLazySubsystem:
    """Tests for LazySubsystem descriptor."""

    def test_lazy_initialization_on_first_access(self):
        """Subsystem should be created on first access."""
        from aragora.debate.lazy_subsystems import LazySubsystem

        factory_mock = MagicMock(return_value="created_instance")

        class TestClass:
            subsystem = LazySubsystem(
                "_subsystem",
                factory=factory_mock,
            )

            def __init__(self):
                self._subsystem = None

        obj = TestClass()

        # Factory should not be called yet
        factory_mock.assert_not_called()

        # First access should trigger creation
        result = obj.subsystem
        factory_mock.assert_called_once_with(obj)
        assert result == "created_instance"

        # Second access should return cached value
        factory_mock.reset_mock()
        result2 = obj.subsystem
        factory_mock.assert_not_called()
        assert result2 == "created_instance"

    def test_lazy_initialization_with_condition_true(self):
        """Subsystem should be created when condition is True."""
        from aragora.debate.lazy_subsystems import LazySubsystem

        factory_mock = MagicMock(return_value="created")
        condition_mock = MagicMock(return_value=True)

        class TestClass:
            subsystem = LazySubsystem(
                "_subsystem",
                factory=factory_mock,
                condition=condition_mock,
            )

            def __init__(self):
                self._subsystem = None

        obj = TestClass()
        result = obj.subsystem

        condition_mock.assert_called_once_with(obj)
        factory_mock.assert_called_once_with(obj)
        assert result == "created"

    def test_lazy_initialization_with_condition_false(self):
        """Subsystem should not be created when condition is False."""
        from aragora.debate.lazy_subsystems import LazySubsystem

        factory_mock = MagicMock(return_value="created")
        condition_mock = MagicMock(return_value=False)

        class TestClass:
            subsystem = LazySubsystem(
                "_subsystem",
                factory=factory_mock,
                condition=condition_mock,
            )

            def __init__(self):
                self._subsystem = None

        obj = TestClass()
        result = obj.subsystem

        condition_mock.assert_called_once_with(obj)
        factory_mock.assert_not_called()
        assert result is None

    def test_lazy_initialization_with_on_create_callback(self):
        """on_create callback should be called after creation."""
        from aragora.debate.lazy_subsystems import LazySubsystem

        factory_mock = MagicMock(return_value="created")
        on_create_mock = MagicMock()

        class TestClass:
            subsystem = LazySubsystem(
                "_subsystem",
                factory=factory_mock,
                on_create=on_create_mock,
            )

            def __init__(self):
                self._subsystem = None

        obj = TestClass()
        result = obj.subsystem

        on_create_mock.assert_called_once_with(obj, "created")
        assert result == "created"

    def test_lazy_initialization_handles_factory_exception(self):
        """Factory exceptions should be logged and None returned."""
        from aragora.debate.lazy_subsystems import LazySubsystem

        factory_mock = MagicMock(side_effect=RuntimeError("Init failed"))

        class TestClass:
            subsystem = LazySubsystem(
                "_subsystem",
                factory=factory_mock,
            )

            def __init__(self):
                self._subsystem = None

        obj = TestClass()

        with patch("aragora.debate.lazy_subsystems.logger") as logger_mock:
            result = obj.subsystem

        assert result is None
        logger_mock.warning.assert_called_once()

    def test_lazy_set_value(self):
        """Setting value should work via descriptor."""
        from aragora.debate.lazy_subsystems import LazySubsystem

        class TestClass:
            subsystem = LazySubsystem(
                "_subsystem",
                factory=lambda self: "default",
            )

            def __init__(self):
                self._subsystem = None

        obj = TestClass()
        obj.subsystem = "custom_value"

        # Should return the set value, not call factory
        assert obj.subsystem == "custom_value"


class TestLazyFactories:
    """Tests for lazy factory functions."""

    def test_create_lazy_checkpoint_manager_disabled(self):
        """Should return None when checkpointing disabled."""
        from aragora.debate.lazy_subsystems import create_lazy_checkpoint_manager

        arena_mock = MagicMock()
        arena_mock.protocol.enable_checkpointing = False

        result = create_lazy_checkpoint_manager(arena_mock)
        assert result is None

    def test_create_lazy_knowledge_mound_disabled(self):
        """Should return None when knowledge features disabled."""
        from aragora.debate.lazy_subsystems import create_lazy_knowledge_mound

        arena_mock = MagicMock()
        arena_mock.enable_knowledge_retrieval = False
        arena_mock.enable_knowledge_ingestion = False

        result = create_lazy_knowledge_mound(arena_mock)
        assert result is None

    def test_create_lazy_population_manager_disabled(self):
        """Should return None when auto_evolve disabled."""
        from aragora.debate.lazy_subsystems import create_lazy_population_manager

        arena_mock = MagicMock()
        arena_mock.auto_evolve = False

        result = create_lazy_population_manager(arena_mock)
        assert result is None

    def test_create_lazy_prompt_evolver_disabled(self):
        """Should return None when prompt evolution disabled."""
        from aragora.debate.lazy_subsystems import create_lazy_prompt_evolver

        arena_mock = MagicMock()
        arena_mock.protocol.enable_prompt_evolution = False

        result = create_lazy_prompt_evolver(arena_mock)
        assert result is None

    def test_create_lazy_cross_debate_memory_disabled(self):
        """Should return None when cross-debate memory disabled."""
        from aragora.debate.lazy_subsystems import create_lazy_cross_debate_memory

        arena_mock = MagicMock()
        arena_mock.enable_cross_debate_memory = False

        result = create_lazy_cross_debate_memory(arena_mock)
        assert result is None
