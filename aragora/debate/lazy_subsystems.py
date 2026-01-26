"""
Lazy subsystem initialization for Arena.

Provides decorators and helpers for deferring expensive subsystem
initialization until first access. This reduces Arena startup time
and memory usage when subsystems aren't actually used.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LazySubsystem(Generic[T]):
    """Descriptor for lazy subsystem initialization.

    Usage:
        class MyClass:
            expensive_subsystem = LazySubsystem(
                "_expensive_subsystem",
                factory=lambda self: ExpensiveClass(),
                condition=lambda self: self.enable_expensive,
            )

    The subsystem is only created when first accessed, and only if
    the condition (if provided) returns True.
    """

    def __init__(
        self,
        private_attr: str,
        factory: Callable[["Arena"], T],
        condition: Optional[Callable[["Arena"], bool]] = None,
        on_create: Optional[Callable[["Arena", T], None]] = None,
    ):
        """Initialize lazy subsystem descriptor.

        Args:
            private_attr: Name of private attribute to store instance
            factory: Callable that creates the subsystem instance
            condition: Optional condition that must be True to create
            on_create: Optional callback after creation
        """
        self.private_attr = private_attr
        self.factory = factory
        self.condition = condition
        self.on_create = on_create

    def __get__(self, obj: Optional["Arena"], objtype: type = None) -> Optional[T]:
        if obj is None:
            return self  # type: ignore

        # Check if already initialized
        cached = getattr(obj, self.private_attr, None)
        if cached is not None:
            return cached

        # Check condition
        if self.condition and not self.condition(obj):
            return None

        # Create and cache
        try:
            instance = self.factory(obj)
            setattr(obj, self.private_attr, instance)
            logger.debug(f"[lazy] Initialized {self.private_attr}")

            if self.on_create and instance is not None:
                self.on_create(obj, instance)

            return instance
        except Exception as e:
            logger.warning(f"[lazy] Failed to initialize {self.private_attr}: {e}")
            # Cache None to prevent repeated attempts
            setattr(obj, self.private_attr, None)
            return None

    def __set__(self, obj: "Arena", value: Optional[T]) -> None:
        setattr(obj, self.private_attr, value)


def lazy_property(
    condition: Optional[Callable[["Arena"], bool]] = None,
    on_create: Optional[Callable[["Arena", Any], None]] = None,
):
    """Decorator for lazy property initialization.

    Usage:
        @lazy_property(condition=lambda self: self.enable_feature)
        def my_subsystem(self):
            return ExpensiveClass()

    Args:
        condition: Optional condition that must be True to create
        on_create: Optional callback after creation
    """

    def decorator(func: Callable[["Arena"], T]) -> property:
        private_attr = f"_lazy_{func.__name__}"

        def wrapper(self: "Arena") -> Optional[T]:
            # Check cache
            cached = getattr(self, private_attr, None)
            if cached is not None:
                return cached

            # Check condition
            if condition and not condition(self):
                return None

            # Create and cache
            try:
                instance = func(self)
                setattr(self, private_attr, instance)
                logger.debug(f"[lazy] Initialized {func.__name__}")

                if on_create and instance is not None:
                    on_create(self, instance)

                return instance
            except Exception as e:
                logger.warning(f"[lazy] Failed to initialize {func.__name__}: {e}")
                setattr(self, private_attr, None)
                return None

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        return property(wrapper)

    return decorator


def create_lazy_checkpoint_manager(arena: "Arena"):
    """Factory for lazy checkpoint manager creation."""
    if not arena.protocol.enable_checkpointing:
        return None

    try:
        from aragora.debate.checkpoint import CheckpointManager, DatabaseCheckpointStore

        manager = CheckpointManager(store=DatabaseCheckpointStore())
        logger.debug("[lazy] Auto-created CheckpointManager with database store")
        return manager
    except ImportError:
        logger.warning("[lazy] CheckpointManager not available")
        return None


def create_lazy_knowledge_mound(arena: "Arena"):
    """Factory for lazy knowledge mound creation."""
    if not (arena.enable_knowledge_retrieval or arena.enable_knowledge_ingestion):
        return None

    try:
        from aragora.knowledge.mound import get_knowledge_mound

        workspace_id = getattr(arena, "org_id", None) or "default"
        mound = get_knowledge_mound(
            workspace_id=workspace_id,
            auto_initialize=True,
        )
        logger.info(
            f"[lazy] Auto-created KM instance for debate "
            f"(retrieval={arena.enable_knowledge_retrieval}, "
            f"ingestion={arena.enable_knowledge_ingestion})"
        )
        return mound
    except ImportError as e:
        logger.warning(f"[lazy] Could not create KnowledgeMound: {e}")
        return None
    except (RuntimeError, ConnectionError, OSError) as e:
        logger.warning(f"[lazy] KM infrastructure error: {e}")
        return None
    except Exception as e:
        logger.exception(f"[lazy] Unexpected KM error: {e}")
        return None


def create_lazy_population_manager(arena: "Arena"):
    """Factory for lazy population manager creation."""
    if not arena.auto_evolve:
        return None

    try:
        from aragora.genesis.breeding import PopulationManager

        manager = PopulationManager()
        logger.info("[lazy] population_manager auto-initialized for genome evolution")
        return manager
    except ImportError:
        logger.warning("[lazy] PopulationManager not available")
        return None
    except Exception as e:
        logger.warning(f"[lazy] Failed to initialize PopulationManager: {e}")
        return None


def create_lazy_prompt_evolver(arena: "Arena"):
    """Factory for lazy prompt evolver creation."""
    if not arena.protocol.enable_prompt_evolution:
        return None

    try:
        from aragora.evolution.evolver import PromptEvolver

        evolver = PromptEvolver()
        logger.debug("[lazy] Auto-created PromptEvolver for pattern extraction")
        return evolver
    except ImportError:
        logger.warning("[lazy] PromptEvolver not available")
        return None


def create_lazy_cross_debate_memory(arena: "Arena"):
    """Factory for lazy cross-debate memory creation."""
    if not getattr(arena, "enable_cross_debate_memory", True):
        return None

    try:
        from aragora.memory.cross_debate import CrossDebateMemory

        memory = CrossDebateMemory()
        logger.debug("[lazy] Auto-created CrossDebateMemory")
        return memory
    except ImportError:
        logger.warning("[lazy] CrossDebateMemory not available")
        return None
    except Exception as e:
        logger.warning(f"[lazy] Failed to initialize CrossDebateMemory: {e}")
        return None
