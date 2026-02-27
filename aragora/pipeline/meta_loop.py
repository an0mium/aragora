"""Meta-Loop Trigger."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MetaLoopConfig:
    objective: str = "Improve Aragora's self-improvement system"
    target_modules: list[str] = field(default_factory=list)
    max_files_changed: int = 10
    require_test_pass: bool = True
    require_human_approval: bool = True
    cycle_count: int = 0
    cooldown_cycles: int = 10

    @property
    def is_eligible(self) -> bool:
        return self.cycle_count == 0 or self.cooldown_cycles <= 0


@dataclass
class MetaLoopTarget:
    module: str
    description: str
    priority: float = 0.5
    risk: str = "low"
    estimated_files: int = 1
    rationale: str = ""


@dataclass
class MetaLoopResult:
    targets_identified: list[MetaLoopTarget] = field(default_factory=list)
    targets_executed: list[MetaLoopTarget] = field(default_factory=list)
    targets_skipped: list[MetaLoopTarget] = field(default_factory=list)
    total_files_changed: int = 0
    tests_passed: bool = False
    quality_before: float = 0.0
    quality_after: float = 0.0
    approved: bool = False

    @property
    def quality_delta(self) -> float:
        return self.quality_after - self.quality_before

    @property
    def improved(self) -> bool:
        return self.quality_after > self.quality_before


class MetaLoopTrigger:
    def __init__(
        self,
        config: MetaLoopConfig | None = None,
        quality_scorer: Callable[[], float] | None = None,
        knowledge_mound: Any | None = None,
    ) -> None:
        self._config = config or MetaLoopConfig()
        self._quality_scorer = quality_scorer
        self._km = knowledge_mound
        self._history: list[MetaLoopResult] = []
        self._cycle_counter = 0

    def should_trigger(self) -> bool:
        if self._cycle_counter < self._config.cooldown_cycles:
            return False
        if self._quality_scorer is not None:
            current_quality = self._quality_scorer()
            if current_quality > 0.9:
                return False
        return True

    def increment_cycle(self) -> None:
        self._cycle_counter += 1

    def identify_targets(
        self, pipeline_outcomes: list[dict[str, Any]] | None = None
    ) -> list[MetaLoopTarget]:
        targets: list[MetaLoopTarget] = []
        if not pipeline_outcomes:
            targets.append(
                MetaLoopTarget(
                    module="aragora/interrogation",
                    description="Improve question quality and relevance",
                    priority=0.7,
                    risk="low",
                    estimated_files=2,
                    rationale="Interrogation quality directly impacts spec quality",
                )
            )
            return targets
        failure_modules: dict[str, int] = {}
        for outcome in pipeline_outcomes:
            if not outcome.get("execution_succeeded", True):
                module = outcome.get("failed_module", "unknown")
                failure_modules[module] = failure_modules.get(module, 0) + 1
        for module, count in sorted(failure_modules.items(), key=lambda x: x[1], reverse=True)[:3]:
            targets.append(
                MetaLoopTarget(
                    module=module,
                    description=f"Fix recurring failures ({count} occurrences)",
                    priority=min(1.0, count / 5),
                    risk="medium" if count > 3 else "low",
                    estimated_files=2,
                    rationale=f"Failed {count} times in recent pipeline runs",
                )
            )
        return targets

    def execute(
        self,
        targets: list[MetaLoopTarget],
        executor: Callable[[MetaLoopTarget], bool] | None = None,
    ) -> MetaLoopResult:
        result = MetaLoopResult()
        result.targets_identified = list(targets)
        if self._quality_scorer is not None:
            result.quality_before = self._quality_scorer()
        safe_targets = sorted(targets, key=lambda t: t.priority, reverse=True)
        if self._config.require_human_approval:
            result.targets_skipped = safe_targets
            result.approved = False
            self._history.append(result)
            return result
        files_budget = self._config.max_files_changed
        for target in safe_targets:
            if target.estimated_files > files_budget:
                result.targets_skipped.append(target)
                continue
            if target.risk == "high":
                result.targets_skipped.append(target)
                continue
            if executor is not None:
                success = executor(target)
                if success:
                    result.targets_executed.append(target)
                    files_budget -= target.estimated_files
                    result.total_files_changed += target.estimated_files
                else:
                    result.targets_skipped.append(target)
            else:
                result.targets_skipped.append(target)
        if self._quality_scorer is not None:
            result.quality_after = self._quality_scorer()
        result.approved = True
        self._cycle_counter = 0
        self._history.append(result)
        if self._km is not None:
            try:
                self._km.ingest(
                    {
                        "type": "meta_loop_result",
                        "targets_executed": len(result.targets_executed),
                        "targets_skipped": len(result.targets_skipped),
                        "quality_delta": result.quality_delta,
                        "improved": result.improved,
                    }
                )
            except Exception:
                logger.warning("Failed to record meta-loop result to KM")
        return result

    def get_history(self, limit: int = 10) -> list[MetaLoopResult]:
        return self._history[-limit:]
