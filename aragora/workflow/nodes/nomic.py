"""
Nomic Loop Step for executing self-improvement cycles within workflows.

Wraps the NomicStateMachine and phase implementations to enable:
- Self-improvement as a workflow step
- Phase-selective execution
- Integration with workflow checkpointing
- Human approval gates for code changes
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


class NomicLoopStep(BaseStep):
    """
    Workflow step that executes Nomic Loop phases.

    Wraps the NomicStateMachine for self-improvement cycles within workflows.
    Can run a subset of phases or the full cycle.

    Config options:
        cycles: int - Number of nomic cycles to run (default: 1)
        phases: List[str] - Phases to execute (default: all 6)
            Available: context, debate, design, implement, verify, commit
        workspace_id: str - Workspace for knowledge storage
        enable_code_execution: bool - Allow code changes (default: False)
        require_approval: bool - Require human approval for changes (default: True)
        checkpoint_dir: str - Directory for nomic checkpoints
        timeout_seconds: float - Timeout per phase (default: 300)
        recovery_enabled: bool - Enable automatic recovery (default: True)
        max_retries: int - Maximum retries per phase (default: 3)
        agents: List[str] - Agent types for debate (default: ["claude", "gpt4"])

    Usage:
        step = NomicLoopStep(
            name="Self-Improvement Cycle",
            config={
                "cycles": 1,
                "phases": ["context", "debate", "design"],
                "workspace_id": "my_workspace",
                "enable_code_execution": False,
            }
        )
        result = await step.execute(context)
    """

    # Available phases in order
    ALL_PHASES = ["context", "debate", "design", "implement", "verify", "commit"]

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._state_machine = None
        self._current_phase_idx = 0
        self._cycle_count = 0

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the nomic loop step."""
        config = {**self._config, **context.current_step_config}

        # Extract configuration
        cycles = config.get("cycles", 1)
        phases = config.get("phases", self.ALL_PHASES)
        workspace_id = config.get("workspace_id", context.get_input("workspace_id", "default"))
        enable_code_execution = config.get("enable_code_execution", False)
        require_approval = config.get("require_approval", True)
        timeout_seconds = config.get("timeout_seconds", 300.0)
        recovery_enabled = config.get("recovery_enabled", True)
        max_retries = config.get("max_retries", 3)
        agents = config.get("agents", ["claude", "gpt4"])

        # Validate phases
        phases = [p for p in phases if p in self.ALL_PHASES]
        if not phases:
            logger.warning("No valid phases specified, using all phases")
            phases = self.ALL_PHASES

        logger.info(
            f"NomicLoopStep '{self.name}' starting: "
            f"cycles={cycles}, phases={phases}, workspace={workspace_id}"
        )

        try:
            from aragora.nomic import (  # noqa: F401
                NomicStateMachine,
                NomicState,
                create_nomic_state_machine,
                CheckpointManager,
                RecoveryManager,
            )
            from aragora.nomic.phases import (
                ContextPhase,
                DebatePhase,
                DesignPhase,
                ImplementPhase,
                VerifyPhase,
                CommitPhase,
                DebateConfig,
                DesignConfig,
            )

            # Map phase names to state machine states

            # Build phase instances
            phase_instances: Dict[str, Any] = {}

            if "context" in phases:
                phase_instances["context"] = ContextPhase(  # type: ignore[call-arg]
                    workspace_id=workspace_id,
                    timeout_seconds=timeout_seconds,
                )

            if "debate" in phases:
                debate_config = DebateConfig(  # type: ignore[call-arg]
                    agents=agents,
                    rounds=config.get("debate_rounds", 3),
                    consensus_mechanism=config.get("consensus_mechanism", "weighted"),
                )
                phase_instances["debate"] = DebatePhase(  # type: ignore[call-arg]
                    config=debate_config,
                    timeout_seconds=timeout_seconds,
                )

            if "design" in phases:
                design_config = DesignConfig(  # type: ignore[call-arg]
                    require_approval=require_approval,
                    max_scope=config.get("max_scope", 3),
                )
                phase_instances["design"] = DesignPhase(  # type: ignore[call-arg]
                    config=design_config,
                    timeout_seconds=timeout_seconds,
                )

            if "implement" in phases:
                phase_instances["implement"] = ImplementPhase(  # type: ignore[call-arg]
                    enable_code_execution=enable_code_execution,
                    require_approval=require_approval,
                    timeout_seconds=timeout_seconds,
                )

            if "verify" in phases:
                phase_instances["verify"] = VerifyPhase(  # type: ignore[call-arg]
                    timeout_seconds=timeout_seconds,
                )

            if "commit" in phases:
                phase_instances["commit"] = CommitPhase(  # type: ignore[call-arg]
                    require_approval=require_approval,
                    timeout_seconds=timeout_seconds,
                )

            # Execute cycles
            results: List[Dict[str, Any]] = []
            for cycle in range(cycles):
                self._cycle_count = cycle + 1
                logger.info(f"Starting nomic cycle {cycle + 1}/{cycles}")

                cycle_result: Dict[str, Any] = {
                    "cycle": cycle + 1,
                    "phases": {},
                    "success": True,
                }

                # Execute each phase in order
                for phase_name in phases:
                    if phase_name not in phase_instances:
                        continue

                    self._current_phase_idx = self.ALL_PHASES.index(phase_name)
                    phase = phase_instances[phase_name]

                    logger.info(f"Executing phase: {phase_name}")

                    try:
                        # Create phase context from workflow context
                        phase_context = self._build_phase_context(context, cycle_result)

                        # Execute the phase
                        phase_result = await phase.execute(phase_context)  # type: ignore[call-arg]

                        cycle_result["phases"][phase_name] = {
                            "success": True,
                            "result": phase_result,
                        }

                        # Store phase output in workflow context for next phase
                        context.set_state(f"nomic_{phase_name}_result", phase_result)

                    except Exception as e:
                        logger.error(f"Phase {phase_name} failed: {e}")
                        cycle_result["phases"][phase_name] = {
                            "success": False,
                            "error": str(e),
                        }
                        cycle_result["success"] = False

                        # Check if we should continue or abort
                        if not recovery_enabled:
                            raise

                        # Try recovery if enabled
                        retries = 0
                        while retries < max_retries:
                            retries += 1
                            logger.info(
                                f"Retrying phase {phase_name} (attempt {retries}/{max_retries})"
                            )
                            try:
                                phase_result = await phase.execute(phase_context)  # type: ignore[call-arg]
                                cycle_result["phases"][phase_name] = {
                                    "success": True,
                                    "result": phase_result,
                                    "retries": retries,
                                }
                                cycle_result["success"] = True
                                break
                            except Exception as retry_error:
                                logger.error(f"Retry {retries} failed: {retry_error}")

                        if not cycle_result["phases"][phase_name].get("success"):
                            # All retries exhausted
                            break

                results.append(cycle_result)

                # Check if cycle failed and we should stop
                if not cycle_result["success"] and not config.get("continue_on_failure", False):
                    break

            # Aggregate results
            return {
                "cycles_completed": len(results),
                "cycles_requested": cycles,
                "all_successful": all(r["success"] for r in results),
                "results": results,
                "phases_executed": phases,
                "workspace_id": workspace_id,
            }

        except ImportError as e:
            logger.error(f"Failed to import nomic module: {e}")
            return {
                "success": False,
                "error": f"Nomic module not available: {e}",
            }

    def _build_phase_context(
        self,
        workflow_context: WorkflowContext,
        cycle_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build context dict for phase execution."""
        return {
            "workflow_id": workflow_context.workflow_id,
            "workflow_inputs": workflow_context.inputs,
            "workflow_state": workflow_context.state,
            "previous_phases": cycle_result.get("phases", {}),
            "cycle_number": cycle_result.get("cycle", 1),
        }

    async def checkpoint(self) -> Dict[str, Any]:
        """Save nomic step state for checkpointing."""
        return {
            "current_phase_idx": self._current_phase_idx,
            "cycle_count": self._cycle_count,
        }

    async def restore(self, state: Dict[str, Any]) -> None:
        """Restore nomic step state from checkpoint."""
        self._current_phase_idx = state.get("current_phase_idx", 0)
        self._cycle_count = state.get("cycle_count", 0)

    def validate_config(self) -> bool:
        """Validate nomic step configuration."""
        phases = self._config.get("phases", self.ALL_PHASES)
        if phases:
            invalid_phases = [p for p in phases if p not in self.ALL_PHASES]
            if invalid_phases:
                logger.warning(f"Invalid phases in config: {invalid_phases}")
                return False
        return True
