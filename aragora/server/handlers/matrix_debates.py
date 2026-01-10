"""
Matrix debates endpoint handlers.

Endpoints:
- POST /api/debates/matrix - Run parallel scenario debates
- GET /api/debates/matrix/{id} - Get matrix debate results
- GET /api/debates/matrix/{id}/scenarios - Get all scenario results
- GET /api/debates/matrix/{id}/conclusions - Get universal/conditional conclusions
"""

import asyncio
import logging
from typing import Any, Optional
import uuid

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
)

logger = logging.getLogger(__name__)


class MatrixDebatesHandler(BaseHandler):
    """Handler for matrix debate endpoints (parallel scenario exploration)."""

    ROUTES = [
        "/api/debates/matrix",
        "/api/debates/matrix/",
    ]

    AUTH_REQUIRED_ENDPOINTS = [
        "/api/debates/matrix",
    ]

    @handle_errors
    async def handle_get(self, handler, path: str, query_params: dict) -> HandlerResult:
        """Handle GET requests for matrix debates."""
        parts = path.rstrip("/").split("/")

        # GET /api/debates/matrix/{id}
        if len(parts) >= 5 and parts[3] == "matrix":
            matrix_id = parts[4]

            # GET /api/debates/matrix/{id}/scenarios
            if len(parts) >= 6 and parts[5] == "scenarios":
                return await self._get_scenarios(handler, matrix_id)

            # GET /api/debates/matrix/{id}/conclusions
            if len(parts) >= 6 and parts[5] == "conclusions":
                return await self._get_conclusions(handler, matrix_id)

            return await self._get_matrix_debate(handler, matrix_id)

        return error_response("Not found", 404)

    @handle_errors
    async def handle_post(self, handler, path: str, data: dict) -> HandlerResult:
        """Handle POST requests for matrix debates.

        POST /api/debates/matrix - Run parallel scenario debates
        """
        if not path.rstrip("/").endswith("/debates/matrix"):
            return error_response("Not found", 404)

        return await self._run_matrix_debate(handler, data)

    async def _run_matrix_debate(self, handler, data: dict) -> HandlerResult:
        """Run parallel scenario debates.

        Request body:
            task: str - Base debate topic/question
            agents: list[str] - Agent names to participate
            scenarios: list[dict] - List of scenario configurations
                - name: str - Scenario name
                - parameters: dict - Scenario-specific parameters
                - constraints: list[str] - Additional constraints
                - is_baseline: bool - Whether this is the baseline scenario
            max_rounds: int - Maximum rounds per scenario (default: 3)
        """
        task = data.get("task")
        if not task:
            return error_response("task is required", 400)

        scenarios = data.get("scenarios", [])
        if not scenarios:
            return error_response("At least one scenario is required", 400)

        agent_names = data.get("agents", [])
        max_rounds = data.get("max_rounds", 3)

        try:
            from aragora.debate.scenarios import MatrixDebateRunner, ScenarioConfig

            # Load agents
            agents = await self._load_agents(agent_names)
            if not agents:
                return error_response("No valid agents found", 400)

            # Create matrix runner
            runner = MatrixDebateRunner(
                base_task=task,
                agents=agents,
            )

            # Add scenarios
            for scenario_data in scenarios:
                config = ScenarioConfig(
                    name=scenario_data.get("name", f"Scenario {len(runner.scenarios) + 1}"),
                    parameters=scenario_data.get("parameters", {}),
                    constraints=scenario_data.get("constraints", []),
                    is_baseline=scenario_data.get("is_baseline", False),
                )
                runner.add_scenario(config)

            # Generate matrix ID
            matrix_id = str(uuid.uuid4())

            # Run all scenarios in parallel
            results = await runner.run_all(max_rounds=max_rounds)

            # Build response
            return json_response({
                "matrix_id": matrix_id,
                "task": task,
                "scenario_count": len(results.scenario_results),
                "results": [r.to_dict() for r in results.scenario_results],
                "universal_conclusions": results.universal_conclusions,
                "conditional_conclusions": results.conditional_conclusions,
                "comparison_matrix": results.comparison_matrix,
            })

        except ImportError as e:
            logger.warning(f"Matrix debate module not available, using fallback: {e}")
            return await self._run_matrix_debate_fallback(handler, data)
        except Exception as e:
            logger.exception(f"Matrix debate failed: {e}")
            return error_response(f"Matrix debate failed: {str(e)}", 500)

    async def _run_matrix_debate_fallback(self, handler, data: dict) -> HandlerResult:
        """Fallback implementation using Arena directly for each scenario."""
        from aragora.debate.orchestrator import Arena, ArenaConfig
        from aragora.core import Environment, DebateProtocol

        task = data.get("task")
        scenarios = data.get("scenarios", [])
        agent_names = data.get("agents", [])
        max_rounds = data.get("max_rounds", 3)

        try:
            agents = await self._load_agents(agent_names)
            if not agents:
                return error_response("No valid agents found", 400)

            matrix_id = str(uuid.uuid4())
            results = []
            all_conclusions = []

            # Run scenarios in parallel
            async def run_scenario(scenario_data: dict) -> dict:
                name = scenario_data.get("name", "Unnamed")
                parameters = scenario_data.get("parameters", {})
                constraints = scenario_data.get("constraints", [])

                # Build scenario task with parameters and constraints
                scenario_task = f"{task}"
                if parameters:
                    param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
                    scenario_task += f"\n\nParameters: {param_str}"
                if constraints:
                    scenario_task += f"\n\nConstraints: {', '.join(constraints)}"

                # Run debate
                env = Environment(task=scenario_task)
                protocol = DebateProtocol(rounds=max_rounds)
                arena = Arena(env, agents, protocol)

                result = await arena.run()

                return {
                    "scenario_name": name,
                    "parameters": parameters,
                    "constraints": constraints,
                    "is_baseline": scenario_data.get("is_baseline", False),
                    "winner": result.winner,
                    "final_answer": result.final_answer,
                    "confidence": result.confidence,
                    "consensus_reached": result.consensus_reached,
                    "rounds_used": result.rounds_used,
                }

            # Run all scenarios concurrently
            tasks = [run_scenario(s) for s in scenarios]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            valid_results = []
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Scenario failed: {r}")
                else:
                    valid_results.append(r)
                    if r.get("final_answer"):
                        all_conclusions.append({
                            "scenario": r["scenario_name"],
                            "conclusion": r["final_answer"],
                            "confidence": r["confidence"],
                        })

            # Find universal conclusions (conclusions that appear in all scenarios)
            universal_conclusions = self._find_universal_conclusions(valid_results)

            # Find conditional conclusions (conclusions specific to scenarios)
            conditional_conclusions = self._find_conditional_conclusions(valid_results)

            return json_response({
                "matrix_id": matrix_id,
                "task": task,
                "scenario_count": len(valid_results),
                "results": valid_results,
                "universal_conclusions": universal_conclusions,
                "conditional_conclusions": conditional_conclusions,
                "comparison_matrix": self._build_comparison_matrix(valid_results),
            })

        except Exception as e:
            logger.exception(f"Matrix debate fallback failed: {e}")
            return error_response(f"Matrix debate failed: {str(e)}", 500)

    def _find_universal_conclusions(self, results: list[dict]) -> list[str]:
        """Find conclusions that are consistent across all scenarios."""
        if not results:
            return []

        # Simple heuristic: if all scenarios reached consensus, that's universal
        consensus_results = [r for r in results if r.get("consensus_reached")]
        if len(consensus_results) == len(results):
            return ["All scenarios reached consensus"]

        return []

    def _find_conditional_conclusions(self, results: list[dict]) -> list[dict]:
        """Find conclusions that depend on specific scenarios."""
        conditional = []
        for r in results:
            if r.get("final_answer"):
                conditional.append({
                    "condition": f"When {r['scenario_name']}",
                    "parameters": r.get("parameters", {}),
                    "conclusion": r["final_answer"],
                    "confidence": r.get("confidence", 0),
                })
        return conditional

    def _build_comparison_matrix(self, results: list[dict]) -> dict:
        """Build a comparison matrix of scenarios."""
        return {
            "scenarios": [r["scenario_name"] for r in results],
            "consensus_rate": sum(1 for r in results if r.get("consensus_reached")) / max(len(results), 1),
            "avg_confidence": sum(r.get("confidence", 0) for r in results) / max(len(results), 1),
            "avg_rounds": sum(r.get("rounds_used", 0) for r in results) / max(len(results), 1),
        }

    async def _load_agents(self, agent_names: list[str]) -> list:
        """Load agents by name."""
        try:
            from aragora.agents import load_agents
            return load_agents(agent_names or ["claude", "gpt4"])
        except Exception as e:
            logger.warning(f"Failed to load agents: {e}")
            return []

    async def _get_matrix_debate(self, handler, matrix_id: str) -> HandlerResult:
        """Get a matrix debate by ID."""
        storage = getattr(handler, "storage", None)
        if not storage:
            return error_response("Storage not configured", 503)

        try:
            matrix = await storage.get_matrix_debate(matrix_id)
            if not matrix:
                return error_response("Matrix debate not found", 404)

            return json_response(matrix)
        except Exception as e:
            logger.error(f"Failed to get matrix debate {matrix_id}: {e}")
            return error_response("Failed to retrieve matrix debate", 500)

    async def _get_scenarios(self, handler, matrix_id: str) -> HandlerResult:
        """Get all scenario results for a matrix debate."""
        storage = getattr(handler, "storage", None)
        if not storage:
            return error_response("Storage not configured", 503)

        try:
            scenarios = await storage.get_matrix_scenarios(matrix_id)
            return json_response({"matrix_id": matrix_id, "scenarios": scenarios})
        except Exception as e:
            logger.error(f"Failed to get scenarios for {matrix_id}: {e}")
            return error_response("Failed to retrieve scenarios", 500)

    async def _get_conclusions(self, handler, matrix_id: str) -> HandlerResult:
        """Get conclusions for a matrix debate."""
        storage = getattr(handler, "storage", None)
        if not storage:
            return error_response("Storage not configured", 503)

        try:
            conclusions = await storage.get_matrix_conclusions(matrix_id)
            return json_response({
                "matrix_id": matrix_id,
                "universal_conclusions": conclusions.get("universal", []),
                "conditional_conclusions": conclusions.get("conditional", []),
            })
        except Exception as e:
            logger.error(f"Failed to get conclusions for {matrix_id}: {e}")
            return error_response("Failed to retrieve conclusions", 500)
