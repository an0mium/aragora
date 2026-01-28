"""
Matrix debates endpoint handlers.

Endpoints:
- POST /api/debates/matrix - Run parallel scenario debates
- GET /api/debates/matrix/{id} - Get matrix debate results
- GET /api/debates/matrix/{id}/scenarios - Get all scenario results
- GET /api/debates/matrix/{id}/conclusions - Get universal/conditional conclusions
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from typing import TypeAlias

    # Type alias for agent instances (from base.py)
    AgentInstance: TypeAlias = Any  # Could be APIAgent | CLIAgent


@runtime_checkable
class ScenarioConfigProtocol(Protocol):
    """Protocol for scenario configuration objects."""

    name: str
    parameters: dict[str, Any]
    constraints: list[str]
    is_baseline: bool


@runtime_checkable
class MatrixResultProtocol(Protocol):
    """Protocol for matrix debate result objects."""

    @property
    def scenario_results(self) -> list[Any]: ...

    @property
    def universal_conclusions(self) -> list[str]: ...

    @property
    def conditional_conclusions(self) -> dict[str, list[str]]: ...

    @property
    def comparison_matrix(self) -> dict[str, Any]: ...


@runtime_checkable
class MatrixRunnerProtocol(Protocol):
    """Protocol for matrix debate runner objects."""

    @property
    def scenarios(self) -> list[Any]: ...

    def add_scenario(self, config: Any) -> None: ...

    async def run_all(self, max_rounds: int = 3) -> MatrixResultProtocol: ...


from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    safe_error_message,
)
from ..secure import SecureHandler, ForbiddenError, UnauthorizedError
from ..utils.rate_limit import RateLimiter, get_client_ip
from aragora.resilience_patterns import with_timeout

logger = logging.getLogger(__name__)

# RBAC permissions for matrix debates
DEBATES_READ_PERMISSION = "debates:read"
DEBATES_CREATE_PERMISSION = "debates:create"

# Rate limiter for matrix debates (5 requests per minute - parallel debates are expensive)
_matrix_limiter = RateLimiter(requests_per_minute=5)


class MatrixDebatesHandler(SecureHandler):
    """Handler for matrix debate endpoints (parallel scenario exploration).

    RBAC Protected:
    - debates:read - required for GET endpoints
    - debates:create - required for POST endpoints
    """

    ROUTES = [
        "/api/v1/debates/matrix",
        "/api/v1/debates/matrix/",
    ]

    AUTH_REQUIRED_ENDPOINTS = [
        "/api/v1/debates/matrix",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/v1/debates/matrix")

    @handle_errors("matrix debates GET")
    async def handle_get(
        self, handler: Any, path: str, query_params: dict[str, Any]
    ) -> HandlerResult:
        """Handle GET requests for matrix debates with RBAC."""
        # RBAC: Require authentication and debates:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, DEBATES_READ_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(f"Matrix debates GET access denied: {e}")
            return error_response(str(e), 403)

        parts = path.rstrip("/").split("/")

        # GET /api/v1/debates/matrix/{id}
        # Path structure: ['', 'api', 'v1', 'debates', 'matrix', '{id}', ...]
        if len(parts) >= 6 and parts[4] == "matrix":
            matrix_id = parts[5]

            # GET /api/v1/debates/matrix/{id}/scenarios
            if len(parts) >= 7 and parts[6] == "scenarios":
                return await self._get_scenarios(handler, matrix_id)

            # GET /api/v1/debates/matrix/{id}/conclusions
            if len(parts) >= 7 and parts[6] == "conclusions":
                return await self._get_conclusions(handler, matrix_id)

            return await self._get_matrix_debate(handler, matrix_id)

        return error_response("Not found", 404)

    @handle_errors("matrix debates POST")
    async def handle_post(self, handler: Any, path: str, data: dict[str, Any]) -> HandlerResult:
        """Handle POST requests for matrix debates with RBAC.

        POST /api/debates/matrix - Run parallel scenario debates
        """
        if not path.rstrip("/").endswith("/debates/matrix"):
            return error_response("Not found", 404)

        # RBAC: Require authentication and debates:create permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, DEBATES_CREATE_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(f"Matrix debates POST access denied: {e}")
            return error_response(str(e), 403)

        # Rate limit check (5/min - expensive parallel operations)
        client_ip = get_client_ip(handler)
        if not _matrix_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for matrix debates: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        logger.debug("POST /api/debates/matrix - running matrix debate")
        return await self._run_matrix_debate(handler, data)

    @with_timeout(180.0)
    async def _run_matrix_debate(self, handler: Any, data: dict[str, Any]) -> HandlerResult:
        """Run parallel scenario debates.

        Request body:
            task: str - Base debate topic/question (10-5000 chars)
            agents: list[str] - Agent names to participate (2-10 agents)
            scenarios: list[dict] - List of scenario configurations (1-10 scenarios)
                - name: str - Scenario name (max 100 chars)
                - parameters: dict - Scenario-specific parameters
                - constraints: list[str] - Additional constraints
                - is_baseline: bool - Whether this is the baseline scenario
            max_rounds: int - Maximum rounds per scenario (1-10, default: 3)
        """
        # Validate task
        task = data.get("task")
        if not task:
            return error_response("task is required", 400)
        if not isinstance(task, str):
            return error_response("task must be a string", 400)
        task = task.strip()
        if len(task) < 10:
            return error_response("task must be at least 10 characters", 400)
        if len(task) > 5000:
            return error_response("task must be at most 5000 characters", 400)

        # Validate scenarios
        scenarios = data.get("scenarios", [])
        if not isinstance(scenarios, list):
            return error_response("scenarios must be an array", 400)
        if not scenarios:
            return error_response("At least one scenario is required", 400)
        if len(scenarios) > 10:
            return error_response("Maximum 10 scenarios allowed", 400)

        # Validate each scenario
        for i, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                return error_response(f"scenarios[{i}] must be an object", 400)
            name = scenario.get("name", "")
            if name and len(name) > 100:
                return error_response(f"scenarios[{i}].name too long (max 100 chars)", 400)
            if "parameters" in scenario and not isinstance(scenario["parameters"], dict):
                return error_response(f"scenarios[{i}].parameters must be an object", 400)
            if "constraints" in scenario:
                if not isinstance(scenario["constraints"], list):
                    return error_response(f"scenarios[{i}].constraints must be an array", 400)
                if len(scenario["constraints"]) > 10:
                    return error_response(f"scenarios[{i}].constraints too many (max 10)", 400)

        # Validate agents
        agent_names = data.get("agents", [])
        if not isinstance(agent_names, list):
            return error_response("agents must be an array", 400)
        if len(agent_names) > 10:
            return error_response("Maximum 10 agents allowed", 400)
        for i, name in enumerate(agent_names):
            if not isinstance(name, str):
                return error_response(f"agents[{i}] must be a string", 400)
            if len(name) > 50:
                return error_response(f"agents[{i}] name too long (max 50 chars)", 400)

        # Validate max_rounds
        max_rounds = data.get("max_rounds", 3)
        if not isinstance(max_rounds, int):
            try:
                max_rounds = int(max_rounds)
            except (ValueError, TypeError):
                return error_response("max_rounds must be an integer", 400)
        if max_rounds < 1:
            return error_response("max_rounds must be at least 1", 400)
        if max_rounds > 10:
            return error_response("max_rounds must be at most 10", 400)

        try:
            # Dynamic import of scenario module classes
            # These classes may have a different API than our Protocol definitions,
            # so we use cast() and handle ImportError gracefully with fallback
            from typing import cast

            scenarios_module = __import__(
                "aragora.debate.scenarios", fromlist=["MatrixDebateRunner", "ScenarioConfig"]
            )

            # Check if the expected API exists - if not, fall back to our implementation
            if not hasattr(scenarios_module, "ScenarioConfig") or not hasattr(
                scenarios_module, "MatrixDebateRunner"
            ):
                raise ImportError("Required scenario classes not found")

            ScenarioConfig = scenarios_module.ScenarioConfig
            MatrixDebateRunner = scenarios_module.MatrixDebateRunner

            # Load agents
            agents = await self._load_agents(agent_names)
            if not agents:
                return error_response("No valid agents found", 400)

            # Create matrix runner - cast to our Protocol for type checking
            runner = cast(
                MatrixRunnerProtocol,
                MatrixDebateRunner(
                    base_task=task,
                    agents=agents,
                ),
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
            return json_response(
                {
                    "matrix_id": matrix_id,
                    "task": task,
                    "scenario_count": len(results.scenario_results),
                    "results": [r.to_dict() for r in results.scenario_results],
                    "universal_conclusions": results.universal_conclusions,
                    "conditional_conclusions": results.conditional_conclusions,
                    "comparison_matrix": results.comparison_matrix,
                }
            )

        except ImportError as e:
            logger.warning(f"Matrix debate module not available, using fallback: {e}")
            return await self._run_matrix_debate_fallback(handler, data)
        except Exception as e:
            logger.exception(f"Matrix debate failed: {e}")
            return error_response(safe_error_message(e, "matrix debate"), 500)

    async def _run_matrix_debate_fallback(
        self, handler: Any, data: dict[str, Any]
    ) -> HandlerResult:
        """Fallback implementation using Arena directly for each scenario."""
        from aragora.core import DebateProtocol, Environment
        from aragora.debate.orchestrator import Arena

        task = data.get("task")
        scenarios = data.get("scenarios", [])
        agent_names = data.get("agents", [])
        max_rounds = data.get("max_rounds", 3)

        try:
            agents = await self._load_agents(agent_names)
            if not agents:
                return error_response("No valid agents found", 400)

            matrix_id = str(uuid.uuid4())
            all_conclusions: list[dict[str, Any]] = []

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
                protocol = DebateProtocol(
                    rounds=max_rounds,
                    convergence_detection=False,
                    early_stopping=False,
                )
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
            scenario_tasks = [run_scenario(s) for s in scenarios]
            gather_results = await asyncio.gather(*scenario_tasks, return_exceptions=True)

            # Process results
            valid_results: list[dict[str, Any]] = []
            for r in gather_results:
                if isinstance(r, BaseException):
                    logger.error(f"Scenario failed: {r}")
                else:
                    valid_results.append(r)
                    if r.get("final_answer"):
                        all_conclusions.append(
                            {
                                "scenario": r["scenario_name"],
                                "conclusion": r["final_answer"],
                                "confidence": r["confidence"],
                            }
                        )

            # Find universal conclusions (conclusions that appear in all scenarios)
            universal_conclusions = self._find_universal_conclusions(valid_results)

            # Find conditional conclusions (conclusions specific to scenarios)
            conditional_conclusions = self._find_conditional_conclusions(valid_results)

            return json_response(
                {
                    "matrix_id": matrix_id,
                    "task": task,
                    "scenario_count": len(valid_results),
                    "results": valid_results,
                    "universal_conclusions": universal_conclusions,
                    "conditional_conclusions": conditional_conclusions,
                    "comparison_matrix": self._build_comparison_matrix(valid_results),
                }
            )

        except Exception as e:
            logger.exception(f"Matrix debate fallback failed: {e}")
            return error_response(safe_error_message(e, "matrix debate"), 500)

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
                conditional.append(
                    {
                        "condition": f"When {r['scenario_name']}",
                        "parameters": r.get("parameters", {}),
                        "conclusion": r["final_answer"],
                        "confidence": r.get("confidence", 0),
                    }
                )
        return conditional

    def _build_comparison_matrix(self, results: list[dict]) -> dict:
        """Build a comparison matrix of scenarios."""
        return {
            "scenarios": [r["scenario_name"] for r in results],
            "consensus_rate": sum(1 for r in results if r.get("consensus_reached"))
            / max(len(results), 1),
            "avg_confidence": sum(r.get("confidence", 0) for r in results) / max(len(results), 1),
            "avg_rounds": sum(r.get("rounds_used", 0) for r in results) / max(len(results), 1),
        }

    async def _load_agents(self, agent_names: list[str]) -> list[Any]:
        """Load agents by name."""
        try:
            from typing import cast

            from aragora.agents.base import AgentType, create_agent

            names = agent_names or ["claude", "openai"]
            agents: list[Any] = []
            for name in names:
                try:
                    # Cast string to AgentType - create_agent will raise ValueError
                    # if the name is not a valid agent type
                    agent = create_agent(cast(AgentType, name))
                    agents.append(agent)
                except Exception as e:
                    logger.warning(f"Failed to create agent {name}: {e}")
            return agents
        except Exception as e:
            logger.warning(f"Failed to load agents: {e}")
            return []

    async def _get_matrix_debate(self, handler: Any, matrix_id: str) -> HandlerResult:
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

    async def _get_scenarios(self, handler: Any, matrix_id: str) -> HandlerResult:
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

    async def _get_conclusions(self, handler: Any, matrix_id: str) -> HandlerResult:
        """Get conclusions for a matrix debate."""
        storage = getattr(handler, "storage", None)
        if not storage:
            return error_response("Storage not configured", 503)

        try:
            conclusions = await storage.get_matrix_conclusions(matrix_id)
            return json_response(
                {
                    "matrix_id": matrix_id,
                    "universal_conclusions": conclusions.get("universal", []),
                    "conditional_conclusions": conclusions.get("conditional", []),
                }
            )
        except Exception as e:
            logger.error(f"Failed to get conclusions for {matrix_id}: {e}")
            return error_response("Failed to retrieve conclusions", 500)
