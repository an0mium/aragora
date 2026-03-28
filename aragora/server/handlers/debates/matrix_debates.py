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

from aragora.config import DEFAULT_ROUNDS
from aragora.debate.scenarios import score_result_candidate, select_best_result
from aragora.server.versioning.compat import strip_version_prefix

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

    async def run_all(self, max_rounds: int = DEFAULT_ROUNDS) -> MatrixResultProtocol: ...


from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    safe_error_message,
)
from ..openapi_decorator import api_endpoint
from ..secure import SecureHandler, ForbiddenError, UnauthorizedError
from ..utils.rate_limit import RateLimiter, get_client_ip
from aragora.resilience import with_timeout

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

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/debates/matrix",
        "/api/v1/debates/matrix/",
        "/api/v1/matrix-debates",
        "/api/v1/matrix-debates/",
        "/api/v1/matrix-debates/*",
    ]

    AUTH_REQUIRED_ENDPOINTS = [
        "/api/v1/debates/matrix",
        "/api/v1/matrix-debates",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = strip_version_prefix(path)
        return normalized.startswith("/api/debates/matrix") or normalized.startswith(
            "/api/matrix-debates"
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/matrix/{matrix_id}",
        summary="Get matrix debate",
        description="Get the results of a matrix debate with parallel scenario exploration.",
        tags=["Debates", "Matrix Debates"],
        parameters=[
            {"name": "matrix_id", "in": "path", "required": True, "schema": {"type": "string"}}
        ],
        responses={
            "200": {"description": "Matrix debate results"},
            "401": {"description": "Authentication required"},
            "403": {"description": "Permission denied"},
            "404": {"description": "Matrix debate not found"},
        },
        operation_id="get_matrix_debate",
    )
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
            logger.warning("Matrix debates GET access denied: %s", e)
            return error_response("Permission denied", 403)

        normalized = strip_version_prefix(path)
        if normalized.startswith("/api/matrix-debates"):
            normalized = normalized.replace("/api/matrix-debates", "/api/debates/matrix", 1)
        segments = normalized.strip("/").split("/")

        # GET /api/debates/matrix/{id}
        # Path structure: ['api', 'debates', 'matrix', '{id}', ...]
        if len(segments) >= 4 and segments[2] == "matrix":
            matrix_id = segments[3]

            # GET /api/debates/matrix/{id}/scenarios
            if len(segments) >= 5 and segments[4] == "scenarios":
                return await self._get_scenarios(handler, matrix_id)

            # GET /api/debates/matrix/{id}/conclusions
            if len(segments) >= 5 and segments[4] == "conclusions":
                return await self._get_conclusions(handler, matrix_id)

            return await self._get_matrix_debate(handler, matrix_id)

        return error_response("Not found", 404)

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/matrix",
        summary="Create matrix debate",
        description="Run parallel scenario debates to explore a topic under different conditions.",
        tags=["Debates", "Matrix Debates"],
        operation_id="create_matrix_debate",
        responses={
            "200": {"description": "Matrix debate created and executed"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Authentication required"},
            "403": {"description": "Permission denied"},
            "429": {"description": "Rate limit exceeded"},
            "500": {"description": "Matrix debate failed"},
        },
    )
    @handle_errors("matrix debates POST")
    async def handle_post(self, *args: Any, **kwargs: Any) -> HandlerResult:
        """Handle POST requests for matrix debates with RBAC.

        POST /api/debates/matrix - Run parallel scenario debates
        """
        handler = None
        path = ""
        data: dict[str, Any] = {}

        if len(args) >= 3:
            if isinstance(args[0], str):
                path = args[0]
                handler = args[2]
                data, error = self.read_json_body_validated(handler)
                if error:
                    return error
            else:
                handler = args[0]
                path = args[1]
                data = args[2] or {}
        else:
            handler = kwargs.get("handler")
            path = kwargs.get("path", "")
            data = kwargs.get("data") or kwargs.get("body") or {}
            if handler is None:
                return error_response("Invalid request", 400)
            if not data:
                data, error = self.read_json_body_validated(handler)
                if error:
                    return error

        normalized = strip_version_prefix(path)
        if normalized.startswith("/api/matrix-debates"):
            normalized = normalized.replace("/api/matrix-debates", "/api/debates/matrix", 1)
        if not normalized.rstrip("/").endswith("/debates/matrix"):
            return error_response("Not found", 404)

        # RBAC: Require authentication and debates:create permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, DEBATES_CREATE_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning("Matrix debates POST access denied: %s", e)
            return error_response("Permission denied", 403)

        # Rate limit check (5/min - expensive parallel operations)
        client_ip = get_client_ip(handler)
        if not _matrix_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for matrix debates: %s", client_ip)
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
                - agents: list[str] - Optional scenario-specific agent override
            model_combinations: list[dict|list[str]] - Optional agent sets to expand across
                - name: str - Combination label
                - agents: list[str] - Agent names for this combination
                - is_baseline: bool - Whether this combination is the baseline
            max_rounds: int - Maximum rounds per scenario (1-10, default: global debate default)
        """
        # Validate task (accept "question" as alias for frontend compatibility)
        task = data.get("task") or data.get("question")
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
            if "agents" in scenario:
                agent_error = self._validate_agent_names(
                    scenario["agents"],
                    f"scenarios[{i}].agents",
                    allow_empty=False,
                )
                if agent_error:
                    return error_response(agent_error, 400)

        # Validate optional model combinations
        raw_model_combinations = data.get("model_combinations", [])
        if not isinstance(raw_model_combinations, list):
            return error_response("model_combinations must be an array", 400)
        if len(raw_model_combinations) > 10:
            return error_response("Maximum 10 model_combinations allowed", 400)
        model_combinations, model_combo_error = self._normalize_model_combinations(
            raw_model_combinations
        )
        if model_combo_error:
            return error_response(model_combo_error, 400)

        if not scenarios and not model_combinations:
            return error_response("At least one scenario or model combination is required", 400)

        # Validate agents
        agent_names = data.get("agents", [])
        agent_error = self._validate_agent_names(agent_names, "agents")
        if agent_error:
            return error_response(agent_error, 400)

        # Validate max_rounds
        max_rounds = data.get("max_rounds", DEFAULT_ROUNDS)
        if not isinstance(max_rounds, int):
            try:
                max_rounds = int(max_rounds)
            except (ValueError, TypeError):
                return error_response("max_rounds must be an integer", 400)
        if max_rounds < 1:
            return error_response("max_rounds must be at least 1", 400)
        if max_rounds > 10:
            return error_response("max_rounds must be at most 10", 400)

        scenario_variants = self._build_scenario_variants(scenarios, model_combinations)
        if len(scenario_variants) > 25:
            return error_response("Maximum 25 expanded scenario runs allowed", 400)

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
            for scenario_data in scenario_variants:
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
            result_payloads = [
                r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results.scenario_results
            ]
            best_result = self._build_best_result_payload(result_payloads)
            comparison_matrix = results.comparison_matrix
            if isinstance(comparison_matrix, dict):
                comparison_matrix = dict(comparison_matrix)
                comparison_matrix["best_result"] = best_result
                comparison_matrix["model_combination_count"] = len(model_combinations)

            # Build response
            return json_response(
                {
                    "matrix_id": matrix_id,
                    "task": task,
                    "scenario_count": len(results.scenario_results),
                    "results": result_payloads,
                    "universal_conclusions": results.universal_conclusions,
                    "conditional_conclusions": results.conditional_conclusions,
                    "comparison_matrix": comparison_matrix,
                    "best_result": best_result,
                    "model_combination_count": len(model_combinations),
                }
            )

        except ImportError as e:
            logger.warning("Matrix debate module not available, using fallback: %s", e)
            return await self._run_matrix_debate_fallback(handler, data)
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError) as e:
            logger.exception("Matrix debate failed: %s", e)
            return error_response(safe_error_message(e, "matrix debate"), 500)

    async def _run_matrix_debate_fallback(
        self, handler: Any, data: dict[str, Any]
    ) -> HandlerResult:
        """Fallback implementation using Arena directly for each scenario."""
        from aragora.core import DebateProtocol, Environment
        from aragora.debate.orchestrator import Arena

        task = data.get("task") or data.get("question")
        scenarios = self._build_scenario_variants(
            data.get("scenarios", []),
            data.get("model_combinations", []),
        )
        agent_names = data.get("agents", [])
        max_rounds = data.get("max_rounds", DEFAULT_ROUNDS)

        try:
            for scenario_data in scenarios:
                scenario_agent_names = scenario_data.get("agents", agent_names)
                loaded_agents = await self._load_agents(scenario_agent_names)
                if not loaded_agents:
                    combo_name = scenario_data.get("model_combination_name")
                    if combo_name:
                        return error_response(
                            f"No valid agents found for model combination '{combo_name}'",
                            400,
                        )
                    return error_response("No valid agents found", 400)

            matrix_id = str(uuid.uuid4())
            all_conclusions: list[dict[str, Any]] = []
            ctx = getattr(self, "ctx", {}) or {}
            document_store = ctx.get("document_store")
            evidence_store = ctx.get("evidence_store")

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

                scenario_agent_names = scenario_data.get("agents", agent_names)
                agents = await self._load_agents(scenario_agent_names)

                # Run debate
                env = Environment(task=scenario_task)
                protocol = DebateProtocol(
                    rounds=max_rounds,
                    convergence_detection=False,
                    early_stopping=False,
                )
                arena = Arena(
                    env,
                    agents,
                    protocol,
                    document_store=document_store,
                    evidence_store=evidence_store,
                )

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
                    "agents": scenario_agent_names,
                    "model_combination_name": scenario_data.get("model_combination_name"),
                    "source_scenario_name": scenario_data.get("source_scenario_name", name),
                }

            # Run all scenarios concurrently
            scenario_tasks = [run_scenario(s) for s in scenarios]
            gather_results = await asyncio.gather(*scenario_tasks, return_exceptions=True)

            # Process results
            valid_results: list[dict[str, Any]] = []
            for r in gather_results:
                if isinstance(r, BaseException):
                    logger.error("Scenario failed: %s", r)
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
            best_result = self._build_best_result_payload(valid_results)
            comparison_matrix = self._build_comparison_matrix(valid_results)
            comparison_matrix["best_result"] = best_result
            comparison_matrix["model_combination_count"] = len(data.get("model_combinations", []))

            return json_response(
                {
                    "matrix_id": matrix_id,
                    "task": task,
                    "scenario_count": len(valid_results),
                    "results": valid_results,
                    "universal_conclusions": universal_conclusions,
                    "conditional_conclusions": conditional_conclusions,
                    "comparison_matrix": comparison_matrix,
                    "best_result": best_result,
                    "model_combination_count": len(data.get("model_combinations", [])),
                }
            )

        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError) as e:
            logger.exception("Matrix debate fallback failed: %s", e)
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
                        "condition": self._format_condition_label(r),
                        "parameters": r.get("parameters", {}),
                        "conclusion": r["final_answer"],
                        "confidence": r.get("confidence", 0),
                    }
                )
        return conditional

    def _build_comparison_matrix(self, results: list[dict]) -> dict:
        """Build a comparison matrix of scenarios."""
        ranked_results = [
            {
                "scenario_name": r["scenario_name"],
                "selection_score": score_result_candidate(r),
                "confidence": r.get("confidence", 0),
                "consensus_reached": r.get("consensus_reached", False),
            }
            for r in sorted(results, key=score_result_candidate, reverse=True)
        ]
        best_result = self._build_best_result_payload(results)
        return {
            "scenarios": [r["scenario_name"] for r in results],
            "consensus_rate": sum(1 for r in results if r.get("consensus_reached"))
            / max(len(results), 1),
            "avg_confidence": sum(r.get("confidence", 0) for r in results) / max(len(results), 1),
            "avg_rounds": sum(r.get("rounds_used", 0) for r in results) / max(len(results), 1),
            "ranked_results": ranked_results,
            "best_result": best_result,
        }

    def _validate_agent_names(
        self,
        agent_names: Any,
        field_name: str,
        *,
        allow_empty: bool = True,
    ) -> str | None:
        """Validate a list of agent names."""
        if not isinstance(agent_names, list):
            return f"{field_name} must be an array"
        if not allow_empty and not agent_names:
            return f"{field_name} must include at least one agent"
        if len(agent_names) > 10:
            return f"Maximum 10 agents allowed for {field_name}"
        for i, name in enumerate(agent_names):
            if not isinstance(name, str):
                return f"{field_name}[{i}] must be a string"
            if len(name) > 50:
                return f"{field_name}[{i}] name too long (max 50 chars)"
        return None

    def _normalize_model_combinations(
        self, raw_model_combinations: list[Any]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Normalize model combination inputs into a consistent shape."""
        normalized: list[dict[str, Any]] = []
        for i, combination in enumerate(raw_model_combinations):
            field_name = f"model_combinations[{i}]"
            if isinstance(combination, list):
                agent_error = self._validate_agent_names(
                    combination,
                    f"{field_name}.agents",
                    allow_empty=False,
                )
                if agent_error:
                    return [], agent_error
                normalized.append(
                    {
                        "name": " + ".join(combination),
                        "agents": list(combination),
                        "is_baseline": i == 0,
                    }
                )
                continue

            if not isinstance(combination, dict):
                return [], f"{field_name} must be an object or array of agent names"

            combo_agents = combination.get("agents")
            agent_error = self._validate_agent_names(
                combo_agents,
                f"{field_name}.agents",
                allow_empty=False,
            )
            if agent_error:
                return [], agent_error

            combo_name = combination.get("name") or " + ".join(combo_agents)
            if not isinstance(combo_name, str):
                return [], f"{field_name}.name must be a string"
            if len(combo_name) > 100:
                return [], f"{field_name}.name too long (max 100 chars)"

            normalized.append(
                {
                    "name": combo_name,
                    "agents": list(combo_agents),
                    "is_baseline": bool(combination.get("is_baseline", False)),
                }
            )
        return normalized, None

    def _build_scenario_variants(
        self,
        scenarios: list[dict[str, Any]],
        model_combinations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expand base scenarios across model combinations."""
        if not model_combinations:
            return [dict(scenario) for scenario in scenarios]

        base_scenarios = scenarios or [{"name": "Base question", "is_baseline": True}]
        expanded: list[dict[str, Any]] = []
        default_combo_baseline = not any(combo.get("is_baseline") for combo in model_combinations)

        for scenario in base_scenarios:
            source_name = scenario.get("name") or "Scenario"
            for index, combo in enumerate(model_combinations):
                variant = dict(scenario)
                variant["source_scenario_name"] = source_name
                variant["model_combination_name"] = combo["name"]
                variant["agents"] = list(combo["agents"])
                variant["name"] = (
                    combo["name"]
                    if len(base_scenarios) == 1 and source_name == "Base question"
                    else f"{source_name} [{combo['name']}]"
                )
                combo_baseline = combo.get("is_baseline", False) or (
                    default_combo_baseline and index == 0
                )
                variant["is_baseline"] = bool(
                    scenario.get("is_baseline", False) or combo_baseline
                ) and (len(base_scenarios) == 1 or scenario.get("is_baseline", False))
                if len(base_scenarios) == 1:
                    variant["is_baseline"] = bool(
                        scenario.get("is_baseline", False) or combo_baseline
                    )
                expanded.append(variant)
        return expanded

    def _build_best_result_payload(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the selected best result with its deterministic score."""
        selected = select_best_result(results)
        if not isinstance(selected, dict):
            return None

        payload = dict(selected)
        payload["selection_score"] = score_result_candidate(selected)
        payload["selection_strategy"] = "consensus_confidence_answer"
        return payload

    def _format_condition_label(self, result: dict[str, Any]) -> str:
        """Format a human-readable label for a conditional conclusion."""
        scenario_name = result.get("source_scenario_name") or result.get(
            "scenario_name", "scenario"
        )
        model_combination_name = result.get("model_combination_name")
        if model_combination_name and model_combination_name != scenario_name:
            return f"When {scenario_name} using {model_combination_name}"
        if model_combination_name:
            return f"When using {model_combination_name}"
        return f"When {scenario_name}"

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
                except (ImportError, ValueError, TypeError, KeyError, AttributeError) as e:
                    logger.warning("Failed to create agent %s: %s", name, e)
            return agents
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Failed to load agents: %s", e)
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
        except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
            logger.error("Failed to get matrix debate %s: %s", matrix_id, e)
            return error_response("Failed to retrieve matrix debate", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/matrix/{matrix_id}/scenarios",
        summary="Get matrix debate scenarios",
        description="Get all scenario results from a matrix debate.",
        tags=["Debates", "Matrix Debates"],
        parameters=[
            {"name": "matrix_id", "in": "path", "required": True, "schema": {"type": "string"}}
        ],
        responses={
            "200": {"description": "List of scenario results"},
            "401": {"description": "Authentication required"},
            "503": {"description": "Storage not configured"},
        },
    )
    async def _get_scenarios(self, handler: Any, matrix_id: str) -> HandlerResult:
        """Get all scenario results for a matrix debate."""
        storage = getattr(handler, "storage", None)
        if not storage:
            return error_response("Storage not configured", 503)

        try:
            scenarios = await storage.get_matrix_scenarios(matrix_id)
            return json_response({"matrix_id": matrix_id, "scenarios": scenarios})
        except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
            logger.error("Failed to get scenarios for %s: %s", matrix_id, e)
            return error_response("Failed to retrieve scenarios", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/matrix/{matrix_id}/conclusions",
        summary="Get matrix debate conclusions",
        description="Get universal and conditional conclusions from a matrix debate.",
        tags=["Debates", "Matrix Debates"],
        parameters=[
            {"name": "matrix_id", "in": "path", "required": True, "schema": {"type": "string"}}
        ],
        responses={
            "200": {"description": "Universal and conditional conclusions"},
            "401": {"description": "Authentication required"},
            "503": {"description": "Storage not configured"},
        },
    )
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
        except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
            logger.error("Failed to get conclusions for %s: %s", matrix_id, e)
            return error_response("Failed to retrieve conclusions", 500)
