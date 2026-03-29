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
            agents: list[str | dict] - Default agents to participate (2-10 agents)
            scenarios: list[dict] - List of scenario configurations (1-10 scenarios)
                - name: str - Scenario name (max 100 chars)
                - parameters: dict - Scenario-specific parameters
                - constraints: list[str] - Additional constraints
                - is_baseline: bool - Whether this is the baseline scenario
            model_combinations: list[dict] - Optional model/team combinations
                - name: str - Combination name (max 100 chars)
                - agents: list[str | dict] - Agents for this combination
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

        # Validate scenarios / combinations
        scenarios = data.get("scenarios", [])
        model_combinations = self._get_model_combinations(data)
        if not isinstance(scenarios, list):
            return error_response("scenarios must be an array", 400)
        if len(scenarios) > 10:
            return error_response("Maximum 10 scenarios allowed", 400)
        if not isinstance(model_combinations, list):
            return error_response("model_combinations must be an array", 400)
        if len(model_combinations) > 10:
            return error_response("Maximum 10 model combinations allowed", 400)
        if not scenarios and not model_combinations:
            return error_response(
                "At least one scenario or model combination is required",
                400,
            )

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

        # Validate default agents
        agent_names = data.get("agents", [])
        agent_error = self._validate_agents_payload(
            agent_names,
            field_name="agents",
            allow_empty=bool(model_combinations),
        )
        if agent_error is not None:
            return agent_error

        combination_error = self._validate_model_combinations(model_combinations, agent_names)
        if combination_error is not None:
            return combination_error

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

        # Combination runs rely on the fallback executor, which supports
        # per-run agent overrides and best-result selection.
        if model_combinations:
            return await self._run_matrix_debate_fallback(handler, data)

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
        scenarios = data.get("scenarios", [])
        agent_specs = data.get("agents", [])
        model_combinations = self._get_model_combinations(data)
        max_rounds = data.get("max_rounds", DEFAULT_ROUNDS)

        try:
            matrix_id = str(uuid.uuid4())
            ctx = getattr(self, "ctx", {}) or {}
            document_store = ctx.get("document_store")
            evidence_store = ctx.get("evidence_store")

            # Run scenarios / combinations in parallel.
            async def run_variant(
                scenario_data: dict[str, Any] | None = None,
                combination_data: dict[str, Any] | None = None,
            ) -> dict[str, Any]:
                scenario_name = (scenario_data or {}).get("name", "Base question")
                parameters = (scenario_data or {}).get("parameters", {})
                constraints = (scenario_data or {}).get("constraints", [])
                combination_name = (combination_data or {}).get("name")
                active_agent_specs = (
                    (combination_data or {}).get("agents", agent_specs)
                    if combination_data
                    else agent_specs
                )
                agents = await self._load_agents(active_agent_specs)
                if not agents:
                    label = combination_name or scenario_name
                    raise ValueError(f"No valid agents found for {label}")

                # Build scenario task with parameters and constraints
                scenario_task = f"{task}"
                if parameters:
                    param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
                    scenario_task += f"\n\nParameters: {param_str}"
                if constraints:
                    scenario_task += f"\n\nConstraints: {', '.join(constraints)}"
                if combination_name:
                    scenario_task += f"\n\nModel combination: {combination_name}"

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

                run_result = {
                    "scenario_name": scenario_name
                    if scenario_data
                    else (combination_name or "Base question"),
                    "variant_name": self._variant_name(
                        scenario_name=scenario_name if scenario_data else None,
                        combination_name=combination_name,
                    ),
                    "combination_name": combination_name,
                    "agents": self._describe_agent_specs(active_agent_specs),
                    "parameters": parameters,
                    "constraints": constraints,
                    "is_baseline": (scenario_data or {}).get("is_baseline", False),
                    "winner": result.winner,
                    "final_answer": result.final_answer,
                    "confidence": result.confidence,
                    "consensus_reached": result.consensus_reached,
                    "rounds_used": result.rounds_used,
                }
                run_result["selection_score"] = self._selection_score(run_result)
                return run_result

            scenario_combinations: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
            if scenarios and model_combinations:
                for scenario in scenarios:
                    for combination in model_combinations:
                        scenario_combinations.append((scenario, combination))
            elif scenarios:
                scenario_combinations.extend((scenario, None) for scenario in scenarios)
            else:
                scenario_combinations.extend(
                    (None, combination) for combination in model_combinations
                )

            gather_results = await asyncio.gather(
                *[
                    run_variant(scenario, combination)
                    for scenario, combination in scenario_combinations
                ],
                return_exceptions=True,
            )

            # Process results
            valid_results: list[dict[str, Any]] = []
            for r in gather_results:
                if isinstance(r, BaseException):
                    logger.error("Scenario failed: %s", r)
                else:
                    valid_results.append(r)

            if not valid_results:
                return error_response("No valid agents found", 400)

            # Find universal conclusions (conclusions that appear in all scenarios)
            universal_conclusions = self._find_universal_conclusions(valid_results)

            # Find conditional conclusions (conclusions specific to scenarios)
            conditional_conclusions = self._find_conditional_conclusions(valid_results)
            best_result = self._select_best_result(valid_results)
            best_combination = None
            if best_result and best_result.get("combination_name"):
                best_combination = {
                    "name": best_result["combination_name"],
                    "agents": best_result.get("agents", []),
                    "selection_score": best_result.get("selection_score", 0.0),
                }

            return json_response(
                {
                    "matrix_id": matrix_id,
                    "task": task,
                    "scenario_count": len(valid_results),
                    "results": valid_results,
                    "universal_conclusions": universal_conclusions,
                    "conditional_conclusions": conditional_conclusions,
                    "comparison_matrix": self._build_comparison_matrix(valid_results),
                    "best_result": best_result,
                    "best_combination": best_combination,
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
                condition_label = self._result_label(r)
                conditional.append(
                    {
                        "condition": f"When {condition_label}",
                        "parameters": r.get("parameters", {}),
                        "conclusion": r["final_answer"],
                        "confidence": r.get("confidence", 0),
                    }
                )
        return conditional

    def _build_comparison_matrix(self, results: list[dict]) -> dict:
        """Build a comparison matrix of scenarios."""
        best_result = self._select_best_result(results)
        return {
            "scenarios": list(dict.fromkeys(r["scenario_name"] for r in results)),
            "combinations": list(
                dict.fromkeys(
                    r["combination_name"]
                    for r in results
                    if isinstance(r.get("combination_name"), str)
                )
            ),
            "consensus_rate": sum(1 for r in results if r.get("consensus_reached"))
            / max(len(results), 1),
            "avg_confidence": sum(r.get("confidence", 0) for r in results) / max(len(results), 1),
            "avg_rounds": sum(r.get("rounds_used", 0) for r in results) / max(len(results), 1),
            "best_result": best_result,
        }

    def _get_model_combinations(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return model combination inputs with backward-compatible aliases."""
        raw = data.get("model_combinations")
        if raw is None:
            raw = data.get("agent_combinations")
        if raw is None:
            raw = data.get("combinations")
        if raw is None:
            return []
        return raw

    def _validate_agents_payload(
        self,
        agent_payload: Any,
        *,
        field_name: str,
        allow_empty: bool = False,
    ) -> HandlerResult | None:
        """Validate agent lists used by default teams and per-run combinations."""
        if agent_payload is None:
            agent_payload = []
        if not isinstance(agent_payload, list):
            return error_response(f"{field_name} must be an array", 400)
        if not agent_payload and not allow_empty:
            return error_response(f"{field_name} must include at least one agent", 400)
        if len(agent_payload) > 10:
            return error_response(f"Maximum 10 agents allowed in {field_name}", 400)

        for i, agent in enumerate(agent_payload):
            if isinstance(agent, str):
                if len(agent) > 200:
                    return error_response(
                        f"{field_name}[{i}] is too long (max 200 chars)",
                        400,
                    )
                continue

            if not isinstance(agent, dict):
                return error_response(
                    f"{field_name}[{i}] must be a string or object",
                    400,
                )

            provider = agent.get("provider") or agent.get("agent_type")
            if not isinstance(provider, str) or not provider.strip():
                return error_response(
                    f"{field_name}[{i}] must include provider",
                    400,
                )
            if len(provider) > 50:
                return error_response(
                    f"{field_name}[{i}].provider too long (max 50 chars)",
                    400,
                )
            model = agent.get("model")
            if model is not None and (not isinstance(model, str) or len(model) > 200):
                return error_response(
                    f"{field_name}[{i}].model must be a string up to 200 chars",
                    400,
                )

        return None

    def _validate_model_combinations(
        self,
        model_combinations: list[dict[str, Any]],
        default_agents: list[Any],
    ) -> HandlerResult | None:
        """Validate named model/team combinations."""
        for i, combination in enumerate(model_combinations):
            if not isinstance(combination, dict):
                return error_response(f"model_combinations[{i}] must be an object", 400)

            name = combination.get("name")
            if name is not None and not isinstance(name, str):
                return error_response(f"model_combinations[{i}].name must be a string", 400)
            if isinstance(name, str) and len(name) > 100:
                return error_response(
                    f"model_combinations[{i}].name too long (max 100 chars)",
                    400,
                )

            combo_agents = combination.get("agents", default_agents)
            agent_error = self._validate_agents_payload(
                combo_agents,
                field_name=f"model_combinations[{i}].agents",
                allow_empty=bool(default_agents),
            )
            if agent_error is not None:
                return agent_error

        return None

    def _coerce_agent_specs(self, agent_payload: list[Any]) -> list[Any]:
        """Coerce raw request agent payload into AgentSpec objects."""
        from aragora.agents.spec import AgentSpec

        specs: list[AgentSpec] = []
        for i, item in enumerate(agent_payload or []):
            if isinstance(item, AgentSpec):
                specs.append(item)
                continue
            if isinstance(item, str):
                specs.append(AgentSpec.parse(item, _warn=False))
                continue
            if not isinstance(item, dict):
                raise ValueError(f"Invalid agent spec at index {i}")

            provider = item.get("provider") or item.get("agent_type")
            if not provider:
                raise ValueError(f"Agent spec at index {i} missing provider")

            specs.append(
                AgentSpec(
                    provider=str(provider),
                    model=item.get("model"),
                    persona=item.get("persona"),
                    role=item.get("role"),
                    name=item.get("name"),
                    hierarchy_role=item.get("hierarchy_role"),
                )
            )

        return specs

    def _describe_agent_specs(self, agent_payload: list[Any]) -> list[str]:
        """Create readable labels for combination output."""
        labels: list[str] = []
        for spec in self._coerce_agent_specs(agent_payload):
            label = spec.provider
            if spec.model:
                label = f"{label}|{spec.model}"
            labels.append(label)
        return labels

    def _variant_name(
        self,
        *,
        scenario_name: str | None,
        combination_name: str | None,
    ) -> str:
        """Build a stable result label for cross-product runs."""
        if scenario_name and combination_name:
            return f"{scenario_name} [{combination_name}]"
        if combination_name:
            return combination_name
        return scenario_name or "Base question"

    def _result_label(self, result: dict[str, Any]) -> str:
        """Get the display label for a single matrix result."""
        label = (
            result.get("variant_name")
            or result.get("combination_name")
            or result.get("scenario_name")
        )
        return str(label) if label else "Matrix run"

    def _selection_score(self, result: dict[str, Any]) -> float:
        """Compute a transparent score for best-run selection."""
        confidence = float(result.get("confidence", 0.0) or 0.0)
        rounds_used = int(result.get("rounds_used", 0) or 0)
        consensus_bonus = 1.0 if result.get("consensus_reached") else 0.0
        answer_bonus = 0.1 if str(result.get("final_answer", "")).strip() else 0.0
        round_penalty = min(rounds_used, 100) * 0.001
        return round(consensus_bonus + confidence + answer_bonus - round_penalty, 4)

    def _best_result_sort_key(self, result: dict[str, Any]) -> tuple[float, float, int, int]:
        """Stable ordering for selecting the best run in a matrix batch."""
        return (
            1.0 if result.get("consensus_reached") else 0.0,
            float(result.get("confidence", 0.0) or 0.0),
            1 if str(result.get("final_answer", "")).strip() else 0,
            -int(result.get("rounds_used", 0) or 0),
        )

    def _select_best_result(self, results: list[dict]) -> dict[str, Any] | None:
        """Pick the strongest completed run from a matrix batch."""
        if not results:
            return None

        best = dict(max(results, key=self._best_result_sort_key))
        best["variant_name"] = self._result_label(best)
        best["selection_score"] = self._selection_score(best)
        best["selection_reason"] = (
            "Ranked by consensus reached, confidence, answer completeness, then fewer rounds."
        )
        return best

    async def _load_agents(self, agent_specs: list[Any]) -> list[Any]:
        """Load agents from string or object specs."""
        try:
            from typing import cast

            from aragora.agents.base import AgentType, create_agent
            from aragora.agents.personas.helpers import apply_persona_to_agent

            specs = self._coerce_agent_specs(agent_specs or ["claude", "openai"])
            agents: list[Any] = []
            for i, spec in enumerate(specs):
                role = spec.role
                if role is None:
                    if i == 0:
                        role = "proposer"
                    elif i == len(specs) - 1 and len(specs) > 1:
                        role = "synthesizer"
                    else:
                        role = "critic"
                try:
                    agent = create_agent(
                        model_type=cast(AgentType, spec.provider),
                        name=spec.name,
                        role=role,
                        model=spec.model,
                    )
                    if spec.persona:
                        apply_persona_to_agent(agent, spec.persona)
                    agents.append(agent)
                except (ImportError, ValueError, TypeError, KeyError, AttributeError) as e:
                    logger.warning("Failed to create agent %s: %s", spec.provider, e)
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
