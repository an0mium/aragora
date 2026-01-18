"""
Decision Step for conditional branching in workflows.

Provides flexible decision-making logic:
- Expression-based conditions
- Multiple branch support
- Default fallback handling
- Integration with AI-based decisions
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from aragora.workflow.safe_eval import SafeEvalError, safe_eval, safe_eval_bool
from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


class DecisionStep(BaseStep):
    """
    Decision step for conditional branching in workflows.

    Evaluates conditions and returns which branch to take.
    The workflow engine uses the output to determine the next step.

    Config options:
        conditions: List[dict] - Conditions to evaluate in order
            [{
                "name": "high_risk",
                "expression": "outputs.risk_score > 0.8",
                "next_step": "manual_review"
            }, ...]
        default_branch: str - Branch to take if no conditions match
        evaluation_mode: str - "first_match" or "all" (default: "first_match")
        ai_fallback: bool - Use AI to decide if no conditions match (default: False)
        ai_prompt: str - Prompt for AI decision (if ai_fallback is True)

    Usage:
        step = DecisionStep(
            name="Risk Routing",
            config={
                "conditions": [
                    {
                        "name": "high_risk",
                        "expression": "step.risk_assessment.score > 0.8",
                        "next_step": "manual_review"
                    },
                    {
                        "name": "medium_risk",
                        "expression": "step.risk_assessment.score > 0.5",
                        "next_step": "enhanced_review"
                    },
                ],
                "default_branch": "auto_approve",
            }
        )
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the decision step."""
        config = {**self._config, **context.current_step_config}

        conditions = config.get("conditions", [])
        default_branch = config.get("default_branch", "")
        evaluation_mode = config.get("evaluation_mode", "first_match")

        matched_branches = []
        evaluation_results = []

        for condition in conditions:
            name = condition.get("name", "unnamed")
            expression = condition.get("expression", "True")
            next_step = condition.get("next_step", "")

            try:
                result = self._evaluate_condition(expression, context)
                evaluation_results.append({
                    "name": name,
                    "expression": expression,
                    "result": result,
                    "next_step": next_step,
                })

                if result:
                    matched_branches.append({
                        "name": name,
                        "next_step": next_step,
                    })

                    if evaluation_mode == "first_match":
                        break

            except Exception as e:
                logger.warning(f"Condition evaluation failed for '{name}': {e}")
                evaluation_results.append({
                    "name": name,
                    "expression": expression,
                    "result": False,
                    "error": str(e),
                })

        # Determine final decision
        if matched_branches:
            selected_branch = matched_branches[0]
            decision = selected_branch["next_step"]
            decision_name = selected_branch["name"]
        elif config.get("ai_fallback", False):
            # Use AI to make decision
            ai_decision = await self._ai_decision(config, context)
            decision = ai_decision.get("next_step", default_branch)
            decision_name = ai_decision.get("name", "ai_decision")
            evaluation_results.append({
                "name": "ai_fallback",
                "result": True,
                "ai_reasoning": ai_decision.get("reasoning", ""),
            })
        else:
            decision = default_branch
            decision_name = "default"

        return {
            "decision": decision,
            "decision_name": decision_name,
            "matched_branches": matched_branches,
            "evaluations": evaluation_results,
            "next_step": decision,  # For workflow engine compatibility
        }

    def _evaluate_condition(self, expression: str, context: WorkflowContext) -> bool:
        """Safely evaluate a condition expression."""
        # Build evaluation namespace
        namespace = {
            "inputs": context.inputs,
            "outputs": context.step_outputs,
            "step": context.step_outputs,  # Alias for convenience
            "state": context.state,
            # Safe builtins
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "all": all,
            "any": any,
            # Comparison helpers
            "contains": lambda a, b: b in a if a else False,
            "startswith": lambda a, b: a.startswith(b) if isinstance(a, str) else False,
            "endswith": lambda a, b: a.endswith(b) if isinstance(a, str) else False,
        }

        # Add step outputs as direct variables for simpler expressions
        for step_id, output in context.step_outputs.items():
            safe_name = step_id.replace("-", "_").replace(".", "_")
            namespace[safe_name] = output

        try:
            return safe_eval_bool(expression, namespace)
        except SafeEvalError as e:
            logger.debug(f"Expression evaluation failed: {expression} -> {e}")
            raise

    async def _ai_decision(self, config: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Use AI to make a decision when no conditions match."""
        ai_prompt = config.get("ai_prompt", "")
        conditions = config.get("conditions", [])

        if not ai_prompt:
            # Build default prompt
            ai_prompt = "Based on the following context, which branch should be taken?\n\n"
            ai_prompt += f"Context:\n{self._summarize_context(context)}\n\n"
            ai_prompt += "Available branches:\n"
            for cond in conditions:
                ai_prompt += f"- {cond.get('name', 'unnamed')}: {cond.get('next_step', '')}\n"
            ai_prompt += f"- default: {config.get('default_branch', 'continue')}\n\n"
            ai_prompt += "Respond with just the branch name."

        try:
            from aragora.agents import create_agent

            agent = create_agent("claude")
            response = await agent.generate(ai_prompt)

            # Parse response to find matching branch
            response_lower = response.lower().strip()
            for cond in conditions:
                if cond.get("name", "").lower() in response_lower:
                    return {
                        "name": cond.get("name"),
                        "next_step": cond.get("next_step"),
                        "reasoning": response,
                    }

            return {
                "name": "default",
                "next_step": config.get("default_branch", ""),
                "reasoning": response,
            }

        except Exception as e:
            logger.warning(f"AI decision failed: {e}")
            return {
                "name": "error",
                "next_step": config.get("default_branch", ""),
                "error": str(e),
            }

    def _summarize_context(self, context: WorkflowContext) -> str:
        """Create a summary of context for AI decision."""
        parts = []

        if context.inputs:
            parts.append("Inputs:")
            for k, v in list(context.inputs.items())[:5]:
                parts.append(f"  - {k}: {str(v)[:200]}")

        if context.step_outputs:
            parts.append("\nStep Outputs:")
            for k, v in list(context.step_outputs.items())[:5]:
                if isinstance(v, dict):
                    v_str = str({kk: str(vv)[:100] for kk, vv in list(v.items())[:3]})
                else:
                    v_str = str(v)[:200]
                parts.append(f"  - {k}: {v_str}")

        return "\n".join(parts)


class SwitchStep(BaseStep):
    """
    Switch step for value-based branching (similar to switch/case).

    Config options:
        value: str - Expression to evaluate (e.g., "inputs.category")
        cases: Dict[str, str] - Map of values to next steps
            {"legal": "legal_review", "technical": "tech_review"}
        default: str - Default next step if no case matches

    Usage:
        step = SwitchStep(
            name="Route by Category",
            config={
                "value": "inputs.document_type",
                "cases": {
                    "contract": "contract_review",
                    "invoice": "invoice_processing",
                    "policy": "policy_review",
                },
                "default": "general_review",
            }
        )
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute switch step."""
        config = {**self._config, **context.current_step_config}

        value_expr = config.get("value", "")
        cases = config.get("cases", {})
        default = config.get("default", "")

        # Evaluate value expression using AST-based evaluator
        try:
            # Create a dict-like wrapper that supports attribute access
            class DictWrapper(dict):
                def __getattr__(self, key):
                    try:
                        return self[key]
                    except KeyError:
                        raise AttributeError(key)

            namespace = {
                "inputs": DictWrapper(context.inputs),
                "outputs": DictWrapper(context.step_outputs),
                "state": DictWrapper(context.state),
            }
            value = safe_eval(value_expr, namespace)
        except SafeEvalError as e:
            logger.warning(f"Value expression failed: {e}")
            value = None

        # Find matching case
        value_str = str(value) if value is not None else ""
        next_step = cases.get(value_str, cases.get(value, default))

        return {
            "value": value_str,
            "matched_case": value_str if value_str in cases else "default",
            "next_step": next_step,
        }
