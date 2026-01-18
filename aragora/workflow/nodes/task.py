"""
Task Step for generic task execution in workflows.

Provides a flexible task step that can execute various operations:
- Python functions
- HTTP requests
- Shell commands (sandboxed)
- Custom actions
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from aragora.workflow.safe_eval import SafeEvalError, safe_eval
from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


# Registry of task handlers
_task_handlers: Dict[str, Callable] = {}


def register_task_handler(name: str, handler: Callable) -> None:
    """Register a task handler function."""
    _task_handlers[name] = handler
    logger.debug(f"Registered task handler: {name}")


def get_task_handler(name: str) -> Optional[Callable]:
    """Get a registered task handler."""
    return _task_handlers.get(name)


class TaskStep(BaseStep):
    """
    Generic task step for flexible workflow operations.

    Config options:
        task_type: str - Type of task (function, http, transform, validate, aggregate)
        handler: str - Name of registered handler (for function type)
        url: str - URL for HTTP requests (for http type)
        method: str - HTTP method (default: GET)
        headers: dict - HTTP headers
        body: dict - HTTP request body (can use {placeholder} syntax)
        transform: str - Python expression for data transformation
        validation: dict - Validation rules
        inputs: List[str] - Input step IDs to aggregate
        output_format: str - Format of output (json, text, list)

    Task Types:
        - function: Execute a registered Python function
        - http: Make an HTTP request
        - transform: Transform data using expressions
        - validate: Validate data against rules
        - aggregate: Combine outputs from multiple steps

    Usage:
        # Transform task
        step = TaskStep(
            name="Extract Key Points",
            config={
                "task_type": "transform",
                "transform": "[p['content'] for p in inputs.paragraphs if p.get('important')]",
                "output_format": "list",
            }
        )

        # HTTP task
        step = TaskStep(
            name="Notify Webhook",
            config={
                "task_type": "http",
                "url": "https://api.example.com/webhook",
                "method": "POST",
                "headers": {"Authorization": "Bearer {api_token}"},
                "body": {"result": "{step.analysis.summary}"},
            }
        )
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the task step."""
        config = {**self._config, **context.current_step_config}
        task_type = config.get("task_type", "function")

        try:
            if task_type == "function":
                return await self._execute_function(config, context)
            elif task_type == "http":
                return await self._execute_http(config, context)
            elif task_type == "transform":
                return await self._execute_transform(config, context)
            elif task_type == "validate":
                return await self._execute_validate(config, context)
            elif task_type == "aggregate":
                return await self._execute_aggregate(config, context)
            else:
                return {"success": False, "error": f"Unknown task type: {task_type}"}

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_function(self, config: Dict[str, Any], context: WorkflowContext) -> Any:
        """Execute a registered function handler."""
        handler_name = config.get("handler", "")
        handler = get_task_handler(handler_name)

        if not handler:
            return {"success": False, "error": f"Handler not found: {handler_name}"}

        # Build arguments from config
        args = self._interpolate_dict(config.get("args", {}), context)

        # Execute handler
        if asyncio.iscoroutinefunction(handler):
            result = await handler(context, **args)
        else:
            result = handler(context, **args)

        return {"success": True, "result": result}

    async def _execute_http(self, config: Dict[str, Any], context: WorkflowContext) -> Any:
        """Execute an HTTP request."""
        try:
            import aiohttp
        except ImportError:
            return {"success": False, "error": "aiohttp not installed"}

        url = self._interpolate_text(config.get("url", ""), context)
        method = config.get("method", "GET").upper()
        headers = self._interpolate_dict(config.get("headers", {}), context)
        body = self._interpolate_dict(config.get("body", {}), context)
        timeout = config.get("timeout_seconds", 30)

        try:
            async with aiohttp.ClientSession() as session:
                kwargs = {"headers": headers, "timeout": aiohttp.ClientTimeout(total=timeout)}
                if method in ("POST", "PUT", "PATCH") and body:
                    kwargs["json"] = body

                async with session.request(method, url, **kwargs) as response:
                    response_text = await response.text()

                    # Try to parse as JSON
                    try:
                        import json
                        response_data = json.loads(response_text)
                    except Exception:
                        response_data = response_text

                    return {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "response": response_data,
                        "headers": dict(response.headers),
                    }

        except asyncio.TimeoutError:
            return {"success": False, "error": f"Request timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_transform(self, config: Dict[str, Any], context: WorkflowContext) -> Any:
        """Execute a data transformation."""
        transform_expr = config.get("transform", "")
        output_format = config.get("output_format", "auto")

        if not transform_expr:
            return {"success": False, "error": "No transform expression provided"}

        # Build namespace for transformation
        namespace = {
            "inputs": context.inputs,
            "outputs": context.step_outputs,
            "state": context.state,
            # Safe builtins
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "sorted": sorted,
            "filter": filter,
            "map": map,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "zip": zip,
            "enumerate": enumerate,
            "range": range,
        }

        # Add step outputs as direct variables
        for step_id, output in context.step_outputs.items():
            safe_name = step_id.replace("-", "_").replace(".", "_")
            namespace[safe_name] = output

        try:
            result = safe_eval(transform_expr, namespace)

            # Format output
            if output_format == "list" and not isinstance(result, list):
                result = list(result) if hasattr(result, "__iter__") else [result]
            elif output_format == "json":
                import json
                result = json.dumps(result, default=str)
            elif output_format == "text":
                result = str(result)

            return {"success": True, "result": result}

        except SafeEvalError as e:
            return {"success": False, "error": f"Transform failed: {e}"}

    async def _execute_validate(self, config: Dict[str, Any], context: WorkflowContext) -> Any:
        """Execute data validation."""
        rules = config.get("validation", {})
        data_expr = config.get("data", "inputs")

        # Get data to validate
        namespace = {
            "inputs": context.inputs,
            "outputs": context.step_outputs,
            "state": context.state,
        }
        try:
            data = safe_eval(data_expr, namespace)
        except SafeEvalError as e:
            return {"success": False, "valid": False, "error": f"Invalid data expression: {e}"}

        # Validate against rules
        errors = []
        warnings = []

        for field, rule in rules.items():
            value = data.get(field) if isinstance(data, dict) else getattr(data, field, None)

            # Required check
            if rule.get("required", False) and value is None:
                errors.append(f"Field '{field}' is required")
                continue

            if value is None:
                continue

            # Type check
            expected_type = rule.get("type")
            if expected_type:
                type_map = {"string": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}
                if expected_type in type_map and not isinstance(value, type_map[expected_type]):
                    errors.append(f"Field '{field}' must be {expected_type}")

            # Min/max for numbers
            if isinstance(value, (int, float)):
                if "min" in rule and value < rule["min"]:
                    errors.append(f"Field '{field}' must be >= {rule['min']}")
                if "max" in rule and value > rule["max"]:
                    errors.append(f"Field '{field}' must be <= {rule['max']}")

            # Min/max length for strings
            if isinstance(value, str):
                if "min_length" in rule and len(value) < rule["min_length"]:
                    errors.append(f"Field '{field}' must be at least {rule['min_length']} chars")
                if "max_length" in rule and len(value) > rule["max_length"]:
                    errors.append(f"Field '{field}' must be at most {rule['max_length']} chars")

            # Pattern match
            if "pattern" in rule and isinstance(value, str):
                import re
                if not re.match(rule["pattern"], value):
                    errors.append(f"Field '{field}' does not match pattern")

            # Allowed values
            if "enum" in rule and value not in rule["enum"]:
                errors.append(f"Field '{field}' must be one of: {rule['enum']}")

            # Custom expression
            if "expression" in rule:
                try:
                    ns = {"value": value, "field": field, "data": data, **namespace}
                    if not safe_eval(rule["expression"], ns):
                        errors.append(rule.get("message", f"Field '{field}' failed validation"))
                except SafeEvalError:
                    warnings.append(f"Could not evaluate expression for '{field}'")

        return {
            "success": len(errors) == 0,
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    async def _execute_aggregate(self, config: Dict[str, Any], context: WorkflowContext) -> Any:
        """Aggregate outputs from multiple steps."""
        input_steps = config.get("inputs", [])
        mode = config.get("mode", "merge")  # merge, list, first_valid

        if not input_steps:
            # Use all previous step outputs
            input_steps = list(context.step_outputs.keys())

        values = []
        for step_id in input_steps:
            if step_id in context.step_outputs:
                values.append(context.step_outputs[step_id])

        if mode == "list":
            return {"success": True, "result": values}
        elif mode == "first_valid":
            for v in values:
                if v is not None and v != "" and v != {}:
                    return {"success": True, "result": v}
            return {"success": False, "result": None, "error": "No valid values found"}
        else:  # merge
            result = {}
            for v in values:
                if isinstance(v, dict):
                    result.update(v)
            return {"success": True, "result": result}

    def _interpolate_text(self, template: str, context: WorkflowContext) -> str:
        """Interpolate text template with context values."""
        text = template
        for key, value in context.inputs.items():
            text = text.replace(f"{{{key}}}", str(value))
        for step_id, output in context.step_outputs.items():
            if isinstance(output, str):
                text = text.replace(f"{{step.{step_id}}}", output)
            elif isinstance(output, dict):
                for k, v in output.items():
                    text = text.replace(f"{{step.{step_id}.{k}}}", str(v))
        for key, value in context.state.items():
            text = text.replace(f"{{state.{key}}}", str(value))
        return text

    def _interpolate_dict(self, data: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Interpolate dictionary values with context."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._interpolate_text(value, context)
            elif isinstance(value, dict):
                result[key] = self._interpolate_dict(value, context)
            elif isinstance(value, list):
                result[key] = [
                    self._interpolate_text(v, context) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result


# Built-in task handlers


def _handler_log(context: WorkflowContext, message: str = "", level: str = "info") -> Dict[str, Any]:
    """Log a message."""
    log_func = getattr(logger, level, logger.info)
    log_func(f"[{context.workflow_id}] {message}")
    return {"logged": True, "message": message, "level": level}


def _handler_set_state(context: WorkflowContext, key: str = "", value: Any = None) -> Dict[str, Any]:
    """Set a state value."""
    context.set_state(key, value)
    return {"key": key, "value": value}


def _handler_delay(context: WorkflowContext, seconds: float = 1.0) -> Dict[str, Any]:
    """Delay execution (use in async context)."""
    import time
    time.sleep(seconds)
    return {"delayed_seconds": seconds}


# Register built-in handlers
register_task_handler("log", _handler_log)
register_task_handler("set_state", _handler_set_state)
register_task_handler("delay", _handler_delay)
