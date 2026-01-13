"""
Handler validation decorators.

Provides decorators for automatic validation of handler requests
including body schemas, query parameters, and path segments.
"""

from functools import wraps
from typing import Callable, Dict, Optional, Tuple

from .schema import validate_against_schema
from .query_params import safe_query_int


def validate_request(
    schema: Optional[dict] = None,
    required_params: Optional[list] = None,
    path_validators: Optional[Dict[str, Callable]] = None,
) -> Callable:
    """Decorator for validating handler requests.

    Provides automatic validation of request bodies and query parameters
    for handler methods. Returns error responses early if validation fails.

    Args:
        schema: Schema dict for validating POST body (uses validate_against_schema)
        required_params: List of required query parameter names
        path_validators: Dict mapping path param names to validation functions

    Returns:
        Decorator function

    Example:
        @validate_request(
            schema=DEBATE_START_SCHEMA,
            required_params=["task"],
            path_validators={"debate_id": validate_debate_id}
        )
        def _handle_start_debate(self, path, query, body, handler):
            # body is already validated and parsed
            task = body["task"]
            ...

        @validate_request(required_params=["limit"])
        def _handle_list(self, path, query, handler):
            limit = safe_query_int(query, "limit", 10)
            ...

    Usage Pattern:
        The decorator assumes the handler method receives these args:
        - self: The handler instance
        - path: URL path string
        - query: Query params dict
        - body (optional): Parsed JSON body (for POST handlers)
        - handler: Server handler object

        For POST handlers with schema, the body is automatically parsed
        and validated, then passed to the handler.

        If validation fails, returns an error response dict with
        {"error": "...", "status": 400}.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract args - support multiple call patterns
            path = args[0] if args else kwargs.get("path", "")
            query = args[1] if len(args) > 1 else kwargs.get("query", {})

            # Validate required query params
            if required_params:
                for param in required_params:
                    val = query.get(param)
                    if val is None or (isinstance(val, list) and not val):
                        return {
                            "error": f"Missing required parameter: {param}",
                            "status": 400,
                        }

            # Validate path segments if validators provided
            if path_validators:
                parts = path.strip("/").split("/")
                for name, validator in path_validators.items():
                    # Try to find the segment in the path
                    # Common patterns: /api/debates/{id}, /api/agent/{name}/history
                    try:
                        if name == "debate_id" and len(parts) >= 3:
                            segment = parts[2]  # /api/debates/{id}
                        elif name == "agent" and len(parts) >= 3:
                            segment = parts[2]  # /api/agent/{name}
                        elif name in kwargs:
                            segment = kwargs[name]
                        else:
                            continue  # Skip if not found

                        is_valid, err = validator(segment)
                        if not is_valid:
                            return {"error": err, "status": 400}
                    except (IndexError, TypeError):
                        pass  # Path structure doesn't match, skip

            # For schemas, we need the body - caller must pass it
            if schema:
                body = kwargs.get("body")
                if body is None and len(args) > 2:
                    body = args[2]

                if body is not None:
                    result = validate_against_schema(body, schema)
                    if not result.is_valid:
                        return {"error": result.error, "status": 400}

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def validate_post_body(schema: dict) -> Callable:
    """Decorator for validating POST request bodies only.

    Simplified decorator that only validates the request body against
    a schema. Use for POST endpoints that need body validation.

    Args:
        schema: Schema dict for body validation

    Returns:
        Decorator function

    Example:
        @validate_post_body(DEBATE_START_SCHEMA)
        def _handle_start(self, body, handler):
            task = body["task"]  # Already validated
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Body should be first positional arg or in kwargs
            body = args[0] if args else kwargs.get("body", {})

            if not isinstance(body, dict):
                return {"error": "Request body must be a JSON object", "status": 400}

            result = validate_against_schema(body, schema)
            if not result.is_valid:
                return {"error": result.error, "status": 400}

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def validate_query_params(
    required: Optional[list] = None,
    int_params: Optional[Dict[str, Tuple[int, int, int]]] = None,
    string_params: Optional[Dict[str, Tuple[str, int]]] = None,
) -> Callable:
    """Decorator for validating query parameters.

    Args:
        required: List of required parameter names
        int_params: Dict mapping param names to (default, min, max) tuples
        string_params: Dict mapping param names to (default, max_length) tuples

    Returns:
        Decorator function

    Example:
        @validate_query_params(
            required=["agent"],
            int_params={"limit": (10, 1, 100), "offset": (0, 0, 10000)},
            string_params={"sort": ("created_at", 64)}
        )
        def _handle_list(self, query, handler):
            ...
    """

    def decorator(func: Callable) -> Callable:
        """Create a validation wrapper for the given handler function."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """Validate query parameters before invoking the handler.

            1. Extract query dict from kwargs or positional args
            2. Check required parameters exist and are non-empty
            3. Validate int params: parse, apply bounds checking
            4. Validate string params: check max length
            5. Return error dict with status 400 on validation failure
            6. Call handler on success
            """
            # Query should be in kwargs or as a positional arg
            query = kwargs.get("query")
            if query is None:
                # Check positional args - typically (self, path, query, ...)
                for arg in args:
                    if isinstance(arg, dict):
                        query = arg
                        break

            if query is None:
                query = {}

            # Check required params
            if required:
                for param in required:
                    val = query.get(param)
                    if val is None or (isinstance(val, list) and not val):
                        return {
                            "error": f"Missing required parameter: {param}",
                            "status": 400,
                        }

            # Validate int params
            if int_params:
                for param, (default, min_val, max_val) in int_params.items():
                    try:
                        raw = query.get(param)
                        if raw is not None:
                            if isinstance(raw, list):
                                raw = raw[0]
                            val = int(raw)
                            if val < min_val or val > max_val:
                                return {
                                    "error": f"Parameter '{param}' must be between {min_val} and {max_val}",
                                    "status": 400,
                                }
                    except (ValueError, TypeError):
                        return {
                            "error": f"Parameter '{param}' must be an integer",
                            "status": 400,
                        }

            # Validate string params
            if string_params:
                for param, (default, max_len) in string_params.items():
                    raw = query.get(param, default)
                    if isinstance(raw, list):
                        raw = raw[0] if raw else default
                    if raw and len(str(raw)) > max_len:
                        return {
                            "error": f"Parameter '{param}' exceeds maximum length of {max_len}",
                            "status": 400,
                        }

            return func(self, *args, **kwargs)

        return wrapper

    return decorator
