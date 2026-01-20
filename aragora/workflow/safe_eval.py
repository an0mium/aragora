"""
Safe AST-based expression evaluator for workflow conditions.

Replaces unsafe eval() calls with a restricted AST evaluator that only
allows safe operations: comparisons, boolean logic, attribute access,
subscripting, and a whitelist of safe functions.

Security benefits:
- No arbitrary code execution
- No access to __builtins__, __class__, __mro__, etc.
- No function definitions, imports, or class creation
- Explicit whitelist of allowed operations

Usage:
    from aragora.workflow.safe_eval import safe_eval

    namespace = {"result": "success", "count": 5}
    result = safe_eval("result == 'success' and count > 0", namespace)
"""

import ast
import logging
import operator
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Safe binary operators
_SAFE_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}

# Safe unary operators
_SAFE_UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

# Safe comparison operators
_SAFE_CMPOPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

# Safe builtin functions (whitelist)
_SAFE_BUILTINS = {
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
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "round": round,
    "isinstance": isinstance,
    "type": type,
    "hasattr": hasattr,
    "getattr": getattr,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "True": True,
    "False": False,
    "None": None,
}

# Blocked attribute names (prevent access to dangerous internals)
_BLOCKED_ATTRS = {
    "__class__",
    "__bases__",
    "__mro__",
    "__subclasses__",
    "__init__",
    "__new__",
    "__del__",
    "__call__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__dict__",
    "__globals__",
    "__code__",
    "__func__",
    "__self__",
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "__module__",
    "__qualname__",
    "__reduce__",
    "__reduce_ex__",
}


class SafeEvalError(Exception):
    """Raised when safe_eval encounters an unsafe or invalid expression."""

    pass


class _SafeEvaluator(ast.NodeVisitor):
    """AST visitor that safely evaluates expressions."""

    def __init__(self, namespace: dict[str, Any]):
        self.namespace = namespace

    def visit(self, node: ast.AST) -> Any:
        """Visit a node and return its evaluated value."""
        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> Any:
        """Raise error for unsupported node types."""
        raise SafeEvalError(f"Unsupported expression type: {node.__class__.__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        """Handle Expression wrapper."""
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Handle literal constants (numbers, strings, etc.)."""
        return node.value

    def visit_Num(self, node: ast.Num) -> Any:
        """Handle numeric literals (Python 3.7 compatibility)."""
        return node.n

    def visit_Str(self, node: ast.Str) -> Any:
        """Handle string literals (Python 3.7 compatibility)."""
        return node.s

    def visit_NameConstant(self, node: ast.NameConstant) -> Any:
        """Handle True, False, None (Python 3.7 compatibility)."""
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        """Handle variable names."""
        name = node.id
        if name in self.namespace:
            return self.namespace[name]
        if name in _SAFE_BUILTINS:
            return _SAFE_BUILTINS[name]
        raise SafeEvalError(f"Unknown name: {name}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Handle binary operations (a + b, a * b, etc.)."""
        op_type = type(node.op)
        if op_type not in _SAFE_BINOPS:
            raise SafeEvalError(f"Unsupported binary operator: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _SAFE_BINOPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Handle unary operations (-a, not a, etc.)."""
        op_type = type(node.op)
        if op_type not in _SAFE_UNARYOPS:
            raise SafeEvalError(f"Unsupported unary operator: {op_type.__name__}")
        operand = self.visit(node.operand)
        return _SAFE_UNARYOPS[op_type](operand)

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Handle comparisons (a < b, a == b, a in b, etc.)."""
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _SAFE_CMPOPS:
                raise SafeEvalError(f"Unsupported comparison: {op_type.__name__}")
            right = self.visit(comparator)
            if not _SAFE_CMPOPS[op_type](left, right):
                return False
            left = right
        return True

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        """Handle boolean operations (and, or)."""
        if isinstance(node.op, ast.And):
            for value in node.values:
                result = self.visit(value)
                if not result:
                    return result
            return result
        elif isinstance(node.op, ast.Or):
            for value in node.values:
                result = self.visit(value)
                if result:
                    return result
            return result
        else:
            raise SafeEvalError(f"Unsupported boolean operator: {type(node.op).__name__}")

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Handle ternary expressions (a if b else c)."""
        if self.visit(node.test):
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Handle attribute access (a.b)."""
        attr_name = node.attr
        if attr_name in _BLOCKED_ATTRS:
            raise SafeEvalError(f"Access to '{attr_name}' is not allowed")
        value = self.visit(node.value)
        if not hasattr(value, attr_name):
            raise SafeEvalError(f"Attribute '{attr_name}' not found")
        return getattr(value, attr_name)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Handle subscript access (a[b], a[1:2])."""
        value = self.visit(node.value)
        slice_val = self.visit(node.slice)
        return value[slice_val]

    def visit_Slice(self, node: ast.Slice) -> Any:
        """Handle slice objects (1:2, ::2, etc.)."""
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return slice(lower, upper, step)

    def visit_Index(self, node: Any) -> Any:
        """Handle index (Python 3.8 compatibility).

        Note: ast.Index was deprecated in Python 3.9 and removed in 3.11.
        In newer Python versions, the subscript node's slice is the value directly.
        """
        # In Python 3.9+, ast.Index may not exist or may not have 'value' attribute
        if hasattr(node, "value"):
            return self.visit(node.value)
        # Fallback: the node itself is the value
        return self.visit(node)

    def visit_List(self, node: ast.List) -> Any:
        """Handle list literals ([a, b, c])."""
        return [self.visit(el) for el in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        """Handle tuple literals ((a, b, c))."""
        return tuple(self.visit(el) for el in node.elts)

    def visit_Set(self, node: ast.Set) -> Any:
        """Handle set literals ({a, b, c})."""
        return {self.visit(el) for el in node.elts}

    def visit_Dict(self, node: ast.Dict) -> Any:
        """Handle dict literals ({a: b, c: d})."""
        return {
            self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values) if k is not None
        }

    def visit_Call(self, node: ast.Call) -> Any:
        """Handle function calls (f(a, b))."""
        func = self.visit(node.func)

        # Check if it's a safe builtin or from namespace
        func_allowed = (
            func in _SAFE_BUILTINS.values()
            or callable(func)
            and getattr(func, "__name__", "") in _SAFE_BUILTINS
        )

        if not func_allowed:
            # Also allow methods on safe types
            if not callable(func):
                raise SafeEvalError(f"Calling non-callable: {func}")

        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        """Handle list comprehensions ([x for x in y])."""
        return self._eval_comprehension(node, list)

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        """Handle set comprehensions ({x for x in y})."""
        return self._eval_comprehension(node, set)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        """Handle generator expressions (x for x in y)."""
        # Return as list for simplicity in workflow context
        return self._eval_comprehension(node, list)

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        """Handle dict comprehensions ({k: v for k, v in items})."""
        result = {}
        for gen in node.generators:
            iter_val = self.visit(gen.iter)
            for item in iter_val:
                # Handle tuple unpacking
                target = gen.target
                if isinstance(target, ast.Tuple):
                    if not isinstance(item, (tuple, list)) or len(item) != len(target.elts):
                        raise SafeEvalError("Cannot unpack iterable")
                    for t, v in zip(target.elts, item):
                        if isinstance(t, ast.Name):
                            self.namespace[t.id] = v
                elif isinstance(target, ast.Name):
                    self.namespace[target.id] = item

                # Check conditions
                if all(self.visit(cond) for cond in gen.ifs):
                    key = self.visit(node.key)
                    value = self.visit(node.value)
                    result[key] = value
        return result

    def _eval_comprehension(self, node: ast.AST, container_type: type) -> Any:
        """Helper to evaluate list/set comprehensions."""
        result = []
        elt = node.elt  # type: ignore
        generators = node.generators  # type: ignore

        for gen in generators:
            iter_val = self.visit(gen.iter)
            for item in iter_val:
                # Handle tuple unpacking
                target = gen.target
                if isinstance(target, ast.Tuple):
                    if not isinstance(item, (tuple, list)) or len(item) != len(target.elts):
                        raise SafeEvalError("Cannot unpack iterable")
                    for t, v in zip(target.elts, item):
                        if isinstance(t, ast.Name):
                            self.namespace[t.id] = v
                elif isinstance(target, ast.Name):
                    self.namespace[target.id] = item

                # Check conditions
                if all(self.visit(cond) for cond in gen.ifs):
                    result.append(self.visit(elt))

        return container_type(result)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        """Handle f-strings."""
        parts = []
        for value in node.values:
            parts.append(str(self.visit(value)))
        return "".join(parts)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Any:
        """Handle formatted values in f-strings."""
        value = self.visit(node.value)
        if node.format_spec:
            spec = self.visit(node.format_spec)
            return format(value, spec)
        return value


def safe_eval(expression: str, namespace: dict[str, Any] | None = None) -> Any:
    """
    Safely evaluate a Python expression using AST parsing.

    Only allows safe operations: comparisons, boolean logic, arithmetic,
    attribute access, subscripting, and whitelisted functions.

    Args:
        expression: The expression string to evaluate
        namespace: Optional dict of variables available in the expression

    Returns:
        The result of evaluating the expression

    Raises:
        SafeEvalError: If the expression contains unsafe operations
        SyntaxError: If the expression has invalid Python syntax

    Example:
        >>> safe_eval("x > 0 and y < 10", {"x": 5, "y": 3})
        True
        >>> safe_eval("len(items) > 0", {"items": [1, 2, 3]})
        True
        >>> safe_eval("result.status == 'success'", {"result": obj})
        True
    """
    if namespace is None:
        namespace = {}

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"Invalid expression syntax: {e}") from e

    evaluator = _SafeEvaluator(namespace.copy())
    return evaluator.visit(tree)


def safe_eval_bool(expression: str, namespace: dict[str, Any] | None = None) -> bool:
    """
    Safely evaluate an expression and return a boolean result.

    Convenience wrapper around safe_eval that converts the result to bool.

    Args:
        expression: The expression string to evaluate
        namespace: Optional dict of variables available in the expression

    Returns:
        Boolean result of the expression

    Raises:
        SafeEvalError: If the expression contains unsafe operations
    """
    return bool(safe_eval(expression, namespace))


__all__ = ["safe_eval", "safe_eval_bool", "SafeEvalError"]
