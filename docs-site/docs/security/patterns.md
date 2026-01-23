---
title: Security Patterns in Aragora
description: Security Patterns in Aragora
---

# Security Patterns in Aragora

This document describes the security patterns, controls, and best practices used throughout the Aragora codebase. It serves as both documentation and a guide for contributors.

## Executive Summary

Aragora implements defense-in-depth security with:
- **No unsafe dynamic code execution** - Only 1 sandboxed `exec()` site
- **No `eval()` usage** - Replaced with AST-based evaluators
- **No `shell=True`** - All subprocess calls use list-based arguments
- **Input validation** - Blocklist patterns for dangerous code
- **Process isolation** - Subprocess sandboxing for untrusted code

---

## Dynamic Code Execution

### Sandboxed Exec (proofs.py)

**Location**: `aragora/verification/proofs.py`

**Purpose**: Execute user-provided proof assertions in a controlled environment.

**Security Controls**:
1. **Input validation**: `_validate_code_safety()` checks for dangerous patterns before execution
2. **Builtin restriction**: Uses `{"__builtins__": SAFE_BUILTINS}` whitelist
3. **Process isolation**: Execution happens in a subprocess that can be killed
4. **Timeout enforcement**: Hard timeout prevents infinite loops
5. **Pattern blocklist**: Blocks `__class__`, `__import__`, `exec(`, `eval(`, `open(`, etc.

```python
# Blocked patterns (proofs.py)
DANGEROUS_PATTERNS = [
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__builtins__", "__import__",
    "exec(", "eval(", "compile(", "open(", "getattr(",
    "os.", "sys.", "subprocess", "importlib",
    # ... and more
]

# Safe builtins whitelist (proofs.py)
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "float", "int",
    "len", "list", "max", "min", "range", "set", "str",
    "sum", "tuple", "type", "zip",
    # Explicitly excluded: __import__, open, exec, eval, compile
}
```

**Risk Level**: LOW (properly sandboxed)

---

### AST-Based Condition Evaluation (policy/engine.py)

**Location**: `aragora/policy/engine.py`

**Purpose**: Evaluate policy condition expressions without using `eval()`.

**Security Controls**:
1. **AST parsing**: Uses `ast.parse()` to parse conditions
2. **Node whitelist**: Only specific AST node types are allowed
3. **Explicit blocking**: `ast.Attribute`, `ast.Call`, `ast.Subscript` are blocked
4. **Context isolation**: Only provided context variables are accessible

```python
# Blocked node types (engine.py:259-260)
elif isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
    raise ValueError(f"Blocked node type for security: {type(node).__name__}")

# Supported expressions:
# - Comparisons: x == 1, y != "foo", z < 10
# - Membership: x in ["a", "b"]
# - Boolean: x == 1 and y == 2
# - Negation: not x
```

**Risk Level**: MINIMAL (no code execution, AST-only)

---

## Subprocess Execution

### General Pattern

All subprocess calls in Aragora use the **list-based argument form** which prevents shell injection:

```python
# CORRECT - list form (no shell injection)
subprocess.run(["git", "commit", "-m", message], shell=False)

# INCORRECT - shell form (vulnerable)
subprocess.run(f"git commit -m \{message\}", shell=True)  # NEVER USED
```

### Key Files Using Subprocess

| File | Purpose | shell= |
|------|---------|--------|
| `utils/subprocess_runner.py` | General command execution | `False` |
| `tools/code.py` | Git operations | `False` |
| `broadcast/mixer.py` | FFmpeg audio/video | `False` |
| `verification/proofs.py` | Sandboxed proof execution | `False` |
| `connectors/github.py` | GitHub CLI operations | `False` |
| `nomic/phases/commit.py` | Auto-commit operations | `False` |

### Subprocess Runner (subprocess_runner.py)

**Location**: `aragora/utils/subprocess_runner.py`

**Comment**: `shell=False,  # Never use shell=True`

This module provides the standard interface for subprocess execution throughout Aragora with:
- Explicit `shell=False` enforcement
- Timeout support
- Output capture
- Error handling

---

## Content Security Policy

### HTTP Headers (unified_server.py)

**Location**: `aragora/server/unified_server.py`

The server sets strict CSP headers:

```python
# Security headers
self.send_header("X-Content-Type-Options", "nosniff")
self.send_header("X-Frame-Options", "DENY")
self.send_header("X-XSS-Protection", "1; mode=block")
self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")

# CSP - 'unsafe-eval' removed for security
self.send_header(
    "Content-Security-Policy",
    "default-src 'self'; "
    "script-src 'self'; "  # No 'unsafe-eval' or 'unsafe-inline'
    "style-src 'self' 'unsafe-inline'; "  # Needed for CSS-in-JS
    "img-src 'self' data: https:; "
    # ...
)
```

---

## Input Validation

### API Input Validation

All API endpoints use Pydantic models for input validation:

```python
# Example from handlers/debates.py
class CreateDebateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=1000)
    agents: list[str] = Field(..., min_items=2, max_items=10)
    rounds: int = Field(default=3, ge=1, le=20)
```

### SQL Injection Prevention

Database operations use parameterized queries via SQLAlchemy:

```python
# Safe - parameterized query
session.execute(
    select(Debate).where(Debate.id == debate_id)
)

# Never used - string interpolation
# f"SELECT * FROM debates WHERE id = \{debate_id\}"
```

---

## Authentication & Authorization

### Token Management

- Tokens use cryptographically secure random generation
- Tokens have configurable TTL
- Token blacklist support for revocation
- bcrypt for password hashing (not MD5/SHA1)

### Session Security

- Session cookies are HttpOnly and Secure
- CSRF protection via SameSite cookie attribute
- Session timeout enforcement

---

## Environment & Secrets

### Best Practices

1. **`.env` in `.gitignore`**: Secrets never committed
2. **`.env.example`**: Template without actual values
3. **pydantic-settings**: Validates env vars at startup
4. **File permissions**: `.env` should be mode 600

### Required Environment Variables

See `docs/ENVIRONMENT.md` for the complete reference.

---

## Security Testing

### Automated Checks

The CI pipeline includes:
- **Bandit**: Static security analysis for Python
- **Safety**: Dependency vulnerability scanning
- **Ruff**: Linting includes security rules
- **Codebase scanning**: Dependency and CVE lookup via `aragora.analysis.codebase`

### Manual Testing

For security-critical changes:
1. Review all dynamic code execution paths
2. Verify subprocess calls use list form
3. Check input validation on new endpoints
4. Test authentication bypass scenarios

---

## Reporting Security Issues

If you discover a security vulnerability, please:
1. **Do not** open a public issue
2. Email security concerns to the maintainers
3. Include steps to reproduce
4. Allow reasonable time for a fix before disclosure

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-13 | Initial security audit and documentation | Claude Code |
