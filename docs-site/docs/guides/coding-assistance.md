---
title: Coding Assistance
description: Coding Assistance
---

# Coding Assistance

Aragora includes a coding assistance module for automated test generation and
quality scaffolding. It is designed to support engineering teams that want
repeatable, reviewable test suites for changed code.

## What It Does

- Parses Python source with `ast` to identify functions and branches.
- Generates test cases for happy paths, boundary cases, and error handling.
- Emits test code for multiple frameworks (pytest, unittest, Jest, etc.).
- Produces structured test plans that can feed downstream review workflows.

Core module: `aragora/coding/test_generator.py`

## Supported Frameworks

`pytest`, `unittest`, `jest`, `mocha`, `vitest`, `rspec`, `go_test`, `rust_test`

## Usage

### Generate Tests for a Function

```python
from aragora.coding import TestFramework, generate_tests_for_function

code = """

def add(a: int, b: int) -> int:
    if a < 0 or b < 0:
        raise ValueError("no negatives")
    return a + b
"""

cases, test_code = generate_tests_for_function(
    code,
    function_name="add",
    framework=TestFramework.PYTEST,
)

print(test_code)
```

### Generate Tests for a File

```python
from aragora.coding import TestFramework, generate_tests_for_file

suite = generate_tests_for_file("aragora/core/decision.py", framework=TestFramework.PYTEST)
print(suite.to_code())
```

## Notes

- The generator uses static analysis and heuristics. It does not execute code.
- Generated tests should be reviewed before use in production.
- For multi-agent review or refactoring workflows, combine this module with
  Gauntlet or deliberation templates.
- The module is a library utility; it is not exposed as an HTTP endpoint by
  default.
