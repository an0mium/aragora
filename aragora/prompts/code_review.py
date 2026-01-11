"""
Specialized prompts for AI Red Team code review.

These prompts are designed to focus agents on finding real, actionable issues
rather than style nits or false positives.
"""

from __future__ import annotations

from typing import Optional

# Maximum diff size before truncation (50KB)
MAX_DIFF_SIZE = 50000


SECURITY_PROMPT = """
**Security Review Focus**

Look for vulnerabilities that could be exploited:

- **Injection** - SQL, NoSQL, OS command, LDAP injection
- **XSS** - Cross-site scripting (reflected, stored, DOM-based)
- **CSRF** - Missing or weak cross-site request forgery protection
- **Auth Bypass** - Authentication/authorization flaws
- **Secrets** - Hardcoded credentials, API keys, tokens in code
- **Deserialization** - Insecure deserialization of untrusted data
- **Path Traversal** - File path manipulation (../../etc/passwd)
- **SSRF** - Server-side request forgery
- **XXE** - XML external entity injection
- **Cryptography** - Weak algorithms, improper key handling

Rate each finding:
- CRITICAL: Exploitable with significant impact (data breach, RCE)
- HIGH: Exploitable but requires specific conditions
- MEDIUM: Potential vulnerability, needs more context
- LOW: Defense in depth improvement
"""

PERFORMANCE_PROMPT = """
**Performance Review Focus**

Look for issues that cause slowdowns or resource exhaustion:

- **N+1 Queries** - Database queries in loops
- **Algorithmic Complexity** - O(n^2) or worse operations on large data
- **Memory Issues** - Unbounded collections, memory leaks, large allocations
- **Missing Pagination** - Loading all records without limits
- **Blocking Operations** - Synchronous I/O in async code paths
- **Caching Opportunities** - Repeated expensive computations
- **Connection Leaks** - Unclosed database/HTTP connections
- **Inefficient Loops** - Unnecessary iterations, redundant work
- **Large Payloads** - Returning more data than needed
- **Missing Indexes** - Database queries without index hints

Rate each finding:
- CRITICAL: Will cause outage under normal load
- HIGH: Significant degradation (>2x slowdown)
- MEDIUM: Noticeable impact at scale
- LOW: Optimization opportunity
"""

QUALITY_PROMPT = """
**Code Quality Review Focus**

Look for issues that cause bugs or maintenance burden:

- **Error Handling** - Missing try/catch, swallowed exceptions, unclear errors
- **Edge Cases** - Null checks, empty arrays, boundary conditions
- **Input Validation** - Missing or incomplete validation at boundaries
- **Resource Management** - Unclosed files, connections, cleanup in finally
- **Race Conditions** - Shared state without synchronization
- **Type Safety** - Implicit conversions, any types, missing null checks
- **Logic Errors** - Off-by-one, incorrect operators, inverted conditions
- **Dead Code** - Unreachable code, unused variables
- **Complexity** - Functions doing too much, deep nesting
- **API Contracts** - Breaking changes, inconsistent return types

Rate each finding:
- CRITICAL: Will cause data corruption or crash
- HIGH: Bug that will manifest in production
- MEDIUM: Potential bug under certain conditions
- LOW: Code smell or maintainability issue
"""

# Combined default prompt
DEFAULT_FOCUS_AREAS = ["security", "performance", "quality"]


def get_focus_prompts(focus_areas: Optional[list[str]] = None) -> str:
    """Get combined focus prompts for specified areas.

    Args:
        focus_areas: List of focus areas (security, performance, quality)
                    Defaults to all areas.

    Returns:
        Combined prompt string with all requested focus areas.
    """
    focus = focus_areas or DEFAULT_FOCUS_AREAS
    prompts = []

    if "security" in focus:
        prompts.append(SECURITY_PROMPT)
    if "performance" in focus:
        prompts.append(PERFORMANCE_PROMPT)
    if "quality" in focus:
        prompts.append(QUALITY_PROMPT)

    return "\n\n".join(prompts)


def build_review_prompt(
    diff: str,
    focus_areas: Optional[list[str]] = None,
    additional_context: Optional[str] = None,
) -> str:
    """Build a complete code review prompt.

    Args:
        diff: The diff content to review
        focus_areas: List of focus areas (security, performance, quality)
        additional_context: Optional extra context about the codebase

    Returns:
        Complete prompt string ready for agent consumption.
    """
    focus_text = get_focus_prompts(focus_areas)

    # Truncate diff if too large
    if len(diff) > MAX_DIFF_SIZE:
        diff = diff[:MAX_DIFF_SIZE] + "\n\n[... diff truncated ...]"

    context_section = ""
    if additional_context:
        context_section = f"""
## Additional Context
{additional_context}
"""

    return f"""You are an expert code reviewer performing a security and quality audit.
Analyze the following diff carefully and identify real, actionable issues.

## Review Focus Areas
{focus_text}
{context_section}
## Diff to Review
```diff
{diff}
```

## Response Format

For each issue found, provide:

1. **Severity**: CRITICAL, HIGH, MEDIUM, or LOW
2. **Category**: Security, Performance, or Quality
3. **Location**: File and line number (if identifiable from diff)
4. **Issue**: Clear, specific description of the problem
5. **Suggestion**: Concrete fix or mitigation

**Important Guidelines:**
- Focus on REAL issues, not style preferences
- Be specific about the vulnerability or bug
- Provide actionable suggestions
- If you're uncertain, say so
- If no issues in a category, explicitly state "No issues found"

Begin your review:"""


# Agent role-specific prompts
SECURITY_REVIEWER_ROLE = """
You are a senior security engineer specializing in application security.
Your expertise includes OWASP Top 10, secure coding practices, and threat modeling.
Focus primarily on security vulnerabilities but note other critical issues.
"""

PERFORMANCE_REVIEWER_ROLE = """
You are a senior performance engineer specializing in system optimization.
Your expertise includes database optimization, caching strategies, and scalability.
Focus primarily on performance issues but note security concerns.
"""

QUALITY_REVIEWER_ROLE = """
You are a senior software architect specializing in code quality and maintainability.
Your expertise includes design patterns, error handling, and testing strategies.
Focus primarily on quality issues but note security and performance concerns.
"""


def get_role_prompt(role: str) -> str:
    """Get role-specific system prompt for an agent.

    Args:
        role: One of security_reviewer, performance_reviewer, quality_reviewer

    Returns:
        Role-specific prompt string.
    """
    roles = {
        "security_reviewer": SECURITY_REVIEWER_ROLE,
        "performance_reviewer": PERFORMANCE_REVIEWER_ROLE,
        "quality_reviewer": QUALITY_REVIEWER_ROLE,
    }
    return roles.get(role, "")
