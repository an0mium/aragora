# Security Audit Report

**Date**: January 20, 2026
**Scanner**: Bandit v1.9.3
**Scope**: `aragora/` Python codebase (486,798 lines scanned)

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| HIGH | 0 | All fixed |
| MEDIUM | 135 | Reviewed (mostly false positives) |
| LOW | 0 | None |

## HIGH Severity Issues - FIXED

### 1. Shell Injection (B602) - FIXED
**Location**: `aragora/autonomous/loop_enhancement.py`
- Lines 701, 728, 784: `subprocess` calls with `shell=True`
- **Fix**: Refactored to use `shlex.split()` and list-based `subprocess.run()`
- **Risk**: Could allow command injection via file paths
- **Status**: Fixed

### 2. Weak Hash (B324) - FIXED
**Location**: Multiple files
- `aragora/debate/phases/context_init.py:413`
- `aragora/knowledge/mound/checkpoint.py:195`
- `aragora/knowledge/mound/federated_query.py:198, 671`
- **Fix**: Added `usedforsecurity=False` parameter to MD5 calls
- **Risk**: MD5 is weak for cryptographic purposes, but these were used for cache keys
- **Status**: Fixed

## MEDIUM Severity Issues - Reviewed

### B608: SQL Injection (87 instances)
- **Finding**: f-string SQL queries detected
- **Analysis**: All instances use parameterized queries with `?` placeholders
- **Risk**: Low - false positives, queries are properly parameterized
- **Action**: No changes needed

### B310: URL Open (25 instances)
- **Finding**: `urlopen` usage with potential user input
- **Analysis**: Used in controlled contexts (API clients, webhook handlers)
- **Risk**: Low - inputs are validated
- **Action**: No changes needed

### B314: XML Parsing (6 instances)
- **Finding**: XML parsing without defused XML
- **Analysis**: Used for GraphML export (trusted internal data)
- **Risk**: Low - no external XML parsing
- **Action**: No changes needed

### B104: Hardcoded Bind (4 instances)
- **Finding**: Server bound to `0.0.0.0`
- **Analysis**: Expected behavior for containerized servers
- **Risk**: Low - network policies handle access control
- **Action**: No changes needed

## Frontend Security

### XSS Vectors
- **Finding**: 5 uses of `dangerouslySetInnerHTML`
- **Analysis**:
  - 4 instances: Static icon constants (safe)
  - 1 instance: `embed_html` from API (sanitized server-side)
- **Risk**: Low with proper server sanitization
- **Recommendation**: Add DOMPurify for defense-in-depth

## Secret Detection

- **Finding**: No hardcoded secrets detected
- **Analysis**: All credential patterns are placeholder examples in docs
- **Status**: Pass

## Recommendations

1. **Add Content Security Policy** headers for XSS mitigation
2. **Implement rate limiting** on API endpoints (documented separately)
3. **Add DOMPurify** for client-side HTML sanitization
4. **Regular dependency audits** with `pip-audit` and `npm audit`

## Compliance Checklist

- [x] No HIGH severity issues
- [x] SQL injection protected (parameterized queries)
- [x] No hardcoded secrets
- [x] Subprocess calls secured
- [x] Hash functions marked for non-security use
- [x] XSS vectors reviewed

---
*This audit covers static analysis only. Consider penetration testing for production deployment.*
