# Security Audit: Subprocess Calls

**Audit Date:** January 2026
**Auditor:** Claude (automated security review)
**Scope:** All `subprocess.run()` and `subprocess.Popen()` calls in the `aragora/` directory

## Executive Summary

Audited **16 subprocess calls** across **9 files**. All calls use `shell=False` (preventing shell injection). One minor issue found in `cli/review.py` where `shell=False` should be explicitly set for clarity.

**Overall Risk Level:** LOW

## Audit Checklist

For each subprocess call, verified:
- [x] Uses `shell=False` (or has documented exception)
- [x] Input is sanitized/validated before use
- [x] No direct user input in command without validation
- [x] Timeout specified
- [x] Error handling present

## Findings by File

### HIGH PRIORITY FILES

#### 1. aragora/tools/code.py (7 calls)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 371-377 | `git rev-parse --git-dir` | LOW | SAFE - Fixed command |
| 388-394 | `git checkout -b {branch_name}` | MEDIUM | SAFE - shell=False, arg-based |
| 465-471 | `python -m pytest` | LOW | SAFE - Fixed command, timeout=180s |
| 490-494 | `python -m py_compile` + files | LOW | SAFE - Files from glob(), shell=False |
| 518-524 | `git add -A` | LOW | SAFE - Fixed command |
| 525-530 | `git commit -m {message}` | MEDIUM | SAFE - shell=False, message as single arg |
| 542-548 | `git reset --hard HEAD~1` | LOW | SAFE - Fixed command |

**Notes:**
- `branch_name` and `message` come from `CodeProposal` which is agent-generated
- With `shell=False`, these are passed as arguments, not shell-parsed
- No injection possible via git command arguments

#### 2. aragora/implement/executor.py (1 call)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 176-183 | `git diff --stat` | LOW | SAFE - Fixed command, timeout=180s |

### MEDIUM PRIORITY FILES

#### 3. aragora/verification/proofs.py (1 call)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 257-264 | `sys.executable + temp_file` | LOW | WELL SECURED |

**Mitigations in place:**
- `shell=False`
- Filtered environment via `_get_safe_subprocess_env()` - prevents API key leakage
- Code validation via `_validate_code_safety()` - blocks dangerous patterns
- `SAFE_BUILTINS` whitelist - restricts available functions
- `DANGEROUS_PATTERNS` blocklist - prevents sandbox escapes
- Explicit timeout (5 seconds default)
- Temp file cleanup in finally block
- This is **intentionally** a code execution sandbox with appropriate security

#### 4. aragora/verification/formal.py (1 call)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 248-254 | `lean --version` | LOW | SAFE - Fixed command, timeout=5s |

#### 5. aragora/connectors/github.py (2 calls)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 95-101 | `gh auth status` | LOW | SAFE - Fixed command, timeout=30s |
| 115-126 | `gh` commands (async) | LOW | WELL SECURED |

**Mitigations for async gh calls:**
- `VALID_REPO_PATTERN` regex validates `owner/repo` format
- `VALID_NUMBER_PATTERN` regex validates issue/PR numbers
- `ALLOWED_STATES` frozenset restricts state values
- `MAX_QUERY_LENGTH = 500` prevents query abuse
- Uses `asyncio.create_subprocess_exec` (no shell)
- Timeout=30s on all operations

#### 6. aragora/reasoning/provenance_enhanced.py (1 call)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 163-170 | `git` commands | LOW | SAFE - Fixed commands, internal args |

**Notes:**
- Only runs fixed git commands (`rev-parse`, `show`, etc.)
- Args constructed internally, not from user input
- `shell=False`, timeout=30s

### LOW PRIORITY FILES

#### 7. aragora/broadcast/mixer.py (2 calls)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 40-42 | `ffprobe` | LOW | SAFE - Internal paths, timeout=10s |
| 184 | `ffmpeg` | LOW | SAFE - Internal paths, timeout=300s |

**Mitigations:**
- File paths from internal `Path` objects
- Uses `tempfile.TemporaryDirectory` for secure temp files
- `shell=False`

#### 8. aragora/broadcast/video_gen.py (1 call)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 584-596 | `ffprobe` | LOW | SAFE - Internal paths, shell=False |

#### 9. aragora/cli/review.py (1 call)

| Line | Command | Risk | Status |
|------|---------|------|--------|
| 497-502 | `gh pr diff` | MEDIUM | MOSTLY SAFE - see note |

**Note:** Missing explicit `shell=False` parameter. While `subprocess.run()` defaults to `shell=False`, this should be explicit for security clarity. The PR number and repo come from URL parsing.

## Issues Identified

### Issue #1: Missing Explicit shell=False (cli/review.py:497)

**Severity:** LOW
**File:** `aragora/cli/review.py`
**Line:** 497

```python
result = subprocess.run(
    gh_cmd,
    capture_output=True,
    text=True,
    timeout=30,
    # Missing: shell=False
)
```

**Recommendation:** Add explicit `shell=False` for security clarity, even though it's the default.

## Risk Matrix

| Category | LOW | MEDIUM | HIGH |
|----------|-----|--------|------|
| Shell Injection | 0 | 0 | 0 |
| Command Injection | 0 | 0 | 0 |
| Timeout Missing | 0 | 0 | 0 |
| User Input in Command | 0 | 4* | 0 |

*MEDIUM: User-influenced input (agent output, URL parsing) but properly passed as arguments with shell=False

## Conclusion

The codebase demonstrates good security practices for subprocess usage:

1. **All calls use `shell=False`** - Prevents shell injection attacks
2. **Timeouts are specified** - Prevents hanging processes
3. **Error handling is present** - Graceful failure handling
4. **Input validation exists** where needed (GitHub connector regex patterns)
5. **Sandbox execution is well-secured** (proofs.py has multiple layers)

### Recommendations

1. Add explicit `shell=False` to `cli/review.py:497` for clarity
2. Continue current practices for any new subprocess calls
3. Consider adding input validation for git branch names in `tools/code.py`

## Appendix: Security Patterns Used

### Pattern 1: Fixed Commands
```python
subprocess.run(["git", "status"], shell=False, timeout=30)
```

### Pattern 2: Validated User Input
```python
if not VALID_REPO_PATTERN.match(repo):
    raise ValueError(f"Invalid repo format: {repo}")
subprocess.run(["gh", "repo", "view", repo], shell=False)
```

### Pattern 3: Sandboxed Execution
```python
env = _get_safe_subprocess_env()  # Filter sensitive env vars
subprocess.run([sys.executable, temp_file], env=env, timeout=5, shell=False)
```

### Pattern 4: Async Subprocess with Timeout
```python
proc = await asyncio.create_subprocess_exec("gh", *args, ...)
stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
```

---

# Bandit Static Analysis Report

**Audit Date:** January 2026
**Tool:** Bandit 1.7+

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| High | 0 | Clean |
| Medium | 80 | Triaged |
| Low | 137 | Acceptable |

## Medium Findings Triage

### B608: SQL Injection Vector (56 findings)

**Status**: False Positives / Low Risk

Most B608 findings flag f-string SQL construction. Our patterns:

1. **Column name interpolation** (relationships.py, storage modules)
   - Column names from internal logic, not user input
   - VALUES always parameterized (`?` placeholders)
   - Example: `SET {col} = {col} + 1 WHERE agent_a = ?`

2. **Table name interpolation** (base_store.py, schema modules)
   - Table names are constants/class attributes, not user input
   - All user-provided values use parameterized queries

**Mitigation**: All user input is parameterized, never interpolated.

### B310: URL Scheme Audit (19 findings)

**Status**: Tolerated Risk

These flag `urllib.request.urlopen()` allowing `file://` schemes.

**Analysis**:
- Used in research connectors (Wikipedia, web scraping)
- URLs are hardcoded (API endpoints) or validated
- External URLs are user-initiated (not automatic)

**Mitigation**: URL validation restricts schemes to `http://`, `https://`.

### B314: XML Parsing (2 findings)

**Status**: Tolerated Risk

Located in RSS feed parsing and export handling.

**Analysis**:
- XML from trusted sources (configured RSS feeds)
- No user-uploaded XML processing
- defusedxml not required for trusted internal data

### B104: Bind All Interfaces (3 findings)

**Status**: Intentional for Development

Default `0.0.0.0` binding for container deployments.

**Mitigation**: `ARAGORA_HOST` environment variable for explicit binding in production.

## Low Findings (137)

Acceptable findings include:
- `assert` statements in non-production code
- `try/except/pass` patterns for optional features
- Hardcoded temporary directory paths

## Security Controls in Place

- Input validation (size limits, format validation)
- Rate limiting (per-endpoint, per-user)
- Authentication (JWT tokens, API keys)
- CORS with origin allowlist
- Security headers (X-Frame-Options, CSP, etc.)
- Error message sanitization (redacts secrets)
- SQL parameterization for all user input
- Path traversal protection
