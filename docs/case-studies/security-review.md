# Case Study: Multi-Model Code Review Catches Authentication Bypass

> **Note:** This is a worked example demonstrating how Aragora's multi-model
> review process catches vulnerabilities that a single AI reviewer would miss.
> The company, code, and specific findings are fictional but model realistic
> patterns encountered in production codebases.

## The Scenario

A platform engineering team at a B2B SaaS company submits PR #1847, adding
OAuth2 token validation to their API gateway. The diff introduces a new
`verify_access_token()` function that every authenticated endpoint will call.
Getting this wrong means every protected route is vulnerable.

The diff under review:

```python
# gateway/auth/token_validator.py

import time
import json
import base64
import hashlib

from gateway.auth.keys import get_signing_key
from gateway.auth.models import TokenClaims
from gateway.auth.store import token_store


def verify_access_token(token: str, required_scopes: list[str] | None = None) -> TokenClaims:
    """Verify an OAuth2 access token and return its claims.

    Args:
        token: The bearer token from the Authorization header.
        required_scopes: If provided, the token must contain all of these scopes.

    Returns:
        TokenClaims with the validated token's subject, scopes, and expiry.

    Raises:
        AuthenticationError: If the token is invalid or expired.
    """
    # Decode the token parts
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthenticationError("Malformed token")

    header_b64, payload_b64, signature_b64 = parts

    # Verify signature
    signing_key = get_signing_key()
    expected_sig = hashlib.sha256(
        f"{header_b64}.{payload_b64}.{signing_key}".encode()
    ).hexdigest()

    if expected_sig != base64.urlsafe_b64decode(signature_b64).decode():
        raise AuthenticationError("Invalid signature")

    # Decode payload
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
    claims = TokenClaims(
        subject=payload["sub"],
        scopes=payload.get("scope", "").split(" "),
        expires_at=payload["exp"],
        issued_at=payload["iat"],
    )

    # Check expiry
    if claims.expires_at < time.time():
        raise AuthenticationError("Token expired")

    # Check revocation
    if token_store.is_revoked(claims.subject):
        raise AuthenticationError("Token revoked")

    return claims
```

---

## Single-Model Review

A team runs the diff through a single AI code reviewer. The model produces a
reasonable review, catching the obvious issues:

> **AI Review (single model):**
>
> 1. **Missing error handling on `json.loads`** -- if the base64 payload is
>    valid base64 but not valid JSON, the function raises an unhandled
>    `json.JSONDecodeError` that will bubble up as a 500 instead of a 401.
>
> 2. **Missing error handling on `base64.urlsafe_b64decode`** -- malformed
>    base64 input will raise `binascii.Error`.
>
> 3. **KeyError on `payload["sub"]` and `payload["exp"]`** -- if the
>    payload is valid JSON but missing required fields, the function raises
>    `KeyError` instead of `AuthenticationError`.
>
> **Verdict:** Address the error handling gaps before merging.

These are all real issues. But the review misses two subtle vulnerabilities
that require specialized knowledge to detect.

---

## Aragora Multi-Model Review

The same diff is reviewed using Aragora's multi-agent debate:

```bash
git diff main -- gateway/auth/token_validator.py | aragora review \
    --agents anthropic-api,openai-api,gemini-api \
    --rounds 3 \
    --focus security
```

Three models independently review the code, then critique each other's
findings across three rounds of structured debate. Here is the PR comment
produced by `format_github_comment()`:

---

## Multi Agent Code Review

**3 agents reviewed this PR** (anthropic-api, openai-api, gemini-api)

<details open>
<summary><strong>Unanimous Issues</strong> - All AI models agree</summary>

- Missing error handling: `json.loads`, `base64.urlsafe_b64decode`, and dict key access can raise unhandled exceptions that leak as HTTP 500 errors instead of returning 401
- `is_revoked()` checks the subject (user ID) but not the specific token -- a user with multiple active sessions would have all tokens revoked if any one is revoked

</details>

<details open>
<summary><strong>Critical & High Severity Issues</strong> (3 found)</summary>

- **CRITICAL**: Signature comparison uses `!=` (string equality) which is vulnerable to timing side-channel attacks -- an attacker can determine the correct signature byte-by-byte by measuring response times
- **HIGH**: No scope validation -- `required_scopes` parameter is accepted but never checked against `claims.scopes`, allowing any valid token to access any endpoint regardless of granted permissions
- **HIGH**: Custom HMAC-SHA256 signature verification instead of a standard JWT library (e.g., PyJWT) -- the hand-rolled scheme concatenates the signing key directly into the hash input, which is not a proper HMAC construction and is weaker than HMAC-SHA256

</details>

<details>
<summary><strong>Split Opinions</strong> - Agents disagree on these</summary>

| Topic | For | Against |
|-------|-----|---------|
| Timing attack via string comparison is ex... | anthropic-api, openai-api | gemini-api |
| Token should be checked against a blocklist... | openai-api, gemini-api | anthropic-api |
| Should reject tokens with `iat` in the future | anthropic-api | openai-api, gemini-api |

</details>

<details>
<summary><strong>Risk Areas</strong> - Manual review recommended</summary>

- `get_signing_key()` is called on every request -- verify it is cached and not reading from disk/network each time
- The base64 padding workaround (`+ "=="`) may silently accept malformed tokens that a strict decoder would reject

</details>

---
*Agreement score: 78% | Powered by [Aragora](https://github.com/an0mium/aragora) - Multi Agent Decision Making*

---

### How Each Agent Contributed

**Agent 1 (anthropic-api)** found the error handling gaps and flagged the
timing attack on signature comparison. It also noted the hand-rolled crypto
scheme deviates from standard HMAC construction.

**Agent 2 (openai-api)** independently flagged the timing attack and was the
first to notice the missing scope validation -- the `required_scopes` parameter
is accepted but the function never actually compares it against
`claims.scopes`. It also identified the per-subject (rather than per-token)
revocation check as a correctness bug.

**Agent 3 (gemini-api)** caught the scope validation gap and the error
handling issues, but did not flag the timing attack. In the debate rounds,
it argued that the timing attack is theoretical because network jitter would
mask the timing signal. The other two agents countered with citations showing
that statistical analysis over thousands of requests can reliably extract
timing differences even over the network, and that `hmac.compare_digest()`
exists specifically for this reason.

The disagreement itself was informative: it surfaced the exact reasoning for
and against the vulnerability, giving the team a complete picture.

---

## The Receipt

After the review debate, Aragora generates a `DecisionReceipt` -- a
tamper-evident, cryptographically hashed audit artifact. This is the condensed
receipt for PR #1847, matching the structure from
`aragora.gauntlet.receipt_models.DecisionReceipt`:

```json
{
  "receipt_id": "a3f7c912-4e8b-4d1a-b6c3-9f2e1d8a5b47",
  "gauntlet_id": "review-a3f7c912",
  "timestamp": "2026-01-15T14:23:07.891Z",
  "schema_version": "1.0",

  "input_summary": "PR review: https://github.com/acme/platform/pull/1847",
  "input_hash": "e4a3b2c1d5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",

  "verdict": "CONDITIONAL",
  "confidence": 0.78,
  "robustness_score": 0.62,

  "risk_summary": {
    "critical": 1,
    "high": 2,
    "medium": 0,
    "low": 0,
    "total": 3
  },

  "vulnerability_details": [
    {
      "agent": "openai-api",
      "issue": "Timing side-channel in signature comparison allows byte-by-byte signature extraction",
      "target": "gateway/auth/token_validator.py:29",
      "severity": "CRITICAL"
    },
    {
      "agent": "openai-api",
      "issue": "required_scopes parameter is accepted but never validated against token scopes",
      "target": "gateway/auth/token_validator.py:14",
      "severity": "HIGH"
    },
    {
      "agent": "anthropic-api",
      "issue": "Hand-rolled HMAC-SHA256 construction is weaker than standard HMAC and should use hmac module",
      "target": "gateway/auth/token_validator.py:26",
      "severity": "HIGH"
    }
  ],

  "verdict_reasoning": "Review found 3 issue(s) with 78% agent agreement. One critical timing side-channel and two high-severity design flaws require remediation before merge. Scope validation gap means any valid token can access any endpoint.",

  "dissenting_views": [
    "Timing attack via string comparison is exploitable over the network",
    "Token should be checked against a per-token blocklist, not per-subject"
  ],

  "consensus_proof": {
    "reached": true,
    "confidence": 0.78,
    "supporting_agents": ["anthropic-api", "openai-api"],
    "dissenting_agents": ["gemini-api"],
    "method": "multi_agent_review",
    "evidence_hash": "e4a3b2c1d5f6a7b8"
  },

  "provenance_chain": [
    {
      "timestamp": "2026-01-15T14:23:07.891Z",
      "event_type": "review_finding",
      "agent": "anthropic-api",
      "description": "[CRITICAL] Timing side-channel in signature comparison allows byte-by-byte",
      "evidence_hash": "7a3f9c2e1b4d8f6a"
    },
    {
      "timestamp": "2026-01-15T14:23:07.891Z",
      "event_type": "review_finding",
      "agent": "openai-api",
      "description": "[HIGH] required_scopes parameter is accepted but never validated against token",
      "evidence_hash": "b2c4d6e8f0a1c3e5"
    },
    {
      "timestamp": "2026-01-15T14:23:07.891Z",
      "event_type": "review_finding",
      "agent": "anthropic-api",
      "description": "[HIGH] Hand-rolled HMAC-SHA256 construction is weaker than standard HMAC",
      "evidence_hash": "d1e3f5a7b9c2d4e6"
    },
    {
      "timestamp": "2026-01-15T14:23:07.891Z",
      "event_type": "unanimous_critique",
      "agent": null,
      "description": "Missing error handling: json.loads, base64.urlsafe_b64decode, and dict key",
      "evidence_hash": "f0a2b4c6d8e1f3a5"
    },
    {
      "timestamp": "2026-01-15T14:23:07.891Z",
      "event_type": "split_opinion",
      "agent": null,
      "description": "Timing attack via string comparison is exploitabl (for: ['anthropic-api', 'openai-api'], against: ['gemini-api'])",
      "evidence_hash": "a1b3c5d7e9f2a4b6"
    },
    {
      "timestamp": "2026-01-15T14:23:07.891Z",
      "event_type": "verdict",
      "agent": null,
      "description": "Review verdict: CONDITIONAL (agreement: 78.0%, 3 issue(s) found)"
    }
  ],

  "config_used": {
    "pr_url": "https://github.com/acme/platform/pull/1847",
    "reviewer_agents": ["anthropic-api", "openai-api", "gemini-api"],
    "source": "aragora_review"
  },

  "artifact_hash": "c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8"
}
```

Key receipt properties:

- **`verdict: "CONDITIONAL"`** -- high-severity issues found but agreement
  score is above the 0.7 threshold, so it is not a hard `FAIL`. The team
  must address the findings before merge.
- **`confidence: 0.78`** -- derived from the inter-agent agreement score.
  Two of three agents agreed on the critical finding.
- **`consensus_proof.dissenting_agents: ["gemini-api"]`** -- Gemini's
  disagreement on the timing attack is recorded permanently. If the
  vulnerability is later confirmed (or refuted), the dissent record shows
  which model had the correct assessment.
- **`artifact_hash`** -- SHA-256 content-addressable hash. Any modification
  to the receipt (changing the verdict, removing a finding) invalidates the
  hash, making tampering detectable.

---

## Outcome

The team addresses all three findings before merging:

1. **Timing attack** -- replaced `!=` string comparison with
   `hmac.compare_digest()`, which runs in constant time regardless of input.
2. **Scope validation** -- added the missing check:
   ```python
   if required_scopes:
       missing = set(required_scopes) - set(claims.scopes)
       if missing:
           raise AuthorizationError(f"Missing scopes: {', '.join(missing)}")
   ```
3. **Crypto construction** -- replaced the hand-rolled scheme with
   `hmac.new(signing_key, msg, hashlib.sha256)` for proper HMAC, and
   added a backlog item to migrate to PyJWT with RS256.

Six months later, the company undergoes a penetration test. The pentest report
confirms:

> "The API gateway's token validation uses constant-time comparison for
> signature verification and enforces scope-based access control. No timing
> side-channel or scope escalation vulnerabilities were identified."

The original `DecisionReceipt` from the Aragora review is attached as
evidence in the company's SOC 2 audit trail, demonstrating that the
vulnerability was identified and remediated during code review -- before
it reached production.

---

## Key Takeaway

**The disagreement between models was the signal.** A single model reviewed
this code and produced a reasonable report focused on error handling. It
missed the timing attack and the scope validation gap -- two vulnerabilities
that require specific security domain expertise to recognize.

When three models reviewed the same code independently and then debated
their findings:

- The **timing attack** was flagged by two out of three models. The third
  model's counter-argument ("network jitter masks the signal") was recorded
  but overruled by the majority, who cited the existence of
  `hmac.compare_digest()` as evidence that the Python standard library
  considers this a real attack vector.

- The **scope validation gap** was caught by two models but missed by the
  first. The debate process surfaced it as a unanimous issue after the
  second round, when the models critiqued each other's reviews.

- The **agreement score of 0.78** told the team exactly how much confidence
  to place in the findings. A score of 1.0 would mean every model agreed
  on everything. A score of 0.78 means there is meaningful disagreement
  worth investigating -- which is precisely what led to the timing attack
  being flagged as a split opinion rather than buried in a single model's
  output.

The cost of running three models instead of one was approximately 3x the
token cost. The cost of shipping a timing side-channel in an authentication
gateway to production would have been orders of magnitude higher.

---

*To run a multi-model code review on your own PRs:*

```bash
# Review from stdin
git diff main | aragora review --focus security

# Review a GitHub PR directly
aragora review https://github.com/your-org/repo/pull/123

# Generate a Decision Receipt
aragora review https://github.com/your-org/repo/pull/123 --gauntlet

# CI mode (exit non-zero on critical findings)
git diff main | aragora review --ci --sarif results.sarif
```

*See the [review CLI documentation](../../aragora/cli/review.py) for all options.*
