# Golden Path 3: Adversarial Code Review

Multiple agents review a code diff from different perspectives -- one as the author defending the change, one as a security reviewer looking for vulnerabilities, and one as a senior engineer balancing pragmatism with correctness.

## What it demonstrates

- Reviewing code diffs with multi-agent adversarial debate
- Custom agent proposals tailored to specific review findings
- Structured findings with severity ratings (CRITICAL / WARNING / INFO)
- Critique exchange where agents challenge each other's assessments
- Generating a review verdict with decision receipt

## Run it

```bash
python examples/golden_paths/code_review/main.py
```

No API keys required.

## Expected output

```
================================================================
  Aragora Golden Path: Adversarial Code Review
================================================================

--- Code Diff Under Review ---
  diff --git a/api/auth.py b/api/auth.py
  ...
  +    cursor.execute(
  +        f"INSERT INTO sessions (user_id, token) VALUES ('{user_id}', '{token}')"
  +    )
  ...

Reviewers: code-author, security-reviewer, senior-engineer
Rounds:    2

--- Review Findings ---

[security-reviewer]:
    REVIEW ASSESSMENT: CRITICAL ISSUES FOUND.
    1. [CRITICAL] SQL Injection (line +22, +29): f-string SQL queries...
    2. [CRITICAL] Hardcoded Secret (line 15): SECRET_KEY is hardcoded...
    3. [WARNING] No connection pooling...
    ...

--- Final Verdict ---
  Result:     CONDITIONAL
  Consensus:  Reached
  Confidence: 73%
  Receipt:    DR-20260224-...
```

## Sample diff

The example includes a deliberately vulnerable code diff with:

| Finding | Severity | CWE |
|---------|----------|-----|
| SQL injection via f-strings | CRITICAL | CWE-89 |
| Hardcoded secret key | CRITICAL | CWE-798 |
| No connection pooling | WARNING | -- |
| /tmp storage path | WARNING | -- |
| Missing type hints | INFO | -- |

## Key APIs used

| Import | Purpose |
|--------|---------|
| `aragora_debate.Arena` | Debate orchestrator for code review |
| `aragora_debate.StyledMockAgent` | Reviewers with custom findings |
| `aragora_debate.ReceiptBuilder` | Audit trail for the review decision |

## Next steps

- Pipe real `git diff` output into the review question
- Export findings as SARIF for VS Code integration
- Connect to GitHub Actions with `aragora review --format github`
- See `examples/slack_code_review/` for Slack-integrated reviews
