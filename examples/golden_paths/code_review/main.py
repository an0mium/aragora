#!/usr/bin/env python3
"""
Golden Path 3: Adversarial Code Review
=======================================

Agents review a code diff adversarially. One agent proposes the code is
acceptable, another critiques it for bugs and security issues, and a third
synthesizes a final review verdict. The debate produces structured findings
with severity ratings and a decision receipt.

No API keys required -- uses StyledMockAgent with custom proposals.

Usage:
    python examples/golden_paths/code_review/main.py

Expected runtime: <5 seconds
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running as a standalone script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aragora_debate import Arena, DebateConfig, ReceiptBuilder, StyledMockAgent


# ----------------------------------------------------------------
# Sample code diff to review
# ----------------------------------------------------------------

SAMPLE_DIFF = """\
diff --git a/api/auth.py b/api/auth.py
index 3a4b5c6..7d8e9f0 100644
--- a/api/auth.py
+++ b/api/auth.py
@@ -12,8 +12,25 @@ from datetime import datetime, timedelta
 from fastapi import APIRouter, HTTPException, Request
 from jose import jwt

+import sqlite3
+
 router = APIRouter()
 SECRET_KEY = "hardcoded-secret-key-do-not-use"
+DB_PATH = "/tmp/sessions.db"
+
+def create_session(user_id: str, token: str) -> None:
+    conn = sqlite3.connect(DB_PATH)
+    cursor = conn.cursor()
+    cursor.execute(
+        f"INSERT INTO sessions (user_id, token) VALUES ('{user_id}', '{token}')"
+    )
+    conn.commit()
+    conn.close()
+
+def get_user(user_id: str) -> dict:
+    conn = sqlite3.connect(DB_PATH)
+    cursor = conn.cursor()
+    cursor.execute(f"SELECT * FROM users WHERE id = '{user_id}'")
+    row = cursor.fetchone()
+    conn.close()
+    return {"id": row[0], "name": row[1]} if row else None

 @router.post("/login")
 async def login(request: Request):
"""


# ----------------------------------------------------------------
# Build review agents with code-review-specific proposals
# ----------------------------------------------------------------

def build_review_agents() -> list[StyledMockAgent]:
    """Create agents with code-review-focused proposals and critiques."""
    return [
        StyledMockAgent(
            "code-author",
            style="supportive",
            proposal=(
                "REVIEW ASSESSMENT: The changes add session management to the auth module. "
                "The implementation is straightforward and functional. SQLite is a reasonable "
                "choice for session storage in a single-server deployment. The new functions "
                "follow the existing module's patterns. I recommend APPROVAL with minor "
                "suggestions for error handling improvements."
            ),
            critique_issues=[
                "The critique raises valid points about SQL injection, but the code runs "
                "behind authentication middleware that validates user_id format",
                "Hardcoded secret key is flagged but this is a development-only setting",
            ],
        ),
        StyledMockAgent(
            "security-reviewer",
            style="critical",
            proposal=(
                "REVIEW ASSESSMENT: CRITICAL ISSUES FOUND.\n"
                "1. [CRITICAL] SQL Injection (line +22, +29): f-string SQL queries with "
                "unsanitized user input. Both create_session() and get_user() are vulnerable "
                "to SQL injection attacks. Must use parameterized queries.\n"
                "2. [CRITICAL] Hardcoded Secret (line 15): SECRET_KEY is hardcoded in source. "
                "Must use environment variable or secrets manager.\n"
                "3. [WARNING] No connection pooling: Opening/closing connections per request "
                "will cause performance issues under load.\n"
                "4. [WARNING] /tmp storage (line 17): Session DB in /tmp is not persistent "
                "across restarts and is world-readable.\n"
                "5. [INFO] Missing type hints on return value of get_user() (returns dict | None).\n"
                "Verdict: REJECT until critical issues are resolved."
            ),
            critique_issues=[
                "SQL injection via f-string formatting is a critical vulnerability (CWE-89)",
                "Hardcoded SECRET_KEY violates CWE-798 (hard-coded credentials)",
                "No connection pooling will cause resource exhaustion under load",
                "/tmp/sessions.db is world-readable and not persistent",
            ],
        ),
        StyledMockAgent(
            "senior-engineer",
            style="balanced",
            proposal=(
                "REVIEW ASSESSMENT: Mixed findings.\n"
                "The session management feature addresses a real need, but the implementation "
                "has security gaps that must be fixed before merge.\n\n"
                "MUST FIX (blocking):\n"
                "- SQL injection: Use parameterized queries (cursor.execute('...?', (param,)))\n"
                "- Hardcoded secret: Move to env var with os.environ['AUTH_SECRET_KEY']\n\n"
                "SHOULD FIX (non-blocking):\n"
                "- Add connection pooling or use a context manager pattern\n"
                "- Move DB path to a configurable, non-tmp location\n"
                "- Add return type annotation: -> dict | None\n\n"
                "Verdict: CONDITIONAL APPROVAL -- merge after fixing the two blocking issues."
            ),
            critique_issues=[
                "SQL injection is a valid blocking concern that must be resolved",
                "The hardcoded key should use environment variables, but is not a merge blocker "
                "if this is behind a feature flag",
            ],
        ),
    ]


async def main() -> None:
    print("=" * 64)
    print("  Aragora Golden Path: Adversarial Code Review")
    print("=" * 64)
    print()

    # ----------------------------------------------------------------
    # Step 1: Show the diff being reviewed
    # ----------------------------------------------------------------
    print("--- Code Diff Under Review ---")
    for line in SAMPLE_DIFF.strip().split("\n"):
        print(f"  {line}")
    print()

    # ----------------------------------------------------------------
    # Step 2: Set up the review debate
    # ----------------------------------------------------------------
    agents = build_review_agents()

    config = DebateConfig(
        rounds=2,
        consensus_method="majority",
        early_stopping=True,
    )

    arena = Arena(
        question=(
            "Review the following code diff for bugs, security vulnerabilities, "
            "performance concerns, and style issues. Provide a structured review "
            "with severity ratings (CRITICAL / WARNING / INFO) for each finding.\n\n"
            f"```diff\n{SAMPLE_DIFF}\n```"
        ),
        agents=agents,
        config=config,
    )

    print(f"Reviewers: {', '.join(a.name for a in agents)}")
    print(f"Rounds:    {config.rounds}")
    print()

    # ----------------------------------------------------------------
    # Step 3: Run the review debate
    # ----------------------------------------------------------------
    result = await arena.run()

    # ----------------------------------------------------------------
    # Step 4: Display structured findings
    # ----------------------------------------------------------------
    print("--- Review Findings ---")
    print()
    for agent_name, proposal_text in result.proposals.items():
        print(f"[{agent_name}]:")
        # Indent each line for readability
        for line in proposal_text.split("\n"):
            print(f"    {line}")
        print()

    # Show the critique exchange
    if result.critiques:
        print(f"--- Critique Exchange ({len(result.critiques)} critiques) ---")
        for critique in result.critiques:
            severity_str = f"severity {critique.severity:.0f}/10"
            print(f"  [{critique.agent} -> {critique.target_agent}] ({severity_str})")
            for issue in critique.issues[:2]:
                short_issue = issue[:100] + ("..." if len(issue) > 100 else "")
                print(f"    - {short_issue}")
        print()

    # Show voting
    if result.votes:
        print("--- Review Votes ---")
        for vote in result.votes:
            print(f"  [{vote.agent}] -> {vote.choice} "
                  f"(confidence: {vote.confidence:.0%})")
        print()

    # ----------------------------------------------------------------
    # Step 5: Final verdict
    # ----------------------------------------------------------------
    verdict_map = {
        "approved": "PASS",
        "approved_with_conditions": "CONDITIONAL",
        "needs_review": "NEEDS DISCUSSION",
        "rejected": "FAIL",
    }
    verdict_str = "UNKNOWN"
    if result.receipt:
        verdict_str = verdict_map.get(result.receipt.verdict.value, result.receipt.verdict.value)

    print("--- Final Verdict ---")
    print(f"  Result:     {verdict_str}")
    print(f"  Consensus:  {'Reached' if result.consensus_reached else 'Not reached'}")
    print(f"  Confidence: {result.confidence:.0%}")
    if result.receipt:
        print(f"  Receipt:    {result.receipt.receipt_id}")
    print()

    # Print the receipt markdown
    if result.receipt:
        print("--- Decision Receipt ---")
        print(result.receipt.to_markdown())
    print()

    print("In production, this review would be posted as a PR comment via")
    print("the GitHub or Slack integration, with SARIF export for IDE support.")


if __name__ == "__main__":
    asyncio.run(main())
