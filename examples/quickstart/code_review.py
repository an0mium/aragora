#!/usr/bin/env python3
"""
Multi-Model Code Review with SARIF Output.

Demonstrates Aragora's code review workflow: multiple models independently
review a diff, debate their findings, and produce SARIF-format output
suitable for CI/CD integration.

In demo mode, uses mocked responses. For real reviews:
    git diff main | aragora review

Usage:
    python examples/quickstart/code_review.py
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# --- Mock data representing a code review debate ---

SAMPLE_DIFF = """\
--- a/server/auth.py
+++ b/server/auth.py
@@ -42,7 +42,8 @@ def verify_token(token: str) -> dict:
-    decoded = jwt.decode(token, SECRET_KEY)
+    decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
+    if decoded.get("exp", 0) < time.time():
+        raise TokenExpiredError(decoded["sub"])
     return decoded
"""


@dataclass
class ReviewFinding:
    """A single finding from the code review debate."""

    rule_id: str
    message: str
    level: str  # "error", "warning", "note"
    file: str
    line: int
    models_agreed: list[str] = field(default_factory=list)
    models_dissented: list[str] = field(default_factory=list)
    consensus: str = "unanimous"


def run_mock_review() -> list[ReviewFinding]:
    """Simulate a multi-model code review debate."""
    return [
        ReviewFinding(
            rule_id="SEC-001",
            message=(
                "jwt.decode without verify=True is deprecated in PyJWT 2.x. "
                "The fix correctly adds algorithms but should also add "
                "options={'verify_exp': True} instead of manual exp check."
            ),
            level="warning",
            file="server/auth.py",
            line=44,
            models_agreed=["claude", "gpt", "gemini"],
            consensus="unanimous",
        ),
        ReviewFinding(
            rule_id="SEC-002",
            message=(
                "TokenExpiredError includes decoded['sub'] which may leak "
                "user identifiers in error logs. Consider logging the sub "
                "separately at debug level."
            ),
            level="warning",
            file="server/auth.py",
            line=45,
            models_agreed=["claude", "gpt"],
            models_dissented=["gemini"],
            consensus="majority",
        ),
        ReviewFinding(
            rule_id="PERF-001",
            message=(
                "time.time() returns wall clock time which can jump on NTP "
                "sync. For token expiry, this is acceptable but worth noting."
            ),
            level="note",
            file="server/auth.py",
            line=44,
            models_agreed=["gemini"],
            models_dissented=["claude", "gpt"],
            consensus="minority",
        ),
    ]


def findings_to_sarif(findings: list[ReviewFinding]) -> dict:
    """Convert review findings to SARIF 2.1.0 format."""
    results = []
    rules = []
    seen_rules: set[str] = set()

    for f in findings:
        if f.rule_id not in seen_rules:
            rules.append(
                {
                    "id": f.rule_id,
                    "shortDescription": {"text": f.rule_id},
                    "properties": {"consensus": f.consensus},
                }
            )
            seen_rules.add(f.rule_id)

        results.append(
            {
                "ruleId": f.rule_id,
                "level": f.level,
                "message": {
                    "text": f"[{f.consensus}] {f.message}",
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": f.file},
                            "region": {"startLine": f.line},
                        }
                    }
                ],
                "properties": {
                    "modelsAgreed": f.models_agreed,
                    "modelsDissented": f.models_dissented,
                    "consensusLevel": f.consensus,
                },
            }
        )

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "aragora-review",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/an0mium/aragora",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }


def main():
    print("Aragora Multi-Model Code Review (Demo)")
    print("=" * 45)
    print(f"\nReviewing diff ({len(SAMPLE_DIFF.splitlines())} lines)...")
    print()

    # Run mock review (in production: aragora review runs real debates)
    findings = run_mock_review()

    # Display findings grouped by consensus
    unanimous = [f for f in findings if f.consensus == "unanimous"]
    majority = [f for f in findings if f.consensus == "majority"]
    minority = [f for f in findings if f.consensus == "minority"]

    if unanimous:
        print("UNANIMOUS (all models agree -- fix these):")
        for f in unanimous:
            print(f"  [{f.level.upper()}] {f.file}:{f.line} - {f.message[:80]}...")
        print()

    if majority:
        print("MAJORITY (most models agree -- review these):")
        for f in majority:
            agreed = ", ".join(f.models_agreed)
            dissented = ", ".join(f.models_dissented)
            print(f"  [{f.level.upper()}] {f.file}:{f.line} - {f.message[:80]}...")
            print(f"    Agreed: {agreed} | Dissented: {dissented}")
        print()

    if minority:
        print("MINORITY (single model flagged -- investigate if relevant):")
        for f in minority:
            print(f"  [{f.level.upper()}] {f.file}:{f.line} - {f.message[:80]}...")
        print()

    # Generate SARIF
    sarif = findings_to_sarif(findings)
    sarif_json = json.dumps(sarif, indent=2)
    print(f"SARIF output ({len(sarif['runs'][0]['results'])} results):")
    print(sarif_json[:500])
    print("...")

    print("\nTo run a real multi-model code review:")
    print("  git diff main | aragora review")
    print("  aragora review --pr https://github.com/org/repo/pull/123 --format sarif")


if __name__ == "__main__":
    main()
