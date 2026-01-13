"""
MCP Gauntlet Tools.

Document stress-testing with adversarial AI.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def run_gauntlet_tool(
    content: str,
    content_type: str = "spec",
    profile: str = "quick",
) -> Dict[str, Any]:
    """
    Run gauntlet stress-test on content.

    Args:
        content: The content to stress-test
        content_type: Type of content (spec, code, policy, architecture)
        profile: Test profile (quick, thorough, code, security, gdpr, hipaa)

    Returns:
        Dict with verdict, risk score, and vulnerabilities found
    """
    from aragora.gauntlet import GauntletConfig, GauntletRunner
    from aragora.gauntlet.config import AttackCategory

    if not content:
        return {"error": "Content is required"}

    # Configure based on profile
    if profile == "security":
        attack_categories = [AttackCategory.SECURITY, AttackCategory.ADVERSARIAL_INPUT]
    elif profile == "code":
        attack_categories = [AttackCategory.LOGIC, AttackCategory.EDGE_CASE]
    else:  # quick/thorough/gdpr/hipaa
        attack_categories = [
            AttackCategory.SECURITY,
            AttackCategory.LOGIC,
            AttackCategory.ARCHITECTURE,
        ]

    config = GauntletConfig(
        name=f"{profile}_gauntlet",
        input_type=content_type,
        attack_categories=attack_categories,
        attack_rounds=2 if profile == "quick" else 3,
    )

    runner = GauntletRunner(config)
    result = await runner.run(content)

    vulnerabilities = getattr(result, "vulnerabilities", [])

    return {
        "verdict": result.verdict.value if hasattr(result, "verdict") else "unknown",
        "risk_score": getattr(result, "risk_score", 0),
        "vulnerabilities_count": len(vulnerabilities),
        "vulnerabilities": [
            {
                "category": v.category,
                "severity": v.severity,
                "description": v.description,
            }
            for v in vulnerabilities[:10]  # Limit to 10
        ],
        "content_type": content_type,
        "profile": profile,
    }


__all__ = ["run_gauntlet_tool"]
