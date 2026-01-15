#!/usr/bin/env python3
"""
Security Audit Debate Example
=============================

Multi-agent security assessment using red team and blue team
perspectives to identify vulnerabilities and mitigations.

Use case: Comprehensive security analysis with adversarial thinking.

Time: ~4-6 minutes
Requirements: At least one API key

Usage:
    python examples/gallery/security_audit_debate.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora import Arena, Environment, DebateProtocol
from aragora.agents.base import create_agent


SYSTEM_DESCRIPTION = """
E-commerce Platform Security Audit Target

Architecture:
- React frontend (SPA) hosted on CloudFront
- Node.js API gateway with JWT authentication
- Python microservices (orders, inventory, payments)
- PostgreSQL for transactions, Redis for sessions
- Stripe integration for payments
- AWS deployment with VPC, public/private subnets

Authentication:
- JWT tokens (24h expiry)
- OAuth2 social login (Google, Facebook)
- Password stored with bcrypt
- MFA available but optional

API Endpoints:
- /api/auth/* - Authentication flows
- /api/orders/* - Order management
- /api/payments/* - Payment processing
- /api/admin/* - Admin operations (IP whitelist)

Current Security Measures:
- WAF with OWASP ruleset
- Rate limiting (1000 req/min per IP)
- TLS 1.3 everywhere
- Secrets in AWS Secrets Manager
"""


async def run_security_audit_debate():
    """Run a multi-agent security audit debate."""

    print("\n" + "=" * 60)
    print("ARAGORA: Security Audit Debate")
    print("=" * 60)

    # Red team vs Blue team agents
    agent_configs = [
        ("anthropic-api", "red_team"),  # Attack perspective
        ("openai-api", "blue_team"),  # Defense perspective
        ("gemini", "compliance"),  # Compliance perspective
    ]

    agents = []
    for agent_type, role in agent_configs:
        try:
            agent = create_agent(model_type=agent_type, name=f"{role}", role=role)  # type: ignore
            agents.append(agent)
            print(f"  + {agent.name} ready ({role})")
        except Exception as e:
            print(f"  - {agent_type} unavailable: {str(e)[:40]}")

    if len(agents) < 2:
        print("\nError: Need at least 2 agents. Check API keys.")
        return None

    env = Environment(
        task=f"""Perform a security audit of this e-commerce platform.

{SYSTEM_DESCRIPTION}

Red team: Identify attack vectors and vulnerabilities.
Blue team: Propose mitigations and security improvements.
Compliance: Check against OWASP Top 10 and PCI-DSS requirements.

Provide:
1. Top 5 vulnerabilities ranked by severity
2. Specific attack scenarios
3. Recommended mitigations
4. Compliance gaps
5. Prioritized remediation roadmap""",
        context="Security audit for PCI-DSS compliance",
    )

    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        enable_calibration=True,
    )

    print(f"\nRunning security audit with {len(agents)} agents...")

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    print(f"\n{'='*60}")
    print("SECURITY AUDIT RESULTS")
    print(f"{'='*60}")
    print(f"Consensus: {'Yes' if result.consensus_reached else 'No'}")
    print(f"Confidence: {result.confidence:.0%}")

    print(f"\n--- Security Findings ---")
    answer = result.final_answer
    print(answer[:2000] if len(answer) > 2000 else answer)

    return result


if __name__ == "__main__":
    result = asyncio.run(run_security_audit_debate())
    if result and result.consensus_reached:
        print("\n[SUCCESS] Security audit completed with consensus!")
