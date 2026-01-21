#!/usr/bin/env python3
"""
Incident Postmortem Debate Example
===================================

Multi-agent analysis of a production incident to identify root causes,
contributing factors, and prevention measures.

Use case: Blameless postmortems, incident analysis, prevention planning.

Time: ~4-6 minutes
Requirements: At least one API key

Usage:
    python examples/gallery/incident_postmortem_debate.py
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


INCIDENT_REPORT = """
INCIDENT: Production Database Outage
Severity: SEV-1
Duration: 4 hours 23 minutes
Impact: 100% of users unable to access platform

TIMELINE (UTC):
14:00 - Deploy of v2.3.1 containing database migration
14:05 - Migration starts, adds new index on 500M row table
14:15 - Database CPU spikes to 100%, read queries timeout
14:20 - PagerDuty alerts fire, on-call engineer paged
14:25 - Engineer identifies migration as cause
14:30 - Attempt to cancel migration fails (transaction locked)
14:45 - Decision made to failover to replica
15:00 - Failover initiated but replica also affected (replication lag)
15:30 - Decision to restore from backup
16:45 - Backup restoration complete
17:30 - Traffic gradually restored, monitoring confirms stability
18:23 - All-clear given, incident closed

CONTRIBUTING FACTORS IDENTIFIED:
1. Migration ran on production without prior load testing
2. No migration size limit in deploy pipeline
3. Runbook for database issues was 2 years out of date
4. Replica was configured with same connection pool (propagated locks)

TEAM CONTEXT:
- Migration author: Senior engineer (5 years experience)
- Reviewer: Staff engineer (approved without load test request)
- On-call: Junior engineer (first SEV-1 response)
- No dedicated DBA on team
"""


async def run_postmortem_debate():
    """Run a multi-agent incident postmortem debate."""

    # Create agents with different perspectives
    agent_configs = [
        ("anthropic-api", "systems_engineer"),
        ("openai-api", "process_analyst"),
        ("gemini", "human_factors"),
    ]

    agents = []
    for agent_type, role in agent_configs:
        try:
            agent = create_agent(model_type=agent_type, name=f"{role}", role=role)  # type: ignore
            agents.append(agent)
        except Exception:
            pass

    if len(agents) < 2:
        return None

    env = Environment(
        task=f"""Analyze this production incident using blameless postmortem principles.

{INCIDENT_REPORT}

Focus on:
1. Root cause analysis (5 Whys)
2. Contributing factors beyond the immediate trigger
3. What went well during response
4. Process and system improvements

Provide:
1. True root cause(s) - not just "human error"
2. Systemic factors that enabled the incident
3. Specific, measurable action items with owners
4. Metrics to prevent recurrence
5. Timeline improvement opportunities""",
        context="Blameless postmortem for engineering organization",
    )

    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        early_stopping=True,
    )

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    return result


if __name__ == "__main__":
    result = asyncio.run(run_postmortem_debate())
    if result and result.consensus_reached:
        pass
