#!/usr/bin/env python3
"""
10_advanced_workflow.py - End-to-end workflow combining multiple SDK features.

This advanced example demonstrates a complete workflow that:
1. Queries knowledge for context
2. Configures custom agents with personas
3. Runs a debate with streaming and consensus tracking
4. Handles errors with retries and fallbacks
5. Stores results and evidence

Usage:
    python 10_advanced_workflow.py --dry-run
    python 10_advanced_workflow.py --topic "How should we handle PII in our ML pipeline?"
"""

import argparse
import asyncio
from datetime import datetime
from aragora_sdk import DebateConfig, Agent, AgentPersona
from aragora_sdk.knowledge import KnowledgeMound, Evidence, KnowledgeQuery
from aragora_sdk.streaming import StreamingClient, EventType, ReconnectPolicy
from aragora_sdk.consensus import ConsensusTracker, ConvergenceMetrics
from aragora_sdk.resilience import RetryPolicy


class WorkflowOrchestrator:
    """Orchestrates a complete debate workflow with all SDK features."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.metrics = {"events": 0, "rounds": 0, "knowledge_items": 0}
        self.convergence_data = []

    async def gather_knowledge(self, topic: str) -> list:
        """Step 1: Query knowledge base for relevant context."""
        print("\n[1/5] Gathering organizational knowledge...")

        if self.dry_run:
            print("  [DRY RUN] Would query Knowledge Mound")
            return [{"title": "Mock Policy", "confidence": 0.85}]

        knowledge = KnowledgeMound()
        query = KnowledgeQuery(text=topic, limit=5, min_confidence=0.6)
        results = await knowledge.search(query)

        self.metrics["knowledge_items"] = len(results)
        print(f"  Found {len(results)} relevant knowledge items")
        return results

    def configure_agents(self) -> list:
        """Step 2: Configure specialized agents with personas."""
        print("\n[2/5] Configuring specialized agents...")

        agents = [
            Agent(
                name="compliance_expert",
                model="claude-sonnet-4-20250514",
                persona=AgentPersona(
                    role="Compliance Officer",
                    expertise=["GDPR", "CCPA", "data privacy", "regulatory compliance"],
                    tone="thorough and risk-aware",
                    priorities=["legal compliance", "risk mitigation"],
                ),
                vote_weight=1.5,
                fallback=Agent(name="claude_fallback", model="gpt-4o"),
            ),
            Agent(
                name="security_architect",
                model="gpt-4o",
                persona=AgentPersona(
                    role="Security Architect",
                    expertise=["encryption", "access control", "threat modeling"],
                    tone="technical and security-focused",
                    priorities=["data security", "defense in depth"],
                ),
                vote_weight=1.3,
            ),
            Agent(
                name="ml_engineer",
                model="gemini-2.0-flash",
                persona=AgentPersona(
                    role="ML Engineer",
                    expertise=["machine learning", "data pipelines", "MLOps"],
                    tone="practical and implementation-focused",
                    priorities=["model performance", "data quality", "efficiency"],
                ),
                vote_weight=1.0,
            ),
        ]

        print(f"  Configured {len(agents)} specialized agents")
        for agent in agents:
            print(f"    - {agent.persona.role} (weight: {agent.vote_weight})")

        return agents

    async def run_streaming_debate(
        self,
        topic: str,
        agents: list,
        knowledge_context: list,
    ) -> dict:
        """Step 3: Run debate with streaming and consensus tracking."""
        print("\n[3/5] Running debate with streaming...")

        if self.dry_run:
            print("  [DRY RUN] Would run streaming debate")
            print("  [DRY RUN] Simulating 3 rounds...")
            for i in range(3):
                self.metrics["rounds"] += 1
                self.convergence_data.append({"round": i + 1, "agreement": 0.5 + (i * 0.15)})
            return {
                "consensus_reached": True,
                "decision": "Mock decision for dry run",
                "confidence": 0.82,
                "debate_id": "dry-run-001",
            }

        # Configure resilience
        retry_policy = RetryPolicy(max_retries=3, initial_delay=1.0)
        reconnect_policy = ReconnectPolicy(max_retries=3)

        # Consensus tracking callback
        def on_convergence(metrics: ConvergenceMetrics):
            self.metrics["rounds"] += 1
            self.convergence_data.append(
                {
                    "round": self.metrics["rounds"],
                    "agreement": metrics.agreement_level,
                }
            )
            print(f"    Round {self.metrics['rounds']}: {metrics.agreement_level:.1%} agreement")

        # Event handler
        async def on_event(event_type: str, data: dict):
            self.metrics["events"] += 1

        # Initialize streaming client
        client = StreamingClient(
            retry_policy=retry_policy,
            reconnect_policy=reconnect_policy,
        )

        tracker = ConsensusTracker(on_round_complete=on_convergence)

        config = DebateConfig(
            topic=topic,
            agents=agents,
            rounds=4,
            consensus_threshold=0.7,
            knowledge_context=knowledge_context,
            consensus_tracker=tracker,
            use_personas=True,
            collect_evidence=True,
        )

        # Run with streaming
        client.on(EventType.AGENT_MESSAGE, on_event)
        client.on(EventType.VOTE, on_event)

        async with client.connect() as stream:
            result = await stream.run_debate(config)

        return result.to_dict()

    async def store_evidence(self, result: dict, topic: str) -> None:
        """Step 4: Store evidence from debate outcome."""
        print("\n[4/5] Storing evidence and results...")

        if self.dry_run:
            print("  [DRY RUN] Would store evidence to Knowledge Mound")
            return

        if result.get("consensus_reached"):
            knowledge = KnowledgeMound()
            evidence = Evidence(
                claim=result["decision"],
                confidence=result["confidence"],
                sources=["compliance_expert", "security_architect", "ml_engineer"],
                debate_id=result.get("debate_id"),
                metadata={
                    "topic": topic,
                    "timestamp": datetime.utcnow().isoformat(),
                    "convergence_data": self.convergence_data,
                },
            )
            await knowledge.store_evidence(evidence)
            print(f"  Stored evidence with {evidence.confidence:.1%} confidence")

    def generate_report(self, topic: str, result: dict) -> dict:
        """Step 5: Generate final report."""
        print("\n[5/5] Generating report...")

        report = {
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat(),
            "result": {
                "consensus_reached": result.get("consensus_reached"),
                "decision": result.get("decision"),
                "confidence": result.get("confidence"),
            },
            "metrics": {
                "total_events": self.metrics["events"],
                "total_rounds": self.metrics["rounds"],
                "knowledge_items_used": self.metrics["knowledge_items"],
            },
            "convergence": self.convergence_data,
        }

        print(f"\n{'=' * 50}")
        print("WORKFLOW COMPLETE")
        print(f"{'=' * 50}")
        print(f"Topic: {topic}")
        print(f"Consensus: {result.get('consensus_reached')}")
        print(f"Decision: {result.get('decision', 'N/A')[:80]}...")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"Rounds: {self.metrics['rounds']}")
        print(f"Events processed: {self.metrics['events']}")
        print(f"Knowledge items: {self.metrics['knowledge_items']}")

        return report


async def run_advanced_workflow(topic: str, dry_run: bool = False) -> dict:
    """Execute the complete advanced workflow."""

    orchestrator = WorkflowOrchestrator(dry_run=dry_run)

    # Step 1: Gather knowledge
    knowledge_context = await orchestrator.gather_knowledge(topic)

    # Step 2: Configure agents
    agents = orchestrator.configure_agents()

    # Step 3: Run streaming debate
    result = await orchestrator.run_streaming_debate(topic, agents, knowledge_context)

    # Step 4: Store evidence
    await orchestrator.store_evidence(result, topic)

    # Step 5: Generate report
    report = orchestrator.generate_report(topic, result)

    return report


def main():
    parser = argparse.ArgumentParser(description="Run advanced debate workflow")
    parser.add_argument(
        "--topic",
        default="How should we handle PII in our ML pipeline?",
        help="Topic for the debate",
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    args = parser.parse_args()

    result = asyncio.run(run_advanced_workflow(args.topic, args.dry_run))
    return result


if __name__ == "__main__":
    main()
