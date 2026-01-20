#!/usr/bin/env python3
"""
Strategic Positioning Debate for Aragora.

Runs a multi-agent debate on Aragora's market positioning and generates
an audio file of the debate for listening.

Usage:
    python scripts/strategic_debate.py
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_strategic_debate():
    """Run a debate on Aragora's strategic positioning."""
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.core import Environment
    from aragora.agents.base import create_agent

    # The strategic question based on the analysis provided
    task = """
Aragora is a multi-agent debate framework with impressive technical depth but unclear market positioning.

CRITICAL STRATEGIC QUESTIONS TO DEBATE:

1. WHO IS THE CUSTOMER? What's their hair-on-fire problem that Aragora solves?
   - Current: AI researchers (small market) or "people wanting multiple AI perspectives" (vague)
   - Better question: Who would pay $100/month TODAY?

2. WHAT'S THE 10-WORD PITCH?
   - Current implicit pitch is a feature list, not a value proposition
   - Need: "__________ for __________" (like "Cursor: AI-first code editor")

3. WHAT'S THE WEDGE?
   - Every successful dev tool has ONE thing it does 10x better
   - The Nomic Loop is unique but unexplained
   - The debate framework is cool but when do you actually NEED it?

4. WHERE'S THE MAGIC MOMENT?
   - For Cursor: First time AI autocompletes a complex function
   - For ChatGPT: First time it explains your code perfectly
   - For Aragora: ???

STRATEGIC OPTIONS TO EVALUATE:

A) "Second Opinion" Tool - "Five AI experts debate your decision before you commit"
B) "Omnivorous Decision Engine" - "Any source, any channel, multi-agent consensus"
C) "Self-Healing Codebase" - "Your codebase autonomously fixes bugs and improves itself"
D) "AI Parliament" for Content - "AI content that shows all sides, not just one"

Debate these options and provide a concrete recommendation with:
- Clear target customer persona
- 10-word pitch
- The wedge (what's 10x better)
- The magic moment
- First 3 features to build
"""

    context = """
Background on Aragora's current capabilities:
- 163K+ lines of production Python code
- 15,000+ tests with comprehensive coverage
- 12+ LLM providers (Anthropic, OpenAI, Mistral, DeepSeek, etc.)
- Nomic Loop for autonomous self-improvement
- Real-time debate visualization via WebSocket
- ELO ranking system for agent performance
- Memory tiers (fast/medium/slow/glacial) for learning
- Consensus mechanisms (majority, unanimous, judge)

The technology is impressive. The positioning needs work.
What concrete positioning should Aragora pursue to find product-market fit?
"""

    env = Environment(task=task, context=context)

    # Create diverse agents for multi-perspective debate
    logger.info("Creating debate agents...")
    agents = []

    # Try to create agents based on available API keys
    agent_configs = [
        ("anthropic-api", "claude_strategist", "proposer"),
        ("openai-api", "gpt_analyst", "critic"),
        ("openrouter", "deepseek_synthesizer", "synthesizer"),
    ]

    for agent_type, name, role in agent_configs:
        try:
            agent = create_agent(agent_type, name=name, role=role)
            agents.append(agent)
            logger.info(f"  Created {name} ({agent_type})")
        except Exception as e:
            logger.warning(f"  Could not create {name}: {e}")

    if len(agents) < 2:
        logger.error("Need at least 2 agents to run a debate")
        return None

    # Configure debate protocol for strategic discussion
    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        asymmetric_stances=True,  # Force perspective diversity
        topology="all-to-all",  # Everyone critiques everyone
        agreement_intensity=5,  # Balanced (not too adversarial or collaborative)
        enable_research=False,  # Focus on the provided context
        early_stopping=True,
        early_stop_threshold=0.7,
        convergence_detection=True,
        timeout_seconds=600,  # 10 minute max
    )

    logger.info(f"Starting strategic positioning debate with {len(agents)} agents...")
    logger.info(f"  Rounds: {protocol.rounds}")
    logger.info(f"  Consensus: {protocol.consensus}")

    arena = Arena(env, agents, protocol)
    result = await arena.run()

    return result


async def generate_audio(result, output_dir: Path):
    """Generate audio from debate result."""
    try:
        from aragora.broadcast import broadcast_debate
        from aragora.debate.traces import DebateTrace

        # Create a trace from the result for the broadcast system
        trace_data = {
            "debate_id": f"strategic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "task": result.task if hasattr(result, "task") else "Strategic positioning debate",
            "events": [],
            "final_answer": result.final_answer,
            "consensus_reached": result.consensus_reached,
            "confidence": result.confidence,
        }

        # Add messages as events
        for i, msg in enumerate(result.messages):
            trace_data["events"].append(
                {
                    "type": "proposal",
                    "agent": msg.agent,
                    "content": msg.content,
                    "round": (
                        msg.round_num
                        if hasattr(msg, "round_num")
                        else i // len(result.messages) + 1
                    ),
                }
            )

        # Save trace
        trace_path = output_dir / "debate_trace.json"
        with open(trace_path, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)

        logger.info(f"Saved debate trace to {trace_path}")

        # Try to generate audio
        trace = DebateTrace.load(trace_path)
        audio_path = await broadcast_debate(trace, output_dir=output_dir)

        return audio_path

    except ImportError as e:
        logger.warning(f"Audio generation requires broadcast dependencies: {e}")
        logger.info("Install with: pip install aragora[broadcast]")
        return None
    except Exception as e:
        logger.warning(f"Could not generate audio: {e}")
        return None


def save_results(result, output_dir: Path):
    """Save debate results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full result as JSON
    result_path = output_dir / "debate_result.json"
    result_data = {
        "final_answer": result.final_answer,
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "rounds_used": result.rounds_used,
        "duration_seconds": result.duration_seconds,
        "dissenting_views": result.dissenting_views if hasattr(result, "dissenting_views") else [],
        "votes": (
            [{"agent": v.agent, "choice": v.choice, "reasoning": v.reasoning} for v in result.votes]
            if result.votes
            else []
        ),
        "generated_at": datetime.now().isoformat(),
    }

    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"Saved result JSON to {result_path}")

    # Save readable summary
    summary_path = output_dir / "debate_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Aragora Strategic Positioning Debate\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Consensus Reached:** {'Yes' if result.consensus_reached else 'No'}\n")
        f.write(f"**Confidence:** {result.confidence:.0%}\n")
        f.write(f"**Rounds Used:** {result.rounds_used}\n")
        f.write(f"**Duration:** {result.duration_seconds:.1f}s\n\n")
        f.write("---\n\n")
        f.write("## Final Strategic Recommendation\n\n")
        f.write(result.final_answer)
        f.write("\n\n---\n\n")

        if hasattr(result, "dissenting_views") and result.dissenting_views:
            f.write("## Alternative Perspectives\n\n")
            for i, view in enumerate(result.dissenting_views, 1):
                f.write(f"### Alternative {i}\n\n")
                f.write(view)
                f.write("\n\n")

        if result.votes:
            f.write("## Voting Breakdown\n\n")
            for vote in result.votes:
                f.write(f"- **{vote.agent}**: {vote.choice}\n")
                if vote.reasoning:
                    f.write(f"  - *Reasoning:* {vote.reasoning[:200]}...\n")
            f.write("\n")

    logger.info(f"Saved readable summary to {summary_path}")

    return result_path, summary_path


async def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("ARAGORA STRATEGIC POSITIONING DEBATE")
    print("=" * 70 + "\n")

    output_dir = Path(".nomic/strategic_debate")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the debate
    result = await run_strategic_debate()

    if not result:
        logger.error("Debate failed to produce results")
        return 1

    # Display results
    print("\n" + "=" * 70)
    print("DEBATE RESULTS")
    print("=" * 70)
    print(f"\nConsensus Reached: {'Yes' if result.consensus_reached else 'No'}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Rounds Used: {result.rounds_used}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print("\n" + "-" * 70)
    print("STRATEGIC RECOMMENDATION:")
    print("-" * 70)
    print(result.final_answer)
    print("-" * 70 + "\n")

    # Save results
    save_results(result, output_dir)

    # Try to generate audio
    print("\nGenerating audio version of debate...")
    audio_path = await generate_audio(result, output_dir)

    if audio_path:
        print(f"\nAudio saved to: {audio_path}")
        print("You can listen to the debate using any audio player.")
    else:
        print("\nAudio generation skipped (install aragora[broadcast] for audio)")

    print(f"\nAll outputs saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
