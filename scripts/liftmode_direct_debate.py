#!/usr/bin/env python3
"""
LiftMode business turnaround debate using direct API calls.
Bypasses Arena to avoid heavy initialization.
"""

import os
import sys

# Disable heavy ML imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/sentence_transformers'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Don't download models

import asyncio
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing modules...", flush=True)
from aragora.agents.registry import AgentRegistry
from aragora.agents.api_agents.gemini import GeminiAgent  # noqa: F401
from aragora.agents.api_agents.grok import GrokAgent  # noqa: F401
print("Imports complete.", flush=True)


# =============================================================================
# BUSINESS CONTEXT
# =============================================================================

COMPANY_CONTEXT = """
## Company Overview
- **Company**: Synaptent LLC d/b/a LiftMode.com
- **Founded**: 2010, Chicago IL
- **Business**: E-commerce supplements/nootropics
- **Building**: Owned by co-owner (rent accrued as debt, not cash)

## Current Financial Situation (2025 P&L Analysis)
- **Revenue**: $778,642/year (~$65k/month)
- **Gross Margin**: 54.8% ($426k gross profit)
- **Net Loss**: -$634,385/year (~$53k/month reported loss)
- **Actual Cash Burn**: ~$24k/month (rent is "soft debt" to building owner)

## Major Expenses (Monthly)
| Expense | Amount | % of Revenue |
|---------|--------|--------------|
| Rent (accrued to co-owner) | $21,000 | 32% |
| Admin/Professional | $19,200 | 30% |
| Director of Operations | $15,000 | 23% |
| Software/Tech | $8,000-12,700 | 12-20% |
| Marketing | $1,100 | 2% |

## Balance Sheet Highlights
- **Total Assets**: $2.08M (of which $1.5M is inventory)
- **Loan from Owner**: $776k (funding losses)
- **Inventory**: ~2 years of stock at current sales rate

## Critical Business Event
- **November 2023**: Phenibut discontinued due to FDA pressure
- **Pre-phenibut revenue**: ~$327k/month
- **Post-phenibut revenue**: ~$65k/month (80% decline)

## Top Products (Post-Phenibut Era)
1. MT55 Kanna Extract - $120k revenue (67-75% margin)
2. Kava Honey products - $28k (44-66% margin)
3. MoodLift Capsules - $20k (85% margin)
4. Energy Caps - $17k (73% margin)

## Breakeven Target: End of 2026
"""

DEBATE_PROMPT = """
# Strategic Business Decision: Synaptent LLC / LiftMode.com

## The Challenge
LiftMode.com is burning approximately $24,000/month in actual cash. The company has been funding losses through owner loans totaling $776k. Revenue dropped from ~$327k/month to ~$65k/month after Phenibut discontinuation in November 2023.

## Key Decisions Needed

1. **Personnel**: Director of Operations costs $15k/month. Retain, reduce, or eliminate?

2. **Revenue Growth**: Marketing spend is only $1,100/month (2% of revenue). Where should focus be?
   - Double down on Kanna (proven demand, 67-75% margin)
   - Push high-margin products (PPAP, PEA, Icariin at 90%+ margin)
   - Develop B2B/wholesale channel
   - International expansion

3. **Cost Reduction**: Beyond personnel, what can be cut?
   - Software costs ($8-12k/month)
   - Professional fees ($19k/month)

4. **Strategic Direction**:
   - Execute aggressive turnaround
   - Seek acquisition/merger
   - Orderly wind-down
   - Pivot business model

## Business Context
{context}

Please provide a comprehensive strategic recommendation that:
1. Is financially viable (achieves path to breakeven)
2. Is executable with current resources
3. Shows month-by-month milestones
4. Identifies risks and contingencies

Be specific with numbers and timelines.
"""


async def run_direct_debate():
    """Run debate using direct API calls."""
    print("=" * 80)
    print("LIFTMODE BUSINESS TURNAROUND DEBATE (Direct API)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Create output directory
    output_dir = Path("output/liftmode_debate")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create agents
    print("\nCreating agents...", flush=True)

    agents = []

    # Gemini 3 Pro
    try:
        gemini = AgentRegistry.create(
            "gemini",
            name="strategic-consultant",
            model="gemini-3-pro-preview",
            role="proposer",
            use_cache=False,
            timeout=300,
        )
        gemini.system_prompt = "You are a strategic business consultant specializing in e-commerce turnarounds. Focus on practical, executable recommendations with clear financial justification."
        agents.append(gemini)
        print("  Added: Gemini 3 Pro (strategic-consultant)", flush=True)
    except Exception as e:
        print(f"  Gemini not available: {e}", flush=True)

    # Grok 4
    try:
        grok = AgentRegistry.create(
            "grok",
            name="cfo-advisor",
            model="grok-4",
            role="critic",
            use_cache=False,
            timeout=300,
        )
        grok.system_prompt = "You are a CFO with expertise in distressed company operations. Prioritize cash flow preservation and realistic revenue projections. Challenge assumptions and identify risks."
        agents.append(grok)
        print("  Added: Grok 4 (cfo-advisor)", flush=True)
    except Exception as e:
        print(f"  Grok not available: {e}", flush=True)

    if len(agents) < 2:
        print("\nError: Need at least 2 agents. Check API keys.")
        return None

    # Prepare the debate prompt
    full_prompt = DEBATE_PROMPT.format(context=COMPANY_CONTEXT)

    # Run multiple rounds
    rounds = []
    context = []  # Conversation context

    for round_num in range(1, 5):  # 4 rounds
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print("=" * 60)

        round_messages = []

        for agent in agents:
            print(f"\n{agent.name} thinking...", flush=True)

            # Build prompt with context
            if round_num == 1:
                prompt = full_prompt
            else:
                # Include previous responses
                prev_context = "\n\n".join([
                    f"**{m['agent']}**: {m['content'][:500]}..."
                    for m in round_messages[-len(agents):]
                ])
                prompt = f"""Based on the previous discussion:

{prev_context}

Please provide your refined perspective on the strategic recommendations for LiftMode.com.
Focus on areas of agreement and remaining concerns.
Be specific about implementation priorities and timeline.

Original question: {full_prompt[:1000]}..."""

            try:
                # Generate response
                response = await agent.generate(prompt, context=None)

                round_messages.append({
                    "agent": agent.name,
                    "round": round_num,
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                })

                print(f"\n{agent.name} response ({len(response)} chars):", flush=True)
                print(response[:500] + "..." if len(response) > 500 else response, flush=True)

            except Exception as e:
                print(f"Error from {agent.name}: {e}", flush=True)
                round_messages.append({
                    "agent": agent.name,
                    "round": round_num,
                    "content": f"Error: {e}",
                    "timestamp": datetime.now().isoformat(),
                })

        rounds.append(round_messages)

    # Save transcript
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = output_dir / f"transcript_{timestamp}.md"

    with open(transcript_path, "w") as f:
        f.write("# LiftMode Business Turnaround Debate\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Agents**: {', '.join(a.name for a in agents)}\n\n")
        f.write("---\n\n")
        f.write("## Business Context\n\n")
        f.write(COMPANY_CONTEXT)
        f.write("\n\n---\n\n")
        f.write("## Debate Transcript\n\n")

        for i, round_msgs in enumerate(rounds, 1):
            f.write(f"### Round {i}\n\n")
            for msg in round_msgs:
                f.write(f"**{msg['agent']}**:\n\n{msg['content']}\n\n---\n\n")

    print(f"\n\nTranscript saved to: {transcript_path}")

    # Save JSON for further processing
    json_path = output_dir / f"debate_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "agents": [a.name for a in agents],
            "context": COMPANY_CONTEXT,
            "rounds": rounds,
        }, f, indent=2)

    print(f"JSON saved to: {json_path}")

    return rounds


if __name__ == "__main__":
    print("Starting LiftMode debate...", flush=True)
    result = asyncio.run(run_direct_debate())
    print("\nDebate completed.", flush=True)
