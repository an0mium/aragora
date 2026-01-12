#!/usr/bin/env python3
"""
LiftMode Expanded Strategy Debate - Multi-Agent Discussion
Topics: DOO Elimination, B2B Strategy, Marketing, Product Innovation, Aragora as Revenue
"""

import os
import sys

# Disable heavy ML imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp/sentence_transformers'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import asyncio
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing modules...", flush=True)
from aragora.agents.registry import AgentRegistry
from aragora.agents.api_agents.gemini import GeminiAgent  # noqa: F401
from aragora.agents.api_agents.grok import GrokAgent  # noqa: F401
from aragora.agents.api_agents.anthropic import AnthropicAPIAgent  # noqa: F401
from aragora.agents.api_agents.openrouter import OpenRouterAgent, DeepSeekReasonerAgent  # noqa: F401
print("Imports complete.", flush=True)


# =============================================================================
# EXPANDED BUSINESS CONTEXT WITH PRODUCT DATA
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

## PRODUCT ANALYSIS (Post-Phenibut Era: Nov 2023 - Present)
- Total Revenue Analyzed: $1,856,620
- Unique Products: 515 SKUs
- Average Margin: 63.2%

### Top Categories by Revenue:
1. **Kanna Products**: $605,777 (33% of revenue, 62% margin) - THE ANCHOR
2. **Powders (Other)**: $435,791 (24%, 79% margin)
3. **Kava Products**: $151,745 (8%, 57% margin)
4. **PEA**: $76,441 (4%, 87% margin) - HIGH MARGIN UPSELL
5. **Icariin**: $67,381 (4%, 80% margin) - HIGH MARGIN UPSELL
6. **MoodLift**: $56,583 (3%, 72% margin)

### Top 10 Products (Post-Phenibut):
1. MT55 Kanna Extract 5g - $150,625 (63% margin)
2. MT55 Kanna Extract 2g - $91,261 (64% margin)
3. MT55 Kanna Extract 1g - $77,784 (63% margin)
4. Kava Honey 8 oz - $51,640 (62% margin)
5. MoodLift Capsules 30ct - $49,073 (80% margin)
6. PEA 200g - $48,536 (80% margin)
7. Kava Extract Powder 30g - $46,658 (73% margin)
8. Baicalein Powder 20g - $39,514 (81% margin)
9. Energy Caps 120ct - $38,159 (66% margin)
10. Baicalm 120ct - $37,085 (56% margin)

### GROWTH STARS (Trending Up >20%):
- Bitter Orange Extract Capsules (+305%)
- NACET 96% (+250%)
- Magnolia Bark Extract (+110%)
- Caffeine + L-Theanine Caps (+101%)
- L-THP Tablets (+66%)
- Icariin Tablets (+63%)

### DECLINING PRODUCTS (Need Action):
- 5-HTP Powder (-92%)
- Kanna Drink Strawberry (-84%)
- NMN 10g (-80%)
- Kanna XK6 5g (-79%)

### HIGH-MARGIN UPSELL OPPORTUNITIES (>80% margin):
- MoodLift Capsules (80% margin, $49k)
- PEA 200g (80% margin, $49k)
- Sleep Caps (82% margin, $32k)
- Icariin 98% (81% margin, $17k)
- Oleamide (87% margin, $19k)

## MONTHLY TREND (Recent):
- Last 3 months revenue: $143,812
- Prior 3 months revenue: $186,518
- Trend: -22.9% (CONCERNING)

## Balance Sheet Highlights
- **Total Assets**: $2.08M (of which $1.5M is inventory)
- **Loan from Owner**: $776k (funding losses)
- **Inventory**: ~2 years of stock at current sales rate

## Critical Business Event
- **November 2023**: Phenibut discontinued due to FDA pressure
- **Pre-phenibut revenue**: ~$327k/month
- **Post-phenibut revenue**: ~$65k/month (80% decline)

## ARAGORA - POTENTIAL FUTURE REVENUE SOURCE
- December software expense includes Aragora development
- Aragora is a multi-agent AI debate platform (this debate is running on it)
- Could be commercialized as SaaS product for enterprise decision-making
- Target markets: Consulting firms, strategy teams, research orgs, product teams
- Current stage: Internal tool, needs productization
"""

DEBATE_TOPICS = {
    "doo_elimination": """
## TOPIC 1: Director of Operations Elimination - Execution Strategy

The previous debate reached consensus that the $15k/month DOO role must be eliminated.
This is the single biggest lever for cash flow (62% of monthly burn).

### Key Questions:
1. **How to execute the transition without operational disruption?**
   - What specific duties need to be covered?
   - Who takes over what (owner, warehouse lead, outsourced)?
   - What's the 30/60/90 day handover plan?

2. **What's the communication strategy?**
   - Severance considerations?
   - How to handle institutional knowledge transfer?

3. **Risk mitigation:**
   - What if fulfillment quality drops?
   - How do we monitor for issues?
   - Contingency if the owner can't handle increased workload?

4. **Financial modeling:**
   - If we promote warehouse lead with $500-1k raise, net savings = $14-14.5k/mo
   - Breakeven impact?
""",

    "b2b_and_marketing": """
## TOPIC 2: B2B Wholesale Strategy & Marketing Growth

The company spends only $1,100/month on marketing (2% of revenue) which is effectively zero.
Kanna is 33% of revenue with 62% margin - it's the hero product.

### Key Questions for B2B Strategy:
1. **Which products are best suited for wholesale?**
   - Kanna MT55 (top seller, proven demand)
   - Kava products (fits wellness/bar niche)
   - Any others?

2. **Target customers:**
   - Kava bars
   - Vape/smoke shops
   - Wellness boutiques
   - Other supplement brands (white label?)
   - Gyms/fitness centers

3. **Pricing & Terms:**
   - What discount for volume (20-30%)?
   - Payment terms (Net 30?)
   - Minimum orders?

4. **Execution:**
   - Who does outreach (owner? commission-only rep?)
   - Sales materials needed?
   - Timeline to first B2B revenue?

### Key Questions for Marketing:
1. **Budget:**
   - Previous debate suggested $5-10k/month (funded by DOO savings)
   - What's the right allocation?

2. **Channel priorities:**
   - SEO/Content for Kanna (long-term)
   - Email reactivation of legacy Phenibut customers
   - Influencer/affiliate (biohackers, nootropics YouTubers)
   - Paid ads (Google, Meta)?

3. **Product focus:**
   - Lead with Kanna (hero product)
   - Upsell high-margin products (PEA, Icariin, MoodLift)
   - Bundles/stacks (Focus Stack, Mood Stack, Sleep Stack)?

4. **Metrics:**
   - Target CAC?
   - Expected ROI per channel?
   - How long until payback?
""",

    "product_innovation": """
## TOPIC 3: New Product Development & Revenue Diversification

The company has strong margins but is overly dependent on Kanna.
Recent trend shows -22.9% revenue decline in last 3 months vs prior 3 months.

### Product Opportunities:
1. **Functional Stacks (Bundles):**
   - Mood Stack: Kanna + MoodLift + L-Theanine
   - Focus Stack: Kanna + Caffeine + L-THP
   - Sleep Stack: Magnolia + Oleamide + Melatonin
   - Energy Stack: Caffeine + L-Theanine + PEA

2. **Format Innovation:**
   - Ready-to-drink Kanna beverages (existing Kanna Drink is declining -84%)
   - Gummies (high margin, consumer-friendly)
   - Sublingual drops
   - Stick packs for convenience

3. **New Product Lines:**
   - Functional mushrooms (Lion's Mane, Reishi) - regulatory safe
   - Adaptogens (more Ashwagandha variations)
   - Sports/fitness supplements (pre-workout stacks)

4. **Inventory Optimization:**
   - What to liquidate (declining products)?
   - Bundle slow movers with best sellers?
   - How to turn $1.5M inventory into cash?

### Key Questions:
1. What's the investment required for new products?
2. Timeline to launch?
3. Which has best ROI potential?
4. Regulatory considerations?
""",

    "aragora_revenue": """
## TOPIC 4: Aragora as Future Revenue Stream

December software expenses include development of Aragora - the multi-agent AI debate platform
powering this very discussion. This represents a potential pivot to B2B SaaS revenue.

### What is Aragora?
- Multi-agent debate framework where AI agents discuss, critique, and improve responses
- Uses heterogeneous AI models (Claude, Gemini, Grok, etc.)
- Implements self-improvement through the "Nomic Loop"
- Memory systems, consensus detection, ELO rankings for agents

### Potential Markets:
1. **Consulting Firms:**
   - Strategic decision support
   - Scenario analysis
   - Multi-perspective research

2. **Enterprise Strategy Teams:**
   - Business case evaluation
   - Risk assessment debates
   - M&A due diligence

3. **Research Organizations:**
   - Literature review synthesis
   - Hypothesis generation
   - Peer review simulation

4. **Product Teams:**
   - Feature prioritization debates
   - Design decision analysis
   - Technical architecture review

### Key Questions:
1. **What's the go-to-market strategy?**
   - SaaS platform vs. consulting services?
   - Pricing model (usage-based, subscription)?
   - Target customer profile?

2. **Product readiness:**
   - What needs to be built for commercialization?
   - Security/compliance requirements?
   - Documentation/onboarding?

3. **Financial projections:**
   - Development cost to MVP?
   - Potential ARR if successful?
   - Time to revenue?

4. **Strategic fit:**
   - Does this distract from LiftMode turnaround?
   - Could Aragora revenue eventually exceed supplements?
   - Resource allocation between both businesses?
"""
}


async def run_expanded_debate():
    """Run expanded debate with 4 agents on multiple topics."""
    print("=" * 80)
    print("LIFTMODE EXPANDED STRATEGY DEBATE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Create output directory
    output_dir = Path("output/liftmode_debate")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create agents
    print("\nCreating agents...", flush=True)
    agents = []

    # Agent 1: Gemini 3 Pro - Strategic Consultant
    try:
        gemini = AgentRegistry.create(
            "gemini",
            name="strategic-consultant",
            model="gemini-3-pro-preview",
            role="proposer",
            use_cache=False,
            timeout=300,
        )
        gemini.system_prompt = """You are a strategic business consultant specializing in e-commerce turnarounds and growth strategy.
Focus on practical, executable recommendations with clear financial justification.
Consider both short-term survival and long-term growth opportunities."""
        agents.append(gemini)
        print("  Added: Gemini 3 Pro (strategic-consultant)", flush=True)
    except Exception as e:
        print(f"  Gemini not available: {e}", flush=True)

    # Agent 2: Grok 4 - CFO Advisor
    try:
        grok = AgentRegistry.create(
            "grok",
            name="cfo-advisor",
            model="grok-4",
            role="critic",
            use_cache=False,
            timeout=300,
        )
        grok.system_prompt = """You are a CFO with expertise in distressed company operations and cash flow management.
Prioritize cash preservation and realistic revenue projections. Challenge optimistic assumptions.
Focus on unit economics and break-even analysis."""
        agents.append(grok)
        print("  Added: Grok 4 (cfo-advisor)", flush=True)
    except Exception as e:
        print(f"  Grok not available: {e}", flush=True)

    # Agent 3: Claude - Marketing & Innovation Lead
    try:
        claude = AgentRegistry.create(
            "anthropic-api",
            name="marketing-lead",
            model="claude-sonnet-4-20250514",
            role="proposer",
            use_cache=False,
            timeout=300,
        )
        claude.system_prompt = """You are a marketing and product innovation expert specializing in DTC e-commerce and supplement brands.
Focus on customer acquisition, retention, and new product development.
Consider brand positioning, channel strategy, and product-market fit."""
        agents.append(claude)
        print("  Added: Claude Sonnet (marketing-lead)", flush=True)
    except Exception as e:
        print(f"  Claude not available: {e}", flush=True)

    # Agent 4: DeepSeek - Technology & Operations
    try:
        deepseek = AgentRegistry.create(
            "deepseek-r1",
            name="tech-ops-advisor",
            role="critic",
            use_cache=False,
            timeout=300,
        )
        deepseek.system_prompt = """You are a technology and operations expert with experience in B2B SaaS and e-commerce operations.
Focus on operational efficiency, technology investments, and scalability.
Evaluate the Aragora opportunity from a tech product perspective."""
        agents.append(deepseek)
        print("  Added: DeepSeek R1 (tech-ops-advisor)", flush=True)
    except Exception as e:
        print(f"  DeepSeek not available: {e}", flush=True)

    if len(agents) < 2:
        print("\nError: Need at least 2 agents. Check API keys.")
        return None

    print(f"\n{len(agents)} agents ready for debate.")

    # Run debate on each topic
    all_results = {}

    topics_to_run = ["doo_elimination", "b2b_and_marketing", "product_innovation", "aragora_revenue"]

    for topic_key in topics_to_run:
        topic_prompt = DEBATE_TOPICS[topic_key]

        print(f"\n{'='*80}")
        print(f"TOPIC: {topic_key.upper().replace('_', ' ')}")
        print("=" * 80)

        rounds = []

        for round_num in range(1, 3):  # 2 rounds per topic to save time
            print(f"\n--- Round {round_num} ---")
            round_messages = []

            for agent in agents:
                print(f"\n{agent.name} thinking...", flush=True)

                if round_num == 1:
                    prompt = f"""# LiftMode Strategic Analysis

## Business Context
{COMPANY_CONTEXT}

{topic_prompt}

Please provide your analysis and recommendations. Be specific with numbers, timelines, and actionable steps.
"""
                else:
                    prev_responses = "\n\n".join([
                        f"**{m['agent']}**: {m['content'][:800]}..."
                        for m in round_messages
                    ])
                    prompt = f"""Based on the previous responses from other advisors:

{prev_responses}

Please provide your refined perspective. Focus on:
1. Areas of agreement
2. Remaining concerns or disagreements
3. Specific next steps you recommend

Be concise but specific."""

                try:
                    response = await agent.generate(prompt, context=None)
                    round_messages.append({
                        "agent": agent.name,
                        "round": round_num,
                        "content": response,
                        "timestamp": datetime.now().isoformat(),
                    })
                    print(f"\n{agent.name} ({len(response)} chars):", flush=True)
                    print(response[:400] + "..." if len(response) > 400 else response, flush=True)
                except Exception as e:
                    print(f"Error from {agent.name}: {e}", flush=True)
                    round_messages.append({
                        "agent": agent.name,
                        "round": round_num,
                        "content": f"Error: {e}",
                        "timestamp": datetime.now().isoformat(),
                    })

            rounds.append(round_messages)

        all_results[topic_key] = rounds

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save transcript
    transcript_path = output_dir / f"expanded_debate_{timestamp}.md"
    with open(transcript_path, "w") as f:
        f.write("# LiftMode Expanded Strategy Debate\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Agents**: {', '.join(a.name for a in agents)}\n\n")
        f.write("---\n\n")
        f.write("## Business Context\n\n")
        f.write(COMPANY_CONTEXT)
        f.write("\n\n---\n\n")

        for topic_key, rounds in all_results.items():
            f.write(f"# {topic_key.upper().replace('_', ' ')}\n\n")
            f.write(DEBATE_TOPICS[topic_key])
            f.write("\n\n## Debate Transcript\n\n")

            for i, round_msgs in enumerate(rounds, 1):
                f.write(f"### Round {i}\n\n")
                for msg in round_msgs:
                    f.write(f"**{msg['agent']}**:\n\n{msg['content']}\n\n---\n\n")

    print(f"\n\nTranscript saved to: {transcript_path}")

    # Save JSON
    json_path = output_dir / f"expanded_debate_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "agents": [a.name for a in agents],
            "context": COMPANY_CONTEXT,
            "topics": all_results,
        }, f, indent=2)

    print(f"JSON saved to: {json_path}")

    return all_results


if __name__ == "__main__":
    print("Starting LiftMode expanded debate...", flush=True)
    result = asyncio.run(run_expanded_debate())
    print("\nDebate completed.", flush=True)
