#!/usr/bin/env python3
"""
Aragora Identity Debate V2 - Enhanced Dogfooding with Full Agent Diversity

This script runs a multi-agent debate to define Aragora's core identity.
It demonstrates the system's self-awareness by injecting comprehensive
context about its own capabilities.

V2 Improvements:
- OpenRouter fallback explicitly enabled for all native API agents
- 8 frontier agents from diverse providers (not just 5)
- Prior debate context injection to build on previous insights
- Enhanced self-knowledge emphasizing data ingestion, bidirectional comms, memory

Usage:
    python scripts/identity_debate.py

Requirements:
    - OPENROUTER_API_KEY (required - provides fallback + frontier models)
    - ANTHROPIC_API_KEY (optional - will fallback to OpenRouter if exhausted)
    - OPENAI_API_KEY (optional - will fallback to OpenRouter if exhausted)
    - GEMINI_API_KEY (optional)
    - XAI_API_KEY (optional)
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Comprehensive self-knowledge context for agents (V2 - enhanced emphasis on data/comms/memory)
ARAGORA_SELF_KNOWLEDGE = """
## Aragora Codebase Facts (as of 2026-01-22)

**Scale:**
- 710,762 lines of production code
- 734,919 lines of test code
- 1,543 Python modules across 97 subsystems
- 43,500+ passing tests
- Test-to-code ratio: 1.03:1

**Core Capabilities:**

### 1. OMNIVOROUS DATA INGESTION (96% mature) - CRITICAL CAPABILITY
   - **Document Processing**: Unstructured.io + Docling integration for 25+ formats
   - **Supported Formats**: PDF, DOCX, XLSX, PPTX, HTML, Markdown, JSON, XML, CSV, TXT, RTF, ODT, EPUB
   - **Multimodal**: Images (PNG, JPG, WebP via vision models), audio (Whisper transcription), video (frame extraction)
   - **Specialized Sources**: SEC filings (EDGAR), ArXiv papers, GitHub repos, SQL databases, S3 buckets, SharePoint
   - **Chunking Strategies**: Semantic, fixed-size, sentence-based, paragraph-based with overlap control
   - **Embeddings**: OpenAI, Sentence-Transformers, Cohere for vector search
   - **24 data connectors** with bidirectional sync

### 2. BIDIRECTIONAL COMMUNICATION (91% mature) - CRITICAL CAPABILITY
   - **Inbound Channels**: Slack, Discord, Teams, Telegram, WhatsApp, Voice (WebRTC), GitHub webhooks, REST API, WebSocket
   - **Outbound Routing**: Debate results automatically routed back to originating channel
   - **80+ WebSocket events** for real-time streaming
   - **Audience Participation**: Live voting, suggestions, and feedback during debates
   - **Voice Integration**: Speech-to-text input, text-to-speech output
   - **GitHub Integration**: PR comments, issue responses, code review feedback

### 3. INSTITUTIONAL MEMORY & LEARNING (92% mature) - CRITICAL CAPABILITY
   - **4-Tier Memory Architecture**:
     - Fast (1 min TTL): Immediate context within debate
     - Medium (1 hour TTL): Session-level learning
     - Slow (7 day TTL): Cross-session patterns
     - Glacial (30 day TTL): Long-term institutional knowledge
   - **Surprise-Based Retention**: Unexpected findings prioritized for longer storage
   - **ConsensusMemory**: Tracks settled vs contested topics across all debates
   - **CritiqueStore**: Per-agent performance history for calibration
   - **Knowledge Mound**: Unified graph with 9 bidirectional adapters
   - **Cross-Session Learning**: Agents remember outcomes from previous debates

### 4. LONG CONTEXT MASTERY (88% mature) - CRITICAL CAPABILITY
   - **RLM (Recursive Language Model)**: Hierarchical context compression via arXiv:2512.24601
   - **Context Windows**: Up to 2M tokens with Gemini 3 Pro
   - **Intelligent Chunking**: Preserves semantic boundaries during compression
   - **REPL Navigation**: Interactive drilling into compressed context
   - **Evidence Grounding**: Citations and provenance tracked across compressions

### 5. DEBATE ENGINE (95% mature)
   - Arena orchestrator with 9-round Hegelian protocol (Thesis->Antithesis->Synthesis)
   - 8 consensus mechanisms: majority, unanimous, judge, byzantine, weighted, supermajority, any, none
   - Convergence detection via semantic similarity
   - Crux identification for load-bearing assumptions
   - Trickster detection for hollow consensus

### 6. AGENT LAYER (98% mature)
   - 15+ AI providers: Claude, GPT, Gemini, Grok, Mistral, DeepSeek, Qwen, Kimi, Llama, Yi
   - OpenRouter fallback on ANY provider failure (429, 400 billing, timeouts)
   - Per-agent reliability weights and calibration tracking
   - Position tracking and flip detection

### 7. NOMIC LOOP - Self-Improvement (60% mature - experimental)
   - Autonomous self-improvement: debate->design->implement->verify->commit
   - Constitutional verification (rules govern the rule-changer)
   - Protected file verification and automatic backups
   - Human approval gates for dangerous changes
   - Task decomposition for complex multi-step improvements

### 8. GAUNTLET (94% mature)
   - Adversarial stress-testing with attack types
   - Red team, devil's advocate, scaling critic, compliance personas
   - Decision receipts with cryptographic audit trails
   - 5 compliance presets: GDPR, HIPAA, SOX, AI Act, Security

### 9. ENTERPRISE (85-95% mature)
   - OIDC/SAML SSO (Azure AD, Okta, Google, Auth0, Keycloak)
   - RBAC v2 with 6 default roles and 50+ permissions
   - Multi-tenancy with quota tracking and cost metering
   - Immutable audit logging

**Architectural Principles:**
- **Dialectical Reasoning**: Contradiction drives improvement, not avoided
- **Omnivorous I/O**: Ingests from ANY source, outputs to ANY channel
- **Defensible**: Every decision produces audit-ready receipts with evidence chains
- **Self-Governing**: Nomic Loop enables autonomous improvement with constitutional constraints
- **Institutional**: Accumulates organizational knowledge across all interactions

**What Makes Aragora Unique (vs ChatGPT/Copilot/single-model wrappers):**
- NOT single-model: Heterogeneous 15+ provider ensemble argues toward truth
- NOT black-box: Full provenance and explainability for every decision
- NOT stateless: Institutional memory accumulates across debates and sessions
- NOT just chat: Structured Hegelian debate protocol with phases, roles, synthesis
- NOT passive: Self-improving via Nomic Loop with constitutional constraints
- NOT text-only: Multimodal ingestion (docs, images, audio, video) + multi-channel output
- NOT isolated: 24 data connectors + bidirectional chat platform integration

**Use Cases:**
- Code review at scale (24/7 multi-model consensus before human review)
- Compliance audits (GDPR, HIPAA, SOC 2 attack personas with audit trails)
- Architectural decisions (multi-perspective analysis with dissent recorded)
- Security red-teaming (systematic adversarial stress testing)
- Document analysis (cross-reference thousands of documents with citations)
- Inbox management (priority extraction, signal/noise separation)
- Enterprise knowledge repository (unified querying across all data sources)
"""

# Context from the previous identity debate (to build upon)
PRIOR_DEBATE_CONTEXT = """
## Prior Identity Debate (2026-01-22)

A previous debate with limited participation (only 1/5 agents responded due to API credit issues) produced:

**Tagline:** "Aragora: Debate-driven AI for defensible decisions and continuous self-improvement"

**Description:** "Aragora is an AI platform that uses structured debate across diverse agents
to arrive at well-reasoned, auditable decisions. Unlike simple chatbots, it fosters
institutional learning and autonomously improves through a self-governing Nomic Loop."

**Feedback on this framing:** It underemphasizes several CRITICAL capabilities:

1. **Knowledge & Data Ingestion** - Aragora is not just a debate engine, it's an OMNIVOROUS
   data platform that ingests 25+ document formats, images, audio, video, and connects to
   24 data sources. This is a key differentiator from chat-only tools.

2. **Bidirectional Communication** - Aragora receives queries FROM and sends results TO
   multiple channels (Slack, Discord, Teams, Telegram, WhatsApp, voice, GitHub). It's not
   just an API - it's embedded in the organization's communication flow.

3. **Institutional Memory** - The 4-tier memory system with surprise-based retention and
   cross-session learning means Aragora ACCUMULATES organizational knowledge. This is
   fundamentally different from stateless chat tools.

4. **Very Long Context** - RLM compression enables million-token contexts with intelligent
   navigation. This enables comprehensive document analysis that other tools cannot match.

Please address these gaps in your proposals. The ideal tagline and description should
capture ALL of these capabilities, not just the debate mechanism.
"""

DEBATE_TASK = """
Define Aragora's core identity in a single compelling sentence.

## Context
You are debating the identity of Aragora, a 710K+ line codebase with 43,500+ tests.
A previous debate produced a tagline that was criticized for underemphasizing key capabilities.
Your task is to improve upon it.

## Key Capabilities to Emphasize (often overlooked)
1. **Omnivorous Data Ingestion**: 25+ document formats, images, audio, video, 24 data connectors
2. **Institutional Memory**: 4-tier memory, surprise-based retention, Knowledge Mound, cross-session learning
3. **Bidirectional Communication**: Receives queries AND sends results to Slack, Discord, Teams, Telegram, WhatsApp, voice
4. **Debate & Synthesis**: Hegelian dialectics (Thesis->Antithesis->Synthesis), multi-agent consensus
5. **Long Context Mastery**: RLM compression, million-token contexts, intelligent navigation
6. **Self-Improvement**: Nomic Loop (debate->design->implement->verify with constitutional constraints)

## What Makes Aragora Unique (vs ChatGPT/Copilot)
- NOT a chatbot: Structured debate protocol with phases, roles, and evidence chains
- NOT a copilot: Institutional learning that ACCUMULATES organizational knowledge
- NOT single-model: Heterogeneous 15+ provider ensemble that argues toward truth
- NOT stateless: Remembers outcomes, builds knowledge graphs, improves itself
- NOT text-only: Multimodal ingestion + multi-channel bidirectional output

## Perspectives to Consider
1. **Technical**: What does the system actually do at its core?
2. **Business**: What problem does it solve for enterprises?
3. **Philosophical**: What is its intellectual foundation?
4. **Competitive**: How is it different from other AI tools?

## Deliverables
1. ONE recommended tagline (under 15 words) - must capture MORE than just "debate"
2. ONE full description (under 50 words) - must mention data ingestion AND institutional memory
3. Brief justification for why this framing is most compelling

Focus on what's unique and defensible, not generic AI buzzwords.
"""


async def create_agents():
    """Create diverse agents for the debate (V2 - 8 agents with fallback enabled).

    V2 Improvements:
    - All native API agents have enable_fallback=True
    - 4 additional frontier models via OpenRouter: Mistral, Qwen, Kimi, DeepSeek
    - Better error handling for missing providers
    """
    from aragora.agents.registry import AgentRegistry

    agents = []
    available_providers = []

    # Check which API keys are available
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_grok = bool(os.environ.get("XAI_API_KEY"))

    if has_anthropic:
        available_providers.append("anthropic")
    if has_openai:
        available_providers.append("openai")
    if has_openrouter:
        available_providers.append("openrouter")
    if has_gemini:
        available_providers.append("gemini")
    if has_grok:
        available_providers.append("grok")

    logger.info(f"Available providers: {available_providers}")

    if not available_providers:
        raise RuntimeError(
            "No API keys found. Set at least one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY"
        )

    # === NATIVE API AGENTS (with fallback enabled) ===

    # Claude - Philosophical depth, nuanced synthesis
    if has_anthropic:
        agents.append(
            AgentRegistry.create(
                "anthropic-api",
                name="claude_philosopher",
                role="synthesizer",
                model="claude-sonnet-4-20250514",
                enable_fallback=True,  # V2: Enable OpenRouter fallback
            )
        )
        logger.info("Added Claude (philosopher/synthesizer) with fallback")

    # GPT-4o - Pragmatic business framing
    if has_openai:
        agents.append(
            AgentRegistry.create(
                "openai-api",
                name="gpt_pragmatist",
                role="analyst",
                model="gpt-4o",
                enable_fallback=True,  # V2: Enable OpenRouter fallback
            )
        )
        logger.info("Added GPT-4o (pragmatist/analyst) with fallback")

    # Gemini - Creative alternatives, multimodal perspective
    if has_gemini:
        agents.append(
            AgentRegistry.create(
                "gemini",
                name="gemini_creative",
                role="proposer",
                model="gemini-2.0-flash",
                enable_fallback=True,  # V2: Enable OpenRouter fallback
            )
        )
        logger.info("Added Gemini (creative/proposer) with fallback")

    # Grok - Contrarian perspective, unconventional takes
    if has_grok:
        agents.append(
            AgentRegistry.create(
                "grok",
                name="grok_contrarian",
                role="devil_advocate",
                model="grok-2",
                enable_fallback=True,  # V2: Enable OpenRouter fallback
            )
        )
        logger.info("Added Grok (contrarian/devil's advocate) with fallback")

    # === OPENROUTER FRONTIER MODELS (only if OpenRouter key is set) ===

    if has_openrouter:
        # DeepSeek V3 - Technical precision, engineering focus
        agents.append(
            AgentRegistry.create(
                "openrouter",
                name="deepseek_engineer",
                role="critic",
                model="deepseek/deepseek-chat",
            )
        )
        logger.info("Added DeepSeek V3 (engineer/critic) via OpenRouter")

        # Mistral Large - European perspective, balanced analysis
        agents.append(
            AgentRegistry.create(
                "openrouter",
                name="mistral_analyst",
                role="analyst",
                model="mistralai/mistral-large-2411",
            )
        )
        logger.info("Added Mistral Large (analyst) via OpenRouter")

        # Qwen 3 235B - Large-scale reasoning, integrative thinking
        agents.append(
            AgentRegistry.create(
                "openrouter",
                name="qwen_integrator",
                role="synthesizer",
                model="qwen/qwen3-235b-a22b",
            )
        )
        logger.info("Added Qwen 3 235B (integrator/synthesizer) via OpenRouter")

        # Kimi K2 - Challenger perspective, fresh takes
        agents.append(
            AgentRegistry.create(
                "openrouter",
                name="kimi_challenger",
                role="devil_advocate",
                model="moonshotai/kimi-k2-instruct",
            )
        )
        logger.info("Added Kimi K2 (challenger/devil's advocate) via OpenRouter")
    else:
        logger.warning(
            "OpenRouter not available - skipping frontier models (Mistral, Qwen, Kimi, DeepSeek)"
        )

    # Ensure at least 2 agents for a meaningful debate
    if len(agents) < 2:
        logger.warning(f"Only {len(agents)} agents available. Need at least 2 for debate.")
        # Add additional OpenRouter models if we don't have enough and OpenRouter is available
        if has_openrouter and len(agents) < 2:
            agents.append(
                AgentRegistry.create(
                    "openrouter",
                    name="llama_generalist",
                    role="proposer",
                    model="meta-llama/llama-3.3-70b-instruct",
                )
            )
            logger.info("Added Llama 3.3 70B (generalist) via OpenRouter")

    if len(agents) < 2:
        raise RuntimeError(
            f"Only {len(agents)} agent(s) available. Need at least 2 for a debate. "
            "Add more API keys or set OPENROUTER_API_KEY."
        )

    logger.info(f"Created {len(agents)} agents for debate (V2 enhanced)")
    return agents


async def run_identity_debate():
    """Run the Aragora identity debate (V2 - with prior context)."""
    from aragora.core import Environment
    from aragora.debate.orchestrator import Arena
    from aragora.debate.protocol import DebateProtocol

    print("=" * 70)
    print("ARAGORA IDENTITY DEBATE V2")
    print("Enhanced: 8 agents, fallback enabled, prior context injected")
    print("=" * 70)
    print()

    # Combine self-knowledge with prior debate context (V2 enhancement)
    full_context = f"{ARAGORA_SELF_KNOWLEDGE}\n\n{PRIOR_DEBATE_CONTEXT}"

    # Create environment with task and comprehensive context
    env = Environment(task=DEBATE_TASK, context=full_context)

    # Configure debate protocol
    protocol = DebateProtocol(
        rounds=5,  # Reduced for faster execution; increase to 9 for full debate
        use_structured_phases=True,
        consensus="judge",
        topology="all-to-all",
        agreement_intensity=4,  # Slightly adversarial for diversity
        enable_trickster=True,  # Detect hollow consensus
        enable_calibration=True,
        timeout_seconds=900,  # 15 minutes max
        round_timeout_seconds=120,  # 2 min per round
        convergence_detection=True,
        convergence_threshold=0.92,
    )

    # Create agents
    agents = await create_agents()

    print(f"Participants: {', '.join(a.name for a in agents)}")
    print(f"Protocol: {protocol.rounds} rounds, {protocol.consensus} consensus")
    print()
    print("Starting debate...")
    print("-" * 70)

    # Run the debate
    arena = Arena(environment=env, agents=agents, protocol=protocol)
    start_time = datetime.now()

    try:
        result = await arena.run()
    except Exception as e:
        logger.error(f"Debate failed: {e}", exc_info=True)
        raise

    duration = (datetime.now() - start_time).total_seconds()

    # Display results
    print()
    print("=" * 70)
    print("DEBATE RESULTS")
    print("=" * 70)
    print()

    consensus_emoji = "Y" if result.consensus_reached else "X"
    print(
        f"[{consensus_emoji}] CONSENSUS: {'Reached' if result.consensus_reached else 'Not reached'}"
    )
    print(f"[*] CONFIDENCE: {result.confidence:.0%}")
    print(f"[i] DURATION: {duration:.0f}s ({result.rounds_used} rounds)")
    print(f"[i] PARTICIPANTS: {', '.join(result.participants)}")
    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print()
    print(result.final_answer)
    print()

    # Show disagreement analysis if available
    if hasattr(result, "disagreement_report") and result.disagreement_report:
        print("=" * 70)
        print("DISAGREEMENT ANALYSIS")
        print("=" * 70)
        print()
        if hasattr(result.disagreement_report, "summary"):
            print(result.disagreement_report.summary())
        else:
            print(str(result.disagreement_report))
        print()

    # Show cruxes if available
    if hasattr(result, "debate_cruxes") and result.debate_cruxes:
        print("=" * 70)
        print("KEY CRUXES (Load-bearing disagreements)")
        print("=" * 70)
        print()
        for i, crux in enumerate(result.debate_cruxes[:5], 1):
            print(f"{i}. {crux}")
        print()

    # Save results to file
    output_dir = Path(__file__).parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "IDENTITY_DEBATE_RESULTS.md"

    with open(output_file, "w") as f:
        f.write("# Aragora Identity Debate Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Participants:** {', '.join(result.participants)}\n\n")
        f.write(f"**Consensus:** {'Reached' if result.consensus_reached else 'Not reached'} ")
        f.write(f"(Confidence: {result.confidence:.0%})\n\n")
        f.write(f"**Duration:** {duration:.0f}s ({result.rounds_used} rounds)\n\n")
        f.write("---\n\n")
        f.write("## Final Answer\n\n")
        f.write(result.final_answer)
        f.write("\n\n---\n\n")
        f.write("## Context Provided to Agents\n\n")
        f.write("```\n")
        f.write(ARAGORA_SELF_KNOWLEDGE[:2000])
        f.write("\n...(truncated)\n```\n")

    print(f"Results saved to: {output_file}")
    print()

    return result


def main():
    """Entry point (V2 - works with available providers)."""
    print()
    print("Checking API keys...")
    print()

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_grok = bool(os.environ.get("XAI_API_KEY"))

    print(
        f"  OPENROUTER_API_KEY: {'[Y] Set (enables fallback + frontier models)' if has_openrouter else '[X] Not set (recommended)'}"
    )
    print(f"  ANTHROPIC_API_KEY:  {'[Y] Set' if has_anthropic else '[X] Not set'}")
    print(f"  OPENAI_API_KEY:     {'[Y] Set' if has_openai else '[X] Not set'}")
    print(f"  GEMINI_API_KEY:     {'[Y] Set' if has_gemini else '[X] Not set'}")
    print(f"  XAI_API_KEY:        {'[Y] Set' if has_grok else '[X] Not set'}")
    print()

    native_count = sum([has_anthropic, has_openai, has_gemini, has_grok])

    if native_count == 0 and not has_openrouter:
        print("[!] ERROR: No API keys found!")
        print(
            "    Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY"
        )
        sys.exit(1)

    if not has_openrouter:
        print("[!] WARNING: OPENROUTER_API_KEY not set")
        print("    - No fallback if native APIs fail")
        print("    - Missing frontier models: Mistral, Qwen, Kimi, DeepSeek")
        print("    Get one at https://openrouter.ai/keys")
        print()

    openrouter_count = 4 if has_openrouter else 0
    print(
        f"[i] {native_count} native APIs + {openrouter_count} OpenRouter models = {native_count + openrouter_count} potential agents"
    )

    try:
        result = asyncio.run(run_identity_debate())
        return 0 if result.consensus_reached else 1
    except KeyboardInterrupt:
        print("\n[!] Debate interrupted by user")
        return 130
    except Exception as e:
        print(f"\n[!] ERROR: {e}")
        logger.exception("Debate failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
