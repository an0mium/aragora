"""
Aragora identity and positioning constants.

This module contains the canonical positioning statements, taglines, and
differentiators for Aragora. All user-facing descriptions should reference
these constants to ensure consistency across the codebase.
"""

# Primary tagline (under 15 words)
TAGLINE = "Control plane for multi-agent robust decisionmaking across org knowledge and channels."

# Short description (under 50 words)
DESCRIPTION_SHORT = (
    "Aragora is the control plane for multi-agent robust decisionmaking—orchestrating "
    "15+ AI models to debate your organization's knowledge (documents, databases, "
    "APIs) and deliver defensible decisions to any channel (Slack, Teams, Discord, voice)."
)

# Full description with differentiators
DESCRIPTION_FULL = (
    "Aragora is the control plane for multi-agent robust decisionmaking—orchestrating "
    "15+ AI models to debate your organization's knowledge (documents, databases, APIs) "
    "and deliver defensible decisions to any channel (Slack, Teams, Discord, voice). "
    "Unlike chatbots, Aragora builds institutional memory and provides full audit trails "
    "for high-stakes decisions."
)

# Elevator pitch (100 words)
ELEVATOR_PITCH = (
    "Aragora is an enterprise control plane that orchestrates multi-agent robust decisionmaking "
    "across your organization's knowledge and communication channels.\n\n"
    "It ingests anything—25+ document formats, databases, APIs—debates it across 15+ "
    "frontier AI models, and delivers defensible answers wherever your team works. "
    "The structured robust decisionmaking protocol produces decisions with evidence chains, "
    "not black-box outputs.\n\n"
    "Aragora's 4-tier memory builds institutional knowledge that compounds over time. "
    "For legal, finance, compliance, and security teams where accountability matters, "
    "Aragora provides decision assurance with receipts—not 'the AI said so.'"
)

# Key differentiators vs. chatbots and single-model tools
DIFFERENTIATORS = {
    "not_chatbot": "Structured robust decisionmaking protocol with phases, roles, and evidence chains",
    "not_copilot": "Institutional learning that accumulates across sessions",
    "not_single_model": "Heterogeneous ensemble (15+ AI providers) that argues toward truth",
    "not_stateless": "Remembers outcomes, builds knowledge, improves itself",
    "not_text_only": "Ingests documents, images, audio; outputs to chat, voice, APIs",
}

# What Aragora is NOT (competitive positioning)
NOT_STATEMENTS = {
    "chatbot": "NOT a chatbot: Structured robust decisionmaking protocol with phases, roles, and evidence chains",
    "copilot": "NOT a copilot: Institutional learning that ACCUMULATES organizational knowledge",
    "single_model": "NOT single-model: Heterogeneous 15+ provider ensemble that argues toward truth",
    "stateless": "NOT stateless: Remembers outcomes, builds knowledge graphs, improves itself",
    "text_only": "NOT text-only: Multimodal ingestion + multi-channel bidirectional output",
}

# Core capabilities for marketing materials
CORE_CAPABILITIES = {
    "omnivorous_ingestion": {
        "name": "Omnivorous Data Ingestion",
        "description": "25+ document formats, images, audio, video, 24 data connectors",
        "maturity": "96%",
    },
    "institutional_memory": {
        "name": "Institutional Memory",
        "description": "4-tier memory, surprise-based retention, Knowledge Mound, cross-session learning",
        "maturity": "92%",
    },
    "bidirectional_communication": {
        "name": "Bidirectional Communication",
        "description": "Receives queries AND sends results to Slack, Discord, Teams, Telegram, WhatsApp, voice",
        "maturity": "91%",
    },
    "debate_synthesis": {
        "name": "Debate & Synthesis",
        "description": "Structured robust decisionmaking protocol (Thesis→Antithesis→Synthesis), multi-agent consensus",
        "maturity": "95%",
    },
    "long_context": {
        "name": "Long Context Mastery",
        "description": "RLM programmatic navigation, million-token contexts, REPL-like interface",
        "maturity": "88%",
    },
    "self_improvement": {
        "name": "Self-Improvement",
        "description": "Nomic Loop (debate→design→implement→verify with constitutional constraints)",
        "maturity": "85%",
    },
}

# Service offerings for enterprise sales
SERVICE_OFFERINGS = {
    "decision_assurance_pilot": {
        "name": "Decision Assurance Pilot",
        "description": "90-day deployment proving defensible AI decisions for one high-stakes workflow",
    },
    "ai_red_team": {
        "name": "AI Red-Team & Code Review",
        "description": "Multi-agent adversarial analysis of your AI systems or codebase",
    },
    "omnichannel_knowledge_ops": {
        "name": "Omnichannel Knowledge Ops",
        "description": "Connect any data source, deliver insights to any channel",
    },
}

# Terminology guidance for external communications
TERMINOLOGY = {
    # Use externally
    "deliberation": "Use 'robust decisionmaking' instead of 'debate' in external communications",
    "control_plane": "Use 'control plane' to emphasize enterprise orchestration",
    "structured_protocol": "Use 'structured robust decisionmaking protocol' instead of 'Hegelian'",
    "decision_assurance": "Use 'decision assurance' for enterprise positioning",
    # Keep internal
    "nomic_loop": "Keep 'Nomic Loop' for internal/technical documentation",
    "knowledge_mound": "Keep 'Knowledge Mound' for internal product naming",
    "omnivorous": "Keep 'omnivorous' - strong differentiator for data ingestion",
    "decision_receipts": "Keep 'decision receipts' - unique value proposition",
}


def get_tagline() -> str:
    """Return the primary tagline."""
    return TAGLINE


def get_description(full: bool = False) -> str:
    """Return the short or full description."""
    return DESCRIPTION_FULL if full else DESCRIPTION_SHORT


def get_elevator_pitch() -> str:
    """Return the elevator pitch."""
    return ELEVATOR_PITCH
