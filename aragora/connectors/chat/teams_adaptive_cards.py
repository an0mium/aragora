"""
Microsoft Teams Adaptive Cards Builder.

Provides rich Adaptive Card templates for debate interactions in Teams.
See: https://adaptivecards.io/explorer/

Templates:
- Debate starting card with progress indicator
- Round progress card with agent messages
- Voting card for human participation
- Verdict card with detailed breakdown
- Error card with troubleshooting
- Receipt card with decision audit trail

Usage:
    from aragora.connectors.chat.teams_adaptive_cards import TeamsAdaptiveCards

    # Create a verdict card
    card = TeamsAdaptiveCards.verdict_card(
        topic="Should we use microservices?",
        verdict="Yes, for services needing independent scaling",
        confidence=0.85,
        agents=[
            {"name": "Claude", "position": "for", "key_point": "Better scaling"},
            {"name": "GPT-4", "position": "for", "key_point": "Team autonomy"},
            {"name": "Gemini", "position": "against", "key_point": "Complexity cost"},
        ],
        receipt_id="rec_abc123",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentContribution:
    """Represents an agent's contribution to a debate."""

    name: str
    position: str  # "for", "against", "neutral"
    key_point: str
    confidence: float = 0.5
    icon_url: Optional[str] = None


@dataclass
class RoundProgress:
    """Progress information for a debate round."""

    round_number: int
    total_rounds: int
    agent_messages: List[Dict[str, str]] = field(default_factory=list)
    current_consensus: Optional[str] = None


class TeamsAdaptiveCards:
    """Builder for Teams Adaptive Cards."""

    # Agent icon URLs (placeholder - should be configurable)
    AGENT_ICONS = {
        "claude": "https://api.aragora.ai/icons/claude.png",
        "anthropic": "https://api.aragora.ai/icons/claude.png",
        "gpt": "https://api.aragora.ai/icons/openai.png",
        "openai": "https://api.aragora.ai/icons/openai.png",
        "gemini": "https://api.aragora.ai/icons/gemini.png",
        "grok": "https://api.aragora.ai/icons/grok.png",
        "mistral": "https://api.aragora.ai/icons/mistral.png",
        "default": "https://api.aragora.ai/icons/agent.png",
    }

    @classmethod
    def get_agent_icon(cls, agent_name: str) -> str:
        """Get icon URL for an agent."""
        name_lower = agent_name.lower()
        for key, url in cls.AGENT_ICONS.items():
            if key in name_lower:
                return url
        return cls.AGENT_ICONS["default"]

    @classmethod
    def wrap_as_card(
        cls, body: List[Dict[str, Any]], actions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Wrap body elements as a full Adaptive Card."""
        card: Dict[str, Any] = {
            "type": "AdaptiveCard",
            "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.5",
            "body": body,
        }
        if actions:
            card["actions"] = actions
        return card

    @classmethod
    def starting_card(
        cls,
        topic: str,
        initiated_by: str,
        agents: List[str],
        debate_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a card for when a debate is starting."""
        body: List[Dict[str, Any]] = [
            {
                "type": "Container",
                "style": "accent",
                "bleed": True,
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "Debate Starting",
                        "size": "Large",
                        "weight": "Bolder",
                        "color": "Light",
                    }
                ],
            },
            {
                "type": "Container",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": topic,
                        "size": "Medium",
                        "weight": "Bolder",
                        "wrap": True,
                    },
                    {
                        "type": "ColumnSet",
                        "columns": [
                            {
                                "type": "Column",
                                "width": "auto",
                                "items": [
                                    {
                                        "type": "TextBlock",
                                        "text": "Initiated by:",
                                        "isSubtle": True,
                                        "size": "Small",
                                    }
                                ],
                            },
                            {
                                "type": "Column",
                                "width": "stretch",
                                "items": [
                                    {
                                        "type": "TextBlock",
                                        "text": initiated_by,
                                        "size": "Small",
                                        "weight": "Bolder",
                                    }
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                "type": "TextBlock",
                "text": "Participating Agents",
                "weight": "Bolder",
                "spacing": "Medium",
            },
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [
                            {
                                "type": "Image",
                                "url": cls.get_agent_icon(agent),
                                "size": "Small",
                                "style": "Person",
                            }
                            for agent in agents[:4]  # Limit to 4 agents
                        ],
                    },
                    {
                        "type": "Column",
                        "width": "stretch",
                        "verticalContentAlignment": "Center",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": ", ".join(agents),
                                "wrap": True,
                            }
                        ],
                    },
                ],
            },
            {
                "type": "TextBlock",
                "text": "AI agents are deliberating...",
                "isSubtle": True,
                "horizontalAlignment": "Center",
                "spacing": "Large",
            },
        ]

        actions = [
            {
                "type": "Action.Submit",
                "title": "Cancel Debate",
                "style": "destructive",
                "data": {
                    "action": "cancel_debate",
                    "debate_id": debate_id,
                },
            }
        ]

        return cls.wrap_as_card(body, actions)

    @classmethod
    def progress_card(
        cls,
        topic: str,
        progress: RoundProgress,
        debate_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a card showing debate progress."""
        progress_pct = int((progress.round_number / progress.total_rounds) * 100)

        body: List[Dict[str, Any]] = [
            {
                "type": "Container",
                "style": "emphasis",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": f"Round {progress.round_number} of {progress.total_rounds}",
                        "size": "Medium",
                        "weight": "Bolder",
                    }
                ],
            },
            {
                "type": "TextBlock",
                "text": topic,
                "wrap": True,
                "isSubtle": True,
            },
            # Progress bar (simulated with columns)
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": progress_pct,
                        "items": [
                            {
                                "type": "Container",
                                "style": "accent",
                                "height": "8px",
                            }
                        ],
                    },
                    {
                        "type": "Column",
                        "width": 100 - progress_pct,
                        "items": [
                            {
                                "type": "Container",
                                "style": "default",
                                "height": "8px",
                            }
                        ],
                    }
                    if progress_pct < 100
                    else {},
                ],
            },
        ]

        # Add agent messages
        if progress.agent_messages:
            body.append(
                {
                    "type": "TextBlock",
                    "text": "Recent Activity",
                    "weight": "Bolder",
                    "spacing": "Medium",
                }
            )
            for msg in progress.agent_messages[-3:]:  # Last 3 messages
                body.append(
                    {
                        "type": "Container",
                        "items": [
                            {
                                "type": "ColumnSet",
                                "columns": [
                                    {
                                        "type": "Column",
                                        "width": "auto",
                                        "items": [
                                            {
                                                "type": "Image",
                                                "url": cls.get_agent_icon(msg.get("agent", "")),
                                                "size": "Small",
                                                "style": "Person",
                                            }
                                        ],
                                    },
                                    {
                                        "type": "Column",
                                        "width": "stretch",
                                        "items": [
                                            {
                                                "type": "TextBlock",
                                                "text": msg.get("agent", "Agent"),
                                                "weight": "Bolder",
                                                "size": "Small",
                                            },
                                            {
                                                "type": "TextBlock",
                                                "text": msg.get("summary", "")[:200] + "..."
                                                if len(msg.get("summary", "")) > 200
                                                else msg.get("summary", ""),
                                                "wrap": True,
                                                "size": "Small",
                                            },
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                )

        if progress.current_consensus:
            body.append(
                {
                    "type": "TextBlock",
                    "text": f"Emerging consensus: {progress.current_consensus}",
                    "wrap": True,
                    "isSubtle": True,
                    "spacing": "Medium",
                }
            )

        return cls.wrap_as_card(body)

    @classmethod
    def voting_card(
        cls,
        topic: str,
        verdict: str,
        debate_id: str,
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a card for voting on a debate outcome."""
        if options is None:
            options = ["Agree", "Disagree", "Abstain"]

        body: List[Dict[str, Any]] = [
            {
                "type": "Container",
                "style": "emphasis",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "Your Vote Requested",
                        "size": "Medium",
                        "weight": "Bolder",
                    }
                ],
            },
            {
                "type": "TextBlock",
                "text": topic,
                "wrap": True,
                "spacing": "Small",
            },
            {
                "type": "Container",
                "style": "accent",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "AI Verdict",
                        "size": "Small",
                        "isSubtle": True,
                    },
                    {
                        "type": "TextBlock",
                        "text": verdict,
                        "wrap": True,
                        "weight": "Bolder",
                    },
                ],
            },
            {
                "type": "TextBlock",
                "text": "Do you agree with this decision?",
                "spacing": "Medium",
            },
        ]

        actions: List[Dict[str, Any]] = []
        for option in options:
            vote_value = option.lower().replace(" ", "_")
            style = "positive" if option.lower() == "agree" else "default"
            if option.lower() == "disagree":
                style = "destructive"
            actions.append(
                {
                    "type": "Action.Submit",
                    "title": option,
                    "style": style,
                    "data": {
                        "action": "vote",
                        "vote": vote_value,
                        "debate_id": debate_id,
                    },
                }
            )

        return cls.wrap_as_card(body, actions)

    @classmethod
    def verdict_card(
        cls,
        topic: str,
        verdict: str,
        confidence: float,
        agents: List[AgentContribution],
        rounds_completed: int = 3,
        receipt_id: Optional[str] = None,
        debate_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a comprehensive verdict card."""
        confidence_color = (
            "Good" if confidence >= 0.7 else ("Warning" if confidence >= 0.5 else "Attention")
        )

        body: List[Dict[str, Any]] = [
            {
                "type": "Container",
                "style": "good",
                "bleed": True,
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "Debate Complete",
                        "size": "Large",
                        "weight": "Bolder",
                        "color": "Light",
                    }
                ],
            },
            {
                "type": "TextBlock",
                "text": topic,
                "size": "Medium",
                "wrap": True,
                "spacing": "Medium",
            },
            # Verdict section
            {
                "type": "Container",
                "style": "emphasis",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "VERDICT",
                        "size": "Small",
                        "weight": "Bolder",
                        "color": "Accent",
                    },
                    {
                        "type": "TextBlock",
                        "text": verdict,
                        "size": "Medium",
                        "weight": "Bolder",
                        "wrap": True,
                    },
                ],
            },
            # Stats
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Confidence",
                                "size": "Small",
                                "isSubtle": True,
                            },
                            {
                                "type": "TextBlock",
                                "text": f"{confidence:.0%}",
                                "size": "ExtraLarge",
                                "weight": "Bolder",
                                "color": confidence_color,
                            },
                        ],
                    },
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Rounds",
                                "size": "Small",
                                "isSubtle": True,
                            },
                            {
                                "type": "TextBlock",
                                "text": str(rounds_completed),
                                "size": "ExtraLarge",
                                "weight": "Bolder",
                            },
                        ],
                    },
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "Agents",
                                "size": "Small",
                                "isSubtle": True,
                            },
                            {
                                "type": "TextBlock",
                                "text": str(len(agents)),
                                "size": "ExtraLarge",
                                "weight": "Bolder",
                            },
                        ],
                    },
                ],
            },
        ]

        # Agent breakdown
        if agents:
            body.append(
                {
                    "type": "TextBlock",
                    "text": "Agent Breakdown",
                    "weight": "Bolder",
                    "spacing": "Medium",
                }
            )
            for_agents = [a for a in agents if a.position == "for"]
            against_agents = [a for a in agents if a.position == "against"]

            body.append(
                {
                    "type": "ColumnSet",
                    "columns": [
                        {
                            "type": "Column",
                            "width": "stretch",
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": f"FOR ({len(for_agents)})",
                                    "weight": "Bolder",
                                    "color": "Good",
                                },
                                *[
                                    {
                                        "type": "TextBlock",
                                        "text": f"**{a.name}**: {a.key_point}",
                                        "wrap": True,
                                        "size": "Small",
                                    }
                                    for a in for_agents
                                ],
                            ],
                        },
                        {
                            "type": "Column",
                            "width": "stretch",
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": f"AGAINST ({len(against_agents)})",
                                    "weight": "Bolder",
                                    "color": "Attention",
                                },
                                *[
                                    {
                                        "type": "TextBlock",
                                        "text": f"**{a.name}**: {a.key_point}",
                                        "wrap": True,
                                        "size": "Small",
                                    }
                                    for a in against_agents
                                ],
                            ],
                        },
                    ],
                }
            )

        actions: List[Dict[str, Any]] = []
        if receipt_id:
            actions.append(
                {
                    "type": "Action.OpenUrl",
                    "title": "View Full Receipt",
                    "url": f"/api/v1/receipts/{receipt_id}",
                }
            )
        if debate_id:
            actions.append(
                {
                    "type": "Action.Submit",
                    "title": "Agree",
                    "style": "positive",
                    "data": {"action": "vote", "vote": "agree", "debate_id": debate_id},
                }
            )
            actions.append(
                {
                    "type": "Action.Submit",
                    "title": "Disagree",
                    "data": {"action": "vote", "vote": "disagree", "debate_id": debate_id},
                }
            )

        return cls.wrap_as_card(body, actions)

    @classmethod
    def error_card(
        cls,
        title: str,
        message: str,
        suggestions: Optional[List[str]] = None,
        retry_action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an error card with troubleshooting suggestions."""
        body: List[Dict[str, Any]] = [
            {
                "type": "Container",
                "style": "attention",
                "bleed": True,
                "items": [
                    {
                        "type": "TextBlock",
                        "text": title,
                        "size": "Medium",
                        "weight": "Bolder",
                        "color": "Light",
                    }
                ],
            },
            {
                "type": "TextBlock",
                "text": message,
                "wrap": True,
                "spacing": "Medium",
            },
        ]

        if suggestions:
            body.append(
                {
                    "type": "TextBlock",
                    "text": "Suggestions:",
                    "weight": "Bolder",
                    "spacing": "Medium",
                }
            )
            for suggestion in suggestions:
                body.append(
                    {
                        "type": "TextBlock",
                        "text": f"• {suggestion}",
                        "wrap": True,
                        "size": "Small",
                    }
                )

        actions: List[Dict[str, Any]] = []
        if retry_action:
            actions.append(
                {
                    "type": "Action.Submit",
                    "title": "Retry",
                    "data": retry_action,
                }
            )
        actions.append(
            {
                "type": "Action.Submit",
                "title": "Get Help",
                "data": {"action": "help"},
            }
        )

        return cls.wrap_as_card(body, actions)

    @classmethod
    def receipt_card(
        cls,
        receipt_id: str,
        topic: str,
        verdict: str,
        timestamp: str,
        hash_preview: str,
        verification_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a decision receipt card."""
        body = [
            {
                "type": "Container",
                "style": "emphasis",
                "items": [
                    {
                        "type": "ColumnSet",
                        "columns": [
                            {
                                "type": "Column",
                                "width": "auto",
                                "items": [
                                    {
                                        "type": "Image",
                                        "url": "https://api.aragora.ai/icons/receipt.png",
                                        "size": "Small",
                                    }
                                ],
                            },
                            {
                                "type": "Column",
                                "width": "stretch",
                                "items": [
                                    {
                                        "type": "TextBlock",
                                        "text": "Decision Receipt",
                                        "size": "Medium",
                                        "weight": "Bolder",
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Receipt ID", "value": receipt_id[:16] + "..."},
                    {"title": "Topic", "value": topic[:50] + "..." if len(topic) > 50 else topic},
                    {"title": "Timestamp", "value": timestamp},
                    {"title": "Hash", "value": hash_preview[:16] + "..."},
                ],
            },
            {
                "type": "Container",
                "style": "accent",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": "Verdict",
                        "size": "Small",
                        "isSubtle": True,
                    },
                    {
                        "type": "TextBlock",
                        "text": verdict,
                        "wrap": True,
                        "weight": "Bolder",
                    },
                ],
            },
        ]

        actions = [
            {
                "type": "Action.OpenUrl",
                "title": "View Full Receipt",
                "url": f"/api/v1/receipts/{receipt_id}",
            }
        ]
        if verification_url:
            actions.append(
                {
                    "type": "Action.OpenUrl",
                    "title": "Verify Integrity",
                    "url": verification_url,
                }
            )

        return cls.wrap_as_card(body, actions)


__all__ = [
    "TeamsAdaptiveCards",
    "AgentContribution",
    "RoundProgress",
]
