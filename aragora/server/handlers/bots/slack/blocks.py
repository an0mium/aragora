"""
Slack Block Kit Message Builders.

This module contains functions to build Block Kit formatted messages
for debates, consensus results, and modals.
"""

import json
from typing import Any

from aragora.config import DEFAULT_ROUNDS


def build_debate_message_blocks(
    debate_id: str,
    task: str,
    agents: list[str],
    current_round: int,
    total_rounds: int,
    include_vote_buttons: bool = True,
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for a debate message."""
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": " Active Debate",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Task:* {task}",
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Agents:*\n{', '.join(agents)}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Progress:*\nRound {current_round}/{total_rounds}",
                },
            ],
        },
        {"type": "divider"},
    ]

    if include_vote_buttons:
        # Add voting buttons
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Cast your vote:*",
                },
            }
        )

        # Create button for each agent
        buttons = [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": f"Vote {agent}",
                    "emoji": True,
                },
                "style": "primary" if i == 0 else None,
                "action_id": f"vote_{debate_id}_{agent}",
                "value": json.dumps({"debate_id": debate_id, "agent": agent}),
            }
            for i, agent in enumerate(agents[:5])  # Max 5 buttons
        ]

        # Remove None style values
        for btn in buttons:
            if btn.get("style") is None:
                del btn["style"]

        blocks.append(
            {
                "type": "actions",
                "elements": buttons,
            }
        )

        # Add summary button
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": " View Summary",
                            "emoji": True,
                        },
                        "action_id": f"summary_{debate_id}",
                        "value": debate_id,
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": " Provenance",
                            "emoji": True,
                        },
                        "action_id": f"provenance_{debate_id}",
                        "value": debate_id,
                        "url": f"https://aragora.ai/debates/provenance?debate={debate_id}",
                    },
                ],
            }
        )

    # Footer
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f" Aragora | Debate ID: `{debate_id[:8]}...`",
                },
            ],
        }
    )

    return blocks


def build_consensus_message_blocks(
    debate_id: str,
    task: str,
    consensus_reached: bool,
    confidence: float,
    winner: str | None,
    final_answer: str | None,
    vote_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Build Block Kit blocks for consensus result."""
    status_emoji = "" if consensus_reached else ""
    status_text = "Consensus Reached" if consensus_reached else "No Consensus"

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} {status_text}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Task:* {task}",
            },
        },
    ]

    # Results fields
    fields = [
        {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.0%}"},
    ]

    if winner:
        fields.append({"type": "mrkdwn", "text": f"*Winner:*\n{winner}"})

    blocks.append(
        {
            "type": "section",
            "fields": fields,
        }
    )

    # Show user votes if any
    if vote_counts:
        vote_text = "\n".join(
            f"• {agent}: {count} vote{'s' if count != 1 else ''}"
            for agent, count in sorted(vote_counts.items(), key=lambda x: -x[1])
        )
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*User Votes:*\n{vote_text}",
                },
            }
        )

    # Final answer preview
    if final_answer:
        preview = final_answer[:500]
        if len(final_answer) > 500:
            preview += "..."

        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Decision:*\n```{preview}```",
                },
            }
        )

    # Action buttons
    blocks.append(
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": " View Full",
                        "emoji": True,
                    },
                    "url": f"https://aragora.ai/debate/{debate_id}",
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": " Audit Trail",
                        "emoji": True,
                    },
                    "url": f"https://aragora.ai/debates/provenance?debate={debate_id}",
                },
            ],
        }
    )

    return blocks


# Alias for backward compatibility
build_debate_result_blocks = build_consensus_message_blocks


def build_start_debate_modal() -> dict[str, Any]:
    """Build modal for starting a new debate."""
    return {
        "type": "modal",
        "callback_id": "start_debate_modal",
        "title": {
            "type": "plain_text",
            "text": "Start Debate",
        },
        "submit": {
            "type": "plain_text",
            "text": "Start",
        },
        "close": {
            "type": "plain_text",
            "text": "Cancel",
        },
        "blocks": [
            {
                "type": "input",
                "block_id": "task_block",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "task_input",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "What should the agents debate?",
                    },
                    "multiline": True,
                },
                "label": {
                    "type": "plain_text",
                    "text": "Debate Task",
                },
            },
            {
                "type": "input",
                "block_id": "agents_block",
                "element": {
                    "type": "multi_static_select",
                    "action_id": "agents_select",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Select agents",
                    },
                    "options": [
                        {"text": {"type": "plain_text", "text": "Claude"}, "value": "claude"},
                        {"text": {"type": "plain_text", "text": "GPT-4"}, "value": "gpt4"},
                        {"text": {"type": "plain_text", "text": "Gemini"}, "value": "gemini"},
                        {"text": {"type": "plain_text", "text": "Mistral"}, "value": "mistral"},
                        {"text": {"type": "plain_text", "text": "DeepSeek"}, "value": "deepseek"},
                        {"text": {"type": "plain_text", "text": "Grok"}, "value": "grok"},
                        {"text": {"type": "plain_text", "text": "Qwen"}, "value": "qwen"},
                        {"text": {"type": "plain_text", "text": "Kimi"}, "value": "kimi"},
                    ],
                },
                "label": {
                    "type": "plain_text",
                    "text": "Agents",
                },
            },
            {
                "type": "input",
                "block_id": "rounds_block",
                "element": {
                    "type": "static_select",
                    "action_id": "rounds_select",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Number of rounds",
                    },
                    "options": [
                        {"text": {"type": "plain_text", "text": "3 rounds"}, "value": "3"},
                        {"text": {"type": "plain_text", "text": "5 rounds"}, "value": "5"},
                        {"text": {"type": "plain_text", "text": "8 rounds"}, "value": "8"},
                        {"text": {"type": "plain_text", "text": "9 rounds"}, "value": "9"},
                    ],
                    "initial_option": {
                        "text": {"type": "plain_text", "text": f"{DEFAULT_ROUNDS} rounds"},
                        "value": str(DEFAULT_ROUNDS),
                    },
                },
                "label": {
                    "type": "plain_text",
                    "text": "Rounds",
                },
            },
        ],
    }


# Private alias for backward compatibility
_build_start_debate_modal = build_start_debate_modal


__all__ = [
    "build_debate_message_blocks",
    "build_consensus_message_blocks",
    "build_debate_result_blocks",
    "build_start_debate_modal",
    "_build_start_debate_modal",
]
