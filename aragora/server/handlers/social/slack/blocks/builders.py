"""
Slack Block Kit builders for debate messages.

Provides functions for building rich Slack messages:
- starting_blocks: Initial debate announcement
- round_update_blocks: Progress updates during debates
- agent_response_blocks: Individual agent responses
- result_blocks: Final debate results
"""

from __future__ import annotations

from typing import Any


def build_starting_blocks(
    topic: str,
    user_id: str,
    debate_id: str,
    agents: list[str] | None = None,
    expected_rounds: int | None = None,
) -> list[dict[str, Any]]:
    """Build Slack blocks for debate start message.

    Args:
        topic: Debate topic
        user_id: User who started the debate
        debate_id: Unique debate identifier
        agents: Optional list of participating agents
        expected_rounds: Optional expected number of rounds

    Returns:
        List of Slack block elements
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Debate Starting...",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Topic:* {topic}",
            },
        },
    ]

    # Add agents and rounds info if provided
    context_parts = [f"Requested by <@{user_id}> | ID: `{debate_id}`"]
    if agents:
        context_parts.append(f"Agents: {', '.join(agents)}")
    if expected_rounds:
        context_parts.append(f"Rounds: {expected_rounds}")

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": " | ".join(context_parts),
                },
            ],
        }
    )

    return blocks


def build_round_update_blocks(
    round_num: int,
    total_rounds: int,
    agent: str,
    phase: str = "analyzing",
) -> list[dict[str, Any]]:
    """Build Slack blocks for round progress update.

    Args:
        round_num: Current round number
        total_rounds: Total rounds in debate
        agent: Name of agent that responded
        phase: Current debate phase (analyzing, critique, voting, complete)

    Returns:
        List of Slack block elements
    """
    # Visual progress bar using block characters
    progress_bar = ":black_large_square:" * round_num + ":white_large_square:" * (
        total_rounds - round_num
    )

    # Phase emoji
    phase_emojis = {
        "analyzing": ":mag:",
        "critique": ":speech_balloon:",
        "voting": ":ballot_box:",
        "complete": ":white_check_mark:",
    }
    phase_emoji = phase_emojis.get(phase, ":hourglass_flowing_sand:")

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{phase_emoji} *Round {round_num}/{total_rounds}*\n"
                    f"`{progress_bar}`\n"
                    f"_{agent} responded_"
                ),
            },
        },
    ]

    return blocks


# Agent emoji mapping for visual distinction
AGENT_EMOJIS = {
    "anthropic-api": ":robot_face:",
    "openai-api": ":brain:",
    "gemini": ":gem:",
    "grok": ":zap:",
    "mistral": ":wind_face:",
    "deepseek": ":mag:",
}


def build_agent_response_blocks(
    agent: str,
    response: str,
    round_num: int,
) -> list[dict[str, Any]]:
    """Build Slack blocks for an individual agent response.

    Args:
        agent: Name of agent that responded
        response: The agent's response content
        round_num: Current round number

    Returns:
        List of Slack block elements
    """
    emoji = AGENT_EMOJIS.get(agent.lower(), ":speech_balloon:")

    # Truncate response for Slack (max 3000 chars in section)
    truncated = response[:2800] + "..." if len(response) > 2800 else response

    blocks: list[dict[str, Any]] = [
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{emoji} *{agent}* | Round {round_num}",
                }
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": truncated,
            },
        },
        {"type": "divider"},
    ]

    return blocks


def build_result_blocks(
    topic: str,
    result: Any,
    user_id: str,
    receipt_url: str | None = None,
    debate_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build Slack blocks for debate result message.

    Args:
        topic: Debate topic
        result: DebateResult object with consensus info
        user_id: User who started the debate
        receipt_url: Optional URL to receipt page
        debate_id: Optional debate ID for tracking

    Returns:
        List of Slack block elements with rich formatting
    """
    # Status indicators
    status_emoji = ":white_check_mark:" if result.consensus_reached else ":warning:"
    status_text = "Consensus Reached" if result.consensus_reached else "No Consensus"

    # Confidence visualization (filled/empty circles)
    confidence_filled = int(result.confidence * 5)
    confidence_bar = ":large_blue_circle:" * confidence_filled + ":white_circle:" * (
        5 - confidence_filled
    )

    # Participant names (show up to 4)
    participant_names = result.participants[:4] if result.participants else []
    participants_text = ", ".join(participant_names)
    if len(result.participants) > 4:
        participants_text += f" +{len(result.participants) - 4}"

    blocks: list[dict[str, Any]] = [
        {"type": "divider"},
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} Debate Complete",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Status:*\n{status_text}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Confidence:*\n{confidence_bar} {result.confidence:.0%}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Rounds:*\n{result.rounds_used}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Participants:*\n{participants_text}",
                },
            ],
        },
    ]

    # Add conclusion section
    if result.conclusion:
        # Truncate for Slack
        conclusion = (
            result.conclusion[:2800] + "..." if len(result.conclusion) > 2800 else result.conclusion
        )
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Conclusion:*\n{conclusion}",
                },
            }
        )

    # Add actions if debate_id is provided
    if debate_id:
        actions: list[dict[str, Any]] = []

        # View details button
        actions.append(
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Details",
                    "emoji": True,
                },
                "action_id": f"view_details_{debate_id}",
                "value": debate_id,
            }
        )

        # Receipt link if available
        if receipt_url:
            actions.append(
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View Receipt",
                        "emoji": True,
                    },
                    "url": receipt_url,
                    "action_id": f"view_receipt_{debate_id}",
                }
            )

        # Vote buttons
        actions.extend(
            [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": ":thumbsup: Agree",
                        "emoji": True,
                    },
                    "action_id": f"vote_agree_{debate_id}",
                    "value": debate_id,
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": ":thumbsdown: Disagree",
                        "emoji": True,
                    },
                    "action_id": f"vote_disagree_{debate_id}",
                    "value": debate_id,
                    "style": "danger",
                },
            ]
        )

        blocks.append(
            {
                "type": "actions",
                "elements": actions,
            }
        )

    # Context footer
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Requested by <@{user_id}>"
                    + (f" | ID: `{debate_id}`" if debate_id else ""),
                }
            ],
        }
    )

    return blocks


def build_gauntlet_result_blocks(
    statement: str,
    result: Any,
    user_id: str,
    receipt_url: str | None = None,
) -> list[dict[str, Any]]:
    """Build Slack blocks for gauntlet (stress-test) result message.

    Args:
        statement: The statement being validated
        result: Gauntlet result object
        user_id: User who started the gauntlet
        receipt_url: Optional URL to receipt page

    Returns:
        List of Slack block elements
    """
    # Validation status
    verdict = getattr(result, "verdict", "unknown")
    if verdict == "valid":
        verdict_emoji = ":white_check_mark:"
        verdict_text = "Statement Validated"
    elif verdict == "invalid":
        verdict_emoji = ":x:"
        verdict_text = "Statement Refuted"
    else:
        verdict_emoji = ":warning:"
        verdict_text = "Inconclusive"

    blocks: list[dict[str, Any]] = [
        {"type": "divider"},
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{verdict_emoji} Gauntlet Complete",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Statement:*\n_{statement}_",
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Verdict:*\n{verdict_text}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Confidence:*\n{getattr(result, 'confidence', 0):.0%}",
                },
            ],
        },
    ]

    # Add findings if present
    findings = getattr(result, "findings", [])
    if findings:
        finding_text = "\n".join(f"• {f}" for f in findings[:5])
        if len(findings) > 5:
            finding_text += f"\n_...and {len(findings) - 5} more_"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Key Findings:*\n{finding_text}",
                },
            }
        )

    # Receipt link
    if receipt_url:
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View Full Report",
                            "emoji": True,
                        },
                        "url": receipt_url,
                    }
                ],
            }
        )

    # Context footer
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Requested by <@{user_id}>",
                }
            ],
        }
    )

    return blocks


def build_search_result_blocks(
    query: str,
    results: list[dict[str, Any]],
    total: int = 0,
) -> list[dict[str, Any]]:
    """Build Slack blocks for search results.

    Args:
        query: Search query
        results: List of search result dictionaries
        total: Total number of results (may be > len(results))

    Returns:
        List of Slack block elements
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f'Search Results for "{query}"',
                "emoji": True,
            },
        },
    ]

    if not results:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "_No results found._",
                },
            }
        )
    else:
        for i, result in enumerate(results[:10], 1):
            title = result.get("title", result.get("topic", "Untitled"))
            snippet = result.get("snippet", result.get("conclusion", ""))[:200]
            result_type = result.get("type", "debate")

            type_emoji = {
                "debate": ":speech_balloon:",
                "evidence": ":page_facing_up:",
                "consensus": ":handshake:",
            }.get(result_type, ":mag:")

            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{type_emoji} *{i}. {title}*\n{snippet}...",
                    },
                }
            )

        if total > 10:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_Showing 10 of {total} results_",
                        }
                    ],
                }
            )

    return blocks


__all__ = [
    "build_starting_blocks",
    "build_round_update_blocks",
    "build_agent_response_blocks",
    "build_result_blocks",
    "build_gauntlet_result_blocks",
    "build_search_result_blocks",
    "AGENT_EMOJIS",
]
