"""
Slack Block Kit formatter for decision receipts.

Formats receipts using Slack's Block Kit for rich message display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .formatter import ReceiptFormatter, register_formatter

if TYPE_CHECKING:
    from aragora.export.decision_receipt import DecisionReceipt


@register_formatter
class SlackReceiptFormatter(ReceiptFormatter):
    """Format decision receipts for Slack using Block Kit."""

    @property
    def channel_type(self) -> str:
        return "slack"

    def format(
        self,
        receipt: "DecisionReceipt",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format receipt as Slack Block Kit blocks.

        Options:
            compact: bool - Use compact format (default: False)
            include_agents: bool - Include agent details (default: True)
            include_evidence: bool - Include evidence section (default: True)
        """
        options = options or {}
        compact = options.get("compact", False)
        include_agents = options.get("include_agents", True)
        include_evidence = options.get("include_evidence", True)

        blocks: List[Dict[str, Any]] = []

        # Header
        confidence = receipt.confidence_score or 0
        confidence_emoji = self._get_confidence_emoji(confidence)

        blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{confidence_emoji} Decision Receipt",
                    "emoji": True,
                },
            }
        )

        # Topic/Question
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Topic:* {receipt.topic or receipt.question or 'N/A'}",
                },
            }
        )

        blocks.append({"type": "divider"})

        # Decision with confidence
        decision_text = receipt.decision or "No decision reached"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Decision:*\n{decision_text}",
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": f"{confidence:.0%} Confidence",
                        "emoji": True,
                    },
                    "style": "primary" if confidence >= 0.8 else None,
                    "action_id": "view_receipt",
                },
            }
        )

        if not compact:
            # Key Arguments
            if receipt.key_arguments:
                args_text = "\n".join(f"- {arg}" for arg in receipt.key_arguments[:5])
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Key Arguments:*\n{args_text}",
                        },
                    }
                )

            # Risks
            if receipt.risks:
                risks_text = "\n".join(f"- :warning: {risk}" for risk in receipt.risks[:3])
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Risks Identified:*\n{risks_text}",
                        },
                    }
                )

            # Dissenting Views
            if receipt.dissenting_views:
                dissent_text = "\n".join(f"- {view}" for view in receipt.dissenting_views[:3])
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Dissenting Views:*\n{dissent_text}",
                        },
                    }
                )

        # Agents section
        if include_agents and receipt.agents:
            agent_names = ", ".join(receipt.agents[:5])
            if len(receipt.agents) > 5:
                agent_names += f" +{len(receipt.agents) - 5} more"
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f":robot_face: *Agents:* {agent_names}",
                        },
                    ],
                }
            )

        # Evidence section
        if include_evidence and receipt.evidence:
            evidence_count = len(receipt.evidence)
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f":bookmark: {evidence_count} evidence sources cited",
                        },
                    ],
                }
            )

        blocks.append({"type": "divider"})

        # Footer with metadata
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Receipt ID: `{receipt.receipt_id}` | "
                        f"Rounds: {receipt.rounds or 'N/A'} | "
                        f"Generated by Aragora",
                    },
                ],
            }
        )

        return {"blocks": blocks}

    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji based on confidence level."""
        if confidence >= 0.9:
            return ":white_check_mark:"
        if confidence >= 0.7:
            return ":large_green_circle:"
        if confidence >= 0.5:
            return ":large_yellow_circle:"
        return ":red_circle:"
