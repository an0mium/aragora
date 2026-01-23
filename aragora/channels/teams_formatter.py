"""
Microsoft Teams Adaptive Cards formatter for decision receipts.

Formats receipts using Teams' Adaptive Cards for rich message display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .formatter import ReceiptFormatter, register_formatter

if TYPE_CHECKING:
    from aragora.export.decision_receipt import DecisionReceipt


@register_formatter
class TeamsReceiptFormatter(ReceiptFormatter):
    """Format decision receipts for Microsoft Teams using Adaptive Cards."""

    @property
    def channel_type(self) -> str:
        return "teams"

    def format(
        self,
        receipt: "DecisionReceipt",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format receipt as Teams Adaptive Card.

        Options:
            compact: bool - Use compact format (default: False)
            include_agents: bool - Include agent details (default: True)
        """
        options = options or {}
        compact = options.get("compact", False)

        confidence = receipt.confidence_score or 0
        confidence_color = self._get_confidence_color(confidence)

        body: List[Dict[str, Any]] = []

        # Header with topic
        body.append(
            {
                "type": "TextBlock",
                "size": "Large",
                "weight": "Bolder",
                "text": "Decision Receipt",
                "color": "Accent",
            }
        )

        body.append(
            {
                "type": "TextBlock",
                "text": receipt.topic or receipt.question or "N/A",
                "wrap": True,
                "weight": "Bolder",
            }
        )

        # Confidence indicator
        body.append(
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [
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
                                "text": "Confidence",
                                "size": "Small",
                                "color": "Dark",
                            },
                            {
                                "type": "TextBlock",
                                "text": self._get_confidence_label(confidence),
                                "weight": "Bolder",
                            },
                        ],
                    },
                ],
            }
        )

        # Decision
        body.append(
            {
                "type": "TextBlock",
                "text": "Decision",
                "weight": "Bolder",
                "spacing": "Medium",
            }
        )
        body.append(
            {
                "type": "TextBlock",
                "text": receipt.decision or "No decision reached",
                "wrap": True,
            }
        )

        if not compact:
            # Key Arguments
            if receipt.key_arguments:
                body.append(
                    {
                        "type": "TextBlock",
                        "text": "Key Arguments",
                        "weight": "Bolder",
                        "spacing": "Medium",
                    }
                )
                for arg in receipt.key_arguments[:5]:
                    body.append(
                        {
                            "type": "TextBlock",
                            "text": f"- {arg}",
                            "wrap": True,
                            "spacing": "None",
                        }
                    )

            # Risks
            if receipt.risks:
                body.append(
                    {
                        "type": "TextBlock",
                        "text": "Risks Identified",
                        "weight": "Bolder",
                        "spacing": "Medium",
                        "color": "Warning",
                    }
                )
                for risk in receipt.risks[:3]:
                    body.append(
                        {
                            "type": "TextBlock",
                            "text": f"- {risk}",
                            "wrap": True,
                            "spacing": "None",
                            "color": "Warning",
                        }
                    )

        # Agents
        if receipt.agents:
            body.append(
                {
                    "type": "FactSet",
                    "facts": [
                        {
                            "title": "Agents",
                            "value": ", ".join(receipt.agents[:5]),
                        },
                        {
                            "title": "Rounds",
                            "value": str(receipt.rounds or "N/A"),
                        },
                    ],
                    "spacing": "Medium",
                }
            )

        # Footer
        body.append(
            {
                "type": "TextBlock",
                "text": f"Receipt ID: {receipt.receipt_id}",
                "size": "Small",
                "color": "Dark",
                "spacing": "Medium",
            }
        )

        return {
            "type": "AdaptiveCard",
            "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": body,
            "actions": [
                {
                    "type": "Action.OpenUrl",
                    "title": "View Full Receipt",
                    "url": f"https://aragora.ai/receipts/{receipt.receipt_id}",
                },
            ],
        }

    def _get_confidence_color(self, confidence: float) -> str:
        """Get Adaptive Card color based on confidence."""
        if confidence >= 0.8:
            return "Good"
        if confidence >= 0.5:
            return "Warning"
        return "Attention"

    def _get_confidence_label(self, confidence: float) -> str:
        """Get human-readable confidence label."""
        if confidence >= 0.9:
            return "Very High"
        if confidence >= 0.7:
            return "High"
        if confidence >= 0.5:
            return "Moderate"
        return "Low"
