"""
Discord embed formatter for decision receipts.

Formats receipts using Discord's embed format for rich message display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .formatter import ReceiptFormatter, register_formatter

if TYPE_CHECKING:
    from aragora.export.decision_receipt import DecisionReceipt


@register_formatter
class DiscordReceiptFormatter(ReceiptFormatter):
    """Format decision receipts for Discord using embeds."""

    @property
    def channel_type(self) -> str:
        return "discord"

    def format(
        self,
        receipt: "DecisionReceipt",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format receipt as Discord embed.

        Options:
            compact: bool - Use compact format (default: False)
        """
        options = options or {}
        compact = options.get("compact", False)

        confidence = receipt.confidence_score or 0
        color = self._get_embed_color(confidence)

        fields: List[Dict[str, Any]] = []

        # Decision field
        fields.append(
            {
                "name": "Decision",
                "value": (receipt.decision or "No decision reached")[:1024],
                "inline": False,
            }
        )

        # Confidence field
        confidence_bar = self._make_confidence_bar(confidence)
        fields.append(
            {
                "name": "Confidence",
                "value": f"{confidence_bar} {confidence:.0%}",
                "inline": True,
            }
        )

        # Rounds field
        fields.append(
            {
                "name": "Rounds",
                "value": str(receipt.rounds or "N/A"),
                "inline": True,
            }
        )

        # Agents field
        if receipt.agents:
            agent_list = ", ".join(receipt.agents[:5])
            if len(receipt.agents) > 5:
                agent_list += f" (+{len(receipt.agents) - 5})"
            fields.append(
                {
                    "name": "Agents",
                    "value": agent_list,
                    "inline": True,
                }
            )

        if not compact:
            # Key Arguments
            if receipt.key_arguments:
                args_text = "\n".join(f"- {arg[:100]}" for arg in receipt.key_arguments[:3])
                fields.append(
                    {
                        "name": "Key Arguments",
                        "value": args_text[:1024],
                        "inline": False,
                    }
                )

            # Risks
            if receipt.risks:
                risks_text = "\n".join(f"- {risk[:100]}" for risk in receipt.risks[:3])
                fields.append(
                    {
                        "name": "Risks",
                        "value": risks_text[:1024],
                        "inline": False,
                    }
                )

            # Dissenting Views
            if receipt.dissenting_views:
                dissent_text = "\n".join(f"- {view[:100]}" for view in receipt.dissenting_views[:2])
                fields.append(
                    {
                        "name": "Dissenting Views",
                        "value": dissent_text[:1024],
                        "inline": False,
                    }
                )

        embed = {
            "title": "Decision Receipt",
            "description": (receipt.topic or receipt.question or "")[:4096],
            "color": color,
            "fields": fields,
            "footer": {
                "text": f"Receipt ID: {receipt.receipt_id} | Powered by Aragora",
            },
        }

        if receipt.timestamp:
            embed["timestamp"] = receipt.timestamp

        return {"embeds": [embed]}

    def _get_embed_color(self, confidence: float) -> int:
        """Get Discord embed color based on confidence."""
        if confidence >= 0.8:
            return 0x57F287  # Green
        if confidence >= 0.5:
            return 0xFEE75C  # Yellow
        return 0xED4245  # Red

    def _make_confidence_bar(self, confidence: float, length: int = 10) -> str:
        """Create a visual confidence bar using Discord formatting."""
        filled = int(confidence * length)
        empty = length - filled
        return "█" * filled + "░" * empty
