"""ReceiptBuilder — construct cryptographic decision receipts from debate results.

Receipts provide an auditable trail of how a decision was reached, who agreed,
who dissented, and what evidence was evaluated.  They can be exported as JSON,
Markdown, or HTML.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aragora_debate.types import (
    Consensus,
    ConsensusMethod,
    DebateResult,
    DecisionReceipt,
    Verdict,
)


@dataclass
class ReceiptBuilder:
    """Builds :class:`DecisionReceipt` objects.

    The simplest path is :meth:`from_result` which auto-generates a receipt
    from a completed :class:`DebateResult`::

        receipt = ReceiptBuilder.from_result(result)
        print(receipt.to_markdown())
    """

    _receipt: DecisionReceipt | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_result(cls, result: DebateResult) -> DecisionReceipt:
        """Create a receipt directly from a DebateResult."""
        consensus = result.consensus or Consensus(
            reached=result.consensus_reached,
            method=ConsensusMethod.MAJORITY,
            confidence=result.confidence,
            supporting_agents=result.participants,
            dissenting_agents=[],
        )

        verdict = cls._infer_verdict(consensus)

        receipt = DecisionReceipt(
            receipt_id=f"DR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
            question=result.task,
            verdict=verdict,
            confidence=result.confidence,
            consensus=consensus,
            agents=result.participants,
            rounds_used=result.rounds_used,
            claims=list(result.claims),
            evidence=list(result.evidence),
            metadata={
                "debate_id": result.id,
                "status": result.status,
                "duration_seconds": result.duration_seconds,
                "total_tokens": result.total_tokens,
                "total_cost_usd": result.total_cost_usd,
            },
        )

        # Calculate content hash for integrity
        receipt.signature_algorithm = "SHA-256-content-hash"
        receipt.signature = cls._content_hash(receipt)

        return receipt

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    @staticmethod
    def sign_hmac(receipt: DecisionReceipt, key: str | bytes) -> DecisionReceipt:
        """Sign a receipt with HMAC-SHA256.

        Parameters
        ----------
        receipt : DecisionReceipt
            The receipt to sign.
        key : str | bytes
            The signing key.

        Returns
        -------
        DecisionReceipt
            The same receipt, mutated with ``signature`` and
            ``signature_algorithm`` set.
        """
        if isinstance(key, str):
            key = key.encode()
        payload = json.dumps(receipt.to_dict(), sort_keys=True, default=str).encode()
        sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
        receipt.signature = sig
        receipt.signature_algorithm = "HMAC-SHA256"
        return receipt

    @staticmethod
    def verify_hmac(receipt: DecisionReceipt, key: str | bytes) -> bool:
        """Verify an HMAC-SHA256 signed receipt."""
        if receipt.signature_algorithm != "HMAC-SHA256" or not receipt.signature:
            return False
        if isinstance(key, str):
            key = key.encode()
        saved_sig = receipt.signature
        # Temporarily clear signature fields to re-compute
        receipt.signature = None
        receipt.signature_algorithm = None
        payload = json.dumps(receipt.to_dict(), sort_keys=True, default=str).encode()
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        # Restore
        receipt.signature = saved_sig
        receipt.signature_algorithm = "HMAC-SHA256"
        return hmac.compare_digest(saved_sig, expected)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(receipt: DecisionReceipt, indent: int = 2) -> str:
        """Serialize a receipt to JSON."""
        return json.dumps(receipt.to_dict(), indent=indent, default=str)

    @staticmethod
    def to_html(receipt: DecisionReceipt) -> str:
        """Render a receipt as a standalone HTML page."""
        import html as html_mod

        esc = html_mod.escape
        dissent_rows = ""
        for d in receipt.consensus.dissents:
            reasons = "<br>".join(esc(r) for r in d.reasons)
            alt = esc(d.alternative_view or "")
            dissent_rows += (
                f"<tr><td>{esc(d.agent)}</td>"
                f"<td>{reasons}</td>"
                f"<td>{alt}</td></tr>\n"
            )

        dissent_section = ""
        if dissent_rows:
            dissent_section = f"""
    <h2>Dissenting Views</h2>
    <table>
      <thead><tr><th>Agent</th><th>Reasons</th><th>Alternative</th></tr></thead>
      <tbody>{dissent_rows}</tbody>
    </table>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Decision Receipt {esc(receipt.receipt_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .verdict {{ font-size: 1.3rem; font-weight: bold; }}
    .verdict-approved {{ color: #16a34a; }}
    .verdict-rejected {{ color: #dc2626; }}
    .verdict-review {{ color: #d97706; }}
    .meta {{ color: #666; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Decision Receipt {esc(receipt.receipt_id)}</h1>
  <p><strong>Question:</strong> {esc(receipt.question)}</p>
  <p class="verdict verdict-{'approved' if 'approved' in receipt.verdict.value else 'rejected' if receipt.verdict == Verdict.REJECTED else 'review'}">
    Verdict: {esc(receipt.verdict.value.replace('_', ' ').title())}
  </p>
  <table>
    <tr><td><strong>Confidence</strong></td><td>{receipt.confidence:.0%}</td></tr>
    <tr><td><strong>Consensus</strong></td><td>{'Reached' if receipt.consensus.reached else 'Not reached'} ({esc(receipt.consensus.method.value)})</td></tr>
    <tr><td><strong>Agreement</strong></td><td>{receipt.consensus.agreement_ratio:.0%}</td></tr>
    <tr><td><strong>Agents</strong></td><td>{esc(', '.join(receipt.agents))}</td></tr>
    <tr><td><strong>Rounds</strong></td><td>{receipt.rounds_used}</td></tr>
  </table>
  {dissent_section}
  <p class="meta">Generated {esc(receipt.timestamp)}</p>
  {f'<p class="meta">Signature ({esc(receipt.signature_algorithm or "")}): <code>{esc((receipt.signature or "")[:32])}...</code></p>' if receipt.signature else ''}
</body>
</html>"""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_verdict(consensus: Consensus) -> Verdict:
        """Infer a verdict from consensus state."""
        if not consensus.reached:
            if consensus.confidence >= 0.4:
                return Verdict.NEEDS_REVIEW
            return Verdict.REJECTED
        if consensus.dissents:
            return Verdict.APPROVED_WITH_CONDITIONS
        if consensus.confidence >= 0.8:
            return Verdict.APPROVED
        return Verdict.APPROVED_WITH_CONDITIONS

    @staticmethod
    def _content_hash(receipt: DecisionReceipt) -> str:
        """SHA-256 hash of receipt content for integrity checking."""
        payload = json.dumps(
            {
                "question": receipt.question,
                "verdict": receipt.verdict.value,
                "confidence": receipt.confidence,
                "agents": receipt.agents,
                "rounds_used": receipt.rounds_used,
                "timestamp": receipt.timestamp,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()
