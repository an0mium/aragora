"""Hilbert-style recursive proof decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aragora.verification.formal import (
    FormalProofResult,
    FormalProofStatus,
    FormalVerificationManager,
)


@dataclass
class HilbertProofNode:
    """A node in the recursive proof tree."""

    claim: str
    depth: int
    result: FormalProofResult | None = None
    children: list["HilbertProofNode"] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        if self.result is not None:
            return self.result.is_verified
        return all(child.is_verified for child in self.children)

    @property
    def status(self) -> str:
        if self.result is not None:
            return self.result.status.value
        return "decomposed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "depth": self.depth,
            "status": self.status,
            "is_verified": self.is_verified,
            "result": self.result.to_dict() if self.result else None,
            "children": [child.to_dict() for child in self.children],
        }


class HilbertProver:
    """Recursive proof decomposition over FormalVerificationManager."""

    def __init__(
        self,
        manager: FormalVerificationManager,
        max_depth: int = 2,
        min_subclaims: int = 2,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.manager = manager
        self.max_depth = max_depth
        self.min_subclaims = min_subclaims
        self.timeout_seconds = timeout_seconds

    async def prove(
        self,
        claim: str,
        claim_type: str | None = None,
        context: str = "",
    ) -> HilbertProofNode:
        return await self._prove_recursive(claim, claim_type, context, depth=0)

    async def _prove_recursive(
        self,
        claim: str,
        claim_type: str | None,
        context: str,
        depth: int,
    ) -> HilbertProofNode:
        result = await self.manager.attempt_formal_verification(
            claim=claim,
            claim_type=claim_type,
            context=context,
            timeout_seconds=self.timeout_seconds,
        )

        node = HilbertProofNode(claim=claim, depth=depth, result=result)

        if result.is_verified:
            return node

        if depth >= self.max_depth:
            return node

        if result.status not in (
            FormalProofStatus.NOT_SUPPORTED,
            FormalProofStatus.TRANSLATION_FAILED,
            FormalProofStatus.PROOF_FAILED,
        ):
            return node

        subclaims = self._decompose_claim(claim)
        if len(subclaims) < self.min_subclaims:
            return node

        for subclaim in subclaims:
            child = await self._prove_recursive(
                subclaim, claim_type=claim_type, context=context, depth=depth + 1
            )
            node.children.append(child)

        return node

    def _decompose_claim(self, claim: str) -> list[str]:
        text = claim.strip()
        if not text:
            return []

        lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
        bullet_lines = [line for line in lines if line.startswith(("-", "*"))]
        if bullet_lines:
            return [line.lstrip("-* ").strip() for line in bullet_lines if line.strip()]

        separators = [" and ", ";", ". ", " then "]
        for sep in separators:
            if sep in text.lower():
                parts = [p.strip() for p in text.split(sep) if p.strip()]
                if len(parts) >= self.min_subclaims:
                    return parts

        return []
