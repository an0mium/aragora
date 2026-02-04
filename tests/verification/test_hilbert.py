import asyncio

import pytest

from aragora.verification.formal import (
    FormalLanguage,
    FormalProofResult,
    FormalProofStatus,
)
from aragora.verification.hilbert import HilbertProver


class FakeManager:
    async def attempt_formal_verification(self, claim: str, **kwargs):
        if "and" in claim:
            return FormalProofResult(
                status=FormalProofStatus.NOT_SUPPORTED,
                language=FormalLanguage.Z3_SMT,
            )
        return FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
        )


@pytest.mark.asyncio
async def test_hilbert_prover_decomposes() -> None:
    prover = HilbertProver(manager=FakeManager(), max_depth=1, min_subclaims=2)
    node = await prover.prove("A and B")
    assert node.status == "not_supported"
    assert len(node.children) == 2
    assert all(child.is_verified for child in node.children)
