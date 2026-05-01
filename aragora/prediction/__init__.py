"""Prediction markets — AGT-04 synthetic GitHub prediction substrate.

All public symbols are importable regardless of the feature flag.
The flag (``ARAGORA_PREDICTION_MARKETS_ENABLED``) only gates the runtime
behaviour of :class:`InMemoryStakeableClaimStore` and the resolution adapter.
"""

from aragora.prediction.stakeable_claim import (
    GithubResolutionAdapterStub,
    InMemoryStakeableClaimStore,
    QuestionType,
    ResolutionStatus,
    StakeableClaim,
)

__all__ = [
    "GithubResolutionAdapterStub",
    "InMemoryStakeableClaimStore",
    "QuestionType",
    "ResolutionStatus",
    "StakeableClaim",
]
