from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("page_path", "wrapper_path"),
    [
        (
            "aragora/live/src/app/(app)/spectate/[debateId]/page.tsx",
            "aragora/live/src/app/(app)/spectate/[debateId]/SpectateClient.tsx",
        ),
        (
            "aragora/live/src/app/(app)/gauntlet/[[...id]]/page.tsx",
            "aragora/live/src/app/(app)/gauntlet/[[...id]]/GauntletLiveWrapper.tsx",
        ),
        (
            "aragora/live/src/app/(app)/agent/[[...name]]/page.tsx",
            "aragora/live/src/app/(app)/agent/[[...name]]/AgentProfileWrapper.tsx",
        ),
    ],
)
def test_client_side_dynamic_live_pages_allow_runtime_params(
    page_path: str,
    wrapper_path: str,
) -> None:
    """Client-side ID routes must not be locked to placeholder static params."""

    page_source = (REPO_ROOT / page_path).read_text()
    wrapper_source = (REPO_ROOT / wrapper_path).read_text()

    assert "generateStaticParams" in page_source
    assert "useParams(" in wrapper_source
    assert "export const dynamicParams = true;" in page_source
    assert "dynamicParams = false" not in page_source
