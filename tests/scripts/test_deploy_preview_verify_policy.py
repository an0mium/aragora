from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    REPO_ROOT / ".github/workflows/deploy-frontend.yml",
    REPO_ROOT / ".github/workflows/deploy-secure.yml",
)


def test_canonical_url_verification_is_blocking_in_deploy_workflows() -> None:
    for workflow_path in WORKFLOWS:
        text = workflow_path.read_text(encoding="utf-8")
        assert 'FRONTEND_URL="${LIVE_FRONTEND_BASE_URL:-https://aragora.ai}"' in text
        assert 'bash scripts/verify_frontend_routes.sh "$FRONTEND_URL"' in text


def test_preview_url_verification_policy_is_explicit_in_deploy_workflows() -> None:
    for workflow_path in WORKFLOWS:
        text = workflow_path.read_text(encoding="utf-8")

        assert "VERCEL_PREVIEW_AUTH_HEADER" in text
        assert "VERCEL_PREVIEW_AUTH_BYPASS_TOKEN" in text
        assert "Preview verification is non-blocking" in text
        assert "VERIFY_FRONTEND_SOFT_FAIL=1" in text
        assert "VERIFY_FRONTEND_ANNOTATION_LEVEL=warning" in text
