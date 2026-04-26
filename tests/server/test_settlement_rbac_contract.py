"""Regression coverage for settlement RBAC enforcement."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import patch

from aragora.server.handlers.settlements import SettlementHandler


def test_settlement_post_handler_requires_write_permission() -> None:
    """Member users should be denied before POST settlement logic runs."""
    member_ctx = SimpleNamespace(
        is_authenticated=True,
        user_id="member-1",
        role="member",
        error_reason=None,
    )
    body = json.dumps(
        {
            "outcome": "correct",
            "evidence": "proof",
            "settled_by": "member-1",
        }
    ).encode("utf-8")
    http_handler = SimpleNamespace(
        headers={"Content-Length": str(len(body))},
        rfile=io.BytesIO(body),
    )

    with patch(
        "aragora.billing.jwt_auth.extract_user_from_request",
        lambda handler, user_store=None: member_ctx,
    ):
        result = SettlementHandler(ctx={}).handle_post(
            "/api/v1/settlements/settle-123/settle",
            {},
            http_handler,
        )

    assert result is not None
    assert result.status_code == 403
    assert json.loads(result.body)["error"] == "Permission denied"
