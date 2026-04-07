"""
Tests for the FastAPI legacy v1 compatibility bridge.

These cover the live UI's remaining v1 audit-trail and receipt paths so they
do not silently regress while the frontend still depends on them.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi import create_app
from aragora.server.fastapi.dependencies.auth import require_authenticated


@pytest.fixture
def mock_audit_store():
    store = MagicMock()
    store.list_trails.return_value = [
        {
            "trail_id": "trail-123",
            "gauntlet_id": "gauntlet-123",
            "created_at": "2026-04-07T18:00:00Z",
            "verdict": "PASS",
            "confidence": 0.9,
            "total_findings": 1,
            "duration_seconds": 12.0,
            "checksum": "trail-checksum",
        }
    ]
    store.count_trails.return_value = 1
    store.get_trail.return_value = {
        "trail_id": "trail-123",
        "gauntlet_id": "gauntlet-123",
        "checksum": "trail-checksum",
    }
    store.get_trail_by_gauntlet.return_value = None

    store.list_receipts.return_value = [
        {
            "receipt_id": "receipt-123",
            "gauntlet_id": "gauntlet-123",
            "timestamp": "2026-04-07T18:00:00Z",
            "verdict": "PASS",
            "confidence": 0.9,
            "risk_level": "LOW",
            "findings_count": 1,
            "checksum": "receipt-checksum",
        }
    ]
    store.count_receipts.return_value = 1
    store.get_receipt.return_value = {
        "receipt_id": "receipt-123",
        "gauntlet_id": "gauntlet-123",
        "verdict": "PASS",
        "confidence": 0.9,
        "checksum": "receipt-checksum",
    }
    store.get_receipt_by_gauntlet.return_value = None
    return store


@pytest.fixture
def client(mock_audit_store):
    app = create_app()
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
    }
    auth_ctx = AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id="ws-1",
        roles={"member"},
        permissions={
            "audit:read",
            "audit:verify",
            "audit:receipts.read",
            "audit:receipts.verify",
        },
    )
    app.dependency_overrides[require_authenticated] = lambda: auth_ctx
    with patch(
        "aragora.storage.audit_trail_store.get_audit_trail_store",
        return_value=mock_audit_store,
    ):
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client


def test_v1_list_receipts_route_available(client):
    response = client.get("/api/v1/receipts?limit=10&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["receipts"][0]["receipt_id"] == "receipt-123"


def test_v1_verify_receipt_route_available(client):
    response = client.post("/api/v1/receipts/receipt-123/verify")

    assert response.status_code == 200
    data = response.json()
    assert data["receipt_id"] == "receipt-123"
    assert "valid" in data


def test_v1_list_audit_trails_route_available(client):
    response = client.get("/api/v1/audit-trails?limit=10&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["trails"][0]["trail_id"] == "trail-123"


def test_v1_verify_audit_trail_route_available(client):
    response = client.post("/api/v1/audit-trails/trail-123/verify")

    assert response.status_code == 200
    data = response.json()
    assert data["trail_id"] == "trail-123"
    assert "valid" in data
