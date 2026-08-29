from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.verify_provider_neutral_canary as verifier
from aragora.gauntlet.odr_signing import (
    generate_signing_key,
    public_key_pem,
    sign_odr_receipt,
)
from scripts.verify_provider_neutral_canary import Response, _write_state, run_verification
from tests.gauntlet.test_odr_signing import _valid_odr

IMAGE = "ghcr.io/synaptent/aragora@sha256:" + "a" * 64
SHA = "b" * 40
TOKEN = "ara_private_canary_token_123456"


class FakeTransport:
    def __init__(self, pem: str, *, build_sha: str = SHA) -> None:
        from cryptography.hazmat.primitives import serialization

        public_key = serialization.load_pem_public_key(pem.encode("utf-8"))
        from aragora.gauntlet.odr_signing import compute_key_id

        self.pem = pem
        self.key_id = compute_key_id(public_key)
        self.build_sha = build_sha
        self.webhook_id = "probe-123"
        self.callback_url = ""
        self.auth_headers: list[str] = []

    @staticmethod
    def _response(status: int, payload=None, headers=None) -> Response:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        return Response(status, headers or {}, body)

    def request(self, method, url, *, headers=None, payload=None) -> Response:
        if headers and "Authorization" in headers:
            self.auth_headers.append(headers["Authorization"])
        path = url.split("?", 1)[0]
        if path.endswith("/healthz"):
            return self._response(200)
        if path.endswith("/api/health"):
            return self._response(200, {"status": "healthy"})
        if path.endswith("/health/build"):
            return self._response(200, {"sha": self.build_sha})
        if path.endswith("/api/v2/receipts/signing-key"):
            return self._response(
                200,
                {
                    "algorithm": "Ed25519",
                    "key_id": self.key_id,
                    "public_key_pem": self.pem,
                },
            )
        if path.endswith("/.well-known/aragora-odr-signing-key"):
            return Response(200, {"X-Aragora-Key-Id": self.key_id}, self.pem.encode("utf-8"))
        if method == "POST" and path.endswith("/api/v1/webhook-configs"):
            assert payload["events"] == ["*"]
            self.callback_url = payload["url"]
            return self._response(
                201,
                {
                    "webhook": {
                        "id": self.webhook_id,
                        "url": self.callback_url,
                        "secret": "must-not-appear-in-report",
                    }
                },
            )
        if method == "GET" and path.endswith(f"/api/v1/webhook-configs/{self.webhook_id}"):
            return self._response(
                200, {"webhook": {"id": self.webhook_id, "url": self.callback_url}}
            )
        if method == "DELETE" and path.endswith(f"/api/v1/webhook-configs/{self.webhook_id}"):
            return self._response(204)
        return self._response(404)


def _args(tmp_path: Path, receipt_path: Path, phase: str) -> argparse.Namespace:
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir(mode=0o700, exist_ok=True)
    token_path = secret_dir / "canary-auth-token"
    token_path.write_text(TOKEN, encoding="utf-8")
    token_path.chmod(0o600)
    return argparse.Namespace(
        phase=phase,
        base_url="https://canary.example.invalid",
        expected_sha=SHA,
        expected_image_digest=IMAGE,
        observed_image_digest=IMAGE,
        database_proof_id="snapshot-proof-123",
        secrets_dir=str(secret_dir),
        receipt_file=str(receipt_path),
        persistence_state=str(tmp_path / "persistence.json"),
        output=str(tmp_path / f"{phase}.json"),
    )


def _receipt(tmp_path: Path):
    key = generate_signing_key()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(sign_odr_receipt(_valid_odr(), key)), encoding="utf-8")
    return key, receipt_path


def test_before_and_after_restart_proof(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)
    transport = FakeTransport(public_key_pem(key))
    before_args = _args(tmp_path, receipt_path, "before-restart")

    before = run_verification(before_args, transport)

    assert before["ok"] is True
    state_path = Path(before_args.persistence_state)
    assert state_path.stat().st_mode & 0o777 == 0o600
    after_args = _args(tmp_path, receipt_path, "after-restart")
    after_args.persistence_state = str(state_path)
    after = run_verification(after_args, transport)
    assert after["ok"] is True
    assert transport.auth_headers
    assert TOKEN not in json.dumps(before)
    assert "must-not-appear-in-report" not in json.dumps(before)


def test_legacy_webhook_fallback_persists_selected_endpoint(tmp_path: Path) -> None:
    state_path = tmp_path / "legacy-persistence.json"

    class LegacyTransport:
        def __init__(self) -> None:
            self.callback_url = ""
            self.requests: list[tuple[str, str]] = []

        def request(self, method, url, *, headers=None, payload=None) -> Response:
            path = url.split("?", 1)[0]
            self.requests.append((method, path))
            if method == "POST" and path.endswith("/api/v1/webhook-configs"):
                return Response(404, {}, b"")
            if method == "POST" and path.endswith("/api/v1/webhooks"):
                assert payload is not None
                assert payload == {
                    "webhook_url": payload["webhook_url"],
                    "events": ["test.event"],
                    "platform": "generic",
                }
                self.callback_url = payload["webhook_url"]
                body = {
                    "subscription": {
                        "id": "legacy-probe",
                        "webhook_url": self.callback_url,
                    }
                }
                return Response(201, {}, json.dumps(body).encode("utf-8"))
            if path.endswith("/api/v1/webhooks/legacy-probe"):
                body = {
                    "subscription": {
                        "id": "legacy-probe",
                        "webhook_url": self.callback_url,
                    }
                }
                if method == "GET":
                    return Response(200, {}, json.dumps(body).encode("utf-8"))
                if method == "DELETE":
                    return Response(204, {}, b"")
            raise AssertionError(f"unexpected request: {method} {path}")

    transport = LegacyTransport()
    before = verifier._persistence_before(transport, "https://canary.invalid", {}, state_path)
    assert before["ok"] is True
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["endpoint"] == "/api/v1/webhooks"

    after = verifier._persistence_after(transport, "https://canary.invalid", {}, state_path)
    assert after["ok"] is True
    assert transport.requests == [
        ("POST", "https://canary.invalid/api/v1/webhook-configs"),
        ("POST", "https://canary.invalid/api/v1/webhooks"),
        ("GET", "https://canary.invalid/api/v1/webhooks/legacy-probe"),
        ("GET", "https://canary.invalid/api/v1/webhooks/legacy-probe"),
        ("DELETE", "https://canary.invalid/api/v1/webhooks/legacy-probe"),
    ]


def test_wrong_build_sha_fails_closed(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)
    transport = FakeTransport(public_key_pem(key), build_sha="c" * 40)

    report = run_verification(_args(tmp_path, receipt_path, "before-restart"), transport)

    assert report["ok"] is False
    assert report["checks"]["build"]["ok"] is False


def test_image_or_database_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)
    transport = FakeTransport(public_key_pem(key))
    args = _args(tmp_path, receipt_path, "before-restart")
    args.observed_image_digest = "ghcr.io/synaptent/aragora@sha256:" + "d" * 64

    report = run_verification(args, transport)

    assert report["ok"] is False
    assert report["checks"]["identity"]["ok"] is False


def test_signing_key_endpoint_mismatch_fails_closed(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)
    transport = FakeTransport(public_key_pem(key))
    transport.key_id = "ed25519-wrong"

    report = run_verification(_args(tmp_path, receipt_path, "before-restart"), transport)

    assert report["ok"] is False
    assert report["checks"]["signing_keys"]["ok"] is False


def test_missing_token_is_recorded_in_failed_artifact(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)
    args = _args(tmp_path, receipt_path, "before-restart")
    (Path(args.secrets_dir) / "canary-auth-token").unlink()

    report = run_verification(args, FakeTransport(public_key_pem(key)))

    assert report["ok"] is False
    assert report["checks"]["custody"]["ok"] is False
    assert report["checks"]["custody"]["error_type"] in {
        "RuntimeError",
        "SecretNotFoundError",
    }


def test_transport_exception_is_recorded_without_secret_text(tmp_path: Path) -> None:
    _, receipt_path = _receipt(tmp_path)

    class BrokenTransport:
        def request(self, *args, **kwargs):
            raise OSError(f"network failed with {TOKEN}")

    report = run_verification(_args(tmp_path, receipt_path, "before-restart"), BrokenTransport())

    assert report["ok"] is False
    serialized = json.dumps(report)
    assert "OSError" in serialized
    assert TOKEN not in serialized


def test_http_origin_is_rejected_before_token_use(tmp_path: Path) -> None:
    _, receipt_path = _receipt(tmp_path)
    args = _args(tmp_path, receipt_path, "before-restart")
    args.base_url = "http://canary.example.invalid"

    class UnusedTransport:
        def request(self, *args, **kwargs):
            raise AssertionError("invalid HTTP origin reached the network")

    report = run_verification(args, UnusedTransport())

    assert report["ok"] is False
    assert report["checks"]["inputs"]["ok"] is False


def test_report_file_is_owner_only(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    _write_state(output, {"ok": False})
    assert output.stat().st_mode & 0o777 == 0o600


def test_invalid_auth_token_is_recorded(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)
    args = _args(tmp_path, receipt_path, "before-restart")
    token_path = Path(args.secrets_dir) / "canary-auth-token"
    token_path.write_text("server-bootstrap-token", encoding="utf-8")
    token_path.chmod(0o600)

    report = run_verification(args, FakeTransport(public_key_pem(key)))

    assert report["ok"] is False
    assert report["checks"]["custody"]["error_type"] == "RuntimeError"


def test_webhook_auth_rejection_is_recorded(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)

    class RejectingTransport(FakeTransport):
        def request(self, method, url, *, headers=None, payload=None):
            if method == "POST" and url.endswith("/api/v1/webhook-configs"):
                assert headers == {"Authorization": f"Bearer {TOKEN}"}
                return self._response(401, {"error": "Authentication required"})
            return super().request(method, url, headers=headers, payload=payload)

    report = run_verification(
        _args(tmp_path, receipt_path, "before-restart"),
        RejectingTransport(public_key_pem(key)),
    )

    assert report["ok"] is False
    assert report["checks"]["persistence"]["create_status"] == 401


def test_failed_read_reports_cleanup_status(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)

    class ReadFailTransport(FakeTransport):
        def request(self, method, url, *, headers=None, payload=None):
            if method == "GET" and "/api/v1/webhook-configs/" in url:
                return self._response(500)
            return super().request(method, url, headers=headers, payload=payload)

    report = run_verification(
        _args(tmp_path, receipt_path, "before-restart"),
        ReadFailTransport(public_key_pem(key)),
    )

    assert report["ok"] is False
    persistence = report["checks"]["persistence"]
    assert persistence["read_status"] == 500
    assert persistence["cleanup_status"] == 204


def test_state_write_failure_cleans_up_created_subscription(tmp_path: Path, monkeypatch) -> None:
    key, receipt_path = _receipt(tmp_path)

    class CleanupTrackingTransport(FakeTransport):
        deleted = False

        def request(self, method, url, *, headers=None, payload=None):
            if method == "DELETE":
                self.deleted = True
            return super().request(method, url, headers=headers, payload=payload)

    transport = CleanupTrackingTransport(public_key_pem(key))
    monkeypatch.setattr(
        verifier,
        "_write_state",
        lambda *args: (_ for _ in ()).throw(OSError("state unavailable")),
    )

    report = run_verification(_args(tmp_path, receipt_path, "before-restart"), transport)

    assert report["ok"] is False
    assert report["checks"]["persistence"]["error_type"] == "OSError"
    assert transport.deleted is True


def test_after_restart_delete_failure_is_recorded(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)

    class DeleteFailTransport(FakeTransport):
        fail_delete = False

        def request(self, method, url, *, headers=None, payload=None):
            if self.fail_delete and method == "DELETE":
                return self._response(500)
            return super().request(method, url, headers=headers, payload=payload)

    transport = DeleteFailTransport(public_key_pem(key))
    before_args = _args(tmp_path, receipt_path, "before-restart")
    assert run_verification(before_args, transport)["ok"] is True
    transport.fail_delete = True
    after_args = _args(tmp_path, receipt_path, "after-restart")
    after_args.persistence_state = before_args.persistence_state

    report = run_verification(after_args, transport)

    assert report["ok"] is False
    assert report["checks"]["persistence"]["delete_status"] == 500


def test_corrupt_after_restart_state_is_recorded(tmp_path: Path) -> None:
    key, receipt_path = _receipt(tmp_path)
    args = _args(tmp_path, receipt_path, "after-restart")
    Path(args.persistence_state).write_text("not json", encoding="utf-8")

    report = run_verification(args, FakeTransport(public_key_pem(key)))

    assert report["ok"] is False
    assert report["checks"]["persistence"]["error_type"] == "JSONDecodeError"
