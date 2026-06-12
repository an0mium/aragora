"""Tests for ``aragora.trail.rekor`` (ODR-7 Sigstore Rekor client, #8231).

Network-free by construction: the HTTP transport and the signer are both
injected fakes. The single live test at the bottom is skip-by-default and
never runs in CI (set ``ARAGORA_LIVE_REKOR=1`` to exercise it).
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import pytest

from aragora.trail import rekor

HEAD_HASH = "ab" * 32  # 64 lowercase hex chars
ENTRY_UUID = "c" * 64
FAKE_SIG = base64.b64encode(b"fake-der-signature").decode("ascii")
FAKE_PUB = base64.b64encode(b"-----BEGIN PUBLIC KEY-----\nfake\n").decode("ascii")


def _fake_signer(digest: str) -> tuple[str, str]:
    assert digest == HEAD_HASH
    return FAKE_SIG, FAKE_PUB


def _entry_body(sha256_hex: str = HEAD_HASH) -> str:
    """Base64 canonical hashedrekord body, as Rekor stores it."""
    return base64.b64encode(
        json.dumps(rekor.build_hashedrekord(sha256_hex, FAKE_SIG, FAKE_PUB)).encode()
    ).decode("ascii")


def _entry_payload(
    uuid: str = ENTRY_UUID,
    log_index: int = 12345,
    integrated_time: int = 1760000000,
    body: str | None = None,
) -> dict[str, Any]:
    return {
        uuid: {
            "logID": "d" * 64,
            "logIndex": log_index,
            "integratedTime": integrated_time,
            "body": body if body is not None else _entry_body(),
        }
    }


class FakeHttp:
    """Scripted transport recording every request."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body
        self.calls: list[tuple[str, str, bytes | None]] = []

    def __call__(self, method: str, url: str, body: bytes | None) -> tuple[int, bytes]:
        self.calls.append((method, url, body))
        return self.status, self.body


class TestPayloadShape:
    def test_hashedrekord_matches_rekor_v1_schema(self) -> None:
        proposed = rekor.build_hashedrekord(HEAD_HASH, FAKE_SIG, FAKE_PUB)
        assert proposed["apiVersion"] == "0.0.1"
        assert proposed["kind"] == "hashedrekord"
        spec = proposed["spec"]
        assert spec["data"]["hash"] == {"algorithm": "sha256", "value": HEAD_HASH}
        assert spec["signature"]["content"] == FAKE_SIG
        assert spec["signature"]["publicKey"]["content"] == FAKE_PUB

    @pytest.mark.parametrize(
        "bad",
        ["", "abc", "Z" * 64, HEAD_HASH.upper(), HEAD_HASH + "ab", None, 1234],
    )
    def test_non_sha256_input_is_rejected(self, bad: Any) -> None:
        with pytest.raises(rekor.RekorError, match="sha256"):
            rekor.build_hashedrekord(bad, FAKE_SIG, FAKE_PUB)

    def test_invalid_hash_never_reaches_the_transport(self) -> None:
        http = FakeHttp(201, b"{}")
        with pytest.raises(rekor.RekorError):
            rekor.submit_hash("not-a-hash", http=http, signer=_fake_signer)
        assert http.calls == []


class TestParseEntryResponse:
    def test_parses_single_entry_map(self) -> None:
        entry = rekor.parse_entry_response(_entry_payload())
        assert entry.uuid == ENTRY_UUID
        assert entry.log_index == 12345
        assert entry.integrated_time == 1760000000
        assert entry.log_id == "d" * 64
        assert entry.as_anchor_record() == {
            "log_index": 12345,
            "uuid": ENTRY_UUID,
            "integrated_time": 1760000000,
        }

    @pytest.mark.parametrize(
        "payload",
        [
            {},  # empty map
            {"a": {}, "b": {}},  # two entries
            ["not", "a", "map"],
            {ENTRY_UUID: "not-an-object"},
            {ENTRY_UUID: {"logIndex": "NaN", "integratedTime": 1}},
            {ENTRY_UUID: {"integratedTime": 1}},  # missing logIndex
        ],
    )
    def test_rejects_malformed_payloads(self, payload: Any) -> None:
        with pytest.raises(rekor.RekorError):
            rekor.parse_entry_response(payload)


class TestSubmitHash:
    def test_posts_hashedrekord_and_returns_entry(self) -> None:
        http = FakeHttp(201, json.dumps(_entry_payload()).encode())
        entry = rekor.submit_hash(HEAD_HASH, http=http, signer=_fake_signer)
        assert entry.log_index == 12345
        assert entry.uuid == ENTRY_UUID
        method, url, body = http.calls[0]
        assert method == "POST"
        assert url == "https://rekor.sigstore.dev/api/v1/log/entries"
        sent = json.loads(body or b"")
        assert sent["kind"] == "hashedrekord"
        assert sent["spec"]["data"]["hash"]["value"] == HEAD_HASH

    @pytest.mark.parametrize("status", [400, 409, 500, 503])
    def test_non_201_raises_and_returns_nothing(self, status: int) -> None:
        http = FakeHttp(status, b"nope")
        with pytest.raises(rekor.RekorError, match=f"HTTP {status}"):
            rekor.submit_hash(HEAD_HASH, http=http, signer=_fake_signer)

    def test_non_json_201_raises(self) -> None:
        http = FakeHttp(201, b"<html>surprise</html>")
        with pytest.raises(rekor.RekorError, match="non-JSON"):
            rekor.submit_hash(HEAD_HASH, http=http, signer=_fake_signer)

    def test_base_url_override(self) -> None:
        http = FakeHttp(201, json.dumps(_entry_payload()).encode())
        rekor.submit_hash(
            HEAD_HASH, base_url="https://rekor.example/", http=http, signer=_fake_signer
        )
        assert http.calls[0][1] == "https://rekor.example/api/v1/log/entries"


class TestEphemeralSigner:
    def test_signature_verifies_over_the_digest(self) -> None:
        cryptography = pytest.importorskip("cryptography")
        del cryptography
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, utils

        sig_b64, pub_b64 = rekor._ephemeral_sign(HEAD_HASH)
        public_key = serialization.load_pem_public_key(base64.b64decode(pub_b64))
        assert isinstance(public_key, ec.EllipticCurvePublicKey)
        public_key.verify(  # raises InvalidSignature on mismatch
            base64.b64decode(sig_b64),
            bytes.fromhex(HEAD_HASH),
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )

    def test_keys_are_ephemeral_not_reused(self) -> None:
        pytest.importorskip("cryptography")
        _, pub_a = rekor._ephemeral_sign(HEAD_HASH)
        _, pub_b = rekor._ephemeral_sign(HEAD_HASH)
        assert pub_a != pub_b


class TestFetchAndConsistency:
    def test_fetch_entry_by_uuid(self) -> None:
        http = FakeHttp(200, json.dumps(_entry_payload()).encode())
        entry = rekor.fetch_entry(ENTRY_UUID, http=http)
        assert entry.log_index == 12345
        method, url, body = http.calls[0]
        assert method == "GET"
        assert url.endswith(f"/api/v1/log/entries/{ENTRY_UUID}")
        assert body is None

    def test_fetch_rejects_bad_uuid_without_network(self) -> None:
        http = FakeHttp(200, b"{}")
        with pytest.raises(rekor.RekorError, match="uuid"):
            rekor.fetch_entry("../../etc/passwd", http=http)
        assert http.calls == []

    def test_consistency_check_passes_on_matching_digest(self) -> None:
        http = FakeHttp(200, json.dumps(_entry_payload()).encode())
        entry = rekor.verify_inclusion_consistency(ENTRY_UUID, HEAD_HASH, http=http)
        assert entry.uuid == ENTRY_UUID

    def test_consistency_check_fails_on_digest_mismatch(self) -> None:
        other = "ef" * 32
        http = FakeHttp(200, json.dumps(_entry_payload(body=_entry_body(other))).encode())
        with pytest.raises(rekor.RekorError, match="does not match"):
            rekor.verify_inclusion_consistency(ENTRY_UUID, HEAD_HASH, http=http)

    def test_consistency_check_fails_on_non_hashedrekord(self) -> None:
        body = base64.b64encode(json.dumps({"kind": "intoto"}).encode()).decode()
        http = FakeHttp(200, json.dumps(_entry_payload(body=body)).encode())
        with pytest.raises(rekor.RekorError, match="not a hashedrekord"):
            rekor.verify_inclusion_consistency(ENTRY_UUID, HEAD_HASH, http=http)


class TestTransportGuards:
    def test_default_transport_refuses_plain_http(self) -> None:
        with pytest.raises(rekor.RekorError, match="https"):
            rekor._default_http("GET", "http://rekor.sigstore.dev/x", None)


@pytest.mark.skipif(
    os.environ.get("ARAGORA_LIVE_REKOR") != "1",
    reason="Set ARAGORA_LIVE_REKOR=1 to run the live Rekor smoke test (network)",
)
class TestLiveSmoke:
    def test_submit_and_verify_round_trip(self) -> None:
        import hashlib

        digest = hashlib.sha256(os.urandom(32)).hexdigest()
        entry = rekor.submit_hash(digest)
        assert entry.log_index > 0
        fetched = rekor.verify_inclusion_consistency(entry.uuid, digest)
        assert fetched.log_index == entry.log_index
