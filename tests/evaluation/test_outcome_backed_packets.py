from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from aragora.evaluation.outcome_backed_corpus import load_visible_cases
from aragora.evaluation.outcome_backed_packets import (
    MaterializedSource,
    SourcePacketError,
    build_packet_set,
    fetch_source,
    render_case_packet,
    write_packet_set,
)


CORPUS_DIR = Path("docs/benchmarks/decision_quality/tranches")


class FakeResponse:
    def __init__(self, body: bytes, *, final_url: str = "https://example.test/source") -> None:
        self.body = body
        self.final_url = final_url

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        return self.body if amount < 0 else self.body[:amount]

    def geturl(self) -> str:
        return self.final_url


def _source(body: bytes = b"source body") -> dict[str, str]:
    return {
        "source_id": "source-1",
        "title": "Frozen source",
        "url": "https://example.test/source",
        "published_at": "2025-01-01T00:00:00Z",
        "content_sha256": hashlib.sha256(body).hexdigest(),
    }


def _case(index: int, *, split: str = "development") -> dict[str, object]:
    return {
        "case_id": f"case-{index:02d}",
        "domain": "software_engineering",
        "split": split,
        "title": f"Case {index}",
        "decision_prompt": "Choose one bounded action.",
        "forecast_question": "What is the probability option A is outcome-aligned?",
        "forecast_option_id": "option-a",
        "options": [
            {"option_id": "option-a", "label": "A", "description": "Action A"},
            {"option_id": "option-b", "label": "B", "description": "Action B"},
        ],
        "information_cutoff": "2025-01-02T00:00:00Z",
        "sources": [_source()],
    }


def _materialized(body: bytes = b"source body") -> MaterializedSource:
    return MaterializedSource(
        content=body.decode(),
        media_type="text/plain; charset=utf-8",
        content_encoding="utf-8",
        **_source(body),
    )


def test_visible_loader_never_reads_outcome_sidecars(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text
    reads: list[str] = []

    def tracked(path: Path, *args: object, **kwargs: object) -> str:
        reads.append(path.name)
        if path.name.endswith(".outcomes.json"):
            raise AssertionError("visible loader opened an outcome sidecar")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)

    cases = load_visible_cases(CORPUS_DIR)

    assert len(cases) == 24
    assert sum(case["split"] == "development" for case in cases) == 16
    assert all(name.endswith(".corpus.json") for name in reads)


def test_fetch_source_verifies_exact_bytes_before_decoding() -> None:
    body = "verified source\n".encode()
    opened: list[tuple[str, float]] = []

    def open_url(request: object, *, timeout: float) -> FakeResponse:
        opened.append((request.full_url, timeout))  # type: ignore[attr-defined]
        return FakeResponse(body)

    result = fetch_source(_source(body), timeout_seconds=2.5, open_url=open_url)

    assert result.content == "verified source\n"
    assert result.media_type == "text/plain; charset=utf-8"
    assert result.content_encoding == "utf-8"
    assert result.content_sha256 == hashlib.sha256(body).hexdigest()
    assert opened == [("https://example.test/source", 2.5)]


@pytest.mark.parametrize(
    ("source", "response", "match"),
    [
        (_source(), FakeResponse(b"changed"), "hash mismatch"),
        (
            _source(),
            FakeResponse(b"source body", final_url="http://example.test/source"),
            "redirected outside",
        ),
        (_source(b"\xff"), FakeResponse(b"\xff"), "neither PDF nor valid UTF-8"),
    ],
)
def test_fetch_source_fails_closed_on_integrity_defects(
    source: dict[str, str], response: FakeResponse, match: str
) -> None:
    with pytest.raises(SourcePacketError, match=match):
        fetch_source(source, open_url=lambda *_args, **_kwargs: response)


def test_fetch_source_sanitizes_transport_failure() -> None:
    def fail(*_args: object, **_kwargs: object) -> FakeResponse:
        raise URLError("token=super-secret")

    with pytest.raises(SourcePacketError) as exc_info:
        fetch_source(_source(), open_url=fail)

    assert "URLError" in str(exc_info.value)
    assert "super-secret" not in str(exc_info.value)


def test_fetch_source_preserves_pdf_bytes_losslessly() -> None:
    body = b"%PDF-1.7\nopaque\x00bytes\xff"

    result = fetch_source(
        _source(body),
        open_url=lambda *_args, **_kwargs: FakeResponse(body),
    )

    assert result.media_type == "application/pdf"
    assert result.content_encoding == "base64"
    assert result.content == "JVBERi0xLjcKb3BhcXVlAGJ5dGVz/w=="


def test_packet_is_deterministic_and_outcome_blind() -> None:
    case = _case(1)

    first = render_case_packet(case, [_materialized()])
    second = render_case_packet(dict(reversed(list(case.items()))), [_materialized()])

    assert first == second
    encoded = json.dumps(first, sort_keys=True)
    for forbidden in (
        "authoritative_sources",
        "correct_option_id",
        "cruxes",
        "resolution_summary",
        "resolved_at",
    ):
        assert forbidden not in encoded
    assert first["packet_sha256"] == second["packet_sha256"]


def test_packet_rejects_structural_outcome_fields() -> None:
    case = _case(1)
    case["correct_option_id"] = "option-a"

    with pytest.raises(SourcePacketError, match="outcome-only key"):
        render_case_packet(case, [_materialized()])


def test_build_packet_set_fetches_shared_source_once() -> None:
    calls: list[str] = []

    def fetch(source: dict[str, object]) -> MaterializedSource:
        calls.append(str(source["source_id"]))
        return _materialized()

    manifest, packets = build_packet_set(
        [_case(index) for index in range(16)],
        split="development",
        fetch=fetch,
    )

    assert calls == ["source-1"]
    assert manifest["packet_count"] == 16
    assert manifest["source_count"] == 1
    assert list(packets) == [f"case-{index:02d}" for index in range(16)]


def test_build_packet_set_requires_complete_split() -> None:
    with pytest.raises(SourcePacketError, match="expected 16 development cases"):
        build_packet_set([_case(1)], split="development", fetch=lambda _: _materialized())


def test_write_packet_set_refuses_stale_packet_residue(tmp_path: Path) -> None:
    (tmp_path / "old.packet.json").write_text("{}\n")
    manifest, packets = build_packet_set(
        [_case(index) for index in range(16)],
        split="development",
        fetch=lambda _: _materialized(),
    )

    with pytest.raises(SourcePacketError, match="stale packets"):
        write_packet_set(tmp_path, manifest, packets)
