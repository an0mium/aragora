"""JCS canonicalization tests — RFC 8785 examples + digest stability."""

from __future__ import annotations

from aragora_verify import jcs_canonicalize, odr_content_digest

from _fixtures import valid_odr


def test_member_ordering_is_independent_of_insertion_order() -> None:
    a = jcs_canonicalize({"b": 1, "a": 2, "c": 3})
    b = jcs_canonicalize({"c": 3, "a": 2, "b": 1})
    assert a == b == b'{"a":2,"b":1,"c":3}'


def test_no_insignificant_whitespace() -> None:
    assert jcs_canonicalize({"x": [1, 2, {"y": "z"}]}) == b'{"x":[1,2,{"y":"z"}]}'


def test_literals_and_null() -> None:
    assert jcs_canonicalize(True) == b"true"
    assert jcs_canonicalize(False) == b"false"
    assert jcs_canonicalize(None) == b"null"


def test_rfc8785_number_serialization() -> None:
    # Shortest round-trip per ECMAScript Number::toString.
    assert jcs_canonicalize(0) == b"0"
    assert jcs_canonicalize(-0.0) == b"0"
    assert jcs_canonicalize(1.0) == b"1"
    assert jcs_canonicalize(1.5) == b"1.5"
    assert jcs_canonicalize(1000000000000000000000.0) == b"1e+21"
    assert jcs_canonicalize(0.000001) == b"0.000001"
    assert jcs_canonicalize(1e-7) == b"1e-7"


def test_utf16_codeunit_sort_orders_supplementary_after_bmp() -> None:
    # 'é' (U+00E9) sorts before an emoji (supplementary plane) by UTF-16 units.
    out = jcs_canonicalize({"\U0001f600": 1, "é": 2})
    assert out.index(b"\xc3\xa9") < out.index(b"\xf0\x9f\x98\x80")


def test_unicode_strings_emitted_raw_utf8() -> None:
    assert jcs_canonicalize({"k": "café"}) == '{"k":"café"}'.encode("utf-8")


def test_control_chars_lowercase_escaped() -> None:
    assert jcs_canonicalize("") == b'"\\u0001"'


def test_digest_excludes_signatures() -> None:
    doc = valid_odr()
    digest_unsigned = odr_content_digest(doc)
    doc_with_sig = dict(doc)
    doc_with_sig["signatures"] = [{"alg": "Ed25519", "key_id": "x", "signature": "y"}]
    assert odr_content_digest(doc_with_sig) == digest_unsigned


def test_digest_is_stable_and_hex_sha256() -> None:
    digest = odr_content_digest(valid_odr())
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert odr_content_digest(valid_odr()) == digest
