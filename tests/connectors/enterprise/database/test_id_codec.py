"""Comprehensive unit tests for aragora.connectors.enterprise.database.id_codec."""

from __future__ import annotations

import base64
import json
import uuid

import pytest

from aragora.connectors.enterprise.database.id_codec import (
    PKType,
    _is_hex_hash,
    _pad_base64,
    decode_pk,
    detect_pk_type,
    encode_pk,
    generate_evidence_id,
    is_legacy_id,
    parse_evidence_id,
)


# ---------------------------------------------------------------------------
# 1. PKType enum sanity
# ---------------------------------------------------------------------------
class TestPKTypeEnum:
    def test_values(self):
        assert PKType.INTEGER.value == "i"
        assert PKType.STRING.value == "s"
        assert PKType.UUID.value == "u"
        assert PKType.COMPOSITE.value == "c"


# ---------------------------------------------------------------------------
# 2. detect_pk_type
# ---------------------------------------------------------------------------
class TestDetectPKType:
    """Test primary key type detection."""

    # Integers
    def test_positive_int(self):
        assert detect_pk_type(42) == PKType.INTEGER

    def test_zero(self):
        assert detect_pk_type(0) == PKType.INTEGER

    def test_negative_int(self):
        assert detect_pk_type(-7) == PKType.INTEGER

    def test_large_int(self):
        assert detect_pk_type(10**18) == PKType.INTEGER

    # UUID strings
    def test_uuid_lowercase(self):
        assert detect_pk_type("a1b2c3d4-e5f6-7890-abcd-ef1234567890") == PKType.UUID

    def test_uuid_uppercase(self):
        assert detect_pk_type("A1B2C3D4-E5F6-7890-ABCD-EF1234567890") == PKType.UUID

    def test_uuid_mixed_case(self):
        assert detect_pk_type("a1B2c3D4-E5f6-7890-AbCd-eF1234567890") == PKType.UUID

    def test_uuid_from_uuid_module(self):
        u = uuid.uuid4()
        assert detect_pk_type(str(u)) == PKType.UUID

    # Composite (list / tuple)
    def test_list_composite(self):
        assert detect_pk_type([1, "abc"]) == PKType.COMPOSITE

    def test_tuple_composite(self):
        assert detect_pk_type((10, 20)) == PKType.COMPOSITE

    def test_empty_list_composite(self):
        assert detect_pk_type([]) == PKType.COMPOSITE

    def test_single_element_tuple(self):
        assert detect_pk_type((42,)) == PKType.COMPOSITE

    # Strings
    def test_simple_string(self):
        assert detect_pk_type("hello") == PKType.STRING

    def test_empty_string(self):
        assert detect_pk_type("") == PKType.STRING

    def test_numeric_string(self):
        # Not an int instance, so treated as string
        assert detect_pk_type("123") == PKType.STRING

    def test_uuid_missing_hyphens(self):
        # Without hyphens it does not match UUID regex
        assert detect_pk_type("a1b2c3d4e5f67890abcdef1234567890") == PKType.STRING

    def test_unicode_string(self):
        assert detect_pk_type("\u00e9\u00e0\u00fc") == PKType.STRING

    def test_string_with_special_chars(self):
        assert detect_pk_type("key/with:colons!@#") == PKType.STRING

    def test_bool_detected_as_int(self):
        # In Python bool is a subclass of int
        assert detect_pk_type(True) == PKType.INTEGER


# ---------------------------------------------------------------------------
# 3. encode_pk / decode_pk roundtrip
# ---------------------------------------------------------------------------
class TestEncodePK:
    """Test encode_pk returns expected type indicators and values."""

    def test_integer_encoding(self):
        type_ind, val = encode_pk(42)
        assert type_ind == "i"
        assert val == "42"

    def test_zero_encoding(self):
        type_ind, val = encode_pk(0)
        assert type_ind == "i"
        assert val == "0"

    def test_negative_encoding(self):
        type_ind, val = encode_pk(-99)
        assert type_ind == "i"
        assert val == "-99"

    def test_large_int_encoding(self):
        big = 2**63
        type_ind, val = encode_pk(big)
        assert type_ind == "i"
        assert val == str(big)

    def test_uuid_encoding_strips_hyphens(self):
        u = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        type_ind, val = encode_pk(u)
        assert type_ind == "u"
        assert val == "a1b2c3d4e5f67890abcdef1234567890"
        assert "-" not in val

    def test_uuid_encoding_lowercased(self):
        u = "A1B2C3D4-E5F6-7890-ABCD-EF1234567890"
        type_ind, val = encode_pk(u)
        assert type_ind == "u"
        assert val == "a1b2c3d4e5f67890abcdef1234567890"

    def test_string_encoding_base64url(self):
        type_ind, val = encode_pk("hello")
        assert type_ind == "s"
        # base64url("hello") = "aGVsbG8" (no padding)
        assert val == base64.urlsafe_b64encode(b"hello").decode().rstrip("=")

    def test_empty_string_encoding(self):
        type_ind, val = encode_pk("")
        assert type_ind == "s"
        assert val == ""  # base64 of empty bytes is empty

    def test_string_with_special_chars_encoding(self):
        s = "key/with:colons!@#$"
        type_ind, val = encode_pk(s)
        assert type_ind == "s"
        # Should be valid base64url (no + or /)
        assert "+" not in val

    def test_unicode_string_encoding(self):
        s = "\u00e9\u00e0\u00fc"
        type_ind, val = encode_pk(s)
        assert type_ind == "s"

    def test_composite_encoding_list(self):
        comp = [1, "abc"]
        type_ind, val = encode_pk(comp)
        assert type_ind == "c"
        # Verify the value is base64url encoded JSON
        padded = _pad_base64(val)
        decoded_json = base64.urlsafe_b64decode(padded).decode()
        assert json.loads(decoded_json) == [1, "abc"]

    def test_composite_encoding_tuple(self):
        comp = (10, 20)
        type_ind, val = encode_pk(comp)
        assert type_ind == "c"

    def test_explicit_pk_type_override(self):
        # Force treating "42" as a string
        type_ind, val = encode_pk("42", pk_type=PKType.STRING)
        assert type_ind == "s"


class TestDecodePK:
    """Test decode_pk for each type."""

    def test_integer_decode(self):
        assert decode_pk("i", "42") == 42

    def test_zero_decode(self):
        assert decode_pk("i", "0") == 0

    def test_negative_decode(self):
        assert decode_pk("i", "-99") == -99

    def test_uuid_decode_reformats(self):
        encoded = "a1b2c3d4e5f67890abcdef1234567890"
        result = decode_pk("u", encoded)
        assert result == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_string_decode(self):
        raw = "hello"
        encoded = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
        assert decode_pk("s", encoded) == "hello"

    def test_empty_string_decode(self):
        encoded = base64.urlsafe_b64encode(b"").decode().rstrip("=")
        assert decode_pk("s", encoded) == ""

    def test_composite_decode(self):
        comp = [1, "abc"]
        json_str = json.dumps(comp, separators=(",", ":"))
        encoded = base64.urlsafe_b64encode(json_str.encode()).decode().rstrip("=")
        result = decode_pk("c", encoded)
        assert result == [1, "abc"]


class TestRoundtrip:
    """Test encode -> decode roundtrip for every type."""

    @pytest.mark.parametrize(
        "pk_value",
        [0, 1, -1, 42, 2**63, -(2**31)],
        ids=["zero", "one", "neg_one", "42", "big", "neg_big"],
    )
    def test_integer_roundtrip(self, pk_value):
        type_ind, encoded = encode_pk(pk_value)
        decoded = decode_pk(type_ind, encoded)
        assert decoded == pk_value

    @pytest.mark.parametrize(
        "pk_value",
        [
            "hello",
            "with spaces",
            "key/with:colons",
            "",
            "\u00e9\u00e0\u00fc\u4e16\u754c",
            "a" * 500,
            "line\nnewline",
            "tab\there",
            'quote"inside',
        ],
        ids=[
            "simple",
            "spaces",
            "special",
            "empty",
            "unicode",
            "long",
            "newline",
            "tab",
            "quote",
        ],
    )
    def test_string_roundtrip(self, pk_value):
        type_ind, encoded = encode_pk(pk_value)
        decoded = decode_pk(type_ind, encoded)
        assert decoded == pk_value

    def test_uuid_roundtrip(self):
        u = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        type_ind, encoded = encode_pk(u)
        decoded = decode_pk(type_ind, encoded)
        assert decoded == u.lower()

    def test_uuid_uppercase_roundtrip(self):
        u = "A1B2C3D4-E5F6-7890-ABCD-EF1234567890"
        type_ind, encoded = encode_pk(u)
        decoded = decode_pk(type_ind, encoded)
        # Encoding lowercases, so decoded is lowercase
        assert decoded == u.lower()

    @pytest.mark.parametrize(
        "pk_value",
        [
            [1, 2, 3],
            [1, "abc"],
            ["a", "b", "c"],
            [1],
            [],
            (10, 20),
            (1, "two", 3.0),
        ],
        ids=[
            "int_list",
            "mixed_list",
            "str_list",
            "single",
            "empty",
            "tuple",
            "mixed_tuple",
        ],
    )
    def test_composite_roundtrip(self, pk_value):
        type_ind, encoded = encode_pk(pk_value)
        decoded = decode_pk(type_ind, encoded)
        assert decoded == list(pk_value)


# ---------------------------------------------------------------------------
# 4. _pad_base64
# ---------------------------------------------------------------------------
class TestPadBase64:
    """Test base64 padding helper."""

    def test_no_padding_needed(self):
        assert _pad_base64("abcd") == "abcd"  # len 4, padding 0

    def test_one_pad(self):
        assert _pad_base64("abc") == "abc="  # len 3, need 1

    def test_two_pad(self):
        assert _pad_base64("ab") == "ab=="  # len 2, need 2

    def test_three_pad(self):
        assert _pad_base64("a") == "a==="  # len 1, need 3

    def test_empty_string(self):
        assert _pad_base64("") == ""  # len 0, 4-0%4=4, no padding


# ---------------------------------------------------------------------------
# 5. generate_evidence_id
# ---------------------------------------------------------------------------
class TestGenerateEvidenceID:
    """Test evidence ID generation for all prefixes."""

    def test_postgres_int_pk(self):
        eid = generate_evidence_id("pg", "mydb", "users", 42)
        assert eid == "pg:mydb:users:i:42"

    def test_mysql_string_pk(self):
        eid = generate_evidence_id("mysql", "mydb", "items", "abc")
        parts = eid.split(":")
        assert parts[0] == "mysql"
        assert parts[1] == "mydb"
        assert parts[2] == "items"
        assert parts[3] == "s"
        # Decode the encoded pk to verify
        decoded = decode_pk(parts[3], parts[4])
        assert decoded == "abc"

    def test_mssql_uuid_pk(self):
        u = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        eid = generate_evidence_id("mssql", "db1", "tbl", u)
        parts = eid.split(":")
        assert parts[0] == "mssql"
        assert parts[3] == "u"
        assert parts[4] == "a1b2c3d4e5f67890abcdef1234567890"

    def test_mongo_composite_pk(self):
        eid = generate_evidence_id("mongo", "nosql_db", "docs", [1, "two"])
        parts = eid.split(":")
        assert parts[0] == "mongo"
        assert parts[3] == "c"
        # Roundtrip the pk
        decoded = decode_pk(parts[3], parts[4])
        assert decoded == [1, "two"]

    def test_snowflake_with_account(self):
        eid = generate_evidence_id("sf", "warehouse", "sales", 99, account="acme_org")
        assert eid.startswith("sf:acme_org:warehouse:sales:i:99")
        parts = eid.split(":")
        assert len(parts) == 6
        assert parts[1] == "acme_org"

    def test_snowflake_without_account(self):
        # sf without account should use 5-part format
        eid = generate_evidence_id("sf", "warehouse", "sales", 99)
        parts = eid.split(":")
        assert len(parts) == 5
        assert parts[0] == "sf"

    def test_non_sf_prefix_ignores_account(self):
        # account is only used when prefix == "sf"
        eid = generate_evidence_id("pg", "mydb", "users", 1, account="ignored")
        parts = eid.split(":")
        assert len(parts) == 5
        assert parts[0] == "pg"


# ---------------------------------------------------------------------------
# 6. parse_evidence_id - new format (5-part and 6-part)
# ---------------------------------------------------------------------------
class TestParseEvidenceIDNewFormat:
    """Test parsing new reversible evidence IDs."""

    def test_parse_5_part_integer(self):
        eid = "pg:mydb:users:i:42"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["prefix"] == "pg"
        assert result["database"] == "mydb"
        assert result["table"] == "users"
        assert result["pk_type"] == "i"
        assert result["pk_value"] == 42
        assert result["is_legacy"] is False

    def test_parse_5_part_string(self):
        _, encoded = encode_pk("hello")
        eid = f"mysql:db:tbl:s:{encoded}"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["pk_type"] == "s"
        assert result["pk_value"] == "hello"
        assert result["is_legacy"] is False

    def test_parse_5_part_uuid(self):
        u = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        _, encoded = encode_pk(u)
        eid = f"mssql:db:tbl:u:{encoded}"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["pk_type"] == "u"
        assert result["pk_value"] == u

    def test_parse_5_part_composite(self):
        _, encoded = encode_pk([1, "two"])
        eid = f"pg:db:tbl:c:{encoded}"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["pk_type"] == "c"
        assert result["pk_value"] == [1, "two"]

    def test_parse_6_part_snowflake(self):
        eid = "sf:acme:warehouse:sales:i:99"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["prefix"] == "sf"
        assert result["account"] == "acme"
        assert result["database"] == "warehouse"
        assert result["table"] == "sales"
        assert result["pk_type"] == "i"
        assert result["pk_value"] == 99
        assert result["is_legacy"] is False

    def test_parse_6_part_snowflake_string(self):
        _, encoded = encode_pk("row_key")
        eid = f"sf:acme:wh:tbl:s:{encoded}"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["pk_value"] == "row_key"
        assert result["account"] == "acme"


# ---------------------------------------------------------------------------
# 7. parse_evidence_id - legacy hash-based IDs
# ---------------------------------------------------------------------------
class TestParseEvidenceIDLegacy:
    """Test parsing legacy hash-based evidence IDs."""

    def test_legacy_4_part_12_char_hash(self):
        # 12-char hex hash
        eid = "pg:mydb:users:abcdef123456"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["is_legacy"] is True
        assert result["prefix"] == "pg"
        assert result["database"] == "mydb"
        assert result["table"] == "users"
        assert result["pk_type"] is None
        assert result["pk_value"] is None
        assert result["pk_hash"] == "abcdef123456"

    def test_legacy_4_part_16_char_hash(self):
        eid = "mongo:nosql:docs:abcdef1234567890"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["is_legacy"] is True
        assert result["pk_hash"] == "abcdef1234567890"

    def test_legacy_5_part_snowflake(self):
        # sf:account:db:table:hash
        eid = "sf:acme:warehouse:sales:abcdef123456"
        result = parse_evidence_id(eid)
        assert result is not None
        assert result["is_legacy"] is True
        assert result["prefix"] == "sf"
        assert result["account"] == "acme"
        assert result["database"] == "warehouse"
        assert result["table"] == "sales"
        assert result["pk_hash"] == "abcdef123456"

    def test_legacy_4_part_not_hex(self):
        # Not a hex hash (contains 'g')
        eid = "pg:mydb:users:abcdefghijkl"
        result = parse_evidence_id(eid)
        assert result is None

    def test_legacy_4_part_wrong_length_hash(self):
        # Hash is 10 chars, not 12 or 16
        eid = "pg:mydb:users:abcdef1234"
        result = parse_evidence_id(eid)
        assert result is None


# ---------------------------------------------------------------------------
# 8. is_legacy_id
# ---------------------------------------------------------------------------
class TestIsLegacyID:
    """Test legacy ID detection."""

    def test_legacy_4_part(self):
        assert is_legacy_id("pg:mydb:users:abcdef123456") is True

    def test_legacy_5_part_snowflake(self):
        assert is_legacy_id("sf:acme:wh:tbl:abcdef123456") is True

    def test_new_5_part(self):
        assert is_legacy_id("pg:mydb:users:i:42") is False

    def test_new_6_part_snowflake(self):
        assert is_legacy_id("sf:acme:wh:tbl:i:99") is False

    def test_invalid_id_returns_false(self):
        # parse_evidence_id returns None, so is_legacy returns False
        assert is_legacy_id("totally_invalid") is False

    def test_empty_string_returns_false(self):
        assert is_legacy_id("") is False


# ---------------------------------------------------------------------------
# 9. _is_hex_hash
# ---------------------------------------------------------------------------
class TestIsHexHash:
    """Test hex hash detection helper."""

    def test_12_char_hex(self):
        assert _is_hex_hash("abcdef123456") is True

    def test_16_char_hex(self):
        assert _is_hex_hash("abcdef1234567890") is True

    def test_wrong_length_10(self):
        assert _is_hex_hash("abcdef1234") is False

    def test_wrong_length_14(self):
        assert _is_hex_hash("abcdef12345678") is False

    def test_non_hex_chars(self):
        assert _is_hex_hash("abcdefghijkl") is False

    def test_uppercase_hex_rejected(self):
        # The function checks lowercase only
        assert _is_hex_hash("ABCDEF123456") is False

    def test_empty_string(self):
        assert _is_hex_hash("") is False


# ---------------------------------------------------------------------------
# 10. Edge cases and error handling
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases and malformed inputs."""

    def test_parse_single_part(self):
        assert parse_evidence_id("single") is None

    def test_parse_two_parts(self):
        assert parse_evidence_id("a:b") is None

    def test_parse_three_parts(self):
        assert parse_evidence_id("a:b:c") is None

    def test_parse_seven_parts(self):
        assert parse_evidence_id("a:b:c:d:e:f:g") is None

    def test_parse_empty_string(self):
        assert parse_evidence_id("") is None

    def test_parse_5_part_invalid_pk_type(self):
        eid = "pg:db:tbl:x:42"
        result = parse_evidence_id(eid)
        assert result is None

    def test_parse_5_part_decode_failure(self):
        # c type with invalid base64
        eid = "pg:db:tbl:c:!!invalid!!"
        result = parse_evidence_id(eid)
        assert result is None

    def test_parse_6_part_invalid_pk_type(self):
        eid = "sf:acme:db:tbl:z:42"
        result = parse_evidence_id(eid)
        assert result is None

    def test_parse_6_part_decode_failure(self):
        eid = "sf:acme:db:tbl:c:!!invalid!!"
        result = parse_evidence_id(eid)
        assert result is None

    def test_generate_and_parse_roundtrip_pg(self):
        eid = generate_evidence_id("pg", "mydb", "users", 42)
        parsed = parse_evidence_id(eid)
        assert parsed is not None
        assert parsed["prefix"] == "pg"
        assert parsed["database"] == "mydb"
        assert parsed["table"] == "users"
        assert parsed["pk_value"] == 42
        assert parsed["is_legacy"] is False

    def test_generate_and_parse_roundtrip_sf(self):
        eid = generate_evidence_id("sf", "wh", "sales", "row_key", account="acme")
        parsed = parse_evidence_id(eid)
        assert parsed is not None
        assert parsed["prefix"] == "sf"
        assert parsed["account"] == "acme"
        assert parsed["pk_value"] == "row_key"
        assert parsed["is_legacy"] is False

    def test_generate_and_parse_roundtrip_uuid(self):
        u = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        eid = generate_evidence_id("pg", "db", "tbl", u)
        parsed = parse_evidence_id(eid)
        assert parsed is not None
        assert parsed["pk_value"] == u

    def test_generate_and_parse_roundtrip_composite(self):
        comp = [1, "abc", 3.14]
        eid = generate_evidence_id("mongo", "nosql", "docs", comp)
        parsed = parse_evidence_id(eid)
        assert parsed is not None
        assert parsed["pk_value"] == comp

    def test_base64_padding_1_mod(self):
        """Strings producing base64 lengths needing 1 pad char."""
        s = "ab"  # base64 -> "YWI" (3 chars, needs 1 pad)
        type_ind, encoded = encode_pk(s)
        decoded = decode_pk(type_ind, encoded)
        assert decoded == s

    def test_base64_padding_2_mod(self):
        """Strings producing base64 lengths needing 2 pad chars."""
        s = "a"  # base64 -> "YQ" (2 chars, needs 2 pads)
        type_ind, encoded = encode_pk(s)
        decoded = decode_pk(type_ind, encoded)
        assert decoded == s

    def test_base64_padding_0_mod(self):
        """Strings producing base64 lengths needing no padding."""
        s = "abc"  # base64 -> "YWJj" (4 chars, no pad)
        type_ind, encoded = encode_pk(s)
        decoded = decode_pk(type_ind, encoded)
        assert decoded == s

    def test_string_with_colons_roundtrip(self):
        """Colons in PK must not break colon-delimited evidence ID."""
        s = "key:with:colons"
        eid = generate_evidence_id("pg", "db", "tbl", s)
        parsed = parse_evidence_id(eid)
        # Colons in the base64 encoded string would add extra parts,
        # but base64url does not use colons, so this should work.
        assert parsed is not None
        assert parsed["pk_value"] == s

    def test_int_pk_with_explicit_type(self):
        """Passing explicit PKType.INTEGER."""
        type_ind, val = encode_pk(42, pk_type=PKType.INTEGER)
        assert type_ind == "i"
        assert decode_pk(type_ind, val) == 42

    def test_composite_preserves_order(self):
        comp = [3, 1, 2]
        type_ind, encoded = encode_pk(comp)
        decoded = decode_pk(type_ind, encoded)
        assert decoded == [3, 1, 2]

    def test_generate_all_standard_prefixes(self):
        """Ensure all documented prefixes produce valid IDs."""
        for prefix in ("pg", "mysql", "mssql", "mongo"):
            eid = generate_evidence_id(prefix, "db", "tbl", 1)
            parsed = parse_evidence_id(eid)
            assert parsed is not None, f"Failed for prefix {prefix}"
            assert parsed["prefix"] == prefix
            assert parsed["pk_value"] == 1
