"""
Encryption performance benchmarks.

Measures encryption/decryption performance for:
- Single field encryption
- Batch field encryption
- Various payload sizes
- Migration throughput
"""

from __future__ import annotations

import os
import time
import json
import secrets
from typing import Dict, Any, List

import pytest

from .conftest import SimpleBenchmark


# Skip if no encryption key available
pytestmark = pytest.mark.skipif(
    not os.environ.get("ARAGORA_ENCRYPTION_KEY"), reason="ARAGORA_ENCRYPTION_KEY not set"
)


def _generate_test_data(size_bytes: int) -> str:
    """Generate random test data of specified size."""
    return secrets.token_urlsafe(size_bytes)


def _generate_test_record(num_fields: int = 5, field_size: int = 100) -> Dict[str, Any]:
    """Generate test record with multiple fields."""
    return {
        "id": secrets.token_hex(8),
        "name": "Test Record",
        **{f"field_{i}": _generate_test_data(field_size) for i in range(num_fields)},
    }


class TestEncryptionLatency:
    """Benchmark encryption latency for various operations."""

    def test_single_field_encrypt_latency(self):
        """Measure latency for encrypting a single field."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()
        data = {"api_key": "sk-test-key-123456789"}

        bench = SimpleBenchmark("single_field_encrypt")

        # Warmup
        for _ in range(3):
            service.encrypt_fields(data.copy(), ["api_key"])

        # Benchmark
        for _ in range(100):
            bench(lambda: service.encrypt_fields(data.copy(), ["api_key"]))

        assert bench.mean < 0.01  # Should be under 10ms
        print(
            f"\nSingle field encrypt: mean={bench.mean * 1000:.3f}ms, "
            f"min={bench.min * 1000:.3f}ms, max={bench.max * 1000:.3f}ms"
        )

    def test_single_field_decrypt_latency(self):
        """Measure latency for decrypting a single field."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()
        data = {"api_key": "sk-test-key-123456789"}
        encrypted = service.encrypt_fields(data.copy(), ["api_key"])

        bench = SimpleBenchmark("single_field_decrypt")

        # Warmup
        for _ in range(3):
            service.decrypt_fields(encrypted.copy(), ["api_key"])

        # Benchmark
        for _ in range(100):
            bench(lambda: service.decrypt_fields(encrypted.copy(), ["api_key"]))

        assert bench.mean < 0.01  # Should be under 10ms
        print(
            f"\nSingle field decrypt: mean={bench.mean * 1000:.3f}ms, "
            f"min={bench.min * 1000:.3f}ms, max={bench.max * 1000:.3f}ms"
        )

    def test_multiple_field_encrypt_latency(self):
        """Measure latency for encrypting multiple fields."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()
        data = {
            "api_key": "sk-test-key-123456789",
            "api_secret": "secret-987654321",
            "access_token": "access-token-abcdef",
            "refresh_token": "refresh-token-ghijkl",
        }
        fields = list(data.keys())

        bench = SimpleBenchmark("multi_field_encrypt")

        for _ in range(100):
            bench(lambda: service.encrypt_fields(data.copy(), fields))

        assert bench.mean < 0.05  # Should be under 50ms
        print(
            f"\n4-field encrypt: mean={bench.mean * 1000:.3f}ms, "
            f"min={bench.min * 1000:.3f}ms, max={bench.max * 1000:.3f}ms"
        )

    def test_large_payload_encrypt_latency(self):
        """Measure latency for encrypting large payloads."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()

        sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
        results = []

        for size in sizes:
            data = {"large_field": _generate_test_data(size)}
            bench = SimpleBenchmark(f"encrypt_{size}b")

            for _ in range(20):
                bench(lambda: service.encrypt_fields(data.copy(), ["large_field"]))

            results.append((size, bench.mean))
            print(f"\n{size}B encrypt: mean={bench.mean * 1000:.3f}ms")

        # Verify linear or sub-linear scaling (not exponential)
        for i in range(1, len(results)):
            ratio = results[i][0] / results[i - 1][0]  # Size ratio (10x)
            time_ratio = results[i][1] / results[i - 1][1]  # Time ratio
            # Time should grow slower than data size (not 10x slower for 10x data)
            assert time_ratio < ratio * 2, "Encryption scaling is too slow"


class TestEncryptionThroughput:
    """Benchmark encryption throughput."""

    def test_batch_encrypt_throughput(self):
        """Measure records per second for batch encryption."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()
        sensitive_fields = ["api_key", "api_secret"]

        # Generate batch of records
        records = [
            {
                "id": f"record_{i}",
                "api_key": f"key_{i}",
                "api_secret": f"secret_{i}",
                "name": f"Record {i}",
            }
            for i in range(100)
        ]

        start = time.perf_counter()
        for record in records:
            service.encrypt_fields(record.copy(), sensitive_fields)
        elapsed = time.perf_counter() - start

        records_per_sec = len(records) / elapsed
        print(f"\nBatch encrypt throughput: {records_per_sec:.1f} records/sec")

        # Should handle at least 500 records per second
        assert records_per_sec > 500, f"Throughput too low: {records_per_sec}"

    def test_mixed_workload_throughput(self):
        """Measure throughput for mixed encrypt/decrypt workload."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()
        sensitive_fields = ["api_key", "refresh_token"]

        # Pre-generate encrypted data
        plaintext_records = [
            {
                "id": f"record_{i}",
                "api_key": f"key_{i}",
                "refresh_token": f"token_{i}",
            }
            for i in range(50)
        ]
        encrypted_records = [
            service.encrypt_fields(r.copy(), sensitive_fields) for r in plaintext_records
        ]

        # Mix: 70% reads (decrypt), 30% writes (encrypt)
        operations = 0
        start = time.perf_counter()

        for i in range(100):
            if i % 10 < 7:
                # Decrypt (read)
                service.decrypt_fields(encrypted_records[i % 50].copy(), sensitive_fields)
            else:
                # Encrypt (write)
                service.encrypt_fields(plaintext_records[i % 50].copy(), sensitive_fields)
            operations += 1

        elapsed = time.perf_counter() - start
        ops_per_sec = operations / elapsed
        print(f"\nMixed workload: {ops_per_sec:.1f} ops/sec")

        # Should handle at least 500 ops per second
        assert ops_per_sec > 500


class TestMigrationPerformance:
    """Benchmark migration performance."""

    def test_migration_throughput(self):
        """Measure migration throughput."""
        from aragora.security.encryption import get_encryption_service
        from aragora.security.migration import EncryptionMigrator, needs_migration

        service = get_encryption_service()
        migrator = EncryptionMigrator(encryption_service=service)

        sensitive_fields = ["api_key", "password"]

        # Generate records
        records = [
            {
                "id": f"record_{i}",
                "api_key": f"plaintext_key_{i}",
                "password": f"plaintext_password_{i}",
                "name": f"Record {i}",
            }
            for i in range(100)
        ]

        migrated_count = 0
        start = time.perf_counter()

        for record in records:
            if needs_migration(record, sensitive_fields):
                migrator.migrate_record(record, sensitive_fields, record_id=record["id"])
                migrated_count += 1

        elapsed = time.perf_counter() - start
        records_per_sec = migrated_count / elapsed

        print(f"\nMigration throughput: {records_per_sec:.1f} records/sec")
        assert records_per_sec > 200, f"Migration too slow: {records_per_sec}"

    def test_migration_skip_already_encrypted(self):
        """Verify skipping already-encrypted records is fast."""
        from aragora.security.encryption import get_encryption_service
        from aragora.security.migration import needs_migration

        service = get_encryption_service()
        sensitive_fields = ["api_key"]

        # Pre-encrypt records
        records = [
            service.encrypt_fields({"id": f"record_{i}", "api_key": f"key_{i}"}, sensitive_fields)
            for i in range(1000)
        ]

        start = time.perf_counter()
        checked = 0
        for record in records:
            if needs_migration(record, sensitive_fields):
                checked += 1
        elapsed = time.perf_counter() - start

        print(f"\nSkip check: {len(records) / elapsed:.1f} records/sec, {checked} needed migration")

        # Checking should be very fast (> 10,000 records/sec)
        assert len(records) / elapsed > 10000
        assert checked == 0  # All should be already encrypted


class TestMemoryUsage:
    """Benchmark memory characteristics of encryption."""

    def test_encryption_memory_overhead(self):
        """Measure memory overhead of encrypted vs plaintext."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()

        # Test with realistic payload sizes
        # Encryption adds fixed overhead: ~100-130B for nonce, tag, algorithm, and base64 encoding
        test_values = [
            ("medium-length-api-key-value-here", 600),  # ~32B, high relative overhead
            ("x" * 100, 200),  # 100B
            ("x" * 1000, 50),  # 1KB - base64 is ~33% overhead plus fixed
            ("y" * 10000, 50),  # 10KB - amortized overhead is low
        ]

        for value, max_overhead in test_values:
            data = {"field": value}
            encrypted = service.encrypt_fields(data.copy(), ["field"])

            plain_size = len(json.dumps(data))
            enc_size = len(json.dumps(encrypted))
            overhead = (enc_size - plain_size) / plain_size * 100

            print(f"\nPlaintext {len(value)}B -> Encrypted {enc_size}B ({overhead:+.1f}% overhead)")

            # Verify overhead is within expected bounds
            # AES-GCM + base64 encoding has ~37% data expansion + ~100B fixed overhead
            assert overhead < max_overhead, f"Too much overhead for {len(value)}B: {overhead}%"


class TestRoundTrip:
    """Verify encryption round-trip correctness."""

    def test_roundtrip_various_types(self):
        """Verify round-trip preserves data values for common secret patterns."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()

        # Test realistic secret values (API keys, tokens, passwords)
        # The encryption service preserves string values exactly
        test_cases = [
            {"api_key": "sk-1234567890abcdef"},
            {"unicode_key": "key_\u4e16\u754c_123"},
            {"empty_token": ""},
            {"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"},
            {"special_password": "P@ssw0rd!#$%^&*()"},
            {"multiline_cert": "-----BEGIN CERT-----\nABC123\n-----END CERT-----"},
            {"long_key": "x" * 100000},
        ]

        for data in test_cases:
            field = list(data.keys())[0]
            original_value = data[field]

            encrypted = service.encrypt_fields(data.copy(), [field])
            decrypted = service.decrypt_fields(encrypted, [field])

            assert decrypted[field] == original_value, f"Round-trip failed for {field}"

    def test_associated_data_binding(self):
        """Verify associated data prevents cross-record attacks."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()

        # Encrypt with associated data
        data = {"api_key": "secret123"}
        encrypted = service.encrypt_fields(data.copy(), ["api_key"], associated_data="record_1")

        # Decrypt with same associated data should work
        decrypted = service.decrypt_fields(
            encrypted.copy(), ["api_key"], associated_data="record_1"
        )
        assert decrypted["api_key"] == "secret123"

        # Decrypt with different associated data should fail
        try:
            service.decrypt_fields(encrypted.copy(), ["api_key"], associated_data="record_2")
            pytest.fail("Should have failed with wrong associated data")
        except Exception:
            pass  # Expected - authentication failed
