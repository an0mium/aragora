"""
SDK Benchmark Suite.

Performance benchmarks for the Aragora Python SDK.

Run with:
    pytest tests/benchmarks/test_sdk_benchmarks.py -v --benchmark-only

Requires pytest-benchmark:
    pip install pytest-benchmark
"""

import asyncio
import json
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pytest_benchmark")

# Check if aragora_sdk is installed
try:
    import aragora_sdk

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

# Skip all SDK tests if aragora_sdk is not installed
pytestmark = pytest.mark.skipif(not HAS_SDK, reason="aragora_sdk not installed")


class TestSDKClientBenchmarks:
    """Benchmark SDK client operations."""

    @pytest.fixture
    def mock_response(self):
        """Create mock API response."""
        return {
            "id": "review_123",
            "consensus": {
                "status": "reached",
                "position": "The design meets security requirements.",
                "confidence": 0.85,
            },
            "dissenting_opinions": [],
            "findings": [
                {
                    "title": "Input Validation",
                    "description": "Add validation for user inputs",
                    "severity": "medium",
                }
            ],
            "personas": ["security", "performance"],
            "rounds_used": 3,
        }

    def test_review_result_parsing(self, benchmark, mock_response):
        """Benchmark parsing review result from API response."""
        from aragora_sdk.models import ReviewResult

        def parse_result():
            return ReviewResult(**mock_response)

        result = benchmark(parse_result)
        assert result.id == "review_123"
        assert result.consensus.status == "reached"

    def test_review_request_serialization(self, benchmark):
        """Benchmark serializing review request."""
        from aragora_sdk.models import ReviewRequest

        def serialize():
            request = ReviewRequest(
                spec="# Design Document\n\nThis is a test specification...",
                personas=["security", "sox", "performance"],
                rounds=3,
                include_receipt=True,
            )
            return request.model_dump(exclude_none=True)

        result = benchmark(serialize)
        assert result["personas"] == ["security", "sox", "performance"]


class TestModelsBenchmarks:
    """Benchmark model operations."""

    def test_consensus_result_creation(self, benchmark):
        """Benchmark ConsensusResult creation."""
        from aragora_sdk.models import ConsensusResult, ConsensusStatus

        def create():
            return ConsensusResult(
                status=ConsensusStatus.REACHED,
                position="All agents agree the design is sound.",
                confidence=0.92,
            )

        result = benchmark(create)
        assert result.status == ConsensusStatus.REACHED

    def test_decision_receipt_creation(self, benchmark):
        """Benchmark DecisionReceipt creation with full data."""
        from aragora_sdk.models import (
            Critique,
            DecisionReceipt,
            Position,
            Vote,
        )

        def create():
            return DecisionReceipt(
                id="receipt_123",
                checksum="abc123def456",
                timestamp=datetime.now(),
                positions=[
                    Position(
                        id="pos_1",
                        agent="security",
                        content="The design needs input validation.",
                        round=1,
                        timestamp=datetime.now(),
                    )
                    for _ in range(5)
                ],
                critiques=[
                    Critique(
                        id="crit_1",
                        agent="performance",
                        target_agent="security",
                        content="I agree with the input validation concern.",
                        round=1,
                        timestamp=datetime.now(),
                    )
                    for _ in range(10)
                ],
                votes=[
                    Vote(
                        agent="security",
                        position_id="pos_1",
                        support=True,
                        confidence=0.8,
                    )
                    for _ in range(5)
                ],
            )

        result = benchmark(create)
        assert len(result.positions) == 5
        assert len(result.critiques) == 10

    def test_dissenting_opinion_serialization(self, benchmark):
        """Benchmark DissentingOpinion serialization."""
        from aragora_sdk.models import DissentingOpinion

        opinion = DissentingOpinion(
            agent="pci_dss",
            position="The tokenization approach should clarify PAN handling.",
            reasoning="PCI-DSS requires explicit documentation of CHD flows.",
            concerns=["Log redaction not specified", "Key rotation missing"],
            confidence=0.75,
        )

        def serialize():
            return opinion.model_dump()

        result = benchmark(serialize)
        assert result["agent"] == "pci_dss"


class TestExceptionBenchmarks:
    """Benchmark exception handling."""

    def test_aragora_error_creation(self, benchmark):
        """Benchmark error creation."""
        from aragora_sdk.exceptions import AragoraError

        def create_error():
            return AragoraError("Test error message", status_code=500)

        result = benchmark(create_error)
        assert result.message == "Test error message"

    def test_rate_limit_error_creation(self, benchmark):
        """Benchmark rate limit error with retry info."""
        from aragora_sdk.exceptions import RateLimitError

        def create_error():
            return RateLimitError("Rate limit exceeded", retry_after=60)

        result = benchmark(create_error)
        assert result.retry_after == 60


class TestURLBuildingBenchmarks:
    """Benchmark URL construction."""

    def test_url_join(self, benchmark):
        """Benchmark URL path joining."""
        from urllib.parse import urljoin

        base = "https://api.aragora.ai"

        def build_url():
            return urljoin(base, "/api/review")

        result = benchmark(build_url)
        assert result == "https://api.aragora.ai/api/review"


class TestValidationBenchmarks:
    """Benchmark input validation."""

    def test_persona_validation(self, benchmark):
        """Benchmark persona list validation."""
        valid_personas = {
            "security",
            "performance",
            "sox",
            "pci_dss",
            "hipaa",
            "gdpr",
            "architecture",
        }

        def validate(personas):
            return all(p in valid_personas for p in personas)

        personas = ["security", "sox", "hipaa"]
        result = benchmark(lambda: validate(personas))
        assert result is True

    def test_rounds_validation(self, benchmark):
        """Benchmark rounds parameter validation."""

        def validate_rounds(rounds):
            if not isinstance(rounds, int):
                raise ValueError("rounds must be an integer")
            if rounds < 1 or rounds > 10:
                raise ValueError("rounds must be between 1 and 10")
            return True

        result = benchmark(lambda: validate_rounds(3))
        assert result is True


class TestJSONProcessingBenchmarks:
    """Benchmark JSON processing for API communication."""

    @pytest.fixture
    def large_response(self):
        """Create a large API response for benchmarking."""
        return {
            "id": "review_123",
            "consensus": {
                "status": "reached",
                "position": "A" * 5000,  # Large text
                "confidence": 0.85,
            },
            "dissenting_opinions": [
                {
                    "agent": f"agent_{i}",
                    "position": f"Position text {i}" * 100,
                    "concerns": [f"Concern {j}" for j in range(5)],
                }
                for i in range(3)
            ],
            "findings": [
                {
                    "title": f"Finding {i}",
                    "description": f"Description {i}" * 50,
                    "severity": "medium",
                }
                for i in range(20)
            ],
            "decision_receipt": {
                "id": "receipt_123",
                "checksum": "abcdef123456",
                "positions": [{"content": f"Position {i}" * 100} for i in range(10)],
                "critiques": [{"content": f"Critique {i}" * 50} for i in range(30)],
            },
        }

    def test_json_encode_large_response(self, benchmark, large_response):
        """Benchmark encoding large response to JSON."""

        def encode():
            return json.dumps(large_response)

        result = benchmark(encode)
        assert len(result) > 10000

    def test_json_decode_large_response(self, benchmark, large_response):
        """Benchmark decoding large JSON response."""
        json_str = json.dumps(large_response)

        def decode():
            return json.loads(json_str)

        result = benchmark(decode)
        assert result["id"] == "review_123"


class TestAsyncBenchmarks:
    """Benchmark async operations."""

    @pytest.mark.asyncio
    async def test_async_context_manager_overhead(self, benchmark):
        """Benchmark async context manager overhead."""

        class MockClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        async def use_client():
            async with MockClient() as client:
                return client

        result = benchmark(lambda: asyncio.get_event_loop().run_until_complete(use_client()))
        assert result is not None


# SDK Performance Thresholds (in milliseconds)
SDK_PERFORMANCE_THRESHOLDS = {
    "review_result_parsing": 1.0,  # 1ms
    "review_request_serialization": 0.5,  # 0.5ms
    "consensus_result_creation": 0.1,  # 0.1ms
    "decision_receipt_creation": 1.0,  # 1ms
    "json_encode_large": 5.0,  # 5ms
    "json_decode_large": 5.0,  # 5ms
    "url_building": 0.01,  # 0.01ms
    "validation": 0.1,  # 0.1ms
}


def check_sdk_performance(benchmark_results: dict[str, float]) -> bool:
    """
    Check if SDK performance meets thresholds.

    Args:
        benchmark_results: Dict mapping benchmark name to latency in ms

    Returns:
        True if all benchmarks pass, False otherwise
    """
    for name, latency in benchmark_results.items():
        threshold = SDK_PERFORMANCE_THRESHOLDS.get(name)
        if threshold and latency > threshold:
            return False
    return True
