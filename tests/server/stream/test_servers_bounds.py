"""
Tests for WebSocket and voice stream configuration bounds validation.

Tests that configuration parameters are properly validated and clamped to safe bounds,
preventing misconfiguration that could lead to resource exhaustion or denial of service.
"""

from __future__ import annotations

import importlib
import logging
import os
from unittest import mock

import pytest


class TestWebSocketBounds:
    """Tests for WebSocket configuration bounds in servers.py."""

    def test_ws_conn_rate_default(self):
        """Test default connection rate is within bounds."""
        # Import fresh module with default env
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove any existing env vars
            os.environ.pop("ARAGORA_WS_CONN_RATE", None)
            os.environ.pop("ARAGORA_WS_MAX_PER_IP", None)

            # Force reimport
            import aragora.server.stream.servers as servers_module

            importlib.reload(servers_module)

            assert servers_module.WS_CONNECTIONS_PER_IP_PER_MINUTE == 30
            assert 1 <= servers_module.WS_CONNECTIONS_PER_IP_PER_MINUTE <= 1000

    def test_ws_conn_rate_below_minimum(self, caplog):
        """Test connection rate below minimum is clamped to 1."""
        with mock.patch.dict(os.environ, {"ARAGORA_WS_CONN_RATE": "0"}, clear=False):
            import aragora.server.stream.servers as servers_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(servers_module)

            assert servers_module.WS_CONNECTIONS_PER_IP_PER_MINUTE == 1
            assert "ARAGORA_WS_CONN_RATE=0 out of bounds [1, 1000]" in caplog.text

    def test_ws_conn_rate_above_maximum(self, caplog):
        """Test connection rate above maximum is clamped to 1000."""
        with mock.patch.dict(os.environ, {"ARAGORA_WS_CONN_RATE": "9999"}, clear=False):
            import aragora.server.stream.servers as servers_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(servers_module)

            assert servers_module.WS_CONNECTIONS_PER_IP_PER_MINUTE == 1000
            assert "ARAGORA_WS_CONN_RATE=9999 out of bounds [1, 1000]" in caplog.text

    def test_ws_conn_rate_valid_value(self, caplog):
        """Test valid connection rate passes without warning."""
        with mock.patch.dict(os.environ, {"ARAGORA_WS_CONN_RATE": "50"}, clear=False):
            import aragora.server.stream.servers as servers_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(servers_module)

            assert servers_module.WS_CONNECTIONS_PER_IP_PER_MINUTE == 50
            assert "ARAGORA_WS_CONN_RATE" not in caplog.text

    def test_ws_max_per_ip_default(self):
        """Test default max connections per IP is within bounds."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARAGORA_WS_CONN_RATE", None)
            os.environ.pop("ARAGORA_WS_MAX_PER_IP", None)

            import aragora.server.stream.servers as servers_module

            importlib.reload(servers_module)

            assert servers_module.WS_MAX_CONNECTIONS_PER_IP == 10
            assert 1 <= servers_module.WS_MAX_CONNECTIONS_PER_IP <= 100

    def test_ws_max_per_ip_below_minimum(self, caplog):
        """Test max per IP below minimum is clamped to 1."""
        with mock.patch.dict(os.environ, {"ARAGORA_WS_MAX_PER_IP": "-5"}, clear=False):
            import aragora.server.stream.servers as servers_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(servers_module)

            assert servers_module.WS_MAX_CONNECTIONS_PER_IP == 1
            assert "ARAGORA_WS_MAX_PER_IP=-5 out of bounds [1, 100]" in caplog.text

    def test_ws_max_per_ip_above_maximum(self, caplog):
        """Test max per IP above maximum is clamped to 100."""
        with mock.patch.dict(os.environ, {"ARAGORA_WS_MAX_PER_IP": "500"}, clear=False):
            import aragora.server.stream.servers as servers_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(servers_module)

            assert servers_module.WS_MAX_CONNECTIONS_PER_IP == 100
            assert "ARAGORA_WS_MAX_PER_IP=500 out of bounds [1, 100]" in caplog.text

    def test_ws_max_per_ip_valid_value(self, caplog):
        """Test valid max per IP passes without warning."""
        with mock.patch.dict(os.environ, {"ARAGORA_WS_MAX_PER_IP": "25"}, clear=False):
            import aragora.server.stream.servers as servers_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(servers_module)

            assert servers_module.WS_MAX_CONNECTIONS_PER_IP == 25
            assert "ARAGORA_WS_MAX_PER_IP" not in caplog.text


class TestVoiceStreamBounds:
    """Tests for voice stream configuration bounds in voice_stream.py."""

    def test_voice_chunk_size_default(self):
        """Test default chunk size is within bounds."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Clear all voice-related env vars
            for key in list(os.environ.keys()):
                if key.startswith("ARAGORA_VOICE"):
                    del os.environ[key]

            import aragora.server.stream.voice_stream as voice_module

            importlib.reload(voice_module)

            # Default is 16000 * 2 * 3 = 96000 bytes
            assert voice_module.VOICE_CHUNK_SIZE_BYTES == 96000
            assert 1024 <= voice_module.VOICE_CHUNK_SIZE_BYTES <= 1048576

    def test_voice_chunk_size_below_minimum(self, caplog):
        """Test chunk size below minimum is clamped to 1KB."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_CHUNK_SIZE": "100"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_CHUNK_SIZE_BYTES == 1024
            assert "ARAGORA_VOICE_CHUNK_SIZE=100 out of bounds [1024, 1048576]" in caplog.text

    def test_voice_chunk_size_above_maximum(self, caplog):
        """Test chunk size above maximum is clamped to 1MB."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_CHUNK_SIZE": "2000000"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_CHUNK_SIZE_BYTES == 1048576
            assert "ARAGORA_VOICE_CHUNK_SIZE=2000000 out of bounds [1024, 1048576]" in caplog.text

    def test_voice_max_session_below_minimum(self, caplog):
        """Test max session below minimum is clamped to 10 seconds."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_MAX_SESSION": "5"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_SESSION_SECONDS == 10
            assert "ARAGORA_VOICE_MAX_SESSION=5 out of bounds [10, 3600]" in caplog.text

    def test_voice_max_session_above_maximum(self, caplog):
        """Test max session above maximum is clamped to 1 hour."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_MAX_SESSION": "7200"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_SESSION_SECONDS == 3600
            assert "ARAGORA_VOICE_MAX_SESSION=7200 out of bounds [10, 3600]" in caplog.text

    def test_voice_max_buffer_below_minimum(self, caplog):
        """Test max buffer below minimum is clamped to 1MB."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_MAX_BUFFER": "512000"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_BUFFER_BYTES == 1048576
            assert (
                "ARAGORA_VOICE_MAX_BUFFER=512000 out of bounds [1048576, 536870912]" in caplog.text
            )

    def test_voice_max_buffer_above_maximum(self, caplog):
        """Test max buffer above maximum is clamped to 512MB."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_MAX_BUFFER": "1000000000"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_BUFFER_BYTES == 536870912
            assert (
                "ARAGORA_VOICE_MAX_BUFFER=1000000000 out of bounds [1048576, 536870912]"
                in caplog.text
            )

    def test_voice_transcribe_interval_below_minimum(self, caplog):
        """Test transcribe interval below minimum is clamped to 100ms."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_INTERVAL": "50"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_TRANSCRIBE_INTERVAL_MS == 100
            assert "ARAGORA_VOICE_INTERVAL=50 out of bounds [100, 30000]" in caplog.text

    def test_voice_transcribe_interval_above_maximum(self, caplog):
        """Test transcribe interval above maximum is clamped to 30 seconds."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_INTERVAL": "60000"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_TRANSCRIBE_INTERVAL_MS == 30000
            assert "ARAGORA_VOICE_INTERVAL=60000 out of bounds [100, 30000]" in caplog.text

    def test_voice_max_sessions_per_ip_below_minimum(self, caplog):
        """Test max sessions per IP below minimum is clamped to 1."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_MAX_SESSIONS_IP": "0"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_SESSIONS_PER_IP == 1
            assert "ARAGORA_VOICE_MAX_SESSIONS_IP=0 out of bounds [1, 50]" in caplog.text

    def test_voice_max_sessions_per_ip_above_maximum(self, caplog):
        """Test max sessions per IP above maximum is clamped to 50."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_MAX_SESSIONS_IP": "100"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_SESSIONS_PER_IP == 50
            assert "ARAGORA_VOICE_MAX_SESSIONS_IP=100 out of bounds [1, 50]" in caplog.text

    def test_voice_max_bytes_per_minute_below_minimum(self, caplog):
        """Test max bytes per minute below minimum is clamped to 100KB."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_RATE_BYTES": "50000"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_BYTES_PER_MINUTE == 102400
            assert "ARAGORA_VOICE_RATE_BYTES=50000 out of bounds [102400, 52428800]" in caplog.text

    def test_voice_max_bytes_per_minute_above_maximum(self, caplog):
        """Test max bytes per minute above maximum is clamped to 50MB."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_RATE_BYTES": "100000000"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_BYTES_PER_MINUTE == 52428800
            assert (
                "ARAGORA_VOICE_RATE_BYTES=100000000 out of bounds [102400, 52428800]" in caplog.text
            )

    def test_voice_all_valid_values(self, caplog):
        """Test all valid voice parameters pass without warnings."""
        env_vars = {
            "ARAGORA_VOICE_CHUNK_SIZE": "65536",  # 64KB
            "ARAGORA_VOICE_MAX_SESSION": "600",  # 10 minutes
            "ARAGORA_VOICE_MAX_BUFFER": "10485760",  # 10MB
            "ARAGORA_VOICE_INTERVAL": "5000",  # 5 seconds
            "ARAGORA_VOICE_MAX_SESSIONS_IP": "10",
            "ARAGORA_VOICE_RATE_BYTES": "10485760",  # 10MB/min
        }
        with mock.patch.dict(os.environ, env_vars, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_CHUNK_SIZE_BYTES == 65536
            assert voice_module.VOICE_MAX_SESSION_SECONDS == 600
            assert voice_module.VOICE_MAX_BUFFER_BYTES == 10485760
            assert voice_module.VOICE_TRANSCRIBE_INTERVAL_MS == 5000
            assert voice_module.VOICE_MAX_SESSIONS_PER_IP == 10
            assert voice_module.VOICE_MAX_BYTES_PER_MINUTE == 10485760

            # No warnings should be logged for valid values
            assert "out of bounds" not in caplog.text


class TestClampWithWarning:
    """Tests for the _clamp_with_warning helper function."""

    def test_clamp_at_minimum_boundary(self, caplog):
        """Test clamping at exactly minimum boundary does not warn."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_INTERVAL": "100"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_TRANSCRIBE_INTERVAL_MS == 100
            assert "ARAGORA_VOICE_INTERVAL" not in caplog.text

    def test_clamp_at_maximum_boundary(self, caplog):
        """Test clamping at exactly maximum boundary does not warn."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_INTERVAL": "30000"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_TRANSCRIBE_INTERVAL_MS == 30000
            assert "ARAGORA_VOICE_INTERVAL" not in caplog.text

    def test_clamp_negative_value(self, caplog):
        """Test clamping negative values."""
        with mock.patch.dict(os.environ, {"ARAGORA_VOICE_MAX_SESSIONS_IP": "-10"}, clear=False):
            import aragora.server.stream.voice_stream as voice_module

            with caplog.at_level(logging.WARNING):
                importlib.reload(voice_module)

            assert voice_module.VOICE_MAX_SESSIONS_PER_IP == 1
            assert "ARAGORA_VOICE_MAX_SESSIONS_IP=-10 out of bounds [1, 50]" in caplog.text
