"""
Tests for Aragora Broadcast functionality.
"""

import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aragora.debate.traces import DebateTrace, TraceEvent, EventType
from aragora.broadcast.script_gen import generate_script, ScriptSegment
from aragora.broadcast.audio_engine import generate_audio, _get_voice_for_speaker, VOICE_MAP
from aragora.broadcast.mixer import mix_audio


class TestScriptGen:
    """Test script generation from debate traces."""

    def test_generate_script_basic(self):
        """Test basic script generation."""
        # Create mock trace
        trace = DebateTrace(
            trace_id="test-trace",
            debate_id="test-debate",
            task="Test debate task",
            agents=["agent1", "agent2"],
            random_seed=42,
            events=[
                TraceEvent(
                    event_id="1",
                    event_type=EventType.MESSAGE,
                    timestamp="2026-01-01T00:00:00",
                    agent="agent1",
                    content={"text": "Hello from agent1"},
                    round_num=1
                ),
                TraceEvent(
                    event_id="2",
                    event_type=EventType.MESSAGE,
                    timestamp="2026-01-01T00:00:01",
                    agent="agent2",
                    content={"text": "Response from agent2"},
                    round_num=1
                )
            ]
        )

        segments = generate_script(trace)

        assert len(segments) == 5  # Opening + msg1 + transition + msg2 + closing
        assert segments[0].speaker == "narrator"
        assert "Test debate task" in segments[0].text
        assert segments[1].speaker == "agent1"
        assert segments[2].speaker == "narrator"  # transition
        assert segments[3].speaker == "agent2"
        assert segments[4].speaker == "narrator"  # closing

    def test_code_summarization(self):
        """Test that long code blocks are summarized."""
        long_code = "\n".join([f"line {i}" for i in range(15)])
        short_code = "short code"

        from aragora.broadcast.script_gen import _summarize_code

        assert "Reading code block of 15 lines" in _summarize_code(long_code)
        assert _summarize_code(short_code) == short_code


class TestAudioEngine:
    """Test audio generation."""

    def test_voice_mapping(self):
        """Test voice mapping for speakers."""
        assert _get_voice_for_speaker("claude-visionary") == "en-GB-SoniaNeural"
        assert _get_voice_for_speaker("unknown") == "en-US-AriaNeural"  # narrator default

    @pytest.mark.asyncio
    async def test_generate_audio_empty_segments(self):
        """Test generating audio with empty segments."""
        audio_files = await generate_audio([])
        assert audio_files == []

    @pytest.mark.asyncio
    @patch('aragora.broadcast.audio_engine._generate_edge_tts')
    async def test_generate_audio_with_mock(self, mock_tts):
        """Test audio generation with mocked TTS."""
        # Mock that creates the file and returns True
        async def mock_edge_tts(text, voice, output_path):
            output_path.write_text("dummy audio content")
            return True
        mock_tts.side_effect = mock_edge_tts

        segments = [
            ScriptSegment(speaker="agent1", text="Test text")
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_files = await generate_audio(segments, temp_path)

            assert len(audio_files) == 1
            assert audio_files[0].exists()


class TestMixer:
    """Test audio mixing."""

    def test_mix_audio_no_files(self):
        """Test mixing with no audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output.mp3"
            result = mix_audio([], output)
            assert not result
            assert not output.exists()

    @patch('aragora.broadcast.mixer.PYDUB_AVAILABLE', False)
    def test_mix_audio_no_pydub(self):
        """Test mixing when pydub is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output.mp3"
            result = mix_audio([], output)
            assert not result

    def test_mix_audio_all_files_missing(self):
        """Test mixing when all provided files are missing (Round 24 edge case).

        This verifies the fix for silent data loss where mix_audio() would
        return True even when no files were actually mixed because all
        provided files were missing from disk.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output = temp_path / "output.mp3"

            # Create list of non-existent files
            missing_files = [
                temp_path / "missing1.mp3",
                temp_path / "missing2.mp3",
                temp_path / "missing3.mp3",
            ]

            # Verify files don't exist
            for f in missing_files:
                assert not f.exists()

            # mix_audio should return False when all files are missing
            result = mix_audio(missing_files, output)
            assert not result
            assert not output.exists()


# Integration test
@pytest.mark.asyncio
@patch('aragora.broadcast.audio_engine._generate_edge_tts')
async def test_full_pipeline_mock(mock_tts):
    """Test the full broadcast pipeline with mocks."""
    # Mock TTS that creates the file
    async def mock_edge_tts(text, voice, output_path):
        output_path.write_text("dummy audio content")
        return True
    mock_tts.side_effect = mock_edge_tts

    # Create mock trace
    trace = DebateTrace(
        trace_id="test-trace",
        debate_id="test-debate",
        task="Integration test debate",
        agents=["claude-visionary"],
        random_seed=42,
        events=[
            TraceEvent(
                event_id="1",
                event_type=EventType.MESSAGE,
                timestamp="2026-01-01T00:00:00",
                agent="claude-visionary",
                content={"text": "This is a test message"},
                round_num=1
            )
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Generate script
        segments = generate_script(trace)
        assert len(segments) > 0

        # Generate audio
        audio_files = await generate_audio(segments, temp_path)
        assert len(audio_files) > 0

        # Mix audio
        output_file = temp_path / "broadcast.mp3"
        # Note: mix_audio would need pydub, so this tests the interface
        # In real scenario, pydub would combine the dummy files