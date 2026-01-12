#!/usr/bin/env python3
"""
Generate Aragora Broadcast podcast from debate trace.

Usage:
    python scripts/generate_broadcast.py --trace_id <id> --db aragora_traces.db --output debate.mp3
    python scripts/generate_broadcast.py --trace_file path/to/trace.json --output debate.mp3
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.debate.traces import DebateTrace, DebateReplayer
from aragora.broadcast.script_gen import generate_script
from aragora.broadcast.audio_engine import generate_audio
from aragora.broadcast.mixer import mix_audio, mix_audio_with_ffmpeg


async def main():
    parser = argparse.ArgumentParser(description="Generate Aragora Broadcast podcast")
    parser.add_argument("--trace_id", help="Trace ID to load from database")
    parser.add_argument("--trace_file", help="Path to trace JSON file")
    parser.add_argument("--db", default="aragora_traces.db", help="Trace database path")
    parser.add_argument("--output", required=True, help="Output MP3 file path")
    parser.add_argument("--temp_dir", help="Temporary directory for audio segments")
    parser.add_argument(
        "--tts-backend",
        help="Force TTS backend (elevenlabs, polly, xtts, edge-tts, pyttsx3)",
    )
    parser.add_argument(
        "--tts-order",
        help="Comma-separated backend priority order",
    )

    args = parser.parse_args()

    if not args.trace_id and not args.trace_file:
        print("Error: Must specify --trace_id or --trace_file")
        return 1

    if args.tts_backend:
        os.environ["ARAGORA_TTS_BACKEND"] = args.tts_backend
        os.environ.setdefault("TTS_BACKEND", args.tts_backend)
    if args.tts_order:
        os.environ["ARAGORA_TTS_ORDER"] = args.tts_order

    # Load trace
    if args.trace_file:
        trace_path = Path(args.trace_file)
        if not trace_path.exists():
            print(f"Trace file not found: {trace_path}")
            return 1
        trace = DebateTrace.load(trace_path)
    else:
        try:
            replayer = DebateReplayer.from_database(args.trace_id, db_path=args.db)
        except ValueError:
            if args.trace_id.startswith("trace-"):
                print(f"Trace not found: {args.trace_id}")
                return 1
            try:
                replayer = DebateReplayer.from_database(f"trace-{args.trace_id}", db_path=args.db)
            except ValueError:
                print(f"Trace not found: {args.trace_id}")
                return 1
        trace = replayer.trace

    print(f"Loaded debate trace: {trace.task[:100]}...")
    print(f"Found {len(trace.events)} events")

    # Generate script
    print("Generating script...")
    segments = generate_script(trace)
    print(f"Generated {len(segments)} script segments")

    # Generate audio
    print("Generating audio segments...")
    temp_dir = Path(args.temp_dir) if args.temp_dir else None
    audio_files = await generate_audio(segments, temp_dir)
    print(f"Generated {len(audio_files)} audio files")

    if not audio_files:
        print("No audio files generated")
        return 1

    # Mix audio
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Mixing audio to {output_path}...")
    success = mix_audio(audio_files, output_path)

    if not success:
        print("Primary mixing failed, trying ffmpeg fallback...")
        success = mix_audio_with_ffmpeg(audio_files, output_path)

    if success:
        print(f"Successfully created broadcast: {output_path}")
        print(f"File size: {output_path.stat().st_size} bytes")
    else:
        print("Failed to mix audio")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
