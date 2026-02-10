# Broadcast Module - Debate Podcast Generation

The broadcast module converts decision stress-test debates into audio podcasts, complete with text-to-speech synthesis, audio mixing, and podcast distribution via RSS feeds.

## Quick Start

### Generate Audio from a Debate

```bash
# Generate podcast audio via API
curl -X POST https://api.aragora.ai/api/debates/{debate_id}/broadcast

# Response:
{
  "debate_id": "abc123",
  "status": "generated",
  "audio_url": "/audio/abc123.mp3",
  "duration_seconds": 180
}
```

### Listen to Generated Audio

```bash
# Stream or download the audio
curl https://api.aragora.ai/audio/abc123.mp3 -o debate.mp3
```

### Subscribe to Podcast Feed

```bash
# Get iTunes-compatible RSS feed
curl https://api.aragora.ai/api/podcast/feed.xml

# Get JSON episode listing
curl https://api.aragora.ai/api/podcast/episodes
```

## Programmatic Usage

```python
from aragora.broadcast import broadcast_debate
from aragora.debate.traces import DebateTrace
from pathlib import Path

# Load a debate trace
trace = DebateTrace.load(Path("debate_trace.json"))

# Generate audio (returns path to MP3 file)
audio_path = await broadcast_debate(trace)
print(f"Generated: {audio_path}")
```

## Architecture

```
DebateTrace (events)
    │
    ▼ (script_gen.py)
ScriptSegment[] (narrator + agent turns)
    │
    ▼ (audio_engine.py)
MP3 files (one per segment)
    │
    ▼ (mixer.py)
Single MP3 (concatenated)
    │
    ▼ (storage.py)
.nomic/audio/{debate_id}.mp3 + metadata.json
    │
    ▼ (rss_gen.py)
iTunes RSS feed / JSON episode list
```

## Components

### Script Generation (`script_gen.py`)

Converts debate events into a podcast script with narrator transitions.

```python
from aragora.broadcast.script_gen import generate_script, ScriptSegment

segments = generate_script(trace)
# Returns list of ScriptSegment(speaker, text, voice_id)
```

**Features:**
- Extracts MESSAGE events from debate trace
- Adds narrator transitions: "Now, {agent} responds..."
- Summarizes long code blocks: "Reading code block of N lines..."
- Wraps content with intro/outro narration

### Audio Engine (`audio_engine.py`)

Text-to-speech synthesis using configurable backends (ElevenLabs, Amazon Polly, Coqui XTTS v2, edge-tts) with fallbacks.

```python
from aragora.broadcast.audio_engine import AudioEngine

engine = AudioEngine()
audio_path = await engine.generate_audio("Hello world", "narrator")
```

**Features:**
- Primary: `elevenlabs` (highest quality, diverse voices)
- Secondary: `polly` (AWS neural voices, SSML + lexicons)
- Secondary: `xtts` (Coqui XTTS v2, local, GPU recommended)
- Fallback: `edge-tts` (Microsoft neural voices)
- Final fallback: `pyttsx3` (offline, lower quality)
- Voice mapping per agent (edge-tts) with per-backend overrides via env
- Retry with exponential backoff (edge-tts)
- 60-second timeout per segment
- VTT subtitle generation (edge-tts)

**Voice Mapping:**
| Agent | Voice |
|-------|-------|
| narrator | en-US-AriaNeural |
| claude-visionary | en-GB-SoniaNeural |
| codex-engineer | en-US-GuyNeural |
| gemini-visionary | en-AU-NatashaNeural |
| grok-lateral-thinker | en-US-ChristopherNeural |

### TTS Provider Configuration

Install optional providers:
```bash
pip install "aragora[broadcast-elevenlabs]"  # ElevenLabs (cloud)
pip install "aragora[broadcast-polly]"       # Amazon Polly (cloud, AWS)
pip install "aragora[broadcast-xtts]"        # Coqui XTTS v2 (local)
```

Backend selection and ordering:
```bash
export ARAGORA_TTS_ORDER=elevenlabs,polly,xtts,edge-tts,pyttsx3
export ARAGORA_TTS_BACKEND=elevenlabs  # force a backend
```

CLI overrides (one-off runs):
```bash
python scripts/generate_broadcast.py --trace_id trace-123 \
  --output debate.mp3 \
  --tts-backend elevenlabs \
  --tts-order elevenlabs,xtts,edge-tts
```

ElevenLabs (cloud, best quality):
```bash
export ARAGORA_ELEVENLABS_API_KEY=...
export ARAGORA_ELEVENLABS_MODEL_ID=eleven_multilingual_v2
export ARAGORA_ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
# Optional per-agent map (JSON)
export ARAGORA_ELEVENLABS_VOICE_MAP='{"narrator":"21m00Tcm4TlvDq8ikWAM","claude-visionary":"pNInz6obpgDQGcFmaJgB"}'
```

Coqui XTTS v2 (local, GPU recommended):
```bash
export ARAGORA_XTTS_DEVICE=auto   # auto, cuda, cpu
export ARAGORA_XTTS_LANGUAGE=en
export ARAGORA_XTTS_MODEL_PATH=tts_models/multilingual/multi-dataset/xtts_v2
# Optional per-agent speaker WAVs (JSON)
export ARAGORA_XTTS_SPEAKER_WAV_MAP='{"narrator":"/path/to/narrator.wav"}'
```

Amazon Polly (cloud, AWS):
```bash
export AWS_REGION=us-east-1
export ARAGORA_POLLY_ENGINE=neural
export ARAGORA_POLLY_TEXT_TYPE=text  # or ssml
export ARAGORA_POLLY_VOICE_ID=Joanna
# Optional per-agent map (JSON)
export ARAGORA_POLLY_VOICE_MAP='{"narrator":"Joanna","claude-visionary":"Matthew"}'
# Optional lexicons (comma-separated)
export ARAGORA_POLLY_LEXICONS=product-terms,brand-names
```

### Audio Mixer (`mixer.py`)

Concatenates audio segments into a single podcast file.

```python
from aragora.broadcast.mixer import AudioMixer

mixer = AudioMixer()
output_path = await mixer.mix_segments(segment_paths, output_path)
```

**Features:**
- Primary: `pydub` library (pure Python)
- Fallback: `ffmpeg` CLI (concat demuxer)
- 5-minute timeout for encoding
- Automatic temporary file cleanup

### Audio Storage (`storage.py`)

Persistent storage for generated audio files.

```python
from aragora.broadcast.storage import AudioFileStore
from pathlib import Path

store = AudioFileStore(Path(".nomic/audio"))

# Save audio
stored_path = store.save(
    debate_id="abc123",
    audio_path=Path("/tmp/debate.mp3"),
    duration_seconds=180
)

# Retrieve
audio_path = store.get_path("abc123")
metadata = store.get_metadata("abc123")

# List all
all_audio = store.list_all()
```

**Features:**
- Storage location: `.nomic/audio/`
- JSON metadata sidecars with duration, size, timestamp
- Path traversal protection
- Audio format whitelist: mp3, wav, m4a, ogg, flac, aac
- Magic byte validation
- 100 MB file size limit
- In-memory caching
- Orphaned file cleanup

### RSS Feed Generation (`rss_gen.py`)

iTunes-compatible podcast feed generation.

```python
from aragora.broadcast.rss_gen import PodcastFeedGenerator, PodcastConfig, PodcastEpisode

config = PodcastConfig(
    title="Aragora Debates",
    description="Multi-agent AI debates",
    author="Aragora",
    email="podcast@aragora.ai",
    category="Technology",
)

generator = PodcastFeedGenerator(config)
episodes = [
    PodcastEpisode(
        guid="abc123",
        title="Debate: Rate Limiting Strategy",
        description="AI agents discuss...",
        audio_url="https://api.aragora.ai/audio/abc123.mp3",
        duration_seconds=180,
    )
]

feed_xml = generator.generate_feed(episodes)
```

**Features:**
- iTunes-compatible RSS 2.0 format
- Episode numbering and seasons
- Duration formatting (HH:MM:SS)
- CDATA sections for HTML content
- XML-escaped content

### Video Generation (`video_gen.py`)

Convert audio to video for YouTube uploads.

```python
from aragora.broadcast.video_gen import create_video

video_path = await create_video(
    audio_path=Path("debate.mp3"),
    output_path=Path("debate.mp4"),
    thumbnail_path=Path("thumbnail.png"),  # Optional
    waveform=False,  # Static image or animated waveform
)
```

**Features:**
- Static videos: thumbnail + audio
- Waveform videos: animated audio visualization
- Resolution: 320x240 to 3840x2160
- Bitrate: 64-320 kbps
- FFmpeg encoding with 10-minute timeout
- ImageMagick support for better graphics

### Social Media (`social.py` via `rss_gen.py`)

Generate social media summaries.

```python
from aragora.broadcast.rss_gen import create_debate_summary

summary = create_debate_summary(
    task="Rate Limiting Strategy",
    verdict="Implement token bucket algorithm",
    agents=["claude-visionary", "gemini-visionary"]
)
# Returns: "AI Debate: Rate Limiting Strategy..."
```

**Features:**
- Twitter-optimized (280 char limit)
- Emoji indicators
- Hashtag generation

## API Endpoints

### POST `/api/debates/{id}/broadcast`

Generate podcast audio from a debate.

**Rate Limited:** 3 requests/minute (TTS is CPU-intensive)

**Request:**
```bash
curl -X POST https://api.aragora.ai/api/debates/abc123/broadcast
```

**Response:**
```json
{
  "debate_id": "abc123",
  "status": "generated",
  "audio_url": "/audio/abc123.mp3",
  "audio_path": "/path/to/.nomic/audio/abc123.mp3",
  "duration_seconds": 180
}
```

**Status Values:**
- `generated`: New audio created
- `exists`: Audio already existed (cached)

### GET `/audio/{id}.mp3`

Serve audio file with caching headers.

**Headers:**
- `Content-Type: audio/mpeg`
- `Cache-Control: public, max-age=86400`
- `Accept-Ranges: bytes`

### GET `/api/podcast/feed.xml`

iTunes-compatible RSS feed.

**Response:** XML RSS 2.0 feed

**Headers:**
- `Content-Type: application/rss+xml; charset=utf-8`
- `Cache-Control: public, max-age=300`

### GET `/api/podcast/episodes`

JSON listing of podcast episodes.

**Query Parameters:**
- `limit`: Max episodes to return (default: 50)

**Response:**
```json
{
  "episodes": [
    {
      "debate_id": "abc123",
      "task": "Rate Limiting Strategy",
      "agents": ["claude-visionary", "gemini-visionary"],
      "audio_url": "https://api.aragora.ai/audio/abc123.mp3",
      "duration_seconds": 180,
      "file_size_bytes": 2880000,
      "generated_at": "2026-01-09T10:00:00Z"
    }
  ],
  "count": 1,
  "feed_url": "/api/podcast/feed.xml"
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_AUDIO_DIR` | Audio storage directory | `.nomic/audio` |
| `EDGE_TTS_TIMEOUT` | TTS generation timeout | `60` |

### Dependencies

**Required (for full functionality):**
```bash
pip install aragora[broadcast]
# Installs: edge-tts, pydub, pyttsx3
```

**Optional:**
- `ffmpeg` - For audio mixing fallback and video generation
- `mutagen` - For audio metadata extraction
- `imagemagick` - For better video thumbnails

## Testing

```bash
# Run broadcast tests
pytest tests/test_broadcast*.py -v

# Test modules:
# - test_broadcast_script.py - Script generation
# - test_broadcast_audio.py - TTS generation
# - test_broadcast_mixer.py - Audio mixing
# - test_broadcast_storage.py - File storage
# - test_broadcast_rss.py - RSS feed
# - test_broadcast_video.py - Video generation
# - test_handlers_broadcast.py - API endpoints
# - test_handlers_audio.py - Audio serving
```

## Narrator Flow Example

A debate about "API Rate Limiting" would produce:

1. **Intro:** "Welcome to Aragora Broadcast. Today's debate is about: API Rate Limiting..."
2. **Round 1:** "Now, claude-visionary responds." → [Claude's proposal]
3. **Round 1:** "Now, gemini-visionary responds." → [Gemini's critique]
4. **Code:** "Reading code block of 15 lines..." (for long code snippets)
5. **...more rounds...**
6. **Outro:** "That concludes this Aragora debate. The consensus was: Implement token bucket. Thank you for listening."

## Security

- **Path Traversal Protection:** Debate IDs validated before file access
- **Audio Format Whitelist:** Only allowed formats accepted
- **Magic Byte Validation:** File type verified by content, not extension
- **File Size Limits:** 100 MB maximum
- **Rate Limiting:** 3 requests/minute for generation
