"""
Broadcast pipeline orchestrator.

Orchestrates the full debate-to-publication flow:
1. Audio generation (TTS via edge-tts)
2. Video generation (FFmpeg)
3. RSS feed creation (iTunes-compatible)
4. Metadata persistence

Usage:
    from aragora.broadcast.pipeline import BroadcastPipeline, BroadcastOptions

    pipeline = BroadcastPipeline(nomic_dir=Path(".nomic"))
    result = await pipeline.run("debate-123", BroadcastOptions(video_enabled=True))
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.debate.traces import DebateTrace

logger = logging.getLogger(__name__)


@dataclass
class BroadcastOptions:
    """Options for the broadcast pipeline."""

    # Audio generation
    audio_enabled: bool = True
    audio_format: str = "mp3"

    # Video generation
    video_enabled: bool = False
    video_resolution: tuple[int, int] = (1920, 1080)
    thumbnail_path: Optional[str] = None

    # RSS/Podcast
    generate_rss_episode: bool = True

    # Metadata
    custom_title: Optional[str] = None
    custom_description: Optional[str] = None
    episode_number: Optional[int] = None
    season_number: Optional[int] = None


@dataclass
class PipelineResult:
    """Result of the broadcast pipeline."""

    debate_id: str
    success: bool
    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    rss_episode_guid: Optional[str] = None
    duration_seconds: Optional[int] = None
    error_message: Optional[str] = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    steps_completed: list[str] = field(default_factory=list)


class BroadcastPipeline:
    """
    Orchestrates the full broadcast pipeline.

    Ties together audio generation, video conversion, and RSS feed creation
    into a single, configurable workflow.
    """

    def __init__(
        self,
        nomic_dir: Path,
        audio_store=None,
        rss_generator=None,
    ):
        """
        Initialize the broadcast pipeline.

        Args:
            nomic_dir: Base directory for nomic files
            audio_store: Optional AudioStore for persistence
            rss_generator: Optional PodcastFeedGenerator
        """
        self.nomic_dir = Path(nomic_dir)
        self.audio_store = audio_store
        self.rss_generator = rss_generator
        self.traces_dir = self.nomic_dir / "traces"
        self.audio_dir = self.nomic_dir / "audio"
        self.video_dir = self.nomic_dir / "video"

        # Ensure output directories exist
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _load_trace(self, debate_id: str) -> Optional["DebateTrace"]:
        """Load a debate trace by ID."""
        try:
            from aragora.debate.traces import DebateTrace

            trace_path = self.traces_dir / f"{debate_id}.json"
            if not trace_path.exists():
                logger.warning(f"Trace not found: {trace_path}")
                return None

            return DebateTrace.load(trace_path)
        except Exception as e:
            logger.error(f"Failed to load trace {debate_id}: {e}")
            return None

    async def run(
        self,
        debate_id: str,
        options: Optional[BroadcastOptions] = None,
    ) -> PipelineResult:
        """
        Run the full broadcast pipeline.

        Args:
            debate_id: The debate ID to broadcast
            options: Pipeline options

        Returns:
            PipelineResult with paths and status
        """
        options = options or BroadcastOptions()
        result = PipelineResult(debate_id=debate_id, success=False)

        # Load trace
        trace = self._load_trace(debate_id)
        if not trace:
            result.error_message = "Debate trace not found"
            return result

        # Step 1: Generate audio
        audio_path: Optional[Path] = None
        if options.audio_enabled:
            audio_path = await self._generate_audio(trace, options)
            if audio_path:
                result.audio_path = audio_path
                result.steps_completed.append("audio")
                result.duration_seconds = self._get_audio_duration(audio_path)
                logger.info(f"Audio generated: {audio_path}")
            else:
                result.error_message = "Audio generation failed"
                return result

        # Step 2: Generate video (if enabled)
        if options.video_enabled and audio_path:
            video_path = await self._generate_video(
                audio_path,
                trace,
                options,
            )
            if video_path:
                result.video_path = video_path
                result.steps_completed.append("video")
                logger.info(f"Video generated: {video_path}")
            else:
                logger.warning("Video generation failed, continuing without video")

        # Step 3: Create RSS episode (if enabled)
        if options.generate_rss_episode and audio_path:
            episode_guid = self._create_rss_episode(
                trace,
                audio_path,
                result.duration_seconds or 0,
                options,
            )
            if episode_guid:
                result.rss_episode_guid = episode_guid
                result.steps_completed.append("rss")
                logger.info(f"RSS episode created: {episode_guid}")

        # Step 4: Persist to audio store (if available)
        if self.audio_store and audio_path:
            try:
                stored_path = self.audio_store.save(
                    debate_id=debate_id,
                    audio_path=audio_path,
                    duration_seconds=result.duration_seconds,
                )
                result.audio_path = stored_path
                result.steps_completed.append("storage")
            except Exception as e:
                logger.warning(f"Failed to persist to audio store: {e}")

        result.success = "audio" in result.steps_completed
        return result

    async def _generate_audio(
        self,
        trace: "DebateTrace",
        options: BroadcastOptions,
    ) -> Optional[Path]:
        """Generate audio from debate trace."""
        try:
            from aragora.broadcast import broadcast_debate

            output_path = self.audio_dir / f"{trace.id}.{options.audio_format}"
            result = await broadcast_debate(trace, output_path, options.audio_format)
            return result
        except ImportError:
            logger.error("Broadcast module not available")
            return None
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None

    async def _generate_video(
        self,
        audio_path: Path,
        trace: "DebateTrace",
        options: BroadcastOptions,
    ) -> Optional[Path]:
        """Generate video from audio and trace."""
        try:
            from aragora.broadcast.video_gen import generate_video

            output_path = self.video_dir / f"{trace.id}.mp4"

            title = options.custom_title or trace.task[:100]
            description = options.custom_description or f"Debate: {trace.task}"

            result = await generate_video(
                audio_path=audio_path,
                output_path=output_path,
                title=title,
                description=description,
                resolution=options.video_resolution,
                thumbnail_path=options.thumbnail_path,
            )
            return output_path if result else None
        except ImportError:
            logger.warning("Video generation module not available")
            return None
        except Exception as e:
            logger.warning(f"Video generation failed: {e}")
            return None

    def _create_rss_episode(
        self,
        trace: "DebateTrace",
        audio_path: Path,
        duration_seconds: int,
        options: BroadcastOptions,
    ) -> Optional[str]:
        """Create RSS episode entry."""
        if not self.rss_generator:
            # Try to create a default generator
            try:
                from aragora.broadcast.rss_gen import (
                    PodcastFeedGenerator,
                    PodcastConfig,
                    PodcastEpisode,
                )

                if not self.rss_generator:
                    config = PodcastConfig()
                    self.rss_generator = PodcastFeedGenerator(config)
            except ImportError:
                logger.warning("RSS generator not available")
                return None

        try:
            from aragora.broadcast.rss_gen import PodcastEpisode

            # Extract agents from trace
            agents = list(set(e.agent for e in trace.events if e.agent))

            episode = PodcastEpisode(
                guid=trace.id,
                title=options.custom_title or f"Debate: {trace.task[:80]}",
                description=(options.custom_description or trace.task)[:500],
                content=self._format_show_notes(trace),
                audio_url=f"/audio/{trace.id}.mp3",
                pub_date=datetime.now().isoformat(),
                duration_seconds=duration_seconds,
                file_size_bytes=audio_path.stat().st_size if audio_path.exists() else 0,
                episode_number=options.episode_number,
                season_number=options.season_number,
                agents=agents,
            )

            self.rss_generator.add_episode(episode)
            return episode.guid
        except Exception as e:
            logger.warning(f"Failed to create RSS episode: {e}")
            return None

    def _format_show_notes(self, trace: "DebateTrace") -> str:
        """Format debate trace as show notes."""
        lines = [
            f"# {trace.task}",
            "",
            "## Participants",
        ]

        # List unique agents
        agents = list(set(e.agent for e in trace.events if e.agent))
        for agent in agents:
            lines.append(f"- {agent}")

        lines.extend([
            "",
            "## Summary",
            f"This debate featured {len(agents)} AI agents discussing: {trace.task}",
            "",
            f"Total rounds: {len([e for e in trace.events if e.event_type.value == 'round_start'])}",
        ])

        return "\n".join(lines)

    def _get_audio_duration(self, audio_path: Path) -> Optional[int]:
        """Get audio duration in seconds."""
        try:
            from mutagen.mp3 import MP3

            audio = MP3(audio_path)
            return int(audio.info.length)
        except ImportError:
            logger.debug("mutagen not available for duration extraction")
            return None
        except Exception as e:
            logger.debug(f"Failed to get audio duration: {e}")
            return None

    def get_rss_feed(self) -> Optional[str]:
        """Get the current RSS feed XML."""
        if not self.rss_generator:
            return None
        try:
            return self.rss_generator.generate()
        except Exception as e:
            logger.error(f"Failed to generate RSS feed: {e}")
            return None


async def run_pipeline(
    debate_id: str,
    nomic_dir: Path,
    options: Optional[BroadcastOptions] = None,
) -> PipelineResult:
    """
    Convenience function to run the broadcast pipeline.

    Args:
        debate_id: The debate ID to broadcast
        nomic_dir: Base directory for nomic files
        options: Pipeline options

    Returns:
        PipelineResult with paths and status
    """
    pipeline = BroadcastPipeline(nomic_dir=nomic_dir)
    return await pipeline.run(debate_id, options)
