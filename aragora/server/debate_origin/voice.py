"""Voice synthesis for debate origin result routing.

Provides TTS synthesis for sending voice summaries of debate results
to chat platforms that support audio messages.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DebateOrigin

logger = logging.getLogger(__name__)


async def _synthesize_voice(result: dict[str, Any], origin: DebateOrigin) -> str | None:
    """Synthesize voice message from debate result using TTS.

    Returns path to audio file or None if TTS fails.
    """
    try:
        from aragora.connectors.chat.tts_bridge import get_tts_bridge

        bridge = get_tts_bridge()

        # Create concise voice summary
        consensus = "reached" if result.get("consensus_reached", False) else "not reached"
        confidence = result.get("confidence", 0)
        answer = result.get("final_answer", "No conclusion available.")

        # Truncate for voice (keep it brief)
        if len(answer) > 300:
            answer = answer[:300] + ". See full text for details."

        voice_text = (
            f"Debate complete. Consensus was {consensus} "
            f"with {confidence:.0%} confidence. "
            f"Conclusion: {answer}"
        )

        # Synthesize
        audio_path = await bridge.synthesize_response(voice_text, voice="consensus")
        return audio_path

    except ImportError:
        logger.debug("TTS bridge not available")
        return None
    except Exception as e:
        logger.warning(f"TTS synthesis failed: {e}")
        return None
