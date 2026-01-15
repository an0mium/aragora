"""
Minimal Stabilization: Sanitize agent outputs to prevent crashes.
This module provides utilities for cleaning agent responses.
"""

import logging

logger = logging.getLogger(__name__)


class OutputSanitizer:
    """
    Sanitizes agent outputs to remove problematic characters.
    Prevents null byte crashes and other encoding issues.
    """

    @staticmethod
    def sanitize_agent_output(raw_output: str, agent_name: str) -> str:
        """
        MINIMAL STABILIZATION: Remove null bytes and control chars causing crashes.

        Args:
            raw_output: Raw string from agent
            agent_name: Name of the agent for logging

        Returns:
            Sanitized output string
        """
        if not isinstance(raw_output, str):
            logger.warning(
                f"[Stability] Agent {agent_name} returned non-string output: {type(raw_output)}"
            )
            return "(Agent output type error)"

        original_len = len(raw_output)

        # Critical: Remove null bytes (proven failure mode)
        if "\x00" in raw_output:
            count = raw_output.count("\x00")
            logger.warning(f"[Stability] Removed {count} null bytes from {agent_name}")
            raw_output = raw_output.replace("\x00", "")

        # Remove dangerous control characters (NUL through backspace, vertical tab,
        # form feed, escape sequences, and DEL). Preserves tabs (\x09), newlines (\x0a),
        # and carriage returns (\x0d).
        import re

        raw_output = re.sub(r"[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", raw_output)

        # Log if significant content was stripped (helps debug empty output issues)
        stripped_chars = original_len - len(raw_output)
        if stripped_chars > 0:
            logger.debug(
                f"[Stability] {agent_name}: stripped {stripped_chars} control chars "
                f"(original: {original_len} chars)"
            )

        result = raw_output.strip()
        if not result:
            logger.warning(
                f"[Stability] {agent_name}: output empty after sanitization "
                f"(original was {original_len} chars, stripped {stripped_chars})"
            )
            return "(Agent produced empty output)"

        return result

    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """
        Sanitize prompt before sending to CLI agent.

        Removes null bytes and control characters that cause CLI tools
        to fail with 'embedded null byte' errors.

        Args:
            prompt: Raw prompt string

        Returns:
            Sanitized prompt string
        """
        if not prompt:
            return prompt
        if not isinstance(prompt, str):
            return str(prompt)
        # Remove null bytes (proven failure mode for CLI tools)
        if "\x00" in prompt:
            prompt = prompt.replace("\x00", "")
        # Remove other control characters (preserve newlines \n, tabs \t, carriage returns \r)
        import re

        prompt = re.sub(r"[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", prompt)
        return prompt
