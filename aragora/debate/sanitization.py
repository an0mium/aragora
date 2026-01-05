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
        MINIMAL STABILIZATION: Remove null bytes causing crashes.
        
        Args:
            raw_output: Raw string from agent
            agent_name: Name of the agent for logging
            
        Returns:
            Sanitized output string
        """
        if not isinstance(raw_output, str):
            logger.warning(f"[Stability] Agent {agent_name} returned non-string output: {type(raw_output)}")
            return "(Agent output type error)"
        
        # Critical: Remove null bytes (proven failure mode)
        if '\x00' in raw_output:
            count = raw_output.count('\x00')
            logger.warning(f"[Stability] Removed {count} null bytes from {agent_name}")
            raw_output = raw_output.replace('\x00', '')
        
        # Additional sanitization for other common issues
        # Remove other control characters except newlines and tabs
        import re
        raw_output = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', raw_output)
        
        return raw_output.strip() or "(Agent produced empty output)"