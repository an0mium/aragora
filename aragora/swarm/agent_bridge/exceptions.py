from __future__ import annotations


class AgentBridgeError(RuntimeError):
    """Base error for agent bridge failures."""


class TransportError(AgentBridgeError):
    """Base error for harness transport failures."""


class TransportNotAvailableError(TransportError):
    """Raised when a harness binary is unavailable."""


class TransportLaunchError(TransportError):
    """Raised when a launch command exits unsuccessfully."""


class TransportResumeError(TransportError):
    """Raised when a resume command exits unsuccessfully."""


class TransportOutputParseError(TransportError):
    """Raised when harness output cannot be parsed."""


class FooterValidationError(AgentBridgeError):
    """Raised when a footer block is present but invalid."""


class MissingSessionIdentityError(TransportOutputParseError):
    """Raised when a harness result omits the required session identity."""
