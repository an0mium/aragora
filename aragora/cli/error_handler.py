"""
CLI Error Handler with Recovery Suggestions

Provides structured error handling for the Aragora CLI with actionable
recovery suggestions to help users fix common issues.

Usage:
    from aragora.cli.error_handler import CLIErrorHandler, handle_cli_error

    try:
        # CLI operation
    except Exception as e:
        handle_cli_error(e)
"""

import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    """Categories of CLI errors."""

    API_KEY = "api_key"
    NETWORK = "network"
    SERVER = "server"
    AGENT = "agent"
    CONFIG = "config"
    FILE = "file"
    VALIDATION = "validation"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


@dataclass
class RecoverySuggestion:
    """A suggestion for recovering from an error."""

    title: str
    steps: list[str]
    command: Optional[str] = None  # Optional command to run


@dataclass
class CLIError:
    """Structured CLI error with context and recovery suggestions."""

    message: str
    category: ErrorCategory
    suggestions: list[RecoverySuggestion] = field(default_factory=list)
    details: Optional[str] = None
    original_error: Optional[Exception] = None

    def format(self, verbose: bool = False) -> str:
        """Format the error for display."""
        lines = []

        # Header
        lines.append(f"\n[ERROR] {self.message}")

        # Details
        if self.details:
            lines.append(f"\nDetails: {self.details}")

        # Suggestions
        if self.suggestions:
            lines.append("\nTo fix this:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"\n  {i}. {suggestion.title}")
                for step in suggestion.steps:
                    lines.append(f"     - {step}")
                if suggestion.command:
                    lines.append(f"     $ {suggestion.command}")

        # Verbose traceback
        if verbose and self.original_error:
            lines.append("\n--- Traceback ---")
            lines.append(
                "".join(
                    traceback.format_exception(
                        type(self.original_error),
                        self.original_error,
                        self.original_error.__traceback__,
                    )
                )
            )

        return "\n".join(lines)


class CLIErrorHandler:
    """
    Error handler that classifies errors and provides recovery suggestions.

    Common error patterns and their suggestions:
    - Missing API key: How to set environment variables
    - Rate limited: Use fallback providers or wait
    - Server unavailable: Start server or use demo mode
    - Invalid config: Show config location and format
    - File not found: Check path and permissions
    """

    # Error classification patterns
    API_KEY_PATTERNS = [
        "api_key",
        "apikey",
        "api key",
        "missing key",
        "authentication",
        "unauthorized",
        "401",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "sk-ant",
        "sk-",
    ]

    NETWORK_PATTERNS = [
        "connection refused",
        "timeout",
        "network",
        "unreachable",
        "dns",
        "host",
        "could not connect",
        "errno 61",
        "errno 111",
    ]

    RATE_LIMIT_PATTERNS = [
        "rate limit",
        "429",
        "too many requests",
        "quota",
        "exceeded",
        "throttl",
    ]

    SERVER_PATTERNS = [
        "500",
        "502",
        "503",
        "504",
        "internal server error",
        "server error",
        "service unavailable",
    ]

    PERMISSION_PATTERNS = [
        "permission denied",
        "access denied",
        "forbidden",
        "errno 13",
        "eacces",
    ]

    @classmethod
    def classify_error(cls, error: Exception) -> ErrorCategory:
        """Classify an error into a category."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check patterns
        for pattern in cls.API_KEY_PATTERNS:
            if pattern in error_str:
                return ErrorCategory.API_KEY

        for pattern in cls.NETWORK_PATTERNS:
            if pattern in error_str:
                return ErrorCategory.NETWORK

        for pattern in cls.RATE_LIMIT_PATTERNS:
            if pattern in error_str:
                return ErrorCategory.NETWORK  # Rate limits are network-related

        for pattern in cls.SERVER_PATTERNS:
            if pattern in error_str:
                return ErrorCategory.SERVER

        for pattern in cls.PERMISSION_PATTERNS:
            if pattern in error_str:
                return ErrorCategory.PERMISSION

        # Check exception types
        if "config" in error_type or "config" in error_str:
            return ErrorCategory.CONFIG

        if "filenotfound" in error_type or "no such file" in error_str:
            return ErrorCategory.FILE

        if "validation" in error_type or "invalid" in error_str:
            return ErrorCategory.VALIDATION

        if error_type in ("agenterror", "agentnotfounderror", "agenttimeouterror"):
            return ErrorCategory.AGENT

        return ErrorCategory.UNKNOWN

    @classmethod
    def get_suggestions(cls, category: ErrorCategory, error: Exception) -> list[RecoverySuggestion]:
        """Get recovery suggestions for an error category."""
        suggestions = []
        error_str = str(error).lower()

        if category == ErrorCategory.API_KEY:
            suggestions.append(
                RecoverySuggestion(
                    title="Set an API key in your environment",
                    steps=[
                        "Choose at least one AI provider:",
                        "  Anthropic (Claude): export ANTHROPIC_API_KEY='sk-ant-...'",
                        "  OpenAI (GPT): export OPENAI_API_KEY='sk-...'",
                        "  Mistral: export MISTRAL_API_KEY='...'",
                    ],
                    command="aragora doctor",
                )
            )
            suggestions.append(
                RecoverySuggestion(
                    title="Use a .env file",
                    steps=[
                        "Copy the starter config: cp .env.starter .env",
                        "Edit .env and add your API key",
                    ],
                )
            )
            suggestions.append(
                RecoverySuggestion(
                    title="Run diagnostics",
                    steps=["Check your configuration with the doctor command"],
                    command="aragora doctor",
                )
            )

        elif category == ErrorCategory.NETWORK:
            if any(p in error_str for p in cls.RATE_LIMIT_PATTERNS):
                suggestions.append(
                    RecoverySuggestion(
                        title="Provider rate limit exceeded",
                        steps=[
                            "Add a fallback provider (OpenRouter handles rate limits automatically):",
                            "  export OPENROUTER_API_KEY='sk-or-...'",
                            "Or wait 60 seconds and retry",
                        ],
                    )
                )
                suggestions.append(
                    RecoverySuggestion(
                        title="Use different agents",
                        steps=["Try agents from a different provider"],
                        command="aragora ask 'your question' --agents gemini,mistral-large",
                    )
                )
            else:
                suggestions.append(
                    RecoverySuggestion(
                        title="Check your network connection",
                        steps=[
                            "Verify internet connectivity",
                            "Check if the API endpoint is accessible",
                            "Try disabling VPN/proxy if enabled",
                        ],
                    )
                )

        elif category == ErrorCategory.SERVER:
            suggestions.append(
                RecoverySuggestion(
                    title="Aragora server not running",
                    steps=["Start the server with:"],
                    command="aragora serve",
                )
            )
            suggestions.append(
                RecoverySuggestion(
                    title="Run in demo mode (no server required)",
                    steps=["Add --demo flag to run locally without server"],
                    command="aragora ask 'your question' --demo",
                )
            )

        elif category == ErrorCategory.AGENT:
            suggestions.append(
                RecoverySuggestion(
                    title="Check available agents",
                    steps=["View all available agents and their status"],
                    command="aragora agents",
                )
            )
            suggestions.append(
                RecoverySuggestion(
                    title="Use different agents",
                    steps=["Try a different combination of agents"],
                    command="aragora ask 'your question' --agents anthropic-api,openai-api",
                )
            )

        elif category == ErrorCategory.CONFIG:
            suggestions.append(
                RecoverySuggestion(
                    title="Show current configuration",
                    steps=["View your config settings"],
                    command="aragora config show",
                )
            )
            suggestions.append(
                RecoverySuggestion(
                    title="Reset to defaults",
                    steps=["Remove custom config to use defaults"],
                    command="aragora config reset",
                )
            )

        elif category == ErrorCategory.FILE:
            suggestions.append(
                RecoverySuggestion(
                    title="Check file path",
                    steps=[
                        "Verify the file exists at the specified path",
                        "Use an absolute path if relative path doesn't work",
                    ],
                )
            )

        elif category == ErrorCategory.PERMISSION:
            suggestions.append(
                RecoverySuggestion(
                    title="Check file permissions",
                    steps=[
                        "Ensure you have read/write access to the directory",
                        "Try running with appropriate permissions",
                    ],
                )
            )

        else:  # UNKNOWN
            suggestions.append(
                RecoverySuggestion(
                    title="Run diagnostics",
                    steps=["Check your setup for common issues"],
                    command="aragora doctor",
                )
            )
            suggestions.append(
                RecoverySuggestion(
                    title="Get help",
                    steps=[
                        "View documentation: aragora --help",
                        "Report issues: https://github.com/an0mium/aragora/issues",
                    ],
                )
            )

        return suggestions

    @classmethod
    def create_error(cls, exception: Exception) -> CLIError:
        """Create a structured CLI error from an exception."""
        category = cls.classify_error(exception)
        suggestions = cls.get_suggestions(category, exception)

        # Create user-friendly message
        error_msg = str(exception)
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."

        return CLIError(
            message=error_msg,
            category=category,
            suggestions=suggestions,
            original_error=exception,
        )


def handle_cli_error(
    error: Exception,
    verbose: bool = False,
    exit_code: int = 1,
    exit_on_error: bool = True,
) -> CLIError:
    """
    Handle a CLI error with user-friendly output.

    Args:
        error: The exception to handle
        verbose: Show full traceback
        exit_code: Exit code to use (default 1)
        exit_on_error: Whether to exit after handling (default True)

    Returns:
        The structured CLIError (if exit_on_error is False)
    """
    cli_error = CLIErrorHandler.create_error(error)
    print(cli_error.format(verbose=verbose), file=sys.stderr)

    if exit_on_error:
        sys.exit(exit_code)

    return cli_error


def cli_error_handler(verbose: bool = False):
    """
    Decorator for CLI commands to handle errors gracefully.

    Usage:
        @cli_error_handler()
        def my_command(args):
            # command implementation
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print("\n[Interrupted]", file=sys.stderr)
                sys.exit(130)
            except SystemExit:
                raise
            except Exception as e:
                handle_cli_error(e, verbose=verbose)

        return wrapper

    return decorator


# Common error shortcuts
def api_key_error(provider: str = "AI") -> CLIError:
    """Create an API key error."""
    return CLIError(
        message=f"No {provider} API key configured",
        category=ErrorCategory.API_KEY,
        suggestions=CLIErrorHandler.get_suggestions(ErrorCategory.API_KEY, Exception()),
    )


def server_unavailable_error(url: str = "http://localhost:8080") -> CLIError:
    """Create a server unavailable error."""
    return CLIError(
        message=f"Aragora server not available at {url}",
        category=ErrorCategory.SERVER,
        suggestions=CLIErrorHandler.get_suggestions(ErrorCategory.SERVER, Exception()),
    )


def rate_limit_error(provider: str) -> CLIError:
    """Create a rate limit error."""
    error = Exception(f"Rate limit exceeded for {provider}")
    return CLIError(
        message=f"Rate limit exceeded for {provider}",
        category=ErrorCategory.NETWORK,
        suggestions=CLIErrorHandler.get_suggestions(ErrorCategory.NETWORK, error),
    )
