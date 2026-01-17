"""
Export format operations handler mixin.

Extracted from handler.py for modularity. Provides export formatting methods
for debates in various formats (CSV, HTML, TXT, MD, LaTeX).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

from ..base import (
    HandlerResult,
    error_response,
    json_response,
    require_storage,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    """Protocol defining the interface expected by ExportOperationsMixin.

    This protocol enables proper type checking for mixin classes that
    expect to be mixed into a class providing these methods/attributes.
    """

    ctx: Dict[str, Any]

    def get_storage(self) -> Optional[Any]:
        """Get debate storage instance."""
        ...


class ExportOperationsMixin:
    """Mixin providing export formatting operations for DebatesHandler."""

    @require_storage
    def _export_debate(
        self: _DebatesHandlerProtocol,
        handler: Any,
        debate_id: str,
        format: str,
        table: str,
    ) -> HandlerResult:
        """Export debate in specified format."""
        from aragora.exceptions import (
            DatabaseError,
            RecordNotFoundError,
            StorageError,
        )

        valid_formats = {"json", "csv", "html", "txt", "md"}
        if format not in valid_formats:
            return error_response(f"Invalid format: {format}. Valid: {valid_formats}", 400)

        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            if format == "json":
                return json_response(debate)
            elif format == "csv":
                return _format_csv(debate, table)
            elif format == "txt":
                return _format_txt(debate)
            elif format == "md":
                return _format_md(debate)
            elif format in ("latex", "tex"):
                return _format_latex(debate)
            else:  # format == "html"
                return _format_html(debate)

        except RecordNotFoundError:
            logger.info("Export failed - debate not found: %s", debate_id)
            return error_response(f"Debate not found: {debate_id}", 404)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Export failed for %s (format=%s): %s: %s",
                debate_id,
                format,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return error_response("Database error during export", 500)
        except ValueError as e:
            logger.warning("Export failed for %s - invalid format: %s", debate_id, e)
            return error_response(f"Invalid export format: {e}", 400)


def _format_csv(debate: dict, table: str) -> HandlerResult:
    """Format debate as CSV for the specified table type."""
    from aragora.server.debate_export import format_debate_csv

    result = format_debate_csv(debate, table)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


def _format_html(debate: dict) -> HandlerResult:
    """Format debate as standalone HTML page."""
    from aragora.server.debate_export import format_debate_html

    result = format_debate_html(debate)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


def _format_txt(debate: dict) -> HandlerResult:
    """Format debate as plain text transcript."""
    from aragora.server.debate_export import format_debate_txt

    result = format_debate_txt(debate)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


def _format_md(debate: dict) -> HandlerResult:
    """Format debate as Markdown transcript."""
    from aragora.server.debate_export import format_debate_md

    result = format_debate_md(debate)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


def _format_latex(debate: dict) -> HandlerResult:
    """Format debate as LaTeX document."""
    from aragora.server.debate_export import format_debate_latex

    result = format_debate_latex(debate)
    return HandlerResult(
        status_code=200,
        content_type=result.content_type,
        body=result.content,
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


__all__ = ["ExportOperationsMixin"]
