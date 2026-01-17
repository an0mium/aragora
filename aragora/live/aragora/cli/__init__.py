"""
Aragora CLI - Command-line interface for document management and auditing.

Usage:
    python -m aragora documents upload ./files/*.pdf
    python -m aragora documents list
    python -m aragora audit create --documents doc1,doc2
    python -m aragora audit start session-123
    python -m aragora audit status session-123
"""

from aragora.cli.main import main
from aragora.cli.documents import documents_cli
from aragora.cli.audit import audit_cli

__all__ = ["main", "documents_cli", "audit_cli"]
