"""
Document Connectors for cloud storage.

Supports:
- S3-compatible storage (AWS S3, MinIO, etc.)
- SharePoint/OneDrive (planned)
- Google Drive (planned)
"""

from aragora.connectors.enterprise.documents.s3 import S3Connector

__all__ = [
    "S3Connector",
]
