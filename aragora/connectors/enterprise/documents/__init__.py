"""
Document Connectors for cloud storage.

Supports:
- S3-compatible storage (AWS S3, MinIO, etc.)
- Microsoft SharePoint Online
- Google Drive
"""

from aragora.connectors.enterprise.documents.s3 import S3Connector
from aragora.connectors.enterprise.documents.sharepoint import SharePointConnector
from aragora.connectors.enterprise.documents.gdrive import GoogleDriveConnector

__all__ = [
    "S3Connector",
    "SharePointConnector",
    "GoogleDriveConnector",
]
