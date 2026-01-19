"""
Document Connectors for cloud storage.

Supports:
- S3-compatible storage (AWS S3, MinIO, etc.)
- Microsoft SharePoint Online
- Microsoft OneDrive
- Google Drive
- Dropbox
"""

from aragora.connectors.enterprise.documents.s3 import S3Connector
from aragora.connectors.enterprise.documents.sharepoint import SharePointConnector
from aragora.connectors.enterprise.documents.gdrive import GoogleDriveConnector
from aragora.connectors.enterprise.documents.onedrive import OneDriveConnector
from aragora.connectors.enterprise.documents.dropbox import DropboxConnector

__all__ = [
    "S3Connector",
    "SharePointConnector",
    "GoogleDriveConnector",
    "OneDriveConnector",
    "DropboxConnector",
]
