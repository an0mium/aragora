"""
Legal Connectors.

Integrations for legal workflow automation:
- DocuSign e-signature
"""

from aragora.connectors.legal.docusign import (
    DocuSignConnector,
    DocuSignCredentials,
    DocuSignEnvironment,
    Envelope,
    EnvelopeCreateRequest,
    EnvelopeStatus,
    Recipient,
    RecipientType,
    Document,
    SignatureTab,
    get_mock_envelope,
)
from aragora.connectors.legal.westlaw import WestlawConnector
from aragora.connectors.legal.lexis import LexisConnector

__all__ = [
    "DocuSignConnector",
    "DocuSignCredentials",
    "DocuSignEnvironment",
    "Envelope",
    "EnvelopeCreateRequest",
    "EnvelopeStatus",
    "Recipient",
    "RecipientType",
    "Document",
    "SignatureTab",
    "get_mock_envelope",
    "WestlawConnector",
    "LexisConnector",
]
