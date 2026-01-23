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
]
