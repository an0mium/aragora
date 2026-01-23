"""
Low-Code Platform Connectors.

Integrations for low-code database and application platforms:
- Airtable (bases, tables, records)
- Knack (objects, records, views)
- Notion - see enterprise/collaboration
- Google Sheets - planned
"""

from aragora.connectors.lowcode.airtable import (
    AirtableConnector,
    AirtableCredentials,
    AirtableBase,
    AirtableTable,
    AirtableField,
    AirtableView,
    AirtableRecord,
    Attachment,
    AirtableError,
    FieldType as AirtableFieldType,
    get_mock_record as get_mock_airtable_record,
    get_mock_base as get_mock_airtable_base,
)
from aragora.connectors.lowcode.knack import (
    KnackConnector,
    KnackCredentials,
    KnackObject,
    KnackField,
    KnackRecord,
    KnackView,
    KnackScene,
    KnackError,
    FieldType as KnackFieldType,
    get_mock_record as get_mock_knack_record,
    get_mock_object as get_mock_knack_object,
)

__all__ = [
    # Airtable
    "AirtableConnector",
    "AirtableCredentials",
    "AirtableBase",
    "AirtableTable",
    "AirtableField",
    "AirtableView",
    "AirtableRecord",
    "Attachment",
    "AirtableError",
    "AirtableFieldType",
    "get_mock_airtable_record",
    "get_mock_airtable_base",
    # Knack
    "KnackConnector",
    "KnackCredentials",
    "KnackObject",
    "KnackField",
    "KnackRecord",
    "KnackView",
    "KnackScene",
    "KnackError",
    "KnackFieldType",
    "get_mock_knack_record",
    "get_mock_knack_object",
]
