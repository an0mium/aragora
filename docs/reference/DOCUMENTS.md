# Document Ingestion & Processing

Aragora ingests documents as evidence for debates, audits, and workflows. The
DocumentConnector uses the unified DocumentParser to extract text, tables, and
metadata into structured evidence objects.

## Supported Formats

The parser supports common business and technical formats, including:

- **Office**: PDF, DOC/DOCX, XLS/XLSX, PPT/PPTX
- **Text**: TXT, Markdown, RST, HTML, RTF
- **Structured data**: JSON, YAML, XML, CSV
- **Notebooks**: IPYNB
- **E-books**: EPUB, MOBI
- **Archives**: ZIP, TAR, GZIP (expanded before parsing)

Source: `aragora/connectors/documents/parser.py`

## Connector Usage

```python
from aragora.connectors.documents import DocumentConnector

connector = DocumentConnector(max_pages=100, extract_tables=True)

# Parse a file from disk
results = await connector.search_file("/path/to/report.pdf")

# Parse raw bytes
with open("policy.docx", "rb") as f:
    results = await connector.search_bytes(f.read(), filename="policy.docx")

# Search within parsed content
matches = await connector.search("access control", limit=5)
```

Source: `aragora/connectors/documents/connector.py`

## Where Documents Flow

- **Evidence system** for grounded debate (see [EVIDENCE.md](EVIDENCE.md))
- **Gauntlet** for adversarial audits (see [GAUNTLET.md](GAUNTLET.md))
- **Knowledge Mound** for long-term storage (see [KNOWLEDGE_MOUND.md](KNOWLEDGE_MOUND.md))

## Configuration Notes

Document parsing limits and table extraction are configured in the connector
constructor. For production workloads, tune page limits and max content size to
match your document sizes and compute budget.
