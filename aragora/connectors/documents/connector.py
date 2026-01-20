"""
Document Connector.

Integrates DocumentParser with the EvidenceCollector to enable
omnivorous document ingestion for debate evidence.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aragora.connectors.base import Connector, Evidence
from aragora.connectors.documents.parser import (
    DocumentFormat,
    DocumentParser,
    ParsedDocument,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


class DocumentConnector(Connector):
    """Connector for parsing documents as evidence sources.

    Supports PDF, DOCX, XLSX, PPTX, HTML, JSON, YAML, XML, CSV formats.
    Integrates with EvidenceCollector for omnivorous document ingestion.

    Usage:
        connector = DocumentConnector()

        # Parse from file path
        results = await connector.search_file("/path/to/report.pdf")

        # Parse from bytes
        results = await connector.search_bytes(pdf_bytes, filename="report.pdf")

        # Search parsed documents
        results = await connector.search("machine learning", limit=5)
    """

    def __init__(
        self,
        max_pages: int = 100,
        extract_tables: bool = True,
        extract_metadata: bool = True,
        max_content_size: int = 500_000,
    ):
        """Initialize the document connector.

        Args:
            max_pages: Maximum pages to parse per document
            extract_tables: Whether to extract tables from documents
            extract_metadata: Whether to extract document metadata
            max_content_size: Maximum content size in bytes
        """
        self.parser = DocumentParser(
            max_pages=max_pages,
            extract_tables=extract_tables,
            extract_metadata=extract_metadata,
            max_content_size=max_content_size,
        )
        self._parsed_docs: Dict[str, ParsedDocument] = {}
        self._reliability_scores: Dict[str, float] = {
            "pdf": 0.85,  # Academic/official documents
            "docx": 0.80,  # Word documents
            "xlsx": 0.85,  # Spreadsheets (data)
            "pptx": 0.70,  # Presentations (summarized)
            "html": 0.65,  # Web content
            "json": 0.90,  # Structured data
            "yaml": 0.90,  # Config/structured data
            "xml": 0.85,  # Structured data
            "csv": 0.90,  # Tabular data
            "txt": 0.60,  # Plain text
            "md": 0.70,  # Markdown
        }

    @property
    def name(self) -> str:
        """Connector name."""
        return "documents"

    @property
    def source_type(self) -> SourceType:
        """The source type for this connector."""
        return SourceType.DOCUMENT

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """Fetch a specific piece of evidence by ID.

        Args:
            evidence_id: Unique identifier (format: doc_id_chunk_N or doc_id_table_N)

        Returns:
            Evidence object or None if not found
        """
        # Parse evidence ID to get document and chunk/table info
        parts = evidence_id.rsplit("_", 2)
        if len(parts) < 3:
            return None

        doc_id = parts[0]
        evidence_type = parts[1]  # "chunk" or "table"
        index = int(parts[2]) if parts[2].isdigit() else 0

        doc = self._parsed_docs.get(doc_id)
        if doc is None:
            return None

        if evidence_type == "chunk" and index < len(doc.chunks):
            chunk = doc.chunks[index]
            return Evidence(
                id=evidence_id,
                content=chunk.content,
                source=self.name,
                reliability_score=self._get_reliability(doc.format),
                metadata={
                    "document_id": doc_id,
                    "chunk_index": index,
                    "page": chunk.page,
                    "format": doc.format.value if doc.format else "unknown",
                },
            )
        elif evidence_type == "table" and index < len(doc.tables):
            table = doc.tables[index]
            # Handle both DocumentTable and raw list formats
            if hasattr(table, "to_markdown"):
                content = table.to_markdown()
            else:
                content = str(table)
            return Evidence(
                id=evidence_id,
                content=content,
                source=self.name,
                reliability_score=self._get_reliability(doc.format),
                metadata={
                    "document_id": doc_id,
                    "table_index": index,
                    "format": doc.format.value if doc.format else "unknown",
                },
            )

        return None

    def _get_reliability(self, format: DocumentFormat) -> float:
        """Get reliability score for a document format."""
        return self._reliability_scores.get(format.value if format else "unknown", 0.5)

    async def connect(self) -> bool:
        """Connect to document sources (always succeeds)."""
        return True

    async def disconnect(self) -> None:
        """Disconnect and clear cached documents."""
        self._parsed_docs.clear()

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search across all parsed documents.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching evidence results
        """
        results = []
        query_lower = query.lower()
        query_terms = query_lower.split()

        for doc_id, doc in self._parsed_docs.items():
            # Search in content
            for i, chunk in enumerate(doc.chunks):
                chunk_lower = chunk.content.lower()

                # Calculate relevance score
                term_matches = sum(1 for term in query_terms if term in chunk_lower)
                if term_matches == 0:
                    continue

                relevance = term_matches / len(query_terms)

                results.append({
                    "id": f"{doc_id}_chunk_{i}",
                    "title": f"{doc.title or doc.filename} (Section {i + 1})",
                    "content": chunk.content,
                    "url": doc.metadata.get("source_path", ""),
                    "source": "document",
                    "format": doc.format.value if doc.format else "unknown",
                    "relevance": relevance,
                    "reliability": self._get_reliability(doc.format),
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "page": chunk.page,
                        "section": chunk.section,
                        "filename": doc.filename,
                    },
                })

            # Search in tables
            for j, table in enumerate(doc.tables):
                table_str = str(table.data).lower()
                term_matches = sum(1 for term in query_terms if term in table_str)
                if term_matches == 0:
                    continue

                relevance = term_matches / len(query_terms)

                # Format table as text
                table_content = self._format_table(table.data, table.headers)

                results.append({
                    "id": f"{doc_id}_table_{j}",
                    "title": f"{doc.title or doc.filename} (Table {j + 1})",
                    "content": table_content,
                    "url": doc.metadata.get("source_path", ""),
                    "source": "document_table",
                    "format": doc.format.value if doc.format else "unknown",
                    "relevance": relevance,
                    "reliability": self._get_reliability(doc.format) + 0.05,  # Tables are higher reliability
                    "metadata": {
                        "doc_id": doc_id,
                        "table_index": j,
                        "page": table.page,
                        "caption": table.caption,
                        "filename": doc.filename,
                    },
                })

        # Sort by relevance and limit
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return results[:limit]

    async def parse_file(
        self,
        file_path: Union[str, Path],
    ) -> Optional[ParsedDocument]:
        """Parse a document file and add to search index.

        Args:
            file_path: Path to the document file

        Returns:
            ParsedDocument if successful, None otherwise
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path

        if not path.exists():
            logger.warning(f"Document file not found: {path}")
            return None

        try:
            content = path.read_bytes()
            doc = self.parser.parse(content, filename=path.name)

            if doc:
                doc_id = self._generate_doc_id(path)
                doc.metadata["source_path"] = str(path)
                doc.metadata["parsed_at"] = datetime.now().isoformat()
                self._parsed_docs[doc_id] = doc
                logger.info(f"Parsed document: {path.name} ({len(doc.chunks)} chunks, {len(doc.tables)} tables)")
                return doc

        except Exception as e:
            logger.error(f"Failed to parse document {path}: {e}")

        return None

    async def parse_bytes(
        self,
        content: bytes,
        filename: str,
        format: Optional[DocumentFormat] = None,
    ) -> Optional[ParsedDocument]:
        """Parse document bytes and add to search index.

        Args:
            content: Document content as bytes
            filename: Original filename (for format detection)
            format: Optional explicit format

        Returns:
            ParsedDocument if successful, None otherwise
        """
        try:
            doc = self.parser.parse(content, filename=filename, format=format)

            if doc:
                doc_id = self._generate_doc_id_from_content(content, filename)
                doc.metadata["parsed_at"] = datetime.now().isoformat()
                self._parsed_docs[doc_id] = doc
                logger.info(f"Parsed document: {filename} ({len(doc.chunks)} chunks)")
                return doc

        except Exception as e:
            logger.error(f"Failed to parse document bytes ({filename}): {e}")

        return None

    async def parse_from_url(
        self,
        url: str,
        content: bytes,
        filename: Optional[str] = None,
    ) -> Optional[ParsedDocument]:
        """Parse document fetched from URL.

        Args:
            url: Source URL
            content: Document content as bytes
            filename: Optional filename (derived from URL if not provided)

        Returns:
            ParsedDocument if successful, None otherwise
        """
        # Derive filename from URL if not provided
        if not filename:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            filename = Path(parsed.path).name or "document"

        doc = await self.parse_bytes(content, filename)
        if doc:
            doc.metadata["source_url"] = url

        return doc

    def get_parsed_documents(self) -> Dict[str, ParsedDocument]:
        """Get all parsed documents."""
        return self._parsed_docs.copy()

    def clear_documents(self) -> None:
        """Clear all parsed documents."""
        self._parsed_docs.clear()

    def remove_document(self, doc_id: str) -> bool:
        """Remove a specific parsed document.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if removed, False if not found
        """
        if doc_id in self._parsed_docs:
            del self._parsed_docs[doc_id]
            return True
        return False

    def _generate_doc_id(self, path: Path) -> str:
        """Generate unique document ID from path."""
        content = f"{path.absolute()}:{path.stat().st_mtime}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_doc_id_from_content(self, content: bytes, filename: str) -> str:
        """Generate unique document ID from content."""
        hasher = hashlib.sha256()
        hasher.update(content)
        hasher.update(filename.encode())
        return hasher.hexdigest()[:16]

    def _get_reliability(self, format: Optional[DocumentFormat]) -> float:
        """Get reliability score for document format."""
        if format is None:
            return 0.60
        return self._reliability_scores.get(format.value, 0.60)

    def _format_table(
        self,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
    ) -> str:
        """Format table data as text."""
        lines = []

        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 40)

        for row in data[:20]:  # Limit rows
            lines.append(" | ".join(str(cell)[:50] for cell in row))

        if len(data) > 20:
            lines.append(f"... ({len(data) - 20} more rows)")

        return "\n".join(lines)


class DocumentEvidence:
    """Helper class for creating evidence snippets from documents."""

    @staticmethod
    def from_parsed_document(
        doc: ParsedDocument,
        max_snippet_length: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Convert ParsedDocument to evidence snippets.

        Args:
            doc: Parsed document
            max_snippet_length: Maximum snippet length

        Returns:
            List of evidence dictionaries for EvidenceCollector
        """
        snippets = []
        base_reliability = 0.75

        # Get format-specific reliability
        format_scores = {
            DocumentFormat.PDF: 0.85,
            DocumentFormat.DOCX: 0.80,
            DocumentFormat.XLSX: 0.90,
            DocumentFormat.PPTX: 0.70,
            DocumentFormat.JSON: 0.90,
            DocumentFormat.YAML: 0.90,
            DocumentFormat.CSV: 0.90,
        }
        if doc.format:
            base_reliability = format_scores.get(doc.format, 0.75)

        # Create snippets from chunks
        for i, chunk in enumerate(doc.chunks):
            content = chunk.content[:max_snippet_length]
            if len(chunk.content) > max_snippet_length:
                content += "..."

            snippet = {
                "id": f"doc_{doc.filename}_{i}",
                "source": "document",
                "title": f"{doc.title or doc.filename}",
                "content": content,
                "url": doc.metadata.get("source_path", doc.metadata.get("source_url", "")),
                "reliability_score": base_reliability,
                "metadata": {
                    "filename": doc.filename,
                    "format": doc.format.value if doc.format else "unknown",
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_index": i,
                    "total_chunks": len(doc.chunks),
                },
            }
            snippets.append(snippet)

        # Create snippets from tables
        for j, table in enumerate(doc.tables):
            table_text = DocumentEvidence._format_table_text(table.data, table.headers)
            if len(table_text) > max_snippet_length:
                table_text = table_text[:max_snippet_length] + "..."

            snippet = {
                "id": f"doc_{doc.filename}_table_{j}",
                "source": "document_table",
                "title": f"{doc.title or doc.filename} - Table {j + 1}",
                "content": table_text,
                "url": doc.metadata.get("source_path", doc.metadata.get("source_url", "")),
                "reliability_score": base_reliability + 0.05,  # Tables are structured data
                "metadata": {
                    "filename": doc.filename,
                    "format": doc.format.value if doc.format else "unknown",
                    "page": table.page,
                    "caption": table.caption,
                    "table_index": j,
                    "total_tables": len(doc.tables),
                },
            }
            snippets.append(snippet)

        return snippets

    @staticmethod
    def _format_table_text(
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
    ) -> str:
        """Format table as text for evidence snippet."""
        lines = []

        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 40)

        for row in data[:15]:  # Limit for snippet
            lines.append(" | ".join(str(cell)[:30] for cell in row))

        if len(data) > 15:
            lines.append(f"[{len(data) - 15} more rows...]")

        return "\n".join(lines)


__all__ = [
    "DocumentConnector",
    "DocumentEvidence",
]
