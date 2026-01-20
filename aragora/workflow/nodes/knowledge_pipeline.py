"""
Knowledge Pipeline Step for document ingestion within workflows.

Wraps the KnowledgePipeline to enable:
- Document processing as a workflow step
- Multi-source ingestion
- Knowledge Mound integration
- Connector-based data fetching
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


class KnowledgePipelineStep(BaseStep):
    """
    Workflow step that executes document ingestion via KnowledgePipeline.

    Processes documents from various sources and stores them in the
    Knowledge Mound for later retrieval.

    Config options:
        sources: List[str] - Document sources to process
            Can be file paths, URLs, or connector IDs
        workspace_id: str - Target workspace for storage
        chunk_strategy: str - Chunking strategy (default: "semantic")
            Options: semantic, sliding, recursive, sentence
        chunk_size: int - Target chunk size (default: 512)
        chunk_overlap: int - Overlap between chunks (default: 64)
        embedding_model: str - Embedding model (default: "text-embedding-3-small")
        use_knowledge_mound: bool - Store in Knowledge Mound (default: True)
        extract_facts: bool - Extract facts from documents (default: True)
        connector_type: str - Connector for remote sources (default: "local_docs")
            Options: local_docs, web, github, confluence, notion
        connector_config: Dict - Connector-specific configuration
        timeout_seconds: float - Processing timeout (default: 600)
        batch_size: int - Documents per batch (default: 10)

    Usage:
        step = KnowledgePipelineStep(
            name="Ingest Contracts",
            config={
                "sources": ["/path/to/contracts/", "https://docs.example.com"],
                "workspace_id": "legal",
                "chunk_strategy": "semantic",
                "extract_facts": True,
            }
        )
        result = await step.execute(context)
    """

    CHUNK_STRATEGIES = ["semantic", "sliding", "recursive", "sentence"]
    CONNECTOR_TYPES = ["local_docs", "web", "github", "confluence", "notion"]

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._pipeline = None
        self._documents_processed = 0

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the knowledge pipeline step."""
        config = {**self._config, **context.current_step_config}

        # Extract configuration
        sources = config.get("sources", [])
        workspace_id = config.get("workspace_id", context.get_input("workspace_id", "default"))
        chunk_strategy = config.get("chunk_strategy", "semantic")
        chunk_size = config.get("chunk_size", 512)
        chunk_overlap = config.get("chunk_overlap", 64)
        embedding_model = config.get("embedding_model", "text-embedding-3-small")
        use_knowledge_mound = config.get("use_knowledge_mound", True)
        extract_facts = config.get("extract_facts", True)
        connector_type = config.get("connector_type", "local_docs")
        connector_config = config.get("connector_config", {})
        timeout_seconds = config.get("timeout_seconds", 600.0)
        config.get("batch_size", 10)

        # Also check workflow inputs for sources
        input_sources = context.get_input("sources", [])
        if input_sources:
            sources = sources + input_sources

        if not sources:
            logger.warning(f"No sources specified for KnowledgePipelineStep '{self.name}'")
            return {
                "success": False,
                "error": "No sources specified",
                "documents_processed": 0,
            }

        # Validate chunk strategy
        if chunk_strategy not in self.CHUNK_STRATEGIES:
            logger.warning(f"Unknown chunk strategy '{chunk_strategy}', using 'semantic'")
            chunk_strategy = "semantic"

        logger.info(
            f"KnowledgePipelineStep '{self.name}' starting: "
            f"sources={len(sources)}, workspace={workspace_id}, strategy={chunk_strategy}"
        )

        try:
            from aragora.knowledge.pipeline import KnowledgePipeline, PipelineConfig
            from aragora.documents.chunking import ChunkingConfig

            # Build pipeline configuration
            chunking_config = ChunkingConfig(
                strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            pipeline_config = PipelineConfig(
                workspace_id=workspace_id,
                chunking=chunking_config,
                embedding_model=embedding_model,
                use_knowledge_mound=use_knowledge_mound,
                extract_facts=extract_facts,
                timeout_seconds=timeout_seconds,
            )

            # Create and start pipeline
            self._pipeline = KnowledgePipeline(pipeline_config)
            await self._pipeline.start()

            # Process each source
            results = []
            errors = []

            for source in sources:
                try:
                    result = await self._process_source(
                        source=source,
                        connector_type=connector_type,
                        connector_config=connector_config,
                    )
                    results.append(result)
                    self._documents_processed += result.get("documents", 0)
                except Exception as e:
                    logger.error(f"Failed to process source '{source}': {e}")
                    errors.append(
                        {
                            "source": source,
                            "error": str(e),
                        }
                    )

            # Stop pipeline
            await self._pipeline.stop()

            return {
                "success": len(errors) == 0,
                "documents_processed": self._documents_processed,
                "sources_processed": len(results),
                "sources_failed": len(errors),
                "results": results,
                "errors": errors,
                "workspace_id": workspace_id,
                "chunk_strategy": chunk_strategy,
            }

        except ImportError as e:
            logger.error(f"Failed to import knowledge pipeline: {e}")
            return {
                "success": False,
                "error": f"Knowledge pipeline not available: {e}",
                "documents_processed": 0,
            }

    async def _process_source(
        self,
        source: str,
        connector_type: str,
        connector_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single source through the pipeline."""
        source_path = Path(source) if not source.startswith(("http://", "https://")) else None

        # Check if it's a local path
        if source_path and source_path.exists():
            if source_path.is_dir():
                return await self._process_directory(source_path)
            else:
                return await self._process_file(source_path)

        # Check if it's a URL
        if source.startswith(("http://", "https://")):
            return await self._process_url(source, connector_config)

        # Try as a connector ID
        return await self._process_with_connector(
            source=source,
            connector_type=connector_type,
            connector_config=connector_config,
        )

    async def _process_directory(self, directory: Path) -> Dict[str, Any]:
        """Process all documents in a directory."""
        documents_processed = 0
        chunks_created = 0

        for file_path in directory.rglob("*"):
            if file_path.is_file() and self._is_supported_file(file_path):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    result = await self._pipeline.process_document(
                        content=content,
                        filename=file_path.name,
                        metadata={"path": str(file_path)},
                    )
                    documents_processed += 1
                    chunks_created += result.get("chunks_created", 0)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")

        return {
            "source": str(directory),
            "type": "directory",
            "documents": documents_processed,
            "chunks": chunks_created,
        }

    async def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file."""
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        result = await self._pipeline.process_document(
            content=content,
            filename=file_path.name,
            metadata={"path": str(file_path)},
        )

        return {
            "source": str(file_path),
            "type": "file",
            "documents": 1,
            "chunks": result.get("chunks_created", 0),
        }

    async def _process_url(
        self,
        url: str,
        connector_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a URL source."""
        try:
            from aragora.connectors.web import WebConnector

            connector = WebConnector(config=connector_config)
            content = await connector.fetch(url)

            result = await self._pipeline.process_document(
                content=content,
                filename=url.split("/")[-1] or "web_document",
                metadata={"url": url},
            )

            return {
                "source": url,
                "type": "url",
                "documents": 1,
                "chunks": result.get("chunks_created", 0),
            }

        except ImportError:
            logger.warning("Web connector not available, skipping URL")
            return {
                "source": url,
                "type": "url",
                "documents": 0,
                "error": "Web connector not available",
            }

    async def _process_with_connector(
        self,
        source: str,
        connector_type: str,
        connector_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process using a specific connector."""
        try:
            # Dynamic connector import based on type
            connector_module = {
                "local_docs": "aragora.connectors.local_docs",
                "web": "aragora.connectors.web",
                "github": "aragora.connectors.github",
                "confluence": "aragora.connectors.confluence",
                "notion": "aragora.connectors.notion",
            }.get(connector_type)

            if not connector_module:
                return {
                    "source": source,
                    "type": connector_type,
                    "documents": 0,
                    "error": f"Unknown connector type: {connector_type}",
                }

            import importlib

            module = importlib.import_module(connector_module)
            connector_class = getattr(module, f"{connector_type.title().replace('_', '')}Connector")
            connector = connector_class(config=connector_config)

            # Fetch documents from connector
            documents = await connector.fetch_all(source)
            documents_processed = 0
            chunks_created = 0

            for doc in documents:
                result = await self._pipeline.process_document(
                    content=doc.get("content", ""),
                    filename=doc.get("filename", "document"),
                    metadata=doc.get("metadata", {}),
                )
                documents_processed += 1
                chunks_created += result.get("chunks_created", 0)

            return {
                "source": source,
                "type": connector_type,
                "documents": documents_processed,
                "chunks": chunks_created,
            }

        except Exception as e:
            logger.error(f"Connector processing failed: {e}")
            return {
                "source": source,
                "type": connector_type,
                "documents": 0,
                "error": str(e),
            }

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file type is supported for processing."""
        supported_extensions = {
            ".txt",
            ".md",
            ".rst",
            ".html",
            ".htm",
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".csv",
        }
        return file_path.suffix.lower() in supported_extensions

    async def checkpoint(self) -> Dict[str, Any]:
        """Save pipeline step state for checkpointing."""
        return {
            "documents_processed": self._documents_processed,
        }

    async def restore(self, state: Dict[str, Any]) -> None:
        """Restore pipeline step state from checkpoint."""
        self._documents_processed = state.get("documents_processed", 0)

    def validate_config(self) -> bool:
        """Validate pipeline step configuration."""
        chunk_strategy = self._config.get("chunk_strategy", "semantic")
        if chunk_strategy not in self.CHUNK_STRATEGIES:
            logger.warning(f"Invalid chunk strategy: {chunk_strategy}")
            return False
        return True
