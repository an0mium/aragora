"""
Knowledge Pipeline - End-to-end document processing with embedding and fact extraction.

Orchestrates the complete flow from document upload to queryable knowledge:
1. Document parsing (via Unstructured/Docling)
2. Chunking (semantic/sliding/recursive strategies)
3. Embedding (via Weaviate or in-memory)
4. Fact extraction (via agents)
5. Fact storage (via FactStore)
6. Knowledge Mound integration (optional unified knowledge layer)

Usage:
    from aragora.knowledge.pipeline import KnowledgePipeline

    pipeline = KnowledgePipeline(workspace_id="ws_123")
    await pipeline.start()

    # Process documents
    result = await pipeline.process_document(content, "contract.pdf")

    # Query the knowledge base
    answer = await pipeline.query("What are the payment terms?")

    await pipeline.stop()

With Knowledge Mound (unified enterprise storage):
    from aragora.knowledge.pipeline import KnowledgePipeline, PipelineConfig
    from aragora.knowledge.mound import MoundConfig, MoundBackend

    config = PipelineConfig(
        workspace_id="enterprise",
        use_knowledge_mound=True,
        mound_config=MoundConfig(
            backend=MoundBackend.HYBRID,
            postgres_url="postgresql://user:pass@host/db",
            redis_url="redis://localhost:6379",
        ),
    )
    pipeline = KnowledgePipeline(config)
    await pipeline.start()

    # Documents and facts are automatically synced to Knowledge Mound
    result = await pipeline.process_document(content, "contract.pdf")

    # Federated query across all knowledge sources
    answer = await pipeline.query("What are the payment terms?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

from aragora.documents.chunking import (
    ChunkingConfig,
    auto_select_strategy,
    get_chunking_strategy,
)
from aragora.documents.models import DocumentChunk, DocumentStatus, IngestedDocument
from aragora.knowledge.embeddings import (
    ChunkMatch,
    EmbeddingConfig,
    InMemoryEmbeddingService,
    WeaviateEmbeddingService,
)
from aragora.knowledge.fact_store import FactStore, InMemoryFactStore
from aragora.knowledge.query_engine import DatasetQueryEngine, QueryOptions, SimpleQueryEngine
from aragora.knowledge.types import Fact, QueryResult, ValidationStatus

# Optional Knowledge Mound imports
try:
    from aragora.knowledge.mound import (
        KnowledgeMound,
        MoundConfig,
        MoundBackend,
        IngestionRequest,
        KnowledgeSource,
        ConfidenceLevel,
    )

    MOUND_AVAILABLE = True
except ImportError:
    MOUND_AVAILABLE = False
    KnowledgeMound = None  # type: ignore
    MoundConfig = None  # type: ignore
    MoundBackend = None  # type: ignore

logger = logging.getLogger(__name__)

# Optional imports
try:
    from aragora.documents.ingestion import (
        UnstructuredParser,
        UNSTRUCTURED_AVAILABLE,
    )
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    UnstructuredParser = None  # type: ignore


@dataclass
class PipelineConfig:
    """Configuration for the knowledge pipeline."""

    # Workspace
    workspace_id: str = "default"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: Optional[str] = None  # Auto-select if None

    # Embedding
    use_weaviate: bool = False
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None

    # Fact extraction
    extract_facts: bool = True
    min_fact_confidence: float = 0.5

    # Storage
    fact_db_path: Optional[Path] = None

    # Knowledge Mound integration
    use_knowledge_mound: bool = False
    mound_config: Optional[Any] = None  # MoundConfig if available

    # Concurrency
    max_concurrent_embeddings: int = 10
    embedding_batch_size: int = 100


@dataclass
class ProcessingResult:
    """Result of processing a document through the pipeline."""

    document_id: str
    filename: str
    workspace_id: str
    chunk_count: int
    embedded_count: int
    fact_count: int
    total_tokens: int
    duration_ms: int
    success: bool
    error: Optional[str] = None
    document: Optional[IngestedDocument] = None
    chunks: list[DocumentChunk] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "workspace_id": self.workspace_id,
            "chunk_count": self.chunk_count,
            "embedded_count": self.embedded_count,
            "fact_count": self.fact_count,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


class KnowledgePipeline:
    """
    End-to-end document processing pipeline for enterprise knowledge management.

    Combines document ingestion, chunking, embedding, and fact extraction
    into a unified pipeline that produces a queryable knowledge base.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        fact_store: Optional[Union[FactStore, InMemoryFactStore]] = None,
        embedding_service: Optional[
            Union[WeaviateEmbeddingService, InMemoryEmbeddingService]
        ] = None,
        agents: Optional[list] = None,
        knowledge_mound: Optional[Any] = None,  # KnowledgeMound if available
    ):
        """
        Initialize the knowledge pipeline.

        Args:
            config: Pipeline configuration
            fact_store: Optional pre-configured fact store
            embedding_service: Optional pre-configured embedding service
            agents: Optional list of agents for fact extraction
            knowledge_mound: Optional pre-configured KnowledgeMound for unified storage
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self._fact_store = fact_store
        self._embedding_service = embedding_service
        self._agents = agents or []
        self._query_engine: Optional[Union[DatasetQueryEngine, SimpleQueryEngine]] = None
        self._knowledge_mound = knowledge_mound

        # Parser
        self._parser: Optional[Any] = None

        # State
        self._running = False
        self._stats = {
            "documents_processed": 0,
            "chunks_embedded": 0,
            "facts_extracted": 0,
            "queries_answered": 0,
            "mound_synced": 0,
        }

        # Callbacks
        self._on_progress: Optional[Callable[[str, float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[str, float, str], None]) -> None:
        """Set progress callback: callback(document_id, progress, message)."""
        self._on_progress = callback

    def _report_progress(self, document_id: str, progress: float, message: str) -> None:
        """Report progress if callback is set."""
        if self._on_progress:
            self._on_progress(document_id, progress, message)

    async def start(self) -> None:
        """Start the pipeline and initialize components."""
        if self._running:
            return

        logger.info(f"Starting knowledge pipeline for workspace {self.config.workspace_id}")

        # Initialize fact store
        if self._fact_store is None:
            if self.config.fact_db_path:
                self._fact_store = FactStore(db_path=self.config.fact_db_path)
            else:
                try:
                    self._fact_store = FactStore()
                except Exception as e:
                    logger.warning(f"Failed to create FactStore, using in-memory: {e}")
                    self._fact_store = InMemoryFactStore()

        # Initialize embedding service
        if self._embedding_service is None:
            if self.config.use_weaviate:
                try:
                    self._embedding_service = WeaviateEmbeddingService(
                        EmbeddingConfig(
                            weaviate_url=self.config.weaviate_url,
                            weaviate_api_key=self.config.weaviate_api_key,
                        )
                    )
                    self._embedding_service.connect()
                except Exception as e:
                    logger.warning(f"Failed to connect to Weaviate, using in-memory: {e}")
                    self._embedding_service = InMemoryEmbeddingService()
            else:
                self._embedding_service = InMemoryEmbeddingService()

        # Initialize Knowledge Mound if configured
        if self.config.use_knowledge_mound and MOUND_AVAILABLE and self._knowledge_mound is None:
            try:
                mound_config = self.config.mound_config or MoundConfig()
                self._knowledge_mound = KnowledgeMound(
                    config=mound_config,
                    workspace_id=self.config.workspace_id,
                )
                await self._knowledge_mound.initialize()
                logger.info("Knowledge Mound initialized for pipeline")
            except Exception as e:
                logger.warning(f"Failed to initialize Knowledge Mound: {e}")
                self._knowledge_mound = None

        # Initialize parser
        if UNSTRUCTURED_AVAILABLE and UnstructuredParser:
            self._parser = UnstructuredParser()
        else:
            logger.warning("Unstructured not available, text-only parsing")

        # Initialize query engine
        self._query_engine = SimpleQueryEngine(
            fact_store=self._fact_store,
            embedding_service=self._embedding_service,
        )

        self._running = True
        logger.info("Knowledge pipeline started")

    async def stop(self) -> None:
        """Stop the pipeline and cleanup resources."""
        if not self._running:
            return

        self._running = False

        # Cleanup embedding service
        if self._embedding_service:
            try:
                self._embedding_service.close()
            except Exception as e:
                logger.debug(f"Error closing embedding service: {e}")

        # Cleanup Knowledge Mound
        if self._knowledge_mound:
            try:
                await self._knowledge_mound.close()
            except Exception as e:
                logger.debug(f"Error closing knowledge mound: {e}")

        logger.info("Knowledge pipeline stopped")

    async def process_document(
        self,
        content: bytes,
        filename: str,
        tags: Optional[list[str]] = None,
        extract_facts: Optional[bool] = None,
    ) -> ProcessingResult:
        """
        Process a single document through the pipeline.

        Args:
            content: Raw file content
            filename: Original filename
            tags: Optional tags for categorization
            extract_facts: Override config setting for fact extraction

        Returns:
            ProcessingResult with document, chunks, and facts
        """
        if not self._running:
            await self.start()

        start_time = datetime.now()
        document_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename.replace('.', '_')}"

        try:
            self._report_progress(document_id, 0.0, "Starting document processing...")

            # Step 1: Parse document (20%)
            self._report_progress(document_id, 0.1, "Parsing document...")
            document, text = await self._parse_document(content, filename, document_id, tags)

            # Step 2: Chunk document (40%)
            self._report_progress(document_id, 0.3, "Chunking document...")
            chunks = await self._chunk_document(document, text)

            # Step 3: Embed chunks (70%)
            self._report_progress(document_id, 0.5, "Embedding chunks...")
            embedded_count = await self._embed_chunks(chunks, document_id)

            # Step 4: Extract facts (70%)
            facts = []
            should_extract = (
                extract_facts if extract_facts is not None else self.config.extract_facts
            )
            if should_extract and self._agents:
                self._report_progress(document_id, 0.7, "Extracting facts...")
                facts = await self._extract_facts(document, chunks)

            # Step 5: Sync to Knowledge Mound (90%)
            mound_synced = 0
            if self._knowledge_mound and MOUND_AVAILABLE:
                self._report_progress(document_id, 0.85, "Syncing to Knowledge Mound...")
                mound_synced = await self._sync_to_mound(document, chunks, facts, tags)

            # Finalize
            self._report_progress(document_id, 1.0, "Processing complete")

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Update stats
            self._stats["documents_processed"] += 1
            self._stats["chunks_embedded"] += embedded_count
            self._stats["facts_extracted"] += len(facts)
            self._stats["mound_synced"] += mound_synced

            return ProcessingResult(
                document_id=document_id,
                filename=filename,
                workspace_id=self.config.workspace_id,
                chunk_count=len(chunks),
                embedded_count=embedded_count,
                fact_count=len(facts),
                total_tokens=sum(c.token_count for c in chunks),
                duration_ms=duration_ms,
                success=True,
                document=document,
                chunks=chunks,
                facts=facts,
            )

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return ProcessingResult(
                document_id=document_id,
                filename=filename,
                workspace_id=self.config.workspace_id,
                chunk_count=0,
                embedded_count=0,
                fact_count=0,
                total_tokens=0,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            )

    async def process_batch(
        self,
        files: list[tuple[bytes, str]],
        tags: Optional[list[str]] = None,
    ) -> list[ProcessingResult]:
        """
        Process multiple documents.

        Args:
            files: List of (content, filename) tuples
            tags: Optional tags for all documents

        Returns:
            List of ProcessingResult for each document
        """
        results = []
        for content, filename in files:
            result = await self.process_document(content, filename, tags)
            results.append(result)
        return results

    async def _parse_document(
        self,
        content: bytes,
        filename: str,
        document_id: str,
        tags: Optional[list[str]],
    ) -> tuple[IngestedDocument, str]:
        """Parse document content and extract text."""
        if self._parser and UNSTRUCTURED_AVAILABLE:
            document = self._parser.parse_to_document(
                content=content,
                filename=filename,
                workspace_id=self.config.workspace_id,
                tags=tags or [],
            )
            # Override the auto-generated ID
            document.id = document_id
            return document, document.text
        else:
            # Fallback: treat as text
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1")

            document = IngestedDocument(
                id=document_id,
                filename=filename,
                workspace_id=self.config.workspace_id,
                content_type="text/plain",
                file_size=len(content),
                status=DocumentStatus.PROCESSING,
                text=text,
            )
            return document, text

    async def _chunk_document(
        self,
        document: IngestedDocument,
        text: str,
    ) -> list[DocumentChunk]:
        """Chunk the document text."""
        # Auto-select strategy if not specified
        strategy_name = self.config.chunking_strategy
        if not strategy_name:
            strategy_name = auto_select_strategy(text, document.filename)

        # Ensure valid strategy
        if strategy_name not in ("semantic", "sliding", "recursive", "fixed"):
            strategy_name = "semantic"

        config = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )

        strategy = get_chunking_strategy(strategy_name, **config.__dict__)  # type: ignore
        chunks = strategy.chunk(text=text, document_id=document.id)

        # Update document
        document.chunk_count = len(chunks)
        document.chunk_ids = [c.id for c in chunks]
        document.chunking_strategy = strategy_name
        document.chunk_size = self.config.chunk_size
        document.chunk_overlap = self.config.chunk_overlap
        document.total_tokens = sum(c.token_count for c in chunks)
        document.status = DocumentStatus.INDEXED

        return chunks

    async def _embed_chunks(
        self,
        chunks: list[DocumentChunk],
        document_id: str,
    ) -> int:
        """Embed chunks into the vector store."""
        if not self._embedding_service:
            return 0

        # Convert DocumentChunks to embedding format
        chunk_data = [
            {
                "chunk_id": c.id,
                "document_id": c.document_id,
                "content": c.content,
                "chunk_index": c.sequence,
                "file_path": "",
                "file_type": "",
                "topics": [],
            }
            for c in chunks
        ]

        # Embed in batches
        total_embedded = 0
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(chunk_data), batch_size):
            batch = chunk_data[i : i + batch_size]
            count = await self._embedding_service.embed_chunks(batch, self.config.workspace_id)
            total_embedded += count

        return total_embedded

    async def _extract_facts(
        self,
        document: IngestedDocument,
        chunks: list[DocumentChunk],
    ) -> list[Fact]:
        """Extract facts from document using agents."""
        if not self._agents or not self._fact_store:
            return []

        facts = []

        # Use first agent for fact extraction
        agent = self._agents[0]

        # Build extraction prompt
        # Take first N chunks for context
        context_chunks = chunks[:5]
        chunk_texts = "\n\n".join(
            f"[Chunk {c.sequence}]: {c.content[:500]}..." for c in context_chunks
        )

        prompt = f"""Analyze the following document excerpts and extract specific factual statements.

Document: {document.filename}

Excerpts:
{chunk_texts}

Extract 5-10 specific, verifiable facts from this document.
Format each fact on its own line starting with "FACT: "
Include dates, numbers, names, and specific claims where possible."""

        try:
            response = await agent.generate(prompt, [])

            # Parse facts from response
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("FACT:"):
                    statement = line[5:].strip()
                    if len(statement) > 10:
                        fact = self._fact_store.add_fact(
                            statement=statement,
                            workspace_id=self.config.workspace_id,
                            source_documents=[document.id],
                            evidence_ids=[c.id for c in context_chunks],
                            confidence=self.config.min_fact_confidence,
                            validation_status=ValidationStatus.UNVERIFIED,
                        )
                        facts.append(fact)

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")

        return facts

    async def _sync_to_mound(
        self,
        document: IngestedDocument,
        chunks: list[DocumentChunk],
        facts: list[Fact],
        tags: Optional[list[str]] = None,
    ) -> int:
        """
        Sync document, chunks, and facts to the Knowledge Mound.

        Args:
            document: Ingested document
            chunks: Document chunks
            facts: Extracted facts
            tags: Optional tags

        Returns:
            Number of items synced
        """
        if not self._knowledge_mound or not MOUND_AVAILABLE:
            return 0

        synced = 0
        try:
            # Store the document as a knowledge item
            doc_request = IngestionRequest(
                content=document.text or "",
                source=KnowledgeSource.DOCUMENT,
                workspace_id=self.config.workspace_id,
                confidence=ConfidenceLevel.HIGH,
                metadata={
                    "document_id": document.id,
                    "filename": document.filename,
                    "content_type": document.content_type,
                    "file_size": document.file_size,
                    "chunk_count": len(chunks),
                    "tags": tags or [],
                },
            )
            result = await self._knowledge_mound.store(doc_request)
            if result.success:
                synced += 1

            # Store each chunk as a knowledge item linked to the document
            for chunk in chunks[:50]:  # Limit chunks to avoid overload
                chunk_request = IngestionRequest(
                    content=chunk.content,
                    source=KnowledgeSource.DOCUMENT,
                    workspace_id=self.config.workspace_id,
                    confidence=ConfidenceLevel.HIGH,
                    metadata={
                        "chunk_id": chunk.id,
                        "document_id": document.id,
                        "sequence": chunk.sequence,
                        "token_count": chunk.token_count,
                        "parent_node_id": result.node_id if result.success else None,
                    },
                )
                chunk_result = await self._knowledge_mound.store(chunk_request)
                if chunk_result.success:
                    synced += 1

            # Store each fact as a knowledge item
            for fact in facts:
                fact_request = IngestionRequest(
                    content=fact.statement,
                    source=KnowledgeSource.EXTRACTION,
                    workspace_id=self.config.workspace_id,
                    confidence=ConfidenceLevel.from_score(fact.confidence),
                    metadata={
                        "fact_id": fact.id,
                        "document_id": document.id,
                        "validation_status": (
                            fact.validation_status.value if fact.validation_status else "unverified"
                        ),
                        "source_documents": fact.source_documents,
                    },
                )
                fact_result = await self._knowledge_mound.store(fact_request)
                if fact_result.success:
                    synced += 1

            logger.debug(f"Synced {synced} items to Knowledge Mound for document {document.id}")

        except Exception as e:
            logger.warning(f"Failed to sync to Knowledge Mound: {e}")

        return synced

    async def query(
        self,
        question: str,
        options: Optional[QueryOptions] = None,
        use_mound: bool = True,
    ) -> QueryResult:
        """
        Query the knowledge base.

        Args:
            question: Natural language question
            options: Query options
            use_mound: Whether to include Knowledge Mound in the query (if available)

        Returns:
            QueryResult with answer and supporting facts
        """
        if not self._running:
            await self.start()

        if not self._query_engine:
            raise RuntimeError("Query engine not initialized")

        # Standard query through query engine
        result = await self._query_engine.query(question, self.config.workspace_id, options)

        # Augment with Knowledge Mound results if available
        if use_mound and self._knowledge_mound and MOUND_AVAILABLE:
            try:
                mound_result = await self._knowledge_mound.query(
                    query=question,
                    workspace_id=self.config.workspace_id,
                    limit=10,
                )
                # Merge mound results into the answer context
                if mound_result.items and hasattr(result, "context"):
                    mound_context = "\n".join(
                        f"[Knowledge Mound] {item.content[:200]}..."
                        for item in mound_result.items[:3]
                    )
                    if result.context:
                        result.context = f"{result.context}\n\n{mound_context}"
                    else:
                        result.context = mound_context
            except Exception as e:
                logger.debug(f"Knowledge Mound query augmentation skipped: {e}")

        self._stats["queries_answered"] += 1
        return result

    async def query_mound(
        self,
        query: str,
        limit: int = 20,
        sources: Optional[list[str]] = None,
    ) -> Any:
        """
        Query the Knowledge Mound directly for unified cross-source search.

        Args:
            query: Search query
            limit: Maximum results
            sources: Optional source filters (e.g., ["document", "debate", "consensus"])

        Returns:
            QueryResult from Knowledge Mound, or None if not available
        """
        if not self._running:
            await self.start()

        if not self._knowledge_mound or not MOUND_AVAILABLE:
            logger.warning("Knowledge Mound not available for direct query")
            return None

        return await self._knowledge_mound.query(
            query=query,
            workspace_id=self.config.workspace_id,
            sources=sources or ["all"],
            limit=limit,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ChunkMatch]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching chunks
        """
        if not self._running:
            await self.start()

        if not self._embedding_service:
            return []

        return await self._embedding_service.hybrid_search(query, self.config.workspace_id, limit)

    async def get_facts(
        self,
        query: Optional[str] = None,
        limit: int = 50,
        min_confidence: float = 0.0,
    ) -> list[Fact]:
        """
        Get facts from the knowledge base.

        Args:
            query: Optional search query
            limit: Maximum facts
            min_confidence: Minimum confidence filter

        Returns:
            List of facts
        """
        if not self._running:
            await self.start()

        if not self._fact_store:
            return []

        from aragora.knowledge.types import FactFilters

        filters = FactFilters(
            workspace_id=self.config.workspace_id,
            min_confidence=min_confidence,
            limit=limit,
        )

        if query:
            return self._fact_store.query_facts(query, filters)
        else:
            return self._fact_store.list_facts(filters)

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        embedding_stats = {}
        if self._embedding_service:
            embedding_stats = self._embedding_service.get_statistics(self.config.workspace_id)

        fact_stats = {}
        if self._fact_store:
            fact_stats = self._fact_store.get_statistics(self.config.workspace_id)

        mound_stats = {}
        if self._knowledge_mound:
            try:
                # Sync call - stats are usually lightweight
                mound_stats = {
                    "enabled": True,
                    "synced_items": self._stats.get("mound_synced", 0),
                }
            except (RuntimeError, AttributeError, KeyError) as e:
                logger.debug(f"Could not get mound stats: {e}")
                mound_stats = {"enabled": True, "error": "stats unavailable"}
        else:
            mound_stats = {"enabled": False}

        return {
            "running": self._running,
            "workspace_id": self.config.workspace_id,
            "pipeline_stats": self._stats,
            "embedding_stats": embedding_stats,
            "fact_stats": fact_stats,
            "mound_stats": mound_stats,
        }

    async def get_stale_knowledge(
        self,
        threshold: float = 0.5,
        limit: int = 50,
    ) -> list[Any]:
        """
        Get stale knowledge items that may need revalidation.

        Args:
            threshold: Minimum staleness score (0.0 to 1.0)
            limit: Maximum items to return

        Returns:
            List of StalenessCheck results from Knowledge Mound
        """
        if not self._knowledge_mound or not MOUND_AVAILABLE:
            return []

        try:
            return await self._knowledge_mound.get_stale_knowledge(
                threshold=threshold,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"Failed to get stale knowledge: {e}")
            return []

    async def get_culture_profile(self) -> Any:
        """
        Get the organizational culture profile from observed debates.

        Returns:
            CultureProfile from Knowledge Mound, or None if not available
        """
        if not self._knowledge_mound or not MOUND_AVAILABLE:
            return None

        try:
            return await self._knowledge_mound.get_culture_profile()
        except Exception as e:
            logger.warning(f"Failed to get culture profile: {e}")
            return None

    async def observe_debate(self, debate_result: Any) -> None:
        """
        Observe a debate result to accumulate cultural patterns.

        Args:
            debate_result: Result from Arena.run() or similar debate execution
        """
        if not self._knowledge_mound or not MOUND_AVAILABLE:
            return

        try:
            await self._knowledge_mound.observe_debate(debate_result)
        except Exception as e:
            logger.warning(f"Failed to observe debate: {e}")

    @property
    def knowledge_mound(self) -> Any:
        """Get the Knowledge Mound instance (if available)."""
        return self._knowledge_mound


# Convenience function
async def create_pipeline(
    workspace_id: str,
    use_weaviate: bool = False,
    weaviate_url: str = "http://localhost:8080",
    use_knowledge_mound: bool = False,
    mound_config: Optional[Any] = None,
) -> KnowledgePipeline:
    """
    Create and start a knowledge pipeline.

    Args:
        workspace_id: Workspace identifier
        use_weaviate: Whether to use Weaviate for embeddings
        weaviate_url: Weaviate server URL
        use_knowledge_mound: Whether to enable Knowledge Mound integration
        mound_config: Optional MoundConfig for Knowledge Mound

    Returns:
        Started KnowledgePipeline
    """
    config = PipelineConfig(
        workspace_id=workspace_id,
        use_weaviate=use_weaviate,
        weaviate_url=weaviate_url,
        use_knowledge_mound=use_knowledge_mound,
        mound_config=mound_config,
    )
    pipeline = KnowledgePipeline(config)
    await pipeline.start()
    return pipeline
