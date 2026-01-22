"""
Memory Steps for Knowledge Mound integration.

Provides workflow steps for reading and writing to the Knowledge Mound:
- MemoryReadStep: Query and retrieve knowledge
- MemoryWriteStep: Store new knowledge with relationships
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


class MemoryReadStep(BaseStep):
    """
    Memory read step for querying the Knowledge Mound.

    Config options:
        query: str - Query string (can use {input} placeholders)
        query_type: str - "semantic", "keyword", or "hybrid" (default: "hybrid")
        sources: List[str] - Source types to query (e.g., ["fact", "consensus"])
        domain_filter: str - Domain to filter by (e.g., "legal/contracts")
        min_confidence: float - Minimum confidence threshold
        limit: int - Maximum results to return (default: 10)
        include_graph: bool - Include related graph nodes (default: False)
        graph_depth: int - Graph expansion depth (default: 1)
        tenant_id: str - Tenant for multi-tenant isolation

    Usage:
        step = MemoryReadStep(
            name="Retrieve Contract Knowledge",
            config={
                "query": "What are the termination requirements for {contract_type}?",
                "domain_filter": "legal/contracts",
                "min_confidence": 0.7,
                "limit": 5,
            }
        )
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the memory read step."""
        config = {**self._config, **context.current_step_config}

        # Build query from template and context
        query_template = config.get("query", "")
        query = self._interpolate_query(query_template, context)

        if not query:
            logger.warning(f"Empty query for memory read step '{self.name}'")
            return {"items": [], "total_count": 0, "query": query}

        # Get Knowledge Mound instance
        try:
            from aragora.knowledge.mound import KnowledgeMound
            from aragora.knowledge.mound.types import UnifiedQueryRequest

            # Build query request
            request = UnifiedQueryRequest(
                query=query,
                tenant_id=config.get("tenant_id", context.metadata.get("tenant_id", "default")),
                search_mode=config.get("query_type", "hybrid"),
                domain_filter=config.get("domain_filter"),
                min_confidence=config.get("min_confidence"),
                include_graph=config.get("include_graph", False),
                graph_depth=config.get("graph_depth", 1),
                limit=config.get("limit", 10),
            )

            # Execute query
            mound = KnowledgeMound(workspace_id=request.tenant_id)
            await mound.initialize()

            result = await mound.query(
                query=request.query,
                sources=config.get("sources"),
                limit=request.limit,
            )

            logger.info(f"Memory read '{self.name}': found {len(result.items)} items")

            return {
                "items": [item.to_dict() for item in result.items],
                "total_count": result.total_count,
                "query": query,
                "execution_time_ms": result.execution_time_ms,
            }

        except ImportError:
            logger.warning("Knowledge Mound not available, returning empty result")
            return {
                "items": [],
                "total_count": 0,
                "query": query,
                "error": "Knowledge Mound not available",
            }

        except Exception as e:
            logger.error(f"Memory read failed: {e}")
            return {"items": [], "total_count": 0, "query": query, "error": str(e)}

    def _interpolate_query(self, template: str, context: WorkflowContext) -> str:
        """Interpolate query template with context values."""
        query = template

        # Replace {input_name} with input values
        for key, value in context.inputs.items():
            query = query.replace(f"{{{key}}}", str(value))

        # Replace {step.step_id} with step outputs
        for step_id, output in context.step_outputs.items():
            if isinstance(output, str):
                query = query.replace(f"{{step.{step_id}}}", output)
            elif isinstance(output, dict) and "response" in output:
                query = query.replace(f"{{step.{step_id}}}", str(output["response"]))

        # Replace {state.key} with state values
        for key, value in context.state.items():
            query = query.replace(f"{{state.{key}}}", str(value))

        return query


class MemoryWriteStep(BaseStep):
    """
    Memory write step for storing knowledge in the Knowledge Mound.

    Config options:
        content: str - Content to store (can use {input} placeholders)
        source_type: str - Source type (fact, consensus, etc.)
        domain: str - Domain path (e.g., "legal/contracts")
        confidence: float - Confidence score (default: 0.5)
        importance: float - Importance score (default: 0.5)
        relationships: List[dict] - Relationships to create
            [{type: "supports", target: "km_123"}, ...]
        tenant_id: str - Tenant for multi-tenant isolation
        deduplicate: bool - Check for duplicates (default: True)
        metadata: dict - Additional metadata to store

    Usage:
        step = MemoryWriteStep(
            name="Store Contract Analysis",
            config={
                "content": "{analysis_result}",
                "source_type": "consensus",
                "domain": "legal/contracts",
                "confidence": 0.85,
                "relationships": [
                    {"type": "derived_from", "target": "{source_document_id}"}
                ],
            }
        )
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the memory write step."""
        config = {**self._config, **context.current_step_config}

        # Build content from template and context
        content_template = config.get("content", "")
        content = self._interpolate_content(content_template, context)

        if not content:
            logger.warning(f"Empty content for memory write step '{self.name}'")
            return {"success": False, "error": "Empty content"}

        try:
            from aragora.knowledge.mound import KnowledgeMound, IngestionRequest, KnowledgeSource

            tenant_id = config.get("tenant_id", context.metadata.get("tenant_id", "default"))

            # Parse source type
            source_type_str = config.get("source_type", "fact").upper()
            try:
                source_type = KnowledgeSource[source_type_str]
            except KeyError:
                source_type = KnowledgeSource.FACT

            # Build ingestion request
            request = IngestionRequest(
                content=content,
                workspace_id=tenant_id,
                source_type=source_type,
                confidence=config.get("confidence", 0.5),
                topics=[config.get("domain", "general")],
                metadata={
                    "workflow_id": context.workflow_id,
                    "step_id": context.current_step_id,
                    **config.get("metadata", {}),
                },
            )

            # Parse relationships
            relationships_config = config.get("relationships", [])
            for rel in relationships_config:
                rel_type = rel.get("type", "supports")
                target = self._interpolate_content(rel.get("target", ""), context)
                if target:
                    if rel_type == "supports":
                        request.supports.append(target)
                    elif rel_type == "contradicts":
                        request.contradicts.append(target)
                    elif rel_type == "derived_from":
                        request.derived_from.append(target)

            # Execute write
            mound = KnowledgeMound(workspace_id=tenant_id)
            await mound.initialize()

            result = await mound.store(request)  # type: ignore[arg-type]

            logger.info(f"Memory write '{self.name}': stored as {result.node_id}")

            return {
                "success": result.success,
                "node_id": result.node_id,
                "deduplicated": result.deduplicated,
                "existing_node_id": result.existing_node_id,
                "relationships_created": result.relationships_created,
            }

        except ImportError:
            logger.warning("Knowledge Mound not available, write skipped")
            return {"success": False, "error": "Knowledge Mound not available"}

        except Exception as e:
            logger.error(f"Memory write failed: {e}")
            return {"success": False, "error": str(e)}

    def _interpolate_content(self, template: str, context: WorkflowContext) -> str:
        """Interpolate content template with context values."""
        content = template

        # Replace {input_name} with input values
        for key, value in context.inputs.items():
            content = content.replace(f"{{{key}}}", str(value))

        # Replace {step.step_id} with step outputs
        for step_id, output in context.step_outputs.items():
            if isinstance(output, str):
                content = content.replace(f"{{step.{step_id}}}", output)
            elif isinstance(output, dict):
                # Try common output keys
                if "response" in output:
                    content = content.replace(f"{{step.{step_id}}}", str(output["response"]))
                elif "content" in output:
                    content = content.replace(f"{{step.{step_id}}}", str(output["content"]))
                elif "result" in output:
                    content = content.replace(f"{{step.{step_id}}}", str(output["result"]))

        # Replace {state.key} with state values
        for key, value in context.state.items():
            content = content.replace(f"{{state.{key}}}", str(value))

        return content
