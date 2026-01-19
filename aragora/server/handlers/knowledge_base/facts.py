"""
Facts Operations Mixin for Knowledge Handler.

Provides fact CRUD operations:
- List facts with filtering
- Get specific fact
- Create/update/delete facts
- Add relations between facts
- Get contradictions
- Verify facts with agents
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.knowledge import (
    FactFilters,
    FactRelationType,
    ValidationStatus,
)

from ..base import (
    HandlerResult,
    error_response,
    get_bool_param,
    get_bounded_float_param,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
    ttl_cache,
)

if TYPE_CHECKING:
    from aragora.knowledge import DatasetQueryEngine, FactStore, SimpleQueryEngine

logger = logging.getLogger(__name__)

# Cache TTLs
CACHE_TTL_FACTS = 60  # 1 minute for fact listings


class FactsHandlerProtocol(Protocol):
    """Protocol for handlers that use FactsOperationsMixin."""

    def _get_fact_store(self) -> "FactStore": ...
    def _get_query_engine(self) -> "DatasetQueryEngine | SimpleQueryEngine": ...
    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...


class FactsOperationsMixin:
    """Mixin providing fact CRUD operations for KnowledgeHandler."""

    @ttl_cache(ttl_seconds=CACHE_TTL_FACTS, key_prefix="knowledge_facts", skip_first=True)
    @handle_errors("list facts")
    def _handle_list_facts(self: FactsHandlerProtocol, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/facts - List facts."""
        workspace_id = get_bounded_string_param(query_params, "workspace_id", None, max_length=100)
        topic = get_bounded_string_param(query_params, "topic", None, max_length=200)
        min_confidence = get_bounded_float_param(
            query_params, "min_confidence", 0.0, min_val=0.0, max_val=1.0
        )
        status = get_bounded_string_param(query_params, "status", None, max_length=50)
        include_superseded = get_bool_param(query_params, "include_superseded", False)
        limit = get_clamped_int_param(query_params, "limit", 50, min_val=1, max_val=200)
        offset = get_clamped_int_param(query_params, "offset", 0, min_val=0, max_val=10000)

        filters = FactFilters(
            workspace_id=workspace_id,
            topics=[topic] if topic else None,
            min_confidence=min_confidence,
            validation_status=ValidationStatus(status) if status else None,
            include_superseded=include_superseded,
            limit=limit,
            offset=offset,
        )

        store = self._get_fact_store()
        facts = store.list_facts(filters)

        return json_response(
            {
                "facts": [f.to_dict() for f in facts],
                "total": len(facts),
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("get fact")
    def _handle_get_fact(self: FactsHandlerProtocol, fact_id: str) -> HandlerResult:
        """Handle GET /api/knowledge/facts/:id - Get specific fact."""
        store = self._get_fact_store()
        fact = store.get_fact(fact_id)

        if not fact:
            return error_response(f"Fact not found: {fact_id}", 404)

        return json_response(fact.to_dict())

    @handle_errors("create fact")
    def _handle_create_fact(self: FactsHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/facts - Create new fact."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        statement = data.get("statement", "")
        if not statement:
            return error_response("Statement is required", 400)

        workspace_id = data.get("workspace_id", "default")

        store = self._get_fact_store()
        fact = store.add_fact(
            statement=statement,
            workspace_id=workspace_id,
            evidence_ids=data.get("evidence_ids", []),
            source_documents=data.get("source_documents", []),
            confidence=data.get("confidence", 0.5),
            topics=data.get("topics", []),
            metadata=data.get("metadata", {}),
        )

        return json_response(fact.to_dict(), status=201)

    @handle_errors("update fact")
    def _handle_update_fact(self: FactsHandlerProtocol, fact_id: str, handler: Any) -> HandlerResult:
        """Handle PUT /api/knowledge/facts/:id - Update fact."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        store = self._get_fact_store()

        kwargs = {}
        if "confidence" in data:
            kwargs["confidence"] = data["confidence"]
        if "validation_status" in data:
            kwargs["validation_status"] = ValidationStatus(data["validation_status"])
        if "evidence_ids" in data:
            kwargs["evidence_ids"] = data["evidence_ids"]
        if "topics" in data:
            kwargs["topics"] = data["topics"]
        if "metadata" in data:
            kwargs["metadata"] = data["metadata"]
        if "superseded_by" in data:
            kwargs["superseded_by"] = data["superseded_by"]

        updated = store.update_fact(fact_id, **kwargs)

        if not updated:
            return error_response(f"Fact not found: {fact_id}", 404)

        return json_response(updated.to_dict())

    @handle_errors("delete fact")
    def _handle_delete_fact(self: FactsHandlerProtocol, fact_id: str, handler: Any) -> HandlerResult:
        """Handle DELETE /api/knowledge/facts/:id - Delete fact."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        store = self._get_fact_store()
        deleted = store.delete_fact(fact_id)

        if not deleted:
            return error_response(f"Fact not found: {fact_id}", 404)

        return json_response({"deleted": True, "fact_id": fact_id})

    @handle_errors("verify fact")
    def _handle_verify_fact(self: FactsHandlerProtocol, fact_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/facts/:id/verify - Verify fact."""
        from aragora.knowledge import DatasetQueryEngine
        from aragora.server.http_utils import run_async as _run_async

        store = self._get_fact_store()
        fact = store.get_fact(fact_id)

        if not fact:
            return error_response(f"Fact not found: {fact_id}", 404)

        engine = self._get_query_engine()

        if not isinstance(engine, DatasetQueryEngine):
            store.update_fact(
                fact_id,
                metadata={
                    **fact.metadata,
                    "_pending_verification": True,
                    "_verification_queued_at": __import__("time").time(),
                },
            )
            return json_response({
                "fact_id": fact_id,
                "verified": None,
                "status": "queued",
                "message": "Agent verification not currently available. Fact queued for verification when capability becomes available.",
            })

        try:
            verified = _run_async(engine.verify_fact(fact_id))
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return error_response(f"Verification failed: {e}", 500)

        return json_response(verified.to_dict())

    @handle_errors("get contradictions")
    def _handle_get_contradictions(self: FactsHandlerProtocol, fact_id: str) -> HandlerResult:
        """Handle GET /api/knowledge/facts/:id/contradictions."""
        store = self._get_fact_store()

        fact = store.get_fact(fact_id)
        if not fact:
            return error_response(f"Fact not found: {fact_id}", 404)

        contradictions = store.get_contradictions(fact_id)

        return json_response(
            {
                "fact_id": fact_id,
                "contradictions": [c.to_dict() for c in contradictions],
                "count": len(contradictions),
            }
        )

    @handle_errors("get relations")
    def _handle_get_relations(self: FactsHandlerProtocol, fact_id: str, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/facts/:id/relations."""
        store = self._get_fact_store()

        fact = store.get_fact(fact_id)
        if not fact:
            return error_response(f"Fact not found: {fact_id}", 404)

        relation_type_str = get_bounded_string_param(query_params, "type", None, max_length=50)
        relation_type = FactRelationType(relation_type_str) if relation_type_str else None

        as_source = get_bool_param(query_params, "as_source", True)
        as_target = get_bool_param(query_params, "as_target", True)

        relations = store.get_relations(
            fact_id,
            relation_type=relation_type,
            as_source=as_source,
            as_target=as_target,
        )

        return json_response(
            {
                "fact_id": fact_id,
                "relations": [r.to_dict() for r in relations],
                "count": len(relations),
            }
        )

    @handle_errors("add relation")
    def _handle_add_relation(self: FactsHandlerProtocol, fact_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/facts/:id/relations - Add relation from fact."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        target_fact_id = data.get("target_fact_id")
        if not target_fact_id:
            return error_response("target_fact_id is required", 400)

        relation_type_str = data.get("relation_type")
        if not relation_type_str:
            return error_response("relation_type is required", 400)

        try:
            relation_type = FactRelationType(relation_type_str)
        except ValueError:
            return error_response(f"Invalid relation_type: {relation_type_str}", 400)

        store = self._get_fact_store()

        if not store.get_fact(fact_id):
            return error_response(f"Source fact not found: {fact_id}", 404)
        if not store.get_fact(target_fact_id):
            return error_response(f"Target fact not found: {target_fact_id}", 404)

        relation = store.add_relation(
            source_fact_id=fact_id,
            target_fact_id=target_fact_id,
            relation_type=relation_type,
            confidence=data.get("confidence", 0.5),
            created_by=data.get("created_by", ""),
            metadata=data.get("metadata"),
        )

        return json_response(relation.to_dict(), status=201)

    @handle_errors("add relation bulk")
    def _handle_add_relation_bulk(self: FactsHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/facts/relations - Add relation between facts."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        source_fact_id = data.get("source_fact_id")
        target_fact_id = data.get("target_fact_id")
        relation_type_str = data.get("relation_type")

        if not source_fact_id:
            return error_response("source_fact_id is required", 400)
        if not target_fact_id:
            return error_response("target_fact_id is required", 400)
        if not relation_type_str:
            return error_response("relation_type is required", 400)

        try:
            relation_type = FactRelationType(relation_type_str)
        except ValueError:
            return error_response(f"Invalid relation_type: {relation_type_str}", 400)

        store = self._get_fact_store()

        relation = store.add_relation(
            source_fact_id=source_fact_id,
            target_fact_id=target_fact_id,
            relation_type=relation_type,
            confidence=data.get("confidence", 0.5),
            created_by=data.get("created_by", ""),
            metadata=data.get("metadata"),
        )

        return json_response(relation.to_dict(), status=201)
