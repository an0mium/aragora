"""
MongoDB Enterprise Connector.

Features:
- Incremental sync using _id or custom timestamp fields
- Change streams for real-time updates
- Collection filtering with projection support
- Aggregation pipeline support for complex queries
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


class MongoDBConnector(EnterpriseConnector):
    """
    MongoDB connector for enterprise data sync.

    Supports:
    - Incremental sync using timestamp or _id fields
    - Real-time updates via change streams
    - Collection-level filtering
    - Projection support for field selection
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "test",
        collections: Optional[List[str]] = None,
        timestamp_field: str = "updated_at",
        content_fields: Optional[List[str]] = None,
        title_field: Optional[str] = None,
        use_change_streams: bool = False,
        connection_string: Optional[str] = None,
        **kwargs,
    ):
        connector_id = f"mongodb_{host}_{database}"
        super().__init__(connector_id=connector_id, **kwargs)

        self.host = host
        self.port = port
        self.database_name = database
        self.collections = collections or []
        self.timestamp_field = timestamp_field
        self.content_fields = content_fields
        self.title_field = title_field
        self.use_change_streams = use_change_streams
        self.connection_string = connection_string

        self._client = None
        self._db = None
        self._change_stream_task = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def name(self) -> str:
        return f"MongoDB ({self.database_name})"

    async def _get_client(self):
        """Get or create MongoDB client."""
        if self._client is not None:
            return self._client

        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            # Build connection string
            if self.connection_string:
                conn_str = self.connection_string
            else:
                username = await self.credentials.get_credential("MONGO_USER")
                password = await self.credentials.get_credential("MONGO_PASSWORD")

                if username and password:
                    conn_str = f"mongodb://{username}:{password}@{self.host}:{self.port}/{self.database_name}"
                else:
                    conn_str = f"mongodb://{self.host}:{self.port}/{self.database_name}"

            self._client = AsyncIOMotorClient(conn_str)
            self._db = self._client[self.database_name]
            return self._client

        except ImportError:
            logger.error("motor not installed. Run: pip install motor")
            raise

    async def _discover_collections(self) -> List[str]:
        """Discover collections in the database."""
        await self._get_client()
        if self._db is None:
            raise RuntimeError("Database not initialized")
        collections = await self._db.list_collection_names()
        # Filter out system collections
        return [c for c in collections if not c.startswith("system.")]

    def _document_to_content(self, doc: Dict[str, Any]) -> str:
        """Convert a document to text content for indexing."""
        if self.content_fields:
            filtered = {k: v for k, v in doc.items() if k in self.content_fields}
        else:
            # Exclude metadata fields
            filtered = {k: v for k, v in doc.items() if not k.startswith("_")}

        # Convert to readable format
        parts = []
        for key, value in filtered.items():
            if value is not None:
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, (dict, list)):
                    value = json.dumps(value, default=str, indent=2)
                elif hasattr(value, "__str__"):
                    value = str(value)
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    def _get_document_title(self, doc: Dict[str, Any], collection: str) -> str:
        """Extract title from document."""
        if self.title_field and doc.get(self.title_field):
            return str(doc[self.title_field])

        # Try common title fields
        for field in ["title", "name", "subject", "label", "description"]:
            if doc.get(field):
                return str(doc[field])[:100]

        # Fallback to collection and ID
        return f"{collection} #{str(doc.get('_id', 'unknown'))[:12]}"

    def _infer_domain(self, collection: str) -> str:
        """Infer domain from collection name."""
        collection_lower = collection.lower()

        if any(t in collection_lower for t in ["user", "account", "profile", "auth"]):
            return "operational/users"
        elif any(t in collection_lower for t in ["order", "invoice", "payment", "transaction"]):
            return "financial/transactions"
        elif any(t in collection_lower for t in ["product", "inventory", "catalog"]):
            return "operational/products"
        elif any(t in collection_lower for t in ["log", "audit", "event"]):
            return "operational/logs"
        elif any(t in collection_lower for t in ["config", "setting"]):
            return "technical/configuration"
        elif any(t in collection_lower for t in ["document", "file", "attachment"]):
            return "general/documents"
        elif any(t in collection_lower for t in ["message", "chat", "notification"]):
            return "operational/communications"

        return "general/database"

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield items to sync from MongoDB collections.

        Uses timestamp fields for incremental sync when available.
        """
        await self._get_client()
        if self._db is None:
            raise RuntimeError("Database not initialized")

        # Get collections to sync
        collections = self.collections or await self._discover_collections()
        state.items_total = len(collections)

        for collection_name in collections:
            try:
                collection = self._db[collection_name]

                # Build query filter
                query: Dict[str, Any] = {}

                if state.last_item_timestamp:
                    query[self.timestamp_field] = {"$gt": state.last_item_timestamp}
                elif state.cursor:
                    # Use cursor for pagination
                    try:
                        cursor_data = json.loads(state.cursor)
                        if cursor_data.get("collection") == collection_name:
                            from bson import ObjectId

                            query["_id"] = {"$gt": ObjectId(cursor_data["last_id"])}
                    except (json.JSONDecodeError, Exception) as e:
                        logger.debug(f"Failed to parse cursor, starting from beginning: {e}")

                # Sort by timestamp or _id
                sort_field = self.timestamp_field if state.last_item_timestamp else "_id"

                cursor = collection.find(query).sort(sort_field, 1).limit(batch_size)

                async for doc in cursor:
                    doc_id = str(doc.get("_id", ""))

                    # Generate content
                    content = self._document_to_content(doc)
                    title = self._get_document_title(doc, collection_name)

                    # Get timestamp
                    updated_at = datetime.now(timezone.utc)
                    if doc.get(self.timestamp_field):
                        ts_value = doc[self.timestamp_field]
                        if isinstance(ts_value, datetime):
                            updated_at = (
                                ts_value.replace(tzinfo=timezone.utc)
                                if ts_value.tzinfo is None
                                else ts_value
                            )

                    # Create sync item
                    item_id = f"mongo:{self.database_name}:{collection_name}:{hashlib.sha256(doc_id.encode()).hexdigest()[:12]}"

                    yield SyncItem(
                        id=item_id,
                        content=content[:100000],
                        source_type="database",
                        source_id=f"mongodb://{self.host}:{self.port}/{self.database_name}/{collection_name}/{doc_id}",
                        title=title,
                        url=f"mongodb://{self.host}/{self.database_name}/{collection_name}?_id={doc_id}",
                        updated_at=updated_at,
                        domain=self._infer_domain(collection_name),
                        confidence=0.85,
                        metadata={
                            "database": self.database_name,
                            "collection": collection_name,
                            "document_id": doc_id,
                            "fields": [k for k in doc.keys() if not k.startswith("_")],
                        },
                    )

                    # Update cursor
                    state.cursor = json.dumps(
                        {
                            "collection": collection_name,
                            "last_id": doc_id,
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to sync collection {collection_name}: {e}")
                state.errors.append(f"{collection_name}: {str(e)}")
                continue

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list:
        """
        Search across collections using text search or regex.

        Works best with collections that have text indexes.
        """
        await self._get_client()
        if self._db is None:
            raise RuntimeError("Database not initialized")
        results = []

        collections = self.collections or await self._discover_collections()

        for collection_name in collections[:5]:  # Limit to first 5 collections
            try:
                collection = self._db[collection_name]

                # Try text search first
                try:
                    cursor = (
                        collection.find(
                            {"$text": {"$search": query}}, {"score": {"$meta": "textScore"}}
                        )
                        .sort([("score", {"$meta": "textScore"})])
                        .limit(limit)
                    )

                    async for doc in cursor:
                        results.append(
                            {
                                "collection": collection_name,
                                "data": {k: v for k, v in doc.items() if k != "score"},
                                "score": doc.get("score", 0),
                            }
                        )
                    continue
                except Exception as e:
                    logger.debug(f"Text search not available, falling back to regex: {e}")

                # Fallback to regex search on string fields
                # Get a sample document to find string fields
                sample = await collection.find_one()
                if not sample:
                    continue

                string_fields = [
                    k for k, v in sample.items() if isinstance(v, str) and not k.startswith("_")
                ]

                if string_fields:
                    or_conditions = [
                        {field: {"$regex": query, "$options": "i"}} for field in string_fields[:3]
                    ]

                    cursor = collection.find({"$or": or_conditions}).limit(limit)
                    async for doc in cursor:
                        results.append(
                            {
                                "collection": collection_name,
                                "data": {
                                    k: (
                                        str(v)
                                        if not isinstance(v, (str, int, float, bool, type(None)))
                                        else v
                                    )
                                    for k, v in doc.items()
                                },
                                "score": 0.5,
                            }
                        )

            except Exception as e:
                logger.debug(f"Search failed on {collection_name}: {e}")
                continue

        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:limit]

    async def fetch(self, evidence_id: str):
        """Fetch a specific document by evidence ID."""
        if not evidence_id.startswith("mongo:"):
            return None

        parts = evidence_id.split(":")
        if len(parts) < 4:
            return None

        database, _collection_name, _doc_hash = parts[1], parts[2], parts[3]

        if database != self.database_name:
            return None

        # We can't reverse the hash, so this is limited
        logger.debug(f"[{self.name}] Fetch not implemented for hash-based IDs")
        return None

    async def start_change_stream(self):
        """Start change stream for real-time updates."""
        if not self.use_change_streams:
            return

        await self._get_client()

        async def change_stream_loop():
            try:
                pipeline = [{"$match": {"operationType": {"$in": ["insert", "update", "replace"]}}}]

                async with self._db.watch(pipeline) as stream:
                    logger.info(f"[{self.name}] Change stream started")
                    async for change in stream:
                        await self._handle_change(change)
            except Exception as e:
                logger.error(f"[{self.name}] Change stream error: {e}")

        self._change_stream_task = asyncio.create_task(change_stream_loop())

    async def _handle_change(self, change: Dict[str, Any]):
        """Handle a change stream event."""
        operation = change.get("operationType")
        collection = change.get("ns", {}).get("coll")
        change.get("documentKey", {})

        logger.info(f"[{self.name}] Change: {operation} on {collection}")

        # Trigger incremental sync
        asyncio.create_task(self.sync(max_items=10))

    async def stop_change_stream(self):
        """Stop the change stream."""
        if self._change_stream_task:
            self._change_stream_task.cancel()
            try:
                await self._change_stream_task
            except asyncio.CancelledError:
                pass
            self._change_stream_task = None

    async def close(self):
        """Close MongoDB client."""
        await self.stop_change_stream()
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle webhook for database changes."""
        collection = payload.get("collection")
        operation = payload.get("operation")

        if collection and operation:
            logger.info(f"[{self.name}] Webhook: {operation} on {collection}")
            asyncio.create_task(self.sync(max_items=10))
            return True

        return False

    async def aggregate(
        self,
        collection_name: str,
        pipeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run an aggregation pipeline on a collection.

        Useful for complex queries and analytics.
        """
        await self._get_client()
        if self._db is None:
            raise RuntimeError("Database not initialized")

        collection = self._db[collection_name]
        results = []

        cursor = collection.aggregate(pipeline)
        async for doc in cursor:
            # Convert ObjectId to string for serialization
            doc_dict = {}
            for k, v in doc.items():
                if hasattr(v, "__str__") and k == "_id":
                    doc_dict[k] = str(v)
                else:
                    doc_dict[k] = v
            results.append(doc_dict)

        return results
