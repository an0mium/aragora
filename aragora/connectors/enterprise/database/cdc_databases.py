"""
Database-specific CDC Handlers.

Provides concrete CDC handler implementations for PostgreSQL,
MySQL, and MongoDB, plus convenience factory functions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

from .cdc_models import (
    CDCConfig,
    CDCSourceType,
    ChangeEvent,
    ChangeOperation,
    ResumeToken,
    ResumeTokenStore,
)
from .cdc_handlers import (
    ChangeEventHandler,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base CDC Handler
# =============================================================================


class BaseCDCHandler(ABC):
    """
    Abstract base class for database-specific CDC handlers.

    Provides common functionality for:
    - Connection management with retry logic
    - Event filtering
    - Health monitoring
    - Graceful shutdown
    """

    def __init__(
        self,
        connector_id: str,
        source_type: CDCSourceType,
        config: CDCConfig | None = None,
        event_handler: ChangeEventHandler | None = None,
        token_store: ResumeTokenStore | None = None,
    ):
        self.connector_id = connector_id
        self.source_type = source_type
        self.config = config or CDCConfig()
        self.event_handler = event_handler
        self.token_store = token_store or ResumeTokenStore()

        self._running = False
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._events_processed = 0
        self._last_event_time: datetime | None = None
        self._errors: list[str] = []
        self._connected = False

    @property
    def is_running(self) -> bool:
        """Check if the CDC handler is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if connected to the database."""
        return self._connected

    @property
    def stats(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "connector_id": self.connector_id,
            "source_type": self.source_type.value,
            "running": self._running,
            "connected": self._connected,
            "events_processed": self._events_processed,
            "last_event_time": (
                self._last_event_time.isoformat() if self._last_event_time else None
            ),
            "errors": self._errors[-10:],  # Last 10 errors
        }

    def get_resume_token(self) -> str | None:
        """Get the last saved resume token."""
        token = self.token_store.get(self.connector_id)
        return token.token if token else None

    def _should_process_event(self, event: ChangeEvent) -> bool:
        """Check if an event should be processed based on filter config."""
        # Check table filters
        if self.config.include_tables:
            if event.table not in self.config.include_tables:
                return False

        if self.config.exclude_tables:
            if event.table in self.config.exclude_tables:
                return False

        # Check operation filter
        if self.config.include_operations:
            if event.operation not in self.config.include_operations:
                return False

        return True

    async def _process_event(self, event: ChangeEvent) -> bool:
        """
        Process a single change event through the handler chain.

        Returns True if successfully processed.
        """
        if not self._should_process_event(event):
            logger.debug(f"Skipping filtered event: {event.table}/{event.operation.value}")
            return True

        try:
            if self.event_handler:
                success = await self.event_handler.handle(event)
            else:
                # No handler configured - just log
                logger.info(f"CDC event: {event.operation.value} on {event.qualified_table}")
                success = True

            if success:
                self._events_processed += 1
                self._last_event_time = event.timestamp

                # Persist resume token if available
                if event.resume_token:
                    self.token_store.save(
                        ResumeToken(
                            connector_id=self.connector_id,
                            source_type=self.source_type,
                            token=event.resume_token,
                            timestamp=event.timestamp,
                            sequence_number=event.sequence_number,
                        )
                    )

            return success

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            error_msg = f"Failed to process event {event.id}: {e}"
            logger.error(error_msg)
            self._errors.append(error_msg)
            return False

    async def _retry_with_backoff(
        self,
        operation: Callable[[], Any],
        operation_name: str = "operation",
    ) -> Any:
        """Execute an operation with exponential backoff retry."""
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                return operation()

            except (OSError, ConnectionError, asyncio.TimeoutError) as e:
                if attempt == self.config.max_retries:
                    logger.error(f"{operation_name} failed after {attempt + 1} attempts: {e}")
                    raise

                logger.warning(
                    f"{operation_name} attempt {attempt + 1}/{self.config.max_retries + 1} "
                    f"failed: {e}. Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

                # Apply backoff
                delay = min(
                    delay * self.config.retry_backoff_multiplier,
                    self.config.max_retry_delay_seconds,
                )

    @abstractmethod
    async def _connect(self) -> None:
        """Establish connection to the database. Override in subclasses."""
        pass

    @abstractmethod
    async def _disconnect(self) -> None:
        """Close connection to the database. Override in subclasses."""
        pass

    @abstractmethod
    async def _listen_loop(self) -> None:
        """Main loop for listening to CDC events. Override in subclasses."""
        pass

    async def start(self) -> None:
        """Start the CDC handler."""
        if self._running:
            logger.warning(f"CDC handler {self.connector_id} is already running")
            return

        logger.info(f"Starting CDC handler: {self.connector_id}")
        self._running = True
        self._stop_event.clear()

        try:
            await self._retry_with_backoff(self._connect, "Connection")
            self._connected = True
            self._task = asyncio.create_task(self._run_loop())
        except (OSError, ConnectionError, RuntimeError) as e:
            self._running = False
            logger.error(f"Failed to start CDC handler {self.connector_id}: {e}")
            raise

    async def _run_loop(self) -> None:
        """Run the main listen loop with reconnection support."""
        while self._running and not self._stop_event.is_set():
            try:
                await self._listen_loop()
            except asyncio.CancelledError:
                logger.info(f"CDC handler {self.connector_id} cancelled")
                break
            except (OSError, ConnectionError) as e:
                if self._running:
                    logger.warning(f"CDC connection lost: {e}. Reconnecting...")
                    self._connected = False
                    try:
                        await self._retry_with_backoff(self._connect, "Reconnection")
                        self._connected = True
                    except (OSError, ConnectionError) as e2:
                        logger.error(f"Reconnection failed: {e2}")
                        break

    async def stop(self) -> None:
        """Stop the CDC handler gracefully."""
        if not self._running:
            return

        logger.info(f"Stopping CDC handler: {self.connector_id}")
        self._running = False
        self._stop_event.set()

        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.debug("stop encountered an error: %s", e)
            self._task = None

        await self._disconnect()
        self._connected = False


# =============================================================================
# PostgreSQL CDC Handler
# =============================================================================


class PostgresCDCHandler(BaseCDCHandler):
    """
    PostgreSQL CDC handler using LISTEN/NOTIFY.

    Listens to PostgreSQL notification channels for real-time
    change events. Requires:
    - asyncpg library
    - Database triggers to emit NOTIFY on changes

    Example trigger:
        CREATE OR REPLACE FUNCTION notify_changes()
        RETURNS TRIGGER AS $$
        BEGIN
            PERFORM pg_notify(
                'table_changes',
                json_build_object(
                    'operation', TG_OP,
                    'table', TG_TABLE_NAME,
                    'schema', TG_TABLE_SCHEMA,
                    'new_data', row_to_json(NEW),
                    'old_data', row_to_json(OLD)
                )::text
            );
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

    Usage:
        handler = PostgresCDCHandler(
            host="localhost",
            database="mydb",
            channels=["table_changes"],
            event_handler=KnowledgeMoundHandler(),
        )
        await handler.start()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        schema: str = "public",
        channels: list[str] | None = None,
        username: str | None = None,
        password: str | None = None,
        ssl: bool = False,
        **kwargs: Any,
    ):
        connector_id = kwargs.pop("connector_id", f"pg_cdc_{host}_{database}")
        super().__init__(
            connector_id=connector_id,
            source_type=CDCSourceType.POSTGRESQL,
            **kwargs,
        )

        self.host = host
        self.port = port
        self.database = database
        self.schema = schema
        self.channels = channels or ["table_changes"]
        self.username = username
        self.password = password
        self.ssl = ssl

        self._conn: Any = None
        self._notification_queue: asyncio.Queue = asyncio.Queue()

    async def _connect(self) -> None:
        """Establish asyncpg connection."""
        try:
            import asyncpg
        except ImportError:
            logger.warning("asyncpg not installed. PostgreSQL CDC requires: pip install asyncpg")
            raise ImportError("asyncpg is required for PostgreSQL CDC")

        # Build connection parameters
        conn_params: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
        }

        if self.username:
            conn_params["user"] = self.username
        if self.password:
            conn_params["password"] = self.password
        if self.ssl:
            conn_params["ssl"] = "require"

        self._conn = await asyncpg.connect(**conn_params)

        # Set up notification callback
        async def notification_handler(
            connection: Any, pid: int, channel: str, payload: str
        ) -> None:
            await self._notification_queue.put((channel, payload))

        # Subscribe to channels
        for channel in self.channels:
            await self._conn.add_listener(channel, notification_handler)
            logger.info(f"[PostgresCDC] Listening on channel: {channel}")

    async def _disconnect(self) -> None:
        """Close asyncpg connection."""
        if self._conn:
            for channel in self.channels:
                try:
                    await self._conn.remove_listener(channel, lambda *_: None)
                except (OSError, RuntimeError) as e:
                    logger.debug("disconnect encountered an error: %s", e)
            await self._conn.close()
            self._conn = None

    async def _listen_loop(self) -> None:
        """Process notifications from the queue."""
        while self._running and not self._stop_event.is_set():
            try:
                # Use timeout to allow checking stop event periodically
                channel, payload = await asyncio.wait_for(
                    self._notification_queue.get(),
                    timeout=1.0,
                )

                event = ChangeEvent.from_postgres_notify(
                    payload=payload,
                    channel=channel,
                    connector_id=self.connector_id,
                    database=self.database,
                    schema=self.schema,
                )

                await self._process_event(event)

            except asyncio.TimeoutError:
                # Check for stop event
                continue
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"[PostgresCDC] Failed to parse notification: {e}")

    async def execute_query(self, query: str, *args: Any) -> list[Any]:
        """Execute a query on the connection (for setup/maintenance)."""
        if not self._conn:
            raise RuntimeError("Not connected to PostgreSQL")
        return await self._conn.fetch(query, *args)


# =============================================================================
# MySQL CDC Handler
# =============================================================================


class MySQLCDCHandler(BaseCDCHandler):
    """
    MySQL CDC handler using binary log (binlog) replication.

    Connects to MySQL as a replication slave and receives
    row-level change events. Requires:
    - mysql-replication library
    - Binary logging enabled on MySQL server
    - Replication permissions for the user

    MySQL server configuration:
        [mysqld]
        server-id = 1
        log_bin = mysql-bin
        binlog_format = ROW
        binlog_row_image = FULL

    User permissions:
        GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'user'@'%';

    Usage:
        handler = MySQLCDCHandler(
            host="localhost",
            database="mydb",
            username="replicator",
            password="secret",
            event_handler=KnowledgeMoundHandler(),
        )
        await handler.start()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str | None = None,
        tables: list[str] | None = None,
        username: str | None = None,
        password: str | None = None,
        server_id: int = 100,
        blocking: bool = True,
        resume_stream: bool = True,
        **kwargs: Any,
    ):
        connector_id = kwargs.pop("connector_id", f"mysql_cdc_{host}_{database or 'all'}")
        super().__init__(
            connector_id=connector_id,
            source_type=CDCSourceType.MYSQL,
            **kwargs,
        )

        self.host = host
        self.port = port
        self.database = database
        self.tables = tables
        self.username = username
        self.password = password
        self.server_id = server_id
        self.blocking = blocking
        self.resume_stream = resume_stream

        self._stream: Any = None
        self._stream_thread: Any = None

    async def _connect(self) -> None:
        """Create binlog stream reader."""
        try:
            from pymysqlreplication import BinLogStreamReader
            from pymysqlreplication.row_event import (
                DeleteRowsEvent,
                UpdateRowsEvent,
                WriteRowsEvent,
            )
        except ImportError:
            logger.warning(
                "mysql-replication not installed. MySQL CDC requires: pip install mysql-replication"
            )
            raise ImportError("mysql-replication is required for MySQL CDC")

        mysql_settings = {
            "host": self.host,
            "port": self.port,
            "user": self.username or "root",
            "passwd": self.password or "",
        }

        # Get resume position if available
        log_file = None
        log_pos = None
        resume_token = self.get_resume_token()
        if resume_token and self.resume_stream:
            try:
                token_data = json.loads(resume_token)
                log_file = token_data.get("log_file")
                log_pos = token_data.get("log_pos")
                logger.info(f"[MySQLCDC] Resuming from {log_file}:{log_pos}")
            except json.JSONDecodeError as e:
                logger.debug("Failed to parse JSON data: %s", e)

        stream_kwargs: dict[str, Any] = {
            "connection_settings": mysql_settings,
            "server_id": self.server_id,
            "blocking": self.blocking,
            "only_events": [WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent],
            "resume_stream": self.resume_stream,
        }

        if self.database:
            stream_kwargs["only_schemas"] = [self.database]
        if self.tables:
            stream_kwargs["only_tables"] = self.tables
        if log_file:
            stream_kwargs["log_file"] = log_file
        if log_pos:
            stream_kwargs["log_pos"] = log_pos

        # Create stream in a thread-safe way
        self._stream = BinLogStreamReader(**stream_kwargs)
        logger.info(f"[MySQLCDC] Connected to binlog stream on {self.host}")

    async def _disconnect(self) -> None:
        """Close binlog stream."""
        if self._stream:
            self._stream.close()
            self._stream = None

    async def _listen_loop(self) -> None:
        """Process binlog events in an async-friendly way."""
        try:
            from pymysqlreplication.row_event import (
                DeleteRowsEvent,
                UpdateRowsEvent,
                WriteRowsEvent,
            )
        except ImportError:
            return

        loop = asyncio.get_running_loop()

        def get_next_event() -> Any:
            """Get next event from binlog stream (blocking)."""
            if self._stream is None:
                return None
            for binlog_event in self._stream:
                return binlog_event
            return None

        while self._running and not self._stop_event.is_set():
            try:
                # Run blocking binlog read in executor
                binlog_event = await asyncio.wait_for(
                    loop.run_in_executor(None, get_next_event),
                    timeout=1.0,
                )

                if binlog_event is None:
                    continue

                # Map event type to operation
                if isinstance(binlog_event, WriteRowsEvent):
                    operation = ChangeOperation.INSERT
                elif isinstance(binlog_event, UpdateRowsEvent):
                    operation = ChangeOperation.UPDATE
                elif isinstance(binlog_event, DeleteRowsEvent):
                    operation = ChangeOperation.DELETE
                else:
                    continue

                # Process each row in the event
                for row in binlog_event.rows:
                    if operation == ChangeOperation.UPDATE:
                        data = row.get("after_values", {})
                        old_data = row.get("before_values", {})
                        fields_changed = [k for k in data.keys() if data.get(k) != old_data.get(k)]
                    elif operation == ChangeOperation.DELETE:
                        data = None
                        old_data = row.get("values", {})
                        fields_changed = []
                    else:  # INSERT
                        data = row.get("values", {})
                        old_data = None
                        fields_changed = []

                    # Extract primary key
                    pk_value = None
                    if data:
                        pk_value = data.get("id") or data.get("_id")
                    elif old_data:
                        pk_value = old_data.get("id") or old_data.get("_id")

                    # Build resume token from binlog position
                    resume_token = json.dumps(
                        {
                            "log_file": self._stream.log_file if self._stream else None,
                            "log_pos": self._stream.log_pos if self._stream else None,
                        }
                    )

                    event = ChangeEvent(
                        id="",
                        source_type=CDCSourceType.MYSQL,
                        connector_id=self.connector_id,
                        operation=operation,
                        timestamp=datetime.now(timezone.utc),
                        database=binlog_event.schema,
                        table=binlog_event.table,
                        primary_key={"id": pk_value} if pk_value else None,
                        data=data,
                        old_data=old_data,
                        fields_changed=fields_changed,
                        resume_token=resume_token,
                    )

                    await self._process_event(event)

            except asyncio.TimeoutError:
                # Allow checking stop event
                continue
            except (OSError, ConnectionError) as e:
                logger.error(f"[MySQLCDC] Binlog error: {e}")
                raise


# =============================================================================
# MongoDB CDC Handler
# =============================================================================


class MongoDBCDCHandler(BaseCDCHandler):
    """
    MongoDB CDC handler using change streams.

    Watches MongoDB collections for changes using the
    change streams API. Requires:
    - motor (async MongoDB driver) library
    - MongoDB replica set or sharded cluster

    Usage:
        handler = MongoDBCDCHandler(
            host="localhost",
            database="mydb",
            collections=["users", "orders"],
            event_handler=KnowledgeMoundHandler(),
        )
        await handler.start()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "test",
        collections: list[str] | None = None,
        connection_string: str | None = None,
        username: str | None = None,
        password: str | None = None,
        watch_database: bool = True,
        full_document: str = "updateLookup",
        full_document_before_change: str | None = None,
        **kwargs: Any,
    ):
        connector_id = kwargs.pop("connector_id", f"mongo_cdc_{host}_{database}")
        super().__init__(
            connector_id=connector_id,
            source_type=CDCSourceType.MONGODB,
            **kwargs,
        )

        self.host = host
        self.port = port
        self.database_name = database
        self.collections = collections
        self.connection_string = connection_string
        self.username = username
        self.password = password
        self.watch_database = watch_database
        self.full_document = full_document
        self.full_document_before_change = full_document_before_change

        self._client: Any = None
        self._db: Any = None

    async def _connect(self) -> None:
        """Connect to MongoDB."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            logger.warning("motor not installed. MongoDB CDC requires: pip install motor")
            raise ImportError("motor is required for MongoDB CDC")

        # Build connection string
        if self.connection_string:
            conn_str = self.connection_string
        elif self.username and self.password:
            conn_str = (
                f"mongodb://{self.username}:{self.password}@"  # nosec
                f"{self.host}:{self.port}/{self.database_name}"
            )
        else:
            conn_str = f"mongodb://{self.host}:{self.port}/{self.database_name}"

        self._client = AsyncIOMotorClient(conn_str)
        self._db = self._client[self.database_name]

        # Verify connection
        await self._client.admin.command("ping")
        logger.info(f"[MongoDBCDC] Connected to {self.host}:{self.port}/{self.database_name}")

    async def _disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    async def _listen_loop(self) -> None:
        """Watch for changes using change streams."""
        # Build pipeline for filtering operations
        pipeline = [
            {"$match": {"operationType": {"$in": ["insert", "update", "replace", "delete"]}}}
        ]

        # Filter by collections if specified
        if self.collections and not self.watch_database:
            pipeline[0]["$match"]["ns.coll"] = {"$in": self.collections}

        # Build watch options
        watch_options: dict[str, Any] = {
            "pipeline": pipeline,
            "full_document": self.full_document,
        }

        # Add before-change image if supported (MongoDB 6.0+)
        if self.full_document_before_change:
            watch_options["full_document_before_change"] = self.full_document_before_change

        # Get resume token
        resume_token = self.get_resume_token()
        if resume_token:
            try:
                watch_options["resume_after"] = json.loads(resume_token)
                logger.info("[MongoDBCDC] Resuming from saved token")
            except json.JSONDecodeError:
                logger.warning("[MongoDBCDC] Invalid resume token, starting fresh")

        # Determine what to watch
        if self.collections and len(self.collections) == 1:
            # Watch a single collection
            target = self._db[self.collections[0]]
        elif self.watch_database:
            # Watch entire database
            target = self._db
        else:
            # Watch all collections explicitly listed
            target = self._db

        try:
            async with target.watch(**watch_options) as stream:
                logger.info("[MongoDBCDC] Change stream started")
                async for change in stream:
                    if self._stop_event.is_set():
                        break

                    # Filter by collection if watching database
                    ns = change.get("ns", {})
                    coll = ns.get("coll", "")
                    if self.collections and coll not in self.collections:
                        continue

                    event = ChangeEvent.from_mongodb_change(
                        change=change,
                        connector_id=self.connector_id,
                    )

                    await self._process_event(event)

        except asyncio.CancelledError:
            raise
        except (OSError, ConnectionError) as e:
            logger.error(f"[MongoDBCDC] Change stream error: {e}")
            raise


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_postgres_cdc(
    host: str = "localhost",
    port: int = 5432,
    database: str = "postgres",
    channels: list[str] | None = None,
    username: str | None = None,
    password: str | None = None,
    event_handler: ChangeEventHandler | None = None,
    config: CDCConfig | None = None,
) -> PostgresCDCHandler:
    """
    Create a PostgreSQL CDC handler.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        channels: NOTIFY channels to listen on
        username: Database username
        password: Database password
        event_handler: Handler for change events
        config: CDC configuration

    Returns:
        Configured PostgresCDCHandler
    """
    return PostgresCDCHandler(
        host=host,
        port=port,
        database=database,
        channels=channels,
        username=username,
        password=password,
        event_handler=event_handler,
        config=config,
    )


def create_mysql_cdc(
    host: str = "localhost",
    port: int = 3306,
    database: str | None = None,
    tables: list[str] | None = None,
    username: str | None = None,
    password: str | None = None,
    server_id: int = 100,
    event_handler: ChangeEventHandler | None = None,
    config: CDCConfig | None = None,
) -> MySQLCDCHandler:
    """
    Create a MySQL CDC handler.

    Args:
        host: MySQL host
        port: MySQL port
        database: Database to watch (None = all)
        tables: Tables to watch (None = all)
        username: Database username (needs REPLICATION privileges)
        password: Database password
        server_id: Unique server ID for replication
        event_handler: Handler for change events
        config: CDC configuration

    Returns:
        Configured MySQLCDCHandler
    """
    return MySQLCDCHandler(
        host=host,
        port=port,
        database=database,
        tables=tables,
        username=username,
        password=password,
        server_id=server_id,
        event_handler=event_handler,
        config=config,
    )


def create_mongodb_cdc(
    host: str = "localhost",
    port: int = 27017,
    database: str = "test",
    collections: list[str] | None = None,
    connection_string: str | None = None,
    username: str | None = None,
    password: str | None = None,
    event_handler: ChangeEventHandler | None = None,
    config: CDCConfig | None = None,
) -> MongoDBCDCHandler:
    """
    Create a MongoDB CDC handler.

    Args:
        host: MongoDB host
        port: MongoDB port
        database: Database to watch
        collections: Collections to watch (None = all)
        connection_string: Full connection string (overrides host/port)
        username: Database username
        password: Database password
        event_handler: Handler for change events
        config: CDC configuration

    Returns:
        Configured MongoDBCDCHandler
    """
    return MongoDBCDCHandler(
        host=host,
        port=port,
        database=database,
        collections=collections,
        connection_string=connection_string,
        username=username,
        password=password,
        event_handler=event_handler,
        config=config,
    )
