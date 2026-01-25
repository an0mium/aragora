"""
Database Connectors for enterprise data sources.

Supports:
- PostgreSQL (with LISTEN/NOTIFY for real-time)
- MongoDB (with change streams)
- MySQL (with binlog CDC)
- SQL Server (with CDC and Change Tracking)
- Snowflake (with change tracking and time travel)

CDC (Change Data Capture):
- Unified ChangeEvent model for all databases
- Resume token persistence for reliable streaming
- Knowledge Mound integration for real-time updates
"""

from aragora.connectors.enterprise.database.postgres import PostgreSQLConnector
from aragora.connectors.enterprise.database.mongodb import MongoDBConnector
from aragora.connectors.enterprise.database.mysql import MySQLConnector
from aragora.connectors.enterprise.database.sqlserver import SQLServerConnector
from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector
from aragora.connectors.enterprise.database.cdc import (
    ChangeEvent,
    ChangeOperation,
    CDCSourceType,
    ResumeToken,
    ResumeTokenStore,
    ChangeEventHandler,
    KnowledgeMoundHandler,
    CallbackHandler,
    CompositeHandler,
    CDCStreamManager,
)

__all__ = [
    # Connectors
    "PostgreSQLConnector",
    "MongoDBConnector",
    "MySQLConnector",
    "SQLServerConnector",
    "SnowflakeConnector",
    # CDC
    "ChangeEvent",
    "ChangeOperation",
    "CDCSourceType",
    "ResumeToken",
    "ResumeTokenStore",
    "ChangeEventHandler",
    "KnowledgeMoundHandler",
    "CallbackHandler",
    "CompositeHandler",
    "CDCStreamManager",
]
