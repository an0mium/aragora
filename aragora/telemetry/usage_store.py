"""
Persistent analytics store for AI model usage telemetry.

Stores ModelUsageRecord entries in SQLite (dev) with indexes optimised for
the most common analytics queries: per-debate rollups, per-model breakdowns,
time-range scans, and cost attribution by workspace/agent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from aragora.storage.base_store import SQLiteStore
from aragora.telemetry.models import (
    COLUMNS,
    AggregationInterval,
    ModelUsageAggregate,
    ModelUsageRecord,
    SessionUsageSummary,
    UsageQuery,
)

logger = logging.getLogger(__name__)

# Valid GROUP BY dimensions (whitelist to prevent SQL injection)
_VALID_GROUP_DIMENSIONS = frozenset(
    {
        "provider",
        "model",
        "model_version",
        "agent_id",
        "agent_name",
        "workspace_id",
        "operation",
        "debate_id",
    }
)


class ModelUsageStore(SQLiteStore):
    """SQLite-backed store for AI model usage telemetry.

    Schema is optimised for:
    - Inserting individual usage records at high throughput
    - Querying per-debate session summaries
    - Aggregating by model, provider, agent, or workspace
    - Time-range scans for dashboards and billing
    """

    SCHEMA_NAME = "model_usage"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS model_usage (
            id              TEXT PRIMARY KEY,
            debate_id       TEXT,
            session_id      TEXT,
            workspace_id    TEXT NOT NULL DEFAULT '',
            agent_id        TEXT NOT NULL DEFAULT '',
            agent_name      TEXT NOT NULL DEFAULT '',
            provider        TEXT NOT NULL DEFAULT '',
            model           TEXT NOT NULL DEFAULT '',
            model_version   TEXT NOT NULL DEFAULT '',
            tokens_input    INTEGER NOT NULL DEFAULT 0,
            tokens_output   INTEGER NOT NULL DEFAULT 0,
            tokens_cached   INTEGER NOT NULL DEFAULT 0,
            tokens_total    INTEGER NOT NULL DEFAULT 0,
            latency_ms      REAL NOT NULL DEFAULT 0.0,
            time_to_first_token_ms REAL,
            operation       TEXT NOT NULL DEFAULT 'other',
            cost_usd        REAL NOT NULL DEFAULT 0.0,
            timestamp       TEXT NOT NULL,
            round_number    INTEGER,
            metadata_json   TEXT NOT NULL DEFAULT '{}'
        );

        -- Primary lookup: all records for a debate
        CREATE INDEX IF NOT EXISTS idx_mu_debate_id
            ON model_usage(debate_id);

        -- Session-level rollup
        CREATE INDEX IF NOT EXISTS idx_mu_session_id
            ON model_usage(session_id);

        -- Time-range scans (dashboards, billing windows)
        CREATE INDEX IF NOT EXISTS idx_mu_timestamp
            ON model_usage(timestamp);

        -- Cost attribution by workspace
        CREATE INDEX IF NOT EXISTS idx_mu_workspace_ts
            ON model_usage(workspace_id, timestamp);

        -- Per-model analytics
        CREATE INDEX IF NOT EXISTS idx_mu_provider_model
            ON model_usage(provider, model);

        -- Agent performance drilldown
        CREATE INDEX IF NOT EXISTS idx_mu_agent_id
            ON model_usage(agent_id);

        -- Composite for filtered time-range by debate
        CREATE INDEX IF NOT EXISTS idx_mu_debate_ts
            ON model_usage(debate_id, timestamp);
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, record: ModelUsageRecord) -> None:
        """Insert a single usage record."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_usage (
                    id, debate_id, session_id, workspace_id,
                    agent_id, agent_name, provider, model, model_version,
                    tokens_input, tokens_output, tokens_cached, tokens_total,
                    latency_ms, time_to_first_token_ms, operation, cost_usd,
                    timestamp, round_number, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.debate_id,
                    record.session_id,
                    record.workspace_id,
                    record.agent_id,
                    record.agent_name,
                    record.provider,
                    record.model,
                    record.model_version,
                    record.tokens_input,
                    record.tokens_output,
                    record.tokens_cached,
                    record.tokens_total,
                    record.latency_ms,
                    record.time_to_first_token_ms,
                    record.operation.value,
                    float(record.cost_usd),
                    record.timestamp.isoformat(),
                    record.round_number,
                    json.dumps(record.metadata),
                ),
            )

    def insert_batch(self, records: list[ModelUsageRecord]) -> int:
        """Insert multiple records in a single transaction.

        Returns:
            Number of records inserted.
        """
        if not records:
            return 0

        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO model_usage (
                    id, debate_id, session_id, workspace_id,
                    agent_id, agent_name, provider, model, model_version,
                    tokens_input, tokens_output, tokens_cached, tokens_total,
                    latency_ms, time_to_first_token_ms, operation, cost_usd,
                    timestamp, round_number, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.id,
                        r.debate_id,
                        r.session_id,
                        r.workspace_id,
                        r.agent_id,
                        r.agent_name,
                        r.provider,
                        r.model,
                        r.model_version,
                        r.tokens_input,
                        r.tokens_output,
                        r.tokens_cached,
                        r.tokens_total,
                        r.latency_ms,
                        r.time_to_first_token_ms,
                        r.operation.value,
                        float(r.cost_usd),
                        r.timestamp.isoformat(),
                        r.round_number,
                        json.dumps(r.metadata),
                    )
                    for r in records
                ],
            )
        return len(records)

    # ------------------------------------------------------------------
    # Read – single / filtered
    # ------------------------------------------------------------------

    def get_by_id(self, record_id: str) -> ModelUsageRecord | None:
        """Fetch a single record by ID."""
        cols = ", ".join(COLUMNS)
        row = self.fetch_one(
            f"SELECT {cols} FROM model_usage WHERE id = ?",  # noqa: S608
            (record_id,),
        )
        if row is None:
            return None
        return ModelUsageRecord.from_row(row)

    def query(self, q: UsageQuery) -> list[ModelUsageRecord]:
        """Return records matching the query filters."""
        where, params = q.to_where_clause()
        cols = ", ".join(COLUMNS)
        sql = (
            f"SELECT {cols} FROM model_usage"  # noqa: S608
            f" WHERE {where}"
            " ORDER BY timestamp DESC"
            " LIMIT ? OFFSET ?"
        )
        params.extend([q.limit, q.offset])
        rows = self.fetch_all(sql, tuple(params))
        return [ModelUsageRecord.from_row(r) for r in rows]

    def count_records(self, q: UsageQuery | None = None) -> int:
        """Count records matching the query, or all records."""
        if q is None:
            return self.count("model_usage")
        where, params = q.to_where_clause()
        return self.count("model_usage", where, tuple(params))

    # ------------------------------------------------------------------
    # Read – debate session summary
    # ------------------------------------------------------------------

    def get_session_summary(self, debate_id: str) -> SessionUsageSummary | None:
        """Compute an aggregated usage summary for a debate session."""
        row = self.fetch_one(
            """
            SELECT
                debate_id,
                MIN(session_id)            AS session_id,
                COUNT(*)                   AS total_requests,
                SUM(tokens_input)          AS total_tokens_input,
                SUM(tokens_output)         AS total_tokens_output,
                SUM(tokens_total)          AS total_tokens,
                SUM(cost_usd)              AS total_cost_usd,
                AVG(latency_ms)            AS avg_latency_ms,
                MAX(latency_ms)            AS max_latency_ms,
                MIN(latency_ms)            AS min_latency_ms,
                MIN(timestamp)             AS first_request_at,
                MAX(timestamp)             AS last_request_at
            FROM model_usage
            WHERE debate_id = ?
            GROUP BY debate_id
            """,
            (debate_id,),
        )
        if row is None:
            return None

        # Fetch distinct models and providers
        model_rows = self.fetch_all(
            "SELECT DISTINCT model FROM model_usage WHERE debate_id = ?",
            (debate_id,),
        )
        provider_rows = self.fetch_all(
            "SELECT DISTINCT provider FROM model_usage WHERE debate_id = ?",
            (debate_id,),
        )

        # Compute p95 latency
        latency_rows = self.fetch_all(
            "SELECT latency_ms FROM model_usage WHERE debate_id = ? ORDER BY latency_ms",
            (debate_id,),
        )
        p95 = _percentile([r[0] for r in latency_rows], 0.95)

        first_ts = _parse_ts(row[10])
        last_ts = _parse_ts(row[11])
        duration = (last_ts - first_ts).total_seconds() if first_ts and last_ts else 0.0

        return SessionUsageSummary(
            debate_id=row[0],
            session_id=row[1],
            total_requests=row[2],
            total_tokens_input=row[3],
            total_tokens_output=row[4],
            total_tokens=row[5],
            total_cost_usd=Decimal(str(row[6])),
            avg_latency_ms=row[7],
            p95_latency_ms=p95,
            max_latency_ms=row[8],
            min_latency_ms=row[9],
            models_used=[r[0] for r in model_rows],
            providers_used=[r[0] for r in provider_rows],
            duration_seconds=duration,
            first_request_at=first_ts,
            last_request_at=last_ts,
        )

    # ------------------------------------------------------------------
    # Read – aggregations
    # ------------------------------------------------------------------

    def aggregate_by(
        self,
        dimension: str,
        q: UsageQuery | None = None,
    ) -> list[ModelUsageAggregate]:
        """Aggregate usage records grouped by a dimension.

        Args:
            dimension: Column to group by (model, provider, agent_id, etc.).
            q: Optional query filters applied before aggregation.

        Returns:
            List of aggregated rows sorted by total_cost_usd descending.

        Raises:
            ValueError: If dimension is not in the allow-list.
        """
        if dimension not in _VALID_GROUP_DIMENSIONS:
            raise ValueError(
                f"Invalid dimension '{dimension}'. "
                f"Must be one of: {sorted(_VALID_GROUP_DIMENSIONS)}"
            )

        where = "1=1"
        params: list[Any] = []
        if q is not None:
            where, params = q.to_where_clause()

        sql = f"""
            SELECT
                '{dimension}'              AS group_key,
                {dimension}                AS group_value,
                COUNT(*)                   AS total_requests,
                SUM(tokens_input)          AS total_tokens_input,
                SUM(tokens_output)         AS total_tokens_output,
                SUM(tokens_total)          AS total_tokens,
                SUM(cost_usd)              AS total_cost_usd,
                AVG(latency_ms)            AS avg_latency_ms
            FROM model_usage
            WHERE {where}
            GROUP BY {dimension}
            ORDER BY total_cost_usd DESC
        """  # noqa: S608

        rows = self.fetch_all(sql, tuple(params))
        return [
            ModelUsageAggregate(
                group_key=r[0],
                group_value=r[1] or "",
                total_requests=r[2],
                total_tokens_input=r[3],
                total_tokens_output=r[4],
                total_tokens=r[5],
                total_cost_usd=Decimal(str(r[6])),
                avg_latency_ms=r[7],
            )
            for r in rows
        ]

    def get_time_series(
        self,
        interval: AggregationInterval = AggregationInterval.HOURLY,
        q: UsageQuery | None = None,
    ) -> list[dict[str, Any]]:
        """Return token/cost totals bucketed by time interval.

        Args:
            interval: Time bucket size.
            q: Optional query filters.

        Returns:
            List of dicts with keys: bucket, total_requests, total_tokens, total_cost_usd.
        """
        bucket_expr = _interval_to_strftime(interval)

        where = "1=1"
        params: list[Any] = []
        if q is not None:
            where, params = q.to_where_clause()

        sql = f"""
            SELECT
                {bucket_expr}              AS bucket,
                COUNT(*)                   AS total_requests,
                SUM(tokens_total)          AS total_tokens,
                SUM(cost_usd)              AS total_cost_usd,
                AVG(latency_ms)            AS avg_latency_ms
            FROM model_usage
            WHERE {where}
            GROUP BY bucket
            ORDER BY bucket
        """  # noqa: S608

        rows = self.fetch_all(sql, tuple(params))
        return [
            {
                "bucket": r[0],
                "total_requests": r[1],
                "total_tokens": r[2],
                "total_cost_usd": float(r[3]),
                "avg_latency_ms": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_before(self, before: datetime) -> int:
        """Delete records older than *before*. Returns count deleted."""
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM model_usage WHERE timestamp < ?",
                (before.isoformat(),),
            )
            return cursor.rowcount


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def _parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _interval_to_strftime(interval: AggregationInterval) -> str:
    """Map interval enum to a SQLite strftime expression."""
    mapping = {
        AggregationInterval.HOURLY: "strftime('%Y-%m-%dT%H:00', timestamp)",
        AggregationInterval.DAILY: "strftime('%Y-%m-%d', timestamp)",
        AggregationInterval.WEEKLY: "strftime('%Y-W%W', timestamp)",
        AggregationInterval.MONTHLY: "strftime('%Y-%m', timestamp)",
    }
    return mapping[interval]
