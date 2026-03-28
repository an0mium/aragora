"""
Analytics store for AI model usage telemetry.

SQLite-backed store optimised for:
  - High-throughput inserts of per-request ModelUsageRecords
  - Efficient debate-level aggregation for post-debate receipts
  - Time-range, provider, and model queries for dashboards
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from aragora.storage.base_store import SQLiteStore
from aragora.telemetry.models import (
    DebateUsageSummary,
    ModelUsageRecord,
    RequestStatus,
)

logger = logging.getLogger(__name__)


class TelemetryAnalyticsStore(SQLiteStore):
    """
    Persistent store for model-usage telemetry records.

    Schema is designed for efficient receipt generation:
      - Composite indexes on (debate_id, timestamp) and (workspace_id, timestamp)
      - Covering index on provider/model for cost-breakdown queries
      - Partial index on status for error-rate dashboards

    Usage::

        store = TelemetryAnalyticsStore("telemetry.db")
        store.record(usage_record)
        summary = store.summarise_debate("debate-123")
    """

    SCHEMA_NAME = "telemetry_analytics"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS model_usage (
            id               TEXT PRIMARY KEY,
            debate_id        TEXT NOT NULL DEFAULT '',
            workspace_id     TEXT NOT NULL DEFAULT '',
            agent_id         TEXT NOT NULL DEFAULT '',
            agent_name       TEXT NOT NULL DEFAULT '',
            round_number     INTEGER,
            provider         TEXT NOT NULL DEFAULT '',
            model            TEXT NOT NULL DEFAULT '',
            model_version    TEXT,
            input_tokens     INTEGER NOT NULL DEFAULT 0,
            output_tokens    INTEGER NOT NULL DEFAULT 0,
            cached_tokens    INTEGER NOT NULL DEFAULT 0,
            latency_ms       REAL    NOT NULL DEFAULT 0.0,
            cost_usd         TEXT    NOT NULL DEFAULT '0',
            status           TEXT    NOT NULL DEFAULT 'success',
            error_message    TEXT,
            timestamp        TEXT    NOT NULL,
            metadata         TEXT    NOT NULL DEFAULT '{}'
        );

        -- Receipt generation: aggregate all records for a debate
        CREATE INDEX IF NOT EXISTS idx_mu_debate_ts
            ON model_usage(debate_id, timestamp);

        -- Dashboard: workspace-scoped time-range queries
        CREATE INDEX IF NOT EXISTS idx_mu_workspace_ts
            ON model_usage(workspace_id, timestamp);

        -- Cost breakdown by provider/model
        CREATE INDEX IF NOT EXISTS idx_mu_provider_model
            ON model_usage(provider, model);

        -- Error-rate queries (partial index on non-success)
        CREATE INDEX IF NOT EXISTS idx_mu_errors
            ON model_usage(status) WHERE status != 'success';

        -- Agent-level attribution
        CREATE INDEX IF NOT EXISTS idx_mu_agent
            ON model_usage(agent_id, debate_id);
    """

    def __init__(self, db_path: str | Path = "telemetry_analytics.db", **kwargs: Any):
        super().__init__(db_path, **kwargs)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, usage: ModelUsageRecord) -> None:
        """Insert a single usage record."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_usage (
                    id, debate_id, workspace_id, agent_id, agent_name,
                    round_number, provider, model, model_version,
                    input_tokens, output_tokens, cached_tokens,
                    latency_ms, cost_usd, status, error_message,
                    timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    usage.id,
                    usage.debate_id,
                    usage.workspace_id,
                    usage.agent_id,
                    usage.agent_name,
                    usage.round_number,
                    usage.provider,
                    usage.model,
                    usage.model_version,
                    usage.tokens.input_tokens,
                    usage.tokens.output_tokens,
                    usage.tokens.cached_tokens,
                    usage.latency_ms,
                    str(usage.cost_usd),
                    usage.status.value,
                    usage.error_message,
                    usage.timestamp.isoformat(),
                    json.dumps(usage.metadata),
                ),
            )

    def record_batch(self, records: list[ModelUsageRecord]) -> int:
        """Insert many records in a single transaction. Returns count inserted."""
        if not records:
            return 0
        rows = [
            (
                r.id,
                r.debate_id,
                r.workspace_id,
                r.agent_id,
                r.agent_name,
                r.round_number,
                r.provider,
                r.model,
                r.model_version,
                r.tokens.input_tokens,
                r.tokens.output_tokens,
                r.tokens.cached_tokens,
                r.latency_ms,
                str(r.cost_usd),
                r.status.value,
                r.error_message,
                r.timestamp.isoformat(),
                json.dumps(r.metadata),
            )
            for r in records
        ]
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO model_usage (
                    id, debate_id, workspace_id, agent_id, agent_name,
                    round_number, provider, model, model_version,
                    input_tokens, output_tokens, cached_tokens,
                    latency_ms, cost_usd, status, error_message,
                    timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------
    # Read — single record
    # ------------------------------------------------------------------

    def get(self, record_id: str) -> ModelUsageRecord | None:
        """Fetch a single record by ID."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM model_usage WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query_by_debate(self, debate_id: str) -> list[ModelUsageRecord]:
        """All records for a debate, ordered by timestamp."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM model_usage WHERE debate_id = ? ORDER BY timestamp",
                (debate_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query_by_workspace(
        self,
        workspace_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[ModelUsageRecord]:
        """Records for a workspace within an optional time window."""
        clauses = ["workspace_id = ?"]
        params: list[Any] = [workspace_id]
        if since:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            clauses.append("timestamp <= ?")
            params.append(until.isoformat())
        params.append(limit)
        sql = (
            f"SELECT * FROM model_usage WHERE {' AND '.join(clauses)} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query_errors(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[ModelUsageRecord]:
        """Non-success records (uses partial index)."""
        clauses = ["status != 'success'"]
        params: list[Any] = []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        params.append(limit)
        sql = (
            f"SELECT * FROM model_usage WHERE {' AND '.join(clauses)} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Aggregation — receipt generation
    # ------------------------------------------------------------------

    def summarise_debate(self, debate_id: str) -> DebateUsageSummary:
        """
        Build an aggregated usage summary for a debate.

        This is the primary entry point for post-debate receipt generation:
        one SQL round-trip for the scalar aggregates, plus lightweight
        Python grouping for breakdowns.
        """
        records = self.query_by_debate(debate_id)
        if not records:
            return DebateUsageSummary(debate_id=debate_id)

        summary = DebateUsageSummary(
            debate_id=debate_id,
            workspace_id=records[0].workspace_id,
        )

        latencies: list[float] = []

        for r in records:
            summary.total_requests += 1
            if r.status == RequestStatus.SUCCESS:
                summary.successful_requests += 1
            else:
                summary.failed_requests += 1

            summary.total_input_tokens += r.tokens.input_tokens
            summary.total_output_tokens += r.tokens.output_tokens
            summary.total_cached_tokens += r.tokens.cached_tokens
            summary.total_cost_usd += r.cost_usd

            latencies.append(r.latency_ms)

            # Provider breakdown
            summary.cost_by_provider[r.provider] = (
                summary.cost_by_provider.get(r.provider, Decimal("0")) + r.cost_usd
            )

            # Model breakdown
            summary.cost_by_model[r.model] = (
                summary.cost_by_model.get(r.model, Decimal("0")) + r.cost_usd
            )

            # Agent token totals
            summary.tokens_by_agent[r.agent_id] = (
                summary.tokens_by_agent.get(r.agent_id, 0) + r.tokens.total
            )

            # Round breakdown
            if r.round_number is not None:
                summary.requests_by_round[r.round_number] = (
                    summary.requests_by_round.get(r.round_number, 0) + 1
                )

        # Latency stats
        if latencies:
            latencies.sort()
            summary.min_latency_ms = latencies[0]
            summary.max_latency_ms = latencies[-1]
            summary.avg_latency_ms = sum(latencies) / len(latencies)
            p95_idx = int(len(latencies) * 0.95)
            summary.p95_latency_ms = latencies[min(p95_idx, len(latencies) - 1)]

        # Time range
        summary.started_at = records[0].timestamp
        summary.ended_at = records[-1].timestamp

        return summary

    def cost_by_provider(
        self,
        workspace_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Decimal]:
        """Aggregate cost grouped by provider for a workspace."""
        clauses = ["workspace_id = ?"]
        params: list[Any] = [workspace_id]
        if since:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            clauses.append("timestamp <= ?")
            params.append(until.isoformat())
        sql = (
            f"SELECT provider, SUM(CAST(cost_usd AS REAL)) as total "
            f"FROM model_usage WHERE {' AND '.join(clauses)} "
            f"GROUP BY provider ORDER BY total DESC"
        )
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {row[0]: Decimal(str(row[1])) for row in rows if row[0]}

    def total_tokens(
        self,
        workspace_id: str,
        *,
        since: datetime | None = None,
    ) -> dict[str, int]:
        """Total input/output/cached tokens for a workspace."""
        clauses = ["workspace_id = ?"]
        params: list[Any] = [workspace_id]
        if since:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        sql = (
            f"SELECT "
            f"  COALESCE(SUM(input_tokens), 0), "
            f"  COALESCE(SUM(output_tokens), 0), "
            f"  COALESCE(SUM(cached_tokens), 0) "
            f"FROM model_usage WHERE {' AND '.join(clauses)}"
        )
        with self.connection() as conn:
            row = conn.execute(sql, params).fetchone()
        return {
            "input_tokens": int(row[0]),
            "output_tokens": int(row[1]),
            "cached_tokens": int(row[2]),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: Any) -> ModelUsageRecord:
        """Convert a sqlite3.Row or tuple to ModelUsageRecord."""
        # Support both Row (dict-like) and tuple
        if hasattr(row, "keys"):
            d = dict(row)
        else:
            cols = [
                "id",
                "debate_id",
                "workspace_id",
                "agent_id",
                "agent_name",
                "round_number",
                "provider",
                "model",
                "model_version",
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "latency_ms",
                "cost_usd",
                "status",
                "error_message",
                "timestamp",
                "metadata",
            ]
            d = dict(zip(cols, row))
        return ModelUsageRecord.from_dict(d)
