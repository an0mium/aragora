"""
Supabase client for aragora persistence.

Handles all database operations and real-time subscriptions.
"""

import os
import asyncio
from datetime import datetime
from typing import Optional, Callable
import logging

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

from aragora.persistence.models import (
    NomicCycle,
    DebateArtifact,
    StreamEvent,
    AgentMetrics,
)

logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    Client for Supabase persistence.

    Usage:
        client = SupabaseClient()
        await client.save_cycle(cycle)
        await client.save_event(event)

        # Real-time subscriptions
        client.subscribe_to_events(loop_id, callback)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """
        Initialize Supabase client.

        Args:
            url: Supabase project URL (or SUPABASE_URL env var)
            key: Supabase service role key (or SUPABASE_KEY env var)
        """
        if not SUPABASE_AVAILABLE:
            logger.warning("supabase-py not installed. Run: pip install supabase")
            self.client = None
            return

        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            logger.warning(
                "Supabase credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
            self.client = None
            return

        self.client: Client = create_client(self.url, self.key)
        logger.info(f"Supabase client initialized for {self.url}")

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        return self.client is not None

    # -------------------------------------------------------------------------
    # Nomic Cycles
    # -------------------------------------------------------------------------

    async def save_cycle(self, cycle: NomicCycle) -> Optional[str]:
        """Save or update a nomic cycle."""
        if not self.is_configured:
            return None

        try:
            data = cycle.to_dict()
            if cycle.id:
                # Update existing
                result = self.client.table("nomic_cycles").update(data).eq("id", cycle.id).execute()
            else:
                # Insert new
                result = self.client.table("nomic_cycles").insert(data).execute()

            if result.data:
                return result.data[0].get("id")
            return None
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to save cycle: {e}")
            return None

    async def get_cycle(self, loop_id: str, cycle_number: int) -> Optional[NomicCycle]:
        """Get a specific cycle."""
        if not self.is_configured:
            return None

        try:
            result = self.client.table("nomic_cycles")\
                .select("*")\
                .eq("loop_id", loop_id)\
                .eq("cycle_number", cycle_number)\
                .single()\
                .execute()

            if result.data:
                return self._dict_to_cycle(result.data)
            return None
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to get cycle: {e}")
            return None

    async def list_cycles(
        self,
        loop_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[NomicCycle]:
        """List cycles, optionally filtered by loop_id."""
        if not self.is_configured:
            return []

        try:
            query = self.client.table("nomic_cycles")\
                .select("*")\
                .order("started_at", desc=True)\
                .limit(limit)\
                .offset(offset)

            if loop_id:
                query = query.eq("loop_id", loop_id)

            result = query.execute()
            return [self._dict_to_cycle(d) for d in result.data]
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to list cycles: {e}")
            return []

    def _dict_to_cycle(self, data: dict) -> NomicCycle:
        """Convert database row to NomicCycle."""
        # Safe datetime parsing with fallback
        try:
            started_at = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            started_at = datetime.now()

        try:
            completed_at = datetime.fromisoformat(data["completed_at"].replace("Z", "+00:00")) if data.get("completed_at") else None
        except (ValueError, TypeError):
            completed_at = None

        return NomicCycle(
            id=data.get("id"),
            loop_id=data["loop_id"],
            cycle_number=data["cycle_number"],
            phase=data["phase"],
            stage=data.get("stage", ""),
            started_at=started_at,
            completed_at=completed_at,
            success=data.get("success"),
            git_commit=data.get("git_commit"),
            task_description=data.get("task_description"),
            total_tasks=data.get("total_tasks", 0),
            completed_tasks=data.get("completed_tasks", 0),
            error_message=data.get("error_message"),
        )

    # -------------------------------------------------------------------------
    # Debate Artifacts
    # -------------------------------------------------------------------------

    async def save_debate(self, debate: DebateArtifact) -> Optional[str]:
        """Save a debate artifact."""
        if not self.is_configured:
            return None

        try:
            data = debate.to_dict()
            result = self.client.table("debate_artifacts").insert(data).execute()

            if result.data:
                return result.data[0].get("id")
            return None
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to save debate: {e}")
            return None

    async def get_debate(self, debate_id: str) -> Optional[DebateArtifact]:
        """Get a specific debate by ID."""
        if not self.is_configured:
            return None

        try:
            result = self.client.table("debate_artifacts")\
                .select("*")\
                .eq("id", debate_id)\
                .single()\
                .execute()

            if result.data:
                return self._dict_to_debate(result.data)
            return None
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to get debate: {e}")
            return None

    async def list_debates(
        self,
        loop_id: Optional[str] = None,
        phase: Optional[str] = None,
        limit: int = 50,
    ) -> list[DebateArtifact]:
        """List debate artifacts."""
        if not self.is_configured:
            return []

        try:
            query = self.client.table("debate_artifacts")\
                .select("*")\
                .order("created_at", desc=True)\
                .limit(limit)

            if loop_id:
                query = query.eq("loop_id", loop_id)
            if phase:
                query = query.eq("phase", phase)

            result = query.execute()
            return [self._dict_to_debate(d) for d in result.data]
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to list debates: {e}")
            return []

    def _dict_to_debate(self, data: dict) -> DebateArtifact:
        """Convert database row to DebateArtifact."""
        return DebateArtifact(
            id=data.get("id"),
            loop_id=data["loop_id"],
            cycle_number=data["cycle_number"],
            phase=data["phase"],
            task=data["task"],
            agents=data["agents"],
            transcript=data["transcript"],
            consensus_reached=data["consensus_reached"],
            confidence=data["confidence"],
            winning_proposal=data.get("winning_proposal"),
            vote_tally=data.get("vote_tally"),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        )

    # -------------------------------------------------------------------------
    # Stream Events
    # -------------------------------------------------------------------------

    async def save_event(self, event: StreamEvent) -> Optional[str]:
        """Save a stream event."""
        if not self.is_configured:
            return None

        try:
            data = event.to_dict()
            result = self.client.table("stream_events").insert(data).execute()

            if result.data:
                return result.data[0].get("id")
            return None
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to save event: {e}")
            return None

    async def save_events_batch(self, events: list[StreamEvent]) -> int:
        """Save multiple events in a batch."""
        if not self.is_configured or not events:
            return 0

        try:
            data = [e.to_dict() for e in events]
            result = self.client.table("stream_events").insert(data).execute()
            return len(result.data) if result.data else 0
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to save events batch: {e}")
            return 0

    async def get_events(
        self,
        loop_id: str,
        cycle: Optional[int] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[StreamEvent]:
        """Get events with optional filters."""
        if not self.is_configured:
            return []

        try:
            query = self.client.table("stream_events")\
                .select("*")\
                .eq("loop_id", loop_id)\
                .order("timestamp", desc=False)\
                .limit(limit)

            if cycle is not None:
                query = query.eq("cycle", cycle)
            if event_type:
                query = query.eq("event_type", event_type)
            if since:
                query = query.gte("timestamp", since.isoformat())

            result = query.execute()
            return [self._dict_to_event(d) for d in result.data]
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to get events: {e}")
            return []

    def _dict_to_event(self, data: dict) -> StreamEvent:
        """Convert database row to StreamEvent."""
        return StreamEvent(
            id=data.get("id"),
            loop_id=data["loop_id"],
            cycle=data["cycle"],
            event_type=data["event_type"],
            event_data=data["event_data"],
            agent=data.get("agent"),
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
        )

    # -------------------------------------------------------------------------
    # Agent Metrics
    # -------------------------------------------------------------------------

    async def save_metrics(self, metrics: AgentMetrics) -> Optional[str]:
        """Save agent metrics."""
        if not self.is_configured:
            return None

        try:
            data = metrics.to_dict()
            result = self.client.table("agent_metrics").insert(data).execute()

            if result.data:
                return result.data[0].get("id")
            return None
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to save metrics: {e}")
            return None

    async def get_agent_stats(
        self,
        agent_name: str,
        limit: int = 100,
    ) -> list[AgentMetrics]:
        """Get historical metrics for an agent."""
        if not self.is_configured:
            return []

        try:
            result = self.client.table("agent_metrics")\
                .select("*")\
                .eq("agent_name", agent_name)\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .execute()

            return [self._dict_to_metrics(d) for d in result.data]
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to get agent stats: {e}")
            return []

    def _dict_to_metrics(self, data: dict) -> AgentMetrics:
        """Convert database row to AgentMetrics."""
        return AgentMetrics(
            id=data.get("id"),
            loop_id=data["loop_id"],
            cycle=data["cycle"],
            agent_name=data["agent_name"],
            model=data["model"],
            phase=data["phase"],
            messages_sent=data.get("messages_sent", 0),
            proposals_made=data.get("proposals_made", 0),
            critiques_given=data.get("critiques_given", 0),
            votes_won=data.get("votes_won", 0),
            votes_received=data.get("votes_received", 0),
            consensus_contributions=data.get("consensus_contributions", 0),
            avg_response_time_ms=data.get("avg_response_time_ms"),
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
        )

    # -------------------------------------------------------------------------
    # Real-time Subscriptions
    # -------------------------------------------------------------------------

    def subscribe_to_events(
        self,
        loop_id: str,
        callback: Callable[[StreamEvent], None],
    ):
        """
        Subscribe to real-time events for a loop.

        Note: Requires Supabase Realtime to be enabled on the table.
        """
        if not self.is_configured:
            logger.warning("Supabase not configured, skipping subscription")
            return None

        try:
            channel = self.client.channel(f"events:{loop_id}")

            def handle_insert(payload):
                event = self._dict_to_event(payload["new"])
                callback(event)

            channel.on_postgres_changes(
                event="INSERT",
                schema="public",
                table="stream_events",
                filter=f"loop_id=eq.{loop_id}",
                callback=handle_insert,
            ).subscribe()

            logger.info(f"Subscribed to events for loop {loop_id}")
            return channel
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to subscribe to events: {e}")
            return None

    # -------------------------------------------------------------------------
    # Analytics Queries
    # -------------------------------------------------------------------------

    async def get_loop_summary(self, loop_id: str) -> dict:
        """Get summary statistics for a loop."""
        if not self.is_configured:
            return {}

        try:
            # Get cycles
            cycles = await self.list_cycles(loop_id=loop_id, limit=1000)

            # Get debates
            debates = await self.list_debates(loop_id=loop_id, limit=1000)

            # Calculate stats
            successful_cycles = sum(1 for c in cycles if c.success)
            total_debates = len(debates)
            consensus_rate = (
                sum(1 for d in debates if d.consensus_reached) / total_debates
                if total_debates > 0 else 0
            )

            return {
                "loop_id": loop_id,
                "total_cycles": len(cycles),
                "successful_cycles": successful_cycles,
                "success_rate": successful_cycles / len(cycles) if cycles else 0,
                "total_debates": total_debates,
                "consensus_rate": consensus_rate,
                "phases_completed": {
                    phase: sum(1 for c in cycles if c.phase == phase and c.success)
                    for phase in ["debate", "design", "implement", "verify", "commit"]
                },
            }
        except (TypeError, ValueError, KeyError, OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to get loop summary: {e}")
            return {}
