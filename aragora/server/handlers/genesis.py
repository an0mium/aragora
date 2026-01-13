"""
Genesis (evolution visibility) endpoint handlers.

Endpoints:
- GET /api/genesis/stats - Get overall genesis statistics
- GET /api/genesis/events - Get recent genesis events
- GET /api/genesis/lineage/:genome_id - Get genome ancestry
- GET /api/genesis/tree/:debate_id - Get debate tree structure
- GET /api/genesis/genomes - List all genomes
- GET /api/genesis/genomes/top - Get top genomes by fitness
- GET /api/genesis/genomes/:genome_id - Get single genome details
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from aragora.server.validation import validate_debate_id, validate_genome_id
from aragora.utils.optional_imports import try_import

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    json_response,
    safe_json_parse,
)
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for genesis endpoints (10 requests per minute - evolution ops are expensive)
_genesis_limiter = RateLimiter(requests_per_minute=10)

# Lazy imports for optional dependencies using centralized utility
_genesis_imports, GENESIS_AVAILABLE = try_import(
    "aragora.genesis.ledger", "GenesisLedger", "GenesisEventType"
)
GenesisLedger = _genesis_imports["GenesisLedger"]
GenesisEventType = _genesis_imports["GenesisEventType"]

_genome_imports, GENOME_AVAILABLE = try_import(
    "aragora.genesis.genome", "GenomeStore", "AgentGenome"
)
GenomeStore = _genome_imports["GenomeStore"]
AgentGenome = _genome_imports["AgentGenome"]

from aragora.server.error_utils import safe_error_message as _safe_error_message


class GenesisHandler(BaseHandler):
    """Handler for genesis (evolution visibility) endpoints."""

    ROUTES = [
        "/api/genesis/stats",
        "/api/genesis/events",
        "/api/genesis/genomes",
        "/api/genesis/genomes/top",
        "/api/genesis/population",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Dynamic routes
        if path.startswith("/api/genesis/lineage/"):
            return True
        if path.startswith("/api/genesis/tree/"):
            return True
        if path.startswith("/api/genesis/genomes/") and path != "/api/genesis/genomes/top":
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route genesis requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _genesis_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for genesis endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        nomic_dir = self.ctx.get("nomic_dir")

        if path == "/api/genesis/stats":
            return self._get_genesis_stats(nomic_dir)

        if path == "/api/genesis/events":
            limit = get_int_param(query_params, "limit", 20)
            limit = min(limit, 100)
            event_type = query_params.get("event_type")
            if isinstance(event_type, list):
                event_type = event_type[0] if event_type else None
            return self._get_genesis_events(nomic_dir, limit, event_type)

        if path == "/api/genesis/genomes":
            limit = get_int_param(query_params, "limit", 50)
            limit = min(limit, 200)
            offset = get_int_param(query_params, "offset", 0)
            return self._get_genomes(nomic_dir, limit, offset)

        if path == "/api/genesis/genomes/top":
            limit = get_int_param(query_params, "limit", 10)
            limit = min(limit, 50)
            return self._get_top_genomes(nomic_dir, limit)

        if path == "/api/genesis/population":
            return self._get_population(nomic_dir)

        if path.startswith("/api/genesis/genomes/"):
            # Block path traversal attempts
            if ".." in path:
                return error_response("Invalid genome ID", 400)
            genome_id = path.split("/")[-1]
            is_valid, err = validate_genome_id(genome_id)
            if not is_valid:
                return error_response(err, 400)
            return self._get_genome(nomic_dir, genome_id)

        if path.startswith("/api/genesis/lineage/"):
            # Block path traversal attempts
            if ".." in path:
                return error_response("Invalid genome ID", 400)
            genome_id = path.split("/")[-1]
            is_valid, err = validate_genome_id(genome_id)
            if not is_valid:
                return error_response(err, 400)
            return self._get_genome_lineage(nomic_dir, genome_id)

        if path.startswith("/api/genesis/tree/"):
            # Block path traversal attempts
            if ".." in path:
                return error_response("Invalid debate ID", 400)
            debate_id = path.split("/")[-1]
            is_valid, err = validate_debate_id(debate_id)
            if not is_valid:
                return error_response(err, 400)
            return self._get_debate_tree(nomic_dir, debate_id)

        return None

    def _get_genesis_stats(self, nomic_dir: Optional[Path]) -> HandlerResult:
        """Get overall genesis statistics for evolution visibility."""
        if not GENESIS_AVAILABLE:
            return error_response("Genesis module not available", 503)

        try:
            ledger_path = ".nomic/genesis.db"
            if nomic_dir:
                ledger_path = str(nomic_dir / "genesis.db")

            ledger = GenesisLedger(ledger_path)

            # Count events by type
            event_counts = {}
            for event_type in GenesisEventType:
                events = ledger.get_events_by_type(event_type)
                event_counts[event_type.value] = len(events)

            # Get recent births and deaths
            births = ledger.get_events_by_type(GenesisEventType.AGENT_BIRTH)
            deaths = ledger.get_events_by_type(GenesisEventType.AGENT_DEATH)

            # Get fitness updates for trend
            fitness_updates = ledger.get_events_by_type(GenesisEventType.FITNESS_UPDATE)
            avg_fitness_change = 0.0
            if fitness_updates:
                changes = [e.data.get("change", 0) for e in fitness_updates[-50:]]
                avg_fitness_change = sum(changes) / len(changes) if changes else 0.0

            return json_response(
                {
                    "event_counts": event_counts,
                    "total_events": sum(event_counts.values()),
                    "total_births": len(births),
                    "total_deaths": len(deaths),
                    "net_population_change": len(births) - len(deaths),
                    "avg_fitness_change_recent": round(avg_fitness_change, 4),
                    "integrity_verified": ledger.verify_integrity(),
                    "merkle_root": ledger.get_merkle_root()[:32] + "...",
                }
            )
        except Exception as e:
            return error_response(_safe_error_message(e, "genesis_stats"), 500)

    def _get_genesis_events(
        self, nomic_dir: Optional[Path], limit: int, event_type: Optional[str]
    ) -> HandlerResult:
        """Get recent genesis events."""
        if not GENESIS_AVAILABLE:
            return error_response("Genesis module not available", 503)

        try:
            ledger_path = ".nomic/genesis.db"
            if nomic_dir:
                ledger_path = str(nomic_dir / "genesis.db")

            # Filter by type if specified
            if event_type:
                try:
                    etype = GenesisEventType(event_type)
                    ledger = GenesisLedger(ledger_path)
                    events = ledger.get_events_by_type(etype)[-limit:]
                    return json_response(
                        {
                            "events": [e.to_dict() for e in events],
                            "count": len(events),
                            "filter": event_type,
                        }
                    )
                except ValueError:
                    return error_response(f"Unknown event type: {event_type}", 400)

            # Get all recent events
            ledger = GenesisLedger(ledger_path)
            with ledger.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT event_id, event_type, timestamp, parent_event_id, content_hash, data
                    FROM genesis_events
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                events = []
                for row in cursor.fetchall():
                    events.append(
                        {
                            "event_id": row[0],
                            "event_type": row[1],
                            "timestamp": row[2],
                            "parent_event_id": row[3],
                            "content_hash": row[4][:16] + "..." if row[4] else None,
                            "data": safe_json_parse(row[5], {}),
                        }
                    )

            return json_response(
                {
                    "events": events,
                    "count": len(events),
                }
            )
        except Exception as e:
            return error_response(_safe_error_message(e, "genesis_events"), 500)

    def _get_genome_lineage(self, nomic_dir: Optional[Path], genome_id: str) -> HandlerResult:
        """Get the lineage (ancestry) of a genome."""
        if not GENESIS_AVAILABLE:
            return error_response("Genesis module not available", 503)

        try:
            ledger_path = ".nomic/genesis.db"
            if nomic_dir:
                ledger_path = str(nomic_dir / "genesis.db")

            ledger = GenesisLedger(ledger_path)
            lineage = ledger.get_lineage(genome_id)

            if lineage:
                return json_response(
                    {
                        "genome_id": genome_id,
                        "lineage": lineage,
                        "generations": len(lineage),
                    }
                )
            else:
                return error_response(f"Genome not found: {genome_id}", 404)

        except Exception as e:
            return error_response(_safe_error_message(e, "genome_lineage"), 500)

    def _get_debate_tree(self, nomic_dir: Optional[Path], debate_id: str) -> HandlerResult:
        """Get the fractal tree structure for a debate."""
        if not GENESIS_AVAILABLE:
            return error_response("Genesis module not available", 503)

        try:
            ledger_path = ".nomic/genesis.db"
            if nomic_dir:
                ledger_path = str(nomic_dir / "genesis.db")

            ledger = GenesisLedger(ledger_path)
            tree = ledger.get_debate_tree(debate_id)

            return json_response(
                {
                    "debate_id": debate_id,
                    "tree": tree.to_dict(),
                    "total_nodes": len(tree.nodes),
                }
            )

        except Exception as e:
            return error_response(_safe_error_message(e, "debate_tree"), 500)

    def _get_genomes(self, nomic_dir: Optional[Path], limit: int, offset: int) -> HandlerResult:
        """Get all genomes with pagination."""
        if not GENOME_AVAILABLE:
            return error_response("Genesis genome module not available", 503)

        try:
            db_path = ".nomic/genesis.db"
            if nomic_dir:
                db_path = str(nomic_dir / "genesis.db")

            store = GenomeStore(db_path)
            all_genomes = store.get_all()

            # Apply pagination
            total = len(all_genomes)
            paginated = all_genomes[offset : offset + limit]

            return json_response(
                {
                    "genomes": [g.to_dict() for g in paginated],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )

        except Exception as e:
            return error_response(_safe_error_message(e, "genomes_list"), 500)

    def _get_top_genomes(self, nomic_dir: Optional[Path], limit: int) -> HandlerResult:
        """Get top genomes by fitness score."""
        if not GENOME_AVAILABLE:
            return error_response("Genesis genome module not available", 503)

        try:
            db_path = ".nomic/genesis.db"
            if nomic_dir:
                db_path = str(nomic_dir / "genesis.db")

            store = GenomeStore(db_path)
            top_genomes = store.get_top_by_fitness(limit)

            return json_response(
                {
                    "genomes": [g.to_dict() for g in top_genomes],
                    "count": len(top_genomes),
                }
            )

        except Exception as e:
            return error_response(_safe_error_message(e, "genomes_top"), 500)

    def _get_genome(self, nomic_dir: Optional[Path], genome_id: str) -> HandlerResult:
        """Get a single genome by ID."""
        if not GENOME_AVAILABLE:
            return error_response("Genesis genome module not available", 503)

        try:
            db_path = ".nomic/genesis.db"
            if nomic_dir:
                db_path = str(nomic_dir / "genesis.db")

            store = GenomeStore(db_path)
            genome = store.get(genome_id)

            if genome:
                return json_response(
                    {
                        "genome": genome.to_dict(),
                    }
                )
            else:
                return error_response(f"Genome not found: {genome_id}", 404)

        except Exception as e:
            return error_response(_safe_error_message(e, "genome_get"), 500)

    def _get_population(self, nomic_dir: Optional[Path]) -> HandlerResult:
        """Get the active population and its status.

        Returns:
            Population details including genomes, generation, and average fitness
        """
        try:
            db_path = ".nomic/genesis.db"
            if nomic_dir:
                db_path = str(nomic_dir / "genesis.db")

            # Try to import PopulationManager
            try:
                from aragora.genesis.breeding import PopulationManager
            except ImportError:
                return error_response("Genesis breeding module not available", 503)

            manager = PopulationManager(db_path=db_path)

            # Get or create population with common agent names
            population = manager.get_or_create_population(
                base_agents=["claude", "gemini", "codex", "grok"]
            )

            # Build response
            genomes_list: list[dict] = []
            result: dict = {
                "population_id": population.population_id,
                "generation": population.generation,
                "size": population.size,
                "average_fitness": population.average_fitness,
                "genomes": genomes_list,
                "best_genome": None,
                "debate_history_count": len(population.debate_history),
            }

            # Add genome summaries
            for genome in population.genomes:
                # Get top traits and expertise
                top_traits = (
                    genome.get_dominant_traits(3)
                    if hasattr(genome, "get_dominant_traits")
                    else list(genome.traits.keys())[:3]
                )
                top_expertise = list(genome.expertise.keys())[:3] if genome.expertise else []

                result["genomes"].append(
                    {
                        "genome_id": genome.genome_id,
                        "agent_name": genome.name,  # Use 'name' not 'agent_name'
                        "fitness_score": genome.fitness_score,
                        "generation": genome.generation,
                        "personality_traits": top_traits,
                        "expertise_domains": top_expertise,
                    }
                )

            # Add best genome
            best = population.best_genome
            if best:
                result["best_genome"] = {
                    "genome_id": best.genome_id,
                    "agent_name": best.name,  # Use 'name' not 'agent_name'
                    "fitness_score": best.fitness_score,
                }

            return json_response(result)

        except Exception as e:
            return error_response(_safe_error_message(e, "population"), 500)
