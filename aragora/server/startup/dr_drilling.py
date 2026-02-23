"""DR drill scheduler startup integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.scheduler.dr_drill_scheduler import DRDrillScheduler

logger = logging.getLogger(__name__)

# Module-level reference for shutdown coordination
_dr_drill_scheduler: DRDrillScheduler | None = None


def get_dr_drill_scheduler() -> DRDrillScheduler | None:
    """Get the running DR drill scheduler instance (if any).

    Used by the shutdown sequence to gracefully stop the scheduler
    before closing database connections.
    """
    return _dr_drill_scheduler


async def start_dr_drilling() -> DRDrillScheduler | None:
    """Initialize and start automated DR drilling if enabled.

    Starts the DR drill scheduler that runs scheduled disaster recovery
    drills (monthly backup restoration, quarterly failover, weekly integrity)
    for SOC 2 CC9 compliance.

    Environment Variables:
        ARAGORA_DR_DRILL_ENABLED: Set to "true" to enable (default: false)
        ARAGORA_DR_DRILL_MONTHLY_DAY: Day of month for monthly drills (default: 15)
        ARAGORA_DR_DRILL_TARGET_RTO_SECONDS: Target RTO in seconds (default: 3600)
        ARAGORA_DR_DRILL_TARGET_RPO_SECONDS: Target RPO in seconds (default: 300)
        ARAGORA_DR_DRILL_STORAGE_PATH: Path for drill result storage (default: in-memory)
        ARAGORA_DR_DRILL_DRY_RUN: Set to "true" for dry-run mode (default: false)

    Returns:
        The scheduler instance or None if disabled.
    """
    global _dr_drill_scheduler
    import os

    if os.environ.get("ARAGORA_DR_DRILL_ENABLED", "false").lower() not in ("true", "1", "yes"):
        logger.debug("DR drilling disabled (set ARAGORA_DR_DRILL_ENABLED=true to enable)")
        return None

    try:
        from aragora.scheduler.dr_drill_scheduler import DRDrillConfig, DRDrillScheduler

        config = DRDrillConfig(
            monthly_drill_day=int(os.environ.get("ARAGORA_DR_DRILL_MONTHLY_DAY", "15")),
            quarterly_drill_months=[3, 6, 9, 12],
            annual_drill_month=1,
            target_rto_seconds=float(os.environ.get("ARAGORA_DR_DRILL_TARGET_RTO_SECONDS", "3600")),
            target_rpo_seconds=float(os.environ.get("ARAGORA_DR_DRILL_TARGET_RPO_SECONDS", "300")),
            storage_path=os.environ.get("ARAGORA_DR_DRILL_STORAGE_PATH"),
            dry_run=os.environ.get("ARAGORA_DR_DRILL_DRY_RUN", "false").lower()
            in ("true", "1", "yes"),
        )

        scheduler = DRDrillScheduler(config)
        await scheduler.start()

        # Store reference for shutdown coordination
        _dr_drill_scheduler = scheduler

        logger.info(
            "DR drill scheduler started (monthly_day=%s, target_rto=%ss, target_rpo=%ss, dry_run=%s)", config.monthly_drill_day, config.target_rto_seconds, config.target_rpo_seconds, config.dry_run
        )
        return scheduler

    except ImportError as e:
        logger.debug("DR drill scheduler not available: %s", e)
    except (RuntimeError, OSError, ValueError, TypeError) as e:
        logger.warning("Failed to start DR drill scheduler: %s", e)

    return None


async def stop_dr_drilling() -> None:
    """Stop the DR drill scheduler if running.

    Called during server shutdown to gracefully stop the scheduler.
    """
    global _dr_drill_scheduler

    if _dr_drill_scheduler is not None:
        try:
            await _dr_drill_scheduler.stop()
            logger.info("DR drill scheduler stopped")
        except (RuntimeError, OSError) as e:
            logger.warning("Error stopping DR drill scheduler: %s", e)
        finally:
            _dr_drill_scheduler = None


__all__ = [
    "get_dr_drill_scheduler",
    "start_dr_drilling",
    "stop_dr_drilling",
]
