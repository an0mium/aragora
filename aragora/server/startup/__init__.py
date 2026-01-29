"""
Server startup initialization tasks.

This module handles the startup sequence for the unified server,
including monitoring, tracing, background tasks, and schedulers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Re-export all public symbols from submodules for backward compatibility
from aragora.server.startup.validation import (
    check_agent_credentials,
    check_connector_dependencies,
    check_production_requirements,
    validate_backend_connectivity,
    validate_database_connectivity,
    validate_redis_connectivity,
    validate_storage_backend,
)
from aragora.server.startup.redis import (
    init_redis_ha,
    init_redis_state_backend,
)
from aragora.server.startup.observability import (
    init_error_monitoring,
    init_opentelemetry,
    init_otlp_exporter,
    init_prometheus_metrics,
)
from aragora.server.startup.background import (
    init_background_tasks,
    init_circuit_breaker_persistence,
    init_pulse_scheduler,
    init_state_cleanup_task,
    init_stuck_debate_watchdog,
)
from aragora.server.startup.control_plane import (
    get_mayor_coordinator,
    get_witness_behavior,
    init_control_plane_coordinator,
    init_mayor_coordinator,
    init_persistent_task_queue,
    init_shared_control_plane_state,
    init_witness_patrol,
)
from aragora.server.startup.knowledge_mound import (
    get_km_config_from_env,
    init_km_adapters,
    init_knowledge_mound_from_env,
    init_tts_integration,
)
from aragora.server.startup.workers import (
    get_gauntlet_worker,
    init_backup_scheduler,
    init_durable_job_queue_recovery,
    init_gauntlet_run_recovery,
    init_gauntlet_worker,
    init_notification_worker,
    init_slo_webhooks,
    init_webhook_dispatcher,
    init_workflow_checkpoint_persistence,
)
from aragora.server.startup.security import (
    _get_degraded_status,
    init_access_review_scheduler,
    init_approval_gate_recovery,
    init_decision_router,
    init_deployment_validation,
    init_graphql_routes,
    init_key_rotation_scheduler,
    init_rbac_distributed_cache,
)
from aragora.server.startup.database import (  # noqa: F401
    close_postgres_pool,
    init_postgres_pool,
)

logger = logging.getLogger(__name__)


async def _log_pool_health(checkpoint: str) -> None:
    """Log pool health at a startup checkpoint (diagnostic, temporary)."""
    try:
        from aragora.storage.pool_manager import get_pool_info, get_shared_pool, is_pool_initialized

        if not is_pool_initialized():
            return
        info = get_pool_info()
        pool = get_shared_pool()
        msg = (
            f"[pool-check:{checkpoint}] "
            f"size={info.get('pool_size')}, free={info.get('free_connections')}"
        )
        # Quick health probe
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                msg += ", health=OK"
            except Exception as e:
                msg += f", health=FAIL({type(e).__name__}: {e})"
        logger.warning(msg)
    except Exception as e:
        logger.warning(f"[pool-check:{checkpoint}] error: {e}")


async def run_startup_sequence(
    nomic_dir: Path | None = None,
    stream_emitter: Any | None = None,
    graceful_degradation: bool = True,
) -> dict:
    """Run the full server startup sequence.

    Args:
        nomic_dir: Path to nomic directory
        stream_emitter: Optional event emitter for debates
        graceful_degradation: If True, enter degraded mode on failure instead of crashing.
            The server will start but return 503 for most endpoints until the issue is resolved.
            Defaults to True for production resilience.

    Returns:
        Dictionary with startup status for each component. If graceful_degradation is True
        and startup fails, returns a minimal status dict with degraded=True.

    Raises:
        RuntimeError: If production requirements are not met AND graceful_degradation is False
    """
    import os

    from aragora.control_plane.leader import is_distributed_state_required
    from aragora.server.degraded_mode import (
        set_degraded,
        DegradedErrorCode,
    )

    # Check production requirements first (fail fast or enter degraded mode)
    missing_requirements = check_production_requirements()
    if missing_requirements:
        for req in missing_requirements:
            logger.error(f"Missing production requirement: {req}")

        error_msg = f"Production requirements not met: {', '.join(missing_requirements)}"

        if graceful_degradation:
            # Determine the error code based on what's missing
            error_code = DegradedErrorCode.CONFIG_ERROR
            if any("ENCRYPTION_KEY" in r for r in missing_requirements):
                error_code = DegradedErrorCode.ENCRYPTION_KEY_MISSING
            elif any("REDIS" in r for r in missing_requirements):
                error_code = DegradedErrorCode.REDIS_UNAVAILABLE
            elif any("DATABASE" in r for r in missing_requirements):
                error_code = DegradedErrorCode.DATABASE_UNAVAILABLE

            set_degraded(
                reason=error_msg,
                error_code=error_code,
                details={"missing_requirements": missing_requirements},
            )
            return _get_degraded_status()

        raise RuntimeError(error_msg)

    # Validate actual backend connectivity (not just config presence)
    env = os.environ.get("ARAGORA_ENV", "development")
    is_production = env == "production"
    distributed_required = is_distributed_state_required()
    require_database = os.environ.get("ARAGORA_REQUIRE_DATABASE", "").lower() in (
        "true",
        "1",
        "yes",
    )

    connectivity = await validate_backend_connectivity(
        require_redis=distributed_required,
        require_database=require_database,
        timeout_seconds=10.0 if is_production else 5.0,
    )

    if not connectivity["valid"]:
        for error in connectivity["errors"]:
            logger.error(f"Backend connectivity failure: {error}")

        error_msg = f"Backend connectivity validation failed: {'; '.join(connectivity['errors'])}"

        if graceful_degradation:
            # Determine error code based on what failed
            error_code = DegradedErrorCode.BACKEND_CONNECTIVITY
            if any("Redis" in e for e in connectivity["errors"]):
                error_code = DegradedErrorCode.REDIS_UNAVAILABLE
            elif any("PostgreSQL" in e for e in connectivity["errors"]):
                error_code = DegradedErrorCode.DATABASE_UNAVAILABLE

            set_degraded(
                reason=error_msg,
                error_code=error_code,
                details={
                    "connectivity": connectivity,
                    "distributed_required": distributed_required,
                    "require_database": require_database,
                },
            )
            return _get_degraded_status()

        raise RuntimeError(error_msg)

    # Validate storage backend configuration
    storage_backend = validate_storage_backend()
    if not storage_backend["valid"]:
        for error in storage_backend["errors"]:
            logger.error(f"Storage backend error: {error}")

        error_msg = f"Storage backend validation failed: {'; '.join(storage_backend['errors'])}"

        if graceful_degradation:
            set_degraded(
                reason=error_msg,
                error_code=DegradedErrorCode.DATABASE_UNAVAILABLE,
                details={"storage_backend": storage_backend},
            )
            return _get_degraded_status()

        raise RuntimeError(error_msg)

    # Run auto-migrations if enabled
    migration_results: dict[str, Any] = {"skipped": True}
    if os.environ.get("ARAGORA_AUTO_MIGRATE_ON_STARTUP", "").lower() == "true":
        try:
            from aragora.server.auto_migrations import run_auto_migrations

            migration_results = await run_auto_migrations()
            if migration_results.get("success"):
                logger.info("Auto-migrations completed successfully")
            elif not migration_results.get("skipped"):
                logger.warning(f"Auto-migrations had issues: {migration_results}")
        except Exception as e:
            logger.error(f"Auto-migration failed: {e}")
            migration_results = {"error": str(e), "skipped": False}

    # Validate database schema (in consolidated mode)
    schema_validation = {"success": True, "errors": [], "warnings": []}
    try:
        from aragora.persistence.validator import validate_consolidated_schema

        schema_result = validate_consolidated_schema()
        schema_validation = {
            "success": schema_result.success,
            "errors": schema_result.errors,
            "warnings": schema_result.warnings,
        }

        if not schema_result.success:
            for error in schema_result.errors:
                logger.error(f"Database schema validation error: {error}")

            # Only fail if explicitly required (production environments may need migration)
            require_valid_schema = os.environ.get("ARAGORA_REQUIRE_VALID_SCHEMA", "").lower() in (
                "true",
                "1",
                "yes",
            )

            if require_valid_schema:
                error_msg = f"Database schema validation failed: {'; '.join(schema_result.errors)}"

                if graceful_degradation:
                    set_degraded(
                        reason=error_msg,
                        error_code=DegradedErrorCode.DATABASE_UNAVAILABLE,
                        details={"schema_validation": schema_validation},
                    )
                    return _get_degraded_status()

                raise RuntimeError(error_msg)
            else:
                logger.warning(
                    "Database schema validation failed but ARAGORA_REQUIRE_VALID_SCHEMA not set. "
                    "Server will start with potentially incomplete schema. "
                    "Run: python -m aragora.persistence.migrations.consolidate --migrate"
                )
        else:
            for warning in schema_result.warnings:
                logger.warning(f"Database schema: {warning}")

    except ImportError:
        logger.debug("Database validator not available - skipping schema validation")

    import time as time_mod

    status: dict[str, Any] = {
        "_startup_start_time": time_mod.time(),  # For duration calculation
        "backend_connectivity": connectivity,
        "storage_backend": storage_backend,
        "migrations": migration_results,
        "schema_validation": schema_validation,
        "redis_ha": {"enabled": False, "mode": "standalone", "healthy": False},
        "error_monitoring": False,
        "opentelemetry": False,
        "otlp_exporter": False,
        "prometheus": False,
        "circuit_breakers": 0,
        "background_tasks": False,
        "pulse_scheduler": False,
        "state_cleanup": False,
        "watchdog_task": None,
        "control_plane_coordinator": None,
        "km_adapters": False,
        "workflow_checkpoint_persistence": False,
        "shared_control_plane_state": False,
        "tts_integration": False,
        "persistent_task_queue": 0,
        "webhook_dispatcher": False,
        "slo_webhooks": False,
        "gauntlet_runs_recovered": 0,
        "durable_jobs_recovered": 0,
        "gauntlet_worker": False,
        "redis_state_backend": False,
        "key_rotation_scheduler": False,
        "access_review_scheduler": False,
        "rbac_distributed_cache": False,
        "notification_worker": False,
        "graphql": False,
        "backup_scheduler": False,
        "witness_patrol": False,
        "mayor_coordinator": False,
        "postgres_pool": {"enabled": False},
    }

    # Initialize PostgreSQL connection pool FIRST (event-loop bound)
    # This MUST happen before any subsystems that need database access
    status["postgres_pool"] = await init_postgres_pool()

    # --- Pool diagnostic: check health immediately after creation ---
    await _log_pool_health("after_init")

    # Initialize Redis HA early (other components may depend on it)
    status["redis_ha"] = await init_redis_ha()

    # Initialize in parallel where possible
    status["error_monitoring"] = await init_error_monitoring()
    status["opentelemetry"] = await init_opentelemetry()
    status["otlp_exporter"] = await init_otlp_exporter()
    status["prometheus"] = await init_prometheus_metrics()

    # Sequential initialization for components with dependencies
    status["circuit_breakers"] = init_circuit_breaker_persistence(nomic_dir)
    status["background_tasks"] = init_background_tasks(nomic_dir)
    status["pulse_scheduler"] = await init_pulse_scheduler(stream_emitter)
    status["state_cleanup"] = init_state_cleanup_task()
    status["watchdog_task"] = await init_stuck_debate_watchdog()
    status["control_plane_coordinator"] = await init_control_plane_coordinator()
    status["shared_control_plane_state"] = await init_shared_control_plane_state()

    await _log_pool_health("after_control_plane")

    # Initialize Witness Patrol for Gas Town agent monitoring
    status["witness_patrol"] = await init_witness_patrol()

    # Initialize Mayor Coordinator for distributed leadership
    status["mayor_coordinator"] = await init_mayor_coordinator()

    status["km_adapters"] = await init_km_adapters()
    status["workflow_checkpoint_persistence"] = init_workflow_checkpoint_persistence()
    status["tts_integration"] = await init_tts_integration()
    status["persistent_task_queue"] = await init_persistent_task_queue()

    await _log_pool_health("after_task_queue")

    # Initialize webhooks (dispatcher must be initialized before SLO webhooks)
    status["webhook_dispatcher"] = init_webhook_dispatcher()
    status["slo_webhooks"] = init_slo_webhooks()

    # Recover stale gauntlet runs from previous session
    status["gauntlet_runs_recovered"] = init_gauntlet_run_recovery()

    await _log_pool_health("after_gauntlet_recovery")

    # Recover and re-enqueue interrupted jobs from durable queue (if enabled)
    status["durable_jobs_recovered"] = await init_durable_job_queue_recovery()

    await _log_pool_health("after_job_recovery")

    # Start gauntlet worker for durable queue processing (if enabled)
    status["gauntlet_worker"] = await init_gauntlet_worker()

    await _log_pool_health("after_gauntlet_worker")

    # Start backup scheduler for automated backups and DR drills (if enabled)
    status["backup_scheduler"] = await init_backup_scheduler()

    # Start notification dispatcher worker for queue processing
    status["notification_worker"] = await init_notification_worker()

    # Initialize Redis state backend for horizontal scaling
    status["redis_state_backend"] = await init_redis_state_backend()

    # Initialize DecisionRouter with platform response handlers
    status["decision_router"] = await init_decision_router()

    # Initialize key rotation scheduler for automated encryption key management
    status["key_rotation_scheduler"] = await init_key_rotation_scheduler()

    # Initialize access review scheduler for SOC 2 CC6.1/CC6.2 compliance
    status["access_review_scheduler"] = await init_access_review_scheduler()

    # Initialize RBAC distributed cache for horizontal scaling
    status["rbac_distributed_cache"] = await init_rbac_distributed_cache()

    # Recover pending approval requests from governance store
    status["approval_gate_recovery"] = await init_approval_gate_recovery()

    # Initialize GraphQL API routes (if enabled)
    status["graphql"] = init_graphql_routes(None)

    # Run comprehensive deployment validation and log results
    status["deployment_validation"] = await init_deployment_validation()

    # Record startup completion time and store report
    import time as time_mod

    startup_end_time = time_mod.time()
    startup_duration = startup_end_time - (status.get("_startup_start_time", startup_end_time))

    try:
        from aragora.server.startup_transaction import (
            StartupReport,
            set_last_startup_report,
        )

        # Generate report from status
        components_initialized = [k for k, v in status.items() if v and not k.startswith("_")]
        components_failed = [k for k, v in status.items() if v is False]

        report = StartupReport(
            success=len(components_failed) == 0,
            total_duration_seconds=startup_duration,
            slo_seconds=30.0,
            slo_met=startup_duration <= 30.0,
            components_initialized=len(components_initialized),
            components_failed=components_failed,
            checkpoints=[],
            error=None,
        )
        set_last_startup_report(report)

        if startup_duration > 30.0:
            logger.warning(
                f"[STARTUP] Completed in {startup_duration:.2f}s (exceeds 30s SLO target)"
            )
        else:
            logger.info(
                f"[STARTUP] Completed in {startup_duration:.2f}s "
                f"({len(components_initialized)} components)"
            )

        status["startup_report"] = report.to_dict()

    except ImportError:
        logger.debug("startup_transaction module not available - skipping report")
        status["startup_duration_seconds"] = round(startup_duration, 2)

    return status

__all__ = [
    "check_connector_dependencies",
    "check_agent_credentials",
    "check_production_requirements",
    "validate_redis_connectivity",
    "validate_database_connectivity",
    "validate_backend_connectivity",
    "validate_storage_backend",
    "init_redis_ha",
    "init_error_monitoring",
    "init_opentelemetry",
    "init_otlp_exporter",
    "init_prometheus_metrics",
    "init_circuit_breaker_persistence",
    "init_background_tasks",
    "init_pulse_scheduler",
    "init_state_cleanup_task",
    "init_stuck_debate_watchdog",
    "init_control_plane_coordinator",
    "init_shared_control_plane_state",
    "init_tts_integration",
    "init_persistent_task_queue",
    "init_km_adapters",
    "init_workflow_checkpoint_persistence",
    "init_backup_scheduler",
    "init_webhook_dispatcher",
    "init_slo_webhooks",
    "init_gauntlet_run_recovery",
    "init_durable_job_queue_recovery",
    "init_gauntlet_worker",
    "get_gauntlet_worker",
    "init_redis_state_backend",
    "init_decision_router",
    "init_key_rotation_scheduler",
    "init_access_review_scheduler",
    "init_rbac_distributed_cache",
    "init_approval_gate_recovery",
    "init_notification_worker",
    "init_graphql_routes",
    "init_deployment_validation",
    "init_witness_patrol",
    "get_witness_behavior",
    "init_mayor_coordinator",
    "get_mayor_coordinator",
    "run_startup_sequence",
    "get_km_config_from_env",
    "init_knowledge_mound_from_env",
]
