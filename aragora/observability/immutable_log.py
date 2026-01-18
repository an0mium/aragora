"""
Immutable Audit Logging System.

Provides append-only, tamper-evident audit trails for compliance requirements.
Supports multiple backends: local file, S3 Object Lock, AWS QLDB.

Features:
- Hash chain verification (blockchain-style)
- Cryptographic tamper detection
- Multiple storage backends
- Query and export capabilities
- Daily hash anchors for external verification

Usage:
    from aragora.observability.immutable_log import (
        ImmutableAuditLog,
        get_audit_log,
        AuditEntry,
        AuditBackend,
    )

    # Get the audit log
    log = get_audit_log()

    # Append an entry
    entry = await log.append(
        event_type="finding_created",
        actor="user@example.com",
        resource_type="finding",
        resource_id="f-123",
        action="create",
        details={"severity": "high"},
    )

    # Verify integrity
    is_valid, errors = await log.verify_integrity()

    # Query entries
    entries = await log.query(
        start_time=datetime.now() - timedelta(days=7),
        event_types=["finding_created", "finding_updated"],
    )

    # Export for compliance
    await log.export_range(start, end, format="json", path="/tmp/audit.json")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Re-export types for backward compatibility
from aragora.observability.log_types import (
    AuditBackend,
    AuditEntry,
    DailyAnchor,
    VerificationResult,
)

# Re-export backends for backward compatibility
from aragora.observability.log_backends import (
    AuditLogBackend,
    LocalFileBackend,
    S3ObjectLockBackend,
)

logger = logging.getLogger(__name__)

# Re-export all types at module level for backward compatibility
__all__ = [
    "AuditBackend",
    "AuditEntry",
    "AuditLogBackend",
    "DailyAnchor",
    "ImmutableAuditLog",
    "LocalFileBackend",
    "S3ObjectLockBackend",
    "VerificationResult",
    "audit_data_exported",
    "audit_document_accessed",
    "audit_document_uploaded",
    "audit_finding_created",
    "audit_finding_updated",
    "audit_session_started",
    "get_audit_log",
    "init_audit_log",
]


class ImmutableAuditLog:
    """
    Main interface for immutable audit logging.

    Provides append-only, tamper-evident audit trails with:
    - Hash chain verification
    - Multiple storage backends
    - Daily anchors for external verification
    - Query and export capabilities
    """

    # Genesis hash for the first entry
    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        backend: AuditLogBackend,
        signing_key: Optional[bytes] = None,
    ):
        self.backend = backend
        self.signing_key = signing_key
        self._lock = asyncio.Lock()
        self._last_entry: Optional[AuditEntry] = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Load last entry for hash chain continuity."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self._last_entry = await self.backend.get_last_entry()
            self._initialized = True

    def _compute_merkle_root(self, hashes: list[str]) -> str:
        """Compute Merkle root from list of hashes."""
        if not hashes:
            return self.GENESIS_HASH

        # Pad to power of 2
        while len(hashes) & (len(hashes) - 1) != 0:
            hashes.append(hashes[-1])

        # Build tree
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = next_level

        return hashes[0]

    def _sign_entry(self, entry: AuditEntry) -> Optional[str]:
        """Sign entry hash with HMAC if signing key is configured."""
        if not self.signing_key:
            return None

        import hmac

        signature = hmac.new(
            self.signing_key,
            entry.entry_hash.encode(),
            hashlib.sha256,
        ).hexdigest()

        return signature

    async def append(
        self,
        event_type: str,
        actor: str,
        resource_type: str,
        resource_id: str,
        action: str,
        actor_type: str = "user",
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEntry:
        """
        Append a new entry to the immutable log.

        Args:
            event_type: Type of event (e.g., "finding_created")
            actor: User ID or system identifier
            resource_type: Type of resource (e.g., "finding", "document")
            resource_id: ID of the affected resource
            action: Action performed (e.g., "create", "update")
            actor_type: Type of actor ("user", "system", "agent")
            details: Additional event details
            correlation_id: Request correlation ID
            workspace_id: Workspace ID
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            The created audit entry
        """
        await self._ensure_initialized()

        async with self._lock:
            # Determine sequence and previous hash
            if self._last_entry:
                sequence = self._last_entry.sequence_number + 1
                previous_hash = self._last_entry.entry_hash
            else:
                sequence = 1
                previous_hash = self.GENESIS_HASH

            # Create entry
            entry = AuditEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                sequence_number=sequence,
                previous_hash=previous_hash,
                entry_hash="",  # Computed below
                event_type=event_type,
                actor=actor,
                actor_type=actor_type,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                details=details or {},
                correlation_id=correlation_id,
                workspace_id=workspace_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            # Compute and set hash
            entry.entry_hash = entry.compute_hash()

            # Sign if configured
            entry.signature = self._sign_entry(entry)

            # Persist
            await self.backend.append(entry)
            self._last_entry = entry

            logger.debug(
                f"Audit entry appended: seq={sequence} type={event_type} "
                f"resource={resource_type}/{resource_id}"
            )

            return entry

    async def verify_integrity(
        self,
        start_sequence: Optional[int] = None,
        end_sequence: Optional[int] = None,
    ) -> VerificationResult:
        """
        Verify the integrity of the hash chain.

        Args:
            start_sequence: Start of range to verify (default: 1)
            end_sequence: End of range (default: last entry)

        Returns:
            Verification result with any errors found
        """
        await self._ensure_initialized()

        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []
        first_error_seq: Optional[int] = None

        # Determine range
        if start_sequence is None:
            start_sequence = 1
        if end_sequence is None:
            if self._last_entry:
                end_sequence = self._last_entry.sequence_number
            else:
                return VerificationResult(
                    is_valid=True,
                    entries_checked=0,
                    errors=[],
                    warnings=["No entries to verify"],
                    verification_time_ms=0,
                )

        # Verify entries
        entries = await self.backend.get_entries_range(start_sequence, end_sequence)

        if not entries:
            return VerificationResult(
                is_valid=True,
                entries_checked=0,
                errors=[],
                warnings=["No entries in range"],
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Get previous entry for chain verification
        expected_prev_hash = self.GENESIS_HASH
        if start_sequence > 1:
            prev_entry = await self.backend.get_by_sequence(start_sequence - 1)
            if prev_entry:
                expected_prev_hash = prev_entry.entry_hash
            else:
                warnings.append(f"Could not find entry {start_sequence - 1} for chain verification")

        # Verify each entry
        for entry in entries:
            # Verify hash chain
            if entry.previous_hash != expected_prev_hash:
                error = (
                    f"Chain broken at seq={entry.sequence_number}: "
                    f"expected prev_hash={expected_prev_hash[:16]}..., "
                    f"got={entry.previous_hash[:16]}..."
                )
                errors.append(error)
                if first_error_seq is None:
                    first_error_seq = entry.sequence_number

            # Verify entry hash
            computed_hash = entry.compute_hash()
            if entry.entry_hash != computed_hash:
                error = (
                    f"Hash mismatch at seq={entry.sequence_number}: "
                    f"stored={entry.entry_hash[:16]}..., "
                    f"computed={computed_hash[:16]}..."
                )
                errors.append(error)
                if first_error_seq is None:
                    first_error_seq = entry.sequence_number

            # Verify signature if present
            if entry.signature and self.signing_key:
                expected_sig = self._sign_entry(entry)
                if entry.signature != expected_sig:
                    error = f"Invalid signature at seq={entry.sequence_number}"
                    errors.append(error)
                    if first_error_seq is None:
                        first_error_seq = entry.sequence_number

            expected_prev_hash = entry.entry_hash

        elapsed_ms = (time.time() - start_time) * 1000

        return VerificationResult(
            is_valid=len(errors) == 0,
            entries_checked=len(entries),
            errors=errors,
            warnings=warnings,
            first_error_sequence=first_error_seq,
            verification_time_ms=elapsed_ms,
        )

    async def create_daily_anchor(self, date: Optional[str] = None) -> Optional[DailyAnchor]:
        """
        Create a daily anchor for external verification.

        Args:
            date: Date in YYYY-MM-DD format (default: yesterday)

        Returns:
            The created anchor, or None if no entries for date
        """
        if date is None:
            date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        # Parse date
        anchor_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_time = anchor_date
        end_time = anchor_date + timedelta(days=1)

        # Get entries for the day
        entries = await self.backend.query(
            start_time=start_time,
            end_time=end_time,
            limit=100000,  # Large limit for full day
        )

        if not entries:
            return None

        # Compute Merkle root
        entry_hashes = [e.entry_hash for e in entries]
        merkle_root = self._compute_merkle_root(entry_hashes)

        # Create anchor
        anchor = DailyAnchor(
            date=date,
            first_sequence=entries[0].sequence_number,
            last_sequence=entries[-1].sequence_number,
            entry_count=len(entries),
            merkle_root=merkle_root,
            chain_hash=entries[-1].entry_hash,
            created_at=datetime.now(timezone.utc),
        )

        await self.backend.save_anchor(anchor)
        logger.info(f"Created daily anchor for {date}: {len(entries)} entries")

        return anchor

    async def verify_anchor(self, date: str) -> VerificationResult:
        """
        Verify entries against a daily anchor.

        Args:
            date: Date to verify

        Returns:
            Verification result
        """
        anchor = await self.backend.get_anchor(date)
        if not anchor:
            return VerificationResult(
                is_valid=False,
                entries_checked=0,
                errors=[f"No anchor found for {date}"],
                warnings=[],
            )

        # Get entries for the anchor range
        entries = await self.backend.get_entries_range(
            anchor.first_sequence,
            anchor.last_sequence,
        )

        errors = []
        warnings = []

        # Verify entry count
        if len(entries) != anchor.entry_count:
            errors.append(
                f"Entry count mismatch: expected {anchor.entry_count}, got {len(entries)}"
            )

        # Verify Merkle root
        entry_hashes = [e.entry_hash for e in entries]
        computed_root = self._compute_merkle_root(entry_hashes)
        if computed_root != anchor.merkle_root:
            errors.append(
                f"Merkle root mismatch: expected {anchor.merkle_root[:16]}..., "
                f"got {computed_root[:16]}..."
            )

        # Verify chain hash
        if entries and entries[-1].entry_hash != anchor.chain_hash:
            errors.append(
                f"Chain hash mismatch: expected {anchor.chain_hash[:16]}..., "
                f"got {entries[-1].entry_hash[:16]}..."
            )

        # Also verify the chain integrity
        chain_result = await self.verify_integrity(
            anchor.first_sequence,
            anchor.last_sequence,
        )

        errors.extend(chain_result.errors)
        warnings.extend(chain_result.warnings)

        return VerificationResult(
            is_valid=len(errors) == 0,
            entries_checked=len(entries),
            errors=errors,
            warnings=warnings,
            verification_time_ms=chain_result.verification_time_ms,
        )

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
        actors: Optional[list[str]] = None,
        resource_types: Optional[list[str]] = None,
        resource_ids: Optional[list[str]] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """
        Query audit entries with filters.

        Args:
            start_time: Start of time range
            end_time: End of time range
            event_types: Filter by event types
            actors: Filter by actors
            resource_types: Filter by resource types
            resource_ids: Filter by resource IDs
            workspace_id: Filter by workspace
            limit: Maximum entries to return
            offset: Number of entries to skip

        Returns:
            List of matching entries
        """
        return await self.backend.query(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            actors=actors,
            resource_types=resource_types,
            resource_ids=resource_ids,
            workspace_id=workspace_id,
            limit=limit,
            offset=offset,
        )

    async def export_range(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json",
        path: Optional[str] = None,
    ) -> str | bytes:
        """
        Export audit entries for compliance reporting.

        Args:
            start_time: Start of export range
            end_time: End of export range
            format: Output format ("json", "csv", "jsonl")
            path: Optional file path to write to

        Returns:
            Exported data as string/bytes, or path if written to file
        """
        entries = await self.backend.query(
            start_time=start_time,
            end_time=end_time,
            limit=1000000,  # Large limit for export
        )

        if format == "json":
            data = json.dumps(
                {
                    "export_time": datetime.now(timezone.utc).isoformat(),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "entry_count": len(entries),
                    "entries": [e.to_dict() for e in entries],
                },
                indent=2,
            )
        elif format == "jsonl":
            lines = [json.dumps(e.to_dict()) for e in entries]
            data = "\n".join(lines)
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            if entries:
                writer = csv.DictWriter(output, fieldnames=entries[0].to_dict().keys())
                writer.writeheader()
                for entry in entries:
                    writer.writerow(entry.to_dict())
            data = output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

        if path:
            Path(path).write_text(data)
            return path

        return data

    async def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Get audit log statistics.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Statistics dictionary
        """
        await self._ensure_initialized()

        total_count = await self.backend.count(start_time, end_time)
        last_entry = self._last_entry

        return {
            "total_entries": total_count,
            "last_sequence": last_entry.sequence_number if last_entry else 0,
            "last_entry_time": last_entry.timestamp.isoformat() if last_entry else None,
            "last_entry_hash": last_entry.entry_hash if last_entry else None,
        }


# Global instance
_audit_log: Optional[ImmutableAuditLog] = None
_lock = threading.Lock()


def get_audit_log() -> ImmutableAuditLog:
    """Get the global audit log instance."""
    global _audit_log

    if _audit_log is None:
        with _lock:
            if _audit_log is None:
                # Default to local file backend
                log_dir = os.environ.get(
                    "ARAGORA_AUDIT_LOG_DIR",
                    ".nomic/audit_logs",
                )
                signing_key = os.environ.get("ARAGORA_AUDIT_SIGNING_KEY")
                signing_key_bytes = signing_key.encode() if signing_key else None

                backend = LocalFileBackend(log_dir)
                _audit_log = ImmutableAuditLog(
                    backend=backend,
                    signing_key=signing_key_bytes,
                )

    return _audit_log


def init_audit_log(
    backend: AuditBackend = AuditBackend.LOCAL,
    signing_key: Optional[bytes] = None,
    **backend_kwargs: Any,
) -> ImmutableAuditLog:
    """
    Initialize the global audit log with specific configuration.

    Args:
        backend: Backend type to use
        signing_key: Optional signing key for HMAC signatures
        **backend_kwargs: Backend-specific configuration

    Returns:
        The initialized audit log
    """
    global _audit_log

    backend_impl: AuditLogBackend
    if backend == AuditBackend.LOCAL:
        log_dir = backend_kwargs.get("log_dir", ".nomic/audit_logs")
        backend_impl = LocalFileBackend(log_dir)
    elif backend == AuditBackend.S3_OBJECT_LOCK:
        backend_impl = S3ObjectLockBackend(
            bucket=backend_kwargs["bucket"],
            prefix=backend_kwargs.get("prefix", "audit-logs/"),
            region=backend_kwargs.get("region", "us-east-1"),
            retention_days=backend_kwargs.get("retention_days", 2555),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    with _lock:
        _audit_log = ImmutableAuditLog(
            backend=backend_impl,
            signing_key=signing_key,
        )

    return _audit_log


# Convenience functions for common audit events
async def audit_finding_created(
    finding_id: str,
    actor: str,
    workspace_id: str,
    severity: str,
    category: str,
    **kwargs,
) -> AuditEntry:
    """Log finding creation."""
    return await get_audit_log().append(
        event_type="finding_created",
        actor=actor,
        resource_type="finding",
        resource_id=finding_id,
        action="create",
        workspace_id=workspace_id,
        details={"severity": severity, "category": category, **kwargs},
    )


async def audit_finding_updated(
    finding_id: str,
    actor: str,
    changes: dict[str, Any],
    **kwargs,
) -> AuditEntry:
    """Log finding update."""
    return await get_audit_log().append(
        event_type="finding_updated",
        actor=actor,
        resource_type="finding",
        resource_id=finding_id,
        action="update",
        details={"changes": changes, **kwargs},
    )


async def audit_document_uploaded(
    document_id: str,
    actor: str,
    workspace_id: str,
    filename: str,
    **kwargs,
) -> AuditEntry:
    """Log document upload."""
    return await get_audit_log().append(
        event_type="document_uploaded",
        actor=actor,
        resource_type="document",
        resource_id=document_id,
        action="upload",
        workspace_id=workspace_id,
        details={"filename": filename, **kwargs},
    )


async def audit_document_accessed(
    document_id: str,
    actor: str,
    access_type: str = "read",
    **kwargs,
) -> AuditEntry:
    """Log document access."""
    return await get_audit_log().append(
        event_type="document_accessed",
        actor=actor,
        resource_type="document",
        resource_id=document_id,
        action=access_type,
        details=kwargs,
    )


async def audit_session_started(
    session_id: str,
    actor: str,
    workspace_id: str,
    session_type: str,
    **kwargs,
) -> AuditEntry:
    """Log audit session start."""
    return await get_audit_log().append(
        event_type="session_started",
        actor=actor,
        resource_type="audit_session",
        resource_id=session_id,
        action="start",
        workspace_id=workspace_id,
        details={"session_type": session_type, **kwargs},
    )


async def audit_data_exported(
    export_id: str,
    actor: str,
    resource_type: str,
    resource_ids: list[str],
    format: str,
    **kwargs,
) -> AuditEntry:
    """Log data export."""
    return await get_audit_log().append(
        event_type="data_exported",
        actor=actor,
        resource_type="export",
        resource_id=export_id,
        action="export",
        details={
            "exported_type": resource_type,
            "exported_ids": resource_ids,
            "format": format,
            **kwargs,
        },
    )
