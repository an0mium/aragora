# aragora.backup

Automated backup orchestration with verification, encryption, and disaster recovery.
Supports full, incremental, and differential backups of SQLite databases with
compressed storage, integrity checks, and configurable retention policies.

## Modules

| File | Purpose |
|------|---------|
| `manager.py` | `BackupManager` -- create, verify, and restore backups (local/S3/GCS) |
| `scheduler.py` | `BackupScheduler` -- cron-style scheduling with DR drill integration |
| `encryption.py` | AES-256-GCM encryption, key rotation, streaming support |
| `replication_monitor.py` | Monitor replication lag and primary/standby health |
| `monitoring.py` | Prometheus metrics for backup age, size, RPO/RTO compliance |

## Key Features

- **Backup types** -- full, incremental, differential
- **Integrity verification** -- SHA-256 checksums and dry-run restore tests
- **Retention policies** -- automatic expiry enforcement per schedule
- **Encryption** -- AES-256-GCM with key management and rotation
- **Scheduling** -- hourly / daily / weekly / monthly with DR drill hooks
- **Observability** -- Prometheus metrics for backup age, replication lag, RPO/RTO
- **Disaster recovery** -- automated DR drills validate restore capability

## Usage

```python
from aragora.backup import BackupManager, get_backup_manager

# Global singleton
manager = get_backup_manager()

# Create a backup
backup = manager.create_backup("/path/to/database.db")

# Verify integrity
result = manager.verify_backup(backup.id)

# Dry-run restore
manager.restore_backup(backup.id, "/path/to/restore.db", dry_run=True)
```

### Scheduled Backups

```python
from aragora.backup import BackupSchedule, start_backup_scheduler
import datetime

schedule = BackupSchedule(
    daily=datetime.time(2, 0),   # 2 AM daily
    enable_dr_drills=True,       # monthly DR drills
)
scheduler = await start_backup_scheduler(manager, schedule)
```

### Encryption

```python
from aragora.backup import encrypt_backup, decrypt_backup

encrypted = encrypt_backup(backup_path, key)
decrypt_backup(encrypted, key, output_path)
```
