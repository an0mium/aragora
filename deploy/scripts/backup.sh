#!/bin/sh
# Aragora PostgreSQL Backup Script
# Runs daily backups with optional S3 upload

set -e

BACKUP_DIR="/backups"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

backup_database() {
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="${BACKUP_DIR}/aragora_${TIMESTAMP}.sql.gz"

    log "Starting database backup to ${BACKUP_FILE}"

    # Create backup with custom format, compressed
    pg_dump -Fc | gzip > "${BACKUP_FILE}"

    BACKUP_SIZE=$(ls -lh "${BACKUP_FILE}" | awk '{print $5}')
    log "Backup completed: ${BACKUP_FILE} (${BACKUP_SIZE})"

    # Upload to S3 if configured
    if [ -n "${S3_BUCKET}" ] && [ -n "${AWS_ACCESS_KEY_ID}" ]; then
        log "Uploading backup to S3: s3://${S3_BUCKET}/"

        # Install AWS CLI if not present
        if ! command -v aws &> /dev/null; then
            apk add --no-cache aws-cli
        fi

        aws s3 cp "${BACKUP_FILE}" "s3://${S3_BUCKET}/aragora/$(basename ${BACKUP_FILE})"
        log "S3 upload completed"
    fi

    # Clean old backups
    log "Cleaning backups older than ${RETENTION_DAYS} days"
    find "${BACKUP_DIR}" -name "*.sql.gz" -mtime +${RETENTION_DAYS} -delete

    REMAINING=$(ls -1 ${BACKUP_DIR}/*.sql.gz 2>/dev/null | wc -l)
    log "Cleanup complete. ${REMAINING} backups remaining."
}

# Main loop
log "Aragora Backup Service started"
log "Backup directory: ${BACKUP_DIR}"
log "Retention: ${RETENTION_DAYS} days"
log "S3 bucket: ${S3_BUCKET:-not configured}"

while true; do
    # Run backup
    backup_database || log "ERROR: Backup failed!"

    # Wait 24 hours
    log "Next backup in 24 hours"
    sleep 86400
done
