"""
SecurityRepository - Account lockout and login security.

Extracted from UserStore for better modularity. Manages brute-force
protection through progressive account lockouts based on failed login attempts.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SecurityRepository:
    """
    Repository for account security and lockout operations.

    Implements progressive lockout policy:
    - 5 failed attempts: 15 minute lockout
    - 10 failed attempts: 1 hour lockout
    - 20 failed attempts: 24 hour lockout

    Lockouts are automatically cleared after the duration expires.
    """

    # Lockout thresholds
    LOCKOUT_THRESHOLD_1 = 5  # 5 attempts -> 15 min lockout
    LOCKOUT_THRESHOLD_2 = 10  # 10 attempts -> 1 hour lockout
    LOCKOUT_THRESHOLD_3 = 20  # 20 attempts -> 24 hour lockout

    # Lockout durations in seconds
    LOCKOUT_DURATION_1 = 15 * 60  # 15 minutes
    LOCKOUT_DURATION_2 = 60 * 60  # 1 hour
    LOCKOUT_DURATION_3 = 24 * 60 * 60  # 24 hours

    def __init__(self, transaction_fn: Callable[[], AbstractContextManager[sqlite3.Cursor]]) -> None:
        """
        Initialize the security repository.

        Args:
            transaction_fn: Function that returns a transaction context manager
                           with a cursor.
        """
        self._transaction = transaction_fn

    def is_account_locked(self, email: str) -> tuple[bool, Optional[datetime], int]:
        """
        Check if an account is currently locked.

        Args:
            email: User's email address

        Returns:
            Tuple of (is_locked, lockout_until, failed_attempts)
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT failed_login_attempts, lockout_until
                FROM users
                WHERE email = ?
                """,
                (email,),
            )
            row = cursor.fetchone()

            if not row:
                return False, None, 0

            failed_attempts = row[0] or 0
            lockout_until_str = row[1]

            if not lockout_until_str:
                return False, None, failed_attempts

            lockout_until = datetime.fromisoformat(lockout_until_str)
            now = datetime.now()

            if now < lockout_until:
                return True, lockout_until, failed_attempts
            else:
                # Lockout expired
                return False, None, failed_attempts

    def record_failed_login(self, email: str) -> tuple[int, Optional[datetime]]:
        """
        Record a failed login attempt and potentially lock the account.

        Args:
            email: User's email address

        Returns:
            Tuple of (new_attempt_count, lockout_until_if_locked)
        """
        now = datetime.now()

        with self._transaction() as cursor:
            # Increment failed attempts
            cursor.execute(
                """
                UPDATE users
                SET failed_login_attempts = COALESCE(failed_login_attempts, 0) + 1,
                    last_failed_login_at = ?,
                    updated_at = ?
                WHERE email = ?
                """,
                (now.isoformat(), now.isoformat(), email),
            )

            # Get new count
            cursor.execute(
                "SELECT failed_login_attempts FROM users WHERE email = ?",
                (email,),
            )
            row = cursor.fetchone()

            if not row:
                return 0, None

            failed_attempts = row[0]
            lockout_until = None

            # Determine if lockout is needed
            if failed_attempts >= self.LOCKOUT_THRESHOLD_3:
                lockout_until = now + timedelta(seconds=self.LOCKOUT_DURATION_3)
            elif failed_attempts >= self.LOCKOUT_THRESHOLD_2:
                lockout_until = now + timedelta(seconds=self.LOCKOUT_DURATION_2)
            elif failed_attempts >= self.LOCKOUT_THRESHOLD_1:
                lockout_until = now + timedelta(seconds=self.LOCKOUT_DURATION_1)

            if lockout_until:
                cursor.execute(
                    """
                    UPDATE users
                    SET lockout_until = ?
                    WHERE email = ?
                    """,
                    (lockout_until.isoformat(), email),
                )
                logger.warning(
                    f"Account locked: email={email}, attempts={failed_attempts}, "
                    f"locked_until={lockout_until.isoformat()}"
                )

            return failed_attempts, lockout_until

    def reset_failed_login_attempts(self, email: str) -> bool:
        """
        Reset failed login attempts after successful login.

        Args:
            email: User's email address

        Returns:
            True if reset was successful
        """
        now = datetime.now()

        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET failed_login_attempts = 0,
                    lockout_until = NULL,
                    last_login_at = ?,
                    updated_at = ?
                WHERE email = ?
                """,
                (now.isoformat(), now.isoformat(), email),
            )
            return cursor.rowcount > 0

    def get_lockout_info(self, email: str) -> dict:
        """
        Get detailed lockout information for an account.

        Args:
            email: User's email address

        Returns:
            Dict with lockout details including:
            - email: The email address
            - is_locked: Whether the account is currently locked
            - failed_attempts: Number of failed login attempts
            - lockout_until: ISO timestamp when lockout expires (if locked)
            - lockout_remaining_seconds: Seconds until unlock (if locked)
            - lockout_remaining_minutes: Minutes until unlock (if locked)
            - warning: Warning message if approaching lockout
        """
        is_locked, lockout_until, failed_attempts = self.is_account_locked(email)

        info = {
            "email": email,
            "is_locked": is_locked,
            "failed_attempts": failed_attempts,
            "lockout_until": lockout_until.isoformat() if lockout_until else None,
        }

        # Calculate remaining lockout time
        if is_locked and lockout_until:
            remaining = (lockout_until - datetime.now()).total_seconds()
            info["lockout_remaining_seconds"] = max(0, int(remaining))
            info["lockout_remaining_minutes"] = max(0, int(remaining / 60))

        # Warn if approaching lockout
        if not is_locked:
            if failed_attempts >= self.LOCKOUT_THRESHOLD_1 - 2:
                remaining_attempts = self.LOCKOUT_THRESHOLD_1 - failed_attempts
                info["warning"] = (
                    f"Account will be locked after {remaining_attempts} more failed attempts"
                )

        return info
