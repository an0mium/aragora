"""Tests for NotificationConfigStore multi-tenant storage."""

import pytest
import tempfile
import os
from pathlib import Path

from aragora.storage.notification_config_store import (
    NotificationConfigStore,
    StoredEmailConfig,
    StoredTelegramConfig,
    StoredEmailRecipient,
    get_notification_config_store,
    reset_notification_config_store,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test_notifications.db")
        yield db_path


@pytest.fixture
def store(temp_db):
    """Create a fresh store for each test."""
    reset_notification_config_store()
    return NotificationConfigStore(db_path=temp_db)


class TestEmailConfigStorage:
    """Test email configuration storage."""

    @pytest.mark.asyncio
    async def test_save_and_get_email_config(self, store):
        """Test saving and retrieving email config."""
        config = StoredEmailConfig(
            org_id="org-123",
            provider="smtp",
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_username="user@example.com",
            smtp_password="secret123",
            from_email="noreply@example.com",
        )

        await store.save_email_config(config)
        retrieved = await store.get_email_config("org-123")

        assert retrieved is not None
        assert retrieved.org_id == "org-123"
        assert retrieved.smtp_host == "smtp.example.com"
        assert retrieved.smtp_port == 587
        assert retrieved.from_email == "noreply@example.com"

    @pytest.mark.asyncio
    async def test_get_nonexistent_email_config(self, store):
        """Test retrieving config that doesn't exist."""
        retrieved = await store.get_email_config("nonexistent-org")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_update_email_config(self, store):
        """Test updating existing email config."""
        config1 = StoredEmailConfig(
            org_id="org-123",
            smtp_host="smtp1.example.com",
        )
        await store.save_email_config(config1)

        config2 = StoredEmailConfig(
            org_id="org-123",
            smtp_host="smtp2.example.com",
        )
        await store.save_email_config(config2)

        retrieved = await store.get_email_config("org-123")
        assert retrieved.smtp_host == "smtp2.example.com"

    @pytest.mark.asyncio
    async def test_delete_email_config(self, store):
        """Test deleting email config."""
        config = StoredEmailConfig(
            org_id="org-123",
            smtp_host="smtp.example.com",
        )
        await store.save_email_config(config)

        deleted = await store.delete_email_config("org-123")
        assert deleted is True

        retrieved = await store.get_email_config("org-123")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_config(self, store):
        """Test deleting config that doesn't exist."""
        deleted = await store.delete_email_config("nonexistent-org")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_tenant_isolation_email(self, store):
        """Test that email configs are isolated by org_id."""
        config1 = StoredEmailConfig(
            org_id="org-1",
            smtp_host="smtp1.example.com",
        )
        config2 = StoredEmailConfig(
            org_id="org-2",
            smtp_host="smtp2.example.com",
        )

        await store.save_email_config(config1)
        await store.save_email_config(config2)

        retrieved1 = await store.get_email_config("org-1")
        retrieved2 = await store.get_email_config("org-2")

        assert retrieved1.smtp_host == "smtp1.example.com"
        assert retrieved2.smtp_host == "smtp2.example.com"


class TestTelegramConfigStorage:
    """Test telegram configuration storage."""

    @pytest.mark.asyncio
    async def test_save_and_get_telegram_config(self, store):
        """Test saving and retrieving telegram config."""
        config = StoredTelegramConfig(
            org_id="org-123",
            bot_token="123456:ABC-DEF",
            chat_id="-1001234567890",
            notify_on_consensus=True,
            notify_on_error=False,
        )

        await store.save_telegram_config(config)
        retrieved = await store.get_telegram_config("org-123")

        assert retrieved is not None
        assert retrieved.org_id == "org-123"
        assert retrieved.chat_id == "-1001234567890"
        assert retrieved.notify_on_consensus is True
        assert retrieved.notify_on_error is False

    @pytest.mark.asyncio
    async def test_get_nonexistent_telegram_config(self, store):
        """Test retrieving telegram config that doesn't exist."""
        retrieved = await store.get_telegram_config("nonexistent-org")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_tenant_isolation_telegram(self, store):
        """Test that telegram configs are isolated by org_id."""
        config1 = StoredTelegramConfig(
            org_id="org-1",
            bot_token="token1",
            chat_id="chat1",
        )
        config2 = StoredTelegramConfig(
            org_id="org-2",
            bot_token="token2",
            chat_id="chat2",
        )

        await store.save_telegram_config(config1)
        await store.save_telegram_config(config2)

        retrieved1 = await store.get_telegram_config("org-1")
        retrieved2 = await store.get_telegram_config("org-2")

        assert retrieved1.chat_id == "chat1"
        assert retrieved2.chat_id == "chat2"


class TestEmailRecipientsStorage:
    """Test email recipients storage."""

    @pytest.mark.asyncio
    async def test_add_and_get_recipients(self, store):
        """Test adding and retrieving recipients."""
        recipient = StoredEmailRecipient(
            org_id="org-123",
            email="user@example.com",
            name="Test User",
            preferences={"digest": True},
        )

        await store.add_recipient(recipient)
        recipients = await store.get_recipients("org-123")

        assert len(recipients) == 1
        assert recipients[0].email == "user@example.com"
        assert recipients[0].name == "Test User"
        assert recipients[0].preferences == {"digest": True}

    @pytest.mark.asyncio
    async def test_get_recipients_empty(self, store):
        """Test retrieving recipients when none exist."""
        recipients = await store.get_recipients("org-123")
        assert recipients == []

    @pytest.mark.asyncio
    async def test_add_multiple_recipients(self, store):
        """Test adding multiple recipients."""
        for i in range(3):
            recipient = StoredEmailRecipient(
                org_id="org-123",
                email=f"user{i}@example.com",
            )
            await store.add_recipient(recipient)

        recipients = await store.get_recipients("org-123")
        assert len(recipients) == 3

    @pytest.mark.asyncio
    async def test_update_recipient(self, store):
        """Test updating recipient (upsert by email)."""
        recipient1 = StoredEmailRecipient(
            org_id="org-123",
            email="user@example.com",
            name="Original Name",
        )
        await store.add_recipient(recipient1)

        recipient2 = StoredEmailRecipient(
            org_id="org-123",
            email="user@example.com",
            name="Updated Name",
        )
        await store.add_recipient(recipient2)

        recipients = await store.get_recipients("org-123")
        assert len(recipients) == 1
        assert recipients[0].name == "Updated Name"

    @pytest.mark.asyncio
    async def test_remove_recipient(self, store):
        """Test removing a recipient."""
        recipient = StoredEmailRecipient(
            org_id="org-123",
            email="user@example.com",
        )
        await store.add_recipient(recipient)

        removed = await store.remove_recipient("org-123", "user@example.com")
        assert removed is True

        recipients = await store.get_recipients("org-123")
        assert len(recipients) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_recipient(self, store):
        """Test removing recipient that doesn't exist."""
        removed = await store.remove_recipient("org-123", "nonexistent@example.com")
        assert removed is False

    @pytest.mark.asyncio
    async def test_clear_recipients(self, store):
        """Test clearing all recipients for an org."""
        for i in range(5):
            await store.add_recipient(
                StoredEmailRecipient(org_id="org-123", email=f"user{i}@example.com")
            )

        cleared = await store.clear_recipients("org-123")
        assert cleared == 5

        recipients = await store.get_recipients("org-123")
        assert len(recipients) == 0

    @pytest.mark.asyncio
    async def test_tenant_isolation_recipients(self, store):
        """Test that recipients are isolated by org_id."""
        await store.add_recipient(StoredEmailRecipient(org_id="org-1", email="user1@example.com"))
        await store.add_recipient(StoredEmailRecipient(org_id="org-2", email="user2@example.com"))

        recipients1 = await store.get_recipients("org-1")
        recipients2 = await store.get_recipients("org-2")

        assert len(recipients1) == 1
        assert recipients1[0].email == "user1@example.com"
        assert len(recipients2) == 1
        assert recipients2[0].email == "user2@example.com"


class TestDataclassConversions:
    """Test dataclass to_dict and from_dict methods."""

    def test_email_config_to_dict(self):
        """Test StoredEmailConfig.to_dict()."""
        config = StoredEmailConfig(
            org_id="org-123",
            smtp_host="smtp.example.com",
        )
        d = config.to_dict()
        assert d["org_id"] == "org-123"
        assert d["smtp_host"] == "smtp.example.com"

    def test_email_config_from_dict(self):
        """Test StoredEmailConfig.from_dict()."""
        d = {
            "org_id": "org-123",
            "smtp_host": "smtp.example.com",
            "extra_field": "ignored",
        }
        config = StoredEmailConfig.from_dict(d)
        assert config.org_id == "org-123"
        assert config.smtp_host == "smtp.example.com"

    def test_telegram_config_to_dict(self):
        """Test StoredTelegramConfig.to_dict()."""
        config = StoredTelegramConfig(
            org_id="org-123",
            bot_token="token",
            chat_id="chat",
        )
        d = config.to_dict()
        assert d["org_id"] == "org-123"
        assert d["bot_token"] == "token"

    def test_recipient_to_dict(self):
        """Test StoredEmailRecipient.to_dict()."""
        recipient = StoredEmailRecipient(
            org_id="org-123",
            email="user@example.com",
            name="Test",
        )
        d = recipient.to_dict()
        assert d["org_id"] == "org-123"
        assert d["email"] == "user@example.com"
        assert d["name"] == "Test"


class TestSingletonBehavior:
    """Test singleton store behavior."""

    def test_get_notification_config_store_singleton(self, temp_db):
        """Test that get_notification_config_store returns singleton."""
        reset_notification_config_store()

        # Override with environment variable
        os.environ["ARAGORA_DATA_DIR"] = str(Path(temp_db).parent)

        try:
            store1 = get_notification_config_store()
            store2 = get_notification_config_store()
            assert store1 is store2
        finally:
            os.environ.pop("ARAGORA_DATA_DIR", None)
            reset_notification_config_store()

    def test_reset_clears_singleton(self, temp_db):
        """Test that reset_notification_config_store clears singleton."""
        reset_notification_config_store()

        os.environ["ARAGORA_DATA_DIR"] = str(Path(temp_db).parent)

        try:
            store1 = get_notification_config_store()
            reset_notification_config_store()
            store2 = get_notification_config_store()
            assert store1 is not store2
        finally:
            os.environ.pop("ARAGORA_DATA_DIR", None)
            reset_notification_config_store()
