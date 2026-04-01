"""End-to-end tests for key rotation with encrypted payloads."""

from __future__ import annotations

import os
import secrets

import pytest
from cryptography.exceptions import InvalidTag

from aragora.config.secrets import reset_secret_manager
from aragora.security.encryption import EncryptionService, init_encryption_service
from aragora.security.key_rotation import KeyRotationConfig, KeyRotationScheduler, RotationStatus

_ENV_KEYS = ("ARAGORA_ENCRYPTION_KEY", "ARAGORA_ENCRYPTION_REQUIRED", "ARAGORA_ENV")


@pytest.fixture
def isolated_encryption_service(tmp_path):
    """Provide an isolated global encryption service backed by temp test keys."""
    import aragora.security.encryption as enc_module

    saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    saved_singleton = enc_module._encryption_service

    os.environ.pop("ARAGORA_ENCRYPTION_KEY", None)
    os.environ.pop("ARAGORA_ENCRYPTION_REQUIRED", None)
    os.environ.pop("ARAGORA_ENV", None)
    enc_module._encryption_service = None
    reset_secret_manager()

    initial_key_path = tmp_path / "key_v1.hex"
    initial_key_path.write_text(secrets.token_hex(32))
    service = init_encryption_service(master_key=bytes.fromhex(initial_key_path.read_text()))

    try:
        yield service, initial_key_path
    finally:
        enc_module._encryption_service = saved_singleton
        reset_secret_manager()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.asyncio
async def test_rotate_key_reencrypts_data_and_preserves_access(
    isolated_encryption_service, tmp_path
):
    """Rotating a key should preserve access after re-encryption and reject stale keys."""
    service, initial_key_path = isolated_encryption_service
    scheduler = KeyRotationScheduler(config=KeyRotationConfig(re_encrypt_on_rotation=True))
    associated_data = "record-123"
    plaintext = "rotate this secret"

    encrypted_v1 = service.encrypt(plaintext, associated_data=associated_data)
    assert service.decrypt_string(encrypted_v1, associated_data=associated_data) == plaintext

    old_key_path = tmp_path / "stale_key.hex"
    old_key_path.write_text(initial_key_path.read_text())

    job = await scheduler.rotate_now(key_id=service.get_active_key_id())

    assert job.status == RotationStatus.COMPLETED
    assert job.old_version == 1
    assert job.new_version == 2

    # Existing ciphertext remains readable through the retained overlap key.
    assert service.decrypt_string(encrypted_v1, associated_data=associated_data) == plaintext

    encrypted_v2 = service.re_encrypt(encrypted_v1, associated_data=associated_data)
    assert encrypted_v2.key_version == 2
    assert service.decrypt_string(encrypted_v2, associated_data=associated_data) == plaintext

    stale_service = EncryptionService(master_key=bytes.fromhex(old_key_path.read_text()))
    with pytest.raises(InvalidTag):
        stale_service.decrypt_string(encrypted_v2, associated_data=associated_data)
