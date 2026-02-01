"""
Tests for wallet and transaction signing module.

Tests cover:
- SignerType enum values and member count
- WalletSigner default construction
- from_private_key with mocked eth_account
- from_keystore with mocked eth_account (happy path, file not found, wrong password)
- from_env with private key and keystore environment variables
- from_env with missing credentials
- sign_transaction (configured and not configured)
- sign_and_send (field filling, preserving existing fields, return type)
- sign_message (configured and not configured)
- Security: _account excluded from repr
- Module __all__ exports
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from aragora.blockchain.wallet import SignerType, WalletSigner


def _make_mock_eth_account():
    """Create a mock eth_account module structure for sys.modules injection."""
    mock_module = ModuleType("eth_account")
    mock_account_cls = MagicMock()
    mock_module.Account = mock_account_cls

    mock_messages = ModuleType("eth_account.messages")
    mock_messages.encode_defunct = MagicMock()

    return mock_module, mock_messages, mock_account_cls


class TestSignerType:
    """Tests for SignerType enum."""

    def test_private_key_value(self):
        assert SignerType.PRIVATE_KEY.value == "private_key"

    def test_keystore_value(self):
        assert SignerType.KEYSTORE.value == "keystore"

    def test_external_value(self):
        assert SignerType.EXTERNAL.value == "external"

    def test_member_count(self):
        assert len(SignerType) == 3


class TestWalletSigner:
    """Tests for WalletSigner class."""

    def test_create_default(self):
        signer = WalletSigner()
        assert signer.signer_type == SignerType.PRIVATE_KEY
        assert signer.address == ""
        assert signer._account is None

    def test_from_private_key(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0x1234567890123456789012345678901234567890"
        mock_account_cls.from_key.return_value = mock_acct

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_private_key("0x" + "a" * 64)

        assert signer.signer_type == SignerType.PRIVATE_KEY
        assert signer.address == "0x1234567890123456789012345678901234567890"
        assert signer._account is mock_acct
        mock_account_cls.from_key.assert_called_once_with("0x" + "a" * 64)

    def test_from_private_key_without_prefix(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0xAddr"
        mock_account_cls.from_key.return_value = mock_acct

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            WalletSigner.from_private_key("deadbeef")
        mock_account_cls.from_key.assert_called_once_with("deadbeef")

    def test_from_env_with_private_key(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0xABCD1234567890ABCD1234567890ABCD12345678"
        mock_account_cls.from_key.return_value = mock_acct

        env = {"ERC8004_WALLET_KEY": "0x" + "b" * 64}
        with patch.dict(os.environ, env, clear=False):
            with patch.dict(sys.modules, {"eth_account": mock_mod}):
                signer = WalletSigner.from_env()
        assert signer.address == "0xABCD1234567890ABCD1234567890ABCD12345678"

    def test_from_env_with_keystore(self, tmp_path):
        mock_mod, mock_messages, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0xKEYSTORE12345678901234567890123456789012"
        mock_account_cls.decrypt.return_value = b"\xaa" * 32
        mock_account_cls.from_key.return_value = mock_acct

        keystore_file = tmp_path / "keystore.json"
        keystore_file.write_text('{"version": 3, "crypto": {}}')

        env = {
            "ERC8004_KEYSTORE_PATH": str(keystore_file),
            "ERC8004_KEYSTORE_PASSWORD": "testpass",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ERC8004_WALLET_KEY", None)
            with patch.dict(sys.modules, {"eth_account": mock_mod}):
                signer = WalletSigner.from_env()
        assert signer.address == "0xKEYSTORE12345678901234567890123456789012"
        assert signer.signer_type == SignerType.KEYSTORE

    def test_from_env_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ERC8004_WALLET_KEY", None)
            os.environ.pop("ERC8004_KEYSTORE_PATH", None)
            with pytest.raises(ValueError, match="No wallet credentials"):
                WalletSigner.from_env()

    def test_sign_transaction(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0x1234567890123456789012345678901234567890"
        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\x00" * 32
        mock_acct.sign_transaction.return_value = mock_signed
        mock_account_cls.from_key.return_value = mock_acct

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_private_key("0x" + "a" * 64)

        tx = {
            "to": "0xDEADBEEF12345678901234567890123456789012",
            "value": 0,
            "gas": 100000,
            "gasPrice": 20_000_000_000,
            "nonce": 0,
            "chainId": 1,
        }
        signed = signer.sign_transaction(tx)
        assert signed.raw_transaction == b"\x00" * 32
        mock_acct.sign_transaction.assert_called_once_with(tx)

    def test_sign_transaction_not_configured(self):
        signer = WalletSigner()
        with pytest.raises(ValueError, match="Signer not configured"):
            signer.sign_transaction({"to": "0x123"})

    def test_sign_and_send(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0x1234567890123456789012345678901234567890"
        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\xab" * 32
        mock_acct.sign_transaction.return_value = mock_signed
        mock_account_cls.from_key.return_value = mock_acct

        mock_w3 = MagicMock()
        mock_w3.eth.get_transaction_count.return_value = 5
        mock_w3.eth.chain_id = 1
        mock_w3.eth.gas_price = 20_000_000_000
        mock_w3.eth.send_raw_transaction.return_value = b"\xcd" * 32

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_private_key("0x" + "a" * 64)
        tx = {"to": "0xDEADBEEF12345678901234567890123456789012", "value": 1000}
        tx_hash = signer.sign_and_send(mock_w3, tx)

        assert tx_hash == (b"\xcd" * 32).hex()
        mock_w3.eth.send_raw_transaction.assert_called_once()

    def test_sign_and_send_fills_missing_fields(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0x1234567890123456789012345678901234567890"
        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\xab" * 32
        mock_acct.sign_transaction.return_value = mock_signed
        mock_account_cls.from_key.return_value = mock_acct

        mock_w3 = MagicMock()
        mock_w3.eth.get_transaction_count.return_value = 10
        mock_w3.eth.chain_id = 137
        mock_w3.eth.gas_price = 30_000_000_000
        mock_w3.eth.send_raw_transaction.return_value = b"\xef" * 32

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_private_key("0x" + "a" * 64)
        tx = {"to": "0xDEADBEEF12345678901234567890123456789012"}
        signer.sign_and_send(mock_w3, tx)

        signed_tx = mock_acct.sign_transaction.call_args[0][0]
        assert signed_tx["from"] == "0x1234567890123456789012345678901234567890"
        assert signed_tx["nonce"] == 10
        assert signed_tx["chainId"] == 137
        assert signed_tx["gasPrice"] == 30_000_000_000

    def test_sign_and_send_preserves_existing_fields(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0xSender"
        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\xaa"
        mock_acct.sign_transaction.return_value = mock_signed
        mock_account_cls.from_key.return_value = mock_acct

        mock_w3 = MagicMock()
        mock_w3.eth.send_raw_transaction.return_value = b"\xbb"

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_private_key("0x" + "c" * 64)
        tx = {
            "to": "0xRecipient",
            "from": "0xCustomSender",
            "nonce": 42,
            "chainId": 137,
            "maxFeePerGas": 100,
        }
        signer.sign_and_send(mock_w3, tx)

        signed_tx = mock_acct.sign_transaction.call_args[0][0]
        assert signed_tx["from"] == "0xCustomSender"
        assert signed_tx["nonce"] == 42
        assert signed_tx["chainId"] == 137
        # maxFeePerGas present, so gasPrice should NOT be set
        assert "gasPrice" not in signed_tx

    def test_sign_and_send_returns_hex_string(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0xAddr"
        mock_signed = MagicMock()
        mock_signed.raw_transaction = b"\x01"
        mock_acct.sign_transaction.return_value = mock_signed
        mock_account_cls.from_key.return_value = mock_acct

        mock_w3 = MagicMock()
        mock_w3.eth.get_transaction_count.return_value = 0
        mock_w3.eth.chain_id = 1
        mock_w3.eth.gas_price = 1
        mock_w3.eth.send_raw_transaction.return_value = bytes.fromhex("abcdef")

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_private_key("0x" + "d" * 64)
        result = signer.sign_and_send(mock_w3, {"to": "0x0", "gas": 21000})
        assert result == "abcdef"

    def test_sign_message(self):
        mock_mod, mock_messages, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0x1234567890123456789012345678901234567890"
        mock_signature = MagicMock()
        mock_acct.sign_message.return_value = mock_signature
        mock_account_cls.from_key.return_value = mock_acct
        mock_messages.encode_defunct.return_value = "encoded_message"

        with patch.dict(
            sys.modules,
            {"eth_account": mock_mod, "eth_account.messages": mock_messages},
        ):
            signer = WalletSigner.from_private_key("0x" + "a" * 64)
            signature = signer.sign_message("Hello, World!")

        assert signature is mock_signature
        mock_messages.encode_defunct.assert_called_once_with(text="Hello, World!")

    def test_sign_message_not_configured(self):
        signer = WalletSigner()
        with pytest.raises(ValueError, match="Signer not configured"):
            signer.sign_message("test")


class TestWalletSignerKeystore:
    """Tests for keystore-based wallet creation."""

    def test_from_keystore(self, tmp_path):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0xKEYSTORE12345678901234567890123456789012"
        mock_account_cls.decrypt.return_value = b"\xaa" * 32
        mock_account_cls.from_key.return_value = mock_acct

        keystore_file = tmp_path / "keystore.json"
        keystore_file.write_text('{"version": 3, "crypto": {}}')

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_keystore(str(keystore_file), "password123")
        assert signer.signer_type == SignerType.KEYSTORE
        assert signer.address == "0xKEYSTORE12345678901234567890123456789012"

    def test_from_keystore_file_not_found(self):
        mock_mod, _, _ = _make_mock_eth_account()
        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            with pytest.raises(FileNotFoundError, match="Keystore file not found"):
                WalletSigner.from_keystore("/nonexistent/keystore.json", "password")

    def test_from_keystore_wrong_password(self, tmp_path):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_account_cls.decrypt.side_effect = ValueError("MAC mismatch")

        keystore_file = tmp_path / "keystore.json"
        keystore_file.write_text('{"version": 3, "crypto": {}}')

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            with pytest.raises(ValueError, match="MAC mismatch"):
                WalletSigner.from_keystore(str(keystore_file), "wrong_password")


class TestWalletSignerSecurity:
    """Security-related tests for WalletSigner."""

    def test_account_not_in_repr(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0x1234567890123456789012345678901234567890"
        mock_account_cls.from_key.return_value = mock_acct

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_private_key("0x" + "a" * 64)
        repr_str = repr(signer)
        assert "_account" not in repr_str

    def test_signer_type_in_repr(self):
        mock_mod, _, mock_account_cls = _make_mock_eth_account()
        mock_acct = MagicMock()
        mock_acct.address = "0x1234567890123456789012345678901234567890"
        mock_account_cls.from_key.return_value = mock_acct

        with patch.dict(sys.modules, {"eth_account": mock_mod}):
            signer = WalletSigner.from_private_key("0x" + "a" * 64)
        repr_str = repr(signer)
        assert "signer_type" in repr_str
        assert "address" in repr_str


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_exports_signer_type(self):
        from aragora.blockchain.wallet import __all__

        assert "SignerType" in __all__

    def test_exports_wallet_signer(self):
        from aragora.blockchain.wallet import __all__

        assert "WalletSigner" in __all__

    def test_all_count(self):
        from aragora.blockchain.wallet import __all__

        assert len(__all__) == 2
