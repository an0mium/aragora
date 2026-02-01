"""
Wallet signer for ERC-8004 blockchain transactions.

Supports multiple signing methods:
- Raw private key (from environment variable)
- Keystore file (encrypted JSON)
- External signer (for hardware wallets or custody solutions)

Security notes:
- Private keys are never logged or serialized.
- Keystore passwords are read from env and not stored in memory beyond init.
- Use keystore or external signers in production.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SignerType(Enum):
    """Supported signer types."""

    PRIVATE_KEY = "private_key"
    KEYSTORE = "keystore"
    EXTERNAL = "external"


@dataclass
class WalletSigner:
    """Signs Ethereum transactions for ERC-8004 registry operations.

    Usage:
        # From environment variable
        signer = WalletSigner.from_env()

        # From private key (testing only)
        signer = WalletSigner.from_private_key("0x...")

        # From keystore file
        signer = WalletSigner.from_keystore("/path/to/keystore.json", "password")

        # Sign and send a transaction
        tx_hash = signer.sign_and_send(w3, tx_dict)
    """

    signer_type: SignerType = SignerType.PRIVATE_KEY
    address: str = ""
    _account: Any = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> WalletSigner:
        """Create a signer from environment variables.

        Checks in order:
        1. ERC8004_WALLET_KEY - raw private key
        2. ERC8004_KEYSTORE_PATH + ERC8004_KEYSTORE_PASSWORD - keystore file

        Returns:
            Configured WalletSigner.

        Raises:
            ValueError: If no wallet credentials are configured.
        """
        private_key = os.getenv("ERC8004_WALLET_KEY")
        if private_key:
            return cls.from_private_key(private_key)

        keystore_path = os.getenv("ERC8004_KEYSTORE_PATH")
        keystore_password = os.getenv("ERC8004_KEYSTORE_PASSWORD", "")
        if keystore_path:
            return cls.from_keystore(keystore_path, keystore_password)

        raise ValueError(
            "No wallet credentials configured. Set ERC8004_WALLET_KEY or "
            "ERC8004_KEYSTORE_PATH + ERC8004_KEYSTORE_PASSWORD."
        )

    @classmethod
    def from_private_key(cls, private_key: str) -> WalletSigner:
        """Create a signer from a raw private key.

        Args:
            private_key: Hex-encoded private key (with or without 0x prefix).

        Returns:
            Configured WalletSigner.
        """
        from eth_account import Account

        account = Account.from_key(private_key)
        logger.info(f"Wallet signer initialized: {account.address}")
        return cls(
            signer_type=SignerType.PRIVATE_KEY,
            address=account.address,
            _account=account,
        )

    @classmethod
    def from_keystore(cls, keystore_path: str, password: str) -> WalletSigner:
        """Create a signer from an encrypted keystore file.

        Args:
            keystore_path: Path to the keystore JSON file.
            password: Decryption password.

        Returns:
            Configured WalletSigner.

        Raises:
            FileNotFoundError: If keystore file doesn't exist.
            ValueError: If keystore decryption fails.
        """
        from eth_account import Account

        path = Path(keystore_path)
        if not path.exists():
            raise FileNotFoundError(f"Keystore file not found: {keystore_path}")

        keystore_json = json.loads(path.read_text())
        private_key = Account.decrypt(keystore_json, password)
        account = Account.from_key(private_key)
        logger.info(f"Wallet signer initialized from keystore: {account.address}")
        return cls(
            signer_type=SignerType.KEYSTORE,
            address=account.address,
            _account=account,
        )

    def sign_transaction(self, tx_dict: dict[str, Any]) -> Any:
        """Sign a transaction dictionary.

        Args:
            tx_dict: Transaction parameters (to, value, data, gas, etc.).

        Returns:
            SignedTransaction object.

        Raises:
            ValueError: If signer is not configured for signing.
        """
        if self._account is None:
            raise ValueError("Signer not configured. Use from_env() or from_private_key().")

        return self._account.sign_transaction(tx_dict)

    def sign_and_send(self, w3: Any, tx_dict: dict[str, Any]) -> str:
        """Sign a transaction and send it to the network.

        Args:
            w3: Web3 instance.
            tx_dict: Transaction parameters.

        Returns:
            Transaction hash as hex string.
        """
        # Fill in missing transaction fields
        if "from" not in tx_dict:
            tx_dict["from"] = self.address
        if "nonce" not in tx_dict:
            tx_dict["nonce"] = w3.eth.get_transaction_count(self.address)
        if "chainId" not in tx_dict:
            tx_dict["chainId"] = w3.eth.chain_id
        if "gasPrice" not in tx_dict and "maxFeePerGas" not in tx_dict:
            tx_dict["gasPrice"] = w3.eth.gas_price

        signed = self.sign_transaction(tx_dict)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()

    def sign_message(self, message: str) -> Any:
        """Sign a message (EIP-191 personal sign).

        Args:
            message: Message to sign.

        Returns:
            SignedMessage object.
        """
        if self._account is None:
            raise ValueError("Signer not configured.")

        from eth_account.messages import encode_defunct

        msg = encode_defunct(text=message)
        return self._account.sign_message(msg)


__all__ = [
    "SignerType",
    "WalletSigner",
]
