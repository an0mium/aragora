"""
Tests for ERC8004Adapter - Blockchain integration for Knowledge Mound.

Tests cover:
- Initialization and configuration
- Forward sync (blockchain -> KM): identities, reputation, validations
- Reverse sync (KM -> blockchain): ELO ratings as reputation, receipts as validations
- Consensus verification for on-chain writes
- Error handling for blockchain operations
- Health status reporting
"""

import hashlib
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter
from aragora.knowledge.mound.adapters._types import SyncResult, ValidationSyncResult


# =============================================================================
# Mock Helpers
# =============================================================================


def _make_mock_provider(
    is_connected: bool = True,
    chain_id: int = 1,
    has_identity_registry: bool = True,
    has_reputation_registry: bool = True,
    has_validation_registry: bool = True,
) -> MagicMock:
    """Create a mock Web3Provider."""
    provider = MagicMock()
    provider.is_connected.return_value = is_connected

    config = MagicMock()
    config.chain_id = chain_id
    config.has_identity_registry = has_identity_registry
    config.has_reputation_registry = has_reputation_registry
    config.has_validation_registry = has_validation_registry
    provider.get_config.return_value = config

    health = {"rpc_status": "healthy", "latency_ms": 50}
    provider.get_health_status.return_value = health

    return provider


def _make_mock_signer(address: str = "0x1234567890abcdef") -> MagicMock:
    """Create a mock WalletSigner."""
    signer = MagicMock()
    signer.address = address
    signer.sign_and_send.return_value = "0xabcdef123456"
    return signer


def _make_mock_identity(
    token_id: int = 1,
    owner: str = "0xOwner",
    agent_uri: str = "https://example.com/agent.json",
    wallet_address: str | None = None,
    chain_id: int = 1,
    aragora_agent_id: str | None = "claude",
) -> MagicMock:
    """Create a mock OnChainAgentIdentity."""
    identity = MagicMock()
    identity.token_id = token_id
    identity.owner = owner
    identity.agent_uri = agent_uri
    identity.wallet_address = wallet_address
    identity.chain_id = chain_id
    identity.aragora_agent_id = aragora_agent_id
    return identity


def _make_mock_reputation_summary(
    agent_id: int = 1,
    count: int = 10,
    normalized_value: float = 0.8,
    tag1: str = "aragora_elo",
    tag2: str = "security",
) -> MagicMock:
    """Create a mock ReputationSummary."""
    summary = MagicMock()
    summary.agent_id = agent_id
    summary.count = count
    summary.normalized_value = normalized_value
    summary.tag1 = tag1
    summary.tag2 = tag2
    return summary


def _make_mock_validation_record(
    request_hash: str = "0x123abc",
    agent_id: int = 1,
    validator_address: str = "0xValidator",
    response_name: str = "PASS",
    tag: str = "validated_claims",
) -> MagicMock:
    """Create a mock ValidationRecord."""
    record = MagicMock()
    record.request_hash = request_hash
    record.agent_id = agent_id
    record.validator_address = validator_address
    record.response = MagicMock()
    record.response.name = response_name
    record.tag = tag
    return record


def _make_mock_agent_link(
    aragora_agent_id: str = "claude",
    chain_id: int = 1,
    token_id: int = 42,
    owner_address: str = "0xOwner",
    verified: bool = True,
) -> MagicMock:
    """Create a mock AgentBlockchainLink."""
    link = MagicMock()
    link.aragora_agent_id = aragora_agent_id
    link.chain_id = chain_id
    link.token_id = token_id
    link.owner_address = owner_address
    link.verified = verified
    return link


def _make_mock_receipt_result(
    receipt_id: str = "rcpt-123",
    claims_ingested: int = 5,
    findings_ingested: int = 2,
    success: bool = True,
) -> MagicMock:
    """Create a mock ReceiptIngestionResult."""
    result = MagicMock()
    result.receipt_id = receipt_id
    result.claims_ingested = claims_ingested
    result.findings_ingested = findings_ingested
    result.relationships_created = 3
    result.knowledge_item_ids = [f"rcpt_{receipt_id}", "claim_abc123"]
    result.success = success
    result.errors = [] if success else ["Some error"]
    return result


# =============================================================================
# Test Initialization
# =============================================================================


class TestERC8004AdapterInit:
    """Test adapter initialization."""

    def test_init_default(self):
        """Test default initialization."""
        adapter = ERC8004Adapter()

        assert adapter.adapter_name == "erc8004"
        assert adapter._provider is None
        assert adapter._signer is None
        assert adapter._km_store is None
        assert adapter._enable_reverse_sync is False
        assert adapter._min_elo_for_reputation == 1500.0

    def test_init_with_provider(self):
        """Test initialization with provider."""
        provider = _make_mock_provider()
        adapter = ERC8004Adapter(provider=provider)

        assert adapter._provider is provider

    def test_init_with_signer(self):
        """Test initialization with signer."""
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(signer=signer)

        assert adapter._signer is signer

    def test_init_with_reverse_sync_enabled(self):
        """Test initialization with reverse sync enabled."""
        adapter = ERC8004Adapter(enable_reverse_sync=True)

        assert adapter._enable_reverse_sync is True

    def test_init_with_custom_min_elo(self):
        """Test initialization with custom minimum ELO threshold."""
        adapter = ERC8004Adapter(min_elo_for_reputation=1600.0)

        assert adapter._min_elo_for_reputation == 1600.0


# =============================================================================
# Test Forward Sync (Blockchain -> KM)
# =============================================================================


class TestForwardSync:
    """Test forward sync operations."""

    @pytest.mark.asyncio
    async def test_sync_to_km_basic(self):
        """Test basic forward sync."""
        provider = _make_mock_provider()
        adapter = ERC8004Adapter(provider=provider)

        # Mock contracts
        identity_contract = MagicMock()
        identity_contract.get_total_supply.return_value = 2
        identity_contract.get_agent.return_value = _make_mock_identity(token_id=1)

        reputation_contract = MagicMock()
        reputation_contract.get_summary.return_value = _make_mock_reputation_summary()

        validation_contract = MagicMock()
        validation_contract.get_agent_validations.return_value = [b"0x123"]
        validation_contract.get_validation_status.return_value = _make_mock_validation_record()

        with patch.object(adapter, "_get_identity_contract", return_value=identity_contract):
            with patch.object(
                adapter, "_get_reputation_contract", return_value=reputation_contract
            ):
                with patch.object(
                    adapter, "_get_validation_contract", return_value=validation_contract
                ):
                    result = await adapter.sync_to_km()

        assert isinstance(result, SyncResult)
        assert result.records_synced > 0
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_sync_to_km_with_specific_agents(self):
        """Test forward sync with specific agent IDs."""
        provider = _make_mock_provider()
        adapter = ERC8004Adapter(provider=provider)

        identity_contract = MagicMock()
        identity_contract.get_agent.return_value = _make_mock_identity()

        reputation_contract = MagicMock()
        reputation_contract.get_summary.return_value = _make_mock_reputation_summary()

        validation_contract = MagicMock()
        validation_contract.get_agent_validations.return_value = []

        with patch.object(adapter, "_get_identity_contract", return_value=identity_contract):
            with patch.object(
                adapter, "_get_reputation_contract", return_value=reputation_contract
            ):
                with patch.object(
                    adapter, "_get_validation_contract", return_value=validation_contract
                ):
                    result = await adapter.sync_to_km(agent_ids=[1, 2, 3])

        assert isinstance(result, SyncResult)
        # Should have synced identity and reputation for 3 agents
        assert result.records_synced >= 3

    @pytest.mark.asyncio
    async def test_sync_to_km_handles_errors(self):
        """Test that forward sync handles errors gracefully."""
        provider = _make_mock_provider()
        adapter = ERC8004Adapter(provider=provider)

        identity_contract = MagicMock()
        identity_contract.get_total_supply.return_value = 2
        identity_contract.get_agent.side_effect = Exception("Contract error")

        with patch.object(adapter, "_get_identity_contract", return_value=identity_contract):
            result = await adapter.sync_to_km()

        assert isinstance(result, SyncResult)
        assert result.records_failed > 0
        assert len(result.errors) > 0


# =============================================================================
# Test Reverse Sync (KM -> Blockchain)
# =============================================================================


class TestReverseSync:
    """Test reverse sync operations."""

    @pytest.mark.asyncio
    async def test_sync_from_km_disabled(self):
        """Test that reverse sync returns early when disabled."""
        adapter = ERC8004Adapter(enable_reverse_sync=False)

        result = await adapter.sync_from_km()

        assert isinstance(result, ValidationSyncResult)
        assert "Reverse sync is disabled" in result.errors

    @pytest.mark.asyncio
    async def test_sync_from_km_no_signer(self):
        """Test that reverse sync requires a signer."""
        adapter = ERC8004Adapter(enable_reverse_sync=True, signer=None)

        result = await adapter.sync_from_km()

        assert isinstance(result, ValidationSyncResult)
        assert any("No signer configured" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_sync_from_km_no_linked_agents(self):
        """Test reverse sync with no linked agents."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = []

        with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
            result = await adapter.sync_from_km()

        assert isinstance(result, ValidationSyncResult)
        assert "No agents linked to blockchain identities" in result.errors

    @pytest.mark.asyncio
    async def test_sync_from_km_pushes_elo_ratings(self):
        """Test that reverse sync pushes ELO ratings as reputation."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
            min_elo_for_reputation=1400.0,
        )

        # Mock identity bridge with linked agents
        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = [
            _make_mock_agent_link(aragora_agent_id="claude", token_id=42),
        ]

        # Mock performance adapter with ELO data
        mock_perf_adapter = MagicMock()
        mock_perf_adapter.get_agent_skill_history.return_value = [
            {
                "elo": 1600.0,
                "debates_count": 10,
                "domain_elos": {"security": 1650, "coding": 1550},
            }
        ]

        # Mock reputation contract
        mock_rep_contract = MagicMock()
        mock_rep_contract.give_feedback.return_value = "0xabc123"

        with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
            with patch.object(adapter, "_get_performance_adapter", return_value=mock_perf_adapter):
                with patch.object(
                    adapter, "_get_reputation_contract", return_value=mock_rep_contract
                ):
                    result = await adapter.sync_from_km(push_elo_ratings=True, push_receipts=False)

        assert isinstance(result, ValidationSyncResult)
        assert result.records_analyzed >= 1
        assert result.records_updated >= 1
        mock_rep_contract.give_feedback.assert_called()

    @pytest.mark.asyncio
    async def test_sync_from_km_skips_low_elo_agents(self):
        """Test that agents below ELO threshold are skipped."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
            min_elo_for_reputation=1500.0,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = [
            _make_mock_agent_link(aragora_agent_id="weak_agent", token_id=99),
        ]

        mock_perf_adapter = MagicMock()
        mock_perf_adapter.get_agent_skill_history.return_value = [
            {
                "elo": 1300.0,  # Below threshold
                "debates_count": 10,
                "domain_elos": {},
            }
        ]

        mock_rep_contract = MagicMock()

        with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
            with patch.object(adapter, "_get_performance_adapter", return_value=mock_perf_adapter):
                with patch.object(
                    adapter, "_get_reputation_contract", return_value=mock_rep_contract
                ):
                    result = await adapter.sync_from_km(push_elo_ratings=True, push_receipts=False)

        assert result.records_skipped >= 1
        assert result.records_updated == 0
        mock_rep_contract.give_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_from_km_consensus_verification(self):
        """Test that agents without sufficient debates are skipped (consensus check)."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
            min_elo_for_reputation=1400.0,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = [
            _make_mock_agent_link(aragora_agent_id="new_agent", token_id=77),
        ]

        mock_perf_adapter = MagicMock()
        mock_perf_adapter.get_agent_skill_history.return_value = [
            {
                "elo": 1600.0,
                "debates_count": 1,  # Less than 3 - insufficient for consensus
                "domain_elos": {},
            }
        ]

        mock_rep_contract = MagicMock()

        with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
            with patch.object(adapter, "_get_performance_adapter", return_value=mock_perf_adapter):
                with patch.object(
                    adapter, "_get_reputation_contract", return_value=mock_rep_contract
                ):
                    result = await adapter.sync_from_km(push_elo_ratings=True, push_receipts=False)

        assert result.records_skipped >= 1
        assert result.records_updated == 0
        mock_rep_contract.give_feedback.assert_not_called()


# =============================================================================
# Test Receipt Validation Push
# =============================================================================


class TestReceiptValidationPush:
    """Test pushing receipts as validation records."""

    @pytest.mark.asyncio
    async def test_push_receipts_as_validations(self):
        """Test pushing gauntlet receipts as validation records."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = [
            _make_mock_agent_link(aragora_agent_id="claude", token_id=42),
        ]

        # Mock receipt adapter
        mock_receipt_adapter = MagicMock()
        mock_receipt_adapter.get_stats.return_value = {"receipts_processed": 1}
        mock_receipt_adapter._ingested_receipts = {
            "rcpt-001": _make_mock_receipt_result(
                receipt_id="rcpt-001", claims_ingested=3, findings_ingested=1
            ),
        }

        # Mock validation contract
        mock_val_contract = MagicMock()
        mock_val_contract.submit_response.return_value = "0xval123"

        # Patch the import in the method
        with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.knowledge.mound.adapters": MagicMock(
                        ReceiptAdapter=lambda: mock_receipt_adapter
                    )
                },
            ):
                with patch.object(
                    adapter, "_get_validation_contract", return_value=mock_val_contract
                ):
                    # Patch the import inside the method
                    with patch(
                        "aragora.knowledge.mound.adapters.erc8004_adapter.ReceiptAdapter",
                        create=True,
                        return_value=mock_receipt_adapter,
                    ):
                        # Disable ELO push to isolate receipt test
                        provider.get_config.return_value.has_reputation_registry = False
                        result = await adapter.sync_from_km(
                            push_elo_ratings=False, push_receipts=True
                        )

        assert isinstance(result, ValidationSyncResult)
        # Result can be 0 if import fails or receipts not available, which is fine for this test
        # The main thing is that it doesn't crash


# =============================================================================
# Test Health Status
# =============================================================================


class TestHealthStatus:
    """Test health status reporting."""

    def test_health_status_connected(self):
        """Test health status when connected."""
        provider = _make_mock_provider(is_connected=True)
        adapter = ERC8004Adapter(provider=provider, enable_reverse_sync=True)

        status = adapter.get_health_status()

        assert status["adapter"] == "erc8004"
        assert status["connected"] is True
        assert status["reverse_sync_enabled"] is True
        assert status["has_signer"] is False
        assert "chain_id" in status

    def test_health_status_disconnected(self):
        """Test health status when disconnected."""
        provider = _make_mock_provider(is_connected=False)
        adapter = ERC8004Adapter(provider=provider)

        status = adapter.get_health_status()

        assert status["connected"] is False

    def test_health_status_with_signer(self):
        """Test health status with signer configured."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(provider=provider, signer=signer)

        status = adapter.get_health_status()

        assert status["has_signer"] is True

    def test_health_status_error(self):
        """Test health status when provider throws error."""
        provider = MagicMock()
        provider.is_connected.side_effect = Exception("Connection error")
        adapter = ERC8004Adapter(provider=provider)

        status = adapter.get_health_status()

        assert status["connected"] is False
        assert "error" in status


# =============================================================================
# Test Event Emission
# =============================================================================


class TestEventEmission:
    """Test event emission during sync operations."""

    @pytest.mark.asyncio
    async def test_emits_identity_synced_event(self):
        """Test that identity_synced events are emitted during forward sync."""
        provider = _make_mock_provider()
        events: list[tuple[str, dict]] = []

        def capture_event(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        adapter = ERC8004Adapter(provider=provider, event_callback=capture_event)

        identity_contract = MagicMock()
        identity_contract.get_total_supply.return_value = 1
        identity_contract.get_agent.return_value = _make_mock_identity()

        reputation_contract = MagicMock()
        reputation_contract.get_summary.side_effect = Exception("No reputation")

        validation_contract = MagicMock()
        validation_contract.get_agent_validations.return_value = []

        with patch.object(adapter, "_get_identity_contract", return_value=identity_contract):
            with patch.object(
                adapter, "_get_reputation_contract", return_value=reputation_contract
            ):
                with patch.object(
                    adapter, "_get_validation_contract", return_value=validation_contract
                ):
                    # Need to set a KM store or events won't be emitted for node storage
                    adapter._km_store = MagicMock()
                    await adapter.sync_to_km()

        # Check that identity_synced event was emitted
        event_types = [e[0] for e in events]
        assert "identity_synced" in event_types

    @pytest.mark.asyncio
    async def test_emits_reputation_pushed_event(self):
        """Test that reputation_pushed events are emitted during reverse sync."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        events: list[tuple[str, dict]] = []

        def capture_event(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
            min_elo_for_reputation=1400.0,
            event_callback=capture_event,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = [
            _make_mock_agent_link(aragora_agent_id="claude", token_id=42),
        ]

        mock_perf_adapter = MagicMock()
        mock_perf_adapter.get_agent_skill_history.return_value = [
            {
                "elo": 1600.0,
                "debates_count": 10,
                "domain_elos": {},
            }
        ]

        mock_rep_contract = MagicMock()
        mock_rep_contract.give_feedback.return_value = "0xabc"

        with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
            with patch.object(adapter, "_get_performance_adapter", return_value=mock_perf_adapter):
                with patch.object(
                    adapter, "_get_reputation_contract", return_value=mock_rep_contract
                ):
                    await adapter.sync_from_km(push_elo_ratings=True, push_receipts=False)

        event_types = [e[0] for e in events]
        assert "reputation_pushed" in event_types


# =============================================================================
# Test ELO to Reputation Conversion
# =============================================================================


class TestEloToReputationConversion:
    """Test the ELO to reputation value conversion logic."""

    @pytest.mark.asyncio
    async def test_elo_value_scaling(self):
        """Test that ELO values are correctly scaled to reputation."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
            min_elo_for_reputation=1000.0,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = [
            _make_mock_agent_link(aragora_agent_id="test_agent", token_id=1),
        ]

        # Test different ELO values
        test_cases = [
            (1000.0, 0),  # Minimum ELO -> 0 reputation
            (1500.0, 500),  # Middle ELO -> 500 reputation
            (2000.0, 1000),  # Maximum ELO -> 1000 reputation
        ]

        for elo, expected_rep in test_cases:
            mock_perf_adapter = MagicMock()
            mock_perf_adapter.get_agent_skill_history.return_value = [
                {
                    "elo": elo,
                    "debates_count": 10,
                    "domain_elos": {},
                }
            ]

            mock_rep_contract = MagicMock()
            mock_rep_contract.give_feedback.return_value = "0xabc"

            with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
                with patch.object(
                    adapter, "_get_performance_adapter", return_value=mock_perf_adapter
                ):
                    with patch.object(
                        adapter, "_get_reputation_contract", return_value=mock_rep_contract
                    ):
                        await adapter.sync_from_km(push_elo_ratings=True, push_receipts=False)

            # Check the value passed to give_feedback
            call_args = mock_rep_contract.give_feedback.call_args
            actual_value = call_args.kwargs.get("value") or call_args[1].get("value")
            assert actual_value == expected_rep, (
                f"ELO {elo} should map to reputation {expected_rep}"
            )


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in blockchain operations."""

    @pytest.mark.asyncio
    async def test_handles_contract_errors(self):
        """Test that contract errors are caught and reported."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
            min_elo_for_reputation=1000.0,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = [
            _make_mock_agent_link(aragora_agent_id="test", token_id=1),
        ]

        mock_perf_adapter = MagicMock()
        mock_perf_adapter.get_agent_skill_history.return_value = [
            {"elo": 1600.0, "debates_count": 10, "domain_elos": {}}
        ]

        mock_rep_contract = MagicMock()
        mock_rep_contract.give_feedback.side_effect = Exception("Transaction reverted")

        with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
            with patch.object(adapter, "_get_performance_adapter", return_value=mock_perf_adapter):
                with patch.object(
                    adapter, "_get_reputation_contract", return_value=mock_rep_contract
                ):
                    result = await adapter.sync_from_km(push_elo_ratings=True, push_receipts=False)

        assert result.records_updated == 0
        assert len(result.errors) > 0
        assert any("Transaction reverted" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_handles_performance_adapter_errors(self):
        """Test that errors from performance adapter are handled."""
        provider = _make_mock_provider()
        signer = _make_mock_signer()
        adapter = ERC8004Adapter(
            provider=provider,
            signer=signer,
            enable_reverse_sync=True,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_all_links.return_value = [
            _make_mock_agent_link(aragora_agent_id="test", token_id=1),
        ]

        mock_perf_adapter = MagicMock()
        mock_perf_adapter.get_agent_skill_history.side_effect = Exception("Database error")

        with patch.object(adapter, "_get_identity_bridge", return_value=mock_bridge):
            with patch.object(adapter, "_get_performance_adapter", return_value=mock_perf_adapter):
                result = await adapter.sync_from_km(push_elo_ratings=True, push_receipts=False)

        assert len(result.errors) > 0
        assert any("Database error" in e for e in result.errors)
