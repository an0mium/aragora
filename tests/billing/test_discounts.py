"""
Tests for discount codes and volume pricing.

Tests cover:
- Creating discount codes
- Validating codes
- Applying codes
- Volume discount tiers
- Code expiration and limits
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from decimal import Decimal
import tempfile
import pytest

from aragora.billing.discounts import (
    DiscountType,
    DiscountCodeStatus,
    DiscountCode,
    DiscountUsage,
    VolumeTier,
    VolumeDiscount,
    ApplyCodeResult,
    DiscountManager,
)


@pytest.fixture
def discount_manager():
    """Create a discount manager with temp database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "discounts.db"
        manager = DiscountManager(db_path=str(db_path))
        yield manager


class TestDiscountCodeCreation:
    """Tests for creating discount codes."""

    @pytest.mark.asyncio
    async def test_create_percentage_code(self, discount_manager):
        """Can create a percentage discount code."""
        code = await discount_manager.create_code(
            code="WELCOME20",
            discount_percent=20.0,
            description="Welcome discount",
        )

        assert code.code == "WELCOME20"
        assert code.discount_type == DiscountType.PERCENTAGE
        assert code.discount_percent == 20.0
        assert code.status == DiscountCodeStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_create_fixed_amount_code(self, discount_manager):
        """Can create a fixed amount discount code."""
        code = await discount_manager.create_code(
            code="SAVE50",
            discount_amount_cents=5000,
            description="$50 off",
        )

        assert code.code == "SAVE50"
        assert code.discount_type == DiscountType.FIXED_AMOUNT
        assert code.discount_amount_cents == 5000

    @pytest.mark.asyncio
    async def test_create_code_with_expiration(self, discount_manager):
        """Can create a code with expiration date."""
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        code = await discount_manager.create_code(
            code="SUMMER30",
            discount_percent=30.0,
            expires_at=expires,
        )

        assert code.expires_at is not None
        assert code.expires_at == expires

    @pytest.mark.asyncio
    async def test_create_code_with_max_uses(self, discount_manager):
        """Can create a code with usage limit."""
        code = await discount_manager.create_code(
            code="LIMITED100",
            discount_percent=25.0,
            max_uses=100,
        )

        assert code.max_uses == 100

    @pytest.mark.asyncio
    async def test_create_code_uppercase(self, discount_manager):
        """Code is stored uppercase."""
        code = await discount_manager.create_code(
            code="lowercase",
            discount_percent=10.0,
        )

        assert code.code == "LOWERCASE"

    @pytest.mark.asyncio
    async def test_create_duplicate_code_fails(self, discount_manager):
        """Cannot create duplicate codes."""
        await discount_manager.create_code(code="UNIQUE", discount_percent=10.0)

        with pytest.raises(ValueError, match="already exists"):
            await discount_manager.create_code(code="UNIQUE", discount_percent=20.0)

    @pytest.mark.asyncio
    async def test_create_code_invalid_percent(self, discount_manager):
        """Cannot create code with invalid percent."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            await discount_manager.create_code(code="INVALID", discount_percent=150.0)

    @pytest.mark.asyncio
    async def test_create_code_no_discount(self, discount_manager):
        """Cannot create code with no discount."""
        with pytest.raises(ValueError, match="Must specify"):
            await discount_manager.create_code(code="NODISCOUNT")


class TestCodeValidation:
    """Tests for validating discount codes."""

    @pytest.mark.asyncio
    async def test_validate_valid_code(self, discount_manager):
        """Valid code passes validation."""
        await discount_manager.create_code(code="VALID20", discount_percent=20.0)

        result = await discount_manager.validate_code(
            code="VALID20",
            org_id="org_123",
            purchase_amount_cents=10000,
        )

        assert result.valid is True
        assert result.discount_percent == 20.0
        assert result.discount_amount_cents == 2000  # 20% of 10000

    @pytest.mark.asyncio
    async def test_validate_nonexistent_code(self, discount_manager):
        """Nonexistent code fails validation."""
        result = await discount_manager.validate_code(
            code="NOTREAL",
            org_id="org_123",
        )

        assert result.valid is False
        assert result.error_code == "CODE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_validate_expired_code(self, discount_manager):
        """Expired code fails validation."""
        expired = datetime.now(timezone.utc) - timedelta(days=1)
        await discount_manager.create_code(
            code="EXPIRED",
            discount_percent=10.0,
            expires_at=expired,
        )

        result = await discount_manager.validate_code(
            code="EXPIRED",
            org_id="org_123",
        )

        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_min_purchase(self, discount_manager):
        """Code with minimum purchase requirement."""
        await discount_manager.create_code(
            code="MINPURCHASE",
            discount_percent=15.0,
            min_purchase_cents=5000,
        )

        # Below minimum
        result = await discount_manager.validate_code(
            code="MINPURCHASE",
            org_id="org_123",
            purchase_amount_cents=4000,
        )
        assert result.valid is False
        assert result.error_code == "MIN_PURCHASE_NOT_MET"

        # Above minimum
        result = await discount_manager.validate_code(
            code="MINPURCHASE",
            org_id="org_123",
            purchase_amount_cents=6000,
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validate_tier_restriction(self, discount_manager):
        """Code restricted to specific tiers."""
        await discount_manager.create_code(
            code="ENTERPRISE",
            discount_percent=25.0,
            eligible_tiers=["enterprise", "enterprise_plus"],
        )

        # Wrong tier
        result = await discount_manager.validate_code(
            code="ENTERPRISE",
            org_id="org_123",
            tier="professional",
        )
        assert result.valid is False
        assert result.error_code == "TIER_NOT_ELIGIBLE"

        # Correct tier
        result = await discount_manager.validate_code(
            code="ENTERPRISE",
            org_id="org_123",
            tier="enterprise",
        )
        assert result.valid is True


class TestCodeApplication:
    """Tests for applying discount codes."""

    @pytest.mark.asyncio
    async def test_apply_code(self, discount_manager):
        """Can apply a discount code."""
        await discount_manager.create_code(code="APPLY20", discount_percent=20.0)

        result = await discount_manager.apply_code(
            code="APPLY20",
            org_id="org_123",
            purchase_amount_cents=10000,
        )

        assert result.valid is True
        assert result.discount_amount_cents == 2000

    @pytest.mark.asyncio
    async def test_apply_code_records_usage(self, discount_manager):
        """Applying code records usage."""
        code = await discount_manager.create_code(code="TRACK", discount_percent=10.0)

        await discount_manager.apply_code(
            code="TRACK",
            org_id="org_123",
            purchase_amount_cents=10000,
        )

        usage = await discount_manager.get_code_usage(code.id)
        assert len(usage) == 1
        assert usage[0].org_id == "org_123"
        assert usage[0].discount_cents == 1000

    @pytest.mark.asyncio
    async def test_apply_code_increments_total_uses(self, discount_manager):
        """Applying code increments usage counter."""
        await discount_manager.create_code(code="COUNT", discount_percent=10.0)

        await discount_manager.apply_code(
            code="COUNT",
            org_id="org_1",
            purchase_amount_cents=5000,
        )
        await discount_manager.apply_code(
            code="COUNT",
            org_id="org_2",
            purchase_amount_cents=5000,
        )

        code = await discount_manager.get_code("COUNT")
        assert code.total_uses == 2

    @pytest.mark.asyncio
    async def test_apply_code_max_uses_per_org(self, discount_manager):
        """Organization can only use code max_uses_per_org times."""
        await discount_manager.create_code(
            code="ONEUSE",
            discount_percent=10.0,
            max_uses_per_org=1,
        )

        # First use succeeds
        result = await discount_manager.apply_code(
            code="ONEUSE",
            org_id="org_123",
            purchase_amount_cents=5000,
        )
        assert result.valid is True

        # Second use fails
        result = await discount_manager.apply_code(
            code="ONEUSE",
            org_id="org_123",
            purchase_amount_cents=5000,
        )
        assert result.valid is False
        assert result.error_code == "MAX_USES_PER_ORG"

    @pytest.mark.asyncio
    async def test_apply_code_exhausted(self, discount_manager):
        """Code becomes exhausted after max uses."""
        await discount_manager.create_code(
            code="LIMITED",
            discount_percent=10.0,
            max_uses=2,
            max_uses_per_org=2,
        )

        await discount_manager.apply_code(
            code="LIMITED",
            org_id="org_1",
            purchase_amount_cents=5000,
        )
        await discount_manager.apply_code(
            code="LIMITED",
            org_id="org_2",
            purchase_amount_cents=5000,
        )

        code = await discount_manager.get_code("LIMITED")
        assert code.status == DiscountCodeStatus.EXHAUSTED


class TestVolumeDiscounts:
    """Tests for volume-based discounts."""

    @pytest.mark.asyncio
    async def test_get_volume_discount_new_org(self, discount_manager):
        """New org gets default volume tiers."""
        volume = await discount_manager.get_volume_discount("new_org")

        assert volume.org_id == "new_org"
        assert len(volume.tiers) > 0
        assert volume.cumulative_spend_cents == 0
        assert volume.current_discount_percent == 0.0

    @pytest.mark.asyncio
    async def test_update_volume_spend(self, discount_manager):
        """Updating spend calculates correct tier."""
        # Default tiers: $10k=5%, $50k=10%, $100k=15%, $500k=20%

        # Spend below first tier
        volume = await discount_manager.update_volume_spend("org_123", 5000_00)
        assert volume.current_discount_percent == 0.0

        # Spend at first tier
        volume = await discount_manager.update_volume_spend("org_123", 5000_00)
        assert volume.cumulative_spend_cents == 10000_00
        assert volume.current_discount_percent == 5.0

        # Spend at second tier
        volume = await discount_manager.update_volume_spend("org_123", 40000_00)
        assert volume.cumulative_spend_cents == 50000_00
        assert volume.current_discount_percent == 10.0

    def test_volume_discount_calculate(self):
        """VolumeDiscount calculates correct tier."""
        tiers = [
            VolumeTier(min_spend_cents=1000_00, discount_percent=5.0),
            VolumeTier(min_spend_cents=5000_00, discount_percent=10.0),
        ]
        volume = VolumeDiscount(
            org_id="test",
            tiers=tiers,
            cumulative_spend_cents=3000_00,
        )

        discount = volume.calculate_discount()
        assert discount == 5.0
        assert volume.current_discount_percent == 5.0


class TestCodeManagement:
    """Tests for code management operations."""

    @pytest.mark.asyncio
    async def test_list_codes(self, discount_manager):
        """Can list all codes."""
        await discount_manager.create_code(code="CODE1", discount_percent=10.0)
        await discount_manager.create_code(code="CODE2", discount_percent=20.0)

        codes = await discount_manager.list_codes()
        assert len(codes) == 2

    @pytest.mark.asyncio
    async def test_list_codes_by_status(self, discount_manager):
        """Can filter codes by status."""
        await discount_manager.create_code(code="ACTIVE1", discount_percent=10.0)
        await discount_manager.create_code(code="ACTIVE2", discount_percent=20.0)
        await discount_manager.create_code(code="TODISABLE", discount_percent=30.0)
        await discount_manager.disable_code("TODISABLE")

        active_codes = await discount_manager.list_codes(status=DiscountCodeStatus.ACTIVE)
        assert len(active_codes) == 2

        disabled_codes = await discount_manager.list_codes(status=DiscountCodeStatus.DISABLED)
        assert len(disabled_codes) == 1

    @pytest.mark.asyncio
    async def test_disable_code(self, discount_manager):
        """Can disable a code."""
        await discount_manager.create_code(code="TODISABLE", discount_percent=10.0)

        result = await discount_manager.disable_code("TODISABLE")
        assert result is True

        code = await discount_manager.get_code("TODISABLE")
        assert code.status == DiscountCodeStatus.DISABLED


class TestDiscountCodeDataclass:
    """Tests for DiscountCode dataclass."""

    def test_is_valid_active(self):
        """Active code with no restrictions is valid."""
        code = DiscountCode(
            code="TEST",
            discount_type=DiscountType.PERCENTAGE,
            discount_percent=10.0,
        )
        assert code.is_valid is True

    def test_is_valid_expired(self):
        """Expired code is not valid."""
        code = DiscountCode(
            code="TEST",
            discount_type=DiscountType.PERCENTAGE,
            discount_percent=10.0,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert code.is_valid is False

    def test_is_valid_not_yet_active(self):
        """Code not yet active is not valid."""
        code = DiscountCode(
            code="TEST",
            discount_type=DiscountType.PERCENTAGE,
            discount_percent=10.0,
            valid_from=datetime.now(timezone.utc) + timedelta(days=1),
        )
        assert code.is_valid is False

    def test_is_valid_exhausted(self):
        """Exhausted code is not valid."""
        code = DiscountCode(
            code="TEST",
            discount_type=DiscountType.PERCENTAGE,
            discount_percent=10.0,
            max_uses=10,
            total_uses=10,
        )
        assert code.is_valid is False

    def test_to_dict(self):
        """Code converts to dict."""
        code = DiscountCode(
            code="TEST",
            discount_type=DiscountType.PERCENTAGE,
            discount_percent=15.0,
        )
        data = code.to_dict()

        assert data["code"] == "TEST"
        assert data["discount_type"] == "percentage"
        assert data["discount_percent"] == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
