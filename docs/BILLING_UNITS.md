# Billing Units and Token Metering

This document describes how Aragora meters and bills token usage for subscription-based pricing.

## Overview

Aragora uses a **1K token unit** system for usage-based billing. This provides:
- Predictable billing increments
- Efficient metering (fewer API calls)
- Fair handling of remainder tokens

## Billing Unit Size

| Metric | Unit Size | Reasoning |
|--------|-----------|-----------|
| Input Tokens | 1,000 tokens | Standard LLM pricing granularity |
| Output Tokens | 1,000 tokens | Standard LLM pricing granularity |

## Metering Strategy

### Regular Sync (Background Job)

The `UsageSyncService.sync_all()` runs periodically (default: every 5 minutes) and:

1. **Calculates delta tokens** since last sync:
   ```
   delta = current_cumulative - last_synced_cumulative
   ```

2. **Bills in 1K unit increments**:
   ```
   billable_units = delta // 1000
   billable_tokens = billable_units * 1000
   ```

3. **Preserves remainders** for next sync:
   ```
   # Watermark advances only by billed amount, not full delta
   new_watermark = last_watermark + billable_tokens
   ```

### End-of-Period Flush

The `UsageSyncService.flush_period()` method is called at billing period boundaries to bill all remaining tokens:

1. **Bills all accumulated usage** (including sub-1K remainders)
2. **Resets watermarks** for new billing period
3. **Triggered by**:
   - Period transition detection in `sync_all()`
   - Manual trigger via admin API
   - Subscription cancellation

## Example

Consider a user who has used 2,547 input tokens since last sync:

**Regular Sync:**
```
delta = 2,547 tokens
billable_units = 2,547 // 1,000 = 2 units
billable_tokens = 2 * 1,000 = 2,000 tokens

# Reports: 2 units to billing
# Watermark advances by: 2,000 tokens
# Remainder preserved: 547 tokens
```

**Next Sync (user has used 800 more tokens):**
```
new_delta = (3,347 total) - (2,000 synced) = 1,347 tokens
billable_units = 1,347 // 1,000 = 1 unit
billable_tokens = 1,000 tokens

# Reports: 1 unit to billing
# Watermark advances by: 1,000 tokens
# Remainder preserved: 347 tokens
```

**End of Period Flush:**
```
remaining = (total cumulative) - (last synced)
# All remaining tokens billed, watermarks reset
```

## Implementation Details

### Key Files

| File | Purpose |
|------|---------|
| `aragora/billing/usage_sync.py` | Main sync service |
| `aragora/billing/stripe_client.py` | Stripe integration |
| `aragora/billing/models.py` | Usage data models |

### Configuration

```python
UsageSyncConfig(
    min_tokens_threshold=1000,  # Minimum tokens for reporting
    sync_interval_seconds=300,   # 5-minute sync interval
)
```

### Sync Methods

```python
# Regular background sync
sync_service.sync_all()

# Manual flush for period end
sync_service.flush_period(subscription_id)
```

## Period Transition Detection

The sync service automatically detects billing period transitions by:

1. Checking subscription `current_period_end` timestamp
2. If current time > period end, triggers flush before normal sync
3. Prevents double-billing by tracking `last_period_end`

## Considerations

### Why 1K Unit Size?

- **Industry standard**: Most LLM providers bill in 1K token increments
- **Efficient metering**: Reduces API calls to billing provider
- **Predictable costs**: Users can estimate costs easily

### Handling Remainders

- Remainders accumulate across sync cycles
- Never lost, always billed eventually
- Period-end flush ensures all usage is captured

### Edge Cases

| Case | Handling |
|------|----------|
| Sub-1K usage | Accumulates until 1K threshold or period end |
| Subscription cancellation | Immediate flush of all remaining usage |
| Failed sync | Retry with exponential backoff |
| Provider downtime | Queue usage, sync on recovery |

## Monitoring

Key metrics to track:

```
aragora_billing_sync_total{status="success|failed"}
aragora_billing_tokens_synced{type="input|output"}
aragora_billing_remainder_tokens{type="input|output"}
aragora_billing_flush_total{reason="period_end|cancellation"}
```

## Admin Operations

### Manual Flush

```bash
# Via CLI
aragora billing flush --subscription-id sub_xxx

# Via API
POST /api/admin/billing/flush
{"subscription_id": "sub_xxx"}
```

### View Usage

```bash
# Current period usage
aragora billing usage --subscription-id sub_xxx

# Historical usage
aragora billing history --subscription-id sub_xxx --period 2026-01
```
