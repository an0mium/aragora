# Receipt Contract

Canonical receipt contract for new integrations:
- `aragora.receipts.DecisionReceipt`

Implementation target:
- `aragora.gauntlet.receipt.DecisionReceipt`

Legacy compatibility model (still used by some handlers):
- `aragora.export.decision_receipt.DecisionReceipt`

## Import Guidance

Use this for new code:

```python
from aragora.receipts import DecisionReceipt
```

Use this only for legacy compatibility/migration:

```python
from aragora.receipts import LegacyDecisionReceipt
```

## Why

Historically the repo had two `DecisionReceipt` models with overlapping purpose.
This contract makes the canonical path explicit while preserving existing imports.
