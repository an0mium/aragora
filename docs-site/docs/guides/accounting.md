---
title: Accounting & Financial Automation
description: Accounting & Financial Automation
---

# Accounting & Financial Automation

Aragora provides comprehensive accounting automation for SMEs with QuickBooks Online integration.

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| Expense Tracking | Receipt OCR, categorization, duplicate detection | Stable |
| Invoice Processing | PDF extraction, PO matching, anomaly detection | Stable |
| AR Automation | Invoice generation, payment reminders, aging | Stable |
| AP Automation | Payment optimization, batching, forecasting | Stable |
| QBO Integration | Full OAuth 2.0 connector, sync support | Stable |
| Gusto Integration | Payroll runs, employees, journal entries | Stable |

## Quick Start

### Expense Tracking

```python
from aragora.services.expense_tracker import ExpenseTracker

tracker = ExpenseTracker()

# Process receipt image/PDF
expense = await tracker.process_receipt(
    image_data=receipt_bytes,
    employee_id="emp_001",
)

# Auto-categorize expense
category = await tracker.categorize_expense(expense)

# Sync to QuickBooks
result = await tracker.sync_to_qbo([expense.id])
```

### Invoice Processing

```python
from aragora.services.invoice_processor import InvoiceProcessor

processor = InvoiceProcessor()

# Extract invoice from PDF
invoice = await processor.extract_invoice(pdf_bytes)

# Match to Purchase Order
match = await processor.match_to_po(invoice)

# Detect anomalies (duplicates, unusual amounts, new vendors)
anomalies = await processor.detect_anomalies(invoice)

# Route for approval
if invoice.requires_approval:
    await processor.route_for_approval(invoice.id)
```

### AR Automation

```python
from aragora.services.ar_automation import ARAutomation

ar = ARAutomation()

# Generate invoice
invoice = await ar.generate_invoice(
    customer_id="cust_001",
    customer_name="Acme Corp",
    line_items=[
        {"description": "Consulting", "quantity": 10, "unit_price": 150.00, "amount": 1500.00}
    ],
)

# Send to customer
await ar.send_invoice(invoice.id)

# Send payment reminder (escalating)
await ar.send_payment_reminder(invoice.id, escalation_level=1)

# Get aging report
aging = await ar.track_aging()

# Get collection suggestions
suggestions = await ar.suggest_collections()
```

### AP Automation

```python
from aragora.services.ap_automation import APAutomation

ap = APAutomation()

# Add payable invoice
invoice = await ap.add_invoice(
    vendor_id="vendor_001",
    vendor_name="Office Supplies Co",
    total_amount=1000.00,
    early_pay_discount=0.02,  # 2% discount
    discount_deadline=datetime.now() + timedelta(days=10),
)

# Optimize payment timing
schedule = await ap.optimize_payment_timing(
    invoices=[invoice],
    available_cash=Decimal("5000.00"),
    prioritize_discounts=True,
)

# Batch payments
batch = await ap.batch_payments([invoice1, invoice2])

# Forecast cash needs
forecast = await ap.forecast_cash_needs(days_ahead=30)
```

## API Endpoints

### Expense Tracking

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/accounting/expenses/upload` | Upload and process receipt |
| POST | `/api/v1/accounting/expenses` | Create expense manually |
| GET | `/api/v1/accounting/expenses` | List expenses with filters |
| GET | `/api/v1/accounting/expenses/\{id\}` | Get expense by ID |
| PUT | `/api/v1/accounting/expenses/\{id\}` | Update expense |
| DELETE | `/api/v1/accounting/expenses/\{id\}` | Delete expense |
| POST | `/api/v1/accounting/expenses/\{id\}/approve` | Approve expense |
| POST | `/api/v1/accounting/expenses/\{id\}/reject` | Reject expense |
| POST | `/api/v1/accounting/expenses/categorize` | Auto-categorize |
| POST | `/api/v1/accounting/expenses/sync` | Sync to QBO |
| GET | `/api/v1/accounting/expenses/stats` | Get statistics |

### Invoice Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/accounting/invoices/upload` | Upload and extract invoice |
| POST | `/api/v1/accounting/invoices` | Create invoice manually |
| GET | `/api/v1/accounting/invoices` | List invoices |
| GET | `/api/v1/accounting/invoices/\{id\}` | Get invoice by ID |
| POST | `/api/v1/accounting/invoices/\{id\}/approve` | Approve invoice |
| POST | `/api/v1/accounting/invoices/\{id\}/reject` | Reject invoice |
| POST | `/api/v1/accounting/invoices/\{id\}/match-po` | Match to PO |
| POST | `/api/v1/accounting/invoices/\{id\}/schedule` | Schedule payment |

### AR Automation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/accounting/ar/invoices` | Create AR invoice |
| GET | `/api/v1/accounting/ar/invoices` | List AR invoices |
| GET | `/api/v1/accounting/ar/invoices/\{id\}` | Get invoice |
| POST | `/api/v1/accounting/ar/invoices/\{id\}/send` | Send to customer |
| POST | `/api/v1/accounting/ar/invoices/\{id\}/reminder` | Send reminder |
| POST | `/api/v1/accounting/ar/invoices/\{id\}/payment` | Record payment |
| GET | `/api/v1/accounting/ar/aging` | Get aging report |
| GET | `/api/v1/accounting/ar/collections` | Get collection suggestions |
| POST | `/api/v1/accounting/ar/customers` | Add customer |
| GET | `/api/v1/accounting/ar/customers/\{id\}/balance` | Get customer balance |

### AP Automation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/accounting/ap/invoices` | Add payable invoice |
| GET | `/api/v1/accounting/ap/invoices` | List payables |
| GET | `/api/v1/accounting/ap/invoices/\{id\}` | Get invoice |
| POST | `/api/v1/accounting/ap/invoices/\{id\}/payment` | Record payment |
| POST | `/api/v1/accounting/ap/optimize` | Optimize payment timing |
| POST | `/api/v1/accounting/ap/batch` | Create batch payment |
| GET | `/api/v1/accounting/ap/forecast` | Get cash flow forecast |
| GET | `/api/v1/accounting/ap/discounts` | Get discount opportunities |

## QuickBooks Integration

### Setup

1. Create a QuickBooks Developer account
2. Create an app and get credentials
3. Set environment variables:

```bash
export QBO_CLIENT_ID="your-client-id"
export QBO_CLIENT_SECRET="your-client-secret"
export QBO_REDIRECT_URI="https://yourapp.com/callback"
export QBO_ENVIRONMENT="sandbox"  # or "production"
```

### OAuth Flow

```python
from aragora.connectors.accounting.qbo import QuickBooksConnector

qbo = QuickBooksConnector()

# Get authorization URL
auth_url = qbo.get_auth_url()
# Redirect user to auth_url...

# Exchange code for tokens (after callback)
credentials = await qbo.exchange_code(auth_code, realm_id)

# Use connector
customers = await qbo.list_customers()
```

### Sync Operations

```python
# Create expense in QBO
qbo_expense = await qbo.create_expense(
    vendor_id="123",
    account_id="456",
    amount=100.00,
    description="Office supplies",
)

# Create bill (AP)
qbo_bill = await qbo.create_bill(
    vendor_id="123",
    account_id="789",
    amount=500.00,
    due_date=datetime.now() + timedelta(days=30),
)

# Create invoice (AR)
qbo_invoice = await qbo.create_invoice(
    customer_id="456",
    line_items=[...],
)
```

## Gusto Payroll Integration

### Setup

1. Create a Gusto Developer account and OAuth app
2. Configure redirect URIs in the Gusto developer console
3. Set environment variables:

```bash
GUSTO_CLIENT_ID="your-client-id"
GUSTO_CLIENT_SECRET="your-client-secret"
GUSTO_REDIRECT_URI="https://yourapp.com/api/accounting/gusto/callback"
```

### OAuth Flow

```python
from aragora.connectors.accounting.gusto import GustoConnector

gusto = GustoConnector()
auth_url = gusto.get_auth_url()
# Redirect user to auth_url...
```

### Payroll Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/accounting/gusto/status` | Connection status |
| GET | `/api/accounting/gusto/employees` | List employees |
| GET | `/api/accounting/gusto/payrolls` | List payroll runs |
| GET | `/api/accounting/gusto/payrolls/\{payroll_id\}` | Payroll run details |
| POST | `/api/accounting/gusto/payrolls/\{payroll_id\}/journal-entry` | Generate journal entry |

## OCR & Document Processing

The system uses `pdfplumber` for PDF text extraction and optional `pytesseract` for image OCR.

### Supported Formats

- PDF (native text extraction + table parsing)
- PNG/JPEG images (requires pytesseract)

### Receipt Parsing

Extracts:
- Vendor name (from header)
- Total amount
- Tax amount
- Tip amount
- Date
- Line items

### Invoice Parsing

Extracts:
- Vendor name
- Invoice number
- PO number
- Invoice date
- Due date
- Subtotal, tax, total
- Payment terms
- Line items (from tables or text)

## Expense Categorization

### Pattern-Based Matching

```python
VENDOR_PATTERNS = {
    "airline": ExpenseCategory.TRAVEL,
    "uber": ExpenseCategory.TRAVEL,
    "starbucks": ExpenseCategory.MEALS,
    "office depot": ExpenseCategory.OFFICE_SUPPLIES,
    ...
}
```

### LLM Categorization

When pattern matching fails, the system can use LLM for intelligent categorization:

```bash
# Configure API key (Anthropic preferred, OpenAI fallback)
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

## Anomaly Detection

### Invoice Anomalies

| Type | Description | Threshold |
|------|-------------|-----------|
| `duplicate` | Same vendor, amount, date | Exact match |
| `unusual_amount` | Significantly higher than average | 3x average |
| `round_amount` | Suspiciously round numbers | $1000, $5000, $10000 |
| `new_vendor` | First invoice from vendor | N/A |
| `po_mismatch` | Amount doesn't match PO | >10% variance |

### Duplicate Detection

- Hash-based exact matching
- Fuzzy matching within date tolerance
- Amount tolerance comparison

## Approval Workflows

### Approval Levels

| Level | Amount Threshold |
|-------|------------------|
| AUTO | < $500 |
| MANAGER | $500 - $5,000 |
| DIRECTOR | $5,000 - $10,000 |
| EXECUTIVE | > $10,000 |

### Approval Flow

1. Invoice extracted/created
2. Anomalies detected
3. Approval level determined
4. Routed to appropriate approver
5. Approved → scheduled for payment
6. Rejected → returned with reason

## AR Collection Workflow

### Reminder Levels

| Level | Tone | Days Overdue |
|-------|------|--------------|
| FRIENDLY | Polite reminder | 1-15 |
| FIRM | Firmer language | 16-30 |
| URGENT | Urgent notice | 31-60 |
| FINAL | Final notice before collections | 60+ |

### Collection Suggestions

Based on:
- Days overdue
- Amount outstanding
- Customer payment history
- Number of reminders sent

## Performance Considerations

- In-memory storage (for development)
- For production, use database persistence
- Batch operations for bulk processing
- Rate limiting for QBO API calls

## Error Handling

All services return appropriate error responses:

```python
try:
    expense = await tracker.process_receipt(data)
except OCRExtractionError:
    # Handle OCR failure
except QBOSyncError:
    # Handle QBO sync failure
except ValidationError:
    # Handle validation errors
```

## Testing

```bash
# Run accounting tests
pytest tests/test_expense_tracker.py -v
pytest tests/test_invoice_processor.py -v
pytest tests/test_ar_automation.py -v
pytest tests/test_ap_automation.py -v
```
