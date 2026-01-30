# HubSpot Connector Guide

This guide covers the HubSpot CRM connector for managing contacts, companies, deals, and engagements.

## Overview

The `HubSpotConnector` provides integration with HubSpot CRM API:

- Contact management (create, update, search)
- Company management
- Deal pipeline tracking
- Engagements (emails, calls, meetings, notes)
- Associations between objects
- Owner/user management

**Location:** `aragora/connectors/crm/hubspot.py`

---

## Quick Start

```python
from aragora.connectors.crm.hubspot import (
    HubSpotConnector,
    HubSpotCredentials,
)

credentials = HubSpotCredentials(
    access_token="pat-na1-xxx",
)

async with HubSpotConnector(credentials) as hubspot:
    # Create a contact
    contact = await hubspot.create_contact(
        email="john@example.com",
        first_name="John",
        last_name="Doe",
    )
    print(f"Created contact: {contact.id}")
```

---

## Configuration

### Credentials

```python
from dataclasses import dataclass

@dataclass
class HubSpotCredentials:
    access_token: str  # Private app access token
    base_url: str = "https://api.hubapi.com"
```

### Environment Variables

```bash
HUBSPOT_ACCESS_TOKEN=pat-na1-xxx
```

### Getting Your Access Token

1. Go to **Settings** > **Integrations** > **Private Apps** in HubSpot
2. Create a new private app
3. Select the required scopes:
   - `crm.objects.contacts.read/write`
   - `crm.objects.companies.read/write`
   - `crm.objects.deals.read/write`
4. Copy the access token

---

## Data Models

### Contact

```python
@dataclass
class Contact:
    id: str
    email: str | None
    first_name: str | None
    last_name: str | None
    phone: str | None
    company: str | None
    job_title: str | None
    lifecycle_stage: str | None
    lead_status: str | None
    owner_id: str | None
    properties: dict[str, Any]
    created_at: datetime | None
    updated_at: datetime | None
    archived: bool

    @property
    def full_name(self) -> str:
        """Returns 'First Last'"""
```

### Company

```python
@dataclass
class Company:
    id: str
    name: str | None
    domain: str | None
    industry: str | None
    phone: str | None
    city: str | None
    state: str | None
    country: str | None
    num_employees: int | None
    annual_revenue: Decimal | None
    lifecycle_stage: str | None
    owner_id: str | None
    properties: dict[str, Any]
    created_at: datetime | None
    updated_at: datetime | None
    archived: bool
```

### Deal

```python
@dataclass
class Deal:
    id: str
    name: str | None
    amount: Decimal | None
    stage: str | None
    pipeline: str | None
    close_date: datetime | None
    deal_type: str | None
    owner_id: str | None
    priority: str | None
    properties: dict[str, Any]
    created_at: datetime | None
    updated_at: datetime | None
    archived: bool
```

### Engagement

```python
@dataclass
class Engagement:
    id: str
    type: EngagementType  # EMAIL, CALL, MEETING, NOTE, TASK
    owner_id: str | None
    timestamp: datetime | None
    subject: str | None
    body: str | None
    direction: str | None  # INBOUND or OUTBOUND
    duration_ms: int | None
    status: str | None
    associated_contact_ids: list[str]
    associated_company_ids: list[str]
    associated_deal_ids: list[str]
    created_at: datetime | None
```

### Pipeline

```python
@dataclass
class Pipeline:
    id: str
    label: str
    display_order: int
    active: bool
    stages: list[PipelineStage]

@dataclass
class PipelineStage:
    id: str
    label: str
    display_order: int
    probability: float
    closed_won: bool
```

---

## Contact Operations

### Create Contact

```python
contact = await hubspot.create_contact(
    email="john@example.com",
    first_name="John",
    last_name="Doe",
    phone="+1234567890",
    company="Acme Inc",
    job_title="Software Engineer",
    lifecycle_stage="lead",
    owner_id="12345",
    custom_properties={
        "preferred_language": "en",
        "source": "website",
    },
)
```

### Get Contact

```python
# Get by ID
contact = await hubspot.get_contact("123456")

# Get with specific properties
contact = await hubspot.get_contact(
    "123456",
    properties=["email", "firstname", "lastname", "company"],
)
```

### Update Contact

```python
contact = await hubspot.update_contact(
    contact_id="123456",
    properties={
        "lifecyclestage": "customer",
        "phone": "+1987654321",
    },
)
```

### Delete Contact

```python
# Soft delete (archive)
await hubspot.delete_contact("123456")
```

### List Contacts

```python
# Get first page
contacts, next_after = await hubspot.get_contacts(limit=100)

# Paginate
while next_after:
    more_contacts, next_after = await hubspot.get_contacts(
        limit=100,
        after=next_after,
    )
    contacts.extend(more_contacts)
```

### Search Contacts

```python
# Simple text search
contacts = await hubspot.search_contacts(query="john")

# Filter search
contacts = await hubspot.search_contacts(
    filters=[
        {"propertyName": "email", "operator": "CONTAINS_TOKEN", "value": "@example.com"},
        {"propertyName": "lifecyclestage", "operator": "EQ", "value": "lead"},
    ],
    limit=50,
)
```

#### Search Operators

| Operator | Description |
|----------|-------------|
| `EQ` | Equals |
| `NEQ` | Not equals |
| `LT` / `LTE` | Less than / or equal |
| `GT` / `GTE` | Greater than / or equal |
| `CONTAINS_TOKEN` | Contains word |
| `NOT_CONTAINS_TOKEN` | Does not contain word |
| `HAS_PROPERTY` | Property has a value |
| `NOT_HAS_PROPERTY` | Property is empty |

---

## Company Operations

### Create Company

```python
company = await hubspot.create_company(
    name="Acme Corporation",
    domain="acme.com",
    industry="Technology",
    phone="+1234567890",
    city="San Francisco",
    state="CA",
    country="USA",
    num_employees=500,
    annual_revenue=Decimal("10000000"),
    owner_id="12345",
)
```

### Get/Update Company

```python
# Get
company = await hubspot.get_company("789012")

# Update
company = await hubspot.update_company(
    company_id="789012",
    properties={"industry": "SaaS"},
)
```

### List and Search Companies

```python
# List with pagination
companies, next_after = await hubspot.get_companies(limit=100)

# Search
companies = await hubspot.search_companies(
    query="acme",
    filters=[
        {"propertyName": "industry", "operator": "EQ", "value": "Technology"},
    ],
)
```

---

## Deal Operations

### Create Deal

```python
deal = await hubspot.create_deal(
    name="Enterprise License - Acme Corp",
    stage="contractsent",
    pipeline="default",
    amount=Decimal("50000"),
    close_date=datetime(2026, 3, 31),
    deal_type="newbusiness",
    owner_id="12345",
    priority="high",
)
```

### Update Deal Stage

```python
# Move deal to a different stage
deal = await hubspot.move_deal_stage(
    deal_id="456789",
    stage="closedwon",
)

# Or update multiple properties
deal = await hubspot.update_deal(
    deal_id="456789",
    properties={
        "dealstage": "closedwon",
        "amount": "55000",
    },
)
```

### List Deals

```python
deals, next_after = await hubspot.get_deals(limit=100)
```

---

## Pipeline Management

### Get Pipelines

```python
# Get all deal pipelines
pipelines = await hubspot.get_pipelines("deals")

for pipeline in pipelines:
    print(f"Pipeline: {pipeline.label}")
    for stage in pipeline.stages:
        print(f"  Stage: {stage.label} (prob: {stage.probability})")
```

### Get Specific Pipeline

```python
pipeline = await hubspot.get_pipeline("deals", "default")
```

---

## Associations

### Create Association

```python
# Associate contact with company
await hubspot.create_association(
    from_object_type="contacts",
    from_object_id="123456",
    to_object_type="companies",
    to_object_id="789012",
)

# Associate deal with contact
await hubspot.create_association(
    from_object_type="deals",
    from_object_id="456789",
    to_object_type="contacts",
    to_object_id="123456",
)
```

### Get Associated Objects

```python
# Get companies associated with a contact
company_ids = await hubspot.get_associated_objects(
    object_type="contacts",
    object_id="123456",
    to_object_type="companies",
)

# Get contacts associated with a deal
contact_ids = await hubspot.get_associated_objects(
    object_type="deals",
    object_id="456789",
    to_object_type="contacts",
)
```

---

## Engagements

### Log Email

```python
engagement = await hubspot.log_email(
    subject="Follow-up on proposal",
    body="Hi John, just following up on the proposal we sent...",
    contact_ids=["123456"],
    owner_id="12345",
    timestamp=datetime.now(),
)
```

### Log Call

```python
engagement = await hubspot.log_call(
    body="Discussed pricing options and timeline",
    contact_ids=["123456"],
    owner_id="12345",
)
```

### Add Note

```python
engagement = await hubspot.add_note(
    body="Customer mentioned interest in enterprise features",
    contact_ids=["123456"],
    company_ids=["789012"],
    deal_ids=["456789"],
    owner_id="12345",
)
```

### Create Generic Engagement

```python
from aragora.connectors.crm.hubspot import EngagementType

engagement = await hubspot.create_engagement(
    engagement_type=EngagementType.MEETING,
    subject="Product demo",
    body="Demonstrated new dashboard features",
    contact_ids=["123456"],
    company_ids=["789012"],
    owner_id="12345",
    timestamp=datetime(2026, 1, 15, 14, 0),
)
```

---

## Owners

### List Owners

```python
owners = await hubspot.get_owners()

for owner in owners:
    print(f"{owner.full_name} ({owner.email})")
```

### Get Owner

```python
owner = await hubspot.get_owner("12345")
```

---

## Error Handling

```python
from aragora.connectors.crm.hubspot import HubSpotError

try:
    contact = await hubspot.create_contact(email="invalid")
except HubSpotError as e:
    print(f"HubSpot error: {e}")
    print(f"Status: {e.status_code}")
    print(f"Details: {e.details}")
```

### Common Errors

| Status | Description |
|--------|-------------|
| 400 | Bad request (invalid properties) |
| 401 | Unauthorized (invalid token) |
| 403 | Forbidden (missing scopes) |
| 404 | Object not found |
| 409 | Conflict (duplicate email) |
| 429 | Rate limited |

---

## Testing

### Mock Data

```python
from aragora.connectors.crm.hubspot import (
    get_mock_contact,
    get_mock_deal,
)

# Get mock objects for testing
contact = get_mock_contact()
deal = get_mock_deal()
```

### Test Account

HubSpot provides a [developer test account](https://developers.hubspot.com/docs/api/developer-tools-resources/creating-a-developer-test-account) for development.

---

## Rate Limits

HubSpot has API rate limits based on your subscription:

| Tier | Limit |
|------|-------|
| Free | 100 requests/10 seconds |
| Starter | 100 requests/10 seconds |
| Professional | 150 requests/10 seconds |
| Enterprise | 200 requests/10 seconds |

The connector handles 429 responses automatically with retry.

---

## Best Practices

1. **Use pagination** - Always paginate large lists
2. **Specify properties** - Only request properties you need
3. **Use search filters** - More efficient than fetching all and filtering
4. **Batch operations** - Use batch APIs for bulk operations (not yet implemented)
5. **Handle associations** - Link related objects for better CRM data

---

## Common Patterns

### Sync Contact to CRM

```python
async def sync_user_to_hubspot(user: User, hubspot: HubSpotConnector):
    """Sync a user to HubSpot, creating or updating as needed."""

    # Search for existing contact
    contacts = await hubspot.search_contacts(
        filters=[
            {"propertyName": "email", "operator": "EQ", "value": user.email}
        ],
    )

    if contacts:
        # Update existing
        return await hubspot.update_contact(
            contact_id=contacts[0].id,
            properties={
                "firstname": user.first_name,
                "lastname": user.last_name,
                "lifecyclestage": "customer",
            },
        )
    else:
        # Create new
        return await hubspot.create_contact(
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            lifecycle_stage="customer",
        )
```

### Track Deal Progress

```python
async def update_deal_from_order(order: Order, hubspot: HubSpotConnector):
    """Update deal based on order status."""

    stage_map = {
        "pending": "contractsent",
        "paid": "closedwon",
        "refunded": "closedlost",
    }

    await hubspot.update_deal(
        deal_id=order.hubspot_deal_id,
        properties={
            "dealstage": stage_map.get(order.status, "contractsent"),
            "amount": str(order.total),
        },
    )
```

---

## Related Documentation

- [HubSpot API Reference](https://developers.hubspot.com/docs/api/overview)
- [HubSpot CRM Objects](https://developers.hubspot.com/docs/api/crm/understanding-the-crm)
- [Connector Patterns Guide](./CONNECTOR_PATTERNS.md)

---

*Last updated: 2026-01-30*
