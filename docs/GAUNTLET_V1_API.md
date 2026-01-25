# Gauntlet API v1

The Gauntlet API provides versioned, OpenAPI-compliant endpoints for stress-testing AI decisions. All endpoints follow RFC 7807 for error responses.

## Base URL

```
/api/v1/gauntlet
```

## Authentication

All endpoints require authentication via Bearer token:
```
Authorization: Bearer <token>
```

## Endpoints

### Get Schema

Retrieve a JSON Schema for a specific type.

```http
GET /api/v1/gauntlet/schema/{type}
```

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | Schema type: `decision-receipt`, `risk-heatmap`, `problem-detail` |

**Response: 200 OK**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "receipt_id": {"type": "string"},
    "gauntlet_id": {"type": "string"},
    ...
  }
}
```

**Error: 404 Not Found**
```json
{
  "type": "https://aragora.ai/problems/not-found",
  "title": "Schema Not Found",
  "status": 404,
  "detail": "Schema type 'invalid' not found. Available: ['decision-receipt', 'risk-heatmap']",
  "available_schemas": ["decision-receipt", "risk-heatmap"]
}
```

---

### Get All Schemas

Retrieve all available JSON Schemas.

```http
GET /api/v1/gauntlet/schemas
```

**Response: 200 OK**
```json
{
  "version": "1.0.0",
  "schemas": {
    "decision-receipt": {...},
    "risk-heatmap": {...},
    "problem-detail": {...}
  },
  "count": 3
}
```

---

### List Templates

List available audit templates with optional filtering.

```http
GET /api/v1/gauntlet/templates
GET /api/v1/gauntlet/templates?category=compliance
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Filter by category: `compliance`, `security`, `legal`, `financial`, `operational` |

**Response: 200 OK**
```json
{
  "templates": [
    {
      "id": "soc2-type2",
      "name": "SOC 2 Type II Audit",
      "category": "compliance",
      "description": "Complete SOC 2 Type II compliance audit template",
      "version": "1.0",
      "regulations": ["SOC 2"],
      "supported_formats": ["json", "markdown", "html"]
    }
  ],
  "count": 1
}
```

**Error: 400 Bad Request**
```json
{
  "type": "https://aragora.ai/problems/validation-error",
  "title": "Invalid Category",
  "status": 400,
  "detail": "Category must be one of: ['compliance', 'security', 'legal', 'financial', 'operational']",
  "valid_categories": ["compliance", "security", "legal", "financial", "operational"]
}
```

---

### Get Template

Retrieve a specific audit template by ID.

```http
GET /api/v1/gauntlet/templates/{id}
```

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | Template identifier |

**Response: 200 OK**
```json
{
  "id": "soc2-type2",
  "name": "SOC 2 Type II Audit",
  "category": "compliance",
  "description": "Complete SOC 2 Type II compliance audit template",
  "version": "1.0",
  "regulations": ["SOC 2"],
  "sections": [...],
  "checks": [...]
}
```

**Error: 404 Not Found**
```json
{
  "type": "https://aragora.ai/problems/not-found",
  "title": "Template Not Found",
  "status": 404,
  "detail": "Template 'invalid' not found. Available: ['soc2-type2', 'gdpr-assessment']",
  "available_templates": ["soc2-type2", "gdpr-assessment"]
}
```

---

### Export Receipt

Export a decision receipt in various formats.

```http
POST /api/v1/gauntlet/{id}/export
```

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | Gauntlet run ID |

**Request Body:**
```json
{
  "format": "json",
  "template_id": "soc2-type2",
  "options": {
    "include_provenance": true,
    "include_config": false,
    "max_vulnerabilities": 100,
    "validate_schema": false
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `format` | string | Yes | Export format: `json`, `markdown`, `html`, `csv`, `sarif` |
| `template_id` | string | No | Audit template to apply |
| `options.include_provenance` | boolean | No | Include provenance chain (default: true) |
| `options.include_config` | boolean | No | Include gauntlet config (default: false) |
| `options.max_vulnerabilities` | integer | No | Max vulnerabilities to include (default: 100) |
| `options.validate_schema` | boolean | No | Validate against JSON schema (default: false) |

**Response: 200 OK**

Content-Type varies by format:
- `json`: `application/json`
- `markdown`: `text/markdown`
- `html`: `text/html`
- `csv`: `text/csv`
- `sarif`: `application/sarif+json`

**JSON Response Example:**
```json
{
  "receipt_id": "rcpt-abc123",
  "gauntlet_id": "gauntlet-xyz789",
  "timestamp": "2026-01-24T09:00:00Z",
  "input_summary": "Decision to approve loan application",
  "input_hash": "sha256:abc123...",
  "risk_summary": {
    "critical": 0,
    "high": 1,
    "medium": 3,
    "low": 5,
    "total": 9
  },
  "attacks_attempted": 150,
  "attacks_successful": 12,
  "probes_run": 50,
  "vulnerabilities_found": 9,
  "verdict": "APPROVED",
  "confidence": 0.87,
  "robustness_score": 0.92
}
```

**Error: 400 Bad Request**
```json
{
  "type": "https://aragora.ai/problems/validation-error",
  "title": "Invalid Format",
  "status": 400,
  "detail": "Format must be one of: ['json', 'markdown', 'html', 'csv', 'sarif']",
  "valid_formats": ["json", "markdown", "html", "csv", "sarif"]
}
```

**Error: 404 Not Found**
```json
{
  "type": "https://aragora.ai/problems/not-found",
  "title": "Gauntlet Not Found",
  "status": 404,
  "detail": "Gauntlet run 'invalid' not found",
  "instance": "/api/v1/gauntlet/invalid"
}
```

---

### Export Heatmap

Export a risk heatmap visualization.

```http
GET /api/v1/gauntlet/{id}/heatmap/export
GET /api/v1/gauntlet/{id}/heatmap/export?format=svg
```

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | Gauntlet run ID |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | `json` | Export format: `json`, `csv`, `svg`, `ascii`, `html` |

**Response: 200 OK**

Content-Type varies by format:
- `json`: `application/json`
- `csv`: `text/csv`
- `svg`: `image/svg+xml`
- `ascii`: `text/plain`
- `html`: `text/html`

**JSON Response Example:**
```json
{
  "dimensions": ["category", "severity"],
  "categories": ["privacy", "security", "compliance"],
  "severities": ["critical", "high", "medium", "low"],
  "data": [
    {"category": "privacy", "severity": "high", "count": 3},
    {"category": "security", "severity": "medium", "count": 5}
  ]
}
```

---

### Validate Receipt

Validate a decision receipt against the JSON schema.

```http
POST /api/v1/gauntlet/validate/receipt
```

**Request Body:**
```json
{
  "receipt_id": "rcpt-abc123",
  "gauntlet_id": "gauntlet-xyz789",
  "timestamp": "2026-01-24T09:00:00Z",
  "verdict": "APPROVED",
  "confidence": 0.87
}
```

**Response: 200 OK**
```json
{
  "valid": true,
  "errors": [],
  "error_count": 0
}
```

**Validation Error Response:**
```json
{
  "valid": false,
  "errors": [
    "'gauntlet_id' is a required property",
    "'timestamp' is not a valid ISO 8601 date"
  ],
  "error_count": 2
}
```

**Error: 400 Bad Request**
```json
{
  "type": "https://aragora.ai/problems/validation-error",
  "title": "Missing Body",
  "status": 400,
  "detail": "Request body with receipt data is required"
}
```

---

## RFC 7807 Error Format

All errors follow RFC 7807 Problem Details format:

```json
{
  "type": "https://aragora.ai/problems/{problem-type}",
  "title": "Human-readable title",
  "status": 400,
  "detail": "Detailed error message",
  "instance": "/api/v1/gauntlet/request-path"
}
```

**Problem Types:**
| Type | Description |
|------|-------------|
| `not-found` | Resource not found (404) |
| `validation-error` | Invalid input (400) |
| `internal-error` | Server error (500) |

**Extension Fields:**

Errors may include additional context:
- `available_schemas`: List of valid schema names
- `available_templates`: List of valid template IDs
- `valid_formats`: List of valid export formats
- `valid_categories`: List of valid category values
- `current_status`: Current status when operation requires different status

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| GET /api/v1/gauntlet/schema/* | 60 req/min |
| GET /api/v1/gauntlet/schemas | 60 req/min |
| GET /api/v1/gauntlet/templates | 30 req/min |
| POST /api/v1/gauntlet/*/export | 10 req/min |
| POST /api/v1/gauntlet/validate/receipt | 30 req/min |

Rate limit headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1706090460
```

---

## SARIF Export Format

SARIF (Static Analysis Results Interchange Format) exports are compatible with GitHub Code Scanning and other security tools:

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "Aragora Gauntlet",
          "version": "1.0.0"
        }
      },
      "results": [
        {
          "ruleId": "aragora/high-risk-decision",
          "level": "warning",
          "message": {
            "text": "High risk vulnerability detected in decision path"
          },
          "locations": [...]
        }
      ]
    }
  ]
}
```

---

## Handler Classes

The API is implemented by these handler classes in `aragora/server/handlers/gauntlet_v1.py`:

| Handler | Endpoints |
|---------|-----------|
| `GauntletSchemaHandler` | GET /api/v1/gauntlet/schema/{type} |
| `GauntletAllSchemasHandler` | GET /api/v1/gauntlet/schemas |
| `GauntletTemplatesListHandler` | GET /api/v1/gauntlet/templates |
| `GauntletTemplateHandler` | GET /api/v1/gauntlet/templates/{id} |
| `GauntletReceiptExportHandler` | POST /api/v1/gauntlet/{id}/export |
| `GauntletHeatmapExportHandler` | GET /api/v1/gauntlet/{id}/heatmap/export |
| `GauntletValidateReceiptHandler` | POST /api/v1/gauntlet/validate/receipt |

---

## See Also

- [Gauntlet Quickstart](./GAUNTLET_QUICKSTART.md)
- [API Reference](./API_REFERENCE.md)
- [Decision Receipts](./DECISION_RECEIPTS.md)
