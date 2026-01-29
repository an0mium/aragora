# Server Module

HTTP/WebSocket API server with 461+ endpoints.

## Quick Start

```bash
# Start server
python -m aragora.server.unified_server --port 8080

# With auto-migration
ARAGORA_AUTO_MIGRATE_ON_STARTUP=true python -m aragora.server.unified_server
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `unified_server.py` | Central HTTP/WebSocket server |
| `startup.py` | Server initialization and health checks |
| `router.py` | O(1) path lookup with LRU dispatch caching |
| `debate_controller.py` | Debate orchestration coordination |
| `debate_origin.py` | Bidirectional chat result routing |

## Architecture

```
server/
├── unified_server.py     # Main entry point
├── startup.py            # Initialization sequence
├── router.py             # Request routing
├── api.py                # RESTful API setup
├── auth.py               # Authentication
├── handlers/             # HTTP endpoint handlers (90+ modules)
│   ├── debates/          # Debate CRUD, analysis, export
│   ├── agents/           # Agent management
│   ├── knowledge/        # Knowledge base operations
│   ├── decisions/        # Decision routing
│   ├── analytics/        # Metrics and dashboards
│   ├── auth/             # OAuth, SSO, MFA
│   ├── integrations/     # Slack, Teams, GitHub, Gmail
│   ├── admin/            # Admin operations
│   ├── workflows/        # DAG execution
│   └── ...               # 20+ more domains
├── stream/               # WebSocket handlers (25 modules)
│   ├── events.py         # Event types
│   ├── emitter.py        # Event emission
│   ├── debate_stream.py  # Debate streaming
│   └── voice_stream.py   # Voice/TTS streaming
├── middleware/           # Request middleware (25 modules)
│   ├── auth.py           # Authentication
│   ├── rate_limit/       # Rate limiting
│   ├── security.py       # CORS, HTTPS
│   ├── tracing.py        # OpenTelemetry
│   ├── tenancy.py        # Multi-tenant isolation
│   └── slo_tracking.py   # SLO monitoring
└── validation/           # Request validation
```

## API Statistics

- **461 API endpoints** across 90+ HTTP handlers
- **22 WebSocket stream types** for real-time updates
- **25 middleware modules** for security, logging, rate limiting
- **28 handler subdirectories** organizing domains

## Key Features

- **Multi-Tenant**: Tenant isolation with RBAC
- **Real-Time**: WebSocket streaming for all events
- **Observability**: Prometheus, OpenTelemetry, structured logging
- **Resilience**: Circuit breakers, rate limiting, graceful degradation

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [docs/API_REFERENCE.md](../../docs/API_REFERENCE.md) - API documentation
