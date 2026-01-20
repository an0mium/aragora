# Changelog

All notable changes to the Aragora TypeScript SDK will be documented in this file.

## [2.0.3] - 2026-01-20

### Added
- ConnectorsAPI for enterprise data source management
  - `list()` - List registered connectors
  - `getStatus()` - Get connector sync status
  - `triggerSync()` - Trigger sync operations
  - `health()` - Connector subsystem health
  - `mongoAggregate()` - Execute MongoDB aggregation pipelines
  - `mongoCollections()` - List MongoDB collections
- Full TypeScript types for connector operations
  - `ConnectorType`, `SyncSchedule`, `ConnectorJob`
  - `MongoAggregateRequest/Response`
  - `ServiceNowRecord`, `ServiceNowUser`
  - `ConnectorHealthResponse`

### Changed
- Version aligned with core Aragora package (2.0.3)

## [2.0.0] - 2026-01-17

### Changed
- Major version bump to align with Aragora core v2.0.0
- All APIs updated for v2 compatibility

## [1.0.0] - 2026-01-14

### Added
- Full API coverage for all major Aragora endpoints
- DebatesAPI for creating and managing debates
- GraphDebatesAPI for branching debate structures
- MatrixDebatesAPI for parallel scenario analysis
- AgentsAPI for agent profiles, rankings, and relationships
- VerificationAPI for formal proof verification
- MemoryAPI for multi-tier memory management
- PluginsAPI for plugin marketplace integration
- PersonasAPI for agent personas
- SystemAPI for system monitoring and maintenance
- AdminAPI for administrative operations
- WebSocket streaming with automatic reconnection
- Async iterator support for debate event streams
- TypeScript types for all API responses

### Notes
- This is the first stable release
- API considered stable - breaking changes will follow semver
- Requires Node.js >= 18.0.0
