/**
 * SDK Namespace APIs
 *
 * Re-exports all namespace APIs for the Aragora SDK.
 */

export { DebatesAPI } from './debates';
export { AgentsAPI } from './agents';
export { WorkflowsAPI } from './workflows';
export { SMEAPI } from './sme';
export { ControlPlaneAPI } from './control-plane';
export { GauntletAPI, type GauntletRunRequest } from './gauntlet';
export { AnalyticsAPI, type AnalyticsDateRange } from './analytics';
export { MemoryAPI, type MemoryStoreOptions, type MemoryRetrieveOptions } from './memory';
export { RBACAPI, type CreateRoleRequest, type UpdateRoleRequest } from './rbac';
