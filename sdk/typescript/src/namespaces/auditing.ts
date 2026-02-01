/**
 * Auditing Namespace API
 *
 * Provides deep audit, capability probing, and red team analysis capabilities.
 */

export interface CapabilityProbeRequest {
  agent: string;
  task: string;
  config?: Record<string, unknown>;
}

export interface CapabilityProbeResult {
  agent: string;
  task: string;
  score?: number;
  capabilities?: Record<string, unknown>;
  weaknesses?: string[];
  recommendations?: string[];
  metadata?: Record<string, unknown>;
}

export interface DeepAuditRequest {
  task: string;
  agents?: string[];
  depth?: 'shallow' | 'standard' | 'deep';
  config?: Record<string, unknown>;
}

export interface DeepAuditResult {
  audit_id: string;
  task: string;
  findings: Array<{
    category: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    evidence?: string;
  }>;
  summary: string;
  metadata?: Record<string, unknown>;
}

export interface RedTeamRequest {
  attack_types?: string[];
  intensity?: 'low' | 'medium' | 'high';
  config?: Record<string, unknown>;
}

export interface RedTeamResult {
  debate_id: string;
  attacks_run: number;
  vulnerabilities_found: number;
  findings: Array<{
    attack_type: string;
    success: boolean;
    description: string;
    severity?: string;
  }>;
  summary: string;
}

export interface AttackType {
  id: string;
  name: string;
  description: string;
  category: string;
}

interface AuditingClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { body?: unknown }
  ): Promise<T>;
}

export class AuditingAPI {
  constructor(private client: AuditingClientInterface) {}

  /**
   * Run capability probes on an agent.
   */
  async probeCapability(body: CapabilityProbeRequest): Promise<CapabilityProbeResult> {
    return this.client.request('POST', '/api/v1/debates/capability-probe', { body });
  }

  /**
   * Run a deep audit on a task.
   */
  async deepAudit(body: DeepAuditRequest): Promise<DeepAuditResult> {
    return this.client.request('POST', '/api/v1/debates/deep-audit', { body });
  }

  /**
   * Run red team analysis on a debate.
   */
  async redTeam(debateId: string, body?: RedTeamRequest): Promise<RedTeamResult> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/red-team`, { body });
  }

  /**
   * Get available red team attack types.
   */
  async getAttackTypes(): Promise<{ attack_types: AttackType[] }> {
    return this.client.request('GET', '/api/v1/redteam/attack-types');
  }
}
