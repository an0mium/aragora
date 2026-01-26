/**
 * Knowledge Namespace API
 *
 * Provides a namespaced interface for knowledge management operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  KnowledgeEntry,
  KnowledgeSearchResult,
  KnowledgeStats,
  KnowledgeMoundNode,
  KnowledgeMoundRelationship,
  KnowledgeMoundQueryResult,
  KnowledgeMoundStats,
  PaginationParams,
} from '../types';

/**
 * Options for searching knowledge.
 */
export interface KnowledgeSearchOptions {
  /** Filter by content type */
  type?: string;
  /** Filter by tags */
  tags?: string[];
  /** Minimum confidence score */
  min_confidence?: number;
  /** Maximum results */
  limit?: number;
}

/**
 * Options for querying the Knowledge Mound.
 */
export interface KnowledgeMoundQueryOptions {
  /** Filter by node types */
  types?: string[];
  /** Relationship depth to traverse */
  depth?: number;
  /** Include relationships in result */
  include_relationships?: boolean;
  /** Maximum results */
  limit?: number;
}

/**
 * Interface for the internal client methods used by KnowledgeAPI.
 */
interface KnowledgeClientInterface {
  // Generic request method for extended endpoints
  request<T = unknown>(method: string, path: string, options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }): Promise<T>;

  // Core knowledge methods
  searchKnowledge(query: string, options?: KnowledgeSearchOptions): Promise<{ results: KnowledgeSearchResult[] }>;
  addKnowledge(entry: KnowledgeEntry): Promise<{ id: string; created_at: string }>;
  getKnowledgeEntry(entryId: string): Promise<KnowledgeEntry>;
  updateKnowledge(entryId: string, updates: Partial<KnowledgeEntry>): Promise<KnowledgeEntry>;
  deleteKnowledge(entryId: string): Promise<{ deleted: boolean }>;
  getKnowledgeStats(): Promise<KnowledgeStats>;
  bulkImportKnowledge(entries: KnowledgeEntry[]): Promise<{
    imported: number;
    failed: number;
    errors?: Array<{ index: number; error: string }>;
  }>;
  queryKnowledgeMound(query: string, options?: KnowledgeMoundQueryOptions): Promise<KnowledgeMoundQueryResult>;
  listKnowledgeMoundNodes(options?: PaginationParams & { type?: string }): Promise<{ nodes: KnowledgeMoundNode[] }>;
  createKnowledgeMoundNode(node: {
    content: string;
    node_type: 'fact' | 'concept' | 'claim' | 'evidence' | 'insight';
    confidence?: number;
    source?: string;
    tags?: string[];
    visibility?: 'private' | 'team' | 'global';
    metadata?: Record<string, unknown>;
  }): Promise<{ id: string; created_at: string }>;
  listKnowledgeMoundRelationships(options?: PaginationParams): Promise<{ relationships: KnowledgeMoundRelationship[] }>;
  createKnowledgeMoundRelationship(relationship: {
    source_id: string;
    target_id: string;
    relationship_type: 'supports' | 'contradicts' | 'elaborates' | 'derived_from' | 'related_to';
    strength?: number;
    confidence?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ id: string; created_at: string }>;
  getKnowledgeMoundStats(): Promise<KnowledgeMoundStats>;
  getStaleKnowledge(options?: { max_age_days?: number; limit?: number }): Promise<{ items: unknown[]; total: number }>;
  revalidateKnowledge(nodeId: string, validation: { valid: boolean; new_confidence?: number; notes?: string }): Promise<{ updated: boolean }>;
  getKnowledgeLineage(nodeId: string, options?: { direction?: 'ancestors' | 'descendants' | 'both'; max_depth?: number }): Promise<{ nodes: unknown[]; relationships: unknown[] }>;
  getRelatedKnowledge(nodeId: string, options?: { relationship_types?: string[]; limit?: number }): Promise<{ nodes: unknown[]; relationships: unknown[] }>;
}

/**
 * Knowledge API namespace.
 *
 * Provides methods for managing knowledge:
 * - Searching and retrieving knowledge entries
 * - Adding and updating knowledge
 * - Working with the Knowledge Mound graph
 * - Knowledge validation and lineage tracking
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Search knowledge
 * const results = await client.knowledge.search('machine learning');
 *
 * // Add new knowledge entry
 * const entry = await client.knowledge.add({
 *   content: 'Neural networks are computational models...',
 *   type: 'fact',
 *   confidence: 0.95,
 * });
 *
 * // Query the Knowledge Mound
 * const graph = await client.knowledge.queryMound('AI safety', { depth: 2 });
 *
 * // Get knowledge statistics
 * const stats = await client.knowledge.stats();
 * ```
 */
export class KnowledgeAPI {
  constructor(private client: KnowledgeClientInterface) {}

  /**
   * Search knowledge entries by query.
   */
  async search(query: string, options?: KnowledgeSearchOptions): Promise<{ results: KnowledgeSearchResult[] }> {
    return this.client.searchKnowledge(query, options);
  }

  /**
   * Add a new knowledge entry.
   */
  async add(entry: KnowledgeEntry): Promise<{ id: string; created_at: string }> {
    return this.client.addKnowledge(entry);
  }

  /**
   * Get a knowledge entry by ID.
   */
  async get(entryId: string): Promise<KnowledgeEntry> {
    return this.client.getKnowledgeEntry(entryId);
  }

  /**
   * Update an existing knowledge entry.
   */
  async update(entryId: string, updates: Partial<KnowledgeEntry>): Promise<KnowledgeEntry> {
    return this.client.updateKnowledge(entryId, updates);
  }

  /**
   * Delete a knowledge entry.
   */
  async delete(entryId: string): Promise<{ deleted: boolean }> {
    return this.client.deleteKnowledge(entryId);
  }

  /**
   * Get knowledge system statistics.
   */
  async stats(): Promise<KnowledgeStats> {
    return this.client.getKnowledgeStats();
  }

  /**
   * Bulk import knowledge entries.
   */
  async bulkImport(entries: KnowledgeEntry[]): Promise<{
    imported: number;
    failed: number;
    errors?: Array<{ index: number; error: string }>;
  }> {
    return this.client.bulkImportKnowledge(entries);
  }

  /**
   * Query the Knowledge Mound graph.
   */
  async queryMound(query: string, options?: KnowledgeMoundQueryOptions): Promise<KnowledgeMoundQueryResult> {
    return this.client.queryKnowledgeMound(query, options);
  }

  /**
   * List Knowledge Mound nodes.
   */
  async listNodes(options?: PaginationParams & { type?: string }): Promise<{ nodes: KnowledgeMoundNode[] }> {
    return this.client.listKnowledgeMoundNodes(options);
  }

  /**
   * Create a new Knowledge Mound node.
   */
  async createNode(node: {
    content: string;
    node_type: 'fact' | 'concept' | 'claim' | 'evidence' | 'insight';
    confidence?: number;
    source?: string;
    tags?: string[];
    visibility?: 'private' | 'team' | 'global';
    metadata?: Record<string, unknown>;
  }): Promise<{ id: string; created_at: string }> {
    return this.client.createKnowledgeMoundNode(node);
  }

  /**
   * List Knowledge Mound relationships.
   */
  async listRelationships(options?: PaginationParams): Promise<{ relationships: KnowledgeMoundRelationship[] }> {
    return this.client.listKnowledgeMoundRelationships(options);
  }

  /**
   * Create a new Knowledge Mound relationship.
   */
  async createRelationship(relationship: {
    source_id: string;
    target_id: string;
    relationship_type: 'supports' | 'contradicts' | 'elaborates' | 'derived_from' | 'related_to';
    strength?: number;
    confidence?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ id: string; created_at: string }> {
    return this.client.createKnowledgeMoundRelationship(relationship);
  }

  /**
   * Get Knowledge Mound statistics.
   */
  async moundStats(): Promise<KnowledgeMoundStats> {
    return this.client.getKnowledgeMoundStats();
  }

  /**
   * Get stale knowledge entries that need revalidation.
   */
  async getStale(options?: { max_age_days?: number; limit?: number }): Promise<{ items: unknown[]; total: number }> {
    return this.client.getStaleKnowledge(options);
  }

  /**
   * Revalidate a knowledge node.
   */
  async revalidate(nodeId: string, validation: { valid: boolean; new_confidence?: number; notes?: string }): Promise<{ updated: boolean }> {
    return this.client.revalidateKnowledge(nodeId, validation);
  }

  /**
   * Get the lineage/provenance of a knowledge node.
   */
  async getLineage(nodeId: string, options?: { direction?: 'ancestors' | 'descendants' | 'both'; max_depth?: number }): Promise<{ nodes: unknown[]; relationships: unknown[] }> {
    return this.client.getKnowledgeLineage(nodeId, options);
  }

  /**
   * Get knowledge nodes related to a given node.
   */
  async getRelated(nodeId: string, options?: { relationship_types?: string[]; limit?: number }): Promise<{ nodes: unknown[]; relationships: unknown[] }> {
    return this.client.getRelatedKnowledge(nodeId, options);
  }

  // =========================================================================
  // Graph Traversal
  // =========================================================================

  /**
   * Traverse the knowledge graph from a starting node.
   */
  async traverseGraph(nodeId: string, options?: { depth?: number; direction?: 'outgoing' | 'incoming' | 'both' }): Promise<{ nodes: unknown[]; relationships: unknown[] }> {
    return this.client.request('GET', `/api/v1/knowledge/mound/graph/${nodeId}`, { params: options });
  }

  /**
   * Get the derivation lineage of a knowledge node.
   */
  async getNodeLineage(nodeId: string, options?: { max_depth?: number }): Promise<{ nodes: unknown[]; path: unknown[] }> {
    return this.client.request('GET', `/api/v1/knowledge/mound/graph/${nodeId}/lineage`, { params: options });
  }

  /**
   * Get immediate neighbors of a knowledge node.
   */
  async getNodeNeighbors(nodeId: string, options?: { limit?: number }): Promise<{ nodes: unknown[]; relationships: unknown[] }> {
    return this.client.request('GET', `/api/v1/knowledge/mound/graph/${nodeId}/related`, { params: options });
  }

  // =========================================================================
  // Visibility & Access Control
  // =========================================================================

  /**
   * Get visibility level of a knowledge node.
   */
  async getVisibility(nodeId: string): Promise<{ visibility: string; access_grants: unknown[] }> {
    return this.client.request('GET', `/api/v1/knowledge/mound/nodes/${nodeId}/visibility`);
  }

  /**
   * Set visibility level of a knowledge node.
   */
  async setVisibility(nodeId: string, visibility: 'private' | 'workspace' | 'shared' | 'public'): Promise<{ updated: boolean }> {
    return this.client.request('PUT', `/api/v1/knowledge/mound/nodes/${nodeId}/visibility`, { json: { visibility } });
  }

  /**
   * List access grants for a knowledge node.
   */
  async listAccessGrants(nodeId: string): Promise<{ grants: unknown[] }> {
    return this.client.request('GET', `/api/v1/knowledge/mound/nodes/${nodeId}/access`);
  }

  /**
   * Grant access to a knowledge node.
   */
  async grantAccess(nodeId: string, grant: { grantee_id: string; grantee_type: 'user' | 'workspace'; permission: 'read' | 'write' | 'admin' }): Promise<{ grant_id: string }> {
    return this.client.request('POST', `/api/v1/knowledge/mound/nodes/${nodeId}/access`, { json: grant });
  }

  /**
   * Revoke access from a knowledge node.
   */
  async revokeAccess(nodeId: string, grantId: string): Promise<{ revoked: boolean }> {
    return this.client.request('DELETE', `/api/v1/knowledge/mound/nodes/${nodeId}/access`, { json: { grant_id: grantId } });
  }

  // =========================================================================
  // Sharing
  // =========================================================================

  /**
   * Share a knowledge item with another workspace or user.
   */
  async share(itemId: string, share: { target_id: string; target_type: 'user' | 'workspace'; permission?: 'read' | 'write'; expires_at?: string }): Promise<{ share_id: string }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/share', { json: { item_id: itemId, ...share } });
  }

  /**
   * Get items shared with the current user/workspace.
   */
  async getSharedWithMe(options?: PaginationParams): Promise<{ items: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/shared-with-me', { params: options });
  }

  /**
   * Get items shared by the current user.
   */
  async getMyShares(options?: PaginationParams): Promise<{ shares: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/my-shares', { params: options });
  }

  /**
   * Revoke a share.
   */
  async revokeShare(shareId: string): Promise<{ revoked: boolean }> {
    return this.client.request('DELETE', '/api/v1/knowledge/mound/share', { json: { share_id: shareId } });
  }

  /**
   * Update share permissions or expiration.
   */
  async updateShare(shareId: string, updates: { permission?: 'read' | 'write'; expires_at?: string }): Promise<{ updated: boolean }> {
    return this.client.request('PATCH', '/api/v1/knowledge/mound/share', { json: { share_id: shareId, ...updates } });
  }

  // =========================================================================
  // Federation
  // =========================================================================

  /**
   * Register a federated region (admin only).
   */
  async registerRegion(region: { name: string; endpoint: string; api_key: string }): Promise<{ region_id: string }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/federation/regions', { json: region });
  }

  /**
   * List all federated regions.
   */
  async listRegions(): Promise<{ regions: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/federation/regions');
  }

  /**
   * Unregister a federated region (admin only).
   */
  async unregisterRegion(regionId: string): Promise<{ unregistered: boolean }> {
    return this.client.request('DELETE', `/api/v1/knowledge/mound/federation/regions/${regionId}`);
  }

  /**
   * Push knowledge to a remote region.
   */
  async syncPush(regionId: string, options?: { scope?: 'all' | 'workspace' | 'selected'; node_ids?: string[] }): Promise<{ synced: number; failed: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/federation/sync/push', { json: { region_id: regionId, ...options } });
  }

  /**
   * Pull knowledge from a remote region.
   */
  async syncPull(regionId: string, options?: { since?: string; limit?: number }): Promise<{ received: number; merged: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/federation/sync/pull', { json: { region_id: regionId, ...options } });
  }

  /**
   * Synchronize with all configured regions.
   */
  async syncAll(options?: { direction?: 'push' | 'pull' | 'both' }): Promise<{ results: unknown[] }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/federation/sync/all', { json: options });
  }

  /**
   * Get federation status and health.
   */
  async getFederationStatus(): Promise<{ regions: unknown[]; last_sync: string; health: string }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/federation/status');
  }

  // =========================================================================
  // Global Knowledge
  // =========================================================================

  /**
   * Store a verified fact as global knowledge (admin only).
   */
  async storeGlobalFact(fact: { content: string; source: string; confidence: number; tags?: string[] }): Promise<{ id: string }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/global', { json: fact });
  }

  /**
   * Query global/system knowledge.
   */
  async queryGlobal(query: string, options?: { limit?: number }): Promise<{ results: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/global', { params: { query, ...options } });
  }

  /**
   * Promote workspace knowledge to global level.
   */
  async promoteToGlobal(nodeId: string, options?: { review_required?: boolean }): Promise<{ promoted: boolean; global_id: string }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/global/promote', { json: { node_id: nodeId, ...options } });
  }

  /**
   * List all system-level verified facts.
   */
  async getSystemFacts(options?: PaginationParams): Promise<{ facts: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/global/facts', { params: options });
  }

  // =========================================================================
  // Deduplication
  // =========================================================================

  /**
   * Find duplicate clusters by similarity threshold.
   */
  async getDuplicateClusters(options?: { threshold?: number; limit?: number }): Promise<{ clusters: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/dedup/clusters', { params: options });
  }

  /**
   * Generate deduplication analysis report.
   */
  async getDedupReport(): Promise<{ total_nodes: number; duplicate_clusters: number; potential_savings: number; recommendations: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/dedup/report');
  }

  /**
   * Merge a specific duplicate cluster.
   */
  async mergeDuplicateCluster(clusterId: string, options?: { primary_id?: string; strategy?: 'newest' | 'highest_confidence' | 'manual' }): Promise<{ merged_id: string; removed_count: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/dedup/merge', { json: { cluster_id: clusterId, ...options } });
  }

  /**
   * Automatically merge exact duplicates.
   */
  async autoMergeExactDuplicates(options?: { dry_run?: boolean }): Promise<{ merged: number; clusters_processed: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/dedup/auto-merge', { json: options });
  }

  // =========================================================================
  // Pruning
  // =========================================================================

  /**
   * Get items eligible for pruning by staleness/age.
   */
  async getPrunableItems(options?: { max_age_days?: number; min_staleness?: number; limit?: number }): Promise<{ items: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/pruning/items', { params: options });
  }

  /**
   * Prune specified items (archive or delete).
   */
  async executePrune(nodeIds: string[], options?: { action?: 'archive' | 'delete' }): Promise<{ pruned: number; archived: number; deleted: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/pruning/execute', { json: { node_ids: nodeIds, ...options } });
  }

  /**
   * Run auto-prune with policy.
   */
  async autoPrune(options?: { policy?: 'conservative' | 'moderate' | 'aggressive'; dry_run?: boolean }): Promise<{ pruned: number; archived: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/pruning/auto', { json: options });
  }

  /**
   * Get pruning history.
   */
  async getPruneHistory(options?: PaginationParams): Promise<{ events: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/pruning/history', { params: options });
  }

  /**
   * Restore an archived item.
   */
  async restorePrunedItem(nodeId: string): Promise<{ restored: boolean }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/pruning/restore', { json: { node_id: nodeId } });
  }

  // =========================================================================
  // Culture
  // =========================================================================

  /**
   * Get organization culture profile.
   */
  async getCulture(): Promise<{ principles: unknown[]; values: unknown[]; policies: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/culture');
  }

  /**
   * Add a culture document.
   */
  async addCultureDocument(document: { type: 'policy' | 'principle' | 'value'; content: string; source?: string }): Promise<{ id: string }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/culture/documents', { json: document });
  }

  /**
   * Promote knowledge to culture level.
   */
  async promoteToCulture(nodeId: string, options?: { type?: 'policy' | 'principle' | 'value' }): Promise<{ promoted: boolean }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/culture/promote', { json: { node_id: nodeId, ...options } });
  }

  // =========================================================================
  // Dashboard & Metrics
  // =========================================================================

  /**
   * Get Knowledge Mound health status and recommendations.
   */
  async getDashboardHealth(): Promise<{ status: string; score: number; recommendations: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/dashboard/health');
  }

  /**
   * Get detailed operational metrics.
   */
  async getDashboardMetrics(): Promise<{ queries: unknown; storage: unknown; performance: unknown }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/dashboard/metrics');
  }

  /**
   * Reset metrics counters.
   */
  async resetDashboardMetrics(): Promise<{ reset: boolean }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/dashboard/metrics/reset');
  }

  /**
   * Get adapter status and health.
   */
  async getDashboardAdapters(): Promise<{ adapters: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/dashboard/adapters');
  }

  // =========================================================================
  // Contradictions
  // =========================================================================

  /**
   * Trigger contradiction detection scan.
   */
  async detectContradictions(options?: { scope?: 'workspace' | 'all' }): Promise<{ detected: number; scan_id: string }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/contradictions/detect', { json: options });
  }

  /**
   * List unresolved contradictions.
   */
  async listContradictions(options?: PaginationParams & { status?: 'unresolved' | 'resolved' | 'all' }): Promise<{ contradictions: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/contradictions', { params: options });
  }

  /**
   * Resolve a contradiction.
   */
  async resolveContradiction(contradictionId: string, resolution: { strategy: 'keep_first' | 'keep_second' | 'merge' | 'invalidate_both'; notes?: string }): Promise<{ resolved: boolean }> {
    return this.client.request('POST', `/api/v1/knowledge/mound/contradictions/${contradictionId}/resolve`, { json: resolution });
  }

  /**
   * Get contradiction statistics.
   */
  async getContradictionStats(): Promise<{ total: number; unresolved: number; resolved: number; by_type: unknown }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/contradictions/stats');
  }

  // =========================================================================
  // Governance
  // =========================================================================

  /**
   * Create a custom role for knowledge governance.
   */
  async createGovernanceRole(role: { name: string; permissions: string[]; description?: string }): Promise<{ role_id: string }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/governance/roles', { json: role });
  }

  /**
   * Assign a role to a user.
   */
  async assignGovernanceRole(userId: string, roleId: string): Promise<{ assigned: boolean }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/governance/roles/assign', { json: { user_id: userId, role_id: roleId } });
  }

  /**
   * Revoke a role from a user.
   */
  async revokeGovernanceRole(userId: string, roleId: string): Promise<{ revoked: boolean }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/governance/roles/revoke', { json: { user_id: userId, role_id: roleId } });
  }

  /**
   * Get user permissions for knowledge governance.
   */
  async getUserGovernancePermissions(userId: string): Promise<{ permissions: string[]; roles: unknown[] }> {
    return this.client.request('GET', `/api/v1/knowledge/mound/governance/permissions/${userId}`);
  }

  /**
   * Check if user has a specific permission.
   */
  async checkGovernancePermission(userId: string, permission: string): Promise<{ allowed: boolean }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/governance/permissions/check', { json: { user_id: userId, permission } });
  }

  /**
   * Query the governance audit trail.
   */
  async queryGovernanceAudit(options?: { user_id?: string; action?: string; since?: string; limit?: number }): Promise<{ events: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/governance/audit', { params: options });
  }

  /**
   * Get user activity history for governance.
   */
  async getUserGovernanceActivity(userId: string, options?: PaginationParams): Promise<{ activities: unknown[]; total: number }> {
    return this.client.request('GET', `/api/v1/knowledge/mound/governance/audit/user/${userId}`, { params: options });
  }

  // =========================================================================
  // Analytics
  // =========================================================================

  /**
   * Analyze domain coverage by topic.
   */
  async analyzeCoverage(options?: { topics?: string[] }): Promise<{ coverage: unknown[]; gaps: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/analytics/coverage', { params: options });
  }

  /**
   * Analyze usage patterns over time.
   */
  async analyzeUsage(options?: { period?: 'day' | 'week' | 'month'; since?: string }): Promise<{ patterns: unknown[]; trends: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/analytics/usage', { params: options });
  }

  /**
   * Record a usage event.
   */
  async recordUsageEvent(event: { node_id: string; event_type: 'query' | 'view' | 'cite' | 'share' | 'export'; metadata?: Record<string, unknown> }): Promise<{ recorded: boolean }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/analytics/usage/record', { json: event });
  }

  /**
   * Capture quality metrics snapshot.
   */
  async captureQualitySnapshot(): Promise<{ snapshot_id: string; metrics: unknown }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/analytics/quality/snapshot');
  }

  /**
   * Get quality metrics trend over time.
   */
  async getQualityTrend(options?: { period?: 'day' | 'week' | 'month'; metrics?: string[] }): Promise<{ trend: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/analytics/quality/trend', { params: options });
  }

  // =========================================================================
  // Extraction
  // =========================================================================

  /**
   * Extract claims/knowledge from a debate.
   */
  async extractFromDebate(debateId: string, options?: { confidence_threshold?: number; auto_promote?: boolean }): Promise<{ extracted: number; claims: unknown[] }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/extraction/debate', { json: { debate_id: debateId, ...options } });
  }

  /**
   * Promote extracted claims to main knowledge.
   */
  async promoteExtracted(claimIds: string[], options?: { target_tier?: string }): Promise<{ promoted: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/extraction/promote', { json: { claim_ids: claimIds, ...options } });
  }

  // =========================================================================
  // Confidence Decay
  // =========================================================================

  /**
   * Apply confidence decay to workspace knowledge.
   */
  async applyConfidenceDecay(options?: { scope?: 'workspace' | 'all'; decay_rate?: number }): Promise<{ affected: number; average_decay: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/confidence/decay', { json: options });
  }

  /**
   * Record a confidence-affecting event.
   */
  async recordConfidenceEvent(nodeId: string, event: { type: 'validation' | 'contradiction' | 'citation' | 'correction'; impact: number; notes?: string }): Promise<{ new_confidence: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/confidence/event', { json: { node_id: nodeId, ...event } });
  }

  /**
   * Get confidence adjustment history for a node.
   */
  async getConfidenceHistory(nodeId: string, options?: PaginationParams): Promise<{ events: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/confidence/history', { params: { node_id: nodeId, ...options } });
  }

  // =========================================================================
  // Export
  // =========================================================================

  /**
   * Export knowledge graph as D3 JSON format for visualization.
   */
  async exportD3(options?: { scope?: 'workspace' | 'all'; depth?: number }): Promise<{ nodes: unknown[]; links: unknown[] }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/export/d3', { params: options });
  }

  /**
   * Export knowledge graph as GraphML XML format.
   */
  async exportGraphML(options?: { scope?: 'workspace' | 'all' }): Promise<string> {
    return this.client.request('GET', '/api/v1/knowledge/mound/export/graphml', { params: options });
  }

  /**
   * Index a repository as knowledge.
   */
  async indexRepository(repositoryUrl: string, options?: { branch?: string; paths?: string[] }): Promise<{ indexed: number; job_id: string }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/index/repository', { json: { url: repositoryUrl, ...options } });
  }

  // =========================================================================
  // Auto-Curation
  // =========================================================================

  /**
   * Get curation policy for workspace.
   */
  async getCurationPolicy(): Promise<{ policy: unknown }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/curation/policy');
  }

  /**
   * Set curation policy.
   */
  async setCurationPolicy(policy: { auto_promote?: boolean; auto_archive_days?: number; quality_threshold?: number }): Promise<{ updated: boolean }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/curation/policy', { json: policy });
  }

  /**
   * Get curation status.
   */
  async getCurationStatus(): Promise<{ last_run: string; pending_actions: number; health: string }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/curation/status');
  }

  /**
   * Trigger a curation run.
   */
  async runCuration(options?: { dry_run?: boolean }): Promise<{ actions_taken: number; promoted: number; archived: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/curation/run', { json: options });
  }

  /**
   * Get curation history.
   */
  async getCurationHistory(options?: PaginationParams): Promise<{ runs: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/curation/history', { params: options });
  }

  /**
   * Get quality scores by tier.
   */
  async getQualityScores(): Promise<{ scores: unknown }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/curation/scores');
  }

  /**
   * Get tier distribution.
   */
  async getTierDistribution(): Promise<{ distribution: unknown }> {
    return this.client.request('GET', '/api/v1/knowledge/mound/curation/tiers');
  }

  // =========================================================================
  // Synchronization (Legacy Memory Integration)
  // =========================================================================

  /**
   * Sync knowledge from ContinuumMemory.
   */
  async syncFromContinuum(options?: { since?: string; limit?: number }): Promise<{ synced: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/sync/continuum', { json: options });
  }

  /**
   * Sync from ConsensusMemory debate outcomes.
   */
  async syncFromConsensus(options?: { since?: string; limit?: number }): Promise<{ synced: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/sync/consensus', { json: options });
  }

  /**
   * Sync from FactStore.
   */
  async syncFromFacts(options?: { since?: string; limit?: number }): Promise<{ synced: number }> {
    return this.client.request('POST', '/api/v1/knowledge/mound/sync/facts', { json: options });
  }
}
