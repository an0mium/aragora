/**
 * Memory Namespace API
 *
 * Provides a namespaced interface for memory management operations:
 * - Core CRUD operations (store, retrieve, update, delete)
 * - Continuum tier management (fast/medium/slow/glacial operations)
 * - Memory search/query (semantic search, filtered queries)
 * - Cross-debate memory (institutional knowledge)
 * - Memory export/import (backup/restore)
 * - Memory snapshots (point-in-time recovery)
 * - Memory analytics/stats
 * - Context management
 * - Maintenance operations (prune, compact, sync, vacuum)
 */

import type {
  MemoryEntry,
  MemoryStats,
  MemoryTier,
  MemoryTierStats,
  MemorySearchParams,
  CritiqueEntry,
  ContinuumStoreOptions,
  ContinuumStoreResult,
  ContinuumRetrieveOptions,
  PaginationParams,
} from '../types';

/**
 * Memory tier type alias.
 */
export type MemoryTierType = 'fast' | 'medium' | 'slow' | 'glacial';

/**
 * Conflict resolution strategy.
 */
export type ConflictResolution = 'latest_wins' | 'merge' | 'manual';

/**
 * Sort order type.
 */
export type SortOrder = 'asc' | 'desc';

/**
 * Pressure level type.
 */
export type PressureLevel = 'low' | 'medium' | 'high' | 'critical';

/**
 * Options for storing memory entries.
 */
export interface MemoryStoreOptions {
  /** Key for the memory entry */
  key: string;
  /** Value to store */
  value: unknown;
  /** Optional TTL in seconds */
  ttl?: number;
  /** Optional tags for categorization */
  tags?: string[];
  /** Optional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Options for retrieving memory entries.
 */
export interface MemoryRetrieveOptions {
  /** Filter by memory tier */
  tier?: MemoryTier;
  /** Filter by tags */
  tags?: string[];
  /** Maximum number of results */
  limit?: number;
}

/**
 * Memory tier statistics response.
 */
export interface MemoryTierStatsResponse {
  fast: { count: number; size_bytes: number; oldest?: string };
  medium: { count: number; size_bytes: number; oldest?: string };
  slow: { count: number; size_bytes: number; oldest?: string };
  glacial: { count: number; size_bytes: number; oldest?: string };
}

/**
 * Memory archive statistics response.
 */
export interface MemoryArchiveStatsResponse {
  total_archived: number;
  archive_size_bytes: number;
  oldest_entry?: string;
  newest_entry?: string;
  compression_ratio?: number;
}

/**
 * Memory pressure response.
 */
export interface MemoryPressureResponse {
  utilization: number;
  pressure_level: PressureLevel;
  recommendations?: string[];
  by_tier?: Record<string, { utilization: number; entries: number }>;
}

/**
 * Memory context response.
 */
export interface MemoryContext {
  context_id: string;
  data: Record<string, unknown>;
  created_at: string;
  updated_at?: string;
  expires_at?: string;
}

/**
 * Prune result response.
 */
export interface PruneResult {
  pruned_count: number;
  freed_bytes: number;
  tiers_affected: string[];
}

/**
 * Compact result response.
 */
export interface CompactResult {
  compacted: boolean;
  entries_merged: number;
  space_saved_bytes: number;
}

/**
 * Sync result response.
 */
export interface SyncResult {
  synced: boolean;
  entries_synced: number;
  conflicts_resolved: number;
  last_sync_at: string;
}

/**
 * Query options for memory.
 */
export interface MemoryQueryOptions {
  filter?: Record<string, unknown>;
  sort_by?: string;
  sort_order?: SortOrder;
  limit?: number;
  offset?: number;
  include_metadata?: boolean;
}

/**
 * Semantic search options.
 */
export interface SemanticSearchOptions {
  /** Tiers to search (all if not specified) */
  tiers?: MemoryTierType[];
  /** Maximum results */
  limit?: number;
  /** Minimum similarity threshold (0.0-1.0) */
  min_similarity?: number;
  /** Whether to include embedding vectors */
  include_embeddings?: boolean;
}

/**
 * Semantic search result.
 */
export interface SemanticSearchResult {
  entries: Array<MemoryEntry & { similarity: number }>;
  total: number;
  query_time_ms?: number;
}

/**
 * Cross-debate memory entry.
 */
export interface CrossDebateEntry {
  id: string;
  content: string;
  debate_id: string;
  topic?: string;
  conclusion?: string;
  confidence: number;
  relevance?: number;
  created_at: string;
  metadata?: Record<string, unknown>;
}

/**
 * Cross-debate memory response.
 */
export interface CrossDebateResponse {
  entries: CrossDebateEntry[];
  total: number;
}

/**
 * Memory export options.
 */
export interface MemoryExportOptions {
  /** Tiers to export (all if not specified) */
  tiers?: MemoryTierType[];
  /** Filter by tags */
  tags?: string[];
  /** Export format */
  format?: 'json' | 'msgpack';
  /** Include entry metadata */
  include_metadata?: boolean;
}

/**
 * Memory export result.
 */
export interface MemoryExportResult {
  export_id?: string;
  data?: unknown;
  download_url?: string;
  format: string;
  entries_count: number;
  size_bytes?: number;
  created_at: string;
}

/**
 * Memory import options.
 */
export interface MemoryImportOptions {
  /** Whether to overwrite existing entries */
  overwrite?: boolean;
  /** Force all entries to specific tier */
  target_tier?: MemoryTierType;
}

/**
 * Memory import result.
 */
export interface MemoryImportResult {
  imported_count: number;
  skipped_count: number;
  overwritten_count: number;
  errors?: string[];
}

/**
 * Memory snapshot.
 */
export interface MemorySnapshot {
  id: string;
  name?: string;
  description?: string;
  created_at: string;
  size_bytes: number;
  entries_count: number;
  tiers_included: MemoryTierType[];
  metadata?: Record<string, unknown>;
}

/**
 * Memory analytics response.
 */
export interface MemoryAnalyticsResponse {
  period_start?: string;
  period_end?: string;
  granularity: string;
  data_points: Array<{
    timestamp: string;
    entries_count: number;
    storage_bytes: number;
    operations: {
      stores: number;
      retrieves: number;
      deletes: number;
    };
  }>;
}

/**
 * Vacuum result response.
 */
export interface VacuumResult {
  vacuumed: boolean;
  space_reclaimed_bytes: number;
  duration_ms: number;
}

/**
 * Rebuild index result response.
 */
export interface RebuildIndexResult {
  rebuilt: boolean;
  indexes_rebuilt: number;
  duration_ms: number;
  tier?: MemoryTierType;
}

/**
 * Interface for the internal client methods used by MemoryAPI.
 */
interface MemoryClientInterface {
  storeMemory(key: string, value: unknown, options?: {
    tier?: MemoryTierType;
    importance?: number;
    tags?: string[];
    ttl_seconds?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ stored: boolean; tier: string }>;
  retrieveMemory(key: string, options?: {
    tier?: MemoryTierType;
  }): Promise<{ value: unknown; tier: string; metadata?: Record<string, unknown> } | null>;
  deleteMemory(key: string, tier?: MemoryTierType): Promise<{ deleted: boolean }>;
  getMemoryStats(): Promise<MemoryStats>;
  searchMemory(params: MemorySearchParams): Promise<{ entries: MemoryEntry[] }>;
  getMemoryTiers(): Promise<{ tiers: MemoryTierStats[] }>;
  getMemoryCritiques(options?: PaginationParams): Promise<{ critiques: CritiqueEntry[] }>;
  storeToContinuum(content: string, options?: ContinuumStoreOptions): Promise<ContinuumStoreResult>;
  retrieveFromContinuum(query: string, options?: ContinuumRetrieveOptions): Promise<{ entries: MemoryEntry[] }>;
  getContinuumStats(): Promise<MemoryStats>;
  consolidateMemory(): Promise<{ success: boolean }>;
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown; json?: unknown }
  ): Promise<T>;
}

/**
 * Memory API namespace.
 *
 * Provides methods for managing memory:
 * - Storing and retrieving key-value entries
 * - Managing memory tiers (fast, medium, slow, glacial)
 * - Searching memory content
 * - Working with the continuum memory system
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Store a memory entry
 * await client.memory.store('user-preference', { theme: 'dark' });
 *
 * // Retrieve memory
 * const entry = await client.memory.retrieve('user-preference');
 *
 * // Search memory
 * const results = await client.memory.search('theme settings');
 *
 * // Get memory statistics
 * const stats = await client.memory.stats();
 * ```
 */
export class MemoryAPI {
  constructor(private client: MemoryClientInterface) {}

  /**
   * Store a value in memory.
   */
  async store(
    key: string,
    value: unknown,
    options?: {
      tier?: 'fast' | 'medium' | 'slow' | 'glacial';
      importance?: number;
      tags?: string[];
      ttl_seconds?: number;
    }
  ): Promise<{ stored: boolean; tier: string }> {
    return this.client.storeMemory(key, value, options);
  }

  /**
   * Retrieve a value from memory by key.
   */
  async retrieve(
    key: string,
    options?: { tier?: 'fast' | 'medium' | 'slow' | 'glacial' }
  ): Promise<{ value: unknown; tier: string; metadata?: Record<string, unknown> } | null> {
    return this.client.retrieveMemory(key, options);
  }

  /**
   * Delete a memory entry by key.
   */
  async delete(
    key: string,
    tier?: 'fast' | 'medium' | 'slow' | 'glacial'
  ): Promise<{ deleted: boolean }> {
    return this.client.deleteMemory(key, tier);
  }

  /**
   * Get memory system statistics.
   */
  async stats(): Promise<MemoryStats> {
    return this.client.getMemoryStats();
  }

  /**
   * Search memory entries by query.
   */
  async search(params: MemorySearchParams): Promise<{ entries: MemoryEntry[] }> {
    return this.client.searchMemory(params);
  }

  /**
   * Get information about memory tiers.
   */
  async tiers(): Promise<{ tiers: MemoryTierStats[] }> {
    return this.client.getMemoryTiers();
  }

  /**
   * Get stored critiques from memory.
   */
  async critiques(options?: PaginationParams): Promise<{ critiques: CritiqueEntry[] }> {
    return this.client.getMemoryCritiques(options);
  }

  /**
   * Store content in the continuum memory system.
   */
  async storeToContinuum(
    content: string,
    options?: ContinuumStoreOptions
  ): Promise<ContinuumStoreResult> {
    return this.client.storeToContinuum(content, options);
  }

  /**
   * Retrieve content from the continuum memory system.
   */
  async retrieveFromContinuum(
    query: string,
    options?: ContinuumRetrieveOptions
  ): Promise<{ entries: MemoryEntry[] }> {
    return this.client.retrieveFromContinuum(query, options);
  }

  /**
   * Get continuum memory statistics.
   */
  async continuumStats(): Promise<MemoryStats> {
    return this.client.getContinuumStats();
  }

  /**
   * Consolidate memory by archiving old entries.
   */
  async consolidate(): Promise<{ success: boolean }> {
    return this.client.consolidateMemory();
  }

  // ===========================================================================
  // Update Operations
  // ===========================================================================

  /**
   * Update an existing memory entry.
   *
   * @param key - The key of the entry to update
   * @param value - The new value
   * @param options - Update options
   *
   * @example
   * ```typescript
   * const result = await client.memory.update('user-prefs', { theme: 'light' });
   * if (result.updated) {
   *   console.log(`Updated in tier: ${result.tier}`);
   * }
   * ```
   */
  async update(
    key: string,
    value: unknown,
    options?: {
      tier?: 'fast' | 'medium' | 'slow' | 'glacial';
      merge?: boolean;
      tags?: string[];
    }
  ): Promise<{ updated: boolean; tier: string }> {
    return this.client.request('PUT', `/api/v1/memory/${encodeURIComponent(key)}`, {
      body: { value, ...options },
    });
  }

  // ===========================================================================
  // Query Operations
  // ===========================================================================

  /**
   * Query memory entries with advanced filtering.
   *
   * @param options - Query options including filters, sorting, and pagination
   *
   * @example
   * ```typescript
   * const results = await client.memory.query({
   *   filter: { tags: ['important'] },
   *   sort_by: 'created_at',
   *   sort_order: 'desc',
   *   limit: 10,
   * });
   * ```
   */
  async query(options: MemoryQueryOptions): Promise<{ entries: MemoryEntry[]; total: number }> {
    return this.client.request('POST', '/api/v1/memory/query', {
      body: options,
    });
  }

  // ===========================================================================
  // Context Management
  // ===========================================================================

  /**
   * Get the current memory context.
   *
   * @param contextId - Optional context ID (defaults to current session)
   *
   * @example
   * ```typescript
   * const context = await client.memory.getContext();
   * console.log(`Context data:`, context.data);
   * ```
   */
  async getContext(contextId?: string): Promise<MemoryContext> {
    const params: Record<string, unknown> = {};
    if (contextId) {
      params.context_id = contextId;
    }
    return this.client.request('GET', '/api/v1/memory/context', { params });
  }

  /**
   * Set or update the memory context.
   *
   * @param data - Context data to set
   * @param options - Context options
   *
   * @example
   * ```typescript
   * await client.memory.setContext(
   *   { user_id: '123', session_id: 'abc' },
   *   { ttl_seconds: 3600 }
   * );
   * ```
   */
  async setContext(
    data: Record<string, unknown>,
    options?: { context_id?: string; ttl_seconds?: number }
  ): Promise<MemoryContext> {
    return this.client.request('POST', '/api/v1/memory/context', {
      body: { data, ...options },
    });
  }

  // ===========================================================================
  // Tier Statistics (matching Python SDK)
  // ===========================================================================

  /**
   * Get statistics for all memory tiers.
   *
   * Matches Python SDK's `get_tier_stats()` method.
   *
   * @example
   * ```typescript
   * const stats = await client.memory.getTierStats();
   * console.log(`Fast tier: ${stats.fast.count} entries`);
   * console.log(`Slow tier: ${stats.slow.count} entries`);
   * ```
   */
  async getTierStats(): Promise<MemoryTierStatsResponse> {
    return this.client.request('GET', '/api/v1/memory/tier-stats');
  }

  /**
   * Get archive statistics.
   *
   * Matches Python SDK's `get_archive_stats()` method.
   *
   * @example
   * ```typescript
   * const stats = await client.memory.getArchiveStats();
   * console.log(`Total archived: ${stats.total_archived}`);
   * console.log(`Archive size: ${stats.archive_size_bytes} bytes`);
   * ```
   */
  async getArchiveStats(): Promise<MemoryArchiveStatsResponse> {
    return this.client.request('GET', '/api/v1/memory/archive-stats');
  }

  /**
   * Get memory pressure and utilization metrics.
   *
   * Matches Python SDK's `get_pressure()` method.
   *
   * @example
   * ```typescript
   * const pressure = await client.memory.getPressure();
   * if (pressure.pressure_level === 'critical') {
   *   console.log('Memory pressure is critical!');
   *   console.log('Recommendations:', pressure.recommendations);
   * }
   * ```
   */
  async getPressure(): Promise<MemoryPressureResponse> {
    return this.client.request('GET', '/api/v1/memory/pressure');
  }

  /**
   * List all memory tiers with detailed statistics.
   *
   * Matches Python SDK's `list_tiers()` method.
   * This is an alias for `tiers()` but matches the Python SDK naming.
   *
   * @example
   * ```typescript
   * const { tiers } = await client.memory.listTiers();
   * for (const tier of tiers) {
   *   console.log(`${tier.name}: ${tier.entry_count} entries`);
   * }
   * ```
   */
  async listTiers(): Promise<{ tiers: MemoryTierStats[] }> {
    return this.client.request('GET', '/api/v1/memory/tiers');
  }

  /**
   * List critique store entries.
   *
   * Matches Python SDK's `list_critiques()` method.
   *
   * @param options - Filter options
   *
   * @example
   * ```typescript
   * const { critiques } = await client.memory.listCritiques({
   *   agent: 'claude',
   *   limit: 20,
   * });
   * ```
   */
  async listCritiques(options?: {
    agent?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ critiques: CritiqueEntry[] }> {
    return this.client.request('GET', '/api/v1/memory/critiques', {
      params: options as Record<string, unknown>,
    });
  }

  // ===========================================================================
  // Continuum Operations (matching Python SDK)
  // ===========================================================================

  /**
   * Retrieve memories from the continuum.
   *
   * Matches Python SDK's `retrieve_continuum()` method.
   *
   * @param query - Search query
   * @param options - Retrieval options
   *
   * @example
   * ```typescript
   * const result = await client.memory.retrieveContinuum('user preferences', {
   *   tiers: ['fast', 'medium'],
   *   limit: 10,
   *   min_importance: 0.5,
   * });
   * ```
   */
  async retrieveContinuum(
    query: string = '',
    options?: {
      tiers?: Array<'fast' | 'medium' | 'slow' | 'glacial'>;
      limit?: number;
      min_importance?: number;
    }
  ): Promise<{ entries: MemoryEntry[]; total: number }> {
    const params: Record<string, unknown> = {
      query,
      limit: options?.limit ?? 10,
      min_importance: options?.min_importance ?? 0.0,
    };
    if (options?.tiers) {
      params.tiers = options.tiers.join(',');
    }
    return this.client.request('GET', '/api/v1/memory/continuum/retrieve', { params });
  }

  // ===========================================================================
  // Maintenance Operations
  // ===========================================================================

  /**
   * Prune old or low-importance memory entries.
   *
   * @param options - Prune options
   *
   * @example
   * ```typescript
   * const result = await client.memory.prune({
   *   older_than_days: 30,
   *   min_importance: 0.1,
   *   tiers: ['fast', 'medium'],
   * });
   * console.log(`Pruned ${result.pruned_count} entries`);
   * console.log(`Freed ${result.freed_bytes} bytes`);
   * ```
   */
  async prune(options?: {
    older_than_days?: number;
    min_importance?: number;
    tiers?: Array<'fast' | 'medium' | 'slow' | 'glacial'>;
    dry_run?: boolean;
  }): Promise<PruneResult> {
    return this.client.request('POST', '/api/v1/memory/prune', {
      body: options,
    });
  }

  /**
   * Compact memory storage by merging related entries.
   *
   * @param options - Compact options
   *
   * @example
   * ```typescript
   * const result = await client.memory.compact({
   *   tier: 'slow',
   *   merge_threshold: 0.9,
   * });
   * console.log(`Merged ${result.entries_merged} entries`);
   * console.log(`Saved ${result.space_saved_bytes} bytes`);
   * ```
   */
  async compact(options?: {
    tier?: 'fast' | 'medium' | 'slow' | 'glacial';
    merge_threshold?: number;
  }): Promise<CompactResult> {
    return this.client.request('POST', '/api/v1/memory/compact', {
      body: options,
    });
  }

  /**
   * Synchronize memory across distributed systems.
   *
   * @param options - Sync options
   *
   * @example
   * ```typescript
   * const result = await client.memory.sync({
   *   target: 'all',
   *   conflict_resolution: 'latest_wins',
   * });
   * if (result.synced) {
   *   console.log(`Synced ${result.entries_synced} entries`);
   * }
   * ```
   */
  async sync(options?: {
    target?: string;
    conflict_resolution?: 'latest_wins' | 'merge' | 'manual';
    tiers?: Array<'fast' | 'medium' | 'slow' | 'glacial'>;
  }): Promise<SyncResult> {
    return this.client.request('POST', '/api/v1/memory/sync', {
      body: options,
    });
  }

  // ===========================================================================
  // Tier-Specific Operations
  // ===========================================================================

  /**
   * Get entries from a specific tier.
   *
   * @param tier - The memory tier
   * @param options - Retrieval options
   *
   * @example
   * ```typescript
   * const { entries } = await client.memory.getTier('fast', { limit: 50 });
   * console.log(`${entries.length} entries in fast tier`);
   * ```
   */
  async getTier(
    tier: 'fast' | 'medium' | 'slow' | 'glacial',
    options?: { limit?: number; offset?: number }
  ): Promise<{ entries: MemoryEntry[]; total: number }> {
    return this.client.request('GET', `/api/v1/memory/tier/${tier}`, {
      params: options as Record<string, unknown>,
    });
  }

  /**
   * Move an entry between tiers.
   *
   * @param key - The entry key
   * @param fromTier - Source tier
   * @param toTier - Destination tier
   *
   * @example
   * ```typescript
   * await client.memory.moveTier('important-data', 'fast', 'slow');
   * ```
   */
  async moveTier(
    key: string,
    fromTier: MemoryTierType,
    toTier: MemoryTierType
  ): Promise<{ moved: boolean; key: string; from_tier: string; to_tier: string }> {
    return this.client.request('POST', `/api/v1/memory/${encodeURIComponent(key)}/move`, {
      body: { from_tier: fromTier, to_tier: toTier },
    });
  }

  // ===========================================================================
  // Semantic Search (matching Python SDK)
  // ===========================================================================

  /**
   * Perform semantic search across memory entries.
   *
   * Matches Python SDK's `semantic_search()` method.
   *
   * @param query - Natural language query
   * @param options - Search options
   *
   * @example
   * ```typescript
   * const results = await client.memory.semanticSearch('previous decisions about pricing', {
   *   tiers: ['slow', 'glacial'],
   *   limit: 10,
   *   min_similarity: 0.7,
   * });
   * for (const entry of results.entries) {
   *   console.log(`${entry.content} (similarity: ${entry.similarity})`);
   * }
   * ```
   */
  async semanticSearch(
    query: string,
    options?: SemanticSearchOptions
  ): Promise<SemanticSearchResult> {
    const body: Record<string, unknown> = {
      query,
      limit: options?.limit ?? 10,
      min_similarity: options?.min_similarity ?? 0.7,
      include_embeddings: options?.include_embeddings ?? false,
    };
    if (options?.tiers) {
      body.tiers = options.tiers;
    }
    return this.client.request('POST', '/api/v1/memory/semantic-search', { body });
  }

  // ===========================================================================
  // Cross-Debate Memory (Institutional Knowledge)
  // ===========================================================================

  /**
   * Get cross-debate institutional knowledge.
   *
   * Matches Python SDK's `get_cross_debate()` method.
   *
   * @param options - Retrieval options
   *
   * @example
   * ```typescript
   * const result = await client.memory.getCrossDebate({
   *   topic: 'pricing strategy',
   *   limit: 10,
   *   min_relevance: 0.5,
   * });
   * for (const entry of result.entries) {
   *   console.log(`From debate ${entry.debate_id}: ${entry.content}`);
   * }
   * ```
   */
  async getCrossDebate(options?: {
    topic?: string;
    limit?: number;
    min_relevance?: number;
  }): Promise<CrossDebateResponse> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 10,
      min_relevance: options?.min_relevance ?? 0.5,
    };
    if (options?.topic) {
      params.topic = options.topic;
    }
    return this.client.request('GET', '/api/v1/memory/cross-debate', { params });
  }

  /**
   * Store cross-debate knowledge from a debate outcome.
   *
   * Matches Python SDK's `store_cross_debate()` method.
   *
   * @param content - Knowledge content
   * @param debateId - Source debate ID
   * @param options - Storage options
   *
   * @example
   * ```typescript
   * const result = await client.memory.storeCrossDebate(
   *   'Key insight about pricing: value-based pricing works best for enterprise',
   *   'debate-123',
   *   {
   *     topic: 'pricing',
   *     conclusion: 'Value-based pricing recommended',
   *     confidence: 0.85,
   *   }
   * );
   * ```
   */
  async storeCrossDebate(
    content: string,
    debateId: string,
    options?: {
      topic?: string;
      conclusion?: string;
      confidence?: number;
      metadata?: Record<string, unknown>;
    }
  ): Promise<{ id: string; stored: boolean }> {
    const body: Record<string, unknown> = {
      content,
      debate_id: debateId,
    };
    if (options?.topic) body.topic = options.topic;
    if (options?.conclusion) body.conclusion = options.conclusion;
    if (options?.confidence !== undefined) body.confidence = options.confidence;
    if (options?.metadata) body.metadata = options.metadata;
    return this.client.request('POST', '/api/v1/memory/cross-debate', { body });
  }

  /**
   * Inject institutional knowledge into a debate.
   *
   * Matches Python SDK's `inject_institutional()` method.
   *
   * @param debateId - Target debate ID
   * @param options - Injection options
   *
   * @example
   * ```typescript
   * const result = await client.memory.injectInstitutional('debate-456', {
   *   topic: 'pricing strategy',
   *   max_entries: 5,
   * });
   * console.log(`Injected ${result.injected_count} knowledge entries`);
   * ```
   */
  async injectInstitutional(
    debateId: string,
    options?: {
      topic?: string;
      max_entries?: number;
    }
  ): Promise<{ debate_id: string; injected_count: number }> {
    const body: Record<string, unknown> = {
      debate_id: debateId,
      max_entries: options?.max_entries ?? 5,
    };
    if (options?.topic) body.topic = options.topic;
    return this.client.request('POST', '/api/v1/memory/cross-debate/inject', { body });
  }

  // ===========================================================================
  // Export/Import (Backup/Restore)
  // ===========================================================================

  /**
   * Export memory entries for backup.
   *
   * Matches Python SDK's `export_memory()` method.
   *
   * @param options - Export options
   *
   * @example
   * ```typescript
   * const result = await client.memory.exportMemory({
   *   tiers: ['slow', 'glacial'],
   *   format: 'json',
   *   include_metadata: true,
   * });
   * console.log(`Exported ${result.entries_count} entries`);
   * ```
   */
  async exportMemory(options?: MemoryExportOptions): Promise<MemoryExportResult> {
    const body: Record<string, unknown> = {
      format: options?.format ?? 'json',
      include_metadata: options?.include_metadata ?? true,
    };
    if (options?.tiers) body.tiers = options.tiers;
    if (options?.tags) body.tags = options.tags;
    return this.client.request('POST', '/api/v1/memory/export', { body });
  }

  /**
   * Import memory entries from backup.
   *
   * Matches Python SDK's `import_memory()` method.
   *
   * @param data - Memory data to import
   * @param options - Import options
   *
   * @example
   * ```typescript
   * const result = await client.memory.importMemory(backupData, {
   *   overwrite: false,
   *   target_tier: 'slow',
   * });
   * console.log(`Imported ${result.imported_count} entries`);
   * ```
   */
  async importMemory(
    data: Record<string, unknown> | Array<Record<string, unknown>>,
    options?: MemoryImportOptions
  ): Promise<MemoryImportResult> {
    const body: Record<string, unknown> = {
      data,
      overwrite: options?.overwrite ?? false,
    };
    if (options?.target_tier) body.target_tier = options.target_tier;
    return this.client.request('POST', '/api/v1/memory/import', { body });
  }

  // ===========================================================================
  // Snapshots (Point-in-Time Recovery)
  // ===========================================================================

  /**
   * Create a point-in-time snapshot of memory.
   *
   * Matches Python SDK's `create_snapshot()` method.
   *
   * @param options - Snapshot options
   *
   * @example
   * ```typescript
   * const snapshot = await client.memory.createSnapshot({
   *   name: 'pre-migration-backup',
   *   description: 'Backup before schema migration',
   * });
   * console.log(`Created snapshot ${snapshot.id}`);
   * ```
   */
  async createSnapshot(options?: {
    name?: string;
    description?: string;
  }): Promise<MemorySnapshot> {
    const body: Record<string, unknown> = {};
    if (options?.name) body.name = options.name;
    if (options?.description) body.description = options.description;
    return this.client.request('POST', '/api/v1/memory/snapshots', { body });
  }

  /**
   * List available memory snapshots.
   *
   * Matches Python SDK's `list_snapshots()` method.
   *
   * @param options - Pagination options
   *
   * @example
   * ```typescript
   * const result = await client.memory.listSnapshots({ limit: 10 });
   * for (const snapshot of result.snapshots) {
   *   console.log(`${snapshot.name}: ${snapshot.entries_count} entries`);
   * }
   * ```
   */
  async listSnapshots(options?: {
    limit?: number;
    offset?: number;
  }): Promise<{ snapshots: MemorySnapshot[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 20,
      offset: options?.offset ?? 0,
    };
    return this.client.request('GET', '/api/v1/memory/snapshots', { params });
  }

  /**
   * Restore memory from a snapshot.
   *
   * Matches Python SDK's `restore_snapshot()` method.
   *
   * @param snapshotId - Snapshot ID to restore
   * @param options - Restore options
   *
   * @example
   * ```typescript
   * const result = await client.memory.restoreSnapshot('snapshot-123', {
   *   overwrite: true,
   * });
   * if (result.restored) {
   *   console.log('Memory restored successfully');
   * }
   * ```
   */
  async restoreSnapshot(
    snapshotId: string,
    options?: { overwrite?: boolean }
  ): Promise<{ restored: boolean; snapshot_id: string; entries_restored: number }> {
    const body: Record<string, unknown> = {
      overwrite: options?.overwrite ?? false,
    };
    return this.client.request('POST', `/api/v1/memory/snapshots/${snapshotId}/restore`, { body });
  }

  /**
   * Delete a memory snapshot.
   *
   * Matches Python SDK's `delete_snapshot()` method.
   *
   * @param snapshotId - Snapshot ID to delete
   *
   * @example
   * ```typescript
   * const result = await client.memory.deleteSnapshot('snapshot-123');
   * if (result.deleted) {
   *   console.log('Snapshot deleted');
   * }
   * ```
   */
  async deleteSnapshot(snapshotId: string): Promise<{ deleted: boolean; snapshot_id: string }> {
    return this.client.request('DELETE', `/api/v1/memory/snapshots/${snapshotId}`);
  }

  // ===========================================================================
  // Tier Promotion/Demotion
  // ===========================================================================

  /**
   * Promote an entry to a faster tier.
   *
   * Matches Python SDK's `promote()` method.
   *
   * @param key - The entry key
   * @param options - Promotion options
   *
   * @example
   * ```typescript
   * const result = await client.memory.promote('frequently-accessed-data', {
   *   reason: 'High access frequency',
   * });
   * console.log(`Promoted to tier: ${result.new_tier}`);
   * ```
   */
  async promote(
    key: string,
    options?: { reason?: string }
  ): Promise<{ promoted: boolean; key: string; old_tier: string; new_tier: string }> {
    const body: Record<string, unknown> = {};
    if (options?.reason) body.reason = options.reason;
    return this.client.request('POST', `/api/v1/memory/${encodeURIComponent(key)}/promote`, { body });
  }

  /**
   * Demote an entry to a slower tier.
   *
   * Matches Python SDK's `demote()` method.
   *
   * @param key - The entry key
   * @param options - Demotion options
   *
   * @example
   * ```typescript
   * const result = await client.memory.demote('old-data', {
   *   reason: 'Low access frequency',
   * });
   * console.log(`Demoted to tier: ${result.new_tier}`);
   * ```
   */
  async demote(
    key: string,
    options?: { reason?: string }
  ): Promise<{ demoted: boolean; key: string; old_tier: string; new_tier: string }> {
    const body: Record<string, unknown> = {};
    if (options?.reason) body.reason = options.reason;
    return this.client.request('POST', `/api/v1/memory/${encodeURIComponent(key)}/demote`, { body });
  }

  // ===========================================================================
  // Context Management (Extended)
  // ===========================================================================

  /**
   * Clear the memory context.
   *
   * Matches Python SDK's `clear_context()` method.
   *
   * @param contextId - Context ID to clear (current session if not specified)
   *
   * @example
   * ```typescript
   * await client.memory.clearContext('session-123');
   * ```
   */
  async clearContext(contextId?: string): Promise<{ cleared: boolean; context_id?: string }> {
    const params: Record<string, unknown> = {};
    if (contextId) params.context_id = contextId;
    return this.client.request('DELETE', '/api/v1/memory/context', { params });
  }

  // ===========================================================================
  // Analytics
  // ===========================================================================

  /**
   * Get memory analytics over time.
   *
   * Matches Python SDK's `get_analytics()` method.
   *
   * @param options - Analytics options
   *
   * @example
   * ```typescript
   * const analytics = await client.memory.getAnalytics({
   *   start_time: '2024-01-01T00:00:00Z',
   *   end_time: '2024-01-31T23:59:59Z',
   *   granularity: 'day',
   * });
   * for (const point of analytics.data_points) {
   *   console.log(`${point.timestamp}: ${point.entries_count} entries`);
   * }
   * ```
   */
  async getAnalytics(options?: {
    start_time?: string;
    end_time?: string;
    granularity?: 'minute' | 'hour' | 'day';
  }): Promise<MemoryAnalyticsResponse> {
    const params: Record<string, unknown> = {
      granularity: options?.granularity ?? 'hour',
    };
    if (options?.start_time) params.start_time = options.start_time;
    if (options?.end_time) params.end_time = options.end_time;
    return this.client.request('GET', '/api/v1/memory/analytics', { params });
  }

  // ===========================================================================
  // Additional Maintenance Operations
  // ===========================================================================

  /**
   * Run vacuum operation to reclaim storage space.
   *
   * Matches Python SDK's `vacuum()` method.
   *
   * @example
   * ```typescript
   * const result = await client.memory.vacuum();
   * console.log(`Reclaimed ${result.space_reclaimed_bytes} bytes`);
   * ```
   */
  async vacuum(): Promise<VacuumResult> {
    return this.client.request('POST', '/api/v1/memory/vacuum', { body: {} });
  }

  /**
   * Rebuild memory search indices.
   *
   * Matches Python SDK's `rebuild_index()` method.
   *
   * @param options - Rebuild options
   *
   * @example
   * ```typescript
   * const result = await client.memory.rebuildIndex({ tier: 'slow' });
   * if (result.rebuilt) {
   *   console.log(`Rebuilt ${result.indexes_rebuilt} indexes`);
   * }
   * ```
   */
  async rebuildIndex(options?: {
    tier?: MemoryTierType;
  }): Promise<RebuildIndexResult> {
    const body: Record<string, unknown> = {};
    if (options?.tier) body.tier = options.tier;
    return this.client.request('POST', '/api/v1/memory/rebuild-index', { body });
  }

  // ===========================================================================
  // Critique Operations (Extended)
  // ===========================================================================

  /**
   * Store a critique in memory.
   *
   * Matches Python SDK's `store_critique()` method.
   *
   * @param critique - The critique content
   * @param agent - Agent that generated the critique
   * @param options - Storage options
   *
   * @example
   * ```typescript
   * const result = await client.memory.storeCritique(
   *   'The argument lacks supporting evidence',
   *   'claude',
   *   {
   *     debate_id: 'debate-123',
   *     target_agent: 'gpt4',
   *     score: 0.8,
   *   }
   * );
   * ```
   */
  async storeCritique(
    critique: string,
    agent: string,
    options?: {
      debate_id?: string;
      target_agent?: string;
      score?: number;
      metadata?: Record<string, unknown>;
    }
  ): Promise<{ id: string; stored: boolean }> {
    const body: Record<string, unknown> = {
      critique,
      agent,
    };
    if (options?.debate_id) body.debate_id = options.debate_id;
    if (options?.target_agent) body.target_agent = options.target_agent;
    if (options?.score !== undefined) body.score = options.score;
    if (options?.metadata) body.metadata = options.metadata;
    return this.client.request('POST', '/api/v1/memory/critiques', { body });
  }

  /**
   * List memory entries.
   */
  async listEntries(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/memory/entries', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get memory search index.
   */
  async searchIndex(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/memory/search-index', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * Search memory timeline.
   */
  async searchTimeline(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/memory/search-timeline', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get memory viewer.
   */
  async getViewer(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/memory/viewer', { params }) as Promise<Record<string, unknown>>;
  }
}
