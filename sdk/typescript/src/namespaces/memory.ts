/**
 * Memory Namespace API
 *
 * Provides a namespaced interface for memory management operations.
 * This wraps the flat client methods for a more intuitive API.
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
  pressure_level: 'low' | 'medium' | 'high' | 'critical';
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
  sort_order?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
  include_metadata?: boolean;
}

/**
 * Interface for the internal client methods used by MemoryAPI.
 */
interface MemoryClientInterface {
  storeMemory(key: string, value: unknown, options?: {
    tier?: 'fast' | 'medium' | 'slow' | 'glacial';
    importance?: number;
    tags?: string[];
    ttl_seconds?: number;
  }): Promise<{ stored: boolean; tier: string }>;
  retrieveMemory(key: string, options?: {
    tier?: 'fast' | 'medium' | 'slow' | 'glacial';
  }): Promise<{ value: unknown; tier: string; metadata?: Record<string, unknown> } | null>;
  deleteMemory(key: string, tier?: 'fast' | 'medium' | 'slow' | 'glacial'): Promise<{ deleted: boolean }>;
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
    options?: { params?: Record<string, unknown>; body?: unknown }
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
    fromTier: 'fast' | 'medium' | 'slow' | 'glacial',
    toTier: 'fast' | 'medium' | 'slow' | 'glacial'
  ): Promise<{ moved: boolean; key: string; from_tier: string; to_tier: string }> {
    return this.client.request('POST', `/api/v1/memory/${encodeURIComponent(key)}/move`, {
      body: { from_tier: fromTier, to_tier: toTier },
    });
  }
}
