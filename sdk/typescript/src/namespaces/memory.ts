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
  MemorySearchResult,
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
 * Interface for the internal client methods used by MemoryAPI.
 */
interface MemoryClientInterface {
  storeMemory(key: string, value: unknown, options?: { ttl?: number }): Promise<{ success: boolean }>;
  retrieveMemory(key: string): Promise<MemoryEntry | null>;
  deleteMemory(key: string): Promise<{ deleted: boolean }>;
  getMemoryStats(): Promise<MemoryStats>;
  searchMemory(query: string, options?: MemoryRetrieveOptions): Promise<MemorySearchResult>;
  getMemoryTiers(): Promise<{ tiers: Array<{ name: MemoryTier; count: number; size: number }> }>;
  getMemoryCritiques(options?: PaginationParams): Promise<{ critiques: CritiqueEntry[] }>;
  storeToContinuum(content: string, options?: ContinuumStoreOptions): Promise<ContinuumStoreResult>;
  retrieveFromContinuum(query: string, options?: ContinuumRetrieveOptions): Promise<MemoryEntry[]>;
  getContinuumStats(): Promise<MemoryStats>;
  consolidateMemory(): Promise<{ consolidated: number; freed: number }>;
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
  async retrieve(key: string): Promise<MemoryEntry | null> {
    return this.client.retrieveMemory(key);
  }

  /**
   * Delete a memory entry by key.
   */
  async delete(key: string): Promise<{ deleted: boolean }> {
    return this.client.deleteMemory(key);
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
  async search(query: string, options?: MemoryRetrieveOptions): Promise<MemorySearchResult> {
    return this.client.searchMemory(query, options);
  }

  /**
   * Get information about memory tiers.
   */
  async tiers(): Promise<{ tiers: Array<{ name: MemoryTier; count: number; size: number }> }> {
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
  ): Promise<MemoryEntry[]> {
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
  async consolidate(): Promise<{ consolidated: number; freed: number }> {
    return this.client.consolidateMemory();
  }
}
