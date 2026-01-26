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
  searchKnowledge(query: string, options?: KnowledgeSearchOptions): Promise<{ results: KnowledgeSearchResult[] }>;
  addKnowledge(entry: KnowledgeEntry): Promise<{ id: string; created_at: string }>;
  getKnowledgeEntry(entryId: string): Promise<KnowledgeEntry>;
  updateKnowledge(entryId: string, updates: Partial<KnowledgeEntry>): Promise<KnowledgeEntry>;
  deleteKnowledge(entryId: string): Promise<{ deleted: boolean }>;
  getKnowledgeStats(): Promise<KnowledgeStats>;
  bulkImportKnowledge(entries: KnowledgeEntry[]): Promise<{ imported: number; errors: string[] }>;
  queryKnowledgeMound(query: string, options?: KnowledgeMoundQueryOptions): Promise<KnowledgeMoundQueryResult>;
  listKnowledgeMoundNodes(options?: PaginationParams & { type?: string }): Promise<{ nodes: KnowledgeMoundNode[] }>;
  createKnowledgeMoundNode(node: Partial<KnowledgeMoundNode>): Promise<KnowledgeMoundNode>;
  listKnowledgeMoundRelationships(options?: PaginationParams): Promise<{ relationships: KnowledgeMoundRelationship[] }>;
  createKnowledgeMoundRelationship(relationship: Partial<KnowledgeMoundRelationship>): Promise<KnowledgeMoundRelationship>;
  getKnowledgeMoundStats(): Promise<KnowledgeMoundStats>;
  getStaleKnowledge(options?: { days?: number; limit?: number }): Promise<{ entries: KnowledgeEntry[] }>;
  revalidateKnowledge(nodeId: string, validation: { valid: boolean; notes?: string }): Promise<KnowledgeMoundNode>;
  getKnowledgeLineage(nodeId: string, options?: { depth?: number }): Promise<{ lineage: KnowledgeMoundNode[] }>;
  getRelatedKnowledge(nodeId: string, options?: { types?: string[]; limit?: number }): Promise<{ related: KnowledgeMoundNode[] }>;
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
  async bulkImport(entries: KnowledgeEntry[]): Promise<{ imported: number; errors: string[] }> {
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
  async createNode(node: Partial<KnowledgeMoundNode>): Promise<KnowledgeMoundNode> {
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
  async createRelationship(relationship: Partial<KnowledgeMoundRelationship>): Promise<KnowledgeMoundRelationship> {
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
  async getStale(options?: { days?: number; limit?: number }): Promise<{ entries: KnowledgeEntry[] }> {
    return this.client.getStaleKnowledge(options);
  }

  /**
   * Revalidate a knowledge node.
   */
  async revalidate(nodeId: string, validation: { valid: boolean; notes?: string }): Promise<KnowledgeMoundNode> {
    return this.client.revalidateKnowledge(nodeId, validation);
  }

  /**
   * Get the lineage/provenance of a knowledge node.
   */
  async getLineage(nodeId: string, options?: { depth?: number }): Promise<{ lineage: KnowledgeMoundNode[] }> {
    return this.client.getKnowledgeLineage(nodeId, options);
  }

  /**
   * Get knowledge nodes related to a given node.
   */
  async getRelated(nodeId: string, options?: { types?: string[]; limit?: number }): Promise<{ related: KnowledgeMoundNode[] }> {
    return this.client.getRelatedKnowledge(nodeId, options);
  }
}
