/**
 * Repository Namespace API
 *
 * Provides a namespaced interface for codebase repository indexing operations.
 * This wraps the flat client methods for indexing, entity extraction, and graph queries.
 */

import type { PaginationParams } from '../types';

/**
 * Repository index request.
 */
export interface IndexRepositoryRequest {
  repository_url?: string;
  local_path?: string;
  branch?: string;
  include_patterns?: string[];
  exclude_patterns?: string[];
  max_file_size_kb?: number;
  extract_entities?: boolean;
  build_graph?: boolean;
}

/**
 * Repository index response.
 */
export interface IndexRepositoryResponse {
  index_id: string;
  status: 'pending' | 'indexing' | 'completed' | 'failed';
  started_at: string;
  estimated_completion_at?: string;
}

/**
 * Incremental index request.
 */
export interface IncrementalIndexRequest {
  index_id: string;
  changed_files?: string[];
  since_commit?: string;
}

/**
 * Repository index status.
 */
export interface IndexStatus {
  index_id: string;
  status: 'pending' | 'indexing' | 'completed' | 'failed';
  progress_percent: number;
  files_indexed: number;
  files_total: number;
  entities_extracted: number;
  relationships_found: number;
  started_at: string;
  completed_at?: string;
  error?: string;
}

/**
 * Code entity from repository.
 */
export interface CodeEntity {
  id: string;
  type: 'class' | 'function' | 'method' | 'variable' | 'module' | 'interface' | 'type';
  name: string;
  qualified_name: string;
  file_path: string;
  line_start: number;
  line_end: number;
  language: string;
  docstring?: string;
  signature?: string;
  visibility: 'public' | 'private' | 'protected' | 'internal';
  metadata: Record<string, unknown>;
}

/**
 * Entity relationship.
 */
export interface EntityRelationship {
  source_id: string;
  target_id: string;
  type: 'calls' | 'imports' | 'extends' | 'implements' | 'uses' | 'defines' | 'contains';
  weight: number;
  metadata: Record<string, unknown>;
}

/**
 * Repository graph.
 */
export interface RepositoryGraph {
  nodes: CodeEntity[];
  edges: EntityRelationship[];
  statistics: {
    total_nodes: number;
    total_edges: number;
    avg_degree: number;
    max_depth: number;
  };
}

/**
 * Entity filter options.
 */
export interface EntityFilterParams extends PaginationParams {
  type?: string;
  language?: string;
  file_pattern?: string;
  name_pattern?: string;
  visibility?: string;
}

/**
 * Batch index request for multiple repositories.
 */
export interface BatchIndexRequest {
  repositories: IndexRepositoryRequest[];
  parallel?: boolean;
  max_concurrent?: number;
}

/**
 * Batch index response.
 */
export interface BatchIndexResponse {
  batch_id: string;
  index_ids: string[];
  status: 'pending' | 'processing' | 'completed' | 'partial_failure';
}

/**
 * Interface for the internal client methods used by RepositoryAPI.
 */
interface RepositoryClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: unknown }
  ): Promise<T>;
}

/**
 * Repository API namespace.
 *
 * Provides methods for codebase indexing and entity extraction:
 * - Full and incremental repository indexing
 * - Entity extraction (classes, functions, etc.)
 * - Relationship graph building
 * - Entity search and filtering
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Index a repository
 * const { index_id } = await client.repository.index({
 *   repository_url: 'https://github.com/org/repo',
 *   branch: 'main',
 *   extract_entities: true,
 *   build_graph: true,
 * });
 *
 * // Check status
 * const status = await client.repository.getStatus(index_id);
 *
 * // Get entities
 * const { entities } = await client.repository.listEntities(index_id, {
 *   type: 'class',
 *   language: 'python',
 * });
 *
 * // Get relationship graph
 * const graph = await client.repository.getGraph(index_id);
 * ```
 */
export class RepositoryAPI {
  constructor(private client: RepositoryClientInterface) {}

  /**
   * Start a full repository index.
   */
  async index(request: IndexRepositoryRequest): Promise<IndexRepositoryResponse> {
    return this.client.request<IndexRepositoryResponse>('POST', '/api/v1/repository/index', {
      json: request,
    });
  }

  /**
   * Run an incremental index update.
   */
  async incrementalIndex(request: IncrementalIndexRequest): Promise<IndexRepositoryResponse> {
    return this.client.request<IndexRepositoryResponse>('POST', '/api/v1/repository/incremental', {
      json: request,
    });
  }

  /**
   * Index multiple repositories in batch.
   */
  async batchIndex(request: BatchIndexRequest): Promise<BatchIndexResponse> {
    return this.client.request<BatchIndexResponse>('POST', '/api/v1/repository/batch', {
      json: request,
    });
  }

  /**
   * Get index status.
   */
  async getStatus(indexId: string): Promise<IndexStatus> {
    return this.client.request<IndexStatus>('POST', `/api/v1/repository/${indexId}/status`);
  }

  /**
   * List entities from an indexed repository.
   */
  async listEntities(
    indexId: string,
    params?: EntityFilterParams
  ): Promise<{ entities: CodeEntity[]; total: number }> {
    const queryParams = params ? `?${new URLSearchParams(params as Record<string, string>)}` : '';
    return this.client.request<{ entities: CodeEntity[]; total: number }>(
      'GET',
      `/api/v1/repository/${indexId}/entities${queryParams}`
    );
  }

  /**
   * Get the relationship graph for an indexed repository.
   */
  async getGraph(indexId: string): Promise<RepositoryGraph> {
    return this.client.request<RepositoryGraph>('POST', `/api/v1/repository/${indexId}/graph`);
  }

  /**
   * Get a specific entity by ID.
   */
  async getEntity(indexId: string, entityId: string): Promise<CodeEntity> {
    return this.client.request<CodeEntity>('GET', `/api/v1/repository/${indexId}/entities/${entityId}`);
  }

  /**
   * Wait for index to complete.
   */
  async waitForIndex(
    indexId: string,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<IndexStatus> {
    const pollInterval = options?.pollIntervalMs || 2000;
    const timeout = options?.timeoutMs || 300000; // 5 minutes default
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const status = await this.getStatus(indexId);
      if (status.status === 'completed' || status.status === 'failed') {
        return status;
      }
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Index ${indexId} did not complete within ${timeout}ms`);
  }

  /**
   * Index a repository and wait for completion.
   */
  async indexAndWait(
    request: IndexRepositoryRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<IndexStatus> {
    const { index_id } = await this.index(request);
    return this.waitForIndex(index_id, options);
  }

  /**
   * Get repository info by ID.
   *
   * @param repositoryId - Repository ID
   * @returns Repository details
   */
  async getRepository(repositoryId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/repository/${encodeURIComponent(repositoryId)}`);
  }

  /**
   * Delete a repository index.
   *
   * @param repositoryId - Repository ID
   * @returns Deletion result
   */
  async deleteRepository(repositoryId: string): Promise<{ deleted: boolean }> {
    return this.client.request('GET', `/api/v1/repository/${encodeURIComponent(repositoryId)}`);
  }
}

export default RepositoryAPI;
