/**
 * Aragora SDK Integration Tests
 *
 * Tests for advanced features including:
 * - Batch debate operations
 * - Graph debates
 * - Matrix debates
 * - Evidence API
 * - Memory/Continuum operations
 * - Verification API
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { AragoraClient, AragoraError } from '../src/client';
import { DebateStatus, VerificationStatus } from '../src/types';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Integration Tests', () => {
  let client: AragoraClient;

  beforeEach(() => {
    client = new AragoraClient({
      baseUrl: 'http://localhost:8080',
      apiKey: 'test-api-key',
      retry: { maxRetries: 0 }, // Disable retries for tests
    });
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ===========================================================================
  // Batch Debate Operations
  // ===========================================================================

  describe('Batch Debates API', () => {
    describe('batchDebates.submit()', () => {
      it('should submit a batch of debates', async () => {
        const mockResponse = {
          success: true,
          batch_id: 'batch-123',
          items_queued: 3,
          estimated_completion_minutes: 15,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.batchDebates.submit({
          items: [
            { question: 'Should we use microservices?' },
            { question: 'Is GraphQL better than REST?' },
            { question: 'Should we adopt TypeScript?' },
          ],
          max_parallel: 2,
        });

        expect(result.success).toBe(true);
        expect(result.batch_id).toBe('batch-123');
        expect(result.items_queued).toBe(3);
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/debates/batch',
          expect.objectContaining({
            method: 'POST',
          })
        );
      });

      it('should submit batch with webhook configuration', async () => {
        const mockResponse = {
          success: true,
          batch_id: 'batch-456',
          items_queued: 2,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.batchDebates.submit({
          items: [
            { question: 'Question 1', priority: 1 },
            { question: 'Question 2', priority: 2 },
          ],
          webhook_url: 'https://example.com/webhook',
          webhook_headers: { 'X-Custom-Header': 'value' },
        });

        expect(result.batch_id).toBe('batch-456');

        const callArgs = mockFetch.mock.calls[0];
        const requestBody = JSON.parse(callArgs[1].body);
        expect(requestBody.webhook_url).toBe('https://example.com/webhook');
        expect(requestBody.webhook_headers['X-Custom-Header']).toBe('value');
      });
    });

    describe('batchDebates.status()', () => {
      it('should get batch status', async () => {
        const mockResponse = {
          batch_id: 'batch-123',
          status: 'processing',
          total_items: 3,
          completed_items: 1,
          failed_items: 0,
          items: [
            { question: 'Q1', status: 'completed', debate_id: 'debate-1' },
            { question: 'Q2', status: 'processing' },
            { question: 'Q3', status: 'pending' },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.batchDebates.status('batch-123');

        expect(result.batch_id).toBe('batch-123');
        expect(result.status).toBe('processing');
        expect(result.completed_items).toBe(1);
        expect(result.items).toHaveLength(3);
      });

      it('should return completed status with all items done', async () => {
        const mockResponse = {
          batch_id: 'batch-123',
          status: 'completed',
          total_items: 2,
          completed_items: 2,
          failed_items: 0,
          items: [
            { question: 'Q1', status: 'completed', debate_id: 'debate-1' },
            { question: 'Q2', status: 'completed', debate_id: 'debate-2' },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.batchDebates.status('batch-123');

        expect(result.status).toBe('completed');
        expect(result.completed_items).toBe(2);
      });

      it('should handle partial failure status', async () => {
        const mockResponse = {
          batch_id: 'batch-123',
          status: 'partial_failure',
          total_items: 3,
          completed_items: 2,
          failed_items: 1,
          items: [
            { question: 'Q1', status: 'completed', debate_id: 'debate-1' },
            { question: 'Q2', status: 'completed', debate_id: 'debate-2' },
            { question: 'Q3', status: 'failed', error: 'Timeout exceeded' },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.batchDebates.status('batch-123');

        expect(result.status).toBe('partial_failure');
        expect(result.failed_items).toBe(1);
        expect(result.items[2].error).toBe('Timeout exceeded');
      });
    });

    describe('batchDebates.queueStatus()', () => {
      it('should get queue status', async () => {
        const mockResponse = {
          queue_length: 15,
          processing: 3,
          completed_today: 42,
          failed_today: 2,
          avg_wait_seconds: 120,
          estimated_wait_seconds: 180,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.batchDebates.queueStatus();

        expect(result.queue_length).toBe(15);
        expect(result.processing).toBe(3);
        expect(result.completed_today).toBe(42);
        expect(result.avg_wait_seconds).toBe(120);
      });
    });

    describe('batchDebates.waitForCompletion()', () => {
      it('should poll until batch completes', async () => {
        // First poll: processing
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            batch_id: 'batch-123',
            status: 'processing',
            total_items: 2,
            completed_items: 1,
            failed_items: 0,
            items: [],
          }),
        });

        // Second poll: completed
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            batch_id: 'batch-123',
            status: 'completed',
            total_items: 2,
            completed_items: 2,
            failed_items: 0,
            items: [
              { question: 'Q1', status: 'completed', debate_id: 'debate-1' },
              { question: 'Q2', status: 'completed', debate_id: 'debate-2' },
            ],
          }),
        });

        const result = await client.batchDebates.waitForCompletion('batch-123', {
          pollInterval: 10,
          timeout: 1000,
        });

        expect(result.status).toBe('completed');
        expect(mockFetch).toHaveBeenCalledTimes(2);
      });

      it('should timeout if batch takes too long', async () => {
        mockFetch.mockResolvedValue({
          ok: true,
          json: () => Promise.resolve({
            batch_id: 'batch-123',
            status: 'processing',
            total_items: 2,
            completed_items: 0,
            failed_items: 0,
            items: [],
          }),
        });

        await expect(
          client.batchDebates.waitForCompletion('batch-123', {
            pollInterval: 10,
            timeout: 50,
          })
        ).rejects.toThrow(AragoraError);
      });
    });

    describe('batchDebates.submitAndWait()', () => {
      it('should submit and wait for completion', async () => {
        // Submit response
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            success: true,
            batch_id: 'batch-789',
            items_queued: 1,
          }),
        });

        // Status poll: completed
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            batch_id: 'batch-789',
            status: 'completed',
            total_items: 1,
            completed_items: 1,
            failed_items: 0,
            items: [{ question: 'Q1', status: 'completed', debate_id: 'debate-1' }],
          }),
        });

        const result = await client.batchDebates.submitAndWait(
          { items: [{ question: 'Test question' }] },
          { pollInterval: 10, timeout: 1000 }
        );

        expect(result.status).toBe('completed');
        expect(mockFetch).toHaveBeenCalledTimes(2);
      });
    });
  });

  // ===========================================================================
  // Graph Debates API
  // ===========================================================================

  describe('Graph Debates API', () => {
    describe('graphDebates.create()', () => {
      it('should create a graph debate', async () => {
        const mockResponse = {
          debate_id: 'graph-debate-123',
          status: 'running',
          task: 'Analyze system architecture',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.graphDebates.create({
          task: 'Analyze system architecture',
          agents: ['anthropic-api', 'openai-api'],
          max_rounds: 5,
          branch_threshold: 0.3,
          max_branches: 4,
        });

        expect(result.debate_id).toBe('graph-debate-123');
        expect(result.status).toBe('running');
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/debates/graph',
          expect.objectContaining({
            method: 'POST',
          })
        );
      });
    });

    describe('graphDebates.get()', () => {
      it('should get a graph debate by ID', async () => {
        const mockResponse = {
          debate_id: 'graph-debate-123',
          task: 'Analyze system architecture',
          status: 'completed',
          agents: ['anthropic-api', 'openai-api'],
          branches: [
            {
              branch_id: 'branch-main',
              name: 'main',
              is_main: true,
              nodes: [
                {
                  node_id: 'node-1',
                  content: 'Initial proposal',
                  agent_id: 'anthropic-api',
                  node_type: 'proposal',
                  round: 1,
                },
              ],
            },
            {
              branch_id: 'branch-alt',
              name: 'alternative',
              is_main: false,
              nodes: [
                {
                  node_id: 'node-2',
                  content: 'Alternative approach',
                  agent_id: 'openai-api',
                  node_type: 'proposal',
                  parent_id: 'node-1',
                  round: 2,
                },
              ],
            },
          ],
          consensus: {
            reached: true,
            conclusion: 'Use microservices with event-driven architecture',
            confidence: 0.85,
            supporting_agents: ['anthropic-api', 'openai-api'],
          },
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.graphDebates.get('graph-debate-123');

        expect(result.debate_id).toBe('graph-debate-123');
        expect(result.branches).toHaveLength(2);
        expect(result.branches[0].is_main).toBe(true);
        expect(result.consensus?.reached).toBe(true);
      });
    });

    describe('graphDebates.getBranches()', () => {
      it('should get branches for a graph debate', async () => {
        const mockResponse = {
          branches: [
            {
              branch_id: 'branch-main',
              name: 'main',
              is_main: true,
              nodes: [],
              created_at: '2024-01-01T00:00:00Z',
            },
            {
              branch_id: 'branch-perf',
              name: 'performance-focus',
              is_main: false,
              nodes: [],
              created_at: '2024-01-01T01:00:00Z',
            },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.graphDebates.getBranches('graph-debate-123');

        expect(result).toHaveLength(2);
        expect(result[0].branch_id).toBe('branch-main');
        expect(result[1].name).toBe('performance-focus');
      });
    });

    describe('graphDebates.run()', () => {
      it('should create and wait for graph debate completion', async () => {
        // Create response
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            debate_id: 'graph-new',
            status: 'running',
            task: 'Test task',
          }),
        });

        // Poll: completed
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            debate_id: 'graph-new',
            task: 'Test task',
            status: 'completed',
            agents: [],
            branches: [],
            consensus: { reached: true, confidence: 0.8, supporting_agents: [] },
          }),
        });

        const result = await client.graphDebates.run({ task: 'Test task' });

        expect(result.status).toBe('completed');
      });
    });

    describe('graphDebates.waitForCompletion()', () => {
      it('should poll until graph debate completes', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            debate_id: 'graph-123',
            task: 'Test',
            status: 'running',
            agents: [],
            branches: [],
          }),
        });

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            debate_id: 'graph-123',
            task: 'Test',
            status: 'completed',
            agents: [],
            branches: [],
          }),
        });

        const result = await client.graphDebates.waitForCompletion('graph-123', {
          pollInterval: 10,
          timeout: 1000,
        });

        expect(result.status).toBe('completed');
      });
    });
  });

  // ===========================================================================
  // Matrix Debates API
  // ===========================================================================

  describe('Matrix Debates API', () => {
    describe('matrixDebates.create()', () => {
      it('should create a matrix debate', async () => {
        const mockResponse = {
          matrix_id: 'matrix-123',
          status: 'running',
          task: 'Evaluate caching strategies',
          scenario_count: 3,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.matrixDebates.create({
          task: 'Evaluate caching strategies',
          agents: ['anthropic-api', 'openai-api'],
          scenarios: [
            { name: 'baseline', parameters: { cache_size: 'medium' }, is_baseline: true },
            { name: 'large-cache', parameters: { cache_size: 'large' } },
            { name: 'distributed', parameters: { cache_type: 'distributed' } },
          ],
          max_rounds: 3,
        });

        expect(result.matrix_id).toBe('matrix-123');
        expect(result.scenario_count).toBe(3);
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/debates/matrix',
          expect.objectContaining({
            method: 'POST',
          })
        );
      });
    });

    describe('matrixDebates.get()', () => {
      it('should get a matrix debate by ID', async () => {
        const mockResponse = {
          matrix_id: 'matrix-123',
          task: 'Evaluate caching strategies',
          status: 'completed',
          agents: ['anthropic-api', 'openai-api'],
          scenarios: [
            {
              scenario_name: 'baseline',
              consensus: { reached: true, confidence: 0.9, supporting_agents: [] },
              key_findings: ['LRU cache is effective for read-heavy workloads'],
            },
            {
              scenario_name: 'large-cache',
              consensus: { reached: true, confidence: 0.85, supporting_agents: [] },
              key_findings: ['Larger cache improves hit rate by 15%'],
              differences_from_baseline: ['Higher memory usage', 'Longer warmup time'],
            },
          ],
          conclusions: {
            universal: ['Caching improves response times'],
            conditional: { 'high-memory': ['Use large cache for read-heavy workloads'] },
            contradictions: [],
          },
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.matrixDebates.get('matrix-123');

        expect(result.matrix_id).toBe('matrix-123');
        expect(result.scenarios).toHaveLength(2);
        expect(result.conclusions?.universal).toContain('Caching improves response times');
      });
    });

    describe('matrixDebates.getConclusions()', () => {
      it('should get conclusions for a matrix debate', async () => {
        const mockResponse = {
          universal: ['All scenarios benefit from caching'],
          conditional: {
            'write-heavy': ['Consider write-through caching'],
            'read-heavy': ['Larger cache size recommended'],
          },
          contradictions: ['Scenario A suggests X while Scenario B suggests Y'],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.matrixDebates.getConclusions('matrix-123');

        expect(result.universal).toHaveLength(1);
        expect(result.conditional['write-heavy']).toContain('Consider write-through caching');
        expect(result.contradictions).toHaveLength(1);
      });
    });

    describe('matrixDebates.run()', () => {
      it('should create and wait for matrix debate completion', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            matrix_id: 'matrix-new',
            status: 'running',
            task: 'Test',
            scenario_count: 2,
          }),
        });

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            matrix_id: 'matrix-new',
            task: 'Test',
            status: 'completed',
            agents: [],
            scenarios: [],
          }),
        });

        const result = await client.matrixDebates.run({ task: 'Test' });

        expect(result.status).toBe('completed');
      });
    });

    describe('matrixDebates.waitForCompletion()', () => {
      it('should timeout if matrix debate takes too long', async () => {
        mockFetch.mockResolvedValue({
          ok: true,
          json: () => Promise.resolve({
            matrix_id: 'matrix-slow',
            task: 'Slow task',
            status: 'running',
            agents: [],
            scenarios: [],
          }),
        });

        await expect(
          client.matrixDebates.waitForCompletion('matrix-slow', {
            pollInterval: 10,
            timeout: 50,
          })
        ).rejects.toThrow(AragoraError);
      });
    });
  });

  // ===========================================================================
  // Evidence API
  // ===========================================================================

  describe('Evidence API', () => {
    describe('evidence.list()', () => {
      it('should list evidence with pagination', async () => {
        const mockResponse = {
          evidence: [
            {
              id: 'ev-1',
              source: 'arxiv',
              title: 'Research Paper 1',
              snippet: 'Key finding from research...',
              reliability_score: 0.9,
            },
            {
              id: 'ev-2',
              source: 'wikipedia',
              title: 'Encyclopedia Entry',
              snippet: 'Background information...',
              reliability_score: 0.75,
            },
          ],
          total: 50,
          limit: 10,
          offset: 0,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.list({ limit: 10, offset: 0 });

        expect(result.evidence).toHaveLength(2);
        expect(result.total).toBe(50);
      });

      it('should filter evidence by source and reliability', async () => {
        const mockResponse = {
          evidence: [
            {
              id: 'ev-1',
              source: 'arxiv',
              title: 'High Quality Paper',
              snippet: 'Research finding...',
              reliability_score: 0.95,
            },
          ],
          total: 1,
          limit: 10,
          offset: 0,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.list({
          source: 'arxiv',
          min_reliability: 0.9,
        });

        expect(result.evidence).toHaveLength(1);
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('source=arxiv'),
          expect.anything()
        );
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('min_reliability=0.9'),
          expect.anything()
        );
      });
    });

    describe('evidence.get()', () => {
      it('should get evidence by ID', async () => {
        const mockResponse = {
          evidence: {
            id: 'ev-123',
            source: 'pubmed',
            title: 'Medical Study',
            snippet: 'Study findings indicate...',
            url: 'https://pubmed.ncbi.nlm.nih.gov/12345',
            reliability_score: 0.92,
            freshness_score: 0.8,
            metadata: { authors: ['Dr. Smith'], year: 2024 },
          },
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.get('ev-123');

        expect(result.id).toBe('ev-123');
        expect(result.source).toBe('pubmed');
        expect(result.reliability_score).toBe(0.92);
      });
    });

    describe('evidence.search()', () => {
      it('should search evidence with query', async () => {
        const mockResponse = {
          query: 'machine learning optimization',
          results: [
            {
              id: 'ev-1',
              source: 'arxiv',
              title: 'ML Optimization Techniques',
              snippet: 'Recent advances in ML optimization...',
              reliability_score: 0.88,
            },
            {
              id: 'ev-2',
              source: 'papers',
              title: 'Deep Learning Optimizers',
              snippet: 'Comparison of Adam, SGD...',
              reliability_score: 0.85,
            },
          ],
          count: 2,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.search({
          query: 'machine learning optimization',
          limit: 10,
          min_reliability: 0.8,
        });

        expect(result.query).toBe('machine learning optimization');
        expect(result.results).toHaveLength(2);
        expect(result.count).toBe(2);
      });

      it('should search with context options', async () => {
        const mockResponse = {
          query: 'climate change',
          results: [],
          count: 0,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        await client.evidence.search({
          query: 'climate change',
          context: {
            topic: 'environmental science',
            required_sources: ['nature', 'science'],
            recency_weight: 0.8,
          },
        });

        const callArgs = mockFetch.mock.calls[0];
        const requestBody = JSON.parse(callArgs[1].body);
        expect(requestBody.context.topic).toBe('environmental science');
        expect(requestBody.context.required_sources).toContain('nature');
      });
    });

    describe('evidence.collect()', () => {
      it('should collect evidence for a task', async () => {
        const mockResponse = {
          task: 'Analyze AI safety approaches',
          keywords: ['AI safety', 'alignment', 'interpretability'],
          snippets: [
            {
              id: 'ev-new-1',
              source: 'arxiv',
              title: 'AI Alignment Research',
              snippet: 'Current approaches to AI alignment...',
              reliability_score: 0.9,
            },
          ],
          count: 1,
          total_searched: 150,
          average_reliability: 0.85,
          average_freshness: 0.7,
          saved_ids: ['ev-new-1'],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.collect({
          task: 'Analyze AI safety approaches',
          connectors: ['arxiv', 'semantic_scholar'],
        });

        expect(result.task).toBe('Analyze AI safety approaches');
        expect(result.keywords).toContain('AI safety');
        expect(result.snippets).toHaveLength(1);
        expect(result.saved_ids).toContain('ev-new-1');
      });

      it('should collect evidence for a specific debate', async () => {
        const mockResponse = {
          task: 'Database selection',
          keywords: ['database', 'SQL', 'NoSQL'],
          snippets: [],
          count: 0,
          total_searched: 50,
          average_reliability: 0,
          average_freshness: 0,
          saved_ids: [],
          debate_id: 'debate-456',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.collect({
          task: 'Database selection',
          debate_id: 'debate-456',
          round: 2,
        });

        expect(result.debate_id).toBe('debate-456');
      });
    });

    describe('evidence.forDebate()', () => {
      it('should get evidence for a debate', async () => {
        const mockResponse = {
          debate_id: 'debate-123',
          evidence: [
            {
              id: 'ev-1',
              source: 'web',
              title: 'Supporting Article',
              snippet: 'Evidence supporting position...',
              reliability_score: 0.8,
            },
          ],
          count: 1,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.forDebate('debate-123');

        expect(result.debate_id).toBe('debate-123');
        expect(result.evidence).toHaveLength(1);
      });

      it('should get evidence for a specific round', async () => {
        const mockResponse = {
          debate_id: 'debate-123',
          round: 2,
          evidence: [],
          count: 0,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        await client.evidence.forDebate('debate-123', 2);

        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('round=2'),
          expect.anything()
        );
      });
    });

    describe('evidence.associateWithDebate()', () => {
      it('should associate evidence with a debate', async () => {
        const mockResponse = {
          debate_id: 'debate-123',
          associated: ['ev-1', 'ev-2'],
          count: 2,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.associateWithDebate(
          'debate-123',
          ['ev-1', 'ev-2'],
          { round: 1, relevance_score: 0.9 }
        );

        expect(result.debate_id).toBe('debate-123');
        expect(result.associated).toHaveLength(2);
        expect(result.count).toBe(2);
      });
    });

    describe('evidence.statistics()', () => {
      it('should get evidence statistics', async () => {
        const mockResponse = {
          statistics: {
            total_evidence: 1500,
            total_debates: 200,
            avg_reliability: 0.82,
            sources_distribution: {
              arxiv: 500,
              wikipedia: 400,
              web: 300,
              pubmed: 200,
              other: 100,
            },
          },
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.statistics();

        expect(result.total_evidence).toBe(1500);
        expect(result.avg_reliability).toBe(0.82);
        expect(result.sources_distribution['arxiv']).toBe(500);
      });
    });

    describe('evidence.delete()', () => {
      it('should delete evidence by ID', async () => {
        const mockResponse = { deleted: true };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.evidence.delete('ev-123');

        expect(result).toBe(true);
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/evidence/ev-123',
          expect.objectContaining({ method: 'DELETE' })
        );
      });
    });
  });

  // ===========================================================================
  // Memory/Continuum API
  // ===========================================================================

  describe('Memory API', () => {
    describe('memory.analytics()', () => {
      it('should get memory analytics', async () => {
        const mockResponse = {
          tiers: [
            {
              tier_name: 'fast',
              entry_count: 100,
              avg_access_frequency: 15,
              promotion_rate: 0.2,
              demotion_rate: 0.05,
              hit_rate: 0.95,
            },
            {
              tier_name: 'medium',
              entry_count: 500,
              avg_access_frequency: 5,
              promotion_rate: 0.15,
              demotion_rate: 0.1,
              hit_rate: 0.8,
            },
            {
              tier_name: 'slow',
              entry_count: 2000,
              avg_access_frequency: 1,
              promotion_rate: 0.05,
              demotion_rate: 0.2,
              hit_rate: 0.6,
            },
          ],
          total_entries: 2600,
          learning_velocity: 0.85,
          promotion_effectiveness: 0.72,
          recommendations: [
            { type: 'promotion', description: 'Consider promoting frequently accessed slow tier items', impact: 'medium' },
          ],
          period_days: 30,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.analytics(30);

        expect(result.tiers).toHaveLength(3);
        expect(result.total_entries).toBe(2600);
        expect(result.learning_velocity).toBe(0.85);
      });
    });

    describe('memory.retrieve()', () => {
      it('should retrieve memories from continuum', async () => {
        const mockResponse = {
          memories: [
            {
              memory_id: 'mem-1',
              content: 'Previous debate conclusion about microservices',
              tier: 'fast',
              created_at: '2024-01-01T00:00:00Z',
              access_count: 15,
            },
            {
              memory_id: 'mem-2',
              content: 'Learning about API design patterns',
              tier: 'fast',
              created_at: '2024-01-02T00:00:00Z',
              access_count: 10,
            },
          ],
          tier: 'fast',
          total: 100,
          limit: 10,
          offset: 0,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.retrieve({ tier: 'fast', limit: 10 });

        expect(result.memories).toHaveLength(2);
        expect(result.tier).toBe('fast');
        expect(result.total).toBe(100);
      });

      it('should retrieve memories with query filter', async () => {
        const mockResponse = {
          memories: [],
          tier: 'all',
          total: 0,
          limit: 10,
          offset: 0,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        await client.memory.retrieve({ query: 'microservices architecture' });

        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('query=microservices'),
          expect.anything()
        );
      });
    });

    describe('memory.consolidate()', () => {
      it('should trigger memory consolidation', async () => {
        const mockResponse = {
          success: true,
          memories_consolidated: 150,
          promotions: 25,
          demotions: 40,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.consolidate();

        expect(result.success).toBe(true);
        expect(result.memories_consolidated).toBe(150);
        expect(result.promotions).toBe(25);
        expect(result.demotions).toBe(40);
      });
    });

    describe('memory.cleanup()', () => {
      it('should cleanup expired memories', async () => {
        const mockResponse = {
          success: true,
          memories_removed: 50,
          bytes_freed: 1024000,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.cleanup({ tier: 'glacial', max_age_days: 30 });

        expect(result.success).toBe(true);
        expect(result.memories_removed).toBe(50);
        expect(result.bytes_freed).toBe(1024000);
      });
    });

    describe('memory.tiers()', () => {
      it('should get tier statistics', async () => {
        const mockResponse = {
          tiers: [
            { tier_name: 'fast', entry_count: 100, avg_access_frequency: 10, promotion_rate: 0.2, demotion_rate: 0.05, hit_rate: 0.9 },
            { tier_name: 'medium', entry_count: 500, avg_access_frequency: 3, promotion_rate: 0.1, demotion_rate: 0.1, hit_rate: 0.7 },
            { tier_name: 'slow', entry_count: 1000, avg_access_frequency: 0.5, promotion_rate: 0.05, demotion_rate: 0.15, hit_rate: 0.5 },
            { tier_name: 'glacial', entry_count: 5000, avg_access_frequency: 0.1, promotion_rate: 0.01, demotion_rate: 0.3, hit_rate: 0.2 },
          ],
          total_entries: 6600,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.tiers();

        expect(result.tiers).toHaveLength(4);
        expect(result.total_entries).toBe(6600);
      });
    });

    describe('memory.archiveStats()', () => {
      it('should get archive statistics', async () => {
        const mockResponse = {
          total_archived: 10000,
          size_bytes: 52428800,
          oldest_entry: '2023-01-01T00:00:00Z',
          newest_entry: '2024-01-15T00:00:00Z',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.archiveStats();

        expect(result.total_archived).toBe(10000);
        expect(result.size_bytes).toBe(52428800);
      });
    });

    describe('memory.pressure()', () => {
      it('should get memory pressure status', async () => {
        const mockResponse = {
          pressure_level: 'medium',
          utilization_percent: 65,
          recommended_action: 'Consider running cleanup on glacial tier',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.pressure();

        expect(result.pressure_level).toBe('medium');
        expect(result.utilization_percent).toBe(65);
        expect(result.recommended_action).toBeDefined();
      });

      it('should report critical pressure level', async () => {
        const mockResponse = {
          pressure_level: 'critical',
          utilization_percent: 95,
          recommended_action: 'Immediate cleanup required',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.pressure();

        expect(result.pressure_level).toBe('critical');
        expect(result.utilization_percent).toBe(95);
      });
    });

    describe('memory.delete()', () => {
      it('should delete a memory by ID', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: true }),
        });

        const result = await client.memory.delete('mem-123');

        expect(result).toBe(true);
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/memory/continuum/mem-123',
          expect.objectContaining({ method: 'DELETE' })
        );
      });
    });

    describe('memory.snapshot()', () => {
      it('should create a memory snapshot', async () => {
        const mockResponse = {
          snapshot_id: 'snap-123',
          timestamp: '2024-01-15T12:00:00Z',
          success: true,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.snapshot();

        expect(result.snapshot_id).toBe('snap-123');
        expect(result.success).toBe(true);
      });
    });
  });

  // ===========================================================================
  // Verification API
  // ===========================================================================

  describe('Verification API', () => {
    describe('verification.verify()', () => {
      it('should verify a valid claim', async () => {
        const mockResponse = {
          status: 'valid',
          claim: 'All primes greater than 2 are odd',
          formal_translation: 'forall p: Prime, p > 2 -> odd(p)',
          proof: 'By contradiction: assume even prime p > 2, then p = 2k for some k > 1, so p is divisible by 2, contradicting primality.',
          duration_ms: 250,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.verification.verify({
          claim: 'All primes greater than 2 are odd',
        });

        expect(result.status).toBe('valid');
        expect(result.formal_translation).toBeDefined();
        expect(result.proof).toBeDefined();
      });

      it('should return invalid status with counterexample', async () => {
        const mockResponse = {
          status: 'invalid',
          claim: 'All integers are positive',
          formal_translation: 'forall n: Integer, n > 0',
          counterexample: 'n = -1',
          duration_ms: 100,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.verification.verify({
          claim: 'All integers are positive',
        });

        expect(result.status).toBe('invalid');
        expect(result.counterexample).toBe('n = -1');
      });

      it('should verify with specific backend', async () => {
        const mockResponse = {
          status: 'valid',
          claim: 'Test claim',
          duration_ms: 500,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        await client.verification.verify({
          claim: 'Test claim',
          backend: 'lean',
          timeout: 5000,
        });

        const callArgs = mockFetch.mock.calls[0];
        const requestBody = JSON.parse(callArgs[1].body);
        expect(requestBody.backend).toBe('lean');
        expect(requestBody.timeout).toBe(5000);
      });

      it('should handle unknown status', async () => {
        const mockResponse = {
          status: 'unknown',
          claim: 'Complex mathematical conjecture',
          error_message: 'Unable to determine validity within timeout',
          duration_ms: 30000,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.verification.verify({
          claim: 'Complex mathematical conjecture',
        });

        expect(result.status).toBe('unknown');
        expect(result.error_message).toBeDefined();
      });
    });

    describe('verification.status()', () => {
      it('should get verification system status', async () => {
        const mockResponse = {
          available: true,
          backends: [
            { name: 'z3', available: true, version: '4.12.1' },
            { name: 'lean', available: true, version: '4.2.0' },
            { name: 'coq', available: false },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.verification.status();

        expect(result.available).toBe(true);
        expect(result.backends).toHaveLength(3);
        expect(result.backends[0].name).toBe('z3');
        expect(result.backends[0].available).toBe(true);
        expect(result.backends[2].available).toBe(false);
      });

      it('should report unavailable when no backends', async () => {
        const mockResponse = {
          available: false,
          backends: [],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.verification.status();

        expect(result.available).toBe(false);
        expect(result.backends).toHaveLength(0);
      });
    });

    describe('verification.verifyBatch()', () => {
      it('should verify multiple claims in batch', async () => {
        const mockResponse = {
          results: [
            { status: 'valid', claim: 'Claim 1', duration_ms: 100 },
            { status: 'invalid', claim: 'Claim 2', counterexample: 'x = 0', duration_ms: 150 },
            { status: 'unknown', claim: 'Claim 3', duration_ms: 5000 },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.verification.verifyBatch([
          'Claim 1',
          'Claim 2',
          'Claim 3',
        ]);

        expect(result).toHaveLength(3);
        expect(result[0].status).toBe('valid');
        expect(result[1].status).toBe('invalid');
        expect(result[2].status).toBe('unknown');
      });
    });
  });

  // ===========================================================================
  // Error Handling for Advanced Features
  // ===========================================================================

  describe('Error Handling', () => {
    it('should handle batch submission errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid batch request',
          code: 'VALIDATION_ERROR',
        }),
      });

      await expect(
        client.batchDebates.submit({ items: [] })
      ).rejects.toThrow(AragoraError);
    });

    it('should handle graph debate not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'Graph debate not found',
          code: 'NOT_FOUND',
        }),
      });

      await expect(
        client.graphDebates.get('nonexistent')
      ).rejects.toThrow(AragoraError);
    });

    it('should handle evidence search errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({
          error: 'Search service unavailable',
          code: 'SERVICE_UNAVAILABLE',
        }),
      });

      await expect(
        client.evidence.search({ query: 'test' })
      ).rejects.toThrow(AragoraError);
    });

    it('should handle verification timeout', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 408,
        json: () => Promise.resolve({
          error: 'Verification timed out',
          code: 'TIMEOUT',
        }),
      });

      await expect(
        client.verification.verify({ claim: 'Complex claim' })
      ).rejects.toThrow(AragoraError);
    });

    it('should handle memory operation errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 507,
        json: () => Promise.resolve({
          error: 'Insufficient storage',
          code: 'STORAGE_FULL',
        }),
      });

      await expect(
        client.memory.consolidate()
      ).rejects.toThrow(AragoraError);
    });
  });
});
