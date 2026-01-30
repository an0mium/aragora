/**
 * Aragora SDK Comprehensive Test Suite
 *
 * This test file provides comprehensive coverage for the TypeScript SDK including:
 * - Client initialization and configuration
 * - Authentication (API key, Bearer token)
 * - Debates namespace (create, get, list)
 * - Control plane namespace methods
 * - Facts namespace (CRUD operations)
 * - Error handling (rate limits, auth errors, network errors)
 * - Request retries and timeouts
 * - Type safety validation
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { AragoraClient, createClient } from '../client';
import { AragoraError } from '../types';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// =============================================================================
// Test Suite 1: Client Initialization and Configuration
// =============================================================================

describe('Client Initialization and Configuration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should create a client with minimal config', () => {
    const client = createClient({ baseUrl: 'https://api.aragora.ai' });
    expect(client).toBeInstanceOf(AragoraClient);
  });

  it('should create a client with API key', () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-api-key-12345',
    });
    expect(client).toBeInstanceOf(AragoraClient);
  });

  it('should normalize base URL with trailing slash', () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai/',
      apiKey: 'test-key',
    });
    expect(client.getBaseUrl()).toBe('https://api.aragora.ai');
  });

  it('should normalize base URL with multiple trailing slashes', () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai///',
      apiKey: 'test-key',
    });
    expect(client.getBaseUrl()).toBe('https://api.aragora.ai');
  });

  it('should create client with custom headers', async () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      headers: { 'X-Custom-Header': 'custom-value' },
      retryEnabled: false,
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ status: 'healthy' })),
    });

    await client.getHealth();

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-Custom-Header': 'custom-value',
        }),
      })
    );
  });

  it('should create client with custom timeout', () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      timeout: 60000,
    });
    expect(client).toBeInstanceOf(AragoraClient);
  });

  it('should expose all expected namespaces', () => {
    const client = createClient({ baseUrl: 'https://api.aragora.ai' });

    expect(client.debates).toBeDefined();
    expect(client.agents).toBeDefined();
    expect(client.controlPlane).toBeDefined();
    expect(client.facts).toBeDefined();
    expect(client.auth).toBeDefined();
    expect(client.memory).toBeDefined();
    expect(client.knowledge).toBeDefined();
    expect(client.workflows).toBeDefined();
  });
});

// =============================================================================
// Test Suite 2: Authentication
// =============================================================================

describe('Authentication', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should include API key as Bearer token in requests', async () => {
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'my-api-key-12345',
      retryEnabled: false,
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ status: 'healthy' })),
    });

    await client.getHealth();

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer my-api-key-12345',
        }),
      })
    );
  });

  it('should work without API key for public endpoints', async () => {
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      retryEnabled: false,
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ status: 'healthy' })),
    });

    await client.getHealth();

    expect(mockFetch).toHaveBeenCalled();
    const calledHeaders = mockFetch.mock.calls[0][1].headers;
    expect(calledHeaders.Authorization).toBeUndefined();
  });

  it('should handle 401 unauthorized errors', async () => {
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'invalid-key',
      retryEnabled: false,
    });

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: () => Promise.resolve({
        error: 'Invalid API key',
        code: 'INVALID_TOKEN'
      }),
    });

    await expect(client.getHealth()).rejects.toThrow('Invalid API key');
  });

  it('should handle 403 forbidden errors', async () => {
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'valid-key-no-permission',
      retryEnabled: false,
    });

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 403,
      json: () => Promise.resolve({
        error: 'Access denied',
        code: 'FORBIDDEN'
      }),
    });

    await expect(client.getHealth()).rejects.toThrow('Access denied');
  });

  it('should include content-type and accept headers', async () => {
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      retryEnabled: false,
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ debates: [] })),
    });

    await client.debates.list();

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        }),
      })
    );
  });
});

// =============================================================================
// Test Suite 3: Debates Namespace
// =============================================================================

describe('Debates Namespace', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });
  });

  it('should create a debate', async () => {
    const mockResponse = {
      debate_id: 'debate-123',
      status: 'pending',
      task: 'Should we use microservices?',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockResponse)),
    });

    const result = await client.debates.create({
      task: 'Should we use microservices?',
      agents: ['claude', 'gpt-4'],
      rounds: 3,
    });

    expect(result.debate_id).toBe('debate-123');
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/debate'),
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('should get a debate by ID', async () => {
    const mockDebate = {
      id: 'debate-456',
      task: 'Test debate',
      status: 'completed',
      consensus: { reached: true, confidence: 0.95 },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockDebate)),
    });

    const result = await client.debates.get('debate-456');

    expect(result.id).toBe('debate-456');
    expect(result.status).toBe('completed');
  });

  it('should list debates with pagination', async () => {
    const mockResponse = {
      debates: [
        { id: 'd1', task: 'Debate 1', status: 'completed' },
        { id: 'd2', task: 'Debate 2', status: 'pending' },
      ],
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockResponse)),
    });

    const result = await client.debates.list({ limit: 10, offset: 0 });

    expect(result.debates).toHaveLength(2);
    expect(result.debates[0].id).toBe('d1');
  });

  it('should list debates with status filter', async () => {
    const mockResponse = {
      debates: [{ id: 'd1', task: 'Debate 1', status: 'completed' }],
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockResponse)),
    });

    const result = await client.debates.list({ status: 'completed' });

    expect(result.debates).toHaveLength(1);
    expect(result.debates[0].status).toBe('completed');
  });

  it('should get debate messages', async () => {
    const mockMessages = {
      messages: [
        { id: 'm1', agent: 'claude', content: 'First message', round: 1 },
        { id: 'm2', agent: 'gpt-4', content: 'Second message', round: 1 },
      ],
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockMessages)),
    });

    const result = await client.debates.getMessages('debate-123');

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0].agent).toBe('claude');
  });

  it('should export debate as markdown', async () => {
    const mockExport = {
      format: 'markdown',
      content: '# Debate Export\n\n...',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockExport)),
    });

    const result = await client.debates.export('debate-123', 'markdown');

    expect(result.format).toBe('markdown');
  });

  it('should fork a debate', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ debate_id: 'forked-debate-789' })),
    });

    const result = await client.debates.fork('debate-123', { branch_point: 2 });

    expect(result.debate_id).toBe('forked-debate-789');
  });

  it('should get debate convergence', async () => {
    const mockConvergence = {
      convergence_score: 0.85,
      areas_of_agreement: ['point1', 'point2'],
      areas_of_disagreement: ['point3'],
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockConvergence)),
    });

    const result = await client.debates.getConvergence('debate-123');

    expect(result.convergence_score).toBe(0.85);
  });
});

// =============================================================================
// Test Suite 4: Control Plane Namespace
// =============================================================================

describe('Control Plane Namespace', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });
  });

  it('should expose control plane sub-namespaces', () => {
    expect(client.controlPlane.agents).toBeDefined();
    expect(client.controlPlane.tasks).toBeDefined();
    expect(client.controlPlane.policies).toBeDefined();
    expect(client.controlPlane.schedules).toBeDefined();
    expect(client.controlPlane.deliberations).toBeDefined();
    expect(client.controlPlane.auditLogs).toBeDefined();
    expect(client.controlPlane.violations).toBeDefined();
    expect(client.controlPlane.metrics).toBeDefined();
  });

  it('should get control plane health', async () => {
    const mockHealth = {
      status: 'healthy',
      agents_total: 10,
      agents_active: 8,
      tasks_pending: 5,
      tasks_running: 3,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockHealth)),
    });

    const result = await client.controlPlane.getHealth();

    expect(result.status).toBe('healthy');
    expect(result.agents_total).toBe(10);
  });

  it('should register an agent', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ registered: true, agent_id: 'worker-1' })),
    });

    const result = await client.controlPlane.agents.register({
      agent_id: 'worker-1',
      name: 'Analysis Worker',
      capabilities: ['code-review', 'security-scan'],
    });

    expect(result.registered).toBe(true);
    expect(result.agent_id).toBe('worker-1');
  });

  it('should list registered agents', async () => {
    const mockAgents = {
      agents: [
        { agent_id: 'a1', name: 'Worker 1', status: 'idle', capabilities: ['review'] },
        { agent_id: 'a2', name: 'Worker 2', status: 'busy', capabilities: ['scan'] },
      ],
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockAgents)),
    });

    const result = await client.controlPlane.agents.list();

    expect(result.agents).toHaveLength(2);
  });

  it('should submit a task', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ task_id: 'task-123', status: 'pending' })),
    });

    const result = await client.controlPlane.tasks.submit({
      task_type: 'code-review',
      payload: { repo: 'my-repo', pr: 123 },
      priority: 'high',
    });

    expect(result.task_id).toBe('task-123');
  });

  it('should create a policy', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ policy_id: 'policy-123', created: true })),
    });

    const result = await client.controlPlane.policies.create({
      name: 'Rate Limit Policy',
      rules: [{ condition: 'requests_per_minute > 100', action: 'deny' }],
      enabled: true,
    });

    expect(result.policy_id).toBe('policy-123');
    expect(result.created).toBe(true);
  });

  it('should schedule a task', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({
        schedule_id: 'sched-123',
        next_run_at: '2024-01-15T10:00:00Z'
      })),
    });

    const result = await client.controlPlane.schedules.create({
      task_type: 'cleanup',
      payload: { target: 'logs' },
      cron: '0 0 * * *',
    });

    expect(result.schedule_id).toBe('sched-123');
  });
});

// =============================================================================
// Test Suite 5: Facts Namespace
// =============================================================================

describe('Facts Namespace', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });
  });

  it('should create a fact', async () => {
    const mockFact = {
      id: 'fact-123',
      content: 'TypeScript is statically typed',
      source: 'ts-docs',
      confidence: 0.95,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockFact)),
    });

    const result = await client.facts.createFact({
      content: 'TypeScript is statically typed',
      source: 'ts-docs',
      confidence: 0.95,
    });

    expect(result.id).toBe('fact-123');
    expect(result.confidence).toBe(0.95);
  });

  it('should get a fact by ID', async () => {
    const mockFact = {
      id: 'fact-456',
      content: 'Test fact',
      source: 'test',
      confidence: 0.9,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockFact)),
    });

    const result = await client.facts.getFact('fact-456');

    expect(result.id).toBe('fact-456');
  });

  it('should update a fact', async () => {
    const mockUpdatedFact = {
      id: 'fact-123',
      content: 'Updated content',
      source: 'updated-source',
      confidence: 0.8,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-02T00:00:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockUpdatedFact)),
    });

    const result = await client.facts.updateFact('fact-123', {
      content: 'Updated content',
      confidence: 0.8,
    });

    expect(result.content).toBe('Updated content');
    expect(result.confidence).toBe(0.8);
  });

  it('should delete a fact', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(''),
    });

    await expect(client.facts.deleteFact('fact-123')).resolves.toBeUndefined();
  });

  it('should list facts with pagination', async () => {
    const mockResponse = {
      facts: [
        { id: 'f1', content: 'Fact 1', source: 's1', confidence: 0.9, created_at: '2024-01-01T00:00:00Z', updated_at: '2024-01-01T00:00:00Z' },
      ],
      total: 100,
      offset: 0,
      limit: 10,
      has_more: true,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockResponse)),
    });

    const result = await client.facts.listFacts({ limit: 10, offset: 0 });

    expect(result.facts).toHaveLength(1);
    expect(result.total).toBe(100);
    expect(result.has_more).toBe(true);
  });

  it('should search facts', async () => {
    const mockResults = [
      {
        id: 'f1',
        content: 'Machine learning fact',
        source: 'ml-docs',
        confidence: 0.9,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        relevance: 0.95,
      },
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ results: mockResults })),
    });

    const result = await client.facts.searchFacts('machine learning', { semantic: true });

    expect(result).toHaveLength(1);
    expect(result[0].relevance).toBe(0.95);
  });

  it('should create a relationship', async () => {
    const mockRel = {
      id: 'rel-123',
      source_fact_id: 'f1',
      target_fact_id: 'f2',
      relationship_type: 'supports',
      weight: 0.9,
      created_at: '2024-01-01T00:00:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockRel)),
    });

    const result = await client.facts.createRelationship({
      source_fact_id: 'f1',
      target_fact_id: 'f2',
      relationship_type: 'supports',
      weight: 0.9,
    });

    expect(result.id).toBe('rel-123');
    expect(result.relationship_type).toBe('supports');
  });

  it('should batch create facts', async () => {
    const mockResponse = {
      created: [
        { id: 'f1', content: 'Fact 1', source: 's1', confidence: 0.9, created_at: '2024-01-01T00:00:00Z', updated_at: '2024-01-01T00:00:00Z' },
        { id: 'f2', content: 'Fact 2', source: 's2', confidence: 0.8, created_at: '2024-01-01T00:00:00Z', updated_at: '2024-01-01T00:00:00Z' },
      ],
      failed: [],
      total_created: 2,
      total_failed: 0,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockResponse)),
    });

    const result = await client.facts.batchCreateFacts([
      { content: 'Fact 1', source: 's1' },
      { content: 'Fact 2', source: 's2' },
    ]);

    expect(result.total_created).toBe(2);
    expect(result.failed).toHaveLength(0);
  });

  it('should get fact statistics', async () => {
    const mockStats = {
      total_facts: 1000,
      total_relationships: 500,
      by_source: { docs: 400, manual: 600 },
      average_confidence: 0.85,
      created_last_24h: 50,
      updated_last_24h: 100,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockStats)),
    });

    const result = await client.facts.getStats();

    expect(result.total_facts).toBe(1000);
    expect(result.average_confidence).toBe(0.85);
  });
});

// =============================================================================
// Test Suite 6: Error Handling
// =============================================================================

describe('Error Handling', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });
  });

  it('should handle 400 bad request errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve({
        error: 'Invalid request parameters',
        code: 'BAD_REQUEST'
      }),
    });

    await expect(client.debates.create({ task: '' } as any))
      .rejects.toThrow('Invalid request parameters');
  });

  it('should handle 404 not found errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: () => Promise.resolve({
        error: 'Debate not found',
        code: 'NOT_FOUND'
      }),
    });

    await expect(client.debates.get('nonexistent'))
      .rejects.toThrow('Debate not found');
  });

  it('should handle 429 rate limit errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 429,
      json: () => Promise.resolve({
        error: 'Rate limit exceeded',
        code: 'RATE_LIMITED',
        retry_after: 60
      }),
    });

    try {
      await client.getHealth();
      expect.fail('Should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(AragoraError);
      expect((error as AragoraError).message).toBe('Rate limit exceeded');
      expect((error as AragoraError).status).toBe(429);
    }
  });

  it('should handle 500 internal server errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.resolve({
        error: 'Internal server error',
        code: 'INTERNAL_ERROR'
      }),
    });

    await expect(client.getHealth())
      .rejects.toThrow('Internal server error');
  });

  it('should handle 503 service unavailable errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: () => Promise.resolve({
        error: 'Service unavailable',
        code: 'SERVICE_UNAVAILABLE'
      }),
    });

    await expect(client.getHealth())
      .rejects.toThrow('Service unavailable');
  });

  it('should handle network errors', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    await expect(client.getHealth())
      .rejects.toThrow('Network error');
  });

  it('should handle malformed JSON responses', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error('Invalid JSON')),
      statusText: 'Internal Server Error',
    });

    await expect(client.getHealth())
      .rejects.toThrow();
  });

  it('should preserve error code in AragoraError', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: () => Promise.resolve({
        error: 'Invalid token',
        code: 'INVALID_TOKEN',
        trace_id: 'trace-123'
      }),
    });

    try {
      await client.getHealth();
      expect.fail('Should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(AragoraError);
      expect((error as AragoraError).code).toBe('INVALID_TOKEN');
      expect((error as AragoraError).traceId).toBe('trace-123');
    }
  });
});

// =============================================================================
// Test Suite 7: Request Retries and Timeouts
// =============================================================================

describe('Request Retries and Timeouts', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should retry on 500 errors when retries enabled', async () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: true,
      maxRetries: 3,
    });

    // First two calls fail, third succeeds
    mockFetch
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Server error' }),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Server error' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ status: 'healthy' })),
      });

    const result = await client.getHealth();

    expect(result.status).toBe('healthy');
    expect(mockFetch).toHaveBeenCalledTimes(3);
  });

  it('should not retry on 4xx client errors', async () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: true,
      maxRetries: 3,
    });

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ error: 'Bad request', code: 'BAD_REQUEST' }),
    });

    await expect(client.getHealth()).rejects.toThrow('Bad request');
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('should fail after max retries exhausted', async () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: true,
      maxRetries: 2,
    });

    mockFetch
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Server error' }),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Server error' }),
      });

    await expect(client.getHealth()).rejects.toThrow('Server error');
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('should not retry when retries disabled', async () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ error: 'Server error' }),
    });

    await expect(client.getHealth()).rejects.toThrow('Server error');
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('should handle timeout errors', async () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      timeout: 100,
      retryEnabled: false,
    });

    const abortError = new Error('The operation was aborted');
    abortError.name = 'AbortError';
    mockFetch.mockRejectedValueOnce(abortError);

    await expect(client.getHealth()).rejects.toThrow('Request timeout');
  });

  it('should use exponential backoff on retries', async () => {
    const client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: true,
      maxRetries: 2,
    });

    const startTime = Date.now();

    mockFetch
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'Server error' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ status: 'healthy' })),
      });

    await client.getHealth();

    const elapsed = Date.now() - startTime;
    // First retry should wait at least 1 second (2^0 * 1000)
    expect(elapsed).toBeGreaterThanOrEqual(900); // Allow some timing variance
  });
});

// =============================================================================
// Test Suite 8: Type Safety Validation
// =============================================================================

describe('Type Safety Validation', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });
  });

  it('should return properly typed debate response', async () => {
    const mockDebate = {
      id: 'debate-123',
      task: 'Test task',
      status: 'completed',
      agents: ['claude', 'gpt-4'],
      rounds: 3,
      created_at: '2024-01-01T00:00:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockDebate)),
    });

    const result = await client.debates.get('debate-123');

    // Type assertions to verify the shape
    expect(typeof result.id).toBe('string');
    expect(typeof result.task).toBe('string');
    expect(typeof result.status).toBe('string');
  });

  it('should return properly typed fact response', async () => {
    const mockFact = {
      id: 'fact-123',
      content: 'Test content',
      source: 'test-source',
      confidence: 0.95,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockFact)),
    });

    const result = await client.facts.getFact('fact-123');

    expect(typeof result.id).toBe('string');
    expect(typeof result.content).toBe('string');
    expect(typeof result.confidence).toBe('number');
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it('should return properly typed paginated response', async () => {
    const mockResponse = {
      facts: [
        { id: 'f1', content: 'Fact 1', source: 's1', confidence: 0.9, created_at: '2024-01-01T00:00:00Z', updated_at: '2024-01-01T00:00:00Z' },
      ],
      total: 100,
      offset: 0,
      limit: 10,
      has_more: true,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockResponse)),
    });

    const result = await client.facts.listFacts();

    expect(Array.isArray(result.facts)).toBe(true);
    expect(typeof result.total).toBe('number');
    expect(typeof result.has_more).toBe('boolean');
  });

  it('should return properly typed control plane health', async () => {
    const mockHealth = {
      status: 'healthy',
      agents_total: 10,
      agents_active: 8,
      tasks_pending: 5,
      tasks_running: 3,
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockHealth)),
    });

    const result = await client.controlPlane.getHealth();

    expect(['healthy', 'degraded', 'unhealthy']).toContain(result.status);
    expect(typeof result.agents_total).toBe('number');
    expect(typeof result.agents_active).toBe('number');
  });

  it('should handle empty response bodies', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(''),
    });

    // DELETE operations typically return empty bodies
    const result = await client.facts.deleteFact('fact-123');
    expect(result).toBeUndefined();
  });

  it('should handle relationship type enum values', async () => {
    const mockRel = {
      id: 'rel-123',
      source_fact_id: 'f1',
      target_fact_id: 'f2',
      relationship_type: 'supports',
      weight: 0.9,
      created_at: '2024-01-01T00:00:00Z',
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify(mockRel)),
    });

    const result = await client.facts.createRelationship({
      source_fact_id: 'f1',
      target_fact_id: 'f2',
      relationship_type: 'supports',
    });

    const validTypes = [
      'supports', 'contradicts', 'elaborates', 'derived_from',
      'related_to', 'precedes', 'follows', 'causes', 'caused_by'
    ];
    expect(validTypes).toContain(result.relationship_type);
  });
});

// =============================================================================
// Test Suite 9: Additional Edge Cases
// =============================================================================

describe('Additional Edge Cases', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });
  });

  it('should handle query parameters correctly', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ debates: [] })),
    });

    await client.debates.list({ limit: 20, offset: 10, status: 'completed' });

    const calledUrl = mockFetch.mock.calls[0][0];
    expect(calledUrl).toContain('limit=20');
    expect(calledUrl).toContain('offset=10');
    expect(calledUrl).toContain('status=completed');
  });

  it('should handle undefined query parameters', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ debates: [] })),
    });

    await client.debates.list({ limit: 10, offset: undefined });

    const calledUrl = mockFetch.mock.calls[0][0];
    expect(calledUrl).toContain('limit=10');
    expect(calledUrl).not.toContain('offset');
  });

  it('should send correct HTTP methods', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ id: 'fact-123' })),
    });

    await client.facts.createFact({ content: 'Test', source: 'test' });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ method: 'POST' })
    );
  });

  it('should serialize request body as JSON for debates', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ debate_id: 'debate-123', status: 'pending' })),
    });

    const debateRequest = { task: 'Test task', agents: ['claude', 'gpt-4'], rounds: 3 };
    await client.debates.create(debateRequest);

    const requestBody = mockFetch.mock.calls[0][1].body;
    expect(JSON.parse(requestBody)).toEqual(debateRequest);
  });

  it('should handle special characters in IDs', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      text: () => Promise.resolve(JSON.stringify({ id: 'fact-with-special/chars' })),
    });

    await client.facts.getFact('fact-with-special/chars');

    expect(mockFetch).toHaveBeenCalled();
  });
});
