/**
 * Aragora SDK Debates API Tests
 *
 * Comprehensive tests for the debates API including:
 * - CRUD operations
 * - Polling and waiting
 * - Status normalization
 * - Error handling
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { AragoraClient, AragoraError } from '../src/client';
import { DebateStatus, normalizeStatus, LEGACY_STATUS_MAP } from '../src/types';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Debates API', () => {
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

  describe('debates.create()', () => {
    it('should create a debate with required fields', async () => {
      const mockResponse = {
        debate_id: 'debate-123',
        status: 'running',
        task: 'Design a caching system',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.debates.create({
        task: 'Design a caching system',
      });

      expect(result.debate_id).toBe('debate-123');
      expect(result.status).toBe('running');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/debates',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('caching system'),
        })
      );
    });

    it('should create a debate with optional fields', async () => {
      const mockResponse = {
        debate_id: 'debate-456',
        status: 'running',
        task: 'Complex question',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.debates.create({
        task: 'Complex question',
        agents: ['anthropic-api', 'openai-api'],
        max_rounds: 5,
      });

      expect(result.debate_id).toBe('debate-456');

      // Verify request body contains all fields
      const callArgs = mockFetch.mock.calls[0];
      const requestBody = JSON.parse(callArgs[1].body);
      expect(requestBody.agents).toEqual(['anthropic-api', 'openai-api']);
      expect(requestBody.max_rounds).toBe(5);
    });

    it('should handle creation errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ error: 'Invalid question' }),
      });

      await expect(client.debates.create({ task: '' }))
        .rejects.toThrow(AragoraError);
    });

    it('should handle rate limiting', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({ error: 'Rate limit exceeded' }),
      });

      await expect(client.debates.create({ task: 'Test' }))
        .rejects.toThrow();
    });
  });

  describe('debates.get()', () => {
    it('should get a debate by ID', async () => {
      const mockResponse = {
        id: 'debate-123',
        task: 'Test debate',
        status: 'running',
        rounds_completed: 2,
        consensus: {
          reached: false,
          confidence: 0.6,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.debates.get('debate-123');

      expect(result.id).toBe('debate-123');
      expect(result.status).toBe(DebateStatus.RUNNING);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/debates/debate-123',
        expect.objectContaining({ method: 'GET' })
      );
    });

    it('should normalize legacy status values', async () => {
      const mockResponse = {
        id: 'debate-123',
        status: 'active', // Legacy status
        task: 'Test debate',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.debates.get('debate-123');

      // Status should be normalized to 'running'
      expect(result.status).toBe(DebateStatus.RUNNING);
    });

    it('should handle not found errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: 'Debate not found' }),
      });

      await expect(client.debates.get('nonexistent'))
        .rejects.toThrow(AragoraError);
    });
  });

  describe('debates.list()', () => {
    it('should list debates with default pagination', async () => {
      const mockResponse = {
        debates: [
          { id: 'debate-1', task: 'Task 1', status: 'running' },
          { id: 'debate-2', task: 'Task 2', status: 'completed' },
        ],
        count: 2,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.debates.list();

      expect(result).toHaveLength(2);
      expect(result[0].id).toBe('debate-1');
    });

    it('should list debates with custom pagination', async () => {
      const mockResponse = {
        debates: [{ id: 'debate-3', task: 'Task 3', status: 'completed' }],
        count: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.debates.list({ limit: 10, offset: 20 });

      expect(result).toHaveLength(1);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('limit=10'),
        expect.anything()
      );
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('offset=20'),
        expect.anything()
      );
    });

    it('should normalize status in list results', async () => {
      const mockResponse = {
        debates: [
          { id: 'debate-1', task: 'Task 1', status: 'active' }, // Legacy
          { id: 'debate-2', task: 'Task 2', status: 'concluded' }, // Legacy
        ],
        count: 2,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.debates.list();

      expect(result[0].status).toBe(DebateStatus.RUNNING);
      expect(result[1].status).toBe(DebateStatus.COMPLETED);
    });
  });

  describe('debates.waitForCompletion()', () => {
    it('should poll until debate completes', async () => {
      // First poll: running
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'debate-123',
          status: 'running',
          task: 'Test',
        }),
      });

      // Second poll: running
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'debate-123',
          status: 'running',
          task: 'Test',
        }),
      });

      // Third poll: completed
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'debate-123',
          status: 'completed',
          task: 'Test',
          consensus: { reached: true, conclusion: 'Final answer' },
        }),
      });

      const result = await client.debates.waitForCompletion('debate-123', {
        pollInterval: 10, // Fast polling for test
        timeout: 1000,
      });

      expect(result.status).toBe(DebateStatus.COMPLETED);
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });

    it('should timeout if debate takes too long', async () => {
      // Always return running
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          id: 'debate-123',
          status: 'running',
          task: 'Test',
        }),
      });

      await expect(client.debates.waitForCompletion('debate-123', {
        pollInterval: 10,
        timeout: 50,
      })).rejects.toThrow();
    });
  });

  describe('debates.run()', () => {
    it('should create and wait for debate completion', async () => {
      // Create response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          debate_id: 'debate-new',
          status: 'running',
          task: 'New debate',
        }),
      });

      // First poll
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'debate-new',
          status: 'running',
          task: 'New debate',
        }),
      });

      // Second poll: completed
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'debate-new',
          status: 'completed',
          task: 'New debate',
          consensus: { reached: true },
        }),
      });

      const result = await client.debates.run({ task: 'New debate' });

      expect(result.status).toBe(DebateStatus.COMPLETED);
    });
  });
});

describe('Status Normalization', () => {
  describe('normalizeStatus()', () => {
    it('should normalize legacy active to running', () => {
      expect(normalizeStatus('active')).toBe(DebateStatus.RUNNING);
    });

    it('should normalize legacy concluded to completed', () => {
      expect(normalizeStatus('concluded')).toBe(DebateStatus.COMPLETED);
    });

    it('should normalize legacy archived to completed', () => {
      expect(normalizeStatus('archived')).toBe(DebateStatus.COMPLETED);
    });

    it('should normalize legacy created to pending', () => {
      expect(normalizeStatus('created')).toBe(DebateStatus.PENDING);
    });

    it('should normalize legacy in_progress to running', () => {
      expect(normalizeStatus('in_progress')).toBe(DebateStatus.RUNNING);
    });

    it('should pass through canonical values unchanged', () => {
      expect(normalizeStatus('running')).toBe(DebateStatus.RUNNING);
      expect(normalizeStatus('completed')).toBe(DebateStatus.COMPLETED);
      expect(normalizeStatus('pending')).toBe(DebateStatus.PENDING);
      expect(normalizeStatus('failed')).toBe(DebateStatus.FAILED);
      expect(normalizeStatus('cancelled')).toBe(DebateStatus.CANCELLED);
      expect(normalizeStatus('paused')).toBe(DebateStatus.PAUSED);
    });

    it('should handle enum values', () => {
      expect(normalizeStatus(DebateStatus.RUNNING)).toBe(DebateStatus.RUNNING);
      expect(normalizeStatus(DebateStatus.COMPLETED)).toBe(DebateStatus.COMPLETED);
    });
  });

  describe('LEGACY_STATUS_MAP', () => {
    it('should contain all expected mappings', () => {
      expect(LEGACY_STATUS_MAP['active']).toBe(DebateStatus.RUNNING);
      expect(LEGACY_STATUS_MAP['concluded']).toBe(DebateStatus.COMPLETED);
      expect(LEGACY_STATUS_MAP['archived']).toBe(DebateStatus.COMPLETED);
      expect(LEGACY_STATUS_MAP['created']).toBe(DebateStatus.PENDING);
      expect(LEGACY_STATUS_MAP['in_progress']).toBe(DebateStatus.RUNNING);
    });
  });
});

describe('Error Handling', () => {
  let client: AragoraClient;

  beforeEach(() => {
    client = new AragoraClient({
      baseUrl: 'http://localhost:8080',
      apiKey: 'test-api-key',
      retry: { maxRetries: 0 }, // Disable retries for error tests
    });
    mockFetch.mockReset();
  });

  it('should throw AragoraError for API errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ error: 'Internal server error' }),
    });

    try {
      await client.debates.get('test');
      expect.fail('Should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(AragoraError);
      expect((e as AragoraError).status).toBe(500);
    }
  });

  it('should handle network errors', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    await expect(client.debates.get('test')).rejects.toThrow('Network error');
  });

  it('should handle malformed JSON responses', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.reject(new Error('Invalid JSON')),
    });

    await expect(client.debates.get('test')).rejects.toThrow();
  });
});
