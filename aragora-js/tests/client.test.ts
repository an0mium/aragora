/**
 * Aragora SDK Client Tests
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { AragoraClient, AragoraError } from '../src/client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('AragoraClient', () => {
  let client: AragoraClient;

  beforeEach(() => {
    client = new AragoraClient({
      baseUrl: 'http://localhost:8080',
      apiKey: 'test-api-key',
    });
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create client with base URL', () => {
      const c = new AragoraClient({ baseUrl: 'http://localhost:8080' });
      expect(c).toBeInstanceOf(AragoraClient);
    });

    it('should strip trailing slash from base URL', () => {
      const c = new AragoraClient({ baseUrl: 'http://localhost:8080/' });
      expect(c).toBeInstanceOf(AragoraClient);
    });
  });

  describe('health()', () => {
    it('should return health check response', async () => {
      const mockResponse = {
        status: 'healthy',
        version: '1.0.0',
        uptime_seconds: 3600,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.health();
      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/health',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            Authorization: 'Bearer test-api-key',
          }),
        })
      );
    });
  });

  describe('healthDeep()', () => {
    it('should return deep health check response', async () => {
      const mockResponse = {
        status: 'healthy',
        healthy: true,
        checks: {
          storage: { healthy: true, status: 'connected' },
          elo_system: { healthy: true, status: 'connected' },
        },
        warnings: [],
        response_time_ms: 42.5,
        timestamp: '2024-01-01T00:00:00Z',
        version: '1.0.0',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.healthDeep();
      expect(result.healthy).toBe(true);
      expect(result.checks.storage.healthy).toBe(true);
    });
  });

  describe('debates', () => {
    describe('create()', () => {
      it('should create a debate', async () => {
        const mockResponse = {
          debate_id: 'debate-123',
          status: 'created',
          task: 'Test debate',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.debates.create({
          task: 'Test debate',
          agents: ['claude', 'gpt4'],
        });

        expect(result.debate_id).toBe('debate-123');
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/debates',
          expect.objectContaining({
            method: 'POST',
            body: JSON.stringify({
              task: 'Test debate',
              agents: ['claude', 'gpt4'],
            }),
          })
        );
      });
    });

    describe('get()', () => {
      it('should get a debate by ID', async () => {
        const mockResponse = {
          debate_id: 'debate-123',
          task: 'Test debate',
          status: 'completed',
          agents: ['claude', 'gpt4'],
          rounds: [],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.debates.get('debate-123');
        expect(result.debate_id).toBe('debate-123');
      });
    });

    describe('list()', () => {
      it('should list debates with options', async () => {
        const mockResponse = {
          debates: [
            { debate_id: 'debate-1', task: 'Task 1', status: 'completed', agents: [], rounds: [] },
            { debate_id: 'debate-2', task: 'Task 2', status: 'in_progress', agents: [], rounds: [] },
          ],
          total: 2,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.debates.list({ limit: 10, offset: 0 });
        expect(result).toHaveLength(2);
      });
    });
  });

  describe('pulse', () => {
    describe('trending()', () => {
      it('should get trending topics', async () => {
        const mockResponse = {
          topics: [
            { topic: 'AI Safety', source: 'hackernews', score: 100 },
            { topic: 'Climate Tech', source: 'reddit', score: 85 },
          ],
          sources: ['hackernews', 'reddit'],
          updated_at: '2024-01-01T00:00:00Z',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.pulse.trending();
        expect(result.topics).toHaveLength(2);
        expect(result.topics[0].topic).toBe('AI Safety');
      });
    });

    describe('suggest()', () => {
      it('should suggest a trending topic', async () => {
        const mockResponse = {
          suggestion: { topic: 'AI Regulation', source: 'hackernews', score: 95 },
          reason: 'High engagement and debate potential',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.pulse.suggest();
        expect(result.suggestion.topic).toBe('AI Regulation');
      });
    });

    describe('schedulerStatus()', () => {
      it('should get scheduler status', async () => {
        const mockResponse = {
          running: true,
          paused: false,
          interval_minutes: 60,
          debates_today: 5,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.pulse.schedulerStatus();
        expect(result.running).toBe(true);
        expect(result.debates_today).toBe(5);
      });
    });
  });

  describe('documents', () => {
    describe('list()', () => {
      it('should list documents', async () => {
        const mockResponse = {
          documents: [
            {
              doc_id: 'doc-1',
              filename: 'test.pdf',
              content_type: 'application/pdf',
              size_bytes: 1024,
              uploaded_at: '2024-01-01T00:00:00Z',
            },
          ],
          total: 1,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.documents.list();
        expect(result).toHaveLength(1);
        expect(result[0].filename).toBe('test.pdf');
      });
    });

    describe('formats()', () => {
      it('should get supported formats', async () => {
        const mockResponse = {
          formats: [
            { extension: '.pdf', content_type: 'application/pdf', max_size_mb: 10 },
            { extension: '.txt', content_type: 'text/plain', max_size_mb: 5 },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.documents.formats();
        expect(result.formats).toHaveLength(2);
      });
    });
  });

  describe('breakpoints', () => {
    describe('pending()', () => {
      it('should list pending breakpoints', async () => {
        const mockResponse = {
          breakpoints: [
            {
              breakpoint_id: 'bp-1',
              debate_id: 'debate-123',
              round: 2,
              reason: 'High uncertainty',
              status: 'pending',
              created_at: '2024-01-01T00:00:00Z',
            },
          ],
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.breakpoints.pending();
        expect(result).toHaveLength(1);
        expect(result[0].reason).toBe('High uncertainty');
      });
    });

    describe('resolve()', () => {
      it('should resolve a breakpoint', async () => {
        const mockResponse = {
          breakpoint_id: 'bp-1',
          status: 'resolved',
          action_taken: 'continue',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.breakpoints.resolve('bp-1', {
          action: 'continue',
          feedback: 'Proceed with the debate',
        });

        expect(result.status).toBe('resolved');
        expect(result.action_taken).toBe('continue');
      });
    });
  });

  describe('error handling', () => {
    it('should throw AragoraError on HTTP error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: 'Not found', code: 'NOT_FOUND' }),
      });

      await expect(client.debates.get('nonexistent')).rejects.toThrow(AragoraError);
    });

    it('should throw AragoraError with correct properties', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () =>
          Promise.resolve({
            error: 'Rate limit exceeded',
            code: 'RATE_LIMIT',
          }),
      });

      try {
        await client.debates.get('test');
      } catch (error) {
        expect(error).toBeInstanceOf(AragoraError);
        if (error instanceof AragoraError) {
          expect(error.status).toBe(429);
          expect(error.code).toBe('RATE_LIMIT');
          expect(error.message).toBe('Rate limit exceeded');
        }
      }
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(client.health()).rejects.toThrow(AragoraError);
    });

    it('should handle timeout', async () => {
      const client = new AragoraClient({
        baseUrl: 'http://localhost:8080',
        timeout: 50, // Very short timeout
      });

      // Mock fetch that respects the AbortSignal
      mockFetch.mockImplementationOnce((_url: string, options: RequestInit) => {
        return new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            resolve({
              ok: true,
              json: () => Promise.resolve({}),
            });
          }, 200);

          // Listen for abort signal
          if (options?.signal) {
            options.signal.addEventListener('abort', () => {
              clearTimeout(timeout);
              const error = new Error('The operation was aborted');
              error.name = 'AbortError';
              reject(error);
            });
          }
        });
      });

      await expect(client.health()).rejects.toThrow('timed out');
    });
  });

  describe('gauntlet', () => {
    describe('run()', () => {
      it('should start a gauntlet run', async () => {
        const mockResponse = {
          gauntlet_id: 'gauntlet-123',
          status: 'running',
          persona: 'security',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.gauntlet.run({
          input_content: 'function test() { return eval(userInput); }',
          input_type: 'code',
          persona: 'security' as const,
        });

        expect(result.gauntlet_id).toBe('gauntlet-123');
        expect(result.persona).toBe('security');
      });
    });

    describe('getReceipt()', () => {
      it('should get gauntlet receipt', async () => {
        const mockResponse = {
          gauntlet_id: 'gauntlet-123',
          status: 'completed',
          score: 75,
          findings: [
            {
              severity: 'high',
              category: 'security',
              description: 'Potential code injection via eval()',
            },
          ],
          duration_seconds: 45,
          persona: 'security',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.gauntlet.getReceipt('gauntlet-123');
        expect(result.score).toBe(75);
        expect(result.findings).toHaveLength(1);
      });
    });
  });

  describe('verification', () => {
    describe('verify()', () => {
      it('should verify a claim', async () => {
        const mockResponse = {
          status: 'valid',
          claim: 'All primes > 2 are odd',
          formal_translation: 'forall p: Prime, p > 2 -> odd(p)',
          proof: 'By contradiction...',
          duration_ms: 150,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.verification.verify({
          claim: 'All primes > 2 are odd',
        });

        expect(result.status).toBe('valid');
        expect(result.formal_translation).toBeDefined();
      });
    });
  });

  describe('memory', () => {
    describe('analytics()', () => {
      it('should get memory analytics', async () => {
        const mockResponse = {
          tiers: [
            { tier_name: 'fast', entry_count: 100, avg_access_frequency: 10, promotion_rate: 0.3, demotion_rate: 0.1, hit_rate: 0.85 },
            { tier_name: 'slow', entry_count: 500, avg_access_frequency: 2, promotion_rate: 0.1, demotion_rate: 0.2, hit_rate: 0.7 },
          ],
          total_entries: 600,
          learning_velocity: 0.8,
          promotion_effectiveness: 0.75,
          recommendations: [],
          period_days: 30,
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        });

        const result = await client.memory.analytics(30);
        expect(result.tiers).toHaveLength(2);
        expect(result.total_entries).toBe(600);
      });
    });
  });
});
