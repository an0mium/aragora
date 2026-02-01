/**
 * Gauntlet Namespace Test Suite
 *
 * Tests for the gauntlet namespace including:
 * - Running gauntlets (async and wait modes)
 * - Listing and retrieving receipts
 * - Verifying receipt integrity
 * - Exporting receipts in various formats
 * - Managing personas
 * - Viewing results and heatmaps
 * - Comparing gauntlet runs
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AragoraClient, createClient } from '../client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Gauntlet Namespace', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });
  });

  // ===========================================================================
  // Run Gauntlet
  // ===========================================================================

  describe('Run Gauntlet', () => {
    it('should run a gauntlet (non-blocking)', async () => {
      const mockResponse = {
        gauntlet_id: 'gauntlet-001',
        status: 'pending',
        status_url: '/api/gauntlet/gauntlet-001/status',
        estimated_duration_seconds: 120,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.run({
        input: 'Our API is secure against SQL injection',
        personas: ['adversarial', 'compliance'],
        profile: 'security',
      });

      expect(result.gauntlet_id).toBe('gauntlet-001');
      expect(result.status).toBe('pending');
      expect(result.status_url).toBeDefined();
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/gauntlet'),
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('should run a gauntlet with minimal options', async () => {
      const mockResponse = {
        gauntlet_id: 'gauntlet-002',
        status: 'running',
        status_url: '/api/gauntlet/gauntlet-002/status',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.run({
        input: 'Simple claim to test',
      });

      expect(result.gauntlet_id).toBe('gauntlet-002');
    });

    it('should run a gauntlet with parallel execution', async () => {
      const mockResponse = {
        gauntlet_id: 'gauntlet-003',
        status: 'running',
        status_url: '/api/gauntlet/gauntlet-003/status',
        estimated_duration_seconds: 60,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.run({
        input: 'Claim to evaluate',
        personas: ['adversarial', 'edge_case', 'stress'],
        parallel: true,
        metadata: { source: 'automated-test' },
      });

      expect(result.gauntlet_id).toBe('gauntlet-003');
      expect(result.estimated_duration_seconds).toBe(60);
    });

    it('should run gauntlet and wait for completion', async () => {
      const mockRunResponse = {
        gauntlet_id: 'gauntlet-001',
        status: 'pending',
        status_url: '/api/gauntlet/gauntlet-001/status',
      };

      const mockCompletedRun = {
        id: 'gauntlet-001',
        name: 'Security Gauntlet',
        status: 'completed',
        created_at: '2024-01-15T10:00:00Z',
        started_at: '2024-01-15T10:00:01Z',
        completed_at: '2024-01-15T10:02:00Z',
        input: 'Our API is secure against SQL injection',
        results: {
          passed: 8,
          failed: 2,
          total: 10,
          pass_rate: 0.8,
          risk_score: 0.25,
        },
      };

      // First call: runGauntlet
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRunResponse)),
      });

      // Second call: getGauntlet (polling)
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockCompletedRun)),
      });

      const result = await client.gauntlet.runAndWait({
        input: 'Our API is secure against SQL injection',
        personas: ['adversarial'],
      });

      expect(result.id).toBe('gauntlet-001');
      expect(result.status).toBe('completed');
      expect(result.results.pass_rate).toBe(0.8);
    });

    it('should run gauntlet and wait with custom timeout', async () => {
      const mockRunResponse = {
        gauntlet_id: 'gauntlet-004',
        status: 'pending',
        status_url: '/api/gauntlet/gauntlet-004/status',
      };

      const mockCompletedRun = {
        id: 'gauntlet-004',
        status: 'completed',
        created_at: '2024-01-15T10:00:00Z',
        completed_at: '2024-01-15T10:01:00Z',
        input: 'Quick test',
        results: {
          passed: 5,
          failed: 0,
          total: 5,
          pass_rate: 1.0,
          risk_score: 0.0,
        },
      };

      // First call: runGauntlet
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRunResponse)),
      });

      // Second call: getGauntlet (polling)
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockCompletedRun)),
      });

      const result = await client.gauntlet.runAndWait(
        { input: 'Quick test' },
        { pollIntervalMs: 500, timeoutMs: 30000 }
      );

      expect(result.status).toBe('completed');
      expect(result.results.pass_rate).toBe(1.0);
    });
  });

  // ===========================================================================
  // Receipts
  // ===========================================================================

  describe('Receipts', () => {
    it('should list gauntlet receipts', async () => {
      const mockResponse = {
        receipts: [
          {
            id: 'receipt-001',
            debate_id: 'gauntlet-001',
            verdict: 'pass',
            confidence: 0.95,
            consensus_reached: true,
            created_at: '2024-01-15T10:02:00Z',
          },
          {
            id: 'receipt-002',
            debate_id: 'gauntlet-002',
            verdict: 'fail',
            confidence: 0.88,
            consensus_reached: true,
            created_at: '2024-01-15T11:00:00Z',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.list({ limit: 10 });

      expect(result.receipts).toHaveLength(2);
      expect(result.receipts[0].id).toBe('receipt-001');
      expect(result.receipts[0].verdict).toBe('pass');
    });

    it('should list receipts with verdict filter', async () => {
      const mockResponse = {
        receipts: [
          {
            id: 'receipt-003',
            debate_id: 'gauntlet-003',
            verdict: 'fail',
            confidence: 0.75,
            created_at: '2024-01-15T12:00:00Z',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.list({ verdict: 'fail' });

      expect(result.receipts).toHaveLength(1);
      expect(result.receipts[0].verdict).toBe('fail');
    });

    it('should get a specific receipt', async () => {
      const mockReceipt = {
        id: 'receipt-001',
        debate_id: 'gauntlet-001',
        verdict: 'pass',
        confidence: 0.95,
        consensus_reached: true,
        participating_agents: ['claude', 'gpt-4', 'gemini'],
        dissenting_agents: [],
        evidence: [
          { type: 'test', name: 'SQL injection test', result: 'pass' },
          { type: 'test', name: 'XSS test', result: 'pass' },
        ],
        reasoning: 'All security tests passed with high confidence',
        hash: 'sha256:abc123def456',
        created_at: '2024-01-15T10:02:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockReceipt)),
      });

      const result = await client.gauntlet.get('receipt-001');

      expect(result.id).toBe('receipt-001');
      expect(result.verdict).toBe('pass');
      expect(result.participating_agents).toHaveLength(3);
      expect(result.evidence).toHaveLength(2);
      expect(result.hash).toBeDefined();
    });

    it('should verify receipt integrity', async () => {
      const mockVerification = {
        valid: true,
        hash: 'sha256:abc123def456',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockVerification)),
      });

      const result = await client.gauntlet.verify('receipt-001');

      expect(result.valid).toBe(true);
      expect(result.hash).toBe('sha256:abc123def456');
    });

    it('should detect invalid receipt', async () => {
      const mockVerification = {
        valid: false,
        hash: 'sha256:tampered',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockVerification)),
      });

      const result = await client.gauntlet.verify('receipt-tampered');

      expect(result.valid).toBe(false);
    });
  });

  // ===========================================================================
  // Export
  // ===========================================================================

  describe('Export', () => {
    it('should export receipt as JSON', async () => {
      const mockExport = {
        format: 'json',
        content: '{"id":"receipt-001","verdict":"pass"}',
        filename: 'gauntlet-receipt-001.json',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExport)),
      });

      const result = await client.gauntlet.export('receipt-001', 'json');

      expect(result.format).toBe('json');
      expect(result.content).toBeDefined();
    });

    it('should export receipt as HTML', async () => {
      const mockExport = {
        format: 'html',
        content: '<html><body>Gauntlet Report</body></html>',
        filename: 'gauntlet-receipt-001.html',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExport)),
      });

      const result = await client.gauntlet.export('receipt-001', 'html');

      expect(result.format).toBe('html');
      expect(result.content).toContain('<html>');
    });

    it('should export receipt as Markdown', async () => {
      const mockExport = {
        format: 'markdown',
        content: '# Gauntlet Report\n\n## Summary\n\nVerdict: Pass',
        filename: 'gauntlet-receipt-001.md',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExport)),
      });

      const result = await client.gauntlet.export('receipt-001', 'markdown');

      expect(result.format).toBe('markdown');
      expect(result.content).toContain('# Gauntlet Report');
    });

    it('should export receipt as SARIF', async () => {
      const mockExport = {
        format: 'sarif',
        content: '{"$schema":"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json","version":"2.1.0"}',
        filename: 'gauntlet-receipt-001.sarif',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExport)),
      });

      const result = await client.gauntlet.export('receipt-001', 'sarif');

      expect(result.format).toBe('sarif');
      expect(result.content).toContain('sarif-schema');
    });
  });

  // ===========================================================================
  // Personas
  // ===========================================================================

  describe('Personas', () => {
    it('should list available personas', async () => {
      const mockResponse = {
        personas: [
          {
            id: 'adversarial-1',
            name: 'Adversarial Attacker',
            description: 'Attempts to find vulnerabilities',
            category: 'adversarial',
            severity: 'high',
            tags: ['security', 'attack'],
            enabled: true,
          },
          {
            id: 'edge-case-1',
            name: 'Edge Case Explorer',
            description: 'Tests boundary conditions',
            category: 'edge_case',
            severity: 'medium',
            tags: ['testing', 'boundaries'],
            enabled: true,
          },
          {
            id: 'compliance-1',
            name: 'Compliance Checker',
            description: 'Verifies regulatory compliance',
            category: 'compliance',
            severity: 'critical',
            tags: ['compliance', 'audit'],
            enabled: true,
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.listPersonas();

      expect(result.personas).toHaveLength(3);
      expect(result.personas[0].category).toBe('adversarial');
    });

    it('should filter personas by category', async () => {
      const mockResponse = {
        personas: [
          {
            id: 'compliance-1',
            name: 'Compliance Checker',
            description: 'Verifies regulatory compliance',
            category: 'compliance',
            severity: 'critical',
            tags: ['compliance', 'audit'],
            enabled: true,
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.listPersonas({ category: 'compliance' });

      expect(result.personas).toHaveLength(1);
      expect(result.personas[0].category).toBe('compliance');
    });

    it('should filter personas by enabled status', async () => {
      const mockResponse = {
        personas: [
          {
            id: 'disabled-1',
            name: 'Disabled Persona',
            description: 'Currently disabled',
            category: 'custom',
            severity: 'low',
            tags: [],
            enabled: false,
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.listPersonas({ enabled: false });

      expect(result.personas).toHaveLength(1);
      expect(result.personas[0].enabled).toBe(false);
    });
  });

  // ===========================================================================
  // Results
  // ===========================================================================

  describe('Results', () => {
    it('should list gauntlet results', async () => {
      const mockResponse = {
        results: [
          {
            id: 'result-001',
            gauntlet_id: 'gauntlet-001',
            scenario: 'SQL Injection Test',
            persona: 'adversarial',
            status: 'pass',
            input: "'; DROP TABLE users;--",
            output: 'Query blocked',
            duration_ms: 150,
            timestamp: '2024-01-15T10:01:00Z',
          },
          {
            id: 'result-002',
            gauntlet_id: 'gauntlet-001',
            scenario: 'XSS Test',
            persona: 'adversarial',
            status: 'pass',
            input: '<script>alert("xss")</script>',
            output: 'Script escaped',
            duration_ms: 100,
            timestamp: '2024-01-15T10:01:01Z',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.listResults({ gauntlet_id: 'gauntlet-001' });

      expect(result.results).toHaveLength(2);
      expect(result.results[0].status).toBe('pass');
    });

    it('should filter results by status', async () => {
      const mockResponse = {
        results: [
          {
            id: 'result-003',
            gauntlet_id: 'gauntlet-002',
            scenario: 'Rate Limit Test',
            status: 'fail',
            input: 'Burst 1000 requests',
            output: 'Rate limit not enforced',
            findings: [
              { type: 'vulnerability', message: 'Missing rate limiting', severity: 'high' },
            ],
            timestamp: '2024-01-15T11:00:00Z',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.listResults({ status: 'fail' });

      expect(result.results).toHaveLength(1);
      expect(result.results[0].status).toBe('fail');
      expect(result.results[0].findings).toBeDefined();
    });
  });

  // ===========================================================================
  // Heatmap
  // ===========================================================================

  describe('Heatmap', () => {
    it('should get heatmap as JSON', async () => {
      const mockHeatmap = {
        gauntlet_id: 'gauntlet-001',
        dimensions: {
          categories: ['security', 'performance', 'compliance'],
          personas: ['adversarial', 'edge_case', 'stress'],
        },
        matrix: [
          [0.9, 0.8, 0.95],
          [0.85, 0.7, 0.9],
          [0.75, 0.6, 0.8],
        ],
        hotspots: [
          { category: 'performance', persona: 'stress', risk: 0.4 },
        ],
        generated_at: '2024-01-15T10:05:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHeatmap)),
      });

      const result = await client.gauntlet.getHeatmap('gauntlet-001', 'json');

      expect(result.gauntlet_id).toBe('gauntlet-001');
      expect(result.dimensions.categories).toHaveLength(3);
      expect(result.matrix).toHaveLength(3);
      expect(result.hotspots).toHaveLength(1);
    });

    it('should get heatmap as SVG', async () => {
      const mockHeatmap = {
        gauntlet_id: 'gauntlet-001',
        dimensions: {
          categories: ['security', 'performance'],
          personas: ['adversarial', 'edge_case'],
        },
        matrix: [
          [0.9, 0.8],
          [0.85, 0.7],
        ],
        svg: '<svg xmlns="http://www.w3.org/2000/svg">...</svg>',
        hotspots: [],
        generated_at: '2024-01-15T10:05:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHeatmap)),
      });

      const result = await client.gauntlet.getHeatmap('gauntlet-001', 'svg');

      expect(result.gauntlet_id).toBe('gauntlet-001');
    });
  });

  // ===========================================================================
  // Compare
  // ===========================================================================

  describe('Compare', () => {
    it('should compare two gauntlet runs', async () => {
      const mockComparison = {
        gauntlet_a: 'gauntlet-001',
        gauntlet_b: 'gauntlet-002',
        comparison: {
          pass_rate_delta: 0.15,
          risk_score_delta: -0.1,
          new_failures: ['rate-limiting'],
          resolved_failures: ['xss-protection'],
          common_failures: ['sql-injection'],
        },
        by_category: [
          { category: 'security', a_pass_rate: 0.9, b_pass_rate: 0.85, delta: -0.05 },
          { category: 'performance', a_pass_rate: 0.7, b_pass_rate: 0.85, delta: 0.15 },
        ],
        recommendation: 'investigate',
        generated_at: '2024-01-15T12:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockComparison)),
      });

      const result = await client.gauntlet.compare('gauntlet-001', 'gauntlet-002');

      expect(result.gauntlet_a).toBe('gauntlet-001');
      expect(result.gauntlet_b).toBe('gauntlet-002');
      expect(result.comparison.pass_rate_delta).toBe(0.15);
      expect(result.recommendation).toBe('investigate');
    });

    it('should recommend promotion when improved', async () => {
      const mockComparison = {
        gauntlet_a: 'gauntlet-old',
        gauntlet_b: 'gauntlet-new',
        comparison: {
          pass_rate_delta: 0.2,
          risk_score_delta: -0.3,
          new_failures: [],
          resolved_failures: ['xss', 'csrf', 'sqli'],
          common_failures: [],
        },
        by_category: [],
        recommendation: 'promote',
        generated_at: '2024-01-15T12:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockComparison)),
      });

      const result = await client.gauntlet.compare('gauntlet-old', 'gauntlet-new');

      expect(result.recommendation).toBe('promote');
      expect(result.comparison.resolved_failures).toHaveLength(3);
    });

    it('should recommend block when degraded', async () => {
      const mockComparison = {
        gauntlet_a: 'gauntlet-baseline',
        gauntlet_b: 'gauntlet-regression',
        comparison: {
          pass_rate_delta: -0.4,
          risk_score_delta: 0.5,
          new_failures: ['auth-bypass', 'data-leak'],
          resolved_failures: [],
          common_failures: [],
        },
        by_category: [],
        recommendation: 'block',
        generated_at: '2024-01-15T12:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockComparison)),
      });

      const result = await client.gauntlet.compare('gauntlet-baseline', 'gauntlet-regression');

      expect(result.recommendation).toBe('block');
      expect(result.comparison.new_failures).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('Error Handling', () => {
    it('should handle gauntlet not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'Gauntlet not found',
          code: 'NOT_FOUND',
        }),
      });

      await expect(client.gauntlet.get('nonexistent'))
        .rejects.toThrow('Gauntlet not found');
    });

    it('should handle invalid input', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Input is required',
          code: 'MISSING_FIELD',
          field: 'input',
        }),
      });

      await expect(client.gauntlet.run({ input: '' }))
        .rejects.toThrow('Input is required');
    });

    it('should handle invalid persona', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Unknown persona: invalid-persona',
          code: 'INVALID_VALUE',
        }),
      });

      await expect(
        client.gauntlet.run({
          input: 'Test',
          personas: ['invalid-persona'],
        })
      ).rejects.toThrow('Unknown persona');
    });

    it('should handle rate limiting', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({
          error: 'Rate limit exceeded',
          code: 'RATE_LIMITED',
          retry_after: 60,
        }),
      });

      await expect(client.gauntlet.run({ input: 'Test' }))
        .rejects.toThrow('Rate limit exceeded');
    });

    it('should handle invalid export format', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid export format',
          code: 'INVALID_VALUE',
        }),
      });

      // @ts-expect-error - Testing invalid format
      await expect(client.gauntlet.export('receipt-001', 'pdf'))
        .rejects.toThrow('Invalid export format');
    });
  });

  // ===========================================================================
  // Type Safety
  // ===========================================================================

  describe('Type Safety', () => {
    it('should return properly typed gauntlet run response', async () => {
      const mockResponse = {
        gauntlet_id: 'gauntlet-001',
        status: 'pending',
        status_url: '/api/gauntlet/gauntlet-001/status',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.run({ input: 'Test claim' });

      expect(typeof result.gauntlet_id).toBe('string');
      expect(['pending', 'running']).toContain(result.status);
      expect(typeof result.status_url).toBe('string');
    });

    it('should return properly typed receipt', async () => {
      const mockReceipt = {
        id: 'receipt-001',
        debate_id: 'gauntlet-001',
        verdict: 'pass',
        confidence: 0.95,
        consensus_reached: true,
        participating_agents: ['claude', 'gpt-4'],
        created_at: '2024-01-15T10:02:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockReceipt)),
      });

      const result = await client.gauntlet.get('receipt-001');

      expect(typeof result.id).toBe('string');
      expect(typeof result.verdict).toBe('string');
      expect(typeof result.confidence).toBe('number');
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
      expect(typeof result.consensus_reached).toBe('boolean');
      expect(Array.isArray(result.participating_agents)).toBe(true);
    });

    it('should return properly typed persona', async () => {
      const mockResponse = {
        personas: [
          {
            id: 'persona-001',
            name: 'Test Persona',
            description: 'A test persona',
            category: 'adversarial',
            severity: 'high',
            tags: ['test'],
            enabled: true,
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.listPersonas();
      const persona = result.personas[0];

      expect(typeof persona.id).toBe('string');
      expect(typeof persona.name).toBe('string');
      expect(['adversarial', 'edge_case', 'stress', 'compliance', 'custom'])
        .toContain(persona.category);
      expect(['low', 'medium', 'high', 'critical']).toContain(persona.severity);
      expect(typeof persona.enabled).toBe('boolean');
      expect(Array.isArray(persona.tags)).toBe(true);
    });

    it('should return properly typed gauntlet result', async () => {
      const mockResponse = {
        results: [
          {
            id: 'result-001',
            gauntlet_id: 'gauntlet-001',
            scenario: 'Test Scenario',
            status: 'pass',
            duration_ms: 100,
            timestamp: '2024-01-15T10:00:00Z',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.gauntlet.listResults();
      const gauntletResult = result.results[0];

      expect(typeof gauntletResult.id).toBe('string');
      expect(typeof gauntletResult.gauntlet_id).toBe('string');
      expect(['pass', 'fail', 'error', 'skip']).toContain(gauntletResult.status);
      expect(typeof gauntletResult.duration_ms).toBe('number');
    });

    it('should return properly typed comparison', async () => {
      const mockComparison = {
        gauntlet_a: 'g1',
        gauntlet_b: 'g2',
        comparison: {
          pass_rate_delta: 0.1,
          risk_score_delta: -0.05,
          new_failures: [],
          resolved_failures: [],
          common_failures: [],
        },
        by_category: [],
        recommendation: 'promote',
        generated_at: '2024-01-15T12:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockComparison)),
      });

      const result = await client.gauntlet.compare('g1', 'g2');

      expect(typeof result.gauntlet_a).toBe('string');
      expect(typeof result.gauntlet_b).toBe('string');
      expect(['promote', 'investigate', 'block']).toContain(result.recommendation);
      expect(typeof result.comparison.pass_rate_delta).toBe('number');
    });
  });
});
