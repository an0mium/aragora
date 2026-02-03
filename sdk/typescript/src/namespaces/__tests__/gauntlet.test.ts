/**
 * Gauntlet Namespace Tests
 *
 * Comprehensive tests for the GauntletAPI namespace class.
 * Tests all methods including:
 * - Running gauntlets (blocking and non-blocking)
 * - Listing and retrieving receipts
 * - Verifying receipt integrity
 * - Exporting receipts
 * - Persona management
 * - Results and heatmaps
 * - Gauntlet comparison
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { GauntletAPI } from '../gauntlet';

interface MockClient {
  runGauntlet: Mock;
  runGauntletAndWait: Mock;
  listGauntletReceipts: Mock;
  getGauntletReceipt: Mock;
  verifyGauntletReceipt: Mock;
  exportGauntletReceipt: Mock;
  listGauntletPersonas: Mock;
  listGauntletResults: Mock;
  getGauntletHeatmap: Mock;
  compareGauntlets: Mock;
}

function createMockClient(): MockClient {
  return {
    runGauntlet: vi.fn(),
    runGauntletAndWait: vi.fn(),
    listGauntletReceipts: vi.fn(),
    getGauntletReceipt: vi.fn(),
    verifyGauntletReceipt: vi.fn(),
    exportGauntletReceipt: vi.fn(),
    listGauntletPersonas: vi.fn(),
    listGauntletResults: vi.fn(),
    getGauntletHeatmap: vi.fn(),
    compareGauntlets: vi.fn(),
  };
}

describe('GauntletAPI', () => {
  let api: GauntletAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    vi.clearAllMocks();
    mockClient = createMockClient();
    api = new GauntletAPI(mockClient as any);
  });

  // ===========================================================================
  // Running Gauntlets
  // ===========================================================================

  describe('Running Gauntlets', () => {
    it('should run a gauntlet (non-blocking)', async () => {
      const mockResponse = {
        gauntlet_id: 'g_123',
        status: 'running',
      };
      mockClient.runGauntlet.mockResolvedValue(mockResponse);

      const request = {
        input: 'Our API is secure against SQL injection',
        personas: ['adversarial', 'compliance'],
      };
      const result = await api.run(request);

      expect(mockClient.runGauntlet).toHaveBeenCalledWith(request);
      expect(result.gauntlet_id).toBe('g_123');
      expect(result.status).toBe('running');
    });

    it('should run a gauntlet and wait for completion', async () => {
      const mockRun = {
        gauntlet_id: 'g_124',
        status: 'completed',
        winner: 'claude',
        rounds: [
          { round: 1, attacker_score: 0.7, defender_score: 0.6 },
          { round: 2, attacker_score: 0.8, defender_score: 0.65 },
        ],
      };
      mockClient.runGauntletAndWait.mockResolvedValue(mockRun);

      const request = { input: 'Test claim' };
      const result = await api.runAndWait(request, { pollIntervalMs: 1000, timeoutMs: 30000 });

      expect(mockClient.runGauntletAndWait).toHaveBeenCalledWith(request, {
        pollIntervalMs: 1000,
        timeoutMs: 30000,
      });
      expect(result.status).toBe('completed');
    });

    it('should run a gauntlet and wait with default options', async () => {
      const mockRun = { gauntlet_id: 'g_125', status: 'completed' };
      mockClient.runGauntletAndWait.mockResolvedValue(mockRun);

      const request = { input: 'Another claim' };
      await api.runAndWait(request);

      expect(mockClient.runGauntletAndWait).toHaveBeenCalledWith(request, undefined);
    });
  });

  // ===========================================================================
  // Receipt Management
  // ===========================================================================

  describe('Receipt Management', () => {
    it('should list receipts without filters', async () => {
      const mockReceipts = {
        receipts: [
          { receipt_id: 'rcpt_1', topic: 'Topic 1', timestamp: '2024-01-20' },
          { receipt_id: 'rcpt_2', topic: 'Topic 2', timestamp: '2024-01-19' },
        ],
      };
      mockClient.listGauntletReceipts.mockResolvedValue(mockReceipts);

      const result = await api.list();

      expect(mockClient.listGauntletReceipts).toHaveBeenCalledWith(undefined);
      expect(result.receipts).toHaveLength(2);
    });

    it('should list receipts with verdict filter', async () => {
      const mockReceipts = { receipts: [{ receipt_id: 'rcpt_1' }] };
      mockClient.listGauntletReceipts.mockResolvedValue(mockReceipts);

      await api.list({ verdict: 'pass', limit: 10, offset: 0 });

      expect(mockClient.listGauntletReceipts).toHaveBeenCalledWith({
        verdict: 'pass',
        limit: 10,
        offset: 0,
      });
    });

    it('should get receipt by ID', async () => {
      const mockReceipt = {
        receipt_id: 'rcpt_456',
        gauntlet_id: 'g_123',
        topic: 'Microservices adoption',
        hash: 'sha256:abc123',
        timestamp: '2024-01-20T10:05:00Z',
        verified: true,
      };
      mockClient.getGauntletReceipt.mockResolvedValue(mockReceipt);

      const result = await api.get('rcpt_456');

      expect(mockClient.getGauntletReceipt).toHaveBeenCalledWith('rcpt_456');
      expect(result.receipt_id).toBe('rcpt_456');
      expect(result.hash).toContain('sha256');
    });
  });

  // ===========================================================================
  // Verification
  // ===========================================================================

  describe('Verification', () => {
    it('should verify a receipt', async () => {
      const mockVerification = {
        valid: true,
        hash: 'sha256:abc123def456',
      };
      mockClient.verifyGauntletReceipt.mockResolvedValue(mockVerification);

      const result = await api.verify('rcpt_456');

      expect(mockClient.verifyGauntletReceipt).toHaveBeenCalledWith('rcpt_456');
      expect(result.valid).toBe(true);
      expect(result.hash).toContain('sha256');
    });

    it('should detect invalid receipt', async () => {
      const mockVerification = {
        valid: false,
        hash: '',
      };
      mockClient.verifyGauntletReceipt.mockResolvedValue(mockVerification);

      const result = await api.verify('rcpt_fake');

      expect(mockClient.verifyGauntletReceipt).toHaveBeenCalledWith('rcpt_fake');
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
        content: '{"receipt_id": "rcpt_456"}',
      };
      mockClient.exportGauntletReceipt.mockResolvedValue(mockExport);

      const result = await api.export('rcpt_456', 'json');

      expect(mockClient.exportGauntletReceipt).toHaveBeenCalledWith('rcpt_456', 'json');
      expect(result.format).toBe('json');
    });

    it('should export receipt as HTML', async () => {
      const mockExport = { format: 'html', content: '<html>...</html>' };
      mockClient.exportGauntletReceipt.mockResolvedValue(mockExport);

      const result = await api.export('rcpt_456', 'html');

      expect(mockClient.exportGauntletReceipt).toHaveBeenCalledWith('rcpt_456', 'html');
      expect(result.format).toBe('html');
    });

    it('should export receipt as markdown', async () => {
      const mockExport = { format: 'markdown', content: '# Receipt' };
      mockClient.exportGauntletReceipt.mockResolvedValue(mockExport);

      await api.export('rcpt_456', 'markdown');

      expect(mockClient.exportGauntletReceipt).toHaveBeenCalledWith('rcpt_456', 'markdown');
    });

    it('should export receipt as SARIF', async () => {
      const mockExport = { format: 'sarif', content: '{}' };
      mockClient.exportGauntletReceipt.mockResolvedValue(mockExport);

      await api.export('rcpt_456', 'sarif');

      expect(mockClient.exportGauntletReceipt).toHaveBeenCalledWith('rcpt_456', 'sarif');
    });
  });

  // ===========================================================================
  // Personas
  // ===========================================================================

  describe('Personas', () => {
    it('should list personas without filters', async () => {
      const mockPersonas = {
        personas: [
          { persona_id: 'p_1', name: 'Adversarial', category: 'security' },
          { persona_id: 'p_2', name: 'Compliance', category: 'governance' },
        ],
      };
      mockClient.listGauntletPersonas.mockResolvedValue(mockPersonas);

      const result = await api.listPersonas();

      expect(mockClient.listGauntletPersonas).toHaveBeenCalledWith(undefined);
      expect(result.personas).toHaveLength(2);
    });

    it('should list personas with category filter', async () => {
      const mockPersonas = { personas: [{ persona_id: 'p_1' }] };
      mockClient.listGauntletPersonas.mockResolvedValue(mockPersonas);

      await api.listPersonas({ category: 'security', enabled: true });

      expect(mockClient.listGauntletPersonas).toHaveBeenCalledWith({
        category: 'security',
        enabled: true,
      });
    });
  });

  // ===========================================================================
  // Results and Heatmap
  // ===========================================================================

  describe('Results and Heatmap', () => {
    it('should list results without filters', async () => {
      const mockResults = {
        results: [
          { gauntlet_id: 'g_1', status: 'completed', score: 0.85 },
          { gauntlet_id: 'g_2', status: 'completed', score: 0.72 },
        ],
      };
      mockClient.listGauntletResults.mockResolvedValue(mockResults);

      const result = await api.listResults();

      expect(mockClient.listGauntletResults).toHaveBeenCalledWith(undefined);
      expect(result.results).toHaveLength(2);
    });

    it('should list results with filters', async () => {
      const mockResults = { results: [] };
      mockClient.listGauntletResults.mockResolvedValue(mockResults);

      await api.listResults({ gauntlet_id: 'g_1', status: 'completed', limit: 5 });

      expect(mockClient.listGauntletResults).toHaveBeenCalledWith({
        gauntlet_id: 'g_1',
        status: 'completed',
        limit: 5,
      });
    });

    it('should get heatmap as JSON', async () => {
      const mockHeatmap = {
        gauntlet_id: 'g_1',
        cells: [{ x: 0, y: 0, value: 0.85 }],
      };
      mockClient.getGauntletHeatmap.mockResolvedValue(mockHeatmap);

      const result = await api.getHeatmap('g_1', 'json');

      expect(mockClient.getGauntletHeatmap).toHaveBeenCalledWith('g_1', 'json');
      expect(result.gauntlet_id).toBe('g_1');
    });

    it('should get heatmap as SVG', async () => {
      const mockHeatmap = { gauntlet_id: 'g_1', svg: '<svg>...</svg>' };
      mockClient.getGauntletHeatmap.mockResolvedValue(mockHeatmap);

      await api.getHeatmap('g_1', 'svg');

      expect(mockClient.getGauntletHeatmap).toHaveBeenCalledWith('g_1', 'svg');
    });
  });

  // ===========================================================================
  // Comparison
  // ===========================================================================

  describe('Comparison', () => {
    it('should compare two gauntlet runs', async () => {
      const mockComparison = {
        gauntlet_1: 'g_1',
        gauntlet_2: 'g_2',
        score_diff: 0.13,
        common_findings: 5,
        unique_to_first: 2,
        unique_to_second: 3,
      };
      mockClient.compareGauntlets.mockResolvedValue(mockComparison);

      const result = await api.compare('g_1', 'g_2');

      expect(mockClient.compareGauntlets).toHaveBeenCalledWith('g_1', 'g_2');
      expect(result.score_diff).toBe(0.13);
      expect(result.common_findings).toBe(5);
    });
  });
});
