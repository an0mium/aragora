/**
 * Gauntlet Namespace Tests
 *
 * Comprehensive tests for the gauntlet namespace API including:
 * - Running gauntlets
 * - Receipt management
 * - Gauntlet history
 * - Verification
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { GauntletNamespace } from '../gauntlet';

interface MockClient {
  request: Mock;
}

describe('GauntletNamespace', () => {
  let api: GauntletNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new GauntletNamespace(mockClient as any);
  });

  // ===========================================================================
  // Running Gauntlets
  // ===========================================================================

  describe('Running Gauntlets', () => {
    it('should run gauntlet with topic', async () => {
      const mockResult = {
        gauntlet_id: 'g_123',
        topic: 'Should we adopt microservices?',
        status: 'completed',
        rounds: 5,
        attacker: 'claude',
        defender: 'gpt4',
        winner: 'claude',
        receipt_id: 'rcpt_456',
        started_at: '2024-01-20T10:00:00Z',
        completed_at: '2024-01-20T10:05:00Z',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.run({ topic: 'Should we adopt microservices?' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gauntlet/run', {
        json: { topic: 'Should we adopt microservices?' },
      });
      expect(result.gauntlet_id).toBe('g_123');
      expect(result.winner).toBe('claude');
    });

    it('should run gauntlet with custom agents', async () => {
      const mockResult = {
        gauntlet_id: 'g_124',
        attacker: 'gemini',
        defender: 'mistral',
        status: 'completed',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.run({
        topic: 'Test topic',
        attacker: 'gemini',
        defender: 'mistral',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gauntlet/run', {
        json: { topic: 'Test topic', attacker: 'gemini', defender: 'mistral' },
      });
      expect(result.attacker).toBe('gemini');
    });

    it('should run gauntlet with custom rounds', async () => {
      const mockResult = { gauntlet_id: 'g_125', rounds: 10 };
      mockClient.request.mockResolvedValue(mockResult);

      await api.run({ topic: 'Test', rounds: 10 });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gauntlet/run', {
        json: { topic: 'Test', rounds: 10 },
      });
    });

    it('should run gauntlet with context', async () => {
      const mockResult = { gauntlet_id: 'g_126' };
      mockClient.request.mockResolvedValue(mockResult);

      await api.run({
        topic: 'Architecture decision',
        context: 'We have 50 microservices already',
        domain: 'technology',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gauntlet/run', {
        json: {
          topic: 'Architecture decision',
          context: 'We have 50 microservices already',
          domain: 'technology',
        },
      });
    });
  });

  // ===========================================================================
  // Receipt Management
  // ===========================================================================

  describe('Receipt Management', () => {
    it('should get receipt by ID', async () => {
      const mockReceipt = {
        receipt_id: 'rcpt_456',
        gauntlet_id: 'g_123',
        topic: 'Microservices adoption',
        hash: 'sha256:abc123...',
        timestamp: '2024-01-20T10:05:00Z',
        winner: 'claude',
        confidence: 0.85,
        rounds_summary: [
          { round: 1, attacker_score: 0.7, defender_score: 0.6 },
          { round: 2, attacker_score: 0.8, defender_score: 0.65 },
        ],
        verified: true,
      };
      mockClient.request.mockResolvedValue(mockReceipt);

      const result = await api.getReceipt('rcpt_456');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gauntlet/receipts/rcpt_456');
      expect(result.hash).toContain('sha256');
      expect(result.verified).toBe(true);
    });

    it('should list receipts', async () => {
      const mockReceipts = {
        receipts: [
          { receipt_id: 'rcpt_1', topic: 'Topic 1', timestamp: '2024-01-20' },
          { receipt_id: 'rcpt_2', topic: 'Topic 2', timestamp: '2024-01-19' },
        ],
        total: 2,
        has_more: false,
      };
      mockClient.request.mockResolvedValue(mockReceipts);

      const result = await api.listReceipts();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gauntlet/receipts', {
        params: undefined,
      });
      expect(result.receipts).toHaveLength(2);
    });

    it('should list receipts with pagination', async () => {
      const mockReceipts = { receipts: [], total: 100 };
      mockClient.request.mockResolvedValue(mockReceipts);

      await api.listReceipts({ limit: 10, offset: 20 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gauntlet/receipts', {
        params: { limit: 10, offset: 20 },
      });
    });

    it('should list receipts by gauntlet', async () => {
      const mockReceipts = { receipts: [{ receipt_id: 'rcpt_1' }] };
      mockClient.request.mockResolvedValue(mockReceipts);

      await api.listReceipts({ gauntlet_id: 'g_123' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gauntlet/receipts', {
        params: { gauntlet_id: 'g_123' },
      });
    });

    it('should export receipt as PDF', async () => {
      const mockExport = {
        download_url: 'https://storage.example.com/receipts/rcpt_456.pdf',
        expires_at: '2024-01-20T11:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.exportReceipt('rcpt_456', 'pdf');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/gauntlet/receipts/rcpt_456/export',
        { params: { format: 'pdf' } }
      );
      expect(result.download_url).toContain('.pdf');
    });

    it('should export receipt as JSON', async () => {
      const mockExport = { data: { receipt_id: 'rcpt_456' } };
      mockClient.request.mockResolvedValue(mockExport);

      await api.exportReceipt('rcpt_456', 'json');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/gauntlet/receipts/rcpt_456/export',
        { params: { format: 'json' } }
      );
    });
  });

  // ===========================================================================
  // Gauntlet History
  // ===========================================================================

  describe('Gauntlet History', () => {
    it('should get gauntlet by ID', async () => {
      const mockGauntlet = {
        gauntlet_id: 'g_123',
        topic: 'Microservices',
        status: 'completed',
        attacker: 'claude',
        defender: 'gpt4',
        winner: 'claude',
        rounds: [
          {
            round: 1,
            attacker_argument: 'Scalability benefits...',
            defender_response: 'Complexity costs...',
            scores: { attacker: 0.7, defender: 0.6 },
          },
        ],
        receipt_id: 'rcpt_456',
      };
      mockClient.request.mockResolvedValue(mockGauntlet);

      const result = await api.get('g_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gauntlet/g_123');
      expect(result.rounds).toHaveLength(1);
    });

    it('should list gauntlets', async () => {
      const mockGauntlets = {
        gauntlets: [
          { gauntlet_id: 'g_1', topic: 'Topic 1', status: 'completed' },
          { gauntlet_id: 'g_2', topic: 'Topic 2', status: 'in_progress' },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockGauntlets);

      const result = await api.list();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gauntlet', {
        params: undefined,
      });
      expect(result.gauntlets).toHaveLength(2);
    });

    it('should list gauntlets with status filter', async () => {
      const mockGauntlets = { gauntlets: [] };
      mockClient.request.mockResolvedValue(mockGauntlets);

      await api.list({ status: 'completed' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gauntlet', {
        params: { status: 'completed' },
      });
    });
  });

  // ===========================================================================
  // Verification
  // ===========================================================================

  describe('Verification', () => {
    it('should verify receipt', async () => {
      const mockVerification = {
        valid: true,
        receipt_id: 'rcpt_456',
        hash_verified: true,
        timestamp_verified: true,
        chain_verified: true,
        verification_time: '2024-01-20T10:10:00Z',
      };
      mockClient.request.mockResolvedValue(mockVerification);

      const result = await api.verifyReceipt('rcpt_456');

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/gauntlet/receipts/rcpt_456/verify'
      );
      expect(result.valid).toBe(true);
      expect(result.hash_verified).toBe(true);
    });

    it('should detect invalid receipt', async () => {
      const mockVerification = {
        valid: false,
        receipt_id: 'rcpt_fake',
        hash_verified: false,
        error: 'Hash mismatch',
      };
      mockClient.request.mockResolvedValue(mockVerification);

      const result = await api.verifyReceipt('rcpt_fake');

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Hash mismatch');
    });

    it('should verify with external hash', async () => {
      const mockVerification = { valid: true };
      mockClient.request.mockResolvedValue(mockVerification);

      await api.verifyReceipt('rcpt_456', { expected_hash: 'sha256:abc123' });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/gauntlet/receipts/rcpt_456/verify',
        { json: { expected_hash: 'sha256:abc123' } }
      );
    });
  });

  // ===========================================================================
  // Statistics
  // ===========================================================================

  describe('Statistics', () => {
    it('should get gauntlet statistics', async () => {
      const mockStats = {
        total_gauntlets: 500,
        completed: 480,
        in_progress: 15,
        failed: 5,
        average_rounds: 5.2,
        win_rates: {
          claude: 0.65,
          gpt4: 0.58,
          gemini: 0.52,
        },
        most_debated_topics: [
          { topic: 'AI Ethics', count: 50 },
          { topic: 'Architecture', count: 45 },
        ],
      };
      mockClient.request.mockResolvedValue(mockStats);

      const result = await api.getStats();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gauntlet/stats');
      expect(result.total_gauntlets).toBe(500);
      expect(result.win_rates.claude).toBe(0.65);
    });
  });
});
