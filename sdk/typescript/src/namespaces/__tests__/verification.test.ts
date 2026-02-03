/**
 * Verification Namespace Tests
 *
 * Comprehensive tests for the verification namespace API including:
 * - Claim verification
 * - Debate conclusion verification
 * - Verification reports
 * - System verification status
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { VerificationAPI } from '../verification';

interface MockClient {
  verifyClaim: Mock;
  getVerificationStatus: Mock;
  verifyDebateConclusion: Mock;
  getVerificationReport: Mock;
}

describe('VerificationAPI Namespace', () => {
  let api: VerificationAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      verifyClaim: vi.fn(),
      getVerificationStatus: vi.fn(),
      verifyDebateConclusion: vi.fn(),
      getVerificationReport: vi.fn(),
    };
    api = new VerificationAPI(mockClient as any);
  });

  // ===========================================================================
  // Claim Verification
  // ===========================================================================

  describe('Claim Verification', () => {
    it('should verify a claim', async () => {
      const mockResult = {
        verified: true,
        confidence: 0.95,
        evidence: ['Source 1', 'Source 2'],
        explanation: 'The claim is supported by multiple reliable sources.',
      };
      mockClient.verifyClaim.mockResolvedValue(mockResult);

      const result = await api.verifyClaim({
        claim: 'The Earth is approximately 4.5 billion years old',
        context: 'Geological science',
      });

      expect(mockClient.verifyClaim).toHaveBeenCalledWith({
        claim: 'The Earth is approximately 4.5 billion years old',
        context: 'Geological science',
      });
      expect(result.verified).toBe(true);
      expect(result.confidence).toBe(0.95);
    });

    it('should verify a claim with sources', async () => {
      const mockResult = {
        verified: false,
        confidence: 0.3,
        evidence: [],
        explanation: 'The claim could not be verified.',
      };
      mockClient.verifyClaim.mockResolvedValue(mockResult);

      const result = await api.verifyClaim({
        claim: 'Unverifiable statement',
        sources: ['https://example.com/source1'],
      });

      expect(mockClient.verifyClaim).toHaveBeenCalledWith({
        claim: 'Unverifiable statement',
        sources: ['https://example.com/source1'],
      });
      expect(result.verified).toBe(false);
    });
  });

  // ===========================================================================
  // System Status
  // ===========================================================================

  describe('System Status', () => {
    it('should get verification system status', async () => {
      const mockStatus = {
        status: 'operational',
        available_backends: ['fact_check', 'source_verify', 'consensus'],
        last_check: '2024-01-20T10:00:00Z',
        uptime_percent: 99.9,
      };
      mockClient.getVerificationStatus.mockResolvedValue(mockStatus);

      const result = await api.status();

      expect(mockClient.getVerificationStatus).toHaveBeenCalled();
      expect(result.status).toBe('operational');
      expect(result.available_backends).toContain('fact_check');
    });
  });

  // ===========================================================================
  // Debate Conclusion Verification
  // ===========================================================================

  describe('Debate Conclusion Verification', () => {
    it('should verify debate conclusion', async () => {
      const mockResult = {
        verified: true,
        confidence: 0.88,
        conclusion_quality: 'high',
        supporting_evidence: ['Round 3 consensus', 'Expert agreement'],
      };
      mockClient.verifyDebateConclusion.mockResolvedValue(mockResult);

      const result = await api.verifyConclusion('debate_123');

      expect(mockClient.verifyDebateConclusion).toHaveBeenCalledWith('debate_123', undefined);
      expect(result.verified).toBe(true);
      expect(result.conclusion_quality).toBe('high');
    });

    it('should verify conclusion with options', async () => {
      const mockResult = {
        verified: true,
        confidence: 0.92,
        evidence: [{ type: 'consensus', round: 5 }],
        counterfactuals: ['Alternative path 1', 'Alternative path 2'],
      };
      mockClient.verifyDebateConclusion.mockResolvedValue(mockResult);

      const result = await api.verifyConclusion('debate_456', {
        include_evidence: true,
        include_counterfactuals: true,
        depth: 'deep',
      });

      expect(mockClient.verifyDebateConclusion).toHaveBeenCalledWith('debate_456', {
        include_evidence: true,
        include_counterfactuals: true,
        depth: 'deep',
      });
      expect(result.counterfactuals).toHaveLength(2);
    });

    it('should verify conclusion with shallow depth', async () => {
      const mockResult = { verified: true, confidence: 0.75 };
      mockClient.verifyDebateConclusion.mockResolvedValue(mockResult);

      await api.verifyConclusion('debate_789', { depth: 'shallow' });

      expect(mockClient.verifyDebateConclusion).toHaveBeenCalledWith('debate_789', {
        depth: 'shallow',
      });
    });
  });

  // ===========================================================================
  // Verification Reports
  // ===========================================================================

  describe('Verification Reports', () => {
    it('should get verification report', async () => {
      const mockReport = {
        debate_id: 'debate_123',
        generated_at: '2024-01-20T10:00:00Z',
        summary: 'The debate reached a verified consensus.',
        verification_steps: [
          { step: 1, action: 'Source check', result: 'passed' },
          { step: 2, action: 'Consistency check', result: 'passed' },
          { step: 3, action: 'Expert review', result: 'passed' },
        ],
        overall_score: 0.95,
        recommendations: ['Consider additional peer review for edge cases'],
      };
      mockClient.getVerificationReport.mockResolvedValue(mockReport);

      const result = await api.getReport('debate_123');

      expect(mockClient.getVerificationReport).toHaveBeenCalledWith('debate_123');
      expect(result.debate_id).toBe('debate_123');
      expect(result.verification_steps).toHaveLength(3);
      expect(result.overall_score).toBe(0.95);
    });

    it('should get report with failed steps', async () => {
      const mockReport = {
        debate_id: 'debate_failed',
        generated_at: '2024-01-20T11:00:00Z',
        summary: 'Verification found inconsistencies.',
        verification_steps: [
          { step: 1, action: 'Source check', result: 'passed' },
          { step: 2, action: 'Consistency check', result: 'failed', reason: 'Contradictory claims' },
        ],
        overall_score: 0.45,
        recommendations: ['Review contradictory claims before finalizing'],
      };
      mockClient.getVerificationReport.mockResolvedValue(mockReport);

      const result = await api.getReport('debate_failed');

      expect(result.overall_score).toBe(0.45);
      expect(result.verification_steps[1].result).toBe('failed');
    });
  });
});
