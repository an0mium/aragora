/**
 * Verification Namespace Test Suite
 *
 * Tests for the verification namespace including:
 * - Claim verification (independent claims)
 * - Debate conclusion verification
 * - Verification reports
 * - System verification status
 * - Backend selection (Z3, Lean, auto)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AragoraClient, createClient } from '../client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Verification Namespace', () => {
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
  // Verify Claim
  // ===========================================================================

  describe('Verify Claim', () => {
    it('should verify a simple claim', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'QED: Claim holds under given axioms',
        duration_ms: 150,
        trace_id: 'trace-001',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'If A implies B, and A is true, then B is true',
        context: 'Propositional logic',
      });

      expect(result.verified).toBe(true);
      expect(result.backend).toBe('z3');
      expect(result.proof).toBeDefined();
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/verification'),
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('should verify a claim with Z3 backend', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'Satisfiability check passed',
        assumptions: ['x > 0', 'y > 0'],
        duration_ms: 200,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'x + y > 0',
        context: 'Given positive integers x and y',
        backend: 'z3',
      });

      expect(result.verified).toBe(true);
      expect(result.backend).toBe('z3');
      expect(result.assumptions).toContain('x > 0');
    });

    it('should verify a claim with Lean backend', async () => {
      const mockResult = {
        verified: true,
        backend: 'lean',
        proof: 'theorem verified: P -> Q -> (P /\\ Q)',
        duration_ms: 500,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'P implies Q implies (P and Q)',
        backend: 'lean',
      });

      expect(result.verified).toBe(true);
      expect(result.backend).toBe('lean');
    });

    it('should use auto backend selection', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'Auto-selected Z3 for arithmetic reasoning',
        duration_ms: 180,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'For all n > 0, n * 2 > n',
        backend: 'auto',
      });

      expect(result.verified).toBe(true);
      expect(['z3', 'lean', 'auto']).toContain(result.backend);
    });

    it('should handle unverified claims with counterexample', async () => {
      const mockResult = {
        verified: false,
        backend: 'z3',
        counterexample: 'x = -1 provides a counterexample',
        duration_ms: 100,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'x * x is always greater than x',
      });

      expect(result.verified).toBe(false);
      expect(result.counterexample).toBeDefined();
      expect(result.counterexample).toContain('-1');
    });

    it('should verify with timeout', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'Verified within timeout',
        duration_ms: 4500,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'Complex mathematical proof',
        timeout_seconds: 10,
      });

      expect(result.verified).toBe(true);
      expect(result.duration_ms).toBeLessThan(10000);
    });

    it('should include assumptions in result', async () => {
      const mockResult = {
        verified: true,
        backend: 'lean',
        proof: 'Proof complete',
        assumptions: [
          'n is a natural number',
          'n > 0',
          'Peano axioms hold',
        ],
        duration_ms: 300,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'There exists a successor for every natural number',
        context: 'Peano arithmetic',
      });

      expect(result.verified).toBe(true);
      expect(result.assumptions).toHaveLength(3);
    });
  });

  // ===========================================================================
  // Verification Status
  // ===========================================================================

  describe('Verification Status', () => {
    it('should get verification system status', async () => {
      const mockStatus = {
        z3_available: true,
        lean_available: true,
        default_backend: 'z3',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.verification.status();

      expect(result.z3_available).toBe(true);
      expect(result.lean_available).toBe(true);
      expect(result.default_backend).toBe('z3');
    });

    it('should handle partial backend availability', async () => {
      const mockStatus = {
        z3_available: true,
        lean_available: false,
        default_backend: 'z3',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.verification.status();

      expect(result.z3_available).toBe(true);
      expect(result.lean_available).toBe(false);
      expect(result.default_backend).toBe('z3');
    });

    it('should handle no backends available', async () => {
      const mockStatus = {
        z3_available: false,
        lean_available: false,
        default_backend: 'auto',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.verification.status();

      expect(result.z3_available).toBe(false);
      expect(result.lean_available).toBe(false);
    });
  });

  // ===========================================================================
  // Verify Debate Conclusion
  // ===========================================================================

  describe('Verify Debate Conclusion', () => {
    it('should verify a debate conclusion', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'Consensus conclusion verified against evidence',
        duration_ms: 350,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyConclusion('debate-123');

      expect(result.verified).toBe(true);
      expect(result.proof).toBeDefined();
    });

    it('should verify with evidence inclusion', async () => {
      const mockResult = {
        verified: true,
        backend: 'lean',
        proof: 'Full evidence chain verified',
        assumptions: ['Evidence A supports conclusion', 'Evidence B is consistent'],
        duration_ms: 800,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyConclusion('debate-123', {
        include_evidence: true,
      });

      expect(result.verified).toBe(true);
      expect(result.assumptions).toBeDefined();
    });

    it('should verify with counterfactual analysis', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'Conclusion robust against counterfactuals',
        duration_ms: 600,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyConclusion('debate-123', {
        include_counterfactuals: true,
      });

      expect(result.verified).toBe(true);
    });

    it('should verify with shallow depth', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'Quick verification passed',
        duration_ms: 50,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyConclusion('debate-123', {
        depth: 'shallow',
      });

      expect(result.verified).toBe(true);
      expect(result.duration_ms).toBeLessThan(100);
    });

    it('should verify with deep depth', async () => {
      const mockResult = {
        verified: true,
        backend: 'lean',
        proof: 'Deep verification with full formal proof',
        assumptions: [
          'All agent claims verified',
          'Evidence chain complete',
          'No logical contradictions',
        ],
        duration_ms: 2500,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyConclusion('debate-123', {
        depth: 'deep',
        include_evidence: true,
        include_counterfactuals: true,
      });

      expect(result.verified).toBe(true);
      expect(result.assumptions).toHaveLength(3);
    });

    it('should handle unverified conclusion', async () => {
      const mockResult = {
        verified: false,
        backend: 'z3',
        counterexample: 'Agent claim contradicts evidence',
        duration_ms: 200,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyConclusion('debate-flawed');

      expect(result.verified).toBe(false);
      expect(result.counterexample).toBeDefined();
    });
  });

  // ===========================================================================
  // Verification Report
  // ===========================================================================

  describe('Verification Report', () => {
    it('should get verification report for a debate', async () => {
      const mockReport = {
        debate_id: 'debate-123',
        verified: true,
        verification_method: 'formal',
        claims_verified: 15,
        claims_failed: 0,
        claims_skipped: 2,
        overall_confidence: 0.95,
        backend_used: 'z3',
        duration_ms: 1200,
        created_at: '2024-01-15T10:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockReport)),
      });

      const result = await client.verification.getReport('debate-123');

      expect(result.debate_id).toBe('debate-123');
      expect(result.verified).toBe(true);
      expect(result.claims_verified).toBe(15);
      expect(result.claims_failed).toBe(0);
    });

    it('should get report with partial verification', async () => {
      const mockReport = {
        debate_id: 'debate-456',
        verified: false,
        verification_method: 'formal',
        claims_verified: 10,
        claims_failed: 3,
        claims_skipped: 1,
        overall_confidence: 0.72,
        backend_used: 'lean',
        failed_claims: [
          { claim: 'Claim A', reason: 'Counterexample found' },
          { claim: 'Claim B', reason: 'Timeout exceeded' },
          { claim: 'Claim C', reason: 'Unsupported logic' },
        ],
        duration_ms: 3500,
        created_at: '2024-01-15T11:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockReport)),
      });

      const result = await client.verification.getReport('debate-456');

      expect(result.debate_id).toBe('debate-456');
      expect(result.verified).toBe(false);
      expect(result.claims_failed).toBe(3);
    });

    it('should include detailed claim breakdown', async () => {
      const mockReport = {
        debate_id: 'debate-789',
        verified: true,
        verification_method: 'hybrid',
        claims_verified: 8,
        claims_failed: 0,
        claims_skipped: 0,
        overall_confidence: 0.98,
        backend_used: 'auto',
        claim_details: [
          { claim: 'Market growth projection', verified: true, backend: 'z3' },
          { claim: 'Cost reduction estimate', verified: true, backend: 'z3' },
          { claim: 'Risk assessment', verified: true, backend: 'lean' },
        ],
        duration_ms: 2000,
        created_at: '2024-01-15T12:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockReport)),
      });

      const result = await client.verification.getReport('debate-789');

      expect(result.debate_id).toBe('debate-789');
      expect(result.verified).toBe(true);
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('Error Handling', () => {
    it('should handle claim syntax error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid claim syntax',
          code: 'INVALID_VALUE',
        }),
      });

      await expect(
        client.verification.verifyClaim({ claim: '>>invalid<<' })
      ).rejects.toThrow('Invalid claim syntax');
    });

    it('should handle backend unavailable', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: () => Promise.resolve({
          error: 'Lean backend unavailable',
          code: 'SERVICE_UNAVAILABLE',
        }),
      });

      await expect(
        client.verification.verifyClaim({
          claim: 'Test claim',
          backend: 'lean',
        })
      ).rejects.toThrow('Lean backend unavailable');
    });

    it('should handle verification timeout', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 408,
        json: () => Promise.resolve({
          error: 'Verification timeout exceeded',
          code: 'AGENT_TIMEOUT',
        }),
      });

      await expect(
        client.verification.verifyClaim({
          claim: 'Very complex claim',
          timeout_seconds: 5,
        })
      ).rejects.toThrow('Verification timeout');
    });

    it('should handle debate not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'Debate not found',
          code: 'NOT_FOUND',
        }),
      });

      await expect(client.verification.verifyConclusion('nonexistent'))
        .rejects.toThrow('Debate not found');
    });

    it('should handle report not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'Verification report not found',
          code: 'NOT_FOUND',
        }),
      });

      await expect(client.verification.getReport('no-report'))
        .rejects.toThrow('Verification report not found');
    });

    it('should handle unauthorized access', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Invalid API key',
          code: 'INVALID_TOKEN',
        }),
      });

      await expect(client.verification.status())
        .rejects.toThrow('Invalid API key');
    });

    it('should handle internal error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({
          error: 'Z3 solver crashed',
          code: 'INTERNAL_ERROR',
        }),
      });

      await expect(
        client.verification.verifyClaim({ claim: 'Test' })
      ).rejects.toThrow('Z3 solver crashed');
    });
  });

  // ===========================================================================
  // Type Safety
  // ===========================================================================

  describe('Type Safety', () => {
    it('should return properly typed verification result', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'QED',
        duration_ms: 100,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'Test claim',
      });

      expect(typeof result.verified).toBe('boolean');
      expect(['z3', 'lean', 'auto']).toContain(result.backend);
      expect(typeof result.duration_ms).toBe('number');
      expect(result.duration_ms).toBeGreaterThan(0);
    });

    it('should return properly typed verification status', async () => {
      const mockStatus = {
        z3_available: true,
        lean_available: false,
        default_backend: 'z3',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.verification.status();

      expect(typeof result.z3_available).toBe('boolean');
      expect(typeof result.lean_available).toBe('boolean');
      expect(['z3', 'lean', 'auto']).toContain(result.default_backend);
    });

    it('should return properly typed verification report', async () => {
      const mockReport = {
        debate_id: 'debate-123',
        verified: true,
        verification_method: 'formal',
        claims_verified: 10,
        claims_failed: 0,
        claims_skipped: 0,
        overall_confidence: 0.95,
        backend_used: 'z3',
        duration_ms: 1000,
        created_at: '2024-01-15T10:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockReport)),
      });

      const result = await client.verification.getReport('debate-123');

      expect(typeof result.debate_id).toBe('string');
      expect(typeof result.verified).toBe('boolean');
      expect(typeof result.claims_verified).toBe('number');
      expect(typeof result.claims_failed).toBe('number');
    });

    it('should handle optional fields in verification result', async () => {
      const mockResult = {
        verified: false,
        backend: 'z3',
        counterexample: 'x = -1',
        duration_ms: 50,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'x > 0',
      });

      expect(result.verified).toBe(false);
      expect(result.proof).toBeUndefined();
      expect(result.counterexample).toBe('x = -1');
    });

    it('should handle assumptions array in result', async () => {
      const mockResult = {
        verified: true,
        backend: 'lean',
        proof: 'Complete',
        assumptions: ['Axiom 1', 'Axiom 2', 'Axiom 3'],
        duration_ms: 200,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'Test with assumptions',
      });

      expect(result.verified).toBe(true);
      expect(Array.isArray(result.assumptions)).toBe(true);
      expect(result.assumptions).toHaveLength(3);
      result.assumptions?.forEach((assumption: string) => {
        expect(typeof assumption).toBe('string');
      });
    });
  });

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe('Edge Cases', () => {
    it('should handle empty claim', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Claim cannot be empty',
          code: 'MISSING_FIELD',
          field: 'claim',
        }),
      });

      await expect(client.verification.verifyClaim({ claim: '' }))
        .rejects.toThrow('Claim cannot be empty');
    });

    it('should handle very long claim', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'Verified after decomposition',
        duration_ms: 5000,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const longClaim = 'A'.repeat(10000);
      const result = await client.verification.verifyClaim({ claim: longClaim });

      expect(result.verified).toBe(true);
    });

    it('should handle special characters in claim', async () => {
      const mockResult = {
        verified: true,
        backend: 'lean',
        proof: 'Special characters handled',
        duration_ms: 150,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'forall x. exists y. P(x) => Q(y)',
      });

      expect(result.verified).toBe(true);
    });

    it('should handle unicode in claim', async () => {
      const mockResult = {
        verified: true,
        backend: 'lean',
        proof: 'Unicode claim verified',
        duration_ms: 100,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'If price rises, demand falls',
      });

      expect(result.verified).toBe(true);
    });

    it('should handle zero duration in result', async () => {
      const mockResult = {
        verified: true,
        backend: 'z3',
        proof: 'Trivially true',
        duration_ms: 0,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.verification.verifyClaim({
        claim: 'true',
      });

      expect(result.verified).toBe(true);
      expect(result.duration_ms).toBe(0);
    });

    it('should handle negative timeout gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Timeout must be positive',
          code: 'INVALID_VALUE',
        }),
      });

      await expect(
        client.verification.verifyClaim({
          claim: 'Test',
          timeout_seconds: -1,
        })
      ).rejects.toThrow('Timeout must be positive');
    });
  });
});
