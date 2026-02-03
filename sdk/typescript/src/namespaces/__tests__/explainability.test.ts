/**
 * Explainability Namespace Tests
 *
 * Comprehensive tests for the explainability namespace API including:
 * - Core explainability (get, getFactors)
 * - Counterfactuals
 * - Provenance
 * - Narrative
 * - Batch processing
 * - Evidence and vote analysis
 * - Comparison
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { ExplainabilityAPI } from '../explainability';

interface MockClient {
  getExplanation: Mock;
  getExplanationFactors: Mock;
  getCounterfactuals: Mock;
  generateCounterfactual: Mock;
  getProvenance: Mock;
  getNarrative: Mock;
  createBatchExplanation: Mock;
  getBatchExplanationStatus: Mock;
  getBatchExplanationResults: Mock;
  getDebateEvidenceV1: Mock;
  getDebateVotePivots: Mock;
  compareExplanations: Mock;
}

describe('ExplainabilityAPI Namespace', () => {
  let api: ExplainabilityAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      getExplanation: vi.fn(),
      getExplanationFactors: vi.fn(),
      getCounterfactuals: vi.fn(),
      generateCounterfactual: vi.fn(),
      getProvenance: vi.fn(),
      getNarrative: vi.fn(),
      createBatchExplanation: vi.fn(),
      getBatchExplanationStatus: vi.fn(),
      getBatchExplanationResults: vi.fn(),
      getDebateEvidenceV1: vi.fn(),
      getDebateVotePivots: vi.fn(),
      compareExplanations: vi.fn(),
    };
    api = new ExplainabilityAPI(mockClient as any);
  });

  // ===========================================================================
  // Core Explainability
  // ===========================================================================

  describe('Core Explainability', () => {
    it('should get full explanation', async () => {
      const mockExplanation = {
        debate_id: 'd_123',
        decision: 'Adopt microservices architecture',
        confidence: 0.85,
        factors: [
          { name: 'Scalability', contribution: 0.3, description: 'Better horizontal scaling' },
          { name: 'Complexity', contribution: -0.1, description: 'Increased operational complexity' },
        ],
        counterfactuals: [],
        provenance: { sources: [] },
      };
      mockClient.getExplanation.mockResolvedValue(mockExplanation);

      const result = await api.get('d_123');

      expect(mockClient.getExplanation).toHaveBeenCalledWith('d_123', undefined);
      expect(result.confidence).toBe(0.85);
      expect(result.factors).toHaveLength(2);
    });

    it('should get explanation with options', async () => {
      const mockExplanation = { debate_id: 'd_123' };
      mockClient.getExplanation.mockResolvedValue(mockExplanation);

      await api.get('d_123', {
        include_factors: true,
        include_counterfactuals: true,
        include_provenance: true,
      });

      expect(mockClient.getExplanation).toHaveBeenCalledWith('d_123', {
        include_factors: true,
        include_counterfactuals: true,
        include_provenance: true,
      });
    });

    it('should get factors', async () => {
      const mockFactors = {
        factors: [
          { name: 'Evidence Quality', contribution: 0.25, description: 'Strong supporting data' },
          { name: 'Agent Agreement', contribution: 0.2, description: 'High consensus' },
          { name: 'Risk', contribution: -0.05, description: 'Minor implementation risk' },
        ],
      };
      mockClient.getExplanationFactors.mockResolvedValue(mockFactors);

      const result = await api.getFactors('d_123');

      expect(mockClient.getExplanationFactors).toHaveBeenCalledWith('d_123', undefined);
      expect(result.factors).toHaveLength(3);
    });

    it('should get factors with min contribution', async () => {
      const mockFactors = { factors: [] };
      mockClient.getExplanationFactors.mockResolvedValue(mockFactors);

      await api.getFactors('d_123', { min_contribution: 0.1 });

      expect(mockClient.getExplanationFactors).toHaveBeenCalledWith('d_123', { min_contribution: 0.1 });
    });
  });

  // ===========================================================================
  // Counterfactuals
  // ===========================================================================

  describe('Counterfactuals', () => {
    it('should get existing counterfactuals', async () => {
      const mockCounterfactuals = {
        debate_id: 'd_123',
        counterfactuals: [
          {
            id: 'cf_1',
            hypothesis: 'Without security constraints',
            predicted_outcome: 'Faster implementation but higher risk',
            confidence: 0.75,
          },
        ],
      };
      mockClient.getCounterfactuals.mockResolvedValue(mockCounterfactuals);

      const result = await api.getCounterfactuals('d_123');

      expect(mockClient.getCounterfactuals).toHaveBeenCalledWith('d_123', undefined);
      expect(result.counterfactuals).toHaveLength(1);
    });

    it('should get counterfactuals with max scenarios', async () => {
      const mockCounterfactuals = { counterfactuals: [] };
      mockClient.getCounterfactuals.mockResolvedValue(mockCounterfactuals);

      await api.getCounterfactuals('d_123', { max_scenarios: 5 });

      expect(mockClient.getCounterfactuals).toHaveBeenCalledWith('d_123', { max_scenarios: 5 });
    });

    it('should generate counterfactual', async () => {
      const mockGenerated = {
        id: 'cf_new',
        hypothesis: 'What if we had a larger budget?',
        predicted_outcome: 'Could adopt more comprehensive solution',
        confidence: 0.7,
        affected_factors: ['Cost constraints', 'Implementation timeline'],
      };
      mockClient.generateCounterfactual.mockResolvedValue(mockGenerated);

      const result = await api.generateCounterfactual('d_123', {
        hypothesis: 'What if we had a larger budget?',
        affected_agents: ['gpt-4', 'claude'],
      });

      expect(mockClient.generateCounterfactual).toHaveBeenCalledWith('d_123', {
        hypothesis: 'What if we had a larger budget?',
        affected_agents: ['gpt-4', 'claude'],
      });
      expect(result.predicted_outcome).toContain('comprehensive');
    });
  });

  // ===========================================================================
  // Provenance
  // ===========================================================================

  describe('Provenance', () => {
    it('should get provenance', async () => {
      const mockProvenance = {
        debate_id: 'd_123',
        sources: [
          { type: 'document', id: 'doc_1', title: 'Architecture RFC', relevance: 0.9 },
          { type: 'debate', id: 'd_100', title: 'Previous Discussion', relevance: 0.7 },
        ],
        reasoning_chain: [
          { step: 1, agent: 'claude', action: 'Proposed microservices', source_refs: ['doc_1'] },
          { step: 2, agent: 'gpt-4', action: 'Agreed with caveats', source_refs: ['d_100'] },
        ],
      };
      mockClient.getProvenance.mockResolvedValue(mockProvenance);

      const result = await api.getProvenance('d_123');

      expect(mockClient.getProvenance).toHaveBeenCalledWith('d_123');
      expect(result.sources).toHaveLength(2);
      expect(result.reasoning_chain).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Narrative
  // ===========================================================================

  describe('Narrative', () => {
    it('should get narrative with default format', async () => {
      const mockNarrative = {
        debate_id: 'd_123',
        format: 'detailed',
        narrative: 'The team decided to adopt microservices architecture...',
        word_count: 250,
      };
      mockClient.getNarrative.mockResolvedValue(mockNarrative);

      const result = await api.getNarrative('d_123');

      expect(mockClient.getNarrative).toHaveBeenCalledWith('d_123', undefined);
      expect(result.narrative).toContain('microservices');
    });

    it('should get executive summary', async () => {
      const mockNarrative = {
        format: 'executive_summary',
        narrative: 'Decision: Adopt microservices. Key benefit: Scalability.',
      };
      mockClient.getNarrative.mockResolvedValue(mockNarrative);

      await api.getNarrative('d_123', { format: 'executive_summary' });

      expect(mockClient.getNarrative).toHaveBeenCalledWith('d_123', { format: 'executive_summary' });
    });
  });

  // ===========================================================================
  // Batch Processing
  // ===========================================================================

  describe('Batch Processing', () => {
    it('should create batch explanation', async () => {
      const mockBatch = {
        batch_id: 'batch_123',
        status_url: '/api/v1/explainability/batch/batch_123/status',
      };
      mockClient.createBatchExplanation.mockResolvedValue(mockBatch);

      const result = await api.createBatch({
        debate_ids: ['d_1', 'd_2', 'd_3'],
        include_factors: true,
        include_narrative: true,
      });

      expect(mockClient.createBatchExplanation).toHaveBeenCalledWith({
        debate_ids: ['d_1', 'd_2', 'd_3'],
        include_factors: true,
        include_narrative: true,
      });
      expect(result.batch_id).toBe('batch_123');
    });

    it('should get batch status', async () => {
      const mockStatus = {
        batch_id: 'batch_123',
        status: 'processing',
        progress_pct: 60,
      };
      mockClient.getBatchExplanationStatus.mockResolvedValue(mockStatus);

      const result = await api.getBatchStatus('batch_123');

      expect(mockClient.getBatchExplanationStatus).toHaveBeenCalledWith('batch_123');
      expect(result.progress_pct).toBe(60);
    });

    it('should get batch results', async () => {
      const mockResults = {
        results: [
          { debate_id: 'd_1', status: 'completed', explanation: {} },
          { debate_id: 'd_2', status: 'completed', explanation: {} },
        ],
      };
      mockClient.getBatchExplanationResults.mockResolvedValue(mockResults);

      const result = await api.getBatchResults('batch_123');

      expect(mockClient.getBatchExplanationResults).toHaveBeenCalledWith('batch_123', undefined);
      expect(result.results).toHaveLength(2);
    });

    it('should get batch results with pagination', async () => {
      const mockResults = { results: [] };
      mockClient.getBatchExplanationResults.mockResolvedValue(mockResults);

      await api.getBatchResults('batch_123', { limit: 10, offset: 20 });

      expect(mockClient.getBatchExplanationResults).toHaveBeenCalledWith('batch_123', { limit: 10, offset: 20 });
    });
  });

  // ===========================================================================
  // Evidence
  // ===========================================================================

  describe('Evidence', () => {
    it('should get evidence', async () => {
      const mockEvidence = {
        debate_id: 'd_123',
        evidence_count: 5,
        evidence_quality_score: 0.85,
        evidence: [
          { id: 'ev_1', type: 'document', title: 'RFC', relevance: 0.95 },
          { id: 'ev_2', type: 'citation', title: 'Industry report', relevance: 0.8 },
        ],
      };
      mockClient.getDebateEvidenceV1.mockResolvedValue(mockEvidence);

      const result = await api.getEvidence('d_123');

      expect(mockClient.getDebateEvidenceV1).toHaveBeenCalledWith('d_123', undefined);
      expect(result.evidence_quality_score).toBe(0.85);
    });

    it('should get evidence with filters', async () => {
      const mockEvidence = { evidence: [] };
      mockClient.getDebateEvidenceV1.mockResolvedValue(mockEvidence);

      await api.getEvidence('d_123', { limit: 10, min_relevance: 0.5 });

      expect(mockClient.getDebateEvidenceV1).toHaveBeenCalledWith('d_123', { limit: 10, min_relevance: 0.5 });
    });
  });

  // ===========================================================================
  // Vote Analysis
  // ===========================================================================

  describe('Vote Analysis', () => {
    it('should get vote pivots', async () => {
      const mockPivots = {
        debate_id: 'd_123',
        total_votes: 5,
        pivotal_votes: 2,
        pivot_threshold: 0.5,
        votes: [
          { agent_id: 'claude', vote: 'for', influence_score: 0.8, reasoning: 'Strong evidence' },
          { agent_id: 'gpt-4', vote: 'for', influence_score: 0.6, reasoning: 'Cost effective' },
        ],
      };
      mockClient.getDebateVotePivots.mockResolvedValue(mockPivots);

      const result = await api.getVotePivots('d_123');

      expect(mockClient.getDebateVotePivots).toHaveBeenCalledWith('d_123', undefined);
      expect(result.pivotal_votes).toBe(2);
    });

    it('should get vote pivots with min influence', async () => {
      const mockPivots = { votes: [] };
      mockClient.getDebateVotePivots.mockResolvedValue(mockPivots);

      await api.getVotePivots('d_123', { min_influence: 0.5 });

      expect(mockClient.getDebateVotePivots).toHaveBeenCalledWith('d_123', { min_influence: 0.5 });
    });
  });

  // ===========================================================================
  // Comparison
  // ===========================================================================

  describe('Comparison', () => {
    it('should compare explanations', async () => {
      const mockComparison = {
        debates: {
          'd_1': { confidence: 0.9 },
          'd_2': { confidence: 0.7 },
        },
        comparison: {
          fields: {
            confidence: [
              { debate_id: 'd_1', value: 0.9 },
              { debate_id: 'd_2', value: 0.7 },
            ],
          },
          summary: 'Debate 1 had higher confidence due to stronger evidence.',
        },
      };
      mockClient.compareExplanations.mockResolvedValue(mockComparison);

      const result = await api.compare({
        debate_ids: ['d_1', 'd_2'],
        compare_fields: ['confidence', 'evidence_quality'],
      });

      expect(mockClient.compareExplanations).toHaveBeenCalledWith({
        debate_ids: ['d_1', 'd_2'],
        compare_fields: ['confidence', 'evidence_quality'],
      });
      expect(result.comparison.summary).toContain('higher confidence');
    });
  });

  // ===========================================================================
  // Helper Methods
  // ===========================================================================

  describe('Helper Methods', () => {
    it('should get top positive factors', () => {
      const result = {
        debate_id: 'd_123',
        factors: [
          { name: 'Factor A', contribution: 0.3 },
          { name: 'Factor B', contribution: -0.1 },
          { name: 'Factor C', contribution: 0.2 },
          { name: 'Factor D', contribution: 0.1 },
        ],
      } as any;

      const topPositive = api.getTopPositiveFactors(result, 2);

      expect(topPositive).toHaveLength(2);
      expect(topPositive[0].name).toBe('Factor A');
      expect(topPositive[1].name).toBe('Factor C');
    });

    it('should get top negative factors', () => {
      const result = {
        debate_id: 'd_123',
        factors: [
          { name: 'Factor A', contribution: 0.3 },
          { name: 'Factor B', contribution: -0.2 },
          { name: 'Factor C', contribution: -0.1 },
          { name: 'Factor D', contribution: -0.3 },
        ],
      } as any;

      const topNegative = api.getTopNegativeFactors(result, 2);

      expect(topNegative).toHaveLength(2);
      expect(topNegative[0].name).toBe('Factor D');
      expect(topNegative[1].name).toBe('Factor B');
    });

    it('should handle missing factors', () => {
      const result = { debate_id: 'd_123' } as any;

      expect(api.getTopPositiveFactors(result, 3)).toHaveLength(0);
      expect(api.getTopNegativeFactors(result, 3)).toHaveLength(0);
    });
  });
});
