/**
 * Consensus Namespace Tests
 *
 * Comprehensive tests for the consensus namespace API including:
 * - Consensus detection
 * - Settled topics
 * - Dissent tracking
 * - Contrarian views
 * - Confidence analysis
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { ConsensusNamespace } from '../consensus';

interface MockClient {
  request: Mock;
}

describe('ConsensusNamespace', () => {
  let api: ConsensusNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new ConsensusNamespace(mockClient as any);
  });

  // ===========================================================================
  // Consensus Detection
  // ===========================================================================

  describe('Consensus Detection', () => {
    it('should check consensus for debate', async () => {
      const mockConsensus = {
        debate_id: 'd_123',
        has_consensus: true,
        confidence: 0.92,
        consensus_type: 'majority',
        position: 'We should adopt microservices for the backend',
        supporting_agents: ['claude', 'gpt4', 'gemini'],
        dissenting_agents: ['mistral'],
        vote_distribution: {
          for: 3,
          against: 1,
          abstain: 0,
        },
      };
      mockClient.request.mockResolvedValue(mockConsensus);

      const result = await api.check('d_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/d_123');
      expect(result.has_consensus).toBe(true);
      expect(result.confidence).toBe(0.92);
    });

    it('should detect no consensus', async () => {
      const mockConsensus = {
        debate_id: 'd_124',
        has_consensus: false,
        confidence: 0.45,
        positions: [
          { position: 'Option A', supporters: ['claude', 'gpt4'] },
          { position: 'Option B', supporters: ['gemini', 'mistral'] },
        ],
      };
      mockClient.request.mockResolvedValue(mockConsensus);

      const result = await api.check('d_124');

      expect(result.has_consensus).toBe(false);
      expect(result.positions).toHaveLength(2);
    });

    it('should get consensus with options', async () => {
      const mockConsensus = { has_consensus: true };
      mockClient.request.mockResolvedValue(mockConsensus);

      await api.check('d_123', { include_reasoning: true, threshold: 0.8 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/d_123', {
        params: { include_reasoning: true, threshold: 0.8 },
      });
    });
  });

  // ===========================================================================
  // Settled Topics
  // ===========================================================================

  describe('Settled Topics', () => {
    it('should list settled topics', async () => {
      const mockSettled = {
        topics: [
          {
            topic_id: 't_1',
            topic: 'Best practices for API design',
            position: 'Use REST with OpenAPI spec',
            confidence: 0.95,
            debates_count: 15,
            last_challenged: '2024-01-10T00:00:00Z',
            status: 'settled',
          },
          {
            topic_id: 't_2',
            topic: 'Database choice for high-write workloads',
            position: 'Use PostgreSQL with proper indexing',
            confidence: 0.88,
            debates_count: 8,
            status: 'settled',
          },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockSettled);

      const result = await api.listSettled();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/settled', {
        params: undefined,
      });
      expect(result.topics).toHaveLength(2);
    });

    it('should list settled by domain', async () => {
      const mockSettled = { topics: [], total: 0 };
      mockClient.request.mockResolvedValue(mockSettled);

      await api.listSettled({ domain: 'technology', min_confidence: 0.9 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/settled', {
        params: { domain: 'technology', min_confidence: 0.9 },
      });
    });

    it('should mark topic as settled', async () => {
      const mockResult = {
        topic_id: 't_new',
        topic: 'Caching strategy',
        position: 'Use Redis for session cache',
        marked_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.markSettled({
        topic: 'Caching strategy',
        position: 'Use Redis for session cache',
        debate_ids: ['d_1', 'd_2'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/consensus/settled', {
        json: {
          topic: 'Caching strategy',
          position: 'Use Redis for session cache',
          debate_ids: ['d_1', 'd_2'],
        },
      });
      expect(result.topic_id).toBe('t_new');
    });

    it('should challenge settled topic', async () => {
      const mockResult = {
        challenge_id: 'ch_1',
        topic_id: 't_1',
        status: 'pending_debate',
        challenger: 'user@example.com',
        reason: 'New evidence suggests alternatives',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.challenge('t_1', {
        reason: 'New evidence suggests alternatives',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/consensus/settled/t_1/challenge', {
        json: { reason: 'New evidence suggests alternatives' },
      });
      expect(result.status).toBe('pending_debate');
    });
  });

  // ===========================================================================
  // Dissent Tracking
  // ===========================================================================

  describe('Dissent Tracking', () => {
    it('should get dissents for debate', async () => {
      const mockDissents = {
        debate_id: 'd_123',
        dissents: [
          {
            agent: 'mistral',
            position: 'Monolith is better for small teams',
            reasoning: 'Operational complexity outweighs benefits',
            confidence: 0.75,
            evidence: ['study_1', 'case_study_2'],
          },
        ],
        total_dissents: 1,
      };
      mockClient.request.mockResolvedValue(mockDissents);

      const result = await api.getDissents('d_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/d_123/dissents');
      expect(result.dissents).toHaveLength(1);
      expect(result.dissents[0].agent).toBe('mistral');
    });

    it('should list all dissents', async () => {
      const mockDissents = {
        dissents: [
          { debate_id: 'd_1', agent: 'mistral', position: 'Alternative view 1' },
          { debate_id: 'd_2', agent: 'grok', position: 'Alternative view 2' },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockDissents);

      const result = await api.listDissents();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/dissents', {
        params: undefined,
      });
      expect(result.dissents).toHaveLength(2);
    });

    it('should list dissents by agent', async () => {
      const mockDissents = { dissents: [], total: 0 };
      mockClient.request.mockResolvedValue(mockDissents);

      await api.listDissents({ agent: 'mistral' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/dissents', {
        params: { agent: 'mistral' },
      });
    });
  });

  // ===========================================================================
  // Contrarian Views
  // ===========================================================================

  describe('Contrarian Views', () => {
    it('should get contrarian analysis', async () => {
      const mockAnalysis = {
        debate_id: 'd_123',
        contrarian_views: [
          {
            view: 'Event sourcing introduces unnecessary complexity',
            agent: 'grok',
            strength: 0.72,
            counter_evidence: ['Large event stores are hard to manage'],
            addressed_by: [],
          },
        ],
        total_views: 1,
      };
      mockClient.request.mockResolvedValue(mockAnalysis);

      const result = await api.getContrarian('d_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/d_123/contrarian');
      expect(result.contrarian_views).toHaveLength(1);
    });

    it('should list strongest contrarian views', async () => {
      const mockViews = {
        views: [
          { topic: 'Microservices', view: 'Monolith better for most', strength: 0.85 },
          { topic: 'NoSQL', view: 'Relational still superior', strength: 0.78 },
        ],
      };
      mockClient.request.mockResolvedValue(mockViews);

      const result = await api.listContrarianViews({ min_strength: 0.7, limit: 10 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/contrarian', {
        params: { min_strength: 0.7, limit: 10 },
      });
      expect(result.views).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Confidence Analysis
  // ===========================================================================

  describe('Confidence Analysis', () => {
    it('should get confidence breakdown', async () => {
      const mockConfidence = {
        debate_id: 'd_123',
        overall_confidence: 0.88,
        breakdown: {
          agent_agreement: 0.92,
          evidence_quality: 0.85,
          argument_strength: 0.87,
          historical_accuracy: 0.90,
        },
        factors: [
          { factor: 'Strong supporting evidence', impact: 0.15 },
          { factor: 'High agent expertise in domain', impact: 0.12 },
          { factor: 'Minor dissent', impact: -0.05 },
        ],
      };
      mockClient.request.mockResolvedValue(mockConfidence);

      const result = await api.getConfidence('d_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/d_123/confidence');
      expect(result.overall_confidence).toBe(0.88);
      expect(result.breakdown.agent_agreement).toBe(0.92);
    });

    it('should get confidence trend', async () => {
      const mockTrend = {
        debate_id: 'd_123',
        trend: [
          { round: 1, confidence: 0.5 },
          { round: 2, confidence: 0.65 },
          { round: 3, confidence: 0.8 },
          { round: 4, confidence: 0.88 },
        ],
        final_confidence: 0.88,
        convergence_rate: 0.127,
      };
      mockClient.request.mockResolvedValue(mockTrend);

      const result = await api.getConfidenceTrend('d_123');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/consensus/d_123/confidence/trend'
      );
      expect(result.trend).toHaveLength(4);
      expect(result.convergence_rate).toBe(0.127);
    });
  });

  // ===========================================================================
  // Consensus Statistics
  // ===========================================================================

  describe('Consensus Statistics', () => {
    it('should get consensus statistics', async () => {
      const mockStats = {
        total_debates_analyzed: 1500,
        consensus_rate: 0.78,
        average_confidence: 0.82,
        by_type: {
          unanimous: 350,
          majority: 820,
          no_consensus: 330,
        },
        by_domain: {
          technology: { debates: 500, consensus_rate: 0.82 },
          business: { debates: 400, consensus_rate: 0.75 },
          science: { debates: 300, consensus_rate: 0.80 },
        },
        average_rounds_to_consensus: 3.2,
      };
      mockClient.request.mockResolvedValue(mockStats);

      const result = await api.getStats();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/stats');
      expect(result.consensus_rate).toBe(0.78);
      expect(result.by_type.unanimous).toBe(350);
    });

    it('should get stats with date filter', async () => {
      const mockStats = { consensus_rate: 0.80 };
      mockClient.request.mockResolvedValue(mockStats);

      await api.getStats({
        start_date: '2024-01-01',
        end_date: '2024-01-31',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/stats', {
        params: { start_date: '2024-01-01', end_date: '2024-01-31' },
      });
    });
  });

  // ===========================================================================
  // Hollow Consensus Detection
  // ===========================================================================

  describe('Hollow Consensus Detection', () => {
    it('should check for hollow consensus', async () => {
      const mockResult = {
        debate_id: 'd_123',
        is_hollow: false,
        indicators: {
          surface_agreement: 0.95,
          deep_alignment: 0.88,
          reasoning_consistency: 0.85,
        },
        risk_score: 0.15,
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.checkHollow('d_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/consensus/d_123/hollow');
      expect(result.is_hollow).toBe(false);
      expect(result.risk_score).toBe(0.15);
    });

    it('should detect hollow consensus', async () => {
      const mockResult = {
        debate_id: 'd_124',
        is_hollow: true,
        indicators: {
          surface_agreement: 0.90,
          deep_alignment: 0.45,
          reasoning_consistency: 0.50,
        },
        risk_score: 0.75,
        warnings: [
          'Agents agree on conclusion but reasoning diverges significantly',
          'Evidence cited by agents does not overlap',
        ],
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.checkHollow('d_124');

      expect(result.is_hollow).toBe(true);
      expect(result.warnings).toHaveLength(2);
    });
  });
});
