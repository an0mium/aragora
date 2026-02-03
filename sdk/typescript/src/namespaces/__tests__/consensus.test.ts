/**
 * Consensus Namespace Tests
 *
 * Comprehensive tests for the ConsensusAPI namespace class.
 * Tests all methods including:
 * - Finding similar debates
 * - Settled topics
 * - Consensus statistics
 * - Dissent tracking
 * - Contrarian views
 * - Risk warnings
 * - Domain history
 * - Demo seeding
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { ConsensusAPI } from '../consensus';

interface MockClient {
  request: Mock;
}

describe('ConsensusAPI', () => {
  let api: ConsensusAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    vi.clearAllMocks();
    mockClient = {
      request: vi.fn(),
    };
    api = new ConsensusAPI(mockClient as any);
  });

  // ===========================================================================
  // Find Similar Debates
  // ===========================================================================

  describe('Find Similar Debates', () => {
    it('should find similar debates for a topic', async () => {
      const mockSimilar = {
        query: 'microservices',
        similar: [
          {
            topic: 'Should we adopt microservices?',
            conclusion: 'Yes, for large teams',
            strength: 'strong',
            confidence: 0.92,
            similarity: 0.95,
            agents: ['claude', 'gpt4'],
            dissent_count: 1,
            timestamp: '2024-01-15T00:00:00Z',
          },
        ],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockSimilar);

      const result = await api.findSimilar({ topic: 'microservices', limit: 5 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/similar', {
        params: { topic: 'microservices', limit: 5 },
      });
      expect(result.similar).toHaveLength(1);
      expect(result.similar[0].confidence).toBe(0.92);
    });

    it('should use default limit when not specified', async () => {
      const mockSimilar = { query: 'caching', similar: [], count: 0 };
      mockClient.request.mockResolvedValue(mockSimilar);

      await api.findSimilar({ topic: 'caching' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/similar', {
        params: { topic: 'caching', limit: 5 },
      });
    });
  });

  // ===========================================================================
  // Settled Topics
  // ===========================================================================

  describe('Settled Topics', () => {
    it('should get settled topics with defaults', async () => {
      const mockSettled = {
        min_confidence: 0.8,
        topics: [
          {
            topic: 'API design best practices',
            conclusion: 'Use REST with OpenAPI spec',
            confidence: 0.95,
            strength: 'strong',
            timestamp: '2024-01-10T00:00:00Z',
          },
        ],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockSettled);

      const result = await api.getSettled();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/settled', {
        params: { min_confidence: 0.8, limit: 20 },
      });
      expect(result.topics).toHaveLength(1);
      expect(result.topics[0].confidence).toBe(0.95);
    });

    it('should get settled topics with custom confidence', async () => {
      const mockSettled = { min_confidence: 0.9, topics: [], count: 0 };
      mockClient.request.mockResolvedValue(mockSettled);

      await api.getSettled({ minConfidence: 0.9, limit: 10 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/settled', {
        params: { min_confidence: 0.9, limit: 10 },
      });
    });
  });

  // ===========================================================================
  // Consensus Statistics
  // ===========================================================================

  describe('Consensus Statistics', () => {
    it('should get consensus statistics', async () => {
      const mockStats = {
        total_topics: 150,
        high_confidence_count: 80,
        domains: ['technology', 'business', 'science'],
        avg_confidence: 0.82,
        total_dissents: 45,
        by_strength: { strong: 60, moderate: 50, weak: 40 },
        by_domain: { technology: 80, business: 40, science: 30 },
      };
      mockClient.request.mockResolvedValue(mockStats);

      const result = await api.getStats();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/stats');
      expect(result.total_topics).toBe(150);
      expect(result.avg_confidence).toBe(0.82);
      expect(result.domains).toHaveLength(3);
    });
  });

  // ===========================================================================
  // Dissent Tracking
  // ===========================================================================

  describe('Dissent Tracking', () => {
    it('should get dissents with defaults', async () => {
      const mockDissents = {
        dissents: [
          {
            id: 'dis_1',
            topic: 'Microservices adoption',
            agent: 'mistral',
            position: 'Monolith is better for small teams',
            reasoning: 'Operational complexity outweighs benefits',
            timestamp: '2024-01-20T00:00:00Z',
          },
        ],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockDissents);

      const result = await api.getDissents();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/dissents', {
        params: { topic: undefined, domain: undefined, limit: 10 },
      });
      expect(result.dissents).toHaveLength(1);
      expect(result.dissents[0].agent).toBe('mistral');
    });

    it('should get dissents filtered by topic', async () => {
      const mockDissents = { dissents: [], count: 0 };
      mockClient.request.mockResolvedValue(mockDissents);

      await api.getDissents({ topic: 'caching', domain: 'technology', limit: 5 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/dissents', {
        params: { topic: 'caching', domain: 'technology', limit: 5 },
      });
    });
  });

  // ===========================================================================
  // Contrarian Views
  // ===========================================================================

  describe('Contrarian Views', () => {
    it('should get contrarian views with defaults', async () => {
      const mockContrarian = {
        views: [
          {
            id: 'cv_1',
            topic: 'NoSQL databases',
            agent: 'grok',
            perspective: 'Relational databases are still superior for most use cases',
            confidence: 0.78,
            timestamp: '2024-01-18T00:00:00Z',
          },
        ],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockContrarian);

      const result = await api.getContrarian();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/contrarian-views', {
        params: { topic: undefined, domain: undefined, limit: 10 },
      });
      expect(result.views).toHaveLength(1);
      expect(result.views[0].confidence).toBe(0.78);
    });

    it('should get contrarian views filtered by domain', async () => {
      const mockContrarian = { views: [], count: 0 };
      mockClient.request.mockResolvedValue(mockContrarian);

      await api.getContrarian({ domain: 'architecture', limit: 20 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/contrarian-views', {
        params: { topic: undefined, domain: 'architecture', limit: 20 },
      });
    });
  });

  // ===========================================================================
  // Risk Warnings
  // ===========================================================================

  describe('Risk Warnings', () => {
    it('should get risk warnings with defaults', async () => {
      const mockWarnings = {
        warnings: [
          {
            id: 'rw_1',
            topic: 'Serverless migration',
            warning: 'Cold start latency may impact user experience',
            severity: 'medium' as const,
            source_agent: 'claude',
            timestamp: '2024-01-19T00:00:00Z',
          },
        ],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockWarnings);

      const result = await api.getRiskWarnings();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/risk-warnings', {
        params: { topic: undefined, domain: undefined, limit: 10 },
      });
      expect(result.warnings).toHaveLength(1);
      expect(result.warnings[0].severity).toBe('medium');
    });

    it('should get risk warnings filtered by topic', async () => {
      const mockWarnings = { warnings: [], count: 0 };
      mockClient.request.mockResolvedValue(mockWarnings);

      await api.getRiskWarnings({ topic: 'security', limit: 5 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/risk-warnings', {
        params: { topic: 'security', domain: undefined, limit: 5 },
      });
    });
  });

  // ===========================================================================
  // Domain History
  // ===========================================================================

  describe('Domain History', () => {
    it('should get domain history with default limit', async () => {
      const mockHistory = {
        domain: 'technology',
        history: [
          {
            topic: 'Microservices vs Monolith',
            conclusion: 'Depends on team size',
            confidence: 0.88,
            timestamp: '2024-01-15T00:00:00Z',
          },
        ],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockHistory);

      const result = await api.getDomainHistory('technology');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/consensus/domain/technology',
        { params: { limit: 50 } }
      );
      expect(result.domain).toBe('technology');
      expect(result.history).toHaveLength(1);
    });

    it('should get domain history with custom limit', async () => {
      const mockHistory = { domain: 'business', history: [], count: 0 };
      mockClient.request.mockResolvedValue(mockHistory);

      await api.getDomainHistory('business', 10);

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/consensus/domain/business',
        { params: { limit: 10 } }
      );
    });

    it('should encode domain name in URL', async () => {
      const mockHistory = { domain: 'AI/ML', history: [], count: 0 };
      mockClient.request.mockResolvedValue(mockHistory);

      await api.getDomainHistory('AI/ML');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/consensus/domain/AI%2FML',
        { params: { limit: 50 } }
      );
    });
  });

  // ===========================================================================
  // Seed Demo
  // ===========================================================================

  describe('Seed Demo', () => {
    it('should seed demo consensus data', async () => {
      const mockResult = { success: true, message: 'Demo data seeded successfully' };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.seedDemo();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/consensus/seed-demo');
      expect(result.success).toBe(true);
      expect(result.message).toContain('seeded');
    });
  });
});
