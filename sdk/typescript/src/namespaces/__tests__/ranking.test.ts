/**
 * Ranking Namespace Tests
 *
 * Comprehensive tests for the ranking namespace API including:
 * - Listing rankings
 * - Getting specific agent ranking
 * - Ranking statistics
 * - Domain filtering
 * - Convenience methods
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { RankingNamespace } from '../ranking';

interface MockClient {
  request: Mock;
}

describe('RankingNamespace', () => {
  let api: RankingNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new RankingNamespace(mockClient as any);
  });

  // ===========================================================================
  // Listing Rankings
  // ===========================================================================

  describe('Listing Rankings', () => {
    it('should list agent rankings', async () => {
      const mockRankings = {
        rankings: [
          {
            rank: 1,
            agent: 'claude',
            elo: 1850,
            wins: 150,
            losses: 30,
            draws: 20,
            total_debates: 200,
            win_rate: 0.75,
            streak: 5,
            streak_type: 'win',
            trend: 'up',
          },
          {
            rank: 2,
            agent: 'gpt4',
            elo: 1800,
            wins: 140,
            losses: 40,
            draws: 20,
            total_debates: 200,
            win_rate: 0.70,
            streak: 2,
            streak_type: 'win',
            trend: 'stable',
          },
        ],
      };
      mockClient.request.mockResolvedValue(mockRankings);

      const result = await api.list();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/rankings', {
        params: undefined,
      });
      expect(result).toHaveLength(2);
      expect(result[0].agent).toBe('claude');
    });

    it('should list rankings with options', async () => {
      const mockRankings = { rankings: [{ rank: 1, agent: 'claude', elo: 1850 }] };
      mockClient.request.mockResolvedValue(mockRankings);

      const result = await api.list({
        limit: 10,
        offset: 5,
        min_debates: 50,
        sort_by: 'win_rate',
        order: 'desc',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/rankings', {
        params: { limit: 10, offset: 5, min_debates: 50, sort_by: 'win_rate', order: 'desc' },
      });
    });

    it('should list rankings by domain', async () => {
      const mockRankings = { rankings: [{ rank: 1, agent: 'claude', elo: 1900 }] };
      mockClient.request.mockResolvedValue(mockRankings);

      const result = await api.listByDomain('technology', { limit: 5 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/rankings', {
        params: { limit: 5, domain: 'technology' },
      });
    });
  });

  // ===========================================================================
  // Getting Specific Agent
  // ===========================================================================

  describe('Getting Specific Agent', () => {
    it('should get agent ranking', async () => {
      const mockRanking = {
        ranking: {
          rank: 1,
          agent: 'claude',
          elo: 1850,
          wins: 150,
          losses: 30,
          draws: 20,
          total_debates: 200,
          win_rate: 0.75,
          streak: 5,
          streak_type: 'win',
          last_active: '2024-01-20T10:00:00Z',
          elo_change_24h: 15,
          trend: 'up',
        },
      };
      mockClient.request.mockResolvedValue(mockRanking);

      const result = await api.get('claude');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/rankings/claude');
      expect(result.elo).toBe(1850);
      expect(result.win_rate).toBe(0.75);
    });

    it('should encode agent name in URL', async () => {
      const mockRanking = { ranking: { agent: 'gpt-4-turbo', elo: 1800 } };
      mockClient.request.mockResolvedValue(mockRanking);

      await api.get('gpt-4-turbo');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/rankings/gpt-4-turbo');
    });
  });

  // ===========================================================================
  // Statistics
  // ===========================================================================

  describe('Statistics', () => {
    it('should get ranking statistics', async () => {
      const mockStats = {
        total_ranked_agents: 15,
        total_debates: 5000,
        average_elo: 1500,
        elo_range: { min: 1200, max: 1900 },
        most_improved: { agent: 'grok', elo_gain: 50 },
        most_active: { agent: 'claude', debates: 200 },
        highest_win_rate: { agent: 'opus', rate: 0.82 },
        longest_streak: { agent: 'claude', streak: 12, type: 'win' },
        last_updated: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockStats);

      const result = await api.getStats();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/ranking/stats');
      expect(result.total_ranked_agents).toBe(15);
      expect(result.most_improved.agent).toBe('grok');
    });
  });

  // ===========================================================================
  // Convenience Methods
  // ===========================================================================

  describe('Convenience Methods', () => {
    it('should get top N agents', async () => {
      const mockRankings = {
        rankings: [
          { rank: 1, agent: 'claude', elo: 1850 },
          { rank: 2, agent: 'gpt4', elo: 1800 },
          { rank: 3, agent: 'gemini', elo: 1750 },
        ],
      };
      mockClient.request.mockResolvedValue(mockRankings);

      const result = await api.getTop(3);

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/rankings', {
        params: { limit: 3, sort_by: 'elo', order: 'desc' },
      });
      expect(result).toHaveLength(3);
    });

    it('should get top 10 by default', async () => {
      const mockRankings = { rankings: [] };
      mockClient.request.mockResolvedValue(mockRankings);

      await api.getTop();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/rankings', {
        params: { limit: 10, sort_by: 'elo', order: 'desc' },
      });
    });

    it('should get recently active agents', async () => {
      const mockRankings = {
        rankings: [
          { rank: 1, agent: 'claude', last_active: '2024-01-20T10:00:00Z' },
          { rank: 2, agent: 'gpt4', last_active: '2024-01-20T09:55:00Z' },
        ],
      };
      mockClient.request.mockResolvedValue(mockRankings);

      const result = await api.getRecentlyActive({ limit: 5 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/rankings', {
        params: { limit: 5, sort_by: 'recent_activity', order: 'desc' },
      });
    });
  });
});
