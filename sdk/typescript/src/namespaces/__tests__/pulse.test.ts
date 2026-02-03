/**
 * Pulse Namespace Tests
 *
 * Comprehensive tests for the pulse namespace API including:
 * - Trending topics
 * - Debate suggestions
 * - Pulse analytics
 * - Scheduler management
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { PulseAPI } from '../pulse';

interface MockClient {
  request: Mock;
}

describe('PulseAPI Namespace', () => {
  let api: PulseAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new PulseAPI(mockClient as any);
  });

  // ===========================================================================
  // Trending Topics
  // ===========================================================================

  describe('Trending Topics', () => {
    it('should get trending topics', async () => {
      const mockTopics = {
        topics: [
          {
            id: 't_1',
            title: 'AI Regulation Debate Heats Up',
            url: 'https://news.example.com/ai-regulation',
            source: 'hackernews',
            score: 450,
            comments_count: 320,
            posted_at: '2024-01-20T08:00:00Z',
            fetched_at: '2024-01-20T10:00:00Z',
            category: 'technology',
            sentiment: 'mixed',
          },
          {
            id: 't_2',
            title: 'New Study on Remote Work Productivity',
            source: 'reddit',
            score: 1200,
            comments_count: 580,
            posted_at: '2024-01-20T07:30:00Z',
            fetched_at: '2024-01-20T10:00:00Z',
            category: 'business',
            sentiment: 'positive',
          },
        ],
        fetched_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockTopics);

      const result = await api.getTrending();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/trending', { params: undefined });
      expect(result.topics).toHaveLength(2);
    });

    it('should get trending by source', async () => {
      const mockTopics = { topics: [], fetched_at: '2024-01-20T10:00:00Z' };
      mockClient.request.mockResolvedValue(mockTopics);

      await api.getTrending({ source: 'hackernews' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/trending', {
        params: { source: 'hackernews' },
      });
    });

    it('should get trending with filters', async () => {
      const mockTopics = { topics: [], fetched_at: '2024-01-20T10:00:00Z' };
      mockClient.request.mockResolvedValue(mockTopics);

      await api.getTrending({
        source: 'reddit',
        category: 'technology',
        limit: 20,
        min_score: 100,
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/trending', {
        params: { source: 'reddit', category: 'technology', limit: 20, min_score: 100 },
      });
    });
  });

  // ===========================================================================
  // Debate Suggestions
  // ===========================================================================

  describe('Debate Suggestions', () => {
    it('should get debate suggestion', async () => {
      const mockSuggestion = {
        topic: {
          id: 't_1',
          title: 'AI Regulation Debate',
          source: 'hackernews',
          score: 450,
        },
        suggested_task: 'Debate the merits of AI regulation in the EU',
        suggested_agents: ['claude', 'gpt-4', 'gemini'],
        relevance_score: 0.92,
        controversy_score: 0.75,
        timeliness_score: 0.95,
      };
      mockClient.request.mockResolvedValue(mockSuggestion);

      const result = await api.suggest();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/suggest', { params: undefined });
      expect(result.suggested_agents).toContain('claude');
      expect(result.relevance_score).toBe(0.92);
    });

    it('should get controversial suggestion', async () => {
      const mockSuggestion = {
        topic: { id: 't_2' },
        controversy_score: 0.95,
      };
      mockClient.request.mockResolvedValue(mockSuggestion);

      await api.suggest({ prefer_controversial: true });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/suggest', {
        params: { prefer_controversial: true },
      });
    });

    it('should get suggestion by source and category', async () => {
      const mockSuggestion = { topic: { id: 't_3' } };
      mockClient.request.mockResolvedValue(mockSuggestion);

      await api.suggest({ source: 'twitter', category: 'politics' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/suggest', {
        params: { source: 'twitter', category: 'politics' },
      });
    });
  });

  // ===========================================================================
  // Analytics
  // ===========================================================================

  describe('Analytics', () => {
    it('should get pulse analytics', async () => {
      const mockAnalytics = {
        total_debates: 150,
        consensus_rate: 0.72,
        average_rounds: 3.5,
        top_sources: [
          { source: 'hackernews', count: 80, consensus_rate: 0.78 },
          { source: 'reddit', count: 50, consensus_rate: 0.68 },
          { source: 'twitter', count: 20, consensus_rate: 0.65 },
        ],
        trending_categories: [
          { category: 'technology', count: 60 },
          { category: 'business', count: 45 },
          { category: 'science', count: 30 },
        ],
        period: '30d',
      };
      mockClient.request.mockResolvedValue(mockAnalytics);

      const result = await api.getAnalytics();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/analytics', { params: undefined });
      expect(result.consensus_rate).toBe(0.72);
      expect(result.top_sources).toHaveLength(3);
    });

    it('should get analytics with filters', async () => {
      const mockAnalytics = { total_debates: 50, period: '7d' };
      mockClient.request.mockResolvedValue(mockAnalytics);

      await api.getAnalytics({ source: 'hackernews', period: '7d' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/analytics', {
        params: { source: 'hackernews', period: '7d' },
      });
    });
  });

  // ===========================================================================
  // Start Debate
  // ===========================================================================

  describe('Start Debate', () => {
    it('should start debate on trending topic', async () => {
      const mockResult = {
        debate_id: 'd_123',
        task: 'Debate: AI Regulation in the EU',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.startDebate('t_1');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/pulse/debate-topic', {
        json: { topic_id: 't_1' },
      });
      expect(result.debate_id).toBe('d_123');
    });

    it('should start debate with custom agents', async () => {
      const mockResult = { debate_id: 'd_124', task: 'Debate topic' };
      mockClient.request.mockResolvedValue(mockResult);

      await api.startDebate('t_1', {
        agents: ['claude', 'gpt-4'],
        protocol: { rounds: 5 },
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/pulse/debate-topic', {
        json: {
          topic_id: 't_1',
          agents: ['claude', 'gpt-4'],
          protocol: { rounds: 5 },
        },
      });
    });
  });

  // ===========================================================================
  // Scheduler Management
  // ===========================================================================

  describe('Scheduler Management', () => {
    it('should get scheduler status', async () => {
      const mockStatus = {
        running: true,
        paused: false,
        last_fetch: '2024-01-20T09:55:00Z',
        next_fetch: '2024-01-20T10:00:00Z',
        fetch_interval_minutes: 5,
        topics_fetched: 1500,
        debates_started: 200,
        errors_count: 3,
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getSchedulerStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/scheduler/status');
      expect(result.running).toBe(true);
      expect(result.topics_fetched).toBe(1500);
    });

    it('should start scheduler', async () => {
      const mockResult = { success: true, message: 'Scheduler started' };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.startScheduler();

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/pulse/scheduler/start');
      expect(result.success).toBe(true);
    });

    it('should stop scheduler', async () => {
      const mockResult = { success: true, message: 'Scheduler stopped' };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.stopScheduler();

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/pulse/scheduler/stop');
      expect(result.success).toBe(true);
    });

    it('should pause scheduler', async () => {
      const mockResult = { success: true, message: 'Scheduler paused' };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.pauseScheduler();

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/pulse/scheduler/pause');
      expect(result.success).toBe(true);
    });

    it('should resume scheduler', async () => {
      const mockResult = { success: true, message: 'Scheduler resumed' };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.resumeScheduler();

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/pulse/scheduler/resume');
      expect(result.success).toBe(true);
    });

    it('should update scheduler config', async () => {
      const mockResult = {
        success: true,
        config: { fetch_interval_minutes: 10, auto_debate: true },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.updateSchedulerConfig({
        fetch_interval_minutes: 10,
        auto_debate: true,
        min_score: 50,
        sources: ['hackernews', 'reddit'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('PATCH', '/api/pulse/scheduler/config', {
        json: {
          fetch_interval_minutes: 10,
          auto_debate: true,
          min_score: 50,
          sources: ['hackernews', 'reddit'],
        },
      });
    });

    it('should get scheduler history', async () => {
      const mockHistory = {
        debates: [
          {
            id: 'sd_1',
            topic_id: 't_1',
            topic_title: 'AI Regulation',
            source: 'hackernews',
            debate_id: 'd_123',
            started_at: '2024-01-20T09:00:00Z',
            completed_at: '2024-01-20T09:15:00Z',
            result: 'consensus',
          },
        ],
        total: 1,
      };
      mockClient.request.mockResolvedValue(mockHistory);

      const result = await api.getSchedulerHistory();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/scheduler/history', { params: undefined });
      expect(result.debates).toHaveLength(1);
    });

    it('should get scheduler history with filters', async () => {
      const mockHistory = { debates: [], total: 0 };
      mockClient.request.mockResolvedValue(mockHistory);

      await api.getSchedulerHistory({ platform: 'reddit', limit: 20, offset: 10 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/pulse/scheduler/history', {
        params: { platform: 'reddit', limit: 20, offset: 10 },
      });
    });
  });
});
