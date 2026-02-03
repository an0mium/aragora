/**
 * Feedback Namespace Tests
 *
 * Comprehensive tests for the feedback namespace API including:
 * - NPS submissions
 * - General feedback
 * - Feature requests
 * - Bug reports
 * - NPS summary
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { FeedbackAPI } from '../feedback';

interface MockClient {
  request: Mock;
}

describe('FeedbackAPI Namespace', () => {
  let api: FeedbackAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new FeedbackAPI(mockClient as any);
  });

  // ===========================================================================
  // NPS Submissions
  // ===========================================================================

  describe('NPS Submissions', () => {
    it('should submit NPS score', async () => {
      const mockResponse = {
        success: true,
        feedback_id: 'fb_123',
        message: 'NPS feedback received',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.submitNPS({ score: 9 });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/nps', {
        json: { score: 9 },
      });
      expect(result.success).toBe(true);
      expect(result.feedback_id).toBe('fb_123');
    });

    it('should submit NPS with comment', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_124', message: 'Thanks!' };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.submitNPS({
        score: 10,
        comment: 'Great product!',
        context: { page: 'dashboard' },
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/nps', {
        json: {
          score: 10,
          comment: 'Great product!',
          context: { page: 'dashboard' },
        },
      });
    });

    it('should reject invalid NPS score below 0', async () => {
      await expect(api.submitNPS({ score: -1 })).rejects.toThrow('NPS score must be between 0 and 10');
    });

    it('should reject invalid NPS score above 10', async () => {
      await expect(api.submitNPS({ score: 11 })).rejects.toThrow('NPS score must be between 0 and 10');
    });

    it('should accept edge case scores', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_125', message: 'OK' };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.submitNPS({ score: 0 });
      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/nps', {
        json: { score: 0 },
      });

      await api.submitNPS({ score: 10 });
      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/nps', {
        json: { score: 10 },
      });
    });
  });

  // ===========================================================================
  // General Feedback
  // ===========================================================================

  describe('General Feedback', () => {
    it('should submit general feedback', async () => {
      const mockResponse = {
        success: true,
        feedback_id: 'fb_200',
        message: 'Feedback received',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.submitFeedback({
        type: 'general',
        comment: 'I have a suggestion about the UI',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/general', {
        json: {
          type: 'general',
          comment: 'I have a suggestion about the UI',
        },
      });
      expect(result.success).toBe(true);
    });

    it('should submit feedback with score and context', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_201', message: 'OK' };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.submitFeedback({
        type: 'debate_quality',
        comment: 'The debate was well-structured',
        score: 8,
        context: { debate_id: 'd_123' },
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/general', {
        json: {
          type: 'debate_quality',
          comment: 'The debate was well-structured',
          score: 8,
          context: { debate_id: 'd_123' },
        },
      });
    });

    it('should reject feedback without comment', async () => {
      await expect(
        api.submitFeedback({ type: 'general', comment: '' })
      ).rejects.toThrow('Comment is required for feedback submission');
    });
  });

  // ===========================================================================
  // Feature Requests
  // ===========================================================================

  describe('Feature Requests', () => {
    it('should submit feature request', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_300', message: 'Feature request received' };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.submitFeatureRequest('Add dark mode support');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/general', {
        json: {
          type: 'feature_request',
          comment: 'Add dark mode support',
          context: undefined,
        },
      });
      expect(result.success).toBe(true);
    });

    it('should submit feature request with context', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_301', message: 'OK' };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.submitFeatureRequest('Add export to CSV', {
        use_case: 'data analysis',
        priority: 'high',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/general', {
        json: {
          type: 'feature_request',
          comment: 'Add export to CSV',
          context: { use_case: 'data analysis', priority: 'high' },
        },
      });
    });
  });

  // ===========================================================================
  // Bug Reports
  // ===========================================================================

  describe('Bug Reports', () => {
    it('should submit bug report', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_400', message: 'Bug report received' };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.submitBugReport('Button not responding on mobile');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/general', {
        json: {
          type: 'bug_report',
          comment: 'Button not responding on mobile',
          context: undefined,
        },
      });
      expect(result.success).toBe(true);
    });

    it('should submit bug report with context', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_401', message: 'OK' };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.submitBugReport('Login fails with SSO', {
        steps_to_reproduce: ['Click SSO button', 'Redirect fails'],
        browser: 'Chrome 120',
        os: 'macOS 14.0',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/general', {
        json: {
          type: 'bug_report',
          comment: 'Login fails with SSO',
          context: {
            steps_to_reproduce: ['Click SSO button', 'Redirect fails'],
            browser: 'Chrome 120',
            os: 'macOS 14.0',
          },
        },
      });
    });
  });

  // ===========================================================================
  // Debate Quality Feedback
  // ===========================================================================

  describe('Debate Quality Feedback', () => {
    it('should submit debate quality feedback', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_500', message: 'OK' };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.submitDebateQualityFeedback(
        'd_123',
        'The agents provided thoughtful analysis'
      );

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/general', {
        json: {
          type: 'debate_quality',
          comment: 'The agents provided thoughtful analysis',
          score: undefined,
          context: { debate_id: 'd_123' },
        },
      });
    });

    it('should submit debate quality feedback with score', async () => {
      const mockResponse = { success: true, feedback_id: 'fb_501', message: 'OK' };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.submitDebateQualityFeedback(
        'd_456',
        'Excellent consensus building',
        9
      );

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/feedback/general', {
        json: {
          type: 'debate_quality',
          comment: 'Excellent consensus building',
          score: 9,
          context: { debate_id: 'd_456' },
        },
      });
    });
  });

  // ===========================================================================
  // NPS Summary
  // ===========================================================================

  describe('NPS Summary', () => {
    it('should get NPS summary with default days', async () => {
      const mockSummary = {
        nps_score: 45,
        total_responses: 150,
        promoters: 80,
        passives: 50,
        detractors: 20,
        period_days: 30,
      };
      mockClient.request.mockResolvedValue(mockSummary);

      const result = await api.getNPSSummary();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/feedback/nps/summary', {
        params: { days: 30 },
      });
      expect(result.nps_score).toBe(45);
      expect(result.promoters).toBe(80);
    });

    it('should get NPS summary with custom days', async () => {
      const mockSummary = {
        nps_score: 50,
        total_responses: 300,
        promoters: 170,
        passives: 100,
        detractors: 30,
        period_days: 90,
      };
      mockClient.request.mockResolvedValue(mockSummary);

      const result = await api.getNPSSummary(90);

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/feedback/nps/summary', {
        params: { days: 90 },
      });
      expect(result.period_days).toBe(90);
    });
  });

  // ===========================================================================
  // Feedback Prompts
  // ===========================================================================

  describe('Feedback Prompts', () => {
    it('should get active prompts', async () => {
      const mockPrompts = {
        prompts: [
          {
            type: 'nps',
            question: 'How likely are you to recommend Aragora?',
            scale: {
              min: 0,
              max: 10,
              labels: { 0: 'Not at all likely', 10: 'Extremely likely' },
            },
            follow_up: 'What is the main reason for your score?',
          },
        ],
      };
      mockClient.request.mockResolvedValue(mockPrompts);

      const result = await api.getPrompts();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/feedback/prompts');
      expect(result.prompts).toHaveLength(1);
      expect(result.prompts[0].type).toBe('nps');
    });
  });
});
