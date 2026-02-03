/**
 * Evaluation Namespace Tests
 *
 * Comprehensive tests for the evaluation namespace API including:
 * - Response evaluation
 * - Response comparison
 * - Dimensions and profiles
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { EvaluationAPI } from '../evaluation';

interface MockClient {
  request: Mock;
}

describe('EvaluationAPI Namespace', () => {
  let api: EvaluationAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new EvaluationAPI(mockClient as any);
  });

  // ===========================================================================
  // Response Evaluation
  // ===========================================================================

  describe('Response Evaluation', () => {
    it('should evaluate response', async () => {
      const mockResult = {
        overall_score: 0.85,
        dimension_scores: {
          accuracy: 0.9,
          clarity: 0.85,
          completeness: 0.8,
        },
        feedback: 'The response is well-structured and accurate.',
        strengths: ['Clear explanation', 'Accurate information'],
        weaknesses: ['Could include more examples'],
        suggestions: ['Add practical examples to illustrate the concept'],
        profile_used: 'default',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.evaluate({
        response: 'Paris is the capital of France.',
        prompt: 'What is the capital of France?',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/evaluate', {
        json: {
          response: 'Paris is the capital of France.',
          prompt: 'What is the capital of France?',
        },
      });
      expect(result.overall_score).toBe(0.85);
      expect(result.strengths).toContain('Clear explanation');
    });

    it('should evaluate with dimensions', async () => {
      const mockResult = {
        overall_score: 0.8,
        dimension_scores: { accuracy: 0.9, helpfulness: 0.7 },
        feedback: 'Good response.',
        strengths: [],
        weaknesses: [],
        suggestions: [],
        profile_used: 'custom',
      };
      mockClient.request.mockResolvedValue(mockResult);

      await api.evaluate({
        response: 'Answer...',
        prompt: 'Question?',
        dimensions: ['accuracy', 'helpfulness'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/evaluate', {
        json: {
          response: 'Answer...',
          prompt: 'Question?',
          dimensions: ['accuracy', 'helpfulness'],
        },
      });
    });

    it('should evaluate with context and profile', async () => {
      const mockResult = {
        overall_score: 0.75,
        dimension_scores: {},
        feedback: '',
        strengths: [],
        weaknesses: [],
        suggestions: [],
        profile_used: 'technical',
      };
      mockClient.request.mockResolvedValue(mockResult);

      await api.evaluate({
        response: 'Technical explanation...',
        prompt: 'Explain X',
        context: 'This is for a technical audience',
        profile: 'technical',
        reference: 'The correct answer is Y',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/evaluate', {
        json: {
          response: 'Technical explanation...',
          prompt: 'Explain X',
          context: 'This is for a technical audience',
          profile: 'technical',
          reference: 'The correct answer is Y',
        },
      });
    });
  });

  // ===========================================================================
  // Response Comparison
  // ===========================================================================

  describe('Response Comparison', () => {
    it('should compare two responses', async () => {
      const mockResult = {
        winner: 'A',
        margin: 0.15,
        response_a_score: 0.85,
        response_b_score: 0.70,
        dimension_comparison: {
          accuracy: { a: 0.9, b: 0.7, winner: 'A' },
          clarity: { a: 0.8, b: 0.75, winner: 'A' },
          completeness: { a: 0.85, b: 0.65, winner: 'A' },
        },
        reasoning: 'Response A provides more accurate and complete information.',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.compare({
        response_a: 'First response...',
        response_b: 'Second response...',
        prompt: 'Original question?',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/evaluate/compare', {
        json: {
          response_a: 'First response...',
          response_b: 'Second response...',
          prompt: 'Original question?',
        },
      });
      expect(result.winner).toBe('A');
      expect(result.margin).toBe(0.15);
    });

    it('should compare with tie result', async () => {
      const mockResult = {
        winner: 'tie',
        margin: 0.02,
        response_a_score: 0.80,
        response_b_score: 0.78,
        dimension_comparison: {},
        reasoning: 'Both responses are of similar quality.',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.compare({
        response_a: 'Response A',
        response_b: 'Response B',
      });

      expect(result.winner).toBe('tie');
    });

    it('should compare with context and dimensions', async () => {
      const mockResult = {
        winner: 'B',
        margin: 0.1,
        response_a_score: 0.7,
        response_b_score: 0.8,
        dimension_comparison: {},
        reasoning: 'Response B is better for technical audience.',
      };
      mockClient.request.mockResolvedValue(mockResult);

      await api.compare({
        response_a: 'Simple explanation',
        response_b: 'Technical explanation',
        prompt: 'Explain concept',
        context: 'Technical audience',
        dimensions: ['technical_accuracy', 'depth'],
        profile: 'technical',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/evaluate/compare', {
        json: {
          response_a: 'Simple explanation',
          response_b: 'Technical explanation',
          prompt: 'Explain concept',
          context: 'Technical audience',
          dimensions: ['technical_accuracy', 'depth'],
          profile: 'technical',
        },
      });
    });
  });

  // ===========================================================================
  // Dimensions
  // ===========================================================================

  describe('Dimensions', () => {
    it('should list dimensions', async () => {
      const mockDimensions = {
        dimensions: [
          {
            id: 'accuracy',
            name: 'Accuracy',
            description: 'How factually correct is the response',
            weight: 1.0,
            criteria: ['Factual correctness', 'No hallucinations'],
          },
          {
            id: 'clarity',
            name: 'Clarity',
            description: 'How clear and understandable is the response',
            weight: 0.8,
            criteria: ['Easy to understand', 'Well-structured'],
          },
          {
            id: 'completeness',
            name: 'Completeness',
            description: 'How thoroughly does the response address the question',
            weight: 0.7,
            criteria: ['Addresses all parts', 'Provides examples'],
          },
        ],
      };
      mockClient.request.mockResolvedValue(mockDimensions);

      const result = await api.listDimensions();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/evaluate/dimensions');
      expect(result.dimensions).toHaveLength(3);
      expect(result.dimensions[0].id).toBe('accuracy');
    });

    it('should get specific dimension', async () => {
      const mockDimensions = {
        dimensions: [
          { id: 'accuracy', name: 'Accuracy', description: '', weight: 1.0, criteria: [] },
          { id: 'clarity', name: 'Clarity', description: '', weight: 0.8, criteria: [] },
        ],
      };
      mockClient.request.mockResolvedValue(mockDimensions);

      const result = await api.getDimension('accuracy');

      expect(result.id).toBe('accuracy');
      expect(result.name).toBe('Accuracy');
    });

    it('should throw error for non-existent dimension', async () => {
      const mockDimensions = { dimensions: [] };
      mockClient.request.mockResolvedValue(mockDimensions);

      await expect(api.getDimension('non_existent')).rejects.toThrow('Dimension not found: non_existent');
    });
  });

  // ===========================================================================
  // Profiles
  // ===========================================================================

  describe('Profiles', () => {
    it('should list profiles', async () => {
      const mockProfiles = {
        profiles: [
          {
            id: 'default',
            name: 'Default',
            description: 'Balanced evaluation profile',
            dimensions: { accuracy: 1.0, clarity: 0.8, completeness: 0.7 },
            default: true,
          },
          {
            id: 'technical',
            name: 'Technical',
            description: 'Emphasizes technical accuracy',
            dimensions: { accuracy: 1.2, technical_depth: 1.0, clarity: 0.6 },
            default: false,
          },
          {
            id: 'creative',
            name: 'Creative',
            description: 'For creative writing tasks',
            dimensions: { creativity: 1.0, engagement: 0.9, clarity: 0.7 },
            default: false,
          },
        ],
      };
      mockClient.request.mockResolvedValue(mockProfiles);

      const result = await api.listProfiles();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/evaluate/profiles');
      expect(result.profiles).toHaveLength(3);
      expect(result.profiles[0].default).toBe(true);
    });

    it('should get specific profile', async () => {
      const mockProfiles = {
        profiles: [
          { id: 'default', name: 'Default', description: '', dimensions: {}, default: true },
          { id: 'technical', name: 'Technical', description: '', dimensions: {}, default: false },
        ],
      };
      mockClient.request.mockResolvedValue(mockProfiles);

      const result = await api.getProfile('technical');

      expect(result.id).toBe('technical');
      expect(result.name).toBe('Technical');
    });

    it('should throw error for non-existent profile', async () => {
      const mockProfiles = { profiles: [] };
      mockClient.request.mockResolvedValue(mockProfiles);

      await expect(api.getProfile('non_existent')).rejects.toThrow('Profile not found: non_existent');
    });
  });
});
