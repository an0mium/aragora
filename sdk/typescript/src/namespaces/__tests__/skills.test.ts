/**
 * Skills Namespace Tests
 *
 * Comprehensive tests for the skills namespace API including:
 * - Skill listing
 * - Skill details
 * - Skill metrics
 * - Skill invocation
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { SkillsAPI } from '../skills';

interface MockClient {
  request: Mock;
}

describe('SkillsAPI Namespace', () => {
  let api: SkillsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new SkillsAPI(mockClient as any);
  });

  // ===========================================================================
  // Skill Listing
  // ===========================================================================

  describe('Skill Listing', () => {
    it('should list all skills', async () => {
      const mockSkills = {
        skills: [
          {
            name: 'web-search',
            version: '1.0.0',
            description: 'Search the web for information',
            capabilities: ['WEB_SEARCH', 'EXTERNAL_API'],
            input_schema: { query: { type: 'string' } },
            tags: ['search', 'web'],
          },
          {
            name: 'code-execution',
            version: '1.2.0',
            description: 'Execute code in a sandboxed environment',
            capabilities: ['CODE_EXECUTION'],
            input_schema: { code: { type: 'string' }, language: { type: 'string' } },
            tags: ['code', 'execution'],
          },
          {
            name: 'evidence-collector',
            version: '2.0.0',
            description: 'Collect and validate evidence for debates',
            capabilities: ['EVIDENCE_COLLECTION', 'WEB_FETCH'],
            input_schema: { topic: { type: 'string' } },
            tags: ['evidence', 'debate'],
          },
        ],
        total: 3,
      };
      mockClient.request.mockResolvedValue(mockSkills);

      const result = await api.list();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/skills');
      expect(result.skills).toHaveLength(3);
      expect(result.total).toBe(3);
    });
  });

  // ===========================================================================
  // Skill Details
  // ===========================================================================

  describe('Skill Details', () => {
    it('should get skill details', async () => {
      const mockSkill = {
        name: 'web-search',
        version: '1.0.0',
        description: 'Search the web for information',
        capabilities: ['WEB_SEARCH', 'EXTERNAL_API'],
        input_schema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            max_results: { type: 'integer', default: 10 },
          },
          required: ['query'],
        },
        output_schema: {
          type: 'object',
          properties: {
            results: { type: 'array' },
            total: { type: 'integer' },
          },
        },
        author: 'aragora-team',
        required_permissions: ['web:search'],
        max_execution_time_seconds: 30,
        debate_compatible: true,
        requires_debate_context: false,
      };
      mockClient.request.mockResolvedValue(mockSkill);

      const result = await api.get('web-search');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/skills/web-search');
      expect(result.name).toBe('web-search');
      expect(result.debate_compatible).toBe(true);
    });

    it('should encode skill name in URL', async () => {
      const mockSkill = { name: 'my-special/skill', version: '1.0.0' };
      mockClient.request.mockResolvedValue(mockSkill);

      await api.get('my-special/skill');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/skills/my-special%2Fskill');
    });
  });

  // ===========================================================================
  // Skill Metrics
  // ===========================================================================

  describe('Skill Metrics', () => {
    it('should get skill metrics', async () => {
      const mockMetrics = {
        skill: 'web-search',
        total_invocations: 5000,
        successful_invocations: 4850,
        failed_invocations: 150,
        average_latency_ms: 250,
        last_invoked: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockMetrics);

      const result = await api.getMetrics('web-search');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/skills/web-search/metrics');
      expect(result.total_invocations).toBe(5000);
      expect(result.average_latency_ms).toBe(250);
    });
  });

  // ===========================================================================
  // Skill Invocation
  // ===========================================================================

  describe('Skill Invocation', () => {
    it('should invoke skill with basic input', async () => {
      const mockResult = {
        status: 'success',
        output: {
          results: [
            { title: 'TypeScript Docs', url: 'https://typescriptlang.org' },
            { title: 'TS Handbook', url: 'https://typescriptlang.org/handbook' },
          ],
          total: 2,
        },
        execution_time_ms: 180,
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.invoke({
        skill: 'web-search',
        input: { query: 'TypeScript best practices' },
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/skills/invoke', {
        json: {
          skill: 'web-search',
          input: { query: 'TypeScript best practices' },
        },
      });
      expect(result.status).toBe('success');
      expect(result.output?.results).toHaveLength(2);
    });

    it('should invoke skill with all options', async () => {
      const mockResult = {
        status: 'success',
        output: { executed: true },
        execution_time_ms: 500,
        metadata: { sandbox_id: 'sb_123' },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.invoke({
        skill: 'code-execution',
        input: { code: 'console.log("hello")', language: 'javascript' },
        user_id: 'u_123',
        permissions: ['code:execute'],
        metadata: { debug: true },
        timeout: 45,
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/skills/invoke', {
        json: {
          skill: 'code-execution',
          input: { code: 'console.log("hello")', language: 'javascript' },
          user_id: 'u_123',
          permissions: ['code:execute'],
          metadata: { debug: true },
          timeout: 45,
        },
      });
    });

    it('should handle skill error', async () => {
      const mockResult = {
        status: 'error',
        error: 'Rate limit exceeded',
        execution_time_ms: 5,
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.invoke({
        skill: 'web-search',
        input: { query: 'test' },
      });

      expect(result.status).toBe('error');
      expect(result.error).toBe('Rate limit exceeded');
    });

    it('should invoke skill without input', async () => {
      const mockResult = {
        status: 'success',
        output: { system_info: { platform: 'linux', version: '5.4.0' } },
      };
      mockClient.request.mockResolvedValue(mockResult);

      await api.invoke({ skill: 'system-info' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/skills/invoke', {
        json: { skill: 'system-info' },
      });
    });
  });
});
