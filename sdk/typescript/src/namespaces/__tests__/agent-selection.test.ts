/**
 * Agent Selection Namespace Tests
 *
 * Comprehensive tests for the AgentSelectionAPI namespace class.
 * Tests all methods including:
 * - Plugin discovery (listPlugins, getDefaults)
 * - Agent scoring (scoreAgents, getBestAgent)
 * - Team selection (selectTeam, assignRoles)
 * - History tracking (getHistory)
 * - Convenience methods (listScorers, listTeamSelectors, listRoleAssigners)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  AgentSelectionAPI,
  type AgentSelectionClientInterface,
  type SelectionPlugin,
  type ScorerPlugin,
  type TeamSelectorPlugin,
  type RoleAssignerPlugin,
} from '../agent-selection';

// Helper to create a mock client
function createMockClient(): AgentSelectionClientInterface {
  return {
    request: vi.fn(),
  };
}

describe('AgentSelectionAPI', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let api: AgentSelectionAPI;

  beforeEach(() => {
    vi.clearAllMocks();
    mockClient = createMockClient();
    api = new AgentSelectionAPI(mockClient);
  });

  // ===========================================================================
  // Plugin Discovery
  // ===========================================================================

  describe('Plugin Discovery', () => {
    it('should list all plugins', async () => {
      const mockPlugins = {
        plugins: [
          {
            name: 'elo-scorer',
            type: 'scorer',
            description: 'Scores based on ELO rating',
            version: '1.0.0',
            enabled: true,
            config_schema: {},
          },
          {
            name: 'diversity-selector',
            type: 'team_selector',
            description: 'Selects diverse teams',
            version: '1.0.0',
            enabled: true,
            config_schema: {},
          },
          {
            name: 'capability-assigner',
            type: 'role_assigner',
            description: 'Assigns roles by capability',
            version: '1.0.0',
            enabled: true,
            config_schema: {},
          },
        ],
      };
      mockClient.request.mockResolvedValueOnce(mockPlugins);

      const result = await api.listPlugins();

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/agent-selection/plugins'
      );
      expect(result.plugins).toHaveLength(3);
      expect(result.plugins[0].name).toBe('elo-scorer');
    });

    it('should get default plugin configuration', async () => {
      const mockDefaults = {
        scorer: 'elo-scorer',
        team_selector: 'diversity-selector',
        role_assigner: 'capability-assigner',
        scorer_weights: { accuracy: 0.4, speed: 0.3, cost: 0.3 },
      };
      mockClient.request.mockResolvedValueOnce(mockDefaults);

      const result = await api.getDefaults();

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/agent-selection/defaults'
      );
      expect(result.scorer).toBe('elo-scorer');
      expect(result.scorer_weights).toHaveProperty('accuracy');
    });
  });

  // ===========================================================================
  // Agent Scoring
  // ===========================================================================

  describe('Agent Scoring', () => {
    it('should score agents with minimal parameters', async () => {
      const mockScores = {
        scores: [
          {
            agent_id: 'claude',
            agent_name: 'Claude',
            overall_score: 0.92,
            dimension_scores: { accuracy: 0.95, speed: 0.88 },
            confidence: 0.85,
            reasoning: 'Strong analytical capabilities',
          },
        ],
        scorer_used: 'elo-scorer',
        scoring_time_ms: 150,
      };
      mockClient.request.mockResolvedValueOnce(mockScores);

      const result = await api.scoreAgents({
        agents: ['claude', 'gpt-4', 'gemini'],
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/agent-selection/score',
        {
          json: { agents: ['claude', 'gpt-4', 'gemini'] },
        }
      );
      expect(result.scores[0].overall_score).toBe(0.92);
    });

    it('should score agents with all parameters', async () => {
      const mockScores = {
        scores: [
          {
            agent_id: 'claude',
            agent_name: 'Claude',
            overall_score: 0.92,
            dimension_scores: { accuracy: 0.95, speed: 0.88 },
            confidence: 0.85,
            reasoning: 'Best for code review',
          },
        ],
        scorer_used: 'custom-scorer',
        scoring_time_ms: 200,
      };
      mockClient.request.mockResolvedValueOnce(mockScores);

      const result = await api.scoreAgents({
        agents: ['claude', 'gpt-4'],
        context: 'security code review',
        dimensions: ['accuracy', 'speed', 'cost'],
        scorer: 'custom-scorer',
        weights: { accuracy: 0.5, speed: 0.3, cost: 0.2 },
        top_k: 1,
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/agent-selection/score',
        {
          json: {
            agents: ['claude', 'gpt-4'],
            context: 'security code review',
            dimensions: ['accuracy', 'speed', 'cost'],
            scorer: 'custom-scorer',
            weights: { accuracy: 0.5, speed: 0.3, cost: 0.2 },
            top_k: 1,
          },
        }
      );
      expect(result.scorer_used).toBe('custom-scorer');
    });

    it('should get best agent with minimal parameters', async () => {
      const mockBest = {
        agent_id: 'claude',
        agent_name: 'Claude',
        score: 0.95,
        reasoning: 'Best suited for code review tasks',
        task_type: 'code_review',
      };
      mockClient.request.mockResolvedValueOnce(mockBest);

      const result = await api.getBestAgent({
        pool: ['claude', 'gpt-4', 'gemini'],
        task_type: 'code_review',
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/agent-selection/best',
        {
          json: {
            pool: ['claude', 'gpt-4', 'gemini'],
            task_type: 'code_review',
          },
        }
      );
      expect(result.agent_id).toBe('claude');
      expect(result.task_type).toBe('code_review');
    });

    it('should get best agent with context', async () => {
      const mockBest = {
        agent_id: 'gpt-4',
        agent_name: 'GPT-4',
        score: 0.93,
        reasoning: 'Best for creative tasks',
        task_type: 'creative',
      };
      mockClient.request.mockResolvedValueOnce(mockBest);

      const result = await api.getBestAgent({
        pool: ['claude', 'gpt-4'],
        task_type: 'creative',
        context: 'Writing marketing copy',
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/agent-selection/best',
        {
          json: {
            pool: ['claude', 'gpt-4'],
            task_type: 'creative',
            context: 'Writing marketing copy',
          },
        }
      );
      expect(result.agent_id).toBe('gpt-4');
    });
  });

  // ===========================================================================
  // Team Selection
  // ===========================================================================

  describe('Team Selection', () => {
    it('should select team with minimal parameters', async () => {
      const mockTeam = {
        team: [
          {
            agent_id: 'claude',
            agent_name: 'Claude',
            role: 'lead',
            score: 0.95,
            strengths: ['analysis', 'reasoning'],
            selection_reasoning: 'Best overall candidate',
          },
          {
            agent_id: 'gpt-4',
            agent_name: 'GPT-4',
            role: 'reviewer',
            score: 0.88,
            strengths: ['creativity', 'language'],
            selection_reasoning: 'Strong complementary skills',
          },
        ],
        team_score: 0.92,
        diversity_score: 0.85,
        coverage_score: 0.90,
        selector_used: 'default-selector',
        role_assigner_used: 'default-assigner',
        selection_time_ms: 300,
      };
      mockClient.request.mockResolvedValueOnce(mockTeam);

      const result = await api.selectTeam({
        pool: ['claude', 'gpt-4', 'gemini', 'mistral'],
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/agent-selection/select-team',
        {
          json: { pool: ['claude', 'gpt-4', 'gemini', 'mistral'] },
        }
      );
      expect(result.team).toHaveLength(2);
      expect(result.team_score).toBe(0.92);
    });

    it('should select team with all parameters', async () => {
      const mockTeam = {
        team: [
          {
            agent_id: 'claude',
            agent_name: 'Claude',
            role: 'security_expert',
            score: 0.98,
            strengths: ['security', 'analysis'],
            selection_reasoning: 'Best security expertise',
          },
        ],
        team_score: 0.98,
        diversity_score: 0.70,
        coverage_score: 0.95,
        selector_used: 'custom-selector',
        role_assigner_used: 'custom-assigner',
        selection_time_ms: 250,
      };
      mockClient.request.mockResolvedValueOnce(mockTeam);

      const result = await api.selectTeam({
        pool: ['claude', 'gpt-4', 'gemini'],
        task_requirements: { domain: 'security', complexity: 'high' },
        team_size: 3,
        constraints: { max_cost: 100 },
        min_team_size: 2,
        max_team_size: 5,
        required_roles: ['security_expert', 'reviewer'],
        excluded_agents: ['mistral'],
        diversity_weight: 0.7,
        selector: 'custom-selector',
        role_assigner: 'custom-assigner',
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/agent-selection/select-team',
        {
          json: {
            pool: ['claude', 'gpt-4', 'gemini'],
            task_requirements: { domain: 'security', complexity: 'high' },
            team_size: 3,
            constraints: { max_cost: 100 },
            min_team_size: 2,
            max_team_size: 5,
            required_roles: ['security_expert', 'reviewer'],
            excluded_agents: ['mistral'],
            diversity_weight: 0.7,
            selector: 'custom-selector',
            role_assigner: 'custom-assigner',
          },
        }
      );
      expect(result.selector_used).toBe('custom-selector');
    });

    it('should assign roles with minimal parameters', async () => {
      const mockAssignments = {
        assignments: [
          {
            agent_id: 'claude',
            agent_name: 'Claude',
            role: 'lead',
            confidence: 0.92,
            reasoning: 'Best analytical skills for lead role',
          },
          {
            agent_id: 'gpt-4',
            agent_name: 'GPT-4',
            role: 'reviewer',
            confidence: 0.88,
            reasoning: 'Strong attention to detail',
          },
        ],
        assigner_used: 'default-assigner',
        assignment_time_ms: 100,
      };
      mockClient.request.mockResolvedValueOnce(mockAssignments);

      const result = await api.assignRoles({
        members: ['claude', 'gpt-4'],
        roles: ['lead', 'reviewer'],
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/agent-selection/assign-roles',
        {
          json: {
            members: ['claude', 'gpt-4'],
            roles: ['lead', 'reviewer'],
          },
        }
      );
      expect(result.assignments).toHaveLength(2);
    });

    it('should assign roles with all parameters', async () => {
      const mockAssignments = {
        assignments: [
          {
            agent_id: 'claude',
            agent_name: 'Claude',
            role: 'architect',
            confidence: 0.95,
            reasoning: 'Strong system design skills',
          },
        ],
        assigner_used: 'custom-assigner',
        assignment_time_ms: 80,
      };
      mockClient.request.mockResolvedValueOnce(mockAssignments);

      const result = await api.assignRoles({
        members: ['claude', 'gpt-4'],
        roles: ['architect', 'developer'],
        task_context: 'Designing a microservices architecture',
        assigner: 'custom-assigner',
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/agent-selection/assign-roles',
        {
          json: {
            members: ['claude', 'gpt-4'],
            roles: ['architect', 'developer'],
            task_context: 'Designing a microservices architecture',
            assigner: 'custom-assigner',
          },
        }
      );
      expect(result.assigner_used).toBe('custom-assigner');
    });
  });

  // ===========================================================================
  // History
  // ===========================================================================

  describe('History', () => {
    it('should get selection history without parameters', async () => {
      const mockHistory = {
        history: [
          {
            id: 'sel-001',
            timestamp: '2024-01-15T10:30:00Z',
            selection_type: 'team',
            request: { pool: ['claude', 'gpt-4'] },
            result: { team: ['claude'] },
            duration_ms: 200,
          },
        ],
        total: 100,
        has_more: true,
      };
      mockClient.request.mockResolvedValueOnce(mockHistory);

      const result = await api.getHistory();

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/agent-selection/history',
        { params: {} }
      );
      expect(result.history).toHaveLength(1);
      expect(result.has_more).toBe(true);
    });

    it('should get selection history with parameters', async () => {
      const mockHistory = {
        history: [
          {
            id: 'sel-002',
            timestamp: '2024-01-16T14:00:00Z',
            selection_type: 'score',
            request: { agents: ['claude'] },
            result: { scores: [] },
            duration_ms: 150,
          },
        ],
        total: 50,
        has_more: false,
      };
      mockClient.request.mockResolvedValueOnce(mockHistory);

      const result = await api.getHistory({
        limit: 10,
        since: '2024-01-15T00:00:00Z',
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/agent-selection/history',
        {
          params: {
            limit: 10,
            since: '2024-01-15T00:00:00Z',
          },
        }
      );
      expect(result.total).toBe(50);
    });
  });

  // ===========================================================================
  // Plugin Details
  // ===========================================================================

  describe('Plugin Details', () => {
    it('should get scorer plugin details', async () => {
      const mockScorer: ScorerPlugin = {
        name: 'elo-scorer',
        type: 'scorer',
        description: 'Scores based on ELO rating',
        version: '1.0.0',
        enabled: true,
        config_schema: {},
        metrics: ['accuracy', 'speed', 'cost'],
        weight_range: { min: 0, max: 1 },
      };
      mockClient.request.mockResolvedValueOnce(mockScorer);

      const result = await api.getScorer('elo-scorer');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/agent-selection/scorers/elo-scorer'
      );
      expect(result.metrics).toContain('accuracy');
    });

    it('should get team selector plugin details', async () => {
      const mockSelector: TeamSelectorPlugin = {
        name: 'diversity-selector',
        type: 'team_selector',
        description: 'Selects diverse teams',
        version: '1.0.0',
        enabled: true,
        config_schema: {},
        strategies: ['round-robin', 'weighted'],
        supports_constraints: true,
      };
      mockClient.request.mockResolvedValueOnce(mockSelector);

      const result = await api.getTeamSelector('diversity-selector');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/agent-selection/team-selectors/diversity-selector'
      );
      expect(result.supports_constraints).toBe(true);
    });

    it('should get role assigner plugin details', async () => {
      const mockAssigner: RoleAssignerPlugin = {
        name: 'capability-assigner',
        type: 'role_assigner',
        description: 'Assigns roles by capability',
        version: '1.0.0',
        enabled: true,
        config_schema: {},
        roles: ['lead', 'reviewer', 'expert'],
        assignment_strategies: ['best-fit', 'round-robin'],
      };
      mockClient.request.mockResolvedValueOnce(mockAssigner);

      const result = await api.getRoleAssigner('capability-assigner');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/agent-selection/role-assigners/capability-assigner'
      );
      expect(result.roles).toContain('lead');
    });

    it('should URL encode plugin names with special characters', async () => {
      const mockScorer: ScorerPlugin = {
        name: 'custom/scorer',
        type: 'scorer',
        description: 'Custom scorer',
        version: '1.0.0',
        enabled: true,
        config_schema: {},
        metrics: [],
        weight_range: { min: 0, max: 1 },
      };
      mockClient.request.mockResolvedValueOnce(mockScorer);

      await api.getScorer('custom/scorer');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/agent-selection/scorers/custom%2Fscorer'
      );
    });
  });

  // ===========================================================================
  // Convenience Methods
  // ===========================================================================

  describe('Convenience Methods', () => {
    const mockPlugins = {
      plugins: [
        {
          name: 'elo-scorer',
          type: 'scorer' as const,
          description: 'ELO scorer',
          version: '1.0.0',
          enabled: true,
          config_schema: {},
          metrics: ['accuracy'],
          weight_range: { min: 0, max: 1 },
        },
        {
          name: 'performance-scorer',
          type: 'scorer' as const,
          description: 'Performance scorer',
          version: '1.0.0',
          enabled: true,
          config_schema: {},
          metrics: ['speed'],
          weight_range: { min: 0, max: 1 },
        },
        {
          name: 'diversity-selector',
          type: 'team_selector' as const,
          description: 'Diversity selector',
          version: '1.0.0',
          enabled: true,
          config_schema: {},
          strategies: ['balanced'],
          supports_constraints: true,
        },
        {
          name: 'capability-assigner',
          type: 'role_assigner' as const,
          description: 'Capability assigner',
          version: '1.0.0',
          enabled: true,
          config_schema: {},
          roles: ['lead'],
          assignment_strategies: ['best-fit'],
        },
      ],
    };

    it('should list scorer plugins only', async () => {
      mockClient.request.mockResolvedValueOnce(mockPlugins);

      const result = await api.listScorers();

      expect(result.scorers).toHaveLength(2);
      expect(result.scorers.every(s => s.type === 'scorer')).toBe(true);
    });

    it('should list team selector plugins only', async () => {
      mockClient.request.mockResolvedValueOnce(mockPlugins);

      const result = await api.listTeamSelectors();

      expect(result.selectors).toHaveLength(1);
      expect(result.selectors[0].name).toBe('diversity-selector');
    });

    it('should list role assigner plugins only', async () => {
      mockClient.request.mockResolvedValueOnce(mockPlugins);

      const result = await api.listRoleAssigners();

      expect(result.assigners).toHaveLength(1);
      expect(result.assigners[0].name).toBe('capability-assigner');
    });

    it('should select team with alternatives', async () => {
      const primaryTeam = {
        team: [
          { agent_id: 'claude', agent_name: 'Claude', role: 'lead', score: 0.95, strengths: [], selection_reasoning: '' },
          { agent_id: 'gpt-4', agent_name: 'GPT-4', role: 'reviewer', score: 0.88, strengths: [], selection_reasoning: '' },
        ],
        team_score: 0.92,
        diversity_score: 0.85,
        coverage_score: 0.90,
        selector_used: 'default',
        role_assigner_used: 'default',
        selection_time_ms: 200,
      };

      const altTeam1 = {
        team: [
          { agent_id: 'gemini', agent_name: 'Gemini', role: 'lead', score: 0.85, strengths: [], selection_reasoning: '' },
          { agent_id: 'mistral', agent_name: 'Mistral', role: 'reviewer', score: 0.82, strengths: [], selection_reasoning: '' },
        ],
        team_score: 0.84,
        diversity_score: 0.80,
        coverage_score: 0.85,
        selector_used: 'default',
        role_assigner_used: 'default',
        selection_time_ms: 180,
      };

      mockClient.request
        .mockResolvedValueOnce(primaryTeam)
        .mockResolvedValueOnce(altTeam1);

      const result = await api.selectTeamWithAlternatives(
        { pool: ['claude', 'gpt-4', 'gemini', 'mistral', 'llama'] },
        1
      );

      expect(result.primary.team).toHaveLength(2);
      expect(result.alternatives).toHaveLength(1);
      expect(mockClient.request).toHaveBeenCalledTimes(2);
    });

    it('should handle team selection with alternatives when not enough agents', async () => {
      const primaryTeam = {
        team: [
          { agent_id: 'claude', agent_name: 'Claude', role: 'lead', score: 0.95, strengths: [], selection_reasoning: '' },
        ],
        team_score: 0.95,
        diversity_score: 0.50,
        coverage_score: 0.80,
        selector_used: 'default',
        role_assigner_used: 'default',
        selection_time_ms: 150,
      };

      mockClient.request
        .mockResolvedValueOnce(primaryTeam)
        .mockRejectedValueOnce(new Error('Not enough agents'));

      const result = await api.selectTeamWithAlternatives(
        { pool: ['claude', 'gpt-4'] },
        2
      );

      expect(result.primary.team).toHaveLength(1);
      expect(result.alternatives).toHaveLength(0);
    });
  });
});
