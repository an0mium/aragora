/**
 * Debates Namespace Tests
 *
 * Tests for the debates namespace API including:
 * - Core CRUD operations (create, get, list, update, delete)
 * - Debate lifecycle (start, stop, pause, resume, cancel)
 * - Analysis endpoints (impasse, rhetorical, trickster, meta-critique)
 * - Summary and verification
 * - Batch operations
 * - Graph and visualization
 * - Export functionality
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AragoraClient, createClient } from '../client';
import { AragoraError } from '../types';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Debates Namespace', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-api-key',
      retryEnabled: false,
    });
  });

  // ===========================================================================
  // Core CRUD Operations
  // ===========================================================================

  describe('Core CRUD Operations', () => {
    it('should create a debate', async () => {
      const mockResponse = {
        debate_id: 'debate-123',
        status: 'pending',
        task: 'Should we use microservices?',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.debates.create({
        task: 'Should we use microservices?',
        agents: ['claude', 'gpt-4'],
        rounds: 3,
      });

      expect(result.debate_id).toBe('debate-123');
      expect(result.status).toBe('pending');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/debate'),
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('should create a debate with auto_select enabled', async () => {
      const mockResponse = {
        debate_id: 'debate-456',
        status: 'pending',
        task: 'Test auto-select',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.debates.create({
        task: 'Test auto-select',
        auto_select: true,
        rounds: 5,
      });

      expect(result.debate_id).toBe('debate-456');
      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.auto_select).toBe(true);
    });

    it('should get a debate by ID', async () => {
      const mockDebate = {
        id: 'debate-789',
        debate_id: 'debate-789',
        task: 'Test debate',
        status: 'completed',
        agents: ['claude', 'gpt-4'],
        consensus: { reached: true, confidence: 0.95 },
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockDebate)),
      });

      const result = await client.debates.get('debate-789');

      expect(result.status).toBe('completed');
      expect(result.consensus?.reached).toBe(true);
    });

    it('should list debates with pagination', async () => {
      const mockResponse = {
        debates: [
          { id: 'd1', task: 'Debate 1', status: 'completed', agents: [], created_at: '2024-01-01T00:00:00Z' },
          { id: 'd2', task: 'Debate 2', status: 'pending', agents: [], created_at: '2024-01-01T00:00:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.debates.list({ limit: 10, offset: 0 });

      expect(result.debates).toHaveLength(2);
      expect(result.debates[0].task).toBe('Debate 1');
    });

    it('should list debates filtered by status', async () => {
      const mockResponse = {
        debates: [
          { id: 'd1', task: 'Completed Debate', status: 'completed', agents: [], created_at: '2024-01-01T00:00:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.debates.list({ status: 'completed' });

      expect(result.debates).toHaveLength(1);
      expect(result.debates[0].status).toBe('completed');
    });

    it('should update a debate', async () => {
      const mockUpdated = {
        id: 'debate-123',
        debate_id: 'debate-123',
        task: 'Updated task',
        status: 'pending',
        agents: ['claude'],
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockUpdated)),
      });

      const result = await client.debates.update('debate-123', { task: 'Updated task' });

      expect(result.task).toBe('Updated task');
    });

    it('should delete a debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ success: true })),
      });

      const result = await client.debates.delete('debate-123');

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/debates/debate-123'),
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });

  // ===========================================================================
  // Messages and Evidence
  // ===========================================================================

  describe('Messages and Evidence', () => {
    it('should get debate messages', async () => {
      const mockMessages = {
        messages: [
          { role: 'assistant', agent: 'claude', content: 'First message', round: 1 },
          { role: 'assistant', agent: 'gpt-4', content: 'Second message', round: 1 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockMessages)),
      });

      const result = await client.debates.getMessages('debate-123');

      expect(result.messages).toHaveLength(2);
      expect(result.messages[0].agent).toBe('claude');
    });

    it('should add a message to a debate', async () => {
      const mockMessage = {
        role: 'user',
        content: 'What about security?',
        timestamp: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockMessage)),
      });

      const result = await client.debates.addMessage('debate-123', 'What about security?', 'user');

      expect(result.content).toBe('What about security?');
    });

    it('should get debate evidence', async () => {
      const mockEvidence = {
        evidence: [
          { id: 'e1', content: 'Research shows...', source: 'paper' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockEvidence)),
      });

      const result = await client.debates.getEvidence('debate-123');

      expect(result.evidence).toBeDefined();
    });

    it('should add evidence to a debate', async () => {
      const mockResult = {
        evidence_id: 'ev-123',
        success: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.debates.addEvidence('debate-123', 'Studies show...', 'research-paper');

      expect(result.evidence_id).toBe('ev-123');
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Debate Lifecycle
  // ===========================================================================

  describe('Debate Lifecycle', () => {
    it('should start a debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ success: true, status: 'running' })),
      });

      const result = await client.debates.start('debate-123');

      expect(result.success).toBe(true);
      expect(result.status).toBe('running');
    });

    it('should stop a debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ success: true, status: 'stopped' })),
      });

      const result = await client.debates.stop('debate-123');

      expect(result.success).toBe(true);
    });

    it('should pause a debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ success: true, status: 'paused' })),
      });

      const result = await client.debates.pause('debate-123');

      expect(result.success).toBe(true);
      expect(result.status).toBe('paused');
    });

    it('should resume a paused debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ success: true, status: 'running' })),
      });

      const result = await client.debates.resume('debate-123');

      expect(result.success).toBe(true);
    });

    it('should cancel a debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ success: true, status: 'cancelled' })),
      });

      const result = await client.debates.cancel('debate-123');

      expect(result.success).toBe(true);
      expect(result.status).toBe('cancelled');
    });
  });

  // ===========================================================================
  // Analysis Endpoints
  // ===========================================================================

  describe('Analysis Endpoints', () => {
    it('should get impasse detection', async () => {
      const mockImpasse = {
        is_impasse: true,
        confidence: 0.85,
        reason: 'Agents are stuck in circular argument',
        stuck_since_round: 3,
        suggested_intervention: 'Introduce new evidence',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockImpasse)),
      });

      const result = await client.debates.getImpasse('debate-123');

      expect(result.is_impasse).toBe(true);
      expect(result.confidence).toBe(0.85);
      expect(result.stuck_since_round).toBe(3);
    });

    it('should get rhetorical analysis', async () => {
      const mockRhetorical = {
        debate_id: 'debate-123',
        observations: [
          {
            agent: 'claude',
            pattern: 'appeal_to_authority',
            description: 'Used expert citation',
            severity: 0.3,
            round: 2,
            timestamp: '2024-01-01T00:00:00Z',
          },
        ],
        summary: {
          total_observations: 1,
          patterns_detected: ['appeal_to_authority'],
          agents_flagged: ['claude'],
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRhetorical)),
      });

      const result = await client.debates.getRhetorical('debate-123');

      expect(result.observations).toHaveLength(1);
      expect(result.observations[0].pattern).toBe('appeal_to_authority');
    });

    it('should get trickster hollow consensus status', async () => {
      const mockTrickster = {
        debate_id: 'debate-123',
        hollow_consensus_detected: false,
        confidence: 0.15,
        indicators: [],
        recommendation: null,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockTrickster)),
      });

      const result = await client.debates.getTrickster('debate-123');

      expect(result.hollow_consensus_detected).toBe(false);
      expect(result.confidence).toBe(0.15);
    });

    it('should get meta-critique', async () => {
      const mockMetaCritique = {
        debate_id: 'debate-123',
        quality_score: 85,
        critique: 'Overall good debate with minor issues',
        strengths: ['Thorough evidence', 'Balanced perspectives'],
        weaknesses: ['Could explore more alternatives'],
        recommendations: ['Consider edge cases'],
        agent_performance: [
          { agent: 'claude', contribution_score: 0.9, critique: 'Excellent' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockMetaCritique)),
      });

      const result = await client.debates.getMetaCritique('debate-123');

      expect(result.quality_score).toBe(85);
      expect(result.strengths).toContain('Thorough evidence');
    });

    it('should get convergence analysis', async () => {
      const mockConvergence = {
        convergence_score: 0.85,
        areas_of_agreement: ['performance', 'scalability'],
        areas_of_disagreement: ['cost'],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockConvergence)),
      });

      const result = await client.debates.getConvergence('debate-123');

      expect(result.convergence_score).toBe(0.85);
    });
  });

  // ===========================================================================
  // Summary and Verification
  // ===========================================================================

  describe('Summary and Verification', () => {
    it('should get debate summary', async () => {
      const mockSummary = {
        debate_id: 'debate-123',
        verdict: 'Microservices recommended for this use case',
        confidence: 0.92,
        key_points: ['Better scalability', 'Independent deployments'],
        dissenting_views: ['Higher initial complexity'],
        evidence_quality: 0.88,
        generated_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSummary)),
      });

      const result = await client.debates.getSummary('debate-123');

      expect(result.verdict).toContain('Microservices');
      expect(result.confidence).toBe(0.92);
    });

    it('should get verification report', async () => {
      const mockReport = {
        debate_id: 'debate-123',
        verified: true,
        confidence: 0.9,
        claims_verified: 8,
        claims_total: 10,
        verification_details: [
          { claim: 'Claim 1', status: 'verified', evidence: ['e1'], confidence: 0.95 },
        ],
        bonuses: [],
        generated_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockReport)),
      });

      const result = await client.debates.getVerificationReport('debate-123');

      expect(result.verified).toBe(true);
      expect(result.claims_verified).toBe(8);
    });

    it('should verify a specific claim', async () => {
      const mockVerification = {
        claim_id: 'claim-456',
        verified: true,
        confidence: 0.88,
        supporting_evidence: ['ev1', 'ev2'],
        counter_evidence: [],
        status: 'verified',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockVerification)),
      });

      const result = await client.debates.verifyClaim('debate-123', 'claim-456');

      expect(result.verified).toBe(true);
      expect(result.status).toBe('verified');
    });
  });

  // ===========================================================================
  // Fork and Follow-up
  // ===========================================================================

  describe('Fork and Follow-up', () => {
    it('should fork a debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ debate_id: 'forked-debate-789' })),
      });

      const result = await client.debates.fork('debate-123', { branch_point: 2 });

      expect(result.debate_id).toBe('forked-debate-789');
    });

    it('should clone a debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ debate_id: 'cloned-debate-111' })),
      });

      const result = await client.debates.clone('debate-123', { preserveAgents: true });

      expect(result.debate_id).toBe('cloned-debate-111');
    });

    it('should get follow-up suggestions', async () => {
      const mockSuggestions = {
        suggestions: [
          {
            id: 's1',
            topic: 'Security implications',
            question: 'What are the security considerations?',
            rationale: 'Security was not fully explored',
            priority: 'high',
            estimated_value: 0.9,
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSuggestions)),
      });

      const result = await client.debates.getFollowupSuggestions('debate-123');

      expect(result).toHaveLength(1);
      expect(result[0].priority).toBe('high');
    });

    it('should create a follow-up debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ debate_id: 'followup-999' })),
      });

      const result = await client.debates.followUp('debate-123', {
        cruxId: 'crux-1',
        context: 'Focus on security',
      });

      expect(result.debate_id).toBe('followup-999');
    });

    it('should list debate forks', async () => {
      const mockForks = {
        forks: [
          {
            fork_id: 'fork-1',
            parent_debate_id: 'debate-123',
            branch_point: 2,
            created_at: '2024-01-01T00:00:00Z',
            status: 'completed',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockForks)),
      });

      const result = await client.debates.listForks('debate-123');

      expect(result).toHaveLength(1);
      expect(result[0].branch_point).toBe(2);
    });
  });

  // ===========================================================================
  // Export and Graph
  // ===========================================================================

  describe('Export and Graph', () => {
    it('should export debate as markdown', async () => {
      const mockExport = {
        format: 'markdown',
        content: '# Debate Export\n\n...',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExport)),
      });

      const result = await client.debates.export('debate-123', 'markdown');

      expect(result.format).toBe('markdown');
      expect(result.content).toContain('# Debate');
    });

    it('should export debate as JSON', async () => {
      const mockExport = {
        format: 'json',
        content: '{"debate_id": "debate-123"}',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExport)),
      });

      const result = await client.debates.export('debate-123', 'json');

      expect(result.format).toBe('json');
    });

    it('should get debate graph', async () => {
      const mockGraph = {
        nodes: [
          { id: 'n1', type: 'claim', content: 'Main claim', agent: 'claude', round: 1 },
          { id: 'n2', type: 'evidence', content: 'Supporting evidence', agent: 'gpt-4', round: 1 },
        ],
        edges: [
          { source: 'n2', target: 'n1', type: 'supports' },
        ],
        metadata: {
          total_nodes: 2,
          total_edges: 1,
          depth: 2,
          branching_factor: 1,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockGraph)),
      });

      const result = await client.debates.getGraph('debate-123');

      expect(result.nodes).toHaveLength(2);
      expect(result.edges).toHaveLength(1);
      expect(result.metadata.total_nodes).toBe(2);
    });

    it('should get graph branches', async () => {
      const mockBranches = {
        branches: [
          { branch_id: 'b1', root_node: 'n1', depth: 3, node_count: 5 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockBranches)),
      });

      const result = await client.debates.getGraphBranches('debate-123');

      expect(result).toHaveLength(1);
      expect(result[0].depth).toBe(3);
    });

    it('should get graph stats', async () => {
      const mockStats = {
        node_count: 10,
        edge_count: 15,
        depth: 4,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStats)),
      });

      const result = await client.debates.getGraphStats('debate-123');

      expect(result.node_count).toBe(10);
    });
  });

  // ===========================================================================
  // Batch Operations
  // ===========================================================================

  describe('Batch Operations', () => {
    it('should submit batch debates', async () => {
      const mockBatch = {
        batch_id: 'batch-123',
        jobs: [
          { job_id: 'job-1', status: 'pending', created_at: '2024-01-01T00:00:00Z' },
          { job_id: 'job-2', status: 'pending', created_at: '2024-01-01T00:00:00Z' },
        ],
        total_jobs: 2,
        submitted_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockBatch)),
      });

      const result = await client.debates.submitBatch([
        { task: 'Should we use Redis?' },
        { task: 'Should we use PostgreSQL?' },
      ]);

      expect(result.batch_id).toBe('batch-123');
      expect(result.total_jobs).toBe(2);
    });

    it('should get batch status', async () => {
      const mockStatus = {
        batch_id: 'batch-123',
        status: 'running',
        total_jobs: 2,
        completed_jobs: 1,
        failed_jobs: 0,
        jobs: [],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.debates.getBatchStatus('batch-123');

      expect(result.status).toBe('running');
      expect(result.completed_jobs).toBe(1);
    });

    it('should get queue status', async () => {
      const mockQueue = {
        pending_count: 5,
        running_count: 3,
        completed_today: 100,
        average_wait_time_ms: 5000,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockQueue)),
      });

      const result = await client.debates.getQueueStatus();

      expect(result.pending_count).toBe(5);
      expect(result.running_count).toBe(3);
    });
  });

  // ===========================================================================
  // Rounds, Agents, and Votes
  // ===========================================================================

  describe('Rounds, Agents, and Votes', () => {
    it('should get debate rounds', async () => {
      const mockRounds = {
        rounds: [
          {
            number: 1,
            proposals: [{ agent: 'claude', content: 'Proposal 1' }],
            critiques: [],
            status: 'completed',
            started_at: '2024-01-01T00:00:00Z',
            ended_at: '2024-01-01T00:01:00Z',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRounds)),
      });

      const result = await client.debates.getRounds('debate-123');

      expect(result).toHaveLength(1);
      expect(result[0].number).toBe(1);
    });

    it('should get agents in a debate', async () => {
      const mockAgents = {
        agents: [
          { name: 'claude', role: 'proposer', model: 'claude-3', elo: 1500, contributions: 10 },
          { name: 'gpt-4', role: 'critic', model: 'gpt-4', elo: 1480, contributions: 8 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockAgents)),
      });

      const result = await client.debates.getAgents('debate-123');

      expect(result).toHaveLength(2);
      expect(result[0].elo).toBe(1500);
    });

    it('should get votes from a debate', async () => {
      const mockVotes = {
        votes: [
          { agent: 'claude', position: 'for', confidence: 0.9, round: 3 },
          { agent: 'gpt-4', position: 'for', confidence: 0.85, round: 3 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockVotes)),
      });

      const result = await client.debates.getVotes('debate-123');

      expect(result).toHaveLength(2);
      expect(result[0].position).toBe('for');
    });

    it('should add user input', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ input_id: 'input-123', success: true })),
      });

      const result = await client.debates.addUserInput('debate-123', 'Consider scalability', 'suggestion');

      expect(result.input_id).toBe('input-123');
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Explainability
  // ===========================================================================

  describe('Explainability', () => {
    it('should get explainability data', async () => {
      const mockExplain = {
        debate_id: 'debate-123',
        narrative: 'The decision was based on...',
        factors: [
          { name: 'performance', weight: 0.4, description: 'Performance considerations' },
          { name: 'cost', weight: 0.3, description: 'Cost analysis' },
        ],
        confidence: 0.92,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExplain)),
      });

      const result = await client.debates.getExplainability('debate-123');

      expect(result.narrative).toContain('decision');
      expect(result.factors).toHaveLength(2);
    });

    it('should get explainability factors', async () => {
      const mockFactors = {
        factors: [
          { name: 'scalability', weight: 0.5, description: 'System scalability', evidence: ['e1'] },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockFactors)),
      });

      const result = await client.debates.getExplainabilityFactors('debate-123');

      expect(result.factors).toHaveLength(1);
      expect(result.factors[0].name).toBe('scalability');
    });

    it('should get explainability narrative', async () => {
      const mockNarrative = {
        text: 'The debate concluded that...',
        key_points: ['Point 1', 'Point 2'],
        audience_level: 'technical',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockNarrative)),
      });

      const result = await client.debates.getExplainabilityNarrative('debate-123');

      expect(result.text).toContain('concluded');
      expect(result.audience_level).toBe('technical');
    });

    it('should create a counterfactual', async () => {
      const mockCounterfactual = {
        predicted_outcome: 'Different conclusion',
        confidence: 0.75,
        impact_analysis: [
          { factor: 'agents', original: 2, modified: 3, impact: 0.2 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockCounterfactual)),
      });

      const result = await client.debates.createCounterfactual('debate-123', {
        agents: ['claude', 'gpt-4', 'gemini'],
      });

      expect(result.predicted_outcome).toBe('Different conclusion');
      expect(result.impact_analysis).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('Error Handling', () => {
    it('should handle 404 for non-existent debate', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'Debate not found',
          code: 'NOT_FOUND',
        }),
      });

      await expect(client.debates.get('nonexistent'))
        .rejects.toThrow('Debate not found');
    });

    it('should handle validation errors on create', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Task is required',
          code: 'MISSING_FIELD',
          field: 'task',
        }),
      });

      await expect(client.debates.create({ task: '' } as any))
        .rejects.toThrow('Task is required');
    });

    it('should handle server errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({
          error: 'Internal server error',
          code: 'INTERNAL_ERROR',
        }),
      });

      await expect(client.debates.get('debate-123'))
        .rejects.toThrow('Internal server error');
    });
  });

  // ===========================================================================
  // Search and Discovery
  // ===========================================================================

  describe('Search and Discovery', () => {
    it('should search debates', async () => {
      const mockResults = {
        debates: [
          { id: 'd1', task: 'Microservices debate', status: 'completed', agents: [], created_at: '2024-01-01T00:00:00Z' },
        ],
        total: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResults)),
      });

      const result = await client.debates.search({
        query: 'microservices',
        limit: 10,
        status: 'completed',
      });

      expect(result.debates).toHaveLength(1);
      expect(result.total).toBe(1);
    });

    it('should get dashboard data', async () => {
      const mockDashboard = {
        active_count: 5,
        completed_today: 20,
        pending_count: 3,
        recent_debates: [],
        trending_topics: ['AI', 'Cloud'],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockDashboard)),
      });

      const result = await client.debates.getDashboard();

      expect(result.active_count).toBe(5);
      expect(result.trending_topics).toContain('AI');
    });
  });
});
