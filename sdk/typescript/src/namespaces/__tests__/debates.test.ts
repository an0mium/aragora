/**
 * Debates Namespace Tests
 *
 * Comprehensive tests for the debates namespace API including:
 * - Core CRUD operations (create, get, list, update, delete)
 * - Debate lifecycle (start, stop, pause, resume, cancel)
 * - Messages and evidence management
 * - Analysis endpoints (impasse, rhetorical, trickster, meta-critique)
 * - Summary and verification
 * - Fork and follow-up operations
 * - Batch operations
 * - Graph and visualization
 * - Explainability features
 * - Broadcasting and publishing
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { DebatesAPI } from '../debates';
import type { Debate, DebateCreateRequest } from '../../types';

// Mock client interface
interface MockClient {
  listDebates: Mock;
  getDebate: Mock;
  getDebateBySlug: Mock;
  createDebate: Mock;
  getDebateMessages: Mock;
  getDebateConvergence: Mock;
  getDebateCitations: Mock;
  getDebateEvidence: Mock;
  forkDebate: Mock;
  exportDebate: Mock;
  updateDebate: Mock;
  getDebateGraphStats: Mock;
  createDebateAndStream: Mock;
  runDebate: Mock;
  request: Mock;
}

describe('DebatesAPI Namespace', () => {
  let api: DebatesAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      listDebates: vi.fn(),
      getDebate: vi.fn(),
      getDebateBySlug: vi.fn(),
      createDebate: vi.fn(),
      getDebateMessages: vi.fn(),
      getDebateConvergence: vi.fn(),
      getDebateCitations: vi.fn(),
      getDebateEvidence: vi.fn(),
      forkDebate: vi.fn(),
      exportDebate: vi.fn(),
      updateDebate: vi.fn(),
      getDebateGraphStats: vi.fn(),
      createDebateAndStream: vi.fn(),
      runDebate: vi.fn(),
      request: vi.fn(),
    };
    api = new DebatesAPI(mockClient as any);
  });

  // ===========================================================================
  // Core CRUD Operations
  // ===========================================================================

  describe('Core CRUD Operations', () => {
    it('should list debates with pagination', async () => {
      const mockDebates = {
        debates: [
          { id: 'd1', task: 'Debate 1', status: 'completed', agents: [], created_at: '2024-01-01T00:00:00Z' },
          { id: 'd2', task: 'Debate 2', status: 'pending', agents: [], created_at: '2024-01-02T00:00:00Z' },
        ],
      };
      mockClient.listDebates.mockResolvedValue(mockDebates);

      const result = await api.list({ limit: 10, offset: 0 });

      expect(mockClient.listDebates).toHaveBeenCalledWith({ limit: 10, offset: 0 });
      expect(result.debates).toHaveLength(2);
      expect(result.debates[0].task).toBe('Debate 1');
    });

    it('should list debates filtered by status', async () => {
      const mockDebates = { debates: [{ id: 'd1', status: 'completed' }] };
      mockClient.listDebates.mockResolvedValue(mockDebates);

      const result = await api.list({ status: 'completed' });

      expect(mockClient.listDebates).toHaveBeenCalledWith({ status: 'completed' });
      expect(result.debates[0].status).toBe('completed');
    });

    it('should get a debate by ID', async () => {
      const mockDebate = {
        id: 'debate-123',
        task: 'Test debate',
        status: 'completed',
        consensus: { reached: true, confidence: 0.95 },
      };
      mockClient.getDebate.mockResolvedValue(mockDebate);

      const result = await api.get('debate-123');

      expect(mockClient.getDebate).toHaveBeenCalledWith('debate-123');
      expect(result.status).toBe('completed');
      expect(result.consensus?.reached).toBe(true);
    });

    it('should get a debate by slug', async () => {
      const mockDebate = { id: 'debate-456', task: 'Slug debate' };
      mockClient.getDebateBySlug.mockResolvedValue(mockDebate);

      const result = await api.getBySlug('my-debate-slug');

      expect(mockClient.getDebateBySlug).toHaveBeenCalledWith('my-debate-slug');
      expect(result.task).toBe('Slug debate');
    });

    it('should create a new debate', async () => {
      const mockResponse = { debate_id: 'new-debate', status: 'pending' };
      mockClient.createDebate.mockResolvedValue(mockResponse);

      const result = await api.create({
        task: 'Should we use microservices?',
        agents: ['claude', 'gpt-4'],
        rounds: 3,
      });

      expect(mockClient.createDebate).toHaveBeenCalledWith({
        task: 'Should we use microservices?',
        agents: ['claude', 'gpt-4'],
        rounds: 3,
      });
      expect(result.debate_id).toBe('new-debate');
    });

    it('should update an existing debate', async () => {
      const mockUpdated = { id: 'debate-123', status: 'paused', notes: 'Paused for review' };
      mockClient.updateDebate.mockResolvedValue(mockUpdated);

      const result = await api.update('debate-123', { status: 'paused', notes: 'Paused for review' });

      expect(mockClient.updateDebate).toHaveBeenCalledWith('debate-123', { status: 'paused', notes: 'Paused for review' });
      expect(result.status).toBe('paused');
    });

    it('should delete a debate', async () => {
      mockClient.request.mockResolvedValue({ success: true });

      const result = await api.delete('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v1/debates/debate-123');
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Messages and Evidence
  // ===========================================================================

  describe('Messages and Evidence', () => {
    it('should get debate messages', async () => {
      const mockMessages = {
        messages: [
          { role: 'assistant', agent: 'claude', content: 'First message' },
          { role: 'assistant', agent: 'gpt-4', content: 'Second message' },
        ],
      };
      mockClient.getDebateMessages.mockResolvedValue(mockMessages);

      const result = await api.getMessages('debate-123');

      expect(mockClient.getDebateMessages).toHaveBeenCalledWith('debate-123');
      expect(result.messages).toHaveLength(2);
    });

    it('should add a message to a debate', async () => {
      const mockMessage = { role: 'user', content: 'What about security?' };
      mockClient.request.mockResolvedValue(mockMessage);

      const result = await api.addMessage('debate-123', 'What about security?', 'user');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/messages', {
        body: { content: 'What about security?', role: 'user' },
      });
      expect(result.content).toBe('What about security?');
    });

    it('should get debate evidence', async () => {
      const mockEvidence = { evidence: [{ id: 'e1', content: 'Research shows...' }] };
      mockClient.getDebateEvidence.mockResolvedValue(mockEvidence);

      const result = await api.getEvidence('debate-123');

      expect(mockClient.getDebateEvidence).toHaveBeenCalledWith('debate-123');
      expect(result.evidence).toBeDefined();
    });

    it('should add evidence to a debate', async () => {
      mockClient.request.mockResolvedValue({ evidence_id: 'ev-123', success: true });

      const result = await api.addEvidence('debate-123', 'Studies show...', 'research-paper');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/evidence', {
        body: { evidence: 'Studies show...', source: 'research-paper', metadata: undefined },
      });
      expect(result.evidence_id).toBe('ev-123');
    });

    it('should get consensus information', async () => {
      mockClient.request.mockResolvedValue({
        reached: true,
        conclusion: 'Microservices recommended',
        confidence: 0.92,
        dissent: [],
      });

      const result = await api.getConsensus('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/consensus');
      expect(result.reached).toBe(true);
      expect(result.confidence).toBe(0.92);
    });
  });

  // ===========================================================================
  // Debate Lifecycle
  // ===========================================================================

  describe('Debate Lifecycle', () => {
    it('should start a debate', async () => {
      mockClient.request.mockResolvedValue({ success: true, status: 'running' });

      const result = await api.start('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/start');
      expect(result.status).toBe('running');
    });

    it('should stop a debate', async () => {
      mockClient.request.mockResolvedValue({ success: true, status: 'stopped' });

      const result = await api.stop('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/stop');
      expect(result.success).toBe(true);
    });

    it('should pause a debate', async () => {
      mockClient.request.mockResolvedValue({ success: true, status: 'paused' });

      const result = await api.pause('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/pause');
      expect(result.status).toBe('paused');
    });

    it('should resume a paused debate', async () => {
      mockClient.request.mockResolvedValue({ success: true, status: 'running' });

      const result = await api.resume('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/resume');
      expect(result.status).toBe('running');
    });

    it('should cancel a debate', async () => {
      mockClient.request.mockResolvedValue({ success: true, status: 'cancelled' });

      const result = await api.cancel('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/cancel');
      expect(result.status).toBe('cancelled');
    });
  });

  // ===========================================================================
  // Analysis Endpoints
  // ===========================================================================

  describe('Analysis Endpoints', () => {
    it('should get impasse detection', async () => {
      mockClient.request.mockResolvedValue({
        is_impasse: true,
        confidence: 0.85,
        reason: 'Circular argument detected',
        stuck_since_round: 3,
        suggested_intervention: 'Introduce new evidence',
      });

      const result = await api.getImpasse('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/impasse');
      expect(result.is_impasse).toBe(true);
      expect(result.stuck_since_round).toBe(3);
    });

    it('should get rhetorical analysis', async () => {
      mockClient.request.mockResolvedValue({
        debate_id: 'debate-123',
        observations: [
          { agent: 'claude', pattern: 'appeal_to_authority', severity: 0.3 },
        ],
        summary: { total_observations: 1, patterns_detected: ['appeal_to_authority'] },
      });

      const result = await api.getRhetorical('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/rhetorical');
      expect(result.observations).toHaveLength(1);
    });

    it('should get trickster hollow consensus status', async () => {
      mockClient.request.mockResolvedValue({
        debate_id: 'debate-123',
        hollow_consensus_detected: false,
        confidence: 0.15,
        indicators: [],
      });

      const result = await api.getTrickster('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/trickster');
      expect(result.hollow_consensus_detected).toBe(false);
    });

    it('should get meta-critique', async () => {
      mockClient.request.mockResolvedValue({
        debate_id: 'debate-123',
        quality_score: 85,
        strengths: ['Thorough evidence'],
        weaknesses: ['Limited scope'],
      });

      const result = await api.getMetaCritique('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debate/debate-123/meta-critique');
      expect(result.quality_score).toBe(85);
    });

    it('should get convergence analysis', async () => {
      mockClient.getDebateConvergence.mockResolvedValue({
        convergence_score: 0.85,
        areas_of_agreement: ['scalability'],
      });

      const result = await api.getConvergence('debate-123');

      expect(mockClient.getDebateConvergence).toHaveBeenCalledWith('debate-123');
    });
  });

  // ===========================================================================
  // Summary and Verification
  // ===========================================================================

  describe('Summary and Verification', () => {
    it('should get debate summary', async () => {
      mockClient.request.mockResolvedValue({
        debate_id: 'debate-123',
        verdict: 'Microservices recommended',
        confidence: 0.92,
        key_points: ['Better scalability'],
      });

      const result = await api.getSummary('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/summary');
      expect(result.verdict).toContain('Microservices');
    });

    it('should get verification report', async () => {
      mockClient.request.mockResolvedValue({
        debate_id: 'debate-123',
        verified: true,
        claims_verified: 8,
        claims_total: 10,
      });

      const result = await api.getVerificationReport('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/verification-report');
      expect(result.claims_verified).toBe(8);
    });

    it('should verify a specific claim', async () => {
      mockClient.request.mockResolvedValue({
        claim_id: 'claim-456',
        verified: true,
        confidence: 0.88,
        status: 'verified',
      });

      const result = await api.verifyClaim('debate-123', 'claim-456');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/verify', {
        body: { claim_id: 'claim-456', evidence: undefined },
      });
      expect(result.verified).toBe(true);
    });
  });

  // ===========================================================================
  // Fork and Follow-up
  // ===========================================================================

  describe('Fork and Follow-up', () => {
    it('should fork a debate', async () => {
      mockClient.forkDebate.mockResolvedValue({ debate_id: 'forked-debate-789' });

      const result = await api.fork('debate-123', { branch_point: 2 });

      expect(mockClient.forkDebate).toHaveBeenCalledWith('debate-123', { branch_point: 2 });
      expect(result.debate_id).toBe('forked-debate-789');
    });

    it('should clone a debate', async () => {
      mockClient.request.mockResolvedValue({ debate_id: 'cloned-debate-111' });

      const result = await api.clone('debate-123', { preserveAgents: true });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/clone', {
        body: { preserveAgents: true },
      });
      expect(result.debate_id).toBe('cloned-debate-111');
    });

    it('should get follow-up suggestions', async () => {
      mockClient.request.mockResolvedValue({
        suggestions: [
          { id: 's1', topic: 'Security', priority: 'high' },
        ],
      });

      const result = await api.getFollowupSuggestions('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/followups');
      expect(result).toHaveLength(1);
      expect(result[0].priority).toBe('high');
    });

    it('should create a follow-up debate', async () => {
      mockClient.request.mockResolvedValue({ debate_id: 'followup-999' });

      const result = await api.followUp('debate-123', { cruxId: 'crux-1' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/followup', {
        body: { cruxId: 'crux-1' },
      });
      expect(result.debate_id).toBe('followup-999');
    });

    it('should list debate forks', async () => {
      mockClient.request.mockResolvedValue({
        forks: [{ fork_id: 'fork-1', branch_point: 2 }],
      });

      const result = await api.listForks('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/forks');
      expect(result).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Batch Operations
  // ===========================================================================

  describe('Batch Operations', () => {
    it('should submit batch debates', async () => {
      mockClient.request.mockResolvedValue({
        batch_id: 'batch-123',
        jobs: [{ job_id: 'job-1' }, { job_id: 'job-2' }],
        total_jobs: 2,
      });

      const result = await api.submitBatch([
        { task: 'Should we use Redis?' },
        { task: 'Should we use PostgreSQL?' },
      ]);

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/batch', {
        body: { requests: [{ task: 'Should we use Redis?' }, { task: 'Should we use PostgreSQL?' }] },
      });
      expect(result.batch_id).toBe('batch-123');
      expect(result.total_jobs).toBe(2);
    });

    it('should get batch status', async () => {
      mockClient.request.mockResolvedValue({
        batch_id: 'batch-123',
        status: 'running',
        completed_jobs: 1,
        total_jobs: 2,
      });

      const result = await api.getBatchStatus('batch-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/batch/batch-123/status');
      expect(result.status).toBe('running');
    });

    it('should get queue status', async () => {
      mockClient.request.mockResolvedValue({
        pending_count: 5,
        running_count: 3,
        completed_today: 100,
        average_wait_time_ms: 5000,
      });

      const result = await api.getQueueStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/queue/status');
      expect(result.pending_count).toBe(5);
    });
  });

  // ===========================================================================
  // Graph and Visualization
  // ===========================================================================

  describe('Graph and Visualization', () => {
    it('should get debate graph', async () => {
      mockClient.request.mockResolvedValue({
        nodes: [
          { id: 'n1', type: 'claim', content: 'Main claim' },
          { id: 'n2', type: 'evidence', content: 'Supporting evidence' },
        ],
        edges: [{ source: 'n2', target: 'n1', type: 'supports' }],
        metadata: { total_nodes: 2, total_edges: 1, depth: 2 },
      });

      const result = await api.getGraph('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/graph/debate-123');
      expect(result.nodes).toHaveLength(2);
      expect(result.edges).toHaveLength(1);
    });

    it('should get graph branches', async () => {
      mockClient.request.mockResolvedValue({
        branches: [{ branch_id: 'b1', depth: 3, node_count: 5 }],
      });

      const result = await api.getGraphBranches('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/graph/debate-123/branches');
      expect(result).toHaveLength(1);
    });

    it('should export debate as markdown', async () => {
      mockClient.exportDebate.mockResolvedValue({
        format: 'markdown',
        content: '# Debate Export\n\n...',
      });

      const result = await api.export('debate-123', 'markdown');

      expect(mockClient.exportDebate).toHaveBeenCalledWith('debate-123', 'markdown');
      expect(result.format).toBe('markdown');
    });
  });

  // ===========================================================================
  // Explainability
  // ===========================================================================

  describe('Explainability', () => {
    it('should get explainability data', async () => {
      mockClient.request.mockResolvedValue({
        debate_id: 'debate-123',
        narrative: 'The decision was based on...',
        factors: [{ name: 'performance', weight: 0.4 }],
        confidence: 0.92,
      });

      const result = await api.getExplainability('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/explainability');
      expect(result.narrative).toContain('decision');
    });

    it('should get explainability factors', async () => {
      mockClient.request.mockResolvedValue({
        factors: [{ name: 'scalability', weight: 0.5 }],
      });

      const result = await api.getExplainabilityFactors('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/explainability/factors');
      expect(result.factors).toHaveLength(1);
    });

    it('should get explainability narrative', async () => {
      mockClient.request.mockResolvedValue({
        text: 'The debate concluded that...',
        key_points: ['Point 1'],
        audience_level: 'technical',
      });

      const result = await api.getExplainabilityNarrative('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/explainability/narrative');
      expect(result.audience_level).toBe('technical');
    });

    it('should create a counterfactual', async () => {
      mockClient.request.mockResolvedValue({
        predicted_outcome: 'Different conclusion',
        confidence: 0.75,
        impact_analysis: [{ factor: 'agents', impact: 0.2 }],
      });

      const result = await api.createCounterfactual('debate-123', {
        agents: ['claude', 'gpt-4', 'gemini'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/explainability/counterfactual', {
        body: { agents: ['claude', 'gpt-4', 'gemini'] },
      });
      expect(result.predicted_outcome).toBe('Different conclusion');
    });
  });

  // ===========================================================================
  // Red Team and Specialized
  // ===========================================================================

  describe('Red Team and Specialized Debates', () => {
    it('should get red team analysis', async () => {
      mockClient.request.mockResolvedValue({
        debate_id: 'debate-123',
        vulnerabilities: [{ severity: 'high', description: 'Input validation issue' }],
        overall_risk: 0.7,
      });

      const result = await api.getRedTeam('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/red-team');
      expect(result.vulnerabilities).toHaveLength(1);
    });

    it('should run capability probe', async () => {
      mockClient.request.mockResolvedValue({
        debate_id: 'probe-123',
        assessment: 'System can handle real-time data',
        capabilities: [{ name: 'real-time', level: 0.8 }],
      });

      const result = await api.capabilityProbe('Can this system handle real-time data?', ['claude']);

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/capability-probe', {
        body: { task: 'Can this system handle real-time data?', agents: ['claude'] },
      });
      expect(result.assessment).toContain('real-time');
    });
  });

  // ===========================================================================
  // Broadcasting
  // ===========================================================================

  describe('Broadcasting and Publishing', () => {
    it('should broadcast debate to channels', async () => {
      mockClient.request.mockResolvedValue({
        success: true,
        channels_notified: ['slack', 'discord'],
      });

      const result = await api.broadcast('debate-123', ['slack', 'discord']);

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/broadcast', {
        body: { channels: ['slack', 'discord'] },
      });
      expect(result.channels_notified).toContain('slack');
    });
  });

  // ===========================================================================
  // Dashboard and History
  // ===========================================================================

  describe('Dashboard and History', () => {
    it('should get debates dashboard', async () => {
      mockClient.request.mockResolvedValue({
        active_count: 5,
        completed_today: 20,
        pending_count: 3,
        recent_debates: [],
        trending_topics: ['AI', 'Cloud'],
      });

      const result = await api.getDashboard();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/dashboard/debates');
      expect(result.active_count).toBe(5);
      expect(result.trending_topics).toContain('AI');
    });

    it('should search debates', async () => {
      mockClient.request.mockResolvedValue({
        debates: [{ id: 'd1', task: 'Microservices debate' }],
        total: 1,
      });

      const result = await api.search({ query: 'microservices', limit: 10 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/search', {
        params: { query: 'microservices', limit: 10 },
      });
      expect(result.total).toBe(1);
    });

    it('should get debate history', async () => {
      mockClient.request.mockResolvedValue({
        debates: [{ id: 'd1', task: 'Past debate' }],
        total: 50,
      });

      const result = await api.getHistory(10, 0);

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/history/debates', {
        params: { limit: 10, offset: 0 },
      });
      expect(result.total).toBe(50);
    });
  });

  // ===========================================================================
  // Rounds, Agents, and Votes
  // ===========================================================================

  describe('Rounds, Agents, and Votes', () => {
    it('should get debate rounds', async () => {
      mockClient.request.mockResolvedValue({
        rounds: [
          { number: 1, proposals: [], critiques: [], status: 'completed' },
        ],
      });

      const result = await api.getRounds('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/rounds');
      expect(result).toHaveLength(1);
    });

    it('should get debate agents', async () => {
      mockClient.request.mockResolvedValue({
        agents: [
          { name: 'claude', role: 'proposer', elo: 1500 },
        ],
      });

      const result = await api.getAgents('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/agents');
      expect(result[0].name).toBe('claude');
    });

    it('should get debate votes', async () => {
      mockClient.request.mockResolvedValue({
        votes: [
          { agent: 'claude', position: 'for', confidence: 0.9 },
        ],
      });

      const result = await api.getVotes('debate-123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/debates/debate-123/votes');
      expect(result[0].position).toBe('for');
    });

    it('should add user input', async () => {
      mockClient.request.mockResolvedValue({ input_id: 'input-123', success: true });

      const result = await api.addUserInput('debate-123', 'Consider scalability', 'suggestion');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/debates/debate-123/user-input', {
        body: { input: 'Consider scalability', type: 'suggestion' },
      });
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Run and Stream
  // ===========================================================================

  describe('Run and Stream', () => {
    it('should run debate to completion', async () => {
      const mockDebate = { id: 'debate-123', status: 'completed' };
      mockClient.runDebate.mockResolvedValue(mockDebate);

      const result = await api.run({ task: 'Quick debate' });

      expect(mockClient.runDebate).toHaveBeenCalledWith({ task: 'Quick debate' }, undefined);
      expect(result.status).toBe('completed');
    });

    it('should create debate and stream', async () => {
      const mockStream = { debate: { debate_id: 'd1' }, stream: {} };
      mockClient.createDebateAndStream.mockResolvedValue(mockStream);

      const result = await api.createAndStream({ task: 'Streamed debate' });

      expect(mockClient.createDebateAndStream).toHaveBeenCalledWith({ task: 'Streamed debate' }, undefined);
      expect(result.debate.debate_id).toBe('d1');
    });
  });
});
