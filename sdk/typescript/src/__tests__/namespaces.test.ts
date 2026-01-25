/**
 * Aragora SDK Namespace API Tests
 *
 * Tests for the namespaced API (client.debates, client.agents, client.workflows).
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createClient } from '../client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Namespace APIs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('debates namespace', () => {
    it('should expose debates namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.debates).toBeDefined();
      expect(typeof client.debates.list).toBe('function');
      expect(typeof client.debates.get).toBe('function');
      expect(typeof client.debates.create).toBe('function');
    });

    it('should list debates via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ debates: [{ id: 'd1', task: 'Test' }] })),
      });

      const result = await client.debates.list();
      expect(result.debates).toHaveLength(1);
      expect(result.debates[0].id).toBe('d1');
    });

    it('should get debate by ID via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ id: 'debate-123', task: 'Test task', status: 'running' })),
      });

      const debate = await client.debates.get('debate-123');
      expect(debate.id).toBe('debate-123');
      expect(debate.task).toBe('Test task');
    });

    it('should create debate via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ debate_id: 'new-debate', status: 'pending' })),
      });

      const result = await client.debates.create({ task: 'Should we use microservices?' });
      expect(result.debate_id).toBe('new-debate');
    });

    it('should export debate via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ format: 'markdown', content: '# Debate Export' })),
      });

      const exported = await client.debates.export('debate-123', 'markdown');
      expect(exported.format).toBe('markdown');
    });

    it('should get debate messages via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ messages: [{ agent: 'claude', content: 'Hello' }] })),
      });

      const result = await client.debates.getMessages('debate-123');
      expect(result.messages).toHaveLength(1);
      expect(result.messages[0].agent).toBe('claude');
    });
  });

  describe('agents namespace', () => {
    it('should expose agents namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.agents).toBeDefined();
      expect(typeof client.agents.list).toBe('function');
      expect(typeof client.agents.get).toBe('function');
      expect(typeof client.agents.getPerformance).toBe('function');
    });

    it('should list agents via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ agents: [{ name: 'claude', provider: 'anthropic' }] })),
      });

      const result = await client.agents.list();
      expect(result.agents).toHaveLength(1);
      expect(result.agents[0].name).toBe('claude');
    });

    it('should get agent by name via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ name: 'claude', provider: 'anthropic', elo: 1500 })),
      });

      const agent = await client.agents.get('claude');
      expect(agent.name).toBe('claude');
      expect(agent.provider).toBe('anthropic');
    });

    it('should get agent performance via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ agent: 'claude', win_rate: 0.65, total_debates: 100 })),
      });

      const perf = await client.agents.getPerformance('claude');
      expect(perf.agent).toBe('claude');
      expect(perf.win_rate).toBe(0.65);
    });

    it('should get head-to-head stats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ agent_a: 'claude', agent_b: 'gpt-4', wins_a: 30, wins_b: 25 })),
      });

      const h2h = await client.agents.getHeadToHead('claude', 'gpt-4');
      expect(h2h.agent_a).toBe('claude');
      expect(h2h.wins_a).toBe(30);
    });

    it('should get agent ELO via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ agent: 'claude', elo: 1650, history: [] })),
      });

      const elo = await client.agents.getElo('claude');
      expect(elo.agent).toBe('claude');
      expect(elo.elo).toBe(1650);
    });
  });

  describe('workflows namespace', () => {
    it('should expose workflows namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.workflows).toBeDefined();
      expect(typeof client.workflows.list).toBe('function');
      expect(typeof client.workflows.get).toBe('function');
      expect(typeof client.workflows.execute).toBe('function');
    });

    it('should list workflows via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ workflows: [{ id: 'w1', name: 'Test Workflow' }] })),
      });

      const result = await client.workflows.list();
      expect(result.workflows).toHaveLength(1);
      expect(result.workflows[0].name).toBe('Test Workflow');
    });

    it('should execute workflow via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ execution_id: 'exec-123' })),
      });

      const result = await client.workflows.execute('workflow-123', { input: 'value' });
      expect(result.execution_id).toBe('exec-123');
    });

    it('should get workflow execution status via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ id: 'exec-123', status: 'running', progress: 0.5 })),
      });

      const execution = await client.workflows.getExecution('exec-123');
      expect(execution.id).toBe('exec-123');
      expect(execution.status).toBe('running');
    });

    it('should list workflow templates via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ templates: [{ id: 't1', name: 'Review PR' }] })),
      });

      const result = await client.workflows.listTemplates();
      expect(result.templates).toHaveLength(1);
      expect(result.templates[0].name).toBe('Review PR');
    });

    it('should simulate workflow via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ simulated: true, steps: 5, estimated_time: 60 })),
      });

      const simulation = await client.workflows.simulate('workflow-123', { dry_run: true });
      expect(simulation.simulated).toBe(true);
    });

    it('should resolve workflow approval via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ id: 'approval-123', status: 'approved' })),
      });

      const result = await client.workflows.resolveApproval('approval-123', { approved: true, comment: 'LGTM' });
      expect(result.status).toBe('approved');
    });
  });

  describe('sme namespace', () => {
    it('should expose sme namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.sme).toBeDefined();
      expect(typeof client.sme.listWorkflows).toBe('function');
      expect(typeof client.sme.executeWorkflow).toBe('function');
      expect(typeof client.sme.getOnboardingStatus).toBe('function');
    });

    it('should list SME workflows via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ workflows: [{ id: 'invoice', name: 'Invoice Generation' }] })),
      });

      const result = await client.sme.listWorkflows();
      expect(result.workflows).toHaveLength(1);
      expect(result.workflows[0].id).toBe('invoice');
    });

    it('should execute SME workflow via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ execution_id: 'exec-sme-123', status: 'pending' })),
      });

      const result = await client.sme.executeWorkflow('invoice', {
        inputs: { customer_email: 'test@example.com' }
      });
      expect(result.execution_id).toBe('exec-sme-123');
    });

    it('should get onboarding status via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ completed: false, step: 3, total_steps: 8 })),
      });

      const status = await client.sme.getOnboardingStatus();
      expect(status.completed).toBe(false);
      expect(status.step).toBe(3);
    });

    it('should complete onboarding via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ completed: true, organization_id: 'org-123', completed_at: '2024-01-15' })),
      });

      const result = await client.sme.completeOnboarding({ first_debate_id: 'debate-1' });
      expect(result.completed).toBe(true);
      expect(result.organization_id).toBe('org-123');
    });

    it('should provide quick invoice helper', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ execution_id: 'exec-inv-123' })),
      });

      const result = await client.sme.quickInvoice({
        customerEmail: 'billing@client.com',
        customerName: 'Client Corp',
        items: [{ name: 'Service', price: 1000 }]
      });
      expect(result.execution_id).toBe('exec-inv-123');
    });

    it('should provide quick inventory check helper', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ execution_id: 'exec-inv-check-123' })),
      });

      const result = await client.sme.quickInventoryCheck({
        productId: 'SKU-001',
        minThreshold: 10,
        notificationEmail: 'ops@company.com'
      });
      expect(result.execution_id).toBe('exec-inv-check-123');
    });

    it('should provide quick report helper', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ execution_id: 'exec-report-123' })),
      });

      const result = await client.sme.quickReport({
        type: 'sales',
        period: 'weekly',
        format: 'pdf'
      });
      expect(result.execution_id).toBe('exec-report-123');
    });

    it('should provide quick followup helper', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ execution_id: 'exec-followup-123' })),
      });

      const result = await client.sme.quickFollowup({
        customerId: 'cust-123',
        type: 'renewal',
        message: 'Your subscription is expiring!'
      });
      expect(result.execution_id).toBe('exec-followup-123');
    });
  });

  describe('backward compatibility', () => {
    it('should still support flat API methods', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      // Verify flat methods still exist
      expect(typeof client.createDebate).toBe('function');
      expect(typeof client.getDebate).toBe('function');
      expect(typeof client.listDebates).toBe('function');
      expect(typeof client.listAgents).toBe('function');
      expect(typeof client.getAgent).toBe('function');
      expect(typeof client.listWorkflows).toBe('function');
      expect(typeof client.executeWorkflow).toBe('function');
    });

    it('flat and namespace APIs should produce same results', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      const mockDebates = { debates: [{ id: 'd1', task: 'Test' }] };

      // First call via namespace
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockDebates)),
      });
      const namespaceResult = await client.debates.list();

      // Second call via flat API
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockDebates)),
      });
      const flatResult = await client.listDebates();

      expect(namespaceResult).toEqual(flatResult);
    });
  });
});
