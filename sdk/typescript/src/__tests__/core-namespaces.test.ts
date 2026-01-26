/**
 * Aragora SDK Core Namespace API Tests
 *
 * Tests for core namespaces: billing, budgets, receipts, gauntlet, analytics,
 * memory, knowledge, tournaments, auth, verification, marketplace, codebase.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createClient } from '../client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Core Namespace APIs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('billing namespace', () => {
    it('should expose billing namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.billing).toBeDefined();
      expect(typeof client.billing.getCurrentPlan).toBe('function');
      expect(typeof client.billing.getUsage).toBe('function');
    });

    it('should get current plan via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          plan_id: 'pro',
          name: 'Pro Plan',
          price_cents: 9900,
          features: ['unlimited_debates', 'priority_support']
        })),
      });

      const plan = await client.billing.getCurrentPlan();
      expect(plan.plan_id).toBe('pro');
      expect(plan.features).toContain('unlimited_debates');
    });

    it('should get usage via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          debates_count: 150,
          api_calls: 5000,
          storage_bytes: 1024000,
          period_start: '2024-01-01',
          period_end: '2024-01-31'
        })),
      });

      const usage = await client.billing.getUsage();
      expect(usage.debates_count).toBe(150);
    });

    it('should get invoices via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          invoices: [{ id: 'inv-123', amount_cents: 9900, status: 'paid' }]
        })),
      });

      const result = await client.billing.getInvoices();
      expect(result.invoices).toHaveLength(1);
    });

    it('should forecast usage via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          projected_cost_cents: 15000,
          projected_debates: 200,
          confidence: 0.85
        })),
      });

      const forecast = await client.billing.forecastUsage();
      expect(forecast.projected_cost_cents).toBe(15000);
    });
  });

  describe('budgets namespace', () => {
    it('should expose budgets namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.budgets).toBeDefined();
      expect(typeof client.budgets.list).toBe('function');
      expect(typeof client.budgets.create).toBe('function');
    });

    it('should list budgets via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          budgets: [{ id: 'b1', name: 'Monthly API', limit_cents: 50000, spent_cents: 25000 }]
        })),
      });

      const result = await client.budgets.list();
      expect(result.budgets).toHaveLength(1);
      expect(result.budgets[0].spent_cents).toBe(25000);
    });

    it('should create budget via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'budget-123',
          name: 'Q1 Budget',
          limit_cents: 100000
        })),
      });

      const result = await client.budgets.create({
        name: 'Q1 Budget',
        limit_cents: 100000,
        alert_threshold: 0.8
      });
      expect(result.id).toBe('budget-123');
    });

    it('should get budget alerts via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          alerts: [{ id: 'a1', budget_id: 'b1', threshold: 0.8, triggered: true }]
        })),
      });

      const result = await client.budgets.getAlerts('budget-123');
      expect(result.alerts).toHaveLength(1);
      expect(result.alerts[0].triggered).toBe(true);
    });
  });

  describe('receipts namespace', () => {
    it('should expose receipts namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.receipts).toBeDefined();
      expect(typeof client.receipts.list).toBe('function');
      expect(typeof client.receipts.get).toBe('function');
    });

    it('should list receipts via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          receipts: [{ id: 'r1', debate_id: 'd1', verdict: 'APPROVED', confidence: 0.95 }]
        })),
      });

      const result = await client.receipts.list();
      expect(result.receipts).toHaveLength(1);
      expect(result.receipts[0].verdict).toBe('APPROVED');
    });

    it('should get receipt by ID via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'receipt-123',
          debate_id: 'debate-456',
          verdict: 'APPROVED',
          hash: 'sha256:abc123'
        })),
      });

      const receipt = await client.receipts.get('receipt-123');
      expect(receipt.id).toBe('receipt-123');
      expect(receipt.hash).toBeDefined();
    });

    it('should verify receipt via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          valid: true,
          hash_match: true,
          verified_at: '2024-01-15T10:00:00Z'
        })),
      });

      const result = await client.receipts.verify('receipt-123');
      expect(result.valid).toBe(true);
    });
  });

  describe('gauntlet namespace', () => {
    it('should expose gauntlet namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.gauntlet).toBeDefined();
      expect(typeof client.gauntlet.run).toBe('function');
      expect(typeof client.gauntlet.getStatus).toBe('function');
    });

    it('should run gauntlet via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          gauntlet_id: 'gauntlet-123',
          status: 'pending',
          status_url: '/api/v1/gauntlet/gauntlet-123'
        })),
      });

      const result = await client.gauntlet.run({ input: 'Test input', profile: 'comprehensive' });
      expect(result.gauntlet_id).toBe('gauntlet-123');
    });

    it('should get gauntlet status via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          gauntlet_id: 'gauntlet-123',
          status: 'completed',
          verdict: 'APPROVED',
          confidence: 0.92
        })),
      });

      const status = await client.gauntlet.getStatus('gauntlet-123');
      expect(status.status).toBe('completed');
      expect(status.verdict).toBe('APPROVED');
    });

    it('should get gauntlet receipt via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'receipt-123',
          gauntlet_id: 'gauntlet-123',
          verdict: 'APPROVED',
          findings: []
        })),
      });

      const receipt = await client.gauntlet.getReceipt('receipt-123');
      expect(receipt.gauntlet_id).toBe('gauntlet-123');
    });
  });

  describe('analytics namespace', () => {
    it('should expose analytics namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.analytics).toBeDefined();
      expect(typeof client.analytics.getDebateMetrics).toBe('function');
      expect(typeof client.analytics.getAgentMetrics).toBe('function');
    });

    it('should get debate metrics via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          total_debates: 500,
          avg_duration_ms: 45000,
          consensus_rate: 0.85
        })),
      });

      const metrics = await client.analytics.getDebateMetrics();
      expect(metrics.total_debates).toBe(500);
      expect(metrics.consensus_rate).toBe(0.85);
    });

    it('should get agent metrics via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          agents: [{ name: 'claude', debates: 100, win_rate: 0.65 }]
        })),
      });

      const metrics = await client.analytics.getAgentMetrics();
      expect(metrics.agents).toHaveLength(1);
      expect(metrics.agents[0].win_rate).toBe(0.65);
    });

    it('should get usage trends via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          period: 'weekly',
          data_points: [
            { date: '2024-01-08', debates: 50 },
            { date: '2024-01-15', debates: 75 }
          ]
        })),
      });

      const trends = await client.analytics.getUsageTrends({ period: 'weekly' });
      expect(trends.data_points).toHaveLength(2);
    });
  });

  describe('memory namespace', () => {
    it('should expose memory namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.memory).toBeDefined();
      expect(typeof client.memory.store).toBe('function');
      expect(typeof client.memory.retrieve).toBe('function');
    });

    it('should store memory via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          stored: true,
          tier: 'fast',
          key: 'mem-123'
        })),
      });

      const result = await client.memory.store('mem-123', { data: 'test' });
      expect(result.stored).toBe(true);
    });

    it('should retrieve memory via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          entries: [{ key: 'mem-123', value: { data: 'test' }, tier: 'fast' }]
        })),
      });

      const result = await client.memory.retrieve({ keys: ['mem-123'] });
      expect(result.entries).toHaveLength(1);
    });

    it('should search memory via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          entries: [{ key: 'mem-1', content: 'test content', score: 0.9 }]
        })),
      });

      const result = await client.memory.search({ query: 'test' });
      expect(result.entries).toHaveLength(1);
    });
  });

  describe('knowledge namespace', () => {
    it('should expose knowledge namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.knowledge).toBeDefined();
      expect(typeof client.knowledge.search).toBe('function');
      expect(typeof client.knowledge.add).toBe('function');
    });

    it('should search knowledge via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          results: [{ id: 'k1', content: 'Knowledge content', score: 0.95 }]
        })),
      });

      const result = await client.knowledge.search({ query: 'test query' });
      expect(result.results).toHaveLength(1);
    });

    it('should add knowledge via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'knowledge-123',
          created_at: '2024-01-15'
        })),
      });

      const result = await client.knowledge.add({ content: 'New knowledge', tags: ['test'] });
      expect(result.id).toBe('knowledge-123');
    });

    it('should get knowledge stats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          total_entries: 1000,
          by_visibility: { private: 500, team: 300, global: 200 }
        })),
      });

      const stats = await client.knowledge.getStats();
      expect(stats.total_entries).toBe(1000);
    });
  });

  describe('tournaments namespace', () => {
    it('should expose tournaments namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.tournaments).toBeDefined();
      expect(typeof client.tournaments.list).toBe('function');
      expect(typeof client.tournaments.create).toBe('function');
    });

    it('should list tournaments via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          tournaments: [{ id: 't1', name: 'Q1 Tournament', status: 'active' }]
        })),
      });

      const result = await client.tournaments.list();
      expect(result.tournaments).toHaveLength(1);
    });

    it('should create tournament via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'tournament-123',
          name: 'New Tournament',
          status: 'pending'
        })),
      });

      const result = await client.tournaments.create({ name: 'New Tournament', format: 'round_robin' });
      expect(result.id).toBe('tournament-123');
    });

    it('should get tournament leaderboard via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          rankings: [{ agent: 'claude', wins: 10, losses: 2, elo: 1650 }]
        })),
      });

      const result = await client.tournaments.getLeaderboard('tournament-123');
      expect(result.rankings).toHaveLength(1);
      expect(result.rankings[0].agent).toBe('claude');
    });
  });

  describe('auth namespace', () => {
    it('should expose auth namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.auth).toBeDefined();
      expect(typeof client.auth.getSession).toBe('function');
      expect(typeof client.auth.listApiKeys).toBe('function');
    });

    it('should get session via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          user_id: 'user-123',
          email: 'user@example.com',
          roles: ['admin'],
          expires_at: '2024-01-16T10:00:00Z'
        })),
      });

      const session = await client.auth.getSession();
      expect(session.user_id).toBe('user-123');
      expect(session.roles).toContain('admin');
    });

    it('should list API keys via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          keys: [{ id: 'key-1', name: 'Production', prefix: 'ak_prod_', created_at: '2024-01-01' }]
        })),
      });

      const result = await client.auth.listApiKeys();
      expect(result.keys).toHaveLength(1);
    });

    it('should create API key via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'key-new',
          name: 'New Key',
          key: 'ak_test_xxx', // Only shown once
          created_at: '2024-01-15'
        })),
      });

      const result = await client.auth.createApiKey({ name: 'New Key', scopes: ['debates.read'] });
      expect(result.key).toBeDefined();
    });
  });

  describe('verification namespace', () => {
    it('should expose verification namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.verification).toBeDefined();
      expect(typeof client.verification.verifyConclusion).toBe('function');
    });

    it('should verify conclusion via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          verified: true,
          confidence: 0.92,
          evidence: ['e1', 'e2']
        })),
      });

      const result = await client.verification.verifyConclusion('debate-123');
      expect(result.verified).toBe(true);
    });

    it('should get verification proof via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          proof_id: 'proof-123',
          type: 'z3',
          valid: true
        })),
      });

      const result = await client.verification.getProof('debate-123');
      expect(result.valid).toBe(true);
    });
  });

  describe('marketplace namespace', () => {
    it('should expose marketplace namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.marketplace).toBeDefined();
      expect(typeof client.marketplace.list).toBe('function');
      expect(typeof client.marketplace.get).toBe('function');
    });

    it('should list marketplace templates via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          templates: [{ id: 't1', name: 'PR Review', category: 'development', rating: 4.5 }]
        })),
      });

      const result = await client.marketplace.list();
      expect(result.templates).toHaveLength(1);
    });

    it('should get template via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'template-123',
          name: 'PR Review Template',
          description: 'Review pull requests',
          downloads: 1500
        })),
      });

      const template = await client.marketplace.get('template-123');
      expect(template.name).toBe('PR Review Template');
    });

    it('should rate template via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          new_rating: 4.6
        })),
      });

      const result = await client.marketplace.rate('template-123', 5);
      expect(result.new_rating).toBe(4.6);
    });

    it('should import template via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          imported_id: 'local-template-123'
        })),
      });

      const result = await client.marketplace.import('template-123');
      expect(result.imported_id).toBe('local-template-123');
    });
  });

  describe('codebase namespace', () => {
    it('should expose codebase namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.codebase).toBeDefined();
      expect(typeof client.codebase.startScan).toBe('function');
      expect(typeof client.codebase.listVulnerabilities).toBe('function');
    });

    it('should start codebase scan via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          scan_id: 'scan-123',
          status: 'in_progress',
          started_at: '2024-01-15T10:00:00Z'
        })),
      });

      const result = await client.codebase.startScan('my-repo', { repo_path: '/path/to/repo' });
      expect(result.scan_id).toBe('scan-123');
    });

    it('should list vulnerabilities via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          vulnerabilities: [{ id: 'v1', severity: 'critical', package: 'lodash@4.17.20' }],
          total: 1
        })),
      });

      const result = await client.codebase.listVulnerabilities('my-repo');
      expect(result.vulnerabilities).toHaveLength(1);
      expect(result.vulnerabilities[0].severity).toBe('critical');
    });

    it('should generate SBOM via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          format: 'cyclonedx',
          components: 150,
          download_url: '/api/v1/codebase/sbom/download'
        })),
      });

      const result = await client.codebase.generateSbom({ repo_path: '/path/to/repo', format: 'cyclonedx' });
      expect(result.format).toBe('cyclonedx');
    });

    it('should analyze metrics via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          total_lines: 50000,
          test_coverage: 0.85,
          complexity_avg: 5.2
        })),
      });

      const result = await client.codebase.analyzeMetrics('my-repo', { repo_path: '/path/to/repo' });
      expect(result.test_coverage).toBe(0.85);
    });

    it('should get dead code report via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          dead_code: [{ file: 'utils.py', line: 45, function: 'unused_helper' }],
          total_dead_lines: 150
        })),
      });

      const result = await client.codebase.getDeadcode('my-repo');
      expect(result.dead_code).toHaveLength(1);
    });
  });

  describe('explainability namespace', () => {
    it('should expose explainability namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.explainability).toBeDefined();
      expect(typeof client.explainability.explain).toBe('function');
      expect(typeof client.explainability.getFactors).toBe('function');
    });

    it('should get explanation via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          debate_id: 'debate-123',
          summary: 'The decision was made based on...',
          confidence: 0.92
        })),
      });

      const result = await client.explainability.explain('debate-123');
      expect(result.summary).toContain('decision');
    });

    it('should get factors via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          factors: [{ name: 'evidence_quality', weight: 0.4, impact: 'positive' }]
        })),
      });

      const result = await client.explainability.getFactors('debate-123');
      expect(result.factors).toHaveLength(1);
    });

    it('should get counterfactuals via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          counterfactuals: [{ scenario: 'If evidence X was stronger...', outcome: 'different' }]
        })),
      });

      const result = await client.explainability.getCounterfactuals('debate-123');
      expect(result.counterfactuals).toHaveLength(1);
    });
  });
});
