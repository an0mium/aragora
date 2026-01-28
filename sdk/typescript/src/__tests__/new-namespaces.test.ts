/**
 * New Namespace API Tests
 *
 * Tests for the recently added namespace APIs:
 * - backups (disaster recovery)
 * - dashboard (analytics)
 * - devices (device management)
 * - expenses (expense tracking)
 * - rlm (recursive language models)
 * - threatIntel (threat intelligence)
 * - unifiedInbox (multi-channel messaging)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createClient } from '../client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('New Namespace APIs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ===========================================================================
  // Backups Namespace
  // ===========================================================================
  describe('backups namespace', () => {
    it('should expose backups namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.backups).toBeDefined();
      expect(typeof client.backups.list).toBe('function');
      expect(typeof client.backups.get).toBe('function');
      expect(typeof client.backups.create).toBe('function');
      expect(typeof client.backups.verify).toBe('function');
      expect(typeof client.backups.delete).toBe('function');
      expect(typeof client.backups.cleanup).toBe('function');
      expect(typeof client.backups.getStats).toBe('function');
    });

    it('should list backups via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              backups: [{ id: 'b1', source_path: '/data', status: 'completed' }],
              pagination: { limit: 20, offset: 0, total: 1, has_more: false },
            })
          ),
      });

      const result = await client.backups.list();
      expect(result.backups).toHaveLength(1);
      expect(result.backups[0].id).toBe('b1');
    });

    it('should get backup by ID via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              id: 'backup-123',
              source_path: '/var/db',
              backup_type: 'full',
              status: 'verified',
              verified: true,
            })
          ),
      });

      const backup = await client.backups.get('backup-123');
      expect(backup.id).toBe('backup-123');
      expect(backup.verified).toBe(true);
    });

    it('should create backup via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              backup: { id: 'new-backup', source_path: '/data', status: 'pending' },
              message: 'Backup started',
            })
          ),
      });

      const result = await client.backups.create('/data', { backup_type: 'incremental' });
      expect(result.backup.id).toBe('new-backup');
      expect(result.message).toBe('Backup started');
    });

    it('should verify backup via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              backup_id: 'backup-123',
              verified: true,
              checksum_valid: true,
              errors: [],
            })
          ),
      });

      const result = await client.backups.verify('backup-123');
      expect(result.verified).toBe(true);
      expect(result.checksum_valid).toBe(true);
    });

    it('should get backup stats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              stats: {
                total_backups: 10,
                verified_backups: 8,
                failed_backups: 1,
                total_size_mb: 500,
              },
              generated_at: '2025-01-27T12:00:00Z',
            })
          ),
      });

      const result = await client.backups.getStats();
      expect(result.stats.total_backups).toBe(10);
      expect(result.stats.verified_backups).toBe(8);
    });

    it('should cleanup backups via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              dry_run: true,
              backup_ids: ['b1', 'b2'],
              count: 2,
              message: 'Would delete 2 backups',
            })
          ),
      });

      const result = await client.backups.cleanup(true);
      expect(result.count).toBe(2);
      expect(result.dry_run).toBe(true);
    });

    it('should delete backup via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              deleted: true,
              backup_id: 'backup-123',
              message: 'Backup deleted',
            })
          ),
      });

      const result = await client.backups.delete('backup-123');
      expect(result.deleted).toBe(true);
    });
  });

  // ===========================================================================
  // Dashboard Namespace
  // ===========================================================================
  describe('dashboard namespace', () => {
    it('should expose dashboard namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.dashboard).toBeDefined();
      expect(typeof client.dashboard.getOverview).toBe('function');
      expect(typeof client.dashboard.getStats).toBe('function');
      expect(typeof client.dashboard.getActivity).toBe('function');
      expect(typeof client.dashboard.getInboxSummary).toBe('function');
      expect(typeof client.dashboard.getQuickActions).toBe('function');
    });

    it('should get dashboard overview via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              user_id: 'user-1',
              generated_at: '2025-01-27T12:00:00Z',
              inbox: { total_unread: 10, high_priority: 3 },
              today: { emails_received: 25, emails_sent: 12 },
              team: { active_members: 5, open_tickets: 8 },
              ai: { emails_categorized: 100, debates_run: 5 },
              cards: [],
            })
          ),
      });

      const overview = await client.dashboard.getOverview();
      expect(overview.inbox.total_unread).toBe(10);
      expect(overview.today.emails_received).toBe(25);
    });

    it('should get stats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              period: 'week',
              generated_at: '2025-01-27T12:00:00Z',
              email_volume: { labels: [], received: [], sent: [] },
              response_time: { labels: [], values: [] },
              summary: { total_emails: 100, avg_daily_emails: 14.3 },
            })
          ),
      });

      const stats = await client.dashboard.getStats('week');
      expect(stats.period).toBe('week');
      expect(stats.summary.total_emails).toBe(100);
    });

    it('should get activity via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              activities: [
                { id: 'a1', type: 'email', title: 'New email', description: 'Test', timestamp: '2025-01-27', priority: 'high', icon: 'mail' },
              ],
              total: 50,
              limit: 20,
              offset: 0,
              has_more: true,
            })
          ),
      });

      const activity = await client.dashboard.getActivity({ limit: 20 });
      expect(activity.activities).toHaveLength(1);
      expect(activity.total).toBe(50);
    });

    it('should get inbox summary via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              generated_at: '2025-01-27T12:00:00Z',
              counts: { unread: 15, starred: 5, snoozed: 3, drafts: 2, trash: 0 },
              by_priority: { critical: 2, high: 5, medium: 8, low: 10 },
              by_category: { inbox: 20, updates: 5 },
              top_labels: [],
              urgent_emails: [],
              pending_actions: [],
            })
          ),
      });

      const summary = await client.dashboard.getInboxSummary();
      expect(summary.counts.unread).toBe(15);
      expect(summary.by_priority.critical).toBe(2);
    });
  });

  // ===========================================================================
  // Devices Namespace
  // ===========================================================================
  describe('devices namespace', () => {
    it('should expose devices namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.devices).toBeDefined();
      expect(typeof client.devices.register).toBe('function');
      expect(typeof client.devices.unregister).toBe('function');
      expect(typeof client.devices.get).toBe('function');
      expect(typeof client.devices.notify).toBe('function');
      expect(typeof client.devices.getHealth).toBe('function');
    });

    it('should register device via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              success: true,
              device_id: 'd1',
              device_type: 'ios',
              registered_at: '2025-01-27T12:00:00Z',
            })
          ),
      });

      const result = await client.devices.register({
        device_type: 'ios',
        push_token: 'token123',
        user_id: 'user-1',
      });
      expect(result.success).toBe(true);
      expect(result.device_id).toBe('d1');
    });

    it('should get device by ID via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              device_id: 'd1',
              user_id: 'user-1',
              device_type: 'ios',
              device_name: 'iPhone 15',
              last_active: '2025-01-27T12:00:00Z',
              notification_count: 42,
              created_at: '2025-01-01T00:00:00Z',
            })
          ),
      });

      const device = await client.devices.get('d1');
      expect(device.device_id).toBe('d1');
      expect(device.device_type).toBe('ios');
    });

    it('should send notification via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              success: true,
              device_id: 'd1',
              message_id: 'msg-123',
              status: 'sent',
              error: null,
            })
          ),
      });

      const result = await client.devices.notify('d1', {
        title: 'Test',
        body: 'Hello world',
      });
      expect(result.success).toBe(true);
      expect(result.status).toBe('sent');
    });

    it('should get device health via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              status: 'healthy',
              fcm: { available: true, last_check: '2025-01-27T12:00:00Z' },
              apns: { available: true, last_check: '2025-01-27T12:00:00Z' },
              web_push: { available: true, last_check: '2025-01-27T12:00:00Z' },
            })
          ),
      });

      const health = await client.devices.getHealth();
      expect(health.status).toBe('healthy');
      expect(health.fcm?.available).toBe(true);
    });
  });

  // ===========================================================================
  // Expenses Namespace
  // ===========================================================================
  describe('expenses namespace', () => {
    it('should expose expenses namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.expenses).toBeDefined();
      expect(typeof client.expenses.list).toBe('function');
      expect(typeof client.expenses.create).toBe('function');
      expect(typeof client.expenses.approve).toBe('function');
      expect(typeof client.expenses.getStats).toBe('function');
    });

    it('should list expenses via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              expenses: [
                { id: 'e1', vendor_name: 'Acme Corp', amount: 99.99, status: 'pending' },
              ],
              total: 1,
              limit: 20,
              offset: 0,
            })
          ),
      });

      const result = await client.expenses.list();
      expect(result.expenses).toHaveLength(1);
      expect(result.expenses[0].vendor_name).toBe('Acme Corp');
    });

    it('should create expense via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              expense: { id: 'e2', vendor_name: 'Office Depot', amount: 150.00, status: 'pending' },
              message: 'Expense created',
            })
          ),
      });

      const result = await client.expenses.create({
        vendor_name: 'Office Depot',
        amount: 150.00,
        category: 'office_supplies',
      });
      expect(result.expense.id).toBe('e2');
      expect(result.expense.amount).toBe(150.00);
    });

    it('should approve expense via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              expense: { id: 'e1', status: 'approved' },
              message: 'Expense approved',
            })
          ),
      });

      const result = await client.expenses.approve('e1');
      expect(result.expense.status).toBe('approved');
    });

    it('should get expense stats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              stats: {
                total_amount: 5000.00,
                expense_count: 25,
                by_category: { travel: 2000, meals: 1000, office_supplies: 500 },
                by_status: { pending: 5, approved: 15, synced: 5 },
              },
            })
          ),
      });

      const result = await client.expenses.getStats();
      expect(result.stats.total_amount).toBe(5000.00);
      expect(result.stats.expense_count).toBe(25);
    });
  });

  // ===========================================================================
  // RLM Namespace
  // ===========================================================================
  describe('rlm namespace', () => {
    it('should expose rlm namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.rlm).toBeDefined();
      expect(typeof client.rlm.getStats).toBe('function');
      expect(typeof client.rlm.getStrategies).toBe('function');
      expect(typeof client.rlm.compress).toBe('function');
      expect(typeof client.rlm.query).toBe('function');
    });

    it('should get RLM stats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              cache: { hits: 100, misses: 20, size: 50 },
              contexts: { stored: 10, ids: ['c1', 'c2'] },
              system: { has_official_rlm: true, compressor_available: true, rlm_available: true },
              timestamp: '2025-01-27T12:00:00Z',
            })
          ),
      });

      const stats = await client.rlm.getStats();
      expect(stats.cache.hits).toBe(100);
      expect(stats.contexts.stored).toBe(10);
    });

    it('should get strategies via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              strategies: {
                peek: { name: 'peek', description: 'Preview top-level', use_case: 'Quick scan' },
                grep: { name: 'grep', description: 'Search content', use_case: 'Finding specific info' },
              },
              default: 'auto',
              documentation: 'https://docs.example.com/rlm',
            })
          ),
      });

      const result = await client.rlm.getStrategies();
      expect(result.default).toBe('auto');
      expect(result.strategies.peek).toBeDefined();
    });

    it('should compress content via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              context_id: 'ctx-123',
              compression_result: {
                original_tokens: 1000,
                compressed_tokens: 200,
                compression_ratio: 0.8,
                levels: { L0: { nodes: 5, tokens: 50 }, L1: { nodes: 2, tokens: 100 } },
                source_type: 'text',
              },
              created_at: '2025-01-27T12:00:00Z',
            })
          ),
      });

      const result = await client.rlm.compress('Long content to compress...', { source_type: 'text' });
      expect(result.context_id).toBe('ctx-123');
      expect(result.compression_result.compression_ratio).toBe(0.8);
    });

    it('should query context via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              answer: 'The answer is 42',
              metadata: {
                context_id: 'ctx-123',
                strategy: 'auto',
                refined: true,
                confidence: 0.95,
              },
              timestamp: '2025-01-27T12:00:00Z',
            })
          ),
      });

      const result = await client.rlm.query('ctx-123', 'What is the answer?');
      expect(result.answer).toBe('The answer is 42');
      expect(result.metadata.confidence).toBe(0.95);
    });
  });

  // ===========================================================================
  // Threat Intel Namespace
  // ===========================================================================
  describe('threatIntel namespace', () => {
    it('should expose threatIntel namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.threatIntel).toBeDefined();
      expect(typeof client.threatIntel.checkURL).toBe('function');
      expect(typeof client.threatIntel.checkIP).toBe('function');
      expect(typeof client.threatIntel.checkHash).toBe('function');
      expect(typeof client.threatIntel.scanEmail).toBe('function');
      expect(typeof client.threatIntel.getStatus).toBe('function');
    });

    it('should check URL via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              target: 'https://evil.example.com',
              is_malicious: true,
              threat_type: 'phishing',
              severity: 'HIGH',
              confidence: 0.95,
              cached: false,
            })
          ),
      });

      const result = await client.threatIntel.checkURL({ url: 'https://evil.example.com' });
      expect(result.is_malicious).toBe(true);
      expect(result.threat_type).toBe('phishing');
    });

    it('should check IP via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              ip_address: '192.0.2.1',
              is_malicious: true,
              abuse_score: 85,
              total_reports: 100,
              country_code: 'XX',
              categories: ['spam', 'botnet'],
              cached: false,
            })
          ),
      });

      const result = await client.threatIntel.checkIP('192.0.2.1');
      expect(result.is_malicious).toBe(true);
      expect(result.abuse_score).toBe(85);
    });

    it('should check hash via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              hash_value: 'abc123',
              hash_type: 'sha256',
              is_malware: true,
              threat_type: 'trojan',
              detection_ratio: '45/70',
              positives: 45,
              total_scanners: 70,
              cached: false,
            })
          ),
      });

      const result = await client.threatIntel.checkHash('abc123');
      expect(result.is_malware).toBe(true);
      expect(result.positives).toBe(45);
    });

    it('should get threat intel status via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              virustotal: { enabled: true, has_key: true, rate_limit: 4 },
              abuseipdb: { enabled: true, has_key: true, rate_limit: 1000 },
              phishtank: { enabled: true, has_key: false, rate_limit: 0 },
              caching: true,
              cache_ttl_hours: 24,
            })
          ),
      });

      const status = await client.threatIntel.getStatus();
      expect(status.virustotal.enabled).toBe(true);
      expect(status.caching).toBe(true);
    });
  });

  // ===========================================================================
  // Unified Inbox Namespace
  // ===========================================================================
  describe('unifiedInbox namespace', () => {
    it('should expose unifiedInbox namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.unifiedInbox).toBeDefined();
      expect(typeof client.unifiedInbox.listAccounts).toBe('function');
      expect(typeof client.unifiedInbox.listMessages).toBe('function');
      expect(typeof client.unifiedInbox.getMessage).toBe('function');
      expect(typeof client.unifiedInbox.triage).toBe('function');
      expect(typeof client.unifiedInbox.getStats).toBe('function');
    });

    it('should list accounts via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              accounts: [
                { id: 'a1', provider: 'gmail', email_address: 'user@example.com', status: 'connected' },
              ],
              total: 1,
            })
          ),
      });

      const result = await client.unifiedInbox.listAccounts();
      expect(result.accounts).toHaveLength(1);
      expect(result.accounts[0].provider).toBe('gmail');
    });

    it('should list messages via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              messages: [
                {
                  id: 'm1',
                  subject: 'Important meeting',
                  sender: { email: 'boss@company.com', name: 'Boss' },
                  is_read: false,
                  priority: { score: 85, tier: 'high', reasons: ['VIP sender'] },
                },
              ],
              total: 100,
              limit: 20,
              offset: 0,
              has_more: true,
            })
          ),
      });

      const result = await client.unifiedInbox.listMessages({ limit: 20 });
      expect(result.messages).toHaveLength(1);
      expect(result.messages[0].priority.tier).toBe('high');
    });

    it('should get message by ID via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              message: {
                id: 'm1',
                subject: 'Re: Project update',
                sender: { email: 'colleague@company.com', name: 'Colleague' },
                is_read: true,
              },
              triage: null,
            })
          ),
      });

      const result = await client.unifiedInbox.getMessage('m1');
      expect(result.message.subject).toBe('Re: Project update');
    });

    it('should triage messages via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              results: [
                {
                  message_id: 'm1',
                  recommended_action: 'respond_urgent',
                  confidence: 0.9,
                  rationale: 'High priority sender, time-sensitive request',
                  agents_involved: ['claude', 'gpt-4'],
                },
              ],
              total_triaged: 1,
            })
          ),
      });

      const result = await client.unifiedInbox.triage({ message_ids: ['m1'] });
      expect(result.results).toHaveLength(1);
      expect(result.results[0].recommended_action).toBe('respond_urgent');
    });

    it('should get inbox stats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              stats: {
                total_accounts: 2,
                total_messages: 500,
                unread_count: 25,
                messages_by_priority: { critical: 2, high: 10, medium: 50, low: 438 },
                pending_triage: 15,
              },
            })
          ),
      });

      const result = await client.unifiedInbox.getStats();
      expect(result.stats.total_accounts).toBe(2);
      expect(result.stats.unread_count).toBe(25);
    });
  });
});
