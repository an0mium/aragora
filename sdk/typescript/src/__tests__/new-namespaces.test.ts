/**
 * New Namespace API Tests
 *
 * Tests for the recently added namespace APIs:
 * - backups (disaster recovery)
 * - dashboard (analytics)
 * - devices (device management)
 * - expenses (expense tracking)
 * - rlm (rate limit management)
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
  });

  describe('dashboard namespace', () => {
    it('should expose dashboard namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.dashboard).toBeDefined();
      expect(typeof client.dashboard.getOverview).toBe('function');
      expect(typeof client.dashboard.getDebateStats).toBe('function');
      expect(typeof client.dashboard.getAgentLeaderboard).toBe('function');
      expect(typeof client.dashboard.getSystemHealth).toBe('function');
    });

    it('should get dashboard overview via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              total_debates: 150,
              active_debates: 5,
              total_agents: 12,
              avg_consensus_rate: 0.85,
            })
          ),
      });

      const overview = await client.dashboard.getOverview();
      expect(overview.total_debates).toBe(150);
      expect(overview.avg_consensus_rate).toBe(0.85);
    });

    it('should get debate stats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              debates_by_day: { '2025-01-26': 10, '2025-01-27': 15 },
              consensus_rate: 0.87,
              average_rounds: 3.2,
            })
          ),
      });

      const stats = await client.dashboard.getDebateStats('7d');
      expect(stats.consensus_rate).toBe(0.87);
      expect(stats.average_rounds).toBe(3.2);
    });

    it('should get agent leaderboard via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              agents: [
                { name: 'claude', elo: 1650, rank: 1 },
                { name: 'gpt-4', elo: 1600, rank: 2 },
              ],
            })
          ),
      });

      const leaderboard = await client.dashboard.getAgentLeaderboard({ limit: 10 });
      expect(leaderboard.agents).toHaveLength(2);
      expect(leaderboard.agents[0].name).toBe('claude');
    });

    it('should get system health via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              status: 'healthy',
              components: {
                database: 'healthy',
                api: 'healthy',
                queue: 'degraded',
              },
              uptime_seconds: 86400,
            })
          ),
      });

      const health = await client.dashboard.getSystemHealth();
      expect(health.status).toBe('healthy');
      expect(health.components.queue).toBe('degraded');
    });
  });

  describe('devices namespace', () => {
    it('should expose devices namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.devices).toBeDefined();
      expect(typeof client.devices.list).toBe('function');
      expect(typeof client.devices.get).toBe('function');
      expect(typeof client.devices.register).toBe('function');
      expect(typeof client.devices.revoke).toBe('function');
    });

    it('should list devices via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              devices: [
                { id: 'd1', name: 'Laptop', type: 'desktop', status: 'active' },
                { id: 'd2', name: 'Phone', type: 'mobile', status: 'active' },
              ],
            })
          ),
      });

      const result = await client.devices.list();
      expect(result.devices).toHaveLength(2);
      expect(result.devices[0].name).toBe('Laptop');
    });

    it('should register device via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              device: { id: 'new-device', name: 'Tablet', status: 'active' },
              token: 'device-token-123',
            })
          ),
      });

      const result = await client.devices.register({
        name: 'Tablet',
        type: 'tablet',
        platform: 'ios',
      });
      expect(result.device.id).toBe('new-device');
      expect(result.token).toBe('device-token-123');
    });

    it('should revoke device via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              revoked: true,
              device_id: 'd1',
              message: 'Device revoked successfully',
            })
          ),
      });

      const result = await client.devices.revoke('d1');
      expect(result.revoked).toBe(true);
    });
  });

  describe('expenses namespace', () => {
    it('should expose expenses namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.expenses).toBeDefined();
      expect(typeof client.expenses.list).toBe('function');
      expect(typeof client.expenses.get).toBe('function');
      expect(typeof client.expenses.create).toBe('function');
      expect(typeof client.expenses.approve).toBe('function');
    });

    it('should list expenses via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              expenses: [
                { id: 'e1', amount: 100.5, currency: 'USD', status: 'pending' },
                { id: 'e2', amount: 250.0, currency: 'USD', status: 'approved' },
              ],
              total_amount: 350.5,
            })
          ),
      });

      const result = await client.expenses.list();
      expect(result.expenses).toHaveLength(2);
      expect(result.total_amount).toBe(350.5);
    });

    it('should create expense via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              expense: {
                id: 'new-expense',
                amount: 75.0,
                currency: 'USD',
                category: 'travel',
                status: 'pending',
              },
            })
          ),
      });

      const result = await client.expenses.create({
        amount: 75.0,
        currency: 'USD',
        category: 'travel',
        description: 'Train ticket',
      });
      expect(result.expense.id).toBe('new-expense');
      expect(result.expense.category).toBe('travel');
    });

    it('should approve expense via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              expense: { id: 'e1', status: 'approved' },
              approved_by: 'manager@company.com',
            })
          ),
      });

      const result = await client.expenses.approve('e1', { comment: 'LGTM' });
      expect(result.expense.status).toBe('approved');
    });
  });

  describe('rlm namespace', () => {
    it('should expose rlm namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.rlm).toBeDefined();
      expect(typeof client.rlm.getLimits).toBe('function');
      expect(typeof client.rlm.getUsage).toBe('function');
      expect(typeof client.rlm.setOverride).toBe('function');
    });

    it('should get rate limits via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              limits: {
                requests_per_minute: 60,
                requests_per_hour: 1000,
                tokens_per_day: 100000,
              },
              tier: 'standard',
            })
          ),
      });

      const result = await client.rlm.getLimits();
      expect(result.limits.requests_per_minute).toBe(60);
      expect(result.tier).toBe('standard');
    });

    it('should get usage via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              usage: {
                requests_this_minute: 25,
                requests_this_hour: 450,
                tokens_today: 50000,
              },
              remaining: {
                requests_per_minute: 35,
                requests_per_hour: 550,
              },
            })
          ),
      });

      const result = await client.rlm.getUsage();
      expect(result.usage.requests_this_minute).toBe(25);
      expect(result.remaining.requests_per_minute).toBe(35);
    });

    it('should set rate limit override via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              override: {
                user_id: 'user-123',
                requests_per_minute: 120,
                expires_at: '2025-02-01T00:00:00Z',
              },
            })
          ),
      });

      const result = await client.rlm.setOverride('user-123', {
        requests_per_minute: 120,
        expires_at: '2025-02-01T00:00:00Z',
      });
      expect(result.override.requests_per_minute).toBe(120);
    });
  });

  describe('threatIntel namespace', () => {
    it('should expose threatIntel namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.threatIntel).toBeDefined();
      expect(typeof client.threatIntel.checkIndicator).toBe('function');
      expect(typeof client.threatIntel.getFeeds).toBe('function');
      expect(typeof client.threatIntel.searchThreats).toBe('function');
    });

    it('should check indicator via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              indicator: '192.0.2.1',
              type: 'ip',
              malicious: true,
              confidence: 0.95,
              sources: ['abuse_db', 'internal'],
              tags: ['malware', 'c2'],
            })
          ),
      });

      const result = await client.threatIntel.checkIndicator('192.0.2.1', 'ip');
      expect(result.malicious).toBe(true);
      expect(result.confidence).toBe(0.95);
      expect(result.tags).toContain('malware');
    });

    it('should get threat feeds via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              feeds: [
                { id: 'abuse_db', name: 'Abuse DB', active: true, last_update: '2025-01-27' },
                { id: 'phishing', name: 'Phishing Feed', active: true, last_update: '2025-01-27' },
              ],
            })
          ),
      });

      const result = await client.threatIntel.getFeeds();
      expect(result.feeds).toHaveLength(2);
      expect(result.feeds[0].id).toBe('abuse_db');
    });

    it('should search threats via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              threats: [
                { id: 't1', name: 'APT29', type: 'apt', severity: 'high' },
                { id: 't2', name: 'Emotet', type: 'malware', severity: 'critical' },
              ],
              total: 2,
            })
          ),
      });

      const result = await client.threatIntel.searchThreats({ severity: 'high' });
      expect(result.threats).toHaveLength(2);
      expect(result.threats[0].type).toBe('apt');
    });
  });

  describe('unifiedInbox namespace', () => {
    it('should expose unifiedInbox namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.unifiedInbox).toBeDefined();
      expect(typeof client.unifiedInbox.list).toBe('function');
      expect(typeof client.unifiedInbox.get).toBe('function');
      expect(typeof client.unifiedInbox.send).toBe('function');
      expect(typeof client.unifiedInbox.reply).toBe('function');
    });

    it('should list messages via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              messages: [
                { id: 'm1', channel: 'slack', from: 'user@company.com', preview: 'Hello!' },
                { id: 'm2', channel: 'email', from: 'client@client.com', preview: 'Question about...' },
              ],
              unread_count: 5,
            })
          ),
      });

      const result = await client.unifiedInbox.list();
      expect(result.messages).toHaveLength(2);
      expect(result.unread_count).toBe(5);
    });

    it('should get message by ID via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              id: 'm1',
              channel: 'slack',
              from: 'user@company.com',
              content: 'Hello, I have a question about...',
              received_at: '2025-01-27T10:00:00Z',
              read: false,
            })
          ),
      });

      const message = await client.unifiedInbox.get('m1');
      expect(message.channel).toBe('slack');
      expect(message.read).toBe(false);
    });

    it('should send message via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              message_id: 'sent-1',
              channel: 'slack',
              sent_at: '2025-01-27T10:05:00Z',
              status: 'delivered',
            })
          ),
      });

      const result = await client.unifiedInbox.send({
        channel: 'slack',
        to: 'user@company.com',
        content: 'Here is your answer...',
      });
      expect(result.message_id).toBe('sent-1');
      expect(result.status).toBe('delivered');
    });

    it('should reply to message via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () =>
          Promise.resolve(
            JSON.stringify({
              message_id: 'reply-1',
              in_reply_to: 'm1',
              channel: 'slack',
              status: 'delivered',
            })
          ),
      });

      const result = await client.unifiedInbox.reply('m1', {
        content: 'Thanks for reaching out!',
      });
      expect(result.in_reply_to).toBe('m1');
      expect(result.status).toBe('delivered');
    });
  });
});
