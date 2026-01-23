'use client';

/**
 * Shared Inbox Dashboard - Team Collaboration Email Management
 *
 * Features:
 * - List of shared inboxes
 * - Message queue with assignment/status
 * - Routing rules management
 * - Team activity overview
 */

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useAuth } from '@/context/AuthContext';

interface SharedInbox {
  id: string;
  workspace_id: string;
  name: string;
  description?: string;
  email_address?: string;
  connector_type?: string;
  team_members: string[];
  admins: string[];
  message_count: number;
  unread_count: number;
  created_at: string;
}

interface SharedInboxMessage {
  id: string;
  inbox_id: string;
  email_id: string;
  subject: string;
  from_address: string;
  snippet: string;
  received_at: string;
  status: string;
  assigned_to?: string;
  tags: string[];
  priority?: string;
}

interface RoutingRule {
  id: string;
  name: string;
  workspace_id: string;
  conditions: Array<{
    field: string;
    operator: string;
    value: string;
  }>;
  actions: Array<{
    type: string;
    target?: string;
  }>;
  priority: number;
  enabled: boolean;
  stats: { total_matches: number };
}

const STATUS_COLORS: Record<string, string> = {
  open: 'bg-acid-blue/20 text-acid-blue border-acid-blue/40',
  assigned: 'bg-acid-purple/20 text-acid-purple border-acid-purple/40',
  in_progress: 'bg-acid-cyan/20 text-acid-cyan border-acid-cyan/40',
  waiting: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
  resolved: 'bg-acid-green/20 text-acid-green border-acid-green/40',
  closed: 'bg-muted/20 text-muted border-muted/40',
};

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={`px-2 py-0.5 text-xs font-mono rounded border ${
        STATUS_COLORS[status] || STATUS_COLORS.open
      }`}
    >
      {status.toUpperCase().replace('_', ' ')}
    </span>
  );
}

function PriorityIndicator({ priority }: { priority?: string }) {
  if (!priority) return null;
  const colors: Record<string, string> = {
    critical: 'text-acid-red',
    high: 'text-acid-orange',
    medium: 'text-acid-yellow',
    low: 'text-acid-cyan',
  };
  const icons: Record<string, string> = {
    critical: '!!!',
    high: '!!',
    medium: '!',
    low: '-',
  };
  return (
    <span className={`text-xs font-mono ${colors[priority] || 'text-muted'}`}>
      {icons[priority] || ''}
    </span>
  );
}

type ActiveView = 'inboxes' | 'messages' | 'rules';

export default function SharedInboxPage() {
  const { config: backendConfig } = useBackend();
  const { tokens } = useAuth();

  const [inboxes, setInboxes] = useState<SharedInbox[]>([]);
  const [selectedInbox, setSelectedInbox] = useState<SharedInbox | null>(null);
  const [messages, setMessages] = useState<SharedInboxMessage[]>([]);
  const [rules, setRules] = useState<RoutingRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState<ActiveView>('inboxes');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [showCreateInbox, setShowCreateInbox] = useState(false);
  const [showCreateRule, setShowCreateRule] = useState(false);

  // Form states
  const [newInboxName, setNewInboxName] = useState('');
  const [newInboxDescription, setNewInboxDescription] = useState('');
  const [newInboxEmail, setNewInboxEmail] = useState('');

  const workspaceId = 'default'; // TODO: Get from auth context

  const fetchInboxes = useCallback(async () => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/v1/inbox/shared?workspace_id=${workspaceId}`,
        {
          headers: { Authorization: `Bearer ${tokens?.access_token || ''}` },
        }
      );
      if (response.ok) {
        const data = await response.json();
        setInboxes(data.inboxes || []);
      } else {
        // Use mock data
        setMockInboxes();
      }
    } catch {
      setMockInboxes();
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api, tokens?.access_token, workspaceId]);

  const setMockInboxes = () => {
    setInboxes([
      {
        id: 'inbox-001',
        workspace_id: 'default',
        name: 'Support',
        description: 'Customer support inquiries',
        email_address: 'support@company.com',
        connector_type: 'gmail',
        team_members: ['user1', 'user2', 'user3'],
        admins: ['admin1'],
        message_count: 24,
        unread_count: 8,
        created_at: new Date(Date.now() - 86400000 * 30).toISOString(),
      },
      {
        id: 'inbox-002',
        workspace_id: 'default',
        name: 'Sales',
        description: 'Sales inquiries and leads',
        email_address: 'sales@company.com',
        connector_type: 'gmail',
        team_members: ['user4', 'user5'],
        admins: ['admin1'],
        message_count: 15,
        unread_count: 3,
        created_at: new Date(Date.now() - 86400000 * 15).toISOString(),
      },
      {
        id: 'inbox-003',
        workspace_id: 'default',
        name: 'HR',
        description: 'Human resources',
        email_address: 'hr@company.com',
        team_members: ['user6'],
        admins: ['admin2'],
        message_count: 7,
        unread_count: 2,
        created_at: new Date(Date.now() - 86400000 * 7).toISOString(),
      },
    ]);
  };

  const fetchMessages = useCallback(
    async (inboxId: string) => {
      try {
        const params = new URLSearchParams();
        if (statusFilter !== 'all') params.set('status', statusFilter);

        const response = await fetch(
          `${backendConfig.api}/api/v1/inbox/shared/${inboxId}/messages?${params}`,
          {
            headers: { Authorization: `Bearer ${tokens?.access_token || ''}` },
          }
        );
        if (response.ok) {
          const data = await response.json();
          setMessages(data.messages || []);
        } else {
          // Mock messages
          setMessages([
            {
              id: 'msg-001',
              inbox_id: inboxId,
              email_id: 'email-001',
              subject: 'Need help with billing',
              from_address: 'customer@example.com',
              snippet: 'Hi, I was charged twice for my subscription last month...',
              received_at: new Date(Date.now() - 3600000).toISOString(),
              status: 'open',
              tags: ['billing', 'urgent'],
              priority: 'high',
            },
            {
              id: 'msg-002',
              inbox_id: inboxId,
              email_id: 'email-002',
              subject: 'Feature request: Dark mode',
              from_address: 'user@example.com',
              snippet: 'Would love to see a dark mode option in the app...',
              received_at: new Date(Date.now() - 7200000).toISOString(),
              status: 'assigned',
              assigned_to: 'user1',
              tags: ['feature-request'],
              priority: 'medium',
            },
            {
              id: 'msg-003',
              inbox_id: inboxId,
              email_id: 'email-003',
              subject: 'Account access issue',
              from_address: 'blocked@example.com',
              snippet: "I can't log into my account after the update...",
              received_at: new Date(Date.now() - 14400000).toISOString(),
              status: 'in_progress',
              assigned_to: 'user2',
              tags: ['account', 'bug'],
              priority: 'critical',
            },
          ]);
        }
      } catch {
        setMessages([]);
      }
    },
    [backendConfig.api, tokens?.access_token, statusFilter]
  );

  const fetchRules = useCallback(async () => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/v1/inbox/routing/rules?workspace_id=${workspaceId}`,
        {
          headers: { Authorization: `Bearer ${tokens?.access_token || ''}` },
        }
      );
      if (response.ok) {
        const data = await response.json();
        setRules(data.rules || []);
      } else {
        // Mock rules
        setRules([
          {
            id: 'rule-001',
            name: 'Urgent Issues',
            workspace_id: 'default',
            conditions: [{ field: 'subject', operator: 'contains', value: 'urgent' }],
            actions: [
              { type: 'label', target: 'urgent' },
              { type: 'assign', target: 'support-lead' },
            ],
            priority: 1,
            enabled: true,
            stats: { total_matches: 45 },
          },
          {
            id: 'rule-002',
            name: 'Billing Inquiries',
            workspace_id: 'default',
            conditions: [{ field: 'subject', operator: 'contains', value: 'billing' }],
            actions: [
              { type: 'label', target: 'billing' },
              { type: 'assign', target: 'billing-team' },
            ],
            priority: 2,
            enabled: true,
            stats: { total_matches: 128 },
          },
          {
            id: 'rule-003',
            name: 'Newsletter Signups',
            workspace_id: 'default',
            conditions: [{ field: 'from', operator: 'contains', value: 'newsletter' }],
            actions: [{ type: 'archive' }],
            priority: 10,
            enabled: false,
            stats: { total_matches: 0 },
          },
        ]);
      }
    } catch {
      setRules([]);
    }
  }, [backendConfig.api, tokens?.access_token, workspaceId]);

  useEffect(() => {
    fetchInboxes();
    fetchRules();
  }, [fetchInboxes, fetchRules]);

  useEffect(() => {
    if (selectedInbox) {
      fetchMessages(selectedInbox.id);
    }
  }, [selectedInbox, fetchMessages]);

  const handleCreateInbox = async () => {
    if (!newInboxName.trim()) return;

    try {
      const response = await fetch(`${backendConfig.api}/api/v1/inbox/shared`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({
          workspace_id: workspaceId,
          name: newInboxName,
          description: newInboxDescription || undefined,
          email_address: newInboxEmail || undefined,
        }),
      });

      if (response.ok) {
        fetchInboxes();
        setShowCreateInbox(false);
        setNewInboxName('');
        setNewInboxDescription('');
        setNewInboxEmail('');
      }
    } catch {
      // Handle error
    }
  };

  const handleAssign = async (messageId: string, assignedTo: string) => {
    if (!selectedInbox) return;

    try {
      await fetch(
        `${backendConfig.api}/api/v1/inbox/shared/${selectedInbox.id}/messages/${messageId}/assign`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${tokens?.access_token || ''}`,
          },
          body: JSON.stringify({ assigned_to: assignedTo }),
        }
      );
      fetchMessages(selectedInbox.id);
    } catch {
      // Handle error
    }
  };

  const handleStatusChange = async (messageId: string, newStatus: string) => {
    if (!selectedInbox) return;

    try {
      await fetch(
        `${backendConfig.api}/api/v1/inbox/shared/${selectedInbox.id}/messages/${messageId}/status`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${tokens?.access_token || ''}`,
          },
          body: JSON.stringify({ status: newStatus }),
        }
      );
      fetchMessages(selectedInbox.id);
    } catch {
      // Handle error
    }
  };

  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <div className="min-h-screen bg-background">
      <Scanlines />
      <CRTVignette />

      <header className="border-b border-border bg-surface/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="hover:text-accent">
              <AsciiBannerCompact />
            </Link>
            <span className="text-muted font-mono text-sm">{'//'} SHARED INBOX</span>
          </div>
          <div className="flex items-center gap-3">
            <Link href="/inbox" className="text-xs font-mono text-muted hover:text-accent">
              Personal Inbox
            </Link>
            <BackendSelector />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {/* Page Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-mono mb-1">SHARED INBOX</h1>
            <p className="text-muted text-sm font-mono">Team collaboration email management</p>
          </div>
          <button
            onClick={() => setShowCreateInbox(true)}
            className="btn btn-primary"
          >
            + New Inbox
          </button>
        </div>

        {/* View Tabs */}
        <div className="border-b border-border mb-6">
          <div className="flex gap-4">
            {[
              { id: 'inboxes' as ActiveView, label: 'INBOXES', count: inboxes.length },
              { id: 'messages' as ActiveView, label: 'MESSAGES', count: messages.length },
              { id: 'rules' as ActiveView, label: 'ROUTING RULES', count: rules.length },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveView(tab.id)}
                className={`px-4 py-2 font-mono text-sm transition-colors flex items-center gap-2 ${
                  activeView === tab.id
                    ? 'text-accent border-b-2 border-accent'
                    : 'text-muted hover:text-foreground'
                }`}
              >
                {tab.label}
                <span className="px-1.5 py-0.5 bg-surface rounded text-xs">{tab.count}</span>
              </button>
            ))}
          </div>
        </div>

        <PanelErrorBoundary panelName="Shared Inbox Content">
          {/* Inboxes View */}
          {activeView === 'inboxes' && (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {loading ? (
                [1, 2, 3].map((i) => (
                  <div key={i} className="card p-4 animate-pulse">
                    <div className="h-6 bg-surface rounded mb-2" />
                    <div className="h-4 bg-surface rounded w-2/3" />
                  </div>
                ))
              ) : inboxes.length === 0 ? (
                <div className="col-span-full card p-12 text-center">
                  <div className="text-4xl mb-4">📬</div>
                  <div className="text-muted font-mono mb-4">No shared inboxes yet</div>
                  <button
                    onClick={() => setShowCreateInbox(true)}
                    className="btn btn-primary"
                  >
                    Create First Inbox
                  </button>
                </div>
              ) : (
                inboxes.map((inbox) => (
                  <div
                    key={inbox.id}
                    onClick={() => {
                      setSelectedInbox(inbox);
                      setActiveView('messages');
                    }}
                    className={`card p-4 cursor-pointer transition-all hover:border-accent/50 ${
                      selectedInbox?.id === inbox.id ? 'border-accent' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="font-mono font-medium">{inbox.name}</h3>
                      {inbox.unread_count > 0 && (
                        <span className="px-2 py-0.5 bg-accent/20 text-accent text-xs font-mono rounded">
                          {inbox.unread_count} new
                        </span>
                      )}
                    </div>
                    {inbox.description && (
                      <p className="text-sm text-muted mb-3">{inbox.description}</p>
                    )}
                    <div className="flex items-center justify-between text-xs font-mono text-muted">
                      <span>{inbox.message_count} messages</span>
                      <span>{inbox.team_members.length} members</span>
                    </div>
                    {inbox.email_address && (
                      <div className="mt-2 text-xs font-mono text-accent/70 truncate">
                        {inbox.email_address}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}

          {/* Messages View */}
          {activeView === 'messages' && (
            <div>
              {/* Inbox Selector & Filters */}
              <div className="flex items-center gap-4 mb-4">
                <select
                  value={selectedInbox?.id || ''}
                  onChange={(e) => {
                    const inbox = inboxes.find((i) => i.id === e.target.value);
                    setSelectedInbox(inbox || null);
                  }}
                  className="input"
                >
                  <option value="">Select Inbox...</option>
                  {inboxes.map((inbox) => (
                    <option key={inbox.id} value={inbox.id}>
                      {inbox.name} ({inbox.message_count})
                    </option>
                  ))}
                </select>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="input"
                >
                  <option value="all">All Status</option>
                  <option value="open">Open</option>
                  <option value="assigned">Assigned</option>
                  <option value="in_progress">In Progress</option>
                  <option value="waiting">Waiting</option>
                  <option value="resolved">Resolved</option>
                </select>
              </div>

              {/* Messages List */}
              {!selectedInbox ? (
                <div className="card p-12 text-center">
                  <div className="text-muted font-mono">Select an inbox to view messages</div>
                </div>
              ) : messages.length === 0 ? (
                <div className="card p-12 text-center">
                  <div className="text-4xl mb-4">📭</div>
                  <div className="text-muted font-mono">No messages in this inbox</div>
                </div>
              ) : (
                <div className="space-y-2">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className="card p-4 hover:border-accent/30 transition-colors"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <PriorityIndicator priority={message.priority} />
                            <span className="font-mono text-sm truncate">
                              {message.subject}
                            </span>
                          </div>
                          <div className="text-xs text-muted mb-2">
                            From: {message.from_address}
                          </div>
                          <p className="text-sm text-muted truncate">{message.snippet}</p>
                          <div className="flex items-center gap-2 mt-2">
                            {message.tags.map((tag) => (
                              <span
                                key={tag}
                                className="px-1.5 py-0.5 bg-surface text-xs font-mono rounded"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="flex flex-col items-end gap-2">
                          <StatusBadge status={message.status} />
                          <span className="text-xs text-muted">
                            {formatDate(message.received_at)}
                          </span>
                          {message.assigned_to && (
                            <span className="text-xs text-accent">@ {message.assigned_to}</span>
                          )}
                        </div>
                      </div>

                      {/* Quick Actions */}
                      <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border">
                        {message.status === 'open' && (
                          <button
                            onClick={() => handleAssign(message.id, 'me')}
                            className="px-2 py-1 text-xs font-mono bg-accent/10 text-accent hover:bg-accent/20 rounded transition-colors"
                          >
                            Claim
                          </button>
                        )}
                        {message.status !== 'resolved' && (
                          <button
                            onClick={() => handleStatusChange(message.id, 'resolved')}
                            className="px-2 py-1 text-xs font-mono bg-acid-green/10 text-acid-green hover:bg-acid-green/20 rounded transition-colors"
                          >
                            Resolve
                          </button>
                        )}
                        <button className="px-2 py-1 text-xs font-mono bg-surface hover:bg-accent/10 rounded transition-colors">
                          View
                        </button>
                        <button className="px-2 py-1 text-xs font-mono bg-surface hover:bg-accent/10 rounded transition-colors">
                          Reply
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Rules View */}
          {activeView === 'rules' && (
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm text-muted font-mono">
                  {rules.length} routing rule{rules.length !== 1 ? 's' : ''}
                </span>
                <button
                  onClick={() => setShowCreateRule(true)}
                  className="btn btn-sm btn-ghost"
                >
                  + Add Rule
                </button>
              </div>

              {rules.length === 0 ? (
                <div className="card p-12 text-center">
                  <div className="text-4xl mb-4">🔀</div>
                  <div className="text-muted font-mono mb-4">No routing rules configured</div>
                  <button
                    onClick={() => setShowCreateRule(true)}
                    className="btn btn-primary"
                  >
                    Create First Rule
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {rules.map((rule) => (
                    <div
                      key={rule.id}
                      className={`card p-4 ${!rule.enabled ? 'opacity-50' : ''}`}
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-mono font-medium">{rule.name}</span>
                            <span className="text-xs text-muted">Priority: {rule.priority}</span>
                            {!rule.enabled && (
                              <span className="px-1.5 py-0.5 bg-muted/20 text-muted text-xs rounded">
                                DISABLED
                              </span>
                            )}
                          </div>
                          <div className="text-xs text-muted mb-2">
                            {rule.conditions.map((c, i) => (
                              <span key={i}>
                                {i > 0 && ' AND '}
                                {c.field} {c.operator} "{c.value}"
                              </span>
                            ))}
                          </div>
                          <div className="flex items-center gap-2">
                            {rule.actions.map((a, i) => (
                              <span
                                key={i}
                                className="px-1.5 py-0.5 bg-accent/10 text-accent text-xs font-mono rounded"
                              >
                                {a.type}
                                {a.target && `: ${a.target}`}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-xs text-muted">
                            {rule.stats.total_matches} matches
                          </div>
                          <div className="flex items-center gap-1 mt-2">
                            <button className="px-2 py-1 text-xs font-mono bg-surface hover:bg-accent/10 rounded transition-colors">
                              Edit
                            </button>
                            <button className="px-2 py-1 text-xs font-mono bg-surface hover:bg-accent/10 rounded transition-colors">
                              Test
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </PanelErrorBoundary>

        {/* Create Inbox Modal */}
        {showCreateInbox && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="card p-6 max-w-md w-full mx-4">
              <h2 className="text-lg font-mono mb-4">Create Shared Inbox</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-mono text-muted mb-1">Name *</label>
                  <input
                    type="text"
                    value={newInboxName}
                    onChange={(e) => setNewInboxName(e.target.value)}
                    placeholder="Support"
                    className="input w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-mono text-muted mb-1">Description</label>
                  <input
                    type="text"
                    value={newInboxDescription}
                    onChange={(e) => setNewInboxDescription(e.target.value)}
                    placeholder="Customer support inquiries"
                    className="input w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-mono text-muted mb-1">Email Address</label>
                  <input
                    type="email"
                    value={newInboxEmail}
                    onChange={(e) => setNewInboxEmail(e.target.value)}
                    placeholder="support@company.com"
                    className="input w-full"
                  />
                </div>
              </div>
              <div className="flex items-center justify-end gap-2 mt-6">
                <button
                  onClick={() => setShowCreateInbox(false)}
                  className="btn btn-ghost"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateInbox}
                  disabled={!newInboxName.trim()}
                  className="btn btn-primary"
                >
                  Create
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Create Rule Modal - Placeholder */}
        {showCreateRule && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="card p-6 max-w-lg w-full mx-4">
              <h2 className="text-lg font-mono mb-4">Create Routing Rule</h2>
              <p className="text-muted text-sm mb-4">
                Configure conditions and actions for automatic message routing.
              </p>
              <div className="p-4 bg-surface rounded text-center text-muted">
                Rule builder coming soon...
              </div>
              <div className="flex items-center justify-end gap-2 mt-6">
                <button
                  onClick={() => setShowCreateRule(false)}
                  className="btn btn-ghost"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-border bg-surface/50 py-4 mt-8">
        <div className="container mx-auto px-4 flex items-center justify-between text-xs text-muted font-mono">
          <span>ARAGORA SHARED INBOX</span>
          <div className="flex items-center gap-4">
            <Link href="/inbox" className="hover:text-accent">
              PERSONAL
            </Link>
            <Link href="/audit" className="hover:text-accent">
              AUDITS
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
