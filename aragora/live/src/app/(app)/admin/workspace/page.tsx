'use client';

import React, { useState, useEffect } from 'react';
import { AdminLayout } from '@/components/admin/AdminLayout';
import { WorkspaceMemberManager, WorkspaceMember, WorkspaceRole } from '@/components/admin/WorkspaceMemberManager';
import { RoleMatrixViewer, Role, Permission } from '@/components/admin/RoleMatrixViewer';
import { CostBreakdownChart, CostItem, BreakdownType, TimeRange } from '@/components/admin/CostBreakdownChart';

// Mock data for demonstration
const MOCK_ROLES: WorkspaceRole[] = [
  { id: 'owner', name: 'Owner', description: 'Full workspace control', permissions: ['*'], isDefault: false },
  { id: 'admin', name: 'Admin', description: 'Manage members and settings', permissions: ['admin:*', 'debates:*', 'knowledge:*'], isDefault: false },
  { id: 'member', name: 'Member', description: 'Create and participate in debates', permissions: ['debates:create', 'debates:read', 'knowledge:read'], isDefault: true },
  { id: 'viewer', name: 'Viewer', description: 'Read-only access', permissions: ['debates:read', 'knowledge:read'], isDefault: false },
];

const MOCK_MEMBERS: WorkspaceMember[] = [
  { id: '1', name: 'Alice Chen', email: 'alice@example.com', role: 'owner', status: 'active', joinedAt: '2024-01-15', workspaceId: 'ws1', permissions: ['*'] },
  { id: '2', name: 'Bob Smith', email: 'bob@example.com', role: 'admin', status: 'active', joinedAt: '2024-02-20', workspaceId: 'ws1', permissions: ['admin:*'] },
  { id: '3', name: 'Carol Davis', email: 'carol@example.com', role: 'member', status: 'active', joinedAt: '2024-03-10', workspaceId: 'ws1', permissions: ['debates:create'] },
  { id: '4', name: 'Dan Wilson', email: 'dan@example.com', role: 'member', status: 'pending', joinedAt: '2024-06-01', workspaceId: 'ws1', permissions: ['debates:create'] },
  { id: '5', name: 'Eve Johnson', email: 'eve@example.com', role: 'viewer', status: 'inactive', joinedAt: '2024-04-15', workspaceId: 'ws1', permissions: ['debates:read'] },
];

const MOCK_PERMISSIONS: Permission[] = [
  { id: 'debates:create', resource: 'debates', action: 'create', description: 'Create new debates' },
  { id: 'debates:read', resource: 'debates', action: 'read', description: 'View debates' },
  { id: 'debates:update', resource: 'debates', action: 'update', description: 'Edit debates' },
  { id: 'debates:delete', resource: 'debates', action: 'delete', description: 'Delete debates' },
  { id: 'knowledge:read', resource: 'knowledge', action: 'read', description: 'View knowledge base' },
  { id: 'knowledge:write', resource: 'knowledge', action: 'write', description: 'Edit knowledge base' },
  { id: 'users:read', resource: 'users', action: 'read', description: 'View user profiles' },
  { id: 'users:manage', resource: 'users', action: 'manage', description: 'Manage users' },
  { id: 'admin:settings', resource: 'admin', action: 'settings', description: 'Manage workspace settings' },
  { id: 'admin:billing', resource: 'admin', action: 'billing', description: 'Manage billing' },
  { id: 'analytics:read', resource: 'analytics', action: 'read', description: 'View analytics' },
  { id: 'workflows:create', resource: 'workflows', action: 'create', description: 'Create workflows' },
  { id: 'workflows:execute', resource: 'workflows', action: 'execute', description: 'Execute workflows' },
];

const MOCK_MATRIX_ROLES: Role[] = [
  { id: 'owner', name: 'Owner', description: 'Full access', permissions: MOCK_PERMISSIONS.map(p => p.id), isBuiltin: true },
  { id: 'admin', name: 'Admin', description: 'Administrative access', permissions: ['debates:create', 'debates:read', 'debates:update', 'debates:delete', 'knowledge:read', 'knowledge:write', 'users:read', 'users:manage', 'admin:settings', 'analytics:read'], isBuiltin: true, parentRole: 'owner' },
  { id: 'member', name: 'Member', description: 'Standard access', permissions: ['debates:create', 'debates:read', 'debates:update', 'knowledge:read', 'analytics:read', 'workflows:create', 'workflows:execute'], isBuiltin: true },
  { id: 'viewer', name: 'Viewer', description: 'Read-only', permissions: ['debates:read', 'knowledge:read', 'analytics:read'], isBuiltin: true },
];

const MOCK_COSTS: CostItem[] = [
  { id: '1', label: 'Claude Debates', cost: 1250.50, category: 'agent', subcategory: 'anthropic' },
  { id: '2', label: 'GPT-4 Analysis', cost: 890.25, category: 'agent', subcategory: 'openai' },
  { id: '3', label: 'Knowledge Ingestion', cost: 450.00, category: 'knowledge' },
  { id: '4', label: 'Workflow Execution', cost: 380.75, category: 'workflow' },
  { id: '5', label: 'Document Storage', cost: 125.00, category: 'storage' },
  { id: '6', label: 'API Requests', cost: 95.50, category: 'api' },
  { id: '7', label: 'Gemini Critiques', cost: 560.00, category: 'agent', subcategory: 'google' },
  { id: '8', label: 'Analytics Processing', cost: 220.00, category: 'analytics' },
  { id: '9', label: 'Gauntlet Runs', cost: 340.00, category: 'debate' },
  { id: '10', label: 'Evidence Collection', cost: 180.00, category: 'knowledge' },
];

type Tab = 'members' | 'roles' | 'costs';

export default function WorkspaceAdminPage() {
  const [activeTab, setActiveTab] = useState<Tab>('members');
  const [members, setMembers] = useState<WorkspaceMember[]>(MOCK_MEMBERS);
  const [loading, setLoading] = useState(false);
  const [breakdownType, setBreakdownType] = useState<BreakdownType>('feature');
  const [_timeRange, setTimeRange] = useState<TimeRange>('30d');

  // Simulate loading
  useEffect(() => {
    setLoading(true);
    const timer = setTimeout(() => setLoading(false), 500);
    return () => clearTimeout(timer);
  }, [activeTab]);

  const handleRoleChange = async (memberId: string, newRole: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    setMembers(prev =>
      prev.map(m => (m.id === memberId ? { ...m, role: newRole } : m))
    );
  };

  const handleInvite = async (email: string, role: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    const newMember: WorkspaceMember = {
      id: `new-${Date.now()}`,
      name: email.split('@')[0],
      email,
      role,
      status: 'pending',
      joinedAt: new Date().toISOString(),
      workspaceId: 'ws1',
      permissions: [],
    };
    setMembers(prev => [...prev, newMember]);
  };

  const handleRemove = async (memberId: string) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    setMembers(prev => prev.filter(m => m.id !== memberId));
  };

  const handleBulkAction = async (action: string, memberIds: string[]) => {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    if (action === 'remove') {
      setMembers(prev => prev.filter(m => !memberIds.includes(m.id)));
    } else if (action === 'deactivate') {
      setMembers(prev =>
        prev.map(m => (memberIds.includes(m.id) ? { ...m, status: 'inactive' as const } : m))
      );
    }
  };

  const tabs: { id: Tab; label: string }[] = [
    { id: 'members', label: 'MEMBERS' },
    { id: 'roles', label: 'ROLES & PERMISSIONS' },
    { id: 'costs', label: 'COST BREAKDOWN' },
  ];

  return (
    <AdminLayout title="Workspace Management">
      {/* Tab Navigation */}
      <div className="flex items-center gap-1 mb-6 border-b border-acid-green/20">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-3 font-mono text-sm transition-colors relative ${
              activeTab === tab.id
                ? 'text-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
          >
            {tab.label}
            {activeTab === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-acid-green" />
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'members' && (
        <WorkspaceMemberManager
          workspaceId="ws1"
          members={members}
          roles={MOCK_ROLES}
          loading={loading}
          onRoleChange={handleRoleChange}
          onInvite={handleInvite}
          onRemove={handleRemove}
          onBulkAction={handleBulkAction}
        />
      )}

      {activeTab === 'roles' && (
        <RoleMatrixViewer
          roles={MOCK_MATRIX_ROLES}
          permissions={MOCK_PERMISSIONS}
          loading={loading}
          groupByResource={true}
          onRoleClick={(role) => console.log('Role clicked:', role)}
          onPermissionClick={(perm) => console.log('Permission clicked:', perm)}
        />
      )}

      {activeTab === 'costs' && (
        <CostBreakdownChart
          data={MOCK_COSTS}
          title="WORKSPACE COST BREAKDOWN"
          breakdownType={breakdownType}
          onBreakdownTypeChange={setBreakdownType}
          onTimeRangeChange={setTimeRange}
          onItemClick={(item) => console.log('Cost item clicked:', item)}
          loading={loading}
        />
      )}
    </AdminLayout>
  );
}
