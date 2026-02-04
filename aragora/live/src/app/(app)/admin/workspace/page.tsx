'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { AdminLayout } from '@/components/admin/AdminLayout';
import { WorkspaceMemberManager, WorkspaceMember, WorkspaceRole } from '@/components/admin/WorkspaceMemberManager';
import { RoleMatrixViewer, Role, Permission } from '@/components/admin/RoleMatrixViewer';
import { CostBreakdownChart, CostItem, BreakdownType, TimeRange } from '@/components/admin/CostBreakdownChart';
import { useAuthenticatedFetch, useAuthFetch } from '@/hooks/useAuthenticatedFetch';
import { useAuth } from '@/context/AuthContext';

// ============================================================================
// Types for API responses
// ============================================================================

interface RBACRole {
  id: string;
  name: string;
  description: string;
  permissions: string[];
  is_default?: boolean;
  is_builtin?: boolean;
  parent_role?: string;
}

interface RBACPermission {
  id: string;
  resource: string;
  action: string;
  description: string;
}

interface WorkspaceMemberResponse {
  id: string;
  user_id: string;
  name?: string;
  email?: string;
  role: string;
  status: 'active' | 'pending' | 'inactive';
  joined_at: string;
  permissions: string[];
}

interface CostDataResponse {
  items: Array<{
    id: string;
    label: string;
    cost: number;
    category: string;
    subcategory?: string;
  }>;
  total: number;
  period: string;
}

type Tab = 'members' | 'roles' | 'costs';

export default function WorkspaceAdminPage() {
  const [activeTab, setActiveTab] = useState<Tab>('members');
  const [breakdownType, setBreakdownType] = useState<BreakdownType>('feature');
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');

  const { organization } = useAuth();
  const { authFetch } = useAuthFetch();

  // Fetch workspace ID from current context
  const workspaceId = organization?.id || 'default';

  // =========================================================================
  // API Data Fetching
  // =========================================================================

  // Fetch workspace members
  const {
    data: membersData,
    loading: membersLoading,
    error: membersError,
    refetch: refetchMembers,
  } = useAuthenticatedFetch<{ members: WorkspaceMemberResponse[] }>(
    `/api/v1/workspaces/${workspaceId}/members`,
    { defaultData: { members: [] } }
  );

  // Fetch RBAC roles
  const {
    data: rolesData,
    loading: rolesLoading,
    error: rolesError,
  } = useAuthenticatedFetch<{ roles: RBACRole[] }>(
    '/api/v1/rbac/roles',
    { defaultData: { roles: [] } }
  );

  // Fetch RBAC permissions
  const {
    data: permissionsData,
    loading: permissionsLoading,
  } = useAuthenticatedFetch<{ permissions: RBACPermission[] }>(
    '/api/v1/rbac/permissions',
    { defaultData: { permissions: [] } }
  );

  // Fetch cost breakdown
  const {
    data: costData,
    loading: costLoading,
  } = useAuthenticatedFetch<CostDataResponse>(
    `/api/v1/billing/costs?workspace_id=${workspaceId}&period=${timeRange}`,
    {
      defaultData: { items: [], total: 0, period: timeRange },
      deps: [timeRange, workspaceId],
    }
  );

  // =========================================================================
  // Transform API data to component props
  // =========================================================================

  const members: WorkspaceMember[] = useMemo(() => {
    if (!membersData?.members) return [];
    return membersData.members.map((m) => ({
      id: m.id,
      name: m.name || m.email?.split('@')[0] || 'Unknown',
      email: m.email || `${m.user_id}@workspace`,
      role: m.role,
      status: m.status,
      joinedAt: m.joined_at,
      workspaceId,
      permissions: m.permissions,
    }));
  }, [membersData, workspaceId]);

  const workspaceRoles: WorkspaceRole[] = useMemo(() => {
    if (!rolesData?.roles) return [];
    return rolesData.roles.map((r) => ({
      id: r.id,
      name: r.name,
      description: r.description,
      permissions: r.permissions,
      isDefault: r.is_default || false,
    }));
  }, [rolesData]);

  const matrixRoles: Role[] = useMemo(() => {
    if (!rolesData?.roles) return [];
    return rolesData.roles.map((r) => ({
      id: r.id,
      name: r.name,
      description: r.description,
      permissions: r.permissions,
      isBuiltin: r.is_builtin || false,
      parentRole: r.parent_role,
    }));
  }, [rolesData]);

  const permissions: Permission[] = useMemo(() => {
    if (!permissionsData?.permissions) return [];
    return permissionsData.permissions.map((p) => ({
      id: p.id,
      resource: p.resource,
      action: p.action,
      description: p.description,
    }));
  }, [permissionsData]);

  const costItems: CostItem[] = useMemo(() => {
    if (!costData?.items) return [];
    return costData.items.map((c) => ({
      id: c.id,
      label: c.label,
      cost: c.cost,
      category: c.category,
      subcategory: c.subcategory,
    }));
  }, [costData]);

  // =========================================================================
  // Member actions (use real API calls)
  // =========================================================================

  const handleRoleChange = async (memberId: string, newRole: string) => {
    try {
      await authFetch(`/api/v1/workspaces/${workspaceId}/members/${memberId}/role`, {
        method: 'PUT',
        body: JSON.stringify({ role: newRole }),
      });
      await refetchMembers();
    } catch (error) {
      console.error('Failed to change role:', error);
    }
  };

  const handleInvite = async (email: string, role: string) => {
    try {
      await authFetch(`/api/v1/workspaces/${workspaceId}/invites`, {
        method: 'POST',
        body: JSON.stringify({ email, role }),
      });
      await refetchMembers();
    } catch (error) {
      console.error('Failed to invite member:', error);
    }
  };

  const handleRemove = async (memberId: string) => {
    try {
      await authFetch(`/api/v1/workspaces/${workspaceId}/members/${memberId}`, {
        method: 'DELETE',
      });
      await refetchMembers();
    } catch (error) {
      console.error('Failed to remove member:', error);
    }
  };

  const handleBulkAction = async (action: string, memberIds: string[]) => {
    try {
      await authFetch(`/api/v1/workspaces/${workspaceId}/members/bulk`, {
        method: 'POST',
        body: JSON.stringify({ action, member_ids: memberIds }),
      });
      await refetchMembers();
    } catch (error) {
      console.error('Failed to perform bulk action:', error);
    }
  };

  // =========================================================================
  // Loading states
  // =========================================================================

  const loading = activeTab === 'members'
    ? membersLoading
    : activeTab === 'roles'
    ? rolesLoading || permissionsLoading
    : costLoading;

  const error = membersError || rolesError;

  const tabs: { id: Tab; label: string }[] = [
    { id: 'members', label: 'MEMBERS' },
    { id: 'roles', label: 'ROLES & PERMISSIONS' },
    { id: 'costs', label: 'COST BREAKDOWN' },
  ];

  return (
    <AdminLayout title="Workspace Management">
      {/* Error Banner */}
      {error && (
        <div className="mb-4 p-3 border border-red-500/30 bg-red-500/10 text-red-400 text-sm font-mono">
          {error}
        </div>
      )}

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
        <>
          {members.length === 0 && !loading && (
            <div className="p-8 text-center border border-acid-green/20 bg-surface/30">
              <p className="text-text-muted text-sm font-mono mb-2">No members found</p>
              <p className="text-text-muted/60 text-xs font-mono">
                Invite team members to collaborate on debates and decisions.
              </p>
            </div>
          )}
          {(members.length > 0 || loading) && (
            <WorkspaceMemberManager
              workspaceId={workspaceId}
              members={members}
              roles={workspaceRoles}
              loading={loading}
              onRoleChange={handleRoleChange}
              onInvite={handleInvite}
              onRemove={handleRemove}
              onBulkAction={handleBulkAction}
            />
          )}
        </>
      )}

      {activeTab === 'roles' && (
        <>
          {matrixRoles.length === 0 && !loading && (
            <div className="p-8 text-center border border-acid-green/20 bg-surface/30">
              <p className="text-text-muted text-sm font-mono mb-2">No roles configured</p>
              <p className="text-text-muted/60 text-xs font-mono">
                Role-based access control is not configured for this workspace.
              </p>
            </div>
          )}
          {(matrixRoles.length > 0 || loading) && (
            <RoleMatrixViewer
              roles={matrixRoles}
              permissions={permissions}
              loading={loading}
              groupByResource={true}
              onRoleClick={(role) => console.log('Role clicked:', role)}
              onPermissionClick={(perm) => console.log('Permission clicked:', perm)}
            />
          )}
        </>
      )}

      {activeTab === 'costs' && (
        <>
          {costItems.length === 0 && !loading && (
            <div className="p-8 text-center border border-acid-green/20 bg-surface/30">
              <p className="text-text-muted text-sm font-mono mb-2">No cost data available</p>
              <p className="text-text-muted/60 text-xs font-mono">
                Cost tracking will appear once debates and workflows are executed.
              </p>
            </div>
          )}
          {(costItems.length > 0 || loading) && (
            <CostBreakdownChart
              data={costItems}
              title="WORKSPACE COST BREAKDOWN"
              breakdownType={breakdownType}
              onBreakdownTypeChange={setBreakdownType}
              onTimeRangeChange={setTimeRange}
              onItemClick={(item) => console.log('Cost item clicked:', item)}
              loading={loading}
            />
          )}
        </>
      )}
    </AdminLayout>
  );
}
