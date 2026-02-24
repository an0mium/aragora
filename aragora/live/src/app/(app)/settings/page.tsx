'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useSWRFetch } from '@/hooks/useSWRFetch';

const SettingsPanel = dynamic(
  () => import('@/components/settings-panel').then(m => ({ default: m.SettingsPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-96 bg-[var(--surface)] rounded" />
      </div>
    ),
  }
);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Permission {
  id: string;
  name: string;
  description?: string;
  resource: string;
  action: string;
}

interface Role {
  id: string;
  name: string;
  description?: string;
  permissions: string[];
  parent?: string | null;
  is_default?: boolean;
  user_count?: number;
}

interface RbacResponse {
  roles?: Role[];
  permissions?: Permission[];
}

// ---------------------------------------------------------------------------
// RBAC Components
// ---------------------------------------------------------------------------

function RoleHierarchy({ roles }: { roles: Role[] }) {
  // Build hierarchy tree
  const rootRoles = roles.filter((r) => !r.parent);
  const childMap = useMemo(() => {
    const map = new Map<string, Role[]>();
    roles.forEach((r) => {
      if (r.parent) {
        const children = map.get(r.parent) || [];
        children.push(r);
        map.set(r.parent, children);
      }
    });
    return map;
  }, [roles]);

  const renderRole = (role: Role, depth: number) => {
    const children = childMap.get(role.name) || childMap.get(role.id) || [];
    return (
      <div key={role.id} style={{ marginLeft: depth * 24 }}>
        <div className="flex items-center gap-2 py-2 px-3 hover:bg-[var(--surface)]/50 transition-colors">
          <span className="text-[var(--acid-green)] font-mono text-xs">
            {depth > 0 ? '\u2514\u2500 ' : '\u25B8 '}
          </span>
          <span className="font-mono text-sm text-[var(--text)]">{role.name}</span>
          {role.is_default && (
            <span className="px-1.5 py-0.5 text-[10px] font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30">
              DEFAULT
            </span>
          )}
          <span className="text-[10px] font-mono text-[var(--text-muted)] ml-auto">
            {role.permissions.length} permissions
            {role.user_count !== undefined && ` | ${role.user_count} users`}
          </span>
        </div>
        {children.map((child) => renderRole(child, depth + 1))}
      </div>
    );
  };

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)]">
      <div className="p-4 border-b border-[var(--border)]">
        <h3 className="text-sm font-mono text-[var(--acid-green)]">{'>'} ROLE HIERARCHY</h3>
      </div>
      <div className="p-4">
        {rootRoles.length === 0 ? (
          <div className="text-center py-4 text-[var(--text-muted)] font-mono text-sm">
            No roles defined
          </div>
        ) : (
          rootRoles.map((role) => renderRole(role, 0))
        )}
      </div>
    </div>
  );
}

function PermissionMatrix({ roles, permissions }: { roles: Role[]; permissions: Permission[] }) {
  // Group permissions by resource
  const resources = useMemo(() => {
    const map = new Map<string, Permission[]>();
    permissions.forEach((p) => {
      const group = map.get(p.resource) || [];
      group.push(p);
      map.set(p.resource, group);
    });
    return Array.from(map.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [permissions]);

  if (permissions.length === 0 || roles.length === 0) {
    return (
      <div className="bg-[var(--surface)] border border-[var(--border)] p-8 text-center">
        <div className="text-[var(--text-muted)] font-mono text-sm">
          No permission data available
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] overflow-hidden">
      <div className="p-4 border-b border-[var(--border)]">
        <h3 className="text-sm font-mono text-[var(--acid-green)]">{'>'} PERMISSION MATRIX</h3>
        <p className="text-[10px] font-mono text-[var(--text-muted)] mt-1">
          {permissions.length} permissions across {resources.length} resources
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[var(--border)] bg-[var(--bg)]">
              <th className="text-left p-3 font-mono text-[var(--text-muted)] sticky left-0 bg-[var(--bg)] min-w-[180px]">
                Permission
              </th>
              {roles.map((role) => (
                <th
                  key={role.id}
                  className="text-center p-3 font-mono text-[var(--text-muted)] min-w-[80px]"
                >
                  {role.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {resources.map(([resource, perms]) => (
              <>
                <tr key={`group-${resource}`} className="bg-[var(--acid-green)]/5">
                  <td
                    colSpan={roles.length + 1}
                    className="p-2 font-mono text-[10px] text-[var(--acid-green)] uppercase tracking-wider"
                  >
                    {resource}
                  </td>
                </tr>
                {perms.map((perm) => (
                  <tr
                    key={perm.id}
                    className="border-b border-[var(--border)]/50 hover:bg-[var(--surface)]/50"
                  >
                    <td className="p-3 font-mono text-[var(--text)] sticky left-0 bg-[var(--surface)]">
                      <div>{perm.name}</div>
                      {perm.description && (
                        <div className="text-[10px] text-[var(--text-muted)]">{perm.description}</div>
                      )}
                    </td>
                    {roles.map((role) => {
                      const hasPermission = role.permissions.includes(perm.id) || role.permissions.includes(perm.name);
                      return (
                        <td key={role.id} className="p-3 text-center">
                          {hasPermission ? (
                            <span className="text-[var(--acid-green)] font-mono">[+]</span>
                          ) : (
                            <span className="text-[var(--text-muted)]/30 font-mono">[-]</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RolesList({ roles }: { roles: Role[] }) {
  return (
    <div className="bg-[var(--surface)] border border-[var(--border)]">
      <div className="p-4 border-b border-[var(--border)]">
        <h3 className="text-sm font-mono text-[var(--acid-green)]">{'>'} ROLES</h3>
      </div>
      <div className="divide-y divide-[var(--border)]">
        {roles.map((role) => (
          <div key={role.id} className="p-4 hover:bg-[var(--bg)] transition-colors">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-[var(--acid-cyan)]">{role.name}</span>
                {role.is_default && (
                  <span className="px-1.5 py-0.5 text-[10px] font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30">
                    DEFAULT
                  </span>
                )}
              </div>
              <span className="text-[10px] font-mono text-[var(--text-muted)]">
                {role.permissions.length} permissions
              </span>
            </div>
            {role.description && (
              <p className="text-xs text-[var(--text-muted)] font-mono mb-2">
                {role.description}
              </p>
            )}
            <div className="flex flex-wrap gap-1">
              {role.permissions.slice(0, 8).map((p) => (
                <span
                  key={p}
                  className="px-1.5 py-0.5 text-[10px] font-mono bg-[var(--bg)] text-[var(--text-muted)] border border-[var(--border)]"
                >
                  {p}
                </span>
              ))}
              {role.permissions.length > 8 && (
                <span className="px-1.5 py-0.5 text-[10px] font-mono text-[var(--text-muted)]">
                  +{role.permissions.length - 8} more
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

type ActiveTab = 'preferences' | 'roles' | 'permissions' | 'hierarchy';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('preferences');

  // Fetch RBAC data
  const { data: rbacData, error: rbacError, isLoading: rbacLoading } = useSWRFetch<RbacResponse>(
    '/api/v1/rbac/roles',
    { refreshInterval: 120000 },
  );

  const roles: Role[] = rbacData?.roles || [];
  const permissions: Permission[] = rbacData?.permissions || [];

  const tabs: { key: ActiveTab; label: string }[] = [
    { key: 'preferences', label: 'PREFERENCES' },
    { key: 'roles', label: 'ROLES' },
    { key: 'permissions', label: 'PERMISSION MATRIX' },
    { key: 'hierarchy', label: 'HIERARCHY' },
  ];

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
        <div className="container mx-auto px-4 py-6">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-[var(--acid-green)] mb-2">
              {'>'} SETTINGS & RBAC
            </h1>
            <p className="text-[var(--text-muted)] font-mono text-sm">
              Configure preferences, manage roles, and review the permission matrix.
            </p>
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-0.5 mb-6 bg-[var(--bg)] border border-[var(--border)] p-0.5 w-fit font-mono text-xs">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-4 py-2 transition-colors ${
                  activeTab === tab.key
                    ? 'bg-[var(--acid-green)] text-[var(--bg)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--acid-green)]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* RBAC Summary Stats */}
          {activeTab !== 'preferences' && (
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="p-4 bg-[var(--surface)] border border-[var(--border)] text-center">
                <div className="text-2xl font-mono text-[var(--acid-green)]">
                  {rbacLoading ? '-' : roles.length}
                </div>
                <div className="text-[10px] font-mono text-[var(--text-muted)]">Roles</div>
              </div>
              <div className="p-4 bg-[var(--surface)] border border-[var(--border)] text-center">
                <div className="text-2xl font-mono text-[var(--acid-cyan)]">
                  {rbacLoading ? '-' : permissions.length}
                </div>
                <div className="text-[10px] font-mono text-[var(--text-muted)]">Permissions</div>
              </div>
              <div className="p-4 bg-[var(--surface)] border border-[var(--border)] text-center">
                <div className="text-2xl font-mono text-purple-400">
                  {rbacLoading
                    ? '-'
                    : new Set(permissions.map((p) => p.resource)).size}
                </div>
                <div className="text-[10px] font-mono text-[var(--text-muted)]">Resources</div>
              </div>
            </div>
          )}

          {/* Error State for RBAC */}
          {rbacError && activeTab !== 'preferences' && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 text-red-400 font-mono text-sm">
              Failed to load RBAC data. The backend may be unavailable.
            </div>
          )}

          {/* Tab Content */}
          {activeTab === 'preferences' && (
            <PanelErrorBoundary panelName="Settings">
              <SettingsPanel />
            </PanelErrorBoundary>
          )}

          {activeTab === 'roles' && (
            <PanelErrorBoundary panelName="Roles">
              {rbacLoading ? (
                <div className="animate-pulse space-y-3">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="h-20 bg-[var(--surface)] rounded" />
                  ))}
                </div>
              ) : (
                <RolesList roles={roles} />
              )}
            </PanelErrorBoundary>
          )}

          {activeTab === 'permissions' && (
            <PanelErrorBoundary panelName="Permission Matrix">
              {rbacLoading ? (
                <div className="animate-pulse">
                  <div className="h-96 bg-[var(--surface)] rounded" />
                </div>
              ) : (
                <PermissionMatrix roles={roles} permissions={permissions} />
              )}
            </PanelErrorBoundary>
          )}

          {activeTab === 'hierarchy' && (
            <PanelErrorBoundary panelName="Role Hierarchy">
              {rbacLoading ? (
                <div className="animate-pulse">
                  <div className="h-48 bg-[var(--surface)] rounded" />
                </div>
              ) : (
                <RoleHierarchy roles={roles} />
              )}
            </PanelErrorBoundary>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-[var(--acid-green)]/20 mt-8">
          <div className="text-[var(--acid-green)]/50 mb-2" aria-hidden="true">
            {'='.repeat(40)}
          </div>
          <p className="text-[var(--text-muted)]">
            {'>'} ARAGORA // SETTINGS & RBAC
          </p>
        </footer>
      </main>
    </>
  );
}
