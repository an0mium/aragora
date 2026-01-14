'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { useAuth } from '@/context/AuthContext';

interface User {
  id: string;
  email: string;
  name: string;
  org_id?: string;
  role: string;
  is_active: boolean;
  email_verified: boolean;
  created_at: string;
  last_login_at?: string;
}

interface UsersResponse {
  users: User[];
  total: number;
  limit: number;
  offset: number;
}

function RoleBadge({ role }: { role: string }) {
  const colors: Record<string, string> = {
    owner: 'bg-acid-magenta/20 text-acid-magenta border-acid-magenta/40',
    admin: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
    member: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    viewer: 'bg-text-muted/20 text-text-muted border-text-muted/40',
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded border ${colors[role] || colors.member}`}>
      {role.toUpperCase()}
    </span>
  );
}

function StatusBadge({ active }: { active: boolean }) {
  return active ? (
    <span className="px-2 py-0.5 text-xs font-mono rounded border bg-acid-green/20 text-acid-green border-acid-green/40">
      ACTIVE
    </span>
  ) : (
    <span className="px-2 py-0.5 text-xs font-mono rounded border bg-acid-red/20 text-acid-red border-acid-red/40">
      INACTIVE
    </span>
  );
}

export default function UsersAdminPage() {
  const { config: backendConfig } = useBackend();
  const { user, isAuthenticated, tokens } = useAuth();
  const token = tokens?.access_token;  // Extract access token;
  const [users, setUsers] = useState<User[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [roleFilter, setRoleFilter] = useState<string>('');
  const [activeOnly, setActiveOnly] = useState(false);
  const limit = 20;

  const fetchUsers = useCallback(async () => {
    if (!token) return;

    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams({
        limit: String(limit),
        offset: String(page * limit),
      });
      if (roleFilter) {
        params.set('role', roleFilter);
      }
      if (activeOnly) {
        params.set('active_only', 'true');
      }

      const res = await fetch(
        `${backendConfig.api}/api/admin/users?${params}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (!res.ok) {
        if (res.status === 403) {
          throw new Error('Admin access required');
        }
        throw new Error(`Failed to fetch users: ${res.status}`);
      }

      const data: UsersResponse = await res.json();
      setUsers(data.users);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch users');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api, token, page, roleFilter, activeOnly]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchUsers();
    }
  }, [fetchUsers, isAuthenticated]);

  const handleToggleActive = async (userId: string, currentActive: boolean) => {
    if (!token) return;

    try {
      const action = currentActive ? 'deactivate' : 'activate';
      const res = await fetch(
        `${backendConfig.api}/api/admin/users/${userId}/${action}`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (!res.ok) {
        throw new Error(`Failed to ${action} user`);
      }

      // Refresh the list
      fetchUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Action failed');
    }
  };

  const isAdmin = isAuthenticated && (user?.role === 'admin' || user?.role === 'owner');
  const totalPages = Math.ceil(total / limit);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-4">
              <Link
                href="/admin"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [ADMIN]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Sub Navigation */}
        <div className="border-b border-acid-green/20 bg-surface/40">
          <div className="container mx-auto px-4">
            <div className="flex gap-4 overflow-x-auto">
              <Link
                href="/admin"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                SYSTEM
              </Link>
              <Link
                href="/admin/organizations"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                ORGANIZATIONS
              </Link>
              <Link
                href="/admin/users"
                className="px-4 py-2 font-mono text-sm text-acid-green border-b-2 border-acid-green"
              >
                USERS
              </Link>
              <Link
                href="/admin/personas"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                PERSONAS
              </Link>
              <Link
                href="/admin/audit"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                AUDIT
              </Link>
              <Link
                href="/admin/revenue"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                REVENUE
              </Link>
              <Link
                href="/admin/training"
                className="px-4 py-2 font-mono text-sm text-text-muted hover:text-text transition-colors"
              >
                TRAINING
              </Link>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6 flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-2xl font-mono text-acid-green mb-2">
                Users
              </h1>
              <p className="text-text-muted font-mono text-sm">
                Manage user accounts and access control.
              </p>
            </div>
            <div className="flex items-center gap-4 flex-wrap">
              <select
                value={roleFilter}
                onChange={(e) => {
                  setRoleFilter(e.target.value);
                  setPage(0);
                }}
                className="bg-surface border border-acid-green/40 text-text font-mono text-sm rounded px-3 py-2"
              >
                <option value="">All Roles</option>
                <option value="owner">Owner</option>
                <option value="admin">Admin</option>
                <option value="member">Member</option>
                <option value="viewer">Viewer</option>
              </select>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={activeOnly}
                  onChange={(e) => {
                    setActiveOnly(e.target.checked);
                    setPage(0);
                  }}
                  className="w-4 h-4 accent-acid-green"
                />
                <span className="font-mono text-sm text-text-muted">Active Only</span>
              </label>
              <button
                onClick={fetchUsers}
                disabled={loading}
                className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
              >
                {loading ? 'Loading...' : 'Refresh'}
              </button>
            </div>
          </div>

          {!isAdmin && (
            <div className="card p-6 mb-6 border-acid-yellow/40">
              <div className="flex items-center gap-2 text-acid-yellow font-mono text-sm">
                <span>!</span>
                <span>Admin access required. Please sign in with an admin account.</span>
              </div>
            </div>
          )}

          {error && (
            <div className="card p-4 mb-6 border-acid-red/40 bg-acid-red/10">
              <p className="text-acid-red font-mono text-sm">{error}</p>
            </div>
          )}

          {/* Stats Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="card p-4">
              <div className="font-mono text-xs text-text-muted">Total Users</div>
              <div className="font-mono text-2xl text-acid-green">{total}</div>
            </div>
            <div className="card p-4">
              <div className="font-mono text-xs text-text-muted">On This Page</div>
              <div className="font-mono text-2xl text-text">{users.length}</div>
            </div>
            <div className="card p-4">
              <div className="font-mono text-xs text-text-muted">Current Page</div>
              <div className="font-mono text-2xl text-acid-cyan">{page + 1} / {totalPages || 1}</div>
            </div>
            <div className="card p-4">
              <div className="font-mono text-xs text-text-muted">Filter</div>
              <div className="font-mono text-lg text-text">{roleFilter || 'All'}</div>
            </div>
          </div>

          {/* Users Table */}
          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-surface border-b border-acid-green/20">
                  <tr>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">USER</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">ROLE</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">STATUS</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">VERIFIED</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">LAST LOGIN</th>
                    <th className="text-left px-4 py-3 font-mono text-xs text-text-muted">ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center">
                        <div className="font-mono text-text-muted animate-pulse">Loading...</div>
                      </td>
                    </tr>
                  )}
                  {!loading && users.length === 0 && (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center">
                        <div className="font-mono text-text-muted">No users found</div>
                      </td>
                    </tr>
                  )}
                  {!loading && users.map((u) => (
                    <tr key={u.id} className="border-b border-acid-green/10 hover:bg-surface/50">
                      <td className="px-4 py-3">
                        <div className="font-mono text-sm text-text">{u.name || u.email.split('@')[0]}</div>
                        <div className="font-mono text-xs text-acid-cyan">{u.email}</div>
                      </td>
                      <td className="px-4 py-3">
                        <RoleBadge role={u.role} />
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge active={u.is_active} />
                      </td>
                      <td className="px-4 py-3">
                        {u.email_verified ? (
                          <span className="text-acid-green font-mono text-xs">YES</span>
                        ) : (
                          <span className="text-text-muted font-mono text-xs">NO</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <div className="font-mono text-xs text-text-muted">
                          {u.last_login_at
                            ? new Date(u.last_login_at).toLocaleString()
                            : 'Never'}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {u.id !== user?.id && (
                            <button
                              onClick={() => handleToggleActive(u.id, u.is_active)}
                              className={`px-2 py-1 font-mono text-xs rounded ${
                                u.is_active
                                  ? 'text-acid-red hover:bg-acid-red/20'
                                  : 'text-acid-green hover:bg-acid-green/20'
                              }`}
                            >
                              {u.is_active ? 'DEACTIVATE' : 'ACTIVATE'}
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t border-acid-green/20">
                <button
                  onClick={() => setPage(p => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="px-3 py-1 font-mono text-sm text-acid-cyan hover:text-acid-green disabled:text-text-muted disabled:cursor-not-allowed"
                >
                  &lt; PREV
                </button>
                <span className="font-mono text-sm text-text-muted">
                  Page {page + 1} of {totalPages}
                </span>
                <button
                  onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="px-3 py-1 font-mono text-sm text-acid-cyan hover:text-acid-green disabled:text-text-muted disabled:cursor-not-allowed"
                >
                  NEXT &gt;
                </button>
              </div>
            )}
          </div>
        </div>
      </main>
    </>
  );
}
