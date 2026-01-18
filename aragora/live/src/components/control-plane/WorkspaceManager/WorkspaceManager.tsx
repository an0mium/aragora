'use client';

import { useState, useMemo } from 'react';
import { WorkspaceSettings } from './WorkspaceSettings';
import { TeamAccessPanel } from './TeamAccessPanel';

export interface WorkspaceMember {
  id: string;
  name: string;
  email: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  joinedAt: string;
  lastActive?: string;
}

export interface Workspace {
  id: string;
  name: string;
  description: string;
  owner: string;
  members: WorkspaceMember[];
  createdAt: string;
  updatedAt: string;
  settings: {
    defaultVertical?: string;
    complianceFrameworks: string[];
    agentLimit: number;
    documentsQuota: number;
    documentsUsed: number;
  };
}

export interface WorkspaceManagerProps {
  workspaces?: Workspace[];
  currentWorkspaceId?: string;
  onWorkspaceSelect?: (workspace: Workspace) => void;
  onWorkspaceCreate?: (name: string, description: string) => void;
  onWorkspaceUpdate?: (workspace: Workspace) => void;
  className?: string;
}

// Mock data
const MOCK_WORKSPACES: Workspace[] = [
  {
    id: 'ws_001',
    name: 'Engineering',
    description: 'Software development team workspace',
    owner: 'admin@company.com',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-16T10:00:00Z',
    members: [
      { id: 'u1', name: 'Alice Chen', email: 'alice@company.com', role: 'owner', joinedAt: '2024-01-01T00:00:00Z', lastActive: '2024-01-16T12:00:00Z' },
      { id: 'u2', name: 'Bob Smith', email: 'bob@company.com', role: 'admin', joinedAt: '2024-01-02T00:00:00Z', lastActive: '2024-01-16T10:00:00Z' },
      { id: 'u3', name: 'Carol Jones', email: 'carol@company.com', role: 'member', joinedAt: '2024-01-05T00:00:00Z' },
    ],
    settings: {
      defaultVertical: 'software',
      complianceFrameworks: ['OWASP', 'CWE'],
      agentLimit: 10,
      documentsQuota: 10000,
      documentsUsed: 2340,
    },
  },
  {
    id: 'ws_002',
    name: 'Legal',
    description: 'Legal and compliance workspace',
    owner: 'legal@company.com',
    createdAt: '2024-01-05T00:00:00Z',
    updatedAt: '2024-01-15T14:00:00Z',
    members: [
      { id: 'u4', name: 'David Lee', email: 'david@company.com', role: 'owner', joinedAt: '2024-01-05T00:00:00Z' },
      { id: 'u5', name: 'Eve Wilson', email: 'eve@company.com', role: 'member', joinedAt: '2024-01-06T00:00:00Z' },
    ],
    settings: {
      defaultVertical: 'legal',
      complianceFrameworks: ['GDPR', 'CCPA'],
      agentLimit: 5,
      documentsQuota: 5000,
      documentsUsed: 890,
    },
  },
  {
    id: 'ws_003',
    name: 'Research',
    description: 'R&D and innovation workspace',
    owner: 'research@company.com',
    createdAt: '2024-01-10T00:00:00Z',
    updatedAt: '2024-01-14T09:00:00Z',
    members: [
      { id: 'u6', name: 'Frank Brown', email: 'frank@company.com', role: 'owner', joinedAt: '2024-01-10T00:00:00Z' },
    ],
    settings: {
      defaultVertical: 'research',
      complianceFrameworks: ['IRB'],
      agentLimit: 8,
      documentsQuota: 8000,
      documentsUsed: 1560,
    },
  },
];

type ViewMode = 'list' | 'settings' | 'team';

export function WorkspaceManager({
  workspaces = MOCK_WORKSPACES,
  currentWorkspaceId,
  onWorkspaceSelect,
  onWorkspaceUpdate,
  className = '',
}: WorkspaceManagerProps) {
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState(
    currentWorkspaceId || (workspaces.length > 0 ? workspaces[0].id : null)
  );
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [showCreateModal, setShowCreateModal] = useState(false);

  const selectedWorkspace = useMemo(() => {
    return workspaces.find((ws) => ws.id === selectedWorkspaceId);
  }, [workspaces, selectedWorkspaceId]);

  const handleWorkspaceClick = (workspace: Workspace) => {
    setSelectedWorkspaceId(workspace.id);
    onWorkspaceSelect?.(workspace);
  };

  const getUsagePercent = (used: number, quota: number) => {
    return Math.min(100, Math.round((used / quota) * 100));
  };

  const getUsageColor = (percent: number) => {
    if (percent >= 90) return 'bg-red-500';
    if (percent >= 70) return 'bg-yellow-500';
    return 'bg-acid-green';
  };

  const getRoleColor = (role: WorkspaceMember['role']) => {
    switch (role) {
      case 'owner':
        return 'text-acid-green';
      case 'admin':
        return 'text-cyan-400';
      case 'member':
        return 'text-text';
      case 'viewer':
        return 'text-text-muted';
    }
  };

  const getVerticalIcon = (vertical?: string) => {
    const icons: Record<string, string> = {
      software: '&#x1F4BB;',
      legal: '&#x2696;',
      healthcare: '&#x1F3E5;',
      accounting: '&#x1F4CA;',
      research: '&#x1F52C;',
    };
    return vertical ? icons[vertical] || '' : '';
  };

  return (
    <div className={`bg-surface border border-border rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-bg flex items-center justify-between">
        <div>
          <h3 className="text-sm font-mono font-bold text-acid-green">
            WORKSPACE MANAGER
          </h3>
          <p className="text-xs text-text-muted mt-1">
            Manage workspaces and team access
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-3 py-1.5 text-xs font-mono bg-acid-green text-bg rounded hover:bg-acid-green/80 transition-colors"
        >
          + NEW WORKSPACE
        </button>
      </div>

      {/* View Mode Tabs */}
      <div className="flex border-b border-border">
        {(['list', 'settings', 'team'] as ViewMode[]).map((mode) => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            disabled={mode !== 'list' && !selectedWorkspace}
            className={`
              px-4 py-2 text-xs font-mono uppercase transition-colors
              ${viewMode === mode ? 'text-acid-green border-b-2 border-acid-green bg-bg' : 'text-text-muted hover:text-text'}
              ${mode !== 'list' && !selectedWorkspace ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            {mode === 'list' ? 'Workspaces' : mode === 'settings' ? 'Settings' : 'Team'}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-4">
        {viewMode === 'list' && (
          <div className="space-y-3">
            {workspaces.map((workspace) => {
              const usagePercent = getUsagePercent(
                workspace.settings.documentsUsed,
                workspace.settings.documentsQuota
              );
              const isSelected = workspace.id === selectedWorkspaceId;

              return (
                <div
                  key={workspace.id}
                  onClick={() => handleWorkspaceClick(workspace)}
                  className={`
                    p-4 rounded-lg border-2 cursor-pointer transition-all
                    ${isSelected ? 'border-acid-green bg-acid-green/5' : 'border-border hover:border-text-muted bg-bg'}
                  `}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span
                        className="text-2xl"
                        dangerouslySetInnerHTML={{ __html: getVerticalIcon(workspace.settings.defaultVertical) || '&#x1F4C1;' }}
                      />
                      <div>
                        <h4 className="font-mono font-bold text-text">{workspace.name}</h4>
                        <p className="text-xs text-text-muted">{workspace.description}</p>
                      </div>
                    </div>
                    {isSelected && (
                      <span className="px-2 py-0.5 text-xs font-mono bg-acid-green/20 text-acid-green rounded">
                        ACTIVE
                      </span>
                    )}
                  </div>

                  {/* Stats Row */}
                  <div className="flex items-center gap-6 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="text-text-muted">Members:</span>
                      <span className="font-mono text-text">{workspace.members.length}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-text-muted">Agents:</span>
                      <span className="font-mono text-text">{workspace.settings.agentLimit}</span>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-text-muted">Documents:</span>
                        <span className="font-mono text-text">
                          {workspace.settings.documentsUsed.toLocaleString()} / {workspace.settings.documentsQuota.toLocaleString()}
                        </span>
                      </div>
                      <div className="h-1.5 bg-surface rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all ${getUsageColor(usagePercent)}`}
                          style={{ width: `${usagePercent}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Compliance Frameworks */}
                  {workspace.settings.complianceFrameworks.length > 0 && (
                    <div className="flex items-center gap-1 mt-3">
                      {workspace.settings.complianceFrameworks.map((fw) => (
                        <span
                          key={fw}
                          className="px-1.5 py-0.5 text-xs font-mono bg-surface border border-border rounded"
                        >
                          {fw}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {viewMode === 'settings' && selectedWorkspace && (
          <WorkspaceSettings
            workspace={selectedWorkspace}
            onSave={(updated) => {
              onWorkspaceUpdate?.(updated);
            }}
          />
        )}

        {viewMode === 'team' && selectedWorkspace && (
          <TeamAccessPanel
            workspace={selectedWorkspace}
            onMemberAdd={(email, role) => {
              console.log('Add member:', email, role);
            }}
            onMemberRemove={(memberId) => {
              console.log('Remove member:', memberId);
            }}
            onRoleChange={(memberId, role) => {
              console.log('Change role:', memberId, role);
            }}
          />
        )}
      </div>

      {/* Create Workspace Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-bg/80 flex items-center justify-center z-50">
          <div className="bg-surface border border-border rounded-lg p-6 w-full max-w-md">
            <h3 className="font-mono font-bold text-acid-green mb-4">CREATE WORKSPACE</h3>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                const form = e.target as HTMLFormElement;
                const name = (form.elements.namedItem('name') as HTMLInputElement).value;
                const description = (form.elements.namedItem('description') as HTMLTextAreaElement).value;
                console.log('Create workspace:', name, description);
                setShowCreateModal(false);
              }}
            >
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-mono text-text-muted mb-1">NAME</label>
                  <input
                    name="name"
                    type="text"
                    required
                    className="w-full px-3 py-2 bg-bg border border-border rounded font-mono text-sm focus:outline-none focus:border-acid-green"
                    placeholder="Workspace name"
                  />
                </div>
                <div>
                  <label className="block text-xs font-mono text-text-muted mb-1">DESCRIPTION</label>
                  <textarea
                    name="description"
                    rows={3}
                    className="w-full px-3 py-2 bg-bg border border-border rounded font-mono text-sm focus:outline-none focus:border-acid-green resize-none"
                    placeholder="Workspace description"
                  />
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="flex-1 px-4 py-2 text-xs font-mono border border-border rounded hover:border-text-muted transition-colors"
                >
                  CANCEL
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 text-xs font-mono bg-acid-green text-bg rounded hover:bg-acid-green/80 transition-colors"
                >
                  CREATE
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
