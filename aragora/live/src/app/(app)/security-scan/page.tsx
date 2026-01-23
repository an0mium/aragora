'use client';

import { Suspense, useState } from 'react';
import { SecurityScanWizard } from '@/components/codebase/SecurityScanWizard';
import { DependencySecurityPanel } from '@/components/codebase/DependencySecurityPanel';
import { useBackend } from '@/components/BackendSelector';
import { useAuth } from '@/context/AuthContext';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

function LoadingFallback() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 bg-[var(--surface)] rounded w-1/3" />
      <div className="h-64 bg-[var(--surface)] rounded" />
    </div>
  );
}

type TabId = 'scan' | 'dependencies';

export default function SecurityScanPage() {
  const [activeTab, setActiveTab] = useState<TabId>('scan');
  const { config: backendConfig } = useBackend();
  const { user, tokens } = useAuth();
  const userId = user?.id || 'default';

  return (
    <div className="container mx-auto px-4 py-6 max-w-5xl">
      {/* Tab Navigation */}
      <div className="flex gap-2 mb-6 border-b border-[var(--border)]">
        <button
          onClick={() => setActiveTab('scan')}
          className={`px-4 py-2 font-mono text-sm border-b-2 transition-colors ${
            activeTab === 'scan'
              ? 'border-[var(--accent)] text-[var(--accent)]'
              : 'border-transparent text-[var(--muted)] hover:text-[var(--text)]'
          }`}
        >
          Security Scan
        </button>
        <button
          onClick={() => setActiveTab('dependencies')}
          className={`px-4 py-2 font-mono text-sm border-b-2 transition-colors ${
            activeTab === 'dependencies'
              ? 'border-[var(--accent)] text-[var(--accent)]'
              : 'border-transparent text-[var(--muted)] hover:text-[var(--text)]'
          }`}
        >
          Dependencies & SBOM
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'scan' && (
        <Suspense fallback={<LoadingFallback />}>
          <SecurityScanWizard />
        </Suspense>
      )}

      {activeTab === 'dependencies' && (
        <Suspense fallback={<LoadingFallback />}>
          <PanelErrorBoundary panelName="Dependency Security">
            <DependencySecurityPanel
              apiBase={backendConfig.api}
              userId={userId}
              authToken={tokens?.access_token}
            />
          </PanelErrorBoundary>
        </Suspense>
      )}
    </div>
  );
}
