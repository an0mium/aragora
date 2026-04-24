'use client';

import { StatusBadge } from '@/components/shared/StatusBadge';

import type { AgentBridgeSessionEntry } from './types';
import { formatBridgeTimestamp, truncateBridgeSessionId } from './types';

interface BridgeRoleCardProps {
  role: string;
  session: AgentBridgeSessionEntry;
}

function getHarnessVariant(harness: string) {
  const normalized = harness.toLowerCase();
  if (normalized.startsWith('claude')) {
    return 'purple' as const;
  }
  if (normalized.startsWith('codex')) {
    return 'info' as const;
  }
  if (normalized.startsWith('droid')) {
    return 'orange' as const;
  }
  return 'neutral' as const;
}

function getSessionStatusVariant(sessionStatus: AgentBridgeSessionEntry['session_status']) {
  switch (sessionStatus) {
    case 'active':
      return 'success' as const;
    case 'completed':
      return 'info' as const;
    case 'failed':
      return 'error' as const;
    default:
      return 'warning' as const;
  }
}

export function BridgeRoleCard({ role, session }: BridgeRoleCardProps) {
  return (
    <article className="rounded-xl border border-white/10 bg-white/5 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-[0.2em] text-white/35">Role</div>
          <h3 className="mt-1 text-lg font-medium text-white">{role}</h3>
          <div className="mt-1 text-sm text-white/55">{session.model}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          <StatusBadge label={session.harness} variant={getHarnessVariant(session.harness)} />
          <StatusBadge
            label={session.session_status}
            variant={getSessionStatusVariant(session.session_status)}
          />
        </div>
      </div>

      <dl className="mt-4 space-y-3 text-sm">
        <div className="flex items-start justify-between gap-4">
          <dt className="text-white/45">Session</dt>
          <dd className="font-theme-data text-right text-white/80" title={session.session_id ?? ''}>
            {truncateBridgeSessionId(session.session_id)}
          </dd>
        </div>
        <div className="flex items-start justify-between gap-4">
          <dt className="text-white/45">Last turn</dt>
          <dd className="text-right text-white/80">
            {formatBridgeTimestamp(session.last_completed_at)}
          </dd>
        </div>
        <div className="flex items-start justify-between gap-4">
          <dt className="text-white/45">Branch</dt>
          <dd className="break-all text-right text-white/70">{session.branch ?? 'n/a'}</dd>
        </div>
      </dl>
    </article>
  );
}
