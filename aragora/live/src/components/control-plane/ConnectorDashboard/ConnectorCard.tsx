'use client';

import { useMemo } from 'react';

export type ConnectorType =
  | 'github'
  | 's3'
  | 'sharepoint'
  | 'postgresql'
  | 'mongodb'
  | 'confluence'
  | 'notion'
  | 'slack'
  | 'fhir'
  | 'gdrive';

export type ConnectorStatus = 'connected' | 'disconnected' | 'syncing' | 'error' | 'configuring';

export interface ConnectorInfo {
  id: string;
  type: ConnectorType;
  name: string;
  description: string;
  status: ConnectorStatus;
  last_sync?: string;
  items_synced?: number;
  error_message?: string;
  sync_progress?: number;
  config?: Record<string, unknown>;
}

export interface ConnectorCardProps {
  connector: ConnectorInfo;
  selected?: boolean;
  onSelect?: (connector: ConnectorInfo) => void;
  onConfigure?: (connector: ConnectorInfo) => void;
  onSync?: (connector: ConnectorInfo) => void;
  onDisconnect?: (connector: ConnectorInfo) => void;
  compact?: boolean;
}

const CONNECTOR_ICONS: Record<ConnectorType, string> = {
  github: '/icons/github.svg',
  s3: '/icons/aws.svg',
  sharepoint: '/icons/sharepoint.svg',
  postgresql: '/icons/postgresql.svg',
  mongodb: '/icons/mongodb.svg',
  confluence: '/icons/confluence.svg',
  notion: '/icons/notion.svg',
  slack: '/icons/slack.svg',
  fhir: '/icons/fhir.svg',
  gdrive: '/icons/gdrive.svg',
};

const CONNECTOR_COLORS: Record<ConnectorType, string> = {
  github: '#181717',
  s3: '#FF9900',
  sharepoint: '#0078D4',
  postgresql: '#336791',
  mongodb: '#47A248',
  confluence: '#0052CC',
  notion: '#000000',
  slack: '#4A154B',
  fhir: '#E01F3D',
  gdrive: '#4285F4',
};

const CONNECTOR_EMOJI: Record<ConnectorType, string> = {
  github: '  ',
  s3: '  ',
  sharepoint: '  ',
  postgresql: '  ',
  mongodb: '  ',
  confluence: '  ',
  notion: '  ',
  slack: '  ',
  fhir: '  ',
  gdrive: '  ',
};

/**
 * ConnectorCard component for displaying individual connector status.
 */
export function ConnectorCard({
  connector,
  selected = false,
  onSelect,
  onConfigure,
  onSync,
  onDisconnect,
  compact = false,
}: ConnectorCardProps) {
  const statusColor = useMemo(() => {
    switch (connector.status) {
      case 'connected':
        return 'text-success';
      case 'syncing':
        return 'text-acid-cyan';
      case 'disconnected':
        return 'text-text-muted';
      case 'configuring':
        return 'text-acid-yellow';
      case 'error':
        return 'text-crimson';
      default:
        return 'text-text-muted';
    }
  }, [connector.status]);

  const statusIndicator = useMemo(() => {
    switch (connector.status) {
      case 'connected':
        return 'bg-success';
      case 'syncing':
        return 'bg-acid-cyan animate-pulse';
      case 'disconnected':
        return 'bg-text-muted';
      case 'configuring':
        return 'bg-acid-yellow';
      case 'error':
        return 'bg-crimson';
      default:
        return 'bg-text-muted';
    }
  }, [connector.status]);

  const formatLastSync = (dateStr?: string) => {
    if (!dateStr) return 'Never';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  const formatItemCount = (count?: number) => {
    if (!count) return '0';
    if (count < 1000) return count.toString();
    if (count < 1000000) return `${(count / 1000).toFixed(1)}K`;
    return `${(count / 1000000).toFixed(1)}M`;
  };

  if (compact) {
    return (
      <div
        onClick={() => onSelect?.(connector)}
        className={`p-3 rounded border cursor-pointer transition-all ${
          selected
            ? 'border-acid-green bg-acid-green/10'
            : 'border-border bg-surface hover:border-acid-green/50'
        }`}
      >
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${statusIndicator}`} />
          <span className="font-mono text-sm">{connector.name}</span>
          <span className={`text-xs font-mono ml-auto ${statusColor}`}>
            {connector.status.toUpperCase()}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`card p-4 transition-all ${
        selected ? 'border-acid-green ring-1 ring-acid-green/30' : ''
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center text-xl"
            style={{ backgroundColor: `${CONNECTOR_COLORS[connector.type]}20` }}
          >
            {CONNECTOR_EMOJI[connector.type]}
          </div>
          <div>
            <h3 className="font-mono font-medium">{connector.name}</h3>
            <p className="text-xs text-text-muted">{connector.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${statusIndicator}`} />
          <span className={`text-xs font-mono uppercase ${statusColor}`}>
            {connector.status}
          </span>
        </div>
      </div>

      {/* Sync Progress */}
      {connector.status === 'syncing' && connector.sync_progress !== undefined && (
        <div className="mb-3">
          <div className="h-1.5 bg-surface rounded overflow-hidden">
            <div
              className="h-full bg-acid-cyan transition-all"
              style={{ width: `${connector.sync_progress * 100}%` }}
            />
          </div>
          <div className="text-xs text-text-muted font-mono mt-1 text-right">
            {Math.round(connector.sync_progress * 100)}%
          </div>
        </div>
      )}

      {/* Error Message */}
      {connector.status === 'error' && connector.error_message && (
        <div className="mb-3 p-2 bg-crimson/10 border border-crimson/30 rounded text-xs text-crimson font-mono">
          {connector.error_message}
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="bg-surface p-2 rounded">
          <div className="text-xs text-text-muted font-mono">Last Sync</div>
          <div className="text-sm font-mono">{formatLastSync(connector.last_sync)}</div>
        </div>
        <div className="bg-surface p-2 rounded">
          <div className="text-xs text-text-muted font-mono">Items</div>
          <div className="text-sm font-mono">{formatItemCount(connector.items_synced)}</div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={() => onConfigure?.(connector)}
          className="flex-1 px-3 py-1.5 text-xs font-mono border border-border rounded hover:border-acid-green transition-colors"
        >
          Configure
        </button>
        {connector.status === 'connected' && (
          <>
            <button
              onClick={() => onSync?.(connector)}
              className="flex-1 px-3 py-1.5 text-xs font-mono bg-acid-green/20 text-acid-green border border-acid-green/30 rounded hover:bg-acid-green/30 transition-colors"
            >
              Sync Now
            </button>
            <button
              onClick={() => onDisconnect?.(connector)}
              className="px-3 py-1.5 text-xs font-mono text-crimson border border-crimson/30 rounded hover:bg-crimson/10 transition-colors"
              title="Disconnect"
            >
              x
            </button>
          </>
        )}
        {connector.status === 'disconnected' && (
          <button
            onClick={() => onConfigure?.(connector)}
            className="flex-1 px-3 py-1.5 text-xs font-mono bg-acid-cyan/20 text-acid-cyan border border-acid-cyan/30 rounded hover:bg-acid-cyan/30 transition-colors"
          >
            Connect
          </button>
        )}
        {connector.status === 'syncing' && (
          <button
            onClick={() => onSync?.(connector)}
            className="flex-1 px-3 py-1.5 text-xs font-mono text-acid-yellow border border-acid-yellow/30 rounded hover:bg-acid-yellow/10 transition-colors"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}

export default ConnectorCard;
