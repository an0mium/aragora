'use client';

import { useState, useMemo } from 'react';

export type ChannelType = 'slack' | 'teams' | 'discord' | 'email';

export interface ChannelOption {
  type: ChannelType;
  name: string;
  icon: string;
  description: string;
  configured: boolean;
  destinations?: Array<{
    id: string;
    name: string;
    type: 'channel' | 'user' | 'email';
  }>;
}

export interface ChannelSelectorProps {
  /** Available channels */
  channels: ChannelOption[];
  /** Currently selected channel */
  selectedChannel: ChannelType | null;
  /** Selected destination within channel */
  selectedDestination: string | null;
  /** Callback when channel is selected */
  onChannelSelect: (channel: ChannelType) => void;
  /** Callback when destination is selected */
  onDestinationSelect: (destinationId: string) => void;
  /** Loading state */
  loading?: boolean;
}

/**
 * Channel selector for receipt delivery.
 */
export function ChannelSelector({
  channels,
  selectedChannel,
  selectedDestination,
  onChannelSelect,
  onDestinationSelect,
  loading = false,
}: ChannelSelectorProps) {
  const [showDestinations, setShowDestinations] = useState(false);

  const selectedChannelData = useMemo(
    () => channels.find((c) => c.type === selectedChannel),
    [channels, selectedChannel]
  );

  const handleChannelClick = (channel: ChannelOption) => {
    if (!channel.configured) return;
    onChannelSelect(channel.type);
    if (channel.destinations && channel.destinations.length > 0) {
      setShowDestinations(true);
    }
  };

  if (loading) {
    return (
      <div className="space-y-3 animate-pulse">
        <div className="grid grid-cols-2 gap-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-20 bg-surface rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Channel Grid */}
      <div className="grid grid-cols-2 gap-3">
        {channels.map((channel) => (
          <button
            key={channel.type}
            onClick={() => handleChannelClick(channel)}
            disabled={!channel.configured}
            className={`p-4 rounded-lg border text-left transition-all ${
              selectedChannel === channel.type
                ? 'border-acid-green bg-acid-green/10'
                : channel.configured
                  ? 'border-border hover:border-acid-green/50 bg-surface'
                  : 'border-border bg-surface opacity-50 cursor-not-allowed'
            }`}
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">{channel.icon}</span>
              <div>
                <div className="font-mono text-sm font-medium">
                  {channel.name}
                </div>
                <div className="text-xs text-text-muted">
                  {channel.configured ? channel.description : 'Not configured'}
                </div>
              </div>
            </div>
            {!channel.configured && (
              <div className="mt-2 text-xs text-yellow-500">
                Configure in settings
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Destination Selector */}
      {showDestinations && selectedChannelData?.destinations && (
        <div className="p-4 bg-surface rounded-lg border border-border">
          <div className="text-sm font-mono font-medium mb-3">
            Select Destination
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {selectedChannelData.destinations.map((dest) => (
              <button
                key={dest.id}
                onClick={() => onDestinationSelect(dest.id)}
                className={`w-full p-3 rounded text-left transition-colors ${
                  selectedDestination === dest.id
                    ? 'bg-acid-green/20 border border-acid-green'
                    : 'bg-bg hover:bg-surface-lighter'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-xs text-text-muted">
                    {dest.type === 'channel' ? '#' : dest.type === 'user' ? '@' : '@'}
                  </span>
                  <span className="font-mono text-sm">{dest.name}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Email Input (for email channel) */}
      {selectedChannel === 'email' && (
        <div className="p-4 bg-surface rounded-lg border border-border">
          <label className="block text-sm font-mono font-medium mb-2">
            Recipient Email
          </label>
          <input
            type="email"
            placeholder="recipient@example.com"
            onChange={(e) => onDestinationSelect(e.target.value)}
            className="w-full px-3 py-2 text-sm bg-bg border border-border rounded
                       focus:border-acid-green focus:outline-none font-mono"
          />
        </div>
      )}
    </div>
  );
}

export default ChannelSelector;
