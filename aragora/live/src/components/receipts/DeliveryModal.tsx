'use client';

import { useState, useCallback, useEffect } from 'react';
import { ChannelSelector, type ChannelType, type ChannelOption } from './ChannelSelector';
import { useAuth } from '@/context/AuthContext';

export interface DeliveryOptions {
  /** Include full receipt details */
  includeDetails: boolean;
  /** Include vulnerability list */
  includeVulnerabilities: boolean;
  /** Include provenance chain */
  includeProvenance: boolean;
  /** Use compact format */
  compact: boolean;
  /** Custom message */
  message?: string;
}

export interface DeliveryModalProps {
  /** Whether modal is open */
  isOpen: boolean;
  /** Callback to close modal */
  onClose: () => void;
  /** Receipt ID to deliver */
  receiptId: string;
  /** Receipt summary for display */
  receiptSummary?: string;
  /** API base URL */
  apiUrl: string;
  /** Callback on successful delivery */
  onDeliverySuccess?: (channel: ChannelType, destination: string) => void;
}

// Default channels - in production, fetch from API
const DEFAULT_CHANNELS: ChannelOption[] = [
  {
    type: 'slack',
    name: 'Slack',
    icon: '?',
    description: 'Send to Slack channel',
    configured: true,
    destinations: [
      { id: 'C01234567', name: 'security-alerts', type: 'channel' },
      { id: 'C01234568', name: 'compliance', type: 'channel' },
      { id: 'C01234569', name: 'general', type: 'channel' },
    ],
  },
  {
    type: 'teams',
    name: 'Microsoft Teams',
    icon: '?',
    description: 'Send to Teams channel',
    configured: true,
    destinations: [
      { id: 'T01234567', name: 'Security Team', type: 'channel' },
      { id: 'T01234568', name: 'Compliance', type: 'channel' },
    ],
  },
  {
    type: 'discord',
    name: 'Discord',
    icon: '?',
    description: 'Send to Discord server',
    configured: false,
    destinations: [],
  },
  {
    type: 'email',
    name: 'Email',
    icon: '?',
    description: 'Send via email',
    configured: true,
    destinations: [],
  },
];

/**
 * Modal for delivering receipts to communication channels.
 */
export function DeliveryModal({
  isOpen,
  onClose,
  receiptId,
  receiptSummary,
  apiUrl,
  onDeliverySuccess,
}: DeliveryModalProps) {
  const { tokens } = useAuth();
  const [channels, setChannels] = useState<ChannelOption[]>(DEFAULT_CHANNELS);
  const [selectedChannel, setSelectedChannel] = useState<ChannelType | null>(null);
  const [selectedDestination, setSelectedDestination] = useState<string | null>(null);
  const [options, setOptions] = useState<DeliveryOptions>({
    includeDetails: true,
    includeVulnerabilities: true,
    includeProvenance: false,
    compact: false,
    message: '',
  });
  const [loading, setLoading] = useState(false);
  const [channelsLoading, setChannelsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Fetch available channels on mount
  useEffect(() => {
    const fetchChannels = async () => {
      setChannelsLoading(true);
      try {
        const headers: HeadersInit = {};
        if (tokens?.access_token) {
          headers['Authorization'] = `Bearer ${tokens.access_token}`;
        }
        const response = await fetch(`${apiUrl}/api/channels`, { headers });
        if (response.ok) {
          const data = await response.json();
          if (data.channels) {
            setChannels(data.channels);
          }
        }
      } catch {
        // Use defaults on error
      } finally {
        setChannelsLoading(false);
      }
    };

    if (isOpen) {
      fetchChannels();
      // Reset state when opening
      setSelectedChannel(null);
      setSelectedDestination(null);
      setError(null);
      setSuccess(false);
    }
  }, [isOpen, apiUrl, tokens?.access_token]);

  const handleDeliver = useCallback(async () => {
    if (!selectedChannel || !selectedDestination) {
      setError('Please select a channel and destination');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' };
      if (tokens?.access_token) {
        headers['Authorization'] = `Bearer ${tokens.access_token}`;
      }
      const response = await fetch(`${apiUrl}/api/v1/receipts/${receiptId}/deliver`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          channel_type: selectedChannel,
          destination: selectedDestination,
          options,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      setSuccess(true);
      onDeliverySuccess?.(selectedChannel, selectedDestination);

      // Close after brief success message
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delivery failed');
    } finally {
      setLoading(false);
    }
  }, [apiUrl, receiptId, selectedChannel, selectedDestination, options, onDeliverySuccess, onClose, tokens?.access_token]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-lg mx-4 bg-bg border border-border rounded-lg shadow-xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div>
            <h2 className="text-lg font-mono font-bold text-acid-green">
              Deliver Receipt
            </h2>
            {receiptSummary && (
              <p className="text-xs text-text-muted mt-1 truncate max-w-sm">
                {receiptSummary}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 text-text-muted hover:text-white transition-colors"
          >
            x
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Success Message */}
          {success && (
            <div className="p-4 bg-acid-green/20 border border-acid-green rounded-lg text-center">
              <div className="text-2xl mb-2">?</div>
              <div className="text-acid-green font-mono">
                Receipt delivered successfully!
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
              {error}
            </div>
          )}

          {!success && (
            <>
              {/* Channel Selector */}
              <div>
                <h3 className="text-sm font-mono font-medium mb-3">
                  Select Channel
                </h3>
                <ChannelSelector
                  channels={channels}
                  selectedChannel={selectedChannel}
                  selectedDestination={selectedDestination}
                  onChannelSelect={setSelectedChannel}
                  onDestinationSelect={setSelectedDestination}
                  loading={channelsLoading}
                />
              </div>

              {/* Delivery Options */}
              <div className="p-4 bg-surface rounded-lg border border-border">
                <h3 className="text-sm font-mono font-medium mb-3">
                  Delivery Options
                </h3>
                <div className="space-y-3">
                  <label className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={options.includeDetails}
                      onChange={(e) =>
                        setOptions({ ...options, includeDetails: e.target.checked })
                      }
                      className="w-4 h-4 accent-acid-green"
                    />
                    <span className="text-sm">Include full details</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={options.includeVulnerabilities}
                      onChange={(e) =>
                        setOptions({ ...options, includeVulnerabilities: e.target.checked })
                      }
                      className="w-4 h-4 accent-acid-green"
                    />
                    <span className="text-sm">Include vulnerability list</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={options.includeProvenance}
                      onChange={(e) =>
                        setOptions({ ...options, includeProvenance: e.target.checked })
                      }
                      className="w-4 h-4 accent-acid-green"
                    />
                    <span className="text-sm">Include provenance chain</span>
                  </label>
                  <label className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      checked={options.compact}
                      onChange={(e) =>
                        setOptions({ ...options, compact: e.target.checked })
                      }
                      className="w-4 h-4 accent-acid-green"
                    />
                    <span className="text-sm">Use compact format</span>
                  </label>
                </div>

                {/* Custom Message */}
                <div className="mt-4">
                  <label className="block text-sm font-mono mb-2">
                    Custom Message (optional)
                  </label>
                  <textarea
                    value={options.message}
                    onChange={(e) =>
                      setOptions({ ...options, message: e.target.value })
                    }
                    placeholder="Add a note to include with the delivery..."
                    className="w-full px-3 py-2 text-sm bg-bg border border-border rounded
                               focus:border-acid-green focus:outline-none resize-none h-20"
                  />
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        {!success && (
          <div className="flex items-center justify-end gap-3 p-4 border-t border-border">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-mono border border-border rounded
                         hover:border-acid-green/50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleDeliver}
              disabled={loading || !selectedChannel || !selectedDestination}
              className="px-4 py-2 text-sm font-mono bg-acid-green text-bg rounded
                         hover:bg-acid-green/80 transition-colors
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Delivering...' : 'Deliver Receipt'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default DeliveryModal;
