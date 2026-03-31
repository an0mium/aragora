'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useAuth } from '@/context/AuthContext';

export interface ReceiptShareModalProps {
  isOpen: boolean;
  onClose: () => void;
  receiptId: string;
  receiptSummary?: string;
  apiUrl: string;
}

interface ShareResponse {
  share_url?: string;
  expires_at?: string;
}

const EXPIRY_OPTIONS = [
  { label: '1 hour', value: 1 },
  { label: '24 hours', value: 24 },
  { label: '7 days', value: 24 * 7 },
  { label: '30 days', value: 24 * 30 },
];

function buildAbsoluteShareUrl(shareUrl: string): string {
  if (typeof window === 'undefined') return shareUrl;
  return new URL(shareUrl, window.location.origin).toString();
}

export function ReceiptShareModal({
  isOpen,
  onClose,
  receiptId,
  receiptSummary,
  apiUrl,
}: ReceiptShareModalProps) {
  const { tokens } = useAuth();
  const [expiresInHours, setExpiresInHours] = useState<number>(24);
  const [maxAccesses, setMaxAccesses] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [expiresAt, setExpiresAt] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setExpiresInHours(24);
    setMaxAccesses('');
    setIsSubmitting(false);
    setShareUrl('');
    setExpiresAt('');
    setError(null);
    setCopied(false);
  }, [isOpen]);

  const maxAccessCount = useMemo(() => {
    const trimmed = maxAccesses.trim();
    if (!trimmed) return null;
    const parsed = Number(trimmed);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : Number.NaN;
  }, [maxAccesses]);

  const copyShareUrl = useCallback(async (nextShareUrl: string) => {
    if (typeof navigator === 'undefined' || !navigator.clipboard) {
      setCopied(false);
      return;
    }

    await navigator.clipboard.writeText(nextShareUrl);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  }, []);

  const handleShare = useCallback(async () => {
    if (Number.isNaN(maxAccessCount)) {
      setError('Max accesses must be a positive whole number');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (tokens?.access_token) {
        headers.Authorization = `Bearer ${tokens.access_token}`;
      }

      const response = await fetch(`${apiUrl}/api/v2/receipts/${receiptId}/share`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          expires_in_hours: expiresInHours,
          max_accesses: maxAccessCount,
        }),
      });

      const payload = (await response.json()) as ShareResponse & { error?: string };
      if (!response.ok || !payload.share_url) {
        throw new Error(payload.error || `Failed to create share link (HTTP ${response.status})`);
      }

      const nextShareUrl = buildAbsoluteShareUrl(payload.share_url);
      setShareUrl(nextShareUrl);
      setExpiresAt(payload.expires_at || '');
      await copyShareUrl(nextShareUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create share link');
    } finally {
      setIsSubmitting(false);
    }
  }, [apiUrl, copyShareUrl, expiresInHours, maxAccessCount, receiptId, tokens?.access_token]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative w-full max-w-lg mx-4 bg-bg border border-border rounded-lg shadow-xl">
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div>
            <h2 className="text-lg font-mono font-bold text-acid-green">Share Receipt</h2>
            {receiptSummary ? (
              <p className="text-xs text-text-muted mt-1 truncate max-w-sm">{receiptSummary}</p>
            ) : null}
          </div>
          <button
            onClick={onClose}
            className="p-2 text-text-muted hover:text-white transition-colors"
            aria-label="Close share receipt modal"
          >
            x
          </button>
        </div>

        <div className="p-4 space-y-4">
          {error ? (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
              {error}
            </div>
          ) : null}

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-2 text-sm font-mono">
              <span className="text-text-muted">Link expiry</span>
              <select
                value={expiresInHours}
                onChange={(event) => setExpiresInHours(Number(event.target.value))}
                className="w-full px-3 py-2 bg-surface border border-border rounded text-text"
              >
                {EXPIRY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-2 text-sm font-mono">
              <span className="text-text-muted">Max opens (optional)</span>
              <input
                type="number"
                min="1"
                inputMode="numeric"
                value={maxAccesses}
                onChange={(event) => setMaxAccesses(event.target.value)}
                placeholder="Unlimited"
                aria-label="Max opens (optional)"
                className="w-full px-3 py-2 bg-surface border border-border rounded text-text"
              />
            </label>
          </div>

          <div className="p-3 bg-surface border border-border rounded text-xs text-text-muted space-y-1">
            <p>Creates a public receipt link using the existing signed backend share token flow.</p>
            <p>The generated link is copied automatically when creation succeeds.</p>
          </div>

          {shareUrl ? (
            <div className="space-y-2 p-3 bg-acid-green/10 border border-acid-green/30 rounded">
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm font-mono text-acid-green">
                  {copied ? 'Copied share link' : 'Share link ready'}
                </div>
                <button
                  onClick={() => void copyShareUrl(shareUrl)}
                  className="px-3 py-1 text-xs font-mono border border-acid-green/40 text-acid-green rounded hover:bg-acid-green/10"
                >
                  {copied ? 'Copied!' : 'Copy again'}
                </button>
              </div>
              <input
                readOnly
                value={shareUrl}
                aria-label="Receipt share link"
                className="w-full px-3 py-2 bg-bg border border-acid-green/20 rounded text-xs text-text"
              />
              {expiresAt ? (
                <div className="text-xs text-text-muted">
                  Expires {new Date(expiresAt).toLocaleString()}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="flex items-center justify-end gap-3 p-4 border-t border-border">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-mono border border-border rounded hover:border-acid-green/50"
          >
            Close
          </button>
          <button
            onClick={() => void handleShare()}
            disabled={isSubmitting}
            className="px-4 py-2 text-sm font-mono bg-acid-green/20 border border-acid-green text-acid-green rounded hover:bg-acid-green/30 disabled:opacity-60"
          >
            {isSubmitting ? 'Creating link...' : shareUrl ? 'Refresh link' : 'Create share link'}
          </button>
        </div>
      </div>
    </div>
  );
}
