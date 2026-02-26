'use client';

import { useState, useCallback } from 'react';

interface ShareToolbarProps {
  debateId: string;
  topic: string;
}

export function ShareToolbar({ debateId, topic }: ShareToolbarProps) {
  const [copied, setCopied] = useState(false);

  const shareUrl = typeof window !== 'undefined'
    ? `${window.location.origin}/debate/${debateId}`
    : `/debate/${debateId}`;

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const input = document.createElement('input');
      input.value = shareUrl;
      document.body.appendChild(input);
      input.select();
      document.execCommand('copy');
      document.body.removeChild(input);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [shareUrl]);

  const handleDownloadImage = useCallback(async () => {
    try {
      const res = await fetch(`/api/og/${debateId}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `aragora-debate-${debateId.slice(0, 8)}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      // Silent fail
    }
  }, [debateId]);

  const tweetText = encodeURIComponent(`"${topic.slice(0, 200)}" \u2014 Multi-AI verdict on @aragora_ai`);
  const twitterUrl = `https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}&text=${tweetText}`;
  const linkedinUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleCopy}
        className="font-mono text-[10px] px-3 py-1.5 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-green)] hover:text-[var(--acid-green)] transition-colors"
        title="Copy share link"
      >
        {copied ? 'Copied!' : 'Copy link'}
      </button>
      <button
        onClick={handleDownloadImage}
        className="font-mono text-[10px] px-3 py-1.5 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-cyan)] hover:text-[var(--acid-cyan)] transition-colors"
        title="Download share image"
      >
        Image
      </button>
      <a
        href={twitterUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="font-mono text-[10px] px-3 py-1.5 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-cyan)] hover:text-[var(--acid-cyan)] transition-colors"
        title="Share on X"
      >
        X
      </a>
      <a
        href={linkedinUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="font-mono text-[10px] px-3 py-1.5 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--purple)] hover:text-[var(--purple)] transition-colors"
        title="Share on LinkedIn"
      >
        LinkedIn
      </a>
    </div>
  );
}
