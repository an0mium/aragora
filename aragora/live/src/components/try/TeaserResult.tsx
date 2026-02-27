'use client';

import Link from 'next/link';

interface TeaserResultProps {
  verdict: string;
  confidence: number;
  explanation: string;
}

export function TeaserResult({ verdict, confidence, explanation }: TeaserResultProps) {
  const confidencePercent = Math.round(confidence * 100);
  const truncated = explanation.length > 200 ? explanation.slice(0, 200) + '...' : explanation;

  return (
    <div className="border border-[var(--acid-green)]/30 bg-[var(--surface)]/50">
      {/* Result Header */}
      <div className="p-4 border-b border-[var(--acid-green)]/20">
        <div className="flex items-center justify-between mb-3">
          <span className="px-3 py-1 text-sm font-mono font-bold bg-[var(--acid-green)]/20 text-[var(--acid-green)] border border-[var(--acid-green)]/30">
            {verdict}
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-[var(--text-muted)]">CONFIDENCE</span>
            <div className="w-24 h-2 bg-[var(--bg)] border border-[var(--acid-green)]/20 overflow-hidden">
              <div
                className="h-full bg-[var(--acid-green)] transition-all duration-500"
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
            <span className="text-xs font-mono text-[var(--acid-green)]">{confidencePercent}%</span>
          </div>
        </div>
      </div>

      {/* Visible Explanation */}
      <div className="p-4 border-b border-[var(--acid-green)]/20">
        <p className="text-sm font-mono text-[var(--text)] leading-relaxed">
          {truncated}
        </p>
      </div>

      {/* Locked Sections */}
      <div className="p-4 space-y-3">
        <div className="flex items-center gap-3 p-3 bg-[var(--bg)]/50 border border-[var(--border)] opacity-60">
          <span className="text-xs font-mono text-[var(--acid-green)]">&#10003;</span>
          <div>
            <span className="text-xs font-mono text-[var(--text-muted)]">SHA-256 DECISION RECEIPT</span>
            <p className="text-xs font-mono text-[var(--text-muted)]/60">Tamper-proof audit trail with agent citations</p>
          </div>
        </div>
        <div className="flex items-center gap-3 p-3 bg-[var(--bg)]/50 border border-[var(--border)] opacity-60">
          <span className="text-xs font-mono text-[var(--acid-green)]">&#10003;</span>
          <div>
            <span className="text-xs font-mono text-[var(--text-muted)]">FULL DEBATE TRANSCRIPT</span>
            <p className="text-xs font-mono text-[var(--text-muted)]/60">Every argument, critique, and dissenting opinion</p>
          </div>
        </div>
        <div className="flex items-center gap-3 p-3 bg-[var(--bg)]/50 border border-[var(--border)] opacity-60">
          <span className="text-xs font-mono text-[var(--acid-green)]">&#10003;</span>
          <div>
            <span className="text-xs font-mono text-[var(--text-muted)]">SHAREABLE LINK</span>
            <p className="text-xs font-mono text-[var(--text-muted)]/60">Send the receipt to your team for review</p>
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="p-4 border-t border-[var(--acid-green)]/20 bg-[var(--acid-green)]/5">
        <Link
          href="/onboarding"
          className="block w-full py-3 text-center font-mono font-bold text-sm bg-[var(--acid-green)] text-[var(--bg)] hover:bg-[var(--acid-green)]/80 transition-colors"
        >
          GET FULL RECEIPTS — START FREE
        </Link>
        <p className="text-center text-xs font-mono text-[var(--text-muted)] mt-2">
          30 seconds to set up. Full audit trails, shareable receipts, team collaboration.
        </p>
        <p className="text-center text-xs font-mono text-[var(--text-muted)] mt-1">
          Already have an account?{' '}
          <Link href="/login" className="text-[var(--acid-cyan)] hover:underline">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
