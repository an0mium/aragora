import Link from 'next/link';

export default function TryLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      {/* Minimal Header */}
      <header className="h-12 border-b border-[var(--acid-green)]/20 bg-[var(--surface)]/50 flex items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-sm font-mono font-bold text-[var(--acid-green)]">ARAGORA</span>
          <span className="text-xs font-mono text-[var(--text-muted)]">{'// LIVE'}</span>
        </Link>
        <Link
          href="/login"
          className="px-3 py-1 text-xs font-mono border border-[var(--acid-green)]/50 text-[var(--acid-green)] hover:bg-[var(--acid-green)]/10 transition-colors"
        >
          SIGN IN
        </Link>
      </header>

      <main>{children}</main>
    </div>
  );
}
