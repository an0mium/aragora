import Link from 'next/link';

interface PublicFooterProps {
  maxWidth?: string;
}

export function PublicFooter({ maxWidth = '72rem' }: PublicFooterProps) {
  return (
    <footer className="border-t border-[var(--border)] mt-12">
      <div
        className="mx-auto flex flex-col items-center gap-4 text-sm text-[var(--text-muted)]"
        style={{ maxWidth, padding: '32px 40px' }}
      >
        <div className="flex flex-wrap justify-center gap-6">
          <Link href="/about" className="hover:text-[var(--text)] transition-colors">
            About
          </Link>
          <Link href="/pricing" className="hover:text-[var(--text)] transition-colors">
            Pricing
          </Link>
          <Link href="/docs" className="hover:text-[var(--text)] transition-colors">
            Docs
          </Link>
          <Link
            href="mailto:support@aragora.ai"
            className="hover:text-[var(--text)] transition-colors"
          >
            Support
          </Link>
        </div>
        <p style={{ fontFamily: 'var(--font-landing)' }}>AI decisions you can trust.</p>
      </div>
    </footer>
  );
}
