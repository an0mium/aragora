'use client';

const FOOTER_LINKS = [
  { href: 'https://aragora.ai', label: 'Live Dashboard' },
  { href: 'https://github.com/aragora', label: 'GitHub' },
  { href: '/about', label: 'API Docs' },
];

export function Footer() {
  return (
    <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-auto">
      <div className="text-acid-green/50 mb-2">{'═'.repeat(40)}</div>
      <p className="text-acid-green font-bold mb-1">
        {'>'} ARAGORA // AI RED TEAM & DECISION STRESS-TEST ENGINE
      </p>
      <p className="text-text-muted/70 text-[11px] italic max-w-md mx-auto mb-4">
        &quot;The self-evolving debate engine behind defensible decisions.&quot;
      </p>
      <div className="flex justify-center gap-4 text-text-muted/50 mb-4">
        {FOOTER_LINKS.map((link, idx) => (
          <span key={link.href} className="flex items-center gap-4">
            {idx > 0 && <span>|</span>}
            <a href={link.href} className="hover:text-acid-green transition-colors">
              {link.label}
            </a>
          </span>
        ))}
      </div>
      <div className="text-acid-green/50">{'═'.repeat(40)}</div>
    </footer>
  );
}
