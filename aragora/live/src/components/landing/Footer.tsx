'use client';

const USE_CASES = [
  {
    icon: '🏗️',
    title: 'Architecture Stress-Test',
    desc: 'Find the $500K flaw before launch — AI stress-tests your architecture in 30 minutes',
  },
  {
    icon: '🔐',
    title: 'Security Red-Team',
    desc: 'Adversarial AI critique catches vulnerabilities before attackers do',
  },
  {
    icon: '📋',
    title: 'Decision Receipts',
    desc: 'Audit-ready transcripts with minority views preserved for compliance',
  },
];

const FOOTER_LINKS = [
  { href: 'https://live.aragora.ai', label: 'Live Dashboard' },
  { href: 'https://status.aragora.ai', label: 'Status' },
  { href: 'https://github.com/an0mium/aragora', label: 'GitHub' },
  { href: '/about', label: 'Docs' },
  { href: '/security', label: 'Security' },
  { href: '/privacy', label: 'Privacy' },
];

export function Footer() {
  return (
    <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-auto">
      {/* Use Cases Section */}
      <div className="text-acid-green/50 mb-4">{'═'.repeat(50)}</div>
      <p className="text-acid-cyan font-bold mb-4 text-sm">
        {'>'} USE CASES
      </p>
      <div className="max-w-2xl mx-auto px-4 mb-6 space-y-3">
        {USE_CASES.map((useCase) => (
          <div key={useCase.title} className="text-left">
            <span className="text-acid-green">
              {useCase.icon} {useCase.title}
            </span>
            <p className="text-text-muted/60 text-[10px] ml-6 mt-0.5">
              {useCase.desc}
            </p>
          </div>
        ))}
      </div>

      {/* Main Footer */}
      <div className="text-acid-green/50 mb-4">{'═'.repeat(50)}</div>
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
      <div className="text-acid-green/50">{'═'.repeat(50)}</div>
    </footer>
  );
}
