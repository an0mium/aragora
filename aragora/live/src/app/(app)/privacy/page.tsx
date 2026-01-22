'use client';

import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ThemeToggle } from '@/components/ThemeToggle';

// Data we collect
const DATA_COLLECTED = [
  {
    category: 'Account Information',
    examples: 'Email, name, organization',
    purpose: 'Account management, authentication',
  },
  {
    category: 'Payment Information',
    examples: 'Billing address, payment method (via Stripe)',
    purpose: 'Subscription billing',
  },
  {
    category: 'Usage Data',
    examples: 'API calls, debates run, tokens consumed',
    purpose: 'Billing, analytics',
  },
  {
    category: 'Log Data',
    examples: 'IP address, browser type, timestamps',
    purpose: 'Security, troubleshooting',
  },
];

// Data retention periods
const RETENTION_PERIODS = [
  { type: 'Account Data', period: 'Duration of account + 30 days' },
  { type: 'Usage Data', period: '2 years' },
  { type: 'Audit Logs', period: '7 years' },
  { type: 'Debate Content', period: 'Configurable (default: 90 days)' },
  { type: 'Payment Records', period: '7 years' },
];

// User rights
const USER_RIGHTS = [
  {
    icon: '📥',
    title: 'Access & Portability',
    description: 'Request a copy of your data. Export debate history in JSON or CSV format.',
  },
  {
    icon: '✏️',
    title: 'Correction',
    description: 'Update account information or request correction of inaccurate data.',
  },
  {
    icon: '🗑️',
    title: 'Deletion',
    description: 'Delete your account and associated data. Data removed within 30 days.',
  },
  {
    icon: '🚫',
    title: 'Restriction',
    description: 'Opt out of marketing. Restrict processing for specific purposes.',
  },
];

// What we don't do
const DONT_DO = [
  'Sell your personal information',
  'Share data for third-party advertising',
  'Use your content to train AI models without consent',
  'Provide data to government agencies without legal process',
];

export default function PrivacyPage() {
  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/" className="text-acid-green font-mono font-bold hover:text-acid-cyan transition-colors">
              [ARAGORA]
            </Link>
            <div className="flex items-center gap-4">
              <Link
                href="/security"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [SECURITY]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Hero */}
        <section className="py-16 px-4 border-b border-acid-green/20">
          <div className="container mx-auto max-w-4xl text-center">
            <div className="text-6xl mb-6">🔒</div>
            <h1 className="text-3xl font-mono text-acid-green mb-4">Privacy Policy</h1>
            <p className="text-text-muted font-mono max-w-2xl mx-auto">
              Your privacy matters. This policy explains how we collect, use, and protect your data.
            </p>
            <p className="text-text-muted/60 font-mono text-xs mt-4">
              Effective Date: January 14, 2026 | Version 1.0.0
            </p>
          </div>
        </section>

        {/* Data Collection */}
        <section className="py-12 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Information We Collect</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {DATA_COLLECTED.map((item) => (
                <div key={item.category} className="border border-acid-green/20 p-4 bg-bg/50">
                  <h3 className="text-acid-cyan font-mono font-bold text-sm mb-2">{item.category}</h3>
                  <p className="text-text-muted text-xs font-mono mb-1">
                    <span className="text-text-muted/60">Examples:</span> {item.examples}
                  </p>
                  <p className="text-text-muted text-xs font-mono">
                    <span className="text-text-muted/60">Purpose:</span> {item.purpose}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* What We Don't Do */}
        <section className="py-12 px-4">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">What We Don&apos;t Do</h2>
            <div className="border border-acid-green/30 p-6 bg-surface/20">
              <ul className="space-y-3">
                {DONT_DO.map((item) => (
                  <li key={item} className="flex items-center gap-3 text-text font-mono text-sm">
                    <span className="text-acid-green">✗</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* Your Rights */}
        <section className="py-12 px-4 bg-surface/30">
          <div className="container mx-auto max-w-5xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Your Rights</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {USER_RIGHTS.map((right) => (
                <div key={right.title} className="border border-acid-green/20 p-5 bg-bg/50">
                  <div className="text-2xl mb-3">{right.icon}</div>
                  <h3 className="text-acid-cyan font-mono font-bold text-sm mb-2">{right.title}</h3>
                  <p className="text-text-muted text-xs font-mono">{right.description}</p>
                </div>
              ))}
            </div>
            <div className="mt-8 text-center">
              <p className="text-text-muted font-mono text-sm mb-4">
                Exercise your rights via account settings or contact us
              </p>
              <a
                href="mailto:privacy@aragora.ai"
                className="inline-flex items-center gap-2 px-6 py-2 border border-acid-green/50 text-acid-green font-mono text-sm hover:bg-acid-green/10 transition-colors"
              >
                privacy@aragora.ai
              </a>
            </div>
          </div>
        </section>

        {/* Data Retention */}
        <section className="py-12 px-4">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Data Retention</h2>
            <div className="border border-acid-green/20 overflow-hidden">
              <table className="w-full text-sm font-mono">
                <thead className="bg-surface/50">
                  <tr>
                    <th className="text-left p-3 text-acid-cyan">Data Type</th>
                    <th className="text-left p-3 text-acid-cyan">Retention Period</th>
                  </tr>
                </thead>
                <tbody>
                  {RETENTION_PERIODS.map((item, idx) => (
                    <tr key={item.type} className={idx % 2 === 0 ? 'bg-bg/30' : ''}>
                      <td className="p-3 text-text">{item.type}</td>
                      <td className="p-3 text-text-muted">{item.period}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-text-muted/60 font-mono text-xs mt-4 text-center">
              After retention periods, data is permanently deleted or anonymized.
            </p>
          </div>
        </section>

        {/* International & Compliance */}
        <section className="py-12 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Compliance</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="border border-acid-green/30 p-5 bg-bg/50">
                <h3 className="text-acid-cyan font-mono font-bold mb-3">GDPR</h3>
                <p className="text-text-muted text-xs font-mono">
                  Full compliance for EU/EEA users including lawful basis for processing,
                  data subject rights, and 72-hour breach notification.
                </p>
              </div>
              <div className="border border-acid-green/30 p-5 bg-bg/50">
                <h3 className="text-acid-cyan font-mono font-bold mb-3">CCPA</h3>
                <p className="text-text-muted text-xs font-mono">
                  California residents have rights to know, delete, and opt out.
                  We do not sell personal information.
                </p>
              </div>
              <div className="border border-acid-green/30 p-5 bg-bg/50">
                <h3 className="text-acid-cyan font-mono font-bold mb-3">SOC 2</h3>
                <p className="text-text-muted text-xs font-mono">
                  Security, Availability, Processing Integrity, and Confidentiality
                  controls audited by third party.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Data Processing */}
        <section className="py-12 px-4">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Data Processors</h2>
            <div className="space-y-3">
              {[
                { name: 'AI Providers (OpenAI, Anthropic)', purpose: 'Process debate requests', safeguard: 'Data Processing Agreements' },
                { name: 'Stripe', purpose: 'Payment processing', safeguard: 'PCI DSS compliant' },
                { name: 'AWS/GCP', purpose: 'Infrastructure hosting', safeguard: 'SOC 2 certified' },
              ].map((processor) => (
                <div key={processor.name} className="flex items-center justify-between border border-acid-green/20 p-4 bg-surface/20">
                  <div>
                    <span className="text-acid-cyan font-mono text-sm">{processor.name}</span>
                    <span className="text-text-muted font-mono text-xs ml-4">{processor.purpose}</span>
                  </div>
                  <span className="text-acid-green/70 font-mono text-xs">{processor.safeguard}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Cookies */}
        <section className="py-12 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Cookies</h2>
            <div className="border border-acid-green/20 p-6 bg-bg/50">
              <h3 className="text-acid-cyan font-mono font-bold mb-4">Essential Cookies Only</h3>
              <ul className="space-y-2 text-text-muted text-sm font-mono">
                <li className="flex items-start gap-2">
                  <span className="text-acid-green">•</span>
                  <span><strong>session_token:</strong> Authentication (session duration)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-acid-green">•</span>
                  <span><strong>csrf_token:</strong> Security (session duration)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-acid-green">•</span>
                  <span><strong>preferences:</strong> User settings (1 year)</span>
                </li>
              </ul>
              <p className="text-text-muted/60 text-xs font-mono mt-4">
                We use self-hosted analytics with no cross-site tracking and no advertising cookies.
              </p>
            </div>
          </div>
        </section>

        {/* Contact */}
        <section className="py-12 px-4">
          <div className="container mx-auto max-w-4xl">
            <h2 className="text-2xl font-mono text-acid-green mb-8 text-center">Contact Us</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="border border-acid-green/20 p-5 bg-surface/20 text-center">
                <h3 className="text-acid-cyan font-mono font-bold mb-2">Privacy Inquiries</h3>
                <a
                  href="mailto:privacy@aragora.ai"
                  className="text-acid-green font-mono text-sm hover:underline"
                >
                  privacy@aragora.ai
                </a>
                <p className="text-text-muted text-xs font-mono mt-2">Response within 3 business days</p>
              </div>
              <div className="border border-acid-green/20 p-5 bg-surface/20 text-center">
                <h3 className="text-acid-cyan font-mono font-bold mb-2">Data Protection Officer</h3>
                <a
                  href="mailto:dpo@aragora.ai"
                  className="text-acid-green font-mono text-sm hover:underline"
                >
                  dpo@aragora.ai
                </a>
                <p className="text-text-muted text-xs font-mono mt-2">For EU/EEA users</p>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-12 border-t border-acid-green/20">
          <div className="container mx-auto px-4">
            <div className="text-acid-green/50 mb-4">
              {'═'.repeat(50)}
            </div>
            <div className="flex justify-center gap-6 mb-6">
              <Link href="/" className="text-acid-cyan hover:text-acid-green transition-colors">
                Home
              </Link>
              <Link href="/security" className="text-acid-cyan hover:text-acid-green transition-colors">
                Security
              </Link>
              <Link href="/terms" className="text-acid-cyan hover:text-acid-green transition-colors">
                Terms
              </Link>
              <a href="https://status.aragora.ai" className="text-acid-cyan hover:text-acid-green transition-colors">
                Status
              </a>
              <a href="mailto:privacy@aragora.ai" className="text-acid-cyan hover:text-acid-green transition-colors">
                Contact
              </a>
            </div>
            <p className="text-text-muted mb-2">
              Your privacy is our priority.
            </p>
            <p className="text-text-muted/60">
              Last updated: January 14, 2026
            </p>
            <div className="text-acid-green/50 mt-4">
              {'═'.repeat(50)}
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
