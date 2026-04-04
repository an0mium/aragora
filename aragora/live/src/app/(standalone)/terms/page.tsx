'use client';

import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ThemeToggle } from '@/components/ThemeToggle';

const TERMS_SECTIONS = [
  {
    title: 'Service Access',
    body:
      'You may use Aragora only in compliance with applicable law and these terms. You are responsible for maintaining the security of your account, credentials, and any systems you connect to the platform.',
  },
  {
    title: 'Acceptable Use',
    body:
      'Do not use Aragora to violate laws, abuse third-party services, interfere with the platform, bypass security controls, or process data you are not authorized to submit. You remain responsible for the decisions and actions taken from platform outputs.',
  },
  {
    title: 'Billing and Trials',
    body:
      'Paid features, usage-based billing, and trial terms may be described in your order form or pricing page. Fees are non-refundable except where required by law or stated in a separate written agreement.',
  },
  {
    title: 'Customer Data',
    body:
      'You retain ownership of the data and prompts you submit. We process that data to provide, secure, and improve the service according to our Privacy Policy and any enterprise agreement in place with your organization.',
  },
  {
    title: 'Availability and Changes',
    body:
      'We may update, improve, or discontinue features as the platform evolves. We aim for reliable service, but Aragora is provided on an as-available basis and may occasionally be unavailable for maintenance, incidents, or third-party dependency failures.',
  },
  {
    title: 'Disclaimers and Liability',
    body:
      'Aragora provides software and generated outputs, not legal, financial, or medical advice. To the maximum extent permitted by law, Aragora disclaims implied warranties and will not be liable for indirect, incidental, special, consequential, or punitive damages.',
  },
];

export default function TermsPage() {
  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link
              href="/"
              className="text-acid-green font-mono font-bold hover:text-acid-cyan transition-colors"
            >
              [ARAGORA]
            </Link>
            <div className="flex items-center gap-4">
              <Link
                href="/privacy"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [PRIVACY]
              </Link>
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

        <section className="py-16 px-4 border-b border-acid-green/20">
          <div className="container mx-auto max-w-4xl text-center">
            <div className="text-6xl mb-6">📜</div>
            <h1 className="text-3xl font-mono text-acid-green mb-4">Terms of Service</h1>
            <p className="text-text-muted font-mono max-w-2xl mx-auto">
              These terms govern your access to Aragora and clarify the basic service,
              data, billing, and usage expectations for the platform.
            </p>
            <p className="text-text-muted/60 font-mono text-xs mt-4">
              Effective Date: April 3, 2026 | Version 1.0.0
            </p>
          </div>
        </section>

        <section className="py-12 px-4">
          <div className="container mx-auto max-w-4xl space-y-6">
            {TERMS_SECTIONS.map((section) => (
              <article
                key={section.title}
                className="border border-acid-green/20 bg-surface/20 p-6"
              >
                <h2 className="text-lg font-mono text-acid-cyan mb-3">{section.title}</h2>
                <p className="text-sm font-mono text-text-muted leading-7">{section.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="py-12 px-4 bg-surface/30">
          <div className="container mx-auto max-w-4xl grid gap-6 md:grid-cols-2">
            <div className="border border-acid-green/20 bg-bg/50 p-5">
              <h2 className="text-lg font-mono text-acid-cyan mb-3">Questions</h2>
              <p className="text-sm font-mono text-text-muted mb-4">
                For contract, billing, or service-term questions, contact the Aragora team.
              </p>
              <a
                href="mailto:legal@aragora.ai"
                className="inline-flex items-center gap-2 px-4 py-2 border border-acid-green/50 text-acid-green font-mono text-sm hover:bg-acid-green/10 transition-colors"
              >
                legal@aragora.ai
              </a>
            </div>
            <div className="border border-acid-green/20 bg-bg/50 p-5">
              <h2 className="text-lg font-mono text-acid-cyan mb-3">Related Policies</h2>
              <div className="flex flex-wrap gap-3">
                <Link
                  href="/privacy"
                  className="px-4 py-2 border border-acid-green/30 text-acid-green font-mono text-sm hover:bg-acid-green/10 transition-colors"
                >
                  Privacy Policy
                </Link>
                <Link
                  href="/security"
                  className="px-4 py-2 border border-acid-green/30 text-acid-green font-mono text-sm hover:bg-acid-green/10 transition-colors"
                >
                  Security Portal
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
