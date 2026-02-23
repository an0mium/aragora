'use client';

import Link from 'next/link';
import { PlaygroundDebate } from '@/components/playground/PlaygroundDebate';

export default function PlaygroundPage() {
  return (
    <main className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      {/* Nav */}
      <nav className="border-b border-[var(--border)] bg-[var(--surface)]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <Link href="/" className="font-mono text-[var(--acid-green)] font-bold text-sm tracking-wider">
            ARAGORA
          </Link>
          <div className="flex items-center gap-4">
            <Link
              href="/login"
              className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
            >
              SIGN IN
            </Link>
            <Link
              href="/signup"
              className="text-xs font-mono px-3 py-1.5 bg-[var(--acid-green)] text-[var(--bg)] hover:bg-[var(--acid-green)]/80 transition-colors font-bold"
            >
              SIGN UP FREE
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="py-12 sm:py-16 px-4">
        <div className="max-w-3xl mx-auto text-center mb-10">
          <h1 className="font-mono text-2xl sm:text-3xl text-[var(--text)] mb-4 leading-tight">
            See a debate in action.{' '}
            <span className="text-[var(--acid-green)]">No signup required.</span>
          </h1>
          <p className="font-mono text-sm text-[var(--text-muted)] max-w-xl mx-auto leading-relaxed">
            Watch 3 AI agents argue an architecture decision, critique each
            other, vote, and produce an audit-ready decision receipt.
          </p>
        </div>

        {/* Debate component */}
        <PlaygroundDebate />
      </section>

      {/* How it differs from chatbots */}
      <section className="py-12 px-4 border-t border-[var(--border)]">
        <div className="max-w-3xl mx-auto">
          <h2 className="font-mono text-lg text-center text-[var(--acid-green)] mb-8">
            {'>'} WHY DEBATE BEATS A SINGLE LLM
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              {
                title: 'ADVERSARIAL',
                desc: 'Agents are structurally incentivized to find flaws in each other\'s reasoning.',
              },
              {
                title: 'MULTI-MODEL',
                desc: 'Claude, GPT, Gemini, Mistral -- diverse architectures reduce shared blind spots.',
              },
              {
                title: 'AUDITABLE',
                desc: 'Every claim is linked to evidence. Dissenting views are preserved, never hidden.',
              },
            ].map((item) => (
              <div
                key={item.title}
                className="border border-[var(--acid-green)]/30 bg-[var(--surface)]/30 p-5"
              >
                <h3 className="font-mono text-sm text-[var(--acid-green)] mb-2">
                  {item.title}
                </h3>
                <p className="font-mono text-xs text-[var(--text-muted)] leading-relaxed">
                  {item.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-4 border-t border-[var(--acid-green)]/30 bg-[var(--surface)]/30">
        <div className="max-w-2xl mx-auto text-center space-y-6">
          <h2 className="font-mono text-xl text-[var(--text)]">
            Ready to vet your own decisions?
          </h2>
          <p className="font-mono text-sm text-[var(--text-muted)]">
            Free tier includes 10 debates/month with real AI models.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/signup"
              className="px-8 py-3 font-mono text-sm font-bold bg-[var(--acid-green)] text-[var(--bg)] hover:bg-[var(--acid-green)]/80 transition-colors"
            >
              CREATE FREE ACCOUNT
            </Link>
            <Link
              href="/"
              className="px-8 py-3 font-mono text-sm border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-green)] hover:text-[var(--acid-green)] transition-colors"
            >
              BACK TO HOME
            </Link>
          </div>
          <div className="pt-4 space-y-1">
            {[
              'No credit card required',
              'Full debate receipts with audit trail',
              '30+ AI agents across 6 providers',
            ].map((perk) => (
              <div key={perk} className="flex items-center justify-center gap-2 text-xs font-mono text-[var(--text-muted)]">
                <span className="text-[var(--acid-green)]">+</span>
                <span>{perk}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
