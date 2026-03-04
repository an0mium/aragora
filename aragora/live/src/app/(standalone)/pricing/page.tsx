'use client';

import { useTheme } from '@/context/ThemeContext';
import { Header } from '@/components/landing/Header';
import { Footer } from '@/components/landing/Footer';
import { PricingSection } from '@/components/landing/PricingSection';

const FAQ = [
  {
    q: 'Can I change plans later?',
    a: 'Yes, you can upgrade or downgrade at any time. Changes take effect immediately and billing is prorated.',
  },
  {
    q: 'Do I need to provide my own API keys?',
    a: 'Yes. Aragora never marks up LLM costs — you bring your own keys for Anthropic, OpenAI, etc. and pay providers directly.',
  },
  {
    q: 'What happens if I exceed my debate limit?',
    a: 'You\u2019ll see a prompt to upgrade. Existing debates remain accessible, but new ones are paused until the next billing cycle or an upgrade.',
  },
  {
    q: 'Do you offer refunds?',
    a: 'We offer a 7-day money-back guarantee. Contact support@aragora.ai for a full refund if you\u2019re not satisfied.',
  },
  {
    q: 'Is there a self-hosted option?',
    a: 'Enterprise plans include self-hosted and dedicated cloud deployment options with SSO, encryption, and SLA.',
  },
];

export default function PricingPage() {
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', color: 'var(--text)' }}>
      <Header />

      <PricingSection />

      {/* FAQ */}
      <section
        className="px-4"
        style={{
          paddingTop: '80px',
          paddingBottom: '80px',
          borderTop: '1px solid var(--border)',
          fontFamily: 'var(--font-landing)',
        }}
      >
        <div className="max-w-2xl mx-auto">
          <p
            className="text-center uppercase tracking-widest"
            style={{
              fontSize: isDark ? '16px' : '18px',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-landing)',
              marginBottom: '20px',
            }}
          >
            {isDark ? '> FAQ' : 'FAQ'}
          </p>

          <h2
            className="text-center"
            style={{
              fontSize: isDark ? '24px' : '28px',
              fontWeight: 600,
              color: 'var(--text)',
              fontFamily: 'var(--font-landing)',
              marginBottom: '48px',
            }}
          >
            Common questions
          </h2>

          <div className="space-y-6">
            {FAQ.map((item) => (
              <div
                key={item.q}
                style={{
                  backgroundColor: 'var(--surface)',
                  borderRadius: 'var(--radius-card)',
                  border: '1px solid var(--border)',
                  padding: '24px',
                }}
              >
                <h3
                  className="font-semibold"
                  style={{
                    fontSize: '15px',
                    color: 'var(--text)',
                    fontFamily: 'var(--font-landing)',
                    marginBottom: '8px',
                  }}
                >
                  {item.q}
                </h3>
                <p
                  style={{
                    fontSize: isDark ? '13px' : '14px',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-landing)',
                    lineHeight: '1.6',
                  }}
                >
                  {item.a}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
