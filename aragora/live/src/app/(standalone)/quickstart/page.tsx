'use client';

import Link from 'next/link';
import { useTheme } from '@/context/ThemeContext';
import { Header } from '@/components/landing/Header';
import { Footer } from '@/components/landing/Footer';
import { ConnectOpenRouterButton } from '@/components/openrouter/ConnectOpenRouterButton';

function CodeBlock({
  children,
  lang,
}: {
  children: string;
  lang?: string;
}) {
  return (
    <div style={{ borderRadius: '10px', overflow: 'hidden', border: '1px solid var(--border)' }}>
      {lang && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '8px 16px',
            backgroundColor: 'var(--surface-elevated)',
            borderBottom: '1px solid var(--border)',
            fontSize: '11px',
            fontFamily: 'var(--font-theme-data, monospace)',
            color: 'var(--text-muted)',
            textTransform: 'uppercase' as const,
            letterSpacing: '0.06em',
          }}
        >
          {lang}
        </div>
      )}
      <pre
        style={{
          margin: 0,
          padding: '16px',
          backgroundColor: 'var(--surface)',
          overflowX: 'auto',
          fontSize: '13px',
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          color: 'var(--text)',
          lineHeight: 1.7,
        }}
      >
        <code>{children}</code>
      </pre>
    </div>
  );
}

function Step({
  number,
  title,
  children,
}: {
  number: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section
      style={{
        borderRadius: '14px',
        border: '1px solid var(--border)',
        backgroundColor: 'color-mix(in srgb, var(--surface) 40%, transparent)',
        padding: '28px 32px',
        marginBottom: '24px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
        <span
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '36px',
            height: '36px',
            fontSize: '14px',
            fontWeight: 600,
            borderRadius: '50%',
            backgroundColor: 'color-mix(in srgb, var(--accent) 12%, transparent)',
            color: 'var(--accent)',
            border: '1px solid color-mix(in srgb, var(--accent) 20%, transparent)',
          }}
        >
          {number}
        </span>
        <h2
          style={{
            margin: 0,
            fontSize: '20px',
            fontWeight: 600,
            color: 'var(--text)',
            fontFamily: 'var(--font-landing)',
          }}
        >
          {title}
        </h2>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>{children}</div>
    </section>
  );
}

export default function QuickstartPage() {
  const { theme } = useTheme();

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: 'var(--bg)',
        color: 'var(--text)',
        fontFamily: 'var(--font-landing)',
      }}
      data-landing-theme={theme}
    >
      <Header />

      <main
        style={{
          maxWidth: '780px',
          margin: '0 auto',
          padding: '56px 24px 80px',
        }}
      >
        {/* Title */}
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <h1
            style={{
              fontSize: '36px',
              fontWeight: 700,
              color: 'var(--accent)',
              marginBottom: '10px',
              fontFamily: 'var(--font-landing)',
            }}
          >
            Quickstart
          </h1>
          <p
            style={{
              color: 'var(--text-muted)',
              fontSize: '18px',
              fontFamily: 'var(--font-landing)',
              margin: 0,
            }}
          >
            Get from zero to a saved decision receipt in under a minute.
          </p>
        </div>

        {/* Step 1 */}
        <Step number={1} title="Install">
          <CodeBlock lang="bash">{`pip install aragora-debate
python -m aragora --version`}</CodeBlock>
        </Step>

        {/* Step 2 */}
        <Step number={2} title="Zero-Key Demo">
          <p style={{ color: 'var(--text-muted)', margin: 0, fontFamily: 'var(--font-landing)' }}>
            No API keys needed. Run the bounded quickstart path locally and save a
            deterministic demo receipt:
          </p>
          <CodeBlock lang="bash">python -m aragora quickstart --demo --no-browser</CodeBlock>
          <p style={{ color: 'var(--text-muted)', fontSize: '14px', margin: 0, fontFamily: 'var(--font-landing)' }}>
            Quickstart reports whether the run was `demo` or `live`, saves the
            artifact to `.aragora/receipts/`, and keeps the first-run path
            bounded.
          </p>
        </Step>

        {/* Step 3 */}
        <Step number={3} title="Add Real AI Models">
          <ConnectOpenRouterButton />
          <p style={{ color: 'var(--text-muted)', margin: 0, fontFamily: 'var(--font-landing)' }}>
            Or set a provider key manually and run the same quickstart flow in
            live mode:
          </p>
          <CodeBlock lang="bash">{`export ANTHROPIC_API_KEY="sk-ant-..."   # Claude
# or
export OPENAI_API_KEY="sk-..."          # GPT`}</CodeBlock>
          <p style={{ color: 'var(--text-muted)', margin: 0, fontFamily: 'var(--font-landing)' }}>
            Then run a live debate:
          </p>
          <CodeBlock lang="bash">python -m aragora quickstart --question "Should we rewrite this service in Go?" --no-browser</CodeBlock>
        </Step>

        {/* Step 4 */}
        <Step number={4} title="Inspect The Receipt">
          <p style={{ color: 'var(--text-muted)', margin: 0, fontFamily: 'var(--font-landing)' }}>
            Quickstart writes a durable artifact for both demo and live runs.
            Inspect it or verify it before moving on:
          </p>
          <CodeBlock lang="bash">{`python -m aragora receipt inspect .aragora/receipts/quickstart-live-receipt.json
python -m aragora receipt verify .aragora/receipts/quickstart-live-receipt.json`}</CodeBlock>
        </Step>

        {/* Step 5 */}
        <Step number={5} title="Self-Host">
          <CodeBlock lang="bash">docker compose -f deploy/demo/docker-compose.yml up</CodeBlock>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
            {[
              { label: 'Landing page', url: 'localhost:3000/landing' },
              { label: 'Try route', url: 'localhost:3000/try' },
              { label: 'Playground', url: 'localhost:3000/playground' },
              { label: 'API docs', url: 'localhost:8080/api/v2/docs' },
            ].map((item) => (
              <div
                key={item.label}
                style={{
                  padding: '12px 14px',
                  borderRadius: '8px',
                  border: '1px solid var(--border)',
                  backgroundColor: 'var(--surface)',
                }}
              >
                <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--accent)' }}>
                  {item.label}
                </span>
                <span style={{ fontSize: '13px', color: 'var(--text-muted)', marginLeft: '8px', fontFamily: "'JetBrains Mono', monospace" }}>
                  {item.url}
                </span>
              </div>
            ))}
          </div>
        </Step>

        {/* Next Steps */}
        <div style={{ marginTop: '48px' }}>
          <h2
            style={{
              fontSize: '13px',
              fontWeight: 600,
              color: 'var(--text-muted)',
              textTransform: 'uppercase' as const,
              letterSpacing: '0.06em',
              marginBottom: '16px',
              fontFamily: 'var(--font-landing)',
            }}
          >
            Next Steps
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px' }}>
            <Link
              href="/try"
              style={{
                padding: '20px',
                borderRadius: '14px',
                border: '1px solid color-mix(in srgb, var(--accent) 25%, transparent)',
                backgroundColor: 'color-mix(in srgb, var(--accent) 5%, transparent)',
                textDecoration: 'none',
                display: 'block',
              }}
            >
              <span style={{ fontSize: '16px', fontWeight: 600, color: 'var(--accent)', fontFamily: 'var(--font-landing)' }}>
                Try a debate now
              </span>
              <p style={{ fontSize: '14px', color: 'var(--text-muted)', margin: '6px 0 0', fontFamily: 'var(--font-landing)' }}>
                No install needed — run in your browser
              </p>
            </Link>
            <Link
              href="/docs"
              style={{
                padding: '20px',
                borderRadius: '14px',
                border: '1px solid var(--border)',
                textDecoration: 'none',
                display: 'block',
              }}
            >
              <span style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text)', fontFamily: 'var(--font-landing)' }}>
                API Reference
              </span>
              <p style={{ fontSize: '14px', color: 'var(--text-muted)', margin: '6px 0 0', fontFamily: 'var(--font-landing)' }}>
                Full REST API documentation
              </p>
            </Link>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
