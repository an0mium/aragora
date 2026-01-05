import type { Metadata } from 'next';
import './globals.css';
import { ErrorBoundary } from '@/components/ErrorBoundary';

export const metadata: Metadata = {
  metadataBase: new URL('https://aragora.ai'),
  title: 'ARAGORA // LIVE',
  description: 'Real-time multi-agent debate system - Watch AI agents debate, evolve, and reach consensus',
  keywords: ['AI', 'multi-agent', 'debate', 'consensus', 'LLM', 'aragora'],
  authors: [{ name: 'Aragora Team' }],
  openGraph: {
    title: 'ARAGORA // LIVE',
    description: 'Real-time multi-agent debate system',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="crt-overlay crt-flicker">
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </body>
    </html>
  );
}
