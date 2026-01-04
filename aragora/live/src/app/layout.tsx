import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
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
        {children}
      </body>
    </html>
  );
}
