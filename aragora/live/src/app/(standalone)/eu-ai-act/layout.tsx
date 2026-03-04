import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'EU AI Act Compliance — Aragora',
  description:
    'Classify your AI system risk level, generate compliance bundles, and prepare for EU AI Act deadlines. Interactive risk assessment with article-level guidance.',
  openGraph: {
    title: 'EU AI Act Compliance — Aragora',
    description:
      'Classify risk, generate compliance bundles, and meet EU AI Act deadlines with Aragora.',
  },
};

export default function EuAiActLayout({ children }: { children: React.ReactNode }) {
  return children;
}
