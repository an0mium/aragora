import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'About — Aragora',
  description:
    'Aragora is the Decision Integrity Platform — orchestrating 43 agent types to adversarially vet decisions, then delivering audit-ready decision receipts to any channel.',
  openGraph: {
    title: 'About — Aragora',
    description:
      'The Decision Integrity Platform. Multiple AI models debate your decisions with confidence scores and full audit trails.',
  },
};

export default function AboutLayout({ children }: { children: React.ReactNode }) {
  return children;
}
