import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Sign Up — Aragora',
  description:
    'Create a free Aragora account. 10 debates per month, 3 agents per debate, markdown receipts. No credit card required.',
  openGraph: {
    title: 'Sign Up — Aragora',
    description: 'Create a free account and start debating decisions with multiple AI models.',
  },
};

export default function SignupLayout({ children }: { children: React.ReactNode }) {
  return children;
}
