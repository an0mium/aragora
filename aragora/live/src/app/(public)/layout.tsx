import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Aragora - Make Better Decisions with AI-Powered Debate',
  description:
    'Submit any decision. Watch 30+ AI agents debate. Get an audit-ready verdict with evidence chains. Free to try.',
};

/**
 * Public layout -- no AppShell, no auth wrapper.
 * Pages under (public) are accessible without authentication.
 */
export default function PublicLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
