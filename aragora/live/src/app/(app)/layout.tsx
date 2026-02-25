import AppLayoutClient from './AppLayoutClient';

// Skip static prerendering for all (app) pages.
// These pages depend on client-side auth, backend config, and browser APIs
// that are unavailable during build-time static generation.
export const dynamic = 'force-dynamic';

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <AppLayoutClient>{children}</AppLayoutClient>;
}
