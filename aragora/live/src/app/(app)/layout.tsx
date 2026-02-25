import AppLayoutClient from './AppLayoutClient';

// All (app) pages are 'use client' and render entirely in the browser.
// No server-side dynamic rendering needed — static shells are fine.

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <AppLayoutClient>{children}</AppLayoutClient>;
}
