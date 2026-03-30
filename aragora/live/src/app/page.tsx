import { redirect } from 'next/navigation';

/**
 * Root route fallback.
 *
 * Runtime deployments already redirect `/` to `/landing/` in next.config.js.
 * Keeping this as a server redirect avoids the client-manifest prerender bug
 * during standalone builds while preserving the same destination.
 */
export default function RootPage() {
  redirect('/landing/');
}
