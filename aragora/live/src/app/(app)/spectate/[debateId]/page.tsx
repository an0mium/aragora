import SpectateClient from './SpectateClient';

// Allow runtime debate IDs in standalone/server mode.
// Static export still uses the placeholder param below.
export const dynamicParams = true;

export async function generateStaticParams() {
  // Return a placeholder so static export has at least one path.
  // Actual debate IDs are resolved client-side via useParams().
  return [{ debateId: '_' }];
}

export default function SpectateDebatePage() {
  return <SpectateClient />;
}
