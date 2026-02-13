import SpectateClient from './SpectateClient';

// For static export - no pages pre-rendered, client-side navigation only
export const dynamicParams = false;

export async function generateStaticParams() {
  // Return a placeholder so static export has at least one path
  // Actual debate IDs are resolved client-side via useParams()
  return [{ debateId: '_' }];
}

export default function SpectateDebatePage() {
  return <SpectateClient />;
}
