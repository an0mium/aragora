import { Metadata } from 'next';
import { DebateViewerWrapper } from './DebateViewerWrapper';

// Allow runtime debate IDs in standalone/server mode.
// Static export still uses the base route param below.
export const dynamicParams = true;

export async function generateStaticParams() {
  // Only generate the base route - client handles the rest
  return [{ id: undefined }];
}

// Generate dynamic metadata for OG cards
export async function generateMetadata(
  props: { params: Promise<{ id?: string[] }> }
): Promise<Metadata> {
  const params = await props.params;
  const debateId = params.id?.[0];

  // Default metadata for base route
  if (!debateId) {
    return {
      title: 'ARAGORA Debate Viewer',
      description: 'Watch AI agents debate and reach consensus in real-time',
    };
  }

  // Keep metadata server-safe and avoid client-only dependencies.
  const shortId = debateId.slice(0, 12);

  return {
    title: `Debate ${shortId} | ARAGORA`,
    description: `Watch debate ${shortId} and follow agent reasoning in real-time.`,
    openGraph: {
      title: `Debate ${shortId}`,
      description: `ARAGORA live debate stream for ${shortId}.`,
      type: 'website',
      siteName: 'ARAGORA // LIVE',
    },
    twitter: {
      card: 'summary',
      title: `Debate ${shortId}`,
      description: `ARAGORA debate ${shortId}`,
    },
  };
}

export default function DebateViewerPage() {
  return <DebateViewerWrapper />;
}
