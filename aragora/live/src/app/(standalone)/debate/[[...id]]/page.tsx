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

  if (!debateId) {
    return {
      title: 'ARAGORA Debate Viewer',
      description: 'Watch AI agents debate and reach consensus in real-time',
    };
  }

  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
  let topic = `Debate ${debateId.slice(0, 12)}`;
  let verdict = '';
  let confidence = 0;
  let participantCount = 0;

  try {
    const res = await fetch(`${apiBase}/api/v1/debates/${debateId}`, {
      next: { revalidate: 3600 },
    });
    if (res.ok) {
      const data = await res.json();
      topic = data.topic || data.question || topic;
      verdict = data.verdict?.replace(/_/g, ' ') || '';
      confidence = Math.round((data.confidence || 0) * 100);
      participantCount = data.participants?.length || 0;
    }
  } catch {
    // Use defaults
  }

  const truncatedTopic = topic.length > 70 ? topic.slice(0, 67) + '...' : topic;
  const description = [
    verdict && `Verdict: ${verdict}`,
    confidence && `${confidence}% confidence`,
    participantCount && `${participantCount} AI analysts`,
  ].filter(Boolean).join(' | ') || 'ARAGORA debate analysis';

  const ogImageUrl = `/api/og/${debateId}`;

  return {
    title: `ARAGORA | ${truncatedTopic}`,
    description,
    openGraph: {
      title: truncatedTopic,
      description,
      type: 'website',
      siteName: 'ARAGORA // LIVE',
      images: [{ url: ogImageUrl, width: 1200, height: 630, alt: `ARAGORA verdict: ${truncatedTopic}` }],
    },
    twitter: {
      card: 'summary_large_image',
      title: `ARAGORA | ${truncatedTopic}`,
      description,
      images: [ogImageUrl],
    },
  };
}

export default function DebateViewerPage() {
  return <DebateViewerWrapper />;
}
