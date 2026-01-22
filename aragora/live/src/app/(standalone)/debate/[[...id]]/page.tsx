import { Metadata } from 'next';
import { DebateViewerWrapper } from './DebateViewerWrapper';
import { fetchDebateById } from '@/utils/supabase';

// For static export with optional catch-all
export const dynamicParams = false;

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

  // Fetch debate data for dynamic metadata
  const debate = await fetchDebateById(debateId);

  if (!debate) {
    return {
      title: 'Debate Not Found | ARAGORA',
      description: 'This debate could not be found.',
    };
  }

  const agentList = debate.agents?.join(', ') || 'AI agents';
  const consensusText = debate.consensus_reached ? 'Consensus reached' : 'No consensus';
  const confidenceText = `${Math.round((debate.confidence || 0) * 100)}% confidence`;

  return {
    title: `${debate.task} | ARAGORA`,
    description: `Debate between ${agentList} - ${consensusText} (${confidenceText})`,
    openGraph: {
      title: debate.task,
      description: `AI consensus debate: ${agentList} - ${consensusText}`,
      type: 'website',
      siteName: 'ARAGORA // LIVE',
    },
    twitter: {
      card: 'summary',
      title: debate.task,
      description: `Debate between ${debate.agents?.length || 0} AI agents - ${consensusText}`,
    },
  };
}

export default function DebateViewerPage() {
  return <DebateViewerWrapper />;
}
