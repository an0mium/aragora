import { ImageResponse } from 'next/og';
import { NextRequest } from 'next/server';

export const runtime = 'edge';

const VERDICT_COLORS: Record<string, string> = {
  approved: '#39ff14',
  approved_with_conditions: '#ffd700',
  needs_review: '#ff8c00',
  rejected: '#ff0040',
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ debateId: string }> }
) {
  const { debateId } = await params;

  // Fetch debate data from the backend API
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

  let topic = 'AI Multi-Model Debate';
  let verdict = 'completed';
  let confidence = 0;
  let participantCount = 0;
  let roundsUsed = 0;
  let dissentCount = 0;

  try {
    const res = await fetch(`${apiBase}/api/v1/debates/${debateId}`, {
      next: { revalidate: 3600 },
    });
    if (res.ok) {
      const data = await res.json();
      topic = data.topic || data.question || topic;
      verdict = data.verdict || (data.consensus_reached ? 'approved' : 'needs_review');
      confidence = Math.round((data.confidence || 0) * 100);
      participantCount = data.participants?.length || 0;
      roundsUsed = data.rounds_used || 0;
      dissentCount = data.dissenting_views?.length || 0;
    }
  } catch {
    // Use defaults if backend is unreachable
  }

  // Truncate topic for display
  const displayTopic = topic.length > 120 ? topic.slice(0, 117) + '...' : topic;
  const verdictLabel = verdict.replace(/_/g, ' ').toUpperCase();
  const verdictColor = VERDICT_COLORS[verdict] || '#00ffff';

  return new ImageResponse(
    (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          width: '100%',
          height: '100%',
          backgroundColor: '#0a0a0a',
          padding: '60px',
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' }}>
          <span style={{ color: '#39ff14', fontSize: '24px', fontWeight: 'bold', letterSpacing: '4px' }}>
            ARAGORA
          </span>
          <span style={{ color: '#666', fontSize: '16px' }}>
            Multi-AI Verdict
          </span>
        </div>

        {/* Topic */}
        <div style={{ display: 'flex', flex: 1, flexDirection: 'column', justifyContent: 'center' }}>
          <p style={{ color: '#e0e0e0', fontSize: '36px', lineHeight: 1.3, marginBottom: '40px', fontWeight: 600 }}>
            &ldquo;{displayTopic}&rdquo;
          </p>

          {/* Verdict bar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '20px',
              padding: '20px 28px',
              borderRadius: '12px',
              border: `2px solid ${verdictColor}40`,
              backgroundColor: `${verdictColor}10`,
              marginBottom: '30px',
            }}
          >
            <span style={{ color: verdictColor, fontSize: '20px', fontWeight: 'bold', letterSpacing: '2px' }}>
              {verdictLabel}
            </span>
            {confidence > 0 && (
              <span style={{ color: '#9a9a9a', fontSize: '20px' }}>
                {confidence}% confidence
              </span>
            )}
          </div>
        </div>

        {/* Footer stats */}
        <div style={{ display: 'flex', gap: '24px', color: '#666', fontSize: '18px' }}>
          {participantCount > 0 && <span>{participantCount} AI analysts</span>}
          {roundsUsed > 0 && <span>{roundsUsed} rounds</span>}
          {dissentCount > 0 && <span>{dissentCount} dissenting views</span>}
          <span style={{ marginLeft: 'auto' }}>aragora.ai</span>
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
    }
  );
}
