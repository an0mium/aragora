import { ImageResponse } from 'next/og';
import { fetchDebate } from './fetchDebate';

export const runtime = 'nodejs';
export const alt = 'Aragora Debate Result';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default async function OGImage(
  props: { params: Promise<{ id?: string[] }> },
) {
  const params = await props.params;
  const debateId = params.id?.[0];

  // Fallback card when no debate ID or fetch fails
  if (!debateId) {
    return fallbackImage('ARAGORA', 'Multi-agent AI debate platform');
  }

  const debate = await fetchDebate(debateId);

  if (!debate) {
    return fallbackImage('ARAGORA', 'Debate not found');
  }

  const confidencePercent = Math.round(debate.confidence * 100);
  const agentCount = debate.participants.length;
  // Truncate topic for visual display
  const displayTopic =
    debate.topic.length > 120
      ? debate.topic.slice(0, 117) + '...'
      : debate.topic;
  const displayVerdict =
    debate.verdict.length > 200
      ? debate.verdict.slice(0, 197) + '...'
      : debate.verdict;

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          backgroundColor: '#0a0a0a',
          padding: '48px 56px',
          fontFamily: 'monospace',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Subtle grid overlay */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundImage:
              'linear-gradient(rgba(57,255,20,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(57,255,20,0.03) 1px, transparent 1px)',
            backgroundSize: '40px 40px',
          }}
        />

        {/* Top: branding + agent count */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div
            style={{
              fontSize: 20,
              color: '#39ff14',
              letterSpacing: '0.15em',
              fontWeight: 700,
            }}
          >
            ARAGORA
          </div>
          <div
            style={{
              fontSize: 16,
              color: '#9a9a9a',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <span
              style={{
                display: 'flex',
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: '#39ff14',
              }}
            />
            {agentCount} AGENTS
          </div>
        </div>

        {/* Middle: topic */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '24px',
            flex: 1,
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              fontSize: displayTopic.length > 80 ? 28 : 36,
              color: '#e0e0e0',
              lineHeight: 1.3,
              fontWeight: 700,
            }}
          >
            {displayTopic}
          </div>

          {/* Verdict badge */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '16px',
            }}
          >
            <div
              style={{
                padding: '8px 16px',
                border: '1px solid rgba(57,255,20,0.5)',
                backgroundColor: 'rgba(57,255,20,0.1)',
                color: '#39ff14',
                fontSize: 16,
                fontWeight: 700,
              }}
            >
              VERDICT
            </div>
            <div
              style={{
                fontSize: 16,
                color: '#9a9a9a',
                flex: 1,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {displayVerdict}
            </div>
          </div>
        </div>

        {/* Bottom: confidence bar + branding */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          {/* Confidence indicator */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
            }}
          >
            <span style={{ fontSize: 14, color: '#9a9a9a' }}>
              CONFIDENCE
            </span>
            <div
              style={{
                display: 'flex',
                width: '160px',
                height: '8px',
                backgroundColor: '#1a1a1a',
                border: '1px solid rgba(57,255,20,0.2)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${confidencePercent}%`,
                  height: '100%',
                  backgroundColor: '#39ff14',
                }}
              />
            </div>
            <span
              style={{ fontSize: 16, color: '#39ff14', fontWeight: 700 }}
            >
              {confidencePercent}%
            </span>
          </div>

          <div style={{ fontSize: 14, color: '#555' }}>
            aragora.ai
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}

/** Generic fallback OG image when no debate data is available. */
function fallbackImage(title: string, subtitle: string) {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#0a0a0a',
          fontFamily: 'monospace',
          position: 'relative',
        }}
      >
        {/* Grid */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundImage:
              'linear-gradient(rgba(57,255,20,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(57,255,20,0.03) 1px, transparent 1px)',
            backgroundSize: '40px 40px',
          }}
        />
        <div
          style={{
            fontSize: 48,
            color: '#39ff14',
            fontWeight: 700,
            letterSpacing: '0.2em',
            marginBottom: '16px',
          }}
        >
          {title}
        </div>
        <div style={{ fontSize: 20, color: '#9a9a9a' }}>
          {subtitle}
        </div>
      </div>
    ),
    { ...size },
  );
}
