import { Suspense } from 'react';
import { getServerClient } from '@/lib/aragora';
import DebateStream from './DebateStream';

interface PageProps {
  params: { id: string };
}

// Fetch initial debate data on server
async function getDebate(id: string) {
  const client = getServerClient();
  try {
    return await client.debates.get(id);
  } catch (error) {
    console.error('Failed to fetch debate:', error);
    return null;
  }
}

export default async function DebateDetailPage({ params }: PageProps) {
  const debate = await getDebate(params.id);

  if (!debate) {
    return (
      <div className="card">
        <h2>Debate not found</h2>
        <p style={{ color: 'var(--text-muted)' }}>
          The debate with ID "{params.id}" could not be found.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div style={{ marginBottom: '2rem' }}>
        <span className={`status-badge status-${debate.status}`}>
          {debate.status}
        </span>
        <h1 style={{ marginTop: '1rem' }}>{debate.task}</h1>
        <p style={{ color: 'var(--text-muted)' }}>
          Created {new Date(debate.created_at).toLocaleString()}
        </p>
      </div>

      <div className="grid" style={{ marginBottom: '2rem' }}>
        <div className="card">
          <h3>Agents</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
            {(debate.agents || []).map((agent: string) => (
              <span
                key={agent}
                style={{
                  padding: '0.25rem 0.75rem',
                  background: 'var(--bg)',
                  borderRadius: '9999px',
                  fontSize: '0.875rem',
                }}
              >
                {agent}
              </span>
            ))}
          </div>
        </div>

        <div className="card">
          <h3>Progress</h3>
          <p style={{ marginTop: '0.5rem' }}>
            Round {debate.current_round || 0} of {debate.total_rounds || 9}
          </p>
          <div style={{ marginTop: '0.5rem', background: 'var(--bg)', borderRadius: '9999px', height: '8px', overflow: 'hidden' }}>
            <div
              style={{
                width: `${((debate.current_round || 0) / (debate.total_rounds || 9)) * 100}%`,
                height: '100%',
                background: 'var(--primary)',
                transition: 'width 0.3s',
              }}
            />
          </div>
        </div>
      </div>

      {/* Real-time stream for running debates */}
      {debate.status === 'running' && (
        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Live Stream</h3>
          <Suspense fallback={<div>Connecting to stream...</div>}>
            <DebateStream debateId={params.id} />
          </Suspense>
        </div>
      )}

      {/* Consensus result for completed debates */}
      {debate.status === 'completed' && debate.consensus && (
        <div className="card" style={{ borderColor: 'var(--primary)' }}>
          <h3 style={{ color: 'var(--primary)', marginBottom: '1rem' }}>Consensus Reached</h3>
          <p style={{ marginBottom: '1rem' }}>{debate.consensus.decision}</p>
          <div style={{ display: 'flex', gap: '2rem' }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Confidence</span>
              <p style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                {(debate.consensus.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Agreement</span>
              <p style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                {debate.consensus.votes_for}/{debate.consensus.votes_for + debate.consensus.votes_against}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Messages history */}
      {debate.messages && debate.messages.length > 0 && (
        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Debate History</h3>
          <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
            {debate.messages.map((msg: any, idx: number) => (
              <div
                key={idx}
                style={{
                  padding: '1rem',
                  borderBottom: '1px solid var(--border)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <strong>{msg.agent}</strong>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                    Round {msg.round} - {msg.phase}
                  </span>
                </div>
                <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
