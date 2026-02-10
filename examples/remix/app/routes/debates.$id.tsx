import type { LoaderFunctionArgs, MetaFunction } from '@remix-run/node';
import { json } from '@remix-run/node';
import { useLoaderData } from '@remix-run/react';
import { getClient } from '~/aragora.server';
import { useEffect, useState } from 'react';

export const meta: MetaFunction<typeof loader> = ({ data }) => [
  { title: data?.debate ? `${data.debate.task} | Aragora` : 'Debate | Aragora' },
];

export async function loader({ params }: LoaderFunctionArgs) {
  const client = getClient();
  try {
    const debate = await client.debates.get(params.id!);
    return json({ debate });
  } catch {
    throw new Response('Debate not found', { status: 404 });
  }
}

function DebateStream({ debateId }: { debateId: string }) {
  const [events, setEvents] = useState<any[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/debate/${debateId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setEvents((prev) => [...prev, { ...data, timestamp: new Date().toISOString() }]);
      } catch {
        // ignore non-JSON messages
      }
    };

    return () => ws.close();
  }, [debateId]);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: connected ? '#22c55e' : '#ef4444',
          }}
        />
        <span style={{ fontSize: '0.875rem', opacity: 0.7 }}>
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {events.length === 0 ? (
        <p style={{ opacity: 0.6 }}>Waiting for events...</p>
      ) : (
        events.map((event, idx) => (
          <div
            key={idx}
            style={{
              padding: '0.75rem',
              borderBottom: '1px solid var(--border, #e5e7eb)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <strong>
                {event.type}
                {event.agent && ` (${event.agent})`}
              </strong>
              <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
            {event.content && (
              <p style={{ opacity: 0.8 }}>
                {event.content.slice(0, 200)}
                {event.content.length > 200 ? '...' : ''}
              </p>
            )}
          </div>
        ))
      )}
    </div>
  );
}

export default function DebateDetailPage() {
  const { debate } = useLoaderData<typeof loader>();

  return (
    <div>
      <div style={{ marginBottom: '2rem' }}>
        <span
          style={{
            padding: '0.25rem 0.75rem',
            borderRadius: '9999px',
            fontSize: '0.75rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            background: debate.status === 'completed' ? '#dcfce7' : debate.status === 'running' ? '#dbeafe' : '#f3f4f6',
            color: debate.status === 'completed' ? '#166534' : debate.status === 'running' ? '#1e40af' : '#374151',
          }}
        >
          {debate.status}
        </span>
        <h1 style={{ marginTop: '1rem' }}>{debate.task}</h1>
        <p style={{ opacity: 0.6 }}>
          Created {new Date(debate.created_at).toLocaleString()}
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
        <div style={{ padding: '1.5rem', border: '1px solid var(--border, #e5e7eb)', borderRadius: '0.5rem' }}>
          <h3>Agents</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
            {(debate.agents || []).map((agent: string) => (
              <span
                key={agent}
                style={{
                  padding: '0.25rem 0.75rem',
                  background: '#f3f4f6',
                  borderRadius: '9999px',
                  fontSize: '0.875rem',
                }}
              >
                {agent}
              </span>
            ))}
          </div>
        </div>

        <div style={{ padding: '1.5rem', border: '1px solid var(--border, #e5e7eb)', borderRadius: '0.5rem' }}>
          <h3>Progress</h3>
          <p style={{ marginTop: '0.5rem' }}>
            Round {debate.current_round || 0} of {debate.total_rounds || 9}
          </p>
          <div style={{ marginTop: '0.5rem', background: '#f3f4f6', borderRadius: '9999px', height: '8px', overflow: 'hidden' }}>
            <div
              style={{
                width: `${((debate.current_round || 0) / (debate.total_rounds || 9)) * 100}%`,
                height: '100%',
                background: '#3b82f6',
                transition: 'width 0.3s',
              }}
            />
          </div>
        </div>
      </div>

      {debate.status === 'running' && (
        <div style={{ padding: '1.5rem', border: '1px solid var(--border, #e5e7eb)', borderRadius: '0.5rem', marginBottom: '1rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Live Stream</h3>
          <DebateStream debateId={debate.debate_id} />
        </div>
      )}

      {debate.status === 'completed' && debate.consensus && (
        <div style={{ padding: '1.5rem', border: '2px solid #3b82f6', borderRadius: '0.5rem', marginBottom: '1rem' }}>
          <h3 style={{ color: '#3b82f6', marginBottom: '1rem' }}>Consensus Reached</h3>
          <p style={{ marginBottom: '1rem' }}>{debate.consensus.decision}</p>
          <div style={{ display: 'flex', gap: '2rem' }}>
            <div>
              <span style={{ opacity: 0.6 }}>Confidence</span>
              <p style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                {(debate.consensus.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <span style={{ opacity: 0.6 }}>Agreement</span>
              <p style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                {debate.consensus.votes_for}/{debate.consensus.votes_for + debate.consensus.votes_against}
              </p>
            </div>
          </div>
        </div>
      )}

      {debate.messages && debate.messages.length > 0 && (
        <div style={{ padding: '1.5rem', border: '1px solid var(--border, #e5e7eb)', borderRadius: '0.5rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Debate History</h3>
          <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
            {debate.messages.map((msg: any, idx: number) => (
              <div
                key={idx}
                style={{
                  padding: '1rem',
                  borderBottom: '1px solid var(--border, #e5e7eb)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <strong>{msg.agent}</strong>
                  <span style={{ fontSize: '0.875rem', opacity: 0.6 }}>
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
