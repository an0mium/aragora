'use client';

import { useEffect, useState } from 'react';
import { getClientSideClient } from '@/lib/aragora';

interface StreamEvent {
  type: string;
  agent?: string;
  content?: string;
  round?: number;
  phase?: string;
  timestamp: string;
}

export default function DebateStream({ debateId }: { debateId: string }) {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const client = getClientSideClient();
    let cleanup: (() => void) | undefined;

    async function connectStream() {
      try {
        const stream = await client.debates.stream(debateId);
        setConnected(true);

        stream.on('message', (event: any) => {
          setEvents(prev => [...prev, {
            ...event,
            timestamp: new Date().toISOString(),
          }]);
        });

        stream.on('error', (err: Error) => {
          setError(err.message);
          setConnected(false);
        });

        stream.on('close', () => {
          setConnected(false);
        });

        cleanup = () => stream.close();
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to connect');
        setConnected(false);
      }
    }

    connectStream();

    return () => {
      if (cleanup) cleanup();
    };
  }, [debateId]);

  if (error) {
    return (
      <div style={{ color: '#ef4444' }}>
        Connection error: {error}
      </div>
    );
  }

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
        <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      <div className="stream-container">
        {events.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>Waiting for events...</p>
        ) : (
          events.map((event, idx) => (
            <div key={idx} className="stream-event">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                <span style={{ color: 'var(--primary)', fontWeight: 500 }}>
                  {event.type}
                  {event.agent && ` (${event.agent})`}
                </span>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
              </div>
              {event.content && (
                <p style={{ color: 'var(--text-muted)' }}>
                  {event.content.slice(0, 200)}
                  {event.content.length > 200 ? '...' : ''}
                </p>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
