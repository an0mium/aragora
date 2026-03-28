'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useTheme } from '@/context/ThemeContext';
import { useBackend } from '@/components/BackendSelector';
import { LiveDebateStream } from '@/components/debate/LiveDebateStream';
import { useDebateWebSocket } from '@/hooks/useDebateWebSocket';
import type { DebateConnectionStatus } from '@/hooks/debate-websocket/types';

const DEMO_TOPICS = [
  'Should we migrate our monolith to microservices this quarter?',
  'Should the product team prioritize reliability fixes over new growth experiments?',
  'Should we standardize on one frontier model or route across multiple providers?',
];

const DEMO_AGENT_FALLBACK = ['anthropic-api', 'openai-api', 'gemini'];

function buildPlaygroundDebateUrl(apiBase: string): string {
  return apiBase === '' ? '/api/v1/playground/debate/' : `${apiBase}/api/v1/playground/debate`;
}

interface LiveBootstrapResponse {
  id?: string;
  debate_id?: string;
  topic?: string;
  question?: string;
  error?: string;
  message?: string;
}

export function LiveDemoSection() {
  const { theme } = useTheme();
  const { config: backendConfig } = useBackend();

  const [debateId, setDebateId] = useState<string | null>(null);
  const [currentTopic, setCurrentTopic] = useState(DEMO_TOPICS[0]);
  const [queuedTopic, setQueuedTopic] = useState<string | null>(null);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [isBootstrapping, setIsBootstrapping] = useState(false);

  const autoStartRef = useRef(true);
  const mountedRef = useRef(true);
  const launchRequestRef = useRef(0);

  const {
    status,
    error,
    errorDetails,
    task,
    agents,
    messages,
    streamingMessages,
    streamEvents,
    reconnectAttempt,
    connectionQuality,
    isPolling,
    reconnect,
  } = useDebateWebSocket({
    debateId: debateId ?? 'landing-live-demo',
    wsUrl: backendConfig.ws,
    enabled: Boolean(debateId),
  });

  const liveIsActive =
    debateId !== null && (status === 'connecting' || status === 'streaming' || status === 'polling');

  const launchDebate = useCallback(async (topic: string) => {
    if (!mountedRef.current) return;

    if (isBootstrapping || liveIsActive) {
      setQueuedTopic(topic);
      return;
    }

    const requestId = launchRequestRef.current + 1;
    launchRequestRef.current = requestId;

    setIsBootstrapping(true);
    setBootstrapError(null);
    setQueuedTopic(null);
    setCurrentTopic(topic);
    setDebateId(null);

    try {
      const response = await fetch(buildPlaygroundDebateUrl(backendConfig.api), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          question: topic,
          rounds: 3,
          agents: 3,
          source: 'landing-live-demo',
          stream: true,
        }),
      });

      const payload = (await response.json().catch(() => null)) as LiveBootstrapResponse | null;

      if (!response.ok) {
        throw new Error(
          payload?.error || payload?.message || `Failed to start live demo (${response.status})`
        );
      }

      const nextDebateId = payload?.id || payload?.debate_id;
      if (!nextDebateId) {
        throw new Error('Live demo started without a debate identifier.');
      }

      if (!mountedRef.current || launchRequestRef.current !== requestId) {
        return;
      }

      setDebateId(nextDebateId);
      setCurrentTopic(payload?.topic || payload?.question || topic);
    } catch (launchError) {
      if (!mountedRef.current || launchRequestRef.current !== requestId) {
        return;
      }

      setBootstrapError(
        launchError instanceof Error
          ? launchError.message
          : 'Failed to start the live landing-page debate.'
      );
    } finally {
      if (mountedRef.current && launchRequestRef.current === requestId) {
        setIsBootstrapping(false);
      }
    }
  }, [backendConfig.api, isBootstrapping, liveIsActive]);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!autoStartRef.current) return;
    autoStartRef.current = false;
    void launchDebate(DEMO_TOPICS[0]);
  }, [launchDebate]);

  useEffect(() => {
    if (!queuedTopic || isBootstrapping) return;
    if (status !== 'complete' && status !== 'error' && debateId !== null) return;

    const timer = window.setTimeout(() => {
      void launchDebate(queuedTopic);
    }, 700);

    return () => window.clearTimeout(timer);
  }, [debateId, isBootstrapping, launchDebate, queuedTopic, status]);

  const handleQueueTopic = (topic: string) => {
    if (topic === currentTopic && liveIsActive) return;

    if (isBootstrapping || liveIsActive) {
      setQueuedTopic(topic);
      return;
    }

    void launchDebate(topic);
  };

  const handleReconnect = () => {
    if (debateId && (status === 'connecting' || status === 'streaming' || status === 'polling')) {
      reconnect();
      return;
    }

    void launchDebate(queuedTopic || currentTopic);
  };

  const viewerStatus: DebateConnectionStatus = isBootstrapping
    ? 'connecting'
    : debateId
      ? status
      : bootstrapError
        ? 'error'
        : 'idle';

  const viewerError = bootstrapError || error;
  const viewerTask = task || currentTopic;
  const viewerAgents = agents.length > 0 ? agents : DEMO_AGENT_FALLBACK;
  const themeLabel = theme === 'dark' ? 'Signal Mode' : 'Studio Mode';

  return (
    <section
      className="px-4"
      style={{
        paddingTop: '96px',
        paddingBottom: '96px',
        borderTop: '1px solid var(--border)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      <div className="mx-auto max-w-6xl">
        <div className="mb-10 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div className="max-w-2xl">
            <p
              className="uppercase tracking-widest"
              style={{
                fontSize: '12px',
                color: 'var(--accent)',
                marginBottom: '12px',
              }}
            >
              Live Debate Feed
            </p>
            <h2
              style={{
                fontSize: 'clamp(28px, 4vw, 42px)',
                lineHeight: '1.05',
                color: 'var(--text)',
                fontFamily: 'var(--font-display, var(--font-landing))',
                marginBottom: '12px',
              }}
            >
              Watch agents argue in public.
            </h2>
            <p
              style={{
                fontSize: '15px',
                lineHeight: '1.7',
                color: 'var(--text-muted)',
                maxWidth: '56ch',
              }}
            >
              The landing page now bootstraps a real debate, attaches a live WebSocket feed, and
              keeps the next topic queued so spectators can stay in the stream.
            </p>
          </div>

          <div
            className="grid gap-3 sm:grid-cols-3"
            style={{ minWidth: 'min(100%, 420px)' }}
          >
            <MetricCard
              label="Connection"
              value={
                viewerStatus === 'streaming'
                  ? 'Live'
                  : viewerStatus === 'polling'
                    ? 'Polling'
                    : viewerStatus === 'complete'
                      ? 'Complete'
                      : viewerStatus === 'error'
                        ? 'Error'
                        : 'Booting'
              }
            />
            <MetricCard label="Queued" value={queuedTopic ? '1 next topic' : 'None'} />
            <MetricCard label="Theme" value={themeLabel} />
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
          <div
            style={{
              backgroundColor: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-card)',
              overflow: 'hidden',
              boxShadow: 'var(--shadow-card)',
            }}
          >
            <div
              className="flex flex-wrap items-center gap-3 border-b px-4 py-3"
              style={{ borderColor: 'var(--border)' }}
            >
              <span
                className="inline-flex items-center gap-2 px-3 py-1"
                style={{
                  border: '1px solid var(--border)',
                  borderRadius: '999px',
                  fontSize: '11px',
                  color: 'var(--text)',
                }}
              >
                <span
                  className={viewerStatus === 'streaming' ? 'animate-pulse' : undefined}
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor:
                      viewerStatus === 'streaming'
                        ? 'var(--accent)'
                        : viewerStatus === 'error'
                          ? '#ef4444'
                          : 'var(--text-muted)',
                  }}
                />
                {debateId ? `Debate ${debateId}` : 'Bootstrapping live debate'}
              </span>

              {queuedTopic && (
                <span
                  className="px-3 py-1"
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: '999px',
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                  }}
                >
                  Queued next: {queuedTopic}
                </span>
              )}
            </div>

            <LiveDebateStream
              status={viewerStatus}
              error={viewerError}
              errorDetails={bootstrapError ? 'Queue another topic or retry the current debate.' : errorDetails}
              task={viewerTask}
              agents={viewerAgents}
              messages={messages}
              streamingMessages={streamingMessages}
              streamEvents={streamEvents}
              reconnectAttempt={reconnectAttempt}
              connectionQuality={connectionQuality}
              isPolling={isPolling}
              onReconnect={handleReconnect}
              onComplete={() => undefined}
            />
          </div>

          <aside className="space-y-4">
            <div
              style={{
                backgroundColor: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-card)',
                padding: '20px',
                boxShadow: 'var(--shadow-card)',
              }}
            >
              <p
                className="uppercase tracking-widest"
                style={{
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                  marginBottom: '12px',
                }}
              >
                Queue Another Topic
              </p>
              <div className="space-y-2">
                {DEMO_TOPICS.map((topic) => {
                  const isCurrent = topic === currentTopic;
                  const isQueued = topic === queuedTopic;

                  return (
                    <button
                      key={topic}
                      type="button"
                      onClick={() => handleQueueTopic(topic)}
                      aria-pressed={isCurrent || isQueued}
                      className="w-full text-left transition-colors"
                      style={{
                        padding: '14px 16px',
                        borderRadius: '14px',
                        border: `1px solid ${
                          isQueued ? 'var(--accent)' : isCurrent ? 'var(--text)' : 'var(--border)'
                        }`,
                        backgroundColor:
                          isQueued || isCurrent ? 'var(--bg)' : 'transparent',
                        color: 'var(--text)',
                      }}
                    >
                      <div
                        className="mb-1 uppercase tracking-widest"
                        style={{
                          fontSize: '10px',
                          color: isQueued ? 'var(--accent)' : 'var(--text-muted)',
                        }}
                      >
                        {isQueued ? 'Queued next' : isCurrent ? 'Now debating' : 'Queue topic'}
                      </div>
                      <div style={{ fontSize: '14px', lineHeight: '1.5' }}>{topic}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div
              style={{
                backgroundColor: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-card)',
                padding: '20px',
                boxShadow: 'var(--shadow-card)',
              }}
            >
              <p
                className="uppercase tracking-widest"
                style={{
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                  marginBottom: '10px',
                }}
              >
                What This Demonstrates
              </p>
              <ul
                style={{
                  margin: 0,
                  paddingLeft: '18px',
                  color: 'var(--text-muted)',
                  fontSize: '13px',
                  lineHeight: '1.8',
                }}
              >
                <li>Real debate bootstrap from the landing page.</li>
                <li>Live transcript updates over WebSocket.</li>
                <li>Serialized launch queue for follow-on debates.</li>
              </ul>
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        backgroundColor: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: '14px',
        padding: '14px 16px',
      }}
    >
      <div
        className="uppercase tracking-widest"
        style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '6px' }}
      >
        {label}
      </div>
      <div style={{ fontSize: '15px', color: 'var(--text)' }}>{value}</div>
    </div>
  );
}
