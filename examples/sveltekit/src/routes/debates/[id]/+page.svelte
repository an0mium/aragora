<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { PageData } from './$types';

  export let data: PageData;
  $: debate = data.debate;

  let events: any[] = [];
  let connected = false;
  let ws: WebSocket | null = null;

  onMount(() => {
    if (debate.status === 'running') {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${protocol}://${window.location.host}/ws/debate/${debate.debate_id}`);

      ws.onopen = () => (connected = true);
      ws.onclose = () => (connected = false);
      ws.onerror = () => (connected = false);
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          events = [...events, { ...msg, timestamp: new Date().toISOString() }];
        } catch {
          // ignore non-JSON messages
        }
      };
    }
  });

  onDestroy(() => {
    if (ws) ws.close();
  });

  function formatConfidence(value: number): string {
    return (value * 100).toFixed(1) + '%';
  }
</script>

<div>
  <div style="margin-bottom: 2rem">
    <span
      class="status-badge"
      class:completed={debate.status === 'completed'}
      class:running={debate.status === 'running'}
    >
      {debate.status}
    </span>
    <h1 style="margin-top: 1rem">{debate.task}</h1>
    <p class="muted">
      Created {new Date(debate.created_at).toLocaleString()}
    </p>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Agents</h3>
      <div class="agent-list">
        {#each debate.agents || [] as agent}
          <span class="agent-tag">{agent}</span>
        {/each}
      </div>
    </div>

    <div class="card">
      <h3>Progress</h3>
      <p>Round {debate.current_round || 0} of {debate.total_rounds || 9}</p>
      <div class="progress-bar">
        <div
          class="progress-fill"
          style="width: {((debate.current_round || 0) / (debate.total_rounds || 9)) * 100}%"
        />
      </div>
    </div>
  </div>

  {#if debate.status === 'running'}
    <div class="card" style="margin-top: 1rem">
      <h3>Live Stream</h3>
      <div class="connection-status">
        <div class="dot" class:connected />
        <span class="muted">{connected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {#if events.length === 0}
        <p class="muted">Waiting for events...</p>
      {:else}
        {#each events as event, idx}
          <div class="stream-event">
            <div class="event-header">
              <strong>
                {event.type}
                {#if event.agent}({event.agent}){/if}
              </strong>
              <span class="timestamp">
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
            {#if event.content}
              <p class="muted">
                {event.content.slice(0, 200)}{event.content.length > 200 ? '...' : ''}
              </p>
            {/if}
          </div>
        {/each}
      {/if}
    </div>
  {/if}

  {#if debate.status === 'completed' && debate.consensus}
    <div class="card consensus" style="margin-top: 1rem">
      <h3>Consensus Reached</h3>
      <p style="margin-bottom: 1rem">{debate.consensus.decision}</p>
      <div class="stats">
        <div>
          <span class="muted">Confidence</span>
          <p class="stat-value">{formatConfidence(debate.consensus.confidence)}</p>
        </div>
        <div>
          <span class="muted">Agreement</span>
          <p class="stat-value">
            {debate.consensus.votes_for}/{debate.consensus.votes_for + debate.consensus.votes_against}
          </p>
        </div>
      </div>
    </div>
  {/if}

  {#if debate.messages && debate.messages.length > 0}
    <div class="card" style="margin-top: 1rem">
      <h3>Debate History</h3>
      <div class="messages">
        {#each debate.messages as msg, idx}
          <div class="message">
            <div class="message-header">
              <strong>{msg.agent}</strong>
              <span class="muted">Round {msg.round} - {msg.phase}</span>
            </div>
            <p style="white-space: pre-wrap">{msg.content}</p>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .card {
    padding: 1.5rem;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
  }

  .consensus {
    border-color: #3b82f6;
    border-width: 2px;
  }

  .consensus h3 {
    color: #3b82f6;
  }

  .muted {
    opacity: 0.6;
  }

  .status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    background: #f3f4f6;
    color: #374151;
  }

  .status-badge.completed {
    background: #dcfce7;
    color: #166534;
  }

  .status-badge.running {
    background: #dbeafe;
    color: #1e40af;
  }

  .agent-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .agent-tag {
    padding: 0.25rem 0.75rem;
    background: #f3f4f6;
    border-radius: 9999px;
    font-size: 0.875rem;
  }

  .progress-bar {
    margin-top: 0.5rem;
    background: #f3f4f6;
    border-radius: 9999px;
    height: 8px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: #3b82f6;
    transition: width 0.3s;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
  }

  .dot.connected {
    background: #22c55e;
  }

  .stream-event {
    padding: 0.75rem;
    border-bottom: 1px solid #e5e7eb;
  }

  .event-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.25rem;
  }

  .timestamp {
    font-size: 0.75rem;
    opacity: 0.6;
  }

  .stats {
    display: flex;
    gap: 2rem;
  }

  .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
  }

  .messages {
    max-height: 500px;
    overflow-y: auto;
  }

  .message {
    padding: 1rem;
    border-bottom: 1px solid #e5e7eb;
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
  }
</style>
