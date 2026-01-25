/**
 * Aragora Web Demo
 *
 * Demonstrates the Aragora TypeScript SDK in a web browser with:
 * - Real-time debate streaming via WebSocket
 * - Tournament creation and management
 * - Agent rankings
 * - Authentication
 */

import { AragoraClient } from 'aragora-js';

// Configuration
const API_URL = 'http://localhost:8080';

// Initialize client
const client = new AragoraClient({ baseUrl: API_URL });

// State
let currentUser: { id: string; email: string; name?: string } | null = null;

// =============================================================================
// DOM Elements
// =============================================================================

// Server status
const serverStatus = document.getElementById('server-status')!;

// Tabs
const tabButtons = document.querySelectorAll<HTMLButtonElement>('.tab-btn');
const tabContents = document.querySelectorAll<HTMLElement>('.tab-content');

// Debate elements
const debateForm = document.getElementById('debate-form') as HTMLFormElement;
const startBtn = document.getElementById('start-btn') as HTMLButtonElement;
const resultsSection = document.getElementById('results-section')!;
const debateStatus = document.querySelector('#debate-status .status-badge')!;
const debateIdSpan = document.getElementById('debate-id')!;
const debateLog = document.getElementById('debate-log')!;
const consensusResult = document.getElementById('consensus-result')!;
const consensusContent = document.getElementById('consensus-content')!;
const useStreamingCheckbox = document.getElementById('use-streaming') as HTMLInputElement;

// Rankings elements
const loadRankingsBtn = document.getElementById('load-rankings') as HTMLButtonElement;
const rankingsTable = document.getElementById('rankings-table')!;

// Tournament elements
const tournamentForm = document.getElementById('tournament-form') as HTMLFormElement;
const refreshTournamentsBtn = document.getElementById('refresh-tournaments') as HTMLButtonElement;
const tournamentsList = document.getElementById('tournaments-list')!;

// Auth elements
const authStatus = document.getElementById('auth-status')!;
const loginPanel = document.getElementById('login-panel')!;
const accountPanel = document.getElementById('account-panel')!;
const apikeysPanel = document.getElementById('apikeys-panel')!;
const loginForm = document.getElementById('login-form') as HTMLFormElement;
const logoutBtn = document.getElementById('logout-btn') as HTMLButtonElement;
const userInfo = document.getElementById('user-info')!;
const loadApikeysBtn = document.getElementById('load-apikeys') as HTMLButtonElement;
const apikeysList = document.getElementById('apikeys-list')!;

// =============================================================================
// Tab Navigation
// =============================================================================

function switchTab(tabId: string): void {
  tabButtons.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  tabContents.forEach(content => {
    content.classList.toggle('active', content.id === `${tabId}-tab`);
    content.classList.toggle('hidden', content.id !== `${tabId}-tab`);
  });
}

// =============================================================================
// Server Health
// =============================================================================

async function checkHealth(): Promise<void> {
  try {
    const health = await client.health.check();
    serverStatus.innerHTML = `
      <span class="status-ok">Connected</span>
      <span class="status-version">v${health.version || 'unknown'}</span>
    `;
  } catch {
    serverStatus.innerHTML = `
      <span class="status-error">Disconnected</span>
      <p class="hint">Start the server: <code>python -m aragora.server.unified_server --port 8080</code></p>
    `;
  }
}

// =============================================================================
// Debate Functions
// =============================================================================

function log(message: string, type: 'info' | 'agent' | 'system' | 'error' = 'info'): void {
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  const time = new Date().toLocaleTimeString();
  entry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
  debateLog.appendChild(entry);
  debateLog.scrollTop = debateLog.scrollHeight;
}

async function startDebateStreaming(topic: string, agents: string[], rounds: number): Promise<void> {
  log('Connecting to WebSocket...', 'system');

  try {
    const ws = client.debates.stream({
      topic,
      agents,
      rounds,
      consensus: 'majority',
    });

    ws.onMessage((event) => {
      switch (event.type) {
        case 'debate_start':
          debateIdSpan.textContent = `ID: ${event.debateId?.slice(0, 8)}...`;
          debateStatus.textContent = 'Running';
          debateStatus.className = 'status-badge status-running';
          log(`Debate started: ${event.debateId}`, 'system');
          break;

        case 'round_start':
          log(`Round ${event.round} starting...`, 'system');
          break;

        case 'agent_message':
          log(`<strong>${event.agent}</strong>: ${(event.content || '').slice(0, 150)}...`, 'agent');
          break;

        case 'critique':
          log(`[Critique by ${event.critic}]: ${event.summary || ''}`, 'info');
          break;

        case 'vote':
          log(`Vote: ${event.voter} voted for ${event.choice}`, 'info');
          break;

        case 'consensus':
          log('Consensus reached!', 'system');
          showConsensus(event.data);
          break;

        case 'debate_end':
          debateStatus.textContent = 'Completed';
          debateStatus.className = 'status-badge status-completed';
          log('Debate completed!', 'system');
          break;

        case 'error':
          log(`Error: ${event.message}`, 'error');
          debateStatus.textContent = 'Error';
          debateStatus.className = 'status-badge status-error';
          break;
      }
    });

    ws.onError((err) => {
      log(`WebSocket error: ${err.message}`, 'error');
      debateStatus.textContent = 'Error';
      debateStatus.className = 'status-badge status-error';
    });

    ws.onClose(() => {
      log('WebSocket closed', 'system');
    });

    await ws.connect();
  } catch (err) {
    log(`Failed to connect: ${(err as Error).message}`, 'error');
    throw err;
  }
}

async function startDebatePolling(topic: string, agents: string[], rounds: number): Promise<void> {
  log('Creating debate (polling mode)...', 'system');

  const debate = await client.debates.create({
    topic,
    agents,
    rounds,
    consensus: 'majority',
  });

  debateIdSpan.textContent = `ID: ${debate.id.slice(0, 8)}...`;
  debateStatus.textContent = 'Running';
  debateStatus.className = 'status-badge status-running';

  log(`Debate started: ${debate.id}`, 'system');

  // Poll for completion
  let result = await client.debates.get(debate.id);
  let lastRound = 0;

  while (result.status !== 'completed') {
    await sleep(2000);
    result = await client.debates.get(debate.id);

    const currentRound = result.rounds_completed || 0;
    if (currentRound > lastRound) {
      log(`Round ${currentRound} completed`, 'system');
      lastRound = currentRound;
    }
  }

  debateStatus.textContent = 'Completed';
  debateStatus.className = 'status-badge status-completed';
  log('Debate completed!', 'system');

  if (result.consensus) {
    showConsensus(result.consensus);
  }
}

function showConsensus(consensus: { reached?: boolean; confidence?: number; conclusion?: string }): void {
  consensusResult.classList.remove('hidden');
  consensusContent.innerHTML = `
    <p><strong>Reached:</strong> ${consensus.reached ? 'Yes' : 'No'}</p>
    <p><strong>Confidence:</strong> ${((consensus.confidence || 0) * 100).toFixed(1)}%</p>
    <div class="consensus-text">${consensus.conclusion || 'No conclusion'}</div>
  `;
}

async function startDebate(e: Event): Promise<void> {
  e.preventDefault();

  const formData = new FormData(debateForm);
  const topic = formData.get('topic') as string;
  const agentsSelect = document.getElementById('agents') as HTMLSelectElement;
  const agents = Array.from(agentsSelect.selectedOptions).map(opt => opt.value);
  const rounds = parseInt(formData.get('rounds') as string, 10);
  const useStreaming = useStreamingCheckbox.checked;

  if (!topic.trim()) {
    alert('Please enter a topic');
    return;
  }

  if (agents.length < 2) {
    alert('Please select at least 2 agents');
    return;
  }

  // Reset UI
  resultsSection.classList.remove('hidden');
  debateLog.innerHTML = '';
  consensusResult.classList.add('hidden');
  debateStatus.textContent = 'Creating...';
  debateStatus.className = 'status-badge status-pending';
  startBtn.disabled = true;

  try {
    log(`Starting debate with ${agents.length} agents...`, 'system');
    log(`Mode: ${useStreaming ? 'WebSocket streaming' : 'Polling'}`, 'system');

    if (useStreaming) {
      await startDebateStreaming(topic, agents, rounds);
    } else {
      await startDebatePolling(topic, agents, rounds);
    }
  } catch (err) {
    debateStatus.textContent = 'Error';
    debateStatus.className = 'status-badge status-error';
    log(`Error: ${(err as Error).message}`, 'error');
  } finally {
    startBtn.disabled = false;
  }
}

// =============================================================================
// Rankings Functions
// =============================================================================

async function loadRankings(): Promise<void> {
  loadRankingsBtn.disabled = true;
  loadRankingsBtn.textContent = 'Loading...';

  try {
    const rankings = await client.agents.rankings();

    const tbody = rankingsTable.querySelector('tbody')!;
    tbody.innerHTML = '';

    rankings.slice(0, 10).forEach((agent, i) => {
      const wins = agent.wins || 0;
      const losses = agent.losses || 0;
      const total = wins + losses;
      const winRate = total > 0 ? ((wins / total) * 100).toFixed(1) : 'N/A';

      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${i + 1}</td>
        <td>${agent.name}</td>
        <td>${agent.elo?.toFixed(0) || 'N/A'}</td>
        <td>${wins}/${losses}</td>
        <td>${winRate}${typeof winRate === 'string' && winRate !== 'N/A' ? '%' : ''}</td>
      `;
      tbody.appendChild(row);
    });

    rankingsTable.classList.remove('hidden');
    loadRankingsBtn.textContent = 'Refresh';
  } catch (err) {
    alert(`Failed to load rankings: ${(err as Error).message}`);
    loadRankingsBtn.textContent = 'Retry';
  } finally {
    loadRankingsBtn.disabled = false;
  }
}

// =============================================================================
// Tournament Functions
// =============================================================================

async function createTournament(e: Event): Promise<void> {
  e.preventDefault();

  const nameInput = document.getElementById('tournament-name') as HTMLInputElement;
  const agentsSelect = document.getElementById('tournament-agents') as HTMLSelectElement;
  const formatSelect = document.getElementById('tournament-format') as HTMLSelectElement;

  const name = nameInput.value;
  const participants = Array.from(agentsSelect.selectedOptions).map(opt => opt.value);
  const format = formatSelect.value;

  if (participants.length < 2) {
    alert('Please select at least 2 participants');
    return;
  }

  try {
    const tournament = await client.tournaments.create({
      name,
      participants,
      format,
    });

    alert(`Tournament created: ${tournament.id}`);
    nameInput.value = '';
    await loadTournaments();
  } catch (err) {
    alert(`Failed to create tournament: ${(err as Error).message}`);
  }
}

async function loadTournaments(): Promise<void> {
  refreshTournamentsBtn.disabled = true;

  try {
    const tournaments = await client.tournaments.list({ limit: 10 });

    if (tournaments.length === 0) {
      tournamentsList.innerHTML = '<p class="hint">No tournaments found. Create one above!</p>';
      return;
    }

    tournamentsList.innerHTML = tournaments.map(t => `
      <div class="tournament-card">
        <h4>${t.name}</h4>
        <p>
          <span class="status-badge status-${t.status}">${t.status}</span>
          <span>${t.format}</span>
        </p>
        <p>${t.participants?.length || 0} participants</p>
        ${t.status === 'completed' ? `<button onclick="viewStandings('${t.id}')">View Standings</button>` : ''}
      </div>
    `).join('');
  } catch (err) {
    tournamentsList.innerHTML = `<p class="error">Failed to load: ${(err as Error).message}</p>`;
  } finally {
    refreshTournamentsBtn.disabled = false;
  }
}

// Make viewStandings available globally
(window as unknown as { viewStandings: (id: string) => void }).viewStandings = async (id: string) => {
  try {
    const standings = await client.tournaments.getStandings(id);
    const html = standings.standings.map((s, i) =>
      `${i + 1}. ${s.participant} - ${s.wins}W/${s.losses}L`
    ).join('\n');
    alert(`Standings:\n${html}`);
  } catch (err) {
    alert(`Failed to load standings: ${(err as Error).message}`);
  }
};

// =============================================================================
// Authentication Functions
// =============================================================================

function updateAuthUI(): void {
  if (currentUser) {
    authStatus.innerHTML = `<span class="user-badge">${currentUser.email}</span>`;
    loginPanel.classList.add('hidden');
    accountPanel.classList.remove('hidden');
    apikeysPanel.classList.remove('hidden');
    userInfo.innerHTML = `
      <p><strong>Email:</strong> ${currentUser.email}</p>
      <p><strong>ID:</strong> ${currentUser.id}</p>
      ${currentUser.name ? `<p><strong>Name:</strong> ${currentUser.name}</p>` : ''}
    `;
  } else {
    authStatus.innerHTML = '<span class="hint">Not logged in</span>';
    loginPanel.classList.remove('hidden');
    accountPanel.classList.add('hidden');
    apikeysPanel.classList.add('hidden');
  }
}

async function handleLogin(e: Event): Promise<void> {
  e.preventDefault();

  const emailInput = document.getElementById('login-email') as HTMLInputElement;
  const passwordInput = document.getElementById('login-password') as HTMLInputElement;

  try {
    const token = await client.auth.login(emailInput.value, passwordInput.value);

    // Store token
    localStorage.setItem('aragora_token', token.accessToken);

    // Get user info
    const user = await client.auth.getCurrentUser();
    currentUser = user;

    updateAuthUI();
    passwordInput.value = '';
  } catch (err) {
    alert(`Login failed: ${(err as Error).message}`);
  }
}

async function handleLogout(): Promise<void> {
  try {
    await client.auth.logout();
  } catch {
    // Ignore logout errors
  }

  localStorage.removeItem('aragora_token');
  currentUser = null;
  updateAuthUI();
}

async function loadApiKeys(): Promise<void> {
  loadApikeysBtn.disabled = true;

  try {
    const keys = await client.auth.listApiKeys();

    if (keys.length === 0) {
      apikeysList.innerHTML = '<p class="hint">No API keys found.</p>';
    } else {
      apikeysList.innerHTML = keys.map(k => `
        <div class="apikey-card">
          <strong>${k.name}</strong>
          <span class="status-badge ${k.isActive ? 'status-ok' : 'status-error'}">${k.isActive ? 'Active' : 'Revoked'}</span>
          <p>Created: ${k.createdAt}</p>
        </div>
      `).join('');
    }

    document.getElementById('create-apikey-form')?.classList.remove('hidden');
  } catch (err) {
    apikeysList.innerHTML = `<p class="error">Failed to load: ${(err as Error).message}</p>`;
  } finally {
    loadApikeysBtn.disabled = false;
  }
}

// =============================================================================
// Utilities
// =============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// Event Listeners
// =============================================================================

// Tabs
tabButtons.forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab!));
});

// Debates
debateForm.addEventListener('submit', startDebate);

// Rankings
loadRankingsBtn.addEventListener('click', loadRankings);

// Tournaments
tournamentForm.addEventListener('submit', createTournament);
refreshTournamentsBtn.addEventListener('click', loadTournaments);

// Auth
loginForm.addEventListener('submit', handleLogin);
logoutBtn.addEventListener('click', handleLogout);
loadApikeysBtn.addEventListener('click', loadApiKeys);

// =============================================================================
// Initialize
// =============================================================================

checkHealth();
updateAuthUI();

// Check for stored token
const storedToken = localStorage.getItem('aragora_token');
if (storedToken) {
  client.auth.getCurrentUser()
    .then(user => {
      currentUser = user;
      updateAuthUI();
    })
    .catch(() => {
      localStorage.removeItem('aragora_token');
    });
}
