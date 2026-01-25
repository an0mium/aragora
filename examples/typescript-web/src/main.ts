/**
 * Aragora Web Demo
 *
 * Demonstrates the Aragora TypeScript SDK in a web browser.
 */

import { AragoraClient } from 'aragora-js';

// Configuration
const API_URL = 'http://localhost:8080';

// Initialize client
const client = new AragoraClient({ baseUrl: API_URL });

// DOM Elements
const serverStatus = document.getElementById('server-status')!;
const debateForm = document.getElementById('debate-form') as HTMLFormElement;
const startBtn = document.getElementById('start-btn') as HTMLButtonElement;
const resultsSection = document.getElementById('results-section')!;
const debateStatus = document.querySelector('#debate-status .status-badge')!;
const debateIdSpan = document.getElementById('debate-id')!;
const debateLog = document.getElementById('debate-log')!;
const consensusResult = document.getElementById('consensus-result')!;
const consensusContent = document.getElementById('consensus-content')!;
const loadRankingsBtn = document.getElementById('load-rankings') as HTMLButtonElement;
const rankingsTable = document.getElementById('rankings-table')!;

// Check server health on load
async function checkHealth(): Promise<void> {
  try {
    const health = await client.health.check();
    serverStatus.innerHTML = `
      <span class="status-ok">Connected</span>
      <span class="status-version">v${health.version || 'unknown'}</span>
    `;
  } catch (error) {
    serverStatus.innerHTML = `
      <span class="status-error">Disconnected</span>
      <p>Start the server: <code>python -m aragora.server.unified_server --port 8080</code></p>
    `;
  }
}

// Log a message to the debate log
function log(message: string, type: 'info' | 'agent' | 'system' = 'info'): void {
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  entry.textContent = message;
  debateLog.appendChild(entry);
  debateLog.scrollTop = debateLog.scrollHeight;
}

// Start a debate
async function startDebate(e: Event): Promise<void> {
  e.preventDefault();

  const formData = new FormData(debateForm);
  const topic = formData.get('topic') as string;
  const agentsSelect = document.getElementById('agents') as HTMLSelectElement;
  const agents = Array.from(agentsSelect.selectedOptions).map(opt => opt.value);
  const rounds = parseInt(formData.get('rounds') as string, 10);

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
    log(`Creating debate with ${agents.length} agents...`, 'system');

    // Create debate
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

      // Log round progress
      const currentRound = result.rounds_completed || 0;
      if (currentRound > lastRound) {
        log(`Round ${currentRound} completed`, 'system');
        lastRound = currentRound;
      }
    }

    // Show results
    debateStatus.textContent = 'Completed';
    debateStatus.className = 'status-badge status-completed';

    log('Debate completed!', 'system');

    if (result.consensus) {
      consensusResult.classList.remove('hidden');
      consensusContent.innerHTML = `
        <p><strong>Reached:</strong> ${result.consensus.reached ? 'Yes' : 'No'}</p>
        <p><strong>Confidence:</strong> ${((result.consensus.confidence || 0) * 100).toFixed(1)}%</p>
        <div class="consensus-text">${result.consensus.conclusion || result.synthesis || 'No conclusion'}</div>
      `;
    }

  } catch (error) {
    debateStatus.textContent = 'Error';
    debateStatus.className = 'status-badge status-error';
    log(`Error: ${(error as Error).message}`, 'system');
  } finally {
    startBtn.disabled = false;
  }
}

// Load agent rankings
async function loadRankings(): Promise<void> {
  loadRankingsBtn.disabled = true;
  loadRankingsBtn.textContent = 'Loading...';

  try {
    const rankings = await client.agents.rankings();

    const tbody = rankingsTable.querySelector('tbody')!;
    tbody.innerHTML = '';

    rankings.slice(0, 10).forEach((agent, i) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${i + 1}</td>
        <td>${agent.name}</td>
        <td>${agent.elo?.toFixed(0) || 'N/A'}</td>
        <td>${agent.wins || 0}/${agent.losses || 0}</td>
      `;
      tbody.appendChild(row);
    });

    rankingsTable.classList.remove('hidden');
    loadRankingsBtn.textContent = 'Refresh';

  } catch (error) {
    alert(`Failed to load rankings: ${(error as Error).message}`);
    loadRankingsBtn.textContent = 'Retry';
  } finally {
    loadRankingsBtn.disabled = false;
  }
}

// Utility: sleep
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Event listeners
debateForm.addEventListener('submit', startDebate);
loadRankingsBtn.addEventListener('click', loadRankings);

// Initialize
checkHealth();
