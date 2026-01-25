/**
 * Aragora Slack Bot
 *
 * A Slack bot that enables users to run AI debates directly from Slack.
 *
 * Commands:
 *   /debate <topic> - Start a new debate
 *   /rankings - View agent rankings
 *   /tournament <name> - Create a tournament
 *
 * Features:
 *   - Real-time debate streaming to Slack threads
 *   - Interactive buttons for debate actions
 *   - Agent ranking leaderboard
 *   - Tournament creation and tracking
 */

import { App, LogLevel } from '@slack/bolt';
import { AragoraClient } from 'aragora-js';
import 'dotenv/config';

// =============================================================================
// Configuration
// =============================================================================

const ARAGORA_URL = process.env.ARAGORA_URL || 'http://localhost:8080';
const SLACK_BOT_TOKEN = process.env.SLACK_BOT_TOKEN;
const SLACK_SIGNING_SECRET = process.env.SLACK_SIGNING_SECRET;
const SLACK_APP_TOKEN = process.env.SLACK_APP_TOKEN;

if (!SLACK_BOT_TOKEN || !SLACK_SIGNING_SECRET) {
  console.error('Error: SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET are required');
  process.exit(1);
}

// Initialize clients
const aragora = new AragoraClient({ baseUrl: ARAGORA_URL });
const app = new App({
  token: SLACK_BOT_TOKEN,
  signingSecret: SLACK_SIGNING_SECRET,
  appToken: SLACK_APP_TOKEN,
  socketMode: !!SLACK_APP_TOKEN,
  logLevel: LogLevel.INFO,
});

// =============================================================================
// Slash Commands
// =============================================================================

/**
 * /debate <topic> - Start a new AI debate
 */
app.command('/debate', async ({ command, ack, say, client }) => {
  await ack();

  const topic = command.text.trim();
  if (!topic) {
    await say({
      text: 'Please provide a debate topic. Usage: `/debate <topic>`',
      response_type: 'ephemeral',
    });
    return;
  }

  // Post initial message
  const result = await say({
    blocks: [
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `*Starting Debate*\n_${topic}_`,
        },
      },
      {
        type: 'context',
        elements: [
          {
            type: 'mrkdwn',
            text: `Requested by <@${command.user_id}>`,
          },
        ],
      },
    ],
  });

  const threadTs = result.ts;
  const channel = command.channel_id;

  try {
    // Stream debate events
    const ws = aragora.debates.stream({
      topic,
      agents: ['anthropic-api', 'openai-api'],
      rounds: 2,
      consensus: 'majority',
    });

    let debateId = '';

    ws.onMessage(async (event) => {
      switch (event.type) {
        case 'debate_start':
          debateId = event.debateId || '';
          await client.chat.postMessage({
            channel,
            thread_ts: threadTs,
            text: `Debate started (ID: ${debateId.slice(0, 8)}...)`,
          });
          break;

        case 'round_start':
          await client.chat.postMessage({
            channel,
            thread_ts: threadTs,
            text: `*Round ${event.round}*`,
          });
          break;

        case 'agent_message':
          const content = (event.content || '').slice(0, 500);
          await client.chat.postMessage({
            channel,
            thread_ts: threadTs,
            blocks: [
              {
                type: 'section',
                text: {
                  type: 'mrkdwn',
                  text: `*${event.agent}*\n${content}${content.length >= 500 ? '...' : ''}`,
                },
              },
            ],
          });
          break;

        case 'consensus':
          const consensus = event.data as {
            reached?: boolean;
            confidence?: number;
            conclusion?: string;
          };
          await client.chat.postMessage({
            channel,
            thread_ts: threadTs,
            blocks: [
              {
                type: 'section',
                text: {
                  type: 'mrkdwn',
                  text: `*Consensus Reached*\n\n` +
                    `Confidence: ${((consensus.confidence || 0) * 100).toFixed(0)}%\n\n` +
                    `${consensus.conclusion || 'No conclusion'}`,
                },
              },
            ],
          });
          break;

        case 'debate_end':
          await client.chat.postMessage({
            channel,
            thread_ts: threadTs,
            blocks: [
              {
                type: 'section',
                text: {
                  type: 'mrkdwn',
                  text: '*Debate Complete*',
                },
              },
              {
                type: 'actions',
                elements: [
                  {
                    type: 'button',
                    text: { type: 'plain_text', text: 'View Full Results' },
                    action_id: 'view_debate',
                    value: debateId,
                  },
                  {
                    type: 'button',
                    text: { type: 'plain_text', text: 'Start Similar Debate' },
                    action_id: 'restart_debate',
                    value: topic,
                  },
                ],
              },
            ],
          });
          break;
      }
    });

    ws.onError(async (err) => {
      await client.chat.postMessage({
        channel,
        thread_ts: threadTs,
        text: `Error: ${err.message}`,
      });
    });

    await ws.connect();
  } catch (error) {
    await client.chat.postMessage({
      channel,
      thread_ts: threadTs,
      text: `Failed to start debate: ${(error as Error).message}`,
    });
  }
});

/**
 * /rankings - View agent rankings
 */
app.command('/rankings', async ({ command, ack, say }) => {
  await ack();

  try {
    const rankings = await aragora.agents.rankings();

    const blocks = [
      {
        type: 'section' as const,
        text: {
          type: 'mrkdwn' as const,
          text: '*Agent Rankings*',
        },
      },
      {
        type: 'divider' as const,
      },
    ];

    rankings.slice(0, 10).forEach((agent, i) => {
      const wins = agent.wins || 0;
      const losses = agent.losses || 0;
      const medal = i === 0 ? ':first_place_medal:' : i === 1 ? ':second_place_medal:' : i === 2 ? ':third_place_medal:' : '';

      blocks.push({
        type: 'section' as const,
        text: {
          type: 'mrkdwn' as const,
          text: `${medal} *${i + 1}. ${agent.name}*\nELO: ${agent.elo?.toFixed(0) || 'N/A'} | W/L: ${wins}/${losses}`,
        },
      });
    });

    await say({
      blocks,
      text: 'Agent Rankings',
    });
  } catch (error) {
    await say({
      text: `Failed to load rankings: ${(error as Error).message}`,
      response_type: 'ephemeral',
    });
  }
});

/**
 * /tournament <name> - Create a tournament
 */
app.command('/tournament', async ({ command, ack, say, client }) => {
  await ack();

  const name = command.text.trim() || `Slack Tournament ${new Date().toLocaleDateString()}`;

  // Show tournament creation modal
  await client.views.open({
    trigger_id: command.trigger_id,
    view: {
      type: 'modal',
      callback_id: 'create_tournament',
      title: {
        type: 'plain_text',
        text: 'Create Tournament',
      },
      submit: {
        type: 'plain_text',
        text: 'Create',
      },
      blocks: [
        {
          type: 'input',
          block_id: 'name_block',
          label: {
            type: 'plain_text',
            text: 'Tournament Name',
          },
          element: {
            type: 'plain_text_input',
            action_id: 'name_input',
            initial_value: name,
          },
        },
        {
          type: 'input',
          block_id: 'agents_block',
          label: {
            type: 'plain_text',
            text: 'Participants',
          },
          element: {
            type: 'multi_static_select',
            action_id: 'agents_select',
            placeholder: {
              type: 'plain_text',
              text: 'Select agents',
            },
            options: [
              { text: { type: 'plain_text', text: 'Claude (Anthropic)' }, value: 'anthropic-api' },
              { text: { type: 'plain_text', text: 'GPT (OpenAI)' }, value: 'openai-api' },
              { text: { type: 'plain_text', text: 'Gemini (Google)' }, value: 'gemini' },
              { text: { type: 'plain_text', text: 'Grok (xAI)' }, value: 'grok' },
              { text: { type: 'plain_text', text: 'Mistral' }, value: 'mistral' },
            ],
            initial_options: [
              { text: { type: 'plain_text', text: 'Claude (Anthropic)' }, value: 'anthropic-api' },
              { text: { type: 'plain_text', text: 'GPT (OpenAI)' }, value: 'openai-api' },
              { text: { type: 'plain_text', text: 'Gemini (Google)' }, value: 'gemini' },
            ],
          },
        },
        {
          type: 'input',
          block_id: 'format_block',
          label: {
            type: 'plain_text',
            text: 'Format',
          },
          element: {
            type: 'static_select',
            action_id: 'format_select',
            options: [
              { text: { type: 'plain_text', text: 'Single Elimination' }, value: 'single_elimination' },
              { text: { type: 'plain_text', text: 'Double Elimination' }, value: 'double_elimination' },
              { text: { type: 'plain_text', text: 'Round Robin' }, value: 'round_robin' },
            ],
            initial_option: { text: { type: 'plain_text', text: 'Single Elimination' }, value: 'single_elimination' },
          },
        },
      ],
      private_metadata: command.channel_id,
    },
  });
});

// =============================================================================
// View Submissions
// =============================================================================

app.view('create_tournament', async ({ ack, view, client }) => {
  await ack();

  const values = view.state.values;
  const name = values.name_block.name_input.value || 'Unnamed Tournament';
  const agents = values.agents_block.agents_select.selected_options?.map(opt => opt.value) || [];
  const format = values.format_block.format_select.selected_option?.value || 'single_elimination';
  const channel = view.private_metadata;

  try {
    const tournament = await aragora.tournaments.create({
      name,
      participants: agents,
      format,
    });

    await client.chat.postMessage({
      channel,
      blocks: [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `*Tournament Created: ${name}*\n\nID: \`${tournament.id}\`\nFormat: ${format}\nParticipants: ${agents.join(', ')}`,
          },
        },
        {
          type: 'actions',
          elements: [
            {
              type: 'button',
              text: { type: 'plain_text', text: 'View Standings' },
              action_id: 'view_standings',
              value: tournament.id,
            },
          ],
        },
      ],
    });
  } catch (error) {
    await client.chat.postMessage({
      channel,
      text: `Failed to create tournament: ${(error as Error).message}`,
    });
  }
});

// =============================================================================
// Button Actions
// =============================================================================

app.action('view_debate', async ({ ack, body, client }) => {
  await ack();

  const debateId = (body as { actions: { value: string }[] }).actions[0].value;

  try {
    const debate = await aragora.debates.get(debateId);

    await client.views.open({
      trigger_id: (body as { trigger_id: string }).trigger_id,
      view: {
        type: 'modal',
        title: { type: 'plain_text', text: 'Debate Details' },
        blocks: [
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `*Topic:* ${debate.topic || 'N/A'}\n*Status:* ${debate.status}\n*Rounds:* ${debate.rounds_completed || 0}`,
            },
          },
          {
            type: 'divider',
          },
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `*Consensus:*\n${debate.consensus?.conclusion || 'No consensus'}`,
            },
          },
        ],
      },
    });
  } catch (error) {
    console.error('Failed to load debate:', error);
  }
});

app.action('restart_debate', async ({ ack, body, say }) => {
  await ack();

  const topic = (body as { actions: { value: string }[] }).actions[0].value;
  await say({
    text: `To start a similar debate, use: \`/debate ${topic}\``,
    response_type: 'ephemeral',
  });
});

app.action('view_standings', async ({ ack, body, client }) => {
  await ack();

  const tournamentId = (body as { actions: { value: string }[] }).actions[0].value;

  try {
    const standings = await aragora.tournaments.getStandings(tournamentId);

    const text = standings.standings
      .map((s, i) => `${i + 1}. ${s.participant} - ${s.wins}W/${s.losses}L`)
      .join('\n');

    await client.views.open({
      trigger_id: (body as { trigger_id: string }).trigger_id,
      view: {
        type: 'modal',
        title: { type: 'plain_text', text: 'Tournament Standings' },
        blocks: [
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: text || 'No standings available yet.',
            },
          },
        ],
      },
    });
  } catch (error) {
    console.error('Failed to load standings:', error);
  }
});

// =============================================================================
// App Mentions
// =============================================================================

app.event('app_mention', async ({ event, say }) => {
  const text = event.text.toLowerCase();

  if (text.includes('debate')) {
    await say({
      thread_ts: event.ts,
      text: 'To start a debate, use the `/debate <topic>` command!',
    });
  } else if (text.includes('ranking')) {
    await say({
      thread_ts: event.ts,
      text: 'To view rankings, use the `/rankings` command!',
    });
  } else {
    await say({
      thread_ts: event.ts,
      text: `Hi <@${event.user}>! I can help you run AI debates.\n\nCommands:\n• \`/debate <topic>\` - Start a debate\n• \`/rankings\` - View agent rankings\n• \`/tournament <name>\` - Create a tournament`,
    });
  }
});

// =============================================================================
// Start the app
// =============================================================================

(async () => {
  const port = Number(process.env.PORT) || 3000;
  await app.start(port);
  console.log(`Aragora Slack Bot running on port ${port}`);
  console.log(`Aragora API: ${ARAGORA_URL}`);
})();
