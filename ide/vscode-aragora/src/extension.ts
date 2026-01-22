/**
 * Aragora VS Code Extension
 *
 * Provides integration with the Aragora multi-agent debate API.
 */

import * as vscode from 'vscode';

interface DebateResult {
  debate_id: string;
  question: string;
  status: string;
  consensus?: {
    final_answer: string;
    confidence: number;
    method: string;
  };
  agents: string[];
  rounds_completed: number;
}

interface Agent {
  id: string;
  name: string;
  provider: string;
  capabilities?: string[];
  status: string;
}

class AragoraClient {
  private apiUrl: string;
  private apiKey: string;

  constructor() {
    const config = vscode.workspace.getConfiguration('aragora');
    this.apiUrl = config.get('apiUrl') || 'https://api.aragora.ai';
    this.apiKey = config.get('apiKey') || '';
  }

  private async fetch<T>(path: string, options: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(`${this.apiUrl}${path}`, {
      ...options,
      headers: { ...headers, ...options.headers },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    return response.json() as Promise<T>;
  }

  async runDebate(question: string, agents: string[], rounds: number): Promise<DebateResult> {
    return this.fetch<DebateResult>('/api/debates', {
      method: 'POST',
      body: JSON.stringify({
        task: question,
        agents,
        protocol: { rounds, consensus: 'majority' },
      }),
    });
  }

  async getDebate(debateId: string): Promise<DebateResult> {
    return this.fetch<DebateResult>(`/api/debates/${debateId}`);
  }

  async listDebates(limit = 10): Promise<{ debates: DebateResult[] }> {
    return this.fetch<{ debates: DebateResult[] }>(`/api/debates?limit=${limit}`);
  }

  async listAgents(): Promise<{ agents: Agent[] }> {
    return this.fetch<{ agents: Agent[] }>('/api/agents');
  }

  async runGauntlet(content: string, contentType = 'code'): Promise<{ session_id: string }> {
    return this.fetch<{ session_id: string }>('/api/gauntlet', {
      method: 'POST',
      body: JSON.stringify({ content, content_type: contentType, profile: 'quick' }),
    });
  }
}

class DebatesTreeProvider implements vscode.TreeDataProvider<DebateItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<DebateItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private client: AragoraClient;
  private debates: DebateResult[] = [];

  constructor(client: AragoraClient) {
    this.client = client;
  }

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  async loadDebates(): Promise<void> {
    try {
      const result = await this.client.listDebates(20);
      this.debates = result.debates || [];
      this.refresh();
    } catch (error) {
      this.debates = [];
      this.refresh();
    }
  }

  getTreeItem(element: DebateItem): vscode.TreeItem {
    return element;
  }

  getChildren(): DebateItem[] {
    return this.debates.map(
      (d) =>
        new DebateItem(
          d.question.substring(0, 50) + (d.question.length > 50 ? '...' : ''),
          d.debate_id,
          d.status,
          d.consensus?.final_answer
        )
    );
  }
}

class DebateItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly debateId: string,
    public readonly status: string,
    public readonly answer?: string
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);

    this.tooltip = answer || 'No consensus reached';
    this.description = status;

    const icon = status === 'completed' ? 'check' : status === 'running' ? 'sync~spin' : 'circle-outline';
    this.iconPath = new vscode.ThemeIcon(icon);
  }
}

class AgentsTreeProvider implements vscode.TreeDataProvider<AgentItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<AgentItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private client: AragoraClient;
  private agents: Agent[] = [];

  constructor(client: AragoraClient) {
    this.client = client;
  }

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  async loadAgents(): Promise<void> {
    try {
      const result = await this.client.listAgents();
      this.agents = result.agents || [];
      this.refresh();
    } catch (error) {
      this.agents = [];
      this.refresh();
    }
  }

  getTreeItem(element: AgentItem): vscode.TreeItem {
    return element;
  }

  getChildren(): AgentItem[] {
    return this.agents.map((a) => new AgentItem(a.name || a.id, a.provider, a.status));
  }
}

class AgentItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly provider: string,
    public readonly status: string
  ) {
    super(label, vscode.TreeItemCollapsibleState.None);

    this.description = provider;
    this.tooltip = `${label} (${provider}) - ${status}`;

    const icon = status === 'available' ? 'robot' : 'circle-slash';
    this.iconPath = new vscode.ThemeIcon(icon);
  }
}

export function activate(context: vscode.ExtensionContext) {
  const client = new AragoraClient();

  // Register tree views
  const debatesProvider = new DebatesTreeProvider(client);
  const agentsProvider = new AgentsTreeProvider(client);

  vscode.window.registerTreeDataProvider('aragora.debates', debatesProvider);
  vscode.window.registerTreeDataProvider('aragora.agents', agentsProvider);

  // Load initial data
  debatesProvider.loadDebates();
  agentsProvider.loadAgents();

  // Run Debate command
  const runDebateCmd = vscode.commands.registerCommand('aragora.runDebate', async () => {
    const config = vscode.workspace.getConfiguration('aragora');
    const defaultAgents = config.get<string>('defaultAgents') || 'claude,gpt-4';
    const defaultRounds = config.get<number>('defaultRounds') || 3;

    const question = await vscode.window.showInputBox({
      prompt: 'Enter the debate question or topic',
      placeHolder: 'What is the best approach for handling errors in TypeScript?',
    });

    if (!question) {
      return;
    }

    const agentsInput = await vscode.window.showInputBox({
      prompt: 'Enter agents (comma-separated)',
      value: defaultAgents,
      placeHolder: 'claude,gpt-4,gemini',
    });

    if (!agentsInput) {
      return;
    }

    const agents = agentsInput.split(',').map((a) => a.trim());

    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: 'Running Aragora Debate',
        cancellable: false,
      },
      async (progress) => {
        progress.report({ message: 'Starting debate...' });

        try {
          const result = await client.runDebate(question, agents, defaultRounds);

          // Poll for completion
          let debate = result;
          while (debate.status === 'running' || debate.status === 'pending') {
            await new Promise((resolve) => setTimeout(resolve, 2000));
            debate = await client.getDebate(result.debate_id);
            progress.report({ message: `Round ${debate.rounds_completed}/${defaultRounds}` });
          }

          // Show result
          const answer = debate.consensus?.final_answer || 'No consensus reached';
          const action = await vscode.window.showInformationMessage(
            `Debate Complete!\n\n${answer.substring(0, 200)}...`,
            'View Full Result',
            'Copy Answer'
          );

          if (action === 'Copy Answer') {
            await vscode.env.clipboard.writeText(answer);
            vscode.window.showInformationMessage('Answer copied to clipboard');
          } else if (action === 'View Full Result') {
            const doc = await vscode.workspace.openTextDocument({
              content: JSON.stringify(debate, null, 2),
              language: 'json',
            });
            await vscode.window.showTextDocument(doc);
          }

          debatesProvider.loadDebates();
        } catch (error) {
          vscode.window.showErrorMessage(`Debate failed: ${error}`);
        }
      }
    );
  });

  // Run Gauntlet command
  const runGauntletCmd = vscode.commands.registerCommand('aragora.runGauntlet', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage('No active editor');
      return;
    }

    const selection = editor.selection;
    const content = editor.document.getText(selection);

    if (!content) {
      vscode.window.showErrorMessage('No text selected');
      return;
    }

    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: 'Running Aragora Gauntlet',
        cancellable: false,
      },
      async () => {
        try {
          const result = await client.runGauntlet(content, 'code');
          vscode.window.showInformationMessage(
            `Gauntlet started! Session ID: ${result.session_id}`
          );
        } catch (error) {
          vscode.window.showErrorMessage(`Gauntlet failed: ${error}`);
        }
      }
    );
  });

  // List Agents command
  const listAgentsCmd = vscode.commands.registerCommand('aragora.listAgents', async () => {
    try {
      const result = await client.listAgents();
      const items = result.agents.map((a) => ({
        label: a.name || a.id,
        description: a.provider,
        detail: `Status: ${a.status} | Capabilities: ${a.capabilities?.join(', ') || 'general'}`,
      }));

      await vscode.window.showQuickPick(items, {
        placeHolder: 'Available Agents',
        matchOnDescription: true,
        matchOnDetail: true,
      });
    } catch (error) {
      vscode.window.showErrorMessage(`Failed to list agents: ${error}`);
    }
  });

  // Show Results command
  const showResultsCmd = vscode.commands.registerCommand('aragora.showResults', async () => {
    debatesProvider.loadDebates();
    vscode.commands.executeCommand('workbench.view.extension.aragora');
  });

  // Configure command
  const configureCmd = vscode.commands.registerCommand('aragora.configure', async () => {
    const action = await vscode.window.showQuickPick(
      [
        { label: 'Set API URL', description: 'Configure the Aragora API endpoint' },
        { label: 'Set API Key', description: 'Configure your API key' },
        { label: 'Set Default Agents', description: 'Configure default agents for debates' },
        { label: 'Set Default Rounds', description: 'Configure default number of rounds' },
      ],
      { placeHolder: 'What would you like to configure?' }
    );

    if (!action) return;

    const config = vscode.workspace.getConfiguration('aragora');

    if (action.label === 'Set API URL') {
      const value = await vscode.window.showInputBox({
        prompt: 'Enter API URL',
        value: config.get('apiUrl'),
      });
      if (value) await config.update('apiUrl', value, true);
    } else if (action.label === 'Set API Key') {
      const value = await vscode.window.showInputBox({
        prompt: 'Enter API Key',
        password: true,
      });
      if (value) await config.update('apiKey', value, true);
    } else if (action.label === 'Set Default Agents') {
      const value = await vscode.window.showInputBox({
        prompt: 'Enter default agents (comma-separated)',
        value: config.get('defaultAgents'),
      });
      if (value) await config.update('defaultAgents', value, true);
    } else if (action.label === 'Set Default Rounds') {
      const value = await vscode.window.showInputBox({
        prompt: 'Enter default number of rounds',
        value: String(config.get('defaultRounds')),
      });
      if (value) await config.update('defaultRounds', parseInt(value, 10), true);
    }
  });

  context.subscriptions.push(
    runDebateCmd,
    runGauntletCmd,
    listAgentsCmd,
    showResultsCmd,
    configureCmd
  );

  // Status bar item
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.text = '$(robot) Aragora';
  statusBar.tooltip = 'Aragora Multi-Agent Debate';
  statusBar.command = 'aragora.runDebate';
  statusBar.show();

  context.subscriptions.push(statusBar);

  vscode.window.showInformationMessage('Aragora extension activated!');
}

export function deactivate() {}
