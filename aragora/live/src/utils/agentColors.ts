/**
 * Centralized agent color scheme definitions.
 * Maps AI model families to consistent color themes across the app.
 */

export interface AgentColorScheme {
  /** Background color class (e.g., 'bg-purple/10') */
  bg: string;
  /** Text color class (e.g., 'text-purple') */
  text: string;
  /** Border color class (e.g., 'border-purple/40') */
  border: string;
  /** Optional glow effect class */
  glow?: string;
  /** Optional tab indicator color */
  tab?: string;
}

/**
 * Color schemes for each AI model family.
 * - Gemini (Google): Purple
 * - Codex (OpenAI): Gold
 * - Claude (Anthropic): Cyan
 * - Grok (xAI): Crimson/Red
 * - Default: Acid Green
 */
export const AGENT_COLORS: Record<string, AgentColorScheme> = {
  gemini: {
    bg: 'bg-purple/10',
    text: 'text-purple',
    border: 'border-purple/40',
    glow: 'shadow-[0_0_10px_rgba(191,0,255,0.1)]',
    tab: 'bg-purple-500',
  },
  codex: {
    bg: 'bg-gold/10',
    text: 'text-gold',
    border: 'border-gold/40',
    glow: 'shadow-[0_0_10px_rgba(255,215,0,0.1)]',
    tab: 'bg-yellow-500',
  },
  claude: {
    bg: 'bg-acid-cyan/10',
    text: 'text-acid-cyan',
    border: 'border-acid-cyan/40',
    glow: 'shadow-[0_0_10px_rgba(0,255,255,0.1)]',
    tab: 'bg-indigo-500',
  },
  grok: {
    bg: 'bg-crimson/10',
    text: 'text-crimson',
    border: 'border-crimson/40',
    glow: 'shadow-[0_0_10px_rgba(255,0,64,0.1)]',
    tab: 'bg-red-500',
  },
  default: {
    bg: 'bg-acid-green/10',
    text: 'text-acid-green',
    border: 'border-acid-green/40',
    glow: '',
    tab: 'bg-gray-500',
  },
};

/**
 * Get colors for an agent by name, using prefix matching.
 * This ensures any variant (e.g., "grok-explorer", "claude-3-opus") gets the right color.
 *
 * @param agentName - The agent/model name to look up
 * @returns The color scheme for the agent
 */
export function getAgentColors(agentName: string): AgentColorScheme {
  const name = agentName.toLowerCase();

  if (name.startsWith('gemini')) return AGENT_COLORS.gemini;
  if (name.startsWith('codex') || name.startsWith('gpt') || name.startsWith('openai')) {
    return AGENT_COLORS.codex;
  }
  if (name.startsWith('claude') || name.startsWith('anthropic')) {
    return AGENT_COLORS.claude;
  }
  if (name.startsWith('grok') || name.startsWith('xai')) {
    return AGENT_COLORS.grok;
  }

  return AGENT_COLORS.default;
}

/**
 * Get just the text color class for an agent.
 * Convenience function for simpler use cases.
 *
 * @param agentName - The agent/model name to look up
 * @returns Just the text color class
 */
export function getAgentTextColor(agentName: string): string {
  return getAgentColors(agentName).text;
}
