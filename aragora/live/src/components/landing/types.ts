export interface HeroSectionProps {
  error: string | null;
  activeDebateId: string | null;
  activeQuestion: string | null;
  apiBase: string;
  onDismissError: () => void;
  onDebateStarted: (debateId: string, question: string) => void;
  onError: (err: string) => void;
}

export interface LandingPreparedDebateOption {
  id: string;
  label: string;
  description: string;
  originalQuestion: string;
  interpretedQuestion: string;
  debatePrompt: string;
  agents: number;
  rounds: number;
  recommended?: boolean;
}

export interface LandingDebatePreflight {
  title: string;
  prompt: string;
  warning?: string;
  options: LandingPreparedDebateOption[];
}
