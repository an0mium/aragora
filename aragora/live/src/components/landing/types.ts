export interface HeroSectionProps {
  error: string | null;
  activeDebateId: string | null;
  activeQuestion: string | null;
  apiBase: string;
  onDismissError: () => void;
  onDebateStarted: (debateId: string, question: string) => void;
  onError: (err: string) => void;
}
