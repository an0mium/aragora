import type { DebateArtifact } from '@/utils/supabase';
import type { TranscriptMessage } from '@/hooks/useDebateWebSocket';
import type { StreamEvent } from '@/types/events';

export interface DebateViewerProps {
  debateId: string;
  wsUrl?: string;
}

export interface LiveDebateViewProps {
  debateId: string;
  status: 'connecting' | 'streaming' | 'complete' | 'error';
  task: string;
  agents: string[];
  messages: TranscriptMessage[];
  streamingMessages: Map<string, StreamingMessage>;
  streamEvents: StreamEvent[];
  hasCitations: boolean;
  showCitations: boolean;
  setShowCitations: (show: boolean) => void;
  showParticipation: boolean;
  setShowParticipation: (show: boolean) => void;
  onShare: () => void;
  copied: boolean;
  onVote: (choice: string, intensity?: number) => void;
  onSuggest: (suggestion: string) => void;
  onAck: (callback: (msgType: string) => void) => () => void;
  onError: (callback: (message: string) => void) => () => void;
  // Scroll handling props
  scrollContainerRef: React.RefObject<HTMLDivElement>;
  onScroll: () => void;
  userScrolled: boolean;
  onResumeAutoScroll: () => void;
}

export interface ArchivedDebateViewProps {
  debate: DebateArtifact;
  onShare: () => void;
  copied: boolean;
}

export interface StreamingMessage {
  agent: string;
  content: string;
  startTime: number;
}

export interface TranscriptMessageCardProps {
  message: TranscriptMessage;
}

export interface StreamingMessageCardProps {
  message: StreamingMessage;
}

export { type TranscriptMessage } from '@/hooks/useDebateWebSocket';
export { type DebateArtifact } from '@/utils/supabase';
