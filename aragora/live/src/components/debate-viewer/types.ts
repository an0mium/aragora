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
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  userScrolled: boolean;
  onResumeAutoScroll: () => void;
  // Crux highlighting props
  cruxes?: CruxClaim[];
  showCruxHighlighting?: boolean;
  setShowCruxHighlighting?: (show: boolean) => void;
}

export interface ArchivedDebateViewProps {
  debate: DebateArtifact;
  onShare: () => void;
  copied: boolean;
}

export interface StreamingMessage {
  agent: string;
  taskId?: string;  // Task ID for composite React key support
  content: string;
  startTime: number;
}

export interface CruxClaim {
  claim_id: string;
  statement: string;
  author: string;
  crux_score?: number;
}

export interface TranscriptMessageCardProps {
  message: TranscriptMessage;
  cruxes?: CruxClaim[];
}

export interface StreamingMessageCardProps {
  message: StreamingMessage;
}

export { type TranscriptMessage } from '@/hooks/useDebateWebSocket';
export { type DebateArtifact } from '@/utils/supabase';
