import { ReviewQueueCard } from './ReviewQueueCard';
import type { ReviewQueueDetail, ReviewQueueItem } from './types';

interface ReviewQueueListProps {
  items: ReviewQueueItem[];
  selectedNumber: number | null;
  expandedNumber: number | null;
  details: Record<number, ReviewQueueDetail>;
  detailLoading: Record<number, boolean>;
  requestChangesOpenFor: number | null;
  requestChangesDrafts: Record<number, string>;
  actionLoading: Record<number, 'approve' | 'request_changes' | 'defer' | null>;
  onSelect: (number: number) => void;
  onToggleExpand: (number: number) => void;
  onApprove: (item: ReviewQueueItem) => void;
  onDefer: (number: number) => void;
  onOpenDiff: (item: ReviewQueueItem) => void;
  onOpenRequestChanges: (number: number) => void;
  onRequestChangesDraftChange: (number: number, value: string) => void;
  onRequestChangesSubmit: (number: number) => void;
  onRequestChangesCancel: () => void;
}

export function ReviewQueueList({
  items,
  selectedNumber,
  expandedNumber,
  details,
  detailLoading,
  requestChangesOpenFor,
  requestChangesDrafts,
  actionLoading,
  onSelect,
  onToggleExpand,
  onApprove,
  onDefer,
  onOpenDiff,
  onOpenRequestChanges,
  onRequestChangesDraftChange,
  onRequestChangesSubmit,
  onRequestChangesCancel,
}: ReviewQueueListProps) {
  return (
    <div className="grid gap-4">
      {items.map((item) => (
        <ReviewQueueCard
          key={item.number}
          item={item}
          selected={selectedNumber === item.number}
          expanded={expandedNumber === item.number}
          detail={details[item.number] || null}
          detailLoading={Boolean(detailLoading[item.number])}
          requestChangesOpen={requestChangesOpenFor === item.number}
          requestChangesDraft={requestChangesDrafts[item.number] || ''}
          actionLoading={actionLoading[item.number] || null}
          onSelect={() => onSelect(item.number)}
          onToggleExpand={() => onToggleExpand(item.number)}
          onApprove={() => onApprove(item)}
          onDefer={() => onDefer(item.number)}
          onOpenDiff={() => onOpenDiff(item)}
          onOpenRequestChanges={() => onOpenRequestChanges(item.number)}
          onRequestChangesDraftChange={(value) => onRequestChangesDraftChange(item.number, value)}
          onRequestChangesSubmit={() => onRequestChangesSubmit(item.number)}
          onRequestChangesCancel={onRequestChangesCancel}
        />
      ))}
    </div>
  );
}
