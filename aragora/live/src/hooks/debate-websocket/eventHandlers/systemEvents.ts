/**
 * Handlers for system events
 * (ack, error, queue_overflow)
 */

import type { EventHandlerContext, ParsedEventData } from './types';
import type { StreamEvent } from '@/types/events';
import { logger } from '@/utils/logger';

/**
 * Handle acknowledgment event
 */
export function handleAckEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  const eventData = data.data;
  const msgType = (eventData?.message_type as string) || '';

  if (ctx.ackCallbackRef.current) {
    ctx.ackCallbackRef.current(msgType);
  }
}

/**
 * Handle error event
 */
export function handleErrorEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  const eventData = data.data;

  // Handle queue overflow notifications
  if (eventData?.error_type === 'queue_overflow') {
    logger.warn('[WebSocket] Server queue overflow:', eventData.message);
    ctx.errorCallbackRef.current?.(
      `Some updates may be missing (${eventData.dropped_count} events dropped)`
    );
    return;
  }

  const errorMsg = (eventData?.message as string) || 'Unknown error';

  if (eventData?.phase === 'initialization') {
    const event: StreamEvent = {
      type: 'error',
      data: (eventData as Record<string, unknown>) || {},
      timestamp: (data.timestamp as number) || Date.now() / 1000,
    };
    ctx.addStreamEvent(event);
  }

  if (ctx.errorCallbackRef.current) {
    ctx.errorCallbackRef.current(errorMsg);
  }

  // If this is a fatal error (e.g., invalid agent type), set error state
  if (eventData?.fatal || eventData?.error_type === 'validation_error') {
    ctx.setStatus('error');
    ctx.setError('Debate failed');
    ctx.setErrorDetails(errorMsg);
    ctx.clearDebateStartTimeout();
  }
}

/**
 * Registry of system event handlers
 */
export const systemHandlers = {
  ack: handleAckEvent,
  error: handleErrorEvent,
};
