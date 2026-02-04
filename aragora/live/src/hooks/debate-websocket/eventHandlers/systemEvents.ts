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
 * Handle authentication revoked event
 * Triggered when token is invalid or expired during an active connection
 */
export function handleAuthRevokedEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  const eventData = data.data;
  const reason = (eventData?.reason as string) || 'Token expired or invalid';

  logger.warn('[WebSocket] Authentication revoked:', reason);

  // Notify via error callback
  ctx.errorCallbackRef.current?.(`Authentication failed: ${reason}`);

  // Call the onAuthRevoked callback if provided
  if (ctx.onAuthRevoked) {
    ctx.onAuthRevoked();
  }

  // Set error state - connection will be closed by server
  ctx.setStatus('error');
  ctx.setError('Authentication required');
  ctx.setErrorDetails(reason);
}

/**
 * Registry of system event handlers
 */
export const systemHandlers = {
  ack: handleAckEvent,
  error: handleErrorEvent,
  auth_revoked: handleAuthRevokedEvent,
};
