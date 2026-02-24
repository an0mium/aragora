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
 * Handle heartbeat event - server liveness signal
 *
 * Updates the last-seen timestamp so the frontend can detect stalls
 * without guessing.  Also checks last_seq from the server to detect
 * missed events - if the server's latest seq is ahead of ours by
 * more than a small tolerance, dropped events may have occurred.
 */
export function handleHeartbeatEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  // Update last activity timestamp (used by stall detection)
  ctx.lastActivityRef.current = Date.now();

  // Check for sequence gaps via server-reported last_seq
  const eventData = data.data;
  const serverLastSeq = eventData?.last_seq as number | undefined;
  if (serverLastSeq && typeof serverLastSeq === 'number' && serverLastSeq > 0) {
    const clientLastSeq = ctx.lastSeqRef.current;
    if (clientLastSeq > 0) {
      const gap = serverLastSeq - clientLastSeq;
      if (gap > 10) {
        logger.warn(
          `[WS] Heartbeat gap detected: server seq=${serverLastSeq}, client seq=${clientLastSeq} (${gap} events behind)`
        );
      }
    }
  }
}

/**
 * Handle replay_start event - server is about to send missed events
 */
export function handleReplayStartEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  const eventData = data.data;
  const count = (eventData?.event_count as number) || 0;
  logger.debug(`[WS] Replay starting: ${count} missed events`);
  ctx.lastActivityRef.current = Date.now();
}

/**
 * Handle replay_end event - all missed events have been replayed
 */
export function handleReplayEndEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  const eventData = data.data;
  const count = (eventData?.replayed_count as number) || 0;
  logger.debug(`[WS] Replay complete: ${count} events replayed`);
  ctx.lastActivityRef.current = Date.now();
}

/**
 * Registry of system event handlers
 */
export const systemHandlers = {
  ack: handleAckEvent,
  error: handleErrorEvent,
  auth_revoked: handleAuthRevokedEvent,
  heartbeat: handleHeartbeatEvent,
  replay_start: handleReplayStartEvent,
  replay_end: handleReplayEndEvent,
};
