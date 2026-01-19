/**
 * Handlers for token streaming events
 * (token_start, token_delta, token_end)
 */

import { logger } from '@/utils/logger';
import type { TranscriptMessage } from '../types';
import { makeStreamingKey } from '../utils';
import type { EventHandlerContext, ParsedEventData } from './types';

/**
 * Handle token_start event - begin streaming for an agent
 */
export function handleTokenStartEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  const eventData = data.data;
  const agent = (data.agent as string) || (eventData?.agent as string);
  const taskId = (data.task_id as string) || '';

  if (!agent) return;

  if (process.env.NODE_ENV === 'development') {
    logger.debug(`[WS] TOKEN_START: ${agent} (task=${taskId || 'default'})`);
  }

  const streamKey = makeStreamingKey(agent, taskId);

  ctx.setStreamingMessages(prev => {
    const updated = new Map(prev);
    updated.set(streamKey, {
      agent,
      taskId,
      content: '',
      isComplete: false,
      startTime: Date.now(),
      expectedSeq: 1,
      pendingTokens: new Map(),
    });
    return updated;
  });

  ctx.setAgents(prev => prev.includes(agent) ? prev : [...prev, agent]);
}

/**
 * Handle token_delta event - append token to streaming message
 */
export function handleTokenDeltaEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  const eventData = data.data;
  const agent = (data.agent as string) || (eventData?.agent as string);
  const taskId = (data.task_id as string) || '';
  const token = (eventData?.token as string) || '';
  const agentSeq = (data.agent_seq as number) || 0;

  if (!agent || !token) return;

  const streamKey = makeStreamingKey(agent, taskId);

  ctx.setStreamingMessages(prev => {
    const updated = new Map(prev);
    const existing = updated.get(streamKey);

    if (existing) {
      // If we have sequence info, use it for ordering
      if (agentSeq > 0) {
        // Check if this is the expected sequence
        if (agentSeq === existing.expectedSeq) {
          // Token is in order - append it
          let newContent = existing.content + token;
          let nextExpected = agentSeq + 1;

          // Check if we have buffered tokens that can now be appended
          const pending = new Map(existing.pendingTokens);
          while (pending.has(nextExpected)) {
            newContent += pending.get(nextExpected)!;
            pending.delete(nextExpected);
            nextExpected++;
          }

          updated.set(streamKey, {
            ...existing,
            content: newContent,
            expectedSeq: nextExpected,
            pendingTokens: pending,
          });
        } else if (agentSeq > existing.expectedSeq) {
          // Token arrived out of order - buffer it
          const pending = new Map(existing.pendingTokens);
          pending.set(agentSeq, token);
          updated.set(streamKey, {
            ...existing,
            pendingTokens: pending,
          });
        }
        // Ignore tokens with seq < expectedSeq (duplicate/old)
      } else {
        // No sequence info - fall back to simple append (backward compat)
        updated.set(streamKey, {
          ...existing,
          content: existing.content + token,
        });
      }
    } else {
      // First token for this agent+task (no token_start received)
      updated.set(streamKey, {
        agent,
        taskId,
        content: token,
        isComplete: false,
        startTime: Date.now(),
        expectedSeq: agentSeq > 0 ? agentSeq + 1 : 1,
        pendingTokens: new Map(),
      });
    }

    return updated;
  });
}

/**
 * Handle token_end event - complete streaming message
 */
export function handleTokenEndEvent(data: ParsedEventData, ctx: EventHandlerContext): void {
  const eventData = data.data;
  const agent = (data.agent as string) || (eventData?.agent as string);
  const taskId = (data.task_id as string) || '';

  if (!agent) return;

  const streamKey = makeStreamingKey(agent, taskId);

  ctx.setStreamingMessages(prev => {
    const updated = new Map(prev);
    const existing = updated.get(streamKey);

    if (existing) {
      // Flush any remaining buffered tokens in order
      let finalContent = existing.content;
      if (existing.pendingTokens.size > 0) {
        const sortedSeqs = Array.from(existing.pendingTokens.keys()).sort((a, b) => a - b);
        for (const seq of sortedSeqs) {
          finalContent += existing.pendingTokens.get(seq)!;
        }
      }

      // Log completion stats in development
      if (process.env.NODE_ENV === 'development') {
        const duration = Date.now() - existing.startTime;
        logger.debug(`[WS] TOKEN_END: ${agent} - ${finalContent.length} chars in ${duration}ms`);
      }

      if (finalContent) {
        const msg: TranscriptMessage = {
          agent,
          content: finalContent,
          timestamp: Date.now() / 1000,
        };

        // Use addMessageIfNew which handles deduplication
        ctx.addMessageIfNew(msg);
      }
    }

    updated.delete(streamKey);
    return updated;
  });
}

/**
 * Registry of token event handlers
 */
export const tokenHandlers = {
  token_start: handleTokenStartEvent,
  token_delta: handleTokenDeltaEvent,
  token_end: handleTokenEndEvent,
};
