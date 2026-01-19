/**
 * Event handlers for WebSocket events
 *
 * Exports all event handler registries and the combined processor.
 */

export type {
  EventHandlerContext,
  ParsedEventData,
  EventHandler,
  EventHandlerRegistry,
} from './types';

export { lifecycleHandlers } from './lifecycleEvents';
export { messageHandlers } from './messageEvents';
export { tokenHandlers } from './tokenEvents';
export { analyticsHandlers } from './analyticsEvents';
export { systemHandlers } from './systemEvents';

import { lifecycleHandlers } from './lifecycleEvents';
import { messageHandlers } from './messageEvents';
import { tokenHandlers } from './tokenEvents';
import { analyticsHandlers } from './analyticsEvents';
import { systemHandlers } from './systemEvents';
import type { EventHandlerRegistry } from './types';

/**
 * Combined registry of all event handlers
 */
export const eventHandlerRegistry: EventHandlerRegistry = {
  ...lifecycleHandlers,
  ...messageHandlers,
  ...tokenHandlers,
  ...analyticsHandlers,
  ...systemHandlers,
};
