/**
 * Audit Components - Visualize audit trails and decision receipts.
 *
 * These components surface the defensible decisions pillar of Aragora's
 * control plane positioning, providing full audit trails with cryptographic
 * integrity verification.
 */

export { AuditTrailViewer } from './AuditTrailViewer';
export type {
  AuditTrailViewerProps,
  AuditTrail,
  AuditEvent,
  AuditEventType,
} from './AuditTrailViewer';

export { DecisionReceiptViewer } from './DecisionReceiptViewer';
export type {
  DecisionReceiptViewerProps,
  DecisionReceipt,
  ReceiptFinding,
  ReceiptDissent,
  ReceiptVerification,
} from './DecisionReceiptViewer';
