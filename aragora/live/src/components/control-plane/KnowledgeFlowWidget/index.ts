/**
 * Knowledge Flow Widget
 *
 * Components for monitoring knowledge ingestion, freshness, and refresh schedules.
 * Provides visibility into the health and currency of organizational knowledge.
 */

export {
  KnowledgeFlowWidget,
  type KnowledgeFlowWidgetProps,
  type KnowledgeFlowStats,
  type RefreshSchedule,
} from './KnowledgeFlowWidget';

export {
  IngestionStatusCard,
  type IngestionStatusCardProps,
  type ConnectorIngestionStatus,
  type IngestionStatus,
} from './IngestionStatusCard';

export {
  KnowledgeAgeHistogram,
  type KnowledgeAgeHistogramProps,
  type AgeDistribution,
} from './KnowledgeAgeHistogram';
