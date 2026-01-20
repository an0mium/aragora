// Auto-generated TypeScript types from OpenAPI spec
// Do not edit manually

// API Configuration
export interface ApiConfig {
  baseUrl: string;
  apiKey?: string;
  headers?: Record<string, string>;
}

export interface Error {
  error: string;
  code?: "INVALID_JSON" | "MISSING_FIELD" | "INVALID_VALUE" | "AUTH_REQUIRED" | "INVALID_TOKEN" | "FORBIDDEN" | "NOT_OWNER" | "NOT_FOUND" | "QUOTA_EXCEEDED" | "RATE_LIMITED" | "INTERNAL_ERROR" | "SERVICE_UNAVAILABLE" | "AGENT_TIMEOUT" | "CONSENSUS_FAILED";
  trace_id?: string;
  field?: string;
  resource_type?: string;
  resource_id?: string;
  limit?: number;
  retry_after?: number;
  resets_at?: string;
  upgrade_url?: string;
  support_url?: string;
}

export interface PaginatedResponse {
  total?: number;
  offset?: number;
  limit?: number;
  has_more?: boolean;
}

export interface Agent {
  name?: string;
  elo?: number;
  matches?: number;
  wins?: number;
  losses?: number;
  calibration_score?: number;
}

export type DebateStatus = "created" | "starting" | "pending" | "running" | "in_progress" | "completed" | "failed" | "cancelled" | "paused" | "active" | "concluded" | "archived";

export interface ConsensusResult {
  reached?: boolean;
  agreement?: number;
  confidence?: number;
  final_answer?: string;
  conclusion?: string;
  supporting_agents?: string[];
  dissenting_agents?: string[];
}

export interface DebateCreateRequest {
  task: string;
  question?: string;
  agents?: string[];
  rounds?: number;
  consensus?: "majority" | "unanimous" | "weighted" | "semantic";
  context?: string;
  auto_select?: boolean;
  auto_select_config?: Record<string, unknown>;
  use_trending?: boolean;
  trending_category?: "tech" | "science" | "politics" | "business" | "health";
}

export interface DebateCreateResponse {
  success: boolean;
  debate_id?: string;
  status?: DebateStatus;
  task?: string;
  agents?: string[];
  websocket_url?: string;
  estimated_duration?: number;
  error?: string;
}

export interface Debate {
  debate_id?: string;
  id?: string;
  slug?: string;
  task?: string;
  topic?: string;
  context?: string;
  status?: DebateStatus;
  outcome?: string;
  final_answer?: string;
  consensus?: ConsensusResult;
  consensus_proof?: Record<string, unknown>;
  consensus_reached?: boolean;
  confidence?: number;
  rounds_used?: number;
  duration_seconds?: number;
  agents?: string[];
  rounds?: Round[];
  created_at?: string;
  completed_at?: string;
  metadata?: Record<string, unknown>;
}

export interface Message {
  role?: "system" | "user" | "assistant";
  content?: string;
  agent?: string;
  agent_id?: string;
  round?: number;
  timestamp?: string;
}

export interface HealthCheck {
  status?: "healthy" | "degraded" | "unhealthy";
  version?: string;
  timestamp?: string;
  checks?: Record<string, Record<string, unknown>>;
  response_time_ms?: number;
}

export interface Consensus {
  reached?: boolean;
  topic?: string;
  verdict?: string;
  confidence?: number;
  participating_agents?: string[];
}

export interface Calibration {
  agent?: string;
  score?: number;
  bucket_stats?: Record<string, unknown>[];
  overconfidence_index?: number;
}

export interface Relationship {
  agent_a?: string;
  agent_b?: string;
  alliance_score?: number;
  rivalry_score?: number;
  total_interactions?: number;
}

export interface OAuthProvider {
  id: string;
  name: string;
}

export interface OAuthProviders {
  providers: OAuthProvider[];
}

export interface Round {
  round_number?: number;
  messages?: Message[];
  votes?: Record<string, unknown>;
  summary?: string;
}

export interface Workspace {
  id: string;
  organization_id: string;
  name: string;
  created_at?: string;
  created_by?: string;
  encrypted?: boolean;
  retention_days?: number;
  sensitivity_level?: string;
  document_count?: number;
  storage_bytes?: number;
}

export interface WorkspaceList {
  workspaces: Workspace[];
  total: number;
}

export interface RetentionPolicy {
  id?: string;
  name?: string;
  retention_days?: number;
  data_types?: string[];
  enabled?: boolean;
  created_at?: string;
}

export interface RetentionPolicyList {
  policies?: RetentionPolicy[];
  total?: number;
}

export interface StepDefinition {
  id?: string;
  name?: string;
  type?: string;
  config?: Record<string, unknown>;
  depends_on?: string[];
}

export interface TransitionRule {
  from_step?: string;
  to_step?: string;
  condition?: string;
}

export interface Workflow {
  id?: string;
  name?: string;
  description?: string;
  version?: string;
  status?: string;
  steps?: StepDefinition[];
  transitions?: TransitionRule[];
  created_at?: string;
  updated_at?: string;
}

export interface WorkflowList {
  workflows?: Workflow[];
  total?: number;
}

export interface WorkflowUpdate {
  name?: string;
  description?: string;
  steps?: StepDefinition[];
  transitions?: TransitionRule[];
}

export interface WorkflowTemplate {
  id?: string;
  name?: string;
  description?: string;
  category?: string;
  steps?: StepDefinition[];
  parameters?: Record<string, unknown>;
}

export interface WorkflowTemplateList {
  templates?: WorkflowTemplate[];
  total?: number;
}

export interface ExecutionList {
  executions?: Record<string, unknown>[];
  total?: number;
}

export interface DecisionReceipt {
  id: string;
  debate_id: string;
  verdict: string;
  confidence?: number;
  consensus_reached?: boolean;
  participating_agents?: string[];
  dissenting_agents?: string[];
  evidence?: Record<string, unknown>[];
  reasoning?: string;
  hash?: string;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

export interface ReceiptList {
  receipts: DecisionReceipt[];
  total: number;
  offset?: number;
  limit?: number;
  has_more?: boolean;
}

export interface RiskHeatmap {
  id: string;
  gauntlet_id?: string;
  categories: string[];
  scores: number[];
  matrix?: number[][];
  overall_risk?: number;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

export interface HeatmapList {
  heatmaps: RiskHeatmap[];
  total: number;
  offset?: number;
  limit?: number;
}

export interface PatternTemplate {
  id: string;
  name: string;
  description?: string;
  pattern_type: "hive_mind" | "map_reduce" | "review_cycle";
  parameters?: Record<string, unknown>;
  example_config?: Record<string, unknown>;
  created_at?: string;
}

export interface PatternTemplateList {
  templates: PatternTemplate[];
  total: number;
}

export interface CheckpointMetadata {
  id: string;
  debate_id?: string;
  workflow_id?: string;
  name?: string;
  description?: string;
  state?: Record<string, unknown>;
  round_number?: number;
  created_at: string;
  created_by?: string;
  size_bytes?: number;
}

export interface CheckpointList {
  checkpoints: CheckpointMetadata[];
  total: number;
  offset?: number;
  limit?: number;
}

export interface RestoreResult {
  success: boolean;
  checkpoint_id: string;
  debate_id?: string;
  workflow_id?: string;
  restored_at?: string;
  state_restored?: boolean;
  message?: string;
}

// API Endpoints
export type ApiEndpoints = {
  'get__api_health': {
    path: '/api/health';
    method: 'GET';
  };
  'get__api_health_detailed': {
    path: '/api/health/detailed';
    method: 'GET';
  };
  'get__api_nomic_state': {
    path: '/api/nomic/state';
    method: 'GET';
  };
  'get__api_nomic_health': {
    path: '/api/nomic/health';
    method: 'GET';
  };
  'get__api_nomic_log': {
    path: '/api/nomic/log';
    method: 'GET';
  };
  'get__api_nomic_risk-register': {
    path: '/api/nomic/risk-register';
    method: 'GET';
  };
  'get__api_modes': {
    path: '/api/modes';
    method: 'GET';
  };
  'get__api_history_cycles': {
    path: '/api/history/cycles';
    method: 'GET';
  };
  'get__api_history_events': {
    path: '/api/history/events';
    method: 'GET';
  };
  'get__api_history_debates': {
    path: '/api/history/debates';
    method: 'GET';
  };
  'get__api_history_summary': {
    path: '/api/history/summary';
    method: 'GET';
  };
  'get__api_system_maintenance': {
    path: '/api/system/maintenance';
    method: 'GET';
  };
  'get__api_openapi': {
    path: '/api/openapi';
    method: 'GET';
  };
  'get__api_agents': {
    path: '/api/agents';
    method: 'GET';
  };
  'get__api_leaderboard': {
    path: '/api/leaderboard';
    method: 'GET';
  };
  'get__api_rankings': {
    path: '/api/rankings';
    method: 'GET';
  };
  'get__api_leaderboard-view': {
    path: '/api/leaderboard-view';
    method: 'GET';
  };
  'get__api_agent_{name}_profile': {
    path: '/api/agent/{name}/profile';
    method: 'GET';
  };
  'get__api_agent_{name}_history': {
    path: '/api/agent/{name}/history';
    method: 'GET';
  };
  'get__api_agent_{name}_calibration': {
    path: '/api/agent/{name}/calibration';
    method: 'GET';
  };
  'get__api_agent_{name}_calibration-curve': {
    path: '/api/agent/{name}/calibration-curve';
    method: 'GET';
  };
  'get__api_agent_{name}_calibration-summary': {
    path: '/api/agent/{name}/calibration-summary';
    method: 'GET';
  };
  'get__api_agent_{name}_consistency': {
    path: '/api/agent/{name}/consistency';
    method: 'GET';
  };
  'get__api_agent_{name}_flips': {
    path: '/api/agent/{name}/flips';
    method: 'GET';
  };
  'get__api_agent_{name}_network': {
    path: '/api/agent/{name}/network';
    method: 'GET';
  };
  'get__api_agent_{name}_rivals': {
    path: '/api/agent/{name}/rivals';
    method: 'GET';
  };
  'get__api_agent_{name}_allies': {
    path: '/api/agent/{name}/allies';
    method: 'GET';
  };
  'get__api_agent_{name}_moments': {
    path: '/api/agent/{name}/moments';
    method: 'GET';
  };
  'get__api_agent_{name}_reputation': {
    path: '/api/agent/{name}/reputation';
    method: 'GET';
  };
  'get__api_agent_{name}_persona': {
    path: '/api/agent/{name}/persona';
    method: 'GET';
  };
  'get__api_agent_{name}_grounded-persona': {
    path: '/api/agent/{name}/grounded-persona';
    method: 'GET';
  };
  'get__api_agent_{name}_identity-prompt': {
    path: '/api/agent/{name}/identity-prompt';
    method: 'GET';
  };
  'get__api_agent_{name}_performance': {
    path: '/api/agent/{name}/performance';
    method: 'GET';
  };
  'get__api_agent_{name}_domains': {
    path: '/api/agent/{name}/domains';
    method: 'GET';
  };
  'get__api_agent_{name}_accuracy': {
    path: '/api/agent/{name}/accuracy';
    method: 'GET';
  };
  'get__api_agent_compare': {
    path: '/api/agent/compare';
    method: 'GET';
  };
  'get__api_matches_recent': {
    path: '/api/matches/recent';
    method: 'GET';
  };
  'get__api_calibration_leaderboard': {
    path: '/api/calibration/leaderboard';
    method: 'GET';
  };
  'get__api_personas': {
    path: '/api/personas';
    method: 'GET';
  };
  'listDebates': {
    path: '/api/debates';
    method: 'GET';
  };
  'createDebate': {
    path: '/api/debates';
    method: 'POST';
  };
  'post__api_debate': {
    path: '/api/debate';
    method: 'POST';
  };
  'get__api_debates_{id}': {
    path: '/api/debates/{id}';
    method: 'GET';
  };
  'get__api_debates_slug_{slug}': {
    path: '/api/debates/slug/{slug}';
    method: 'GET';
  };
  'get__api_debates_{id}_messages': {
    path: '/api/debates/{id}/messages';
    method: 'GET';
  };
  'get__api_debates_{id}_convergence': {
    path: '/api/debates/{id}/convergence';
    method: 'GET';
  };
  'get__api_debates_{id}_citations': {
    path: '/api/debates/{id}/citations';
    method: 'GET';
  };
  'get__api_debates_{id}_evidence': {
    path: '/api/debates/{id}/evidence';
    method: 'GET';
  };
  'get__api_debates_{id}_impasse': {
    path: '/api/debates/{id}/impasse';
    method: 'GET';
  };
  'get__api_debates_{id}_meta-critique': {
    path: '/api/debates/{id}/meta-critique';
    method: 'GET';
  };
  'get__api_debates_{id}_graph_stats': {
    path: '/api/debates/{id}/graph/stats';
    method: 'GET';
  };
  'post__api_debates_{id}_fork': {
    path: '/api/debates/{id}/fork';
    method: 'POST';
  };
  'get__api_debates_{id}_export_{format}': {
    path: '/api/debates/{id}/export/{format}';
    method: 'GET';
  };
  'post__api_debates_{id}_broadcast': {
    path: '/api/debates/{id}/broadcast';
    method: 'POST';
  };
  'post__api_debates_{id}_publish_twitter': {
    path: '/api/debates/{id}/publish/twitter';
    method: 'POST';
  };
  'post__api_debates_{id}_publish_youtube': {
    path: '/api/debates/{id}/publish/youtube';
    method: 'POST';
  };
  'get__api_debates_{id}_red-team': {
    path: '/api/debates/{id}/red-team';
    method: 'GET';
  };
  'get__api_search': {
    path: '/api/search';
    method: 'GET';
  };
  'get__api_dashboard_debates': {
    path: '/api/dashboard/debates';
    method: 'GET';
  };
  'get__api_analytics_disagreements': {
    path: '/api/analytics/disagreements';
    method: 'GET';
  };
  'get__api_analytics_role-rotation': {
    path: '/api/analytics/role-rotation';
    method: 'GET';
  };
  'get__api_analytics_early-stops': {
    path: '/api/analytics/early-stops';
    method: 'GET';
  };
  'get__api_ranking_stats': {
    path: '/api/ranking/stats';
    method: 'GET';
  };
  'get__api_memory_stats': {
    path: '/api/memory/stats';
    method: 'GET';
  };
  'get__api_flips_recent': {
    path: '/api/flips/recent';
    method: 'GET';
  };
  'get__api_flips_summary': {
    path: '/api/flips/summary';
    method: 'GET';
  };
  'get__api_insights_recent': {
    path: '/api/insights/recent';
    method: 'GET';
  };
  'post__api_insights_extract-detailed': {
    path: '/api/insights/extract-detailed';
    method: 'POST';
  };
  'get__api_moments_summary': {
    path: '/api/moments/summary';
    method: 'GET';
  };
  'get__api_moments_timeline': {
    path: '/api/moments/timeline';
    method: 'GET';
  };
  'get__api_moments_trending': {
    path: '/api/moments/trending';
    method: 'GET';
  };
  'get__api_moments_by-type_{type}': {
    path: '/api/moments/by-type/{type}';
    method: 'GET';
  };
  'get__api_consensus_similar': {
    path: '/api/consensus/similar';
    method: 'GET';
  };
  'get__api_consensus_settled': {
    path: '/api/consensus/settled';
    method: 'GET';
  };
  'get__api_consensus_stats': {
    path: '/api/consensus/stats';
    method: 'GET';
  };
  'get__api_consensus_dissents': {
    path: '/api/consensus/dissents';
    method: 'GET';
  };
  'get__api_consensus_contrarian-views': {
    path: '/api/consensus/contrarian-views';
    method: 'GET';
  };
  'get__api_consensus_risk-warnings': {
    path: '/api/consensus/risk-warnings';
    method: 'GET';
  };
  'get__api_consensus_domain_{domain}': {
    path: '/api/consensus/domain/{domain}';
    method: 'GET';
  };
  'get__api_relationships_summary': {
    path: '/api/relationships/summary';
    method: 'GET';
  };
  'get__api_relationships_graph': {
    path: '/api/relationships/graph';
    method: 'GET';
  };
  'get__api_relationships_stats': {
    path: '/api/relationships/stats';
    method: 'GET';
  };
  'get__api_relationship_{agent_a}_{agent_b}': {
    path: '/api/relationship/{agent_a}/{agent_b}';
    method: 'GET';
  };
  'get__api_memory_continuum_retrieve': {
    path: '/api/memory/continuum/retrieve';
    method: 'GET';
  };
  'post__api_memory_continuum_consolidate': {
    path: '/api/memory/continuum/consolidate';
    method: 'POST';
  };
  'post__api_memory_continuum_cleanup': {
    path: '/api/memory/continuum/cleanup';
    method: 'POST';
  };
  'get__api_memory_tier-stats': {
    path: '/api/memory/tier-stats';
    method: 'GET';
  };
  'get__api_memory_archive-stats': {
    path: '/api/memory/archive-stats';
    method: 'GET';
  };
  'get__api_belief-network_{debate_id}_cruxes': {
    path: '/api/belief-network/{debate_id}/cruxes';
    method: 'GET';
  };
  'get__api_belief-network_{debate_id}_load-bearing-claims': {
    path: '/api/belief-network/{debate_id}/load-bearing-claims';
    method: 'GET';
  };
  'get__api_debate_{debate_id}_graph-stats': {
    path: '/api/debate/{debate_id}/graph-stats';
    method: 'GET';
  };
  'get__api_pulse_trending': {
    path: '/api/pulse/trending';
    method: 'GET';
  };
  'get__api_pulse_suggest': {
    path: '/api/pulse/suggest';
    method: 'GET';
  };
  'get__api_metrics': {
    path: '/api/metrics';
    method: 'GET';
  };
  'get__api_metrics_health': {
    path: '/api/metrics/health';
    method: 'GET';
  };
  'get__api_metrics_cache': {
    path: '/api/metrics/cache';
    method: 'GET';
  };
  'get__api_metrics_system': {
    path: '/api/metrics/system';
    method: 'GET';
  };
  'get__metrics': {
    path: '/metrics';
    method: 'GET';
  };
  'get__api_verification_status': {
    path: '/api/verification/status';
    method: 'GET';
  };
  'post__api_verification_formal-verify': {
    path: '/api/verification/formal-verify';
    method: 'POST';
  };
  'post__api_debates_capability-probe': {
    path: '/api/debates/capability-probe';
    method: 'POST';
  };
  'post__api_debates_deep-audit': {
    path: '/api/debates/deep-audit';
    method: 'POST';
  };
  'post__api_probes_capability': {
    path: '/api/probes/capability';
    method: 'POST';
  };
  'get__api_documents': {
    path: '/api/documents';
    method: 'GET';
  };
  'get__api_documents_formats': {
    path: '/api/documents/formats';
    method: 'GET';
  };
  'post__api_documents_upload': {
    path: '/api/documents/upload';
    method: 'POST';
  };
  'get__api_podcast_feed.xml': {
    path: '/api/podcast/feed.xml';
    method: 'GET';
  };
  'get__api_podcast_episodes': {
    path: '/api/podcast/episodes';
    method: 'GET';
  };
  'get__api_youtube_auth': {
    path: '/api/youtube/auth';
    method: 'GET';
  };
  'get__api_youtube_callback': {
    path: '/api/youtube/callback';
    method: 'GET';
  };
  'get__api_youtube_status': {
    path: '/api/youtube/status';
    method: 'GET';
  };
  'get__api_plugins': {
    path: '/api/plugins';
    method: 'GET';
  };
  'get__api_plugins_{name}': {
    path: '/api/plugins/{name}';
    method: 'GET';
  };
  'post__api_plugins_{name}_run': {
    path: '/api/plugins/{name}/run';
    method: 'POST';
  };
  'get__api_laboratory_emergent-traits': {
    path: '/api/laboratory/emergent-traits';
    method: 'GET';
  };
  'get__api_laboratory_cross-pollinations_suggest': {
    path: '/api/laboratory/cross-pollinations/suggest';
    method: 'GET';
  };
  'get__api_tournaments': {
    path: '/api/tournaments';
    method: 'GET';
  };
  'get__api_tournaments_{id}_standings': {
    path: '/api/tournaments/{id}/standings';
    method: 'GET';
  };
  'get__api_genesis_stats': {
    path: '/api/genesis/stats';
    method: 'GET';
  };
  'get__api_genesis_events': {
    path: '/api/genesis/events';
    method: 'GET';
  };
  'get__api_genesis_lineage_{agent}': {
    path: '/api/genesis/lineage/{agent}';
    method: 'GET';
  };
  'get__api_genesis_tree_{agent}': {
    path: '/api/genesis/tree/{agent}';
    method: 'GET';
  };
  'get__api_evolution_{agent}_history': {
    path: '/api/evolution/{agent}/history';
    method: 'GET';
  };
  'get__api_replays': {
    path: '/api/replays';
    method: 'GET';
  };
  'get__api_replays_{id}': {
    path: '/api/replays/{id}';
    method: 'GET';
  };
  'get__api_learning_evolution': {
    path: '/api/learning/evolution';
    method: 'GET';
  };
  'get__api_meta-learning_stats': {
    path: '/api/meta-learning/stats';
    method: 'GET';
  };
  'get__api_critiques_patterns': {
    path: '/api/critiques/patterns';
    method: 'GET';
  };
  'get__api_critiques_archive': {
    path: '/api/critiques/archive';
    method: 'GET';
  };
  'get__api_reputation_all': {
    path: '/api/reputation/all';
    method: 'GET';
  };
  'get__api_routing_best-teams': {
    path: '/api/routing/best-teams';
    method: 'GET';
  };
  'post__api_routing_recommendations': {
    path: '/api/routing/recommendations';
    method: 'POST';
  };
  'get__api_introspection_all': {
    path: '/api/introspection/all';
    method: 'GET';
  };
  'get__api_introspection_leaderboard': {
    path: '/api/introspection/leaderboard';
    method: 'GET';
  };
  'get__api_introspection_agents': {
    path: '/api/introspection/agents';
    method: 'GET';
  };
  'get__api_introspection_agents_{name}': {
    path: '/api/introspection/agents/{name}';
    method: 'GET';
  };
  'get__api_auth_oauth_google': {
    path: '/api/auth/oauth/google';
    method: 'GET';
  };
  'get__api_auth_oauth_google_callback': {
    path: '/api/auth/oauth/google/callback';
    method: 'GET';
  };
  'get__api_auth_oauth_github': {
    path: '/api/auth/oauth/github';
    method: 'GET';
  };
  'get__api_auth_oauth_github_callback': {
    path: '/api/auth/oauth/github/callback';
    method: 'GET';
  };
  'post__api_auth_oauth_link': {
    path: '/api/auth/oauth/link';
    method: 'POST';
  };
  'delete__api_auth_oauth_unlink': {
    path: '/api/auth/oauth/unlink';
    method: 'DELETE';
  };
  'get__api_auth_oauth_providers': {
    path: '/api/auth/oauth/providers';
    method: 'GET';
  };
  'get__api_user_oauth-providers': {
    path: '/api/user/oauth-providers';
    method: 'GET';
  };
  'get__api_workspaces': {
    path: '/api/workspaces';
    method: 'GET';
  };
  'post__api_workspaces': {
    path: '/api/workspaces';
    method: 'POST';
  };
  'get__api_workspaces_{workspace_id}': {
    path: '/api/workspaces/{workspace_id}';
    method: 'GET';
  };
  'delete__api_workspaces_{workspace_id}': {
    path: '/api/workspaces/{workspace_id}';
    method: 'DELETE';
  };
  'post__api_workspaces_{workspace_id}_members': {
    path: '/api/workspaces/{workspace_id}/members';
    method: 'POST';
  };
  'get__api_retention_policies': {
    path: '/api/retention/policies';
    method: 'GET';
  };
  'post__api_retention_policies': {
    path: '/api/retention/policies';
    method: 'POST';
  };
  'post__api_retention_policies_{policy_id}_execute': {
    path: '/api/retention/policies/{policy_id}/execute';
    method: 'POST';
  };
  'get__api_retention_expiring': {
    path: '/api/retention/expiring';
    method: 'GET';
  };
  'post__api_classify': {
    path: '/api/classify';
    method: 'POST';
  };
  'get__api_classify_policy_{level}': {
    path: '/api/classify/policy/{level}';
    method: 'GET';
  };
  'get__api_audit_entries': {
    path: '/api/audit/entries';
    method: 'GET';
  };
  'get__api_audit_report': {
    path: '/api/audit/report';
    method: 'GET';
  };
  'get__api_audit_verify': {
    path: '/api/audit/verify';
    method: 'GET';
  };
  'get__api_workflows': {
    path: '/api/workflows';
    method: 'GET';
  };
  'post__api_workflows': {
    path: '/api/workflows';
    method: 'POST';
  };
  'get__api_workflows_{workflow_id}': {
    path: '/api/workflows/{workflow_id}';
    method: 'GET';
  };
  'put__api_workflows_{workflow_id}': {
    path: '/api/workflows/{workflow_id}';
    method: 'PUT';
  };
  'delete__api_workflows_{workflow_id}': {
    path: '/api/workflows/{workflow_id}';
    method: 'DELETE';
  };
  'post__api_workflows_{workflow_id}_execute': {
    path: '/api/workflows/{workflow_id}/execute';
    method: 'POST';
  };
  'get__api_workflows_{workflow_id}_versions': {
    path: '/api/workflows/{workflow_id}/versions';
    method: 'GET';
  };
  'get__api_workflow-templates': {
    path: '/api/workflow-templates';
    method: 'GET';
  };
  'get__api_workflow-templates_{template_id}': {
    path: '/api/workflow-templates/{template_id}';
    method: 'GET';
  };
  'get__api_workflow-executions': {
    path: '/api/workflow-executions';
    method: 'GET';
  };
  'get__api_workflow-executions_{execution_id}': {
    path: '/api/workflow-executions/{execution_id}';
    method: 'GET';
  };
  'delete__api_workflow-executions_{execution_id}': {
    path: '/api/workflow-executions/{execution_id}';
    method: 'DELETE';
  };
  'get__api_workflow-approvals': {
    path: '/api/workflow-approvals';
    method: 'GET';
  };
  'post__api_workflow-approvals_{approval_id}': {
    path: '/api/workflow-approvals/{approval_id}';
    method: 'POST';
  };
  'get__api_cross-pollination_stats': {
    path: '/api/cross-pollination/stats';
    method: 'GET';
  };
  'get__api_cross-pollination_subscribers': {
    path: '/api/cross-pollination/subscribers';
    method: 'GET';
  };
  'get__api_cross-pollination_bridge': {
    path: '/api/cross-pollination/bridge';
    method: 'GET';
  };
  'get__api_cross-pollination_metrics': {
    path: '/api/cross-pollination/metrics';
    method: 'GET';
  };
  'post__api_cross-pollination_reset': {
    path: '/api/cross-pollination/reset';
    method: 'POST';
  };
  'get__api_cross-pollination_handlers_{handler_name}_circuit-breaker': {
    path: '/api/cross-pollination/handlers/{handler_name}/circuit-breaker';
    method: 'GET';
  };
  'post__api_cross-pollination_handlers_{handler_name}_circuit-breaker': {
    path: '/api/cross-pollination/handlers/{handler_name}/circuit-breaker';
    method: 'POST';
  };
  'get__api_gauntlet_receipts': {
    path: '/api/gauntlet/receipts';
    method: 'GET';
  };
  'get__api_gauntlet_receipts_{receipt_id}': {
    path: '/api/gauntlet/receipts/{receipt_id}';
    method: 'GET';
  };
  'get__api_gauntlet_receipts_{receipt_id}_export': {
    path: '/api/gauntlet/receipts/{receipt_id}/export';
    method: 'GET';
  };
  'post__api_gauntlet_receipts_export_bundle': {
    path: '/api/gauntlet/receipts/export/bundle';
    method: 'POST';
  };
  'get__api_gauntlet_heatmaps': {
    path: '/api/gauntlet/heatmaps';
    method: 'GET';
  };
  'get__api_gauntlet_heatmaps_{heatmap_id}': {
    path: '/api/gauntlet/heatmaps/{heatmap_id}';
    method: 'GET';
  };
  'get__api_gauntlet_heatmaps_{heatmap_id}_export': {
    path: '/api/gauntlet/heatmaps/{heatmap_id}/export';
    method: 'GET';
  };
  'get__api_gauntlet_receipts_{receipt_id}_stream': {
    path: '/api/gauntlet/receipts/{receipt_id}/stream';
    method: 'GET';
  };
  'get__api_patterns': {
    path: '/api/patterns';
    method: 'GET';
  };
  'get__api_patterns_{pattern_id}': {
    path: '/api/patterns/{pattern_id}';
    method: 'GET';
  };
  'post__api_patterns_hive-mind': {
    path: '/api/patterns/hive-mind';
    method: 'POST';
  };
  'post__api_patterns_map-reduce': {
    path: '/api/patterns/map-reduce';
    method: 'POST';
  };
  'post__api_patterns_review-cycle': {
    path: '/api/patterns/review-cycle';
    method: 'POST';
  };
  'get__api_km_checkpoints': {
    path: '/api/km/checkpoints';
    method: 'GET';
  };
  'post__api_km_checkpoints': {
    path: '/api/km/checkpoints';
    method: 'POST';
  };
  'get__api_km_checkpoints_{checkpoint_name}': {
    path: '/api/km/checkpoints/{checkpoint_name}';
    method: 'GET';
  };
  'delete__api_km_checkpoints_{checkpoint_name}': {
    path: '/api/km/checkpoints/{checkpoint_name}';
    method: 'DELETE';
  };
  'post__api_km_checkpoints_{checkpoint_name}_restore': {
    path: '/api/km/checkpoints/{checkpoint_name}/restore';
    method: 'POST';
  };
  'get__api_km_checkpoints_{checkpoint_name}_compare': {
    path: '/api/km/checkpoints/{checkpoint_name}/compare';
    method: 'GET';
  };
  'post__api_km_checkpoints_compare': {
    path: '/api/km/checkpoints/compare';
    method: 'POST';
  };
  'get__api_km_checkpoints_{checkpoint_name}_download': {
    path: '/api/km/checkpoints/{checkpoint_name}/download';
    method: 'GET';
  };
  'get__api_debates_{debate_id}_explainability': {
    path: '/api/debates/{debate_id}/explainability';
    method: 'GET';
  };
  'get__api_debates_{debate_id}_explainability_factors': {
    path: '/api/debates/{debate_id}/explainability/factors';
    method: 'GET';
  };
  'get__api_debates_{debate_id}_explainability_counterfactual': {
    path: '/api/debates/{debate_id}/explainability/counterfactual';
    method: 'GET';
  };
  'post__api_debates_{debate_id}_explainability_counterfactual': {
    path: '/api/debates/{debate_id}/explainability/counterfactual';
    method: 'POST';
  };
  'get__api_debates_{debate_id}_explainability_provenance': {
    path: '/api/debates/{debate_id}/explainability/provenance';
    method: 'GET';
  };
  'get__api_debates_{debate_id}_explainability_narrative': {
    path: '/api/debates/{debate_id}/explainability/narrative';
    method: 'GET';
  };
  'get__api_workflow_templates': {
    path: '/api/workflow/templates';
    method: 'GET';
  };
  'get__api_workflow_templates_{template_id}': {
    path: '/api/workflow/templates/{template_id}';
    method: 'GET';
  };
  'get__api_workflow_templates_{template_id}_package': {
    path: '/api/workflow/templates/{template_id}/package';
    method: 'GET';
  };
  'post__api_workflow_templates_{template_id}_run': {
    path: '/api/workflow/templates/{template_id}/run';
    method: 'POST';
  };
  'get__api_workflow_categories': {
    path: '/api/workflow/categories';
    method: 'GET';
  };
  'get__api_workflow_patterns': {
    path: '/api/workflow/patterns';
    method: 'GET';
  };
  'post__api_workflow_patterns_{pattern_id}_instantiate': {
    path: '/api/workflow/patterns/{pattern_id}/instantiate';
    method: 'POST';
  };
};