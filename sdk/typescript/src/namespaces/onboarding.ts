/**
 * Onboarding Namespace API
 *
 * Provides methods for user onboarding:
 * - Onboarding flow management
 * - Template recommendations
 * - First debate assistance
 * - Quick-start configurations
 * - Analytics
 */

/**
 * Starter template recommended for onboarding
 */
export interface StarterTemplate {
  id: string;
  name: string;
  description: string;
  use_cases: string[];
  agents_count: number;
  rounds: number;
  estimated_minutes: number;
  example_prompt: string;
  tags?: string[];
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
}

/**
 * Onboarding flow state
 */
export interface OnboardingFlow {
  id: string;
  current_step: string;
  completed_steps: string[];
  use_case?: string;
  selected_template_id?: string;
  first_debate_id?: string;
  quick_start_profile?: string;
  team_invites_count?: number;
  progress_percentage: number;
  started_at?: string;
  updated_at?: string;
  completed_at?: string;
  skipped?: boolean;
}

/**
 * Quick-start profile configuration
 */
export interface QuickStartConfig {
  profile: string;
  default_template: string;
  suggested_templates: string[];
  default_agents: string[];
  default_rounds: number;
  focus_areas: string[];
}

/**
 * Onboarding funnel analytics
 */
export interface OnboardingAnalytics {
  started: number;
  first_debate: number;
  completed: number;
  completion_rate: number;
  step_completion?: Record<string, number>;
  total_events?: number;
}

/**
 * Quick-start profile types
 */
export type QuickStartProfile =
  | 'developer'
  | 'security'
  | 'executive'
  | 'product'
  | 'compliance';

/**
 * Flow step action types
 */
export type FlowAction = 'next' | 'previous' | 'complete' | 'skip';

/**
 * Interface for the internal client used by OnboardingAPI.
 */
interface OnboardingClientInterface {
  request<T>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Onboarding API namespace.
 *
 * Provides methods for user onboarding and getting started:
 * - Manage onboarding flow state
 * - Get recommended templates for new users
 * - Start guided first debates
 * - Apply quick-start configurations
 *
 * Essential for new user experience and reducing time-to-value.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Check onboarding status
 * const { needs_onboarding, flow, templates } = await client.onboarding.getFlow();
 *
 * if (needs_onboarding) {
 *   // Initialize onboarding for a developer
 *   await client.onboarding.initFlow({ quick_start_profile: 'developer' });
 *
 *   // Start first debate with a template
 *   const result = await client.onboarding.startFirstDebate({
 *     template_id: templates[0].id,
 *     use_example: true
 *   });
 * }
 *
 * // Get analytics (admin)
 * const analytics = await client.onboarding.getAnalytics();
 * console.log(`Completion rate: ${analytics.completion_rate}%`);
 * ```
 */
export class OnboardingAPI {
  constructor(private client: OnboardingClientInterface) {}

  // ===========================================================================
  // Flow Management
  // ===========================================================================

  /**
   * Get current onboarding flow state.
   */
  async getFlow(options?: {
    user_id?: string;
    organization_id?: string;
  }): Promise<{
    needs_onboarding: boolean;
    exists: boolean;
    flow?: OnboardingFlow;
    recommended_templates: StarterTemplate[];
  }> {
    return this.client.request('GET', '/api/v1/onboarding/flow', { params: options });
  }

  /**
   * Initialize a new onboarding flow.
   */
  async initFlow(options?: {
    use_case?: string;
    quick_start_profile?: QuickStartProfile;
    skip_to_step?: string;
  }): Promise<OnboardingFlow> {
    return this.client.request('POST', '/api/v1/onboarding/flow', { json: options });
  }

  /**
   * Update onboarding step progress.
   */
  async updateStep(options: {
    action: FlowAction;
    step_data?: Record<string, unknown>;
    jump_to_step?: string;
  }): Promise<{
    success: boolean;
    flow: OnboardingFlow;
    next_step?: string;
  }> {
    return this.client.request('PUT', '/api/v1/onboarding/flow/step', { json: options });
  }

  /**
   * Skip the onboarding flow.
   */
  async skip(): Promise<{ success: boolean; flow: OnboardingFlow }> {
    return this.updateStep({ action: 'skip' });
  }

  /**
   * Mark onboarding as complete.
   */
  async complete(): Promise<{ success: boolean; flow: OnboardingFlow }> {
    return this.updateStep({ action: 'complete' });
  }

  // ===========================================================================
  // Templates
  // ===========================================================================

  /**
   * Get recommended starter templates.
   */
  async getTemplates(options?: {
    use_case?: string;
    profile?: QuickStartProfile;
  }): Promise<{ templates: StarterTemplate[] }> {
    return this.client.request('GET', '/api/v1/onboarding/templates', { params: options });
  }

  // ===========================================================================
  // First Debate
  // ===========================================================================

  /**
   * Start a guided first debate.
   *
   * This creates a debate optimized for new users with guidance and
   * simplified configuration.
   */
  async startFirstDebate(options?: {
    template_id?: string;
    topic?: string;
    use_example?: boolean;
  }): Promise<{
    success: boolean;
    debate_id: string;
    message: string;
    guidance?: string[];
  }> {
    return this.client.request('POST', '/api/v1/onboarding/first-debate', { json: options });
  }

  // ===========================================================================
  // Quick-Start
  // ===========================================================================

  /**
   * Apply quick-start configuration.
   *
   * Configures the workspace with defaults appropriate for the selected profile.
   */
  async applyQuickStart(profile: QuickStartProfile): Promise<{
    success: boolean;
    applied_config: QuickStartConfig;
    next_action?: string;
  }> {
    return this.client.request('POST', '/api/v1/onboarding/quick-start', {
      json: { profile },
    });
  }

  // ===========================================================================
  // Analytics
  // ===========================================================================

  /**
   * Get onboarding funnel analytics (admin only).
   */
  async getAnalytics(options?: {
    organization_id?: string;
  }): Promise<OnboardingAnalytics> {
    return this.client.request('GET', '/api/v1/onboarding/analytics', { params: options });
  }
}
