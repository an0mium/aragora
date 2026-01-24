/*
Copyright 2024 Aragora.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AragoraPolicySpec defines the desired state of AragoraPolicy
type AragoraPolicySpec struct {
	// ClusterRef references the target AragoraCluster
	// +kubebuilder:validation:Required
	ClusterRef string `json:"clusterRef"`

	// Priority determines policy precedence (higher = more priority)
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1000
	// +kubebuilder:default=100
	Priority int `json:"priority,omitempty"`

	// Enabled controls whether this policy is active
	// +kubebuilder:default=true
	Enabled bool `json:"enabled,omitempty"`

	// CostLimits defines cost control policies
	// +optional
	CostLimits *CostLimitsPolicy `json:"costLimits,omitempty"`

	// ModelRestrictions defines model usage restrictions
	// +optional
	ModelRestrictions []ModelRestriction `json:"modelRestrictions,omitempty"`

	// RateLimits defines rate limiting policies
	// +optional
	RateLimits *RateLimitsPolicy `json:"rateLimits,omitempty"`

	// ContentFilters defines content filtering policies
	// +optional
	ContentFilters []ContentFilter `json:"contentFilters,omitempty"`

	// AccessControl defines access control policies
	// +optional
	AccessControl *AccessControlPolicy `json:"accessControl,omitempty"`

	// AuditLogging configures audit logging
	// +optional
	AuditLogging *AuditLoggingPolicy `json:"auditLogging,omitempty"`

	// ResourceQuotas defines resource usage limits
	// +optional
	ResourceQuotas *ResourceQuotaPolicy `json:"resourceQuotas,omitempty"`

	// Selector selects which workspaces/tenants this policy applies to
	// +optional
	Selector *PolicySelector `json:"selector,omitempty"`
}

// CostLimitsPolicy defines cost control settings
type CostLimitsPolicy struct {
	// DailyLimitUSD is the daily cost limit in USD
	// +optional
	DailyLimitUSD *resource.Quantity `json:"dailyLimitUSD,omitempty"`

	// MonthlyLimitUSD is the monthly cost limit in USD
	// +optional
	MonthlyLimitUSD *resource.Quantity `json:"monthlyLimitUSD,omitempty"`

	// AlertThresholdPercent triggers an alert at this percentage of budget
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	// +kubebuilder:default=80
	AlertThresholdPercent int `json:"alertThresholdPercent,omitempty"`

	// HardLimit enforces a hard limit (stops processing)
	// +kubebuilder:default=false
	HardLimit bool `json:"hardLimit,omitempty"`

	// NotificationChannels for cost alerts
	// +optional
	NotificationChannels []NotificationChannel `json:"notificationChannels,omitempty"`
}

// NotificationChannel defines a notification target
type NotificationChannel struct {
	// Type is the channel type (slack, email, webhook, pagerduty)
	// +kubebuilder:validation:Enum=slack;email;webhook;pagerduty
	Type string `json:"type"`

	// Endpoint is the notification endpoint
	Endpoint string `json:"endpoint"`

	// SecretRef references a secret for authentication
	// +optional
	SecretRef string `json:"secretRef,omitempty"`
}

// ModelRestriction defines restrictions for a specific model
type ModelRestriction struct {
	// Name is the model name (e.g., "claude-opus-4", "gpt-4")
	Name string `json:"name"`

	// MaxRequestsPerHour limits requests per hour
	// +optional
	MaxRequestsPerHour *int `json:"maxRequestsPerHour,omitempty"`

	// MaxTokensPerRequest limits tokens per request
	// +optional
	MaxTokensPerRequest *int `json:"maxTokensPerRequest,omitempty"`

	// Allowed controls whether this model is allowed
	// +kubebuilder:default=true
	Allowed bool `json:"allowed,omitempty"`

	// AllowedOperations limits which operations can use this model
	// +optional
	AllowedOperations []string `json:"allowedOperations,omitempty"`

	// CostMultiplier adjusts cost accounting for this model
	// +optional
	CostMultiplier *resource.Quantity `json:"costMultiplier,omitempty"`
}

// RateLimitsPolicy defines rate limiting settings
type RateLimitsPolicy struct {
	// DebatesPerMinute limits debates per minute
	// +optional
	DebatesPerMinute *int `json:"debatesPerMinute,omitempty"`

	// RequestsPerMinute limits API requests per minute
	// +optional
	RequestsPerMinute *int `json:"requestsPerMinute,omitempty"`

	// TokensPerMinute limits total tokens per minute
	// +optional
	TokensPerMinute *int `json:"tokensPerMinute,omitempty"`

	// ConcurrentDebates limits concurrent debates
	// +optional
	ConcurrentDebates *int `json:"concurrentDebates,omitempty"`

	// BurstLimit allows temporary burst above rate limit
	// +optional
	BurstLimit *int `json:"burstLimit,omitempty"`

	// PerWorkspace enables per-workspace rate limiting
	// +kubebuilder:default=true
	PerWorkspace bool `json:"perWorkspace,omitempty"`
}

// ContentFilter defines content filtering rules
type ContentFilter struct {
	// Name is the filter name
	Name string `json:"name"`

	// Type is the filter type (blocklist, regex, ml)
	// +kubebuilder:validation:Enum=blocklist;regex;ml
	Type string `json:"type"`

	// Pattern is the filter pattern (for blocklist/regex)
	// +optional
	Pattern string `json:"pattern,omitempty"`

	// Action defines what happens on match (block, warn, log)
	// +kubebuilder:validation:Enum=block;warn;log
	// +kubebuilder:default=log
	Action string `json:"action,omitempty"`

	// Categories for ML-based filtering
	// +optional
	Categories []string `json:"categories,omitempty"`

	// Threshold for ML-based filtering (0-1)
	// +optional
	Threshold *resource.Quantity `json:"threshold,omitempty"`
}

// AccessControlPolicy defines access control settings
type AccessControlPolicy struct {
	// RequireAuthentication requires authentication for all requests
	// +kubebuilder:default=true
	RequireAuthentication bool `json:"requireAuthentication,omitempty"`

	// AllowedRoles lists roles that can access the cluster
	// +optional
	AllowedRoles []string `json:"allowedRoles,omitempty"`

	// AllowedNamespaces lists K8s namespaces that can access the cluster
	// +optional
	AllowedNamespaces []string `json:"allowedNamespaces,omitempty"`

	// AllowedServiceAccounts lists service accounts that can access
	// +optional
	AllowedServiceAccounts []string `json:"allowedServiceAccounts,omitempty"`

	// IPWhitelist lists allowed IP ranges
	// +optional
	IPWhitelist []string `json:"ipWhitelist,omitempty"`

	// MFARequired requires MFA for access
	// +kubebuilder:default=false
	MFARequired bool `json:"mfaRequired,omitempty"`
}

// AuditLoggingPolicy defines audit logging settings
type AuditLoggingPolicy struct {
	// Enabled enables audit logging
	// +kubebuilder:default=true
	Enabled bool `json:"enabled,omitempty"`

	// LogLevel sets the audit log level (none, metadata, request, requestresponse)
	// +kubebuilder:validation:Enum=none;metadata;request;requestresponse
	// +kubebuilder:default=metadata
	LogLevel string `json:"logLevel,omitempty"`

	// RetentionDays is how long to retain audit logs
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=90
	RetentionDays int `json:"retentionDays,omitempty"`

	// ExternalSink sends logs to external system
	// +optional
	ExternalSink *AuditSink `json:"externalSink,omitempty"`
}

// AuditSink defines where to send audit logs
type AuditSink struct {
	// Type is the sink type (elasticsearch, splunk, s3, webhook)
	// +kubebuilder:validation:Enum=elasticsearch;splunk;s3;webhook
	Type string `json:"type"`

	// Endpoint is the sink endpoint
	Endpoint string `json:"endpoint"`

	// SecretRef references credentials
	// +optional
	SecretRef string `json:"secretRef,omitempty"`
}

// ResourceQuotaPolicy defines resource quota settings
type ResourceQuotaPolicy struct {
	// MaxDebatesPerHour limits debates per hour
	// +optional
	MaxDebatesPerHour *int `json:"maxDebatesPerHour,omitempty"`

	// MaxConcurrentAgents limits concurrent agents
	// +optional
	MaxConcurrentAgents *int `json:"maxConcurrentAgents,omitempty"`

	// MaxStorageBytes limits storage usage
	// +optional
	MaxStorageBytes *resource.Quantity `json:"maxStorageBytes,omitempty"`

	// MaxMemoryEntries limits memory entries
	// +optional
	MaxMemoryEntries *int64 `json:"maxMemoryEntries,omitempty"`
}

// PolicySelector selects what this policy applies to
type PolicySelector struct {
	// Workspaces lists specific workspace IDs
	// +optional
	Workspaces []string `json:"workspaces,omitempty"`

	// Tenants lists specific tenant IDs
	// +optional
	Tenants []string `json:"tenants,omitempty"`

	// LabelSelector selects by labels
	// +optional
	LabelSelector *metav1.LabelSelector `json:"labelSelector,omitempty"`

	// MatchAll applies to all workspaces/tenants
	// +kubebuilder:default=false
	MatchAll bool `json:"matchAll,omitempty"`
}

// AragoraPolicyStatus defines the observed state of AragoraPolicy
type AragoraPolicyStatus struct {
	// Phase is the current phase of the policy
	// +kubebuilder:validation:Enum=Pending;Active;Error;Disabled
	Phase PolicyPhase `json:"phase,omitempty"`

	// Applied indicates if the policy is currently applied
	Applied bool `json:"applied,omitempty"`

	// LastAppliedTime is when the policy was last applied
	LastAppliedTime *metav1.Time `json:"lastAppliedTime,omitempty"`

	// AffectedWorkspaces lists workspaces affected by this policy
	AffectedWorkspaces []string `json:"affectedWorkspaces,omitempty"`

	// Violations lists current policy violations
	Violations []PolicyViolation `json:"violations,omitempty"`

	// Conditions represent the latest available observations
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ObservedGeneration is the last observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conflicts lists conflicts with other policies
	Conflicts []PolicyConflict `json:"conflicts,omitempty"`
}

// PolicyPhase represents the phase of an AragoraPolicy
// +kubebuilder:validation:Enum=Pending;Active;Error;Disabled
type PolicyPhase string

const (
	PolicyPhasePending  PolicyPhase = "Pending"
	PolicyPhaseActive   PolicyPhase = "Active"
	PolicyPhaseError    PolicyPhase = "Error"
	PolicyPhaseDisabled PolicyPhase = "Disabled"
)

// PolicyViolation represents a policy violation
type PolicyViolation struct {
	// Type is the violation type
	Type string `json:"type"`

	// WorkspaceID is the affected workspace
	WorkspaceID string `json:"workspaceId"`

	// Message describes the violation
	Message string `json:"message"`

	// Timestamp is when the violation occurred
	Timestamp metav1.Time `json:"timestamp"`

	// Severity is the violation severity
	// +kubebuilder:validation:Enum=low;medium;high;critical
	Severity string `json:"severity"`
}

// PolicyConflict represents a conflict with another policy
type PolicyConflict struct {
	// PolicyName is the conflicting policy name
	PolicyName string `json:"policyName"`

	// ConflictType describes the conflict type
	ConflictType string `json:"conflictType"`

	// Description describes the conflict
	Description string `json:"description"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=ap
// +kubebuilder:printcolumn:name="Cluster",type="string",JSONPath=".spec.clusterRef"
// +kubebuilder:printcolumn:name="Priority",type="integer",JSONPath=".spec.priority"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Applied",type="boolean",JSONPath=".status.applied"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// AragoraPolicy is the Schema for the aragorapolicies API
type AragoraPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AragoraPolicySpec   `json:"spec,omitempty"`
	Status AragoraPolicyStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// AragoraPolicyList contains a list of AragoraPolicy
type AragoraPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AragoraPolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&AragoraPolicy{}, &AragoraPolicyList{})
}
