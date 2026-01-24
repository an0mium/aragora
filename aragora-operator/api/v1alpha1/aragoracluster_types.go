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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AragoraClusterSpec defines the desired state of AragoraCluster
type AragoraClusterSpec struct {
	// Version specifies the Aragora version to deploy
	// +kubebuilder:validation:Pattern=`^v?\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$`
	Version string `json:"version"`

	// Replicas is the number of Aragora control plane replicas
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	Replicas int32 `json:"replicas,omitempty"`

	// Resources defines the compute resources for Aragora pods
	Resources ResourceRequirements `json:"resources,omitempty"`

	// Storage defines the storage configuration
	Storage StorageConfig `json:"storage,omitempty"`

	// Agents configures which AI agents are enabled
	Agents AgentsConfig `json:"agents,omitempty"`

	// Memory configures the memory tier system
	Memory MemoryConfig `json:"memory,omitempty"`

	// HighAvailability configures HA settings
	// +optional
	HighAvailability *HighAvailabilityConfig `json:"highAvailability,omitempty"`

	// Image overrides the default Aragora container image
	// +optional
	Image string `json:"image,omitempty"`

	// ImagePullSecrets for pulling the Aragora image
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// ServiceAccountName is the name of the ServiceAccount to use
	// +optional
	ServiceAccountName string `json:"serviceAccountName,omitempty"`

	// NodeSelector for scheduling pods to specific nodes
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Tolerations for pod scheduling
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// Affinity for pod scheduling
	// +optional
	Affinity *corev1.Affinity `json:"affinity,omitempty"`

	// Environment variables for the Aragora pods
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// SecretRef references a secret containing API keys
	// +optional
	SecretRef *corev1.SecretReference `json:"secretRef,omitempty"`

	// Monitoring configures observability settings
	// +optional
	Monitoring *MonitoringConfig `json:"monitoring,omitempty"`
}

// ResourceRequirements defines compute resources
type ResourceRequirements struct {
	// Memory is the memory limit
	// +kubebuilder:default="2Gi"
	Memory resource.Quantity `json:"memory,omitempty"`

	// CPU is the CPU limit
	// +kubebuilder:default="1"
	CPU resource.Quantity `json:"cpu,omitempty"`

	// RequestsMemory is the memory request
	// +optional
	RequestsMemory *resource.Quantity `json:"requestsMemory,omitempty"`

	// RequestsCPU is the CPU request
	// +optional
	RequestsCPU *resource.Quantity `json:"requestsCPU,omitempty"`
}

// StorageConfig defines storage settings
type StorageConfig struct {
	// Size is the storage size
	// +kubebuilder:default="10Gi"
	Size resource.Quantity `json:"size,omitempty"`

	// StorageClassName is the storage class to use
	// +optional
	StorageClassName string `json:"storageClassName,omitempty"`

	// VolumeMode is the volume mode (Filesystem or Block)
	// +optional
	VolumeMode *corev1.PersistentVolumeMode `json:"volumeMode,omitempty"`
}

// AgentsConfig configures AI agent availability
type AgentsConfig struct {
	// Enabled lists the enabled agent providers
	// +kubebuilder:default={"anthropic","openai"}
	Enabled []string `json:"enabled,omitempty"`

	// Fallback specifies the fallback provider (e.g., "openrouter")
	// +optional
	Fallback string `json:"fallback,omitempty"`

	// DefaultModel specifies the default model for debates
	// +optional
	DefaultModel string `json:"defaultModel,omitempty"`

	// RateLimits defines per-provider rate limits
	// +optional
	RateLimits map[string]RateLimitConfig `json:"rateLimits,omitempty"`
}

// RateLimitConfig defines rate limiting for an agent
type RateLimitConfig struct {
	// RequestsPerMinute is the max requests per minute
	RequestsPerMinute int `json:"requestsPerMinute,omitempty"`

	// TokensPerMinute is the max tokens per minute
	TokensPerMinute int `json:"tokensPerMinute,omitempty"`
}

// MemoryConfig configures the memory tier system
type MemoryConfig struct {
	// Tiers defines memory tier configuration
	Tiers map[string]MemoryTierConfig `json:"tiers,omitempty"`

	// Backend specifies the memory backend (redis, postgres, memory)
	// +kubebuilder:default="memory"
	Backend string `json:"backend,omitempty"`

	// RedisURL for Redis backend
	// +optional
	RedisURL string `json:"redisURL,omitempty"`

	// PostgresURL for Postgres backend
	// +optional
	PostgresURL string `json:"postgresURL,omitempty"`
}

// MemoryTierConfig configures a single memory tier
type MemoryTierConfig struct {
	// TTL is the time-to-live for entries in this tier
	TTL metav1.Duration `json:"ttl,omitempty"`

	// MaxSize is the maximum size of this tier
	MaxSize resource.Quantity `json:"maxSize,omitempty"`
}

// HighAvailabilityConfig configures HA settings
type HighAvailabilityConfig struct {
	// Enabled enables high availability mode
	// +kubebuilder:default=false
	Enabled bool `json:"enabled,omitempty"`

	// MinReplicas is the minimum number of replicas for HA
	// +kubebuilder:validation:Minimum=2
	// +kubebuilder:default=2
	MinReplicas int32 `json:"minReplicas,omitempty"`

	// PodDisruptionBudget enables PDB creation
	// +kubebuilder:default=true
	PodDisruptionBudget bool `json:"podDisruptionBudget,omitempty"`

	// TopologySpreadConstraints for HA distribution
	// +optional
	TopologySpreadConstraints []corev1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
}

// MonitoringConfig configures observability
type MonitoringConfig struct {
	// Enabled enables Prometheus metrics
	// +kubebuilder:default=true
	Enabled bool `json:"enabled,omitempty"`

	// ServiceMonitor creates a ServiceMonitor for Prometheus Operator
	// +kubebuilder:default=false
	ServiceMonitor bool `json:"serviceMonitor,omitempty"`

	// PrometheusRules creates PrometheusRule for alerting
	// +kubebuilder:default=false
	PrometheusRules bool `json:"prometheusRules,omitempty"`

	// TracingEnabled enables OpenTelemetry tracing
	// +kubebuilder:default=false
	TracingEnabled bool `json:"tracingEnabled,omitempty"`

	// TracingEndpoint is the OTLP endpoint for tracing
	// +optional
	TracingEndpoint string `json:"tracingEndpoint,omitempty"`
}

// AragoraClusterStatus defines the observed state of AragoraCluster
type AragoraClusterStatus struct {
	// Phase is the current phase of the cluster
	// +kubebuilder:validation:Enum=Pending;Provisioning;Running;Failed;Terminating
	Phase ClusterPhase `json:"phase,omitempty"`

	// ReadyReplicas is the number of ready replicas
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// CurrentVersion is the currently deployed version
	CurrentVersion string `json:"currentVersion,omitempty"`

	// Endpoint is the cluster API endpoint
	Endpoint string `json:"endpoint,omitempty"`

	// Conditions represent the latest available observations
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// LastReconcileTime is when the cluster was last reconciled
	LastReconcileTime *metav1.Time `json:"lastReconcileTime,omitempty"`

	// ObservedGeneration is the last observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// AgentStatus shows the status of each enabled agent
	AgentStatus map[string]AgentStatus `json:"agentStatus,omitempty"`

	// MemoryStatus shows the status of the memory system
	MemoryStatus *MemoryStatus `json:"memoryStatus,omitempty"`
}

// ClusterPhase represents the phase of an AragoraCluster
// +kubebuilder:validation:Enum=Pending;Provisioning;Running;Failed;Terminating
type ClusterPhase string

const (
	ClusterPhasePending      ClusterPhase = "Pending"
	ClusterPhaseProvisioning ClusterPhase = "Provisioning"
	ClusterPhaseRunning      ClusterPhase = "Running"
	ClusterPhaseFailed       ClusterPhase = "Failed"
	ClusterPhaseTerminating  ClusterPhase = "Terminating"
)

// AgentStatus represents the status of an agent
type AgentStatus struct {
	// Available indicates if the agent is available
	Available bool `json:"available"`

	// LastHeartbeat is the last heartbeat time
	LastHeartbeat *metav1.Time `json:"lastHeartbeat,omitempty"`

	// RequestsPerMinute is the current request rate
	RequestsPerMinute float64 `json:"requestsPerMinute,omitempty"`

	// ErrorRate is the current error rate
	ErrorRate float64 `json:"errorRate,omitempty"`
}

// MemoryStatus represents the status of the memory system
type MemoryStatus struct {
	// Connected indicates if the memory backend is connected
	Connected bool `json:"connected"`

	// UsedBytes is the total used memory
	UsedBytes int64 `json:"usedBytes,omitempty"`

	// TotalEntries is the total number of entries
	TotalEntries int64 `json:"totalEntries,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=ac
// +kubebuilder:printcolumn:name="Version",type="string",JSONPath=".spec.version"
// +kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".spec.replicas"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// AragoraCluster is the Schema for the aragoraclusters API
type AragoraCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AragoraClusterSpec   `json:"spec,omitempty"`
	Status AragoraClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// AragoraClusterList contains a list of AragoraCluster
type AragoraClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AragoraCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&AragoraCluster{}, &AragoraClusterList{})
}
