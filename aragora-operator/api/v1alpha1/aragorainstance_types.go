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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AragoraInstanceSpec defines the desired state of AragoraInstance
type AragoraInstanceSpec struct {
	// ClusterRef references the parent AragoraCluster
	// +kubebuilder:validation:Required
	ClusterRef string `json:"clusterRef"`

	// Role specifies the instance role (worker, coordinator, gateway)
	// +kubebuilder:validation:Enum=worker;coordinator;gateway
	// +kubebuilder:default=worker
	Role InstanceRole `json:"role,omitempty"`

	// Scaling configures auto-scaling behavior
	// +optional
	Scaling *ScalingConfig `json:"scaling,omitempty"`

	// Resources overrides the cluster-level resources
	// +optional
	Resources *ResourceRequirements `json:"resources,omitempty"`

	// Capabilities defines what this instance can handle
	// +optional
	Capabilities []string `json:"capabilities,omitempty"`

	// Priority for task scheduling (higher = more priority)
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	// +kubebuilder:default=50
	Priority int `json:"priority,omitempty"`

	// MaxConcurrentDebates limits concurrent debates on this instance
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=10
	MaxConcurrentDebates int `json:"maxConcurrentDebates,omitempty"`

	// Labels for routing and filtering
	// +optional
	Labels map[string]string `json:"labels,omitempty"`

	// NodeSelector for scheduling
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Tolerations for scheduling
	// +optional
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`
}

// InstanceRole defines the role of an Aragora instance
// +kubebuilder:validation:Enum=worker;coordinator;gateway
type InstanceRole string

const (
	InstanceRoleWorker      InstanceRole = "worker"
	InstanceRoleCoordinator InstanceRole = "coordinator"
	InstanceRoleGateway     InstanceRole = "gateway"
)

// ScalingConfig defines auto-scaling settings
type ScalingConfig struct {
	// Enabled enables auto-scaling
	// +kubebuilder:default=false
	Enabled bool `json:"enabled,omitempty"`

	// Metric is the scaling metric
	// +kubebuilder:validation:Enum=debates_per_second;cpu;memory;queue_depth
	// +kubebuilder:default=debates_per_second
	Metric ScalingMetric `json:"metric,omitempty"`

	// Target is the target value for the metric
	// +kubebuilder:validation:Minimum=1
	Target int `json:"target,omitempty"`

	// MinReplicas is the minimum replica count
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=1
	MinReplicas int32 `json:"minReplicas,omitempty"`

	// MaxReplicas is the maximum replica count
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=10
	MaxReplicas int32 `json:"maxReplicas,omitempty"`

	// ScaleDownStabilizationWindowSeconds is the stabilization window
	// +kubebuilder:default=300
	ScaleDownStabilizationWindowSeconds int32 `json:"scaleDownStabilizationWindowSeconds,omitempty"`

	// Behavior customizes scaling behavior
	// +optional
	Behavior *ScalingBehavior `json:"behavior,omitempty"`
}

// ScalingMetric defines what metric to scale on
// +kubebuilder:validation:Enum=debates_per_second;cpu;memory;queue_depth
type ScalingMetric string

const (
	ScalingMetricDebatesPerSecond ScalingMetric = "debates_per_second"
	ScalingMetricCPU              ScalingMetric = "cpu"
	ScalingMetricMemory           ScalingMetric = "memory"
	ScalingMetricQueueDepth       ScalingMetric = "queue_depth"
)

// ScalingBehavior customizes scaling behavior
type ScalingBehavior struct {
	// ScaleUp defines scale-up policies
	// +optional
	ScaleUp *ScalingPolicy `json:"scaleUp,omitempty"`

	// ScaleDown defines scale-down policies
	// +optional
	ScaleDown *ScalingPolicy `json:"scaleDown,omitempty"`
}

// ScalingPolicy defines a scaling policy
type ScalingPolicy struct {
	// StabilizationWindowSeconds is the stabilization window
	StabilizationWindowSeconds int32 `json:"stabilizationWindowSeconds,omitempty"`

	// Policies are the scaling policies
	Policies []ScalingPolicyRule `json:"policies,omitempty"`
}

// ScalingPolicyRule defines a single scaling policy rule
type ScalingPolicyRule struct {
	// Type is the type of scaling (Pods or Percent)
	// +kubebuilder:validation:Enum=Pods;Percent
	Type string `json:"type"`

	// Value is the value for this policy
	Value int32 `json:"value"`

	// PeriodSeconds is the period for this policy
	PeriodSeconds int32 `json:"periodSeconds"`
}

// AragoraInstanceStatus defines the observed state of AragoraInstance
type AragoraInstanceStatus struct {
	// Phase is the current phase of the instance
	// +kubebuilder:validation:Enum=Pending;Starting;Running;Draining;Terminated;Failed
	Phase InstancePhase `json:"phase,omitempty"`

	// Ready indicates if the instance is ready
	Ready bool `json:"ready,omitempty"`

	// CurrentReplicas is the current number of replicas
	CurrentReplicas int32 `json:"currentReplicas,omitempty"`

	// DesiredReplicas is the desired number of replicas
	DesiredReplicas int32 `json:"desiredReplicas,omitempty"`

	// ActiveDebates is the number of active debates
	ActiveDebates int `json:"activeDebates,omitempty"`

	// TotalDebatesProcessed is the total debates processed
	TotalDebatesProcessed int64 `json:"totalDebatesProcessed,omitempty"`

	// AverageLatencyMs is the average debate latency in milliseconds
	AverageLatencyMs float64 `json:"averageLatencyMs,omitempty"`

	// LastScaleTime is when the instance was last scaled
	LastScaleTime *metav1.Time `json:"lastScaleTime,omitempty"`

	// Conditions represent the latest available observations
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ObservedGeneration is the last observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// PodStatuses shows the status of individual pods
	PodStatuses []PodStatus `json:"podStatuses,omitempty"`
}

// InstancePhase represents the phase of an AragoraInstance
// +kubebuilder:validation:Enum=Pending;Starting;Running;Draining;Terminated;Failed
type InstancePhase string

const (
	InstancePhasePending    InstancePhase = "Pending"
	InstancePhaseStarting   InstancePhase = "Starting"
	InstancePhaseRunning    InstancePhase = "Running"
	InstancePhaseDraining   InstancePhase = "Draining"
	InstancePhaseTerminated InstancePhase = "Terminated"
	InstancePhaseFailed     InstancePhase = "Failed"
)

// PodStatus represents the status of a single pod
type PodStatus struct {
	// Name is the pod name
	Name string `json:"name"`

	// Ready indicates if the pod is ready
	Ready bool `json:"ready"`

	// IP is the pod IP
	IP string `json:"ip,omitempty"`

	// Node is the node the pod is scheduled on
	Node string `json:"node,omitempty"`

	// ActiveDebates is the number of active debates on this pod
	ActiveDebates int `json:"activeDebates,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.scaling.minReplicas,statuspath=.status.currentReplicas,selectorpath=.status.selector
// +kubebuilder:resource:shortName=ai
// +kubebuilder:printcolumn:name="Cluster",type="string",JSONPath=".spec.clusterRef"
// +kubebuilder:printcolumn:name="Role",type="string",JSONPath=".spec.role"
// +kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".status.currentReplicas"
// +kubebuilder:printcolumn:name="Ready",type="boolean",JSONPath=".status.ready"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// AragoraInstance is the Schema for the aragorainstances API
type AragoraInstance struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AragoraInstanceSpec   `json:"spec,omitempty"`
	Status AragoraInstanceStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// AragoraInstanceList contains a list of AragoraInstance
type AragoraInstanceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AragoraInstance `json:"items"`
}

func init() {
	SchemeBuilder.Register(&AragoraInstance{}, &AragoraInstanceList{})
}
