{{/*
Expand the name of the chart.
*/}}
{{- define "aragora.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec). If release name contains chart name it will be used
as a full name.
*/}}
{{- define "aragora.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "aragora.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to all resources.
*/}}
{{- define "aragora.labels" -}}
helm.sh/chart: {{ include "aragora.chart" . }}
{{ include "aragora.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: aragora
{{- end }}

{{/*
Selector labels (shared across all components for service discovery).
*/}}
{{- define "aragora.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aragora.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Backend selector labels.
*/}}
{{- define "aragora.backend.selectorLabels" -}}
{{ include "aragora.selectorLabels" . }}
app.kubernetes.io/component: backend
{{- end }}

{{/*
Frontend selector labels.
*/}}
{{- define "aragora.frontend.selectorLabels" -}}
{{ include "aragora.selectorLabels" . }}
app.kubernetes.io/component: frontend
{{- end }}

{{/*
Create the name of the service account to use.
*/}}
{{- define "aragora.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "aragora.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the name of the secret to use for API keys and credentials.
*/}}
{{- define "aragora.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- include "aragora.fullname" . }}
{{- end }}
{{- end }}

{{/*
Return the PostgreSQL DSN. Prefers explicit secret, then subchart, then external.
*/}}
{{- define "aragora.databaseUrl" -}}
{{- if .Values.secrets.database.postgresUrl }}
{{- .Values.secrets.database.postgresUrl }}
{{- else if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s-postgresql:5432/%s" .Values.postgresql.auth.username (.Values.postgresql.auth.password | default "aragora") (include "aragora.fullname" .) .Values.postgresql.auth.database }}
{{- else if .Values.externalDatabase.host }}
{{- printf "postgresql://%s:%d/%s" .Values.externalDatabase.host (int .Values.externalDatabase.port) .Values.externalDatabase.database }}
{{- end }}
{{- end }}

{{/*
Return the Redis URL. Prefers explicit secret, then subchart, then external.
*/}}
{{- define "aragora.redisUrl" -}}
{{- if .Values.secrets.database.redisUrl }}
{{- .Values.secrets.database.redisUrl }}
{{- else if .Values.redis.enabled }}
{{- printf "redis://%s-redis-master:6379" (include "aragora.fullname" .) }}
{{- else if .Values.externalRedis.host }}
{{- printf "redis://%s:%d" .Values.externalRedis.host (int .Values.externalRedis.port) }}
{{- end }}
{{- end }}

{{/*
Backend image reference.
*/}}
{{- define "aragora.backend.image" -}}
{{- printf "%s:%s" .Values.backend.image.repository (.Values.backend.image.tag | default .Chart.AppVersion) }}
{{- end }}

{{/*
Frontend image reference.
*/}}
{{- define "aragora.frontend.image" -}}
{{- printf "%s:%s" .Values.frontend.image.repository (.Values.frontend.image.tag | default .Chart.AppVersion) }}
{{- end }}
