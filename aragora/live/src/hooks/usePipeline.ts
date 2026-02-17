'use client';

import { useCallback } from 'react';
import { useApi } from './useApi';
import type { PipelineResultResponse, PipelineStageType } from '@/components/pipeline-canvas/types';

interface PipelineCreateResponse {
  pipeline_id: string;
  stage_status: Record<PipelineStageType, string>;
  result: PipelineResultResponse;
  [key: string]: unknown;
}

interface PipelineAdvanceResponse {
  pipeline_id: string;
  advanced_to: string;
  stage_status: Record<PipelineStageType, string>;
  result: PipelineResultResponse;
}

export function usePipeline() {
  const api = useApi<PipelineCreateResponse>();
  const advanceApi = useApi<PipelineAdvanceResponse>();
  const getApi = useApi<PipelineResultResponse>();
  const stageApi = useApi<{ stage: string; data: unknown }>();

  const createFromDebate = useCallback(
    (cartographerData: Record<string, unknown>, autoAdvance = true) =>
      api.post('/api/v1/canvas/pipeline/from-debate', {
        cartographer_data: cartographerData,
        auto_advance: autoAdvance,
      }),
    [api]
  );

  const createFromIdeas = useCallback(
    (ideas: string[], autoAdvance = true) =>
      api.post('/api/v1/canvas/pipeline/from-ideas', {
        ideas,
        auto_advance: autoAdvance,
      }),
    [api]
  );

  const advanceStage = useCallback(
    (pipelineId: string, targetStage: PipelineStageType) =>
      advanceApi.post('/api/v1/canvas/pipeline/advance', {
        pipeline_id: pipelineId,
        target_stage: targetStage,
      }),
    [advanceApi]
  );

  const getPipeline = useCallback(
    (pipelineId: string) => getApi.get(`/api/v1/canvas/pipeline/${pipelineId}`),
    [getApi]
  );

  const getStage = useCallback(
    (pipelineId: string, stage: PipelineStageType) =>
      stageApi.get(`/api/v1/canvas/pipeline/${pipelineId}/stage/${stage}`),
    [stageApi]
  );

  return {
    createFromDebate,
    createFromIdeas,
    advanceStage,
    getPipeline,
    getStage,
    loading: api.loading || advanceApi.loading || getApi.loading,
    error: api.error || advanceApi.error || getApi.error,
  };
}
