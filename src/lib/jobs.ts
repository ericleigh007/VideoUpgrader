import type { RealesrganJobRequest } from "../types";

export const JOBS_WINDOW_LABEL = "jobs";
export const JOBS_VIEW_QUERY_KEY = "view";
export const JOBS_VIEW_QUERY_VALUE = "jobs";
export const REPEAT_PIPELINE_REQUEST_STORAGE_KEY = "videoupgrader.repeat.pipeline.request.v1";

export type AppView = "main" | "jobs";
export type RepeatPipelineRequestAction = "repeat" | "restart";

export interface RepeatPipelineRequestEnvelope {
  request: RealesrganJobRequest;
  requestedAt: number;
  action: RepeatPipelineRequestAction;
}

export function resolveAppView(search: string): AppView {
  try {
    return new URLSearchParams(search).get(JOBS_VIEW_QUERY_KEY) === JOBS_VIEW_QUERY_VALUE ? "jobs" : "main";
  } catch {
    return "main";
  }
}

export function buildJobsWindowUrl(locationLike: Pick<Location, "origin" | "pathname">): string {
  return `${locationLike.origin}${locationLike.pathname}?${JOBS_VIEW_QUERY_KEY}=${JOBS_VIEW_QUERY_VALUE}`;
}

export function buildRepeatPipelineRequestEnvelope(
  request: RealesrganJobRequest,
  requestedAt: number,
  action: RepeatPipelineRequestAction = "repeat",
): RepeatPipelineRequestEnvelope {
  return { request, requestedAt, action };
}

export function parseRepeatPipelineRequestEnvelope(raw: string | null): RepeatPipelineRequestEnvelope | null {
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<RepeatPipelineRequestEnvelope>;
    if (!parsed?.request || typeof parsed.requestedAt !== "number" || !Number.isFinite(parsed.requestedAt)) {
      return null;
    }
    return {
      request: parsed.request,
      requestedAt: parsed.requestedAt,
      action: parsed.action === "restart" ? "restart" : "repeat",
    };
  } catch {
    return null;
  }
}
