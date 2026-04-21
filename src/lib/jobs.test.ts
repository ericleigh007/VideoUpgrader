import { describe, expect, it } from "vitest";
import {
  buildComparisonWindowUrl,
  buildJobsWindowUrl,
  buildRepeatPipelineRequestEnvelope,
  parseRepeatPipelineRequestEnvelope,
  resolveAppView,
} from "./jobs";
import type { RealesrganJobRequest } from "../types";

const baseRequest: RealesrganJobRequest = {
  sourcePath: "C:/fixtures/input.mov",
  modelId: "realesrgan-x4plus",
  colorizationMode: "off",
  colorizerModelId: null,
  colorizationContext: null,
  deepremasterProcessingMode: "standard",
  outputMode: "cropTo4k",
  qualityPreset: "qualityMax",
  interpolationMode: "off",
  interpolationTargetFps: null,
  pytorchRunner: "torch",
  gpuId: 1,
  aspectRatioPreset: "16:9",
  customAspectWidth: null,
  customAspectHeight: null,
  resolutionBasis: "exact",
  targetWidth: 3840,
  targetHeight: 2160,
  cropLeft: 0,
  cropTop: 0,
  cropWidth: 1,
  cropHeight: 1,
  previewMode: false,
  previewDurationSeconds: null,
  segmentDurationSeconds: 10,
  outputPath: "C:/exports/output.mkv",
  codec: "h265",
  container: "mkv",
  tileSize: 128,
  fp16: false,
  crf: 16,
};

describe("jobs helpers", () => {
  it("resolves jobs-only view from the query string", () => {
    expect(resolveAppView("?view=jobs")).toBe("jobs");
    expect(resolveAppView("?view=comparison")).toBe("comparison");
    expect(resolveAppView("?foo=bar")).toBe("main");
    expect(resolveAppView("")).toBe("main");
  });

  it("builds the jobs window URL from the current app location", () => {
    expect(buildJobsWindowUrl({ origin: "http://localhost:1420", pathname: "/" } as Pick<Location, "origin" | "pathname">)).toBe("http://localhost:1420/?view=jobs");
    expect(buildJobsWindowUrl({ origin: "tauri://localhost", pathname: "/index.html" } as Pick<Location, "origin" | "pathname">)).toBe("tauri://localhost/index.html?view=jobs");
  });

  it("builds the comparison window URL from the current app location", () => {
    expect(buildComparisonWindowUrl({ origin: "http://localhost:1420", pathname: "/" } as Pick<Location, "origin" | "pathname">)).toBe("http://localhost:1420/?view=comparison");
    expect(buildComparisonWindowUrl({ origin: "tauri://localhost", pathname: "/index.html" } as Pick<Location, "origin" | "pathname">)).toBe("tauri://localhost/index.html?view=comparison");
  });

  it("round-trips repeat-run envelopes", () => {
    const envelope = buildRepeatPipelineRequestEnvelope(baseRequest, 123456789);

    expect(parseRepeatPipelineRequestEnvelope(JSON.stringify(envelope))).toEqual(envelope);
  });

  it("round-trips restart envelopes", () => {
    const envelope = buildRepeatPipelineRequestEnvelope(baseRequest, 123456789, "restart");

    expect(parseRepeatPipelineRequestEnvelope(JSON.stringify(envelope))).toEqual(envelope);
  });

  it("defaults legacy envelopes to repeat actions", () => {
    expect(parseRepeatPipelineRequestEnvelope(JSON.stringify({ request: baseRequest, requestedAt: 123456789 }))).toEqual({
      request: baseRequest,
      requestedAt: 123456789,
      action: "repeat",
    });
  });

  it("rejects invalid repeat-run envelopes", () => {
    expect(parseRepeatPipelineRequestEnvelope(null)).toBeNull();
    expect(parseRepeatPipelineRequestEnvelope("{bad json")).toBeNull();
    expect(parseRepeatPipelineRequestEnvelope(JSON.stringify({ requestedAt: "soon" }))).toBeNull();
  });
});
