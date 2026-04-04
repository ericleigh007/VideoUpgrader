import type { InterpolationMode, InterpolationTargetFps, SourceVideoSummary } from "../types";

export const interpolationModes: Array<{ value: InterpolationMode; label: string }> = [
  { value: "off", label: "Off" },
  { value: "afterUpscale", label: "Interpolate After Upscale" },
  { value: "interpolateOnly", label: "Interpolate Existing Video" }
];

export const interpolationTargetFpsOptions: InterpolationTargetFps[] = [30, 60];

export function isInterpolationEnabled(mode: InterpolationMode): boolean {
  return mode !== "off";
}

export function buildInterpolationRunLabel(mode: InterpolationMode, isRunning: boolean): string {
  if (isRunning) {
    if (mode === "interpolateOnly") {
      return "Interpolating...";
    }
    if (mode === "afterUpscale") {
      return "Upscaling + Interpolating...";
    }
    return "Upscaling...";
  }

  if (mode === "interpolateOnly") {
    return "Run Interpolation";
  }
  if (mode === "afterUpscale") {
    return "Run Upscale + Interpolation";
  }
  return "Run Upscale";
}

export function buildInterpolationWarning(
  source: SourceVideoSummary | null,
  mode: InterpolationMode,
  targetFps: InterpolationTargetFps,
): string | null {
  if (!source || mode === "off") {
    return null;
  }
  if (source.frameRate < targetFps) {
    return null;
  }
  return `The source is already ${source.frameRate.toFixed(3)} fps, so the selected interpolation target of ${targetFps} fps is not higher. Continue anyway?`;
}