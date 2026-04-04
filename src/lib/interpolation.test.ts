import { describe, expect, it } from "vitest";

import { buildInterpolationRunLabel, buildInterpolationWarning, isInterpolationEnabled } from "./interpolation";

describe("interpolation helpers", () => {
  it("builds the correct run button labels", () => {
    expect(buildInterpolationRunLabel("off", false)).toBe("Run Upscale");
    expect(buildInterpolationRunLabel("afterUpscale", false)).toBe("Run Upscale + Interpolation");
    expect(buildInterpolationRunLabel("interpolateOnly", true)).toBe("Interpolating...");
  });

  it("detects whether interpolation is enabled", () => {
    expect(isInterpolationEnabled("off")).toBe(false);
    expect(isInterpolationEnabled("afterUpscale")).toBe(true);
    expect(isInterpolationEnabled("interpolateOnly")).toBe(true);
  });

  it("emits a warning only when the target fps is not higher than the source", () => {
    const source = {
      path: "C:/fixtures/sample.mp4",
      previewPath: "C:/fixtures/sample.mp4",
      width: 1280,
      height: 720,
      durationSeconds: 5,
      frameRate: 60,
      hasAudio: true,
      container: "mp4"
    };

    expect(buildInterpolationWarning(source, "afterUpscale", 60)).toContain("not higher");
    expect(buildInterpolationWarning(source, "off", 60)).toBeNull();
    expect(buildInterpolationWarning({ ...source, frameRate: 24 }, "interpolateOnly", 60)).toBeNull();
  });
});