import { describe, expect, it } from "vitest";
import { defaultCropRect, planOutputFraming } from "./framing";

describe("planOutputFraming", () => {
  it("preserves aspect ratio inside 4k canvas", () => {
    const plan = planOutputFraming({ width: 1280, height: 720 }, "preserveAspect4k", {
      aspectRatioPreset: "16:9",
      customAspectWidth: null,
      customAspectHeight: null,
      resolutionBasis: "exact",
      targetWidth: 3840,
      targetHeight: 2160,
      cropLeft: null,
      cropTop: null,
      cropWidth: null,
      cropHeight: null
    });
    expect(plan.canvas).toEqual({ width: 3840, height: 2160 });
    expect(plan.scaled).toEqual({ width: 3840, height: 2160 });
  });

  it("fills 4k canvas when cropping is requested", () => {
    const plan = planOutputFraming({ width: 1440, height: 1080 }, "cropTo4k", {
      aspectRatioPreset: "16:9",
      customAspectWidth: null,
      customAspectHeight: null,
      resolutionBasis: "exact",
      targetWidth: 3840,
      targetHeight: 2160,
      cropLeft: 0.25,
      cropTop: 0,
      cropWidth: 0.75,
      cropHeight: 1,
    });
    expect(plan.canvas).toEqual({ width: 3840, height: 2160 });
    expect(plan.scaled.width).toBeGreaterThanOrEqual(3840);
    expect(plan.scaled.height).toBeGreaterThanOrEqual(2160);
    expect(plan.cropWindow.offsetX).toBeGreaterThanOrEqual(0);
  });

  it("returns native x4 output without 4k coercion", () => {
    const plan = planOutputFraming({ width: 640, height: 360 }, "native4x", {
      aspectRatioPreset: "source",
      customAspectWidth: null,
      customAspectHeight: null,
      resolutionBasis: "exact",
      targetWidth: null,
      targetHeight: null,
      cropLeft: null,
      cropTop: null,
      cropWidth: null,
      cropHeight: null
    });
    expect(plan.canvas).toEqual({ width: 2560, height: 1440 });
  });

  it("derives target height from width when requested", () => {
    const plan = planOutputFraming({ width: 1280, height: 720 }, "preserveAspect4k", {
      aspectRatioPreset: "1:1",
      customAspectWidth: null,
      customAspectHeight: null,
      resolutionBasis: "width",
      targetWidth: 2048,
      targetHeight: null,
      cropLeft: null,
      cropTop: null,
      cropWidth: null,
      cropHeight: null
    });
    expect(plan.canvas).toEqual({ width: 2048, height: 2048 });
  });

  it("defaults crop rect to centered aspect-matched selection", () => {
    const cropRect = defaultCropRect({ width: 1280, height: 720 }, {
      aspectRatioPreset: "1:1",
      customAspectWidth: null,
      customAspectHeight: null,
      resolutionBasis: "exact",
      targetWidth: 2048,
      targetHeight: 2048,
      cropLeft: null,
      cropTop: null,
      cropWidth: null,
      cropHeight: null
    });

    expect(cropRect.width).toBeCloseTo(0.5625, 4);
    expect(cropRect.left).toBeCloseTo(0.21875, 4);
  });
});
