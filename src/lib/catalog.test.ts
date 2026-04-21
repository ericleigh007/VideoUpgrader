import { describe, expect, it } from "vitest";
import { getBackendDefinition, getBlindComparisonColorizerModels, getBlindComparisonModels, getModelDefinition, getTopRatedModels, getVisibleColorizerModels, getVisibleModels, modelCatalog } from "./catalog";

describe("modelCatalog", () => {
  it("surfaces the runnable serious models in the current UI", () => {
    const visibleModels = getVisibleModels();

    expect(visibleModels.map((model) => model.value)).toEqual([
      "realesrgan-x4plus",
      "realesrnet-x4plus",
      "bsrgan-x4",
      "swinir-realworld-x4",
      "rvrt-x4"
    ]);
  });

  it("retains compatibility entries for lower-level workers", () => {
    const hiddenValues = modelCatalog.filter((model) => model.supportTier === "compatibility").map((model) => model.value);

    expect(hiddenValues).toContain("realesrgan-x4plus-anime");
    expect(hiddenValues).toContain("realesr-animevideov3-x4");
  });

  it("resolves model metadata by id", () => {
    const model = getModelDefinition("realesrgan-x4plus");

    expect(model.label).toBe("Real-ESRGAN x4 Plus");
    expect(model.summary).toContain("photographic");
    expect(model.loader).toBe("ncnn-portable-binary");
    expect(model.nativeScale).toBe(4);
  });

  it("resolves backend metadata for a model", () => {
    const model = getModelDefinition("realesrgan-x4plus");
    const backend = getBackendDefinition(model.backendId);

    expect(backend.id).toBe("realesrgan-ncnn");
    expect(backend.loader).toBe("ncnn-portable-binary");
    expect(backend.supportsGpuSelection).toBe(true);
  });

  it("captures special handling flags needed by backend-specific pipelines", () => {
    const model = getModelDefinition("realesr-animevideov3-x4");

    expect(model.specialHandling.cropMath).toBe("scale-aware");
    expect(model.specialHandling.variantSuffixByScale).toBe(true);
  });

  it("limits blind comparison to the runnable serious subset", () => {
    const blindComparisonModels = getBlindComparisonModels();

    expect(blindComparisonModels).toHaveLength(4);
    expect(blindComparisonModels.map((model) => model.value)).toEqual([
      "realesrgan-x4plus",
      "realesrnet-x4plus",
      "bsrgan-x4",
      "swinir-realworld-x4"
    ]);
  });

  it("surfaces runnable colorizers for UI and blind comparison", () => {
    expect(getVisibleColorizerModels().map((model) => model.value)).toEqual([
      "ddcolor-modelscope",
      "ddcolor-paper",
      "deoldify-stable",
      "deoldify-video",
      "deepremaster",
      "colormnet"
    ]);

    expect(getBlindComparisonColorizerModels().map((model) => model.value)).toEqual([
      "ddcolor-modelscope",
      "ddcolor-paper",
      "deoldify-stable",
      "deoldify-video",
      "deepremaster",
      "colormnet"
    ]);
  });

  it("surfaces the six ranked target models in quality order", () => {
    const topRatedModels = getTopRatedModels();

    expect(topRatedModels).toHaveLength(6);
    expect(topRatedModels.map((model) => model.value)).toEqual([
      "realesrgan-x4plus",
      "realesrnet-x4plus",
      "bsrgan-x4",
      "hat-realhat-gan-x4",
      "swinir-realworld-x4",
      "rvrt-x4"
    ]);
  });
});