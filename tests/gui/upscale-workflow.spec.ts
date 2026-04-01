import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    let activeJob = null;
    let lastRequest = null;
    const openedPaths = [];
    const appConfig = {
      modelRatings: {},
      blindComparisons: []
    };
    window.__UPSCALER_MOCK__ = {
      async selectVideoFile() {
        return "C:/fixtures/sample-input.mp4";
      },
      async selectOutputFile(_defaultPath, container) {
        return `C:/exports/upscaled-result.${container}`;
      },
      async ensureRuntimeAssets() {
        return {
          ffmpegPath: "C:/tools/ffmpeg.exe",
          realesrganPath: "C:/tools/realesrgan-ncnn-vulkan.exe",
          modelDir: "C:/tools/models",
          availableGpus: [
            { id: 0, name: "Intel(R) Graphics", kind: "integrated" },
            { id: 1, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", kind: "discrete" }
          ],
          defaultGpuId: 1
        };
      },
      async probeSourceVideo(sourcePath) {
        return {
          path: sourcePath,
          width: 1280,
          height: 720,
          durationSeconds: 12.5,
          frameRate: 24,
          hasAudio: true,
          container: "mp4"
        };
      },
      async getAppConfig() {
        return appConfig;
      },
      async saveModelRating(modelId, rating) {
        if (rating === null) {
          delete appConfig.modelRatings[modelId];
        } else {
          appConfig.modelRatings[modelId] = {
            rating,
            updatedAt: "1710000000"
          };
        }
        return appConfig;
      },
      async recordBlindComparisonSelection(selection) {
        appConfig.blindComparisons.push({
          ...selection,
          createdAt: "1710000001"
        });
        return appConfig;
      },
      async startPipeline(request) {
        lastRequest = request;
        if (request.gpuId !== 0) {
          throw new Error(`Expected selected GPU 0, received ${request.gpuId}`);
        }
        if (![
          "realesrgan-x4plus",
          "realesrnet-x4plus",
          "bsrgan-x4",
          "swinir-realworld-x4"
        ].includes(request.modelId)) {
          throw new Error(`Expected selected model realesrgan-x4plus, received ${request.modelId}`);
        }
        const modelSuffix = request.modelId.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
        activeJob = {
          jobId: `mock-job-${modelSuffix}`,
          state: "running",
          progress: {
            phase: "upscaling",
            percent: 62,
            message: "Upscaling extracted frames",
            processedFrames: 180,
            totalFrames: 300,
            extractedFrames: 300,
            upscaledFrames: 180,
            encodedFrames: 0,
            remuxedFrames: 0
          },
          result: {
            outputPath: request.outputPath,
            workDir: "C:/workspace/jobs/mock-job",
            frameCount: 300,
            hadAudio: true,
            codec: request.codec,
            container: request.container,
            runtime: {
              ffmpegPath: "C:/tools/ffmpeg.exe",
              realesrganPath: "C:/tools/realesrgan-ncnn-vulkan.exe",
              modelDir: "C:/tools/models",
              availableGpus: [
                { id: 0, name: "Intel(R) Graphics", kind: "integrated" },
                { id: 1, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", kind: "discrete" }
              ],
              defaultGpuId: 1
            },
            log: [`Completed mock pipeline for ${request.modelId}`, "Remuxed original audio"]
          },
          error: null
        };
        return activeJob.jobId;
      },
      async getPipelineJob() {
        if (!activeJob) {
          throw new Error("No mock job available");
        }

        return {
          ...activeJob,
          state: "succeeded",
          progress: {
            phase: "completed",
            percent: 100,
            message: "Pipeline completed",
            processedFrames: 300,
            totalFrames: 300,
            extractedFrames: 300,
            upscaledFrames: 300,
            encodedFrames: 300,
            remuxedFrames: 300
          }
        };
      },
      async openPathInDefaultApp(path) {
        openedPaths.push(path);
      },
      toPreviewSrc() {
        return "https://example.com/mock-preview.mp4";
      }
    };
  });
});

test("selects a source, previews it, chooses output, and runs the workflow", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("select-video-button").click();
  await expect(page.getByTestId("source-preview")).toBeVisible();
  await expect(page.getByText("C:/fixtures/sample-input.mp4")).toBeVisible();
  await expect(page.getByTestId("selected-model-label")).toContainText("Real-ESRGAN x4 Plus");
  await expect(page.getByTestId("selected-model-summary")).toContainText("photographic");
  await page.getByTestId("model-rating-select").selectOption("4");
  await expect(page.getByTestId("rating-summary")).toContainText("Saved rating: 4/5");
  await expect(page.getByTestId("gpu-select")).toHaveValue("1");
  await page.getByTestId("gpu-select").selectOption("0");

  await page.getByTestId("output-mode-select").selectOption("cropTo4k");
  await page.getByTestId("aspect-ratio-select").selectOption("1:1");
  await page.getByTestId("resolution-basis-select").selectOption("width");
  await page.getByTestId("target-width-input").fill("2048");
  await expect(page.getByTestId("target-height-input")).toHaveValue("2048");
  await expect(page.getByTestId("crop-overlay")).toBeVisible();
  const handle = page.getByTestId("crop-handle-se");
  const box = await handle.boundingBox();
  if (!box) {
    throw new Error("Crop handle not rendered");
  }
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 24, box.y + box.height / 2 + 24);
  await page.mouse.up();

  await page.getByTestId("container-select").selectOption("mkv");
  await page.getByTestId("codec-select").selectOption("h265");
  await page.getByTestId("preview-mode-checkbox").check();
  await page.getByTestId("preview-duration-input").fill("8");
  await page.getByTestId("save-output-button").click();
  await expect(page.getByTestId("output-path-input")).toHaveValue("C:/exports/upscaled-result.mkv");

  await page.getByTestId("run-blind-comparison-button").click();
  await expect(page.getByTestId("blind-preview-sample-1")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-2")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-3")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-4")).toBeVisible();
  await expect(page.getByTestId("comparison-inspector")).toBeVisible();
  await page.getByTestId("blind-open-sample-1").click();
  await page.getByTestId("comparison-focus-diagonals").click();
  await page.getByTestId("comparison-zoom-slider").fill("4");
  await expect(page.locator("[data-testid^='blind-reveal-']")).toHaveCount(0);
  await page.getByTestId("pick-sample-1").click();
  await expect(page.locator("[data-testid^='blind-reveal-']")).toHaveCount(4);
  await expect(page.getByText("Selected winner")).toBeVisible();

  await page.getByTestId("run-upscale-button").click();
  await expect(page.getByTestId("job-progress-panel")).toBeVisible();
  await expect(page.getByTestId("progress-upscaled-frames")).toContainText("300");
  await expect(page.getByTestId("progress-remuxed-frames")).toContainText("300");
  await expect(page.getByTestId("result-output-path")).toContainText("C:/exports/upscaled-result.mkv");
  await expect(page.getByTestId("pipeline-log")).toContainText("Completed mock pipeline for realesrgan-x4plus");
  await expect(page.getByText("Original audio remuxed")).toBeVisible();
  await expect(page.getByTestId("result-preview")).toBeVisible();
  await expect(page.getByText("Blind Picks Logged")).toBeVisible();
});
