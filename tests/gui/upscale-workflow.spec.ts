import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    let activeJob = null;
    let activeConversionJob = null;
    let lastRequest = null;
    const nowSeconds = Math.floor(Date.now() / 1000);
    const openedPaths = [];
    const deletedPaths = [];
    const confirmMessages = [];
    const appConfig = {
      modelRatings: {},
      blindComparisons: []
    };
    window.confirm = (message) => {
      confirmMessages.push(String(message));
      return true;
    };
    window.__UPSCALER_TEST_STATE__ = {
      openedPaths,
      deletedPaths,
      confirmMessages,
      lastRequest: null
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
          previewPath: "C:/fixtures/sample-input-preview.mp4",
          width: 1280,
          height: 720,
          durationSeconds: 12.5,
          frameRate: 24,
          hasAudio: true,
          container: "mp4"
        };
      },
      async startSourceConversionToMp4(sourcePath) {
        activeConversionJob = {
          jobId: "mock-conversion-job",
          sourcePath
        };
        return activeConversionJob.jobId;
      },
      async getSourceConversionJob() {
        if (!activeConversionJob) {
          throw new Error("No mock conversion job available");
        }

        return {
          jobId: activeConversionJob.jobId,
          state: "succeeded",
          progress: {
            phase: "completed",
            percent: 100,
            message: "Source conversion completed",
            processedFrames: 55000,
            totalFrames: 55000,
            extractedFrames: 0,
            upscaledFrames: 0,
            interpolatedFrames: 0,
            encodedFrames: 0,
            remuxedFrames: 0
          },
          result: {
            path: activeConversionJob.sourcePath.replace(/\.avi$/i, "_fastprep.mp4"),
            previewPath: activeConversionJob.sourcePath.replace(/\.avi$/i, "_fastprep.mp4"),
            width: 1280,
            height: 720,
            durationSeconds: 12.5,
            frameRate: 24,
            hasAudio: true,
            container: "mp4"
          },
          error: null
        };
      },
      async cancelSourceConversionJob() {
        activeConversionJob = null;
      },
      async getPathStats(path) {
        const normalized = String(path);
        const isDirectory = /\/jobs$|\/converted-sources$|\/source-previews$|\/mock-job$/i.test(normalized.replace(/\\/g, "/"));
        return {
          path: normalized,
          exists: true,
          isDirectory,
          sizeBytes: isDirectory ? 1024 * 1024 * 12 : 1024 * 1024 * 3
        };
      },
      async getScratchStorageSummary() {
        return {
          jobsRoot: {
            path: "C:/workspace/artifacts/jobs",
            exists: true,
            isDirectory: true,
            sizeBytes: 1024 * 1024 * 24
          },
          convertedSourcesRoot: {
            path: "C:/workspace/artifacts/runtime/converted-sources",
            exists: true,
            isDirectory: true,
            sizeBytes: 1024 * 1024 * 48
          },
          sourcePreviewsRoot: {
            path: "C:/workspace/artifacts/runtime/source-previews",
            exists: true,
            isDirectory: true,
            sizeBytes: 1024 * 1024 * 6
          }
        };
      },
      async listManagedJobs() {
        return [
          {
            jobId: "historic-pipeline-job",
            jobKind: "pipeline",
            label: "Historical Upscale Job",
            state: "succeeded",
            sourcePath: "C:/workspace/fixtures/historic-input.mov",
            modelId: "realesrgan-x4plus",
            codec: "h265",
            container: "mkv",
            progress: {
              phase: "completed",
              percent: 100,
              message: "Pipeline completed",
              processedFrames: 1200,
              totalFrames: 1200,
              extractedFrames: 1200,
              upscaledFrames: 1200,
              interpolatedFrames: 0,
              encodedFrames: 1200,
              remuxedFrames: 1200,
              elapsedSeconds: 80,
              averageFramesPerSecond: 15,
              rollingFramesPerSecond: 0,
              estimatedRemainingSeconds: 0,
              processRssBytes: 1024 * 1024 * 768,
              gpuMemoryUsedBytes: 1024 * 1024 * 8192,
              gpuMemoryTotalBytes: 1024 * 1024 * 24576,
              scratchSizeBytes: 1024 * 1024 * 32,
              outputSizeBytes: 1024 * 1024 * 18,
              extractStageSeconds: 6,
              upscaleStageSeconds: 52,
              interpolateStageSeconds: 0,
              encodeStageSeconds: 15,
              remuxStageSeconds: 7
            },
            recordedCount: 1200,
            scratchPath: "C:/workspace/artifacts/jobs/job_historic-pipeline-job",
            scratchStats: {
              path: "C:/workspace/artifacts/jobs/job_historic-pipeline-job",
              exists: true,
              isDirectory: true,
              sizeBytes: 1024 * 1024 * 32
            },
            outputPath: "C:/workspace/artifacts/outputs/historic-upscale.mkv",
            outputStats: {
              path: "C:/workspace/artifacts/outputs/historic-upscale.mkv",
              exists: true,
              isDirectory: false,
              sizeBytes: 1024 * 1024 * 18
            },
            updatedAt: String(nowSeconds - 3600)
          },
          {
            jobId: "conv_historic-source-job",
            jobKind: "sourceConversion",
            label: "Historical Conversion",
            state: "succeeded",
            sourcePath: "C:/workspace/fixtures/historic-source.avi",
            modelId: null,
            codec: null,
            container: "mp4",
            progress: {
              phase: "completed",
              percent: 100,
              message: "Source conversion completed",
              processedFrames: 55000,
              totalFrames: 55000,
              extractedFrames: 0,
              upscaledFrames: 0,
              interpolatedFrames: 0,
              encodedFrames: 0,
              remuxedFrames: 0
            },
            recordedCount: 55000,
            scratchPath: null,
            scratchStats: null,
            outputPath: "C:/workspace/artifacts/runtime/converted-sources/historic-source.mp4",
            outputStats: {
              path: "C:/workspace/artifacts/runtime/converted-sources/historic-source.mp4",
              exists: true,
              isDirectory: false,
              sizeBytes: 1024 * 1024 * 9
            },
            updatedAt: String(nowSeconds - 86400)
          }
        ];
      },
      async deleteManagedPath(path) {
        deletedPaths.push(path);
        return;
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
        window.__UPSCALER_TEST_STATE__.lastRequest = request;
        if (request.gpuId !== 0) {
          throw new Error(`Expected selected GPU 0, received ${request.gpuId}`);
        }
        if (request.modelId === "swinir-realworld-x4" && !["torch", "tensorrt"].includes(request.pytorchRunner)) {
          throw new Error(`Expected a valid PyTorch runner for SwinIR, received ${request.pytorchRunner}`);
        }
        if (request.previewMode === false && request.segmentDurationSeconds !== 10) {
          throw new Error(`Expected export chunk duration 10, received ${request.segmentDurationSeconds}`);
        }
        if (!["off", "afterUpscale", "interpolateOnly"].includes(request.interpolationMode)) {
          throw new Error(`Expected a valid interpolation mode, received ${request.interpolationMode}`);
        }
        if (request.interpolationMode === "off" && request.interpolationTargetFps !== null) {
          throw new Error(`Expected null interpolation target when interpolation is off, received ${request.interpolationTargetFps}`);
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
            interpolatedFrames: 0,
            encodedFrames: 0,
            remuxedFrames: 0,
            segmentIndex: 1,
            segmentCount: 1,
            segmentProcessedFrames: 180,
            segmentTotalFrames: 300,
            batchIndex: 15,
            batchCount: 25,
            elapsedSeconds: 30,
            averageFramesPerSecond: 6,
            rollingFramesPerSecond: 7.5,
            estimatedRemainingSeconds: 20,
            processRssBytes: 1024 * 1024 * 512,
            gpuMemoryUsedBytes: 1024 * 1024 * 6144,
            gpuMemoryTotalBytes: 1024 * 1024 * 24576,
            scratchSizeBytes: 1024 * 1024 * 12,
            outputSizeBytes: 1024 * 1024 * 3,
            extractStageSeconds: 4,
            upscaleStageSeconds: 18,
            interpolateStageSeconds: 0,
            encodeStageSeconds: 5,
            remuxStageSeconds: 3
          },
          result: {
            outputPath: request.outputPath,
            workDir: "C:/workspace/artifacts/jobs/mock-job",
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
            log: [
              `Completed mock pipeline for ${request.modelId}`,
              "Average throughput: 6.00 fps",
              "Stage timings: extract 4s, upscale 18s, encode 5s, remux 3s",
              "Remuxed original audio"
            ]
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
            ...activeJob.progress,
            phase: "completed",
            percent: 100,
            message: "Pipeline completed",
            processedFrames: 300,
            totalFrames: 300,
            extractedFrames: 300,
            upscaledFrames: 300,
            interpolatedFrames: 0,
            encodedFrames: 300,
            remuxedFrames: 300
          }
        };
      },
      async cancelPipelineJob() {
        if (!activeJob) {
          return;
        }

        activeJob = {
          ...activeJob,
          state: "cancelled",
          result: null,
          error: "Job cancelled by user",
          progress: {
            ...activeJob.progress,
            phase: "failed",
            percent: 100,
            message: "Job cancelled by user"
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

test("plays a real preview fixture through the GUI controls", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.toPreviewSrc = () => "/fixtures/gui-progress-sample.mp4";
    window.__UPSCALER_MOCK__.probeSourceVideo = async (sourcePath) => ({
      path: sourcePath,
      previewPath: sourcePath,
      width: 1280,
      height: 720,
      durationSeconds: 12.5,
      frameRate: 24,
      hasAudio: true,
      container: "mp4"
    });
  });

  await page.getByTestId("select-video-button").click();
  await expect(page.getByTestId("source-preview")).toBeVisible();
  await expect(page.getByTestId("source-preview-play-toggle")).toContainText("Play");

  await page.getByTestId("source-preview-play-toggle").click();

  await page.waitForFunction(() => {
    const video = document.querySelector('[data-testid="source-preview"]') as HTMLVideoElement | null;
    return Boolean(video && !video.paused && video.currentTime > 0.15);
  }, { timeout: 10000 });

  await expect(page.getByTestId("source-preview-play-toggle")).toContainText("Pause");

  const playbackState = await page.evaluate(() => {
    const video = document.querySelector('[data-testid="source-preview"]') as HTMLVideoElement | null;
    if (!video) {
      return null;
    }

    return {
      paused: video.paused,
      currentTime: video.currentTime,
      duration: video.duration,
      readyState: video.readyState,
    };
  });

  expect(playbackState).not.toBeNull();
  expect(playbackState?.paused).toBe(false);
  expect(playbackState?.currentTime ?? 0).toBeGreaterThan(0.15);
  expect(playbackState?.duration ?? 0).toBeGreaterThan(1);
});

test("selects a source, previews it, chooses output, and runs the workflow", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "VideoUpgrader" })).toBeVisible();
  await expect(page.getByTestId("upscaler-section-card")).toContainText("Spatial detail pipeline");
  await expect(page.getByTestId("interpolator-section-card")).toContainText("Motion interpolation workspace");
  await expect(page.getByTestId("frame-rate-workspace-section")).toContainText("Interpolation Workspace");

  await page.evaluate(() => {
    Object.defineProperty(HTMLMediaElement.prototype, "paused", {
      configurable: true,
      get() {
        return this.dataset.mockPaused !== "false";
      }
    });

    Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
      configurable: true,
      get() {
        return Number(this.dataset.mockCurrentTime ?? "0");
      },
      set(value: number) {
        this.dataset.mockCurrentTime = String(value);
      }
    });

    Object.defineProperty(HTMLMediaElement.prototype, "duration", {
      configurable: true,
      get() {
        return Number(this.dataset.mockDuration ?? "12.5");
      }
    });

    HTMLMediaElement.prototype.play = async function playMock() {
      this.dataset.mockPaused = "false";
      this.dispatchEvent(new Event("play"));
    };

    HTMLMediaElement.prototype.pause = function pauseMock() {
      this.dataset.mockPaused = "true";
      this.dispatchEvent(new Event("pause"));
    };
  });

  await page.getByTestId("select-video-button").click();
  await expect(page.getByTestId("source-preview")).toBeVisible();
  await expect(page.getByText("C:/fixtures/sample-input.mp4")).toBeVisible();
  await expect(page.getByTestId("source-preview")).toHaveJSProperty("controls", true);
  await expect(page.getByTestId("source-preview-toolbar")).toBeVisible();
  await expect(page.getByTestId("source-preview-seek")).toBeVisible();
  await expect(page.getByTestId("source-preview-timecode")).toContainText("0:00 / 0:12");
  await expect(page.getByTestId("source-preview-play-toggle")).toContainText("Play");
  await page.getByTestId("source-preview-play-toggle").click();
  await expect(page.getByTestId("source-preview-play-toggle")).toContainText("Pause");
  await expect(page.getByTestId("source-preview")).toHaveJSProperty("paused", false);
  await page.evaluate(() => {
    const video = document.querySelector('[data-testid="source-preview"]') as HTMLVideoElement | null;
    if (video) {
      video.currentTime = 4.25;
      video.dispatchEvent(new Event("timeupdate"));
    }
  });
  await expect(page.getByTestId("source-preview-seek")).toHaveValue("4.25");
  await expect(page.getByTestId("source-preview-timecode")).toContainText("0:04 / 0:12");
  await page.getByTestId("source-preview-seek").evaluate((element: HTMLInputElement) => {
    element.value = "7.5";
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.getByTestId("source-preview")).toHaveJSProperty("currentTime", 7.5);
  await expect(page.getByTestId("source-preview-timecode")).toContainText("0:07 / 0:12");
  await page.getByTestId("source-preview-restart").click();
  await expect(page.getByTestId("source-preview")).toHaveJSProperty("currentTime", 0);
  await expect(page.getByTestId("source-preview-seek")).toHaveValue("0");
  await page.getByTestId("source-preview-play-toggle").click();
  await expect(page.getByTestId("source-preview")).toHaveJSProperty("paused", true);
  await expect(page.locator('[data-testid="model-select"] optgroup[label="Available Now"] option')).toHaveCount(4);
  await expect(page.locator('[data-testid="model-select"] optgroup[label="Planned"] option')).toHaveCount(3);
  await expect(page.locator('[data-testid="model-select"] option[disabled]')).toHaveCount(3);
  await expect(page.locator('[data-testid="model-select"] option[value="hat-realhat-gan-x4"]')).toContainText("not implemented");
  await expect(page.locator('[data-testid="model-select"] option[value="rife-v4.6"]')).toContainText("not implemented");
  await expect(page.getByTestId("target-model-set-card")).toHaveCount(0);
  await expect(page.getByTestId("selected-model-label")).toContainText("Real-ESRGAN x4 Plus");
  await expect(page.getByTestId("selected-model-summary")).toContainText("photographic");
  await page.getByTestId("model-rating-select").selectOption("4");
  await expect(page.getByTestId("rating-summary")).toContainText("Saved rating: 4/5");
  await expect(page.getByTestId("gpu-select")).toHaveValue("1");
  await page.getByTestId("gpu-select").selectOption("0");

  await page.getByTestId("output-mode-select").selectOption("cropTo4k");
  await expect(page.getByTestId("crop-overlay")).toBeVisible();
  await expect(page.getByTestId("crop-overlay-label")).toContainText("Crop Frame");
  await expect(page.getByTestId("crop-nudge-controls")).toBeVisible();
  await expect(page.getByTestId("crop-nudge-right")).toBeDisabled();
  await page.getByTestId("toggle-crop-edit-button").click();
  await expect(page.getByTestId("crop-overlay-label")).toContainText("Crop Editing");
  await expect(page.getByTestId("crop-nudge-right")).toBeEnabled();
  await page.getByTestId("aspect-ratio-select").selectOption("1:1");
  await page.getByTestId("resolution-basis-select").selectOption("width");
  await page.getByTestId("target-width-input").fill("2048");
  await expect(page.getByTestId("target-height-input")).toHaveValue("2048");
  await expect(page.getByTestId("crop-overlay")).toBeVisible();
  const initialCropBox = await page.getByTestId("crop-overlay").boundingBox();
  if (!initialCropBox) {
    throw new Error("Initial crop overlay not rendered");
  }
  if (Math.abs((initialCropBox.width / initialCropBox.height) - 1) > 0.06) {
    throw new Error(`Expected initial crop overlay to remain square, received ${initialCropBox.width}x${initialCropBox.height}`);
  }
  const handle = page.getByTestId("crop-handle-se");
  const box = await handle.boundingBox();
  if (!box) {
    throw new Error("Crop handle not rendered");
  }
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 24, box.y + box.height / 2 - 24);
  await page.mouse.up();
  const resizedCropBox = await page.getByTestId("crop-overlay").boundingBox();
  if (!resizedCropBox) {
    throw new Error("Resized crop overlay not rendered");
  }
  if (Math.abs((resizedCropBox.width / resizedCropBox.height) - 1) > 0.06) {
    throw new Error(`Expected resized crop overlay to remain square, received ${resizedCropBox.width}x${resizedCropBox.height}`);
  }
  await page.getByTestId("crop-nudge-right").click();
  const nudgedCropBox = await page.getByTestId("crop-overlay").boundingBox();
  if (!nudgedCropBox) {
    throw new Error("Nudged crop overlay not rendered");
  }
  if (nudgedCropBox.x <= initialCropBox.x) {
    throw new Error(`Expected crop nudge to move the overlay right, but x moved from ${initialCropBox.x} to ${nudgedCropBox.x}`);
  }
  await page.getByTestId("maximize-crop-button").click();
  const maximizedCropBox = await page.getByTestId("crop-overlay").boundingBox();
  if (!maximizedCropBox) {
    throw new Error("Maximized crop overlay not rendered");
  }
  if (Math.abs((maximizedCropBox.width / maximizedCropBox.height) - 1) > 0.06) {
    throw new Error(`Expected maximized crop overlay to remain square, received ${maximizedCropBox.width}x${maximizedCropBox.height}`);
  }
  if (Math.abs(maximizedCropBox.x - initialCropBox.x) > 1.5) {
    throw new Error(`Expected maximize crop to restore the centered crop position, but x moved from ${initialCropBox.x} to ${maximizedCropBox.x}`);
  }
  await page.getByTestId("aspect-ratio-select").selectOption("16:9");
  const fullFrameCropBox = await page.getByTestId("crop-overlay").boundingBox();
  const previewBox = await page.getByTestId("source-preview").boundingBox();
  if (!fullFrameCropBox || !previewBox) {
    throw new Error("Expected preview and crop overlay to be rendered for 16:9 reset check");
  }
  if (Math.abs(fullFrameCropBox.width - previewBox.width) > 2 || Math.abs(fullFrameCropBox.height - previewBox.height) > 2) {
    throw new Error(`Expected 16:9 crop to fill the full preview frame, but crop was ${fullFrameCropBox.width}x${fullFrameCropBox.height} and preview was ${previewBox.width}x${previewBox.height}`);
  }
  const fullFrameHandle = page.getByTestId("crop-handle-se");
  const fullFrameHandleBox = await fullFrameHandle.boundingBox();
  if (!fullFrameHandleBox) {
    throw new Error("16:9 crop handle not rendered");
  }
  await page.mouse.move(fullFrameHandleBox.x + fullFrameHandleBox.width / 2, fullFrameHandleBox.y + fullFrameHandleBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(fullFrameHandleBox.x + fullFrameHandleBox.width / 2 - 40, fullFrameHandleBox.y + fullFrameHandleBox.height / 2 - 40);
  await page.mouse.up();
  const draggedSixteenNineCropBox = await page.getByTestId("crop-overlay").boundingBox();
  if (!draggedSixteenNineCropBox) {
    throw new Error("Dragged 16:9 crop overlay not rendered");
  }
  const previewAspect = previewBox.width / previewBox.height;
  const draggedAspect = draggedSixteenNineCropBox.width / draggedSixteenNineCropBox.height;
  if (Math.abs(draggedAspect - previewAspect) > 0.06) {
    throw new Error(`Expected dragged 16:9 crop to preserve preview aspect ${previewAspect}, received ${draggedAspect}`);
  }
  await page.getByTestId("toggle-crop-edit-button").click();

  await page.getByTestId("container-select").selectOption("mkv");
  await page.getByTestId("codec-select").selectOption("h265");
  await page.getByTestId("frame-rate-mode-select").selectOption("afterUpscale");
  await page.getByTestId("frame-rate-target-select").selectOption("60");
  await expect(page.getByTestId("interpolation-workspace-summary")).toContainText("Post-upscale interpolation");
  await page.getByTestId("preview-mode-checkbox").check();
  await page.getByTestId("preview-duration-input").fill("8");
  await page.getByTestId("save-output-button").click();
  await expect(page.getByTestId("output-path-input")).toHaveValue("C:/exports/upscaled-result.mkv");

  await page.getByTestId("blind-test-panel-toggle").click();
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
  await expect(page.getByTestId("progress-segment-counter")).toContainText("1/1");
  await expect(page.getByTestId("progress-segment-frames")).toContainText("180/300");
  await expect(page.getByTestId("progress-batch-counter")).toContainText("15/25");
  await expect(page.getByTestId("progress-average-fps")).toContainText("6.00 fps");
  await expect(page.getByTestId("progress-rolling-fps")).toContainText("7.50 fps");
  await expect(page.getByTestId("progress-eta")).toContainText("20s");
  await expect(page.getByTestId("progress-process-rss")).toContainText("512 MB");
  await expect(page.getByTestId("progress-gpu-memory")).toContainText("6.0 GB / 24 GB");
  await expect(page.getByTestId("progress-stage-timings")).toContainText("extract 4s");
  await expect(page.getByTestId("progress-current-activity")).toContainText("Pipeline completed");
  await expect(page.getByTestId("progress-current-detail")).toContainText("segment 1/1");
  await expect(page.getByTestId("progress-last-update")).toContainText("Last update");
  await expect(page.getByTestId("progress-event-log")).toContainText("Pipeline completed");
  await expect(page.getByTestId("result-output-path")).toContainText("C:/exports/upscaled-result.mkv");
  await expect(page.getByTestId("pipeline-log")).toContainText("Completed mock pipeline for realesrgan-x4plus");
  await expect(page.getByTestId("pipeline-log")).toContainText("Stage timings: extract 4s, upscale 18s, encode 5s, remux 3s");
  await expect(page.getByText("Original audio remuxed")).toBeVisible();
  await expect(page.getByTestId("result-preview")).toBeVisible();
  await expect(page.getByText("Blind Picks Logged")).toBeVisible();

  const { lastRequest: workflowRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(workflowRequest?.interpolationMode).toBe("afterUpscale");
  expect(workflowRequest?.interpolationTargetFps).toBe(60);
});

test("shows the PyTorch runner selector only for PyTorch image SR models and passes the selection through", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("select-video-button").click();
  await expect(page.getByTestId("pytorch-runner-select")).toBeHidden();

  await page.getByTestId("model-select").selectOption("swinir-realworld-x4");
  await expect(page.getByTestId("selected-model-label")).toContainText("SwinIR Real-World x4");
  await expect(page.getByTestId("selected-model-summary")).toContainText("Transformer-based");
  await expect(page.getByTestId("pytorch-runner-select")).toBeVisible();
  await expect(page.getByTestId("pytorch-runner-select")).toHaveValue("tensorrt");

  await page.getByTestId("model-select").selectOption("bsrgan-x4");
  await expect(page.getByTestId("pytorch-runner-select")).toHaveValue("torch");

  await page.getByTestId("model-select").selectOption("swinir-realworld-x4");
  await expect(page.getByTestId("pytorch-runner-select")).toHaveValue("tensorrt");
  await page.getByTestId("pytorch-runner-select").selectOption("tensorrt");

  await page.getByTestId("gpu-select").selectOption("0");
  await page.getByTestId("run-upscale-button").click();

  const { lastRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(lastRequest?.modelId).toBe("swinir-realworld-x4");
  expect(lastRequest?.pytorchRunner).toBe("tensorrt");
  expect(lastRequest?.interpolationMode).toBe("off");
  expect(lastRequest?.interpolationTargetFps).toBeNull();
});

test("warns per job when the interpolation target is not higher than the source frame rate", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.probeSourceVideo = async (sourcePath) => ({
      path: sourcePath,
      previewPath: "C:/fixtures/sample-input-preview.mp4",
      width: 1280,
      height: 720,
      durationSeconds: 12.5,
      frameRate: 30,
      hasAudio: true,
      container: "mp4"
    });
  });

  await page.getByTestId("select-video-button").click();
  await page.getByTestId("frame-rate-mode-select").selectOption("interpolateOnly");
  await page.getByTestId("frame-rate-target-select").selectOption("30");
  await page.getByTestId("run-upscale-button").click();

  const { confirmMessages, lastRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(confirmMessages).toHaveLength(1);
  expect(confirmMessages[0]).toContain("selected interpolation target of 30 fps is not higher");
  expect(lastRequest?.interpolationMode).toBe("interpolateOnly");
  expect(lastRequest?.interpolationTargetFps).toBe(30);
});

test("shows not-implemented models in the selector and blocks export when one becomes current", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("select-video-button").click();
  await expect(page.locator('[data-testid="model-select"] option[value="rvrt-x4"]')).toHaveJSProperty("disabled", true);

  await page.evaluate(() => {
    const select = document.querySelector('[data-testid="model-select"]') as HTMLSelectElement | null;
    if (!select) {
      return;
    }
    select.value = "rvrt-x4";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });

  await expect(page.getByTestId("selected-model-label")).toContainText("RVRT x4");
  await expect(page.getByTestId("selected-model-status")).toContainText("not implemented");
  await expect(page.getByTestId("selected-model-availability")).toContainText("not implemented yet");
  await expect(page.getByTestId("run-disabled-reason")).toContainText("export is disabled");
  await expect(page.getByTestId("run-upscale-button")).toBeDisabled();
});

test("upgrades an avi preview in the background and still supports manual conversion", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.selectVideoFile = async () => "C:/fixtures/sample-input.avi";
    window.__UPSCALER_MOCK__.probeSourceVideo = async (sourcePath) => ({
      path: sourcePath,
      previewPath: "C:/fixtures/sample-input-preview.mp4",
      width: 640,
      height: 480,
      durationSeconds: 55,
      frameRate: 29.97,
      hasAudio: true,
      container: "avi"
    });
  });

  await page.getByTestId("select-video-button").click();
  await expect(page.getByText("C:/fixtures/sample-input.avi")).toBeVisible();
  await expect(page.getByTestId("source-preview-mode")).toContainText("Full-length converted preview");
  await expect(page.getByTestId("source-preview-guidance")).toContainText("full-length converted preview");
  await expect(page.getByTestId("run-upscale-button")).toBeEnabled();
  await expect(page.getByTestId("convert-source-to-mp4-button")).toBeVisible();

  await page.getByTestId("convert-source-to-mp4-button").click();
  await expect(page.getByTestId("conversion-progress-panel")).toBeVisible();
  await expect(page.getByText("C:/fixtures/sample-input_fastprep.mp4")).toBeVisible();
  await expect(page.getByText("Source converted to MP4.")).toBeVisible();
});

test("routes webm sources through the converted preview path", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.selectVideoFile = async () => "C:/fixtures/sample-input.webm";
    window.__UPSCALER_MOCK__.probeSourceVideo = async (sourcePath) => ({
      path: sourcePath,
      previewPath: "C:/fixtures/sample-input-preview.mp4",
      width: 1280,
      height: 720,
      durationSeconds: 61,
      frameRate: 23.976,
      hasAudio: true,
      container: "webm"
    });
  });

  await page.getByTestId("select-video-button").click();
  await expect(page.getByText("C:/fixtures/sample-input.webm")).toBeVisible();
  await expect(page.getByTestId("source-preview-mode")).toContainText("Full-length converted preview");
  await expect(page.getByTestId("source-preview-guidance")).toContainText("full-length converted preview");
  await expect(page.getByTestId("convert-source-to-mp4-button")).toBeVisible();
  await expect(page.getByTestId("run-upscale-button")).toBeEnabled();
});

test("shows historical cleanup jobs and runs cleanup actions", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("job-cleanup-panel-toggle").click();
  await expect(page.getByTestId("cleanup-sort-select")).toHaveValue("largest");
  await expect(page.getByTestId("cleanup-jobs-table")).toBeVisible();
  await expect(page.getByTestId("cleanup-job-historic-pipeline-job")).toContainText("Historical Upscale Job");
  await expect(page.getByTestId("cleanup-job-conv_historic-source-job")).toContainText("Historical Conversion");
  await expect(page.getByTestId("cleanup-directory-historic-pipeline-job")).toContainText("job_historic-pipeline-job");
  await expect(page.getByTestId("cleanup-input-historic-pipeline-job")).toContainText("historic-input.mov");
  await expect(page.getByTestId("cleanup-output-historic-pipeline-job")).toContainText("historic-upscale.mkv");
  await expect(page.getByTestId("cleanup-updated-historic-pipeline-job")).toContainText("ago");

  await page.getByTestId("cleanup-search-input").fill("pipeline completed");
  await expect(page.getByText("Historical Upscale Job")).toBeVisible();
  await expect(page.getByTestId("cleanup-empty-filter")).toBeHidden();
  await page.getByTestId("cleanup-search-input").fill("");

  await page.getByTestId("cleanup-filter-cancelled").click();
  await expect(page.getByTestId("cleanup-empty-filter")).toBeVisible();
  await page.getByTestId("cleanup-filter-succeeded").click();
  await expect(page.getByText("Historical Upscale Job")).toBeVisible();

  await page.getByTestId("cleanup-expand-historic-pipeline-job").click();
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Model: realesrgan-x4plus");
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Codec / Container: h265 / mkv");
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Average / Current Throughput: 15.0 fps / calculating");
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Stage Times: extract 6s, upscale 52s, encode 15s, remux 7s");

  await page.getByTestId("cleanup-open-output-historic-pipeline-job").click();
  await page.getByTestId("cleanup-open-scratch-historic-pipeline-job").click();
  await page.getByTestId("cleanup-bulk-output").click();

  await page.getByTestId("cleanup-clear-scratch-historic-pipeline-job").click();
  await page.getByTestId("cleanup-expand-conv_historic-source-job").click();
  await page.getByTestId("cleanup-delete-output-conv_historic-source-job").click();

  const { deletedPaths, openedPaths, confirmMessages } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(openedPaths).toEqual([
    "C:/workspace/artifacts/outputs/historic-upscale.mkv",
    "C:/workspace/artifacts/jobs/job_historic-pipeline-job"
  ]);
  expect(confirmMessages).toHaveLength(1);
  expect(confirmMessages[0]).toContain("Impacted size");
  expect(deletedPaths).toEqual([
    "C:/workspace/artifacts/outputs/historic-upscale.mkv",
    "C:/workspace/artifacts/runtime/converted-sources/historic-source.mp4",
    "C:/workspace/artifacts/jobs/job_historic-pipeline-job",
    "C:/workspace/artifacts/runtime/converted-sources/historic-source.mp4"
  ]);
});
