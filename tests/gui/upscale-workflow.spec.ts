import { expect, test } from "@playwright/test";

async function openUpscaleControls(page) {
  const workspace = page.getByTestId("upscaler-workspace-section");
  if (!(await workspace.isVisible().catch(() => false))) {
    await page.getByTestId("pipeline-toggle-upscale").click();
  }
}

async function openInterpolationControls(page) {
  const workspace = page.getByTestId("frame-rate-workspace-section");
  if (!(await workspace.isVisible().catch(() => false))) {
    await page.getByTestId("pipeline-toggle-interpolation").click();
  }
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    let activeJob = null;
    let activeConversionJob = null;
    let lastRequest = null;
    const nowSeconds = Math.floor(Date.now() / 1000);
    const openedPaths = [];
    const deletedPaths = [];
    const confirmMessages = [];
    const pipelineRequests = [];
    const previewLoadPaths = [];
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
      pipelineRequests,
      previewLoadPaths,
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
          rifePath: "C:/tools/rife-ncnn-vulkan.exe",
          rifeModelRoot: "C:/tools/rife-models",
          availableGpus: [
            { id: 0, name: "Intel(R) Graphics", kind: "integrated" },
            { id: 1, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", kind: "discrete" }
          ],
          defaultGpuId: 1,
          externalResearchRuntimes: {
            "rvrt-x4": {
              kind: "external-command",
              commandEnvVar: "UPSCALER_RVRT_COMMAND",
              configured: false
            }
          }
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
          container: "mp4",
          videoCodec: "h264"
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
            container: "mp4",
            videoCodec: "h264"
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
            pipelineRunDetails: {
              request: {
                sourcePath: "C:/workspace/fixtures/historic-input.mov",
                modelId: "realesrgan-x4plus",
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
                outputPath: "C:/workspace/artifacts/outputs/historic-upscale.mkv",
                codec: "h265",
                container: "mkv",
                tileSize: 128,
                fp16: false,
                crf: 16
              },
              sourceMedia: {
                width: 1280,
                height: 720,
                frameRate: 24,
                durationSeconds: 50,
                frameCount: 1200,
                aspectRatio: 1.7777777778,
                pixelCount: 921600,
                hasAudio: true,
                container: "mov",
                videoCodec: "prores"
              },
              outputMedia: {
                width: 3840,
                height: 2160,
                frameRate: 24,
                durationSeconds: 50,
                frameCount: 1200,
                aspectRatio: 1.7777777778,
                pixelCount: 8294400,
                hasAudio: true,
                container: "mkv",
                videoCodec: "hevc"
              },
              effectiveSettings: {
                effectiveTileSize: 128,
                processedDurationSeconds: 50,
                segmentFrameLimit: 240,
                previewMode: false,
                previewDurationSeconds: null,
                segmentDurationSeconds: 10
              },
              executionPath: "file-io",
              videoEncoder: "libx265",
              videoEncoderLabel: "HEVC (libx265)",
              runner: "torch",
              precision: "fp32",
              averageThroughputFps: 15,
              segmentCount: 5,
              segmentFrameLimit: 240,
              frameCount: 1200,
              hadAudio: true,
              runtime: {
                ffmpegPath: "C:/tools/ffmpeg.exe",
                realesrganPath: "C:/tools/realesrgan-ncnn-vulkan.exe",
                modelDir: "C:/tools/models",
                rifePath: "C:/tools/rife-ncnn-vulkan.exe",
                rifeModelRoot: "C:/tools/rife-models",
                availableGpus: [
                  { id: 0, name: "Intel(R) Graphics", kind: "integrated" },
                  { id: 1, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", kind: "discrete" }
                ],
                defaultGpuId: 1
              }
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
          },
          {
            jobId: "legacy-pipeline-job",
            jobKind: "pipeline",
            label: "Legacy Upscale Job",
            state: "succeeded",
            sourcePath: "C:/workspace/fixtures/legacy-repeat-source.mp4",
            modelId: "realesrgan-x4plus",
            codec: "h264",
            container: "mp4",
            progress: {
              phase: "completed",
              percent: 100,
              message: "Legacy pipeline completed",
              processedFrames: 240,
              totalFrames: 240,
              extractedFrames: 240,
              upscaledFrames: 240,
              interpolatedFrames: 0,
              encodedFrames: 240,
              remuxedFrames: 240,
              elapsedSeconds: 18,
              averageFramesPerSecond: 13.3,
              rollingFramesPerSecond: 0,
              estimatedRemainingSeconds: 0,
              processRssBytes: 1024 * 1024 * 384,
              gpuMemoryUsedBytes: 1024 * 1024 * 2048,
              gpuMemoryTotalBytes: 1024 * 1024 * 24576,
              scratchSizeBytes: 1024 * 1024 * 6,
              outputSizeBytes: 1024 * 1024 * 4,
              extractStageSeconds: 2,
              upscaleStageSeconds: 10,
              interpolateStageSeconds: 0,
              encodeStageSeconds: 5,
              remuxStageSeconds: 1
            },
            recordedCount: 240,
            scratchPath: "C:/workspace/artifacts/jobs/job_legacy-pipeline-job",
            scratchStats: {
              path: "C:/workspace/artifacts/jobs/job_legacy-pipeline-job",
              exists: true,
              isDirectory: true,
              sizeBytes: 1024 * 1024 * 6
            },
            outputPath: "C:/workspace/artifacts/outputs/legacy-repeat-output.mp4",
            outputStats: {
              path: "C:/workspace/artifacts/outputs/legacy-repeat-output.mp4",
              exists: true,
              isDirectory: false,
              sizeBytes: 1024 * 1024 * 4
            },
            updatedAt: String(nowSeconds - 7200)
          },
          {
            jobId: "interrupted-pipeline-job",
            jobKind: "pipeline",
            label: "Interrupted Upscale Job",
            state: "interrupted",
            sourcePath: "C:/workspace/fixtures/interrupted-source.mp4",
            modelId: "realesrgan-x4plus",
            codec: "h265",
            container: "mkv",
            progress: {
              phase: "upscaling",
              percent: 41,
              message: "Worker stopped before the current segment completed",
              processedFrames: 168,
              totalFrames: 400,
              extractedFrames: 200,
              upscaledFrames: 168,
              interpolatedFrames: 0,
              encodedFrames: 0,
              remuxedFrames: 0,
              segmentIndex: 2,
              segmentCount: 4,
              segmentProcessedFrames: 68,
              segmentTotalFrames: 100,
              batchIndex: 3,
              batchCount: 5,
              elapsedSeconds: 22,
              averageFramesPerSecond: 7.6,
              rollingFramesPerSecond: 0,
              estimatedRemainingSeconds: 31,
              processRssBytes: 1024 * 1024 * 448,
              gpuMemoryUsedBytes: 1024 * 1024 * 4096,
              gpuMemoryTotalBytes: 1024 * 1024 * 24576,
              scratchSizeBytes: 1024 * 1024 * 14,
              outputSizeBytes: 1024 * 1024 * 1,
              extractStageSeconds: 4,
              upscaleStageSeconds: 18,
              interpolateStageSeconds: 0,
              encodeStageSeconds: 0,
              remuxStageSeconds: 0
            },
            recordedCount: 400,
            scratchPath: "C:/workspace/artifacts/jobs/job_interrupted-pipeline-job",
            scratchStats: {
              path: "C:/workspace/artifacts/jobs/job_interrupted-pipeline-job",
              exists: true,
              isDirectory: true,
              sizeBytes: 1024 * 1024 * 14
            },
            outputPath: "C:/workspace/artifacts/outputs/interrupted-output.mkv",
            outputStats: {
              path: "C:/workspace/artifacts/outputs/interrupted-output.mkv",
              exists: true,
              isDirectory: false,
              sizeBytes: 1024 * 1024 * 1
            },
            pipelineRunDetails: {
              request: {
                sourcePath: "C:/workspace/fixtures/interrupted-source.mp4",
                modelId: "realesrgan-x4plus",
                outputMode: "preserveAspect4k",
                qualityPreset: "qualityBalanced",
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
                cropLeft: null,
                cropTop: null,
                cropWidth: null,
                cropHeight: null,
                previewMode: false,
                previewDurationSeconds: null,
                segmentDurationSeconds: 10,
                outputPath: "C:/workspace/artifacts/outputs/interrupted-output.mkv",
                codec: "h265",
                container: "mkv",
                tileSize: 128,
                fp16: false,
                crf: 18
              }
            },
            updatedAt: String(nowSeconds - 1800)
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
        pipelineRequests.push(request);
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
          "swinir-realworld-x4",
          "rvrt-x4"
        ].includes(request.modelId)) {
          throw new Error(`Expected selected model realesrgan-x4plus, received ${request.modelId}`);
        }
        const modelSuffix = request.modelId.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
        const interpolationEnabled = request.interpolationMode !== "off";
        activeJob = {
          jobId: `mock-job-${modelSuffix}`,
          state: "running",
          progress: {
            phase: interpolationEnabled ? "interpolating" : "upscaling",
            percent: 62,
            message: interpolationEnabled ? "Interpolating additional frames" : "Upscaling extracted frames",
            processedFrames: interpolationEnabled ? 430 : 180,
            totalFrames: interpolationEnabled ? 750 : 300,
            extractedFrames: 300,
            upscaledFrames: interpolationEnabled ? 300 : 180,
            interpolatedFrames: interpolationEnabled ? 430 : 0,
            encodedFrames: 0,
            remuxedFrames: 0,
            segmentIndex: 1,
            segmentCount: 1,
            segmentProcessedFrames: interpolationEnabled ? 430 : 180,
            segmentTotalFrames: interpolationEnabled ? 750 : 300,
            batchIndex: interpolationEnabled ? null : 15,
            batchCount: interpolationEnabled ? null : 25,
            elapsedSeconds: 30,
            averageFramesPerSecond: interpolationEnabled ? 14.5 : 6,
            rollingFramesPerSecond: interpolationEnabled ? 16.2 : 7.5,
            estimatedRemainingSeconds: interpolationEnabled ? 22 : 20,
            processRssBytes: 1024 * 1024 * 512,
            gpuMemoryUsedBytes: 1024 * 1024 * 6144,
            gpuMemoryTotalBytes: 1024 * 1024 * 24576,
            scratchSizeBytes: 1024 * 1024 * 12,
            outputSizeBytes: 1024 * 1024 * 3,
            extractStageSeconds: 4,
            upscaleStageSeconds: interpolationEnabled ? 18 : 18,
            interpolateStageSeconds: interpolationEnabled ? 11 : 0,
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
            sourceMedia: {
              width: 1280,
              height: 720,
              frameRate: 24,
              durationSeconds: 12.5,
              frameCount: 300,
              aspectRatio: 16 / 9,
              pixelCount: 1280 * 720,
              hasAudio: true,
              container: "webm",
              videoCodec: "vp9"
            },
            outputMedia: {
              width: 3840,
              height: 2160,
              frameRate: interpolationEnabled ? 60 : 24,
              durationSeconds: 12.5,
              frameCount: interpolationEnabled ? 750 : 300,
              aspectRatio: 16 / 9,
              pixelCount: 3840 * 2160,
              hasAudio: true,
              container: request.container,
              videoCodec: request.codec
            },
            effectiveSettings: {
              backendId: request.modelId === "rvrt-x4" ? "pytorch-video-sr" : "pytorch-image-sr",
              qualityPreset: request.qualityPreset,
              requestedTileSize: request.tileSize,
              effectiveTileSize: request.tileSize,
              requestedPrecision: "fp16",
              selectedPrecision: "fp16",
              effectivePrecision: "fp16",
              precisionSource: "backend-default",
              processedDurationSeconds: request.previewDurationSeconds ?? 12.5,
              segmentFrameLimit: interpolationEnabled ? 48 : 0,
              previewMode: request.previewMode,
              previewDurationSeconds: request.previewDurationSeconds,
              segmentDurationSeconds: request.segmentDurationSeconds
            },
            executionPath: interpolationEnabled ? "streaming" : "file-io",
            videoEncoder: request.codec === "h265" ? "hevc_nvenc" : "h264_nvenc",
            videoEncoderLabel: request.codec === "h265" ? "NVIDIA NVENC H.265" : "NVIDIA NVENC H.264",
            runner: request.modelId === "rvrt-x4" ? "torch" : request.pytorchRunner,
            precision: "fp16",
            stageTimings: {
              extractSeconds: 4,
              upscaleSeconds: 18,
              interpolateSeconds: interpolationEnabled ? 11 : 0,
              encodeSeconds: 5,
              remuxSeconds: 3
            },
            resourcePeaks: {
              processRssBytes: 1024 * 1024 * 512,
              gpuMemoryUsedBytes: 1024 * 1024 * 6144,
              gpuMemoryTotalBytes: 1024 * 1024 * 24576,
              scratchSizeBytes: 1024 * 1024 * 12,
              outputSizeBytes: 1024 * 1024 * 3
            },
            averageThroughputFps: interpolationEnabled ? 14.5 : 6,
            segmentCount: interpolationEnabled ? 2 : null,
            segmentFrameLimit: interpolationEnabled ? 48 : null,
            runtime: {
              ffmpegPath: "C:/tools/ffmpeg.exe",
              realesrganPath: "C:/tools/realesrgan-ncnn-vulkan.exe",
              modelDir: "C:/tools/models",
              rifePath: "C:/tools/rife-ncnn-vulkan.exe",
              rifeModelRoot: "C:/tools/rife-models",
              availableGpus: [
                { id: 0, name: "Intel(R) Graphics", kind: "integrated" },
                { id: 1, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", kind: "discrete" }
              ],
              defaultGpuId: 1
            },
            interpolationDiagnostics: interpolationEnabled ? {
              mode: request.interpolationMode,
              sourceFps: 24,
              outputFps: 60,
              sourceFrameCount: 300,
              outputFrameCount: 750,
              segmentCount: 2,
              segmentFrameLimit: 48,
              segmentOverlapFrames: 1,
            } : null,
            log: [
              `Completed mock pipeline for ${request.modelId}`,
              `Average throughput: ${interpolationEnabled ? "14.50" : "6.00"} fps`,
              `Stage timings: extract 4s, upscale 18s, interpolate ${interpolationEnabled ? "11s" : "0s"}, encode 5s, remux 3s`,
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

        if (activeJob.state === "paused" || activeJob.state === "cancelled") {
          return activeJob;
        }

        return {
          ...activeJob,
          state: "succeeded",
          progress: {
            ...activeJob.progress,
            phase: "completed",
            percent: 100,
            message: "Pipeline completed",
            processedFrames: activeJob.progress.totalFrames,
            totalFrames: activeJob.progress.totalFrames,
            extractedFrames: 300,
            upscaledFrames: activeJob.progress.upscaledFrames === 0 ? 300 : activeJob.progress.upscaledFrames,
            interpolatedFrames: activeJob.progress.interpolatedFrames === 0 ? 0 : activeJob.progress.totalFrames,
            encodedFrames: activeJob.progress.totalFrames,
            remuxedFrames: activeJob.progress.totalFrames
          }
        };
      },
      async pausePipelineJob() {
        if (!activeJob) {
          return;
        }

        activeJob = {
          ...activeJob,
          state: "paused",
          progress: {
            ...activeJob.progress,
            phase: "paused",
            message: `Paused: ${activeJob.progress.message}`,
          }
        };
      },
      async resumePipelineJob() {
        if (!activeJob) {
          return;
        }

        activeJob = {
          ...activeJob,
          state: "running",
          progress: {
            ...activeJob.progress,
            phase: activeJob.progress.interpolatedFrames > 0 ? "interpolating" : "upscaling",
            message: activeJob.progress.interpolatedFrames > 0 ? "Resumed: interpolating additional frames" : "Resumed: upscaling extracted frames",
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
      async loadPreviewUrl(path) {
        previewLoadPaths.push(path);
        return `mock-preview://${encodeURIComponent(path)}`;
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
      container: "mp4",
      videoCodec: "h264"
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
  await expect(page.getByTestId("upscaler-section-card")).toContainText("Spatial detail restore");
  await expect(page.getByTestId("interpolator-section-card")).toContainText("Upsampling / interpolation");
  await page.getByTestId("pipeline-toggle-interpolation").click();
  await expect(page.getByTestId("frame-rate-workspace-section")).toContainText("Target Frame Rate");

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
  await expect(page.getByTestId("top-status-eta")).toContainText("Not running");
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
  await openUpscaleControls(page);
  await expect(page.locator('[data-testid="model-select"] optgroup[label="Available Now"] option')).toHaveCount(5);
  await expect(page.locator('[data-testid="model-select"] optgroup[label="Planned"] option')).toHaveCount(1);
  await expect(page.locator('[data-testid="model-select"] option[disabled]')).toHaveCount(1);
  await expect(page.locator('[data-testid="model-select"] option[value="hat-realhat-gan-x4"]')).toContainText("not implemented");
  await expect(page.locator('[data-testid="model-select"] option[value="rife-v4.6"]')).toHaveCount(0);
  await expect(page.getByTestId("target-model-set-card")).toHaveCount(0);
  await expect(page.getByTestId("selected-model-label")).toContainText("Real-ESRGAN x4 Plus");
  await expect(page.getByTestId("selected-model-summary")).toContainText("photographic");
  await page.getByTestId("model-details-card").locator("summary").click();
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
  await openInterpolationControls(page);
  await page.getByTestId("frame-rate-target-select").selectOption("60");
  await expect(page.getByTestId("interpolation-workspace-summary")).toContainText("Post-upscale interpolation");
  await page.getByTestId("preview-mode-checkbox").check();
  await page.getByTestId("preview-duration-input").fill("3");
  await page.getByTestId("save-output-button").click();
  await expect(page.getByTestId("output-path-input")).toHaveValue("C:/exports/upscaled-result.mkv");

  await page.evaluate(() => {
    const video = document.querySelector('[data-testid="source-preview"]');
    if (!(video instanceof HTMLVideoElement)) {
      throw new Error("Source preview video is unavailable for blind offset setup");
    }

    Object.defineProperty(video, "duration", {
      configurable: true,
      value: 12.5,
    });
    video.currentTime = 0;
    video.dispatchEvent(new Event("timeupdate", { bubbles: true }));
  });

  await page.getByTestId("blind-test-panel-toggle").click();
  await expect(page.getByTestId("blind-test-panel")).toContainText("5 available");
  await expect(page.getByTestId("blind-test-panel")).toContainText("4 selected");
  await expect(page.getByTestId("blind-model-option-rvrt-x4")).toContainText("RVRT x4");
  await page.getByTestId("source-preview-seek").evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new Error("Source preview seek slider is unavailable");
    }
    element.value = "2.2";
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.getByTestId("blind-capture-current-preview-position").click();
  await expect(page.getByTestId("blind-start-offset-readout")).toContainText("0:02");
  await page.getByTestId("run-blind-comparison-button").click();
  await expect(page.getByTestId("blind-preview-sample-1")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-2")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-3")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-4")).toBeVisible();
  await expect(page.getByTestId("comparison-inspector")).toBeVisible();
  const blindPreviewLoadsBeforeWorkspace = await page.evaluate(() => {
    const testState = window.__UPSCALER_TEST_STATE__;
    return testState.previewLoadPaths.filter((path) => String(path).includes("/blind/")).length;
  });
  await page.getByTestId("open-comparison-workspace-button").click();
  await expect(page.getByTestId("comparison-workspace-modal")).toBeVisible();
  await expect.poll(async () => {
    const { previewLoadPaths } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
    return previewLoadPaths.filter((path) => String(path).includes("/blind/")).length;
  }).toBeGreaterThan(blindPreviewLoadsBeforeWorkspace);
  await expect(page.getByTestId("comparison-workspace-modal")).toContainText("Source plus 4 blind samples");
  await page.getByTestId("comparison-focus-diagonals").click();
  await page.getByTestId("comparison-zoom-slider").fill("4");
  await expect(page.locator("[data-testid^='blind-reveal-']")).toHaveCount(0);
  await page.getByTestId("comparison-pick-sample-1").click();
  await expect(page.locator("[data-testid^='blind-reveal-']")).toHaveCount(4);
  await expect(page.getByTestId("comparison-workspace-reveal-sample-1")).toContainText("Selected winner");
  await page.getByTestId("comparison-workspace-close").click();

  const { pipelineRequests } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  const blindPreviewRequests = pipelineRequests.filter((request) => request.previewMode === true);
  expect(blindPreviewRequests).toHaveLength(4);
  expect(blindPreviewRequests.every((request) => request.container === "mp4")).toBe(true);
  expect(blindPreviewRequests.every((request) => request.codec === "h264")).toBe(true);
  expect(blindPreviewRequests.every((request) => request.outputPath.endsWith(".mp4"))).toBe(true);
  expect(blindPreviewRequests.every((request) => Number(request.previewStartOffsetSeconds) > 2)).toBe(true);

  await page.getByTestId("run-upscale-button").click();
  await page.getByTestId("job-cleanup-panel-toggle").click();
  await expect(page.getByTestId("job-progress-panel")).toBeVisible();
  await expect(page.getByTestId("progress-upscaled-frames")).toContainText("300");
  await expect(page.getByTestId("progress-interpolated-frames")).toContainText("750");
  await expect(page.getByTestId("progress-remuxed-frames")).toContainText("750");
  await expect(page.getByTestId("progress-segment-counter")).toContainText("1/1");
  await expect(page.getByTestId("progress-segment-frames")).toContainText("430/750");
  await expect(page.getByTestId("progress-average-fps")).toContainText("14.5 fps");
  await expect(page.getByTestId("progress-rolling-fps")).toContainText("16.2 fps");
  await expect(page.getByTestId("progress-eta")).toContainText("22s");
  await expect(page.getByTestId("progress-process-rss")).toContainText("512 MB");
  await expect(page.getByTestId("progress-gpu-memory")).toContainText("6.0 GB / 24 GB");
  await expect(page.getByTestId("progress-stage-timings")).toContainText("extract 4s");
  await expect(page.getByTestId("progress-stage-timings")).toContainText("interpolate 11s");
  await expect(page.getByTestId("progress-current-activity")).toContainText("Pipeline completed");
  await expect(page.getByTestId("progress-current-detail")).toContainText("segment 1/1");
  await expect(page.getByTestId("progress-last-update")).toContainText("Last update");
  await expect(page.getByTestId("progress-event-log")).toContainText("Pipeline completed");
  await expect(page.getByTestId("phase-progress-interpolate")).toContainText("750/750");
  await expect(page.getByTestId("result-output-path")).toContainText("C:/exports/upscaled-result.mkv");
  await expect(page.getByTestId("interpolation-diagnostics-details")).not.toHaveAttribute("open", "");
  await page.getByTestId("interpolation-diagnostics-summary").click();
  await expect(page.getByTestId("interpolation-diagnostics-segment-count")).toContainText("2");
  await expect(page.getByTestId("interpolation-diagnostics-segment-limit")).toContainText("48");
  await expect(page.getByTestId("interpolation-diagnostics-overlap")).toContainText("1 frame");
  await expect(page.getByTestId("pipeline-log")).toContainText("Completed mock pipeline for realesrgan-x4plus");
  await expect(page.getByTestId("pipeline-log")).toContainText("Stage timings: extract 4s, upscale 18s, interpolate 11s, encode 5s, remux 3s");
  await page.getByTestId("result-output-details").locator("summary").click();
  await expect(page.getByTestId("result-output-details")).toContainText("Original audio remuxed");
  await expect(page.getByTestId("result-average-throughput")).toContainText("14.5 fps");
  await expect(page.getByTestId("result-effective-pixels-per-second")).toContainText("MP/s");
  await expect(page.getByTestId("result-stage-timings")).toContainText("extract 4s");
  await expect(page.getByTestId("result-stage-timings")).toContainText("interpolate 11s");
  await expect(page.getByTestId("result-output-details")).toContainText("Video Encoder");
  await expect(page.getByTestId("result-output-details")).toContainText("NVIDIA NVENC H.265");
  await expect(page.getByTestId("result-output-details")).toContainText("Requested Tile Size");
  await expect(page.getByTestId("result-output-details")).toContainText("Effective Tile Size");
  await expect(page.getByTestId("result-preview")).toBeVisible();
  await page.locator(".pipeline-runtime-disclosure").locator("summary").click();
  await expect(page.locator(".pipeline-runtime-disclosure")).toContainText("Blind Picks Logged");

  const { lastRequest: workflowRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(workflowRequest?.interpolationMode).toBe("afterUpscale");
  expect(workflowRequest?.interpolationTargetFps).toBe(60);
});

test("captures a blind comparison start offset and forwards it through preview runs", async ({ page }) => {
  await page.goto("/");

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
  await openUpscaleControls(page);
  await page.getByTestId("gpu-select").selectOption("0");
  await page.getByTestId("preview-mode-checkbox").check();
  await page.getByTestId("preview-duration-input").fill("3");
  await page.getByTestId("blind-test-panel-toggle").click();
  await expect(page.getByTestId("blind-test-panel")).toContainText("4 selected");

  await page.evaluate(() => {
    const video = document.querySelector('[data-testid="source-preview"]');
    if (!(video instanceof HTMLVideoElement)) {
      throw new Error("Source preview video is unavailable for blind offset setup");
    }

    video.dataset.mockDuration = "12.5";
    video.currentTime = 0;
    video.dispatchEvent(new Event("timeupdate", { bubbles: true }));
  });

  await page.getByTestId("source-preview-seek").evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new Error("Source preview seek slider is unavailable");
    }
    element.value = "2.2";
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });

  await page.getByTestId("blind-capture-current-preview-position").click();
  await expect(page.getByTestId("blind-start-offset-readout")).toContainText("0:02");

  await page.getByTestId("run-blind-comparison-button").click();
  await expect(page.getByTestId("blind-preview-sample-1")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-2")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-3")).toBeVisible();
  await expect(page.getByTestId("blind-preview-sample-4")).toBeVisible();
  await expect(page.getByTestId("comparison-inspector")).toBeVisible();

  const { pipelineRequests } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  const blindPreviewRequests = pipelineRequests.filter((request) => request.previewMode === true);
  expect(blindPreviewRequests).toHaveLength(4);
  expect(blindPreviewRequests.every((request) => Number(request.previewStartOffsetSeconds) > 2)).toBe(true);

  await page.getByTestId("open-comparison-workspace-button").click();
  await expect(page.getByTestId("comparison-workspace-modal")).toBeVisible();
  const comparisonSourceViewport = page.getByTestId("comparison-source-viewport");
  const comparisonSourceVideo = comparisonSourceViewport.locator("video");
  const initialTransformOrigin = await comparisonSourceVideo.evaluate((element) => {
    if (!(element instanceof HTMLVideoElement)) {
      throw new Error("Comparison source video is unavailable");
    }
    return element.style.transformOrigin;
  });
  const initialViewportBounds = await comparisonSourceViewport.boundingBox();
  if (!initialViewportBounds) {
    throw new Error("Comparison source viewport bounds are unavailable before wheel zoom");
  }
  await comparisonSourceViewport.click({ position: { x: 80, y: 80 } });
  await expect.poll(async () => await comparisonSourceVideo.evaluate((element) => {
    if (!(element instanceof HTMLVideoElement)) {
      return null;
    }
    return element.style.transformOrigin;
  })).toBe(initialTransformOrigin);
  await comparisonSourceViewport.evaluate((element) => {
    element.dispatchEvent(new WheelEvent("wheel", {
      deltaY: -120,
      bubbles: true,
      cancelable: true,
    }));
  });
  await expect(page.getByTestId("comparison-pane-zoom-readout")).toContainText("1.00x");
  await comparisonSourceViewport.evaluate((element) => {
    element.dispatchEvent(new WheelEvent("wheel", {
      deltaY: -120,
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    }));
  });
  await expect(page.getByTestId("comparison-pane-zoom-readout")).not.toContainText("1.00x");
  await expect(page.getByTestId("comparison-zoom-readout")).toContainText("3.00x");
  const comparisonWorkspaceShell = page.getByTestId("comparison-workspace-modal").locator(".comparison-workspace-shell");
  await expect.poll(async () => {
    const viewportBounds = await comparisonSourceViewport.boundingBox();
    const shellBounds = await comparisonWorkspaceShell.boundingBox();
    if (!viewportBounds || !shellBounds) {
      return 0;
    }
    return viewportBounds.width / shellBounds.width;
  }).toBeGreaterThan(0.48);
  await expect.poll(async () => await comparisonSourceVideo.evaluate((element) => {
    if (!(element instanceof HTMLVideoElement)) {
      return null;
    }
    return element.style.transformOrigin;
  })).toBe(initialTransformOrigin);
  await comparisonSourceViewport.evaluate((element) => {
    element.dispatchEvent(new WheelEvent("wheel", {
      deltaY: -120,
      ctrlKey: true,
      bubbles: true,
      cancelable: true,
    }));
  });
  await expect(page.getByTestId("comparison-zoom-readout")).not.toContainText("3.00x");
  await expect.poll(async () => await comparisonSourceVideo.evaluate((element) => {
    if (!(element instanceof HTMLVideoElement)) {
      return null;
    }
    return element.style.transformOrigin;
  })).toBe(initialTransformOrigin);
  const viewportBounds = await comparisonSourceViewport.boundingBox();
  if (!viewportBounds) {
    throw new Error("Comparison source viewport bounds are unavailable");
  }
  await page.mouse.move(viewportBounds.x + (viewportBounds.width / 2), viewportBounds.y + (viewportBounds.height / 2));
  await page.mouse.down();
  await page.mouse.move(viewportBounds.x + (viewportBounds.width / 2) + 48, viewportBounds.y + (viewportBounds.height / 2) + 32, { steps: 6 });
  await page.mouse.up();
  await expect.poll(async () => await comparisonSourceVideo.evaluate((element) => {
    if (!(element instanceof HTMLVideoElement)) {
      return null;
    }
    return element.style.transformOrigin;
  })).not.toBe(initialTransformOrigin);
  const initialSyncedTimes = await page.evaluate(() => {
    const sourceVideo = document.querySelector('[data-testid="comparison-source-viewport"] video');
    const sampleVideo = document.querySelector('[data-testid="comparison-sample-viewport-sample-1"] video');
    if (!(sourceVideo instanceof HTMLVideoElement) || !(sampleVideo instanceof HTMLVideoElement)) {
      throw new Error("Comparison workspace videos are unavailable for initial sync validation");
    }

    return {
      sourceCurrentTime: sourceVideo.currentTime,
      sampleCurrentTime: sampleVideo.currentTime,
    };
  });

  expect(Math.abs(
    (initialSyncedTimes.sourceCurrentTime - initialSyncedTimes.sampleCurrentTime)
    - Number(blindPreviewRequests[0]?.previewStartOffsetSeconds ?? 0),
  )).toBeLessThan(0.1);
  await page.getByTestId("comparison-time-slider").evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new Error("Comparison timeline slider is unavailable");
    }

    element.value = "24";
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });

  await expect.poll(async () => {
    return await page.evaluate(() => {
      const sourceVideo = document.querySelector('[data-testid="comparison-source-viewport"] video');
      const sampleVideo = document.querySelector('[data-testid="comparison-sample-viewport-sample-1"] video');
      if (!(sourceVideo instanceof HTMLVideoElement) || !(sampleVideo instanceof HTMLVideoElement)) {
        return null;
      }

      return {
        sourceCurrentTime: sourceVideo.currentTime,
        sampleCurrentTime: sampleVideo.currentTime,
      };
    });
  }).not.toBeNull();

  const syncedTimelineTimes = await page.evaluate(() => {
    const sourceVideo = document.querySelector('[data-testid="comparison-source-viewport"] video');
    const sampleVideo = document.querySelector('[data-testid="comparison-sample-viewport-sample-1"] video');
    if (!(sourceVideo instanceof HTMLVideoElement) || !(sampleVideo instanceof HTMLVideoElement)) {
      throw new Error("Comparison workspace videos are unavailable");
    }

    return {
      sourceCurrentTime: sourceVideo.currentTime,
      sampleCurrentTime: sampleVideo.currentTime,
    };
  });

  expect(Math.abs(
    (syncedTimelineTimes.sourceCurrentTime - syncedTimelineTimes.sampleCurrentTime)
    - Number(blindPreviewRequests[0]?.previewStartOffsetSeconds ?? 0),
  )).toBeLessThan(0.1);
  await page.getByTestId("comparison-pick-sample-1").click();

  const appConfig = await page.evaluate(() => window.__UPSCALER_MOCK__.getAppConfig());
  expect(appConfig.blindComparisons).toHaveLength(1);
  expect(Number(appConfig.blindComparisons[0]?.previewStartOffsetSeconds)).toBeGreaterThan(2);
});

test("shows the PyTorch runner selector only for PyTorch image SR models and passes the selection through", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("select-video-button").click();
  await openUpscaleControls(page);
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
      container: "mp4",
      videoCodec: "h264"
    });
  });

  await page.getByTestId("select-video-button").click();
  await openInterpolationControls(page);
  await page.getByTestId("pipeline-toggle-upscale").click();
  await page.getByTestId("frame-rate-target-select").selectOption("30");
  await page.getByTestId("run-upscale-button").click();

  const { confirmMessages, lastRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(confirmMessages).toHaveLength(1);
  expect(confirmMessages[0]).toContain("selected interpolation target of 30 fps is not higher");
  expect(lastRequest?.interpolationMode).toBe("interpolateOnly");
  expect(lastRequest?.interpolationTargetFps).toBe(30);
});

test("keeps export format controls available for interpolation-only jobs and matches supported input settings", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.probeSourceVideo = async (sourcePath) => ({
      path: sourcePath,
      previewPath: "C:/fixtures/sample-input-preview.mp4",
      width: 1920,
      height: 1080,
      durationSeconds: 18,
      frameRate: 24,
      hasAudio: true,
      container: "mkv",
      videoCodec: "hevc"
    });
  });

  await page.getByTestId("select-video-button").click();
  await openInterpolationControls(page);
  await page.getByTestId("pipeline-toggle-upscale").click();

  await expect(page.getByTestId("pipeline-export-settings")).toBeVisible();
  await expect(page.getByTestId("match-input-format-summary")).toContainText("HEVC in MKV");

  await page.getByTestId("codec-select").selectOption("h264");
  await page.getByTestId("container-select").selectOption("mp4");
  await page.getByTestId("match-input-format-button").click();

  await expect(page.getByTestId("codec-select")).toHaveValue("h265");
  await expect(page.getByTestId("container-select")).toHaveValue("mkv");

  await page.getByTestId("run-upscale-button").click();

  const { lastRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(lastRequest?.interpolationMode).toBe("interpolateOnly");
  expect(lastRequest?.codec).toBe("h265");
  expect(lastRequest?.container).toBe("mkv");
});

test("shows overlapped interpolation and encoding progress in the job panel", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.startPipeline = async (request) => {
      window.__UPSCALER_TEST_STATE__.lastRequest = request;
      return "mock-overlap-job";
    };

    window.__UPSCALER_MOCK__.getPipelineJob = async () => ({
      jobId: "mock-overlap-job",
      state: "running",
      progress: {
        phase: "encoding",
        percent: 76,
        message: "Encoding segment 1 while the next segment is already interpolated",
        processedFrames: 180,
        totalFrames: 750,
        extractedFrames: 300,
        upscaledFrames: 300,
        interpolatedFrames: 540,
        encodedFrames: 180,
        remuxedFrames: 0,
        segmentIndex: 1,
        segmentCount: 2,
        segmentProcessedFrames: 180,
        segmentTotalFrames: 600,
        batchIndex: null,
        batchCount: null,
        elapsedSeconds: 34,
        averageFramesPerSecond: 13.8,
        rollingFramesPerSecond: 15.4,
        estimatedRemainingSeconds: 12,
        processRssBytes: 1024 * 1024 * 640,
        gpuMemoryUsedBytes: 1024 * 1024 * 7168,
        gpuMemoryTotalBytes: 1024 * 1024 * 24576,
        scratchSizeBytes: 1024 * 1024 * 16,
        outputSizeBytes: 1024 * 1024 * 5,
        extractStageSeconds: 4,
        upscaleStageSeconds: 18,
        interpolateStageSeconds: 12,
        encodeStageSeconds: 6,
        remuxStageSeconds: 0,
      },
      result: null,
      error: null,
    });
  });

  await page.getByTestId("select-video-button").click();
  await openInterpolationControls(page);
  await page.getByTestId("frame-rate-target-select").selectOption("60");
  await page.getByTestId("run-upscale-button").click();
  await page.getByTestId("job-cleanup-panel-toggle").click();

  await expect(page.getByTestId("job-progress-panel")).toBeVisible();
  await expect(page.getByTestId("progress-message")).toContainText("next segment is already interpolated");
  await expect(page.getByTestId("progress-current-activity")).toContainText("Encoding");
  await expect(page.getByTestId("progress-current-detail")).toContainText("segment 1/2");
  await expect(page.getByTestId("progress-segment-counter")).toContainText("1/2");
  await expect(page.getByTestId("progress-segment-frames")).toContainText("180/600");
  await expect(page.getByTestId("progress-interpolated-frames")).toContainText("540");
  await expect(page.getByTestId("progress-encoded-frames")).toContainText("180");
  await expect(page.getByTestId("phase-progress-interpolate")).toContainText("540/750");
  await expect(page.getByTestId("phase-progress-encode")).toContainText("180/750");
  await expect(page.getByTestId("progress-average-fps")).toContainText("13.8 fps");
  await expect(page.getByTestId("progress-rolling-fps")).toContainText("15.4 fps");
  await expect(page.getByTestId("progress-eta")).toContainText("12s");
  await expect(page.getByTestId("top-status-panel")).toContainText("Pipeline Running");
  await expect(page.getByTestId("top-status-panel")).toContainText("ETA 12s");

  const { lastRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(lastRequest?.interpolationMode).toBe("afterUpscale");
  expect(lastRequest?.interpolationTargetFps).toBe(60);
});

test("refreshes detailed live job progress inside a running segment", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    let pollCount = 0;
    window.__UPSCALER_MOCK__.startPipeline = async (request) => {
      window.__UPSCALER_TEST_STATE__.lastRequest = request;
      return "mock-live-progress-job";
    };

    window.__UPSCALER_MOCK__.getPipelineJob = async () => {
      pollCount += 1;
      const progressed = pollCount >= 3;
      return {
        jobId: "mock-live-progress-job",
        state: "running",
        progress: {
          phase: "upscaling",
          percent: progressed ? 44 : 31,
          message: progressed ? "Upscaling segment 2/4 batch 4/8 (96/300 frames)" : "Upscaling segment 2/4 batch 2/8 (32/300 frames)",
          processedFrames: progressed ? 196 : 132,
          totalFrames: 900,
          extractedFrames: 300,
          upscaledFrames: progressed ? 196 : 132,
          interpolatedFrames: 0,
          encodedFrames: 0,
          remuxedFrames: 0,
          segmentIndex: 2,
          segmentCount: 4,
          segmentProcessedFrames: progressed ? 96 : 32,
          segmentTotalFrames: 300,
          batchIndex: progressed ? 4 : 2,
          batchCount: 8,
          elapsedSeconds: progressed ? 21 : 14,
          averageFramesPerSecond: progressed ? 9.7 : 8.1,
          rollingFramesPerSecond: progressed ? 12.4 : 8.8,
          estimatedRemainingSeconds: progressed ? 28 : 45,
          processRssBytes: 1024 * 1024 * 700,
          gpuMemoryUsedBytes: 1024 * 1024 * 6144,
          gpuMemoryTotalBytes: 1024 * 1024 * 24576,
          scratchSizeBytes: 1024 * 1024 * 18,
          outputSizeBytes: 1024 * 1024 * 2,
          extractStageSeconds: 4,
          upscaleStageSeconds: progressed ? 17 : 10,
          interpolateStageSeconds: 0,
          encodeStageSeconds: 0,
          remuxStageSeconds: 0,
        },
        result: null,
        error: null,
      };
    };
  });

  await page.getByTestId("select-video-button").click();
  await page.getByTestId("run-upscale-button").click();
  await page.getByTestId("job-cleanup-panel-toggle").click();

  await expect(page.getByTestId("job-progress-panel")).toBeVisible();
  await expect(page.getByTestId("progress-segment-frames")).toContainText("32/300");
  await expect(page.getByTestId("progress-batch-counter")).toContainText("2/8");
  await expect(page.getByTestId("progress-eta")).toContainText("45s");

  await expect(page.getByTestId("progress-segment-frames")).toContainText("96/300");
  await expect(page.getByTestId("progress-batch-counter")).toContainText("4/8");
  await expect(page.getByTestId("progress-rolling-fps")).toContainText("12.4 fps");
  await expect(page.getByTestId("progress-average-fps")).toContainText("9.70 fps");
  await expect(page.getByTestId("progress-eta")).toContainText("28s");
});

test("surfaces active pipeline status before the jobs panel is opened", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.startPipeline = async (request) => {
      window.__UPSCALER_TEST_STATE__.lastRequest = request;
      return "mock-status-first-job";
    };

    window.__UPSCALER_MOCK__.getPipelineJob = async () => ({
      jobId: "mock-status-first-job",
      state: "running",
      progress: {
        phase: "upscaling",
        percent: 37,
        message: "Upscaling segment 1/1 (144/390 frames)",
        processedFrames: 144,
        totalFrames: 390,
        extractedFrames: 390,
        upscaledFrames: 144,
        interpolatedFrames: 0,
        encodedFrames: 0,
        remuxedFrames: 0,
        segmentIndex: 1,
        segmentCount: 1,
        segmentProcessedFrames: 144,
        segmentTotalFrames: 390,
        batchIndex: 3,
        batchCount: 8,
        elapsedSeconds: 19,
        averageFramesPerSecond: 7.6,
        rollingFramesPerSecond: 8.4,
        estimatedRemainingSeconds: 32,
        processRssBytes: 1024 * 1024 * 688,
        gpuMemoryUsedBytes: 1024 * 1024 * 6144,
        gpuMemoryTotalBytes: 1024 * 1024 * 24576,
        scratchSizeBytes: 1024 * 1024 * 14,
        outputSizeBytes: 0,
        extractStageSeconds: 3,
        upscaleStageSeconds: 16,
        interpolateStageSeconds: 0,
        encodeStageSeconds: 0,
        remuxStageSeconds: 0,
      },
      result: null,
      error: null,
    });
  });

  await page.getByTestId("select-video-button").click();
  await page.getByTestId("run-upscale-button").click();

  await expect(page.getByTestId("pipeline-launch-state")).toHaveAttribute("data-state", /starting|queued|running/);
  await expect(page.getByTestId("top-status-panel")).toContainText("Pipeline Running");
  await expect(page.getByTestId("top-status-panel")).toContainText("37% complete");
  await expect(page.getByTestId("top-status-panel")).toContainText("ETA 32s");
  await expect(page.getByTestId("top-status-panel")).toContainText("Upscaling segment 1/1 (144/390 frames)");
  await expect(page.getByTestId("job-progress-panel")).toHaveCount(0);

  await page.getByTestId("job-cleanup-panel-toggle").click();
  await expect(page.getByTestId("job-progress-panel")).toBeVisible();
  await expect(page.getByTestId("progress-message")).toContainText("Upscaling segment 1/1 (144/390 frames)");
  await expect(page.getByTestId("progress-segment-frames")).toContainText("144/390");
});

test("shows a starting marker before the first worker poll resolves", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.startPipeline = async (request) => {
      window.__UPSCALER_TEST_STATE__.lastRequest = request;
      await new Promise((resolve) => window.setTimeout(resolve, 250));
      return "mock-launch-delay-job";
    };

    window.__UPSCALER_MOCK__.getPipelineJob = async () => ({
      jobId: "mock-launch-delay-job",
      state: "running",
      progress: {
        phase: "upscaling",
        percent: 12,
        message: "Upscaling segment 1/1 (48/390 frames)",
        processedFrames: 48,
        totalFrames: 390,
        extractedFrames: 390,
        upscaledFrames: 48,
        interpolatedFrames: 0,
        encodedFrames: 0,
        remuxedFrames: 0,
        segmentIndex: 1,
        segmentCount: 1,
        segmentProcessedFrames: 48,
        segmentTotalFrames: 390,
        batchIndex: 1,
        batchCount: 8,
        elapsedSeconds: 6,
        averageFramesPerSecond: 8.0,
        rollingFramesPerSecond: 8.6,
        estimatedRemainingSeconds: 42,
        processRssBytes: 1024 * 1024 * 640,
        gpuMemoryUsedBytes: 1024 * 1024 * 4096,
        gpuMemoryTotalBytes: 1024 * 1024 * 24576,
        scratchSizeBytes: 1024 * 1024 * 10,
        outputSizeBytes: 0,
        extractStageSeconds: 2,
        upscaleStageSeconds: 4,
        interpolateStageSeconds: 0,
        encodeStageSeconds: 0,
        remuxStageSeconds: 0,
      },
      result: null,
      error: null,
    });
  });

  await page.getByTestId("select-video-button").click();
  await page.getByTestId("run-upscale-button").click();

  await expect(page.getByTestId("pipeline-launch-state")).toHaveAttribute("data-state", "starting");
  await expect(page.getByTestId("top-status-panel")).toContainText("Pipeline Starting");
  await expect(page.getByTestId("pipeline-launch-state")).toContainText("Pipeline launch launching");
  await expect(page.getByTestId("pipeline-launch-state")).toHaveAttribute("data-state", /queued|running/);
});

test("surfaces recovered running managed jobs in the main status panel when no live in-memory job exists", async ({ page }) => {
  await page.addInitScript(() => {
    const recoveredManagedJobs = [{
      jobId: "recovered-managed-job",
      jobKind: "pipeline",
      label: "Recovered Running Job",
      state: "running",
      sourcePath: "C:/workspace/fixtures/recovered-input.mov",
      modelId: "rvrt-x4",
      codec: "h265",
      container: "mp4",
      progress: {
        phase: "upscaling",
        percent: 41,
        message: "Recovered running worker process detected outside the app state.",
        processedFrames: 0,
        totalFrames: 0,
        extractedFrames: 0,
        upscaledFrames: 0,
        interpolatedFrames: 0,
        encodedFrames: 0,
        remuxedFrames: 0,
        elapsedSeconds: 52,
        averageFramesPerSecond: 0,
        rollingFramesPerSecond: 0,
        estimatedRemainingSeconds: 18,
        processRssBytes: 1024 * 1024 * 512,
        gpuMemoryUsedBytes: 1024 * 1024 * 4096,
        gpuMemoryTotalBytes: 1024 * 1024 * 24576,
        scratchSizeBytes: 1024 * 1024 * 12,
        outputSizeBytes: 0,
        extractStageSeconds: 0,
        upscaleStageSeconds: 44,
        interpolateStageSeconds: 0,
        encodeStageSeconds: 0,
        remuxStageSeconds: 0,
      },
      recordedCount: 0,
      scratchPath: "C:/workspace/artifacts/jobs/job_recovered-managed-job",
      scratchStats: {
        path: "C:/workspace/artifacts/jobs/job_recovered-managed-job",
        exists: true,
        isDirectory: true,
        sizeBytes: 1024 * 1024 * 12,
      },
      outputPath: "C:/workspace/artifacts/outputs/recovered-output.mp4",
      outputStats: {
        path: "C:/workspace/artifacts/outputs/recovered-output.mp4",
        exists: false,
        isDirectory: false,
        sizeBytes: 0,
      },
      updatedAt: String(Math.floor(Date.now() / 1000)),
    }];

    const installRecoveredJobsMock = () => {
      if (!window.__UPSCALER_MOCK__) {
        return;
      }
      window.__UPSCALER_MOCK__.listManagedJobs = async () => recoveredManagedJobs;
    };

    installRecoveredJobsMock();
    window.addEventListener("load", installRecoveredJobsMock, { once: true });
  });

  await page.goto("/");

  await expect(page.getByTestId("top-status-panel")).toContainText("Recovered Running Job");
  await expect(page.getByTestId("top-status-panel")).toContainText("41% complete");
  await expect(page.getByTestId("top-status-panel")).toContainText("ETA 18s");
  await expect(page.getByTestId("top-status-panel")).toContainText("Recovered running worker process detected outside the app state.");

  await page.getByTestId("job-cleanup-panel-toggle").click();
  await expect(page.getByTestId("cleanup-job-recovered-managed-job")).toContainText("Recovered Running Job");
  await expect(page.getByTestId("cleanup-job-recovered-managed-job")).toContainText("running");
});

test("keeps live job polling fast without hammering managed inventory during an active run", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__ || !window.__UPSCALER_TEST_STATE__) {
      return;
    }

    const pollingState = {
      pipelineCalls: 0,
      managedJobCalls: 0,
      scratchSummaryCalls: 0,
      pathStatsCalls: 0,
    };
    window.__UPSCALER_TEST_STATE__.pollingState = pollingState;

    const originalGetPathStats = window.__UPSCALER_MOCK__.getPathStats?.bind(window.__UPSCALER_MOCK__);
    const originalGetScratchStorageSummary = window.__UPSCALER_MOCK__.getScratchStorageSummary?.bind(window.__UPSCALER_MOCK__);
    const originalListManagedJobs = window.__UPSCALER_MOCK__.listManagedJobs?.bind(window.__UPSCALER_MOCK__);

    window.__UPSCALER_MOCK__.startPipeline = async (request) => {
      window.__UPSCALER_TEST_STATE__.lastRequest = request;
      return "managed-poll-pressure-job";
    };

    window.__UPSCALER_MOCK__.getPipelineJob = async () => {
      pollingState.pipelineCalls += 1;
      const processedFrames = Math.min(300, 60 + pollingState.pipelineCalls * 18);
      return {
        jobId: "managed-poll-pressure-job",
        state: "running",
        progress: {
          phase: "upscaling",
          percent: Math.min(95, 20 + pollingState.pipelineCalls * 4),
          message: `Upscaling active batch ${pollingState.pipelineCalls}`,
          processedFrames,
          totalFrames: 300,
          extractedFrames: 300,
          upscaledFrames: processedFrames,
          interpolatedFrames: 0,
          encodedFrames: 0,
          remuxedFrames: 0,
          segmentIndex: 1,
          segmentCount: 1,
          segmentProcessedFrames: processedFrames,
          segmentTotalFrames: 300,
          batchIndex: Math.min(25, pollingState.pipelineCalls),
          batchCount: 25,
          elapsedSeconds: pollingState.pipelineCalls,
          averageFramesPerSecond: 11.2,
          rollingFramesPerSecond: 12.8,
          estimatedRemainingSeconds: 18,
          processRssBytes: 1024 * 1024 * 640,
          gpuMemoryUsedBytes: 1024 * 1024 * 4096,
          gpuMemoryTotalBytes: 1024 * 1024 * 24576,
          scratchSizeBytes: 1024 * 1024 * 10,
          outputSizeBytes: 1024 * 1024 * 2,
          extractStageSeconds: 3,
          upscaleStageSeconds: pollingState.pipelineCalls,
          interpolateStageSeconds: 0,
          encodeStageSeconds: 0,
          remuxStageSeconds: 0,
        },
        result: null,
        error: null,
      };
    };

    window.__UPSCALER_MOCK__.listManagedJobs = async () => {
      pollingState.managedJobCalls += 1;
      return originalListManagedJobs ? originalListManagedJobs() : [];
    };

    window.__UPSCALER_MOCK__.getScratchStorageSummary = async () => {
      pollingState.scratchSummaryCalls += 1;
      return originalGetScratchStorageSummary
        ? originalGetScratchStorageSummary()
        : {
            jobsRoot: { path: "C:/workspace/artifacts/jobs", exists: true, isDirectory: true, sizeBytes: 0 },
            convertedSourcesRoot: { path: "C:/workspace/artifacts/runtime/converted-sources", exists: true, isDirectory: true, sizeBytes: 0 },
            sourcePreviewsRoot: { path: "C:/workspace/artifacts/runtime/source-previews", exists: true, isDirectory: true, sizeBytes: 0 },
          };
    };

    window.__UPSCALER_MOCK__.getPathStats = async (path) => {
      pollingState.pathStatsCalls += 1;
      return originalGetPathStats
        ? originalGetPathStats(path)
        : {
            path,
            exists: true,
            isDirectory: false,
            sizeBytes: 1024,
          };
    };
  });

  await page.getByTestId("select-video-button").click();
  await page.getByTestId("run-upscale-button").click();
  await page.getByTestId("job-cleanup-panel-toggle").click();

  await expect(page.getByTestId("job-progress-panel")).toBeVisible();
  await page.waitForTimeout(1600);

  const pollingState = await page.evaluate(() => window.__UPSCALER_TEST_STATE__.pollingState);
  expect(pollingState.pipelineCalls).toBeGreaterThanOrEqual(5);
  expect(pollingState.managedJobCalls).toBeLessThanOrEqual(2);
  expect(pollingState.scratchSummaryCalls).toBeLessThanOrEqual(1);
  expect(pollingState.pathStatsCalls).toBeLessThanOrEqual(3);
});

test("surfaces RVRT as setup-required when its external runner is not configured", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("select-video-button").click();
  await openUpscaleControls(page);
  await expect(page.locator('[data-testid="model-select"] option[value="rvrt-x4"]')).toHaveJSProperty("disabled", false);

  await page.getByTestId("model-select").selectOption("rvrt-x4");

  await expect(page.getByTestId("selected-model-label")).toContainText("RVRT x4");
  await expect(page.getByTestId("selected-model-status")).toContainText("setup required");
  await expect(page.getByTestId("model-details-card")).toContainText("research");
  await expect(page.getByTestId("selected-model-availability")).toContainText("UPSCALER_RVRT_COMMAND");
  await expect(page.getByTestId("run-upscale-button")).toBeDisabled();
  await expect(page.getByTestId("run-disabled-reason")).toContainText("UPSCALER_RVRT_COMMAND");
});

test("allows launching RVRT when its external runner is configured", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.ensureRuntimeAssets = async () => ({
      ffmpegPath: "C:/tools/ffmpeg.exe",
      realesrganPath: "C:/tools/realesrgan-ncnn-vulkan.exe",
      modelDir: "C:/tools/models",
      rifePath: "C:/tools/rife-ncnn-vulkan.exe",
      rifeModelRoot: "C:/tools/rife-models",
      availableGpus: [
        { id: 0, name: "Intel(R) Graphics", kind: "integrated" },
        { id: 1, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", kind: "discrete" }
      ],
      defaultGpuId: 1,
      externalResearchRuntimes: {
        "rvrt-x4": {
          kind: "external-command",
          commandEnvVar: "UPSCALER_RVRT_COMMAND",
          configured: true
        }
      }
    });
  });

  await page.getByTestId("select-video-button").click();
  await openUpscaleControls(page);
  await page.getByTestId("model-select").selectOption("rvrt-x4");

  await expect(page.getByTestId("selected-model-label")).toContainText("RVRT x4");
  await expect(page.getByTestId("selected-model-status")).toContainText("runnable");
  await expect(page.getByTestId("run-upscale-button")).toBeEnabled();

  await page.getByTestId("run-upscale-button").click();
  await expect.poll(async () => page.evaluate(() => window.__UPSCALER_TEST_STATE__.lastRequest?.modelId ?? null)).toBe("rvrt-x4");
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
      container: "avi",
      videoCodec: "mpeg4"
    });
  });

  await page.getByTestId("select-video-button").click();
  await expect(page.getByText("C:/fixtures/sample-input.avi")).toBeVisible();
  await expect(page.getByTestId("source-preview-mode")).toContainText("Full-length converted preview");
  await expect(page.getByTestId("source-preview-guidance")).toContainText("full-length converted preview");
  await expect(page.getByTestId("run-upscale-button")).toBeEnabled();
  await expect(page.getByTestId("convert-source-to-mp4-button")).toBeVisible();

  await page.getByTestId("convert-source-to-mp4-button").click();
  await page.getByTestId("job-cleanup-panel-toggle").click();
  await expect(page.getByTestId("conversion-progress-panel")).toBeVisible();
  await expect(page.getByText("C:/fixtures/sample-input_fastprep.mp4")).toBeVisible();
  await expect(page.getByText("Source converted to MP4.")).toBeVisible();
});

test("replaces an avi source with the converted mp4 and clears stale comparison state", async ({ page }) => {
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
      container: "avi",
      videoCodec: "mpeg4"
    });
  });

  await page.getByTestId("select-video-button").click();
  await expect(page.getByText("C:/fixtures/sample-input.avi")).toBeVisible();
  await expect(page.getByTestId("source-preview-mode")).toContainText("Full-length converted preview");
  await openUpscaleControls(page);
  await page.getByTestId("gpu-select").selectOption("0");

  await page.getByTestId("blind-test-panel-toggle").click();
  await page.getByTestId("run-blind-comparison-button").click();
  await expect(page.getByTestId("comparison-inspector")).toBeVisible();
  await expect(page.getByTestId("comparison-ready-note")).toContainText("4 of 4 samples are ready");

  await page.getByTestId("convert-source-to-mp4-button").click();
  await expect(page.getByText("Source converted to MP4.")).toBeVisible();
  await expect(page.getByText("C:/fixtures/sample-input_fastprep.mp4")).toBeVisible();
  await expect(page.getByText("C:/fixtures/sample-input.avi")).toBeHidden();
  await expect(page.getByTestId("source-preview-mode")).toContainText("Direct source playback");
  await expect(page.getByTestId("source-preview-guidance")).toHaveCount(0);
  await expect(page.getByTestId("convert-source-to-mp4-button")).toHaveCount(0);
  await expect(page.getByTestId("comparison-inspector")).toHaveCount(0);
  await expect(page.getByTestId("comparison-workspace-modal")).toHaveCount(0);

  await page.getByTestId("run-upscale-button").click();
  const { lastRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(lastRequest?.sourcePath).toBe("C:/fixtures/sample-input_fastprep.mp4");
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
      container: "webm",
      videoCodec: "vp9"
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
  await expect(page.getByTestId("cleanup-sort-scratch-size")).toContainText("↓");
  await expect(page.getByTestId("cleanup-sort-scratch-size")).toHaveClass(/cleanup-sort-button-active/);
  await expect(page.getByTestId("cleanup-jobs-table")).toBeVisible();
  await expect(page.getByTestId("cleanup-job-historic-pipeline-job")).toContainText("Historical Upscale Job");
  await expect(page.getByTestId("cleanup-job-legacy-pipeline-job")).toContainText("Legacy Upscale Job");
  await expect(page.getByTestId("cleanup-job-conv_historic-source-job")).toContainText("Historical Conversion");
  await expect(page.getByTestId("cleanup-directory-historic-pipeline-job")).toContainText("job_historic-pipeline-job");
  await expect(page.getByTestId("cleanup-scratch-size-historic-pipeline-job")).toContainText("32 MB");
  await expect(page.getByTestId("cleanup-output-size-historic-pipeline-job")).toContainText("18 MB");
  await expect(page.getByTestId("cleanup-row-repeat-legacy-pipeline-job")).toBeVisible();
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
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Codec: h265");
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Container: mkv");
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Average Throughput: 15.0 fps");
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Current Throughput: calculating");
  await expect(page.getByTestId("cleanup-details-historic-pipeline-job")).toContainText("Stage Times: extract 6s, upscale 52s, encode 15s, remux 7s");
  await expect(page.getByTestId("cleanup-clear-scratch-historic-pipeline-job")).toHaveAttribute("title", /intermediate working files/i);

  await page.getByTestId("cleanup-open-output-historic-pipeline-job").click();
  await page.getByTestId("cleanup-open-scratch-historic-pipeline-job").click();
  await page.getByTestId("cleanup-bulk-output").click();

  await page.getByTestId("cleanup-clear-scratch-historic-pipeline-job").click();
  await page.getByTestId("cleanup-expand-conv_historic-source-job").click();
  await expect(page.getByTestId("cleanup-delete-output-conv_historic-source-job")).toHaveAttribute("title", /converted source file created by the app/i);
  await page.getByTestId("cleanup-delete-output-conv_historic-source-job").click();
  await page.getByTestId("clear-jobs-pool-button").click();

  const { deletedPaths, openedPaths, confirmMessages } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(openedPaths).toEqual([
    "C:/workspace/artifacts/outputs/historic-upscale.mkv",
    "C:/workspace/artifacts/jobs/job_historic-pipeline-job"
  ]);
  expect(confirmMessages).toHaveLength(4);
  expect(confirmMessages[0]).toContain("Impacted size");
  expect(confirmMessages[1]).toContain("C:/workspace/artifacts/jobs/job_historic-pipeline-job");
  expect(confirmMessages[1]).toContain("intermediate artifacts");
  expect(confirmMessages[2]).toContain("C:/workspace/artifacts/runtime/converted-sources/historic-source.mp4");
  expect(confirmMessages[2]).toContain("compatibility");
  expect(confirmMessages[3]).toContain("C:/workspace/artifacts/jobs");
  expect(confirmMessages[3]).toContain("scratch pool");
  expect(deletedPaths).toEqual([
    "C:/workspace/artifacts/outputs/historic-upscale.mkv",
    "C:/workspace/artifacts/outputs/legacy-repeat-output.mp4",
    "C:/workspace/artifacts/runtime/converted-sources/historic-source.mp4",
    "C:/workspace/artifacts/jobs/job_historic-pipeline-job",
    "C:/workspace/artifacts/runtime/converted-sources/historic-source.mp4",
    "C:/workspace/artifacts/jobs"
  ]);
});

test("restores a historical pipeline request for repeat runs", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("job-cleanup-panel-toggle").click();
  await page.getByTestId("cleanup-row-repeat-historic-pipeline-job").click();

  await expect(page.getByTestId("input-panel")).toContainText("C:/workspace/fixtures/historic-input.mov");
  await expect(page.getByTestId("selected-model-label")).toContainText("Real-ESRGAN x4 Plus");
  await expect(page.getByTestId("output-path-input")).toHaveValue("C:/workspace/artifacts/outputs/historic-upscale.mkv");
  await expect(page.getByTestId("codec-select")).toHaveValue("h265");
  await expect(page.getByTestId("container-select")).toHaveValue("mkv");
  await expect(page.getByTestId("encoding-quality-preset-select")).toHaveValue("qualityMax");
  await expect(page.getByTestId("preview-mode-checkbox")).not.toBeChecked();
  await expect(page.getByTestId("segment-duration-input")).toHaveValue("10");
});

test("restores legacy pipeline jobs with recorded source/output settings and current advanced defaults", async ({ page }) => {
  await page.goto("/");

  await openUpscaleControls(page);
  await page.getByTestId("quality-preset-select").selectOption("vramSafe");
  await page.getByTestId("job-cleanup-panel-toggle").click();
  await page.getByTestId("cleanup-row-repeat-legacy-pipeline-job").click();

  await expect(page.getByTestId("input-panel")).toContainText("C:/workspace/fixtures/legacy-repeat-source.mp4");
  await expect(page.getByTestId("output-path-input")).toHaveValue("C:/workspace/artifacts/outputs/legacy-repeat-output.mp4");
  await expect(page.getByTestId("codec-select")).toHaveValue("h264");
  await expect(page.getByTestId("container-select")).toHaveValue("mp4");
  await expect(page.getByTestId("quality-preset-select")).toHaveValue("vramSafe");
});

test("restarts interrupted jobs from the jobs table using saved settings", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("job-cleanup-panel-toggle").click();
  await expect(page.getByTestId("cleanup-job-interrupted-pipeline-job")).toContainText("Interrupted Upscale Job");
  await expect(page.getByTestId("cleanup-row-restart-interrupted-pipeline-job")).toBeVisible();
  await expect(page.getByTestId("cleanup-row-repeat-interrupted-pipeline-job")).toBeVisible();
  await expect(page.getByTestId("cleanup-row-repeat-interrupted-pipeline-job")).toContainText("Load Template");
  await page.getByTestId("cleanup-expand-interrupted-pipeline-job").click();
  await expect(page.getByTestId("cleanup-details-interrupted-pipeline-job")).toContainText("Input Path: C:/workspace/fixtures/interrupted-source.mp4");
  await expect(page.getByTestId("cleanup-details-interrupted-pipeline-job")).toContainText("Output Path: C:/workspace/artifacts/outputs/interrupted-output.mkv");
  await expect(page.getByTestId("cleanup-details-interrupted-pipeline-job")).toContainText("loaded as a template for edits first");
  await page.getByTestId("cleanup-row-restart-interrupted-pipeline-job").click();

  await expect(page.getByTestId("input-panel")).toContainText("C:/workspace/fixtures/interrupted-source.mp4");
  await expect(page.getByTestId("output-path-input")).toHaveValue("C:/workspace/artifacts/outputs/interrupted-output.mkv");
  await expect(page.getByTestId("codec-select")).toHaveValue("h265");
  await expect(page.getByTestId("container-select")).toHaveValue("mkv");

  const { lastRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(lastRequest?.sourcePath).toBe("C:/workspace/fixtures/interrupted-source.mp4");
  expect(lastRequest?.outputPath).toBe("C:/workspace/artifacts/outputs/interrupted-output.mkv");
  expect(lastRequest?.qualityPreset).toBe("qualityBalanced");
});

test("restarts cancelled jobs from the jobs table using saved settings", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.startPipeline = async (request) => {
      window.__UPSCALER_TEST_STATE__.lastRequest = request;
      return "mock-job-realesrgan-x4plus";
    };
    window.__UPSCALER_MOCK__.getPipelineJob = async () => ({
      jobId: "mock-job-realesrgan-x4plus",
      state: "running",
      progress: {
        phase: "upscaling",
        percent: 35,
        message: "Upscaling extracted frames",
        processedFrames: 105,
        totalFrames: 300,
        extractedFrames: 300,
        upscaledFrames: 105,
        interpolatedFrames: 0,
        encodedFrames: 0,
        remuxedFrames: 0,
      },
      result: null,
      error: null,
    });
    window.__UPSCALER_MOCK__.cancelPipelineJob = async () => {
      window.__UPSCALER_MOCK__.getPipelineJob = async () => ({
        jobId: "mock-job-realesrgan-x4plus",
        state: "cancelled",
        progress: {
          phase: "failed",
          percent: 100,
          message: "Job cancelled by user",
          processedFrames: 105,
          totalFrames: 300,
          extractedFrames: 300,
          upscaledFrames: 105,
          interpolatedFrames: 0,
          encodedFrames: 0,
          remuxedFrames: 0,
        },
        result: null,
        error: "Job cancelled by user",
      });
    };
  });

  await page.getByTestId("select-video-button").click();
  await page.getByTestId("run-upscale-button").click();
  await page.getByTestId("job-cleanup-panel-toggle").click();
  await page.getByTestId("cleanup-expand-mock-job-realesrgan-x4plus").click();
  await expect(page.getByTestId("cleanup-stop-mock-job-realesrgan-x4plus")).toBeVisible();
  await page.getByTestId("cleanup-stop-mock-job-realesrgan-x4plus").click();

  await expect(page.getByTestId("cleanup-job-mock-job-realesrgan-x4plus")).toContainText("cancelled");
  await expect(page.getByTestId("cleanup-row-restart-mock-job-realesrgan-x4plus")).toBeVisible();
  await expect(page.getByTestId("cleanup-row-repeat-mock-job-realesrgan-x4plus")).toBeVisible();
  await expect(page.getByTestId("cleanup-row-repeat-mock-job-realesrgan-x4plus")).toContainText("Load Template");

  await expect(page.getByTestId("cleanup-details-mock-job-realesrgan-x4plus")).toContainText("Recovery: This incomplete job can be restarted immediately or loaded as a template for edits first.");
  await page.getByTestId("cleanup-row-restart-mock-job-realesrgan-x4plus").click();

  await expect(page.getByTestId("input-panel")).toContainText("C:/fixtures/sample-input.mp4");
  await expect(page.getByTestId("output-path-input")).toHaveValue("artifacts/video-upgrader/outputs/sample-input_realesrgan_x4plus.mp4");

  const { lastRequest } = await page.evaluate(() => window.__UPSCALER_TEST_STATE__);
  expect(lastRequest?.sourcePath).toBe("C:/fixtures/sample-input.mp4");
  expect(lastRequest?.outputPath).toBe("artifacts/video-upgrader/outputs/sample-input_realesrgan_x4plus.mp4");
});

test("pauses and resumes an active pipeline job", async ({ page }) => {
  await page.goto("/");

  await page.evaluate(() => {
    if (!window.__UPSCALER_MOCK__ || !window.__UPSCALER_TEST_STATE__) {
      return;
    }

    const pauseState = {
      state: "running",
      progress: {
        phase: "upscaling",
        percent: 41,
        message: "Upscaling extracted frames",
        processedFrames: 123,
        totalFrames: 300,
        extractedFrames: 300,
        upscaledFrames: 123,
        interpolatedFrames: 0,
        encodedFrames: 0,
        remuxedFrames: 0,
      },
    };
    window.__UPSCALER_TEST_STATE__.pauseState = pauseState;

    window.__UPSCALER_MOCK__.startPipeline = async (request) => {
      window.__UPSCALER_TEST_STATE__.lastRequest = request;
      return "pauseable-live-job";
    };
    window.__UPSCALER_MOCK__.getPipelineJob = async () => ({
      jobId: "pauseable-live-job",
      state: pauseState.state,
      progress: {
        ...pauseState.progress,
      },
      result: null,
      error: null,
    });
    window.__UPSCALER_MOCK__.pausePipelineJob = async () => {
      pauseState.state = "paused";
      pauseState.progress = {
        ...pauseState.progress,
        phase: "paused",
        message: "Paused: upscaling extracted frames",
      };
    };
    window.__UPSCALER_MOCK__.resumePipelineJob = async () => {
      pauseState.state = "running";
      pauseState.progress = {
        ...pauseState.progress,
        phase: "upscaling",
        message: "Resumed: upscaling extracted frames",
      };
    };
  });

  await page.getByTestId("select-video-button").click();
  await page.getByTestId("run-upscale-button").click();
  await page.getByTestId("job-cleanup-panel-toggle").click();
  await page.getByTestId("cleanup-expand-pauseable-live-job").click();

  await expect(page.getByTestId("cleanup-pause-pauseable-live-job")).toBeVisible();
  await page.getByTestId("cleanup-pause-pauseable-live-job").click();
  await expect(page.getByTestId("cleanup-job-pauseable-live-job")).toContainText("paused");
  await expect(page.getByTestId("cleanup-resume-pauseable-live-job")).toBeVisible();

  await page.getByTestId("top-status-pause-button").click();
  await expect(page.getByTestId("cleanup-job-pauseable-live-job")).toContainText("running");
  await expect(page.getByTestId("cleanup-pause-pauseable-live-job")).toBeVisible();
});

test("shows a running pipeline job in the standalone jobs window", async ({ page }) => {
  await page.addInitScript(() => {
    if (!window.__UPSCALER_MOCK__) {
      return;
    }

    window.__UPSCALER_MOCK__.listManagedJobs = async () => ([
      {
        jobId: "running-window-job",
        jobKind: "pipeline",
        label: "Detached Running Job",
        state: "running",
        sourcePath: "C:/workspace/fixtures/running-input.mov",
        modelId: "realesrgan-x4plus",
        codec: "h265",
        container: "mkv",
        progress: {
          phase: "upscaling",
          percent: 47,
          message: "Upscaling segment 3/6 batch 2/4 (244/520 frames)",
          processedFrames: 1184,
          totalFrames: 2520,
          extractedFrames: 1560,
          upscaledFrames: 1184,
          interpolatedFrames: 0,
          encodedFrames: 320,
          remuxedFrames: 0,
          segmentIndex: 3,
          segmentCount: 6,
          segmentProcessedFrames: 244,
          segmentTotalFrames: 520,
          batchIndex: 2,
          batchCount: 4,
          elapsedSeconds: 152,
          averageFramesPerSecond: 7.8,
          rollingFramesPerSecond: 8.4,
          estimatedRemainingSeconds: 171,
          processRssBytes: 1024 * 1024 * 896,
          gpuMemoryUsedBytes: 1024 * 1024 * 6144,
          gpuMemoryTotalBytes: 1024 * 1024 * 24576,
          scratchSizeBytes: 1024 * 1024 * 22,
          outputSizeBytes: 1024 * 1024 * 7,
          extractStageSeconds: 12,
          upscaleStageSeconds: 140,
          interpolateStageSeconds: 0,
          encodeStageSeconds: 18,
          remuxStageSeconds: 0,
        },
        recordedCount: 2520,
        scratchPath: "C:/workspace/artifacts/jobs/job_running-window-job",
        scratchStats: {
          path: "C:/workspace/artifacts/jobs/job_running-window-job",
          exists: true,
          isDirectory: true,
          sizeBytes: 1024 * 1024 * 22,
        },
        outputPath: "C:/workspace/artifacts/outputs/running-window-job.mkv",
        outputStats: {
          path: "C:/workspace/artifacts/outputs/running-window-job.mkv",
          exists: true,
          isDirectory: false,
          sizeBytes: 1024 * 1024 * 7,
        },
        pipelineRunDetails: {
          request: {
            sourcePath: "C:/workspace/fixtures/running-input.mov",
            modelId: "realesrgan-x4plus",
            outputMode: "preserveAspect4k",
            qualityPreset: "qualityBalanced",
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
            cropLeft: null,
            cropTop: null,
            cropWidth: null,
            cropHeight: null,
            previewMode: false,
            previewDurationSeconds: null,
            segmentDurationSeconds: 10,
            outputPath: "C:/workspace/artifacts/outputs/running-window-job.mkv",
            codec: "h265",
            container: "mkv",
            tileSize: 128,
            fp16: false,
            crf: 18,
          },
          sourceMedia: {
            width: 1280,
            height: 720,
            frameRate: 24,
            durationSeconds: 105,
            frameCount: 2520,
            aspectRatio: 1.7777777778,
            pixelCount: 921600,
            hasAudio: true,
            container: "mov",
            videoCodec: "prores",
          },
          outputMedia: {
            width: 3840,
            height: 2160,
            frameRate: 24,
            durationSeconds: 105,
            frameCount: 2520,
            aspectRatio: 1.7777777778,
            pixelCount: 8294400,
            hasAudio: true,
            container: "mkv",
            videoCodec: "hevc",
          },
          effectiveSettings: {
            effectiveTileSize: 128,
            processedDurationSeconds: 105,
            segmentFrameLimit: 520,
            previewMode: false,
            previewDurationSeconds: null,
            segmentDurationSeconds: 10,
          },
          executionPath: "file-io",
          videoEncoder: "libx265",
          videoEncoderLabel: "HEVC (libx265)",
          runner: "torch",
          precision: "fp32",
          averageThroughputFps: 7.8,
          segmentCount: 6,
          segmentFrameLimit: 520,
          frameCount: 2520,
          hadAudio: true,
          runtime: {
            ffmpegPath: "C:/tools/ffmpeg.exe",
            realesrganPath: "C:/tools/realesrgan-ncnn-vulkan.exe",
            modelDir: "C:/tools/models",
            rifePath: "C:/tools/rife-ncnn-vulkan.exe",
            rifeModelRoot: "C:/tools/rife-models",
            availableGpus: [
              { id: 0, name: "Intel(R) Graphics", kind: "integrated" },
              { id: 1, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", kind: "discrete" },
            ],
            defaultGpuId: 1,
          },
        },
        updatedAt: String(Math.floor(Date.now() / 1000) - 15),
      },
    ]);
  });

  await page.goto("/?view=jobs");

  await expect(page.getByTestId("job-cleanup-panel")).toBeVisible();
  await expect(page.getByTestId("cleanup-job-running-window-job")).toContainText("Detached Running Job");
  await expect(page.getByTestId("cleanup-job-running-window-job")).toContainText("running");
  await expect(page.getByTestId("cleanup-input-running-window-job")).toContainText("running-input.mov");
  await expect(page.getByTestId("cleanup-output-running-window-job")).toContainText("running-window-job.mkv");
  await expect(page.getByTestId("cleanup-scratch-size-running-window-job")).toContainText("22 MB");
  await expect(page.getByTestId("cleanup-output-size-running-window-job")).toContainText("7.0 MB");

  await page.getByTestId("cleanup-expand-running-window-job").click();
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Phase: upscaling");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Input Path: C:/workspace/fixtures/running-input.mov");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Scratch Path: C:/workspace/artifacts/jobs/job_running-window-job");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Output Path: C:/workspace/artifacts/outputs/running-window-job.mkv");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Average Throughput: 7.80 fps");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Current Throughput: 8.40 fps");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Elapsed: 2m 32s");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("ETA: 2m 51s");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Execution Path: file-io");
  await expect(page.getByTestId("cleanup-details-running-window-job")).toContainText("Precision: fp32");
});

test("renders the standalone jobs view without a close button and keeps the jobs table horizontally scrollable", async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 900 });
  await page.goto("/?view=jobs");

  await expect(page.getByTestId("job-cleanup-panel")).toBeVisible();
  await expect(page.getByTestId("jobs-window-close")).toHaveCount(0);
  await expect(page.getByTestId("top-status-panel")).toHaveCount(0);
  await expect(page.getByTestId("cleanup-row-repeat-historic-pipeline-job")).toBeVisible();
  await expect(page.getByTestId("cleanup-row-repeat-historic-pipeline-job")).toContainText("Load Template");
  await expect(page.getByTestId("cleanup-row-repeat-legacy-pipeline-job")).toBeVisible();
  await expect(page.getByTestId("cleanup-scratch-size-historic-pipeline-job")).toContainText("32 MB");
  await expect(page.getByTestId("cleanup-output-size-historic-pipeline-job")).toContainText("18 MB");

  const tableMetrics = await page.getByTestId("cleanup-jobs-table-shell").evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));

  expect(tableMetrics.scrollWidth).toBeGreaterThan(tableMetrics.clientWidth);
});

test("renders the standalone comparison view from shared workspace state", async ({ page }) => {
  await page.addInitScript(() => {
    const runtime = {
      ffmpegPath: "C:/tools/ffmpeg.exe",
      realesrganPath: "C:/tools/realesrgan-ncnn-vulkan.exe",
      modelDir: "C:/tools/models",
      rifePath: "C:/tools/rife-ncnn-vulkan.exe",
      rifeModelRoot: "C:/tools/rife-models",
      availableGpus: [
        { id: 0, name: "Intel(R) Graphics", kind: "integrated" },
        { id: 1, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", kind: "discrete" },
      ],
      defaultGpuId: 1,
    };
    const mediaSummary = {
      width: 1280,
      height: 720,
      frameRate: 24,
      durationSeconds: 3,
      frameCount: 72,
      aspectRatio: 1280 / 720,
      pixelCount: 1280 * 720,
      hasAudio: false,
      container: "mp4",
      videoCodec: "h264",
    };
    const completedProgress = {
      phase: "completed",
      percent: 100,
      message: "Blind comparison sample ready",
      processedFrames: 72,
      totalFrames: 72,
      extractedFrames: 72,
      upscaledFrames: 72,
      interpolatedFrames: 0,
      encodedFrames: 72,
      remuxedFrames: 0,
    };
    window.localStorage.setItem("videoupgrader.comparison.workspace.v1", JSON.stringify({
      updatedAt: Date.now(),
      source: {
        path: "C:/fixtures/sample-input.mp4",
        previewPath: "C:/fixtures/sample-input-preview.mp4",
        width: 1280,
        height: 720,
        durationSeconds: 12.5,
        frameRate: 24,
        hasAudio: true,
        container: "mp4",
        videoCodec: "h264",
      },
      blindComparison: {
        state: "ready",
        previewDurationSeconds: 3,
        previewStartOffsetSeconds: 2.25,
        selectedSampleId: null,
        winnerModelId: null,
        revealed: false,
        error: null,
        entries: [
          {
            sampleId: "sample-1",
            anonymousLabel: "Sample A",
            modelId: "realesrgan-x4plus",
            jobId: "job-sample-1",
            status: {
              jobId: "job-sample-1",
              state: "succeeded",
              progress: completedProgress,
              result: {
                outputPath: "C:/exports/blind-sample-a.mp4",
                workDir: "C:/workspace/artifacts/jobs/job-sample-1",
                frameCount: 72,
                hadAudio: false,
                codec: "h264",
                container: "mp4",
                sourceMedia: mediaSummary,
                outputMedia: mediaSummary,
                runtime,
                log: [],
              },
              error: null,
            },
          },
          {
            sampleId: "sample-2",
            anonymousLabel: "Sample B",
            modelId: "bsrgan-x4",
            jobId: "job-sample-2",
            status: {
              jobId: "job-sample-2",
              state: "succeeded",
              progress: completedProgress,
              result: {
                outputPath: "C:/exports/blind-sample-b.mp4",
                workDir: "C:/workspace/artifacts/jobs/job-sample-2",
                frameCount: 72,
                hadAudio: false,
                codec: "h264",
                container: "mp4",
                sourceMedia: mediaSummary,
                outputMedia: mediaSummary,
                runtime,
                log: [],
              },
              error: null,
            },
          },
        ],
      },
    }));
  });

  await page.goto("/?view=comparison");

  await expect(page.getByTestId("top-status-panel")).toHaveCount(0);
  await expect(page.getByTestId("comparison-workspace-window")).toBeVisible();
  await expect(page.getByTestId("comparison-workspace-window")).toContainText("Source plus 2 blind samples");
  await expect(page.getByTestId("comparison-workspace-close")).toHaveCount(0);
  await expect(page.getByTestId("comparison-pane-zoom-readout")).toContainText("1.00x");
  await expect(page.getByTestId("comparison-zoom-readout")).toContainText("3.00x");
});
