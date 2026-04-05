import { chromium, expect } from '@playwright/test';

async function main() {
  const realSourcePath = process.env.REAL_SOURCE_PATH ?? '';
  const realPreviewPath = process.env.REAL_PREVIEW_PATH ?? '';
  const mockContainer = process.env.MOCK_SOURCE_CONTAINER ?? 'mp4';
  const pipelineMode = process.env.DESKTOP_PIPELINE_MODE ?? 'afterUpscale';
  const targetAspect = process.env.TARGET_ASPECT ?? '1:1';
  const previewWaitMs = Number.parseInt(process.env.PREVIEW_WAIT_MS ?? '10000', 10);
  const cdpConnectTimeoutMs = Number.parseInt(process.env.CDP_CONNECT_TIMEOUT_MS ?? '60000', 10);
  const cropNudgeMode = process.env.CROP_NUDGE_MODE ?? 'xy';
  const expectedMockSourcePath = `C:/fixtures/sample-input.${mockContainer}`;
  const [targetAspectWidth, targetAspectHeight] = targetAspect.split(':').map((value) => Number.parseFloat(value));
  const expectedAspect = targetAspectWidth > 0 && targetAspectHeight > 0 ? targetAspectWidth / targetAspectHeight : 1;
  const isRealBackendMode = realSourcePath.length > 0;
  const isInterpolationOnlyMode = pipelineMode === 'interpolateOnly';
  const expectedInterpolationMode = isInterpolationOnlyMode ? 'interpolateOnly' : 'afterUpscale';

  async function openPanelIfCollapsed(page, workspaceTestId, toggleTestId) {
    const workspace = page.getByTestId(workspaceTestId);
    const isVisible = await workspace.isVisible().catch(() => false);
    if (!isVisible) {
      await page.getByTestId(toggleTestId).click();
    }
  }

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9223', { timeout: cdpConnectTimeoutMs });
  try {
    const context = browser.contexts()[0];
    const page = context?.pages().find((candidate) => candidate.url().includes('localhost:1420'));
    if (!page) {
      throw new Error('Could not find the Upscaler desktop webview target');
    }

    await page.bringToFront();
    await page.addInitScript(({ realSourcePath: sourcePath, isRealBackendMode: useRealBackend, mockContainer: container }) => {
      if (useRealBackend) {
        window.__UPSCALER_MOCK__ = {
          async selectVideoFile() {
            return sourcePath;
          },
        };
        return;
      }

      window.__UPSCALER_MOCK__ = {
        async selectVideoFile() {
          return `C:/fixtures/sample-input.${container}`;
        },
        async selectOutputFile(_defaultPath, outputContainer) {
          return `C:/exports/upscaled-result.${outputContainer}`;
        },
        async ensureRuntimeAssets() {
          return {
            ffmpegPath: 'C:/tools/ffmpeg.exe',
            realesrganPath: 'C:/tools/realesrgan-ncnn-vulkan.exe',
            modelDir: 'C:/tools/models',
            rifePath: 'C:/tools/rife-ncnn-vulkan.exe',
            rifeModelRoot: 'C:/tools/rife-models',
            availableGpus: [{ id: 1, name: 'NVIDIA RTX', kind: 'discrete' }],
            defaultGpuId: 1,
          };
        },
        async probeSourceVideo(selectedSourcePath) {
          return {
            path: selectedSourcePath,
            previewPath: container === 'mp4' ? '/fixtures/gui-progress-sample.mp4' : 'C:/fixtures/sample-input-preview.mp4',
            width: 1280,
            height: 720,
            durationSeconds: 12.5,
            frameRate: 24,
            hasAudio: true,
            container,
            videoCodec: 'h264',
          };
        },
        async startSourceConversionToMp4(selectedSourcePath) {
          return `mock-${selectedSourcePath}`;
        },
        async getSourceConversionJob(jobId) {
          return {
            jobId,
            state: 'succeeded',
            progress: {
              phase: 'completed',
              percent: 100,
              message: 'Source conversion completed',
              processedFrames: 1000,
              totalFrames: 1000,
              extractedFrames: 0,
              upscaledFrames: 0,
              interpolatedFrames: 0,
              encodedFrames: 0,
              remuxedFrames: 0,
            },
            result: {
              path: 'C:/fixtures/sample-input_fastprep.mp4',
              previewPath: '/fixtures/gui-progress-sample.mp4',
              width: 1280,
              height: 720,
              durationSeconds: 12.5,
              frameRate: 24,
              hasAudio: true,
              container: 'mp4',
              videoCodec: 'h264',
            },
            error: null,
          };
        },
        async cancelSourceConversionJob() {},
        async getAppConfig() {
          return { modelRatings: {}, blindComparisons: [] };
        },
        async saveModelRating() {
          return { modelRatings: {}, blindComparisons: [] };
        },
        async recordBlindComparisonSelection() {
          return { modelRatings: {}, blindComparisons: [] };
        },
        async startPipeline(request) {
          window.__UPSCALER_TEST_STATE__ = { lastRequest: request };
          return 'mock-pipeline-job';
        },
        async getPipelineJob() {
          return {
            jobId: 'mock-pipeline-job',
            state: 'running',
            progress: {
              phase: 'encoding',
              percent: isInterpolationOnlyMode ? 71 : 76,
              message: isInterpolationOnlyMode
                ? 'Encoding interpolated-only segment 1 while the next segment is already interpolated'
                : 'Encoding segment 1 while the next segment is already interpolated',
              processedFrames: 180,
              totalFrames: 750,
              extractedFrames: 300,
              upscaledFrames: isInterpolationOnlyMode ? 0 : 300,
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
              upscaleStageSeconds: isInterpolationOnlyMode ? 0 : 18,
              interpolateStageSeconds: 12,
              encodeStageSeconds: 6,
              remuxStageSeconds: 0,
            },
            result: null,
            error: null,
          };
        },
        async cancelPipelineJob() {},
        async getPathStats(path) {
          return { path, exists: true, isDirectory: false, sizeBytes: 12345 };
        },
        async getScratchStorageSummary() {
          return {
            jobsRoot: { path: 'C:/workspace/artifacts/jobs', exists: true, isDirectory: true, sizeBytes: 1 },
            convertedSourcesRoot: { path: 'C:/workspace/artifacts/runtime/converted-sources', exists: true, isDirectory: true, sizeBytes: 1 },
            sourcePreviewsRoot: { path: 'C:/workspace/artifacts/runtime/source-previews', exists: true, isDirectory: true, sizeBytes: 1 },
          };
        },
        async listManagedJobs() {
          return [];
        },
        async deleteManagedPath() {},
        async openPathInDefaultApp() {},
        toPreviewSrc(path) {
          return path;
        },
      };
    }, { realSourcePath, isRealBackendMode, mockContainer, isInterpolationOnlyMode });
    await page.evaluate(() => {
      window.localStorage.clear();
      window.sessionStorage.clear();
    }).catch(() => {});
    await page.reload({ waitUntil: 'domcontentloaded' });

    const mockDiagnostics = await page.evaluate(async () => {
      const mock = window.__UPSCALER_MOCK__;
      return {
        hasMock: Boolean(mock),
        mockKeys: mock ? Object.keys(mock).sort() : [],
        selectedPath: mock && typeof mock.selectVideoFile === 'function' ? await mock.selectVideoFile() : null,
      };
    });

    if (!mockDiagnostics.hasMock) {
      throw new Error(`Desktop smoke failed to install the page mock: ${JSON.stringify(mockDiagnostics)}`);
    }

    await page.getByTestId('select-video-button').click();
    if (!isRealBackendMode) {
      const selectedSourceVisible = await page.getByText(expectedMockSourcePath).waitFor({ timeout: previewWaitMs }).then(() => true).catch(() => false);
      if (!selectedSourceVisible) {
        const selectionDiagnostics = await page.evaluate(() => ({
          errorText: document.querySelector('[data-testid="error-text"]')?.textContent ?? null,
          bodyText: document.body.textContent ?? '',
          testState: window.__UPSCALER_TEST_STATE__ ?? null,
        }));
        throw new Error(`Desktop smoke did not switch to the mocked source path ${expectedMockSourcePath}. Diagnostics: ${JSON.stringify(selectionDiagnostics)}`);
      }
    }

    const previewBecameVisible = await page.waitForFunction(() => {
      return Boolean(
        document.querySelector('[data-testid="source-preview"]')
        || document.querySelector('[data-testid="source-preview-play-toggle"]')
      );
    }, undefined, { timeout: previewWaitMs }).then(() => true).catch(() => false);

    if (!previewBecameVisible) {
      const diagnostics = await page.evaluate(async ({ realPreviewPath: previewPath }) => {
        const base = {
          errorText: document.querySelector('[data-testid="error-text"]')?.textContent ?? null,
          heroStatusText: Array.from(document.querySelectorAll('.status-card span, .status-card strong')).map((node) => node.textContent?.trim() ?? '').filter(Boolean),
          bodyText: document.body.textContent ?? '',
        };

        if (!previewPath || !(window.__TAURI_INTERNALS__ && typeof window.__TAURI_INTERNALS__.invoke === 'function')) {
          return base;
        }

        try {
          const encoded = await window.__TAURI_INTERNALS__.invoke('read_preview_file_base64', { path: previewPath.replace(/\\/g, '/') });
          return {
            ...base,
            manualInvoke: {
              ok: true,
              encodedLength: typeof encoded === 'string' ? encoded.length : null,
            },
          };
        } catch (error) {
          return {
            ...base,
            manualInvoke: {
              ok: false,
              error: error instanceof Error ? error.message : String(error),
            },
          };
        }
      }, { realPreviewPath });
      throw new Error(`Source preview did not appear. Diagnostics: ${JSON.stringify(diagnostics)}`);
    }

    await expect(page.getByTestId('source-preview')).toBeVisible();
    if (isRealBackendMode || mockContainer !== 'mp4') {
      await expect(page.getByTestId('source-preview-mode')).toContainText('preview');
    }
    const previewSnapshot = await page.evaluate(() => {
      const video = document.querySelector('[data-testid="source-preview"]');
      return video
        ? {
            paused: video.paused,
            currentTime: video.currentTime,
            duration: video.duration,
            src: video.currentSrc || video.src,
            readyState: video.readyState,
            networkState: video.networkState,
          }
        : null;
    });

    if (!isRealBackendMode) {
      await openPanelIfCollapsed(page, 'frame-rate-workspace-section', 'pipeline-toggle-interpolation');
      if (isInterpolationOnlyMode) {
        await openPanelIfCollapsed(page, 'upscaler-workspace-section', 'pipeline-toggle-upscale');
        await page.getByTestId('pipeline-toggle-upscale').click();
        await expect(page.getByTestId('frame-rate-mode-readout')).toHaveValue('Enabled standalone');
      }
      await page.getByTestId('frame-rate-target-select').selectOption('60');
      await page.getByTestId('save-output-button').click();
      await page.getByTestId('run-upscale-button').click();
    }

    if (!isRealBackendMode) {
      await expect.poll(async () => await page.evaluate(() => window.__UPSCALER_TEST_STATE__?.lastRequest ?? null), {
        timeout: 10000,
      }).not.toBeNull();
    }

    const lastRequest = !isRealBackendMode
      ? await page.evaluate(() => window.__UPSCALER_TEST_STATE__?.lastRequest ?? null)
      : null;

    if (!isRealBackendMode && (lastRequest?.interpolationMode !== expectedInterpolationMode || lastRequest?.interpolationTargetFps !== 60)) {
      throw new Error(`Desktop overlap smoke captured the wrong pipeline request: ${JSON.stringify(lastRequest)}`);
    }
    if (!isRealBackendMode && isInterpolationOnlyMode && lastRequest?.outputPath !== 'C:/exports/upscaled-result.mp4') {
      throw new Error(`Interpolation-only desktop smoke built an unexpected output path: ${JSON.stringify(lastRequest)}`);
    }

    const result = {
      previewSnapshot,
      pipelineMode,
      targetAspect,
      overlapRequest: lastRequest,
      mockDiagnostics,
    };

    await page.screenshot({ path: 'artifacts/runtime/desktop-webview-playback-smoke.png', fullPage: false });
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
});