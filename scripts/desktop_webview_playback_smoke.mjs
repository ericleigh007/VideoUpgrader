import { chromium, expect } from '@playwright/test';

async function main() {
  const realSourcePath = process.env.REAL_SOURCE_PATH ?? '';
  const realPreviewPath = process.env.REAL_PREVIEW_PATH ?? '';
  const mockContainer = process.env.MOCK_SOURCE_CONTAINER ?? 'mp4';
  const targetAspect = process.env.TARGET_ASPECT ?? '1:1';
  const previewWaitMs = Number.parseInt(process.env.PREVIEW_WAIT_MS ?? '10000', 10);
  const cdpConnectTimeoutMs = Number.parseInt(process.env.CDP_CONNECT_TIMEOUT_MS ?? '60000', 10);
  const cropNudgeMode = process.env.CROP_NUDGE_MODE ?? 'xy';
  const [targetAspectWidth, targetAspectHeight] = targetAspect.split(':').map((value) => Number.parseFloat(value));
  const expectedAspect = targetAspectWidth > 0 && targetAspectHeight > 0 ? targetAspectWidth / targetAspectHeight : 1;
  const isRealBackendMode = realSourcePath.length > 0;
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9223', { timeout: cdpConnectTimeoutMs });
  try {
    const context = browser.contexts()[0];
    const page = context?.pages().find((candidate) => candidate.url().includes('localhost:1420'));
    if (!page) {
      throw new Error('Could not find the Upscaler desktop webview target');
    }

    await page.bringToFront();
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.evaluate(({ realSourcePath: sourcePath, isRealBackendMode: useRealBackend, mockContainer: container }) => {
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
        async selectOutputFile(_defaultPath, container) {
          return `C:/exports/upscaled-result.${container}`;
        },
        async ensureRuntimeAssets() {
          return {
            ffmpegPath: 'C:/tools/ffmpeg.exe',
            realesrganPath: 'C:/tools/realesrgan-ncnn-vulkan.exe',
            modelDir: 'C:/tools/models',
            availableGpus: [{ id: 1, name: 'NVIDIA RTX', kind: 'discrete' }],
            defaultGpuId: 1,
          };
        },
        async probeSourceVideo(sourcePath) {
          return {
            path: sourcePath,
            previewPath: container === 'mp4' ? '/fixtures/gui-progress-sample.mp4' : 'C:/fixtures/sample-input-preview.mp4',
            width: 1280,
            height: 720,
            durationSeconds: 12.5,
            frameRate: 24,
            hasAudio: true,
            container,
          };
        },
        async startSourceConversionToMp4(sourcePath) {
          return `mock-${sourcePath}`;
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
        async startPipeline() {
          return 'mock-pipeline-job';
        },
        async getPipelineJob() {
          return {
            jobId: 'mock-pipeline-job',
            state: 'queued',
            progress: {
              phase: 'queued',
              percent: 0,
              message: 'Job queued',
              processedFrames: 0,
              totalFrames: 0,
              extractedFrames: 0,
              upscaledFrames: 0,
              encodedFrames: 0,
              remuxedFrames: 0,
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
    }, { realSourcePath, isRealBackendMode, mockContainer });

    await page.getByTestId('select-video-button').click();
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
    await expect(page.getByTestId('source-preview-play-toggle')).toBeVisible();
    await page.getByTestId('source-preview-play-toggle').click();

    const playbackAdvanced = await page.waitForFunction(() => {
      const video = document.querySelector('[data-testid="source-preview"]');
      return Boolean(video && !video.paused && video.currentTime > 0.15);
    }, undefined, { timeout: 10000 }).then(() => true).catch(() => false);

    if (!playbackAdvanced) {
      const playbackDiagnostics = await page.evaluate(() => {
        const video = document.querySelector('[data-testid="source-preview"]');
        const errorText = document.querySelector('[data-testid="error-text"]')?.textContent ?? null;
        if (!(video instanceof HTMLVideoElement)) {
          return { errorText, video: null };
        }

        return Promise.resolve(fetch(video.currentSrc || video.src, { method: 'GET' }).then(async (response) => ({
          errorText,
          video: {
            currentSrc: video.currentSrc || video.src,
            paused: video.paused,
            ended: video.ended,
            currentTime: video.currentTime,
            duration: video.duration,
            readyState: video.readyState,
            networkState: video.networkState,
            muted: video.muted,
            errorCode: video.error?.code ?? null,
            errorMessage: video.error?.message ?? null,
          },
          fetch: {
            ok: response.ok,
            status: response.status,
            contentType: response.headers.get('content-type'),
            contentLength: response.headers.get('content-length'),
            url: response.url,
          },
        })).catch((fetchError) => ({
          errorText,
          video: {
            currentSrc: video.currentSrc || video.src,
            paused: video.paused,
            ended: video.ended,
            currentTime: video.currentTime,
            duration: video.duration,
            readyState: video.readyState,
            networkState: video.networkState,
            muted: video.muted,
            errorCode: video.error?.code ?? null,
            errorMessage: video.error?.message ?? null,
          },
          fetch: {
            error: fetchError instanceof Error ? fetchError.message : String(fetchError),
          },
        })));
      });
      throw new Error(`Preview playback did not advance. Diagnostics: ${JSON.stringify(playbackDiagnostics)}`);
    }

    await page.getByTestId('output-mode-select').selectOption('cropTo4k');
    await page.getByTestId('aspect-ratio-select').selectOption(targetAspect);
    await expect(page.getByTestId('toggle-crop-edit-button')).toBeVisible();
    await expect(page.getByTestId('maximize-crop-button')).toBeVisible();
    await page.getByTestId('toggle-crop-edit-button').click();
    await expect(page.getByTestId('crop-overlay')).toBeVisible();
    await expect(page.getByTestId('crop-nudge-controls')).toBeVisible();

    const firstCropBox = await page.getByTestId('crop-overlay').boundingBox();
    if (!firstCropBox) {
      throw new Error('Crop overlay did not render a measurable bounding box');
    }
    const firstAspect = firstCropBox.width / firstCropBox.height;
    if (Math.abs(firstAspect - expectedAspect) > 0.08) {
      throw new Error(`Crop overlay aspect ${firstAspect} did not match expected ${expectedAspect} for ${targetAspect}`);
    }

    if (cropNudgeMode === 'vertical') {
      await page.getByTestId('crop-nudge-down').click();
    } else {
      await page.getByTestId('crop-nudge-right').click();
      await page.getByTestId('crop-nudge-down').click();
    }

    const secondCropBox = await page.getByTestId('crop-overlay').boundingBox();
    if (!secondCropBox) {
      throw new Error('Crop overlay vanished after first nudge');
    }
    if (Math.abs(secondCropBox.x - firstCropBox.x) < 4 && Math.abs(secondCropBox.y - firstCropBox.y) < 4) {
      throw new Error(`Crop overlay did not move on first nudge: before=${JSON.stringify(firstCropBox)} after=${JSON.stringify(secondCropBox)}`);
    }

    await page.getByTestId('maximize-crop-button').click();
    const maximizedCropBox = await page.getByTestId('crop-overlay').boundingBox();
    if (!maximizedCropBox) {
      throw new Error('Crop overlay vanished after maximize');
    }
    const maximizedAspect = maximizedCropBox.width / maximizedCropBox.height;
    if (Math.abs(maximizedAspect - expectedAspect) > 0.08) {
      throw new Error(`Maximized crop overlay aspect ${maximizedAspect} did not match expected ${expectedAspect} for ${targetAspect}`);
    }

    await page.getByTestId('source-preview-seek').evaluate((element) => {
      element.value = '1.75';
      element.dispatchEvent(new Event('input', { bubbles: true }));
      element.dispatchEvent(new Event('change', { bubbles: true }));
    });

    await page.waitForFunction(() => {
      const video = document.querySelector('[data-testid="source-preview"]');
      return Boolean(video && video.currentTime >= 1.7);
    }, undefined, { timeout: 10000 });

    const afterSeekTime = await page.evaluate(() => {
      const video = document.querySelector('[data-testid="source-preview"]');
      return video ? video.currentTime : -1;
    });

    if (cropNudgeMode === 'vertical') {
      await page.getByTestId('crop-nudge-up').click();
    } else {
      await page.getByTestId('crop-nudge-left').click();
      await page.getByTestId('crop-nudge-down').click();
    }

    const thirdCropBox = await page.getByTestId('crop-overlay').boundingBox();
    if (!thirdCropBox) {
      throw new Error('Crop overlay vanished after second nudge');
    }
    if (Math.abs(thirdCropBox.x - secondCropBox.x) < 4 && Math.abs(thirdCropBox.y - secondCropBox.y) < 4) {
      throw new Error(`Crop overlay did not move on second nudge: before=${JSON.stringify(secondCropBox)} after=${JSON.stringify(thirdCropBox)}`);
    }

    await page.waitForTimeout(700);

    const playback = await page.evaluate(() => {
      const video = document.querySelector('[data-testid="source-preview"]');
      return video
        ? {
            paused: video.paused,
            currentTime: video.currentTime,
            duration: video.duration,
            src: video.currentSrc || video.src,
          }
        : null;
    });

    if (!playback) {
      throw new Error('Source preview video not found after load');
    }

    if (playback.paused || playback.currentTime <= afterSeekTime + 0.1) {
      throw new Error(`Desktop playback did not advance as expected: ${JSON.stringify(playback)}`);
    }

    const result = {
      playback,
      cropBoxes: {
        first: firstCropBox,
        second: secondCropBox,
        maximized: maximizedCropBox,
        third: thirdCropBox,
      },
      afterSeekTime,
      targetAspect,
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