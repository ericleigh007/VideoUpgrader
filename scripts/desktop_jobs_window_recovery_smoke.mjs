import { chromium, expect } from '@playwright/test';

const STATE_KEY = '__UPSCALER_DESKTOP_JOBS_STATE__';
const DEFAULT_SOURCE_PATH = 'C:/fixtures/desktop-running-input.mov';

async function main() {
  const cdpConnectTimeoutMs = Number.parseInt(process.env.CDP_CONNECT_TIMEOUT_MS ?? '60000', 10);
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9223', { timeout: cdpConnectTimeoutMs });

  try {
    const context = browser.contexts()[0];
    if (!context) {
      throw new Error('Could not find a desktop browser context');
    }

    const installMock = async (targetPage) => targetPage.addInitScript(({ stateKey, defaultSourcePath }) => {
      const emptyState = () => ({
        lifecycle: 'idle',
        lastRequest: null,
        restartCount: 0,
      });

      const readState = () => {
        try {
          const raw = window.localStorage.getItem(stateKey);
          if (!raw) {
            return emptyState();
          }
          const parsed = JSON.parse(raw);
          return {
            ...emptyState(),
            ...parsed,
          };
        } catch {
          return emptyState();
        }
      };

      const writeState = (nextState) => {
        window.localStorage.setItem(stateKey, JSON.stringify(nextState));
        return nextState;
      };

      const currentJobId = (state) => state.restartCount > 0 ? `desktop-restarted-job-${state.restartCount}` : 'desktop-running-job';
      const currentOutputPath = (state) => state.lastRequest?.outputPath ?? 'C:/exports/desktop-running-output.mp4';
      const currentScratchPath = (state) => `C:/workspace/artifacts/jobs/job_${currentJobId(state)}`;

      const buildRunDetails = (state) => ({
        request: state.lastRequest,
        sourceMedia: {
          width: 1280,
          height: 720,
          frameRate: 24,
          durationSeconds: 105,
          frameCount: 2520,
          aspectRatio: 1.7777777778,
          pixelCount: 921600,
          hasAudio: true,
          container: 'mov',
          videoCodec: 'prores',
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
          container: state.lastRequest?.container ?? 'mp4',
          videoCodec: state.lastRequest?.codec === 'h265' ? 'hevc' : 'h264',
        },
        effectiveSettings: {
          effectiveTileSize: state.lastRequest?.tileSize ?? 128,
          processedDurationSeconds: 105,
          segmentFrameLimit: 520,
          previewMode: Boolean(state.lastRequest?.previewMode),
          previewDurationSeconds: state.lastRequest?.previewDurationSeconds ?? null,
          segmentDurationSeconds: state.lastRequest?.segmentDurationSeconds ?? 10,
        },
        executionPath: 'file-io',
        videoEncoder: state.lastRequest?.codec === 'h265' ? 'libx265' : 'libx264',
        videoEncoderLabel: state.lastRequest?.codec === 'h265' ? 'HEVC (libx265)' : 'H.264 (libx264)',
        runner: state.lastRequest?.pytorchRunner ?? 'torch',
        precision: 'fp32',
        averageThroughputFps: 7.8,
        segmentCount: 6,
        segmentFrameLimit: 520,
        frameCount: 2520,
        hadAudio: true,
        runtime: {
          ffmpegPath: 'C:/tools/ffmpeg.exe',
          realesrganPath: 'C:/tools/realesrgan-ncnn-vulkan.exe',
          modelDir: 'C:/tools/models',
          rifePath: 'C:/tools/rife-ncnn-vulkan.exe',
          rifeModelRoot: 'C:/tools/rife-models',
          availableGpus: [
            { id: 0, name: 'Intel(R) Graphics', kind: 'integrated' },
            { id: 1, name: 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition', kind: 'discrete' },
          ],
          defaultGpuId: 1,
        },
      });

      const buildRunningManagedJob = (state) => ({
        jobId: currentJobId(state),
        jobKind: 'pipeline',
        label: state.restartCount > 0 ? 'Desktop Restarted Job' : 'Desktop Running Job',
        state: 'running',
        sourcePath: state.lastRequest.sourcePath,
        modelId: state.lastRequest.modelId,
        codec: state.lastRequest.codec,
        container: state.lastRequest.container,
        progress: {
          phase: 'upscaling',
          percent: state.restartCount > 0 ? 16 : 47,
          message: state.restartCount > 0
            ? 'Restarted job is processing the first segment again'
            : 'Upscaling segment 3/6 batch 2/4 (244/520 frames)',
          processedFrames: state.restartCount > 0 ? 392 : 1184,
          totalFrames: 2520,
          extractedFrames: state.restartCount > 0 ? 520 : 1560,
          upscaledFrames: state.restartCount > 0 ? 392 : 1184,
          interpolatedFrames: 0,
          encodedFrames: state.restartCount > 0 ? 0 : 320,
          remuxedFrames: 0,
          segmentIndex: state.restartCount > 0 ? 1 : 3,
          segmentCount: 6,
          segmentProcessedFrames: state.restartCount > 0 ? 392 : 244,
          segmentTotalFrames: 520,
          batchIndex: state.restartCount > 0 ? 1 : 2,
          batchCount: 4,
          elapsedSeconds: state.restartCount > 0 ? 31 : 152,
          averageFramesPerSecond: state.restartCount > 0 ? 6.9 : 7.8,
          rollingFramesPerSecond: state.restartCount > 0 ? 7.2 : 8.4,
          estimatedRemainingSeconds: state.restartCount > 0 ? 334 : 171,
          processRssBytes: 1024 * 1024 * 896,
          gpuMemoryUsedBytes: 1024 * 1024 * 6144,
          gpuMemoryTotalBytes: 1024 * 1024 * 24576,
          scratchSizeBytes: 1024 * 1024 * (state.restartCount > 0 ? 9 : 22),
          outputSizeBytes: 1024 * 1024 * (state.restartCount > 0 ? 1 : 7),
          extractStageSeconds: 12,
          upscaleStageSeconds: state.restartCount > 0 ? 29 : 140,
          interpolateStageSeconds: 0,
          encodeStageSeconds: state.restartCount > 0 ? 0 : 18,
          remuxStageSeconds: 0,
        },
        recordedCount: 2520,
        scratchPath: currentScratchPath(state),
        scratchStats: {
          path: currentScratchPath(state),
          exists: true,
          isDirectory: true,
          sizeBytes: 1024 * 1024 * (state.restartCount > 0 ? 9 : 22),
        },
        outputPath: currentOutputPath(state),
        outputStats: {
          path: currentOutputPath(state),
          exists: true,
          isDirectory: false,
          sizeBytes: 1024 * 1024 * (state.restartCount > 0 ? 1 : 7),
        },
        pipelineRunDetails: buildRunDetails(state),
        updatedAt: String(Math.floor(Date.now() / 1000) - 5),
      });

      const buildInterruptedManagedJob = (state) => ({
        jobId: currentJobId(state),
        jobKind: 'pipeline',
        label: 'Desktop Interrupted Job',
        state: 'interrupted',
        sourcePath: state.lastRequest.sourcePath,
        modelId: state.lastRequest.modelId,
        codec: state.lastRequest.codec,
        container: state.lastRequest.container,
        progress: {
          phase: 'upscaling',
          percent: 47,
          message: 'Worker stopped before the current segment completed',
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
          rollingFramesPerSecond: 0,
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
        scratchPath: currentScratchPath(state),
        scratchStats: {
          path: currentScratchPath(state),
          exists: true,
          isDirectory: true,
          sizeBytes: 1024 * 1024 * 22,
        },
        outputPath: currentOutputPath(state),
        outputStats: {
          path: currentOutputPath(state),
          exists: true,
          isDirectory: false,
          sizeBytes: 1024 * 1024 * 7,
        },
        pipelineRunDetails: buildRunDetails(state),
        updatedAt: String(Math.floor(Date.now() / 1000) - 2),
      });

      const buildRunningPipelineStatus = (state) => ({
        jobId: currentJobId(state),
        state: 'running',
        progress: {
          phase: 'upscaling',
          percent: state.restartCount > 0 ? 16 : 47,
          message: state.restartCount > 0
            ? 'Restarted job is processing the first segment again'
            : 'Upscaling segment 3/6 batch 2/4 (244/520 frames)',
          processedFrames: state.restartCount > 0 ? 392 : 1184,
          totalFrames: 2520,
          extractedFrames: state.restartCount > 0 ? 520 : 1560,
          upscaledFrames: state.restartCount > 0 ? 392 : 1184,
          interpolatedFrames: 0,
          encodedFrames: state.restartCount > 0 ? 0 : 320,
          remuxedFrames: 0,
          segmentIndex: state.restartCount > 0 ? 1 : 3,
          segmentCount: 6,
          segmentProcessedFrames: state.restartCount > 0 ? 392 : 244,
          segmentTotalFrames: 520,
          batchIndex: state.restartCount > 0 ? 1 : 2,
          batchCount: 4,
          elapsedSeconds: state.restartCount > 0 ? 31 : 152,
          averageFramesPerSecond: state.restartCount > 0 ? 6.9 : 7.8,
          rollingFramesPerSecond: state.restartCount > 0 ? 7.2 : 8.4,
          estimatedRemainingSeconds: state.restartCount > 0 ? 334 : 171,
          processRssBytes: 1024 * 1024 * 896,
          gpuMemoryUsedBytes: 1024 * 1024 * 6144,
          gpuMemoryTotalBytes: 1024 * 1024 * 24576,
          scratchSizeBytes: 1024 * 1024 * (state.restartCount > 0 ? 9 : 22),
          outputSizeBytes: 1024 * 1024 * (state.restartCount > 0 ? 1 : 7),
          extractStageSeconds: 12,
          upscaleStageSeconds: state.restartCount > 0 ? 29 : 140,
          interpolateStageSeconds: 0,
          encodeStageSeconds: state.restartCount > 0 ? 0 : 18,
          remuxStageSeconds: 0,
        },
        result: null,
        error: null,
      });

      if (!window.localStorage.getItem(stateKey)) {
        writeState(emptyState());
      }

      window.__UPSCALER_DESKTOP_TEST__ = {
        interruptActiveJob() {
          const state = readState();
          if (!state.lastRequest) {
            return state;
          }
          return writeState({
            ...state,
            lifecycle: 'interrupted',
          });
        },
        readState() {
          return readState();
        },
      };

      window.__UPSCALER_TEST_STATE__ = {
        lastRequest: null,
      };

      window.__UPSCALER_MOCK__ = {
        async selectVideoFile() {
          return defaultSourcePath;
        },
        async selectOutputFile(_defaultPath, container) {
          return `C:/exports/desktop-jobs-restart.${container}`;
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
        async probeSourceVideo(sourcePath) {
          return {
            path: sourcePath,
            previewPath: 'C:/fixtures/sample-input-preview.mp4',
            width: 1280,
            height: 720,
            durationSeconds: 105,
            frameRate: 24,
            hasAudio: true,
            container: 'mov',
            videoCodec: 'prores',
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
              interpolatedFrames: 0,
              encodedFrames: 0,
              remuxedFrames: 0,
            },
            result: {
              path: 'C:/fixtures/sample-input_fastprep.mp4',
              previewPath: 'C:/fixtures/sample-input_fastprep.mp4',
              width: 1280,
              height: 720,
              durationSeconds: 105,
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
          const current = readState();
          const next = writeState({
            lifecycle: 'running',
            lastRequest: request,
            restartCount: current.lifecycle === 'interrupted' ? current.restartCount + 1 : current.restartCount,
          });
          window.__UPSCALER_TEST_STATE__.lastRequest = request;
          return currentJobId(next);
        },
        async getPipelineJob() {
          const state = readState();
          if (!state.lastRequest || state.lifecycle !== 'running') {
            throw new Error('Mock worker exited unexpectedly');
          }
          return buildRunningPipelineStatus(state);
        },
        async cancelPipelineJob() {
          const state = readState();
          writeState({
            ...state,
            lifecycle: 'interrupted',
          });
        },
        async getPathStats(path) {
          const normalized = String(path);
          const isDirectory = /\/job_desktop-|\/artifacts\/jobs$|\/converted-sources$|\/source-previews$/i.test(normalized.replace(/\\/g, '/'));
          return {
            path: normalized,
            exists: true,
            isDirectory,
            sizeBytes: isDirectory ? 1024 * 1024 * 9 : 1024 * 1024 * 3,
          };
        },
        async getScratchStorageSummary() {
          return {
            jobsRoot: { path: 'C:/workspace/artifacts/jobs', exists: true, isDirectory: true, sizeBytes: 1024 * 1024 * 24 },
            convertedSourcesRoot: { path: 'C:/workspace/artifacts/runtime/converted-sources', exists: true, isDirectory: true, sizeBytes: 1024 * 1024 * 4 },
            sourcePreviewsRoot: { path: 'C:/workspace/artifacts/runtime/source-previews', exists: true, isDirectory: true, sizeBytes: 1024 * 1024 * 2 },
          };
        },
        async listManagedJobs() {
          const state = readState();
          if (!state.lastRequest) {
            return [];
          }
          if (state.lifecycle === 'interrupted') {
            return [buildInterruptedManagedJob(state)];
          }
          if (state.lifecycle === 'running') {
            return [buildRunningManagedJob(state)];
          }
          return [];
        },
        async deleteManagedPath() {},
        async openPathInDefaultApp() {},
        toPreviewSrc(path) {
          return path;
        },
      };
    }, { stateKey: STATE_KEY, defaultSourcePath: DEFAULT_SOURCE_PATH });

    await expect.poll(
      () => context.pages().map((candidate) => candidate.url()),
      { timeout: 10000 },
    ).toContainEqual(expect.stringContaining('localhost:1420'));

    const page = context.pages().find((candidate) => candidate.url().includes('localhost:1420') && !candidate.url().includes('view=jobs'));
    if (!page) {
      throw new Error('Could not find the Upscaler desktop webview target');
    }

    await installMock(page);
    await page.bringToFront();
    await page.evaluate((stateKey) => {
      window.localStorage.clear();
      window.sessionStorage.clear();
      window.localStorage.removeItem(stateKey);
    }, STATE_KEY).catch(() => {});
    await page.reload({ waitUntil: 'domcontentloaded' });

    await page.getByTestId('select-video-button').click();
    await page.getByTestId('save-output-button').click();
    await page.getByTestId('run-upscale-button').click();

    const initialState = await expect.poll(async () => {
      const state = await page.evaluate(() => window.__UPSCALER_DESKTOP_TEST__?.readState() ?? null);
      return state?.lastRequest ? state : null;
    }, { timeout: 10000 }).not.toBeNull();

    const expectedOutputPath = await page.evaluate(() => window.__UPSCALER_DESKTOP_TEST__.readState().lastRequest.outputPath);

    let jobsPage = context.pages().find((candidate) => candidate.url().includes('view=jobs'));
    if (!jobsPage) {
      await page.getByTestId('job-cleanup-panel-toggle').click();
      await expect.poll(
        () => context.pages().map((candidate) => candidate.url()),
        { timeout: 10000 },
      ).toContainEqual(expect.stringContaining('view=jobs'));
      jobsPage = context.pages().find((candidate) => candidate.url().includes('view=jobs'));
    }

    if (!jobsPage) {
      throw new Error('Detached Jobs window did not appear');
    }

    await installMock(jobsPage);
    await jobsPage.reload({ waitUntil: 'domcontentloaded' });
    await jobsPage.waitForLoadState('domcontentloaded');
    await jobsPage.bringToFront();
    await expect(jobsPage.getByTestId('cleanup-job-desktop-running-job')).toContainText('Desktop Running Job');
    await expect(jobsPage.getByTestId('cleanup-input-desktop-running-job')).toContainText('desktop-running-input.mov');
    await expect(jobsPage.getByTestId('cleanup-output-desktop-running-job')).toContainText('desktop-jobs-restart.mp4');
    await jobsPage.getByTestId('cleanup-expand-desktop-running-job').click();
    await expect(jobsPage.getByTestId('cleanup-details-desktop-running-job')).toContainText(`Input Path: ${DEFAULT_SOURCE_PATH}`);
    await expect(jobsPage.getByTestId('cleanup-details-desktop-running-job')).toContainText(`Output Path: ${expectedOutputPath}`);
    await expect(jobsPage.getByTestId('cleanup-details-desktop-running-job')).toContainText('Phase: upscaling');
    await expect(jobsPage.getByTestId('cleanup-details-desktop-running-job')).toContainText('Average Throughput: 7.80 fps');
    await expect(jobsPage.getByTestId('cleanup-details-desktop-running-job')).toContainText('Current Throughput: 8.40 fps');

    await page.evaluate(() => window.__UPSCALER_DESKTOP_TEST__.interruptActiveJob());
    await expect.poll(async () => {
      const state = await page.evaluate(() => window.__UPSCALER_DESKTOP_TEST__.readState());
      return state.lifecycle;
    }, { timeout: 10000 }).toBe('interrupted');
    await expect(jobsPage.getByTestId('cleanup-job-desktop-running-job')).toContainText('Desktop Interrupted Job', { timeout: 10000 });
    await expect(jobsPage.getByTestId('cleanup-row-restart-desktop-running-job')).toBeVisible();
    await jobsPage.getByTestId('cleanup-row-restart-desktop-running-job').click();

    await expect.poll(async () => {
      const state = await page.evaluate(() => window.__UPSCALER_DESKTOP_TEST__.readState());
      return { lifecycle: state.lifecycle, restartCount: state.restartCount, outputPath: state.lastRequest?.outputPath };
    }, { timeout: 10000 }).toEqual({
      lifecycle: 'running',
      restartCount: 1,
      outputPath: expectedOutputPath,
    });

    await jobsPage.reload({ waitUntil: 'domcontentloaded' });
    await jobsPage.waitForLoadState('domcontentloaded');
    await expect(jobsPage.getByTestId('cleanup-job-desktop-restarted-job-1')).toContainText('Desktop Restarted Job');
    await jobsPage.getByTestId('cleanup-expand-desktop-restarted-job-1').click();
    await expect(jobsPage.getByTestId('cleanup-details-desktop-restarted-job-1')).toContainText(`Input Path: ${DEFAULT_SOURCE_PATH}`);
    await expect(jobsPage.getByTestId('cleanup-details-desktop-restarted-job-1')).toContainText(`Output Path: ${expectedOutputPath}`);
    await expect(jobsPage.getByTestId('cleanup-details-desktop-restarted-job-1')).toContainText('Phase: upscaling');

    console.log(JSON.stringify({
      jobsWindowVerified: true,
      interruptedAndRestarted: true,
      outputPath: expectedOutputPath,
    }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
});