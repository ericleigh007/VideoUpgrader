import fs from 'node:fs/promises';
import path from 'node:path';

import { chromium, expect } from '@playwright/test';

const repoRoot = process.cwd();
const cdpUrl = process.env.CDP_URL ?? 'http://127.0.0.1:9223';
const sourcePath = process.env.REAL_SOURCE_PATH ?? path.join(repoRoot, 'public/fixtures/gui-progress-sample.mp4');
const outputRoot = process.env.GUI_BENCHMARK_OUTPUT_ROOT ?? path.join(repoRoot, 'artifacts/outputs/gui-pipeline-benchmark');
const reportPath = process.env.GUI_BENCHMARK_REPORT_PATH ?? path.join(repoRoot, 'artifacts/benchmarks/gui-pipeline-benchmark.json');
const previewDurationSeconds = Number.parseInt(process.env.GUI_BENCHMARK_PREVIEW_SECONDS ?? '1', 10);
const runTimeoutMs = Number.parseInt(process.env.GUI_BENCHMARK_RUN_TIMEOUT_MS ?? '900000', 10);
const targetWidth = process.env.GUI_BENCHMARK_TARGET_WIDTH ?? '640';
const targetHeight = process.env.GUI_BENCHMARK_TARGET_HEIGHT ?? '360';
const scenarioSet = process.env.GUI_BENCHMARK_SCENARIO_SET ?? 'default';

const defaultScenarios = [
  {
    id: 'upscale-realesrgan-h264',
    modelId: 'realesrgan-x4plus',
    upscale: true,
    denoise: false,
    colorize: false,
    interpolation: false,
    codec: 'h264',
    container: 'mp4',
  },
  {
    id: 'upscale-realesrgan-h265',
    modelId: 'realesrgan-x4plus',
    upscale: true,
    denoise: false,
    colorize: false,
    interpolation: false,
    codec: 'h265',
    container: 'mp4',
  },
  {
    id: 'afterupscale-realesrgan-rife-h265',
    modelId: 'realesrgan-x4plus',
    upscale: true,
    denoise: false,
    colorize: false,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h265',
    container: 'mp4',
  },
  {
    id: 'interpolate-only-rife-h264',
    modelId: 'realesrgan-x4plus',
    upscale: false,
    denoise: false,
    colorize: false,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h264',
    container: 'mp4',
  },
  {
    id: 'denoise-hqdn3d-realesrgan-h264',
    modelId: 'realesrgan-x4plus',
    denoiserModelId: 'ffmpeg-hqdn3d-balanced',
    upscale: true,
    denoise: true,
    colorize: false,
    interpolation: false,
    codec: 'h264',
    container: 'mp4',
  },
  {
    id: 'denoise-drunet-realesrgan-rife-h265',
    modelId: 'realesrgan-x4plus',
    denoiserModelId: 'drunet-gray-color-denoise',
    upscale: true,
    denoise: true,
    colorize: false,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h265',
    container: 'mp4',
  },
  {
    id: 'upscale-pytorch-realesrnet-h264',
    modelId: 'realesrnet-x4plus',
    upscale: true,
    denoise: false,
    colorize: false,
    interpolation: false,
    codec: 'h264',
    container: 'mp4',
  },
  {
    id: 'colorize-only-deepremaster-h264',
    modelId: 'realesrgan-x4plus',
    colorizerModelId: 'deepremaster',
    upscale: false,
    denoise: false,
    colorize: true,
    interpolation: false,
    codec: 'h264',
    container: 'mp4',
  },
  {
    id: 'colorize-deepremaster-realesrgan-h264',
    modelId: 'realesrgan-x4plus',
    colorizerModelId: 'deepremaster',
    upscale: true,
    denoise: false,
    colorize: true,
    interpolation: false,
    codec: 'h264',
    container: 'mp4',
  },
];

const highMemory4kScenarios = [
  {
    id: '4k-realesrgan-rife-h264',
    modelId: 'realesrgan-x4plus',
    upscale: true,
    denoise: false,
    colorize: false,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h264',
    container: 'mp4',
  },
  {
    id: '4k-realesrgan-rife-h265',
    modelId: 'realesrgan-x4plus',
    upscale: true,
    denoise: false,
    colorize: false,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h265',
    container: 'mp4',
  },
  {
    id: '4k-hqdn3d-realesrgan-rife-h265',
    modelId: 'realesrgan-x4plus',
    denoiserModelId: 'ffmpeg-hqdn3d-balanced',
    upscale: true,
    denoise: true,
    colorize: false,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h265',
    container: 'mp4',
  },
  {
    id: '4k-drunet-realesrgan-rife-h265',
    modelId: 'realesrgan-x4plus',
    denoiserModelId: 'drunet-gray-color-denoise',
    upscale: true,
    denoise: true,
    colorize: false,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h265',
    container: 'mp4',
  },
  {
    id: '4k-pytorch-realesrnet-rife-h264',
    modelId: 'realesrnet-x4plus',
    upscale: true,
    denoise: false,
    colorize: false,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h264',
    container: 'mp4',
  },
  {
    id: '4k-deepremaster-realesrgan-rife-h264',
    modelId: 'realesrgan-x4plus',
    colorizerModelId: 'deepremaster',
    upscale: true,
    denoise: false,
    colorize: true,
    interpolation: true,
    interpolationTargetFps: '60',
    codec: 'h264',
    container: 'mp4',
  },
];

const scenarioSets = {
  default: defaultScenarios,
  'high-memory-4k': highMemory4kScenarios,
};

const scenarios = scenarioSets[scenarioSet];
if (!scenarios) {
  throw new Error(`Unknown GUI_BENCHMARK_SCENARIO_SET '${scenarioSet}'. Available sets: ${Object.keys(scenarioSets).join(', ')}`);
}

function stageFps(result, progress = null) {
  const timings = result?.stageTimings ?? {};
  const stageSeconds = (stageName) => timings[`${stageName}Seconds`] ?? progress?.[`${stageName}StageSeconds`] ?? null;
  const sourceFrames = result?.interpolationDiagnostics?.sourceFrameCount ?? result?.frameCount ?? progress?.totalFrames ?? null;
  const outputFrames = result?.interpolationDiagnostics?.outputFrameCount ?? result?.frameCount ?? progress?.totalFrames ?? null;
  const fpsFor = (frames, seconds) => {
    const frameCount = Number(frames);
    const elapsed = Number(seconds);
    if (!Number.isFinite(frameCount) || !Number.isFinite(elapsed) || elapsed <= 0) {
      return null;
    }
    return Number((frameCount / elapsed).toFixed(6));
  };
  return {
    extractFps: fpsFor(progress?.extractedFrames ?? sourceFrames, stageSeconds('extract')),
    denoiseFps: fpsFor(progress?.denoisedFrames ?? sourceFrames, stageSeconds('denoise')),
    colorizeFps: fpsFor(progress?.colorizedFrames ?? sourceFrames, stageSeconds('colorize')),
    upscaleFps: fpsFor(progress?.upscaledFrames ?? sourceFrames, stageSeconds('upscale')),
    interpolateFps: fpsFor(progress?.interpolatedFrames ?? outputFrames, stageSeconds('interpolate')),
    encodeFps: fpsFor(progress?.encodedFrames ?? outputFrames, stageSeconds('encode')),
    remuxFps: fpsFor(progress?.remuxedFrames ?? outputFrames, stageSeconds('remux')),
  };
}

async function getDesktopPage(context) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 60000) {
    const page = context.pages().find((candidate) => candidate.url().includes('localhost:1420') && !candidate.url().includes('view='));
    if (page) {
      return page;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  const pageUrls = context.pages().map((candidate) => candidate.url());
  throw new Error(`Could not find the Upscaler desktop webview target. Pages: ${JSON.stringify(pageUrls)}`);
}

async function invokeTauri(page, command, args = {}) {
  return page.evaluate(async ({ command, args }) => {
    if (!(window.__TAURI_INTERNALS__ && typeof window.__TAURI_INTERNALS__.invoke === 'function')) {
      throw new Error('Tauri invoke bridge is unavailable');
    }
    return window.__TAURI_INTERNALS__.invoke(command, args);
  }, { command, args });
}

async function readManagedJob(page, outputPath) {
  return page.evaluate(async ({ outputPath }) => {
    if (!(window.__TAURI_INTERNALS__ && typeof window.__TAURI_INTERNALS__.invoke === 'function')) {
      return null;
    }
    const jobs = await window.__TAURI_INTERNALS__.invoke('list_managed_jobs');
    if (!Array.isArray(jobs)) {
      return null;
    }
    return jobs.find((job) => job?.jobKind === 'pipeline' && job?.outputPath === outputPath) ?? null;
  }, { outputPath });
}

async function snapshot(page, outputPath) {
  const managedJob = await readManagedJob(page, outputPath);
  return page.evaluate(({ managedJob }) => ({
    errorText: document.querySelector('[data-testid="error-text"]')?.textContent?.trim() ?? null,
    topStatus: document.querySelector('[data-testid="top-status-panel"]')?.textContent?.trim() ?? null,
    runDisabledReason: document.querySelector('[data-testid="run-disabled-reason"]')?.textContent?.trim() ?? null,
    resultOutputPath: document.querySelector('[data-testid="result-output-path"]')?.textContent?.trim() ?? null,
    progressMessage: document.querySelector('[data-testid="progress-message"]')?.textContent?.trim() ?? null,
    liveSummary: document.querySelector('[data-testid="progress-live-summary"]')?.textContent?.trim() ?? null,
    managedJob,
  }), { managedJob });
}

async function setSwitch(page, testId, desired) {
  const control = page.getByTestId(testId);
  const current = (await control.getAttribute('aria-checked')) === 'true';
  if (current !== desired) {
    await control.click();
    await expect(control).toHaveAttribute('aria-checked', String(desired));
  }
}

async function optionExists(page, testId, value) {
  const options = await page.getByTestId(testId).locator('option').evaluateAll((nodes) => nodes.map((node) => node.value));
  return options.includes(value);
}

async function selectIfPresent(page, testId, value) {
  if (!value) {
    return false;
  }
  if (!(await optionExists(page, testId, value))) {
    return false;
  }
  await page.getByTestId(testId).selectOption(value);
  return true;
}

async function configureScenario(page, scenario, outputPath) {
  await setSwitch(page, 'pipeline-toggle-denoise', Boolean(scenario.denoise));
  if (scenario.denoise && !(await selectIfPresent(page, 'denoiser-model-select', scenario.denoiserModelId))) {
    return { skipped: true, reason: `Denoiser option is missing: ${scenario.denoiserModelId}` };
  }
  if (scenario.denoise) {
    await expect(page.getByTestId('selected-denoiser-status')).toHaveText(/runnable/i, { timeout: 120000 });
  }

  await setSwitch(page, 'pipeline-toggle-colorization', Boolean(scenario.colorize));
  if (scenario.colorize && !(await selectIfPresent(page, 'colorizer-model-select', scenario.colorizerModelId))) {
    return { skipped: true, reason: `Colorizer option is missing: ${scenario.colorizerModelId}` };
  }

  await setSwitch(page, 'pipeline-toggle-upscale', Boolean(scenario.upscale));
  if (scenario.upscale) {
    await page.getByTestId('model-select').waitFor({ timeout: 30000 });
    if (!(await selectIfPresent(page, 'model-select', scenario.modelId))) {
      return { skipped: true, reason: `Model option is missing: ${scenario.modelId}` };
    }
    await expect(page.getByTestId('selected-model-status')).toHaveText(/runnable/i, { timeout: 120000 });
  }

  await setSwitch(page, 'pipeline-toggle-interpolation', Boolean(scenario.interpolation));
  if (scenario.interpolation && scenario.interpolationTargetFps) {
    await page.getByTestId('frame-rate-target-select').selectOption(scenario.interpolationTargetFps);
  }

  let selectedGpu = null;
  if (scenario.upscale) {
    const gpuSelect = page.getByTestId('gpu-select');
    const gpuOptions = await gpuSelect.locator('option').evaluateAll((options) => options.map((option) => ({ value: option.value, text: option.textContent ?? '' })));
    const preferredGpu = gpuOptions.find((option) => option.value === '1') ?? gpuOptions.find((option) => option.value.length > 0);
    if (preferredGpu) {
      await gpuSelect.selectOption(preferredGpu.value);
      selectedGpu = preferredGpu.value;
    }

    await page.getByTestId('output-mode-select').selectOption('preserveAspect4k');
    await page.getByTestId('resolution-basis-select').selectOption('exact');
    await page.getByTestId('target-width-input').fill(targetWidth);
    await page.getByTestId('target-height-input').fill(targetHeight);
  }

  await page.getByTestId('codec-select').selectOption(scenario.codec);
  await page.getByTestId('container-select').selectOption(scenario.container);

  const previewCheckbox = page.getByTestId('preview-mode-checkbox');
  if (!(await previewCheckbox.isChecked())) {
    await previewCheckbox.check();
  }
  await page.getByTestId('preview-duration-input').fill(String(previewDurationSeconds));

  await page.evaluate((selectedOutputPath) => {
    window.__UPSCALER_BENCH_OUTPUT_PATH__ = selectedOutputPath;
  }, outputPath);
  await page.getByTestId('save-output-button').click();
  await expect.poll(
    async () => await page.getByTestId('output-path-input').inputValue(),
    { timeout: 15000 },
  ).toBe(outputPath);

  return { skipped: false, selectedGpu };
}

async function runScenario(page, scenario) {
  const outputPath = path.join(outputRoot, `${scenario.id}.${scenario.container}`);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.rm(outputPath, { force: true }).catch(() => {});

  const startedAt = Date.now();
  try {
    const config = await configureScenario(page, scenario, outputPath);
    if (config.skipped) {
      return { ...scenario, status: 'skipped', reason: config.reason, outputPath };
    }

    const launchSnapshot = await snapshot(page, outputPath);
    const runButton = page.getByTestId('run-upscale-button');
    if (!(await runButton.isEnabled())) {
      return { ...scenario, status: 'skipped', reason: `Run button disabled: ${launchSnapshot.runDisabledReason ?? launchSnapshot.topStatus ?? 'unknown'}`, outputPath, launchSnapshot };
    }

    await runButton.click();
    let lastSnapshot = null;
    while (Date.now() - startedAt < runTimeoutMs) {
      lastSnapshot = await snapshot(page, outputPath);
      if (lastSnapshot.errorText) {
        throw new Error(lastSnapshot.errorText);
      }
      const state = String(lastSnapshot.managedJob?.state ?? '');
      const progress = lastSnapshot.managedJob?.progress ?? null;
      const outputExists = await fs.stat(outputPath).then((stats) => stats.isFile() && stats.size > 0).catch(() => false);
      const guiReportsCompleted = /Pipeline completed/i.test(`${lastSnapshot.topStatus ?? ''} ${lastSnapshot.progressMessage ?? ''}`);
      if (state === 'succeeded' || lastSnapshot.resultOutputPath === outputPath || (outputExists && guiReportsCompleted)) {
        const result = lastSnapshot.managedJob?.result ?? null;
        return {
          ...scenario,
          status: 'succeeded',
          outputPath,
          selectedGpu: config.selectedGpu,
          wallSeconds: Number(((Date.now() - startedAt) / 1000).toFixed(3)),
          jobState: state,
          progress,
          result,
          stageEffectiveFps: stageFps(result, progress),
        };
      }
      if (['failed', 'cancelled', 'interrupted'].includes(state)) {
        throw new Error(`Managed job ended as ${state}: ${lastSnapshot.managedJob?.error ?? 'no error text'}`);
      }
      await page.waitForTimeout(3000);
    }
    throw new Error(`Timed out after ${runTimeoutMs}ms. Last snapshot: ${JSON.stringify(lastSnapshot)}`);
  } catch (error) {
    return {
      ...scenario,
      status: 'failed',
      outputPath,
      wallSeconds: Number(((Date.now() - startedAt) / 1000).toFixed(3)),
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function main() {
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.mkdir(outputRoot, { recursive: true });

  console.log(`[gui-benchmark] connecting to ${cdpUrl}`);
  const browser = await chromium.connectOverCDP(cdpUrl, { timeout: 60000 });
  try {
    const context = browser.contexts()[0];
    if (!context) {
      throw new Error('No browser context available from CDP session');
    }
    console.log(`[gui-benchmark] attached pages: ${context.pages().map((candidate) => candidate.url()).join(', ')}`);
    const page = await getDesktopPage(context);
    console.log(`[gui-benchmark] using page ${page.url()}`);
    await page.bringToFront();
    await page.addInitScript(({ selectedSourcePath }) => {
      window.__UPSCALER_MOCK__ = {
        async selectVideoFile() {
          return selectedSourcePath;
        },
        async selectOutputFile() {
          return window.__UPSCALER_BENCH_OUTPUT_PATH__ ?? '';
        },
      };
    }, { selectedSourcePath: sourcePath });

    await page.evaluate(({ selectedSourcePath }) => {
      window.__UPSCALER_MOCK__ = {
        async selectVideoFile() {
          return selectedSourcePath;
        },
        async selectOutputFile() {
          return window.__UPSCALER_BENCH_OUTPUT_PATH__ ?? '';
        },
      };
    }, { selectedSourcePath: sourcePath });
    await page.evaluate(() => {
      window.localStorage.clear();
      window.sessionStorage.clear();
    }).catch(() => {});
    await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 }).catch(() => {});
    await page.evaluate(({ selectedSourcePath }) => {
      window.__UPSCALER_MOCK__ = {
        async selectVideoFile() {
          return selectedSourcePath;
        },
        async selectOutputFile() {
          return window.__UPSCALER_BENCH_OUTPUT_PATH__ ?? '';
        },
      };
    }, { selectedSourcePath: sourcePath });
    await page.getByTestId('select-video-button').waitFor({ timeout: 60000 });

    console.log('[gui-benchmark] ensuring runtime assets');
    const runtime = await invokeTauri(page, 'ensure_runtime_assets').catch((error) => ({ error: String(error) }));
    console.log('[gui-benchmark] selecting source');
    await page.getByTestId('select-video-button').click();
    await page.getByTestId('source-preview').waitFor({ timeout: 180000 });

    const results = [];
    for (const scenario of scenarios) {
      console.log(`[gui-benchmark] running ${scenario.id}`);
      const result = await runScenario(page, scenario);
      results.push(result);
      console.log(`[gui-benchmark] ${scenario.id}: ${result.status}`);
      await fs.writeFile(reportPath, JSON.stringify({ sourcePath, outputRoot, reportPath, runtime, scenarioSet, previewDurationSeconds, targetWidth: Number(targetWidth), targetHeight: Number(targetHeight), results }, null, 2) + '\n', 'utf8');
    }

    const summary = {
      total: results.length,
      succeeded: results.filter((result) => result.status === 'succeeded').length,
      skipped: results.filter((result) => result.status === 'skipped').length,
      failed: results.filter((result) => result.status === 'failed').length,
    };
    const payload = { sourcePath, outputRoot, reportPath, runtime, scenarioSet, previewDurationSeconds, targetWidth: Number(targetWidth), targetHeight: Number(targetHeight), summary, results };
    await fs.writeFile(reportPath, JSON.stringify(payload, null, 2) + '\n', 'utf8');
    console.log(JSON.stringify(payload, null, 2));
  } finally {
    await browser.close();
  }
}

await main();