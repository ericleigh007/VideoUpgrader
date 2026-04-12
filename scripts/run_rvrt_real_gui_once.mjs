import { fileURLToPath } from 'node:url';

import { chromium, expect } from '@playwright/test';

const cdpUrl = process.env.CDP_URL ?? 'http://127.0.0.1:9223';
const timeoutMs = Number.parseInt(process.env.RUN_TIMEOUT_MS ?? '1200000', 10);

async function readManagedGuiJob(page) {
  const outputPath = process.env.REAL_OUTPUT_PATH;
  if (!outputPath) {
    throw new Error('REAL_OUTPUT_PATH is required');
  }

  return page.evaluate(async ({ expectedOutputPath }) => {
    if (!(window.__TAURI_INTERNALS__ && typeof window.__TAURI_INTERNALS__.invoke === 'function')) {
      return null;
    }

    const jobs = await window.__TAURI_INTERNALS__.invoke('list_managed_jobs');
    if (!Array.isArray(jobs)) {
      return null;
    }

    return jobs.find((job) => job?.jobKind === 'pipeline' && job?.outputPath === expectedOutputPath) ?? null;
  }, { expectedOutputPath: outputPath });
}

export function guiRunHasStarted(snapshot) {
  if (!snapshot || snapshot.errorText) {
    return false;
  }

  return Boolean(
    snapshot.jobProgressVisible
    || (snapshot.managedJob && ['queued', 'running', 'paused', 'succeeded'].includes(String(snapshot.managedJob.state)))
  );
}

export function guiRunHasCompleted(snapshot) {
  if (!snapshot || snapshot.errorText) {
    return false;
  }

  return Boolean(snapshot.resultOutputPath || snapshot.managedJob?.state === 'succeeded');
}

export function guiRunHasFailed(snapshot) {
  if (!snapshot) {
    return false;
  }

  return Boolean(
    snapshot.errorText
    || ['failed', 'cancelled', 'interrupted'].includes(String(snapshot.managedJob?.state ?? ''))
  );
}

async function captureGuiSnapshot(page) {
  const managedJob = await readManagedGuiJob(page);
  return page.evaluate(({ outputPath: expectedOutputPath, managedJob }) => ({
    errorText: document.querySelector('[data-testid="error-text"]')?.textContent?.trim() ?? null,
    resultOutputPath: document.querySelector('[data-testid="result-output-path"]')?.textContent?.trim() ?? null,
    progressMessage: document.querySelector('[data-testid="progress-message"]')?.textContent?.trim() ?? null,
    progressLiveSummary: document.querySelector('[data-testid="progress-live-summary"]')?.textContent?.trim() ?? null,
    topStatus: document.querySelector('[data-testid="top-status-panel"]')?.textContent?.trim() ?? null,
    currentActivity: document.querySelector('[data-testid="progress-current-activity"]')?.textContent?.trim() ?? null,
    currentDetail: document.querySelector('[data-testid="progress-current-detail"]')?.textContent?.trim() ?? null,
    elapsed: document.querySelector('[data-testid="progress-elapsed"]')?.textContent?.trim() ?? null,
    eta: document.querySelector('[data-testid="progress-eta"]')?.textContent?.trim() ?? null,
    jobProgressVisible: Boolean(document.querySelector('[data-testid="job-progress-panel"]')),
    expectedOutputPath,
    managedJob,
  }), { outputPath, managedJob });
}

async function main() {
  const sourcePath = process.env.REAL_SOURCE_PATH;
  const outputPath = process.env.REAL_OUTPUT_PATH;

  if (!sourcePath) {
    throw new Error('REAL_SOURCE_PATH is required');
  }

  if (!outputPath) {
    throw new Error('REAL_OUTPUT_PATH is required');
  }

  const browser = await chromium.connectOverCDP(cdpUrl, { timeout: 60000 });
  try {
    const context = browser.contexts()[0];
    if (!context) {
      throw new Error('No browser context available from CDP session');
    }

    await expect.poll(
      () => context.pages().map((candidate) => candidate.url()),
      { timeout: 15000 },
    ).toContainEqual(expect.stringContaining('localhost:1420'));

    const page = context.pages().find((candidate) => candidate.url().includes('localhost:1420') && !candidate.url().includes('view=jobs'));
    if (!page) {
      throw new Error('Could not find the Upscaler desktop webview target');
    }

    await page.bringToFront();
    await page.addInitScript(({ selectedSourcePath, selectedOutputPath }) => {
      window.__UPSCALER_MOCK__ = {
        async selectVideoFile() {
          return selectedSourcePath;
        },
        async selectOutputFile() {
          return selectedOutputPath;
        },
      };
    }, { selectedSourcePath: sourcePath, selectedOutputPath: outputPath });

    await page.evaluate(() => {
      window.localStorage.clear();
      window.sessionStorage.clear();
    }).catch(() => {});
    await page.reload({ waitUntil: 'domcontentloaded' });

    await page.getByTestId('select-video-button').click();
    await page.getByTestId('source-preview').waitFor({ timeout: 120000 });

    await page.getByTestId('model-select').selectOption('rvrt-x4');
    await expect(page.getByTestId('selected-model-status')).toHaveText(/runnable/i, { timeout: 120000 });

    const gpuSelect = page.getByTestId('gpu-select');
    const gpuValues = await gpuSelect.locator('option').evaluateAll((options) => options.map((option) => ({ value: option.value, text: option.textContent ?? '' })));
    const preferredGpu = gpuValues.find((option) => option.value === '1') ?? gpuValues.find((option) => option.value.length > 0);
    if (preferredGpu) {
      await gpuSelect.selectOption(preferredGpu.value);
    }

    const interpolationToggle = page.getByTestId('pipeline-toggle-interpolation');
    if ((await interpolationToggle.getAttribute('aria-checked')) === 'true') {
      await interpolationToggle.click();
      await expect(interpolationToggle).toHaveAttribute('aria-checked', 'false');
    }

    await page.getByTestId('output-mode-select').selectOption('preserveAspect4k');
    await page.getByTestId('resolution-basis-select').selectOption('exact');
    await page.getByTestId('target-width-input').fill('3840');
    await page.getByTestId('target-height-input').fill('2160');

    const previewCheckbox = page.getByTestId('preview-mode-checkbox');
    if (!(await previewCheckbox.isChecked())) {
      await previewCheckbox.check();
    }
    await page.getByTestId('preview-duration-input').fill('3');

    await page.getByTestId('save-output-button').click();
    await expect.poll(
      async () => await page.getByTestId('output-path-input').inputValue(),
      { timeout: 10000 },
    ).toBe(outputPath);

    const launchDiagnostics = await page.evaluate(() => ({
      topStatus: document.querySelector('[data-testid="top-status-panel"]')?.textContent?.trim() ?? null,
      modelStatus: document.querySelector('[data-testid="selected-model-status"]')?.textContent?.trim() ?? null,
      disabledReason: document.querySelector('[data-testid="run-disabled-reason"]')?.textContent?.trim() ?? null,
      outputPath: (document.querySelector('[data-testid="output-path-input"]') instanceof HTMLInputElement)
        ? document.querySelector('[data-testid="output-path-input"]').value
        : null,
    }));

    const runButton = page.getByTestId('run-upscale-button');
    if (!(await runButton.isEnabled())) {
      throw new Error(`Run button is disabled before launch: ${JSON.stringify(launchDiagnostics)}`);
    }

    await runButton.click();
    await expect.poll(async () => {
      const snapshot = await captureGuiSnapshot(page);
      return {
        jobProgressVisible: snapshot.jobProgressVisible,
        managedJobState: snapshot.managedJob?.state ?? null,
        managedJobPhase: snapshot.managedJob?.progress?.phase ?? null,
        errorText: snapshot.errorText,
      };
    }, { timeout: 120000 }).toMatchObject({
      errorText: null,
    });

    await expect.poll(async () => {
      const snapshot = await captureGuiSnapshot(page);
      if (guiRunHasFailed(snapshot)) {
        return { started: false, errorText: snapshot.errorText };
      }

      return {
        started: guiRunHasStarted(snapshot),
        errorText: snapshot.errorText,
        managedJobState: snapshot.managedJob?.state ?? null,
        managedJobPhase: snapshot.managedJob?.progress?.phase ?? null,
      };
    }, { timeout: 120000 }).toMatchObject({
      started: true,
      errorText: null,
    });

    const startedAt = Date.now();
    let lastProgressSnapshot = null;
    while (Date.now() - startedAt < timeoutMs) {
      const snapshot = await captureGuiSnapshot(page);
      lastProgressSnapshot = snapshot;

      if (snapshot.errorText) {
        throw new Error(`GUI pipeline failed: ${JSON.stringify(snapshot)}`);
      }

      if (guiRunHasCompleted(snapshot)) {
        console.log(JSON.stringify({
          status: 'succeeded',
          sourcePath,
          outputPath: snapshot.resultOutputPath || snapshot.managedJob?.outputPath || outputPath,
          launchDiagnostics,
          completion: snapshot,
        }, null, 2));
        return;
      }

      if (guiRunHasFailed(snapshot)) {
        throw new Error(`GUI pipeline terminated unexpectedly: ${JSON.stringify(snapshot)}`);
      }

      await page.waitForTimeout(5000);
    }

    throw new Error(`Timed out waiting for RVRT GUI run to finish. Last snapshot: ${JSON.stringify(lastProgressSnapshot)}`);
  } finally {
    await browser.close();
  }
}

const invokedPath = process.argv[1] ? fileURLToPath(new URL(`file://${process.argv[1].replace(/\\/g, '/')}`)) : null;
if (invokedPath && fileURLToPath(import.meta.url) === invokedPath) {
  await main();
}