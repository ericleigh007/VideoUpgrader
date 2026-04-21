import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium, expect } from '@playwright/test';

const cdpUrl = process.env.CDP_URL ?? 'http://127.0.0.1:9223';
const timeoutMs = Number.parseInt(process.env.RUN_TIMEOUT_MS ?? '1200000', 10);
const previewDurationSeconds = process.env.REAL_PREVIEW_DURATION_SECONDS ?? '3';

function parseContextPaths(rawValue) {
  return String(rawValue ?? '')
    .split(path.delimiter)
    .map((value) => value.trim())
    .filter(Boolean);
}

async function readManagedGuiJob(page, outputPath) {
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

async function readContextLibrary(page, sourcePath) {
  return page.evaluate(async ({ activeSourcePath }) => {
    if (!(window.__TAURI_INTERNALS__ && typeof window.__TAURI_INTERNALS__.invoke === 'function')) {
      return null;
    }

    return window.__TAURI_INTERNALS__.invoke('get_source_context_library', { sourcePath: activeSourcePath });
  }, { activeSourcePath: sourcePath });
}

function guiRunHasStarted(snapshot) {
  if (!snapshot || snapshot.errorText) {
    return false;
  }

  return Boolean(
    snapshot.jobProgressVisible
    || (snapshot.managedJob && ['queued', 'running', 'paused', 'succeeded'].includes(String(snapshot.managedJob.state)))
  );
}

function guiRunHasCompleted(snapshot) {
  if (!snapshot || snapshot.errorText) {
    return false;
  }

  return Boolean(snapshot.resultOutputPath || snapshot.managedJob?.state === 'succeeded');
}

function guiRunHasFailed(snapshot) {
  if (!snapshot) {
    return false;
  }

  return Boolean(
    snapshot.errorText
    || ['failed', 'cancelled', 'interrupted'].includes(String(snapshot.managedJob?.state ?? ''))
  );
}

async function captureGuiSnapshot(page, outputPath) {
  const managedJob = await readManagedGuiJob(page, outputPath);
  return page.evaluate(({ expectedOutputPath, managedJob }) => ({
    errorText: document.querySelector('[data-testid="error-text"]')?.textContent?.trim() ?? null,
    resultOutputPath: document.querySelector('[data-testid="result-output-path"]')?.textContent?.trim() ?? null,
    progressMessage: document.querySelector('[data-testid="progress-message"]')?.textContent?.trim() ?? null,
    progressLiveSummary: document.querySelector('[data-testid="progress-live-summary"]')?.textContent?.trim() ?? null,
    topStatus: document.querySelector('[data-testid="top-status-panel"]')?.textContent?.trim() ?? null,
    currentActivity: document.querySelector('[data-testid="progress-current-activity"]')?.textContent?.trim() ?? null,
    currentDetail: document.querySelector('[data-testid="progress-current-detail"]')?.textContent?.trim() ?? null,
    pipelineLaunchState: document.querySelector('[data-testid="pipeline-launch-state"]')?.getAttribute('data-state') ?? null,
    colorContextSummary: document.querySelector('[data-testid="color-context-summary"]')?.textContent?.trim() ?? null,
    selectedColorizerStatus: document.querySelector('[data-testid="selected-colorizer-launch-requirement"]')?.textContent?.trim() ?? null,
    jobProgressVisible: Boolean(document.querySelector('[data-testid="job-progress-panel"]')),
    expectedOutputPath,
    managedJob,
  }), { expectedOutputPath: outputPath, managedJob });
}

async function main() {
  const sourcePath = process.env.REAL_SOURCE_PATH;
  const outputPath = process.env.REAL_OUTPUT_PATH;
  const contextPaths = parseContextPaths(process.env.REAL_CONTEXT_PATHS);

  if (!sourcePath) {
    throw new Error('REAL_SOURCE_PATH is required');
  }

  if (!outputPath) {
    throw new Error('REAL_OUTPUT_PATH is required');
  }

  if (contextPaths.length === 0) {
    throw new Error('REAL_CONTEXT_PATHS is required');
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
    await page.addInitScript(({ selectedSourcePath, selectedOutputPath, selectedContextPaths }) => {
      const existing = window.__UPSCALER_MOCK__ ?? {};
      window.__UPSCALER_MOCK__ = {
        ...existing,
        async selectVideoFile() {
          return selectedSourcePath;
        },
        async selectOutputFile() {
          return selectedOutputPath;
        },
        async selectContextFiles() {
          return selectedContextPaths;
        },
      };
    }, { selectedSourcePath: sourcePath, selectedOutputPath: outputPath, selectedContextPaths: contextPaths });

    await page.evaluate(() => {
      window.localStorage.clear();
      window.sessionStorage.clear();
    }).catch(() => {});
    await page.reload({ waitUntil: 'domcontentloaded' });

    await page.getByTestId('select-video-button').click();
    await page.getByTestId('source-preview').waitFor({ timeout: 120000 });

    const colorizationToggle = page.getByTestId('pipeline-toggle-colorization');
    if ((await colorizationToggle.getAttribute('aria-checked')) !== 'true') {
      await colorizationToggle.click();
      await expect(colorizationToggle).toHaveAttribute('aria-checked', 'true');
    }

    const upscaleToggle = page.getByTestId('pipeline-toggle-upscale');
    if ((await upscaleToggle.getAttribute('aria-checked')) === 'true') {
      await upscaleToggle.click();
      await expect(upscaleToggle).toHaveAttribute('aria-checked', 'false');
    }

    const interpolationToggle = page.getByTestId('pipeline-toggle-interpolation');
    if ((await interpolationToggle.getAttribute('aria-checked')) === 'true') {
      await interpolationToggle.click();
      await expect(interpolationToggle).toHaveAttribute('aria-checked', 'false');
    }

    await page.getByTestId('colorizer-model-select').selectOption('deepremaster');

    const previewCheckbox = page.getByTestId('preview-mode-checkbox');
    if (!(await previewCheckbox.isChecked())) {
      await previewCheckbox.check();
    }
    await page.getByTestId('preview-duration-input').fill(previewDurationSeconds);

    await page.getByTestId('save-output-button').click();
    await expect.poll(
      async () => await page.getByTestId('output-path-input').inputValue(),
      { timeout: 10000 },
    ).toBe(outputPath);

    const initialLibrary = await readContextLibrary(page, sourcePath);
    const initialEntryIds = new Set(Array.isArray(initialLibrary?.entries) ? initialLibrary.entries.map((entry) => entry.entryId) : []);

    await page.getByTestId('color-context-add-button').click();
    await page.getByTestId('color-context-library-card').waitFor({ timeout: 30000 });

    await expect.poll(async () => {
      const library = await readContextLibrary(page, sourcePath);
      const entries = Array.isArray(library?.entries) ? library.entries : [];
      const newlyImportedCount = entries.filter((entry) => !initialEntryIds.has(entry.entryId)).length;
      const selectedCount = await page.locator('[data-testid^="color-context-entry-"]').evaluateAll((nodes) => {
        return nodes.filter((node) => node instanceof HTMLInputElement && node.checked).length;
      });
      return { entryCount: entries.length, newlyImportedCount, selectedCount, libraryId: library?.libraryId ?? null };
    }, { timeout: 30000 }).toMatchObject({
      newlyImportedCount: contextPaths.length,
      selectedCount: contextPaths.length,
    });

    const launchDiagnostics = await page.evaluate(() => ({
      topStatus: document.querySelector('[data-testid="top-status-panel"]')?.textContent?.trim() ?? null,
      colorContextSummary: document.querySelector('[data-testid="color-context-summary"]')?.textContent?.trim() ?? null,
      selectedModelStatus: document.querySelector('[data-testid="selected-model-status"]')?.textContent?.trim() ?? null,
      selectedColorizerReason: document.querySelector('[data-testid="selected-colorizer-launch-requirement"]')?.textContent?.trim() ?? null,
      disabledReason: document.querySelector('[data-testid="run-disabled-reason"]')?.textContent?.trim() ?? null,
      disabledColorizerReason: document.querySelector('[data-testid="run-disabled-colorizer-reason"]')?.textContent?.trim() ?? null,
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
      const snapshot = await captureGuiSnapshot(page, outputPath);
      if (guiRunHasFailed(snapshot)) {
        return { started: false, errorText: snapshot.errorText, managedJobState: snapshot.managedJob?.state ?? null };
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
    let lastSnapshot = null;
    while (Date.now() - startedAt < timeoutMs) {
      const snapshot = await captureGuiSnapshot(page, outputPath);
      lastSnapshot = snapshot;

      if (snapshot.errorText) {
        throw new Error(`GUI pipeline failed: ${JSON.stringify(snapshot)}`);
      }

      if (guiRunHasCompleted(snapshot)) {
        const library = await readContextLibrary(page, sourcePath);
        console.log(JSON.stringify({
          status: 'succeeded',
          sourcePath,
          outputPath: snapshot.resultOutputPath || snapshot.managedJob?.outputPath || outputPath,
          contextPaths,
          launchDiagnostics,
          contextLibrary: {
            libraryId: library?.libraryId ?? null,
            entryCount: Array.isArray(library?.entries) ? library.entries.length : 0,
            refsPath: library?.refsPath ?? null,
          },
          completion: snapshot,
        }, null, 2));
        return;
      }

      if (guiRunHasFailed(snapshot)) {
        throw new Error(`GUI pipeline terminated unexpectedly: ${JSON.stringify(snapshot)}`);
      }

      await page.waitForTimeout(5000);
    }

    throw new Error(`Timed out waiting for DeepRemaster GUI run to finish. Last snapshot: ${JSON.stringify(lastSnapshot)}`);
  } finally {
    await browser.close();
  }
}

const invokedPath = process.argv[1] ? fileURLToPath(new URL(`file://${process.argv[1].replace(/\\/g, '/')}`)) : null;
if (invokedPath && fileURLToPath(import.meta.url) === invokedPath) {
  await main();
}