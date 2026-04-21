import { chromium, expect } from '@playwright/test';
import path from 'node:path';

const sourcePath = process.env.REAL_SOURCE_PATH;
const contextPath = process.env.REAL_CONTEXT_PATH;
const cdpUrl = process.env.CDP_URL ?? 'http://127.0.0.1:9223';

if (!sourcePath) {
  throw new Error('REAL_SOURCE_PATH is required');
}

if (!contextPath) {
  throw new Error('REAL_CONTEXT_PATH is required');
}

async function getDesktopPage(context, timeout = 15000) {
  await expect.poll(
    () => context.pages().map((candidate) => candidate.url()),
    { timeout },
  ).toContainEqual(expect.stringContaining('localhost:1420'));

  const page = context.pages().find((candidate) => candidate.url().includes('localhost:1420') && !candidate.url().includes('view=jobs'));
  if (!page) {
    throw new Error('Could not find the Upscaler desktop webview target');
  }

  return page;
}

async function main() {
  const browser = await chromium.connectOverCDP(cdpUrl, { timeout: 60000 });
  try {
    const context = browser.contexts()[0];
    if (!context) {
      throw new Error('No browser context available from CDP session');
    }

    const page = await getDesktopPage(context);
    await page.addInitScript(
      ({ selectedSourcePath, selectedContextPath }) => {
        const existing = window.__UPSCALER_MOCK__ ?? {};
        window.__UPSCALER_MOCK__ = {
          ...existing,
          async selectVideoFile() {
            return selectedSourcePath;
          },
          async selectContextFiles() {
            return [selectedContextPath];
          },
        };
      },
      { selectedSourcePath: sourcePath, selectedContextPath: contextPath },
    );
    await page.reload({ waitUntil: 'domcontentloaded' });

    await page.getByTestId('select-video-button').click();
    await expect(page.getByText(path.basename(sourcePath), { exact: false })).toBeVisible({ timeout: 20000 });

    const initialLibrary = await page.evaluate(async (activeSourcePath) => {
      return window.__TAURI_INTERNALS__.invoke('get_source_context_library', { sourcePath: activeSourcePath });
    }, sourcePath);
    const initialCount = Array.isArray(initialLibrary?.entries) ? initialLibrary.entries.length : 0;
    const initialEntryIds = new Set(Array.isArray(initialLibrary?.entries) ? initialLibrary.entries.map((entry) => entry.entryId) : []);

    const colorizationToggle = page.getByTestId('pipeline-toggle-colorization');
    if ((await colorizationToggle.getAttribute('aria-checked')) !== 'true') {
      await colorizationToggle.click();
    }

    await page.getByTestId('colorizer-model-select').selectOption('deepremaster');
    await page.getByTestId('color-context-add-button').click();

    await expect(page.getByTestId('color-context-library-card')).toBeVisible({ timeout: 20000 });
    await expect(page.getByTestId('color-context-library-path')).toContainText('artifacts');

    try {
      await expect.poll(
        async () => {
          const library = await page.evaluate(async (activeSourcePath) => {
            return window.__TAURI_INTERNALS__.invoke('get_source_context_library', { sourcePath: activeSourcePath });
          }, sourcePath);
          return Array.isArray(library?.entries) && library.entries.some((entry) => !initialEntryIds.has(entry.entryId));
        },
        { timeout: 20000 },
      ).toBeTruthy();
    } catch (error) {
      const diagnostics = await page.evaluate(async (activeSourcePath) => {
        const library = await window.__TAURI_INTERNALS__.invoke('get_source_context_library', { sourcePath: activeSourcePath });
        return {
          statusText: document.querySelector('[data-testid="top-status-panel"]')?.textContent?.trim() ?? null,
          errorText: document.querySelector('.error-text')?.textContent?.trim() ?? null,
          summaryText: document.querySelector('[data-testid="color-context-summary"]')?.textContent?.trim() ?? null,
          library,
        };
      }, sourcePath);
      throw new Error(`Context import did not appear in the backend library: ${JSON.stringify(diagnostics)}`, { cause: error });
    }

    const importedLibrary = await page.evaluate(async (activeSourcePath) => {
      return window.__TAURI_INTERNALS__.invoke('get_source_context_library', { sourcePath: activeSourcePath });
    }, sourcePath);

    if (!importedLibrary || !Array.isArray(importedLibrary.entries)) {
      throw new Error('Context library was not returned after import');
    }

    const importedEntry = importedLibrary.entries.find((entry) => !initialEntryIds.has(entry.entryId));
    if (!importedEntry) {
      throw new Error('A newly imported context entry was not found in the source library');
    }

    if (importedLibrary.entries.length < initialCount + 1) {
      throw new Error(`Expected context entry count to grow from ${initialCount} to at least ${initialCount + 1}, received ${importedLibrary.entries.length}`);
    }

    await expect(page.getByTestId(`color-context-entry-${importedEntry.entryId}`)).toBeChecked({ timeout: 10000 });

    const summary = {
      sourcePath,
      contextPath,
      libraryId: importedLibrary.libraryId,
      refsPath: importedLibrary.refsPath,
      importedEntryId: importedEntry.entryId,
      entryCount: importedLibrary.entries.length,
    };
    console.log(JSON.stringify({ type: 'desktop-color-context-smoke', status: 'passed', summary }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});