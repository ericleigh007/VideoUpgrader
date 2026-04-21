import path from 'node:path';

import { chromium, expect } from '@playwright/test';

const baseUrl = process.env.BASE_URL ?? 'http://127.0.0.1:4173';
const docsImageRoot = process.env.DOCS_IMAGE_ROOT ?? path.resolve('docs/images');

function buildMockScriptData() {
  const sourcePath = 'C:/fixtures/sample-input.mp4';
  const previewPath = '/fixtures/gui-progress-sample.mp4';
  const libraryId = 'sample-input-7f2d6b4a';
  const refsPath = `C:/workspace/artifacts/context-libraries/${libraryId}/refs`;

  return {
    sourcePath,
    previewPath,
    library: {
      libraryId,
      sourcePath,
      sourceFileName: 'sample-input.mp4',
      sourceFingerprint: '7f2d6b4a',
      folderPath: `C:/workspace/artifacts/context-libraries/${libraryId}`,
      refsPath,
      createdAt: '2026-04-21T19:00:00.000Z',
      entries: [
        {
          entryId: 'dorothy-reference-1',
          fileName: 'dorothy_reference_a.jpg',
          relativePath: 'refs/dorothy_reference_a.jpg',
          absolutePath: `${refsPath}/dorothy_reference_a.jpg`,
          sizeBytes: 183245,
          createdAt: '2026-04-21T19:00:00.000Z',
        },
        {
          entryId: 'dorothy-reference-2',
          fileName: 'dorothy_reference_b.jpg',
          relativePath: 'refs/dorothy_reference_b.jpg',
          absolutePath: `${refsPath}/dorothy_reference_b.jpg`,
          sizeBytes: 142118,
          createdAt: '2026-04-21T19:01:00.000Z',
        },
      ],
    },
  };
}

async function installMock(page, mockData) {
  await page.addInitScript((data) => {
    const sourceContextLibrary = structuredClone(data.library);
    window.__UPSCALER_MOCK__ = {
      async selectVideoFile() {
        return data.sourcePath;
      },
      async selectContextFiles() {
        return sourceContextLibrary.entries.map((entry) => entry.absolutePath);
      },
      async selectOutputFile(_defaultPath, container) {
        return `C:/exports/readme-screenshot.${container}`;
      },
      async ensureRuntimeAssets() {
        return {
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
          externalResearchRuntimes: {
            'rvrt-x4': {
              kind: 'external-command',
              commandEnvVar: 'UPSCALER_RVRT_COMMAND',
              configured: false,
            },
          },
        };
      },
      async probeSourceVideo(sourcePath) {
        return {
          path: sourcePath,
          previewPath: data.previewPath,
          width: 1280,
          height: 720,
          durationSeconds: 12.5,
          frameRate: 24,
          hasAudio: true,
          container: 'mp4',
          videoCodec: 'h264',
        };
      },
      async getSourceContextLibrary(sourcePath) {
        return sourcePath === data.sourcePath ? structuredClone(sourceContextLibrary) : null;
      },
      async importSourceContextFiles(sourcePath, importPaths) {
        if (sourcePath !== data.sourcePath) {
          return structuredClone(sourceContextLibrary);
        }
        for (const importPath of importPaths) {
          const existing = sourceContextLibrary.entries.find((entry) => entry.absolutePath === importPath);
          if (existing) {
            continue;
          }
          const fileName = String(importPath).replace(/\\/g, '/').split('/').filter(Boolean).pop() ?? 'context.jpg';
          sourceContextLibrary.entries.push({
            entryId: `context-${sourceContextLibrary.entries.length + 1}`,
            fileName,
            relativePath: `refs/${fileName}`,
            absolutePath: `${sourceContextLibrary.refsPath}/${fileName}`,
            sizeBytes: 100000,
            createdAt: new Date().toISOString(),
          });
        }
        return structuredClone(sourceContextLibrary);
      },
      async removeSourceContextEntry(sourcePath, entryId) {
        if (sourcePath === data.sourcePath) {
          sourceContextLibrary.entries = sourceContextLibrary.entries.filter((entry) => entry.entryId !== entryId);
        }
        return structuredClone(sourceContextLibrary);
      },
      toPreviewSrc(path) {
        return path;
      },
      async loadPreviewUrl(path) {
        return path;
      },
    };
  }, mockData);
}

async function enableSwitch(page, testId) {
  const toggle = page.getByTestId(testId);
  if ((await toggle.getAttribute('aria-checked')) !== 'true') {
    await toggle.click();
  }
  await expect(toggle).toHaveAttribute('aria-checked', 'true');
}

async function captureScreenshots() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 2200 }, deviceScaleFactor: 1 });
  const mockData = buildMockScriptData();

  try {
    await installMock(page, mockData);
    await page.goto(baseUrl, { waitUntil: 'networkidle' });

    await page.getByTestId('select-video-button').click();
    await expect(page.getByTestId('source-preview')).toBeVisible();

    await enableSwitch(page, 'pipeline-toggle-colorization');
    await page.getByTestId('colorizer-model-select').selectOption('deepremaster');
    await page.getByTestId('deepremaster-processing-mode-select').selectOption('high');
    await enableSwitch(page, 'pipeline-toggle-upscale');
    await enableSwitch(page, 'pipeline-toggle-interpolation');

    await expect(page.getByTestId('color-context-library-card')).toBeVisible();
    await page.getByTestId('color-context-entry-dorothy-reference-1').check();
    await page.getByTestId('color-context-entry-dorothy-reference-2').check();
    await page.getByTestId('frame-rate-target-select').selectOption('60');

    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({
      path: path.join(docsImageRoot, 'main-page-top.png'),
      animations: 'disabled',
      caret: 'hide',
    });

    await page.getByTestId('pipeline-colorization-details').screenshot({
      path: path.join(docsImageRoot, 'right-hand-model-selector.png'),
      animations: 'disabled',
      caret: 'hide',
    });

    await page.getByTestId('processing-track-grid').screenshot({
      path: path.join(docsImageRoot, 'right-hand-w-encoder-and-interpolation.png'),
      animations: 'disabled',
      caret: 'hide',
    });
  } finally {
    await browser.close();
  }
}

captureScreenshots().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});