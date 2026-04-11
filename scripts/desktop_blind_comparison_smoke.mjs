import { chromium, expect } from '@playwright/test';
import path from 'node:path';

const sourcePath = process.env.REAL_SOURCE_PATH;
const cdpUrl = process.env.CDP_URL ?? 'http://127.0.0.1:9223';
const timeoutMs = Number.parseInt(process.env.RUN_TIMEOUT_MS ?? '1800000', 10);

if (!sourcePath) {
  throw new Error('REAL_SOURCE_PATH is required');
}

const sourceLabel = path.basename(sourcePath);

function logStep(step, expectation, extra) {
  const payload = {
    type: 'step',
    at: new Date().toISOString(),
    step,
    expectation,
    ...(extra ? { extra } : {}),
  };
  console.log(JSON.stringify(payload));
}

async function openPanelIfCollapsed(page, workspaceTestId, toggleTestId) {
  const workspace = page.getByTestId(workspaceTestId);
  const isVisible = await workspace.isVisible().catch(() => false);
  if (!isVisible) {
    await page.getByTestId(toggleTestId).click();
  }
}

async function getAppConfig(page) {
  return page.evaluate(async () => {
    if (!(window.__TAURI_INTERNALS__ && typeof window.__TAURI_INTERNALS__.invoke === 'function')) {
      throw new Error('Tauri invoke bridge is unavailable');
    }
    return window.__TAURI_INTERNALS__.invoke('get_app_config');
  });
}

async function snapshotBlindPanel(page) {
  return page.evaluate(() => {
    const sampleCards = Array.from(document.querySelectorAll('[data-testid^="blind-sample-"]'));
    const previews = Array.from(document.querySelectorAll('[data-testid^="blind-preview-"]'));
    const pickButtons = Array.from(document.querySelectorAll('[data-testid^="pick-sample-"]'));
    const revealBlocks = Array.from(document.querySelectorAll('[data-testid^="blind-reveal-"]'));
    const sampleStatuses = sampleCards.map((card) => ({
      id: card.getAttribute('data-testid'),
      text: card.querySelector('.blind-sample-status')?.textContent?.trim() ?? null,
      progress: card.querySelector('.blind-sample-header span:last-child')?.textContent?.trim() ?? null,
    }));
    return {
      runButtonText: document.querySelector('[data-testid="run-blind-comparison-button"]')?.textContent?.trim() ?? null,
      blindSummary: document.querySelector('[data-testid="blind-comparison-panel"] .summary')?.textContent?.trim() ?? null,
      comparisonInspectorVisible: Boolean(document.querySelector('[data-testid="comparison-inspector"]')),
      sampleCount: sampleCards.length,
      previewCount: previews.length,
      pickButtonCount: pickButtons.length,
      revealCount: revealBlocks.length,
      selectedWinnerVisible: document.body.textContent?.includes('Selected winner') ?? false,
      focusHint: document.querySelector('[data-testid="comparison-focus-hint"]')?.textContent?.trim() ?? null,
      sampleStatuses,
      errorText: document.querySelector('[data-testid="error-text"], .error-text')?.textContent?.trim() ?? null,
      statusText: document.querySelector('[data-testid="top-status-panel"]')?.textContent?.trim() ?? null,
    };
  });
}

async function captureComparisonPlaybackSnapshot(page) {
  return page.evaluate(() => {
    const readVideo = (selector) => {
      const node = document.querySelector(selector);
      if (!(node instanceof HTMLVideoElement)) {
        return null;
      }
      const sourceNode = node.querySelector('source');
      return {
        currentTime: node.currentTime,
        paused: node.paused,
        readyState: node.readyState,
        networkState: node.networkState,
        currentSrc: node.currentSrc,
        sourceSrc: sourceNode instanceof HTMLSourceElement ? sourceNode.src : null,
        errorCode: node.error?.code ?? null,
      };
    };

    const readBlindPreview = (sampleId) => readVideo(`[data-testid="blind-preview-${sampleId}"]`);

    return {
      timelineValue: (document.querySelector('[data-testid="comparison-time-slider"]') instanceof HTMLInputElement)
        ? Number(document.querySelector('[data-testid="comparison-time-slider"]').value)
        : null,
      timelineReadout: document.querySelector('[data-testid="comparison-timeline-readout"]')?.textContent?.trim() ?? null,
      blindPreviewSample1: readBlindPreview('sample-1'),
      blindPreviewSample2: readBlindPreview('sample-2'),
      source: readVideo('[data-testid="comparison-source-viewport"] video'),
      sampleA: readVideo('[data-testid="comparison-sample-viewport-sample-1"] video'),
      sampleB: readVideo('[data-testid="comparison-sample-viewport-sample-2"] video'),
    };
  });
}

async function waitForComparisonMediaReady(page, timeout = 30000) {
  await expect.poll(
    async () => await captureComparisonPlaybackSnapshot(page),
    { timeout },
  ).toMatchObject({
    source: expect.objectContaining({ readyState: expect.any(Number) }),
    sampleA: expect.objectContaining({ readyState: expect.any(Number) }),
  });

  await expect.poll(
    async () => {
      const snapshot = await captureComparisonPlaybackSnapshot(page);
      return {
        sourceReadyState: snapshot.source?.readyState ?? 0,
        sampleAReadyState: snapshot.sampleA?.readyState ?? 0,
        sampleBReadyState: snapshot.sampleB?.readyState ?? 0,
      };
    },
    { timeout },
  ).toMatchObject({
    sourceReadyState: expect.any(Number),
    sampleAReadyState: expect.any(Number),
    sampleBReadyState: expect.any(Number),
  });

  const startedAt = Date.now();
  let lastSnapshot = await captureComparisonPlaybackSnapshot(page);
  while (Date.now() - startedAt < timeout) {
    lastSnapshot = await captureComparisonPlaybackSnapshot(page);
    const minReadyState = Math.min(
      lastSnapshot.source?.readyState ?? 0,
      lastSnapshot.sampleA?.readyState ?? 0,
      lastSnapshot.sampleB?.readyState ?? 0,
    );
    if (minReadyState > 1) {
      return;
    }
    await page.waitForTimeout(250);
  }

  throw new Error(`Comparison media never became ready: ${JSON.stringify(lastSnapshot)}`);
}

async function seekComparisonTimeline(page, targetFrame) {
  const comparisonTimeSlider = page.getByTestId('comparison-time-slider');
  const sliderMax = await comparisonTimeSlider.evaluate((element) => Number(element.max || '0'));
  const clampedTargetFrame = Math.min(Math.max(0, targetFrame), sliderMax);
  const sliderBounds = await comparisonTimeSlider.boundingBox();
  if (!sliderBounds) {
    throw new Error('Comparison timeline slider is not visible for scrubbing.');
  }

  const sliderRatio = sliderMax > 0 ? clampedTargetFrame / sliderMax : 0;
  await comparisonTimeSlider.click({
    position: {
      x: Math.max(1, Math.min(sliderBounds.width - 1, sliderRatio * sliderBounds.width)),
      y: sliderBounds.height / 2,
    },
  });

  const timelineReadout = page.getByTestId('comparison-timeline-readout');
  const clickApplied = await expect.poll(
    async () => await timelineReadout.textContent(),
    { timeout: 1500 },
  ).toContain(`Frame ${clampedTargetFrame} /`).then(() => true).catch(() => false);
  if (clickApplied) {
    return clampedTargetFrame;
  }

  await comparisonTimeSlider.focus();
  await comparisonTimeSlider.press('Home');
  for (let index = 0; index < clampedTargetFrame; index += 1) {
    await comparisonTimeSlider.press('ArrowRight');
  }

  await expect(timelineReadout).toContainText(new RegExp(`Frame ${clampedTargetFrame}\\s*/`, 'i'));
  return clampedTargetFrame;
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

async function withDesktopPage(context, action, timeout = 30000) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const page = await getDesktopPage(context, timeout);
    try {
      await page.bringToFront();
      return await action(page);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!message.includes('Target page, context or browser has been closed') || attempt === 2) {
        throw error;
      }
    }
  }

  throw new Error('Could not recover the Upscaler desktop webview target');
}

async function main() {
  const browser = await chromium.connectOverCDP(cdpUrl, { timeout: 60000 });
  const context = browser.contexts()[0];
  if (!context) {
    throw new Error('No browser context available from CDP session');
  }

  let page = await getDesktopPage(context);
    await page.bringToFront();
    await page.addInitScript(({ selectedSourcePath }) => {
      window.__UPSCALER_MOCK__ = {
        async selectVideoFile() {
          return selectedSourcePath;
        },
      };
    }, { selectedSourcePath: sourcePath });

    await page.evaluate(() => {
      window.localStorage.clear();
      window.sessionStorage.clear();
    }).catch(() => {});
    await page.reload({ waitUntil: 'domcontentloaded' });

    const initialConfig = await getAppConfig(page);
    const initialBlindPickCount = Array.isArray(initialConfig?.blindComparisons) ? initialConfig.blindComparisons.length : 0;
    logStep(
      'Loaded desktop app and captured baseline config',
      'The app should be reachable and we should know how many blind picks already exist before this run.',
      { initialBlindPickCount },
    );

    logStep(
      'Selecting the real source clip',
      'The app should probe the real source and show a source preview/workspace state without manual file dialog interaction.',
      { sourcePath },
    );
    await page.getByTestId('select-video-button').click();
    await page.getByTestId('source-preview').waitFor({ timeout: 180000 });
    await expect.poll(
      () => page.evaluate((label) => document.body.textContent?.includes(label) ?? false, sourceLabel),
      { timeout: 180000 },
    ).toBe(true);

    logStep(
      'Configuring a 1-second blind comparison',
      'The blind-test summary should update to describe 1s preview exports so we know the comparison is capped to a short run.',
    );
    const previewCheckbox = page.getByTestId('preview-mode-checkbox');
    if (!(await previewCheckbox.isChecked())) {
      await previewCheckbox.check();
    }
    await page.getByTestId('preview-duration-input').fill('1');
    await openPanelIfCollapsed(page, 'blind-comparison-panel', 'blind-test-panel-toggle');
    await expect(page.getByTestId('blind-comparison-panel')).toContainText('1s preview exports');

    logStep(
      'Capturing a non-zero comparison start offset',
      'The source preview scrubber should accept a later position and the blind-test panel should retain that captured start time for preview generation.',
    );
    await expect.poll(
      async () => await page.getByTestId('source-preview-seek').evaluate((element) => {
        if (!(element instanceof HTMLInputElement)) {
          throw new Error('Source preview seek slider is unavailable');
        }
        return Number(element.max || '0');
      }),
      { timeout: 30000 },
    ).toBeGreaterThan(1);
    await page.getByTestId('source-preview-seek').evaluate((element) => {
      if (!(element instanceof HTMLInputElement)) {
        throw new Error('Source preview seek slider is unavailable');
      }
      element.value = '2.2';
      element.dispatchEvent(new Event('input', { bubbles: true }));
      element.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.getByTestId('blind-capture-current-preview-position').click();
    await expect(page.getByTestId('blind-start-offset-readout')).toContainText('0:02');

    logStep(
      'Starting the blind comparison',
      'We expect four anonymous sample jobs, one per runnable comparison model, to appear and eventually produce preview clips.',
    );
    await page.getByTestId('run-blind-comparison-button').click();
    await expect(page.getByTestId('run-blind-comparison-button')).toContainText('Blind Comparison Running...', { timeout: 10000 });

    let lastSignature = '';
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
      const snapshot = await withDesktopPage(context, async (activePage) => {
        page = activePage;
        return await snapshotBlindPanel(activePage);
      });
      if (snapshot.errorText) {
        throw new Error(`Blind comparison failed: ${JSON.stringify(snapshot)}`);
      }

      const signature = JSON.stringify({
        previewCount: snapshot.previewCount,
        pickButtonCount: snapshot.pickButtonCount,
        statuses: snapshot.sampleStatuses,
      });
      if (signature !== lastSignature) {
        lastSignature = signature;
        logStep(
          'Blind comparison progress update',
          'Sample cards should keep filling in until all four previews are ready and winner-pick buttons appear.',
          snapshot,
        );
      }

      if (snapshot.previewCount === 4 && snapshot.pickButtonCount === 4 && snapshot.comparisonInspectorVisible) {
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 5000));
    }

    page = await getDesktopPage(context, 30000);
    const readySnapshot = await snapshotBlindPanel(page);
    if (readySnapshot.previewCount !== 4 || readySnapshot.pickButtonCount !== 4) {
      throw new Error(`Blind comparison did not become ready in time: ${JSON.stringify(readySnapshot)}`);
    }

    logStep(
      'Exercising comparison inspector controls',
      'The inspector should be visible, allow switching focus presets, and keep the reveal hidden until we actually choose a winner.',
      readySnapshot,
    );
    await expect(page.getByTestId('comparison-inspector')).toBeVisible();
    await expect(page.locator('[data-testid^="blind-reveal-"]')).toHaveCount(0);
    await page.getByTestId('open-comparison-workspace-button').click();
    await expect(page.getByTestId('comparison-workspace-modal')).toBeVisible();
    await page.getByTestId('comparison-focus-diagonals').click();
    await page.getByTestId('comparison-zoom-slider').fill('4');
    await expect(page.getByTestId('comparison-focus-hint')).toContainText(/diagonal|corners|eyes|texture/i);
    await waitForComparisonMediaReady(page);

    logStep(
      'Scrubbing the comparison timeline',
      'The frame slider should move to a mid-range frame and the source and samples should all land on roughly the same playback time.',
    );
    const targetFrame = await page.getByTestId('comparison-time-slider').evaluate((element) => {
      const sliderMax = Number(element.max || '0');
      return Math.min(Math.max(1, Math.floor(sliderMax / 2)), sliderMax);
    });
    await seekComparisonTimeline(page, targetFrame);
    const scrubbedSnapshot = await captureComparisonPlaybackSnapshot(page);
    if (!scrubbedSnapshot.source || !scrubbedSnapshot.sampleA) {
      throw new Error(`Comparison players were not available after scrub: ${JSON.stringify(scrubbedSnapshot)}`);
    }
    if (Math.abs((scrubbedSnapshot.source.currentTime ?? 0) - (scrubbedSnapshot.sampleA.currentTime ?? 0)) > 0.15) {
      throw new Error(`Comparison scrub did not keep players aligned: ${JSON.stringify(scrubbedSnapshot)}`);
    }

    logStep(
      'Playing and pausing comparison playback',
      'Play should advance all players together and pause should leave them stopped on the same frame neighborhood.',
      scrubbedSnapshot,
    );
    await page.getByTestId('comparison-play-toggle').click();
    const playbackStartedAt = Date.now();
    let playingSnapshot = await captureComparisonPlaybackSnapshot(page);
    while (Date.now() - playbackStartedAt < 700) {
      playingSnapshot = await captureComparisonPlaybackSnapshot(page);
      if (playingSnapshot.source && playingSnapshot.sampleA) {
        const sourceAdvanced = (playingSnapshot.source.currentTime ?? 0) > ((scrubbedSnapshot.source?.currentTime ?? 0) + 0.05);
        const sampleAdvanced = (playingSnapshot.sampleA.currentTime ?? 0) > ((scrubbedSnapshot.sampleA?.currentTime ?? 0) + 0.05);
        if (!playingSnapshot.source.paused && !playingSnapshot.sampleA.paused && sourceAdvanced && sampleAdvanced) {
          break;
        }
      }

      await page.waitForTimeout(75);
    }
    if (!playingSnapshot.source || !playingSnapshot.sampleA) {
      throw new Error(`Comparison players were not available during playback: ${JSON.stringify(playingSnapshot)}`);
    }
    if (playingSnapshot.source.paused || playingSnapshot.sampleA.paused) {
      throw new Error(`Comparison playback did not start across players: ${JSON.stringify(playingSnapshot)}`);
    }
    await page.getByTestId('comparison-play-toggle').click();
    const pausedSnapshot = await captureComparisonPlaybackSnapshot(page);
    if (!pausedSnapshot.source || !pausedSnapshot.sampleA) {
      throw new Error(`Comparison players were not available after pause: ${JSON.stringify(pausedSnapshot)}`);
    }
    if (!pausedSnapshot.source.paused || !pausedSnapshot.sampleA.paused) {
      throw new Error(`Comparison playback did not pause across players: ${JSON.stringify(pausedSnapshot)}`);
    }
    if (Math.abs((pausedSnapshot.source.currentTime ?? 0) - (pausedSnapshot.sampleA.currentTime ?? 0)) > 0.2) {
      throw new Error(`Comparison pause left players out of sync: ${JSON.stringify(pausedSnapshot)}`);
    }

    logStep(
      'Picking a blind winner',
      'Reveal badges should stay hidden before the click, then all four samples should reveal their true models and exactly one should show Selected winner.',
    );
    await page.getByTestId('comparison-pick-sample-1').click();
    await expect(page.locator('[data-testid^="blind-reveal-"]')).toHaveCount(4, { timeout: 30000 });
    await expect(page.getByTestId('comparison-workspace-reveal-sample-1')).toContainText('Selected winner', { timeout: 30000 });

    const finalConfig = await getAppConfig(page);
    const finalBlindPickCount = Array.isArray(finalConfig?.blindComparisons) ? finalConfig.blindComparisons.length : initialBlindPickCount;
    const latestBlindPick = Array.isArray(finalConfig?.blindComparisons) ? finalConfig.blindComparisons.at(-1) ?? null : null;
    const finalSnapshot = await snapshotBlindPanel(page);

    if (finalBlindPickCount < initialBlindPickCount + 1) {
      throw new Error(`Blind comparison winner was not persisted: ${JSON.stringify({ initialBlindPickCount, finalBlindPickCount, finalSnapshot })}`);
    }
    if (!latestBlindPick || Number(latestBlindPick.previewStartOffsetSeconds ?? 0) <= 0) {
      throw new Error(`Blind comparison start offset was not persisted: ${JSON.stringify({ latestBlindPick, finalSnapshot })}`);
    }

    console.log(JSON.stringify({
      status: 'succeeded',
      sourcePath,
      initialBlindPickCount,
      finalBlindPickCount,
      blindPickDelta: finalBlindPickCount - initialBlindPickCount,
      previewStartOffsetSeconds: latestBlindPick.previewStartOffsetSeconds,
      finalSnapshot,
    }, null, 2));
}

await main();