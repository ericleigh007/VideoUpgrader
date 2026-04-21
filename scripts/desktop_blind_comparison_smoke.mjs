import { chromium, expect } from '@playwright/test';
import path from 'node:path';

const sourcePath = process.env.REAL_SOURCE_PATH;
const cdpUrl = process.env.CDP_URL ?? 'http://127.0.0.1:9223';
const timeoutMs = Number.parseInt(process.env.RUN_TIMEOUT_MS ?? '1800000', 10);
const previewDurationSeconds = Number.parseInt(process.env.BLIND_PREVIEW_DURATION_SECONDS ?? '1', 10);
const comparisonStartOffsetSeconds = Number.parseFloat(process.env.COMPARISON_START_OFFSET_SECONDS ?? '3.2');
const blindComparisonScenario = process.env.BLIND_COMPARISON_SCENARIO ?? 'upscale';
const exerciseBlindJobControls = /^(1|true|yes)$/i.test(process.env.EXERCISE_BLIND_JOB_CONTROLS ?? '');
const cancelBlindComparison = /^(1|true|yes)$/i.test(process.env.CANCEL_BLIND_COMPARISON ?? '');
const assertFlashAlignment = /^(1|true|yes)$/i.test(process.env.EXPECT_FLASH_ALIGNMENT ?? '');
const avSyncFps = Number.parseFloat(process.env.AV_SYNC_FPS ?? '30');
const avSyncFlashIntervalSeconds = Number.parseFloat(process.env.AV_SYNC_FLASH_INTERVAL_SECONDS ?? '1');
const avSyncFlashDurationSeconds = Number.parseFloat(process.env.AV_SYNC_FLASH_DURATION_SECONDS ?? '0.08');

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

function formatClockLabel(seconds) {
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `${minutes}:${String(remainingSeconds).padStart(2, '0')}`;
}

function describeBlindScenario(scenario) {
  switch (scenario) {
    case 'colorizeOnly':
      return 'colorize-only comparison';
    case 'colorizeBeforeUpscale':
      return 'colorize-plus-upscale comparison';
    case 'interpolateAfterUpscale':
      return 'upscale-plus-interpolation comparison';
    case 'upscale':
    default:
      return 'upscale-only comparison';
  }
}

async function setPipelineSwitch(page, testId, enabled) {
  const toggle = page.getByTestId(testId);
  const currentValue = await toggle.getAttribute('aria-checked');
  const isEnabled = currentValue === 'true';
  if (isEnabled !== enabled) {
    await toggle.click();
  }
}

async function configureBlindComparisonScenario(page, scenario) {
  await setPipelineSwitch(page, 'pipeline-toggle-colorization', scenario === 'colorizeOnly' || scenario === 'colorizeBeforeUpscale');
  await setPipelineSwitch(page, 'pipeline-toggle-upscale', scenario !== 'colorizeOnly');
  await setPipelineSwitch(page, 'pipeline-toggle-interpolation', scenario === 'interpolateAfterUpscale');

  if (scenario === 'colorizeBeforeUpscale') {
    await page.getByTestId('model-select').selectOption('realesrnet-x4plus');
  }

  if (scenario === 'interpolateAfterUpscale') {
    await page.getByTestId('frame-rate-target-select').selectOption('60');
  }
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
        duration: Number.isFinite(node.duration) ? node.duration : null,
        paused: node.paused,
        ended: node.ended,
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

async function captureComparisonFrameBrightness(page) {
  return page.evaluate(() => {
    const readBrightness = (video) => {
      if (!(video instanceof HTMLVideoElement) || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth <= 0 || video.videoHeight <= 0) {
        return null;
      }

      const canvas = document.createElement('canvas');
      canvas.width = 48;
      canvas.height = 27;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) {
        return null;
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      let totalLuma = 0;
      for (let index = 0; index < pixels.length; index += 4) {
        totalLuma += (pixels[index] * 0.299) + (pixels[index + 1] * 0.587) + (pixels[index + 2] * 0.114);
      }
      return totalLuma / (pixels.length / 4);
    };

    const sourceVideo = document.querySelector('[data-testid="comparison-source-viewport"] video');
    const sampleVideos = Array.from(document.querySelectorAll('[data-testid^="comparison-sample-viewport-"] video'));

    return {
      source: readBrightness(sourceVideo),
      samples: sampleVideos.map((video, index) => ({
        index,
        brightness: readBrightness(video),
      })),
    };
  });
}

async function waitForBlindJobActionButtons(page, timeout = 30000) {
  await expect(page.getByTestId('top-status-pause-button')).toBeVisible({ timeout });
  await expect(page.getByTestId('top-status-stop-button')).toBeVisible({ timeout });
}

async function exerciseBlindComparisonControls(page) {
  await waitForBlindJobActionButtons(page);
  await page.getByTestId('top-status-pause-button').click();
  await expect.poll(
    async () => await page.getByTestId('top-status-panel').textContent(),
    { timeout: 15000 },
  ).toContain('Paused');

  await page.getByTestId('top-status-pause-button').click();
  await expect.poll(
    async () => await page.getByTestId('top-status-panel').textContent(),
    { timeout: 15000 },
  ).not.toContain('Paused');
}

async function cancelActiveBlindComparison(page) {
  await waitForBlindJobActionButtons(page);
  await page.getByTestId('top-status-stop-button').click();
  const cancellationState = await expect.poll(
    async () => await snapshotBlindPanel(page),
    { timeout: 30000 },
  ).toMatchObject(expect.objectContaining({
    statusText: expect.stringMatching(/cancelled/i),
  })).then(async () => await snapshotBlindPanel(page));

  return cancellationState;
}

function expectedFlashStateAtLogicalTime(logicalTimeSeconds) {
  const absoluteTimeSeconds = logicalTimeSeconds + comparisonStartOffsetSeconds;
  const normalizedTime = ((absoluteTimeSeconds % avSyncFlashIntervalSeconds) + avSyncFlashIntervalSeconds) % avSyncFlashIntervalSeconds;
  return normalizedTime <= avSyncFlashDurationSeconds;
}

async function verifyFlashAlignment(page) {
  const flashCheckpoints = [0.04, 0.35, 1.04, 1.35, 2.04, 2.35]
    .filter((time) => time < Math.max(0.1, previewDurationSeconds - 0.02));

  for (const logicalTimeSeconds of flashCheckpoints) {
    const targetFrame = Math.max(0, Math.round(logicalTimeSeconds * avSyncFps));
    await seekComparisonTimeline(page, targetFrame);
    await page.waitForTimeout(120);

    const brightness = await captureComparisonFrameBrightness(page);
    const values = [brightness.source, ...brightness.samples.map((sample) => sample.brightness)].filter((value) => typeof value === 'number');
    if (values.length < 3) {
      throw new Error(`Could not read enough comparison frame brightness values: ${JSON.stringify({ logicalTimeSeconds, brightness })}`);
    }

    const expectedFlash = expectedFlashStateAtLogicalTime(logicalTimeSeconds);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    if (expectedFlash) {
      if (minimum < 160 || (maximum - minimum) > 45) {
        throw new Error(`Flash alignment check failed for bright frame: ${JSON.stringify({ logicalTimeSeconds, brightness, minimum, maximum })}`);
      }
    } else if (maximum > 80 || (maximum - minimum) > 35) {
      throw new Error(`Flash alignment check failed for dark frame: ${JSON.stringify({ logicalTimeSeconds, brightness, minimum, maximum })}`);
    }
  }
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
  const initialSliderValue = await comparisonTimeSlider.evaluate((element) => Number(element.value || '0'));
  const sliderBounds = await comparisonTimeSlider.boundingBox();
  if (!sliderBounds) {
    throw new Error('Comparison timeline slider is not visible for scrubbing.');
  }

  const parseFrameReadout = (value) => {
    if (typeof value !== 'string') {
      return null;
    }

    const match = value.match(/Frame\s+(\d+)\s*\//i);
    return match ? Number.parseInt(match[1], 10) : null;
  };

  const sliderRatio = sliderMax > 0 ? clampedTargetFrame / sliderMax : 0;
  await comparisonTimeSlider.click({
    position: {
      x: Math.max(1, Math.min(sliderBounds.width - 1, sliderRatio * sliderBounds.width)),
      y: sliderBounds.height / 2,
    },
  });

  const timelineReadout = page.getByTestId('comparison-timeline-readout');
  const clickAppliedFrame = await expect.poll(
    async () => parseFrameReadout(await timelineReadout.textContent()),
    { timeout: 1500 },
  ).not.toBeNull().then(async () => parseFrameReadout(await timelineReadout.textContent())).catch(() => null);
  if (clickAppliedFrame !== null && clickAppliedFrame !== initialSliderValue) {
    return clickAppliedFrame;
  }

  await comparisonTimeSlider.focus();
  await comparisonTimeSlider.press('Home');
  for (let index = 0; index < clampedTargetFrame; index += 1) {
    await comparisonTimeSlider.press('ArrowRight');
  }

  const finalFrame = await expect.poll(
    async () => parseFrameReadout(await timelineReadout.textContent()),
    { timeout: 5000 },
  ).not.toBeNull().then(async () => parseFrameReadout(await timelineReadout.textContent())).catch(() => null);
  if (finalFrame === null) {
    throw new Error(`Comparison timeline readout did not resolve to a frame after scrubbing toward ${clampedTargetFrame}.`);
  }

  return finalFrame;
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

async function getComparisonPage(context, timeout = 15000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const comparisonPage = context.pages().find((candidate) => candidate.url().includes('localhost:1420') && candidate.url().includes('view=comparison'));
    if (comparisonPage) {
      return comparisonPage;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }

  throw new Error('Could not find the detached comparison webview target');
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
  try {
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
      'Configuring the blind comparison scenario',
      'The pipeline switches and blind-test summary should match the requested comparison mode before preview jobs launch.',
      { scenario: blindComparisonScenario, scenarioLabel: describeBlindScenario(blindComparisonScenario) },
    );
    const previewCheckbox = page.getByTestId('preview-mode-checkbox');
    if (!(await previewCheckbox.isChecked())) {
      await previewCheckbox.check();
    }
    await page.getByTestId('preview-duration-input').fill(String(previewDurationSeconds));
    await configureBlindComparisonScenario(page, blindComparisonScenario);
    await openPanelIfCollapsed(page, 'blind-comparison-panel', 'blind-test-panel-toggle');
    await expect(page.getByTestId('blind-comparison-panel')).toContainText(`${previewDurationSeconds}s preview exports`);

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
    await page.getByTestId('source-preview-seek').evaluate((element, targetSeconds) => {
      if (!(element instanceof HTMLInputElement)) {
        throw new Error('Source preview seek slider is unavailable');
      }
      element.value = String(targetSeconds);
      element.dispatchEvent(new Event('input', { bubbles: true }));
      element.dispatchEvent(new Event('change', { bubbles: true }));
    }, comparisonStartOffsetSeconds);
    await page.getByTestId('blind-capture-current-preview-position').click();
    await expect(page.getByTestId('blind-start-offset-readout')).toContainText(formatClockLabel(comparisonStartOffsetSeconds));

    logStep(
      'Starting the blind comparison',
      'We expect four anonymous sample jobs, one per runnable comparison model, to appear and eventually produce preview clips.',
    );
    await page.getByTestId('run-blind-comparison-button').click();
    await expect(page.getByTestId('run-blind-comparison-button')).toContainText('Blind Comparison Running...', { timeout: 10000 });

    if (exerciseBlindJobControls) {
      logStep(
        'Exercising live blind-comparison controls',
        'The top status pause and stop actions should stay visible while the anonymous comparison jobs are running.',
        { scenario: blindComparisonScenario },
      );
      await exerciseBlindComparisonControls(page);
    }

    if (cancelBlindComparison) {
      logStep(
        'Cancelling the active blind comparison',
        'The top status stop action should cancel the active blind-comparison job and report a cancelled state instead of hanging.',
        { scenario: blindComparisonScenario },
      );
      const cancelledSnapshot = await cancelActiveBlindComparison(page);
      console.log(JSON.stringify({
        status: 'cancelled-as-expected',
        scenario: blindComparisonScenario,
        previewDurationSeconds,
        cancelledSnapshot,
      }, null, 2));
      return;
    }

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
    let comparisonPage = null;
    const inlineModalVisible = await page.getByTestId('comparison-workspace-modal').isVisible().catch(() => false);
    if (inlineModalVisible) {
      comparisonPage = page;
    } else {
      comparisonPage = await getComparisonPage(context, 30000);
      await comparisonPage.bringToFront();
      await expect(comparisonPage.getByTestId('comparison-workspace-window')).toBeVisible();
    }
    await comparisonPage.getByTestId('comparison-focus-diagonals').click();
    await comparisonPage.getByTestId('comparison-zoom-slider').fill('4');
    await expect(comparisonPage.getByTestId('comparison-focus-hint')).toContainText(/diagonal|corners|eyes|texture/i);
    await waitForComparisonMediaReady(comparisonPage);

    logStep(
      'Scrubbing the comparison timeline',
      'The frame slider should move to a mid-range frame and the source and samples should all land on roughly the same playback time.',
    );
    const targetFrame = await comparisonPage.getByTestId('comparison-time-slider').evaluate((element) => {
      const sliderMax = Number(element.max || '0');
      return Math.min(Math.max(1, Math.floor(sliderMax / 2)), sliderMax);
    });
    await seekComparisonTimeline(comparisonPage, targetFrame);
    let scrubbedSnapshot = await captureComparisonPlaybackSnapshot(comparisonPage);
    const landedAtTerminalFrame = Boolean(
      (scrubbedSnapshot.sampleA?.ended)
      || (scrubbedSnapshot.sampleA?.duration !== null && scrubbedSnapshot.sampleA && scrubbedSnapshot.sampleA.duration - scrubbedSnapshot.sampleA.currentTime <= 0.05)
      || (scrubbedSnapshot.source?.duration !== null && scrubbedSnapshot.source && scrubbedSnapshot.source.duration - scrubbedSnapshot.source.currentTime <= 0.05),
    );
    if (landedAtTerminalFrame && targetFrame > 1) {
      const fallbackFrame = Math.max(1, Math.floor(targetFrame / 2));
      await seekComparisonTimeline(comparisonPage, fallbackFrame);
      scrubbedSnapshot = await captureComparisonPlaybackSnapshot(comparisonPage);
    }
    if (!scrubbedSnapshot.source || !scrubbedSnapshot.sampleA) {
      throw new Error(`Comparison players were not available after scrub: ${JSON.stringify(scrubbedSnapshot)}`);
    }
    const scrubOffsetDelta = Math.abs(
      ((scrubbedSnapshot.source.currentTime ?? 0) - (scrubbedSnapshot.sampleA.currentTime ?? 0))
      - comparisonStartOffsetSeconds,
    );
    if (scrubOffsetDelta > 0.2) {
      throw new Error(`Comparison scrub did not keep the source offset aligned: ${JSON.stringify({ scrubbedSnapshot, comparisonStartOffsetSeconds, scrubOffsetDelta })}`);
    }

    if (assertFlashAlignment) {
      logStep(
        'Checking synthetic flash alignment',
        'A time-coded synthetic source should show the same bright and dark states across the source and every comparison sample at the same logical frames.',
        { previewDurationSeconds, comparisonStartOffsetSeconds, avSyncFps, avSyncFlashIntervalSeconds, avSyncFlashDurationSeconds },
      );
      await verifyFlashAlignment(comparisonPage);
    }

    logStep(
      'Playing and pausing comparison playback',
      'Play should advance all players together and pause should leave them stopped on the same frame neighborhood.',
      scrubbedSnapshot,
    );
    await comparisonPage.getByTestId('comparison-play-toggle').click();
    const playbackStartedAt = Date.now();
    let playingSnapshot = await captureComparisonPlaybackSnapshot(comparisonPage);
    let playbackAdvanced = false;
    while (Date.now() - playbackStartedAt < 700) {
      playingSnapshot = await captureComparisonPlaybackSnapshot(comparisonPage);
      if (playingSnapshot.source && playingSnapshot.sampleA) {
        const sourceAdvanced = (playingSnapshot.source.currentTime ?? 0) > ((scrubbedSnapshot.source?.currentTime ?? 0) + 0.05);
        const sourceReachedEnd = Boolean(
          playingSnapshot.source.ended
          || (playingSnapshot.source.duration !== null && playingSnapshot.source.duration - (playingSnapshot.source.currentTime ?? 0) <= 0.12),
        );
        const sampleAdvanced = (playingSnapshot.sampleA.currentTime ?? 0) > ((scrubbedSnapshot.sampleA?.currentTime ?? 0) + 0.05);
        if ((sourceAdvanced || sourceReachedEnd) && sampleAdvanced) {
          playbackAdvanced = true;
          break;
        }
      }

      await comparisonPage.waitForTimeout(75);
    }
    if (!playingSnapshot.source || !playingSnapshot.sampleA) {
      throw new Error(`Comparison players were not available during playback: ${JSON.stringify(playingSnapshot)}`);
    }
    if (!playbackAdvanced) {
      const sourceAdvanced = (playingSnapshot.source.currentTime ?? 0) > ((scrubbedSnapshot.source?.currentTime ?? 0) + 0.05);
      const sourceReachedEnd = Boolean(
        playingSnapshot.source.ended
        || (playingSnapshot.source.duration !== null && playingSnapshot.source.duration - (playingSnapshot.source.currentTime ?? 0) <= 0.12),
      );
      const sampleAdvanced = (playingSnapshot.sampleA.currentTime ?? 0) > ((scrubbedSnapshot.sampleA?.currentTime ?? 0) + 0.05);
      playbackAdvanced = (sourceAdvanced || sourceReachedEnd) && sampleAdvanced;
    }
    if (!playbackAdvanced) {
      throw new Error(`Comparison playback did not start across players: ${JSON.stringify(playingSnapshot)}`);
    }
    let pauseAttempted = false;
    if (!playingSnapshot.source.paused || !playingSnapshot.sampleA.paused) {
      pauseAttempted = true;
      await comparisonPage.getByTestId('comparison-play-toggle').click();
    }
    let pausedSnapshot = await captureComparisonPlaybackSnapshot(comparisonPage);
    if (pauseAttempted && pausedSnapshot.source && pausedSnapshot.sampleA) {
      const pauseWaitStartedAt = Date.now();
      while (Date.now() - pauseWaitStartedAt < 1000) {
        if (pausedSnapshot.source.paused && pausedSnapshot.sampleA.paused) {
          break;
        }

        await comparisonPage.waitForTimeout(75);
        pausedSnapshot = await captureComparisonPlaybackSnapshot(comparisonPage);
      }
    }
    if (!pausedSnapshot.source || !pausedSnapshot.sampleA) {
      throw new Error(`Comparison players were not available after pause: ${JSON.stringify(pausedSnapshot)}`);
    }
    if (pausedSnapshot.source.paused && pausedSnapshot.sampleA.paused) {
      const synchronizedReset = Boolean(
        (pausedSnapshot.source.currentTime ?? 0) === 0
        && (pausedSnapshot.sampleA.currentTime ?? 0) === 0
        && pausedSnapshot.source.readyState === 0
        && pausedSnapshot.sampleA.readyState === 0,
      );
      const pauseOffsetDelta = Math.abs(
        ((pausedSnapshot.source.currentTime ?? 0) - (pausedSnapshot.sampleA.currentTime ?? 0))
        - comparisonStartOffsetSeconds,
      );
      if (!synchronizedReset && pauseOffsetDelta > 0.25) {
        throw new Error(`Comparison pause left players out of sync: ${JSON.stringify({ pausedSnapshot, comparisonStartOffsetSeconds, pauseOffsetDelta })}`);
      }
    } else {
      logStep(
        'Pause playback remained transient',
        'Short preview clips may finish or restart quickly, so playback advancement is the primary assertion and pause-state sampling is treated as best-effort.',
        pausedSnapshot,
      );
    }

    logStep(
      'Picking a blind winner',
      'Reveal badges should stay hidden before the click, then all four samples should reveal their true models and exactly one should show Selected winner.',
    );
    await comparisonPage.getByTestId('comparison-pick-sample-1').click();
    await expect(comparisonPage.getByTestId('comparison-workspace-reveal-sample-1')).toContainText('Selected winner', { timeout: 30000 });

    const finalConfig = await getAppConfig(page);
    const finalBlindPickCount = Array.isArray(finalConfig?.blindComparisons) ? finalConfig.blindComparisons.length : initialBlindPickCount;
    const latestBlindPick = Array.isArray(finalConfig?.blindComparisons) ? finalConfig.blindComparisons.at(-1) ?? null : null;
    const finalSnapshot = await snapshotBlindPanel(page);
    const mainRevealSyncReached = await expect
      .poll(async () => await page.locator('[data-testid^="blind-reveal-"]').count(), { timeout: 10000 })
      .toBe(4)
      .then(() => true)
      .catch(() => false);

    if (!mainRevealSyncReached) {
      throw new Error(`Detached winner reveal did not propagate back to the main window: ${JSON.stringify(finalSnapshot)}`);
    }

    if (finalBlindPickCount < initialBlindPickCount + 1) {
      throw new Error(`Blind comparison winner was not persisted: ${JSON.stringify({ initialBlindPickCount, finalBlindPickCount, finalSnapshot })}`);
    }
    if (!latestBlindPick || Number(latestBlindPick.previewStartOffsetSeconds ?? 0) <= 0) {
      throw new Error(`Blind comparison start offset was not persisted: ${JSON.stringify({ latestBlindPick, finalSnapshot })}`);
    }

    console.log(JSON.stringify({
      status: 'succeeded',
      scenario: blindComparisonScenario,
      sourcePath,
      initialBlindPickCount,
      finalBlindPickCount,
      blindPickDelta: finalBlindPickCount - initialBlindPickCount,
      previewStartOffsetSeconds: latestBlindPick.previewStartOffsetSeconds,
      previewDurationSeconds,
      mainRevealSyncReached,
      finalSnapshot,
    }, null, 2));
  } finally {
    await browser.close().catch(() => {});
  }
}

await main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});