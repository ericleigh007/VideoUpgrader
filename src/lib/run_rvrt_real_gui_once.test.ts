import { describe, expect, it } from "vitest";

import {
  guiRunHasCompleted,
  guiRunHasFailed,
  guiRunHasStarted,
} from "../../scripts/run_rvrt_real_gui_once.mjs";

describe("run_rvrt_real_gui_once helpers", () => {
  it("treats a managed running job as started even when the jobs panel is closed", () => {
    const snapshot = {
      errorText: null,
      jobProgressVisible: false,
      managedJob: {
        state: "running",
        progress: {
          phase: "upscaling",
        },
      },
    };

    expect(guiRunHasStarted(snapshot)).toBe(true);
    expect(guiRunHasCompleted(snapshot)).toBe(false);
    expect(guiRunHasFailed(snapshot)).toBe(false);
  });

  it("treats a managed succeeded job as completed even without a visible progress panel", () => {
    const snapshot = {
      errorText: null,
      resultOutputPath: null,
      jobProgressVisible: false,
      managedJob: {
        state: "succeeded",
      },
    };

    expect(guiRunHasStarted(snapshot)).toBe(true);
    expect(guiRunHasCompleted(snapshot)).toBe(true);
    expect(guiRunHasFailed(snapshot)).toBe(false);
  });

  it("treats managed interrupted states as failures", () => {
    const snapshot = {
      errorText: null,
      jobProgressVisible: false,
      managedJob: {
        state: "interrupted",
      },
    };

    expect(guiRunHasStarted(snapshot)).toBe(false);
    expect(guiRunHasCompleted(snapshot)).toBe(false);
    expect(guiRunHasFailed(snapshot)).toBe(true);
  });
});
