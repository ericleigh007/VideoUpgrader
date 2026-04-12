# Upscaler Agent Guide

This file describes how to work in this repository as it exists today.

## Current Repository Shape

This workspace is a product repository, not a generic GOTCHA scaffold.

Important directories:

- `src/`: React UI for the workstation workflow
- `src-tauri/`: Tauri desktop host and Rust command surface
- `python/`: worker runtime, pipeline, media, model integration, and tests
- `scripts/`: bootstrap, build, run, stop, and smoke-test entrypoints
- `config/`: model catalog and preferences
- `context/`: requirements and implementation planning documents
- `docs/images/`: screenshots and other README assets
- `artifacts/`: generated jobs, benchmarks, runtime assets, and outputs

There is currently no `goals/`, `tools/`, `args/`, `hardprompts/`, or manifest-driven workflow layer inside this repo. Do not assume those directories exist.

## Primary Baseline Documents

Read these first when you need project intent or operational guidance:

1. `context/requirements.md` for the product baseline and acceptance requirements
2. `build_VideoUpgrader.md` for the current local build, run, and test workflow
3. `README.md` for the user-facing product overview and current screenshots

## How To Operate In This Repo

1. Prefer existing repository scripts over ad hoc command sequences.
2. Treat `scripts/*.ps1` and `scripts/*.mjs` as the operational tool surface for this project.
3. Keep changes aligned with the actual product state, not aspirational architecture notes.
4. Update documentation when behavior, setup flow, or validation flow changes.
5. Validate fixes with the narrowest reliable command first, then widen to the full script when practical.

## Script Entry Points

- `scripts/bootstrap.ps1`: install Node and Python dependencies and provision worker runtime assets
- `scripts/build.ps1`: run frontend and Cargo build checks, optionally with tests
- `scripts/run.ps1`: launch the desktop app
- `scripts/test.ps1`: run web and Python automated tests
- `scripts/desktop-test.ps1`: run the aggregate desktop smoke flow
- `scripts/desktop_webview_playback_smoke.mjs`: desktop playback smoke for `afterUpscale` or `interpolateOnly`
- `scripts/desktop_jobs_window_recovery_smoke.mjs`: Jobs window recovery smoke
- `scripts/desktop_blind_comparison_smoke.mjs`: detached blind comparison smoke
- `scripts/stop.ps1`: stop repo-owned dev and worker processes

## Working Rules

- Use the repo-local `.venv` as the default Python environment unless the task explicitly needs another interpreter.
- Do not commit local cache content from `model_zoo/` unless the user explicitly asks for that.
- Prefer minimal, targeted edits over broad rewrites.
- Preserve screenshots, generated assets, and docs only when they are intentionally part of the change.
- If a clean-clone issue appears, reproduce it in a fresh checkout before claiming the fix is complete.

## Documentation Maintenance Rules

- `context/requirements.md` should describe the current product baseline, not future wish lists disguised as implemented behavior.
- `build_VideoUpgrader.md` should describe the scripts and validation flow that actually work today.
- `README.md` should remain user-facing and screenshot-aware.
- This file should stay short and repo-specific. Do not reintroduce references to nonexistent `goals/` or `tools/` manifests.

## Current Validation Reality

The recent clean-room validation established:

- `scripts/bootstrap.ps1` succeeds from a fresh clone
- `scripts/build.ps1` succeeds from a fresh clone
- `scripts/run.ps1` launches the app from a fresh clone
- `scripts/test.ps1` succeeds from a fresh clone
- the direct desktop smokes succeed from a fresh clone

At the time of this update, `scripts/desktop-test.ps1` may still linger under some terminal runners after its component smokes succeed. Treat that as a wrapper-orchestration issue, not proof that the desktop workflows themselves are failing.
