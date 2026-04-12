# Build VideoUpgrader

This file is the operational runbook for building, testing, launching, and debugging VideoUpgrader locally.

It is intentionally separate from AGENTS.md.

## Prerequisites

- Windows
- Node.js 20 or newer
- npm
- Cargo and the Rust toolchain
- FFmpeg available on PATH
- Python 3.10 or newer in the project-local `.venv` directory under the repository root

Preferred Python environment setup:

```powershell
$env:UPSCALER_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
```

Notes:

- Repository scripts resolve Python from `UPSCALER_PYTHON` first, then the repo-local `.venv`, and only then fall back to `python`.
- The intended default for this repo is the local `.venv`, not an unrelated global interpreter.
- If you maintain additional version-specific environments for experiments, keep them separate from the default repo `.venv` and opt into them explicitly.

## One-Time Setup

Bootstrap the workspace:

```powershell
./scripts/bootstrap.ps1
```

What bootstrap does:

- Runs `npm install`
- Upgrades `pip` in the selected Python environment
- Installs `python/requirements.txt`
- Installs the configured PyTorch and TorchVision GPU runtime

Optional bootstrap environment overrides:

- `UPSCALER_TORCH_INDEX_URL`
- `UPSCALER_TORCH_VERSION`
- `UPSCALER_TORCHVISION_VERSION`

## Standard Commands

Run the repository test suite:

```powershell
./scripts/test.ps1
```

This runs:

- `npm run test:web`
- Python tests through `unittest discover` under `python/tests`

Build frontend and desktop host checks:

```powershell
./scripts/build.ps1
```

Build and include tests:

```powershell
./scripts/build.ps1 -RunTests
```

Launch the desktop app:

```powershell
./scripts/run.ps1
```

Useful launch behavior:

- Reuses an existing dev session unless `-RestartExisting` is supplied
- Clears port `1420` before launch
- Automatically runs bootstrap if the selected Python environment is missing `torch`

Generate synthetic benchmark fixtures:

```powershell
./scripts/generate-benchmarks.ps1
```

Run the desktop smoke test:

```powershell
./scripts/desktop-test.ps1
```

This launches the Tauri app with WebView remote debugging enabled and runs the aggregate desktop smoke flow, including playback, Jobs-window recovery, detached blind comparison, and both `afterUpscale` and `interpolateOnly` playback coverage.

Current note:

- The component desktop smokes are the most reliable source of truth during investigation.
- In recent clean-clone validation, the direct sub-smokes passed end to end, while `scripts/desktop-test.ps1` could still linger under some terminal runners after those sub-smokes completed successfully.
- Treat that as wrapper orchestration drift until proven otherwise.

## Current Product State

VideoUpgrader is no longer just a scaffold. The current repository includes:

- A Tauri desktop shell and React workspace-first UI
- Native video file selection and source probing
- Upscale-only, interpolate-only, and upscale-then-interpolate processing modes
- Real-ESRGAN-family and PyTorch image super-resolution model support in the current catalog
- RIFE-based frame interpolation targeting 30 fps and 60 fps
- Progress telemetry surfaced across extract, upscale, interpolation, encode, and remux stages
- Synthetic benchmark generation and worker-side benchmarking tools
- Automated frontend, Python, and desktop smoke-test entrypoints

## Current Build And Runtime Requirements

- Use the project-local `.venv` as the default worker environment
- Ensure `torch` is installed in that environment before running the desktop app or worker-heavy workflows
- Keep `FFmpeg` available on PATH for probing, encode, remux, and validation flows
- Set `PYTHONPATH=python` when invoking worker modules directly outside the repository scripts
- Install `python/requirements-tensorrt.txt` only when you need the optional TensorRT runner path

## Script Layout

- `scripts/common.ps1`: shared environment, process, and prerequisite helpers
- `scripts/bootstrap.ps1`: Node and Python dependency bootstrap, including PyTorch installation
- `scripts/test.ps1`: frontend tests plus Python `unittest` discovery
- `scripts/build.ps1`: optional tests, frontend build, and Cargo build
- `scripts/run.ps1`: Tauri desktop launch with bootstrap fallback when `torch` is missing
- `scripts/desktop-test.ps1`: aggregate desktop smoke wrapper through WebView remote debugging
- `scripts/desktop_webview_playback_smoke.mjs`: direct desktop playback smoke for `afterUpscale` and `interpolateOnly`
- `scripts/desktop_jobs_window_recovery_smoke.mjs`: direct Jobs workspace recovery smoke
- `scripts/desktop_blind_comparison_smoke.mjs`: direct detached blind comparison smoke
- `scripts/generate-benchmarks.ps1`: synthetic benchmark fixture generation
- `scripts/stop.ps1`: stop active dev and repo worker processes

## Benchmark And Worker Outputs

Common output locations:

- `artifacts/benchmarks`: synthetic fixtures and benchmark results
- `artifacts/jobs`: job progress and status JSON
- `artifacts/runtime`: runtime assets and desktop smoke-test logs
- `artifacts/outputs`: generated media outputs when workflows target repository-managed output paths

## Troubleshooting

- If `./scripts/test.ps1` fails because Python modules are missing, run `./scripts/bootstrap.ps1` against the repo `.venv`.
- If the desktop app refuses to launch because port `1420` is busy, stop the existing repo processes with `./scripts/stop.ps1 -IncludeRepoWorkers`.
- If you run worker commands directly and imports fail, set `PYTHONPATH` to the repository `python` directory first.
- If you need TensorRT benchmarking or runtime validation, install `python/requirements-tensorrt.txt` explicitly in the active environment.

## Related Baseline Documents

- `context/requirements.md`: product and acceptance requirements
- `context/implementation_plan.md`: implementation plan tied to the current baseline
- `README.md`: user-facing overview, workflows, and examples