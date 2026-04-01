# Build Upscaler

This file is the operational runbook for building, testing, launching, and debugging Upscaler locally.

It is intentionally separate from AGENTS.md.

## Prerequisites

- Windows
- Node.js 20 or newer
- npm
- Cargo and the Rust toolchain
- Python 3.10 or newer in a virtual environment or available via an explicit environment variable
- FFmpeg available on PATH for future media execution work

Preferred Python variable:

```powershell
$env:UPSCALER_PYTHON='C:/path/to/venv/Scripts/python.exe'
```

If the variable is not set, the scripts fall back to `python`.

## One-Time Setup

```powershell
./scripts/bootstrap.ps1
```

## Standard Commands

Run tests:

```powershell
./scripts/test.ps1
```

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

Generate synthetic benchmark fixtures directly:

```powershell
./scripts/generate-benchmarks.ps1
```

## Current Scope Of The Scaffold

- Tauri desktop shell scaffolded
- React workspace-first dashboard scaffolded
- Real-ESRGAN job-planning contract scaffolded
- Synthetic benchmark generator scaffolded
- TypeScript and Python tests wired into repository scripts

The first slice does not yet execute real video inference. It establishes the contracts, math, scripts, and benchmark generation needed for the next implementation pass.

## Script Layout

- `scripts/common.ps1`: shared environment and prerequisite helpers
- `scripts/bootstrap.ps1`: dependency and Python environment bootstrap
- `scripts/test.ps1`: TypeScript and Python tests
- `scripts/build.ps1`: optional tests plus frontend build and Rust check
- `scripts/run.ps1`: launch the Tauri desktop app
- `scripts/generate-benchmarks.ps1`: generate synthetic benchmark fixtures

## Benchmark Generator Output

The synthetic generator writes under:

```text
artifacts/benchmarks
```

Each run emits:

- A master image sequence
- A degraded low-resolution sequence
- A manifest describing resolutions, degradation settings, and frame paths

## Next Implementation Focus

1. Replace sample source metadata in the frontend with real FFprobe-backed inspection.
2. Execute the Real-ESRGAN job plan through the Python worker.
3. Add encode and audio remux validation via FFmpeg.
