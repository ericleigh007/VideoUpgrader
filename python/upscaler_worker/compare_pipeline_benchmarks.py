from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from upscaler_worker.benchmark_pytorch_pipeline_paths import (
    STAGE_ORDER,
    _derive_stage_effective_fps,
    _derive_stage_frame_counts,
)


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_frame_count(payload: dict[str, Any]) -> int:
    fixture = payload.get("fixture")
    if isinstance(fixture, dict):
        duration_seconds = fixture.get("durationSeconds")
        fps = fixture.get("fps")
        try:
            return max(1, int(round(float(duration_seconds) * float(fps))))
        except (TypeError, ValueError):
            pass
    return 1


def _completed_runs_for_path(payload: dict[str, Any], execution_path: str | None) -> tuple[str, list[dict[str, Any]]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Benchmark payload does not contain a results array")

    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        path_name = str(result.get("executionPath") or "unknown")
        if execution_path is not None and path_name != execution_path:
            continue
        runs = result.get("runs")
        if not isinstance(runs, list):
            continue
        completed = [run for run in runs if isinstance(run, dict) and run.get("status") == "completed"]
        if completed:
            candidates.append((path_name, completed))

    if not candidates:
        if execution_path is None:
            raise ValueError("No completed benchmark runs were found")
        raise ValueError(f"No completed benchmark runs were found for execution path '{execution_path}'")

    return candidates[0]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 6)


def _run_number(run: dict[str, Any], key: str) -> float | None:
    value = run.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_number(run: dict[str, Any], object_key: str, value_key: str) -> float | None:
    value_object = run.get(object_key)
    if not isinstance(value_object, dict):
        return None
    try:
        return float(value_object[value_key])
    except (KeyError, TypeError, ValueError):
        return None


def _ensure_stage_metrics(run: dict[str, Any], fallback_frame_count: int) -> tuple[dict[str, int], dict[str, float]]:
    stage_frame_counts = run.get("stageFrameCounts")
    if not isinstance(stage_frame_counts, dict):
        stage_frame_counts = _derive_stage_frame_counts(run, fallback_frame_count)

    stage_effective_fps = run.get("stageEffectiveFps")
    if not isinstance(stage_effective_fps, dict):
        stage_effective_fps = _derive_stage_effective_fps(run, fallback_frame_count)

    return dict(stage_frame_counts), dict(stage_effective_fps)


def _summarize_runs(payload: dict[str, Any], execution_path: str | None) -> dict[str, Any]:
    selected_path, completed_runs = _completed_runs_for_path(payload, execution_path)
    fallback_frame_count = _fixture_frame_count(payload)

    wall_seconds = [_run_number(run, "wallSeconds") for run in completed_runs]
    throughput = [_run_number(run, "averageThroughputFps") for run in completed_runs]
    peak_rss = [_nested_number(run, "resourcePeaks", "processRssBytes") for run in completed_runs]
    peak_gpu = [_nested_number(run, "resourcePeaks", "gpuMemoryUsedBytes") for run in completed_runs]
    peak_scratch = [_nested_number(run, "resourcePeaks", "scratchSizeBytes") for run in completed_runs]
    max_gpu_utilization = [_nested_number(run, "gpuActivity", "maxUtilizationPercent") for run in completed_runs]

    stage_fps: dict[str, float | None] = {}
    stage_frame_counts: dict[str, int | None] = {}
    for stage in STAGE_ORDER:
        fps_metric = f"{stage}Fps"
        fps_values: list[float] = []
        frame_count_values: list[float] = []
        for run in completed_runs:
            run_stage_frame_counts, run_stage_effective_fps = _ensure_stage_metrics(run, fallback_frame_count)
            if fps_metric in run_stage_effective_fps:
                fps_values.append(float(run_stage_effective_fps[fps_metric]))
            if stage in run_stage_frame_counts:
                frame_count_values.append(float(run_stage_frame_counts[stage]))
        stage_fps[fps_metric] = _median(fps_values)
        stage_frame_counts[stage] = int(statistics.median(frame_count_values)) if frame_count_values else None

    return {
        "executionPath": selected_path,
        "completedRuns": len(completed_runs),
        "medianWallSeconds": _median([value for value in wall_seconds if value is not None]),
        "medianOverallFps": _median([value for value in throughput if value is not None]),
        "medianPeakProcessRssBytes": _median([value for value in peak_rss if value is not None]),
        "medianPeakGpuMemoryUsedBytes": _median([value for value in peak_gpu if value is not None]),
        "medianPeakScratchSizeBytes": _median([value for value in peak_scratch if value is not None]),
        "medianMaxGpuUtilizationPercent": _median([value for value in max_gpu_utilization if value is not None]),
        "stageFrameCounts": stage_frame_counts,
        "stageEffectiveFps": stage_fps,
    }


def _metric_delta(before_value: float | None, after_value: float | None, *, lower_is_better: bool) -> dict[str, float | None]:
    if before_value is None or after_value is None:
        return {
            "before": before_value,
            "after": after_value,
            "delta": None,
            "percentChange": None,
            "improvementPercent": None,
        }

    delta = after_value - before_value
    percent_change = None if before_value == 0 else (delta / before_value) * 100.0
    improvement = -percent_change if lower_is_better and percent_change is not None else percent_change
    return {
        "before": round(before_value, 6),
        "after": round(after_value, 6),
        "delta": round(delta, 6),
        "percentChange": round(percent_change, 4) if percent_change is not None else None,
        "improvementPercent": round(improvement, 4) if improvement is not None else None,
    }


def compare_benchmarks(
    *,
    before_payload: dict[str, Any],
    after_payload: dict[str, Any],
    before_execution_path: str | None,
    after_execution_path: str | None,
) -> dict[str, Any]:
    before_summary = _summarize_runs(before_payload, before_execution_path)
    after_summary = _summarize_runs(after_payload, after_execution_path)

    metrics = {
        "medianWallSeconds": _metric_delta(before_summary["medianWallSeconds"], after_summary["medianWallSeconds"], lower_is_better=True),
        "medianOverallFps": _metric_delta(before_summary["medianOverallFps"], after_summary["medianOverallFps"], lower_is_better=False),
        "medianPeakProcessRssBytes": _metric_delta(before_summary["medianPeakProcessRssBytes"], after_summary["medianPeakProcessRssBytes"], lower_is_better=True),
        "medianPeakGpuMemoryUsedBytes": _metric_delta(before_summary["medianPeakGpuMemoryUsedBytes"], after_summary["medianPeakGpuMemoryUsedBytes"], lower_is_better=False),
        "medianPeakScratchSizeBytes": _metric_delta(before_summary["medianPeakScratchSizeBytes"], after_summary["medianPeakScratchSizeBytes"], lower_is_better=True),
        "medianMaxGpuUtilizationPercent": _metric_delta(before_summary["medianMaxGpuUtilizationPercent"], after_summary["medianMaxGpuUtilizationPercent"], lower_is_better=False),
    }

    stage_effective_fps: dict[str, dict[str, float | None]] = {}
    before_stage_fps = before_summary["stageEffectiveFps"]
    after_stage_fps = after_summary["stageEffectiveFps"]
    for stage in STAGE_ORDER:
        metric_name = f"{stage}Fps"
        stage_effective_fps[metric_name] = _metric_delta(
            before_stage_fps.get(metric_name),
            after_stage_fps.get(metric_name),
            lower_is_better=False,
        )

    return {
        "before": before_summary,
        "after": after_summary,
        "metrics": metrics,
        "stageEffectiveFps": stage_effective_fps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare pipeline benchmark JSON results.")
    parser.add_argument("--input", type=Path, help="Compare two execution paths inside one benchmark JSON file.")
    parser.add_argument("--before", type=Path, help="Benchmark JSON captured before a change.")
    parser.add_argument("--after", type=Path, help="Benchmark JSON captured after a change.")
    parser.add_argument("--before-execution-path", help="Execution path to use as the before side.")
    parser.add_argument("--after-execution-path", help="Execution path to use as the after side.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to write the comparison to.")
    args = parser.parse_args()

    if args.input is not None:
        payload = _load_payload(args.input)
        comparison = compare_benchmarks(
            before_payload=payload,
            after_payload=payload,
            before_execution_path=args.before_execution_path,
            after_execution_path=args.after_execution_path,
        )
    else:
        if args.before is None or args.after is None:
            parser.error("Provide either --input or both --before and --after")
        comparison = compare_benchmarks(
            before_payload=_load_payload(args.before),
            after_payload=_load_payload(args.after),
            before_execution_path=args.before_execution_path,
            after_execution_path=args.after_execution_path,
        )

    payload = json.dumps(comparison, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())