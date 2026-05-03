from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("artifacts/caption-removal-comparison")
RESULTS = ROOT / "results" / "local-opencv"
MANIFEST = ROOT / "sample_manifest.json"


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for sample in data["samples"]:
        sample_id = sample["id"]
        output_dir = RESULTS / sample_id
        work_dir = RESULTS / f"{sample_id}-work"
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
        command = [
            sys.executable,
            "-m",
            "upscaler_worker.cli",
            "remove-hard-captions",
            "--source",
            sample["path"],
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(work_dir),
            "--start-seconds",
            "0",
            "--duration-seconds",
            "8",
            "--fps",
            "8",
            *sample["recommendedBaselineArgs"],
            "--keep-work-dir",
        ]
        print(f"RUN {sample_id}", flush=True)
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        summary_start = completed.stdout.find("{")
        summary = json.loads(completed.stdout[summary_start:]) if summary_start >= 0 else {}
        summary["sampleId"] = sample_id
        summary["subtitleType"] = sample["subtitleType"]
        summaries.append(summary)
    (RESULTS / "summary.json").write_text(json.dumps({"tool": "local-opencv", "results": summaries}, indent=2), encoding="utf-8")
    print(RESULTS / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())