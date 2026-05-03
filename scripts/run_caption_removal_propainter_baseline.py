from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("artifacts/caption-removal-comparison")
OPENCV_SUMMARY = ROOT / "results" / "local-opencv" / "summary.json"
RESULTS = ROOT / "results" / "propainter"
PROPAINTER_ROOT = Path("artifacts/runtime/propainter/ProPainter")


def main() -> int:
    repo_root = Path.cwd().resolve()
    propainter_root = (repo_root / PROPAINTER_ROOT).resolve()
    if not (propainter_root / "inference_propainter.py").exists():
        raise RuntimeError(f"ProPainter was not found at {propainter_root}")

    data = json.loads(OPENCV_SUMMARY.read_text(encoding="utf-8"))
    shutil.rmtree(RESULTS, ignore_errors=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for item in data["results"]:
        sample_id = item["sampleId"]
        work_dir = Path(item["outputs"]["workDir"]).resolve()
        input_dir = work_dir / "input"
        mask_dir = work_dir / "masks"
        output_dir = (RESULTS / sample_id).resolve()
        temp_output = (RESULTS / f"{sample_id}-raw").resolve()
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(temp_output, ignore_errors=True)
        temp_output.mkdir(parents=True, exist_ok=True)

        started_at = time.time()
        command = [
            sys.executable,
            "inference_propainter.py",
            "--video",
            str(input_dir),
            "--mask",
            str(mask_dir),
            "--output",
            str(temp_output),
            "--save_fps",
            str(int(item["sample"]["fps"])),
            "--height",
            str(int(item["sample"]["height"])),
            "--width",
            str(int(item["sample"]["width"])),
            "--fp16",
            "--subvideo_length",
            "50",
            "--neighbor_length",
            "8",
            "--ref_stride",
            "10",
            "--mask_dilation",
            "0",
        ]
        print(f"RUN propainter {sample_id}", flush=True)
        subprocess.run(command, cwd=propainter_root, check=True)

        raw_output = temp_output / input_dir.name
        if not raw_output.exists():
            raise RuntimeError(f"ProPainter did not produce expected output folder {raw_output}")
        shutil.move(str(raw_output), str(output_dir))
        shutil.rmtree(temp_output, ignore_errors=True)
        output_video = output_dir / "inpaint_out.mp4"
        comparison_video = output_dir / "masked_in.mp4"
        summary = {
            "sampleId": sample_id,
            "subtitleType": item["subtitleType"],
            "source": item["source"],
            "sample": item["sample"],
            "detections": item["detections"],
            "outputs": {
                "video": str(output_video),
                "comparisonVideo": str(comparison_video),
                "maskSource": str(mask_dir),
            },
            "seconds": time.time() - started_at,
        }
        (output_dir / f"{sample_id}-propainter.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)

    (RESULTS / "summary.json").write_text(json.dumps({"tool": "propainter", "results": summaries}, indent=2), encoding="utf-8")
    print(RESULTS / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())