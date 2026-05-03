from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path("artifacts/caption-removal-comparison")
OPENCV_SUMMARY = ROOT / "results" / "local-opencv" / "summary.json"
RESULTS = ROOT / "results" / "simple-lama"


def _sequence_frames(root: Path) -> list[Path]:
    return sorted(root.glob("frame_*.png"))


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _render_output_video(*, inpainted_dir: Path, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(inpainted_dir / "frame_%08d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )


def _render_side_by_side_video(*, original_dir: Path, inpainted_dir: Path, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(original_dir / "frame_%08d.png"),
            "-framerate",
            str(fps),
            "-i",
            str(inpainted_dir / "frame_%08d.png"),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2,format=yuv420p[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            str(output_path),
        ]
    )


def _write_lama_frame(lama: object, frame_path: Path, mask_path: Path, output_path: Path) -> None:
    image = Image.open(frame_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if not np.any(np.array(mask)):
        shutil.copyfile(frame_path, output_path)
        return
    result = lama(image, mask)
    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result).astype(np.uint8))
    result.save(output_path)


def main() -> int:
    from simple_lama_inpainting import SimpleLama

    data = json.loads(OPENCV_SUMMARY.read_text(encoding="utf-8"))
    shutil.rmtree(RESULTS, ignore_errors=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    lama = SimpleLama()
    summaries: list[dict[str, object]] = []

    for item in data["results"]:
        sample_id = item["sampleId"]
        fps = int(item["sample"]["fps"])
        work_dir = Path(item["outputs"]["workDir"])
        input_dir = work_dir / "input"
        mask_dir = work_dir / "masks"
        output_dir = RESULTS / sample_id
        frame_dir = output_dir / "frames"
        shutil.rmtree(output_dir, ignore_errors=True)
        frame_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.time()
        frame_paths = _sequence_frames(input_dir)
        for frame_path in frame_paths:
            print(f"RUN simple-lama {sample_id} {frame_path.name}", flush=True)
            _write_lama_frame(lama, frame_path, mask_dir / frame_path.name, frame_dir / frame_path.name)

        output_video = output_dir / f"{sample_id}-simple-lama-caption-removed.mp4"
        comparison_video = output_dir / f"{sample_id}-simple-lama-caption-removal-compare.mp4"
        _render_output_video(inpainted_dir=frame_dir, output_path=output_video, fps=fps)
        _render_side_by_side_video(original_dir=input_dir, inpainted_dir=frame_dir, output_path=comparison_video, fps=fps)
        summary = {
            "sampleId": sample_id,
            "subtitleType": item["subtitleType"],
            "source": item["source"],
            "sample": item["sample"],
            "detections": item["detections"],
            "outputs": {
                "video": str(output_video),
                "comparisonVideo": str(comparison_video),
                "frames": str(frame_dir),
                "maskSource": str(mask_dir),
            },
            "seconds": time.time() - started_at,
        }
        (output_dir / f"{sample_id}-simple-lama.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)

    (RESULTS / "summary.json").write_text(json.dumps({"tool": "simple-lama", "results": summaries}, indent=2), encoding="utf-8")
    print(RESULTS / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())