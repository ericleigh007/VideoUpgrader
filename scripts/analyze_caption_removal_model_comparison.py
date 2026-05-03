from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from create_caption_removal_comparison_samples import FPS as SOURCE_FPS
from create_caption_removal_comparison_samples import _background


ROOT = Path("artifacts/caption-removal-comparison")
TOOL_SUMMARIES = {
    "local-opencv": ROOT / "results" / "local-opencv" / "summary.json",
    "simple-lama": ROOT / "results" / "simple-lama" / "summary.json",
    "propainter": ROOT / "results" / "propainter" / "summary.json",
}
REVIEW_DIR = ROOT / "frames" / "model-comparison"
METRICS_PATH = ROOT / "results" / "model-comparison" / "metrics.json"


def _read_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {path}")
    return frames


def _sequence_frames(root: Path) -> list[Path]:
    return sorted(root.glob("frame_*.png"))


def _load_summaries() -> dict[str, dict[str, dict[str, object]]]:
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for tool, path in TOOL_SUMMARIES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        summaries[tool] = {item["sampleId"]: item for item in data["results"]}
    return summaries


def _mask_bbox(mask: np.ndarray, padding: int = 28) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        height, width = mask.shape[:2]
        return 0, 0, width, height
    height, width = mask.shape[:2]
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + padding + 1),
        min(height, int(ys.max()) + padding + 1),
    )


def _label_tile(frame: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    tile = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _synthetic_clean_frame(sample_id: str, sample_frame_index: int, sample_fps: int) -> np.ndarray | None:
    if not sample_id.startswith("synthetic-"):
        return None
    source_frame_index = int(round(sample_frame_index * SOURCE_FPS / sample_fps))
    return _background(source_frame_index)


def _masked_mae(frame: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    masked = mask > 0
    if not np.any(masked):
        return 0.0
    return float(np.mean(np.abs(frame[masked].astype(np.float32) - clean[masked].astype(np.float32))))


def _masked_flicker(frames: list[np.ndarray], masks: list[np.ndarray]) -> float:
    values: list[float] = []
    for index in range(1, min(len(frames), len(masks))):
        mask = cv2.bitwise_or(masks[index - 1], masks[index]) > 0
        if not np.any(mask):
            continue
        diff = np.abs(frames[index].astype(np.float32) - frames[index - 1].astype(np.float32))
        values.append(float(np.mean(diff[mask])))
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    summaries = _load_summaries()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {"tools": {}, "samples": {}}
    crop_rows: list[np.ndarray] = []

    sample_ids = list(summaries["local-opencv"].keys())
    for sample_id in sample_ids:
        opencv_item = summaries["local-opencv"][sample_id]
        sample_fps = int(opencv_item["sample"]["fps"])
        work_dir = Path(opencv_item["outputs"]["workDir"])
        source_frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in _sequence_frames(work_dir / "input")]
        masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in _sequence_frames(work_dir / "masks")]
        source_frames = [frame for frame in source_frames if frame is not None]
        masks = [mask for mask in masks if mask is not None]
        if not source_frames or not masks:
            raise RuntimeError(f"Missing source frames or masks for {sample_id}")

        sample_metrics: dict[str, object] = {}
        tool_frames: dict[str, list[np.ndarray]] = {}
        for tool, tool_items in summaries.items():
            item = tool_items[sample_id]
            frames = _read_video_frames(Path(item["outputs"]["video"]))
            tool_frames[tool] = frames
            synthetic_errors: list[float] = []
            for frame_index, frame in enumerate(frames[: len(masks)]):
                clean = _synthetic_clean_frame(sample_id, frame_index, sample_fps)
                if clean is None:
                    continue
                synthetic_errors.append(_masked_mae(frame, clean, masks[frame_index]))
            sample_metrics[tool] = {
                "runtimeSeconds": round(float(item["seconds"]), 3),
                "maskedFlicker": round(_masked_flicker(frames, masks), 3),
                "syntheticMaskedMaeToClean": round(float(np.mean(synthetic_errors)), 3) if synthetic_errors else None,
            }

        metrics["samples"][sample_id] = sample_metrics

        for frame_index in (12, 32, 52):
            mask = masks[min(frame_index, len(masks) - 1)]
            x1, y1, x2, y2 = _mask_bbox(mask)
            tiles = [_label_tile(source_frames[frame_index][y1:y2, x1:x2], f"{sample_id} source f{frame_index + 1}", (300, 120))]
            for tool in ("local-opencv", "simple-lama", "propainter"):
                frame = tool_frames[tool][frame_index]
                tiles.append(_label_tile(frame[y1:y2, x1:x2], tool, (300, 120)))
            crop_rows.append(np.hstack(tiles))

    crop_montage_path = REVIEW_DIR / "caption-removal-model-crop-montage.png"
    cv2.imwrite(str(crop_montage_path), np.vstack(crop_rows))
    metrics["cropMontage"] = str(crop_montage_path)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(METRICS_PATH)
    print(crop_montage_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())