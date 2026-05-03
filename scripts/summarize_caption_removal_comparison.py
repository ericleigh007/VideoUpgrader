from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("artifacts/caption-removal-comparison")
SUMMARY = ROOT / "results" / "local-opencv" / "summary.json"
REVIEW_DIR = ROOT / "frames" / "local-opencv"


def _middle_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read representative frame from {video_path}")
    return frame


def _label_tile(frame: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(frame, (426, 240), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tile, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> int:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[np.ndarray] = []
    for item in data["results"]:
        sample_id = item["sampleId"]
        source_frame = _middle_frame(Path(item["source"]))
        removed_frame = _middle_frame(Path(item["outputs"]["video"]))
        overlay_frame = cv2.imread(item["outputs"]["overlayFrame"], cv2.IMREAD_COLOR)
        if overlay_frame is None:
            raise RuntimeError(f"Could not read overlay frame for {sample_id}")
        cv2.imwrite(str(REVIEW_DIR / f"{sample_id}-source.png"), source_frame)
        cv2.imwrite(str(REVIEW_DIR / f"{sample_id}-removed.png"), removed_frame)
        tiles = [
            _label_tile(source_frame, f"{sample_id} source"),
            _label_tile(removed_frame, f"{sample_id} removed"),
            _label_tile(overlay_frame, f"{sample_id} mask"),
        ]
        rows.append(np.hstack(tiles))
    montage = np.vstack(rows)
    montage_path = REVIEW_DIR / "local-opencv-review-montage.png"
    cv2.imwrite(str(montage_path), montage)
    print(montage_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())