from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("artifacts/caption-removal-comparison")
TOOL_SUMMARIES = [
    ("opencv", ROOT / "results" / "local-opencv" / "summary.json"),
    ("lama", ROOT / "results" / "simple-lama" / "summary.json"),
    ("propainter", ROOT / "results" / "propainter" / "summary.json"),
]
REVIEW_DIR = ROOT / "frames" / "model-comparison"


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
    tile = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> int:
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for tool_name, summary_path in TOOL_SUMMARIES:
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries[tool_name] = {item["sampleId"]: item for item in data["results"]}
    if "opencv" not in summaries:
        raise RuntimeError("Local OpenCV summary is required for sample ordering")

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[np.ndarray] = []
    for sample_id, opencv_item in summaries["opencv"].items():
        tiles = [_label_tile(_middle_frame(Path(opencv_item["source"])), f"{sample_id} source")]
        for tool_name in ("opencv", "lama", "propainter"):
            item = summaries.get(tool_name, {}).get(sample_id)
            if item is None:
                blank = np.zeros((180, 320, 3), dtype=np.uint8)
                tiles.append(_label_tile(blank, f"{tool_name} missing"))
                continue
            frame = _middle_frame(Path(item["outputs"]["video"]))
            cv2.imwrite(str(REVIEW_DIR / f"{sample_id}-{tool_name}.png"), frame)
            tiles.append(_label_tile(frame, tool_name))
        rows.append(np.hstack(tiles))

    montage_path = REVIEW_DIR / "caption-removal-model-comparison-montage.png"
    cv2.imwrite(str(montage_path), np.vstack(rows))
    print(montage_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())