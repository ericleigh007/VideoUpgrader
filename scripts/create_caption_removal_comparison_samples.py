from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("artifacts/caption-removal-comparison")
SOURCES = ROOT / "sources"
FPS = 24
WIDTH = 1280
HEIGHT = 720
DURATION_SECONDS = 8
FRAME_COUNT = FPS * DURATION_SECONDS


def _writer(path: Path) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _background(index: int) -> np.ndarray:
    x = np.linspace(0, 1, WIDTH, dtype=np.float32)
    y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]
    wave = np.sin((x * 8.0) + (index * 0.08))[None, :]
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = np.clip(80 + 70 * x + 20 * wave, 0, 255)
    frame[:, :, 1] = np.clip(90 + 65 * y + 25 * np.sin(index * 0.04), 0, 255)
    frame[:, :, 2] = np.clip(110 + 45 * (1.0 - x) + 18 * wave, 0, 255)
    center_x = int((index * 5) % (WIDTH + 220)) - 110
    cv2.circle(frame, (center_x, 240), 92, (160, 120, 92), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 520), (WIDTH, HEIGHT), (58, 74, 88), -1)
    cv2.line(frame, (0, 520), (WIDTH, 480), (122, 145, 153), 4, cv2.LINE_AA)
    return frame


def _outlined_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 3, cv2.LINE_AA)


def create_white_bottom() -> None:
    writer = _writer(SOURCES / "synthetic-white-bottom.mp4")
    for index in range(FRAME_COUNT):
        frame = _background(index)
        _outlined_text(frame, "This subtitle should disappear", (245, 642), 1.45, (255, 255, 255))
        writer.write(frame)
    writer.release()


def create_multiline_bottom() -> None:
    writer = _writer(SOURCES / "synthetic-multiline-bottom.mp4")
    for index in range(FRAME_COUNT):
        frame = _background(index)
        _outlined_text(frame, "Two lines are harder", (330, 590), 1.25, (255, 255, 255))
        _outlined_text(frame, "because the mask gets taller", (235, 650), 1.25, (255, 255, 255))
        writer.write(frame)
    writer.release()


def create_karaoke_moving() -> None:
    writer = _writer(SOURCES / "synthetic-karaoke-moving.mp4")
    for index in range(FRAME_COUNT):
        frame = _background(index)
        x = 90 + int(110 * np.sin(index / 22.0))
        y = 330 + int(38 * np.sin(index / 15.0))
        color = (75, 255, 80) if index % 48 < 24 else (40, 230, 255)
        _outlined_text(frame, "moving colored lyric text", (x, y), 1.18, color)
        writer.write(frame)
    writer.release()


def write_manifest() -> None:
    manifest = {
        "samples": [
            {
                "id": "real-airtag-green-midframe",
                "path": str(SOURCES / "real-airtag-green-midframe.mp4"),
                "kind": "real",
                "subtitleType": "green/yellow outlined captions across mid-frame",
                "recommendedBaselineArgs": [
                    "--bottom-region-fraction", "0.8",
                    "--light-threshold", "245",
                    "--mask-dilate-pixels", "8",
                    "--line-box-padding-pixels", "14",
                    "--line-box-max-gap-pixels", "42",
                    "--inpaint-radius", "9",
                ],
            },
            {
                "id": "real-genz-red-white-bottom",
                "path": str(SOURCES / "real-genz-red-white-bottom.mp4"),
                "kind": "real",
                "subtitleType": "red and white bottom captions over bodycam footage",
                "recommendedBaselineArgs": [
                    "--bottom-region-fraction", "0.3",
                    "--light-threshold", "215",
                    "--color-hue-min", "18",
                    "--color-hue-max", "95",
                    "--mask-dilate-pixels", "6",
                    "--line-box-padding-pixels", "10",
                    "--line-box-max-gap-pixels", "45",
                    "--inpaint-radius", "8",
                ],
            },
            {
                "id": "synthetic-white-bottom",
                "path": str(SOURCES / "synthetic-white-bottom.mp4"),
                "kind": "synthetic",
                "subtitleType": "white bottom subtitle with black outline",
                "recommendedBaselineArgs": [
                    "--bottom-region-fraction", "0.45",
                    "--light-threshold", "180",
                    "--mask-dilate-pixels", "8",
                    "--line-box-padding-pixels", "12",
                    "--line-box-max-gap-pixels", "60",
                    "--inpaint-radius", "8",
                ],
            },
            {
                "id": "synthetic-multiline-bottom",
                "path": str(SOURCES / "synthetic-multiline-bottom.mp4"),
                "kind": "synthetic",
                "subtitleType": "two-line white subtitle over moving background",
                "recommendedBaselineArgs": [
                    "--bottom-region-fraction", "0.5",
                    "--light-threshold", "180",
                    "--mask-dilate-pixels", "8",
                    "--line-box-padding-pixels", "12",
                    "--line-box-max-gap-pixels", "60",
                    "--inpaint-radius", "8",
                ],
            },
            {
                "id": "synthetic-karaoke-moving",
                "path": str(SOURCES / "synthetic-karaoke-moving.mp4"),
                "kind": "synthetic",
                "subtitleType": "moving colored lyric text",
                "recommendedBaselineArgs": [
                    "--bottom-region-fraction", "0.75",
                    "--light-threshold", "245",
                    "--mask-dilate-pixels", "8",
                    "--line-box-padding-pixels", "12",
                    "--line-box-max-gap-pixels", "70",
                    "--inpaint-radius", "8",
                ],
            },
        ],
        "reviewScores": {
            "residualTextScore": "0=readable text remains, 5=no readable text",
            "fillQualityScore": "0=severe smear/ghosting, 5=natural fill",
            "temporalStabilityScore": "0=distracting flicker, 5=stable playback",
            "sceneDamageScore": "0=important scene content damaged, 5=no collateral damage",
        },
    }
    (ROOT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    SOURCES.mkdir(parents=True, exist_ok=True)
    create_white_bottom()
    create_multiline_bottom()
    create_karaoke_moving()
    write_manifest()
    print(ROOT / "sample_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())