from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _gradient_frame(width: int, height: int, frame_index: int, total_frames: int) -> Image.Image:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    phase = frame_index / max(total_frames - 1, 1)

    red = (xx * 255).astype(np.uint8)
    green = (yy * 255).astype(np.uint8)
    blue = (((0.5 + 0.5 * np.sin((xx + phase) * np.pi * 8)) * 255)).astype(np.uint8)
    frame = np.stack([red, green, blue], axis=2)
    image = Image.fromarray(frame, mode="RGB")

    draw = ImageDraw.Draw(image)
    grid_step = max(24, width // 32)
    for x_pos in range(0, width, grid_step):
        draw.line((x_pos, 0, x_pos, height), fill=(255, 255, 255), width=1)
    for y_pos in range(0, height, grid_step):
        draw.line((0, y_pos, width, y_pos), fill=(255, 255, 255), width=1)

    square_size = max(48, width // 20)
    offset = int((width - square_size) * phase)
    draw.rectangle((offset, height // 3, offset + square_size, height // 3 + square_size), outline=(255, 32, 32), width=6)
    draw.text((24, 24), f"Frame {frame_index:03d}", fill=(255, 255, 255))
    draw.text((24, 72), f"{width}x{height}", fill=(255, 220, 64))
    return image


def generate_benchmark_fixture(
    *,
    output_dir: Path,
    name: str,
    frames: int,
    width: int,
    height: int,
    downscale_width: int,
    downscale_height: int,
) -> dict[str, object]:
    fixture_dir = output_dir / name
    master_dir = fixture_dir / "master"
    degraded_dir = fixture_dir / "degraded"
    master_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    for frame_index in range(frames):
      master_image = _gradient_frame(width, height, frame_index, frames)
      degraded_image = master_image.resize((downscale_width, downscale_height), Image.Resampling.BICUBIC)

      master_path = master_dir / f"frame_{frame_index:04d}.png"
      degraded_path = degraded_dir / f"frame_{frame_index:04d}.png"
      master_image.save(master_path)
      degraded_image.save(degraded_path)

      entries.append(
          {
              "master": str(master_path),
              "degraded": str(degraded_path),
          }
      )

    manifest = {
        "name": name,
        "frames": frames,
        "masterResolution": {"width": width, "height": height},
        "degradedResolution": {"width": downscale_width, "height": downscale_height},
        "degradation": "bicubic_downscale",
        "entries": entries,
    }

    manifest_path = fixture_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
