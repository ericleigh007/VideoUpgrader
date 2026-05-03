from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_BOTTOM_REGION_FRACTION = 0.42
DEFAULT_LIGHT_THRESHOLD = 185
DEFAULT_DARK_THRESHOLD = 70
DEFAULT_COLOR_SATURATION_THRESHOLD = 80
DEFAULT_COLOR_VALUE_THRESHOLD = 130
DEFAULT_COLOR_HUE_MIN = 18
DEFAULT_COLOR_HUE_MAX = 95
DEFAULT_MASK_DILATE_PIXELS = 5
DEFAULT_LINE_BOX_PADDING_PIXELS = 0
DEFAULT_LINE_BOX_MAX_GAP_PIXELS = 0
DEFAULT_TEMPORAL_RADIUS = 1
DEFAULT_INPAINT_RADIUS = 3.0
DEFAULT_INPAINT_METHOD = "telea"
DEFAULT_CONTROL_SAFE_BOTTOM_PIXELS = 120


@dataclass(frozen=True)
class CaptionMaskSettings:
    bottom_region_fraction: float = DEFAULT_BOTTOM_REGION_FRACTION
    light_threshold: int = DEFAULT_LIGHT_THRESHOLD
    dark_threshold: int = DEFAULT_DARK_THRESHOLD
    color_saturation_threshold: int = DEFAULT_COLOR_SATURATION_THRESHOLD
    color_value_threshold: int = DEFAULT_COLOR_VALUE_THRESHOLD
    color_hue_min: int = DEFAULT_COLOR_HUE_MIN
    color_hue_max: int = DEFAULT_COLOR_HUE_MAX
    mask_dilate_pixels: int = DEFAULT_MASK_DILATE_PIXELS
    line_box_padding_pixels: int = DEFAULT_LINE_BOX_PADDING_PIXELS
    line_box_max_gap_pixels: int = DEFAULT_LINE_BOX_MAX_GAP_PIXELS
    temporal_radius: int = DEFAULT_TEMPORAL_RADIUS
    max_mask_area_fraction: float = 0.18


def _sequence_frames(root: Path) -> list[Path]:
    return sorted(root.glob("frame_*.png"))


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _extract_sample_frames(
    *,
    ffmpeg: str,
    source: Path,
    input_dir: Path,
    start_seconds: float,
    duration_seconds: float,
    fps: int,
) -> int:
    input_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={fps}",
            str(input_dir / "frame_%08d.png"),
        ]
    )
    return len(_sequence_frames(input_dir))


def _region_top(height: int, bottom_region_fraction: float) -> int:
    fraction = min(0.95, max(0.05, bottom_region_fraction))
    return max(0, min(height - 1, int(round(height * (1.0 - fraction)))))


def _filter_text_components(mask: np.ndarray, roi_height: int, roi_width: int) -> np.ndarray:
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    min_area = max(3, int(round((roi_height * roi_width) * 0.000015)))
    max_area = max(min_area + 1, int(round((roi_height * roi_width) * 0.045)))
    min_component_height = max(4, int(round(roi_height * 0.018)))
    max_component_height = max(8, int(round(roi_height * 0.35)))
    min_component_width = 2
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if width < min_component_width or height < min_component_height or height > max_component_height:
            continue
        fill_ratio = area / max(1, width * height)
        if fill_ratio > 0.9 and width > roi_width * 0.12:
            continue
        filtered[labels == component_index] = 255
    return filtered


def _filter_caption_line_components(mask: np.ndarray, roi_height: int, roi_width: int) -> np.ndarray:
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return np.zeros_like(mask)

    components: list[dict[str, float | int]] = []
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if width <= 0 or height <= 0:
            continue
        components.append(
            {
                "index": component_index,
                "x": x,
                "y": y,
                "right": x + width,
                "width": width,
                "height": height,
                "area": area,
                "center_y": float(centroids[component_index][1]),
            }
        )

    if not components:
        return np.zeros_like(mask)

    keep: set[int] = set()
    for component in components:
        component_width = int(component["width"])
        component_height = int(component["height"])
        component_area = int(component["area"])
        component_fill_ratio = component_area / max(1, component_width * component_height)
        if (
            component_width >= max(42, int(roi_width * 0.16))
            and 6 <= component_height <= max(18, int(roi_height * 0.16))
            and component_fill_ratio <= 0.72
        ):
            keep.add(int(component["index"]))

        center_y = float(component["center_y"])
        row_tolerance = max(10.0, float(component["height"]) * 1.8)
        row = [other for other in components if abs(float(other["center_y"]) - center_y) <= row_tolerance]
        row_area = sum(int(other["area"]) for other in row)
        row_left = min(int(other["x"]) for other in row)
        row_right = max(int(other["right"]) for other in row)
        row_width = row_right - row_left
        if len(row) >= 3 and row_width >= max(32, int(roi_width * 0.045)) and row_area <= roi_height * roi_width * 0.08:
            keep.update(int(other["index"]) for other in row)

    filtered = np.zeros_like(mask)
    for component_index in keep:
        filtered[labels == component_index] = 255
    return filtered


def _caption_line_box_mask(
    mask: np.ndarray,
    roi_height: int,
    roi_width: int,
    padding_pixels: int,
    max_gap_pixels: int,
) -> np.ndarray:
    padding = max(0, int(padding_pixels))
    if padding <= 0:
        return np.zeros_like(mask)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return np.zeros_like(mask)

    components: list[dict[str, float | int]] = []
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if width <= 0 or height <= 0 or area <= 0:
            continue
        components.append(
            {
                "index": component_index,
                "x": x,
                "y": y,
                "right": x + width,
                "bottom": y + height,
                "width": width,
                "height": height,
                "area": area,
                "center_y": float(centroids[component_index][1]),
            }
        )

    line_mask = np.zeros_like(mask)
    seen_rows: set[tuple[int, int, int, int]] = set()
    accepted_boxes: list[tuple[int, int, int, int]] = []
    for component in components:
        center_y = float(component["center_y"])
        row_tolerance = max(10.0, float(component["height"]) * 1.8)
        row = sorted(
            [other for other in components if abs(float(other["center_y"]) - center_y) <= row_tolerance],
            key=lambda item: int(item["x"]),
        )
        clusters: list[list[dict[str, float | int]]] = []
        max_cluster_gap = max(1, int(max_gap_pixels)) if max_gap_pixels > 0 else max(48, int(round(roi_width * 0.08)))
        for row_component in row:
            if not clusters:
                clusters.append([row_component])
                continue
            previous_right = max(int(other["right"]) for other in clusters[-1])
            if int(row_component["x"]) - previous_right <= max_cluster_gap:
                clusters[-1].append(row_component)
            else:
                clusters.append([row_component])

        for cluster in clusters:
            row_left = min(int(other["x"]) for other in cluster)
            row_top = min(int(other["y"]) for other in cluster)
            row_right = max(int(other["right"]) for other in cluster)
            row_bottom = max(int(other["bottom"]) for other in cluster)
            row_width = row_right - row_left
            row_height = row_bottom - row_top
            is_wide_component = any(int(other["width"]) >= max(42, int(roi_width * 0.16)) for other in cluster)
            if len(cluster) < 3 and not is_wide_component:
                continue
            if row_width < max(32, int(roi_width * 0.045)) or row_height > max(18, int(roi_height * 0.18)):
                continue

            x1 = max(0, row_left - padding * 2)
            y1 = max(0, row_top - padding)
            x2 = min(roi_width, row_right + padding * 2)
            y2 = min(roi_height, row_bottom + padding)
            key = (x1, y1, x2, y2)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            accepted_boxes.append(key)
            line_mask[y1:y2, x1:x2] = 255

    for component in components:
        component_center_y = int(round(float(component["center_y"])))
        component_left = int(component["x"])
        component_right = int(component["right"])
        component_top = int(component["y"])
        component_bottom = int(component["bottom"])
        for x1, y1, x2, y2 in accepted_boxes:
            vertically_aligned = y1 - padding <= component_center_y <= y2 + padding
            horizontal_gap = max(x1 - component_right, component_left - x2, 0)
            if not vertically_aligned or horizontal_gap > max(padding * 4, 64):
                continue
            companion_x1 = max(0, component_left - padding * 2)
            companion_y1 = max(0, component_top - padding)
            companion_x2 = min(roi_width, component_right + padding * 2)
            companion_y2 = min(roi_height, component_bottom + padding)
            line_mask[companion_y1:companion_y2, companion_x1:companion_x2] = 255
            break
    return line_mask


def detect_hard_caption_mask(frame_bgr: np.ndarray, settings: CaptionMaskSettings | None = None) -> np.ndarray:
    settings = settings or CaptionMaskSettings()
    height, width = frame_bgr.shape[:2]
    top = _region_top(height, settings.bottom_region_fraction)
    roi = frame_bgr[top:, :]
    roi_height, roi_width = roi.shape[:2]
    if roi_height <= 0 or roi_width <= 0:
        return np.zeros((height, width), dtype=np.uint8)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    light_pixels = gray >= settings.light_threshold
    edges = cv2.Canny(gray, 50, 150)
    edge_nearby = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    colored_hue = (hue >= settings.color_hue_min) & (hue <= settings.color_hue_max)
    colored_pixels = (
        (saturation >= settings.color_saturation_threshold)
        & (value >= settings.color_value_threshold)
        & colored_hue
        & edge_nearby
    )
    candidate = np.where(light_pixels | colored_pixels, 255, 0).astype(np.uint8)

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    raw_colored_candidate = np.where(colored_pixels, 255, 0).astype(np.uint8)
    candidate = _filter_text_components(candidate, roi_height=roi_height, roi_width=roi_width)
    candidate = _filter_caption_line_components(candidate, roi_height=roi_height, roi_width=roi_width)
    if not np.any(candidate):
        return np.zeros((height, width), dtype=np.uint8)

    expand = max(1, int(settings.mask_dilate_pixels))
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, expand * 3), max(3, expand)))
    expanded_text = cv2.dilate(candidate, text_kernel, iterations=1)
    line_boxes = _caption_line_box_mask(
        candidate,
        roi_height=roi_height,
        roi_width=roi_width,
        padding_pixels=settings.line_box_padding_pixels,
        max_gap_pixels=settings.line_box_max_gap_pixels,
    )
    if settings.line_box_padding_pixels > 0 and np.any(line_boxes):
        companion_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, settings.line_box_padding_pixels * 8 + 1), max(3, settings.line_box_padding_pixels * 3 + 1)),
        )
        companion_region = cv2.dilate(line_boxes, companion_kernel, iterations=1)
        colored_companions = cv2.bitwise_and(raw_colored_candidate, companion_region)
        line_boxes = cv2.bitwise_or(line_boxes, cv2.dilate(colored_companions, np.ones((expand, expand), dtype=np.uint8), iterations=1))
    expanded_text = cv2.bitwise_or(expanded_text, line_boxes)
    dark_outline = np.where(gray <= settings.dark_threshold, 255, 0).astype(np.uint8)
    outline_near_text = cv2.bitwise_and(dark_outline, cv2.dilate(expanded_text, np.ones((expand * 2 + 1, expand * 2 + 1), dtype=np.uint8)))
    combined = cv2.bitwise_or(expanded_text, outline_near_text)
    combined = cv2.dilate(combined, np.ones((expand, expand), dtype=np.uint8), iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((max(3, expand * 2), max(3, expand)), dtype=np.uint8))

    if float(np.count_nonzero(combined)) / float(max(1, height * width)) > settings.max_mask_area_fraction:
        return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:, :] = combined
    return mask


def smooth_caption_masks(masks: list[np.ndarray], temporal_radius: int) -> list[np.ndarray]:
    radius = max(0, int(temporal_radius))
    if radius == 0 or len(masks) <= 1:
        return masks
    smoothed: list[np.ndarray] = []
    for index in range(len(masks)):
        start = max(0, index - radius)
        end = min(len(masks), index + radius + 1)
        combined = np.zeros_like(masks[index])
        for neighbor in masks[start:end]:
            combined = cv2.bitwise_or(combined, neighbor)
        smoothed.append(combined)
    return smoothed


def _overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame_bgr.copy()
    color = np.zeros_like(frame_bgr)
    color[:, :, 2] = 255
    alpha = 0.45
    masked = mask > 0
    overlay[masked] = ((frame_bgr[masked].astype(np.float32) * (1.0 - alpha)) + (color[masked].astype(np.float32) * alpha)).astype(
        np.uint8
    )
    return overlay


def _render_side_by_side_video(
    *,
    ffmpeg: str,
    original_dir: Path,
    inpainted_dir: Path,
    output_path: Path,
    fps: int,
    control_safe_bottom_pixels: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
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
            f"[0:v][1:v]hstack=inputs=2[compare];[compare]pad=iw:ih+{max(0, control_safe_bottom_pixels)}:0:0:color=black,format=yuv420p[v]",
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


def _render_output_video(*, ffmpeg: str, inpainted_dir: Path, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
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


def _inpaint_method_flag(method: str) -> int:
    normalized = method.strip().lower()
    if normalized == "telea":
        return cv2.INPAINT_TELEA
    if normalized in {"ns", "navier-stokes", "navier_stokes"}:
        return cv2.INPAINT_NS
    raise ValueError(f"Unsupported caption inpaint method: {method}")


def remove_hard_captions(
    *,
    source: Path,
    output_dir: Path,
    work_dir: Path,
    start_seconds: float = 0.0,
    duration_seconds: float = 4.0,
    fps: int = 12,
    bottom_region_fraction: float = DEFAULT_BOTTOM_REGION_FRACTION,
    light_threshold: int = DEFAULT_LIGHT_THRESHOLD,
    color_hue_min: int = DEFAULT_COLOR_HUE_MIN,
    color_hue_max: int = DEFAULT_COLOR_HUE_MAX,
    mask_dilate_pixels: int = DEFAULT_MASK_DILATE_PIXELS,
    line_box_padding_pixels: int = DEFAULT_LINE_BOX_PADDING_PIXELS,
    line_box_max_gap_pixels: int = DEFAULT_LINE_BOX_MAX_GAP_PIXELS,
    temporal_radius: int = DEFAULT_TEMPORAL_RADIUS,
    inpaint_radius: float = DEFAULT_INPAINT_RADIUS,
    inpaint_method: str = DEFAULT_INPAINT_METHOD,
    keep_work_dir: bool = False,
    control_safe_bottom_pixels: int = DEFAULT_CONTROL_SAFE_BOTTOM_PIXELS,
) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    input_dir = work_dir / "input"
    mask_dir = work_dir / "masks"
    overlay_dir = work_dir / "overlays"
    output_frames_dir = work_dir / "inpainted"
    for directory in (mask_dir, overlay_dir, output_frames_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    extracted_frames = _extract_sample_frames(
        ffmpeg=ffmpeg,
        source=source,
        input_dir=input_dir,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        fps=fps,
    )
    frame_paths = _sequence_frames(input_dir)
    if extracted_frames <= 0 or not frame_paths:
        raise RuntimeError("Caption removal sample extraction did not produce any frames")

    settings = CaptionMaskSettings(
        bottom_region_fraction=bottom_region_fraction,
        light_threshold=light_threshold,
        color_hue_min=color_hue_min,
        color_hue_max=color_hue_max,
        mask_dilate_pixels=mask_dilate_pixels,
        line_box_padding_pixels=line_box_padding_pixels,
        line_box_max_gap_pixels=line_box_max_gap_pixels,
        temporal_radius=temporal_radius,
    )
    frames = [cv2.imread(str(frame_path), cv2.IMREAD_COLOR) for frame_path in frame_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("One or more extracted frames could not be read for caption removal")

    raw_masks = [detect_hard_caption_mask(frame, settings) for frame in frames if frame is not None]
    masks = smooth_caption_masks(raw_masks, settings.temporal_radius)
    inpaint_flag = _inpaint_method_flag(inpaint_method)
    mask_pixels = 0
    masked_frame_count = 0
    for frame_path, frame, mask in zip(frame_paths, frames, masks, strict=True):
        assert frame is not None
        frame_name = frame_path.name
        mask_pixels += int(np.count_nonzero(mask))
        if np.any(mask):
            masked_frame_count += 1
        inpainted = cv2.inpaint(frame, mask, float(inpaint_radius), inpaint_flag)
        if not cv2.imwrite(str(mask_dir / frame_name), mask):
            raise RuntimeError(f"Could not write caption mask frame {frame_name}")
        if not cv2.imwrite(str(overlay_dir / frame_name), _overlay_mask(frame, mask)):
            raise RuntimeError(f"Could not write caption overlay frame {frame_name}")
        if not cv2.imwrite(str(output_frames_dir / frame_name), inpainted):
            raise RuntimeError(f"Could not write inpainted frame {frame_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem.replace(" ", "-").replace("[", "").replace("]", "")[:80]
    sample_suffix = f"s{start_seconds:g}-d{duration_seconds:g}-fps{fps}"
    output_video = output_dir / f"{stem}-{sample_suffix}-caption-removed.mp4"
    comparison_video = output_dir / f"{stem}-{sample_suffix}-caption-removal-compare.mp4"
    overlay_frame = output_dir / f"{stem}-{sample_suffix}-caption-mask-overlay.png"
    _render_output_video(ffmpeg=ffmpeg, inpainted_dir=output_frames_dir, output_path=output_video, fps=fps)
    _render_side_by_side_video(
        ffmpeg=ffmpeg,
        original_dir=input_dir,
        inpainted_dir=output_frames_dir,
        output_path=comparison_video,
        fps=fps,
        control_safe_bottom_pixels=control_safe_bottom_pixels,
    )
    representative_index = max(0, min(len(frame_paths) - 1, len(frame_paths) // 2))
    shutil.copyfile(overlay_dir / frame_paths[representative_index].name, overlay_frame)

    frame_height, frame_width = frames[0].shape[:2] if frames[0] is not None else (0, 0)
    total_pixels = max(1, frame_width * frame_height * len(frame_paths))
    summary = {
        "source": str(source),
        "sample": {
            "startSeconds": start_seconds,
            "durationSeconds": duration_seconds,
            "fps": fps,
            "frames": len(frame_paths),
            "width": frame_width,
            "height": frame_height,
        },
        "settings": {
            "bottomRegionFraction": settings.bottom_region_fraction,
            "lightThreshold": settings.light_threshold,
            "darkThreshold": settings.dark_threshold,
            "colorSaturationThreshold": settings.color_saturation_threshold,
            "colorValueThreshold": settings.color_value_threshold,
            "colorHueMin": settings.color_hue_min,
            "colorHueMax": settings.color_hue_max,
            "maskDilatePixels": settings.mask_dilate_pixels,
            "lineBoxPaddingPixels": settings.line_box_padding_pixels,
            "lineBoxMaxGapPixels": settings.line_box_max_gap_pixels,
            "temporalRadius": settings.temporal_radius,
            "inpaintRadius": inpaint_radius,
            "inpaintMethod": inpaint_method,
        },
        "detections": {
            "maskedFrames": masked_frame_count,
            "maskCoveragePercent": round((mask_pixels / total_pixels) * 100.0, 4),
        },
        "outputs": {
            "video": str(output_video),
            "comparisonVideo": str(comparison_video),
            "overlayFrame": str(overlay_frame),
            "workDir": str(work_dir) if keep_work_dir else None,
        },
        "seconds": time.time() - started_at,
    }
    manifest_path = output_dir / f"{stem}-{sample_suffix}-caption-removal.json"
    summary["outputs"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    return summary