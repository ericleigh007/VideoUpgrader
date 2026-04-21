from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor
from torchvision import transforms

from upscaler_worker.cancellation import ensure_not_cancelled, wait_if_paused
from upscaler_worker.model_catalog import ensure_runnable_model, model_backend_id, model_label
from upscaler_worker.runtime import ensure_deepremaster_runtime


SUPPORTED_DEEPREMASTER_MODELS = {"deepremaster"}
SUPPORTED_FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEEPREMASTER_REFERENCE_MIN_EDGE = 256
DEEPREMASTER_PROCESS_MIN_EDGE = 320
DEEPREMASTER_HIGH_PROCESS_MIN_EDGE = 512
DEEPREMASTER_BLOCK_SIZE = 5
DEEPREMASTER_PROCESSING_MODES = {
    "standard": DEEPREMASTER_PROCESS_MIN_EDGE,
    "high": DEEPREMASTER_HIGH_PROCESS_MIN_EDGE,
}


@dataclass
class LoadedDeepRemasterColorizer:
    model_id: str
    model_label: str
    repo_id: str
    checkpoint_path: Path
    source_root: Path
    model_root: Path
    device: torch.device
    precision_mode: str
    autocast_dtype: torch.dtype | None
    model_r: torch.nn.Module
    model_c: torch.nn.Module
    reference_tensor: torch.Tensor | None
    processing_mode: str
    process_min_edge: int


def _resolve_device(gpu_id: int | None) -> tuple[torch.device, list[str]]:
    notes: list[str] = []
    if not torch.cuda.is_available():
        notes.append("PyTorch CUDA runtime unavailable. Falling back to CPU inference.")
        return torch.device("cpu"), notes

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        notes.append("No CUDA devices reported by PyTorch. Falling back to CPU inference.")
        return torch.device("cpu"), notes

    resolved_gpu_id = 0 if gpu_id is None else gpu_id
    if 0 <= resolved_gpu_id < device_count:
        return torch.device(f"cuda:{resolved_gpu_id}"), notes
    if device_count == 1 and resolved_gpu_id >= 0:
        notes.append(f"Mapped app GPU id {resolved_gpu_id} to PyTorch cuda:0.")
        return torch.device("cuda:0"), notes

    notes.append(f"Requested GPU {resolved_gpu_id} is not available to PyTorch. Using cuda:0 instead.")
    return torch.device("cuda:0"), notes


def _resolve_precision(precision: str | None, device: torch.device) -> tuple[str, torch.dtype | None]:
    if device.type != "cuda":
        return "fp32", None

    selected = (precision or "fp32").strip().lower()
    if selected == "fp16":
        return "fp16", torch.float16
    if selected == "bf16":
        return "bf16", torch.bfloat16
    return "fp32", None


def _ensure_repo_import_path(source_root: Path) -> None:
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)


def _convert_lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    lab_image = lab_image.copy()
    lab_image[:, :, 0:1] = lab_image[:, :, 0:1] * 100.0
    lab_image[:, :, 1:3] = np.clip((lab_image[:, :, 1:3] * 255.0) - 128.0, -100.0, 100.0)
    return skcolor.lab2rgb(lab_image.astype(np.float64))


def _add_margin(image: Image.Image, *, target_w: int, target_h: int) -> Image.Image:
    width, height = image.size
    if width == target_w and height == target_h:
        return image

    scale = max(target_w, target_h) / max(width, height)
    resized_width = max(16, int(width * scale / 16.0) * 16)
    resized_height = max(16, int(height * scale / 16.0) * 16)
    resized = transforms.Resize((resized_height, resized_width), interpolation=Image.BICUBIC)(image)
    x_pad = (target_w - resized_width) // 2
    y_pad = (target_h - resized_height) // 2
    result = Image.new(resized.mode, (target_w, target_h), (0, 0, 0))
    result.paste(resized, (x_pad, y_pad))
    return result


def _prepare_reference_tensor(reference_image_paths: list[str], device: torch.device) -> torch.Tensor | None:
    valid_paths = [Path(path) for path in reference_image_paths if Path(path).is_file()]
    if not valid_paths:
        return None

    refs: list[Image.Image] = []
    aspect_mean = 0.0
    for ref_path in valid_paths:
        ref_image = Image.open(ref_path).convert("RGB")
        width, height = ref_image.size
        aspect_mean += width / max(height, 1)
        refs.append(ref_image)

    aspect_mean /= max(len(refs), 1)
    target_w = int(DEEPREMASTER_REFERENCE_MIN_EDGE * aspect_mean) if aspect_mean > 1 else DEEPREMASTER_REFERENCE_MIN_EDGE
    target_h = DEEPREMASTER_REFERENCE_MIN_EDGE if aspect_mean >= 1 else int(DEEPREMASTER_REFERENCE_MIN_EDGE / max(aspect_mean, 1e-6))
    reference_tensor = torch.empty(len(refs), 3, target_h, target_w, dtype=torch.float32)
    for index, ref_image in enumerate(refs):
        prepared = _add_margin(ref_image, target_w=target_w, target_h=target_h)
        reference_tensor[index] = transforms.ToTensor()(prepared)
        ref_image.close()

    return reference_tensor.view(1, reference_tensor.size(0), reference_tensor.size(1), reference_tensor.size(2), reference_tensor.size(3)).to(device)


def _resolve_processing_mode(processing_mode: str | None) -> tuple[str, int]:
    selected_mode = (processing_mode or "standard").strip().lower()
    if selected_mode not in DEEPREMASTER_PROCESSING_MODES:
        supported = ", ".join(sorted(DEEPREMASTER_PROCESSING_MODES))
        raise ValueError(f"Unsupported DeepRemaster processing mode '{selected_mode}'. Expected one of: {supported}")
    return selected_mode, DEEPREMASTER_PROCESSING_MODES[selected_mode]


def load_runtime_colorizer(
    model_id: str,
    gpu_id: int | None,
    precision: str | None,
    log: list[str],
    reference_image_paths: list[str] | None = None,
    processing_mode: str = "standard",
) -> LoadedDeepRemasterColorizer:
    ensure_runnable_model(model_id)
    if model_id not in SUPPORTED_DEEPREMASTER_MODELS or model_backend_id(model_id) != "pytorch-image-colorization":
        raise ValueError(f"Model '{model_id}' is not a DeepRemaster-compatible colorizer")

    runtime_assets = ensure_deepremaster_runtime()
    source_root = Path(runtime_assets["deepremasterSourceRoot"])
    model_root = Path(runtime_assets["deepremasterModelRoot"])
    checkpoint_path = Path(runtime_assets["deepremasterCheckpointPath"])
    _ensure_repo_import_path(source_root)

    from model.remasternet import NetworkC, NetworkR  # type: ignore[import-not-found]

    device, notes = _resolve_device(gpu_id)
    precision_mode, autocast_dtype = _resolve_precision(precision, device)
    resolved_processing_mode, process_min_edge = _resolve_processing_mode(processing_mode)
    for note in notes:
        log.append(note)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_r = NetworkR()
    model_r.load_state_dict(checkpoint["modelR"])
    model_r = model_r.to(device)
    model_r.eval()

    model_c = NetworkC()
    model_c.load_state_dict(checkpoint["modelC"])
    model_c = model_c.to(device)
    model_c.eval()

    if autocast_dtype is not None:
        model_r = model_r.to(dtype=autocast_dtype)
        model_c = model_c.to(dtype=autocast_dtype)

    reference_tensor = _prepare_reference_tensor(reference_image_paths or [], device)
    log.append(f"Loaded DeepRemaster checkpoint from {checkpoint_path}")
    if reference_tensor is not None:
        log.append(f"Prepared {reference_tensor.size(1)} DeepRemaster reference image(s)")
    else:
        log.append("No DeepRemaster reference images selected; using automatic colorization mode")
    log.append(f"DeepRemaster processing mode: {resolved_processing_mode} (min edge {process_min_edge})")

    return LoadedDeepRemasterColorizer(
        model_id=model_id,
        model_label=model_label(model_id),
        repo_id="satoshiiizuka/siggraphasia2019_remastering",
        checkpoint_path=checkpoint_path,
        source_root=source_root,
        model_root=model_root,
        device=device,
        precision_mode=precision_mode,
        autocast_dtype=autocast_dtype,
        model_r=model_r,
        model_c=model_c,
        reference_tensor=reference_tensor,
        processing_mode=resolved_processing_mode,
        process_min_edge=process_min_edge,
    )


def _prepare_luminance_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frame_l = torch.from_numpy(frame_gray).view(frame_gray.shape[0], frame_gray.shape[1], 1)
    frame_l = frame_l.permute(2, 0, 1).float() / 255.0
    return frame_l


def _resolve_processing_size(width: int, height: int, process_min_edge: int) -> tuple[int, int]:
    min_edge = max(1, min(width, height))
    scale = 1.0 if min_edge == process_min_edge else (process_min_edge / min_edge)
    target_width = max(16, int(round((width * scale) / 16.0)) * 16)
    target_height = max(16, int(round((height * scale) / 16.0)) * 16)
    return target_width, target_height


def _tensor_to_rgb_image(lightness_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> np.ndarray:
    output = torch.cat((lightness_tensor, ab_tensor), dim=0).numpy().transpose((1, 2, 0))
    rgb = _convert_lab_to_rgb(output)
    return np.uint8(np.clip(rgb * 255.0, 0, 255))


def colorize_directory(
    *,
    loaded_model: LoadedDeepRemasterColorizer,
    input_dir: Path,
    output_dir: Path,
    cancel_path: str | None,
    pause_path: str | None,
    progress_callback=None,
) -> int:
    frame_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_FRAME_SUFFIXES)
    if not frame_paths:
        raise RuntimeError("No extracted frames were found for colorization.")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = len(frame_paths)
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=loaded_model.autocast_dtype)
        if loaded_model.device.type == "cuda" and loaded_model.autocast_dtype is not None
        else nullcontext()
    )

    processed_count = 0
    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame '{frame_paths[0].name}' for DeepRemaster colorization")
    source_height, source_width = first_frame.shape[:2]
    target_width, target_height = _resolve_processing_size(
        source_width,
        source_height,
        loaded_model.process_min_edge,
    )
    with inference_context():
        with autocast_context:
            for block_start in range(0, total_frames, DEEPREMASTER_BLOCK_SIZE):
                ensure_not_cancelled(cancel_path)
                wait_if_paused(pause_path, cancel_path=cancel_path)

                block_paths = frame_paths[block_start:block_start + DEEPREMASTER_BLOCK_SIZE]
                block_tensor: torch.Tensor | None = None
                for frame_path in block_paths:
                    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError(f"Failed to read frame '{frame_path.name}' for DeepRemaster colorization")
                    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
                    frame_l = _prepare_luminance_tensor(resized_frame)
                    frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
                    block_tensor = frame_l if block_tensor is None else torch.cat((block_tensor, frame_l), dim=2)

                if block_tensor is None:
                    continue
                block_tensor = block_tensor.to(loaded_model.device)
                output_l = loaded_model.model_r(block_tensor)
                if loaded_model.reference_tensor is None:
                    output_ab = loaded_model.model_c(output_l)
                else:
                    output_ab = loaded_model.model_c(output_l, loaded_model.reference_tensor)

                output_l_cpu = output_l.detach().float().cpu()
                output_ab_cpu = output_ab.detach().float().cpu()
                for local_index, frame_path in enumerate(block_paths):
                    output_rgb = _tensor_to_rgb_image(output_l_cpu[0, :, local_index, :, :], output_ab_cpu[0, :, local_index, :, :])
                    if output_rgb.shape[1] != source_width or output_rgb.shape[0] != source_height:
                        output_rgb = cv2.resize(output_rgb, (source_width, source_height), interpolation=cv2.INTER_CUBIC)
                    output_path = output_dir / frame_path.name
                    Image.fromarray(output_rgb).save(output_path)
                    processed_count += 1
                    if progress_callback is not None:
                        progress_callback(processed_count, total_frames)

    if loaded_model.device.type == "cuda":
        torch.cuda.synchronize(loaded_model.device)
    return processed_count