from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

from upscaler_worker.model_catalog import ensure_runnable_model, model_label, model_runtime_asset
from upscaler_worker.runtime import runtime_root


MODEL_TILE_OVERLAP = 32


@dataclass
class LoadedPytorchModel:
    model_id: str
    model_label: str
    checkpoint_path: Path
    descriptor: ImageModelDescriptor
    device: torch.device
    dtype: torch.dtype
    scale: int
    frame_batch_size: int
    non_blocking: bool


def _checkpoint_root() -> Path:
    return runtime_root() / "pytorch-models"


def ensure_model_checkpoint(model_id: str) -> Path:
    ensure_runnable_model(model_id)
    asset = model_runtime_asset(model_id)
    if asset is None:
        raise ValueError(f"Model '{model_id}' does not declare a downloadable runtime asset")

    file_name = str(asset["fileName"])
    download_url = str(asset["downloadUrl"])
    checkpoint_path = _checkpoint_root() / file_name
    if checkpoint_path.exists():
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")
    urllib.request.urlretrieve(download_url, partial_path)
    partial_path.replace(checkpoint_path)
    return checkpoint_path


def _resolve_device(gpu_id: int | None) -> tuple[torch.device, list[str]]:
    notes: list[str] = []
    if not torch.cuda.is_available():
        notes.append("PyTorch CUDA runtime unavailable. Falling back to CPU inference.")
        return torch.device("cpu"), notes

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        notes.append("No CUDA devices reported by PyTorch. Falling back to CPU inference.")
        return torch.device("cpu"), notes

    if gpu_id is None:
        return torch.device("cuda:0"), notes

    if 0 <= gpu_id < device_count:
        return torch.device(f"cuda:{gpu_id}"), notes

    if device_count == 1 and gpu_id >= 0:
        notes.append(f"Mapped app GPU id {gpu_id} to PyTorch cuda:0.")
        return torch.device("cuda:0"), notes

    notes.append(f"Requested GPU {gpu_id} is not available to PyTorch. Using cuda:0 instead.")
    return torch.device("cuda:0"), notes


def _select_frame_batch_size(device: torch.device, tile_size: int, dtype: torch.dtype) -> int:
    if device.type != "cuda":
        return 1
    if tile_size <= 0:
        return 1
    if dtype == torch.float16:
        if tile_size <= 128:
            return 4
        if tile_size <= 192:
            return 3
        if tile_size <= 256:
            return 2
        return 1
    if tile_size <= 128:
        return 3
    if tile_size <= 192:
        return 2
    return 1


def _estimated_output_bytes(batch_size: int, height: int, width: int, scale: int) -> int:
    return batch_size * 3 * height * scale * width * scale * 4


def _should_stage_tiled_output_on_device(image_tensor: torch.Tensor, scale: int) -> bool:
    if image_tensor.device.type != "cuda":
        return False

    batch_size, _, height, width = image_tensor.shape
    estimated_bytes = _estimated_output_bytes(batch_size, height, width, scale)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(image_tensor.device)
    except RuntimeError:
        return False

    return estimated_bytes <= min(1024 * 1024 * 1024, int(free_bytes * 0.25))


def load_runtime_model(model_id: str, gpu_id: int | None, fp16: bool, tile_size: int, log: list[str]) -> LoadedPytorchModel:
    checkpoint_path = ensure_model_checkpoint(model_id)
    descriptor = ModelLoader().load_from_file(str(checkpoint_path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError(f"Checkpoint for '{model_id}' did not load as an image-to-image model")

    device, notes = _resolve_device(gpu_id)
    log.extend(notes)

    if device.type == "cuda":
        descriptor = descriptor.to(device)
        torch.backends.cudnn.benchmark = True
    else:
        descriptor = descriptor.cpu()

    dtype = torch.float32
    if device.type == "cuda" and fp16 and bool(descriptor.supports_half):
        descriptor = descriptor.half()
        dtype = torch.float16
        log.append("Using fp16 inference for PyTorch image SR.")
    else:
        descriptor = descriptor.float()

    descriptor = descriptor.eval()
    frame_batch_size = _select_frame_batch_size(device, tile_size=tile_size, dtype=dtype)
    if frame_batch_size > 1:
        log.append(f"Using PyTorch micro-batch size {frame_batch_size} for image SR frames.")
    return LoadedPytorchModel(
        model_id=model_id,
        model_label=model_label(model_id),
        checkpoint_path=checkpoint_path,
        descriptor=descriptor,
        device=device,
        dtype=dtype,
        scale=int(descriptor.scale),
        frame_batch_size=frame_batch_size,
        non_blocking=device.type == "cuda",
    )


def _load_frame_array(frame_path: Path) -> np.ndarray:
    with Image.open(frame_path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _load_frame_batch(
    frame_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool,
) -> torch.Tensor:
    arrays = [_load_frame_array(frame_path) for frame_path in frame_paths]
    tensor = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).contiguous()
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)


def _save_frame_array(pixels: np.ndarray, frame_path: Path) -> None:
    Image.fromarray(pixels, mode="RGB").save(frame_path)


def _tensor_to_pixel_batch(image_tensor: torch.Tensor) -> list[np.ndarray]:
    array = image_tensor.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().float().numpy()
    pixels = np.clip((array * 255.0).round(), 0, 255).astype(np.uint8)
    return [pixels[index] for index in range(pixels.shape[0])]


def _run_descriptor(
    descriptor: ImageModelDescriptor,
    image_tensor: torch.Tensor,
    tile_size: int,
    scale: int,
) -> torch.Tensor:
    batch_size, _, height, width = image_tensor.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        with torch.inference_mode():
            return descriptor(image_tensor).detach().cpu().float()

    overlap = min(MODEL_TILE_OVERLAP, max(8, tile_size // 4))
    use_device_output = _should_stage_tiled_output_on_device(image_tensor, scale)
    output_device = image_tensor.device if use_device_output else torch.device("cpu")
    output = torch.zeros((batch_size, 3, height * scale, width * scale), dtype=torch.float32, device=output_device)

    with torch.inference_mode():
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                bottom = min(top + tile_size, height)
                right = min(left + tile_size, width)

                tile_top = max(0, top - overlap)
                tile_left = max(0, left - overlap)
                tile_bottom = min(height, bottom + overlap)
                tile_right = min(width, right + overlap)

                input_tile = image_tensor[:, :, tile_top:tile_bottom, tile_left:tile_right]
                output_tile = descriptor(input_tile).detach()

                crop_top = (top - tile_top) * scale
                crop_left = (left - tile_left) * scale
                crop_bottom = crop_top + (bottom - top) * scale
                crop_right = crop_left + (right - left) * scale

                cropped_tile = output_tile[:, :, crop_top:crop_bottom, crop_left:crop_right]
                if use_device_output:
                    output[:, :, top * scale:bottom * scale, left * scale:right * scale] = cropped_tile.float()
                else:
                    output[:, :, top * scale:bottom * scale, left * scale:right * scale] = cropped_tile.cpu().float()

    if output.device.type == "cuda":
        return output.cpu().float()

    return output


def upscale_frames(
    *,
    loaded_model: LoadedPytorchModel,
    input_frames: list[Path],
    output_frames: list[Path],
    tile_size: int,
) -> int:
    frame_tensor = _load_frame_batch(
        input_frames,
        loaded_model.device,
        loaded_model.dtype,
        loaded_model.non_blocking,
    )
    upscaled_tensor = _run_descriptor(
        loaded_model.descriptor,
        frame_tensor,
        tile_size=tile_size,
        scale=loaded_model.scale,
    )
    pixel_batch = _tensor_to_pixel_batch(upscaled_tensor)
    for frame_path, pixels in zip(output_frames, pixel_batch, strict=True):
        _save_frame_array(pixels, frame_path)

    del frame_tensor
    del upscaled_tensor
    return len(output_frames)