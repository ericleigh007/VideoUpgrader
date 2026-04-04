from __future__ import annotations

from contextlib import nullcontext
import os
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
MIN_ESTIMATED_BATCH_BYTES = 128 * 1024 * 1024
ACTIVATION_MEMORY_MULTIPLIER = 8192
SUPPORTED_PRECISION_MODES = {"fp32", "fp16", "bf16"}
SUPPORTED_TORCH_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
SUPPORTED_PYTORCH_RUNNERS = {"torch", "tensorrt"}
FRAME_BATCH_SIZE_OVERRIDE_ENV = "UPSCALER_PYTORCH_FRAME_BATCH_SIZE"


@dataclass
class LoadedPytorchModel:
    model_id: str
    model_label: str
    checkpoint_path: Path
    descriptor: object
    device: torch.device
    dtype: torch.dtype
    autocast_dtype: torch.dtype | None
    precision_mode: str
    runner: str
    scale: int
    frame_batch_size: int
    non_blocking: bool
    channels_last: bool
    torch_compile_requested: bool
    torch_compile_enabled: bool
    torch_compile_mode: str
    torch_compile_cudagraphs: bool


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


def _query_cuda_memory_bytes(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    try:
        return torch.cuda.mem_get_info(device)
    except (AttributeError, RuntimeError):
        return None, None


def _estimated_batch_bytes_per_frame(tile_size: int, dtype: torch.dtype) -> int:
    bytes_per_channel = 2 if dtype in {torch.float16, torch.bfloat16} else 4
    estimated = tile_size * tile_size * 3 * bytes_per_channel * ACTIVATION_MEMORY_MULTIPLIER
    return max(MIN_ESTIMATED_BATCH_BYTES, estimated)


def _default_cuda_batch_ceiling(tile_size: int, dtype: torch.dtype, preset: str) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        if tile_size <= 128:
            batch_ceiling = 8
        elif tile_size <= 192:
            batch_ceiling = 6
        elif tile_size <= 256:
            batch_ceiling = 4
        elif tile_size <= 384:
            batch_ceiling = 2
        else:
            batch_ceiling = 1
    else:
        if tile_size <= 128:
            batch_ceiling = 4
        elif tile_size <= 192:
            batch_ceiling = 3
        elif tile_size <= 256:
            batch_ceiling = 2
        else:
            batch_ceiling = 1

    if preset == "qualityMax":
        return max(1, batch_ceiling - 1)
    if preset == "vramSafe":
        if tile_size <= 128:
            return min(batch_ceiling, 2)
        return 1
    return batch_ceiling


def _select_frame_batch_size(device: torch.device, tile_size: int, dtype: torch.dtype, preset: str = "qualityBalanced") -> int:
    if device.type != "cuda":
        return 1
    if tile_size <= 0:
        return 1

    batch_ceiling = _default_cuda_batch_ceiling(tile_size, dtype, preset)
    free_bytes, total_bytes = _query_cuda_memory_bytes(device)
    if free_bytes is None or total_bytes is None:
        return batch_ceiling

    if preset == "qualityMax":
        reserve_bytes = max(1024 * 1024 * 1024, int(total_bytes * 0.20))
    elif preset == "vramSafe":
        reserve_bytes = max(1536 * 1024 * 1024, int(total_bytes * 0.45))
    else:
        reserve_bytes = max(1280 * 1024 * 1024, int(total_bytes * 0.30))

    available_batch_bytes = max(0, free_bytes - reserve_bytes)
    if available_batch_bytes <= 0:
        return 1

    estimated_batch_bytes = _estimated_batch_bytes_per_frame(tile_size, dtype)
    adaptive_batch_size = max(1, available_batch_bytes // estimated_batch_bytes)
    return max(1, min(batch_ceiling, int(adaptive_batch_size)))


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


def resolve_torch_compile_mode(mode: str | None) -> str:
    resolved = (mode or "reduce-overhead").strip().lower()
    if resolved not in SUPPORTED_TORCH_COMPILE_MODES:
        supported = ", ".join(sorted(SUPPORTED_TORCH_COMPILE_MODES))
        raise ValueError(f"Unsupported torch compile mode '{mode}'. Expected one of: {supported}")
    return resolved


def resolve_pytorch_runner(runner: str | None) -> str:
    resolved = (runner or "torch").strip().lower()
    if resolved not in SUPPORTED_PYTORCH_RUNNERS:
        supported = ", ".join(sorted(SUPPORTED_PYTORCH_RUNNERS))
        raise ValueError(f"Unsupported PyTorch runner '{runner}'. Expected one of: {supported}")
    return resolved


def resolve_frame_batch_size_override(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None

    text = raw_value.strip()
    if not text:
        return None

    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"Invalid frame batch size override '{raw_value}'. Expected a positive integer.") from error

    if value <= 0:
        raise ValueError(f"Invalid frame batch size override '{raw_value}'. Expected a positive integer.")

    return value


def _configure_cuda_runtime(device: torch.device, log: list[str]) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    log.append("Enabled CUDA fast-paths: cuDNN benchmark, TF32 matmul, and high float32 matmul precision.")


def _maybe_enable_torch_compile(
    descriptor: ImageModelDescriptor,
    device: torch.device,
    log: list[str],
    enabled: bool,
    mode: str = "reduce-overhead",
    cudagraphs: bool = False,
) -> bool:
    if not enabled:
        return False
    if device.type != "cuda":
        log.append("Skipping torch.compile because PyTorch image SR is not running on CUDA.")
        return False
    if not hasattr(torch, "compile"):
        log.append("Skipping torch.compile because this Torch runtime does not provide torch.compile.")
        return False

    model = getattr(descriptor, "model", None)
    if not isinstance(model, torch.nn.Module):
        log.append("Skipping torch.compile because the loaded PyTorch SR model is not an nn.Module.")
        return False

    compile_mode = resolve_torch_compile_mode(mode)
    inductor_config = None
    previous_cudagraphs: bool | None = None
    if cudagraphs:
        if compile_mode == "max-autotune-no-cudagraphs":
            raise ValueError("torch compile cudagraphs cannot be enabled with max-autotune-no-cudagraphs mode")
        try:
            import torch._inductor.config as inductor_config  # type: ignore[import-not-found]

            previous_cudagraphs = getattr(inductor_config.triton, "cudagraphs", None)
            inductor_config.triton.cudagraphs = True
        except Exception as error:  # noqa: BLE001
            log.append(f"Unable to enable Inductor cudagraphs; continuing without cudagraphs ({error}).")
            cudagraphs = False

    try:
        descriptor._model = torch.compile(model, mode=compile_mode, fullgraph=False)
        if cudagraphs:
            log.append(f"Enabled torch.compile for PyTorch image SR model execution ({compile_mode}, cudagraphs on).")
        else:
            log.append(f"Enabled torch.compile for PyTorch image SR model execution ({compile_mode}).")
        return True
    except Exception as error:  # noqa: BLE001
        log.append(f"torch.compile unavailable for this model/runtime; continuing without it ({error}).")
        return False
    finally:
        if inductor_config is not None and previous_cudagraphs is not None:
            inductor_config.triton.cudagraphs = previous_cudagraphs


def _maybe_enable_channels_last(
    descriptor: ImageModelDescriptor,
    device: torch.device,
    log: list[str],
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    if device.type != "cuda":
        log.append("Skipping channels_last because PyTorch image SR is not running on CUDA.")
        return False

    model = getattr(descriptor, "model", None)
    if not isinstance(model, torch.nn.Module):
        log.append("Skipping channels_last because the loaded PyTorch SR model is not an nn.Module.")
        return False

    try:
        model.to(memory_format=torch.channels_last)
        log.append("Enabled channels_last memory format for PyTorch image SR model execution.")
        return True
    except Exception as error:  # noqa: BLE001
        log.append(f"channels_last unavailable for this model/runtime; continuing without it ({error}).")
        return False


def resolve_precision_mode(*, fp16: bool = False, bf16: bool = False, precision: str | None = None) -> str:
    if precision is not None:
        resolved = precision.strip().lower()
        if resolved not in SUPPORTED_PRECISION_MODES:
            supported = ", ".join(sorted(SUPPORTED_PRECISION_MODES))
            raise ValueError(f"Unsupported precision mode '{precision}'. Expected one of: {supported}")
        if fp16 and resolved != "fp16":
            raise ValueError("precision conflicts with fp16 flag")
        if bf16 and resolved != "bf16":
            raise ValueError("precision conflicts with bf16 flag")
        if fp16 and bf16:
            raise ValueError("fp16 and bf16 cannot be requested at the same time")
        return resolved

    if fp16 and bf16:
        raise ValueError("fp16 and bf16 cannot be requested at the same time")
    if fp16:
        return "fp16"
    if bf16:
        return "bf16"
    return "fp32"


def load_runtime_model(
    model_id: str,
    gpu_id: int | None,
    fp16: bool,
    tile_size: int,
    log: list[str],
    preset: str = "qualityBalanced",
    torch_compile_enabled: bool = False,
    torch_compile_mode: str = "reduce-overhead",
    torch_compile_cudagraphs: bool = False,
    bf16: bool = False,
    precision: str | None = None,
    pytorch_runner: str | None = None,
    channels_last_enabled: bool = False,
) -> LoadedPytorchModel:
    precision_mode = resolve_precision_mode(fp16=fp16, bf16=bf16, precision=precision)
    compile_mode = resolve_torch_compile_mode(torch_compile_mode)
    resolved_runner = resolve_pytorch_runner(pytorch_runner)

    checkpoint_path = ensure_model_checkpoint(model_id)
    descriptor = ModelLoader().load_from_file(str(checkpoint_path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError(f"Checkpoint for '{model_id}' did not load as an image-to-image model")

    device, notes = _resolve_device(gpu_id)
    log.extend(notes)
    frame_batch_size_override = resolve_frame_batch_size_override(os.environ.get(FRAME_BATCH_SIZE_OVERRIDE_ENV))
    if frame_batch_size_override is not None:
        log.append(f"Overriding PyTorch frame batch size via {FRAME_BATCH_SIZE_OVERRIDE_ENV}={frame_batch_size_override}.")

    if resolved_runner == "tensorrt":
        if device.type != "cuda":
            log.append("TensorRT runner requested, but CUDA is unavailable. Falling back to PyTorch runner.")
            resolved_runner = "torch"
        else:
            raw_model = getattr(descriptor, "model", None)
            if not isinstance(raw_model, torch.nn.Module):
                log.append("TensorRT runner requested, but the loaded model is not an nn.Module. Falling back to PyTorch runner.")
                resolved_runner = "torch"
            else:
                from upscaler_worker.models.tensorrt_sr import TensorRtImageModelRunner

                _configure_cuda_runtime(device, log)
                if torch_compile_enabled:
                    log.append("Ignoring torch.compile request because the TensorRT runner replaces the PyTorch execution path.")
                runner_precision_mode = precision_mode
                if runner_precision_mode == "fp16":
                    log.append("TensorRT fp16 is currently disabled for the image SR runner because validation produced non-finite SwinIR outputs on this workstation. Using TensorRT fp32 instead.")
                    runner_precision_mode = "fp32"
                batching_dtype = torch.float16 if runner_precision_mode in {"fp16", "bf16"} else torch.float32
                frame_batch_size = frame_batch_size_override or _select_frame_batch_size(
                    device,
                    tile_size=tile_size,
                    dtype=batching_dtype,
                    preset=preset,
                )
                if frame_batch_size > 1:
                    log.append(f"Using PyTorch micro-batch size {frame_batch_size} for TensorRT image SR frames ({preset}).")
                else:
                    log.append(f"Using single-frame TensorRT batches for image SR frames ({preset}).")
                runner = TensorRtImageModelRunner(
                    model_id=model_id,
                    checkpoint_path=checkpoint_path,
                    torch_model=raw_model,
                    device=device,
                    scale=int(descriptor.scale),
                    precision_mode=runner_precision_mode,
                    log=log,
                )
                return LoadedPytorchModel(
                    model_id=model_id,
                    model_label=model_label(model_id),
                    checkpoint_path=checkpoint_path,
                    descriptor=runner,
                    device=device,
                    dtype=torch.float32,
                    autocast_dtype=None,
                    precision_mode=runner_precision_mode,
                    runner=resolved_runner,
                    scale=int(descriptor.scale),
                    frame_batch_size=frame_batch_size,
                    non_blocking=True,
                    channels_last=False,
                    torch_compile_requested=False,
                    torch_compile_enabled=False,
                    torch_compile_mode=compile_mode,
                    torch_compile_cudagraphs=False,
                )

    if device.type == "cuda":
        descriptor = descriptor.to(device)
        _configure_cuda_runtime(device, log)
    else:
        descriptor = descriptor.cpu()

    dtype = torch.float32
    autocast_dtype: torch.dtype | None = None
    batching_dtype = torch.float32
    if device.type == "cuda" and precision_mode == "fp16" and bool(descriptor.supports_half):
        descriptor = descriptor.half()
        dtype = torch.float16
        batching_dtype = torch.float16
        log.append("Using fp16 inference for PyTorch image SR.")
    elif device.type == "cuda" and precision_mode == "fp16":
        descriptor = descriptor.float()
        log.append("FP16 requested, but this PyTorch image SR model/runtime does not support half precision. Using fp32 instead.")
    elif device.type == "cuda" and precision_mode == "bf16" and bool(getattr(descriptor, "supports_bfloat16", False)):
        descriptor = descriptor.float()
        autocast_dtype = torch.bfloat16
        batching_dtype = torch.bfloat16
        log.append("Using bf16 autocast inference for PyTorch image SR.")
    elif device.type == "cuda" and precision_mode == "bf16":
        descriptor = descriptor.float()
        log.append("BF16 requested, but this PyTorch image SR model/runtime does not support bfloat16 precision. Using fp32 instead.")
    else:
        descriptor = descriptor.float()

    descriptor = descriptor.eval()
    channels_last_active = _maybe_enable_channels_last(descriptor, device, log, channels_last_enabled)
    compile_enabled = _maybe_enable_torch_compile(
        descriptor,
        device,
        log,
        torch_compile_enabled,
        mode=compile_mode,
        cudagraphs=torch_compile_cudagraphs,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    frame_batch_size = frame_batch_size_override or _select_frame_batch_size(
        device,
        tile_size=tile_size,
        dtype=batching_dtype,
        preset=preset,
    )
    if frame_batch_size > 1:
        log.append(f"Using PyTorch micro-batch size {frame_batch_size} for image SR frames ({preset}).")
    else:
        log.append(f"Using single-frame PyTorch batches for image SR frames ({preset}).")
    return LoadedPytorchModel(
        model_id=model_id,
        model_label=model_label(model_id),
        checkpoint_path=checkpoint_path,
        descriptor=descriptor,
        device=device,
        dtype=dtype,
        autocast_dtype=autocast_dtype,
        precision_mode=precision_mode,
        runner=resolved_runner,
        scale=int(descriptor.scale),
        frame_batch_size=frame_batch_size,
        non_blocking=device.type == "cuda",
        channels_last=channels_last_active,
        torch_compile_requested=torch_compile_enabled,
        torch_compile_enabled=compile_enabled,
        torch_compile_mode=compile_mode,
        torch_compile_cudagraphs=torch_compile_cudagraphs,
    )


def _load_frame_array(frame_path: Path) -> np.ndarray:
    with Image.open(frame_path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _load_array_batch(
    frame_arrays: list[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool,
    channels_last: bool = False,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(frame_arrays, axis=0)).permute(0, 3, 1, 2).contiguous()
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)


def _load_frame_batch(
    frame_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool,
    channels_last: bool = False,
) -> torch.Tensor:
    arrays = [_load_frame_array(frame_path) for frame_path in frame_paths]
    return _load_array_batch(arrays, device, dtype, non_blocking, channels_last)


def _save_frame_array(pixels: np.ndarray, frame_path: Path) -> None:
    # Scratch frames are transient pipeline intermediates, so favor faster lossless PNG writes over smaller files.
    Image.fromarray(pixels, mode="RGB").save(frame_path, compress_level=1)


def _tensor_to_pixel_batch(image_tensor: torch.Tensor) -> list[np.ndarray]:
    array = image_tensor.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().float().numpy()
    pixels = np.clip((array * 255.0).round(), 0, 255).astype(np.uint8)
    return [pixels[index] for index in range(pixels.shape[0])]


def _run_descriptor(
    descriptor: object,
    image_tensor: torch.Tensor,
    tile_size: int,
    scale: int,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    batch_size, _, height, width = image_tensor.shape
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None and image_tensor.device.type == "cuda"
        else nullcontext()
    )
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        with torch.inference_mode(), autocast_context:
            return descriptor(image_tensor).detach().cpu().float()

    overlap = min(MODEL_TILE_OVERLAP, max(8, tile_size // 4))
    use_device_output = _should_stage_tiled_output_on_device(image_tensor, scale)
    output_device = image_tensor.device if use_device_output else torch.device("cpu")
    output = torch.zeros((batch_size, 3, height * scale, width * scale), dtype=torch.float32, device=output_device)

    with torch.inference_mode(), autocast_context:
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
        loaded_model.channels_last,
    )
    upscaled_tensor = _run_descriptor(
        loaded_model.descriptor,
        frame_tensor,
        tile_size=tile_size,
        scale=loaded_model.scale,
        autocast_dtype=loaded_model.autocast_dtype,
    )
    pixel_batch = _tensor_to_pixel_batch(upscaled_tensor)
    for frame_path, pixels in zip(output_frames, pixel_batch, strict=True):
        _save_frame_array(pixels, frame_path)

    del frame_tensor
    del upscaled_tensor
    return len(output_frames)


def upscale_arrays(
    *,
    loaded_model: LoadedPytorchModel,
    input_arrays: list[np.ndarray],
    tile_size: int,
) -> list[np.ndarray]:
    frame_tensor = _load_array_batch(
        input_arrays,
        loaded_model.device,
        loaded_model.dtype,
        loaded_model.non_blocking,
        loaded_model.channels_last,
    )
    upscaled_tensor = _run_descriptor(
        loaded_model.descriptor,
        frame_tensor,
        tile_size=tile_size,
        scale=loaded_model.scale,
        autocast_dtype=loaded_model.autocast_dtype,
    )
    pixel_batch = _tensor_to_pixel_batch(upscaled_tensor)
    del frame_tensor
    del upscaled_tensor
    return pixel_batch