from __future__ import annotations

import argparse
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from upscaler_worker.runtime import (
    ensure_drunet_denoise_runtime,
    ensure_fastdvdnet_denoise_runtime,
    ensure_scunet_denoise_runtime,
    ensure_swinir_denoise_runtime,
)


SWINIR_MODEL_ID = "swinir-denoise-real"
SCUNET_MODEL_ID = "scunet-real-denoise"
DRUNET_MODEL_ID = "drunet-gray-color-denoise"
FASTDVDNET_MODEL_ID = "fastdvdnet"
FRAMEWISE_DEFAULT_TILE_SIZE = {
    SWINIR_MODEL_ID: 256,
    SCUNET_MODEL_ID: 512,
    DRUNET_MODEL_ID: 0,
}
FRAMEWISE_DEFAULT_BATCH_SIZE = {
    SWINIR_MODEL_ID: 1,
    SCUNET_MODEL_ID: 1,
    DRUNET_MODEL_ID: 4,
}


def _sequence_frames(root: Path) -> list[Path]:
    return sorted(root.glob("frame_*.png"))


def _configure_gpu(gpu_id: int | None) -> None:
    del gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _resolve_device(gpu_id: int | None) -> tuple[torch.device, list[str]]:
    notes: list[str] = []
    if not torch.cuda.is_available():
        if gpu_id is not None:
            raise RuntimeError("A GPU id was requested for AI denoise, but PyTorch CUDA is not available.")
        notes.append("PyTorch CUDA runtime unavailable. Falling back to CPU inference.")
        return torch.device("cpu"), notes

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        if gpu_id is not None:
            raise RuntimeError("A GPU id was requested for AI denoise, but PyTorch reports no CUDA devices.")
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


def _configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _prepare_device(model_id: str, gpu_id: int | None, precision: str) -> torch.device:
    device, notes = _resolve_device(gpu_id)
    _configure_cuda_runtime(device)
    for note in notes:
        print(f"[ai-denoise] {note}", flush=True)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"[ai-denoise] model={model_id} device={device} gpu='{device_name}' precision={precision} "
            f"freeMiB={free_bytes // (1024 * 1024)} totalMiB={total_bytes // (1024 * 1024)}",
            flush=True,
        )
    else:
        print(f"[ai-denoise] model={model_id} device=cpu precision=fp32", flush=True)
    return device


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"Invalid integer for {name}: {raw_value}") from error


def _tile_size_for_model(model_id: str) -> int:
    return _env_int("UPSCALER_AI_DENOISE_TILE_SIZE", FRAMEWISE_DEFAULT_TILE_SIZE[model_id])


def _tile_overlap_for_model() -> int:
    return _env_int("UPSCALER_AI_DENOISE_TILE_OVERLAP", 32)


def _frame_batch_size_for_model(model_id: str, device: torch.device, tile_size: int) -> int:
    if device.type != "cuda" or tile_size > 0:
        return 1
    return max(1, _env_int("UPSCALER_AI_DENOISE_FRAME_BATCH_SIZE", FRAMEWISE_DEFAULT_BATCH_SIZE[model_id]))


def _torch_load(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_rgb_tensor(frame_path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(frame_path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=torch.float32)


def _save_rgb_tensor(tensor: torch.Tensor, frame_path: Path) -> None:
    array = tensor.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    pixels = np.clip((array * 255.0).round(), 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(frame_path, compress_level=1)


def _load_sequence_tensor(frame_paths: list[Path], device: torch.device) -> torch.Tensor:
    arrays: list[np.ndarray] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            arrays.append(np.asarray(rgb, dtype=np.float32) / 255.0)
    tensor = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device=device, dtype=torch.float32)


def _pad_temporal_paths(frame_paths: list[Path], minimum_count: int) -> list[Path]:
    if len(frame_paths) >= minimum_count:
        return frame_paths
    if not frame_paths:
        return frame_paths
    padded = list(frame_paths)
    while len(padded) < minimum_count:
        padded.append(padded[-1])
    return padded


def _state_dict_without_dataparallel(state_dict: object) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint did not contain a PyTorch state dictionary")
    if "params" in state_dict and isinstance(state_dict["params"], dict):
        state_dict = state_dict["params"]
    if "params_ema" in state_dict and isinstance(state_dict["params_ema"], dict):
        state_dict = state_dict["params_ema"]
    return {str(key).removeprefix("module."): value for key, value in state_dict.items()}


def _prepare_source_import(source_root: Path) -> None:
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def _ensure_optional_inference_stubs() -> None:
    if "thop" not in sys.modules:
        thop_stub = types.ModuleType("thop")
        thop_stub.profile = lambda *args, **kwargs: (0, 0)
        sys.modules["thop"] = thop_stub


def _pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return tensor, height, width
    return torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect"), height, width


def _restore_shape(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return tensor[..., :height, :width]


def _run_tiled(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    *,
    tile_size: int,
    tile_overlap: int,
    multiple: int,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    padded, original_height, original_width = _pad_to_multiple(image_tensor, multiple)
    _, channels, height, width = padded.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        with torch.inference_mode():
            output = transform(padded) if transform is not None else model(padded)
        return _restore_shape(output, original_height, original_width)

    tile = min(tile_size, height, width)
    tile = max(multiple, tile - tile % multiple)
    overlap = min(max(0, tile_overlap), max(0, tile - multiple))
    stride = max(multiple, tile - overlap)
    height_indices = list(range(0, max(1, height - tile), stride)) + [max(0, height - tile)]
    width_indices = list(range(0, max(1, width - tile), stride)) + [max(0, width - tile)]
    output = torch.zeros_like(padded)
    weights = torch.zeros_like(output)

    with torch.inference_mode():
        for top in sorted(set(height_indices)):
            for left in sorted(set(width_indices)):
                patch = padded[..., top:top + tile, left:left + tile]
                restored = transform(patch) if transform is not None else model(patch)
                output[..., top:top + tile, left:left + tile].add_(restored)
                weights[..., top:top + tile, left:left + tile].add_(1.0)

    return _restore_shape(output / weights.clamp_min(1.0), original_height, original_width)


def _precision_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision not in {"fp16", "bf16"}:
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _run_framewise_frames(
    *,
    model_id: str,
    model: torch.nn.Module,
    frames: list[Path],
    output_dir: Path,
    device: torch.device,
    precision: str,
    multiple: int,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> int:
    tile_size = _tile_size_for_model(model_id)
    tile_overlap = _tile_overlap_for_model()
    batch_size = _frame_batch_size_for_model(model_id, device, tile_size)
    print(
        f"[ai-denoise] tileSize={tile_size} tileOverlap={tile_overlap} frameBatchSize={batch_size}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with _precision_context(device, precision):
        for batch_start in range(0, len(frames), batch_size):
            batch_paths = frames[batch_start:batch_start + batch_size]
            batch_tensor = torch.cat([_load_rgb_tensor(frame_path, device) for frame_path in batch_paths], dim=0)
            restored = _run_tiled(
                model,
                batch_tensor,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                multiple=multiple,
                transform=transform,
            )
            for frame_path, restored_frame in zip(batch_paths, restored, strict=True):
                _save_rgb_tensor(restored_frame.unsqueeze(0), output_dir / frame_path.name)
    return len(frames)


def _denoise_swinir(input_dir: Path, output_dir: Path, gpu_id: int | None, precision: str) -> int:
    runtime = ensure_swinir_denoise_runtime()
    source_root = Path(runtime["sourceRoot"])
    checkpoint_path = Path(runtime["checkpointPath"])
    _prepare_source_import(source_root)

    from models.network_swinir import SwinIR as SwinIRNet

    device = _prepare_device(SWINIR_MODEL_ID, gpu_id, precision)
    model = SwinIRNet(
        upscale=1,
        in_chans=3,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )
    model.load_state_dict(_state_dict_without_dataparallel(_torch_load(checkpoint_path, device)), strict=True)
    model.eval().to(device)
    return _run_framewise_frames(
        model_id=SWINIR_MODEL_ID,
        model=model,
        frames=_sequence_frames(input_dir),
        output_dir=output_dir,
        device=device,
        precision=precision,
        multiple=8,
    )


def _denoise_scunet(input_dir: Path, output_dir: Path, gpu_id: int | None, precision: str) -> int:
    runtime = ensure_scunet_denoise_runtime()
    source_root = Path(runtime["sourceRoot"])
    checkpoint_path = Path(runtime["checkpointPath"])
    _prepare_source_import(source_root)
    _ensure_optional_inference_stubs()

    from models.network_scunet import SCUNet

    device = _prepare_device(SCUNET_MODEL_ID, gpu_id, precision)
    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(_state_dict_without_dataparallel(_torch_load(checkpoint_path, device)), strict=True)
    model.eval().to(device)
    return _run_framewise_frames(
        model_id=SCUNET_MODEL_ID,
        model=model,
        frames=_sequence_frames(input_dir),
        output_dir=output_dir,
        device=device,
        precision=precision,
        multiple=64,
    )


def _denoise_drunet(input_dir: Path, output_dir: Path, gpu_id: int | None, precision: str) -> int:
    runtime = ensure_drunet_denoise_runtime()
    source_root = Path(runtime["sourceRoot"])
    checkpoint_path = Path(runtime["checkpointPath"])
    _prepare_source_import(source_root)

    from models.network_unet import UNetRes

    device = _prepare_device(DRUNET_MODEL_ID, gpu_id, precision)
    model = UNetRes(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    )
    model.load_state_dict(_state_dict_without_dataparallel(_torch_load(checkpoint_path, device)), strict=True)
    model.eval().to(device)
    sigma = float(os.environ.get("UPSCALER_DRUNET_DENOISE_SIGMA", "25")) / 255.0
    def transform(patch: torch.Tensor) -> torch.Tensor:
        noise_map = torch.full((patch.shape[0], 1, patch.shape[2], patch.shape[3]), sigma, device=patch.device, dtype=patch.dtype)
        return model(torch.cat((patch, noise_map), dim=1))

    return _run_framewise_frames(
        model_id=DRUNET_MODEL_ID,
        model=model,
        frames=_sequence_frames(input_dir),
        output_dir=output_dir,
        device=device,
        precision=precision,
        multiple=8,
        transform=transform,
    )


def _denoise_fastdvdnet(input_dir: Path, output_dir: Path, gpu_id: int | None, precision: str) -> int:
    runtime = ensure_fastdvdnet_denoise_runtime()
    source_root = Path(runtime["sourceRoot"])
    checkpoint_path = Path(runtime["checkpointPath"])
    _prepare_source_import(source_root)

    from fastdvdnet import denoise_seq_fastdvdnet
    from models import FastDVDnet

    device = _prepare_device(FASTDVDNET_MODEL_ID, gpu_id, precision)
    model = FastDVDnet(num_input_frames=5)
    model.load_state_dict(_state_dict_without_dataparallel(_torch_load(checkpoint_path, device)), strict=True)
    model.eval().to(device)
    if device.type == "cuda" and precision == "fp16":
        model.half()
    sigma = float(os.environ.get("UPSCALER_FASTDVDNET_DENOISE_SIGMA", "25")) / 255.0
    chunk_frames = _env_int("UPSCALER_FASTDVDNET_CHUNK_FRAMES", 96 if device.type == "cuda" else 32)
    print(f"[ai-denoise] temporalChunkFrames={chunk_frames} temporalRadius=2", flush=True)
    radius = 2
    frame_paths = _sequence_frames(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start in range(0, len(frame_paths), max(1, chunk_frames)):
        chunk_end = min(len(frame_paths), chunk_start + max(1, chunk_frames))
        expanded_start = max(0, chunk_start - radius)
        expanded_end = min(len(frame_paths), chunk_end + radius)
        expanded_paths = _pad_temporal_paths(frame_paths[expanded_start:expanded_end], 5)
        sequence = _load_sequence_tensor(expanded_paths, device)
        if device.type == "cuda" and precision == "fp16":
            sequence = sequence.half()
        noise_std = torch.tensor([sigma], device=device, dtype=sequence.dtype)
        with torch.inference_mode():
            restored_sequence = denoise_seq_fastdvdnet(sequence, noise_std, 5, model).float()
        trim_start = chunk_start - expanded_start
        trim_end = trim_start + (chunk_end - chunk_start)
        for frame_path, restored in zip(frame_paths[chunk_start:chunk_end], restored_sequence[trim_start:trim_end], strict=True):
            _save_rgb_tensor(restored.unsqueeze(0), output_dir / frame_path.name)

    return len(frame_paths)


def run_ai_denoise(model_id: str, input_dir: Path, output_dir: Path, gpu_id: int | None, precision: str) -> int:
    _configure_gpu(gpu_id)
    runners = {
        SWINIR_MODEL_ID: _denoise_swinir,
        SCUNET_MODEL_ID: _denoise_scunet,
        DRUNET_MODEL_ID: _denoise_drunet,
        FASTDVDNET_MODEL_ID: _denoise_fastdvdnet,
    }
    runner = runners.get(model_id)
    if runner is None:
        raise ValueError(f"Unsupported AI denoise model '{model_id}'")
    return runner(input_dir, output_dir, gpu_id, precision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a repo-provided AI denoiser over frame_*.png input frames.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = parser.parse_args()

    gpu_id = args.gpu_id if args.gpu_id >= 0 else None
    frame_count = run_ai_denoise(args.model, args.input, args.output, gpu_id, args.precision)
    print(f"Denoised {frame_count} frames with {args.model}", flush=True)


if __name__ == "__main__":
    main()